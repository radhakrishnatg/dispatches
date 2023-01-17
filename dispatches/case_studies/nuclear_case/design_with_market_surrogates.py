#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

__author__ = "Radhakrishna Tumbalam Gooty"

# This code requires OMLT v1.0

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from pyomo.common.fileutils import this_file_dir
from pyomo.environ import (
    ConcreteModel,
    Var, 
    NonNegativeReals, 
    value, 
    Expression, 
    Constraint,
    Objective, 
    sqrt,
)

from idaes.apps.grid_integration import MultiPeriodModel
import idaes.logger as idaeslog
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver

from dispatches.case_studies.nuclear_case.nuclear_flowsheet import (
    build_ne_flowsheet,
    fix_dof_and_initialize,
)

import omlt  # omlt can encode the neural networks in Pyomo
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io import load_keras_sequential

PEM_CAPEX = 400
LIFETIME = 30

NP_CAPACITY = 400               # Capacity of the nuclear power plant (in MW)
H2_PROD_RATE = (1000 / 50)      # Hydrogen production rate (in kg/MW-h)
NUM_REP_DAYS = 30               # Number of clusters/representative days

NPP_FOM = 13.7  # Normalized FOM = (120,000 / 8760) = $13.7 per MWh
NPP_VOM = 2.3   # Using a VOM cost of $2.3 per MWh for the nuclear power plant
PEM_FOM = 5.47  # Normalized FOM = (47,900 / 8760) = $5.47 per MWh
PEM_VOM = 1.3   # Using a VOM cost of $1.3 per MWh for the PEM electrolyzer

KW_TO_MW = 1e-3
HOURS_TO_S = 3600
MW_H2 = 2.016e-3

# path for folder that has surrogate models
surrogate_dir = os.path.join(this_file_dir(), "nn_market_surrogates")

# ================= Read the cluster centers (dispatch representative days) ====================
with open(os.path.join(surrogate_dir, "dispatch_cluster_centers.json"), 'r') as f:
    cluster_results = json.load(f)
cluster_center = np.array(cluster_results['model_params']['cluster_centers_'])
cluster_center = cluster_center.reshape(NUM_REP_DAYS, 24)

# add zero/full capacity days to the clustering results. 
full_days = np.array([np.ones(24)])
zero_days = np.array([np.zeros(24)])

# corresponds to the ws, ws[0] is for zero cf days and ws[31] is for full cd days.
cluster_centers = np.concatenate((zero_days, cluster_center, full_days), axis=0)
# ===============================================================================================

# ================ Get input-output scaling factors for dispatch frequency ======================
with open(os.path.join(surrogate_dir, "dispatch_input_output_scaling_data.json"), 'rb') as f:
    dispatch_data = json.load(f)

# Dispatch frequency surrogate
input_bounds_dispatch = {
    i: (dispatch_data['xmin'][i], dispatch_data['xmax'][i]) 
    for i in range(len(dispatch_data['xmin']))
}
scaling_object_dispatch = omlt.OffsetScaling(
    offset_inputs=dispatch_data['xm_inputs'],
    factor_inputs=dispatch_data['xstd_inputs'],
    offset_outputs=dispatch_data['ws_mean'],
    factor_outputs=dispatch_data['ws_std'],
)

# load keras neural networks for weights 
nn_dispatch = keras.models.load_model(os.path.join(surrogate_dir, "keras_dispatch_frequency"))
net_dispatch_defn = load_keras_sequential(nn_dispatch, scaling_object_dispatch, input_bounds_dispatch)
# ==================================================================================================

# ================ Get input-output scaling factors for revenue ====================================
with open(os.path.join(surrogate_dir, "revenue_input_output_scaling_data.json"), 'rb') as f:
    revenue_data = json.load(f)

input_bounds_revenue = {
    i: (revenue_data['xmin'][i], revenue_data['xmax'][i]) 
    for i in range(len(revenue_data['xmin']))
}
scaling_object_revenue = omlt.OffsetScaling(
    offset_inputs=revenue_data["xm_inputs"],
    factor_inputs=revenue_data["xstd_inputs"],
    offset_outputs=revenue_data["y_mean"],
    factor_outputs=revenue_data["y_std"],
)

nn_revenue = keras.models.load_model(os.path.join(surrogate_dir, "keras_revenue"))
revenue_defn = load_keras_sequential(nn_revenue, scaling_object_revenue, input_bounds_revenue)
# ====================================================================================================


def append_revenue_surrogate(m, reserve, max_lmp):
    x = m.threshold_price
    y = m.pem_np_cap_ratio

    if reserve ==10 and max_lmp == 500:
        coeff = {
            "a": 2.98413128e+04, "b": -1.37555194e+07, "c": -1.62067118e+06,
            "d":  5.13815805e+07, "e": -2.96333861e+06, "f": 1.20052698e+08,
        }

    elif reserve == 15 and max_lmp == 500:
        coeff = {
            "a": 3.52883374e+04, "b": -7.91114675e+06, "c": -1.92976788e+06,
            "d": 5.01143499e+07, "e": -2.86623006e+06, "f":  1.09242591e+08,
        }

    elif reserve == 10 and max_lmp == 1000:
        coeff = {
            "a": 3.47844711e+04, "b": -3.13563653e+06, "c": -1.88163364e+06,
            "d":  4.84757224e+07, "e": -3.48739257e+06, "f":  1.68164994e+08,
        }

    elif reserve == 15 and max_lmp == 1000:
        coeff = {
            "a": 3.56735313e+04, "b": 1.48656547e+07, "c": -1.92050122e+06,
            "d": 4.75459030e+07, "e": -3.57109651e+06, "f": 1.43006742e+08,
        }

    m.electricity_revenue = Expression(
        expr=coeff["a"] * x**2 + coeff["b"] * y**2 + coeff["c"] * x + 
             coeff["d"] * y + coeff["e"] * x * y + coeff["f"]
    )

    return


def unfix_dof(m):
    """
    This function unfixes a few degrees of freedom for optimization

    Args:
        m: object containing the integrated nuclear plant flowsheet

    Returns:
        None
    """
    # Unfix split fractions in the power splitter
    m.fs.np_power_split.split_fraction.unfix()

    return


def conceptual_design_dynamic_NE(num_rep_days, reserve=10, max_lmp=500, H2_SELLING_PRICE=2, verbose=False,):
    
    m = ConcreteModel(name='NE_conceptual_design_dynamic_surrogates')

    # Define a variable for the PEM capacity
    m.pem_capacity = Var(
        within=NonNegativeReals,
        doc="Capacity of the PEM electrolyzer [in MW]",
    )

    # Inputs to the frequency surrogates: PEM capacity/NP capacity, threshold price, reserve, max_lmp
    m.pem_np_cap_ratio = Var(
        within=NonNegativeReals,
        bounds=(0.05, 0.5),
        initialize=0.25,
        doc="Ratio of capacities of PEM and nuclear power plant",
    )
    m.threshold_price = Var(
        within=NonNegativeReals,
        initialize=20,
        doc="Threshold LMP below which selling H2 is more profitable",
    )
    m.reserve = Var(
        within=NonNegativeReals,
        doc="Percentage of reserves",
    )
    m.max_lmp = Var(
        within=NonNegativeReals,
        doc="Maximum LMP",
    )
    m.reserve.fix(reserve)
    m.max_lmp.fix(max_lmp)  

    # Relation between PEM capacity and pem_np_cap_ratio
    m.pem_capacity_definition = Constraint(
        expr=m.pem_capacity == m.pem_np_cap_ratio * NP_CAPACITY
    )
    # Threshold price calculation
    m.threshold_price_definition = Constraint(
        expr=m.threshold_price == H2_PROD_RATE * H2_SELLING_PRICE
    )

    # Add NN surrogates for dispatch frequency to the model using omlt
    m.nn_dispatch = omlt.OmltBlock()
    m.nn_dispatch.build_formulation(FullSpaceNNFormulation(net_dispatch_defn))

    # Add NN surrogate for revenue to the model
    m.nn_revenue = omlt.OmltBlock()
    m.nn_revenue.build_formulation(FullSpaceNNFormulation(revenue_defn))

    inputs = [m.threshold_price, m.pem_np_cap_ratio, m.reserve, m.max_lmp]  

    @m.Constraint(range(len(inputs)))
    def input_variable_equality_dispatch(blk, i):
        return inputs[i] == blk.nn_dispatch.inputs[i]

    @m.Constraint(range(len(inputs)))
    def input_variable_equality_revenue(blk, i):
        return inputs[i] == blk.nn_revenue.inputs[i]
    
    m.set_days = [i for i in range(num_rep_days)]

    m.electricity_revenue = Var(within=NonNegativeReals, initialize=1e9)
    m.weights = Var(
        m.set_days, 
        within=NonNegativeReals,
        initialize={x: 1 / num_rep_days for x in m.set_days},
    )
    m.weights_non_neg = Var(
        m.set_days, 
        within=NonNegativeReals,
        initialize={x: 1 / num_rep_days for x in m.set_days},
    )

    # Convert any negative weights to postive weights.
    @m.Constraint(m.set_days)
    def non_neg_weights_definition(blk, i):
        return (
            blk.weights_non_neg[i] == 0.5 * sqrt(blk.nn_dispatch.outputs[i]**2 + 0.001**2) + 0.5 * blk.nn_dispatch.outputs[i]
        )

    # Re-scale weights so that the add up to one
    @m.Constraint(m.set_days)
    def weights_definition(blk, i):
        return (
            blk.weights[i] * sum(blk.weights_non_neg[j] for j in m.set_days) == blk.weights_non_neg[i]
        )

    m.revenue_surrogate = Constraint(expr=m.electricity_revenue == m.nn_revenue.outputs[0])

    # Create the multiperiod model
    m.mp_model = MultiPeriodModel(
        n_time_points=24,
        set_days=m.set_days,
        process_model_func=build_ne_flowsheet,
        linking_variable_func=None,
        initialization_func=fix_dof_and_initialize,
        unfix_dof_func=unfix_dof,
        flowsheet_options={
            "np_capacity": 400,
            "include_tank": False,
        },
        initialization_options={
            "split_frac_grid": 0.95,
        },
        use_stochastic_build=True,
        outlvl=idaeslog.WARNING,
    )

    # Capacity factor definition = 1 - (electricity_to_pem / pem_capacity)
    @m.Constraint(m.mp_model.set_period)
    def capacity_factor_definition(blk, t, d):
        return (
            blk.mp_model.period[t, d].fs.pem.electricity[0] == blk.pem_capacity * 1000 * (1 - cluster_centers[d][t-1])
        )

    assert degrees_of_freedom(m) == 1

    for blk in m.mp_model.period.values():
        blk.fs.h2_revenue = Expression(
            expr=blk.fs.pem.outlet.flow_mol[0] * MW_H2 * HOURS_TO_S * H2_SELLING_PRICE
        )

        blk.fs.operating_cost = Expression(
            expr=blk.fs.np_power_split.electricity[0] * KW_TO_MW * NPP_VOM +
                 blk.fs.pem.electricity[0] * KW_TO_MW * PEM_VOM - blk.fs.h2_revenue
        )

    m.total_operating_cost = Expression(
        expr=sum(
            m.weights[d] * 366 * m.mp_model.period[t, d].fs.operating_cost
            for (t, d) in m.mp_model.set_period
        )
    )

    m.total_hydrogen_revenue = Expression(
        expr=sum(
            m.weights[d] * 366 *m.mp_model.period[t, d].fs.h2_revenue
            for (t, d) in m.mp_model.set_period
        )
    )

    m.pem_cap_cost = Expression(expr=(PEM_CAPEX * 1000 / LIFETIME) * m.pem_capacity)
    m.total_cost = Expression(expr = m.pem_cap_cost + m.total_operating_cost)

    # set objective functions in $
    m.obj = Objective(expr = m.total_cost - m.electricity_revenue)
    
    return m


if __name__ == '__main__':
    h2_price = 1
    mdl = conceptual_design_dynamic_NE(NUM_REP_DAYS + 2, reserve=10, max_lmp=500, H2_SELLING_PRICE=h2_price)

    solver = get_solver()
    solver.solve(mdl, tee=True)
    # pem_cap = [i / 100 for i in range(5, 51, 5)]
    pem_cap = [0.25]
    elec_rev = []
    h2_rev = []
    net_npv = []

    for p in pem_cap:
        mdl.pem_capacity.fix(p * 400)

        solver.solve(mdl)
        elec_rev.append(mdl.electricity_revenue.value / 1e6)
        h2_rev.append(mdl.total_hydrogen_revenue.expr() / 1e6)
        net_npv.append(-mdl.obj.expr() / 1e6)

    print("=" * 80)
    print("H2 selling price: ", h2_price)
    print("Annualized profit: ", -mdl.obj.expr() / 1e6)
    print("Hydrogen revenue: ", mdl.total_hydrogen_revenue.expr() / 1e6)
    print("Electricity revenue: ", mdl.electricity_revenue.value / 1e6)
    print("PEM fraction: ", mdl.pem_capacity.value / 400)

    # plt.plot(pem_cap, elec_rev)
    # plt.xlabel("PEM Capacity Fraction")
    # plt.ylabel("Electricity revenue")
    # plt.title("H2 SP: $" + str(h2_price) + "Max LMP: " + str(1000))
    # plt.show()
    
    # plt.plot(pem_cap, h2_rev)
    # plt.xlabel("PEM Capacity Fraction")
    # plt.ylabel("H2 revenue")
    # plt.title("H2 SP: $" + str(h2_price) + "Max LMP: " + str(1000))
    # plt.show()

    # plt.plot(pem_cap, net_npv)
    # plt.xlabel("PEM Capacity Fraction")
    # plt.ylabel("NPV")
    # plt.title("H2 SP: $" + str(h2_price) + "Max LMP: " + str(1000))
    # plt.show()
