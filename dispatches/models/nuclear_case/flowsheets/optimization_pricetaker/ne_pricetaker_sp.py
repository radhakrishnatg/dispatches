# General python imports
import matplotlib.pyplot as plt
import numpy as np
import json
from itertools import product

# Pyomo imports
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           NonNegativeReals,
                           TransformationFactory,
                           maximize,
                           Block)
from pyomo.network import Arc
from pyomo.common.timing import TicTocTimer

# IDAES imports
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import (Translator,
                                              Mixer,
                                              MomentumMixingType)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock
from idaes.core.util.misc import get_solver
from idaes.core.util import from_json, to_json

# DISPATCHES imports
from dispatches.models.nuclear_case.unit_models. \
    hydrogen_turbine_unit import HydrogenTurbine
from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration as h2_ideal_config
from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration as hturbine_config
import dispatches.models.nuclear_case.properties.h2_reaction \
    as h2_reaction_props
from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer
from dispatches.models.renewables_case.elec_splitter import ElectricalSplitter

# Import additional functions
from process_lmp_signals import append_lmp_signal, append_raven_lmp_signal
from hydrogen_tank_simplified import SimpleHydrogenTank


def build_ne_flowsheet(m):
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Load thermodynamic and reaction packages
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)
    m.fs.h2turbine_props = GenericParameterBlock(default=hturbine_config)
    m.fs.reaction_params = h2_reaction_props.H2ReactionParameterBlock(
        default={"property_package": m.fs.h2turbine_props})

    # Add electrical splitter
    m.fs.np_power_split = ElectricalSplitter(default={
        "num_outlets": 2,
        "outlet_list": ["np_to_grid", "np_to_pem"]})

    # Add PEM electrolyzer
    m.fs.pem = PEM_Electrolyzer(
        default={"property_package": m.fs.h2ideal_props})

    # Add hydrogen tank
    m.fs.h2_tank = SimpleHydrogenTank(default={
        "property_package": m.fs.h2ideal_props})

    # Add translator block
    m.fs.translator = Translator(default={
        "inlet_property_package": m.fs.h2ideal_props,
        "outlet_property_package": m.fs.h2turbine_props})

    # Add translator block constraints
    m.fs.translator.eq_flow_hydrogen = Constraint(
        expr=m.fs.translator.inlet.flow_mol[0] ==
             m.fs.translator.outlet.flow_mol[0])

    m.fs.translator.eq_temperature = Constraint(
        expr=m.fs.translator.inlet.temperature[0] ==
             m.fs.translator.outlet.temperature[0])

    m.fs.translator.eq_pressure = Constraint(
        expr=m.fs.translator.inlet.pressure[0] ==
             m.fs.translator.outlet.pressure[0])

    # Add mixer block
    # using minimize pressure for all inlets and outlet of the mixer
    # because pressure of inlets is already fixed in flowsheet,
    # using equality will over-constrain
    m.fs.mixer = Mixer(default={
        "momentum_mixing_type": MomentumMixingType.minimize,
        "property_package": m.fs.h2turbine_props,
        "inlet_list": ["air_feed", "hydrogen_feed"]})

    # Add hydrogen turbine
    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})

    """
    Connect the individual blocks via Arcs
    """
    # Connect the electrical splitter and PEM
    m.fs.arc_np_to_pem = Arc(
        source=m.fs.np_power_split.np_to_pem_port,
        destination=m.fs.pem.electricity_in
    )

    # Connect the pem electrolyzer and h2 tank
    m.fs.arc_pem_to_h2_tank = Arc(
        source=m.fs.pem.outlet,
        destination=m.fs.h2_tank.inlet
    )

    # Connect h2 tank and translator
    m.fs.arc_h2_tank_to_translator = Arc(
        source=m.fs.h2_tank.outlet_to_turbine,
        destination=m.fs.translator.inlet
    )

    # Connect translator and mixer
    m.fs.arc_translator_to_mixer = Arc(
        source=m.fs.translator.outlet,
        destination=m.fs.mixer.hydrogen_feed
    )

    # Connect mixer and h2 turbine
    m.fs.arc_mixer_to_h2_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def fix_dof_and_initialize(m,
                           np_power_output=1000 * 1e3,
                           pem_outlet_pressure=1.01325,
                           pem_outlet_temperature=300,
                           air_h2_ratio=10.76,
                           compressor_dp=24.01):
    # Fix the dof of the electrical splitter
    m.fs.np_power_split.electricity[0].fix(np_power_output)
    m.fs.np_power_split.split_fraction["np_to_grid", 0].fix(0.5)

    m.fs.np_power_split.initialize()

    # Fix the dof of the electrolyzer
    # Conversion of kW to mol/sec of H2 based on H-tec design of 54.517kW-hr/kg
    m.fs.pem.electricity_to_mol.fix(0.002527406)
    m.fs.pem.outlet.pressure.fix(pem_outlet_pressure * 1e5)
    m.fs.pem.outlet.temperature.fix(pem_outlet_temperature)

    propagate_state(m.fs.arc_np_to_pem)
    m.fs.pem.initialize()

    # Fix the dof of the tank
    m.fs.h2_tank.dt.fix(3600)
    m.fs.h2_tank.tank_holdup_previous.fix(0)
    m.fs.h2_tank.outlet_to_turbine.flow_mol.fix(10)
    m.fs.h2_tank.outlet_to_pipeline.flow_mol.fix(10)
    m.fs.h2_tank.outlet_to_turbine.mole_frac_comp[0, "hydrogen"].fix(1)
    m.fs.h2_tank.outlet_to_pipeline.mole_frac_comp[0, "hydrogen"].fix(1)

    propagate_state(m.fs.arc_pem_to_h2_tank)
    m.fs.h2_tank.initialize()

    # Fix the dof of the translator block
    m.fs.translator.outlet.mole_frac_comp[0, "hydrogen"].fix(0.99)
    m.fs.translator.outlet.mole_frac_comp[0, "oxygen"].fix(0.01 / 4)
    m.fs.translator.outlet.mole_frac_comp[0, "argon"].fix(0.01 / 4)
    m.fs.translator.outlet.mole_frac_comp[0, "nitrogen"].fix(0.01 / 4)
    m.fs.translator.outlet.mole_frac_comp[0, "water"].fix(0.01 / 4)

    propagate_state(m.fs.arc_h2_tank_to_translator)
    m.fs.translator.initialize()

    # Fix the degrees of freedom of mixer
    m.fs.mixer.air_feed.flow_mol[0].fix(
        m.fs.h2_tank.outlet_to_turbine.flow_mol[0].value * air_h2_ratio
    )
    m.fs.mixer.air_feed.temperature[0].fix(pem_outlet_temperature)
    m.fs.mixer.air_feed.pressure[0].fix(pem_outlet_pressure * 1e5)
    m.fs.mixer.air_feed.mole_frac_comp[0, "oxygen"].fix(0.2054)
    m.fs.mixer.air_feed.mole_frac_comp[0, "argon"].fix(0.0032)
    m.fs.mixer.air_feed.mole_frac_comp[0, "nitrogen"].fix(0.7672)
    m.fs.mixer.air_feed.mole_frac_comp[0, "water"].fix(0.0240)
    m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"].fix(2e-4)

    propagate_state(m.fs.arc_translator_to_mixer)
    m.fs.mixer.initialize()

    # Fix the degrees of freedom of H2 turbine
    m.fs.h2_turbine.compressor.deltaP.fix(compressor_dp * 1e5)
    m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)
    m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)
    m.fs.h2_turbine.turbine.deltaP.fix(-compressor_dp * 1e5)
    m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)

    propagate_state(m.fs.arc_mixer_to_h2_turbine)
    m.fs.h2_turbine.initialize()

    assert degrees_of_freedom(m) == 0
    print("    Beginning to initialize the entire flowsheet")
    res = get_solver().solve(m)
    print("    Initialization of the entire flowsheet: ",
          res.solver.termination_condition)


def unfix_dof_optimization(m, air_h2_ratio):
    """
    This function unfixes a few degrees of freedom for optimization
    """
    # Unfix the electricity split in the electrical splitter
    m.fs.np_power_split.split_fraction["np_to_grid", 0].unfix()

    # Unfix the holdup_previous and outflow variables
    m.fs.h2_tank.tank_holdup_previous.unfix()
    m.fs.h2_tank.outlet_to_turbine.flow_mol.unfix()
    m.fs.h2_tank.outlet_to_pipeline.flow_mol.unfix()

    # Unfix the flowrate of air to the mixer
    m.fs.mixer.air_feed.flow_mol.unfix()

    # Add a constraint to maintain the air to hydrogen flow ratio
    m.fs.mixer.air_h2_ratio = Constraint(
        expr=m.fs.mixer.air_feed.flow_mol[0] ==
             air_h2_ratio * m.fs.mixer.hydrogen_feed.flow_mol[0])


def set_lower_bound_on_flows(m):
    m.fs.pem.outlet.flow_mol[0].setlb(0.001)

    m.fs.h2_tank.inlet.flow_mol[0].setlb(0.001)
    m.fs.h2_tank.outlet_to_turbine.flow_mol[0].setlb(0.001)
    m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].setlb(0.001)

    m.fs.translator.inlet.flow_mol[0].setlb(0.001)
    m.fs.translator.outlet.flow_mol[0].setlb(0.001)

    m.fs.mixer.hydrogen_feed.flow_mol[0].setlb(0.001)


def build_scenario_model(m):
    # ps: Object containing the parameters and set information
    ps = m.parent_block()
    set_hours = ps.set_hours
    set_days = ps.set_days
    set_years = ps.set_years

    # Declare first-stage variables (Design decisions)
    m.pem_capacity = Var(within=NonNegativeReals,
                         doc="Maximum capacity of the PEM electrolyzer (in kW)")
    m.tank_capacity = Var(within=NonNegativeReals,
                          doc="Maximum holdup of the tank (in mol)")
    m.h2_turbine_capacity = Var(within=NonNegativeReals,
                                doc="Maximum power output from the turbine (in W)")

    m.period = Block(set_hours, set_days, set_years,
                     rule=build_ne_flowsheet)

    @m.Constraint(set_hours, set_days, set_years)
    def pem_capacity_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.pem.electricity[0] <= m.pem_capacity

    @m.Constraint(set_hours, set_days, set_years)
    def tank_capacity_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.h2_tank.tank_holdup[0] <= m.tank_capacity

    @m.Constraint(set_hours, set_days, set_years)
    def turbine_capacity_constraint(blk, t, d, y):
        return (
                - blk.period[t, d, y].fs.h2_turbine.turbine.work_mechanical[0]
                - blk.period[t, d, y].fs.h2_turbine.compressor.work_mechanical[0] <=
                m.h2_turbine_capacity
        )

    @m.Constraint(set_hours, set_days, set_years)
    def tank_holdup_constraints(blk, t, d, y):
        if t == 1:
            # Pretending that the initial holdup is zero
            return (
                    blk.period[t, d, y].fs.h2_tank.tank_holdup_previous[0] == 0
            )
        else:
            return (
                    blk.period[t, d, y].fs.h2_tank.tank_holdup_previous[0] ==
                    blk.period[t - 1, d, y].fs.h2_tank.tank_holdup[0]
            )
        
    return m


def app_costs_and_revenue(m, ps, scenario):
    """
    ps: Object containing information on sets and parameters
    """

    set_hours = ps.set_hours
    set_days = ps.set_days
    set_years = ps.set_years
    weights_days = ps.weights_days
    weights_years = ps.weights_years
    LMP = ps.LMP[scenario]

    h2_sp = ps.h2_price  # Selling price of hydrogen
    plant_life = ps.plant_life
    tax_rate = ps.tax_rate

    # Note: LHV of hydrogen is 33.3 kWh/kg, pem_capacity is in kW,
    # tank_capacity is in moles, and turbine capacity is in W
    m.capex = Expression(
        expr=(1630 * m.pem_capacity +
              (29 * 33.3 * 2.016e-3) * m.tank_capacity +
              (947 / 1000) * m.h2_turbine_capacity),
        doc="Total capital cost (in USD)"
    )

    m.fixed_om_cost = Expression(
        expr=(47.9 * m.pem_capacity +
              7e-3 * m.h2_turbine_capacity),
        doc="Fixed O&M Cost (in USD)"
    )

    # Variable O&M: PEM: $1.3/MWh and turbine: $4.25/MWh
    @m.Expression(set_years)
    def variable_om_cost(blk, y):
        return (
            (1.3 * 1e-3) * sum(weights_days[y][d] * blk.period[t, d, y].fs.pem.electricity[0]
                               for t in set_hours for d in set_days) +
            (4.25 * 1e-6) * sum(weights_days[y][d] * (
                                - blk.period[t, d, y].fs.h2_turbine.turbine.work_mechanical[0]
                                - blk.period[t, d, y].fs.h2_turbine.compressor.work_mechanical[0])
                                for t in set_hours for d in set_days)
        )

    @m.Expression(set_years)
    def electricity_revenue(blk, y):
        return (
            sum(weights_days[y][d] * LMP[y][d][t] *
                (blk.period[t, d, y].fs.np_power_split.np_to_grid_port.electricity[0] * 1e-3 -
                 blk.period[t, d, y].fs.h2_turbine.turbine.work_mechanical[0] * 1e-6 -
                 blk.period[t, d, y].fs.h2_turbine.compressor.work_mechanical[0] * 1e-6)
                for t in set_hours for d in set_days)
        )

    @m.Expression(set_years)
    def h2_revenue(blk, y):
        return (
            h2_sp * 2.016e-3 * 3600 *
            sum(weights_days[y][d] *
                blk.period[t, d, y].fs.h2_tank.outlet_to_pipeline.flow_mol[0]
                for t in set_hours for d in set_days)
        )

    @m.Expression(set_years)
    def depreciation(blk, y):
        return (
            blk.capex / plant_life
        )

    @m.Expression(set_years)
    def net_profit(blk, y):
        return (
            blk.depreciation[y] + (1 - tax_rate) * (+ blk.h2_revenue[y]
                                                    + blk.electricity_revenue[y]
                                                    - blk.fixed_om_cost
                                                    - blk.variable_om_cost[y]
                                                    - blk.depreciation[y])
        )

    m.npv = Expression(
        expr=sum(weights_years[y] * m.net_profit[y] for y in set_years) - m.capex
    )


def build_stochastic_program(m):
    # Declare first-stage variables (Design decisions)
    m.pem_capacity = Var(within=NonNegativeReals,
                         doc="Maximum capacity of the PEM electrolyzer (in kW)")
    m.tank_capacity = Var(within=NonNegativeReals,
                          doc="Maximum holdup of the tank (in mol)")
    m.h2_turbine_capacity = Var(within=NonNegativeReals,
                                doc="Maximum power output from the turbine (in W)")

    # Build the model for all scenarios
    m.scenarios = Block(m.set_scenarios, rule=build_scenario_model)
    
    # Append cash flows for each scenario
    for s1 in m.set_scenarios:
        app_costs_and_revenue(m.scenarios[s1], m, scenario=s1)

    # Add non-anticipativity constraints
    @m.Constraint(m.set_scenarios)
    def non_anticipativity_pem(blk, s):
        return blk.pem_capacity == blk.scenarios[s].pem_capacity

    @m.Constraint(m.set_scenarios)
    def non_anticipativity_tank(blk, s):
        return blk.tank_capacity == blk.scenarios[s].tank_capacity

    @m.Constraint(m.set_scenarios)
    def non_anticipativity_turbine(blk, s):
        return blk.h2_turbine_capacity == blk.scenarios[s].h2_turbine_capacity


def initialize_model(m):
    blk = ConcreteModel()
    build_ne_flowsheet(blk)
    fix_dof_and_initialize(blk)
    to_json(blk, fname="ne_flowsheet_solution.json")

    for s, y, d, t in product(m.set_scenarios, m.set_years, m.set_days, m.set_hours):
        from_json(m.scenarios[s].period[t, d, y], fname="ne_flowsheet_solution.json")

        # Unfix a few degrees of freedom for optimization
        unfix_dof_optimization(m.scenarios[s].period[t, d, y], air_h2_ratio=10.76)

        # Set non-zero lower bound on flows. Not needed if Ideal thermodynamic
        # package is used. Adding it as a precautionary measure to avoid convergence issues
        set_lower_bound_on_flows(m.scenarios[s].period[t, d, y])

        # Set the demand constraint
        if m.h2_demand is not None:
            # m.scenarios[s].period[t, d, y].fs.h2_tank.outlet_to_pipeline.flow_mol.fix(m.h2_demand)
            m.scenarios[s].period[t, d, y].fs.h2_tank.outlet_to_pipeline.flow_mol.setub(m.h2_demand)


def append_objective_function(m):
    m.expectation_npv = Objective(
        expr=sum(m.weights_scenarios[s] * m.scenarios[s].npv for s in m.set_scenarios),
        sense=maximize
    )
    
    
def generate_plots(m, s, y, d):
    """
    This function generates plots for a given scenario (s),
    year (y), and cluster (d)
    """
    # Plot the results
    time_instances_1 = []
    time_instances_2 = []
    lmp_price = []
    power_schedule = []
    h2_prod = []
    h2_tank_holdup = []
    h2_turbine_power = []
    h2_to_pipeline = []

    for t in m.set_hours:
        blk = m.scenarios[s].period[t, d, y].fs

        time_instances_1.extend([t - 1, t])
        lmp_price.extend([m.LMP[s][y][d][t], m.LMP[s][y][d][t]])
        power_schedule.extend(
            [blk.np_power_split.np_to_grid_port.electricity[0].value / 1000, 
             blk.np_power_split.np_to_grid_port.electricity[0].value / 1000])
        h2_prod.extend(
            [blk.pem.outlet.flow_mol[0].value * 3600 * 2.016e-3,
             blk.pem.outlet.flow_mol[0].value * 3600 * 2.016e-3])

        time_instances_2.append(t)
        h2_tank_holdup.append(blk.h2_tank.tank_holdup[0].value * 2.016e-3)

        h2_turbine_power.extend([
            - blk.h2_turbine.turbine.work_mechanical[0].value * 1e-6
            - blk.h2_turbine.compressor.work_mechanical[0].value * 1e-6,
            - blk.h2_turbine.turbine.work_mechanical[0].value * 1e-6
            - blk.h2_turbine.compressor.work_mechanical[0].value * 1e-6])
        h2_to_pipeline.extend(
            [blk.h2_tank.outlet_to_pipeline.flow_mol[0].value * 3600 * 2.016e-3,
             blk.h2_tank.outlet_to_pipeline.flow_mol[0].value * 3600 * 2.016e-3])

    fig, ax = plt.subplots(2, 2)

    # instantiate a second axes that shares the same x-axis
    ax1 = ax[0, 0].twinx()
    ax2 = ax[0, 1].twinx()
    ax3 = ax[1, 0].twinx()
    ax4 = ax[1, 1].twinx()

    color = 'tab:red'
    ax[0, 0].set_xlabel('time (hr)')
    ax[0, 0].set_ylabel('LMP ($/MWh)', color=color)
    ax[0, 0].plot(time_instances_1, lmp_price, color=color)
    ax[0, 0].tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax1.set_ylabel('NP to grid (MW)', color=color)
    ax1.plot(time_instances_1, power_schedule, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(500, 1005)

    color = 'tab:red'
    ax[0, 1].set_xlabel('time (hr)')
    ax[0, 1].set_ylabel('LMP ($/MWh)', color=color)
    ax[0, 1].plot(time_instances_1, lmp_price, color=color)
    ax[0, 1].tick_params(axis='y', labelcolor=color)

    color = 'magenta'
    ax2.set_ylabel('H2 to pipeline (kg/hr)', color=color)
    ax2.plot(time_instances_1, h2_to_pipeline, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'tab:red'
    ax[1, 0].set_xlabel('time (hr)')
    ax[1, 0].set_ylabel('LMP ($/MWh)', color=color)
    ax[1, 0].plot(time_instances_1, lmp_price, color=color)
    ax[1, 0].tick_params(axis='y', labelcolor=color)

    color = 'tab:green'
    ax3.set_ylabel('H2 production (kg/hr)', color=color)
    ax3.plot(time_instances_1, h2_prod, color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_ylim(0, 8000)

    color = 'tab:red'
    ax[1, 1].set_xlabel('time (hr)')
    ax[1, 1].set_ylabel('LMP ($/MWh)', color=color)
    ax[1, 1].plot(time_instances_1, lmp_price, color=color)
    ax[1, 1].tick_params(axis='y', labelcolor=color)

    color = 'tab:cyan'
    ax4.set_ylabel('H2 Turbine Power (MW)', color=color)
    ax4.plot(time_instances_1, h2_turbine_power, color=color)
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.set_ylim(-0.5, 10)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    fig_1, ax_1 = plt.subplots()

    # instantiate a second axes that shares the same x-axis
    ax1_1 = ax_1.twinx()

    color = 'tab:red'
    ax_1.set_xlabel('time (hr)')
    ax_1.set_ylabel('LMP ($/MWh)', color=color)
    ax_1.plot(time_instances_1, lmp_price, color=color)
    ax_1.tick_params(axis='y', labelcolor=color)

    color = 'darkgoldenrod'
    ax1_1.set_ylabel('Tank holdup (kg)', color=color)
    ax1_1.plot(time_instances_2, h2_tank_holdup, color=color)
    ax1_1.tick_params(axis='y', labelcolor=color)

    fig_1.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def write_results_to_file(m, filename):
    results = {s: {y: {d: {t: {}
                           for t in m.set_hours}
                       for d in m.set_days}
                   for y in m.set_years}
               for s in m.set_scenarios}
    
    for s in m.set_scenarios:
        for y in m.set_years:
            for d in m.set_days:
                for t in m.set_hours:
                    blk = m.scenarios[s].period[t, d, y].fs
                    obj = results[s][y][d][t]
                    
                    obj["np_grid"] = blk.np_power_split.np_to_grid_port.electricity[0].value / 1000
                    obj["np_to_electrolyzer"] = blk.pem.electricity[0].value / 1000
                    obj["h2_production"] = blk.pem.outlet.flow_mol[0].value * 3600 * 2.016e-3
                    obj["tank_holdup"] = blk.h2_tank.tank_holdup[0].value * 2.016e-3
                    obj["h2_to_pipeline"] = blk.h2_tank.outlet_to_pipeline.flow_mol[0].value * 3600 * 2.016e-3
                    obj["h2_to_turbine"] = blk.h2_tank.outlet_to_turbine.flow_mol[0].value * 3600 * 2.016e-3
                    obj["h2_turbine_power"] = (-blk.h2_turbine.compressor.work_mechanical[0].value * 1e-6
                                               - blk.h2_turbine.turbine.work_mechanical[0].value * 1e-6)
                    
    with open(filename, 'w') as fp:
        json.dump(results, fp, indent=4)


def build_and_solve_problem(h2_price, h2_demand):
    mdl = ConcreteModel()

    # Price of hydrogen: $2 per kg
    mdl.h2_price = h2_price

    # Hydrogen demand (in kg/s)
    mdl.h2_demand = h2_demand / 2.016e-3

    # Append LMP signal
    # append_lmp_signal(mdl,
    #                   signal_source="ARPA_E",
    #                   signal_name="MiNg_$100_CAISO")
    append_raven_lmp_signal(mdl,
                            scenarios=[0],
                            years=[2021],
                            plant_life=30,
                            discount_rate=0.08,
                            tax_rate=0.2)

    # Build the two-stage stochastic program
    timer = TicTocTimer()
    timer.tic("Starting to build the stochastic program!")
    build_stochastic_program(mdl)
    timer.toc("Built the stochastic program")

    # Initialize the model
    initialize_model(mdl)
    timer.toc("Initialized the model")

    # Append the objective function
    append_objective_function(mdl)
    timer.toc("Appended the objective function")

    get_solver().solve(mdl, tee=True)
    timer.toc("Solved the optimization problem")

    # print("Revenue from electricity: $M ", mdl.electricity_revenue.expr() / 1e6)
    # print("Revenue from hydrogen   : $M ", mdl.h2_revenue.expr() / 1e6)
    print("NPV                     : $M ", mdl.expectation_npv.expr() / 1e6)

    print("PEM Capacity            : ", mdl.pem_capacity.value * 1e-3, " MW")
    print("Tank Capacity           : ", mdl.tank_capacity.value * 2.016e-3, " kg")
    print("H2 Turbine Capacity     : ", mdl.h2_turbine_capacity.value * 1e-6, " MW")
    
    # for i in range(1, 21):
    #     generate_plots(mdl, 0, 2021, i)
    #
    # write_results_to_file(mdl, "results_var_demand_" + str(h2_demand * 10) + "_price_" + str(h2_price) + ".json")

    return (mdl.expectation_npv.expr() / 1e6,  # NPV
            mdl.pem_capacity.value * 1e-3,  # PEM Capacity
            mdl.tank_capacity.value * 2.016e-3,  # Tank capacity
            mdl.h2_turbine_capacity.value * 1e-6)  # Turbine capacity


if __name__ == "__main__":
    results = {}
    for demand in [0.5, 1, 2]:
        for price in [2, 4, 8]:
            npv, pem, tank, turbine = build_and_solve_problem(h2_price=price,
                                                              h2_demand=demand)

            results[demand, price] = {"npv": npv,
                                      "pem_capacity": pem,
                                      "tank_capacity": tank,
                                      "turbine_capacity": turbine}

            with open("results_summary.json", 'w') as fp:
                json.dump(results, fp, indent=4)


