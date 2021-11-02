# General python imports
import matplotlib.pyplot as plt
import numpy as np
import json

# Pyomo imports
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           NonNegativeReals,
                           TransformationFactory,
                           maximize,
                           Block,
                           Param)
from pyomo.network import Arc

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

# DISPATCHES imports
from dispatches.models.nuclear_case.unit_models.\
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
from process_lmp_signals import append_lmp_signal
from hydrogen_tank_simplified import SimpleHydrogenTank


def get_tank_dimensions(volume):
    # Assuming that the ratio of tank diameter to length is 1:2
    tank_diameter = (2 * volume / np.pi) ** (1 / 3)
    tank_length = 2 * tank_diameter

    return tank_diameter, tank_length


def build_ne_flowsheet(pem_capacity,
                       tank_capacity,
                       h2_turbine_capacity,
                       h2_demand=None,
                       air_h2_ratio=10.76):
    m = ConcreteModel()
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

    # Fix degrees of freedom and initialize
    fix_dof_and_initialize(m)

    # Unfix a few degrees of freedom for optimization
    unfix_dof_optimization(m, air_h2_ratio=air_h2_ratio, h2_demand=h2_demand)
    
    # Set lower bound on flows
    set_lower_bound_on_flows(m)

    # Add capacity constraints
    @m.fs.Constraint(m.fs.time)
    def pem_capacity_constraint(blk, t):
        return blk.pem.electricity[t] <= pem_capacity

    @m.fs.Constraint(m.fs.time)
    def tank_capacity_constraint(blk, t):
        return blk.h2_tank.tank_holdup[t] <= tank_capacity

    @m.fs.Constraint(m.fs.time)
    def turbine_capacity_constraint(blk, t):
        return (
            - blk.h2_turbine.turbine.work_mechanical[t]
            - blk.h2_turbine.compressor.work_mechanical[t] <=
            h2_turbine_capacity
        )

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
    m.fs.translator.eq_flow_hydrogen = Constraint(
        expr=m.fs.translator.inlet.flow_mol[0] ==
        m.fs.translator.outlet.flow_mol[0]
    )

    m.fs.translator.eq_temperature = Constraint(
        expr=m.fs.translator.inlet.temperature[0] ==
        m.fs.translator.outlet.temperature[0]
    )

    m.fs.translator.eq_pressure = Constraint(
        expr=m.fs.translator.inlet.pressure[0] ==
        m.fs.translator.outlet.pressure[0]
    )

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
    get_solver().solve(m, tee=True)


def unfix_dof_optimization(m,
                           air_h2_ratio,
                           h2_demand):
    """
    This function unfixes a few degrees of freedom for optimization
    """
    # Unfix the electricity split in the electrical splitter
    m.fs.np_power_split.split_fraction["np_to_grid", 0].unfix()
    
    # Unfix the holdup_previous and outflow variables
    m.fs.h2_tank.tank_holdup_previous.unfix()
    m.fs.h2_tank.outlet_to_turbine.flow_mol.unfix()
    m.fs.h2_tank.outlet_to_pipeline.flow_mol.unfix()

    if h2_demand is not None:
        m.fs.h2_tank.outlet_to_pipeline.flow_mol.fix(h2_demand)

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


def build_optimization_model(m, h2_demand):
    # Declare first-stage variables (Design decisions)
    m.pem_capacity = Var(within=NonNegativeReals,
                         doc="Maximum capacity of the PEM electrolyzer (in kW)")
    m.tank_capacity = Var(within=NonNegativeReals,
                          doc="Maximum holdup of the tank (in mol)")
    m.h2_turbine_capacity = Var(within=NonNegativeReals,
                                doc="Maximum power output from the turbine (in W)")

    m.sce = Block(m.set_hours, m.set_days)
    fs = build_ne_flowsheet(pem_capacity=m.pem_capacity,
                            tank_capacity=m.tank_capacity,
                            h2_turbine_capacity=m.h2_turbine_capacity,
                            h2_demand=h2_demand)

    for d1 in m.set_days:
        print(f"Generating scenarios for day {d1}")

        for t1 in m.set_hours:
            m.sce[t1, d1].ne = fs.clone()

    @m.Constraint(m.set_hours, m.set_days)
    def tank_holdup_constraints(blk, t, d):
        if t == 1 and d == 1:
            # Pretending that the initial holdup is zero
            return (
                    blk.sce[t, d].ne.fs.h2_tank.tank_holdup_previous[0] == 0
            )
        elif t == 1:
            # At the beginning of each day, compute the change in holdup
            # w.r.t the previous day
            return (
                    blk.sce[t, d].ne.fs.h2_tank.tank_holdup_previous[0] ==
                    blk.sce[24, d - 1].ne.fs.h2_tank.tank_holdup[0]
            )
        else:
            return (
                    blk.sce[t, d].ne.fs.h2_tank.tank_holdup_previous[0] ==
                    blk.sce[t - 1, d].ne.fs.h2_tank.tank_holdup[0]
            )

    return


def app_costs_and_revenue(m, plant_life=30):
    # Note: LHV of hydrogen is 33.3 kWh/kg, pem_capacity is in kW,
    # tank_capacity is in moles, and turbine capacity is in W
    m.capex = Expression(
        expr=(1630 * m.pem_capacity +
              (29 * 33.3 * 2.016e-3) * m.tank_capacity +
              (947 / 1000) * m.h2_turbine_capacity) / plant_life,
        doc="Total capital cost (in USD)"
    )

    m.fixed_om_cost = Expression(
        expr=(47.9 * m.pem_capacity +
              7e-3 * m.h2_turbine_capacity),
        doc="Fixed O&M Cost (in USD)"
    )

    # Variable O&M: PEM: $1.3/MWh and turbine: $4.25/MWh
    m.variable_om_cost = Expression(
        expr=(1.3 * 1e-3) * sum(m.weights_days[d] * m.sce[t, d].ne.fs.pem.electricity[0]
                                for t in m.set_hours for d in m.set_days) +
             (4.25 * 1e-6) * sum(m.weights_days[d] * (
                                    - m.sce[t, d].ne.fs.h2_turbine.turbine.work_mechanical[0]
                                    - m.sce[t, d].ne.fs.h2_turbine.compressor.work_mechanical[0])
                                 for t in m.set_hours for d in m.set_days)
    )

    m.electricity_revenue = Expression(
        expr=sum(m.weights_days[d] * m.LMP[t, d] *
                 (m.sce[t, d].ne.fs.np_power_split.np_to_grid_port.electricity[0] * 1e-3 -
                  m.sce[t, d].ne.fs.h2_turbine.turbine.work_mechanical[0] * 1e-6 -
                  m.sce[t, d].ne.fs.h2_turbine.compressor.work_mechanical[0] * 1e-6)
                 for t in m.set_hours for d in m.set_days)
    )

    m.h2_revenue = Expression(
        expr=m.h2_price * 2.016e-3 * 3600 *
             sum(m.weights_days[d] *
                 m.sce[t, d].ne.fs.h2_tank.outlet_to_pipeline.flow_mol[0]
                 for t in m.set_hours for d in m.set_days)
    )


def append_objective_function(m):
    m.net_revenue = Objective(
        expr=m.electricity_revenue + m.h2_revenue - m.variable_om_cost
             - m.capex - m.fixed_om_cost,
        sense=maximize
    )


def full_year_plotting(m):
    # Plot the results
    time_instances_1 = []
    time_instances_2 = []
    lmp_price = []
    power_schedule = []
    h2_prod = []
    h2_tank_holdup = []
    h2_turbine_power = []
    h2_to_pipeline = []

    for d in m.set_days:
        for t in m.set_hours:
            blk = m.sce[t, d].ne.fs

            time_instances_1.extend([(d - 1) * 24 + t - 1, (d - 1) * 24 + t])
            lmp_price.extend([m.LMP[t, d], m.LMP[t, d]])
            power_schedule.extend(
                [blk.np_power_split.np_to_grid_port.electricity[0].value / 1000,
                 blk.np_power_split.np_to_grid_port.electricity[0].value / 1000])
            h2_prod.extend(
                [blk.pem.outlet.flow_mol[0].value * 3600 * 2.016e-3,
                 blk.pem.outlet.flow_mol[0].value * 3600 * 2.016e-3])

            time_instances_2.append((d - 1) * 24 + t)
            h2_tank_holdup.append(blk.h2_tank.tank_holdup[0].value * 2.016e-3)

            h2_turbine_power.extend(
                [- blk.h2_turbine.turbine.work_mechanical[0].value * 1e-6
                 - blk.h2_turbine.compressor.work_mechanical[0].value * 1e-6,
                 - blk.h2_turbine.turbine.work_mechanical[0].value * 1e-6
                 - blk.h2_turbine.compressor.work_mechanical[0].value * 1e-6])

            h2_to_pipeline.extend(
                [blk.h2_tank.outlet_to_pipeline.flow_mol[0].value * 3600 * 2.016e-3,
                 blk.h2_tank.outlet_to_pipeline.flow_mol[0].value * 3600 * 2.016e-3]
            )

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


def write_results(m):
    results = {}
    counter = 1

    for d in m.set_days:
        for t in m.set_hours:
            fs = m.sce[t, d].ne.fs

            results[counter] = {
                "np_to_grid": fs.np_power_split.np_to_grid_port.electricity[0].value / 1000,
                "np_to_pem": fs.np_power_split.np_to_pem_port.electricity[0].value / 1000,
                "h2_production": fs.pem.outlet.flow_mol[0].value * 2.016e-3 * 3600,
                "h2_to_pipeline": fs.h2_tank.outlet_to_pipeline.flow_mol[0].value * 2.016e-3 * 3600,
                "h2_to_turbine": fs.h2_tank.outlet_to_turbine.flow_mol[0].value * 2.016e-3 * 3600,
                "tank_holdup": fs.h2_tank.tank_holdup[0].value * 2.016e-3,
                "turbine_power": (- fs.h2_turbine.turbine.work_mechanical[0].value * 1e-6
                                  - fs.h2_turbine.compressor.work_mechanical[0].value * 1e-6)
            }

            counter += 1

    with open('ne_results.json', 'w') as fp:
        json.dump(results, fp, indent=4)


if __name__ == '__main__':
    mdl = ConcreteModel()

    # Price of hydrogen: $2 per kg
    mdl.h2_price = Param(initialize=2)

    # Hydrogen demand (in kg/s)
    h2_demand = 1 / 2.016e-3

    # Append LMP signal
    append_lmp_signal(mdl,
                      signal_source="ARPA_E",
                      signal_name="MiNg_$100_CAISO")

    # Build the optimization model
    build_optimization_model(mdl, h2_demand)
    app_costs_and_revenue(mdl)
    append_objective_function(mdl)

    get_solver().solve(mdl, tee=True)
    # full_year_plotting(mdl)
    
    write_results(mdl)

    print("Revenue from electricity: $M ", mdl.electricity_revenue.expr() / 1e6)
    print("Revenue from hydrogen   : $M ", mdl.h2_revenue.expr() / 1e6)
    print("Net revenue             : $M ", mdl.net_revenue.expr() / 1e6)

    print("PEM Capacity            : ", mdl.pem_capacity.value * 1e-3, " MW")
    print("Tank Capacity           : ", mdl.tank_capacity.value * 2.016e-3, " kg")
    print("H2 Turbine Capacity     : ", mdl.h2_turbine_capacity.value * 1e-6, " MW")

    print("hello!")
