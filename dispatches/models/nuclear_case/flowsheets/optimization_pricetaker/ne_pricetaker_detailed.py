# General python imports
import matplotlib.pyplot as plt
import numpy as np

# Pyomo imports
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           SolverFactory,
                           TransformationFactory,
                           maximize,
                           Block,
                           Param)
from pyomo.network import Arc, SequentialDecomposition
from pyomo.util.check_units import assert_units_consistent

# IDAES imports
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import (Translator,
                                              Mixer,
                                              MomentumMixingType,
                                              Valve,
                                              ValveFunctionType)
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
from dispatches.models.nuclear_case.unit_models.hydrogen_tank import HydrogenTank
from dispatches.models.renewables_case.elec_splitter import ElectricalSplitter

# Import additional functions
from process_lmp_signals import append_lmp_signal
from hydrogen_tank_simplified import SimpleHydrogenTank


def get_tank_dimensions(volume):
    # Assuming that the ratio of tank diameter to length is 1:2
    tank_diameter = (2 * volume / np.pi) ** (1 / 3)
    tank_length = 2 * tank_diameter

    return tank_diameter, tank_length


def build_ne_flowsheet(air_h2_ratio=10.76):
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
    m.fs.arc_mixer_to_t2_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    # Fix degrees of freedom and initialize
    fix_dof_and_initialize(m)

    # Unfix a few degrees of freedom for optimization
    unfix_dof_optimization(m, air_h2_ratio=air_h2_ratio)
    
    # Set lower bound on flows
    set_lower_bound_on_flows(m)

    return m


def fix_dof_and_initialize(m,
                           np_power_output=100 * 1e3,
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

    propagate_state(m.fs.arc_mixer_to_t2_turbine)
    m.fs.h2_turbine.initialize()


def unfix_dof_optimization(m,
                           air_h2_ratio):
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


def rule_tank_holdup(m, t, d):
    if t == 1 and d == 1:
        # Pretending that the initial holdup is zero
        return (
            m.sce[t, d].ne.fs.h2_tank.tank_holdup_previous[0] == 0
        )
    elif t == 1:
        # At the beginning of each day, compute the change in holdup
        # w.r.t the previous day
        return (
            m.sce[t, d].ne.fs.h2_tank.tank_holdup_previous[0] ==
            m.sce[24, d - 1].ne.fs.h2_tank.tank_holdup[0]
        )
    else:
        return (
            m.sce[t, d].ne.fs.h2_tank.tank_holdup_previous[0] ==
            m.sce[t - 1, d].ne.fs.h2_tank.tank_holdup[0]
        )


def build_optimization_model(m):
    m.sce = Block(m.set_hours, m.set_days)
    fs = build_ne_flowsheet()

    for t in m.set_hours:
        for d in m.set_days:
            print(f"Generating scenario for ({t}, {d})")
            m.sce[t, d].ne = fs.clone()

    m.tank_holdup_mb = Constraint(m.set_hours, m.set_days,
                                  rule=rule_tank_holdup)
    return


def append_objective_function(m):
    m.electricity_revenue = Expression(
        expr=sum(m.weights_days[d] * m.LMP[t, d] *
                 (m.sce[t, d].ne.fs.np_power_split.np_to_grid_port.electricity[0] +
                  m.sce[t, d].ne.fs.h2_turbine.turbine.work_mechanical[0] -
                  m.sce[t, d].ne.fs.h2_turbine.compressor.work_mechanical[0])
                 for t in m.set_hours for d in m.set_days)
    )

    m.h2_revenue = Expression(
        expr=m.h2_price * 2.016e-3 *
        sum(m.weights_days[d] *
            m.sce[t, d].ne.fs.h2_tank.outlet_to_pipeline.flow_mol[0]
            for t in m.set_hours for d in m.set_days)
    )

    m.net_revenue = Objective(
        expr=m.electricity_revenue + m.h2_revenue,
        sense=maximize
    )


if __name__ == '__main__':
    mdl = ConcreteModel()

    # Price of hydrogen: $2 per kg
    mdl.h2_price = Param(initialize=2)

    # Append LMP signal
    append_lmp_signal(mdl,
                      signal_source="ARPA_E",
                      signal_name="MiNg_$100_CAISO")

    # Build the optimization model
    build_optimization_model(mdl)
    append_objective_function(mdl)

    get_solver().solve(mdl, tee=True)

    print("hello!")
