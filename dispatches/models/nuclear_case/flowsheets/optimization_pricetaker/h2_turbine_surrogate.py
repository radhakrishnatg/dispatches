# Pyomo imports
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           TransformationFactory,
                           value)
from pyomo.network import Arc

# IDAES imports
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import Mixer, MomentumMixingType
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.misc import get_solver
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock

# DISPATCHES imports
from dispatches.models.nuclear_case.unit_models.\
    hydrogen_turbine_unit import HydrogenTurbine
from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration as hturbine_config
import dispatches.models.nuclear_case.properties.h2_reaction \
    as h2_reaction_props


def build_h2_turbine_model(h2_flow=0,
                           air_h2_ratio=10.76,
                           h2_temperature=300,
                           h2_pressure=1.01325,
                           compressor_DP=24.01):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Add thermodynamic and reaction packages
    m.fs.h2turbine_props = GenericParameterBlock(default=hturbine_config)

    m.fs.reaction_params = h2_reaction_props.H2ReactionParameterBlock(
        default={"property_package": m.fs.h2turbine_props})

    # Add mixer block
    # using minimize pressure for all inlets and outlet of the mixer
    # because pressure of inlets is already fixed in flowsheet, using
    # equality will over-constrain
    m.fs.mixer = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.minimize,
            "property_package": m.fs.h2turbine_props,
            "inlet_list": ["air_feed", "hydrogen_feed"]}
    )

    # Hydrogen feed specifications
    m.fs.mixer.hydrogen_feed.flow_mol[0].fix(h2_flow)
    m.fs.mixer.hydrogen_feed.temperature[0].fix(h2_temperature)
    m.fs.mixer.hydrogen_feed.pressure[0].fix(h2_pressure * 1e5)
    m.fs.mixer.hydrogen_feed.mole_frac_comp[0, "hydrogen"].fix(0.99)
    m.fs.mixer.hydrogen_feed.mole_frac_comp[0, "oxygen"].fix(0.01 / 4)
    m.fs.mixer.hydrogen_feed.mole_frac_comp[0, "argon"].fix(0.01 / 4)
    m.fs.mixer.hydrogen_feed.mole_frac_comp[0, "nitrogen"].fix(0.01 / 4)
    m.fs.mixer.hydrogen_feed.mole_frac_comp[0, "water"].fix(0.01 / 4)

    # Air feed specifications
    m.fs.mixer.air_feed.flow_mol[0].fix(air_h2_ratio * h2_flow)
    m.fs.mixer.air_feed.temperature[0].fix(h2_temperature)
    m.fs.mixer.air_feed.pressure[0].fix(h2_pressure * 1e5)
    m.fs.mixer.air_feed.mole_frac_comp[0, "oxygen"].fix(0.2054)
    m.fs.mixer.air_feed.mole_frac_comp[0, "argon"].fix(0.0032)
    m.fs.mixer.air_feed.mole_frac_comp[0, "nitrogen"].fix(0.7672)
    m.fs.mixer.air_feed.mole_frac_comp[0, "water"].fix(0.0240)
    m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"].fix(2e-4)

    # Add the hydrogen turbine
    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})

    m.fs.h2_turbine.compressor.deltaP.fix(compressor_DP * 1e5)
    m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)

    # Specify the Stoichiometric Conversion Rate of hydrogen
    m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)

    m.fs.h2_turbine.turbine.deltaP.fix(-compressor_DP * 1e5)
    m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)

    m.fs.mixer_to_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    #
    print("Degrees of freedom: ", degrees_of_freedom(m.fs))

    # Initialize mixer and h2 turbine
    m.fs.mixer.initialize()
    propagate_state(m.fs.mixer_to_turbine)
    m.fs.h2_turbine.initialize()

    return m


if __name__ == '__main__':

    work = []
    mdl = build_h2_turbine_model()
    get_solver().solve(mdl, tee=True)
    work.append(abs(mdl.fs.h2_turbine.turbine.work_mechanical[0].value)
                - mdl.fs.h2_turbine.compressor.work_mechanical[0].value)

    # work = []
    # for i in range(10, 200, 10):
    #     print("Solving for H2 flow: ", i)
    #     mdl = build_h2_turbine_model(h2_flow=i)
    #     result = get_solver().solve(mdl)
    #     print(result['Solver'][0]['Termination condition'])
    #
    #     work.append(abs(mdl.fs.h2_turbine.turbine.work_mechanical[0].value)
    #                 - mdl.fs.h2_turbine.compressor.work_mechanical[0].value)

    print('hello')








