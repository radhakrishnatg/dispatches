#############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform to Advance Tightly
# Coupled Hybrid Energy Systems program (DISPATCHES), and is copyright © 2021 by the software owners:
# The Regents of the University of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable Energy, LLC, Battelle
# Energy Alliance, LLC, University of Notre Dame du Lac, et al. All rights reserved.

# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the
# U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted
# for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license
# in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform
# publicly and display publicly, and to permit other to do so.
##############################################################################
"""
Nuclear Flowsheet
Author: Konor Frick based on Darice Guittet's
Date: April 20, 2021
"""
import matplotlib.pyplot as plt
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           SolverFactory,
                           TransformationFactory,
                           value)
from pyomo.network import Arc, SequentialDecomposition
from pyomo.util.check_units import assert_units_consistent
from idaes.core import FlowsheetBlock

from dispatches.models.nuclear_case.unit_models.\
    hydrogen_turbine_unit import HydrogenTurbine

from idaes.generic_models.unit_models import (Translator,
                                              Mixer,
                                              MomentumMixingType,
                                              Valve,
                                              ValveFunctionType)


from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state


from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration as h2_ideal_config
from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration as hturbine_config
import dispatches.models.nuclear_case.properties.h2_reaction \
    as h2_reaction_props

from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock

from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer
from dispatches.models.nuclear_case.unit_models.hydrogen_tank import HydrogenTank
from dispatches.models.renewables_case.elec_splitter import ElectricalSplitter

timestep_hrs = 1
H2_mass = 2.016 / 1000

PEM_temp = 300
H2_turb_pressure_bar = 24.7


def add_pem(m, outlet_pressure_bar):
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)

    m.fs.pem = PEM_Electrolyzer(
        default={"property_package": m.fs.h2ideal_props})

    # Conversion of kW to mol/sec of H2. (elec*elec_to_mol) based on H-tec design of 54.517kW-hr/kg
    m.fs.pem.electricity_to_mol.fix(0.002527406)
    m.fs.pem.outlet.pressure.fix(outlet_pressure_bar * 1e5)
    m.fs.pem.outlet.temperature.fix(PEM_temp)
    return m.fs.pem, m.fs.h2ideal_props


def add_h2_tank(m, pem_pres_bar, length_m, valve_Cv):
    m.fs.h2_tank = HydrogenTank(default={"property_package": m.fs.h2ideal_props, "dynamic": False})

    m.fs.h2_tank.tank_diameter.fix(0.1)
    m.fs.h2_tank.tank_length.fix(length_m)

    m.fs.h2_tank.dt[0].fix(timestep_hrs * 3600)
    m.fs.h2_tank.control_volume.properties_in[0].pressure.setub(1e15)
    m.fs.h2_tank.control_volume.properties_out[0].pressure.setub(1e15)
    m.fs.h2_tank.previous_state[0].pressure.setub(1e15)


    # hydrogen tank valve
    m.fs.tank_valve = Valve(
        default={
            "valve_function_callback": ValveFunctionType.linear,
            "property_package": m.fs.h2ideal_props,
            }
    )

    # connect tank to the valve
    m.fs.tank_to_valve = Arc(
        source=m.fs.h2_tank.outlet,
        destination=m.fs.tank_valve.inlet
    )

    m.fs.tank_valve.inlet.pressure[0].setub(1e15)
    # m.fs.tank_valve.outlet.pressure[0].setub(1e15)
    m.fs.tank_valve.outlet.pressure[0].fix(pem_pres_bar * 1e5)

    # NS: tuning valve's coefficient of flow to match the condition
    m.fs.tank_valve.Cv.fix(valve_Cv)
    # NS: unfixing valve opening. This allows for controlling both pressure
    # and flow at the outlet of the valve
    m.fs.tank_valve.valve_opening[0].unfix()
    m.fs.tank_valve.valve_opening[0].setlb(0)

    return m.fs.h2_tank, m.fs.tank_valve


def add_h2_turbine(m, pem_pres_bar):
    m.fs.h2turbine_props = GenericParameterBlock(default=hturbine_config)

    m.fs.reaction_params = h2_reaction_props.H2ReactionParameterBlock(
        default={"property_package": m.fs.h2turbine_props})

    # Add translator block
    m.fs.translator = Translator(
        default={"inlet_property_package": m.fs.h2ideal_props,
                 "outlet_property_package": m.fs.h2turbine_props})

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

    m.fs.translator.mole_frac_hydrogen = Constraint(
        expr=m.fs.translator.outlet.mole_frac_comp[0, "hydrogen"] == 0.99
    )
    m.fs.translator.outlet.mole_frac_comp[0, "oxygen"].fix(0.01/4)
    m.fs.translator.outlet.mole_frac_comp[0, "argon"].fix(0.01/4)
    m.fs.translator.outlet.mole_frac_comp[0, "nitrogen"].fix(0.01/4)
    m.fs.translator.outlet.mole_frac_comp[0, "water"].fix(0.01/4)

    m.fs.translator.inlet.pressure[0].setub(1e15)
    m.fs.translator.outlet.pressure[0].setub(1e15)

    # Add mixer block
    m.fs.mixer = Mixer(
        default={
    # using minimize pressure for all inlets and outlet of the mixer
    # because pressure of inlets is already fixed in flowsheet, using equality will over-constrain
            "momentum_mixing_type": MomentumMixingType.minimize,
            "property_package": m.fs.h2turbine_props,
            "inlet_list":
                ["air_feed", "hydrogen_feed"]}
    )

    m.fs.mixer.air_feed.temperature[0].fix(PEM_temp)
    m.fs.mixer.air_feed.pressure[0].fix(pem_pres_bar * 1e5)
    m.fs.mixer.air_feed.mole_frac_comp[0, "oxygen"].fix(0.2054)
    m.fs.mixer.air_feed.mole_frac_comp[0, "argon"].fix(0.0032)
    m.fs.mixer.air_feed.mole_frac_comp[0, "nitrogen"].fix(0.7672)
    m.fs.mixer.air_feed.mole_frac_comp[0, "water"].fix(0.0240)
    m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"].fix(2e-4)
    m.fs.mixer.mixed_state[0].pressure.setub(1e15)
    m.fs.mixer.air_feed_state[0].pressure.setub(1e15)
    m.fs.mixer.hydrogen_feed_state[0].pressure.setub(1e15)

    # add arcs
    m.fs.translator_to_mixer = Arc(
        source=m.fs.translator.outlet,
        destination=m.fs.mixer.hydrogen_feed
    )
    # Return early without adding Turbine
    # return None, m.fs.mixer, m.fs.translator

    # Add the hydrogen turbine
    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})

    m.fs.h2_turbine.compressor.deltaP.fix((H2_turb_pressure_bar - pem_pres_bar) * 1e5)

    m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)

    # Specify the Stoichiometric Conversion Rate of hydrogen
    # in the equation shown below
    # H2(g) + O2(g) --> H2O(g) + energy
    # Complete Combustion
    m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)

    m.fs.h2_turbine.turbine.deltaP.fix(-(H2_turb_pressure_bar - 1.01325) * 1e5)
    m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)

    m.fs.H2_production = Expression(
        expr=m.fs.pem.outlet.flow_mol[0] * H2_mass)

    m.fs.mixer_to_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    return m.fs.h2_turbine, m.fs.mixer, m.fs.translator


def create_model(pem_bar, valve_cv, tank_len_m):
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    pem, pem_properties = add_pem(m, pem_bar)

    h2_tank, tank_valve = add_h2_tank(m, pem_bar, tank_len_m, valve_cv)

    h2_turbine, h2_mixer, h2_turbine_translator = add_h2_turbine(m, pem_bar)

    # m.fs.splitter = ElectricalSplitter(default={"outlet_list": ["pem"]})

    # Set up network
    # m.fs.nuc_to_splitter = Arc(source=m.fs.nuclear_power, dest=m.fs.splitter.electricity_in)
    # m.fs.splitter_to_pem = Arc(source=m.fs.splitter.pem_port, dest=pem.electricity_in)

    if hasattr(m.fs, "h2_tank"):
        m.fs.pem_to_tank = Arc(source=pem.outlet, dest=h2_tank.inlet)

    if hasattr(m.fs, "translator"):
        m.fs.valve_to_translator = Arc(source=m.fs.tank_valve.outlet,
                                       destination=m.fs.translator.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)
    return m


def set_initial_conditions(m, tank_init_bar):

    # Fix the outlet flow to zero for tank filling type operation
    if hasattr(m.fs, "h2_tank"):
        m.fs.h2_tank.previous_state[0].temperature.fix(PEM_temp)
        m.fs.h2_tank.previous_state[0].pressure.fix(tank_init_bar * 1e5)

    return m


def update_control_vars(m, i, h2_out_mol_per_s, nuclear_offtake):

    # controlling the flow out of the tank (valve inlet is tank outlet)
    m.fs.h2_tank.outlet.flow_mol[0].fix(h2_out_mol_per_s[i])

    m.fs.nuclear_power = nuclear_offtake  # Input in kW. therefore this is a 100MW input. e.g. 100000kW

    # Units are kW; Value here is to prove 54.517 kW makes 1 kg of H2 \
    # 54.517kW*hr/kg H2 based on H-tec systems
    m.fs.pem.electricity_in.electricity.fix(m.fs.nuclear_power)

    # leaving the air feed free so bounds are respected. This results in the initialization failing at times, even if
    # the model itself will continue to solve correctly.
    # Air feed can be back-calculated for a square problem to avoid that
    # if hasattr(m.fs, "mixer"):
    #     m.fs.mixer.air_feed.flow_mol[0].fix(h2_out_mol_per_s[i] * 3)


def initialize_model(m, verbose=False):

    if verbose:
        print("=========INITIALIZING==========")

    # propagate_state(m.fs.splitter_to_pem)

    m.fs.pem.initialize()
    if verbose:
        m.fs.pem.report()

    if hasattr(m.fs, "h2_tank"):
        propagate_state(m.fs.pem_to_tank)

        m.fs.h2_tank.initialize()
        if verbose:
            m.fs.h2_tank.report()

    if hasattr(m.fs, "tank_valve"):
        propagate_state(m.fs.tank_to_valve)

        m.fs.tank_valve.initialize()
        if verbose:
            m.fs.tank_valve.report()

    if hasattr(m.fs, "translator"):
        propagate_state(m.fs.valve_to_translator)
        m.fs.translator.initialize()
        if verbose:
            m.fs.translator.report()

    if hasattr(m.fs, "mixer"):
        propagate_state(m.fs.translator_to_mixer)
        # initial guess of air feed that will be needed to balance out hydrogen feed
        h2_out = value(m.fs.h2_tank.outlet.flow_mol[0])
        m.fs.mixer.air_feed.flow_mol[0].fix(h2_out * 8)
        m.fs.mixer.initialize()
        m.fs.mixer.air_feed.flow_mol[0].unfix()
        if verbose:
            m.fs.mixer.report()

    if hasattr(m.fs, "h2_turbine"):
        propagate_state(m.fs.mixer_to_turbine)
        m.fs.h2_turbine.initialize()
        if verbose:
            m.fs.h2_turbine.report()
    return m


def update_state(m):

    m.fs.h2_tank.previous_state[0].pressure.fix(value(m.fs.h2_tank.control_volume.properties_out[0].pressure))
    m.fs.h2_tank.previous_state[0].temperature.fix(value(m.fs.h2_tank.control_volume.properties_out[0].temperature))


def run_model(pem_bar, nuclear_offtake, tank_len_m, h2_out_mol_per_s, verbose=False, plotting=False):
    valve_cv = 0.0001
    pem_in_kw = []
    tank_in_mol_per_s = []
    tank_holdup_mol = []
    tank_out_mol_per_s = []
    turbine_work_net = []

    m = create_model(pem_bar, valve_cv, tank_len_m)
    m = set_initial_conditions(m, pem_bar * 0.01)
    status_ok = True

    for i in range(0, len(h2_out_mol_per_s)):
        nuclear_offtake_now = nuclear_offtake[i]
        update_control_vars(m, i, h2_out_mol_per_s, nuclear_offtake_now)
        assert_units_consistent(m)
        m = initialize_model(m, verbose)

        if verbose:
            print("=========SOLVING==========")
            print(f"Step {i} with {degrees_of_freedom(m)} DOF")

        solver = SolverFactory('ipopt')
        res = solver.solve(m, tee=verbose)

        if verbose:
            print("#### PEM ###")
            m.fs.pem.report()
            print("#### Tank ###")
            if hasattr(m.fs, "tank_valve"):
                m.fs.h2_tank.report()
                m.fs.tank_valve.report()
            if hasattr(m.fs, "mixer"):
                print("#### Mixer ###")
                m.fs.translator.report()
                m.fs.mixer.report()
            if hasattr(m.fs, "h2_turbine"):
                print("#### Hydrogen Turbine ###")
                m.fs.h2_turbine.report()
                print(res)

        status_ok &= res.Solver.status == 'ok'
        update_state(m)

        if plotting:
            pem_in_kw.append(value(m.fs.nuclear_power))
            tank_in_mol_per_s.append(value(m.fs.h2_tank.inlet.flow_mol[0]))
            tank_out_mol_per_s.append(value(m.fs.h2_tank.outlet.flow_mol[0]))
            tank_holdup_mol.append(value(m.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]))
            turbine_work_net.append(value(m.fs.h2_turbine.turbine.work_mechanical[0]
                                          + m.fs.h2_turbine.compressor.work_mechanical[0])/1000.0)

            #combustion_chamber_pressure.append(value(m.fs.h2_turbine.stoic_reactor.pressure))
    if plotting:
        n = len(h2_out_mol_per_s) - 1
        fig, ax = plt.subplots(3, 1)

        ax[0].set_title("Hydrogen")
        ax[0].set_ylabel("Flow [mol/s]")
        ax[0].grid()
        ax[0].plot(tank_in_mol_per_s, 'g', label="Flow into tank H2")
        ax[0].plot(tank_out_mol_per_s, 'r', label="Flow out of Tank H2")
        ax[0].legend(loc="upper left")
        ax[0].set_xlim((0, n))

        ax[1].set_title("Electricity")
        ax[1].plot(pem_in_kw, 'orange', label="PEM Demand [kw]")
        ax[1].plot(turbine_work_net, 'blue', label="H2 turbine output")
        ax[1].set_ylabel("Power [kW]")
        ax[1].grid()
        ax[1].legend(loc="upper left")
        ax[1].set_xlim((0, n))

        ax[2].set_title("Hydrogen Tank")
        ax[2].plot(tank_holdup_mol, 'g', label="tank holdup [mol]")
        ax[2].set_ylabel("H2 Mols [mol]")
        ax[2].grid()
        ax[2].legend(loc="upper left")
        ax[2].set_xlim((0, n))

        plt.xlabel("Hour")
        fig.tight_layout()

        #print('pem in KW', pem_in_kw)

        plt.show()

    return status_ok, m









