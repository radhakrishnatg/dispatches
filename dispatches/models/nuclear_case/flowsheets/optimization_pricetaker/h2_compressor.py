# Pyomo imports
from pyomo.environ import (ConcreteModel,
                           TransformationFactory,
                           Objective)
from pyomo.network import Arc

# IDAES imports
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import (Heater,
                                              Compressor)
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock
from idaes.power_generation.properties.natural_gas_PR import get_prop

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.util.misc import get_solver

# DISPATCHES imports
from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration as h2_ideal_config
from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration as hturbine_config


def fix_dof(m, stage):
    if stage == 1:
        # Fix the feed conditions
        m.fs.C101.inlet.flow_mol.fix(500)
        m.fs.C101.inlet.pressure.fix(20e5)
        m.fs.C101.inlet.temperature.fix(300)
        m.fs.C101.inlet.mole_frac_comp[0, "H2"].fix(1)

        # Fix the degrees of freedom of the compressor
        m.fs.C101.ratioP.fix(2.5)
        m.fs.C101.efficiency_isentropic.fix(0.65)

        # Fix the degrees of freedom of the heater
        m.fs.H101.deltaP.fix(-0.1e5)
        m.fs.H101.outlet.temperature.fix(273.15 + 40)

    elif stage == 2:
        # Fix the degrees of freedom of the compressor
        m.fs.C102.ratioP.fix(2.5)
        m.fs.C102.efficiency_isentropic.fix(0.65)

        # Fix the degrees of freedom of the heater
        m.fs.H102.deltaP.fix(-0.1e5)
        m.fs.H102.outlet.temperature.fix(273.15 + 40)

    elif stage == 3:
        # Fix the degrees of freedom of the compressor
        m.fs.C103.outlet.pressure.fix(200.1e5)
        m.fs.C103.efficiency_isentropic.fix(0.65)

        # Fix the degrees of freedom of the heater
        m.fs.H103.deltaP.fix(-0.1e5)
        m.fs.H103.outlet.temperature.fix(273.15 + 40)


def build_compression_train(m):
    # Append flowsheet block
    m.fs = FlowsheetBlock(default={'dynamic': False})

    # Append thermodynamic packages
    H2_config = get_prop(components=["H2"], phases=["Vap"])

    # Update the upper bound on pressure and the accentric factor
    # H2_config['components']['H2']['parameter_data']['omega'] = -0.05
    p_bounds = H2_config['state_bounds']['pressure']
    p_bounds = (p_bounds[0], p_bounds[1], 300e5, p_bounds[3])
    H2_config['state_bounds']['pressure'] = p_bounds
    del p_bounds

    m.fs.H2_props = GenericParameterBlock(default=H2_config)

    # Append the first compression stage
    m.fs.C101 = Compressor(default={
        "property_package": m.fs.H2_props})
    m.fs.H101 = Heater(default={
        "property_package": m.fs.H2_props,
        "has_pressure_change": True})
    m.fs.s1 = Arc(source=m.fs.C101.outlet, destination=m.fs.H101.inlet)

    # Append the second compression stage
    m.fs.C102 = Compressor(default={
        "property_package": m.fs.H2_props})
    m.fs.H102 = Heater(default={
        "property_package": m.fs.H2_props,
        "has_pressure_change": True})
    m.fs.s2 = Arc(source=m.fs.H101.outlet, destination=m.fs.C102.inlet)
    m.fs.s3 = Arc(source=m.fs.C102.outlet, destination=m.fs.H102.inlet)

    # Append the third compression stage
    m.fs.C103 = Compressor(default={
        "property_package": m.fs.H2_props})
    m.fs.H103 = Heater(default={
        "property_package": m.fs.H2_props,
        "has_pressure_change": True})
    m.fs.s4 = Arc(source=m.fs.H102.outlet, destination=m.fs.C103.inlet)
    m.fs.s5 = Arc(source=m.fs.C103.outlet, destination=m.fs.H103.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m.fs)
    fix_dof(m, stage=1)
    fix_dof(m, stage=2)
    fix_dof(m, stage=3)

    m.fs.C101.initialize()
    propagate_state(m.fs.s1)
    m.fs.H101.initialize()
    propagate_state(m.fs.s2)
    m.fs.C102.initialize()
    propagate_state(m.fs.s3)
    m.fs.H102.initialize()
    propagate_state(m.fs.s4)
    m.fs.C103.initialize()
    propagate_state(m.fs.s5)
    m.fs.H103.initialize()


if __name__ == "__main__":
    mdl = ConcreteModel()
    build_compression_train(mdl)

    assert degrees_of_freedom(mdl) == 0
    print(f"Degrees of freedom: {degrees_of_freedom(mdl)}")
    get_solver().solve(mdl)

    # Unfix a few degrees of freedom for optimization
    mdl.fs.C101.ratioP.unfix()
    mdl.fs.C102.ratioP.unfix()

    mdl.obj = Objective(
        expr=mdl.fs.C101.work_mechanical[0] +
             mdl.fs.C102.work_mechanical[0] +
             mdl.fs.C103.work_mechanical[0]
    )

    get_solver().solve(mdl, tee=True)

    print('================ STAGE 1 ================================')
    print(f"Compression work  : {mdl.fs.C101.work_mechanical[0].value / 1e6} MW")
    print(f"Outlet pressure   : {mdl.fs.C101.outlet.pressure[0].value / 1e5} bar")
    print(f"Outlet temperature: {mdl.fs.C101.outlet.temperature[0].value - 273.15} C")
    print(f"Heat Duty         : {mdl.fs.H101.heat_duty[0].value / 1e6} MW")
    print()
    print('================ STAGE 2 ================================')
    print(f"Compression work  : {mdl.fs.C102.work_mechanical[0].value / 1e6} MW")
    print(f"Outlet pressure   : {mdl.fs.C102.outlet.pressure[0].value / 1e5} bar")
    print(f"Outlet temperature: {mdl.fs.C102.outlet.temperature[0].value - 273.15} C")
    print(f"Heat Duty         : {mdl.fs.H102.heat_duty[0].value / 1e6} MW")
    print()
    print('================ STAGE 3 ================================')
    print(f"Compression work  : {mdl.fs.C103.work_mechanical[0].value / 1e6} MW")
    print(f"Outlet pressure   : {mdl.fs.C103.outlet.pressure[0].value / 1e5} bar")
    print(f"Outlet temperature: {mdl.fs.C103.outlet.temperature[0].value - 273.15} C")
    print(f"Heat Duty         : {mdl.fs.H103.heat_duty[0].value / 1e6} MW")
    print()
    total_power = (
            mdl.fs.C101.work_mechanical[0].value / 1e6 +
            mdl.fs.C102.work_mechanical[0].value / 1e6 +
            mdl.fs.C103.work_mechanical[0].value / 1e6
    )
    total_energy = total_power / (mdl.fs.C101.inlet.flow_mol[0].value * 0.0020159)
    print('=========================================================')
    print(f"Power requirement   : {total_power} MW for 500 mol/s")
    print(f"Energy requirement  : {total_energy} MJ/kg")
