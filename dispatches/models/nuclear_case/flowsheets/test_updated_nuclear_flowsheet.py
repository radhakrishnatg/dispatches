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
Nuclear Flowsheet Tester
Author: Konor Frick
Date: May 11, 2021
"""
import pytest
import itertools

from .Nuclear_flowsheet_update import *


def test_h2_valve_opening():
    opening = 0.0001
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)

    h2_tank, tank_valve = add_h2_tank(m, 3, 0.3, opening)

    h2_tank.inlet.pressure.fix(8e6)
    h2_tank.inlet.temperature.fix(300)
    h2_tank.inlet.flow_mol[0] = 0
    tank_valve.outlet.pressure.fix(8e6)


def test_model():
    nuclear_offtake = [3.0, 5.0, 10.0] #KW sent to PEM.
    h2_out_mol_per_s = [0.002, 0.003, 0.002]

    ok, m = run_model(pem_bar=3, tank_len_m=0.3,  nuclear_offtake=nuclear_offtake, h2_out_mol_per_s=h2_out_mol_per_s, verbose=True, plotting=False)
    assert ok


    assert value(m.fs.h2_tank.outlet.flow_mol[0]) == 0.002
    assert value(m.fs.pem.electricity_in.electricity[0]) == 10.0
    assert value(m.fs.h2_turbine.turbine.work_mechanical[0] + m.fs.h2_turbine.compressor.work_mechanical[0]) \
           == pytest.approx(-257.86, 1e-2)

def test_model_1():
    nuclear_offtake = [10e3, 10e3, 10e3] #KW sent to PEM.
    h2_out_mol_per_s = [10.0, 15.0, 20.0]

    ok, m = run_model(pem_bar=3, tank_len_m=0.3,  nuclear_offtake=nuclear_offtake, h2_out_mol_per_s=h2_out_mol_per_s, verbose=True, plotting=False)
    assert ok

    assert value(m.fs.pem.electricity_in.electricity[0]) == 10e3
    assert value(m.fs.h2_tank.outlet.flow_mol[0]) == 20.0
    assert value(m.fs.h2_turbine.turbine.work_mechanical[0] + m.fs.h2_turbine.compressor.work_mechanical[0]) \
           == pytest.approx(-2.566e6, 1e-02)
#
#
def test_model_solves_over_range():
    nuclear_offtake = [10e3, 10e3, 10e3] #KW sent to PEM.
    h2_out_mol_per_s = [0.001, 0.003, 30.0]

    # if the above values are modified, could do a larger range than what's below
    PEM_outlet_pressure_bar = (3, 18)
    H2_tank_length_cm = range(3, 10, 3)

    for p, t in itertools.product(PEM_outlet_pressure_bar, H2_tank_length_cm):
        ok = run_model(pem_bar=p, tank_len_m=t / 10, nuclear_offtake=nuclear_offtake, h2_out_mol_per_s=h2_out_mol_per_s)[0]
        if not ok:
            print(p, t)
        assert ok