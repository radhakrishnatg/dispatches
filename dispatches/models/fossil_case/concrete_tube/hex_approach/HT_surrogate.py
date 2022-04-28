# Import Property Packages (IAPWS95 for Water/Steam)
from idaes.generic_models.properties import iapws95
import json
import math
from smooth_piecewise_linear import get_break_points
from pyomo.environ import ConcreteModel, value


def build_prop(pressure):
    
    m = ConcreteModel()
    m.thermo_params = iapws95.Iapws95ParameterBlock()
    m.state = m.thermo_params.build_state_block(
        default={"defined_state": True,
                 "has_phase_equilibrium": True})

    m.state.pressure.fix(pressure)
    m.state.flow_mol.fix(1)
    return m


def get_HT_surrogate_data(pressure, enth_lb=2000, enth_ub=70000,
                          generate_plots=False):
    try:
        # Check if the data already exists in the data bank
        with open('HT_surrogate_data_library.json') as fp:
            ht_data = json.load(fp)

        if str(int(pressure)) in ht_data:
            return ht_data[str(int(pressure))]

    except FileNotFoundError:
        # Library of H-T data doesn't exist
        pass

    P_CR = 22064000  # Critical pressure
    m = build_prop(pressure)

    enth_range = [i for i in range(int(enth_lb), int(enth_ub) + 1, 50)]
    data_train = []
    brk_pts = []

    if pressure < P_CR:
        # Two phase region: Ensure hl_sat and hv_sat are in the data set
        hl_sat = m.state.enth_mol_sat_phase["Liq"].expr()
        hv_sat = m.state.enth_mol_sat_phase["Vap"].expr()
        T_sat = value(m.state.temperature_crit / m.state.tau_sat)
        
        if hl_sat not in enth_range:
            enth_range.append(hl_sat)
        if hv_sat not in enth_range:
            enth_range.append(hv_sat)
            
        enth_range.sort()
        # To prevent zero slope during phase change, we are increasing the
        # temperature of the vapor by 0.001 K
        brk_pts.extend([(hl_sat, T_sat), (hv_sat, T_sat + 0.001)])

    for enth in enth_range:
        m.state.enth_mol.fix(enth)
        temperature = m.state.temperature.expr()

        data_train.append((enth, temperature))

        # if math.isnan(temperature):
        #     data_train.append((enth, temperature))
        # elif len(data_train) == 0:
        #     data_train.append((enth, temperature))
        # elif temperature > data_train[-1][1]:
        #     data_train.append((enth, temperature))
        # else:
        #     # External function returned an incorrect value of temperature
        #     data_train.append((enth, float("nan")))

    flag = True
    while flag:
        if math.isnan(data_train[-1][1]):
            # If the last element temperature is nan, discard it
            data_train.pop(-1)

        else:
            flag = False

    try:
        # Check if the json file already exists
        with open('HT_surrogate_data_library.json') as fp:
            ht_data = json.load(fp)

    except FileNotFoundError:
        ht_data = {}

    # ht_data = {}
    ht_data[int(pressure)] = get_break_points(data_train, tol=3, brk_pts=brk_pts,
                                              generate_plots=generate_plots)

    with open('HT_surrogate_data_library.json', 'w') as fp:
        json.dump(ht_data, fp, indent=4)

    return ht_data[int(pressure)]
