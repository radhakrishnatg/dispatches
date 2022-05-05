import json
import pandas as pd

# Pyomo imports
from pyomo.environ import (ConcreteModel,
                           Block,
                           Objective,
                           RangeSet,
                           maximize,
                           Constraint,
                           value)

# IDAES imports
from idaes.core.util import from_json, to_json, get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog

# DISPATCHES imports
from scpp_concrete_tes import (build_scpc_flowsheet,
                               fix_dof_and_initialize,
                               unfix_dof_for_optimization,
                               set_scaling_factors)


def build_scenario_model(m):
    set_hours = m.set_hours
    set_days = m.set_days

    m.period = Block(set_hours, set_days)
    for t1 in m.set_hours:
        for d1 in m.set_days:
            print(f"Generating flowsheet model for {m.period[t1, d1].name}")
            build_scpc_flowsheet(m.period[t1, d1], include_concrete_tes=True)

    # Connect the initial temperature profiles
    @m.Constraint(set_hours, set_days, RangeSet(20))
    def initial_temperature_constraint(blk, t, d, s):
        if t == 1:
            return Constraint.Skip

        # FIXME: Hard-coding periods (second one) here.
        return (blk.period[t, d].fs.tes.period[1].concrete.init_temperature[s] ==
                blk.period[t - 1, d].fs.tes.period[2].concrete.temperature[s])

    return m


def build_stochastic_program(m):
    # Use this function to set up multiple scenarios
    pass


def initialize_model(m):
    # include_concrete_tes = m.period[1, 1].fs.include_concrete_tes
    # blk = ConcreteModel()
    # build_scpc_flowsheet(blk, include_concrete_tes)
    # fix_dof_and_initialize(blk)
    #
    # initialized_fs = to_json(blk, return_dict=True)
    #
    # for d in m.set_days:
    #     for t in m.set_hours:
    #         from_json(m.period[t, d], sd=initialized_fs)
    #         unfix_dof_for_optimization(m.period[t, d])

    for d in m.set_days:
        for t in m.set_hours:
            if t != 1:
                for s in m.period[t, d].fs.tes.segments:
                    m.period[t, d].fs.tes.period[1].concrete.init_temperature[s].fix(
                        m.period[t - 1, d].fs.tes.period[2].concrete.temperature[s].value)

            if m.LMP[t, d] < 16.7:
                operating_mode = "charge"
            elif m.period[t, d].fs.tes.period[1].concrete.init_temperature[20].value > 400:
                # If the initial temperature of the last segment > 400, discharge
                operating_mode = "discharge"
            else:
                operating_mode = "neutral"

            set_scaling_factors(m.period[t, d])

            fix_dof_and_initialize(m.period[t, d], outlvl=idaeslog.WARNING,
                                   operating_mode=operating_mode)
            unfix_dof_for_optimization(m.period[t, d])

    for d in m.set_days:
        m.period[1, d].fs.tes.period[1].concrete.init_temperature.fix()

    # Test the feasibility of the initial point
    for d in m.set_days:
        for t in m.set_hours:
            m.period[t, d].fs.hp_splitter.split_fraction[:, "outlet_2"].fix()
            m.period[t, d].fs.tes.period[1].tube_discharge.hex[20].inlet.flow_mol.fix()

    print("=" * 80)
    print("Testing the feasibility of the initial point.")
    print("Degrees of freedom: ", degrees_of_freedom(m))
    get_solver().solve(m, tee=True)
    print("=" * 80)

    for d in m.set_days:
        for t in m.set_hours:
            m.period[t, d].fs.hp_splitter.split_fraction[:, "outlet_2"].unfix()
            m.period[t, d].fs.tes.period[1].tube_discharge.hex[20].inlet.flow_mol.unfix()

    for d in m.set_days:
        for t in m.set_hours:
            if m.LMP[t, d] < 16.7:
                m.period[t, d].fs.tes.period[1].tube_discharge.hex[20].inlet.flow_mol.fix(1e-4)
    #         if m.LMP[t, d] > 16.7:
    #             m.period[t, d].fs.hp_splitter.split_fraction[:, "outlet_2"].fix(1e-5)


def append_objective_function(m):
    """
        This function constructs the objective function
        """
    '''
    Note: Without concrete TES unit, the net power output is 691.63 MW
          To produce this power, the main boiler and the reheater need a 
          heat duty of 1259721948.77 J/s and 260604748.89 J/s, respectively.
          Therefore, the fuel cost per hour:
              = 2.11e-9 * 3600 * (1259721948.77 + 260604748.89)
              = 11548.40
          And the fuel cost per MWh = 11548.40 / 691.63 = $16.697/MWh.
          Therefore, if the LMP is less than 16.697, set the discharge flow to 1e-5

    '''
    m.profit = Objective(
        expr=sum(m.LMP[t, d] * m.period[t, d].fs.net_power_output[0] * 1e-6 -
                 m.LMP[t, d] * m.period[t, d].fs.discharge_turbine.work_mechanical[0] * 1e-6 -
                 2.11e-9 * 3600 * m.period[t, d].fs.boiler.control_volume.heat[0] -
                 2.11e-9 * 3600 * m.period[t, d].fs.reheater.control_volume.heat[0]
                 for t in m.set_hours for d in m.set_days),
        sense=maximize
    )


if __name__ == '__main__':
    m = ConcreteModel()
    m.set_days = RangeSet(1)
    m.set_hours = RangeSet(24)

    # LMP = {(1, 1): 74.76927498,
    #        (2, 1): 69.06309832,
    #        (3, 1): 68.29710689,
    #        (4, 1): 68.17900349,
    #        (5, 1): 68.11404845,
    #        (6, 1): 68.0849794,
    #        (7, 1): 68.07800477,
    #        (8, 1): 68.11518176,
    #        (9, 1): 65.88403408,
    #        (10, 1): 55.50713288,
    #        (11, 1): 42.66843556,
    #        (12, 1): 26.72945624,
    #        (13, 1): 21.3099714,
    #        (14, 1): 24.28040398,
    #        (15, 1): 26.87753627,
    #        (16, 1): 26.8105931,
    #        (17, 1): 27.04827637,
    #        (18, 1): 25.32874319,
    #        (19, 1): 30.26194149,
    #        (20, 1): 49.58981051,
    #        (21, 1): 60.74922195,
    #        (22, 1): 70.6046699,
    #        (23, 1): 81.79030183,
    #        (24, 1): 83.20018636}

    LMP = {(1, 1): 74.76927498 - 40,
           (2, 1): 69.06309832 - 40,
           (3, 1): 68.29710689 - 40,
           (4, 1): 68.17900349 - 40,
           (5, 1): 68.11404845 - 40,
           (6, 1): 68.0849794 - 40,
           (7, 1): 68.07800477 - 40,
           (8, 1): 68.11518176 - 40,
           (9, 1): 65.88403408 - 40,
           (10, 1): 55.50713288 - 35,
           (11, 1): 42.66843556 - 22,
           (12, 1): 26.72945624 - 10,
           (13, 1): 21.3099714 - 10,
           (14, 1): 24.28040398 - 10,  # 10,
           (15, 1): 26.87753627 - 15,
           (16, 1): 26.8105931 - 15,
           (17, 1): 27.04827637 - 15,
           (18, 1): 25.32874319 - 10,
           (19, 1): 30.26194149 - 10,
           (20, 1): 49.58981051 - 20,
           (21, 1): 60.74922195 - 30,
           (22, 1): 70.6046699 - 40,
           (23, 1): 81.79030183 - 40,
           (24, 1): 83.20018636 - 40}

    m.LMP = LMP

    build_scenario_model(m)
    initialize_model(m)
    append_objective_function(m)

    print("Degrees of freedom: ", degrees_of_freedom(m))
    get_solver(options={"max_iter": 3000, "bound_push": 1e-6}).solve(m, tee=True)

    sol = {}
    for d in m.set_days:
        sol["hour"] = []
        sol["LMP"] = []
        sol[d] = []
        for t in m.set_hours:
            sol["hour"].append(t)
            sol["LMP"].append(m.LMP[t, d])
            sol[d].append((m.period[t, d].fs.net_power_output[0].value -
                           value(m.period[t, d].fs.discharge_turbine.work_mechanical[0]))
                          / 1e6)

    df_sol = pd.DataFrame(sol)
    df_sol.to_csv("multiperiod_solution.csv")
    
    conc_temp = {
        0: [m.period[1, 1].fs.tes.period[1].concrete.init_temperature[s].value
            for s in m.period[1, 1].fs.tes.segments]}
    for t in m.set_hours:
        conc_temp[t] = [m.period[t, 1].fs.tes.period[2].concrete.temperature[s].value
                        for s in m.period[t, 1].fs.tes.segments]
        
    df_conc_temp = pd.DataFrame(conc_temp)
    df_conc_temp.to_csv("concrete_temperature_profile.csv")
    
    tes_flows = {
        "charge": [m.period[t, 1].fs.hp_splitter.split_fraction[0, "outlet_2"].value
                   for t in m.set_hours],
        "discharge": [m.period[t, 1].fs.tes.period[1].tube_discharge.hex[20].inlet.flow_mol[0].value
                      for t in m.set_hours]}
    df_tes_flows = pd.DataFrame(tes_flows)
    df_tes_flows.to_csv("optimal_tes_flows.csv")

    print("End of the run!")
