# General python imports
import matplotlib.pyplot as plt
import time

# Pyomo imports
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           NonNegativeReals,
                           Block,
                           Param,
                           maximize,
                           SolverFactory)
from pyomo.common.timing import TicTocTimer

# IDAES imports
from idaes.core import FlowsheetBlock

# Import additional functions
from process_lmp_signals import append_lmp_signal, append_raven_lmp_signal


def build_ne_flowsheet(m):
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Declare variables
    m.fs.np_power = Var(within=NonNegativeReals,
                        doc="Power produced by the nuclear plant (MW)")
    m.fs.np_to_grid = Var(within=NonNegativeReals,
                          doc="Power from NP to the grid (MW)")
    m.fs.np_to_electrolyzer = Var(within=NonNegativeReals,
                                  doc="Power from NP to electrolyzer (MW)")
    m.fs.h2_production = Var(within=NonNegativeReals,
                             doc="Hydrogen production rate (kg/hr)")
    m.fs.tank_holdup = Var(within=NonNegativeReals,
                           doc="Hydrogen holdup in the tank (kg)")
    m.fs.tank_holdup_previous = Var(within=NonNegativeReals,
                                    doc="Hold at the beginning of the period (kg)")
    m.fs.h2_to_pipeline = Var(within=NonNegativeReals,
                              doc="Hydrogen flowrate to the pipeline (kg/hr)")
    m.fs.h2_to_turbine = Var(within=NonNegativeReals,
                             doc="Hydrogen flowrate to the turbine (kg/hr)")
    m.fs.h2_turbine_power = Var(within=NonNegativeReals,
                                doc="Power production from H2 turbine (MW)")
    m.fs.net_power = Var(within=NonNegativeReals,
                         doc="Net power to the grid (MW)")

    # Fix power production from NE plant to 1 GW
    m.fs.np_power.fix(1000)

    # Declare Constraints
    m.fs.np_power_balance = Constraint(
        expr=m.fs.np_power == m.fs.np_to_grid + m.fs.np_to_electrolyzer,
        doc="Power balance at the nuclear power plant"
    )
    # Compute the hydrogen production rate
    # H-tec design: 54.517 kW-hr/kg of hydrogen
    m.fs.calc_h2_production_rate = Constraint(
        expr=m.fs.h2_production == (1000 / 54.517) * m.fs.np_to_electrolyzer,
        doc="Computes the hydrogen production rate"
    )
    # Tank holdup calculations (Assuming a delta_t of 1 hr)
    m.fs.tank_mass_balance = Constraint(
        expr=m.fs.tank_holdup - m.fs.tank_holdup_previous ==
             (m.fs.h2_production - m.fs.h2_to_pipeline - m.fs.h2_to_turbine)
    )

    # Compute the power production via h2 turbine
    # For an air_h2_ratio of 10.76, (T, P) of h2 = (300 K, 1 atm),
    # delta_p across compressor and turbine 24.1 bar, the conversion
    # factor is 0.0125 MW-hr/kg hydrogen
    m.fs.calc_turbine_power = Constraint(
        expr=m.fs.h2_turbine_power == 0.0125 * m.fs.h2_to_turbine,
        doc="Computes the power production via h2 turbine"
    )
    # Power balance at the grid
    m.fs.grid_power_balance = Constraint(
        expr=m.fs.net_power == m.fs.np_to_grid + m.fs.h2_turbine_power
    )

    return m


def build_scenario_model(m):
    # ps: Object containing the parameters and set information
    ps = m.parent_block()
    set_hours = ps.set_hours
    set_days = ps.set_days
    set_years = ps.set_years

    # Declare first-stage variables
    m.pem_capacity = Var(within=NonNegativeReals,
                         doc="Maximum capacity of the PEM electrolyzer (in MW)")
    m.tank_capacity = Var(within=NonNegativeReals,
                          doc="Maximum holdup of the tank (in kg)")
    m.h2_turbine_capacity = Var(within=NonNegativeReals,
                                doc="Maximum power output from the turbine (in MW)")

    # Append second-stage decisions and constraints
    m.period = Block(set_hours, set_days, set_years, rule=build_ne_flowsheet)

    # Tank holdup relations
    @m.Constraint(set_hours, set_days, set_years)
    def tank_holdup_relation(blk, t, d, y):
        if t == 1:
            # Pretending that the initial holdup is zero
            return (
                    blk.period[t, d, y].fs.tank_holdup_previous == 0
            )
        else:
            return (
                    blk.period[t, d, y].fs.tank_holdup_previous ==
                    blk.period[t - 1, d, y].fs.tank_holdup
            )

    @m.Constraint(set_hours, set_days, set_years)
    def pem_capacity_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.np_to_electrolyzer <= blk.pem_capacity

    @m.Constraint(set_hours, set_days, set_years)
    def tank_capacity_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.tank_holdup <= blk.tank_capacity

    @m.Constraint(set_hours, set_days, set_years)
    def turbine_capacity_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.h2_turbine_power <= blk.h2_turbine_capacity

    # H2 market
    @m.Constraint(set_hours, set_days, set_years)
    def h2_demand_constraint(blk, t, d, y):
        return blk.period[t, d, y].fs.h2_to_pipeline == ps.h2_demand

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

    # Note: LHV of hydrogen is 33.3 kWh/kg
    m.capex = Expression(
        expr=(1630000 * m.pem_capacity +
              (29 * 33.3) * m.tank_capacity +
              947000 * m.h2_turbine_capacity),
        doc="Total capital cost (in USD)"
    )

    m.fixed_om_cost = Expression(
        expr=(47900 * m.pem_capacity +
              7000 * m.h2_turbine_capacity),
        doc="Fixed O&M Cost (in USD)"
    )

    # Variable O&M: PEM: $1.3/MWh and turbine: $4.25/MWh
    @m.Expression(set_years)
    def variable_om_cost(blk, y):
        return (
            1.3 * sum(weights_days[y][d] * blk.period[t, d, y].fs.np_to_electrolyzer
                      for t in set_hours for d in set_days) +
            4.25 * sum(weights_days[y][d] * blk.period[t, d, y].fs.h2_turbine_power
                       for t in set_hours for d in set_days)
        )

    @m.Expression(set_years)
    def electricity_revenue(blk, y):
        return (
            sum(weights_days[y][d] * LMP[y][d][t] * blk.period[t, d, y].fs.net_power
                for t in set_hours for d in set_days)
        )

    @m.Expression(set_years)
    def h2_revenue(blk, y):
        return (
            h2_sp * sum(weights_days[y][d] * blk.period[t, d, y].fs.h2_to_pipeline
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
        power_schedule.extend([blk.np_to_grid.value, blk.np_to_grid.value])
        h2_prod.extend([blk.h2_production.value, blk.h2_production.value])

        time_instances_2.append(t)
        h2_tank_holdup.append(blk.tank_holdup.value)

        h2_turbine_power.extend([blk.h2_turbine_power.value, blk.h2_turbine_power.value])
        h2_to_pipeline.extend([blk.h2_to_pipeline.value, blk.h2_to_pipeline.value])

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


def write_results_to_file(m):
    results = {s: {y: {d: {t: {"np_power": m.scenarios[s].period[t, d, y].fs.np_power.value,
                               "np_to_electrolyzer": m.scenarios[s].period[t, d, y].np_to_electrolyzer.value,
                               "h2_production": m.scenarios[s].period[t, d, y].h2_production.value,
                               "tank_holdup": m.scenarios[s].period[t, d, y].tank_holdup.value,
                               "h2_to_pipeline": m.scenarios[s].period[t, d, y].h2_to_pipeline.value,
                               "h2_to_turbine": m.scenarios[s].period[t, d, y].h2_to_turbine.value,
                               "h2_turbine_power": m.scenarios[s].period[t, d, y].h2_turbine_power.value}
                           for t in m.set_hours}
                       for d in m.set_days}
                   for y in m.set_years}
               for s in m.set_scenarios}

    with open("stochastic_results.json", 'w') as fp:
        json.dump(results, fp, indent=4)


if __name__ == '__main__':
    start = time.time()

    mdl = ConcreteModel()

    # Price of hydrogen: $2 per kg
    mdl.h2_price = 2

    # Flowrate of hydrogen to the pipeline in (kg/hr)
    mdl.h2_demand = 1 * 3600

    # Append LMP signal
    # append_lmp_signal(mdl,
    #                   signal_source="ARPA_E",
    #                   signal_name="MiNg_$100_CAISO")
    append_raven_lmp_signal(mdl,
                            scenarios=[0, 1, 2, 3, 4],
                            years=[i for i in range(2020, 2051)],
                            plant_life=31,
                            discount_rate=0.08,
                            tax_rate=0.2)

    # Build the two-stage stochastic program
    timer = TicTocTimer()
    timer.tic("Starting to build the stochastic program!")
    build_stochastic_program(mdl)
    timer.toc("Built the stochastic program")

    # Append the objective function
    append_objective_function(mdl)
    timer.toc("Appended the objective function")

    solver = SolverFactory("gurobi")
    solver.solve(mdl, tee=True)
    timer.toc("Solved the optimization problem")

    # full_year_plotting(mdl)

    # print("Revenue from electricity: $M ", mdl.electricity_revenue.expr() / 1e6)
    # print("Revenue from hydrogen   : $M ", mdl.h2_revenue.expr() / 1e6)
    # print("Net profit              : $M ", mdl.net_profit.expr() / 1e6)
    # print("Total capital cost      : $M ", mdl.capex.expr() / 1e6)
    print("Net present value       : $M ", mdl.expectation_npv.expr() / 1e6)
    print()

    print("Minimum PEM Capacity    : ", 54.517 * 1e-3 * mdl.h2_demand, " MW")
    print("PEM Capacity            : ", mdl.pem_capacity.value, " MW")
    print("Tank Capacity           : ", mdl.tank_capacity.value, " kg")
    print("H2 Turbine Capacity     : ", mdl.h2_turbine_capacity.value, " MW")

    end = time.time()
    print(f"Time taken for the run: {end - start} s")
