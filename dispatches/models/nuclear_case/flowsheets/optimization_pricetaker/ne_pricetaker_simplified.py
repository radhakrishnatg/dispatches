# General python imports
import matplotlib.pyplot as plt

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

# IDAES imports
from idaes.core import FlowsheetBlock

# Import additional functions
from process_lmp_signals import append_lmp_signal


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


def build_optimization_model(m):

    # Declare first-stage variables
    m.pem_capacity = Var(within=NonNegativeReals,
                         doc="Maximum capacity of the PEM electrolyzer (in MW)")
    m.tank_capacity = Var(within=NonNegativeReals,
                          doc="Maximum holdup of the tank (in kg)")
    m.h2_turbine_capacity = Var(within=NonNegativeReals,
                                doc="Maximum power output from the turbine (in MW)")

    # Append second-stage decisions and constraints
    m.sce = Block(m.set_hours, m.set_days, rule=build_ne_flowsheet)

    # Tank holdup relations
    @m.Constraint(m.set_hours, m.set_days)
    def tank_holdup_relation(blk, t, d):
        if t == 1 and d == 1:
            # Pretending that the initial holdup is zero
            return (
                    blk.sce[t, d].fs.tank_holdup_previous == 0
            )
        elif t == 1:
            # Holdup at the beginning of a day is the same as the
            # holdup at the end of the previous day
            return (
                    blk.sce[t, d].fs.tank_holdup_previous ==
                    blk.sce[24, d - 1].fs.tank_holdup
            )
        else:
            return (
                    blk.sce[t, d].fs.tank_holdup_previous ==
                    blk.sce[t - 1, d].fs.tank_holdup
            )

    @m.Constraint(m.set_hours, m.set_days)
    def pem_capacity_constraint(blk, t, d):
        return blk.sce[t, d].fs.np_to_electrolyzer <= blk.pem_capacity

    @m.Constraint(m.set_hours, m.set_days)
    def tank_capacity_constraint(blk, t, d):
        return blk.sce[t, d].fs.tank_holdup <= blk.tank_capacity

    @m.Constraint(m.set_hours, m.set_days)
    def turbine_capacity_constraint(blk, t, d):
        return blk.sce[t, d].fs.h2_turbine_power <= blk.h2_turbine_capacity

    # H2 market
    for t1 in m.set_hours:
        for d1 in m.set_days:
            m.sce[t1, d1].fs.h2_to_pipeline.fix(m.h2_market)

    return


def append_costs_and_revenue(m, plant_life=30):
    # Note: LHV of hydrogen is 33.3 kWh/kg
    m.capex = Expression(
        expr=(1.63 * m.pem_capacity +
              (29 * 33.3 * 1e-6) * m.tank_capacity +
              0.947 * m.h2_turbine_capacity) / plant_life,
        doc="Total capital cost (in million USD)"
    )

    m.fixed_om_cost = Expression(
        expr=(0.0479 * m.pem_capacity +
              0.007 * m.h2_turbine_capacity),
        doc="Fixed O&M Cost (in million USD)"
    )

    # Variable O&M: PEM: $1.3/MWh and turbine: $4.25/MWh
    m.variable_om_cost = Expression(
        expr=1.3 * sum(m.weights_days[d] * m.sce[t, d].fs.np_to_electrolyzer
                       for t in m.set_hours for d in m.set_days) +
             4.25 * sum(m.weights_days[d] * m.sce[t, d].fs.h2_turbine_power
                        for t in m.set_hours for d in m.set_days)
    )

    m.electricity_revenue = Expression(
        expr=sum(m.weights_days[d] * m.LMP[t, d] *
                 m.sce[t, d].fs.net_power
                 for t in m.set_hours for d in m.set_days)
    )

    m.h2_revenue = Expression(
        expr=m.h2_price *
             sum(m.weights_days[d] *
                 m.sce[t, d].fs.h2_to_pipeline
                 for t in m.set_hours for d in m.set_days)
    )


def append_objective_function(m):
    m.net_revenue = Objective(
        expr=m.electricity_revenue + m.h2_revenue - m.variable_om_cost
             - 1e6 * (m.capex + m.fixed_om_cost),
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
            blk = m.sce[t, d].fs

            time_instances_1.extend([(d - 1) * 24 + t - 1, (d - 1) * 24 + t])
            lmp_price.extend([m.LMP[t, d], m.LMP[t, d]])
            power_schedule.extend([blk.np_to_grid.value, blk.np_to_grid.value])
            h2_prod.extend([blk.h2_production.value, blk.h2_production.value])

            time_instances_2.append((d - 1) * 24 + t)
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


if __name__ == '__main__':
    mdl = ConcreteModel()
    
    # Price of hydrogen: $2 per kg
    mdl.h2_price = Param(initialize=2)
    
    # Flowrate of hydrogen to the pipeline in (kg/hr)
    mdl.h2_market = 1 * 3600

    # Append LMP signal
    append_lmp_signal(mdl,
                      signal_source="ARPA_E",
                      signal_name="MiNg_$100_CAISO")

    # Build the optimization model
    build_optimization_model(mdl)
    append_costs_and_revenue(mdl)
    append_objective_function(mdl)

    solver = SolverFactory("glpk")
    solver.solve(mdl, tee=True)

    # full_year_plotting(mdl)

    print("Revenue from electricity: $M ", mdl.electricity_revenue.expr() / 1e6)
    print("Revenue from hydrogen   : $M ", mdl.h2_revenue.expr() / 1e6)
    print("Net revenue             : $M ", mdl.net_revenue.expr() / 1e6)

    print("PEM Capacity            : ", mdl.pem_capacity.value, " MW")
    print("Tank Capacity           : ", mdl.tank_capacity.value, " kg")
    print("H2 Turbine Capacity     : ", mdl.h2_turbine_capacity.value, " MW")

    print("Hello!")
