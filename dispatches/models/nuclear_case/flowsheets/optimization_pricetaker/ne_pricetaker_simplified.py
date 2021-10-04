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


def build_ne_flowsheet():
    m = ConcreteModel()
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

    # Fix power production from NE plant
    m.fs.np_power.fix(100)

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


def rule_tank_holdup(m, t, d):
    if t == 1 and d == 1:
        # Pretending that the initial holdup is zero
        return (
            m.sce[t, d].ne.fs.tank_holdup_previous == 0
        )
    elif t == 1:
        # At the beginning of each day, compute the change in holdup
        # w.r.t the previous day
        return (
            m.sce[t, d].ne.fs.tank_holdup_previous ==
            m.sce[24, d - 1].ne.fs.tank_holdup
        )
    else:
        return (
            m.sce[t, d].ne.fs.tank_holdup_previous ==
            m.sce[t - 1, d].ne.fs.tank_holdup
        )


def build_optimization_model(m):
    m.sce = Block(m.set_hours, m.set_days)

    for t in m.set_hours:
        for d in m.set_days:
            m.sce[t, d].ne = build_ne_flowsheet()
            
    m.tank_holdup_mb = Constraint(m.set_hours, m.set_days,
                                  rule=rule_tank_holdup)

    return


def append_objective_function(m):
    m.electricity_revenue = Expression(
        expr=sum(m.weights_days[d] * m.LMP[t, d] *
                 m.sce[t, d].ne.fs.net_power
                 for t in m.set_hours for d in m.set_days)
    )

    m.h2_revenue = Expression(
        expr=m.h2_price *
        sum(m.weights_days[d] *
            m.sce[t, d].ne.fs.h2_to_pipeline
            for t in m.set_hours for d in m.set_days)
    )

    m.net_revenue = Objective(
        expr=m.electricity_revenue + m.h2_revenue,
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

    for d in m.set_days:
        for t in m.set_hours:
            time_instances_1.extend([(d - 1) * 24 + t - 1, (d - 1) * 24 + t])
            lmp_price.extend([m.LMP[t, d], m.LMP[t, d]])
            power_schedule.extend([m.sce[t, d].ne.fs.np_to_grid.value,
                                   m.sce[t, d].ne.fs.np_to_grid.value])
            h2_prod.extend([m.sce[t, d].ne.fs.h2_production.value,
                            m.sce[t, d].ne.fs.h2_production.value])

            time_instances_2.append((d - 1) * 24 + t)
            h2_tank_holdup.append(m.sce[t, d].ne.fs.tank_holdup.value)

    fig, (ax1, ax3) = plt.subplots(1, 2)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()

    color = 'tab:red'
    ax1.set_xlabel('time (hr)')
    ax1.set_ylabel('LMP ($/MWh)', color=color)
    ax1.plot(time_instances_1, lmp_price, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2.set_ylabel('NP to grid (MW)', color=color)
    ax2.plot(time_instances_1, power_schedule, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 105)

    color = 'tab:red'
    ax3.set_xlabel('time (hr)')
    ax3.set_ylabel('LMP ($/MWh)', color=color)
    ax3.plot(time_instances_1, lmp_price, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    color = 'tab:green'
    ax4.set_ylabel('H2 production (kg/hr)', color=color)
    ax4.plot(time_instances_1, h2_prod, color=color)
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.set_ylim(0, 1900)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == '__main__':
    mdl = ConcreteModel()
    
    # Price of hydrogen: $2 per kg
    mdl.h2_price = Param(initialize=2)

    # Append LMP signal
    append_lmp_signal(mdl,
                      signal_source="RTS_GMLC",
                      signal_name="MiNg_$100_CAISO")

    # Build the optimization model
    build_optimization_model(mdl)
    append_objective_function(mdl)

    solver = SolverFactory("glpk")
    solver.solve(mdl, tee=True)

    full_year_plotting(mdl)

    print("Hello!")
