# General python imports
import json
import logging
from importlib import resources
from pathlib import Path

# Pyomo imports
from pyomo.environ import (
    Objective,
    maximize,
)

# IDAES imports
from idaes.apps.grid_integration import MultiPeriodModel

# Nuclear flowsheet function imports
from dispatches.case_studies.nuclear_case.nuclear_flowsheet import (
    build_ne_flowsheet,
    fix_dof_and_initialize,
)

from dispatches.case_studies.nuclear_case.pricetaker_formulation import (
        get_linking_variable_pairs,
        unfix_dof,
        add_capacity_variables,
        append_op_costs_and_revenue,
        append_cashflows,    
    )

import parapint
from mpi4py import MPI


class StochasticData:
    def __init__(self, scenario_data, LMP, weights_days):
        self.design_vars = ["PEM_CAP", "TANK_CAP", "TURBINE_CAP"]
        self.scenarios = list(scenario_data.keys())
        self.scenario_probabilities = scenario_data
        self.LMP = LMP
        self.weights_days = weights_days


def create_scenario(prob_data, scenario):
    num_rep_days = 20
    set_years = [2022]
    h2_demand = 1

    m = MultiPeriodModel(
        n_time_points=24,
        set_days=[i for i in range(1, num_rep_days + 1)],
        set_years=set_years,
        process_model_func=build_ne_flowsheet,
        linking_variable_func=get_linking_variable_pairs,
        initialization_func=fix_dof_and_initialize,
        unfix_dof_func=unfix_dof,
        flowsheet_options={"np_capacity": 1000},
        initialization_options={
            "split_frac_grid": 0.8,
            "tank_holdup_previous": 0,
            "flow_mol_to_pipeline": 10,
            "flow_mol_to_turbine": 10,
        },
        use_stochastic_build=True,
        outlvl=logging.WARNING,
    )

    # Set initial holdup for each day
    for y in m.set_years:
        for d in m.set_days:
            m.period[1, d, y].fs.h2_tank.tank_holdup_previous.fix(0)

            for t in m.set_time:
                # m.period[t, d, y].fs.h2_tank.outlet_to_pipeline.flow_mol.setub(h2_demand / 2.016e-3)
                m.period[t, d, y].fs.h2_tank.outlet_to_pipeline.flow_mol.fix(h2_demand / 2.016e-3)

    add_capacity_variables(m)

    # Define parameters
    m.plant_life = 20                            # Plant lifetime: 20 years
    m.tax_rate = 0.2                             # Corporate tax rate: 20%
    m.discount_rate = 0.08                       # Discount rate: 8%
    m.h2_price = 3                               # Selling price of hydrogen: $3/kg

    lmp_deterministic = prob_data.LMP[scenario]
    m.weights_days = prob_data.weights_days

    for (t, d, y) in m.set_period:
        append_op_costs_and_revenue(
            m.period[t, d, y], 
            lmp=lmp_deterministic[y][d][t],
            h2_price=m.h2_price,
        )

    append_cashflows(m)

    # Define the objective function
    m.obj = Objective(expr=prob_data.scenario_probabilities[scenario] * m.npv, sense=maximize)

    return m


comm: MPI.Comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    logging.basicConfig(level=logging.INFO)


class Problem(parapint.interfaces.MPIStochasticSchurComplementInteriorPointInterface):
    def __init__(self, prob_data):
        self.prob_data = prob_data

        first_stage_var_ids = self.prob_data.design_vars

        super(Problem, self).__init__(
            scenarios=self.prob_data.scenarios,
            nonanticipative_var_identifiers=first_stage_var_ids,
            comm=comm,
        )

    def build_model_for_scenario(self, scenario_identifier):
        m = create_scenario(self.prob_data, scenario_identifier)
        first_stage_vars = {
            "PEM_CAP": m.pem_capacity,
            "TANK_CAP": m.tank_capacity,
            "TURBINE_CAP": m.h2_turbine_capacity,
        }

        return m, first_stage_vars


def main(prob_data, subproblem_solver_class, subproblem_solver_options):
    interface = Problem(prob_data=prob_data)

    linear_solver = parapint.linalg.MPISchurComplementLinearSolver(
        subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(len(prob_data.scenarios))},
        schur_complement_solver=subproblem_solver_class(**subproblem_solver_options)
    )

    options = parapint.algorithms.IPOptions()
    options.linalg.solver = linear_solver

    status = parapint.algorithms.ip_solve(interface=interface, options=options)

    assert status == parapint.algorithms.InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()

    # gather the results and plot
    if rank == 0:
        interface.pyomo_model(scenario_id=prob_data.scenarios[0]).pem_capacity.display()
        interface.pyomo_model(scenario_id=prob_data.scenarios[0]).tank_capacity.display()
        interface.pyomo_model(scenario_id=prob_data.scenarios[0]).h2_turbine_capacity.display()

    return interface


if __name__ == "__main__":
    # cntl[1] is the MA27 pivot tolerance

    with resources.path("dispatches.case_studies.nuclear_case", "lmp_signal.json") as p:
        path_to_file = Path(p).resolve()

    with open(str(path_to_file)) as fp:
        lmp_dataset = json.load(fp)

    lmp_stochastic = {
        scenario: {
            year: {
                cluster: {
                    hour: lmp_dataset[str(scenario)][str(year)][str(cluster)][str(hour)]
                    for hour in range(1, 25)
                } 
                for cluster in range(1, 21)
            }
            for year in [2022]
        }
        for scenario in [0, 1, 2]
    }

    weights_days = {
        year: {
            cluster: lmp_dataset[str(0)][str(year)][str(cluster)]["num_days"]
            for cluster in range(1, 21)
        }
        for year in [2022, 2032]
    }

    problem_data = StochasticData(
        scenario_data={i: 0.2 for i in range(3)},
        LMP=lmp_stochastic,
        weights_days=weights_days,
    )

    main(
        prob_data=problem_data,
        subproblem_solver_class=parapint.linalg.InteriorPointMA27Interface,
        subproblem_solver_options={'cntl_options': {1: 1e-6}},
    )
