import pygad
# import pygad.visualize
# import pygad.utils
# import pygad.helper

from gym_model import Gym, Equipment, machines_per_muscle, GymLayout
from gym_agent import Routine
import mesa.space as space
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from visualisation import draw_layout
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # replace 'TkAgg' with the backend of your choice


@dataclass
class LayoutTemplate:
    """ordered collection of gym equipment locations"""
    locations: List[space.Coordinate]
    gym_width: int
    gym_height: int
    entrance: space.Coordinate

    def instantiate(self, machines: List[Equipment]) -> GymLayout:
        """create a gym layout from this template"""
        assert len(machines) == len(self.locations)
        layout = np.full((self.gym_width, self.gym_height), None, dtype=Equipment)
        for machine, loc in zip(machines, self.locations):
            layout[loc] = machine

        return layout


    def __len__(self):
        return len(self.locations)


    # factory methods
    # NOTE: try to keep location spatially ordered so k-point crossover doesn't mess up the structure

    @staticmethod
    def circular(height: int, width: int) -> 'LayoutTemplate':
        """Machines along the walls, entrance at (x=width//2, y=0). Locations ordered counter-clockwise from bottom left."""
        walls = []
        
        # bottom wall
        for x in range(1, width - 1):
            if x != width//2: walls.append((x, 0))
        
        # right wall
        for y in range(height):
            walls.append((width-1, y))

        # top wall
        for x in range(width - 2, 0, -1):
            walls.append((x, height-1))

        # left wall
        for y in range(height - 1, -1, -1):
            walls.append((0, y))

        return LayoutTemplate(
            locations=walls,
            gym_height=height,
            gym_width=width,
            entrance=(width//2, 0),
        )
    
    @staticmethod
    def square_isles(isle_rows: int, isle_cols: int) -> 'LayoutTemplate':
        """machines along the walls + isles (2x2 squares) across the gym, entrance at bottom middle"""
        tpl = LayoutTemplate.circular(
            height=(isle_rows + 1) * 3,
            width=(isle_cols + 1) * 3
        )
        # Add isles (2x2 squares) across the gym
        for yBL in range(2, tpl.gym_height - 3, 3):
            for xBL in range(2, tpl.gym_width - 3, 3):
                tpl.locations.extend([(xBL, yBL), (xBL + 1, yBL), (xBL, yBL + 1), (xBL + 1, yBL + 1)])
        # NOTE: maybe it would be better to order isles spiralling inwards CCW?

        return tpl
    

def print_evolution_progress(ga_instance):
    _, fitness_vals, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    print(f"INFO: discarded {ga_instance.discarded_sol_count}/{ga_instance.pop_size[0]} solutions")
    print(f"Finished generation {ga_instance.generations_completed}/{ga_instance.num_generations}, max fitness={fitness_vals}") 
    ga_instance.discarded_sol_count = 0


def plot_best_layout(ga_instance):
    best_sol, fitness_vals, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    best_layout = ga_instance.gym_layout_template.instantiate(machines=[Equipment(i) for i in best_sol])
    draw_layout(best_layout, interactive=True, title=
                f"Best layout after {ga_instance.generations_completed}/{ga_instance.num_generations} gens,"
                f"fitness={[round(f, 2) for f in fitness_vals]}")
    

FITNESS_METRICS = ["Efficiency", "Utilization", "~Congestion"]

def efficiency_ratio(agent_df: pd.DataFrame) -> pd.Series:
    """proportion of time spent working out"""
    total_steps = agent_df.groupby("AgentID").size()
    steps_working_out = agent_df[agent_df['State'] == 'WORKING_OUT'].groupby('AgentID').size()

    efficiency_ratio = steps_working_out / total_steps
    efficiency_ratio.fillna(0, inplace=True) # some of them don't even lift (bruh)
    return efficiency_ratio


def gym_quality(ga_instance, solution, solution_idx) -> Tuple[float, ...]:
    """multi-objective fitness function for pygad (will be maximized)"""
    layout = ga_instance.gym_layout_template.instantiate(machines=[Equipment(i) for i in solution])
    total_machines = machines_per_muscle(layout)

    for routine in Routine:
        if not routine.muscle_groups <= total_machines:
            ga_instance.discarded_sol_count += 1
            return (-np.inf,) * len(FITNESS_METRICS) # invalid solution (not enough machines for some routine)
            # could instead cull checklist (+ fitness penalty); or prevent with custom GA functions?
    
    gym = Gym(layout=layout, spawn_location=ga_instance.gym_layout_template.entrance, **ga_instance.gym_constructor_kwargs)
    # TODO: use mesa.batch_run for stat. significant cost function results (if there is a lot of randomness in gym model)
    model_metrics, agent_metrics = gym.run(ga_instance.simulation_steps, progress_bar=False)
    time_avg = model_metrics.mean() # take average of columns (across time steps)
    return (
        efficiency_ratio(agent_metrics).mean(),
        time_avg["Utilization"],
        1 - time_avg["Congestion"] # value from 0 to 1, higher is better
    )


def optimal_gym(layout_template: LayoutTemplate,
                n_generations: int, simulation_cycle_steps: int,
                population_size=8, n_processes=0,
                crossover_method="single_point", mutation_method="random", tournament_participants: Optional[int] = None,
                parents_proportion=0.5, mutation_percents=10,
                plot_intermediate_layouts=False, **gym_kwargs
                ) -> Tuple[GymLayout, ...]:
    """
    Optimize the layout of a gym with multi-objective genetic algorithm, constrained by the layout template.
    If using parallel processing for fitness function evaluation, it probably makes sense to use a population size
    that is at least as large as the number of CPU threads (note that invalid gym layouts won't even use significant CPU time).
    - crossover_method: "single_point", "two_points", "uniform", "scattered", or None
    - mutation_method: "random", "swap", "scramble", "adaptive"
    - tournament_participants: use None for normal (non-tournament) NSGA-II selection. Greater number -> more selection pressure
    - n_processes: number of processes to use for fitness function computation. 0 for sequential, None for default (calculated by concurrent.futures)

    Returns the best layout found, and its fitness values (higher is better).
    """    
    min_machine = 0
    max_machine = len(Equipment) - 1

    ga_instance = pygad.GA(
        fitness_func=gym_quality,
        num_genes=len(layout_template),
        gene_type=int,
        gene_space=range(min_machine, max_machine + 1),
        init_range_low=min_machine, # no action if initial_population exists
        init_range_high=max_machine, # no action if initial_population exists
        mutation_by_replacement=True, # only has effect if mutation_type="random"
        random_mutation_min_val=min_machine, # only has effect if mutation_type="random"
        random_mutation_max_val=max_machine, # only has effect if mutation_type="random"
        on_generation=(plot_best_layout if plot_intermediate_layouts else print_evolution_progress),
        parallel_processing=("process", n_processes),
        num_generations=n_generations,
        
        sol_per_pop=population_size,
        num_parents_mating=round(population_size * parents_proportion),
        keep_elitism=0, # best k solutions kept in next gen (fitness values are cached)
        keep_parents=-1, # -1 to keep all, 0 disables fitness caching! (only has effect if keep_elitism=0)
        crossover_type=crossover_method, # k-point crossover probably bad for non-linear layout templates?
        
        mutation_type=mutation_method, 
        mutation_percent_genes=mutation_percents, 
        parent_selection_type=("nsga2" if tournament_participants is None else "tournament_nsga2"), 
        K_tournament=tournament_participants,
    )
    # TODO: sync random seed between GA and gyms?? https://pygad.readthedocs.io/en/latest/pygad_more.html#random-seed
    
    # attach some parameters to the GA instance, so they can be accessed in fitness function (local function not pickle-able for multiprocessing)
    for attr in ["simulation_steps", "gym_layout_template", "gym_constructor_kwargs", "discarded_sol_count"]:
        assert not hasattr(ga_instance, attr)

    ga_instance.simulation_steps = simulation_cycle_steps
    ga_instance.gym_layout_template = layout_template
    ga_instance.gym_constructor_kwargs = gym_kwargs
    ga_instance.discarded_sol_count = 0

    ga_instance.summary()
    ga_instance.run()
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")
    ga_instance.plot_fitness(label=FITNESS_METRICS)
    # print("Pareto fronts:", len(ga_instance.pareto_fronts), [f.shape for f in ga_instance.pareto_fronts]) # TODO: plot pareto fronts
    best_sol, fitness_vals, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    best_layout = layout_template.instantiate(machines=[Equipment(i) for i in best_sol])
    return best_layout, fitness_vals


if __name__ == "__main__":
    layout, fitness = optimal_gym(
        layout_template=LayoutTemplate.square_isles(isle_rows=2, isle_cols=2),
        n_generations=100, population_size=40, # large pop. probably more important than many gens (problem with lots of local optima)
        crossover_method="two_points", tournament_participants=None,
        mutation_method="random", mutation_percents=15,
        simulation_cycle_steps=250, interarrival_time=2, agent_exercise_duration=20,
        plot_intermediate_layouts=False,
    )

    print(f"Solution fitness: {fitness}")
    draw_layout(layout, title="Optimal gym layout")
    plt.show()