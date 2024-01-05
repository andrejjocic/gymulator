import pygad
# import pygad.visualize
# import pygad.utils
# import pygad.helper

from gym_model import Gym, Equipment, machines_per_muscle, GymLayout
from gym_agent import Routine
import mesa.space as space
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

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

    @staticmethod
    def circular(height: int, width: int) -> 'LayoutTemplate':
        """machines along the walls, entrance at (x=1, y=0)"""

        walls = []
        for x in range(width):
            if x != 1: walls.append((x, 0))
            walls.append((x, height-1))
        
        for y in range(1, height - 1):
            walls.append((0, y))
            walls.append((width-1, y))

        return LayoutTemplate(
            locations=walls,
            gym_height=height,
            gym_width=width,
            entrance=(1, 0),
        )
    
    @staticmethod
    def store_isles(height: int, width: int) -> 'LayoutTemplate':
        """machines along the walls + isles (2x2 squares?) in the middle"""
        raise NotImplementedError()
    

def after_generation(ga_instance):
    print(f"Finished generation {ga_instance.generations_completed}/{ga_instance.num_generations}")
    util, inv_cong = ga_instance.best_solution()[1]
    print(f"Best fitness: util={util:.2f}, ~cong={inv_cong:.2f}")
    print()


def gym_quality(ga_instance, solution, solution_idx) -> Tuple[float, float]:
    """multi-objective fitness function for pygad (will be maximized)"""
    layout = ga_instance.gym_layout_template.instantiate(machines=[Equipment(i) for i in solution])
    total_machines = machines_per_muscle(layout)

    for routine in Routine:
        if not routine.muscle_groups <= total_machines:
            print(f"Invalid solution: not enough machines for {routine.name} routine (fitness = -inf)")
            return (-np.inf, -np.inf) # invalid solution (not enough machines for some routine)
            # could instead cull checklist (+ fitness penalty); or prevent with custom GA functions?
    
    gym = Gym(layout=layout, spawn_location=ga_instance.gym_layout_template.entrance, **ga_instance.gym_constructor_kwargs)
    
    metrics = gym.run(ga_instance.simulation_steps)
    avg_metrics = metrics.mean() # take average of columns (across time steps)
    return (avg_metrics["Utilization"], -avg_metrics["Congestion"])
    # return (avg_metrics["Utilization"], 1 / avg_metrics["Congestion"])
    # TODO: use mesa.batch_run for stat. significant cost function results (if there is a lot of randomness in gym model)


def optimal_gym(layout_template: LayoutTemplate, n_generations: int,
                simulation_cycle_steps: int, n_processes=0, **gym_kwargs
                ) -> Tuple[GymLayout, ...]:
    """
    Optimize the layout of a gym, constrained by the layout template.
    - n_processes: number of processes to use for fitness function computation. 0 for sequential, None for default (calculated by concurrent.futures)

    Returns the best layout found, and its fitness value (higher is better).
    """    
    min_machine = 0
    max_machine = len(Equipment) - 1

    population_size = 8 # should be at least 1 for each (logical) CPU core? NOTE: invalid layouts waste resources

    ga_instance = pygad.GA(
        fitness_func=gym_quality,
        num_genes=len(layout_template),
        gene_type=int,
        gene_space=range(min_machine, max_machine + 1),
        init_range_low=min_machine,
        init_range_high=max_machine,
        mutation_by_replacement=True, # only has effect if mutation_type="random"
        random_mutation_min_val=min_machine, # only has effect if mutation_type="random"
        random_mutation_max_val=max_machine, # only has effect if mutation_type="random"
        on_generation=after_generation,
        parallel_processing=("process", n_processes),
        num_generations=n_generations,
        
        sol_per_pop=population_size,
        num_parents_mating=population_size // 2,
        keep_elitism=2, # best k solutions kept in next gen (fitness values are cached)
        keep_parents=-1, # -1 to keep all, 0 disables fitness caching! (only has effect if keep_elitism=0)
        crossover_type="single_point", # try also two_points
        
        mutation_type="random", 
        parent_selection_type="nsga2", # try also tournament_nsga2
        K_tournament=3, # parents participating in tournament (if any); greater K -> more selection pressure?
        # TODO?: custom initial pop. + crossover + mutation, making sure all routines always feasible
    )
    # TODO: sync random seed between GA and gyms?? https://pygad.readthedocs.io/en/latest/pygad_more.html#random-seed

    if (nproc := ga_instance.parallel_processing[1]) is not None and nproc > ga_instance.sol_per_pop:
        print(f"Warning: population size is smaller than number of processes") # TODO: figure out what is inferred if nproc=None
    
    # attach some parameters to the GA instance, so they can be accessed in fitness function (local function not pickle-able for multiprocessing)
    for attr in ["simulation_steps", "gym_layout_template", "gym_constructor_kwargs"]:
        assert not hasattr(ga_instance, attr)

    ga_instance.simulation_steps = simulation_cycle_steps
    ga_instance.gym_layout_template = layout_template
    ga_instance.gym_constructor_kwargs = gym_kwargs

    ga_instance.summary()
    ga_instance.run()
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")
    ga_instance.plot_fitness(label=["utilization", "~congestion"])
    # print("Pareto fronts (for multi-objective optimization):", ga_instance.pareto_fronts) # TODO: plot pareto fronts

    solution, solution_fitness, solution_idx = ga_instance.best_solution() # FIXME: seems this line re-computes fitness
    best_layout = layout_template.instantiate(machines=[Equipment(i) for i in solution])
    return best_layout, solution_fitness # TODO?: return more solutions


if __name__ == "__main__":
    layout, fitness = optimal_gym(
        layout_template=LayoutTemplate.circular(height=20, width=20),
        n_processes=2,
        simulation_cycle_steps=10, n_generations=3,
        interarrival_time=1, agent_exercise_duration=1
    )
    # print(layout)
