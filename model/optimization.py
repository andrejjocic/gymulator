import pygad
# import pygad.visualize
# import pygad.utils
# import pygad.helper

from gym_model import Gym, Equipment
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

    def instantiate(self, machines: List[Equipment]) -> np.ndarray:
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
        """machines along the walls + isles (width 2) in the middle"""
        raise NotImplementedError()



def optimal_gym(layout_template: LayoutTemplate, n_generations=50, simulation_cycle_steps=1000, **gym_kwargs) -> Gym:
    """Optimize the layout of a gym, constrained by the layout template.
    Returns the best layout found, and its fitness value (higher is better)."""

    def gym_quality(ga_instance, solution, solution_idx) -> Tuple[float, ...]:
        """multi-objective fitness function for pygad (will be maximized)"""
        # print(solution)
        layout = layout_template.instantiate(machines=[Equipment(i) for i in solution])
        gym = Gym(layout=layout, spawn_location=layout_template.entrance, **gym_kwargs)
        
        metrics = gym.run(simulation_cycle_steps)
        avg_metrics = metrics.mean() # take average of columns (across time steps)
        
        return (avg_metrics["Utilization"], 1 / avg_metrics["Congestion"]) # NOTE: better to just negate?
        # TODO: use mesa.batch_run for stat. significant cost function results (if there is any randomness in gym model)
    
    def after_generation(ga_instance):
        print(f"Finished generation {ga_instance.generations_completed}/{n_generations}")
        util, inv_cong = ga_instance.best_solution()[1]
        print(f"Best fitness: util={util:.2f}, 1/cong={inv_cong:.2f}")
        # print("Pareto fronts (for multi-objective optimization):", ga_instance.pareto_fronts)
        print()


    min_machine = 0
    max_machine = len(Equipment) - 1

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
        # parallel_processing=["thread", None], # use default number of threads for fitness function
        # parallel_processing=["process", None], # use default number of processes for fitness function (preferred to threads?)
        
        ### TODO: experiment with below parameters
        num_generations=n_generations,
        sol_per_pop=4,
        
        parent_selection_type="sss", 
        keep_elitism=1, # best k solutions kept in next gen (fitness values are cached)
        keep_parents=-1, # -1 to keep all, 0 disables fitness caching! (only has effect if keep_elitism=0)
        
        num_parents_mating=2,
        crossover_type="single_point", 

        mutation_type="random", 
    )
    # TODO: sync random seed between GA and gyms??
    # print("exploring gene space:", ga_instance.gene_space_unpacked)
    ga_instance.summary()

    ga_instance.run()
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")
    ga_instance.plot_fitness(label=["utilization", "1/congestion"])

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_layout = layout_template.instantiate(machines=[Equipment(i) for i in solution])
    return best_layout, solution_fitness


if __name__ == "__main__":
    layout, fitness = optimal_gym(
        LayoutTemplate.circular(20, 10),
        simulation_cycle_steps=300, n_generations=10,
        interarrival_time=5, agent_exercise_duration=20
    )
    # print(layout)
