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
        """???"""
        raise NotImplementedError()



def optimize_gym(layout_template: LayoutTemplate, steps_per_run=1000, **gym_kwargs) -> Gym:
    """optimize the layout of a gym, constrained by the layout template"""

    def gym_quality(ga_instance, solution, solution_idx) -> Tuple[float, ...]:
        """multi-objective fitness function for pygad (will be maximized)"""
        # print(solution)
        layout = layout_template.instantiate(machines=[Equipment(i) for i in solution])
        gym = Gym(layout=layout, spawn_location=layout_template.entrance, **gym_kwargs)
        
        metrics = gym.run(steps_per_run)
        avg_metrics = metrics.mean() # take average of columns (across time steps)
        
        return (avg_metrics["Utilization"], 1 / avg_metrics["Congestion"])
        # TODO: use mesa.batch_run for staticstically significant cost function results
    
    def on_gen(ga_instance):
        print("Generation : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", ga_instance.best_solution()[1])


    ga_instance = pygad.GA(
        fitness_func=gym_quality,
        num_genes=len(layout_template),
        gene_space=range(len(Equipment)),
        # set random seed to the gym's seed?
        num_generations=3, #50,
        num_parents_mating=4,
        sol_per_pop=8,
        on_generation=on_gen,
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_layout = layout_template.instantiate(machines=[Equipment(i) for i in solution])
    return best_layout, solution_fitness

if __name__ == "__main__":
    layout, fitness = optimize_gym(LayoutTemplate.circular(40, 20), steps_per_run=20, interarrival_time=5, agent_exercise_duration=5)
    # print(layout)
