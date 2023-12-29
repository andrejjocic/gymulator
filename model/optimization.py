import pygad
# import pygad.visualize
# import pygad.utils
# import pygad.helper

from gym_model import Gym, Equipment
import mesa.space as space
from typing import List
from dataclasses import dataclass
import numpy as np

@dataclass
class LayoutTemplate:
    """ordered collection of gym equipment locations"""
    locations: List[space.Coordinate]
    gym_height: int
    gym_width: int
    entrance: space.Coordinate

    def instantiate(self, machines: List[Equipment]) -> np.ndarray:
        """create a gym layout from this template"""
        assert len(machines) == len(self.locations)
        # layout = np.full((self.gym_height, self.gym_width), None, dtype=Equipment)
        layout = np.full((self.gym_width, self.gym_height), None, dtype=Equipment)
        for machine, (x, y) in zip(machines, self.locations):
            # layout[y, x] = machine
            layout[x, y] = machine

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
    """optimize the layout of a gym"""

    def gym_quality(ga_instance, solution, solution_idx) -> float:
        """fitness function for pygad (will be maximized)"""
        # print(solution)
        layout = layout_template.instantiate(machines=[Equipment(i) for i in solution])
        gym = Gym(layout=layout, spawn_location=layout_template.entrance, **gym_kwargs)
        for _ in range(steps_per_run):
            gym.step()

        return 0.0 # TODO: extract quality metrics
        # TODO: use mesa.batch_run for staticstically significant cost function results
    


    ga_instance = pygad.GA(
        num_generations=3,
        num_parents_mating=4,
        fitness_func=gym_quality,
        num_genes=len(layout_template),
        gene_space=range(len(Equipment)),
        sol_per_pop=8,
        # set random seed to the gym's seed?
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


if __name__ == "__main__":
    optimize_gym(LayoutTemplate.circular(5, 5))
