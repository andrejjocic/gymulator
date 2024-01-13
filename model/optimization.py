import pygad
import pygad.visualize
import pygad.utils.nsga2 as nsga
# import pygad.helper

from gym_model import Gym, Equipment, machines_per_muscle, GymLayout
from gym_agent import Routine
import mesa.space as space
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pprint import pprint
from enum import Enum, auto
import pickle

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
    # NOTE: try to keep locations spatially ordered so k-point crossover doesn't mess up the structure

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
    

def efficiency_ratio(agent_df: pd.DataFrame) -> pd.Series:
    """proportion of time spent working out"""
    total_steps = agent_df.groupby("AgentID").size()
    steps_working_out = agent_df[agent_df['State'] == 'WORKING_OUT'].groupby('AgentID').size()

    efficiency_ratio = steps_working_out / total_steps
    efficiency_ratio.fillna(0, inplace=True) # some of them don't even lift (bruh)
    return efficiency_ratio


class GymMetric(Enum):
    EFFICIENCY = auto()
    CONGESTION = auto()
    UTILIZATION = auto()

    def fitness_value(self, model_avg: pd.Series, agent_df: pd.DataFrame) -> float:
        """value from 0 to 1, to be maximized"""
        match self:
            case GymMetric.EFFICIENCY:
                val = efficiency_ratio(agent_df).mean()
            case GymMetric.CONGESTION:
                val = model_avg["Congestion"]
            case GymMetric.UTILIZATION:
                val = model_avg["Utilization"]
            case _:
                raise ValueError(f"unsupported gym metric: {self}")
            
        return val if self.positive else 1 - val
            
    @property
    def positive(self) -> bool:
        """should this metric be maximized?"""
        return self != GymMetric.CONGESTION

    def __str__(self) -> str:
        label = self.name.lower()
        if not self.positive: label = "~" + label
        return label


def optimal_gyms(layout_template: LayoutTemplate,
                n_generations: int, simulation_cycle_steps: int,
                fitness_metrics: List[GymMetric] = [GymMetric.EFFICIENCY, GymMetric.UTILIZATION, GymMetric.CONGESTION],
                population_size=46,
                crossover_method="single_point", mutation_method="random", tournament_participants: Optional[int] = None,
                parents_proportion=0.5, mutation_percents=10,
                plot_evolution=False, 
                **gym_kwargs
                ) -> List[Tuple[GymLayout, Tuple[float, ...]]]:
    """
    Optimize the layout of a gym with multi-objective genetic algorithm, constrained by the layout template.
    - crossover_method: "single_point", "two_points", "uniform", "scattered", or None
    - mutation_method: "random", "swap", "scramble", "adaptive"
    - tournament_participants: use None for normal (non-tournament) NSGA-II selection. Greater number -> more selection pressure

    Returns the Pareto-optimal gym layouts and their fitness values.
    """ 
    def gym_quality(ga_instance, solution, solution_idx) -> Tuple[float, ...]:
        """multi-objective fitness function for pygad (will be maximized)"""
        nonlocal discarded_sol_count

        layout = layout_template.instantiate(machines=[Equipment(i) for i in solution])
        total_machines = machines_per_muscle(layout)

        for routine in Routine:
            if not routine.muscle_groups <= total_machines:
                discarded_sol_count += 1
                return [-np.inf] * len(fitness_metrics) # invalid solution (not enough machines for some routine)
                # could instead cull checklist (+ fitness penalty); or prevent with custom GA functions?
        
        gym = Gym(layout=layout, spawn_location=layout_template.entrance, **gym_kwargs)
        # TODO: use mesa.batch_run for stat. significant cost function results (if there is a lot of randomness in gym model)
        model_metrics, agent_metrics = gym.run(simulation_cycle_steps, progress_bar=False)
        time_avg = model_metrics.mean() # take average of columns (across time steps)
        return [metric.fitness_value(time_avg, agent_metrics) for metric in fitness_metrics]
    

    def print_evolution_progress(ga_instance):
        nonlocal discarded_sol_count
        print(f"INFO: discarded {discarded_sol_count}/{ga_instance.pop_size[0]} solutions")
        print(f"Finished generation {ga_instance.generations_completed}/{ga_instance.num_generations}") 
        discarded_sol_count = 0
        if plot_evolution:
            col_maxes = np.max(ga_instance.last_generation_fitness, axis=0)
            max_fits[ga_instance.generations_completed - 1] = col_maxes
    

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
        on_generation=print_evolution_progress,
        # save_best_solutions=True,
        
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
        # TODO: sync random seed between GA and gyms?? https://pygad.readthedocs.io/en/latest/pygad_more.html#random-seed
    )
    ga_instance.summary()

    discarded_sol_count = 0
    if plot_evolution: max_fits = np.empty((n_generations, len(fitness_metrics)))
    ga_instance.run()

    # print(f"Best (individual) fitness value reached after {ga_instance.best_solution_generation} generations.")
    if plot_evolution:                
        plt.plot(max_fits)
        plt.legend([str(metric) for metric in fitness_metrics])
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.ylim(0, 1)
        plt.title("Max (objective-wise) fitness values over generations")
        plt.show()

    pareto_optimal_front = ga_instance.pareto_fronts[0]
    best_solutions = []
    for sol_idx, fitness in pareto_optimal_front:
        sol = ga_instance.population[sol_idx]
        best_solutions.append((layout_template.instantiate([Equipment(i) for i in sol]), fitness))

    return best_solutions # TODO: sort them by crowding distance? (using ga_instance.crowding_distance)


if __name__ == "__main__":
    isle_rows, isle_cols = 1, 2
    template = LayoutTemplate.square_isles(isle_rows, isle_cols)
    metrics = [GymMetric.EFFICIENCY, GymMetric.UTILIZATION, GymMetric.CONGESTION]

    gyms = optimal_gyms(
        layout_template=template,
        fitness_metrics=metrics,
        n_generations=100, population_size=50,
        crossover_method="two_points", tournament_participants=None,
        mutation_percents=15,
        plot_evolution=True,
        simulation_cycle_steps=250, interarrival_time=2, agent_exercise_duration=30,
    )
    print(f"Found {len(gyms)} Pareto-optimal gyms.")

    
    fig, ax = plt.subplots(nrows=len(metrics), ncols=2, width_ratios=[1, 0.5])
    
    for i, metric in enumerate(metrics):
        best_gym, fits = max(gyms, key=lambda gym_fitness: gym_fitness[1][i])
        draw_layout(best_gym, ax[i, 0], title=f"Layout with best {metric}")
        ax[i, 1].bar(range(len(fits)), fits, tick_label=list(map(str, metrics)))
        ax[i, 1].set_ylim(0, 1)
        ax[i, 1].set_ylabel('Value')
        ax[i, 1].set_title('Fitness Values')

        # TODO: pickle the best gym layouts (put metric in filename)
        name = f"model/layouts/best_{isle_rows}x{isle_cols}_gym_{metric}"
        np.save(name, best_gym)

    plt.show()