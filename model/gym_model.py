import mesa
import mesa.space as space

from enum import Enum, auto
import numpy as np
import copy
import math
from typing import TypeAlias, List, Iterator, Optional, Dict, Any, Set, Tuple
from functools import cached_property
from collections import Counter
import pandas as pd
from tqdm import tqdm
import itertools


class Equipment(Enum):
    """machines etc. that can be used for a particular exercise"""
    BENCH_PRESS = 0
    CHEST_FLY = auto()
    TRICEPS_EXTENSION = auto()
    TRICEPS_PUSHDOWN = auto()
    LAT_PULLDOWN = auto()
    LEG_PRESS = auto()
    LEG_EXTENSION = auto()
    LEG_CURL = auto()
    CALF_RAISE = auto()
    SHOULDER_PRESS = auto()
    LATERAL_RAISE = auto()
    REAR_DELT_FLY = auto()
    SHRUG = auto()
    BICEPS_CURL = auto()
    TRICEPS_DIP = auto()
    ROW = auto()
    SQUAT_RACK = auto()
    HIP_THRUST = auto()
    
    @property
    def muscle(self) -> 'Muscle':
        from gym_agent import Muscle # placed here to avoid circular import

        match self:
            case Equipment.BENCH_PRESS: return Muscle.CHEST
            case Equipment.CHEST_FLY: return Muscle.CHEST
            case Equipment.TRICEPS_EXTENSION: return Muscle.TRICEPS
            case Equipment.TRICEPS_PUSHDOWN: return Muscle.TRICEPS
            case Equipment.LAT_PULLDOWN: return Muscle.LATS
            case Equipment.LEG_PRESS: return Muscle.QUADS
            case Equipment.LEG_EXTENSION: return Muscle.QUADS
            case Equipment.LEG_CURL: return Muscle.HAMSTRINGS
            case Equipment.CALF_RAISE: return Muscle.CALVES
            case Equipment.SHOULDER_PRESS: return Muscle.FRONT_DELTS
            case Equipment.LATERAL_RAISE: return Muscle.SIDE_DELTS
            case Equipment.REAR_DELT_FLY: return Muscle.REAR_DELTS
            case Equipment.SHRUG: return Muscle.TRAPS
            case Equipment.BICEPS_CURL: return Muscle.BICEPS
            case Equipment.TRICEPS_DIP: return Muscle.TRICEPS
            case Equipment.ROW: return Muscle.LATS
            case Equipment.SQUAT_RACK: return Muscle.QUADS
            case Equipment.HIP_THRUST: return Muscle.GLUTES
            case _:
                raise NotImplementedError(f"muscle for {self} not specified")
    

# - maybe machines should be subclasses of some Machine class? (will also need mapping to visual representation)

class EquipmentAgent(mesa.Agent):
    """piece of equipment, wrapped into an agent (for visualization purposes)"""
    model: 'Gym'

    def __init__(self, unique_id: int, model: 'Gym', type: Equipment):
        super().__init__(unique_id, model)
        self.type = type

    @property
    def portrayal(self) -> Dict[str, Any]:
        return {
            "s": 40,
            "color": "red" if self.model.occuped(self.pos) else "green",
            "alpha": 1,
            "marker": "p" # pentagon
        }
    

GymLayout: TypeAlias = np.ndarray[Optional[Equipment]]

def machines_per_muscle(layout: GymLayout) -> Counter['Muscle']:
    """number of unique machines per muscle"""
    return Counter(machine.muscle for machine in set(layout.flat) if machine is not None)
    

class Gym(mesa.Model):
    interarrival_time: int
    """time between arrivals of new trainees (in timesteps)"""
    mean_exercise_duration: int
    spawn_location: space.Coordinate
    spawned_agents: int = 0
    spawn_timer: int = 0
    agent_layer: space._Grid
    equipment_layer: GymLayout

    def __init__(self, interarrival_time: int, agent_exercise_duration=2*60,
                 layout: Optional[np.ndarray] = None, machine_density=0.5, spawn_location: space.Coordinate = (0, 0)):
        """
        - agent_exercise_duration: mean duration of a single exercise (in timesteps)
        - interarrival_time: time between arrivals of new trainees (in timesteps)
        - layout: 2D array of Equipment (None for empty cells). If None, a random layout will be generated.
        NOTE: position (x,y) corresponds to layout[x, y], not layout[y, x]
        """
        self.interarrival_time = interarrival_time
        self.mean_exercise_duration = agent_exercise_duration

        if layout is None:
            assert 0 < machine_density <= 1
            self.equipment_layer, self.spawn_location = self.build_random_layout(machine_density)
        else:
            self.equipment_layer = layout
            self.spawn_location = spawn_location

        if (obstacle := self.machine_at(self.spawn_location)) is not None:
            raise ValueError(f"Spawn location {spawn_location} is occupied by {obstacle.name}")
        
        self.agent_layer = space.MultiGrid(*self.equipment_layer.shape, torus=False) 
        # maybe try hexagonal grid?
        
        self.schedule = mesa.time.RandomActivation(self)
        # maybe we need another scheduler type?
        # - simultaneous activation (to avoid weird deadlocks)
        # - random activation by type (to avoid bias towards agents in certain state / routine)
        
        # spawn the first agent
        self.spawn_trainee()

        # set up data collection
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                # "Efficiency": proportion_working,
                "Utilization": proportion_in_use,
                "Congestion": crowdedness,
                # "Travel": lambda model: model.total_travel_distance
            },
            agent_reporters={
                "State": lambda agent: agent.state.name,
            }
        )


    def machine_at(self, cell: space.Coordinate) -> Optional[Equipment]:
        return self.equipment_layer[cell]
    
    @cached_property
    def num_machines(self) -> int:
        """number of machines in the gym"""
        return np.count_nonzero(self.equipment_layer != None)
    
    @cached_property
    def machines_per_muscle(self) -> Counter:
        """number of unique machines per muscle"""
        return machines_per_muscle(self.equipment_layer)


    def spawn_trainee(self):
        import gym_agent as agent # placed here to avoid circular import
        
        a = agent.GymRat(self.spawned_agents, self, self.mean_exercise_duration)
        self.schedule.add(a)
        self.agent_layer.place_agent(a, self.spawn_location)
        self.spawned_agents += 1

        # schedule next spawn
        self.spawn_timer = self.interarrival_time

    
    def build_random_layout(self, machine_density: float, machine_copies=2) -> Tuple[np.ndarray, space.Coordinate]:
        """generate a random layout of all the machines (each machine appears twice by default)
        - machine_density = (number of machines) / (number of cells)
        - machine_copies = number of copies of each machine
        - returns: layout, valid spawn location"""

        machines = list(Equipment) * machine_copies
        total_cells = math.ceil(len(machines) / machine_density)
        rows = math.ceil(math.sqrt(total_cells))
        cols = math.ceil(total_cells / rows)

        coords = [(x, y) for y in range(rows) for x in range(cols)]
        self.random.shuffle(coords)
        layout = np.full((cols, rows), None, dtype=Equipment) 
        for i, machine in enumerate(machines):
            layout[coords[i]] = machine # FIXME: this can create unreachable machines

        return layout, coords[i + 1]
    

    @property
    def running(self) -> bool:
        # return self.schedule.get_agent_count() > 0
        return True

    @property
    def agents(self) -> List['GymRat']:
        return self.schedule.agents
    
    @cached_property
    def machine_positions(self) -> Set[space.Coordinate]:
        """positions of all machines"""
        return set(zip(*np.where(self.equipment_layer != None)))
    
    def occuped(self, pos: space.Coordinate) -> bool:
        """whether there is an agent at the given position"""
        return not self.agent_layer.is_cell_empty(pos)
    
    @property
    def space(self) -> space._Grid:
        """space of gym elements (required by mesa visualization functions)"""
        elements = copy.deepcopy(self.agent_layer)
        i = self.spawned_agents

        # for pos, val in np.ndenumerate(self.equipment_layer):
        #     if val is not None:
        for pos in self.machine_positions:
            virtual_agent = EquipmentAgent(unique_id=i, model=self, type=self.machine_at(pos))
            i += 1
            elements.place_agent(virtual_agent, pos)

        # could use custom space_drawer in JupyterVis instead of this hack?
        # https://mesa.readthedocs.io/en/stable/tutorials/adv_tutorial_legacy.html
        return elements
    

    def step(self):
        self.spawn_timer -= 1
        if self.spawn_timer == 0:
            self.spawn_trainee()

        self.datacollector.collect(self)
        self.schedule.step()


    def run(self, num_steps: int, progress_bar=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """run the model for a given number of steps and return (model-level metrics, agent-level metrics)"""
        if progress_bar:
            for _ in (pbar := tqdm(range(num_steps), desc="Simulating gym")):
                self.step()
                pbar.set_postfix_str(f"{len(self.agents)} agents")
        else:
            for _ in range(num_steps):
                self.step()
        
        return self.datacollector.get_model_vars_dataframe(), self.datacollector.get_agent_vars_dataframe()


def proportion_working(model: Gym) -> float:
    """fraction of agents that are currently working out"""
    return sum(agent.state.name == "WORKING_OUT" for agent in model.agents) / len(model.agents)
    

def proportion_in_use(model: Gym) -> float:
    """fraction of machines that are currently in use"""
    return sum(1 for pos in model.machine_positions if model.occuped(pos)) / model.num_machines

def congestion_factor(model: Gym, normalize=False) -> float:
    """maximum agents in a single cell (from the ones not working out)"""
    max_crowd = max(len(content) for content, pos in model.agent_layer.coord_iter() if pos not in model.machine_positions) 
    if normalize:
        not_working = sum(1 for agent in model.agents if agent.state.name != "WORKING_OUT")
        return max_crowd / not_working if not_working > 0 else 0
    else:
        return max_crowd
    

def manhattan_distance(a: space.Coordinate, b: space.Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def crowdedness(model: Gym, distance_threshold=2) -> float:
    """proportion of pairs of (idle) agents that are too close to each other (by inter-cell manhattan distance)"""
    agent_positions = [agent.pos for agent in model.agents if agent.state.name != "WORKING_OUT"]
    n = len(agent_positions)
    if n < 2:
        return 0
    
    too_close = 0
    for a1, a2 in itertools.combinations(agent_positions, 2): # NOTE: could probably be optimized
        if manhattan_distance(a1, a2) <= distance_threshold:
            too_close += 1

    return too_close / (n * (n - 1) / 2) # n choose 2