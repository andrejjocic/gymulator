import mesa
import mesa.space as space

from enum import Enum, auto
import numpy as np
import copy
import math
from typing import List, Iterator, Optional, Dict, Any, Set, Tuple
from functools import cached_property
from collections import Counter
import pandas as pd


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
        free = self.model.agent_layer.is_cell_empty(self.pos)

        return {
            "s": 40,
            "color": "green" if free else "red",
            "alpha": 1,
            "marker": "p" # pentagon
        }
    

class Gym(mesa.Model):
    interarrival_time: int
    """time between arrivals of new trainees (in timesteps)"""
    mean_exercise_duration: int
    spawn_location: space.Coordinate
    spawned_agents: int = 0
    spawn_timer: int = 0
    agent_layer: space._Grid
    equipment_layer: np.ndarray[Optional[Equipment]]

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
        self.datacollector = mesa.datacollection.DataCollector(model_reporters={
            "Utilization": lambda model: proportion_working(model),
            "Congestion": lambda model: congestion_factor(model),
        })


    def machine_at(self, cell: space.Coordinate) -> Optional[Equipment]:
        return self.equipment_layer[cell]
    
    @property
    def machines(self) -> Iterator[Equipment]:
        return (machine for machine in self.equipment_layer.flat if machine is not None)
    
    @cached_property
    def machines_per_muscle(self) -> Counter:
        return Counter(machine.muscle for machine in self.machines)


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
    
    @property
    def space(self) -> space._Grid:
        """space of gym elements (required by mesa visualization functions)"""
        elements = copy.deepcopy(self.agent_layer)
        i = self.spawned_agents

        for pos, val in np.ndenumerate(self.equipment_layer):
            if val is not None:
                virtual_agent = EquipmentAgent(unique_id=i, model=self, type=val)
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


    def run(self, num_steps: int) -> pd.DataFrame:
        """run the model for a given number of steps and return the model-level metrics"""
        for _ in range(num_steps):
            self.step()
        
        return self.datacollector.get_model_vars_dataframe()


def proportion_working(model: Gym) -> float:
    """fraction of agents that are currently working out"""
    return sum(agent.state.name == "WORKING_OUT" for agent in model.agents) / len(model.agents)

def congestion_factor(model: Gym) -> float:
    """maximum number of agents in a single cell"""
    return max(len(content) for content, _ in model.agent_layer.coord_iter())
    # TODO: try some other congestion metrics (ideally relative to gym size / number of agents?)