import mesa
import mesa.space as space

from enum import Enum, auto
import numpy as np
import copy
import math
from typing import List, Iterator, Optional, Dict, Any, Set, Tuple


class Equipment(Enum):
    """machines etc. that can be used for a particular exercise"""
    BENCH_PRESS = auto()
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
    


class EquipmentAgent(mesa.Agent):
    """piece of equipment, wrapped into an agent (for visualization purposes)"""
    model: 'Gym'

    def __init__(self, unique_id: int, model: 'Gym', type: Equipment):
        super().__init__(unique_id, model)
        self.type = type

    @property
    def portrayal(self) -> Dict[str, Any]:
        # try:
        #     occupant = next(self.model.agent_layer.iter_cell_list_contents(self.pos))
        #     print(f"using {self}: {occupant}")
        #     free = False
        # except StopIteration:
        #     free = True
        free = self.model.agent_layer.is_cell_empty(self.pos)

        return {
            "s": 40,
            "color": "green" if free else "red",
            "alpha": 1,
            "marker": "p" # pentagon
        }


# TODO: mapping from machine to muscle (limit to one for now)
# - maybe machines should be subclasses of some Machine class? (will also need mapping to visual representation)

class Gym(mesa.Model):
    num_agents: int
    agent_layer: space._Grid
    equipment_layer: np.ndarray[Optional[Equipment]]

    def __init__(self, num_trainees: int, machine_density: float, spawn_location: space.Coordinate = (0, 0)):
        self.num_agents = num_trainees

        # TODO: read layout from file / data structure 
        # self.equipment_layer = np.array([ 
        #     [None] * len(Equipment), # corridor
        #     list(Equipment) # all the machines
        # ]).T
        self.equipment_layer, spawn_location = self.build_random_layout(machine_density=machine_density)
        if self.equipment_layer[spawn_location] is not None:
            raise ValueError(f"Spawn location {spawn_location} is occupied by {self.equipment_layer[spawn_location]}")
        
        self.agent_layer = space.MultiGrid(*self.equipment_layer.shape, torus=False) 
        # maybe try hexagonal grid?
        
        self.schedule = mesa.time.RandomActivation(self)
        # maybe we need another scheduler type?
        # - simultaneous activation (to avoid weird deadlocks)
        # - random activation by type (to avoid bias towards leg dayers??)
        
        # Create agents
        import gym_agent as agent # placed here to avoid circular import

        for i in range(num_trainees):
            a = agent.GymRat(i, self)
            self.schedule.add(a)
            self.agent_layer.place_agent(a, spawn_location)

        # set up data collection
        self.datacollector = mesa.datacollection.DataCollector(model_reporters={
            "Searching": lambda m: sum(1 for a in m.agents if a.state == agent.State.SEARCHING)
        })
    
    def build_random_layout(self, machine_copies=2, machine_density=1/2) -> Tuple[np.ndarray[Optional[Equipment]], space.Coordinate]:
        """generate a random layout of all the machines (each machine appears twice by default)
        - machine_density = (number of machines) / (number of cells)
        - machine_copies = number of copies of each machine
        - returns: layout, valid spawn location"""

        machines = list(Equipment) * machine_copies
        total_cells = math.ceil(len(machines) / machine_density)
        rows = math.ceil(math.sqrt(total_cells))
        cols = math.ceil(total_cells / rows)

        coords = [(y, x) for y in range(rows) for x in range(cols)]
        self.random.shuffle(coords)

        layout = np.full((rows, cols), None, dtype=Equipment)
        for i, machine in enumerate(machines):
            layout[coords[i]] = machine # FIXME: this can create unreachable machines

        return layout, coords[i + 1]
    

    @property
    def running(self) -> bool:
        return self.schedule.get_agent_count() > 0

    @property
    def agents(self) -> List['GymRat']:
        return self.schedule.agents
    
    @property
    def space(self) -> space._Grid:
        """space of gym elements (required by mesa visualization functions)"""
        elements = copy.deepcopy(self.agent_layer) # need deepcopy?
        i = self.num_agents

        for cell, val in np.ndenumerate(self.equipment_layer):
            if val is not None:
                virtual_agent = EquipmentAgent(unique_id=i, model=self, type=val)
                i += 1
                elements.place_agent(virtual_agent, cell)

        # could use custom space_drawer in JupyterVis instead of this hack?
        return elements
    

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()