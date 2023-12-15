import mesa
import mesa.space as space

from enum import Enum, auto
import numpy as np
import copy
from typing import List, Iterator, Optional, Dict, Any


Equipment = Enum("Equipment", "BENCH_PRESS SQUAT_RACK LEG_PRESS LAT_PULLDOWN") # TODO: add more 
"""machines etc. that can be used for a particular exercise"""

class EquipmentAgent(mesa.Agent):
    """piece of equipment, wrapped into an agent (for visualization purposes)"""

    def __init__(self, unique_id: int, model: 'Gym', type: Equipment):
        super().__init__(unique_id, model)
        self.type = type

    @property
    def portrayal(self) -> Dict[str, Any]:
        return {
            "size": 30,
            "color": "black",
        }


# TODO: mapping from machine to muscle (limit to one for now)
# - maybe machines should be subclasses of some Machine class? (will also need mapping to visual representation)

class Gym(mesa.Model):
    num_agents: int
    agent_layer: space._Grid
    equipment_layer: np.ndarray[Optional[Equipment]]

    def __init__(self, num_trainees: int, spawn_location: space.Coordinate = (0, 0)):
        self.num_agents = num_trainees

        self.equipment_layer = np.array([ # TODO: read this from file / data structure
            [None] * len(Equipment), # corridor
            list(Equipment) # all the machines
        ])
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
        self.running = True

    @property
    def agents(self) -> List['GymRat']:
        return self.schedule.agents
    
    @property
    def space(self) -> space._Grid:
        """space of gym elements (required by mesa visualization functions)"""
        elements = copy.copy(self.agent_layer)
        i = self.num_agents

        for cell, val in np.ndenumerate(self.equipment_layer):
            if val is not None:
                virtual_agent = EquipmentAgent(unique_id=i, model=self, type=val)
                i += 1
                elements.place_agent(virtual_agent, cell)

        # could use custom space_drawer in JupyterVis instead of this hack ?
        return elements
    

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

