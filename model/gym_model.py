import mesa
import mesa.space as space

from enum import Enum, auto
import numpy as np
from typing import List, Iterator, Optional, Dict, Any


class Gym(mesa.Model):
    Equipment = Enum("Equipment", "BENCH_PRESS SQUAT_RACK LEG_PRESS LAT_PULLDOWN") # TODO: add more 
    # TODO: mapping from machine to muscle (limit to one for now)
    # - maybe machines should be subclasses of some Machine class? (will also need mapping to visual representation)

    agent_layer: space._Grid
    equipment_layer: np.ndarray[Optional[Equipment]]
    

    def __init__(self, num_trainees: int, spawn_location: space.Coordinate = (0, 0)):
        self.equipment_layer = np.array([ # TODO: read this from file / data structure
            [None] * len(Gym.Equipment), # corridor
            list(Gym.Equipment) # all the machines
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
        import gym_agent # placed here to avoid circular import

        for i in range(num_trainees):
            a = gym_agent.GymRat(i, self)
            self.schedule.add(a)
            self.agent_layer.place_agent(a, spawn_location)

        # set up data collection
        self.datacollector = mesa.datacollection.DataCollector(model_reporters={
            "Searching": lambda m: sum(1 for a in m.agents if a.state == gym_agent.GymRat.State.SEARCHING)
        })
        self.running = True

    @property
    def agents(self) -> List['gym_agent.GymRat']:
        return self.schedule.agents
    
    @property
    def space(self) -> space._Grid:
        """space of gym elements (required by mesa visualization functions)"""
        return self.agent_layer # TODO: merge with equipment_layer (or use custom SpaceDrawer)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

