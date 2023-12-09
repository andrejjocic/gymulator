import mesa
from enum import Enum, auto
from typing import List

class GymRat(mesa.Agent):
    state: 'GymRat.State'

    class State(Enum):
        SEARCHING = auto()
        """looking for a machine to use"""
        # WARMING_UP = auto()
        WORKING_OUT = auto()
        """doing a set"""
        RESTING = auto()
        """resting between sets"""


    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = GymRat.State.SEARCHING

    def advance_state(self):
        # TODO: implement state machine
        # NOTE: use self.model.random for random time intervals, not random.random()!
        ...

    def step(self):
        self.advance_state()
        # TODO: branch on state


class Gym(mesa.Model):
    def __init__(self, num_trainees):
        self.grid = mesa.space.MultiGrid(widht=10, height=10, torus=False) # TODO: actual gym layout
        self.schedule = mesa.time.RandomActivation(self) # do we need staged activation?
        self.running = True

        # Create agents
        for i in range(num_trainees):
            a = GymRat(i, self)
            self.schedule.add(a)

        self.datacollector = mesa.datacollection.DataCollector(model_reporters={
            "Searching": lambda m: sum(1 for a in m.agents if a.state == GymRat.State.SEARCHING)
        })

    @property
    def agents(self) -> List[GymRat]:
        return self.schedule.agents

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()