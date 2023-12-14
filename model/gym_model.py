import mesa
from enum import Enum, auto
from typing import List
from collections import Counter
from transitions import Machine

class GymRat(mesa.Agent):
    state: 'GymRat.State'
    training_queue: Counter
    """muscle -> number of exercises left to do for that muscle"""


    class State(Enum):
        SEARCHING = auto()
        """looking for a machine to use"""
        # WARMING_UP = auto()
        WORKING_OUT = auto()
        """doing a set"""
        RESTING = auto()
        """resting between sets"""

    Workout = Enum("Workout", "PUSH PULL LEGS")

    Muscle = Enum("Muscle",
        names= """
        BICEPS TRICEPS
        FRONT_DELTS SIDE_DELTS REAR_DELTS
        TRAPS LATS
        QUADS HAMSTRINGS GLUTES CALVES"""
    )


    def __init__(self, unique_id, model, workout: 'Workout'):
        super().__init__(unique_id, model)
        self.state = GymRat.State.SEARCHING
        match workout:
            case GymRat.Workout.PUSH:
                self.training_queue = Counter({
                    GymRat.Muscle.CHEST: 2,
                    GymRat.Muscle.TRICEPS: 1,
                    GymRat.Muscle.FRONT_DELTS: 1,
                    GymRat.Muscle.SIDE_DELTS: 1,
                })
            case GymRat.Workout.PULL:
                self.training_queue = Counter({
                    GymRat.Muscle.LATS: 2,
                    GymRat.Muscle.BICEPS: 1,
                    GymRat.Muscle.REAR_DELTS: 1,
                    GymRat.Muscle.TRAPS: 1,
                })
            case GymRat.Workout.LEGS:
                self.training_queue = Counter({
                    GymRat.Muscle.QUADS: 2,
                    GymRat.Muscle.HAMSTRINGS: 1,
                    GymRat.Muscle.GLUTES: 1,
                    GymRat.Muscle.CALVES: 1,
                })

    def advance_state(self):
        # TODO: implement state machine
        # NOTE: use self.model.random for random time intervals, not random.random()!
        ...

    def step(self):
        self.advance_state()
        # TODO: branch on state


class Gym(mesa.Model):
    Machine = Enum("Machine", "BENCH_PRESS SQUAT_RACK LEG_PRESS LAT_PULLDOWN")

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