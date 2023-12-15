import mesa
from gym_model import Gym
import mesa.space as space

from enum import Enum, auto
from collections import Counter
from typing import List, Iterator, Optional, Dict, Any


class State(Enum):
    SEARCHING = auto()
    """looking for a machine to use"""
    # WARMING_UP = auto()
    WORKING_OUT = auto()
    """doing a set"""
    RESTING = auto()
    """resting between sets"""

Routine = Enum("Routine", "PUSH PULL LEGS TEST") # TODO: remove TEST

Muscle = Enum("Muscle",
    names= """
    BICEPS TRICEPS
    FRONT_DELTS SIDE_DELTS REAR_DELTS
    CHEST
    TRAPS LATS
    QUADS HAMSTRINGS GLUTES CALVES"""
)

class GymRat(mesa.Agent):
    unique_id: int
    model: Gym
    state: State
    training_queue: Counter
    """muscle -> number of exercises left to do for that muscle"""
    # TODO: add history of used machines (to make sure the same machine isn't used twice in a workout)

    def __init__(self, unique_id: int, model: Gym, routine: Routine = Routine.TEST):
        super().__init__(unique_id, model)
        self.state = State.SEARCHING

        match routine:
            case Routine.PUSH:
                self.training_queue = Counter({
                    Muscle.CHEST: 2,
                    Muscle.TRICEPS: 1,
                    Muscle.FRONT_DELTS: 1,
                    Muscle.SIDE_DELTS: 1,
                })
            case Routine.PULL:
                self.training_queue = Counter({
                    Muscle.LATS: 2,
                    Muscle.BICEPS: 1,
                    Muscle.REAR_DELTS: 1,
                    Muscle.TRAPS: 1,
                })
            case Routine.LEGS:
                self.training_queue = Counter({
                    Muscle.QUADS: 2,
                    Muscle.HAMSTRINGS: 1,
                    Muscle.GLUTES: 1,
                    Muscle.CALVES: 1,
                })
            case Routine.TEST:
                self.training_queue = Counter({
                    Muscle.CHEST: 1,
                })
            case _:
                raise ValueError(f"Unsupported workout routine: {routine}")
            
    @property
    def gym(self) -> Gym: # can't just rename model to gym (because subclassing)
        return self.model
            
    def field_of_view(self, diagonals=True, radius=1) -> Iterator[space.Coordinate]:
        """the area in which the agent can see gym equipment"""
        yield from self.model.agent_layer.iter_neighborhood(self.pos, moore=diagonals, include_center=False, radius=radius)

    def move_to(self, new_pos: space.Coordinate):
        self.model.agent_layer.move_agent(self, new_pos)

    def step(self):
        """advance the agent's state machine"""
        # NOTE: use self.model.random for random choices / time intervals, not random.random or np.rand
        print(self)
        
        match self.state:
            case State.SEARCHING:
                free_space = [cell for cell in self.field_of_view() if self.model.equipment_layer[cell] is None]
                # print(f"free space from {self.pos}: {free_space}")
                self.move_to(self.random.choice(free_space))
                # TODO: follow exploration path (not random)
                # TODO: go to a free machine, if any
            case _:
                raise NotImplementedError(f"State {self.state} not implemented")

    def __repr__(self) -> str:
        return f"Agent {self.unique_id} @ {self.pos} ({self.state.name})"

    @property
    def portrayal(self) -> Dict[str, Any]:
        return {
            "size": 15,
            "color": ["green", "red", "gray"][self.state.value - 1],
        }
