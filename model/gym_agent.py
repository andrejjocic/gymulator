import mesa
from gym_model import Gym, Equipment
import mesa.space as space

from enum import Enum, auto
from collections import Counter
from typing import List, Iterator, Optional, Dict, Any, Set


class State(Enum):
    SEARCHING = auto()
    """looking for a machine to use"""
    WORKING_OUT = auto()
    """doing a set"""
    RESTING = auto()
    """resting between sets"""

    @property
    def marker_color(self) -> str:
        match self:
            case State.SEARCHING: return "blue"
            case State.WORKING_OUT: return "red"
            case _:
                raise NotImplementedError(f"color of {self} not specified")
            

Routine = Enum("Routine", "PUSH PULL LEGS")

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
    transition_timer: Optional[int]
    """number of steps left until the agent transitions to the next state"""
    routine: Routine
    used_equipment: Set[Equipment]
    training_queue: Counter
    """muscle -> number of exercises left to do for that muscle"""

    def __init__(self, unique_id: int, model: Gym, routine: Optional[Routine] = None):
        super().__init__(unique_id, model)
        self.state = State.SEARCHING
        self.transition_timer = None

        self.routine = self.random.choice(list(Routine)) if routine is None else routine
        self.used_equipment = set()

        match self.routine:
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
            case _:
                raise ValueError(f"Unsupported workout routine: {routine}")
            
    @property
    def gym(self) -> Gym: # can't just rename model to gym (because subclassing)
        return self.model
            
    def field_of_view(self, diagonals=True, radius=1) -> Iterator[space.Coordinate]:
        """the area in which the agent can see gym equipment"""
        # NOTE: radius > 1 means the agent will "teleport" to a machine, but this actually sort of makes sense:
        # if another (polite) trainee saw them moving to the machine, they wouldn't try to race them to it
        yield from self.model.agent_layer.iter_neighborhood(self.pos, moore=diagonals, include_center=False, radius=radius)

    def move_to(self, new_pos: space.Coordinate):
        self.model.agent_layer.move_agent(self, new_pos)

    
    # NOTE: use self.model.random for random choices / time intervals
    
    def exercise_duration(self) -> int:
        """number of steps to perform an exercise (all sets)"""
        # NOTE: one tick is the amount of time it takes to move between two adjacent cells
        return 5 # TODO: make this random (and sensible)

    def step(self):
        """advance the agent's state machine"""
        print(self)
        if self.transition_timer is not None:
            self.transition_timer -= 1
        
        match self.state:
            case State.SEARCHING:   
                fov = list(self.field_of_view())
                for cell in fov:
                    if (machine := self.model.equipment_layer[cell]) is not None:
                        if machine in self.used_equipment:
                            continue
                        if self.training_queue[machine.muscle] > 0:
                            self.move_to(cell) # NOTE: without staged activation, maybe this can cause multiple agents to move to the same machine ??
                            self.state = State.WORKING_OUT
                            self.transition_timer = self.exercise_duration()
                            
                            self.training_queue[machine.muscle] -= 1
                            self.used_equipment.add(machine)
                            break
                else:
                    free_space = [cell for cell in fov if self.model.equipment_layer[cell] is None]
                    # TODO: follow exploration path (not random)
                    self.move_to(self.random.choice(free_space))

            case State.WORKING_OUT:
                if self.transition_timer == 0:
                    if self.finished_workout:
                        self.model.schedule.remove(self)
                    else:
                        self.state = State.SEARCHING
                        self.transition_timer = None
            case _:
                raise NotImplementedError(f"State {self.state} not implemented")
            
    @property
    def finished_workout(self) -> bool:
        return all(count == 0 for count in self.training_queue.values())

    def __repr__(self) -> str:
        rep = f"{self.pos}: {self.routine.name}Rat<{self.unique_id}>: {self.state.name}"
        if self.transition_timer is not None:
            rep += f" ({self.transition_timer})"
        return rep

    @property
    def portrayal(self) -> Dict[str, Any]:
        return {
            "size": 100,
            "color": self.state.marker_color,
        }
