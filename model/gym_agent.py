import mesa
from gym_model import Gym, Equipment
import mesa.space as space

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import copy
from enum import Enum, auto
from collections import Counter
import math
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
            case State.WORKING_OUT: return "orange"
            case _:
                raise NotImplementedError(f"color of {self} not specified")
            

Muscle = Enum("Muscle", start=0, # NOTE: order roughly corresponds to the order of muscles in the body (for easier visualization)
    names= """
    BICEPS TRICEPS
    REAR_DELTS SIDE_DELTS FRONT_DELTS
    CHEST
    TRAPS LATS
    GLUTES HAMSTRINGS QUADS CALVES"""
)

class Routine(Enum):
    PUSH = auto()
    PULL = auto()
    LEGS = auto()

    @property # don't cache this! (modified by caller)
    def muscle_groups(self) -> Counter[Muscle]:
        match self:
            case Routine.PUSH:
                return Counter({
                    Muscle.CHEST: 2,
                    Muscle.TRICEPS: 1,
                    Muscle.FRONT_DELTS: 1,
                    Muscle.SIDE_DELTS: 1,
                })
            case Routine.PULL:
                return Counter({
                    Muscle.LATS: 2,
                    Muscle.BICEPS: 1,
                    Muscle.REAR_DELTS: 1,
                    Muscle.TRAPS: 1,
                })
            case Routine.LEGS:
                return Counter({
                    Muscle.QUADS: 2,
                    Muscle.HAMSTRINGS: 1,
                    Muscle.GLUTES: 1,
                    Muscle.CALVES: 1,
                })
            case _:
                raise ValueError(f"Unsupported workout routine: {self.name}")
            

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
    path: List
    mean_exercise_duration: float

    def __init__(self, unique_id: int, model: Gym, mean_exercise_duration: float, routine: Optional[Routine] = None):
        super().__init__(unique_id, model)
        self.state = State.SEARCHING
        self.transition_timer = None
        self.mean_exercise_duration = mean_exercise_duration # at least make it dependent on routine? (longer for legs)
        # NOTE: one tick is the amount of time it takes to move between two adjacent cells (about 1 second)

        self.routine = self.random.choice(list(Routine)) if routine is None else routine
        self.used_equipment = set()
        self.path = list()
        self.training_queue = self.routine.muscle_groups
            
        if not self.training_queue <= model.machines_per_muscle: # NOTE: redundant check in GA optimizer
            raise ValueError(f"Not enough machines for {self.routine.name} routine")
            
            
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

    
    # NOTE: use self.random for random choices / time intervals
    
    def exercise_duration(self) -> int:
        """number of steps to perform an exercise (all sets)"""
        return int(self.mean_exercise_duration) 
    
    
    def construct_paths(self,grid):
        end_points = list()
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if (i,j) == self.pos:
                    grid[i][j] = -1
                elif grid[i][j] == None:
                    grid[i][j] = 1
                elif self.training_queue[grid[i][j].muscle] > 0 and not grid[i][j] in self.used_equipment:
                    grid[i][j] = 2
                    end_points.append((i,j))
                else:
                    grid[i][j] = 0
        grid = [list(row) for row in zip(*grid)]
        
        paths = list()
        paths = [self.find_path(grid,self.pos,end) for end in end_points]
        distances = [len(path) for path in paths]
        sorted_paths , _ = zip(*sorted(zip(paths,distances), key = lambda x: x[1]))
        return sorted_paths

    def find_path(self, grid, start, end):
        grid = Grid(matrix=grid)
        start_node = grid.node(*start)
        end_node = grid.node(*end)
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path,_ = finder.find_path(start_node, end_node, grid)
        for i in range(len(path)):
            substring = str(path[i]).split("(")[1].split(")")[0]
            substring2 = substring.split(":")
            substring3 = substring2[1].split(" ")
            path[i] = (int(substring2[0]),int(substring3[0]))
        return path

    def step(self):
        """advance the agent's state machine"""
        if self.transition_timer is not None:
            self.transition_timer -= 1
        
        match self.state:
            case State.SEARCHING:   
                fov = list(self.field_of_view())
                for cell in fov:
                    if (machine := self.model.machine_at(cell)) is not None:
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
                    free_space = [cell for cell in fov if self.model.machine_at(cell) is None]
                    if not self.path:
                        self.path = self.random.choice(self.construct_paths(copy.deepcopy(self.model.equipment_layer)))
                        # print(self.path)
                        if not self.path:
                            self.model.schedule.remove(self)
                    if self.path:
                        node = self.path.pop(0)
                        self.move_to(node)
                        """ if (machine := self.model.machine_at(node)) is not None:
                            self.state = State.WORKING_OUT
                            self.transition_timer = self.exercise_duration()
                            self.training_queue[machine.muscle] -= 1
                            self.used_equipment.add(machine)
                            #print(free_space)
                            self.move_to(node) """
                    

            case State.WORKING_OUT:
                if self.transition_timer == 0:
                    if self.finished_workout:
                        self.model.schedule.remove(self)
                    else:
                        self.state = State.SEARCHING
                        self.transition_timer = None
                        self.path = list()
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
        star_points = 5
        return {
            "s": 100,
            "color": self.state.marker_color,
            "alpha": 0.5,
            # "marker": "2", # tri_up
            "marker": (star_points, 1, self.random.uniform(0, 360 / star_points)) # A star-like symbol rotated by angle.
        }
