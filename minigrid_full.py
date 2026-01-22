"""
FULL MINIGRID WITH MINIMAL (FRONT-CELL) OBSERVATION

This implements the COMPLETE official MiniGrid mechanics:
- 7 actions: turn_left, turn_right, forward, pickup, drop, toggle, done
- 11 object types with 6 colors
- Door states: open, closed, locked
- Color-matched key-door mechanics
- Proper reward calculation

BUT with MINIMAL observation:
- Only see the cell DIRECTLY IN FRONT
- Plus what agent is carrying
- Plus what agent is standing on

This is honest partial observability requiring more exploration.
"""

import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, FrozenSet, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import IntEnum


# =============================================================================
# MINIGRID CONSTANTS (matching official implementation exactly)
# =============================================================================

class ObjectType(IntEnum):
    """Object types (OBJECT_TO_IDX from minigrid/core/constants.py)"""
    UNSEEN = 0
    EMPTY = 1
    WALL = 2
    FLOOR = 3
    DOOR = 4
    KEY = 5
    BALL = 6
    BOX = 7
    GOAL = 8
    LAVA = 9
    AGENT = 10

class Color(IntEnum):
    """Colors (COLOR_TO_IDX)"""
    RED = 0
    GREEN = 1
    BLUE = 2
    PURPLE = 3
    YELLOW = 4
    GREY = 5

class DoorState(IntEnum):
    """Door states"""
    OPEN = 0
    CLOSED = 1
    LOCKED = 2

class Direction(IntEnum):
    """Agent facing direction"""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class Action(IntEnum):
    """Official MiniGrid actions"""
    LEFT = 0      # Turn left
    RIGHT = 1     # Turn right  
    FORWARD = 2   # Move forward
    PICKUP = 3    # Pick up object
    DROP = 4      # Drop object
    TOGGLE = 5    # Toggle/activate object
    DONE = 6      # Declare task complete

# Direction vectors
DIR_TO_VEC = {
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
}

# Names for display
OBJECT_NAMES = {
    ObjectType.UNSEEN: "unseen",
    ObjectType.EMPTY: "empty", 
    ObjectType.WALL: "wall",
    ObjectType.FLOOR: "floor",
    ObjectType.DOOR: "door",
    ObjectType.KEY: "key",
    ObjectType.BALL: "ball",
    ObjectType.BOX: "box",
    ObjectType.GOAL: "goal",
    ObjectType.LAVA: "lava",
    ObjectType.AGENT: "agent",
}

COLOR_NAMES = {
    Color.RED: "red",
    Color.GREEN: "green", 
    Color.BLUE: "blue",
    Color.PURPLE: "purple",
    Color.YELLOW: "yellow",
    Color.GREY: "grey",
}

DOOR_STATE_NAMES = {
    DoorState.OPEN: "open",
    DoorState.CLOSED: "closed",
    DoorState.LOCKED: "locked",
}


# =============================================================================
# WORLD OBJECT
# =============================================================================

@dataclass
class WorldObject:
    """An object in the grid world."""
    type: ObjectType
    color: Color = Color.GREY
    state: int = 0  # For doors: DoorState
    
    def encode(self) -> Tuple[int, int, int]:
        return (int(self.type), int(self.color), self.state)
    
    def can_overlap(self) -> bool:
        """Can agent walk on this?"""
        return self.type in [ObjectType.EMPTY, ObjectType.FLOOR, ObjectType.GOAL]
    
    def can_pickup(self) -> bool:
        """Can agent pick this up?"""
        return self.type in [ObjectType.KEY, ObjectType.BALL, ObjectType.BOX]
    
    def can_toggle(self) -> bool:
        """Can this be toggled?"""
        return self.type == ObjectType.DOOR
    
    def __repr__(self):
        name = OBJECT_NAMES.get(self.type, str(self.type))
        if self.type == ObjectType.DOOR:
            state = DOOR_STATE_NAMES.get(DoorState(self.state), str(self.state))
            return f"{COLOR_NAMES[self.color]}_{name}_{state}"
        elif self.type in [ObjectType.KEY, ObjectType.BALL, ObjectType.BOX]:
            return f"{COLOR_NAMES[self.color]}_{name}"
        return name


# =============================================================================
# MINIMAL OBSERVATION (front cell only)
# =============================================================================

class MinimalObs(NamedTuple):
    """
    Minimal observation: just the front cell + carrying + standing on.
    
    This is what the agent ACTUALLY sees at each step.
    """
    front_type: int      # ObjectType of cell in front
    front_color: int     # Color of cell in front  
    front_state: int     # State (for doors)
    carrying_type: int   # What agent is carrying (0 = nothing)
    carrying_color: int  # Color of carried object
    standing_type: int   # What agent is standing on
    standing_color: int  # Color of what standing on
    
    def to_tokens(self) -> FrozenSet[str]:
        """Convert to token set for learner."""
        tokens = set()
        
        # Front cell
        front_name = OBJECT_NAMES.get(ObjectType(self.front_type), f"t{self.front_type}")
        tokens.add(f"front={front_name}")
        
        if self.front_type in [ObjectType.DOOR, ObjectType.KEY, ObjectType.BALL, 
                               ObjectType.BOX, ObjectType.GOAL]:
            color_name = COLOR_NAMES.get(Color(self.front_color), f"c{self.front_color}")
            tokens.add(f"front={color_name}_{front_name}")
        
        if self.front_type == ObjectType.DOOR:
            state_name = DOOR_STATE_NAMES.get(DoorState(self.front_state), f"s{self.front_state}")
            tokens.add(f"front_door={state_name}")
        
        # Carrying
        if self.carrying_type > 0:
            carry_name = OBJECT_NAMES.get(ObjectType(self.carrying_type), f"t{self.carrying_type}")
            carry_color = COLOR_NAMES.get(Color(self.carrying_color), f"c{self.carrying_color}")
            tokens.add(f"carrying={carry_color}_{carry_name}")
            tokens.add("has_item")
        else:
            tokens.add("empty_handed")
        
        # Standing on
        stand_name = OBJECT_NAMES.get(ObjectType(self.standing_type), f"t{self.standing_type}")
        if self.standing_type == ObjectType.GOAL:
            tokens.add("on_goal")
        elif self.standing_type == ObjectType.LAVA:
            tokens.add("on_lava")
        
        return frozenset(tokens)


# =============================================================================
# FULL MINIGRID ENVIRONMENT
# =============================================================================

class FullMiniGridEnv:
    """
    Full MiniGrid environment with all official mechanics.
    
    Uses MINIMAL observation (front cell only).
    """
    
    def __init__(self, width: int, height: int, max_steps: int = None, seed: int = None):
        self.width = width
        self.height = height
        self.max_steps = max_steps or 4 * width * height
        self.rng = random.Random(seed)
        self.seed_value = seed
        
        # Grid storage
        self.grid: List[List[Optional[WorldObject]]] = None
        
        # Agent state
        self.agent_pos: Tuple[int, int] = None
        self.agent_dir: Direction = None
        self.carrying: Optional[WorldObject] = None
        
        # Episode state
        self.step_count = 0
        self.done = False
        self.success = False
        
    def _init_grid(self):
        """Initialize empty grid with walls."""
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Add walls around perimeter
        for x in range(self.width):
            self.grid[0][x] = WorldObject(ObjectType.WALL, Color.GREY)
            self.grid[self.height-1][x] = WorldObject(ObjectType.WALL, Color.GREY)
        for y in range(self.height):
            self.grid[y][0] = WorldObject(ObjectType.WALL, Color.GREY)
            self.grid[y][self.width-1] = WorldObject(ObjectType.WALL, Color.GREY)
    
    def _set(self, x: int, y: int, obj: WorldObject):
        """Place object at position."""
        self.grid[y][x] = obj
    
    def _get(self, x: int, y: int) -> Optional[WorldObject]:
        """Get object at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return WorldObject(ObjectType.WALL, Color.GREY)  # Out of bounds = wall
    
    def _place_agent(self, x: int = None, y: int = None, dir: Direction = None):
        """Place agent at position."""
        if x is None or y is None:
            # Find random empty cell
            empty = [(i, j) for j in range(1, self.height-1) 
                     for i in range(1, self.width-1) if self.grid[j][i] is None]
            if empty:
                x, y = self.rng.choice(empty)
            else:
                x, y = 1, 1
        
        self.agent_pos = (x, y)
        self.agent_dir = dir if dir is not None else Direction(self.rng.randint(0, 3))
    
    def _place_obj(self, obj: WorldObject, x: int = None, y: int = None) -> Tuple[int, int]:
        """Place object at position or random empty cell."""
        if x is not None and y is not None:
            self._set(x, y, obj)
            return (x, y)
        
        # Find random empty cell (not agent position)
        empty = [(i, j) for j in range(1, self.height-1) 
                 for i in range(1, self.width-1) 
                 if self.grid[j][i] is None and (i, j) != self.agent_pos]
        
        if empty:
            x, y = self.rng.choice(empty)
            self._set(x, y, obj)
            return (x, y)
        return None
    
    def reset(self, seed: int = None) -> MinimalObs:
        """Reset environment."""
        if seed is not None:
            self.rng = random.Random(seed)
            self.seed_value = seed
        
        self._init_grid()
        self._gen_grid()  # Subclass implements this
        
        self.carrying = None
        self.step_count = 0
        self.done = False
        self.success = False
        
        return self._get_obs()
    
    def _gen_grid(self):
        """Generate grid contents. Override in subclasses."""
        pass
    
    def _get_obs(self) -> MinimalObs:
        """Get minimal observation (front cell only)."""
        ax, ay = self.agent_pos
        dx, dy = DIR_TO_VEC[self.agent_dir]
        
        # Front cell
        front = self._get(ax + dx, ay + dy)
        if front is None:
            front_type, front_color, front_state = ObjectType.EMPTY, 0, 0
        else:
            front_type, front_color, front_state = front.encode()
        
        # Carrying
        if self.carrying:
            carry_type, carry_color, _ = self.carrying.encode()
        else:
            carry_type, carry_color = 0, 0
        
        # Standing on
        standing = self._get(ax, ay)
        if standing is None:
            stand_type, stand_color = ObjectType.EMPTY, 0
        else:
            stand_type, stand_color, _ = standing.encode()
        
        return MinimalObs(
            front_type=front_type,
            front_color=front_color,
            front_state=front_state,
            carrying_type=carry_type,
            carrying_color=carry_color,
            standing_type=stand_type,
            standing_color=stand_color,
        )
    
    def step(self, action: int) -> Tuple[MinimalObs, float, bool, bool, dict]:
        """
        Execute action.
        
        Returns: (obs, reward, terminated, truncated, info)
        """
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        ax, ay = self.agent_pos
        dx, dy = DIR_TO_VEC[self.agent_dir]
        front_pos = (ax + dx, ay + dy)
        front_obj = self._get(*front_pos)
        
        if action == Action.LEFT:
            self.agent_dir = Direction((self.agent_dir - 1) % 4)
        
        elif action == Action.RIGHT:
            self.agent_dir = Direction((self.agent_dir + 1) % 4)
        
        elif action == Action.FORWARD:
            if front_obj is None or front_obj.can_overlap():
                self.agent_pos = front_pos
            elif front_obj.type == ObjectType.DOOR and front_obj.state == DoorState.OPEN:
                self.agent_pos = front_pos
            elif front_obj.type == ObjectType.LAVA:
                self.agent_pos = front_pos
                terminated = True
                reward = 0  # Death by lava
        
        elif action == Action.PICKUP:
            if front_obj and front_obj.can_pickup() and self.carrying is None:
                self.carrying = front_obj
                self._set(*front_pos, None)
        
        elif action == Action.DROP:
            if self.carrying and (front_obj is None or front_obj.type == ObjectType.EMPTY):
                self._set(*front_pos, self.carrying)
                self.carrying = None
        
        elif action == Action.TOGGLE:
            if front_obj and front_obj.can_toggle():
                if front_obj.type == ObjectType.DOOR:
                    if front_obj.state == DoorState.OPEN:
                        front_obj.state = DoorState.CLOSED
                    elif front_obj.state == DoorState.CLOSED:
                        front_obj.state = DoorState.OPEN
                    elif front_obj.state == DoorState.LOCKED:
                        # Need matching color key
                        if self.carrying and self.carrying.type == ObjectType.KEY:
                            if self.carrying.color == front_obj.color:
                                front_obj.state = DoorState.OPEN
                                self.carrying = None  # Key consumed
        
        elif action == Action.DONE:
            pass  # No effect in most environments
        
        # Check goal
        standing = self._get(*self.agent_pos)
        if standing and standing.type == ObjectType.GOAL:
            terminated = True
            self.success = True
            reward = 1 - 0.9 * (self.step_count / self.max_steps)
        
        # Check timeout
        if self.step_count >= self.max_steps:
            truncated = True
        
        self.done = terminated or truncated
        
        obs = self._get_obs()
        info = {
            "success": self.success,
            "steps": self.step_count,
        }
        
        return obs, reward, terminated, truncated, info


# =============================================================================
# ALL OFFICIAL MINIGRID ENVIRONMENTS
# =============================================================================

class EmptyEnv(FullMiniGridEnv):
    """MiniGrid-Empty-NxN: Navigate to goal in empty room."""
    
    def __init__(self, size: int = 8, agent_start_pos: Tuple[int, int] = None, **kwargs):
        self.agent_start_pos = agent_start_pos
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Goal in bottom-right
        self._set(self.width - 2, self.height - 2, WorldObject(ObjectType.GOAL, Color.GREEN))
        
        # Agent in top-left
        if self.agent_start_pos:
            self._place_agent(*self.agent_start_pos)
        else:
            self._place_agent(1, 1, Direction.RIGHT)


class EmptyRandomEnv(FullMiniGridEnv):
    """MiniGrid-Empty-Random-NxN: Random start position."""
    
    def __init__(self, size: int = 8, **kwargs):
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        self._set(self.width - 2, self.height - 2, WorldObject(ObjectType.GOAL, Color.GREEN))
        self._place_agent()  # Random position


class DoorKeyEnv(FullMiniGridEnv):
    """MiniGrid-DoorKey-NxN: Pick up key, unlock door, reach goal."""
    
    def __init__(self, size: int = 6, **kwargs):
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Vertical wall with door
        split_x = self.width // 2
        for y in range(1, self.height - 1):
            self._set(split_x, y, WorldObject(ObjectType.WALL, Color.GREY))
        
        # Locked door (random y)
        door_y = self.rng.randint(1, self.height - 2)
        door_color = self.rng.choice([Color.RED, Color.BLUE, Color.YELLOW, Color.GREEN])
        self._set(split_x, door_y, WorldObject(ObjectType.DOOR, door_color, DoorState.LOCKED))
        
        # Key on left side (matching color)
        for _ in range(100):
            kx = self.rng.randint(1, split_x - 1)
            ky = self.rng.randint(1, self.height - 2)
            if self.grid[ky][kx] is None:
                self._set(kx, ky, WorldObject(ObjectType.KEY, door_color))
                break
        
        # Goal on right side
        for _ in range(100):
            gx = self.rng.randint(split_x + 1, self.width - 2)
            gy = self.rng.randint(1, self.height - 2)
            if self.grid[gy][gx] is None:
                self._set(gx, gy, WorldObject(ObjectType.GOAL, Color.GREEN))
                break
        
        # Agent on left side
        for _ in range(100):
            ax = self.rng.randint(1, split_x - 1)
            ay = self.rng.randint(1, self.height - 2)
            if self.grid[ay][ax] is None:
                self._place_agent(ax, ay)
                break


class FourRoomsEnv(FullMiniGridEnv):
    """MiniGrid-FourRooms: Navigate through 4 connected rooms."""
    
    def __init__(self, **kwargs):
        super().__init__(19, 19, **kwargs)
    
    def _gen_grid(self):
        mid_x = self.width // 2
        mid_y = self.height // 2
        
        # Vertical wall
        for y in range(1, self.height - 1):
            self._set(mid_x, y, WorldObject(ObjectType.WALL, Color.GREY))
        
        # Horizontal wall
        for x in range(1, self.width - 1):
            self._set(x, mid_y, WorldObject(ObjectType.WALL, Color.GREY))
        
        # Gaps in walls (doors without door objects - just empty)
        gap1_y = self.rng.randint(2, mid_y - 2)
        self._set(mid_x, gap1_y, None)
        
        gap2_y = self.rng.randint(mid_y + 2, self.height - 3)
        self._set(mid_x, gap2_y, None)
        
        gap3_x = self.rng.randint(2, mid_x - 2)
        self._set(gap3_x, mid_y, None)
        
        gap4_x = self.rng.randint(mid_x + 2, self.width - 3)
        self._set(gap4_x, mid_y, None)
        
        # Random goal
        self._place_obj(WorldObject(ObjectType.GOAL, Color.GREEN))
        
        # Random agent
        self._place_agent()


class MultiRoomEnv(FullMiniGridEnv):
    """MiniGrid-MultiRoom-NxS: Series of rooms with doors."""
    
    def __init__(self, num_rooms: int = 4, room_size: int = 5, **kwargs):
        self.num_rooms = num_rooms
        self.room_size = room_size
        width = num_rooms * (room_size - 1) + 1
        super().__init__(width, room_size, **kwargs)
    
    def _gen_grid(self):
        rs = self.room_size
        
        for r in range(self.num_rooms):
            x_offset = r * (rs - 1)
            
            # Room walls (top and bottom already from perimeter)
            # Add vertical wall on right of each room except last
            if r < self.num_rooms - 1:
                wall_x = x_offset + rs - 1
                for y in range(1, self.height - 1):
                    self._set(wall_x, y, WorldObject(ObjectType.WALL, Color.GREY))
                
                # Door in wall
                door_y = self.rng.randint(1, self.height - 2)
                door_color = Color(r % 6)
                self._set(wall_x, door_y, WorldObject(ObjectType.DOOR, door_color, DoorState.CLOSED))
        
        # Goal in last room
        gx = (self.num_rooms - 1) * (rs - 1) + rs // 2
        gy = self.height // 2
        self._set(gx, gy, WorldObject(ObjectType.GOAL, Color.GREEN))
        
        # Agent in first room
        self._place_agent(rs // 2, self.height // 2, Direction.RIGHT)


class LockedRoomEnv(FullMiniGridEnv):
    """MiniGrid-LockedRoom: Multiple rooms with keys and locked doors."""
    
    def __init__(self, **kwargs):
        super().__init__(19, 19, **kwargs)
    
    def _gen_grid(self):
        # Create 3x3 room layout
        room_w = (self.width - 1) // 3
        room_h = (self.height - 1) // 3
        
        colors = [Color.RED, Color.BLUE, Color.YELLOW, Color.GREEN]
        self.rng.shuffle(colors)
        
        # Vertical walls
        for x in [room_w, 2 * room_w]:
            for y in range(1, self.height - 1):
                self._set(x, y, WorldObject(ObjectType.WALL, Color.GREY))
        
        # Horizontal walls
        for y in [room_h, 2 * room_h]:
            for x in range(1, self.width - 1):
                self._set(x, y, WorldObject(ObjectType.WALL, Color.GREY))
        
        # Add doors and keys
        door_positions = [
            (room_w, room_h // 2),
            (2 * room_w, room_h // 2),
            (room_w // 2, room_h),
        ]
        
        for i, (dx, dy) in enumerate(door_positions):
            if i < len(colors):
                self._set(dx, dy, WorldObject(ObjectType.DOOR, colors[i], DoorState.LOCKED))
        
        # Place keys in different rooms
        key_rooms = [(room_w // 2, room_h // 2), 
                     (room_w + room_w // 2, room_h // 2),
                     (room_w // 2, room_h + room_h // 2)]
        
        for i, (kx, ky) in enumerate(key_rooms):
            if i < len(colors):
                self._place_obj(WorldObject(ObjectType.KEY, colors[i]), kx, ky)
        
        # Goal in bottom-right room
        self._set(self.width - 3, self.height - 3, WorldObject(ObjectType.GOAL, Color.GREEN))
        
        # Agent in top-left room
        self._place_agent(2, 2, Direction.RIGHT)


class LavaGapEnv(FullMiniGridEnv):
    """MiniGrid-LavaGap: Cross gap in lava barrier."""
    
    def __init__(self, size: int = 7, **kwargs):
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Lava barrier
        barrier_x = self.width // 2
        gap_y = self.rng.randint(1, self.height - 2)
        
        for y in range(1, self.height - 1):
            if y != gap_y:
                self._set(barrier_x, y, WorldObject(ObjectType.LAVA, Color.RED))
        
        # Agent on left
        self._place_agent(1, self.height // 2, Direction.RIGHT)
        
        # Goal on right (through gap)
        self._set(self.width - 2, gap_y, WorldObject(ObjectType.GOAL, Color.GREEN))


class LavaCrossingEnv(FullMiniGridEnv):
    """MiniGrid-LavaCrossing: Navigate through lava maze."""
    
    def __init__(self, size: int = 9, num_crossings: int = 3, **kwargs):
        self.num_crossings = num_crossings
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Create river of lava with crossings
        for i in range(self.num_crossings):
            river_x = 2 + i * (self.width - 4) // self.num_crossings
            gap_y = self.rng.randint(2, self.height - 3)
            
            for y in range(1, self.height - 1):
                if y != gap_y:
                    self._set(river_x, y, WorldObject(ObjectType.LAVA, Color.RED))
        
        self._place_agent(1, 1, Direction.RIGHT)
        self._set(self.width - 2, self.height - 2, WorldObject(ObjectType.GOAL, Color.GREEN))


class DynamicObstaclesEnv(FullMiniGridEnv):
    """MiniGrid-Dynamic-Obstacles: Moving obstacles to avoid."""
    
    def __init__(self, size: int = 8, n_obstacles: int = 4, **kwargs):
        self.n_obstacles = n_obstacles
        self.obstacles = []  # [(x, y, dx, dy), ...]
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Goal
        self._set(self.width - 2, self.height - 2, WorldObject(ObjectType.GOAL, Color.GREEN))
        
        # Agent
        self._place_agent(1, 1, Direction.RIGHT)
        
        # Obstacles (balls)
        self.obstacles = []
        for _ in range(self.n_obstacles):
            pos = self._place_obj(WorldObject(ObjectType.BALL, Color.BLUE))
            if pos:
                dx, dy = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                self.obstacles.append([pos[0], pos[1], dx, dy])
    
    def step(self, action: int):
        # Move obstacles first
        for obs in self.obstacles:
            x, y, dx, dy = obs
            # Clear current position
            self._set(x, y, None)
            
            # Try to move
            nx, ny = x + dx, y + dy
            next_cell = self._get(nx, ny)
            
            # Bounce off walls
            if next_cell and next_cell.type == ObjectType.WALL:
                dx, dy = -dx, -dy
                nx, ny = x + dx, y + dy
                next_cell = self._get(nx, ny)
            
            # Only move to empty cells
            if next_cell is None or next_cell.type == ObjectType.EMPTY:
                obs[0], obs[1] = nx, ny
                obs[2], obs[3] = dx, dy
            
            # Place obstacle at new position
            self._set(obs[0], obs[1], WorldObject(ObjectType.BALL, Color.BLUE))
        
        # Check collision with agent BEFORE movement
        ax, ay = self.agent_pos
        for obs in self.obstacles:
            if obs[0] == ax and obs[1] == ay:
                return self._get_obs(), -1, True, False, {"collision": True}
        
        # Normal step
        result = super().step(action)
        
        # Check collision AFTER movement
        ax, ay = self.agent_pos
        for obs in self.obstacles:
            if obs[0] == ax and obs[1] == ay:
                return self._get_obs(), -1, True, False, {"collision": True}
        
        return result


class CrossingEnv(FullMiniGridEnv):
    """MiniGrid-Crossing: Cross a road with traffic."""
    
    def __init__(self, size: int = 9, num_lanes: int = 2, **kwargs):
        self.num_lanes = num_lanes
        self.cars = []  # [(x, y, dx), ...]
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Lanes (horizontal roads)
        lane_spacing = (self.height - 2) // (self.num_lanes + 1)
        
        for lane in range(self.num_lanes):
            lane_y = (lane + 1) * lane_spacing
            direction = 1 if lane % 2 == 0 else -1
            
            # Add cars in this lane
            for car_x in range(2, self.width - 2, 3):
                if self.rng.random() < 0.5:
                    self._set(car_x, lane_y, WorldObject(ObjectType.BALL, Color.RED))
                    self.cars.append([car_x, lane_y, direction])
        
        self._place_agent(1, self.height // 2, Direction.RIGHT)
        self._set(self.width - 2, self.height // 2, WorldObject(ObjectType.GOAL, Color.GREEN))


class KeyCorridorEnv(FullMiniGridEnv):
    """MiniGrid-KeyCorridor: Find key in corridor to unlock door."""
    
    def __init__(self, num_rows: int = 3, room_size: int = 4, **kwargs):
        self.num_rows = num_rows
        self.room_size = room_size
        width = (room_size - 1) * 3 + 1
        height = (room_size - 1) * num_rows + 1
        super().__init__(width, height, **kwargs)
    
    def _gen_grid(self):
        rs = self.room_size
        
        # Create corridor with rooms on sides
        for row in range(self.num_rows):
            y_offset = row * (rs - 1)
            
            # Horizontal walls
            if row > 0:
                for x in range(1, self.width - 1):
                    self._set(x, y_offset, WorldObject(ObjectType.WALL, Color.GREY))
            
            # Left room wall
            for y in range(y_offset + 1, y_offset + rs - 1):
                self._set(rs - 1, y, WorldObject(ObjectType.WALL, Color.GREY))
            
            # Right room wall  
            for y in range(y_offset + 1, y_offset + rs - 1):
                self._set(self.width - rs, y, WorldObject(ObjectType.WALL, Color.GREY))
            
            # Doors to rooms
            door_y = y_offset + rs // 2
            if row == 0:
                # Key room (locked)
                key_color = Color.YELLOW
                self._set(rs - 1, door_y, WorldObject(ObjectType.DOOR, key_color, DoorState.LOCKED))
                # Key inside
                self._set(rs // 2, door_y, WorldObject(ObjectType.KEY, key_color))
            elif row == self.num_rows - 1:
                # Goal room (locked)
                self._set(self.width - rs, door_y, WorldObject(ObjectType.DOOR, key_color, DoorState.LOCKED))
                # Goal inside
                self._set(self.width - rs // 2 - 1, door_y, WorldObject(ObjectType.GOAL, Color.GREEN))
        
        # Add gaps in corridor
        for row in range(1, self.num_rows):
            gap_y = row * (rs - 1)
            self._set(rs + 1, gap_y, None)  # Gap in horizontal wall
        
        # Agent in corridor
        self._place_agent(self.width // 2, (rs - 1) // 2, Direction.DOWN)


class MemoryEnv(FullMiniGridEnv):
    """MiniGrid-Memory: Remember which object to pick up."""
    
    def __init__(self, size: int = 8, **kwargs):
        super().__init__(size, size, **kwargs)
        self.target_color = None
    
    def _gen_grid(self):
        # Target shown at start
        self.target_color = self.rng.choice([Color.RED, Color.BLUE, Color.GREEN])
        
        # Place target indicator (ball of target color) at top
        self._set(self.width // 2, 1, WorldObject(ObjectType.BALL, self.target_color))
        
        # Wall to block view after start
        for x in range(1, self.width - 1):
            self._set(x, 2, WorldObject(ObjectType.WALL, Color.GREY))
        
        # Gap to pass through
        self._set(self.width // 2, 2, None)
        
        # Multiple objects at bottom
        colors = [Color.RED, Color.BLUE, Color.GREEN]
        self.rng.shuffle(colors)
        for i, color in enumerate(colors):
            x = 2 + i * 2
            self._set(x, self.height - 2, WorldObject(ObjectType.BALL, color))
        
        # Goal appears when correct ball picked up
        self.goal_pos = (self.width - 2, self.height - 2)
        
        # Agent starts at top (can see target)
        self._place_agent(self.width // 2, 1, Direction.DOWN)
    
    def step(self, action: int):
        result = super().step(action)
        
        # Check if correct ball picked up
        if self.carrying and self.carrying.type == ObjectType.BALL:
            if self.carrying.color == self.target_color:
                # Reveal goal
                self._set(*self.goal_pos, WorldObject(ObjectType.GOAL, Color.GREEN))
        
        return result


class FetchEnv(FullMiniGridEnv):
    """MiniGrid-Fetch: Pick up the object matching mission."""
    
    def __init__(self, size: int = 8, num_objs: int = 3, **kwargs):
        self.num_objs = num_objs
        self.target_obj = None
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        colors = [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]
        types = [ObjectType.KEY, ObjectType.BALL, ObjectType.BOX]
        
        # Place random objects
        placed = []
        for _ in range(self.num_objs):
            obj_type = self.rng.choice(types)
            obj_color = self.rng.choice(colors)
            pos = self._place_obj(WorldObject(obj_type, obj_color))
            if pos:
                placed.append((obj_type, obj_color, pos))
        
        # Pick target (mission)
        if placed:
            self.target_obj = self.rng.choice(placed)[:2]  # (type, color)
        
        self._place_agent()
    
    def step(self, action: int):
        result = super().step(action)
        
        # Check if correct object picked up
        if self.carrying and self.target_obj:
            if (self.carrying.type, self.carrying.color) == self.target_obj:
                result = (result[0], 1.0, True, result[3], result[4])
        
        return result


class GoToDoorEnv(FullMiniGridEnv):
    """MiniGrid-GoToDoor: Navigate to specific colored door."""
    
    def __init__(self, size: int = 6, **kwargs):
        self.target_door = None
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        colors = [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]
        
        # Doors on each wall
        # Top wall
        self._set(self.width // 2, 0, WorldObject(ObjectType.DOOR, colors[0], DoorState.CLOSED))
        # Bottom wall
        self._set(self.width // 2, self.height - 1, WorldObject(ObjectType.DOOR, colors[1], DoorState.CLOSED))
        # Left wall
        self._set(0, self.height // 2, WorldObject(ObjectType.DOOR, colors[2], DoorState.CLOSED))
        # Right wall
        self._set(self.width - 1, self.height // 2, WorldObject(ObjectType.DOOR, colors[3], DoorState.CLOSED))
        
        # Target door
        self.target_door = self.rng.choice(colors)
        
        self._place_agent()
    
    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check if agent is in front of target door
        ax, ay = self.agent_pos
        dx, dy = DIR_TO_VEC[self.agent_dir]
        front = self._get(ax + dx, ay + dy)
        
        if action == Action.DONE and front:
            if front.type == ObjectType.DOOR and front.color == self.target_door:
                reward = 1 - 0.9 * (self.step_count / self.max_steps)
                terminated = True
        
        return obs, reward, terminated, truncated, info


# =============================================================================
# ENVIRONMENT REGISTRY
# =============================================================================

ENVIRONMENTS = {
    # Empty environments
    "Empty-5x5": lambda seed=None: EmptyEnv(size=5, seed=seed),
    "Empty-6x6": lambda seed=None: EmptyEnv(size=6, seed=seed),
    "Empty-8x8": lambda seed=None: EmptyEnv(size=8, seed=seed),
    "Empty-16x16": lambda seed=None: EmptyEnv(size=16, seed=seed),
    "Empty-Random-5x5": lambda seed=None: EmptyRandomEnv(size=5, seed=seed),
    "Empty-Random-6x6": lambda seed=None: EmptyRandomEnv(size=6, seed=seed),
    
    # DoorKey environments
    "DoorKey-5x5": lambda seed=None: DoorKeyEnv(size=5, seed=seed),
    "DoorKey-6x6": lambda seed=None: DoorKeyEnv(size=6, seed=seed),
    "DoorKey-8x8": lambda seed=None: DoorKeyEnv(size=8, seed=seed),
    "DoorKey-16x16": lambda seed=None: DoorKeyEnv(size=16, seed=seed),
    
    # Room environments
    "FourRooms": lambda seed=None: FourRoomsEnv(seed=seed),
    "MultiRoom-N2-S4": lambda seed=None: MultiRoomEnv(num_rooms=2, room_size=4, seed=seed),
    "MultiRoom-N4-S5": lambda seed=None: MultiRoomEnv(num_rooms=4, room_size=5, seed=seed),
    "MultiRoom-N6": lambda seed=None: MultiRoomEnv(num_rooms=6, room_size=4, seed=seed),
    "LockedRoom": lambda seed=None: LockedRoomEnv(seed=seed),
    
    # Lava environments
    "LavaGap-S5": lambda seed=None: LavaGapEnv(size=5, seed=seed),
    "LavaGap-S6": lambda seed=None: LavaGapEnv(size=6, seed=seed),
    "LavaGap-S7": lambda seed=None: LavaGapEnv(size=7, seed=seed),
    "LavaCrossing-S9N1": lambda seed=None: LavaCrossingEnv(size=9, num_crossings=1, seed=seed),
    "LavaCrossing-S9N2": lambda seed=None: LavaCrossingEnv(size=9, num_crossings=2, seed=seed),
    "LavaCrossing-S11N5": lambda seed=None: LavaCrossingEnv(size=11, num_crossings=5, seed=seed),
    
    # Dynamic environments
    "Dynamic-Obstacles-5x5": lambda seed=None: DynamicObstaclesEnv(size=5, n_obstacles=2, seed=seed),
    "Dynamic-Obstacles-6x6": lambda seed=None: DynamicObstaclesEnv(size=6, n_obstacles=3, seed=seed),
    "Dynamic-Obstacles-8x8": lambda seed=None: DynamicObstaclesEnv(size=8, n_obstacles=4, seed=seed),
    
    # Complex environments
    "KeyCorridor-S3R1": lambda seed=None: KeyCorridorEnv(num_rows=1, room_size=3, seed=seed),
    "KeyCorridor-S3R2": lambda seed=None: KeyCorridorEnv(num_rows=2, room_size=3, seed=seed),
    "KeyCorridor-S3R3": lambda seed=None: KeyCorridorEnv(num_rows=3, room_size=3, seed=seed),
    "Memory-S7": lambda seed=None: MemoryEnv(size=7, seed=seed),
    "Memory-S11": lambda seed=None: MemoryEnv(size=11, seed=seed),
    
    # Mission environments
    "Fetch-5x5-N2": lambda seed=None: FetchEnv(size=5, num_objs=2, seed=seed),
    "Fetch-8x8-N3": lambda seed=None: FetchEnv(size=8, num_objs=3, seed=seed),
    "GoToDoor-5x5": lambda seed=None: GoToDoorEnv(size=5, seed=seed),
    "GoToDoor-6x6": lambda seed=None: GoToDoorEnv(size=6, seed=seed),
}


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FULL MINIGRID WITH MINIMAL OBSERVATION")
    print("=" * 70)
    print(f"\nTotal environments: {len(ENVIRONMENTS)}")
    
    print("\n--- Testing Each Environment ---\n")
    
    for name, factory in list(ENVIRONMENTS.items())[:10]:  # Test first 10
        env = factory(seed=42)
        obs = env.reset()
        
        print(f"{name}:")
        print(f"  Grid: {env.width}x{env.height}, Max steps: {env.max_steps}")
        print(f"  Observation tokens: {obs.to_tokens()}")
        
        # Random episode
        done = False
        steps = 0
        while not done and steps < 100:
            action = random.randint(0, 5)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if reward > 0:
                print(f"  Success in {steps} steps! Reward: {reward:.3f}")
                break
        
        if not env.success:
            print(f"  No success in {steps} steps (random policy)")
        print()
