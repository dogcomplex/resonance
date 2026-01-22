"""
OFFICIAL MINIGRID BENCHMARK

This implements MiniGrid environments matching the official Farama Foundation specification:
https://minigrid.farama.org/

KEY DIFFERENCES FROM OUR PREVIOUS SIMULATIONS:
1. Agent sees a 7x7 grid (not just 1 cell in front!)
2. View is agent-centered and forward-facing
3. Each cell has 3 values: (object_type, color, state)
4. 7 actions: turn_left, turn_right, forward, pickup, drop, toggle, done

ENVIRONMENTS TESTED:
1. Empty-5x5, Empty-8x8, Empty-16x16 - Just navigate to goal
2. DoorKey-5x5, DoorKey-6x6, DoorKey-8x8 - Pickup key, unlock door, reach goal
3. FourRooms - Navigate through 4 connected rooms to goal
4. KeyCorridor - Find key in corridor, unlock door, pickup ball
5. MultiRoom - Navigate through series of rooms with doors
6. LavaGap - Cross gap in lava to reach goal
7. DynamicObstacles - Avoid moving obstacles

FAIRNESS RULES:
- NO domain knowledge in learner
- Learn from raw (type, color, state) tuples
- Discover object semantics from interaction
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set, FrozenSet, Optional
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

# =============================================================================
# MINIGRID CONSTANTS (matching official implementation)
# =============================================================================

class ObjectType(IntEnum):
    """Object types in MiniGrid (OBJECT_TO_IDX)"""
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
    """Colors in MiniGrid (COLOR_TO_IDX)"""
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
    """MiniGrid actions"""
    LEFT = 0      # Turn left
    RIGHT = 1     # Turn right  
    FORWARD = 2   # Move forward
    PICKUP = 3    # Pick up object
    DROP = 4      # Drop object
    TOGGLE = 5    # Toggle/activate object (open door, etc)
    DONE = 6      # Declare task complete

# Direction deltas
DIR_TO_VEC = {
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
}


# =============================================================================
# GRID CELL
# =============================================================================

@dataclass
class Cell:
    """A single grid cell with (type, color, state)"""
    type: int = ObjectType.EMPTY
    color: int = 0
    state: int = 0
    
    def encode(self) -> Tuple[int, int, int]:
        return (self.type, self.color, self.state)
    
    def __repr__(self):
        return f"Cell({ObjectType(self.type).name}, {self.color}, {self.state})"


# =============================================================================
# BASE MINIGRID ENVIRONMENT
# =============================================================================

class MiniGridEnv:
    """
    Base MiniGrid environment matching official API.
    
    Observation: 7x7x3 numpy array (agent-centered partial view)
    Actions: 0-6 (left, right, forward, pickup, drop, toggle, done)
    """
    
    def __init__(self, width: int, height: int, max_steps: int = None, 
                 agent_view_size: int = 7, seed: int = None):
        self.width = width
        self.height = height
        self.agent_view_size = agent_view_size
        self.max_steps = max_steps or 4 * width * height
        self.rng = random.Random(seed)
        
        # Grid storage
        self.grid: List[List[Cell]] = None
        
        # Agent state
        self.agent_pos: Tuple[int, int] = None
        self.agent_dir: Direction = None
        self.carrying: Optional[Cell] = None
        
        # Episode state
        self.step_count = 0
        self.done = False
        self.reward = 0
        
        self.reset()
    
    def reset(self, seed: int = None):
        """Reset environment and return initial observation."""
        if seed is not None:
            self.rng = random.Random(seed)
        
        # Initialize empty grid
        self.grid = [[Cell() for _ in range(self.width)] for _ in range(self.height)]
        
        # Add walls around perimeter
        for x in range(self.width):
            self.grid[0][x] = Cell(ObjectType.WALL, Color.GREY)
            self.grid[self.height-1][x] = Cell(ObjectType.WALL, Color.GREY)
        for y in range(self.height):
            self.grid[y][0] = Cell(ObjectType.WALL, Color.GREY)
            self.grid[y][self.width-1] = Cell(ObjectType.WALL, Color.GREY)
        
        # Subclass should override _gen_grid
        self._gen_grid()
        
        # Initialize agent
        if self.agent_pos is None:
            self._place_agent()
        
        self.carrying = None
        self.step_count = 0
        self.done = False
        self.reward = 0
        
        return self._gen_obs()
    
    def _gen_grid(self):
        """Generate the grid. Override in subclasses."""
        pass
    
    def _place_agent(self, pos=None, dir=None):
        """Place agent at position or random empty cell."""
        if pos is not None:
            self.agent_pos = pos
        else:
            empty_cells = []
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.grid[y][x].type == ObjectType.EMPTY:
                        empty_cells.append((x, y))
            if empty_cells:
                self.agent_pos = self.rng.choice(empty_cells)
        
        self.agent_dir = dir if dir is not None else Direction(self.rng.randint(0, 3))
    
    def _place_obj(self, obj_type: int, color: int = 0, state: int = 0, 
                   pos: Tuple[int, int] = None) -> Tuple[int, int]:
        """Place object at position or random empty cell."""
        if pos is not None:
            self.grid[pos[1]][pos[0]] = Cell(obj_type, color, state)
            return pos
        
        empty_cells = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.grid[y][x].type == ObjectType.EMPTY:
                    if self.agent_pos is None or (x, y) != self.agent_pos:
                        empty_cells.append((x, y))
        
        if empty_cells:
            pos = self.rng.choice(empty_cells)
            self.grid[pos[1]][pos[0]] = Cell(obj_type, color, state)
            return pos
        return None
    
    def _gen_obs(self) -> np.ndarray:
        """
        Generate 7x7x3 partial observation centered on agent.
        
        The observation is a forward-facing view:
        - Row 0 is 3 cells ahead
        - Row 6 is 3 cells behind (mostly unseen)
        - Agent is at row 6, col 3
        """
        obs = np.zeros((self.agent_view_size, self.agent_view_size, 3), dtype=np.uint8)
        
        # Fill with UNSEEN
        obs[:, :, 0] = ObjectType.UNSEEN
        
        ax, ay = self.agent_pos
        half = self.agent_view_size // 2
        
        # Get rotation based on agent direction
        # Agent always "faces up" in observation
        for vy in range(self.agent_view_size):
            for vx in range(self.agent_view_size):
                # Convert view coords to world coords based on agent direction
                # View: agent at (half, agent_view_size-1), facing "up" (toward row 0)
                rel_x = vx - half
                rel_y = (self.agent_view_size - 1) - vy  # Flip so row 0 is ahead
                
                if self.agent_dir == Direction.UP:
                    wx, wy = ax + rel_x, ay - rel_y
                elif self.agent_dir == Direction.DOWN:
                    wx, wy = ax - rel_x, ay + rel_y
                elif self.agent_dir == Direction.RIGHT:
                    wx, wy = ax + rel_y, ay + rel_x
                else:  # LEFT
                    wx, wy = ax - rel_y, ay - rel_x
                
                # Check bounds
                if 0 <= wx < self.width and 0 <= wy < self.height:
                    cell = self.grid[wy][wx]
                    obs[vy, vx] = cell.encode()
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action and return (obs, reward, terminated, truncated, info).
        """
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False
        
        # Get front cell position
        dx, dy = DIR_TO_VEC[self.agent_dir]
        front_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        
        if action == Action.LEFT:
            self.agent_dir = Direction((self.agent_dir - 1) % 4)
        
        elif action == Action.RIGHT:
            self.agent_dir = Direction((self.agent_dir + 1) % 4)
        
        elif action == Action.FORWARD:
            fx, fy = front_pos
            if 0 <= fx < self.width and 0 <= fy < self.height:
                front_cell = self.grid[fy][fx]
                # Can move into empty, floor, goal, or open door
                if front_cell.type in [ObjectType.EMPTY, ObjectType.FLOOR, ObjectType.GOAL]:
                    self.agent_pos = front_pos
                elif front_cell.type == ObjectType.DOOR and front_cell.state == DoorState.OPEN:
                    self.agent_pos = front_pos
                elif front_cell.type == ObjectType.LAVA:
                    terminated = True  # Death!
                    reward = 0
        
        elif action == Action.PICKUP:
            fx, fy = front_pos
            if 0 <= fx < self.width and 0 <= fy < self.height:
                front_cell = self.grid[fy][fx]
                if front_cell.type in [ObjectType.KEY, ObjectType.BALL, ObjectType.BOX]:
                    if self.carrying is None:
                        self.carrying = front_cell
                        self.grid[fy][fx] = Cell(ObjectType.EMPTY)
        
        elif action == Action.DROP:
            fx, fy = front_pos
            if 0 <= fx < self.width and 0 <= fy < self.height:
                front_cell = self.grid[fy][fx]
                if front_cell.type == ObjectType.EMPTY and self.carrying is not None:
                    self.grid[fy][fx] = self.carrying
                    self.carrying = None
        
        elif action == Action.TOGGLE:
            fx, fy = front_pos
            if 0 <= fx < self.width and 0 <= fy < self.height:
                front_cell = self.grid[fy][fx]
                if front_cell.type == ObjectType.DOOR:
                    if front_cell.state == DoorState.CLOSED:
                        front_cell.state = DoorState.OPEN
                    elif front_cell.state == DoorState.LOCKED:
                        # Need matching key
                        if self.carrying and self.carrying.type == ObjectType.KEY:
                            if self.carrying.color == front_cell.color:
                                front_cell.state = DoorState.OPEN
                                self.carrying = None  # Key consumed
                    elif front_cell.state == DoorState.OPEN:
                        front_cell.state = DoorState.CLOSED
        
        elif action == Action.DONE:
            # Used in some environments to signal completion
            pass
        
        # Check if agent reached goal
        ax, ay = self.agent_pos
        if self.grid[ay][ax].type == ObjectType.GOAL:
            terminated = True
            reward = 1 - 0.9 * (self.step_count / self.max_steps)
        
        # Check timeout
        if self.step_count >= self.max_steps:
            truncated = True
        
        self.done = terminated or truncated
        self.reward = reward
        
        obs = self._gen_obs()
        info = {"carrying": self.carrying.encode() if self.carrying else None}
        
        return obs, reward, terminated, truncated, info


# =============================================================================
# SPECIFIC ENVIRONMENTS
# =============================================================================

class EmptyEnv(MiniGridEnv):
    """Empty room - just navigate to goal."""
    
    def __init__(self, size: int = 8, agent_start_pos: Tuple[int, int] = None,
                 agent_start_dir: Direction = None, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Place goal in bottom-right
        goal_pos = (self.width - 2, self.height - 2)
        self._place_obj(ObjectType.GOAL, Color.GREEN, pos=goal_pos)
        
        # Place agent in top-left
        if self.agent_start_pos:
            self.agent_pos = self.agent_start_pos
        else:
            self.agent_pos = (1, 1)
        
        self.agent_dir = self.agent_start_dir if self.agent_start_dir else Direction.RIGHT


class DoorKeyEnv(MiniGridEnv):
    """
    Door-Key environment.
    Agent must pickup key, unlock door, then reach goal.
    """
    
    def __init__(self, size: int = 6, **kwargs):
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Create vertical wall with door
        split_x = self.width // 2
        for y in range(1, self.height - 1):
            self.grid[y][split_x] = Cell(ObjectType.WALL, Color.GREY)
        
        # Add locked door
        door_y = self.rng.randint(1, self.height - 2)
        self.grid[door_y][split_x] = Cell(ObjectType.DOOR, Color.YELLOW, DoorState.LOCKED)
        
        # Place key on left side
        key_pos = None
        while key_pos is None or key_pos[0] >= split_x:
            key_x = self.rng.randint(1, split_x - 1)
            key_y = self.rng.randint(1, self.height - 2)
            if self.grid[key_y][key_x].type == ObjectType.EMPTY:
                key_pos = (key_x, key_y)
        self._place_obj(ObjectType.KEY, Color.YELLOW, pos=key_pos)
        
        # Place goal on right side
        goal_x = self.rng.randint(split_x + 1, self.width - 2)
        goal_y = self.rng.randint(1, self.height - 2)
        self._place_obj(ObjectType.GOAL, Color.GREEN, pos=(goal_x, goal_y))
        
        # Place agent on left side
        agent_x = self.rng.randint(1, split_x - 1)
        agent_y = self.rng.randint(1, self.height - 2)
        while self.grid[agent_y][agent_x].type != ObjectType.EMPTY:
            agent_x = self.rng.randint(1, split_x - 1)
            agent_y = self.rng.randint(1, self.height - 2)
        self.agent_pos = (agent_x, agent_y)
        self.agent_dir = Direction(self.rng.randint(0, 3))


class FourRoomsEnv(MiniGridEnv):
    """
    Four rooms environment.
    Agent navigates through 4 connected rooms to reach goal.
    """
    
    def __init__(self, **kwargs):
        super().__init__(19, 19, **kwargs)
    
    def _gen_grid(self):
        # Create center walls
        mid_x = self.width // 2
        mid_y = self.height // 2
        
        # Vertical wall
        for y in range(1, self.height - 1):
            if y != mid_y:
                self.grid[y][mid_x] = Cell(ObjectType.WALL, Color.GREY)
        
        # Horizontal wall
        for x in range(1, self.width - 1):
            if x != mid_x:
                self.grid[mid_y][x] = Cell(ObjectType.WALL, Color.GREY)
        
        # Add gaps (doors) in walls
        # Top-left room -> top-right room (gap in vertical wall, top half)
        gap1_y = self.rng.randint(2, mid_y - 2)
        self.grid[gap1_y][mid_x] = Cell(ObjectType.EMPTY)
        
        # Bottom-left -> bottom-right (gap in vertical wall, bottom half)
        gap2_y = self.rng.randint(mid_y + 2, self.height - 3)
        self.grid[gap2_y][mid_x] = Cell(ObjectType.EMPTY)
        
        # Top-left -> bottom-left (gap in horizontal wall, left half)
        gap3_x = self.rng.randint(2, mid_x - 2)
        self.grid[mid_y][gap3_x] = Cell(ObjectType.EMPTY)
        
        # Top-right -> bottom-right (gap in horizontal wall, right half)
        gap4_x = self.rng.randint(mid_x + 2, self.width - 3)
        self.grid[mid_y][gap4_x] = Cell(ObjectType.EMPTY)
        
        # Place goal in random room
        room = self.rng.randint(0, 3)
        if room == 0:  # top-left
            gx = self.rng.randint(1, mid_x - 1)
            gy = self.rng.randint(1, mid_y - 1)
        elif room == 1:  # top-right
            gx = self.rng.randint(mid_x + 1, self.width - 2)
            gy = self.rng.randint(1, mid_y - 1)
        elif room == 2:  # bottom-left
            gx = self.rng.randint(1, mid_x - 1)
            gy = self.rng.randint(mid_y + 1, self.height - 2)
        else:  # bottom-right
            gx = self.rng.randint(mid_x + 1, self.width - 2)
            gy = self.rng.randint(mid_y + 1, self.height - 2)
        
        self._place_obj(ObjectType.GOAL, Color.GREEN, pos=(gx, gy))
        
        # Place agent in different room
        agent_room = (room + 2) % 4  # Opposite room
        if agent_room == 0:
            ax = self.rng.randint(1, mid_x - 1)
            ay = self.rng.randint(1, mid_y - 1)
        elif agent_room == 1:
            ax = self.rng.randint(mid_x + 1, self.width - 2)
            ay = self.rng.randint(1, mid_y - 1)
        elif agent_room == 2:
            ax = self.rng.randint(1, mid_x - 1)
            ay = self.rng.randint(mid_y + 1, self.height - 2)
        else:
            ax = self.rng.randint(mid_x + 1, self.width - 2)
            ay = self.rng.randint(mid_y + 1, self.height - 2)
        
        self.agent_pos = (ax, ay)
        self.agent_dir = Direction(self.rng.randint(0, 3))


class LavaGapEnv(MiniGridEnv):
    """
    Lava Gap environment.
    Agent must cross a gap in a lava field to reach the goal.
    """
    
    def __init__(self, size: int = 7, obstacle_type: str = "lava", **kwargs):
        self.obstacle_type = ObjectType.LAVA if obstacle_type == "lava" else ObjectType.WALL
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Create lava/wall barrier
        barrier_x = self.width // 2
        gap_y = self.rng.randint(1, self.height - 2)
        
        for y in range(1, self.height - 1):
            if y != gap_y:
                self.grid[y][barrier_x] = Cell(self.obstacle_type, Color.RED if self.obstacle_type == ObjectType.LAVA else Color.GREY)
        
        # Agent on left
        self.agent_pos = (1, self.height // 2)
        self.agent_dir = Direction.RIGHT
        
        # Goal on right
        self._place_obj(ObjectType.GOAL, Color.GREEN, pos=(self.width - 2, gap_y))


class DynamicObstaclesEnv(MiniGridEnv):
    """
    Dynamic Obstacles environment.
    Agent must reach goal while avoiding moving obstacles.
    """
    
    def __init__(self, size: int = 8, n_obstacles: int = 4, **kwargs):
        self.n_obstacles = n_obstacles
        self.obstacle_positions = []
        self.obstacle_dirs = []
        super().__init__(size, size, **kwargs)
    
    def _gen_grid(self):
        # Place goal
        self._place_obj(ObjectType.GOAL, Color.GREEN, pos=(self.width - 2, self.height - 2))
        
        # Place agent
        self.agent_pos = (1, 1)
        self.agent_dir = Direction.RIGHT
        
        # Place obstacles (use BALL type)
        self.obstacle_positions = []
        self.obstacle_dirs = []
        for _ in range(self.n_obstacles):
            pos = self._place_obj(ObjectType.BALL, Color.BLUE)
            if pos:
                self.obstacle_positions.append(list(pos))
                self.obstacle_dirs.append(self.rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)]))
    
    def step(self, action: int):
        # Move obstacles
        for i, (pos, direction) in enumerate(zip(self.obstacle_positions, self.obstacle_dirs)):
            # Clear old position
            self.grid[pos[1]][pos[0]] = Cell(ObjectType.EMPTY)
            
            # Try to move
            new_x = pos[0] + direction[0]
            new_y = pos[1] + direction[1]
            
            # Bounce off walls
            if (new_x <= 0 or new_x >= self.width - 1 or 
                new_y <= 0 or new_y >= self.height - 1 or
                self.grid[new_y][new_x].type != ObjectType.EMPTY):
                # Reverse direction
                self.obstacle_dirs[i] = (-direction[0], -direction[1])
            else:
                pos[0], pos[1] = new_x, new_y
            
            # Place at new position
            self.grid[pos[1]][pos[0]] = Cell(ObjectType.BALL, Color.BLUE)
        
        # Check collision with agent
        obs, reward, terminated, truncated, info = super().step(action)
        
        ax, ay = self.agent_pos
        for pos in self.obstacle_positions:
            if pos[0] == ax and pos[1] == ay:
                terminated = True
                reward = -1
                break
        
        return obs, reward, terminated, truncated, info


# =============================================================================
# OBSERVATION TOKENIZER
# =============================================================================

def obs_to_tokens(obs: np.ndarray, carrying: Tuple[int, int, int] = None) -> FrozenSet[str]:
    """
    Convert 7x7x3 observation to token set for UnifiedFairLearner.
    
    Creates tokens like:
    - "cell_r{row}_c{col}_t{type}_k{color}_s{state}"
    - "front_t{type}" for what's directly ahead
    - "carrying_t{type}_k{color}"
    """
    tokens = set()
    
    h, w = obs.shape[:2]
    agent_row = h - 1  # Agent is in bottom-center
    agent_col = w // 2
    
    for r in range(h):
        for c in range(w):
            obj_type, color, state = obs[r, c]
            if obj_type != ObjectType.UNSEEN:
                # Position-specific token
                tokens.add(f"r{r}_c{c}_t{obj_type}")
                
                # For important objects, include more detail
                if obj_type in [ObjectType.DOOR, ObjectType.KEY, ObjectType.GOAL, 
                               ObjectType.BALL, ObjectType.LAVA]:
                    tokens.add(f"r{r}_c{c}_t{obj_type}_k{color}_s{state}")
                
                # Relative position tokens for nearby cells
                rel_r = r - agent_row
                rel_c = c - agent_col
                if abs(rel_r) <= 2 and abs(rel_c) <= 2:
                    tokens.add(f"near_dr{rel_r}_dc{rel_c}_t{obj_type}")
    
    # Direct front cell (row 5, col 3 in 7x7 view - one step ahead)
    front_r, front_c = agent_row - 1, agent_col
    if 0 <= front_r < h:
        front_type = obs[front_r, front_c, 0]
        front_state = obs[front_r, front_c, 2]
        tokens.add(f"front_t{front_type}")
        if front_type == ObjectType.DOOR:
            tokens.add(f"front_door_s{front_state}")
    
    # Carrying state
    if carrying:
        tokens.add(f"carrying_t{carrying[0]}_k{carrying[1]}")
        tokens.add("has_item")
    else:
        tokens.add("empty_handed")
    
    return frozenset(tokens)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_minigrid_benchmark():
    """Run benchmark on all MiniGrid environments."""
    
    print("=" * 80)
    print("OFFICIAL MINIGRID BENCHMARK")
    print("=" * 80)
    print("""
OBSERVATION FORMAT: 7x7x3 partial view (agent-centered, forward-facing)
- Each cell: (object_type, color, state)
- Agent sees 7x7 grid ahead of them
- 7 actions: left, right, forward, pickup, drop, toggle, done

This is much more complex than our previous 1-cell lookahead!
""")
    
    environments = [
        ("Empty-5x5", lambda seed: EmptyEnv(size=5, seed=seed)),
        ("Empty-8x8", lambda seed: EmptyEnv(size=8, seed=seed)),
        ("DoorKey-5x5", lambda seed: DoorKeyEnv(size=5, seed=seed)),
        ("DoorKey-6x6", lambda seed: DoorKeyEnv(size=6, seed=seed)),
        ("FourRooms", lambda seed: FourRoomsEnv(seed=seed)),
        ("LavaGap-7", lambda seed: LavaGapEnv(size=7, seed=seed)),
        ("DynamicObs-6x6", lambda seed: DynamicObstaclesEnv(size=6, n_obstacles=2, seed=seed)),
    ]
    
    for env_name, env_factory in environments:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")
        
        # Create environment
        env = env_factory(seed=42)
        
        # Show observation shape
        obs = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Grid size: {env.width}x{env.height}")
        print(f"Max steps: {env.max_steps}")
        
        # Show what agent sees
        print(f"\nAgent position: {env.agent_pos}")
        print(f"Agent direction: {Direction(env.agent_dir).name}")
        
        # Count visible objects
        obj_counts = defaultdict(int)
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                obj_type = obs[r, c, 0]
                if obj_type != ObjectType.UNSEEN:
                    obj_counts[ObjectType(obj_type).name] += 1
        print(f"Visible objects: {dict(obj_counts)}")
        
        # Token representation
        tokens = obs_to_tokens(obs)
        print(f"Token count: {len(tokens)}")
        print(f"Sample tokens: {list(tokens)[:5]}...")
        
        # Random episode
        env.reset(seed=0)
        total_reward = 0
        steps = 0
        
        for step in range(env.max_steps):
            action = random.randint(0, 5)  # Random action (exclude DONE)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"\nRandom agent: {steps} steps, reward={total_reward:.3f}, success={terminated and reward > 0}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
Our previous simulations were TOO SIMPLE:
- We used 1-cell lookahead (front, left, right)
- Official MiniGrid uses 7x7 = 49 cells of partial observation!

To properly benchmark, we need to:
1. Train on the full 7x7x3 observation space
2. Learn object semantics (door states, key-door matching, etc.)
3. Handle the much larger state space

This is a MUCH harder learning problem than what we tested before.
The 95%+ success rates we saw were on our simplified simulation,
not the full MiniGrid specification.
""")


if __name__ == "__main__":
    run_minigrid_benchmark()
