"""
Simple GridWorld - Local MiniGrid-like environment

Tests:
- Spatial abstraction (fidelity)
- Object interactions (rules)
- Goal-directed behavior
- Partial observability (optional)
"""
import random
from typing import Set, Tuple

class SimpleGridWorld:
    """
    Grid with agent, walls, keys, doors, and goal.
    
    Objects:
    - Agent (@)
    - Wall (#)
    - Key (k) - pick up by walking over
    - Door (D) - opens if have key
    - Goal (G) - win condition
    
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    def __init__(self, seed=42, size=5):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        
        self.grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # Place walls on edges
        for i in range(self.size):
            self.grid[0][i] = '#'
            self.grid[self.size-1][i] = '#'
            self.grid[i][0] = '#'
            self.grid[i][self.size-1] = '#'
        
        # Place agent
        self.agent_x = 1
        self.agent_y = 1
        
        # Place key, door, goal
        empty = [(x, y) for x in range(1, self.size-1) 
                 for y in range(1, self.size-1) 
                 if (x, y) != (self.agent_x, self.agent_y)]
        
        self.rng.shuffle(empty)
        
        self.key_x, self.key_y = empty[0]
        self.door_x, self.door_y = empty[1]
        self.goal_x, self.goal_y = empty[2]
        
        self.has_key = False
        self.door_open = False
        self.done = False
        self.won = False
        
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        """Generate symbolic state."""
        tokens = set()
        
        # Agent position
        tokens.add(f"agent_{self.agent_x}_{self.agent_y}")
        
        # Key status
        if not self.has_key:
            tokens.add(f"key_{self.key_x}_{self.key_y}")
        else:
            tokens.add("has_key")
        
        # Door status
        if not self.door_open:
            tokens.add(f"door_closed_{self.door_x}_{self.door_y}")
        else:
            tokens.add(f"door_open_{self.door_x}_{self.door_y}")
        
        # Goal
        tokens.add(f"goal_{self.goal_x}_{self.goal_y}")
        
        # Relative positions (for spatial abstraction)
        dx = self.goal_x - self.agent_x
        dy = self.goal_y - self.agent_y
        tokens.add(f"goal_dx_{dx}")
        tokens.add(f"goal_dy_{dy}")
        
        if not self.has_key:
            kx = self.key_x - self.agent_x
            ky = self.key_y - self.agent_y
            tokens.add(f"key_dx_{kx}")
            tokens.add(f"key_dy_{ky}")
        
        tokens.add(f"done_{self.done}")
        
        return tokens
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, True, {}
        
        # Movement deltas
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy
        
        # Check bounds and walls
        if (0 <= new_x < self.size and 0 <= new_y < self.size and 
            self.grid[new_y][new_x] != '#'):
            
            # Check door
            if (new_x, new_y) == (self.door_x, self.door_y) and not self.door_open:
                if self.has_key:
                    self.door_open = True
                else:
                    return self._get_state(), -0.1, False, {}  # Blocked
            
            # Move
            self.agent_x, self.agent_y = new_x, new_y
            
            # Pick up key
            if (new_x, new_y) == (self.key_x, self.key_y) and not self.has_key:
                self.has_key = True
            
            # Check goal
            if (new_x, new_y) == (self.goal_x, self.goal_y):
                self.done = True
                self.won = True
                return self._get_state(), 1.0, True, {}
        
        return self._get_state(), -0.01, False, {}

print("SimpleGridWorld loaded!")
