"""
HONEST MINIGRID RULE LEARNER

Discovers game rules from completely anonymized observations.
Agent sees only local egocentric view, must discover:
- Turn = cyclic rotation of direction
- Forward = conditional movement (blocked by walls/doors)
- Pickup = acquire object if hands empty and object present
- Toggle = change door state
- Geometric structure of space (from accumulated transitions)

This implements the approach from space.txt - showing how spatial
understanding can emerge from simple causal rules.
"""

import random
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional


# =============================================================================
# MINIGRID WORLD (Oracle)
# =============================================================================

class Tile(Enum):
    FLOOR = 0
    WALL = 1
    GOAL = 2
    LAVA = 3
    DOOR_LOCKED = 4
    DOOR_CLOSED = 5
    DOOR_OPEN = 6

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5

DIRECTION_DELTAS = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}


class MinigridWorld:
    """
    Simplified Minigrid environment.
    Agent navigates grid with walls, doors, keys.
    """
    
    def __init__(self, width=7, height=7, seed=None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self):
        # Create grid
        self.grid = [[Tile.FLOOR for _ in range(self.width)] 
                     for _ in range(self.height)]
        
        # Walls around edges
        for x in range(self.width):
            self.grid[0][x] = Tile.WALL
            self.grid[self.height-1][x] = Tile.WALL
        for y in range(self.height):
            self.grid[y][0] = Tile.WALL
            self.grid[y][self.width-1] = Tile.WALL
        
        # Random internal walls
        for _ in range(self.rng.randint(0, 3)):
            wx = self.rng.randint(2, self.width-3)
            wy = self.rng.randint(2, self.height-3)
            self.grid[wy][wx] = Tile.WALL
        
        # Maybe add door
        if self.rng.random() < 0.3:
            dx = self.rng.randint(2, self.width-3)
            dy = self.rng.randint(2, self.height-3)
            if self.grid[dy][dx] == Tile.FLOOR:
                self.grid[dy][dx] = self.rng.choice([Tile.DOOR_LOCKED, Tile.DOOR_CLOSED])
        
        # Goal
        self.grid[self.height-2][self.width-2] = Tile.GOAL
        
        # Agent position
        self.agent_x = self.rng.randint(1, self.width-2)
        self.agent_y = self.rng.randint(1, self.height-2)
        while self.grid[self.agent_y][self.agent_x] != Tile.FLOOR:
            self.agent_x = self.rng.randint(1, self.width-2)
            self.agent_y = self.rng.randint(1, self.height-2)
        
        self.agent_dir = self.rng.choice(list(Direction))
        self.carrying = None
        
        # Objects
        self.objects = {}
        if self.rng.random() < 0.4:
            ox = self.rng.randint(1, self.width-2)
            oy = self.rng.randint(1, self.height-2)
            if (self.grid[oy][ox] == Tile.FLOOR and 
                (ox, oy) != (self.agent_x, self.agent_y)):
                self.objects[(ox, oy)] = "key"
        
        self.done = False
    
    def get_front_pos(self):
        dx, dy = DIRECTION_DELTAS[self.agent_dir]
        return (self.agent_x + dx, self.agent_y + dy)
    
    def get_tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return Tile.WALL
    
    def is_walkable(self, x, y):
        tile = self.get_tile(x, y)
        if tile in [Tile.WALL, Tile.DOOR_LOCKED, Tile.DOOR_CLOSED]:
            return False
        if (x, y) in self.objects:
            return False
        return True
    
    def get_observation(self) -> Dict[str, str]:
        """Get egocentric observation (what learner sees)."""
        dx, dy = DIRECTION_DELTAS[self.agent_dir]
        fx, fy = self.agent_x + dx, self.agent_y + dy
        
        # Left direction
        left_dir = Direction((self.agent_dir.value - 1) % 4)
        ldx, ldy = DIRECTION_DELTAS[left_dir]
        lx, ly = self.agent_x + ldx, self.agent_y + ldy
        
        # Right direction
        right_dir = Direction((self.agent_dir.value + 1) % 4)
        rdx, rdy = DIRECTION_DELTAS[right_dir]
        rx, ry = self.agent_x + rdx, self.agent_y + rdy
        
        return {
            "direction": self.agent_dir.name,
            "front_tile": self.get_tile(fx, fy).name,
            "left_tile": self.get_tile(lx, ly).name,
            "right_tile": self.get_tile(rx, ry).name,
            "front_object": self.objects.get((fx, fy), "none"),
            "carrying": self.carrying or "none",
        }
    
    def step(self, action: Action) -> Tuple[Dict, float, bool]:
        """Execute action, return (observation, reward, done)."""
        reward = 0.0
        
        if action == Action.TURN_LEFT:
            self.agent_dir = Direction((self.agent_dir.value - 1) % 4)
        
        elif action == Action.TURN_RIGHT:
            self.agent_dir = Direction((self.agent_dir.value + 1) % 4)
        
        elif action == Action.FORWARD:
            fx, fy = self.get_front_pos()
            if self.is_walkable(fx, fy):
                self.agent_x = fx
                self.agent_y = fy
                
                tile = self.get_tile(fx, fy)
                if tile == Tile.GOAL:
                    self.done = True
                    reward = 1.0
                elif tile == Tile.LAVA:
                    self.done = True
                    reward = -1.0
        
        elif action == Action.PICKUP:
            fx, fy = self.get_front_pos()
            if (fx, fy) in self.objects and self.carrying is None:
                self.carrying = self.objects[(fx, fy)]
                del self.objects[(fx, fy)]
        
        elif action == Action.DROP:
            fx, fy = self.get_front_pos()
            tile = self.get_tile(fx, fy)
            if self.carrying and tile == Tile.FLOOR and (fx, fy) not in self.objects:
                self.objects[(fx, fy)] = self.carrying
                self.carrying = None
        
        elif action == Action.TOGGLE:
            fx, fy = self.get_front_pos()
            tile = self.get_tile(fx, fy)
            if tile == Tile.DOOR_CLOSED:
                self.grid[fy][fx] = Tile.DOOR_OPEN
            elif tile == Tile.DOOR_OPEN:
                self.grid[fy][fx] = Tile.DOOR_CLOSED
        
        return self.get_observation(), reward, self.done


# =============================================================================
# ANONYMIZER
# =============================================================================

class Anonymizer:
    """Convert observations to anonymous tokens."""
    
    def __init__(self):
        self.var_map = {}
        self.val_maps = defaultdict(dict)
        self.action_map = {}
        self.next_var = 0
        self.next_action = 0
    
    def anonymize(self, obs: Dict, action: str) -> Dict:
        """Anonymize observation and action."""
        anon = {}
        
        for key, val in obs.items():
            if key not in self.var_map:
                self.var_map[key] = f"V{self.next_var}"
                self.next_var += 1
            
            vid = self.var_map[key]
            
            if val not in self.val_maps[vid]:
                self.val_maps[vid][val] = f"{vid}_{chr(65 + len(self.val_maps[vid]))}"
            
            anon[vid] = self.val_maps[vid][val]
        
        if action not in self.action_map:
            self.action_map[action] = f"A{self.next_action}"
            self.next_action += 1
        
        anon["action"] = self.action_map[action]
        
        return anon


# =============================================================================
# RULE LEARNER
# =============================================================================

class RuleLearner:
    """
    Discovers rules from anonymous observations.
    """
    
    def __init__(self, min_support=5, min_confidence=0.9):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.observations = []
        self.rules = []
    
    def observe(self, before: Dict, action: str, after: Dict):
        """Record a transition."""
        self.observations.append({
            "before": before,
            "action": action,
            "after": after,
        })
    
    def find_rules(self, input_vars: List[str], output_var: str) -> List[Dict]:
        """Find rules mapping input_vars -> output_var."""
        groups = defaultdict(lambda: defaultdict(int))
        
        for obs in self.observations:
            # Build input key
            key_parts = []
            for v in input_vars:
                if v == "action":
                    key_parts.append(obs["action"])
                elif v.startswith("before_"):
                    real_v = v[7:]
                    key_parts.append(obs["before"].get(real_v, "?"))
                else:
                    key_parts.append(obs["before"].get(v, "?"))
            
            key = tuple(key_parts)
            
            # Get output
            if output_var.startswith("after_"):
                real_v = output_var[6:]
                out = obs["after"].get(real_v, "?")
            else:
                out = obs["after"].get(output_var, "?")
            
            groups[key][out] += 1
        
        # Find high-confidence rules
        rules = []
        for key, outcomes in groups.items():
            total = sum(outcomes.values())
            if total < self.min_support:
                continue
            
            for out, count in outcomes.items():
                conf = count / total
                if conf >= self.min_confidence:
                    rules.append({
                        "input_vars": input_vars,
                        "input_vals": key,
                        "output_var": output_var,
                        "output_val": out,
                        "confidence": conf,
                        "support": count,
                    })
        
        return sorted(rules, key=lambda r: (-r["confidence"], -r["support"]))
    
    def discover_all_rules(self) -> List[Dict]:
        """Attempt to discover all major rules."""
        all_rules = []
        
        # Get variable names from first observation
        if not self.observations:
            return []
        
        before_vars = list(self.observations[0]["before"].keys())
        after_vars = list(self.observations[0]["after"].keys())
        
        # Try each output variable
        for out_var in after_vars:
            # Try with just action
            rules = self.find_rules(["action"], f"after_{out_var}")
            all_rules.extend(rules)
            
            # Try with action + relevant before vars
            for in_var in before_vars:
                rules = self.find_rules(["action", f"before_{in_var}"], f"after_{out_var}")
                all_rules.extend(rules)
            
            # Try with two before vars + action
            for v1 in before_vars:
                for v2 in before_vars:
                    if v1 < v2:
                        rules = self.find_rules(
                            ["action", f"before_{v1}", f"before_{v2}"], 
                            f"after_{out_var}"
                        )
                        all_rules.extend(rules)
        
        self.rules = all_rules
        return all_rules


# =============================================================================
# MAIN
# =============================================================================

def generate_observations(n_episodes=100, max_steps=50, seed=42):
    """Generate observations from random play."""
    rng = random.Random(seed)
    anonymizer = Anonymizer()
    learner = RuleLearner()
    
    for ep in range(n_episodes):
        world = MinigridWorld(7, 7, seed=rng.randint(0, 99999))
        
        for step in range(max_steps):
            if world.done:
                break
            
            before = world.get_observation()
            action = rng.choice(list(Action))
            after, reward, done = world.step(action)
            
            # Anonymize and record
            anon_before = anonymizer.anonymize(before, "")
            del anon_before["action"]  # Remove dummy action
            anon_action = anonymizer.action_map.get(action.name, f"A{len(anonymizer.action_map)}")
            if action.name not in anonymizer.action_map:
                anonymizer.action_map[action.name] = anon_action
            anon_after = anonymizer.anonymize(after, "")
            del anon_after["action"]
            
            learner.observe(anon_before, anon_action, anon_after)
    
    return learner, anonymizer


if __name__ == "__main__":
    print("=" * 60)
    print("MINIGRID RULE DISCOVERY")
    print("=" * 60)
    
    learner, anonymizer = generate_observations(200, 40)
    print(f"\nCollected {len(learner.observations)} observations")
    
    print("\n--- Anonymization ---")
    print(f"Variables: {anonymizer.var_map}")
    print(f"Actions: {anonymizer.action_map}")
    
    print("\n--- Discovering rules ---")
    rules = learner.discover_all_rules()
    
    # Filter to interesting rules (where output != input)
    interesting = []
    for r in rules:
        # Check if this is a "something changed" rule
        is_identity = False
        for i, v in enumerate(r["input_vars"]):
            if v.replace("before_", "") == r["output_var"].replace("after_", ""):
                if r["input_vals"][i] == r["output_val"]:
                    is_identity = True
                    break
        
        if not is_identity or r["confidence"] < 1.0:
            interesting.append(r)
    
    print(f"\nFound {len(rules)} rules, {len(interesting)} interesting")
    
    for r in interesting[:20]:
        in_str = ", ".join(f"{v}={val}" for v, val in zip(r["input_vars"], r["input_vals"]))
        print(f"  {in_str} => {r['output_var']}={r['output_val']} "
              f"(conf={r['confidence']:.0%}, n={r['support']})")
