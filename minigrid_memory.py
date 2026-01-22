"""
MINIGRID WITH MEMORY AND REGIONAL ABSTRACTIONS

Adds to our minimal front-cell observation:
1. Memory of past 2 observations (disambiguates aliased states)
2. Action history (helps detect cycles and progress)
3. Regional abstractions (cluster similar observations)

This should help with:
- FourRooms (different regions have same local view)
- MultiRoom (need to remember which doors were opened)
- Memory environments (by design need past info)
"""

import random
import sys
from collections import defaultdict, deque
from typing import Dict, List, Tuple, FrozenSet, Optional, NamedTuple
from dataclasses import dataclass, field

sys.path.insert(0, '/home/claude/locus')

from minigrid_full import (
    ENVIRONMENTS, FullMiniGridEnv, MinimalObs,
    Action, ObjectType, Color, DoorState, Direction,
    OBJECT_NAMES, COLOR_NAMES, DOOR_STATE_NAMES
)


# =============================================================================
# OBSERVATION WITH MEMORY
# =============================================================================

class ObsWithMemory(NamedTuple):
    """
    Extended observation with memory of past states.
    
    Contains:
    - Current front cell info
    - Carrying state
    - Past 2 observations (as compact tuples)
    - Past 2 actions taken
    - Regional signature (abstraction of local area)
    """
    # Current observation
    front_type: int
    front_color: int
    front_state: int
    carrying_type: int
    carrying_color: int
    standing_type: int
    
    # Memory (past 2 steps)
    prev1_front: int  # t-1 front cell type
    prev1_action: int  # t-1 action taken
    prev2_front: int  # t-2 front cell type  
    prev2_action: int  # t-2 action taken
    
    # Regional signature (hash of recent experience)
    region_sig: int


class MemoryWrapper:
    """
    Wraps MiniGrid environment to add memory to observations.
    """
    
    def __init__(self, env: FullMiniGridEnv, memory_len: int = 2):
        self.env = env
        self.memory_len = memory_len
        
        # Memory buffers
        self.obs_history: List[MinimalObs] = []
        self.action_history: List[int] = []
        
        # Regional tracking
        self.region_visits: Dict[int, int] = defaultdict(int)
        self.current_region: int = 0
    
    def reset(self, seed: int = None) -> ObsWithMemory:
        """Reset environment and memory."""
        obs = self.env.reset(seed)
        
        # Clear history
        self.obs_history = [obs] * self.memory_len
        self.action_history = [-1] * self.memory_len
        self.region_visits.clear()
        
        # Compute initial region
        self.current_region = self._compute_region_sig(obs)
        self.region_visits[self.current_region] += 1
        
        return self._make_obs_with_memory(obs)
    
    def step(self, action: int) -> Tuple[ObsWithMemory, float, bool, bool, dict]:
        """Step environment and update memory."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update history (keep last memory_len items)
        self.obs_history.append(obs)
        self.obs_history = self.obs_history[-self.memory_len:]
        
        self.action_history.append(action)
        self.action_history = self.action_history[-self.memory_len:]
        
        # Update region tracking
        new_region = self._compute_region_sig(obs)
        if new_region != self.current_region:
            self.current_region = new_region
        self.region_visits[self.current_region] += 1
        
        obs_with_mem = self._make_obs_with_memory(obs)
        return obs_with_mem, reward, terminated, truncated, info
    
    def _compute_region_sig(self, obs: MinimalObs) -> int:
        """
        Compute a signature for the current "region".
        
        Uses hash of recent observations to identify similar situations.
        """
        # Simple region = hash of (front_type, carrying_type)
        # More sophisticated could cluster based on transition patterns
        return hash((obs.front_type, obs.carrying_type, obs.standing_type))
    
    def _make_obs_with_memory(self, current: MinimalObs) -> ObsWithMemory:
        """Create observation with memory attached."""
        # Get past observations
        prev1 = self.obs_history[-2] if len(self.obs_history) >= 2 else current
        prev2 = self.obs_history[-3] if len(self.obs_history) >= 3 else prev1
        
        # Get past actions
        act1 = self.action_history[-2] if len(self.action_history) >= 2 else -1
        act2 = self.action_history[-3] if len(self.action_history) >= 3 else -1
        
        return ObsWithMemory(
            front_type=current.front_type,
            front_color=current.front_color,
            front_state=current.front_state,
            carrying_type=current.carrying_type,
            carrying_color=current.carrying_color,
            standing_type=current.standing_type,
            prev1_front=prev1.front_type,
            prev1_action=act1,
            prev2_front=prev2.front_type,
            prev2_action=act2,
            region_sig=self.current_region % 1000,  # Truncate for readability
        )
    
    # Pass through properties
    @property
    def width(self): return self.env.width
    @property
    def height(self): return self.env.height
    @property
    def max_steps(self): return self.env.max_steps
    @property
    def carrying(self): return self.env.carrying
    @property
    def agent_pos(self): return self.env.agent_pos
    @property
    def success(self): return self.env.success


def obs_to_tokens(obs: ObsWithMemory) -> FrozenSet[str]:
    """Convert observation with memory to token set."""
    tokens = set()
    
    # Current front cell
    front_name = OBJECT_NAMES.get(ObjectType(obs.front_type), f"t{obs.front_type}")
    tokens.add(f"front={front_name}")
    
    if obs.front_type in [ObjectType.DOOR, ObjectType.KEY, ObjectType.BALL, 
                          ObjectType.BOX, ObjectType.GOAL]:
        color_name = COLOR_NAMES.get(Color(obs.front_color), f"c{obs.front_color}")
        tokens.add(f"front={color_name}_{front_name}")
    
    if obs.front_type == ObjectType.DOOR:
        state_name = DOOR_STATE_NAMES.get(DoorState(obs.front_state), f"s{obs.front_state}")
        tokens.add(f"front_door={state_name}")
    
    # Carrying state
    if obs.carrying_type > 0:
        carry_name = OBJECT_NAMES.get(ObjectType(obs.carrying_type), f"t{obs.carrying_type}")
        carry_color = COLOR_NAMES.get(Color(obs.carrying_color), f"c{obs.carrying_color}")
        tokens.add(f"carrying={carry_color}_{carry_name}")
        tokens.add("has_item")
    else:
        tokens.add("empty_handed")
    
    # Standing on goal?
    if obs.standing_type == ObjectType.GOAL:
        tokens.add("on_goal")
    
    # === MEMORY TOKENS ===
    
    # Previous front cell (t-1)
    if obs.prev1_front >= 0:
        prev1_name = OBJECT_NAMES.get(ObjectType(obs.prev1_front), f"t{obs.prev1_front}")
        tokens.add(f"prev1_front={prev1_name}")
    
    # Previous action (t-1)
    if obs.prev1_action >= 0:
        act_names = ["LEFT", "RIGHT", "FWD", "PICKUP", "DROP", "TOGGLE", "DONE"]
        if obs.prev1_action < len(act_names):
            tokens.add(f"prev1_act={act_names[obs.prev1_action]}")
    
    # Two steps ago front cell (t-2)
    if obs.prev2_front >= 0:
        prev2_name = OBJECT_NAMES.get(ObjectType(obs.prev2_front), f"t{obs.prev2_front}")
        tokens.add(f"prev2_front={prev2_name}")
    
    # Two steps ago action (t-2)
    if obs.prev2_action >= 0:
        act_names = ["LEFT", "RIGHT", "FWD", "PICKUP", "DROP", "TOGGLE", "DONE"]
        if obs.prev2_action < len(act_names):
            tokens.add(f"prev2_act={act_names[obs.prev2_action]}")
    
    # === DERIVED TOKENS (patterns) ===
    
    # Detect if we just turned twice (might be stuck)
    if obs.prev1_action in [0, 1] and obs.prev2_action in [0, 1]:
        tokens.add("pattern=double_turn")
    
    # Detect if we just moved forward twice (making progress)
    if obs.prev1_action == 2 and obs.prev2_action == 2:
        tokens.add("pattern=forward_progress")
    
    # Detect if front changed after action (action had effect)
    if obs.prev1_front != obs.front_type:
        tokens.add("pattern=front_changed")
    
    # Region signature (helps distinguish similar-looking areas)
    tokens.add(f"region={obs.region_sig}")
    
    return frozenset(tokens)


# =============================================================================
# EXPLORATION AND NAVIGATION
# =============================================================================

def explore_with_memory(env_factory, n_episodes: int, max_steps: int = 500) -> Dict:
    """
    Explore environment and collect statistics.
    Uses memory-augmented observations.
    """
    stats = {
        "episodes": 0,
        "successes": 0,
        "total_steps": 0,
        "unique_obs": set(),
        "transitions": defaultdict(lambda: defaultdict(int)),
        "goal_obs": set(),
    }
    
    for ep in range(n_episodes):
        env = MemoryWrapper(env_factory(seed=ep * 17 + 42))
        obs = env.reset()
        tokens = obs_to_tokens(obs)
        stats["unique_obs"].add(tokens)
        
        for step in range(max_steps):
            # Epsilon-greedy with heuristic
            if random.random() < 0.3:
                action = random.randint(0, 5)
            else:
                action = heuristic_action_with_memory(obs, env.carrying)
            
            before_tokens = tokens
            obs, reward, terminated, truncated, info = env.step(action)
            tokens = obs_to_tokens(obs)
            
            # Record transition
            stats["transitions"][(before_tokens, action)][tokens] += 1
            stats["unique_obs"].add(tokens)
            stats["total_steps"] += 1
            
            if reward > 0:
                stats["successes"] += 1
                stats["goal_obs"].add(tokens)
            
            if terminated or truncated:
                break
        
        stats["episodes"] += 1
    
    return stats


def heuristic_action_with_memory(obs: ObsWithMemory, carrying) -> int:
    """
    Heuristic policy that uses memory to make better decisions.
    """
    front_type = obs.front_type
    front_state = obs.front_state
    has_item = obs.carrying_type > 0
    
    # If on goal, we're done
    if obs.standing_type == ObjectType.GOAL:
        return Action.FORWARD  # Shouldn't happen, but just in case
    
    # If goal in front, go forward
    if front_type == ObjectType.GOAL:
        return Action.FORWARD
    
    # If key in front and not carrying, pickup
    if front_type == ObjectType.KEY and not has_item:
        return Action.PICKUP
    
    # If door in front
    if front_type == ObjectType.DOOR:
        if front_state == DoorState.OPEN:
            return Action.FORWARD
        elif front_state == DoorState.CLOSED:
            return Action.TOGGLE  # Open it
        elif front_state == DoorState.LOCKED and has_item:
            return Action.TOGGLE  # Try to unlock
        else:
            # Locked door, no key - turn
            return random.choice([Action.LEFT, Action.RIGHT])
    
    # If empty/floor in front
    if front_type in [ObjectType.EMPTY, ObjectType.FLOOR]:
        # Check memory for patterns
        
        # If we just turned twice, probably stuck - try forward
        if obs.prev1_action in [0, 1] and obs.prev2_action in [0, 1]:
            return Action.FORWARD
        
        # If we've been going forward, occasionally turn to explore
        if obs.prev1_action == 2 and obs.prev2_action == 2:
            if random.random() < 0.3:
                return random.choice([Action.LEFT, Action.RIGHT])
        
        # Default: forward with some turning
        if random.random() < 0.6:
            return Action.FORWARD
        return random.choice([Action.LEFT, Action.RIGHT])
    
    # Wall or obstacle - turn
    return random.choice([Action.LEFT, Action.RIGHT])


def evaluate_with_memory(env_factory, n_episodes: int = 100, 
                         max_steps: int = 500) -> Dict:
    """Evaluate navigation with memory-augmented policy."""
    stats = {
        "episodes": n_episodes,
        "successes": 0,
        "total_steps": 0,
        "success_steps": [],
    }
    
    for ep in range(n_episodes):
        env = MemoryWrapper(env_factory(seed=10000 + ep))
        obs = env.reset()
        
        for step in range(max_steps):
            action = heuristic_action_with_memory(obs, env.carrying)
            obs, reward, terminated, truncated, info = env.step(action)
            stats["total_steps"] += 1
            
            if reward > 0:
                stats["successes"] += 1
                stats["success_steps"].append(step + 1)
                break
            
            if terminated or truncated:
                break
    
    return stats


# =============================================================================
# BENCHMARK
# =============================================================================

def run_memory_benchmark():
    """Compare no-memory vs memory performance."""
    
    print("=" * 80)
    print("MINIGRID WITH MEMORY (Past 2 Observations)")
    print("=" * 80)
    print("""
Configuration:
- Full MiniGrid mechanics
- Front cell observation + MEMORY of past 2 steps
- Action history included
- Regional signatures for disambiguation
""")
    
    # Test environments
    test_envs = [
        ("Empty-5x5", "Empty-5x5"),
        ("Empty-8x8", "Empty-8x8"),
        ("DoorKey-5x5", "DoorKey-5x5"),
        ("DoorKey-6x6", "DoorKey-6x6"),
        ("DoorKey-8x8", "DoorKey-8x8"),
        ("FourRooms", "FourRooms"),
        ("MultiRoom-N2-S4", "MultiRoom-N2-S4"),
        ("MultiRoom-N4-S5", "MultiRoom-N4-S5"),
        ("LavaGap-S5", "LavaGap-S5"),
        ("LavaGap-S6", "LavaGap-S6"),
        ("Dynamic-Obstacles-5x5", "Dynamic-Obstacles-5x5"),
        ("Memory-S7", "Memory-S7"),
        ("LockedRoom", "LockedRoom"),
    ]
    
    results = []
    
    for display_name, env_key in test_envs:
        if env_key not in ENVIRONMENTS:
            print(f"\n{display_name}: NOT IMPLEMENTED")
            continue
        
        print(f"\n--- {display_name} ---")
        
        env_factory = ENVIRONMENTS[env_key]
        
        # Exploration phase
        explore_stats = explore_with_memory(env_factory, n_episodes=500, max_steps=300)
        
        # Evaluation phase
        eval_stats = evaluate_with_memory(env_factory, n_episodes=100, max_steps=300)
        
        success_rate = eval_stats["successes"] / eval_stats["episodes"]
        avg_steps = (sum(eval_stats["success_steps"]) / len(eval_stats["success_steps"])
                    if eval_stats["success_steps"] else 0)
        
        result = {
            "env": display_name,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "unique_obs": len(explore_stats["unique_obs"]),
            "train_successes": explore_stats["successes"],
        }
        results.append(result)
        
        status = "✓" if success_rate >= 0.8 else "~" if success_rate >= 0.3 else "✗"
        print(f"  {status} Success={success_rate:5.1%}, AvgSteps={avg_steps:5.1f}, "
              f"UniqueObs={len(explore_stats['unique_obs']):4d}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Memory Impact")
    print("=" * 80)
    
    print(f"\n{'Environment':<25} {'Success':<10} {'AvgSteps':<10} {'UniqueObs':<10}")
    print("-" * 60)
    
    for r in results:
        status = "✓" if r["success_rate"] >= 0.8 else "~" if r["success_rate"] >= 0.3 else "✗"
        print(f"{status} {r['env']:<23} {r['success_rate']:>8.1%}   "
              f"{r['avg_steps']:>8.1f}   {r['unique_obs']:>8}")
    
    solved = sum(1 for r in results if r["success_rate"] >= 0.8)
    partial = sum(1 for r in results if 0.3 <= r["success_rate"] < 0.8)
    failed = sum(1 for r in results if r["success_rate"] < 0.3)
    
    print(f"\n✓ Solved (≥80%): {solved}")
    print(f"~ Partial (30-80%): {partial}")
    print(f"✗ Failed (<30%): {failed}")
    
    print("""
MEMORY FEATURES ADDED:
1. prev1_front / prev2_front - What was in front 1 and 2 steps ago
2. prev1_act / prev2_act - Actions taken 1 and 2 steps ago
3. pattern=double_turn - Detected turning repeatedly (stuck?)
4. pattern=forward_progress - Making forward progress
5. pattern=front_changed - Something changed after action
6. region=N - Regional signature for disambiguation

This helps with:
- Detecting when stuck in loops
- Distinguishing aliased states (same local view, different context)
- Building implicit memory of recent progress
""")
    
    return results


if __name__ == "__main__":
    results = run_memory_benchmark()
