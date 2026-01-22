"""
CONFIGURABLE MINIGRID AGENT

Features:
1. View modes: FRONT (1), THREE_CELL (3), FULL_7X7 (49)
2. Memory depth: 0-7 steps
3. Correlation discovery (e.g., color matching)
4. Probabilistic outcome tracking

Defaults to minimal config, scales up when needed.
"""

import random
from collections import defaultdict
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Dict, Set, Tuple, List
import sys

sys.path.insert(0, '/home/claude/locus')

from minigrid_full import (
    ENVIRONMENTS, Action, ObjectType, DoorState, Color,
    Direction, DIR_TO_VEC, OBJECT_NAMES, COLOR_NAMES
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ViewMode(IntEnum):
    FRONT_ONLY = 1    # Just front cell
    THREE_CELL = 3    # Front + left + right  
    FULL_7X7 = 49     # Full 7x7 grid (as hash)


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    view_mode: ViewMode = ViewMode.FRONT_ONLY
    memory_depth: int = 0  # 0 = no memory, up to 7
    use_correlations: bool = True  # Discover equivalence classes
    explore_rate: float = 0.2  # Random action probability


# =============================================================================
# AGENT
# =============================================================================

class ConfigurableAgent:
    """
    Agent with configurable observation and memory depth.
    
    Automatically discovers correlations like color matching.
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        # Memory buffers
        self.obs_history: List[dict] = []
        self.action_history: List[int] = []
        
        # Learned outcomes: (state_key, action) -> [wins, total]
        self.outcomes: Dict[Tuple, List[int]] = defaultdict(lambda: [0, 0])
        
        # Correlation tracking: (feature_combo) -> [wins, total]
        self.correlations: Dict[Tuple, List[int]] = defaultdict(lambda: [0, 0])
        
        # Discovered equivalence rules
        self.equiv_rules: Set[str] = set()
    
    def reset(self):
        """Reset episode state (keep learned knowledge)."""
        self.obs_history = []
        self.action_history = []
    
    # -------------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------------
    
    def get_obs(self, env) -> dict:
        """Get observation based on view mode."""
        ax, ay = env.agent_pos
        d = env.agent_dir
        dx, dy = DIR_TO_VEC[d]
        
        obs = {
            'front': self._encode_cell(env._get(ax + dx, ay + dy)),
            'carrying': self._encode_item(env.carrying),
            'on_goal': (env._get(ax, ay) is not None and 
                       env._get(ax, ay).type == ObjectType.GOAL),
        }
        
        # 3-cell view
        if self.config.view_mode >= ViewMode.THREE_CELL:
            ld = Direction((d - 1) % 4)
            ldx, ldy = DIR_TO_VEC[ld]
            obs['left'] = self._encode_cell(env._get(ax + ldx, ay + ldy))
            
            rd = Direction((d + 1) % 4)
            rdx, rdy = DIR_TO_VEC[rd]
            obs['right'] = self._encode_cell(env._get(ax + rdx, ay + rdy))
        
        # Full 7x7 view hash
        if self.config.view_mode >= ViewMode.FULL_7X7:
            view_data = []
            for vy in range(-3, 4):
                for vx in range(-3, 4):
                    cell = env._get(ax + vx, ay + vy)
                    view_data.append(self._encode_cell(cell))
            obs['view_hash'] = hash(tuple(view_data)) % 100000
        
        return obs
    
    def _encode_cell(self, obj) -> Tuple[int, int, int]:
        """Encode cell as (type, color, state)."""
        if obj is None:
            return (ObjectType.EMPTY, 0, 0)
        return (obj.type, obj.color, getattr(obj, 'state', 0))
    
    def _encode_item(self, obj) -> Optional[Tuple[int, int]]:
        """Encode carried item as (type, color)."""
        if obj is None:
            return None
        return (obj.type, obj.color)
    
    def get_state_key(self, obs: dict) -> Tuple:
        """Convert observation + memory to hashable state key."""
        parts = [f"F{obs['front']}"]
        
        if 'left' in obs:
            parts.append(f"L{obs['left'][0]}")
            parts.append(f"R{obs['right'][0]}")
        
        if obs['carrying']:
            parts.append(f"C{obs['carrying']}")
        
        if 'view_hash' in obs:
            parts.append(f"V{obs['view_hash']}")
        
        # Add memory
        depth = min(self.config.memory_depth, len(self.obs_history))
        for i in range(depth):
            prev = self.obs_history[-(i+1)]
            parts.append(f"P{i}F{prev['front'][0]}")
            if i < len(self.action_history):
                parts.append(f"P{i}A{self.action_history[-(i+1)]}")
        
        return tuple(parts)
    
    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------
    
    def record(self, obs: dict, action: int, success: bool):
        """Record outcome and check for correlations."""
        # Standard outcome tracking
        key = (self.get_state_key(obs), action)
        self.outcomes[key][1] += 1
        if success:
            self.outcomes[key][0] += 1
        
        # Correlation tracking
        if self.config.use_correlations:
            self._check_correlations(obs, action, success)
    
    def _check_correlations(self, obs: dict, action: int, success: bool):
        """Track correlations for equivalence class discovery."""
        front_type, front_color, front_state = obs['front']
        
        # Color matching for doors
        if front_type == ObjectType.DOOR and obs['carrying']:
            carry_type, carry_color = obs['carrying']
            colors_match = (carry_color == front_color)
            
            key = ('color_match', front_state, action, colors_match)
            self.correlations[key][1] += 1
            if success:
                self.correlations[key][0] += 1
    
    def discover_rules(self):
        """Analyze correlations to discover equivalence rules."""
        for key, (succ, total) in self.correlations.items():
            if total < 3:
                continue
            
            if key[0] == 'color_match' and key[3] == True:
                rate = succ / total
                if rate > 0.8:
                    self.equiv_rules.add("color_match_unlock")
    
    # -------------------------------------------------------------------------
    # Action Selection
    # -------------------------------------------------------------------------
    
    def choose_action(self, obs: dict) -> int:
        """Choose action using heuristics + learned knowledge."""
        # Random exploration
        if random.random() < self.config.explore_rate:
            return random.randint(0, 5)
        
        front_type, front_color, front_state = obs['front']
        has_item = obs['carrying'] is not None
        
        # Strong heuristics
        if obs['on_goal']:
            return Action.DONE
        
        if front_type == ObjectType.GOAL:
            return Action.FORWARD
        
        if front_type == ObjectType.KEY and not has_item:
            return Action.PICKUP
        
        if front_type == ObjectType.DOOR:
            return self._handle_door(obs, front_state, has_item, front_color)
        
        # Check sides for goals (3-cell view)
        if 'left' in obs:
            if obs['left'][0] == ObjectType.GOAL:
                return Action.LEFT
            if obs['right'][0] == ObjectType.GOAL:
                return Action.RIGHT
        
        # Check learned outcomes
        action = self._get_best_learned_action(obs)
        if action is not None:
            return action
        
        # Default movement
        if front_type in [ObjectType.EMPTY, ObjectType.FLOOR]:
            return Action.FORWARD if random.random() < 0.5 else random.choice([Action.LEFT, Action.RIGHT])
        
        return random.choice([Action.LEFT, Action.RIGHT])
    
    def _handle_door(self, obs: dict, state: int, has_item: bool, door_color: int) -> int:
        """Handle door interaction with color matching rule."""
        if state == DoorState.OPEN:
            return Action.FORWARD
        
        if state == DoorState.CLOSED:
            return Action.TOGGLE  # Always toggle closed doors
        
        if state == DoorState.LOCKED:
            if has_item:
                carry_color = obs['carrying'][1]
                
                # Use color matching rule if discovered
                if "color_match_unlock" in self.equiv_rules:
                    if carry_color == door_color:
                        return Action.TOGGLE
                    else:
                        # Wrong key color! Turn to find correct one
                        return random.choice([Action.LEFT, Action.RIGHT])
                else:
                    # Haven't learned yet, try anyway
                    return Action.TOGGLE
        
        return random.choice([Action.LEFT, Action.RIGHT])
    
    def _get_best_learned_action(self, obs: dict) -> Optional[int]:
        """Get best action from learned outcomes."""
        state_key = self.get_state_key(obs)
        
        best_action = None
        best_score = 0.1  # Minimum threshold
        
        for action in [Action.FORWARD, Action.LEFT, Action.RIGHT]:
            key = (state_key, action)
            wins, total = self.outcomes.get(key, [0, 0])
            if total > 0:
                score = wins / total
                if score > best_score:
                    best_score = score
                    best_action = action
        
        return best_action
    
    # -------------------------------------------------------------------------
    # Adaptive Configuration
    # -------------------------------------------------------------------------
    
    def increase_complexity(self):
        """Increase observation/memory complexity when stuck."""
        changed = False
        
        # First try increasing memory
        if self.config.memory_depth < 5:
            self.config.memory_depth += 1
            changed = True
        # Then try increasing view
        elif self.config.view_mode < ViewMode.FULL_7X7:
            self.config.view_mode = ViewMode(min(self.config.view_mode * 3, 49))
            changed = True
        
        return changed


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def run_episode(env, agent: ConfigurableAgent, max_steps: int = 300, 
                training: bool = True) -> Tuple[bool, int]:
    """Run single episode."""
    env.reset()
    agent.reset()
    obs = agent.get_obs(env)
    
    for step in range(max_steps):
        action = agent.choose_action(obs)
        
        agent.obs_history.append(obs)
        agent.action_history.append(action)
        
        _, reward, term, trunc, _ = env.step(action)
        
        if training:
            agent.record(obs, action, reward > 0)
        
        if reward > 0:
            return True, step + 1
        if term or trunc:
            return False, step + 1
        
        obs = agent.get_obs(env)
    
    return False, max_steps


def test_environment(env_name: str, config: AgentConfig = None,
                    n_train: int = 50, n_test: int = 30) -> dict:
    """Test agent on environment."""
    if env_name not in ENVIRONMENTS:
        return {"error": f"Unknown environment: {env_name}"}
    
    agent = ConfigurableAgent(config or AgentConfig())
    env_factory = ENVIRONMENTS[env_name]
    
    # Training
    train_wins = 0
    for ep in range(n_train):
        env = env_factory(seed=ep)
        won, steps = run_episode(env, agent, training=True)
        if won:
            train_wins += 1
        
        # Periodic rule discovery
        if (ep + 1) % 10 == 0:
            agent.discover_rules()
    
    # Final rule discovery
    agent.discover_rules()
    
    # Testing
    test_wins = 0
    test_steps = []
    for ep in range(n_test):
        env = env_factory(seed=1000 + ep)
        won, steps = run_episode(env, agent, training=False)
        if won:
            test_wins += 1
            test_steps.append(steps)
    
    return {
        "env": env_name,
        "config": f"view={agent.config.view_mode.name}, mem={agent.config.memory_depth}",
        "train_rate": train_wins / n_train,
        "test_rate": test_wins / n_test,
        "test_wins": test_wins,
        "avg_steps": sum(test_steps) / len(test_steps) if test_steps else 0,
        "rules": list(agent.equiv_rules),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("CONFIGURABLE MINIGRID AGENT")
    print("=" * 65)
    
    # Quick tests
    configs = [
        ("Minimal", AgentConfig(ViewMode.FRONT_ONLY, 0)),
        ("Memory-2", AgentConfig(ViewMode.FRONT_ONLY, 2)),
        ("3-Cell + Mem-2", AgentConfig(ViewMode.THREE_CELL, 2)),
    ]
    
    envs = ["DoorKey-5x5", "MultiRoom-N2-S4", "FourRooms"]
    
    print("\n(30 train, 20 test per config)\n")
    
    for env_name in envs:
        print(f"{env_name}:")
        for name, cfg in configs:
            result = test_environment(env_name, cfg, n_train=30, n_test=20)
            status = "✓" if result['test_rate'] >= 0.8 else "~" if result['test_rate'] >= 0.3 else "✗"
            rules = f" {result['rules']}" if result['rules'] else ""
            print(f"  {status} {name:18}: {result['test_rate']:5.0%}{rules}")
        print()
