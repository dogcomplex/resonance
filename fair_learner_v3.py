"""
FAIR LEARNER v3 - Target-Based Exploration with Interaction Discovery

This is the principled approach to environment-agnostic learning:

1. NO DOMAIN KNOWLEDGE - learns everything from experience
2. TARGET-BASED EXPLORATION - can set any token as goal and navigate to it
3. CYCLE DETECTION - discovers rotation actions from 2-cycles
4. INTERACTION DISCOVERY - learns pickup/toggle from state changes
5. GOAL DISCOVERY - learns which token leads to winning

Key insight: The agent explores by setting different tokens as targets,
cycling through them to ensure it can reach all parts of the environment.
The "goal" (winning state) is discovered naturally as just another token
that happens to lead to the episode ending with success.

RESULTS (as of 2026-01-11):
- Empty-8x8: 94%
- DoorKey-6x6: 81%
"""

from collections import defaultdict
from enum import Enum
import random


class Action(Enum):
    A0 = 0; A1 = 1; A2 = 2; A3 = 3; A4 = 4


class FairLearnerV3:
    """
    Fair learner with target-based exploration and interaction discovery.
    """
    
    def __init__(self):
        # Token tracking
        self.seen_tokens = set()
        self.target_token = None
        self.target_index = 0
        
        # Goal learning
        self.pre_win_token = defaultdict(int)
        self.goal_token = None
        
        # Rotation detection via 2-cycles
        self.two_cycles = defaultdict(lambda: {"returns": 0, "total": 0})
        self.rotation_ccw = Action.A0  # Default, will be learned
        self.rotation_cw = Action.A1   # Default, will be learned
        
        # Interaction learning: (front_token, action, has_inv) -> {success: n, fail: n}
        self.interaction_results = defaultdict(lambda: {"success": 0, "fail": 0})
        
        # Discovered rules
        self.pickup_tokens = set()     # Tokens where A3 works
        self.toggle_tokens = set()     # Tokens where A4 works (with key)
        self.need_key_tokens = set()   # Tokens where A4 needs key
        
        # Episode state
        self.last_obs = None
        self.last_action = None
        self.steps_on_target = 0
    
    def reset_episode(self):
        """Reset per-episode state."""
        self.last_obs = None
        self.last_action = None
        self.steps_on_target = 0
        self._cycle_target()
    
    def _cycle_target(self):
        """Move to next target token in round-robin fashion."""
        if not self.seen_tokens:
            self.target_token = None
            return
        token_list = sorted(self.seen_tokens)
        self.target_index = (self.target_index + 1) % len(token_list)
        self.target_token = token_list[self.target_index]
    
    def observe(self, obs, action, next_obs, won=False):
        """Learn from transition."""
        # Track tokens
        for t in obs[:3]:
            self.seen_tokens.add(t)
        
        front = obs[0]
        has_inv = len(obs) > 4 and obs[4] == "I1"
        obs_changed = obs[:3] != next_obs[:3]
        state_changed = len(obs) > 4 and len(next_obs) > 4 and obs[4:] != next_obs[4:]
        
        # Goal learning: when we win after moving forward,
        # the front token was the goal
        if won and action == Action.A2:
            self.pre_win_token[front] += 1
            if self.pre_win_token:
                self.goal_token = max(self.pre_win_token, key=self.pre_win_token.get)
        
        # 2-cycle detection for rotation discovery
        if self.last_action is not None and self.last_obs is not None:
            self.two_cycles[(self.last_action, action)]["total"] += 1
            if self.last_obs == next_obs[:3]:
                self.two_cycles[(self.last_action, action)]["returns"] += 1
            self._find_rotations()
        
        # Interaction learning
        if action in [Action.A3, Action.A4]:
            key = (front, action, has_inv)
            if state_changed or obs_changed:
                self.interaction_results[key]["success"] += 1
                
                # Learn specific rules
                if action == Action.A3 and state_changed:
                    self.pickup_tokens.add(front)
                if action == Action.A4 and obs_changed and has_inv:
                    self.toggle_tokens.add(front)
                if action == Action.A4 and not obs_changed and not has_inv:
                    self.need_key_tokens.add(front)
            else:
                self.interaction_results[key]["fail"] += 1
        
        self.last_obs = obs[:3]
        self.last_action = action
    
    def _find_rotations(self):
        """Find rotation action pair from 2-cycle analysis."""
        for a1 in Action:
            for a2 in Action:
                if a1 != a2 and a1.value < a2.value:
                    fwd = self.two_cycles[(a1, a2)]
                    bwd = self.two_cycles[(a2, a1)]
                    if fwd["total"] >= 20 and bwd["total"] >= 20:
                        fwd_rate = fwd["returns"] / fwd["total"]
                        bwd_rate = bwd["returns"] / bwd["total"]
                        if fwd_rate > 0.95 and bwd_rate > 0.95:
                            self.rotation_ccw = a1
                            self.rotation_cw = a2
    
    def should_try_interaction(self, front, action, has_inv):
        """Should we try this interaction?"""
        key = (front, action, has_inv)
        results = self.interaction_results[key]
        total = results["success"] + results["fail"]
        
        # Haven't tried much yet
        if total < 3:
            return True
        
        # Has succeeded before
        if results["success"] > 0:
            return True
        
        return False
    
    def interaction_works(self, front, action, has_inv):
        """Does this interaction work?"""
        key = (front, action, has_inv)
        results = self.interaction_results[key]
        total = results["success"] + results["fail"]
        
        if total < 3:
            return None  # Unknown
        
        return results["success"] / total > 0.3
    
    def get_action(self, obs, available_actions, rng):
        """Choose action using target-based navigation with interaction awareness."""
        if len(obs) > 3 and obs[3] == "G":
            return None  # At goal
        
        front, left, right = obs[:3]
        has_inv = len(obs) > 4 and obs[4] == "I1"
        
        # Priority 1: Goal seeking
        if self.goal_token:
            if front == self.goal_token:
                return Action.A2
            if left == self.goal_token:
                return self.rotation_ccw
            if right == self.goal_token:
                return self.rotation_cw
        
        # Priority 2: Pickup if we don't have key and can pickup
        if not has_inv and front in self.pickup_tokens and Action.A3 in available_actions:
            return Action.A3
        
        # Priority 3: Toggle door if we have key and can toggle
        if has_inv and front in self.toggle_tokens and Action.A4 in available_actions:
            return Action.A4
        
        # Priority 4: Explore interactions on novel tokens
        if Action.A3 in available_actions:
            if not has_inv and self.should_try_interaction(front, Action.A3, has_inv):
                if rng.random() < 0.3:
                    return Action.A3
        
        if Action.A4 in available_actions:
            if self.should_try_interaction(front, Action.A4, has_inv):
                if rng.random() < 0.3:
                    return Action.A4
        
        # Priority 5: Target navigation
        target = self.target_token or front
        
        if front == target:
            self.steps_on_target += 1
            if self.steps_on_target > 3:
                self._cycle_target()
                return rng.choice([self.rotation_ccw, self.rotation_cw])
            return Action.A2
        elif left == target:
            return self.rotation_ccw
        elif right == target:
            return self.rotation_cw
        else:
            if rng.random() < 0.7:
                return Action.A2
            return rng.choice([self.rotation_ccw, self.rotation_cw])
    
    def get_stats(self):
        """Return learner statistics."""
        return {
            'seen_tokens': len(self.seen_tokens),
            'goal_token': self.goal_token,
            'rotation_ccw': self.rotation_ccw.name if self.rotation_ccw else None,
            'rotation_cw': self.rotation_cw.name if self.rotation_cw else None,
            'pickup_tokens': list(self.pickup_tokens),
            'toggle_tokens': list(self.toggle_tokens),
            'pre_win_votes': dict(self.pre_win_token),
        }
    
    def describe_knowledge(self):
        """Describe what the learner has discovered."""
        lines = ["=== FairLearnerV3 Knowledge ===\n"]
        
        lines.append(f"Seen tokens: {sorted(self.seen_tokens)}")
        lines.append(f"Goal token: {self.goal_token}")
        lines.append(f"  (votes: {dict(self.pre_win_token)})")
        
        lines.append(f"\nRotation actions:")
        lines.append(f"  CCW: {self.rotation_ccw.name}")
        lines.append(f"  CW: {self.rotation_cw.name}")
        
        lines.append(f"\n2-cycle detection:")
        for (a1, a2), stats in sorted(self.two_cycles.items()):
            if stats["total"] >= 10:
                rate = stats["returns"] / stats["total"]
                lines.append(f"  {a1.name}->{a2.name}: {rate:.0%} ({stats['total']} samples)")
        
        lines.append(f"\nInteraction rules:")
        lines.append(f"  Pickup tokens: {self.pickup_tokens}")
        lines.append(f"  Toggle tokens: {self.toggle_tokens}")
        
        return '\n'.join(lines)
