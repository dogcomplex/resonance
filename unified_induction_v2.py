"""
Unified Induction v2 - with exact match + intersection fallback

Like V9, we use:
1. Exact match (full state, action) -> effects (100% accurate for seen)
2. Intersection-based rules (for generalization)
"""

import random
from typing import Set, FrozenSet, List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass


class UnifiedInductionV2:
    """
    O(N) Learner with exact match + intersection fallback.
    """
    
    def __init__(self, n_actions: int = 10, num_samples: int = 100):
        self.n_actions = n_actions
        self.num_samples = num_samples
        
        # Exact matches (like V9)
        self.exact_observations: Dict[Tuple[FrozenSet, int], Dict[FrozenSet, int]] = defaultdict(lambda: defaultdict(int))
        
        # Generalized rules (computed via intersection)
        self.observations: List = []
        self.rules = None
        
    def observe(self, state: Set[str], action: int, next_state: Set[str]):
        """Record an observation."""
        before_fs = frozenset(state)
        after_fs = frozenset(next_state)
        
        added = after_fs - before_fs
        removed = before_fs - after_fs
        effects = frozenset({f"+{t}" for t in added} | {f"-{t}" for t in removed})
        
        # Store exact observation
        self.exact_observations[(before_fs, action)][effects] += 1
        
        # Store for intersection-based rules
        self.observations.append({
            'before': before_fs,
            'after': after_fs,
            'action': action,
            'effect': effects
        })
        self.rules = None  # Invalidate cache
    
    def _compute_rules(self):
        """Compute generalized rules via intersection + counting."""
        if self.rules is not None:
            return
            
        by_action_effect = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
                key = (obs['action'], obs['effect'])
                by_action_effect[key].append(obs['before'])
        
        self.rules = {}
        
        for (action, effect), positives in by_action_effect.items():
            if len(positives) < 2:
                continue
            
            # Sample and intersect
            candidates = set()
            for _ in range(min(self.num_samples, len(positives) * 10)):
                k = max(2, len(positives) * 2 // 3)
                sample = random.sample(positives, min(k, len(positives)))
                lhs = frozenset.intersection(*sample)
                if lhs and len(lhs) < 20:  # Avoid overly specific rules
                    candidates.add(lhs)
            
            # Compute probability for each candidate
            for lhs in candidates:
                applicable = [obs for obs in self.observations 
                              if obs['action'] == action and lhs <= obs['before']]
                if not applicable:
                    continue
                    
                hits = sum(1 for obs in applicable if obs['effect'] == effect)
                prob = hits / len(applicable)
                
                key = (action, lhs)
                if key not in self.rules or prob > self.rules[key][1]:
                    self.rules[key] = (effect, prob, len(applicable))
    
    def predict(self, state: Set[str], action: int) -> Set[str]:
        """Predict effects for (state, action)."""
        state_fs = frozenset(state)
        key = (state_fs, action)
        
        # First: try exact match
        if key in self.exact_observations:
            obs = self.exact_observations[key]
            if obs:
                # Return most common effect
                best_effect = max(obs.items(), key=lambda x: x[1])[0]
                return set(best_effect)
        
        # Second: try generalized rules
        self._compute_rules()
        
        best_match = None
        best_specificity = -1
        best_prob = 0
        
        for (rule_action, rule_lhs), (effect, prob, support) in self.rules.items():
            if rule_action != action:
                continue
            if not rule_lhs <= state_fs:
                continue
            
            specificity = len(rule_lhs)
            if specificity > best_specificity or (specificity == best_specificity and prob > best_prob):
                best_match = effect
                best_specificity = specificity
                best_prob = prob
        
        if best_match and best_prob >= 0.5:
            return set(best_match)
        return set()
    
    def close(self):
        pass


# Quick test
if __name__ == "__main__":
    random.seed(42)
    
    class SimpleTicTacToe:
        def __init__(self, seed=42):
            self.rng = random.Random(seed)
            self.reset()
        
        def reset(self, seed=None):
            if seed: self.rng = random.Random(seed)
            self.board = [0] * 9
            self.current = 1
            self.done = False
            return self._state()
        
        def _state(self):
            tokens = {f"cell_{i}_{self.board[i]}" for i in range(9)}
            tokens.add(f"player_{self.current}")
            tokens.add(f"done_{self.done}")
            return tokens
        
        def get_valid_actions(self):
            if self.done: return []
            return [i for i in range(9) if self.board[i] == 0]
        
        def step(self, action):
            if self.done or action not in self.get_valid_actions():
                return self._state(), 0, self.done, {}
            self.board[action] = self.current
            wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
            for a, b, c in wins:
                if self.board[a] == self.board[b] == self.board[c] == self.current:
                    self.done = True
                    return self._state(), 1, True, {}
            if 0 not in self.board:
                self.done = True
                return self._state(), 0, True, {}
            self.current = 3 - self.current
            return self._state(), 0, False, {}
    
    env = SimpleTicTacToe()
    learner = UnifiedInductionV2(n_actions=9)
    seen = set()
    
    # Train
    for ep in range(200):
        state = env.reset(seed=ep)
        for _ in range(20):
            valid = env.get_valid_actions()
            if not valid: break
            action = random.choice(valid)
            seen.add((frozenset(state), action))
            next_state, _, done, _ = env.step(action)
            learner.observe(state, action, next_state)
            state = next_state
            if done: break
    
    # Test
    tp, fp, fn = 0, 0, 0
    for ep in range(100):
        state = env.reset(seed=50000+ep)
        for _ in range(20):
            valid = env.get_valid_actions()
            if not valid: break
            action = random.choice(valid)
            if (frozenset(state), action) not in seen:
                state = env.step(action)[0]
                continue
            next_state, _, done, _ = env.step(action)
            actual = {f"+{t}" for t in (next_state - state)} | {f"-{t}" for t in (state - next_state)}
            predicted = learner.predict(state, action)
            for e in predicted:
                if e in actual: tp += 1
                else: fp += 1
            for e in actual:
                if e not in predicted: fn += 1
            state = next_state
            if done: break
    
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
    print(f"Unified V2 on TicTacToe: F1={f1:.1%}")
    print(f"  Exact observations: {len(learner.exact_observations)}")
    print(f"  Generalized rules: {len(learner.rules or {})}")
