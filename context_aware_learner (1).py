"""
Context-Aware Hierarchical Learner

Integrates context filtering: learns both position-specific and position-independent rules.
Uses the winning insight: separate learners for different token contexts.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Dict, List, FrozenSet, Optional, Tuple
from itertools import combinations
import sys

sys.path.insert(0, '/mnt/user-data/outputs')


@dataclass
class Rule:
    pattern: FrozenSet[str]
    action: int
    effect: str
    probability: float
    observations: int
    is_deterministic: bool = False


class ContextAwareHierarchicalLearner:
    """
    Learns rules at multiple levels of context:
    1. Full state (for position-specific effects)
    2. Relative state (position-independent, for view/dir effects)
    
    The key insight: different effects depend on different token subsets.
    Position changes need position tokens.
    View changes only need view tokens.
    """
    
    def __init__(self, n_actions: int = 7, max_pattern_size: int = 4):
        self.n_actions = n_actions
        self.max_pattern_size = max_pattern_size
        
        # Separate tracking for full vs relative states
        # Full state: all tokens
        self.full_effects: Dict[Tuple[FrozenSet, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.full_counts: Dict[Tuple[FrozenSet, int], int] = defaultdict(int)
        
        # Relative state: no position tokens
        self.rel_effects: Dict[Tuple[FrozenSet, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.rel_counts: Dict[Tuple[FrozenSet, int], int] = defaultdict(int)
        
        # Track which effects are position-dependent vs position-independent
        self.effect_needs_position: Dict[str, bool] = {}
        
        # Statistics
        self.total_observations = 0
        
    def _is_position_token(self, token: str) -> bool:
        """Check if token is position-related."""
        return token.startswith('pos_')
    
    def _is_position_effect(self, effect: str) -> bool:
        """Check if effect involves position change."""
        return 'pos_' in effect
    
    def _get_relative_state(self, state: Set[str]) -> FrozenSet[str]:
        """Remove position tokens from state."""
        return frozenset(t for t in state if not self._is_position_token(t))
    
    def _get_relative_effects(self, effects: Set[str]) -> Set[str]:
        """Remove position-related effects."""
        return {e for e in effects if not self._is_position_effect(e)}
    
    def observe(self, state: Set[str], action: int, next_state: Set[str]):
        """Record an observation, updating both full and relative learners."""
        self.total_observations += 1
        
        # Compute effects
        added = next_state - state
        removed = state - next_state
        effects = {f"+{t}" for t in added} | {f"-{t}" for t in removed}
        
        # Separate position vs relative effects
        pos_effects = {e for e in effects if self._is_position_effect(e)}
        rel_effects = {e for e in effects if not self._is_position_effect(e)}
        
        # Full state key
        full_key = frozenset(state)
        
        # Relative state key (no position)
        rel_state = self._get_relative_state(state)
        
        # Update full learner with position effects
        for effect in pos_effects:
            self.full_effects[(full_key, action)][effect] += 1
            self.effect_needs_position[effect] = True
        self.full_counts[(full_key, action)] += 1
        
        # Update relative learner with relative effects
        for effect in rel_effects:
            self.rel_effects[(rel_state, action)][effect] += 1
            self.effect_needs_position[effect] = False
        self.rel_counts[(rel_state, action)] += 1
        
        # Also track relative effects in full learner (for comparison)
        for effect in rel_effects:
            self.full_effects[(full_key, action)][effect] += 1
    
    def get_rules(self, min_obs: int = 3, min_prob: float = 0.0) -> List[Rule]:
        """
        Extract rules from both learners.
        Returns position-specific rules for position effects,
        and position-independent rules for relative effects.
        """
        rules = []
        
        # Position-independent rules (from relative learner)
        for (pattern, action), effect_counts in self.rel_effects.items():
            total = self.rel_counts[(pattern, action)]
            if total < min_obs:
                continue
            
            for effect, count in effect_counts.items():
                prob = count / total
                if prob >= min_prob:
                    rules.append(Rule(
                        pattern=pattern,
                        action=action,
                        effect=effect,
                        probability=prob,
                        observations=total,
                        is_deterministic=(prob > 0.95 or prob < 0.05)
                    ))
        
        # Position-specific rules (from full learner, only for position effects)
        for (pattern, action), effect_counts in self.full_effects.items():
            total = self.full_counts[(pattern, action)]
            if total < min_obs:
                continue
            
            for effect, count in effect_counts.items():
                if not self._is_position_effect(effect):
                    continue  # Skip relative effects (already covered)
                
                prob = count / total
                if prob >= min_prob:
                    rules.append(Rule(
                        pattern=pattern,
                        action=action,
                        effect=effect,
                        probability=prob,
                        observations=total,
                        is_deterministic=(prob > 0.95 or prob < 0.05)
                    ))
        
        return rules
    
    def predict(self, state: Set[str], action: int, prob_threshold: float = 0.7) -> Set[str]:
        """
        Predict effects for a state-action pair.
        Uses relative learner for view/dir effects (more general).
        Uses full learner for position effects (position-specific).
        """
        predictions = set()
        
        full_key = frozenset(state)
        rel_state = self._get_relative_state(state)
        
        # Predict relative effects using relative learner
        if (rel_state, action) in self.rel_effects:
            total = self.rel_counts[(rel_state, action)]
            if total > 0:
                for effect, count in self.rel_effects[(rel_state, action)].items():
                    if count / total >= prob_threshold:
                        predictions.add(effect)
        
        # Predict position effects using full learner
        if (full_key, action) in self.full_effects:
            total = self.full_counts[(full_key, action)]
            if total > 0:
                for effect, count in self.full_effects[(full_key, action)].items():
                    if self._is_position_effect(effect):
                        if count / total >= prob_threshold:
                            predictions.add(effect)
        
        return predictions
    
    def predict_with_fallback(self, state: Set[str], action: int, 
                               prob_threshold: float = 0.7) -> Tuple[Set[str], str]:
        """
        Predict with hierarchical fallback.
        Returns (predictions, source) where source indicates which learner was used.
        """
        predictions = set()
        sources = []
        
        full_key = frozenset(state)
        rel_state = self._get_relative_state(state)
        
        # Try exact match on relative state
        if (rel_state, action) in self.rel_effects:
            total = self.rel_counts[(rel_state, action)]
            if total > 0:
                for effect, count in self.rel_effects[(rel_state, action)].items():
                    if count / total >= prob_threshold:
                        predictions.add(effect)
                sources.append('rel_exact')
        
        # Try exact match on full state for position effects
        if (full_key, action) in self.full_effects:
            total = self.full_counts[(full_key, action)]
            if total > 0:
                for effect, count in self.full_effects[(full_key, action)].items():
                    if self._is_position_effect(effect):
                        if count / total >= prob_threshold:
                            predictions.add(effect)
                sources.append('full_exact')
        
        # Fallback: find best matching relative pattern
        if not any('rel' in s for s in sources):
            best_match = None
            best_overlap = -1
            
            for (pattern, act), _ in self.rel_effects.items():
                if act != action:
                    continue
                if pattern <= rel_state:  # pattern is subset of state
                    overlap = len(pattern)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = pattern
            
            if best_match is not None:
                total = self.rel_counts[(best_match, action)]
                if total > 0:
                    for effect, count in self.rel_effects[(best_match, action)].items():
                        if count / total >= prob_threshold:
                            predictions.add(effect)
                    sources.append('rel_fallback')
        
        return predictions, '+'.join(sources) if sources else 'none'
    
    def stats(self) -> Dict:
        """Return statistics about the learner."""
        rel_states = len(set(p for p, a in self.rel_counts.keys()))
        full_states = len(set(p for p, a in self.full_counts.keys()))
        
        rel_rules = sum(len(effs) for effs in self.rel_effects.values())
        full_rules = sum(len(effs) for effs in self.full_effects.values())
        
        return {
            'total_observations': self.total_observations,
            'unique_relative_states': rel_states,
            'unique_full_states': full_states,
            'relative_rules': rel_rules,
            'full_rules': full_rules,
            'compression_ratio': rel_states / full_states if full_states > 0 else 0
        }


# Test on MiniGrid
if __name__ == "__main__":
    import random
    from minigrid_official import FourRoomsEnv, EmptyEnv, DoorKeyEnv
    from hierarchical_learner import HierarchicalLearner
    
    def full_tokenize(obs, env):
        tokens = set()
        if hasattr(obs, 'get'):
            image = obs.get('image', obs)
        else:
            image = obs
        
        for rel_pos, name in [(3, 'front'), (1, 'left'), (5, 'right')]:
            row, col = rel_pos // 3, rel_pos % 3
            if row < len(image) and col < len(image[0]):
                cell = image[row][col]
                tokens.add(f"{name}_t{cell[0]}")
        
        if env.carrying:
            tokens.add(f"carry_t{env.carrying.encode()[0]}")
        else:
            tokens.add("carry_none")
        
        tokens.add(f"pos_{env.agent_pos[0]}_{env.agent_pos[1]}")
        tokens.add(f"dir_{env.agent_dir}")
        
        return tokens
    
    print("="*70)
    print("CONTEXT-AWARE LEARNER TEST")
    print("="*70)
    
    # Compare: baseline vs context-aware
    env_factory = lambda seed=None: FourRoomsEnv(seed=seed)
    
    baseline = HierarchicalLearner(n_actions=7)
    context_aware = ContextAwareHierarchicalLearner(n_actions=7)
    
    print("\nTraining on FourRooms (500 episodes)...")
    
    for ep in range(500):
        try:
            env = env_factory(seed=ep)
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            state = full_tokenize(obs, env)
        except:
            continue
        
        for _ in range(40):
            action = random.randint(0, 6)
            try:
                result = env.step(action)
                next_state = full_tokenize(result[0], env)
            except:
                break
            
            baseline.observe(state, action, next_state)
            context_aware.observe(state, action, next_state)
            
            state = next_state
            if result[2]:
                break
    
    # Compare stats
    baseline_rules = baseline.get_stable(min_obs=3)
    ca_stats = context_aware.stats()
    
    print(f"\nBaseline: {len(baseline_rules)} rules")
    print(f"Context-aware stats:")
    for k, v in ca_stats.items():
        print(f"  {k}: {v}")
    
    # Test prediction accuracy
    print("\n" + "-"*70)
    print("Testing prediction accuracy")
    print("-"*70)
    
    def test_learner(predict_fn, env_factory, test_episodes=50):
        tp, fp, fn = 0, 0, 0
        
        for ep in range(test_episodes):
            try:
                env = env_factory(seed=50000 + ep)
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                state = full_tokenize(obs, env)
            except:
                continue
            
            for _ in range(25):
                action = random.randint(0, 6)
                try:
                    result = env.step(action)
                    next_state = full_tokenize(result[0], env)
                except:
                    break
                
                actual = {f"+{t}" for t in (next_state - state)} | {f"-{t}" for t in (state - next_state)}
                predicted = predict_fn(state, action)
                
                for e in predicted:
                    if e in actual:
                        tp += 1
                    else:
                        fp += 1
                for e in actual:
                    if e not in predicted:
                        fn += 1
                
                state = next_state
                if result[2]:
                    break
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall
    
    # Baseline predictor
    def baseline_predict(state, action):
        rules = baseline.get_stable(min_obs=3)
        by_effect = defaultdict(list)
        for r in rules:
            if r.action == action and r.pattern <= state:
                by_effect[r.effect].append(r)
        
        predictions = set()
        for effect, effect_rules in by_effect.items():
            best = max(effect_rules, key=lambda r: len(r.pattern))
            if best.probability >= 0.7:
                predictions.add(effect)
        return predictions
    
    # Context-aware predictor
    def ca_predict(state, action):
        preds, _ = context_aware.predict_with_fallback(state, action, prob_threshold=0.7)
        return preds
    
    baseline_f1, baseline_p, baseline_r = test_learner(baseline_predict, env_factory)
    ca_f1, ca_p, ca_r = test_learner(ca_predict, env_factory)
    
    print(f"\nBaseline:       F1={baseline_f1:.1%} P={baseline_p:.1%} R={baseline_r:.1%}")
    print(f"Context-aware:  F1={ca_f1:.1%} P={ca_p:.1%} R={ca_r:.1%}")
    print(f"Improvement:    {ca_f1 - baseline_f1:+.1%}")
    
    # Test on relative effects only
    print("\n" + "-"*70)
    print("Testing on RELATIVE effects only (view/dir/carry)")
    print("-"*70)
    
    def test_relative(predict_fn, env_factory, test_episodes=50):
        tp, fp, fn = 0, 0, 0
        
        for ep in range(test_episodes):
            try:
                env = env_factory(seed=50000 + ep)
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                state = full_tokenize(obs, env)
            except:
                continue
            
            for _ in range(25):
                action = random.randint(0, 6)
                try:
                    result = env.step(action)
                    next_state = full_tokenize(result[0], env)
                except:
                    break
                
                # Only relative effects
                actual = set()
                for t in (next_state - state):
                    if not t.startswith('pos_'):
                        actual.add(f"+{t}")
                for t in (state - next_state):
                    if not t.startswith('pos_'):
                        actual.add(f"-{t}")
                
                predicted = {e for e in predict_fn(state, action) if 'pos_' not in e}
                
                for e in predicted:
                    if e in actual:
                        tp += 1
                    else:
                        fp += 1
                for e in actual:
                    if e not in predicted:
                        fn += 1
                
                state = next_state
                if result[2]:
                    break
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1, precision, recall
    
    baseline_rel_f1, baseline_rel_p, baseline_rel_r = test_relative(baseline_predict, env_factory)
    ca_rel_f1, ca_rel_p, ca_rel_r = test_relative(ca_predict, env_factory)
    
    print(f"\nBaseline:       F1={baseline_rel_f1:.1%} P={baseline_rel_p:.1%} R={baseline_rel_r:.1%}")
    print(f"Context-aware:  F1={ca_rel_f1:.1%} P={ca_rel_p:.1%} R={ca_rel_r:.1%}")
    print(f"Improvement:    {ca_rel_f1 - baseline_rel_f1:+.1%}")
