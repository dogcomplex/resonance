"""
PIXEL FIDELITY BOOST

The problem: 40% gap between SEEN and UNSEEN accuracy.
The hypothesis: Better fidelity (token abstraction) can close this gap.

Strategy: Find tokens that are INTERCHANGEABLE in their effects.
If g_3_2_bright and g_3_4_bright always lead to same effects,
they should be treated as equivalent.
"""

import random
import sys
import numpy as np
from collections import defaultdict
from typing import Set, FrozenSet, Dict, List

sys.path.insert(0, '/home/claude')

from pixel_environments import PixelPong, PixelBreakout, tokenize_diff


class FidelitySieve:
    """
    Sieve with emergent token abstraction.
    
    Key insight: Tokens are interchangeable if they appear in similar contexts
    with similar effects. We can discover this WITHOUT domain knowledge.
    """
    
    def __init__(self, n_actions=4, min_co_occurrence=5):
        self.n_actions = n_actions
        self.min_co_occurrence = min_co_occurrence
        
        self.observations = []
        self.exact_matches = defaultdict(lambda: defaultdict(int))
        
        # Token equivalence classes (emergent)
        self.token_classes = {}  # token -> class_id
        self.class_members = defaultdict(set)  # class_id -> {tokens}
        
        self.rules = []
        self._dirty = True
    
    def observe(self, state, action, next_state, timestamp=None):
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | {f"-{t}" for t in (before-after)})
        
        self.exact_matches[(before, action)][effect] += 1
        
        self.observations.append({
            'before': before, 'after': after, 'action': action,
            'effect': effect, 'timestamp': timestamp or len(self.observations)
        })
        self._dirty = True
    
    def _discover_equivalences(self):
        """
        Find tokens that behave equivalently.
        
        Two tokens are equivalent if:
        - They appear in similar positions (same grid cell pattern)
        - They co-occur with similar effects
        """
        # Track token -> effect co-occurrences
        token_effect_counts = defaultdict(lambda: defaultdict(int))
        token_total = defaultdict(int)
        
        for obs in self.observations:
            for token in obs['before']:
                token_total[token] += 1
                for e in obs['effect']:
                    token_effect_counts[token][e] += 1
        
        # Find tokens with similar effect distributions
        # (This is emergent - no domain knowledge!)
        self.token_classes = {}
        self.class_members = defaultdict(set)
        class_counter = 0
        
        tokens = list(token_total.keys())
        
        for i, t1 in enumerate(tokens):
            if t1 in self.token_classes:
                continue
            
            # Start new class
            self.token_classes[t1] = class_counter
            self.class_members[class_counter].add(t1)
            
            # Find equivalent tokens
            for t2 in tokens[i+1:]:
                if t2 in self.token_classes:
                    continue
                
                # Compare effect distributions
                if self._are_equivalent(t1, t2, token_effect_counts, token_total):
                    self.token_classes[t2] = class_counter
                    self.class_members[class_counter].add(t2)
            
            class_counter += 1
    
    def _are_equivalent(self, t1, t2, effect_counts, totals):
        """Check if two tokens have similar effect patterns."""
        # Must have similar names (structural equivalence)
        # e.g., g_3_2_bright and g_3_4_bright are candidates
        parts1 = t1.split('_')
        parts2 = t2.split('_')
        
        # Same prefix and suffix, different middle
        if len(parts1) != len(parts2):
            return False
        
        diff_count = sum(1 for p1, p2 in zip(parts1, parts2) if p1 != p2)
        if diff_count > 1:  # Allow 1 component to differ
            return False
        
        # Check effect correlation
        total1 = totals[t1]
        total2 = totals[t2]
        
        if total1 < self.min_co_occurrence or total2 < self.min_co_occurrence:
            return False
        
        # Compare effect distributions
        all_effects = set(effect_counts[t1].keys()) | set(effect_counts[t2].keys())
        
        if not all_effects:
            return True  # Both have no effects
        
        # Cosine similarity of effect vectors
        dot = 0
        mag1 = 0
        mag2 = 0
        
        for e in all_effects:
            v1 = effect_counts[t1][e] / total1
            v2 = effect_counts[t2][e] / total2
            dot += v1 * v2
            mag1 += v1 * v1
            mag2 += v2 * v2
        
        if mag1 == 0 or mag2 == 0:
            return False
        
        similarity = dot / (mag1 ** 0.5 * mag2 ** 0.5)
        return similarity > 0.7  # Threshold for equivalence
    
    def _abstract_state(self, state: FrozenSet[str]) -> FrozenSet[str]:
        """Replace tokens with their class representatives."""
        abstracted = set()
        for token in state:
            if token in self.token_classes:
                class_id = self.token_classes[token]
                # Use first member as representative
                rep = min(self.class_members[class_id])
                abstracted.add(rep)
            else:
                abstracted.add(token)
        return frozenset(abstracted)
    
    def _compute_rules(self):
        if not self._dirty or len(self.observations) < 2:
            return
        
        # Discover token equivalences
        self._discover_equivalences()
        
        n_classes = len(self.class_members)
        n_merged = sum(len(m) for m in self.class_members.values() if len(m) > 1)
        print(f"  Discovered {n_classes} token classes, {n_merged} tokens merged")
        
        # Build rules using abstracted states
        by_ae = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
                abstract_before = self._abstract_state(obs['before'])
                abstract_effect = self._abstract_state(obs['effect'])
                by_ae[(obs['action'], abstract_effect)].append(abstract_before)
        
        self.rules = []
        
        for (action, effect), positives in by_ae.items():
            if len(positives) < 2:
                continue
            
            # Intersection
            for _ in range(10):
                k = min(len(positives), 20)
                sample = random.sample(positives, k)
                lhs = frozenset.intersection(*sample)
                
                if lhs:
                    self.rules.append({
                        'lhs': lhs, 'effect': effect, 'action': action,
                        'probability': 1.0, 'support': len(positives)
                    })
                    break
        
        self._dirty = False
    
    def predict(self, state, action):
        state_fs = frozenset(state)
        
        # Exact match first
        key = (state_fs, action)
        if key in self.exact_matches:
            obs = self.exact_matches[key]
            if obs:
                return set(max(obs.items(), key=lambda x: x[1])[0])
        
        # Abstract state and try rules
        self._compute_rules()
        
        abstract_state = self._abstract_state(state_fs)
        
        best_rule = None
        best_score = -1
        
        for rule in self.rules:
            if rule['action'] != action:
                continue
            if not rule['lhs'] <= abstract_state:
                continue
            
            score = len(rule['lhs'])
            if score > best_score:
                best_score = score
                best_rule = rule
        
        if best_rule:
            return set(best_rule['effect'])
        
        return set()
    
    def close(self):
        pass


def test_generalization(learner_cls, env_class, n_train=100, n_test=50, max_steps=100):
    """Test on SEEN vs UNSEEN states."""
    random.seed(42)
    
    learner = learner_cls(n_actions=4)
    seen_states = set()
    
    for ep in range(n_train):
        env = env_class(seed=ep)
        screen = env.reset()
        prev_screen = screen.copy()
        state = tokenize_diff(screen, prev_screen, grid_size=7)
        
        for step in range(max_steps):
            seen_states.add(frozenset(state))
            action = random.choice(env.get_valid_actions())
            next_screen, _, done, _ = env.step(action)
            next_state = tokenize_diff(next_screen, screen, grid_size=7)
            learner.observe(state, action, next_state, timestamp=ep*max_steps+step)
            prev_screen, screen, state = screen, next_screen, next_state
            if done: break
    
    if hasattr(learner, '_compute_rules'):
        learner._compute_rules()
    
    tp_seen, fp_seen, fn_seen = 0, 0, 0
    tp_unseen, fp_unseen, fn_unseen = 0, 0, 0
    
    for ep in range(n_test):
        env = env_class(seed=50000+ep)
        screen = env.reset()
        prev_screen = screen.copy()
        state = tokenize_diff(screen, prev_screen, grid_size=7)
        
        for step in range(max_steps):
            action = random.choice(env.get_valid_actions())
            next_screen, _, done, _ = env.step(action)
            next_state = tokenize_diff(next_screen, screen, grid_size=7)
            
            actual = {f"+{t}" for t in (next_state - state)} | {f"-{t}" for t in (state - next_state)}
            predicted = learner.predict(state, action)
            
            is_seen = frozenset(state) in seen_states
            
            for e in predicted:
                if e in actual:
                    if is_seen: tp_seen += 1
                    else: tp_unseen += 1
                else:
                    if is_seen: fp_seen += 1
                    else: fp_unseen += 1
            for e in actual:
                if e not in predicted:
                    if is_seen: fn_seen += 1
                    else: fn_unseen += 1
            
            prev_screen, screen, state = screen, next_screen, next_state
            if done: break
    
    f1_seen = 2*tp_seen / (2*tp_seen + fp_seen + fn_seen) if (2*tp_seen + fp_seen + fn_seen) > 0 else 0
    f1_unseen = 2*tp_unseen / (2*tp_unseen + fp_unseen + fn_unseen) if (2*tp_unseen + fp_unseen + fn_unseen) > 0 else 0
    
    return f1_seen, f1_unseen


print("="*70)
print("FIDELITY BOOST EXPERIMENT")
print("="*70)
print("\nComparing baseline vs fidelity-enhanced sieve\n")

from universal_sieve import UniversalSieve

for name, env_class in [("Pong", PixelPong), ("Breakout", PixelBreakout)]:
    print(f"\n--- {name} ---")
    
    print("Baseline (UniversalSieve):")
    f1_seen, f1_unseen = test_generalization(UniversalSieve, env_class)
    print(f"  SEEN: {f1_seen:.1%}, UNSEEN: {f1_unseen:.1%}, Gap: {f1_seen - f1_unseen:.1%}")
    
    print("Fidelity-Enhanced:")
    f1_seen, f1_unseen = test_generalization(FidelitySieve, env_class)
    print(f"  SEEN: {f1_seen:.1%}, UNSEEN: {f1_unseen:.1%}, Gap: {f1_seen - f1_unseen:.1%}")

print("\n" + "="*70)
