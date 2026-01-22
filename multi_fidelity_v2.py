"""
MULTI-FIDELITY INDUCTION V2

Key insight: Fidelity applies to BOTH LHS AND RHS.

Coarse effects: "+has_cooked_material" (ignoring specific material)
Fine effects: "+has_cooked_iron" (specific material)

This allows generalization like:
  has_raw_material + fire → has_cooked_material
even when specific effects differ.
"""

import random
import re
from typing import Set, FrozenSet, List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import math


def coarsen_token(token: str) -> str:
    """Convert a fine token to its coarse version."""
    # Pattern: has_TYPE_MATERIAL → has_TYPE_material
    # e.g., has_raw_iron → has_raw_material
    # e.g., +has_cooked_copper → +has_cooked_material
    
    prefixes = ['has_raw_', 'has_cooked_', '+has_raw_', '-has_raw_', 
                '+has_cooked_', '-has_cooked_']
    
    for prefix in prefixes:
        if token.startswith(prefix):
            # Keep the prefix, replace specific with "material"
            suffix = token[len(prefix):]
            if suffix not in ['material']:  # Don't double-coarsen
                return prefix + "material"
    
    # Position coarsening: pos_X → region_X//4
    if token.startswith('pos_') or token.startswith('+pos_') or token.startswith('-pos_'):
        # This is a simple example - real coarsening would be domain-specific
        pass
    
    return token  # No coarsening available


def coarsen_tokens(tokens: FrozenSet[str]) -> FrozenSet[str]:
    """Coarsen a whole token set, keeping both fine and coarse."""
    result = set(tokens)
    for t in tokens:
        coarse = coarsen_token(t)
        if coarse != t:
            result.add(coarse)
    return frozenset(result)


@dataclass
class Rule:
    lhs: FrozenSet[str]
    effect: FrozenSet[str]
    action: int
    probability: float
    support: int
    is_coarse: bool = False


class MultiFidelityV2:
    """
    Multi-fidelity learner with effect abstraction.
    
    Key feature: Groups by COARSE effects when fine effects differ.
    """
    
    def __init__(self, n_actions: int = 10, num_samples: int = 100):
        self.n_actions = n_actions
        self.num_samples = num_samples
        
        self.observations: List[Dict] = []
        self.exact_matches: Dict[Tuple[FrozenSet, int], Dict[FrozenSet, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: List[Rule] = []
        self._dirty = True
    
    def observe(self, state: Set[str], action: int, next_state: Set[str], timestamp: Optional[int] = None):
        before_fs = frozenset(state)
        after_fs = frozenset(next_state)
        
        added = after_fs - before_fs
        removed = before_fs - after_fs
        effects = frozenset({f"+{t}" for t in added} | {f"-{t}" for t in removed})
        
        # Also compute coarse effects
        coarse_effects = frozenset(coarsen_token(e) for e in effects)
        
        self.exact_matches[(before_fs, action)][effects] += 1
        
        self.observations.append({
            'before': before_fs,
            'after': after_fs,
            'action': action,
            'effect': effects,
            'effect_coarse': coarse_effects,
            'timestamp': timestamp or len(self.observations)
        })
        
        self._dirty = True
    
    def _compute_rules(self):
        if not self._dirty:
            return
        
        self.rules = []
        
        # FINE RULES: Group by exact (action, effect)
        by_fine = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
                by_fine[(obs['action'], obs['effect'])].append(obs['before'])
        
        for (action, effect), positives in by_fine.items():
            if len(positives) < 2:
                continue
            
            # Sample and intersect
            for _ in range(self.num_samples):
                k = max(2, len(positives) * 2 // 3)
                sample = random.sample(positives, min(k, len(positives)))
                lhs = frozenset.intersection(*sample)
                if lhs:
                    applicable = [o for o in self.observations 
                                 if o['action'] == action and lhs <= o['before']]
                    hits = sum(1 for o in applicable if o['effect'] == effect)
                    prob = hits / len(applicable) if applicable else 0
                    
                    if prob > 0.3:
                        self.rules.append(Rule(
                            lhs=lhs, effect=effect, action=action,
                            probability=prob, support=len(applicable),
                            is_coarse=False
                        ))
        
        # COARSE RULES: Group by (action, coarse_effect)
        by_coarse = defaultdict(list)
        for obs in self.observations:
            if obs['effect_coarse']:
                by_coarse[(obs['action'], obs['effect_coarse'])].append(obs['before'])
        
        for (action, effect_coarse), positives in by_coarse.items():
            if len(positives) < 2:
                continue
            
            # Coarsen the LHS too
            coarse_positives = [coarsen_tokens(p) for p in positives]
            
            for _ in range(self.num_samples):
                k = max(2, len(coarse_positives) * 2 // 3)
                sample = random.sample(coarse_positives, min(k, len(coarse_positives)))
                lhs = frozenset.intersection(*sample)
                if lhs:
                    # Check against coarse effects
                    applicable = [o for o in self.observations 
                                 if o['action'] == action and lhs <= coarsen_tokens(o['before'])]
                    hits = sum(1 for o in applicable if o['effect_coarse'] == effect_coarse)
                    prob = hits / len(applicable) if applicable else 0
                    
                    if prob > 0.3:
                        self.rules.append(Rule(
                            lhs=lhs, effect=effect_coarse, action=action,
                            probability=prob, support=len(applicable),
                            is_coarse=True
                        ))
        
        # Deduplicate
        seen = set()
        unique_rules = []
        for r in self.rules:
            key = (r.lhs, r.effect, r.action)
            if key not in seen:
                seen.add(key)
                unique_rules.append(r)
        self.rules = unique_rules
        
        self._dirty = False
    
    def predict(self, state: Set[str], action: int) -> Set[str]:
        state_fs = frozenset(state)
        state_coarse = coarsen_tokens(state_fs)
        
        # Exact match first
        key = (state_fs, action)
        if key in self.exact_matches:
            obs = self.exact_matches[key]
            if obs:
                return set(max(obs.items(), key=lambda x: x[1])[0])
        
        # Try rules
        self._compute_rules()
        
        best_rule = None
        best_score = -1
        
        for rule in self.rules:
            if rule.action != action:
                continue
            
            # Check match (coarse rules match against coarse state)
            if rule.is_coarse:
                if not rule.lhs <= state_coarse:
                    continue
            else:
                if not rule.lhs <= state_fs:
                    continue
            
            # Score: prefer specific > coarse, then by probability
            score = rule.probability
            if not rule.is_coarse:
                score += 1.0  # Prefer fine rules
            score += len(rule.lhs) * 0.01  # Prefer specific LHS
            
            if score > best_score:
                best_score = score
                best_rule = rule
        
        if best_rule and best_rule.probability >= 0.3:
            return set(best_rule.effect)
        
        return set()
    
    def close(self):
        pass


# Test
if __name__ == "__main__":
    random.seed(42)
    
    class GeneralizationCrafting:
        TRAIN_MATERIALS = ['iron', 'copper', 'tin']
        TEST_MATERIALS = ['gold', 'silver', 'bronze']
        
        def __init__(self, seed=42, train_mode=True):
            self.rng = random.Random(seed)
            self.train_mode = train_mode
            self.reset()
        
        def reset(self, seed=None):
            if seed: self.rng = random.Random(seed)
            materials = self.TRAIN_MATERIALS if self.train_mode else self.TEST_MATERIALS
            mat = self.rng.choice(materials)
            self.inventory = {f"raw_{mat}": 1, "fire": 1}
            return self._state()
        
        def _state(self):
            tokens = set()
            for item, qty in self.inventory.items():
                if qty > 0:
                    tokens.add(f"has_{item}")
                    if item.startswith("raw_"):
                        tokens.add("has_raw_material")
                    if item.startswith("cooked_"):
                        tokens.add("has_cooked_material")
            return tokens
        
        def get_valid_actions(self):
            return [0]
        
        def step(self, action):
            if action == 0:
                for item in list(self.inventory.keys()):
                    if item.startswith("raw_") and self.inventory.get(item, 0) > 0:
                        self.inventory[item] -= 1
                        cooked = item.replace("raw_", "cooked_")
                        self.inventory[cooked] = self.inventory.get(cooked, 0) + 1
                        break
            self.inventory = {k: v for k, v in self.inventory.items() if v > 0}
            return self._state(), 0, False, {}
    
    print("Multi-Fidelity V2 - Effect Abstraction Test")
    print("="*60)
    
    learner = MultiFidelityV2(n_actions=5)
    
    # Train
    for ep in range(100):
        env = GeneralizationCrafting(seed=ep, train_mode=True)
        state = env.reset(seed=ep)
        next_state, _, _, _ = env.step(0)
        learner.observe(state, 0, next_state, timestamp=ep)
    
    # Check rules
    learner._compute_rules()
    print(f"Found {len(learner.rules)} rules")
    print("\nCoarse rules:")
    for r in learner.rules:
        if r.is_coarse:
            print(f"  {r.lhs} → {r.effect} ({r.probability:.0%})")
    
    # Test on UNSEEN material
    print("\nTest on unseen material (gold):")
    env = GeneralizationCrafting(seed=50000, train_mode=False)
    state = env.reset(seed=50000)
    print(f"State: {state}")
    
    pred = learner.predict(state, 0)
    print(f"Prediction: {pred}")
    
    next_state, _, _, _ = env.step(0)
    actual = {f"+{t}" for t in next_state - state} | {f"-{t}" for t in state - next_state}
    print(f"Actual: {actual}")
    
    # Check if coarse prediction matches coarse actual
    pred_coarse = {coarsen_token(t) for t in pred}
    actual_coarse = {coarsen_token(t) for t in actual}
    print(f"\nCoarse pred: {pred_coarse}")
    print(f"Coarse actual: {actual_coarse}")
    match = pred_coarse & actual_coarse
    print(f"Overlap: {match}")
