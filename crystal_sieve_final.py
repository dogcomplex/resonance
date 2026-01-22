"""
CRYSTAL SIEVE FINAL - Annealing-Based Rule Learning

The key discovery: Anneal RULES not CLASSES.

Physical analogy refined:
- CLASSES = atomic arrangement (what atoms bond with what)
- RULES = crystal structure (macroscopic properties)

Standard annealing tries to find optimal atomic arrangement.
But the USEFUL property is the crystal structure (rules).

Better approach:
1. Fix atomic arrangement (conservative class merging)
2. Generate candidate crystal structures (multiple rules)
3. Anneal the structures: test against reality, keep what works
4. Final crystal = validated rules that generalize

This combines:
- Wave interference (token equivalence via cosine)
- Crystal annealing (rule validation and selection)
- Purity filtering (only keep rules that predict consistently)
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List
import math

class CrystalSieve:
    """
    Final annealing-based sieve.
    
    Key innovation: Anneal rules, not classes.
    
    Phase 1: NUCLEATION - Build token classes via interference
    Phase 2: GROWTH - Generate candidate rules
    Phase 3: ANNEAL - Filter rules by validation performance
    Phase 4: CRYSTAL - Final rule set
    """
    
    def __init__(self, 
                 coherence=0.95,      # Token merge threshold
                 n_anneal_rounds=3,   # Validation rounds
                 cold_threshold=0.5,  # Rule acceptance temperature
                 min_energy=5):       # Minimum observations per token
        
        self.coherence = coherence
        self.n_anneal_rounds = n_anneal_rounds
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        
        # Wave state
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        
        # Data
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        
        # Results
        self.classes = {}
        self.rules = []
    
    def observe(self, state: Set, action: int, next_state: Set):
        """Accumulate observations."""
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | 
                          {f"-{t}" for t in (before-after)})
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'before': before,
            'effect': effect,
            'action': action
        })
        
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
    
    def _interference(self, t1: str, t2: str) -> float:
        """Compute interference (cosine similarity) between wave patterns."""
        w1, w2 = self.waves[t1], self.waves[t2]
        e1, e2 = self.energy[t1], self.energy[t2]
        
        if e1 == 0 or e2 == 0:
            return 0.0
        
        all_effects = set(w1.keys()) | set(w2.keys())
        
        dot = sum((w1.get(e, 0)/e1) * (w2.get(e, 0)/e2) for e in all_effects)
        norm1 = sum((w1.get(e, 0)/e1)**2 for e in all_effects) ** 0.5
        norm2 = sum((w2.get(e, 0)/e2)**2 for e in all_effects) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def _structural_match(self, t1: str, t2: str) -> bool:
        """Check structural compatibility for merging."""
        if '_' not in t1 or '_' not in t2:
            return True
        p1, p2 = t1.split('_'), t2.split('_')
        if len(p1) != len(p2):
            return False
        return sum(1 for a, b in zip(p1, p2) if a != b) <= 1
    
    def _phase1_nucleation(self):
        """
        NUCLEATION: Build token equivalence classes.
        
        Uses high coherence threshold for conservative merging.
        """
        tokens = [t for t in self.waves if self.energy[t] >= self.min_energy]
        self.classes = {t: t for t in self.waves}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2:
                    continue
                if not self._structural_match(t1, t2):
                    continue
                if self._interference(t1, t2) > self.coherence:
                    self.classes[t2] = self.classes[t1]
    
    def _phase2_growth(self) -> List[Dict]:
        """
        GROWTH: Generate candidate rules.
        
        Creates multiple rules per (action, effect):
        - Full intersection (most specific)
        - Partial intersections (more general)
        """
        def abstract(s):
            return frozenset(self.classes.get(t, t) for t in s)
        
        by_ae = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
                key = (obs['action'], abstract(obs['effect']))
                by_ae[key].append(abstract(obs['before']))
        
        candidates = []
        
        for (action, effect), states in by_ae.items():
            if len(states) < 2:
                continue
            
            # Full intersection
            sample = random.sample(states, min(10, len(states)))
            lhs_full = frozenset.intersection(*sample)
            if lhs_full:
                candidates.append({
                    'lhs': lhs_full,
                    'effect': effect,
                    'action': action,
                    'support': len(states),
                    'temperature': 0.0
                })
            
            # Partial intersections (if enough data)
            if len(states) >= 4:
                for _ in range(3):
                    subset = random.sample(states, len(states) // 2)
                    lhs_partial = frozenset.intersection(*subset)
                    if lhs_partial and lhs_partial != lhs_full:
                        candidates.append({
                            'lhs': lhs_partial,
                            'effect': effect,
                            'action': action,
                            'support': len(subset),
                            'temperature': 0.3
                        })
        
        return candidates
    
    def _phase3_anneal(self, candidates: List[Dict]) -> List[Dict]:
        """
        ANNEAL: Filter rules by validation performance.
        
        Rules that consistently predict correctly get "cold" (confident).
        Rules that fail get "hot" (rejected).
        """
        def abstract(s):
            return frozenset(self.classes.get(t, t) for t in s)
        
        random.shuffle(self.observations)
        n_test = len(self.observations) // 5
        
        for round_i in range(self.n_anneal_rounds):
            test_start = (round_i * n_test) % len(self.observations)
            test_obs = self.observations[test_start:test_start + n_test]
            
            for rule in candidates:
                # Score rule on test data
                tp, fp, fn = 0, 0, 0
                
                for obs in test_obs:
                    state = abstract(obs['before'])
                    actual = abstract(obs['effect'])
                    
                    if rule['action'] == obs['action'] and rule['lhs'] <= state:
                        predicted = rule['effect']
                        for e in predicted:
                            if e in actual: tp += 1
                            else: fp += 1
                        for e in actual:
                            if e not in predicted: fn += 1
                
                # F1 score
                if tp + fp + fn > 0:
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                else:
                    f1 = 0.5  # Neutral if no applicable data
                
                # Update temperature (exponential moving average)
                rule['temperature'] = rule['temperature'] * 0.5 + (1 - f1) * 0.5
        
        # Keep cold (confident) rules
        cold_rules = [r for r in candidates if r['temperature'] < self.cold_threshold]
        
        # Sort by specificity then support
        cold_rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
        
        return cold_rules
    
    def build(self):
        """Build the crystal sieve through all phases."""
        self._phase1_nucleation()
        candidates = self._phase2_growth()
        self.rules = self._phase3_anneal(candidates)
    
    def predict(self, state: Set, action: int) -> Set:
        """Predict using crystallized rules."""
        key = (frozenset(state), action)
        
        # Exact match (memorized)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        # Rule match (generalized)
        abstract_state = frozenset(self.classes.get(t, t) for t in state)
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        
        return set()


if __name__ == "__main__":
    print("CrystalSieve Final")
    print("="*50)
    print("""
    Algorithm: NUCLEATION -> GROWTH -> ANNEAL -> CRYSTAL
    
    Phase 1 (NUCLEATION): Build token classes via wave interference
    Phase 2 (GROWTH): Generate candidate rules via intersection
    Phase 3 (ANNEAL): Filter rules by validation performance  
    Phase 4 (CRYSTAL): Final validated rule set
    
    Key insight: Anneal RULES not CLASSES.
    The atomic structure (classes) is fixed; we anneal the
    macroscopic properties (rules) to find what generalizes.
    """)
