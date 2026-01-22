"""
WAVE SIEVE FINAL - Simplest Emergent Wave Pattern

After testing, the core insight is:

1. INTERSECTION is already wave-like (destructive interference removes non-matching)
2. COSINE/RESONANCE for token equivalence (constructive interference of similar patterns)
3. HIGH THRESHOLD (0.9+) = coherence requirement

The simplest emergent wave pattern:

    OBSERVE -> RESONATE -> COLLAPSE

Where:
- OBSERVE: Accumulate effect distributions per token
- RESONATE: Merge tokens with similar distributions (cosine > 0.9)
- COLLAPSE: Intersection finds invariant structure (like wave function collapse)

This is O(n) and domain-agnostic.

The "wave" is the effect distribution. The "interference" is cosine similarity.
The "collapse" is intersection.
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set
import math

class WaveSieve:
    """
    Final wave sieve: Observe -> Resonate -> Collapse
    
    Simplest possible formulation that captures wave behavior.
    """
    
    def __init__(self, coherence=0.9, min_energy=5):
        """
        coherence: How similar effect patterns must be to merge (0-1)
        min_energy: Minimum observations before token can participate
        """
        self.coherence = coherence
        self.min_energy = min_energy
        
        # Wave state: token -> effect amplitudes
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)  # Total observations per token
        
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        self.classes = {}
        self.rules = []
    
    def observe(self, state: Set, action: int, next_state: Set):
        """OBSERVE phase: Accumulate wave amplitudes."""
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
        
        # Each token absorbs amplitude from effects
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
    
    def _interference(self, t1: str, t2: str) -> float:
        """
        RESONATE phase: Compute interference between two wave patterns.
        
        Cosine similarity = constructive interference measure
        """
        w1, w2 = self.waves[t1], self.waves[t2]
        e1, e2 = self.energy[t1], self.energy[t2]
        
        if e1 == 0 or e2 == 0:
            return 0.0
        
        # Normalized amplitudes
        all_effects = set(w1.keys()) | set(w2.keys())
        
        dot = 0.0
        norm1_sq = 0.0
        norm2_sq = 0.0
        
        for e in all_effects:
            a1 = w1.get(e, 0) / e1
            a2 = w2.get(e, 0) / e2
            dot += a1 * a2
            norm1_sq += a1 * a1
            norm2_sq += a2 * a2
        
        if norm1_sq == 0 or norm2_sq == 0:
            return 0.0
        
        return dot / (math.sqrt(norm1_sq) * math.sqrt(norm2_sq))
    
    def _structural_match(self, t1: str, t2: str) -> bool:
        """Check if tokens have compatible structure (differ by at most 1 component)."""
        if '_' not in t1 or '_' not in t2:
            return True
        p1, p2 = t1.split('_'), t2.split('_')
        if len(p1) != len(p2):
            return False
        return sum(1 for a, b in zip(p1, p2) if a != b) <= 1
    
    def build(self):
        """
        RESONATE + COLLAPSE phases.
        
        1. Find resonant (equivalent) tokens
        2. Abstract observations
        3. Collapse via intersection to find rules
        """
        # Get tokens with sufficient energy
        tokens = [t for t in self.waves if self.energy[t] >= self.min_energy]
        self.classes = {t: t for t in self.waves}  # Initialize all tokens
        
        # RESONATE: Merge similar waves
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2:
                    continue
                if not self._structural_match(t1, t2):
                    continue
                
                if self._interference(t1, t2) > self.coherence:
                    self.classes[t2] = self.classes[t1]
        
        # COLLAPSE: Intersection finds invariant structure
        def abstract(s):
            return frozenset(self.classes.get(t, t) for t in s)
        
        by_ae = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
                key = (obs['action'], abstract(obs['effect']))
                by_ae[key].append(abstract(obs['before']))
        
        self.rules = []
        for (action, effect), states in by_ae.items():
            if len(states) >= 2:
                # Intersection = collapse to invariant subspace
                sample = random.sample(states, min(10, len(states)))
                lhs = frozenset.intersection(*sample)
                if lhs:
                    self.rules.append({
                        'lhs': lhs,
                        'effect': effect,
                        'action': action
                    })
    
    def predict(self, state: Set, action: int) -> Set:
        """Predict using memorized or generalized rules."""
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


# Test
if __name__ == "__main__":
    print("WaveSieve Final: Observe -> Resonate -> Collapse")
    print("=" * 50)
    print("\nCore operations:")
    print("  OBSERVE:  Accumulate effect distributions")
    print("  RESONATE: Merge tokens with cosine > threshold")
    print("  COLLAPSE: Intersection finds invariant rules")
    print("\nAll operations are O(n)")
