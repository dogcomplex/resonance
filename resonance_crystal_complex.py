"""
RESONANCE CRYSTAL - Complex Wave Extension

Key insight from poor_mans_quantum.txt:
- Real parts at even timesteps (t % 2 == 0)
- Imaginary parts at odd timesteps (t % 2 == 1)
- This gives complex amplitudes that capture PHASE

Two tokens causing same effects at different phases = destructive interference
Two tokens causing same effects at same phase = constructive interference
"""

import random
import cmath
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List

class ResonanceCrystalComplex:
    def __init__(self, 
                 coherence=0.95,
                 min_energy=5,
                 cold_threshold=0.5,
                 n_folds=3,
                 use_complex=True):  # Enable complex waves
        
        self.coherence = coherence
        self.min_energy = min_energy
        self.cold_threshold = cold_threshold
        self.n_folds = n_folds
        self.use_complex = use_complex
        
        # Complex wave: token → effect → complex amplitude
        self.waves = defaultdict(lambda: defaultdict(complex))
        self.energy = defaultdict(float)
        
        self.observations = []
        self.timestep = 0
        self.exact = defaultdict(lambda: defaultdict(int))
        self.classes = {}
        self.rules = []
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after - before)} |
                          {f"-{t}" for t in (before - after)})
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'before': before,
            'action': action,
            'effect': effect,
            'timestep': self.timestep
        })
        
        # Accumulate complex wave signature
        # Even timesteps → real part
        # Odd timesteps → imaginary part
        for token in before:
            self.energy[token] += 1
            for e in effect:
                if self.use_complex:
                    if self.timestep % 2 == 0:
                        self.waves[token][e] += 1.0 + 0j  # Real
                    else:
                        self.waves[token][e] += 0 + 1.0j  # Imaginary
                else:
                    self.waves[token][e] += 1.0 + 0j  # Real only
        
        self.timestep += 1
    
    def reset_episode(self):
        """Reset timestep for new episode."""
        self.timestep = 0
    
    def _complex_interference(self, t1: str, t2: str) -> float:
        """
        Compute interference using complex amplitudes.
        
        This captures phase alignment:
        - Same phase (both real or both imaginary) → high interference
        - Opposite phase → low or negative interference
        """
        w1, w2 = self.waves[t1], self.waves[t2]
        e1, e2 = self.energy[t1], self.energy[t2]
        
        if e1 == 0 or e2 == 0:
            return 0.0
        
        all_effects = set(w1.keys()) | set(w2.keys())
        
        # Complex dot product
        dot = sum((w1.get(e, 0)/e1) * (w2.get(e, 0)/e2).conjugate() 
                  for e in all_effects)
        
        # Magnitudes
        norm1 = sum(abs(w1.get(e, 0)/e1)**2 for e in all_effects) ** 0.5
        norm2 = sum(abs(w2.get(e, 0)/e2)**2 for e in all_effects) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Return magnitude of complex interference
        # High when in-phase, low when out-of-phase
        return abs(dot) / (norm1 * norm2)
    
    def _structural_match(self, t1: str, t2: str) -> bool:
        if '_' not in t1 or '_' not in t2: return True
        p1, p2 = t1.split('_'), t2.split('_')
        if len(p1) != len(p2): return False
        return sum(1 for a, b in zip(p1, p2) if a != b) <= 1
    
    def _resonate(self):
        """Build token classes via complex interference."""
        tokens = [t for t in self.waves if self.energy[t] >= self.min_energy]
        self.classes = {t: t for t in self.waves}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2: continue
                if not self._structural_match(t1, t2): continue
                if self._complex_interference(t1, t2) > self.coherence:
                    self.classes[t2] = self.classes[t1]
    
    def _abstract(self, s: FrozenSet) -> FrozenSet:
        return frozenset(self.classes.get(t, t) for t in s)
    
    def _generate_candidates(self) -> List[Dict]:
        by_ae = defaultdict(list)
        for obs in self.observations:
            if not obs['effect']: continue
            key = (obs['action'], self._abstract(obs['effect']))
            by_ae[key].append(self._abstract(obs['before']))
        
        candidates = []
        for (action, effect), states in by_ae.items():
            if len(states) < 2: continue
            
            sample = random.sample(states, min(10, len(states)))
            lhs_full = frozenset.intersection(*sample)
            
            if lhs_full:
                candidates.append({
                    'lhs': lhs_full, 'effect': effect, 'action': action,
                    'support': len(states), 'temperature': 0.0
                })
            
            if len(states) >= 4:
                for _ in range(3):
                    subset = random.sample(states, len(states) // 2)
                    lhs_partial = frozenset.intersection(*subset)
                    if lhs_partial and lhs_partial != lhs_full:
                        candidates.append({
                            'lhs': lhs_partial, 'effect': effect, 'action': action,
                            'support': len(subset), 'temperature': 0.3
                        })
        
        return candidates
    
    def _anneal(self, candidates: List[Dict]):
        """Anneal: patterns that resonate across folds survive."""
        random.shuffle(self.observations)
        n = len(self.observations)
        n_test = n // 5
        
        for fold_idx in range(self.n_folds):
            start = (fold_idx * n_test) % n
            fold = self.observations[start:start + n_test]
            
            for rule in candidates:
                tp, fp, fn = 0, 0, 0
                
                for obs in fold:
                    state = self._abstract(obs['before'])
                    actual = self._abstract(obs['effect'])
                    
                    if rule['action'] == obs['action'] and rule['lhs'] <= state:
                        for e in rule['effect']:
                            if e in actual: tp += 1
                            else: fp += 1
                        for e in actual:
                            if e not in rule['effect']: fn += 1
                
                if tp + fp + fn > 0:
                    prec = tp / (tp + fp) if tp + fp > 0 else 0
                    rec = tp / (tp + fn) if tp + fn > 0 else 0
                    resonance = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
                else:
                    resonance = 0.5
                
                # High resonance → low temperature → survives
                rule['temperature'] = rule['temperature'] * 0.5 + (1 - resonance) * 0.5
    
    def build(self):
        self._resonate()
        candidates = self._generate_candidates()
        self._anneal(candidates)
        self.rules = [c for c in candidates if c['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
    
    def predict(self, state: Set, action: int) -> Set:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        abstract_state = self._abstract(frozenset(state))
        for rule in self.rules:
            if rule['action'] == action and rule['lhs'] <= abstract_state:
                return set(rule['effect'])
        return set()

print("ResonanceCrystalComplex loaded!")
