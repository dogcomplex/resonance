"""
RESONANCE CRYSTAL FIXED - Match CrystalSieve exactly when delta=False
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List

class ResonanceCrystalFixed:
    def __init__(self, 
                 coherence=0.95,
                 min_energy=5,
                 cold_threshold=0.5,
                 n_folds=3,           # Match Crystal's 3 rounds
                 include_delta=True):
        
        self.coherence = coherence
        self.min_energy = min_energy
        self.cold_threshold = cold_threshold
        self.n_folds = n_folds
        self.include_delta = include_delta
        
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        self.observations = []
        self.prev_state = None
        self.exact = defaultdict(lambda: defaultdict(int))
        self.classes = {}
        self.rules = []
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after - before)} |
                          {f"-{t}" for t in (before - after)})
        
        delta = frozenset()
        if self.include_delta and self.prev_state is not None:
            delta = frozenset(
                {f"D+{t}" for t in (before - self.prev_state)} |
                {f"D-{t}" for t in (self.prev_state - before)}
            )
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'before': before,  # Match Crystal's key name
            'delta': delta,
            'action': action,
            'effect': effect
        })
        
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
        
        if self.include_delta:
            for token in delta:
                self.energy[token] += 1
                for e in effect:
                    self.waves[token][e] += 1.0
        
        self.prev_state = before
    
    def reset_episode(self):
        self.prev_state = None
    
    def _interference(self, t1: str, t2: str) -> float:
        w1, w2 = self.waves[t1], self.waves[t2]
        e1, e2 = self.energy[t1], self.energy[t2]
        if e1 == 0 or e2 == 0: return 0.0
        all_effects = set(w1.keys()) | set(w2.keys())
        dot = sum((w1.get(e, 0)/e1) * (w2.get(e, 0)/e2) for e in all_effects)
        norm1 = sum((w1.get(e, 0)/e1)**2 for e in all_effects) ** 0.5
        norm2 = sum((w2.get(e, 0)/e2)**2 for e in all_effects) ** 0.5
        if norm1 == 0 or norm2 == 0: return 0.0
        return dot / (norm1 * norm2)
    
    def _structural_match(self, t1: str, t2: str) -> bool:
        b1 = t1[2:] if t1.startswith('D+') or t1.startswith('D-') else t1
        b2 = t2[2:] if t2.startswith('D+') or t2.startswith('D-') else t2
        if '_' not in b1 or '_' not in b2: return True
        p1, p2 = b1.split('_'), b2.split('_')
        if len(p1) != len(p2): return False
        return sum(1 for a, b in zip(p1, p2) if a != b) <= 1
    
    def _resonate(self):
        tokens = [t for t in self.waves if self.energy[t] >= self.min_energy]
        self.classes = {t: t for t in self.waves}
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2: continue
                if not self._structural_match(t1, t2): continue
                if self._interference(t1, t2) > self.coherence:
                    self.classes[t2] = self.classes[t1]
    
    def _abstract(self, s: FrozenSet) -> FrozenSet:
        result = set()
        for t in s:
            if t.startswith('D+') or t.startswith('D-'):
                base = t[2:]
                result.add(t[:2] + self.classes.get(base, base))
            else:
                result.add(self.classes.get(t, t))
        return frozenset(result)
    
    def _generate_candidates(self) -> List[Dict]:
        by_ae = defaultdict(list)
        
        for obs in self.observations:
            if not obs['effect']: continue
            
            if self.include_delta:
                extended = obs['before'] | obs['delta']
            else:
                extended = obs['before']
            
            key = (obs['action'], self._abstract(obs['effect']))
            by_ae[key].append(self._abstract(extended))
        
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
                for _ in range(3):  # Match Crystal's 3 partials
                    subset = random.sample(states, len(states) // 2)
                    lhs_partial = frozenset.intersection(*subset)
                    if lhs_partial and lhs_partial != lhs_full:
                        candidates.append({
                            'lhs': lhs_partial, 'effect': effect, 'action': action,
                            'support': len(subset), 'temperature': 0.3
                        })
        
        return candidates
    
    def _anneal(self, candidates: List[Dict]):
        random.shuffle(self.observations)
        n = len(self.observations)
        n_test = n // 5  # Match Crystal
        
        for fold_idx in range(self.n_folds):
            start = (fold_idx * n_test) % n
            fold = self.observations[start:start + n_test]
            
            for rule in candidates:
                tp, fp, fn = 0, 0, 0
                
                for obs in fold:
                    if self.include_delta:
                        extended = obs['before'] | obs['delta']
                    else:
                        extended = obs['before']
                    
                    state = self._abstract(extended)
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
                    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
                else:
                    f1 = 0.5
                
                rule['temperature'] = rule['temperature'] * 0.5 + (1 - f1) * 0.5
    
    def build(self):
        self._resonate()
        candidates = self._generate_candidates()
        self._anneal(candidates)
        self.rules = [c for c in candidates if c['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
    
    def predict(self, state: Set, action: int, delta: Set = None) -> Set:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        if self.include_delta and delta:
            extended = frozenset(state) | frozenset(delta)
        else:
            extended = frozenset(state)
        
        abstract_state = self._abstract(extended)
        
        for rule in self.rules:
            if rule['action'] == action and rule['lhs'] <= abstract_state:
                return set(rule['effect'])
        
        return set()

print("ResonanceCrystalFixed loaded!")
