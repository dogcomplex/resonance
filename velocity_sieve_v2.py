"""
VELOCITY SIEVE V2 - With proper abstraction

Issues with v1:
1. Velocity tokens not being abstracted (too sparse)
2. Low support for velocity rules
3. Velocity tokens treated independently from position tokens

New approach:
1. Build position classes FIRST
2. Velocity tokens inherit their position class
3. D+{pos_token} -> D+{pos_class}

This ensures velocity rules can generalize across positions.
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List

class VelocitySieveV2:
    """
    Velocity sieve with proper token abstraction.
    
    Key fix: Delta tokens inherit the class of their base token.
    D+pixel_3_4 inherits the class of pixel_3_4.
    """
    
    def __init__(self, coherence=0.95, cold_threshold=0.5, min_energy=5):
        self.coherence = coherence
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        
        # Position waves (for building classes)
        self.pos_waves = defaultdict(lambda: defaultdict(float))
        self.pos_energy = defaultdict(float)
        
        # All observations
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        
        self.prev_state = None
        
        self.classes = {}  # Position classes
        self.rules = []
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state)
        after = frozenset(next_state)
        
        effect = frozenset({f"+{t}" for t in (after-before)} | 
                          {f"-{t}" for t in (before-after)})
        
        # Compute delta
        if self.prev_state is not None:
            delta = frozenset({f"D+{t}" for t in (before - self.prev_state)} |
                             {f"D-{t}" for t in (self.prev_state - before)})
        else:
            delta = frozenset()
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'state': before,
            'delta': delta,
            'action': action,
            'effect': effect
        })
        
        # Build position waves (only for position tokens)
        for token in before:
            self.pos_energy[token] += 1
            for e in effect:
                self.pos_waves[token][e] += 1.0
        
        self.prev_state = before
    
    def reset_episode(self):
        self.prev_state = None
    
    def _interference(self, t1: str, t2: str) -> float:
        w1, w2 = self.pos_waves[t1], self.pos_waves[t2]
        e1, e2 = self.pos_energy[t1], self.pos_energy[t2]
        
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
        if '_' not in t1 or '_' not in t2:
            return True
        p1, p2 = t1.split('_'), t2.split('_')
        if len(p1) != len(p2):
            return False
        return sum(1 for a, b in zip(p1, p2) if a != b) <= 1
    
    def _build_classes(self):
        """Build position classes only."""
        tokens = [t for t in self.pos_waves if self.pos_energy[t] >= self.min_energy]
        self.classes = {t: t for t in self.pos_waves}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2:
                    continue
                if not self._structural_match(t1, t2):
                    continue
                if self._interference(t1, t2) > self.coherence:
                    self.classes[t2] = self.classes[t1]
    
    def _abstract_token(self, token: str) -> str:
        """
        Abstract a token, handling delta tokens specially.
        
        D+pixel_3_4 -> D+{class of pixel_3_4}
        """
        if token.startswith('D+'):
            base = token[2:]
            base_class = self.classes.get(base, base)
            return f"D+{base_class}"
        elif token.startswith('D-'):
            base = token[2:]
            base_class = self.classes.get(base, base)
            return f"D-{base_class}"
        else:
            return self.classes.get(token, token)
    
    def _abstract_set(self, s: FrozenSet) -> FrozenSet:
        return frozenset(self._abstract_token(t) for t in s)
    
    def _generate_candidates(self) -> List[Dict]:
        by_ae = defaultdict(list)
        
        for obs in self.observations:
            if obs['effect']:
                # Abstract state + delta together
                extended = obs['state'] | obs['delta']
                abstract_extended = self._abstract_set(extended)
                abstract_effect = self._abstract_set(obs['effect'])
                
                key = (obs['action'], abstract_effect)
                by_ae[key].append(abstract_extended)
        
        candidates = []
        
        for (action, effect), states in by_ae.items():
            if len(states) < 2:
                continue
            
            sample = random.sample(states, min(10, len(states)))
            lhs = frozenset.intersection(*sample)
            
            if lhs:
                has_vel = any(t.startswith('D') for t in lhs)
                candidates.append({
                    'lhs': lhs,
                    'effect': effect,
                    'action': action,
                    'support': len(states),
                    'temperature': 0.0,
                    'has_velocity': has_vel
                })
            
            if len(states) >= 4:
                for _ in range(3):
                    subset = random.sample(states, len(states) // 2)
                    lhs_partial = frozenset.intersection(*subset)
                    if lhs_partial and lhs_partial != lhs:
                        has_vel = any(t.startswith('D') for t in lhs_partial)
                        candidates.append({
                            'lhs': lhs_partial,
                            'effect': effect,
                            'action': action,
                            'support': len(subset),
                            'temperature': 0.2,
                            'has_velocity': has_vel
                        })
        
        return candidates
    
    def _anneal(self, candidates: List[Dict]):
        random.shuffle(self.observations)
        n_test = len(self.observations) // 5
        
        for round_i in range(3):
            test_start = (round_i * n_test) % len(self.observations)
            test_obs = self.observations[test_start:test_start + n_test]
            
            for rule in candidates:
                tp, fp, fn = 0, 0, 0
                
                for obs in test_obs:
                    extended = obs['state'] | obs['delta']
                    state = self._abstract_set(extended)
                    actual = self._abstract_set(obs['effect'])
                    
                    if rule['action'] == obs['action'] and rule['lhs'] <= state:
                        predicted = rule['effect']
                        for e in predicted:
                            if e in actual: tp += 1
                            else: fp += 1
                        for e in actual:
                            if e not in predicted: fn += 1
                
                if tp + fp + fn > 0:
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                else:
                    f1 = 0.5
                
                rule['temperature'] = rule['temperature'] * 0.5 + (1 - f1) * 0.5
    
    def build(self):
        self._build_classes()
        candidates = self._generate_candidates()
        self._anneal(candidates)
        
        self.rules = [r for r in candidates if r['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
    
    def predict(self, state: Set, action: int, delta: Set = None) -> Set:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        if delta is None:
            delta = set()
        
        extended = frozenset(state) | frozenset(delta)
        abstract_state = self._abstract_set(extended)
        
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        
        return set()


print("VelocitySieveV2 loaded!")
