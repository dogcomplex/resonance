"""
UNIFIED TEMPORAL V2 - Filter low-support velocity rules

Key fix: Velocity rules need HIGHER support threshold
since they're more specific.
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List

class UnifiedTemporalV2:
    def __init__(self, coherence=0.95, cold_threshold=0.5, min_energy=5,
                 vel_min_support=5):  # Higher support for velocity rules
        self.coherence = coherence
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        self.vel_min_support = vel_min_support
        
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        self.prev_state = None
        self.classes = {}
        self.rules = []
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | 
                          {f"-{t}" for t in (before-after)})
        
        delta = frozenset()
        if self.prev_state:
            delta = frozenset({f"V+{t}" for t in (before - self.prev_state)} |
                             {f"V-{t}" for t in (self.prev_state - before)})
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'state': before, 'delta': delta, 'action': action, 'effect': effect
        })
        
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
        
        self.prev_state = before
    
    def reset_episode(self): self.prev_state = None
    
    def _interference(self, t1: str, t2: str) -> float:
        w1, w2 = self.waves[t1], self.waves[t2]
        e1, e2 = self.energy[t1], self.energy[t2]
        if e1 == 0 or e2 == 0: return 0.0
        all_e = set(w1) | set(w2)
        dot = sum((w1.get(e,0)/e1)*(w2.get(e,0)/e2) for e in all_e)
        n1 = sum((w1.get(e,0)/e1)**2 for e in all_e)**0.5
        n2 = sum((w2.get(e,0)/e2)**2 for e in all_e)**0.5
        return dot/(n1*n2) if n1 and n2 else 0.0
    
    def _structural_match(self, t1: str, t2: str) -> bool:
        if '_' not in t1 or '_' not in t2: return True
        p1, p2 = t1.split('_'), t2.split('_')
        return len(p1)==len(p2) and sum(a!=b for a,b in zip(p1,p2))<=1
    
    def _build_classes(self):
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
            if t.startswith('V+') or t.startswith('V-'):
                base = t[2:]
                result.add(t[:2] + self.classes.get(base, base))
            else:
                result.add(self.classes.get(t, t))
        return frozenset(result)
    
    def _generate_candidates(self) -> List[Dict]:
        # Generate position-only candidates (no velocity in LHS)
        by_ae_pos = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
                key = (obs['action'], self._abstract(obs['effect']))
                by_ae_pos[key].append(self._abstract(obs['state']))
        
        # Generate position+velocity candidates
        by_ae_vel = defaultdict(list)
        for obs in self.observations:
            if obs['effect'] and obs['delta']:
                extended = obs['state'] | obs['delta']
                key = (obs['action'], self._abstract(obs['effect']))
                by_ae_vel[key].append(self._abstract(extended))
        
        candidates = []
        
        # Position-only rules
        for (action, effect), states in by_ae_pos.items():
            if len(states) < 2: continue
            sample = random.sample(states, min(10, len(states)))
            lhs = frozenset.intersection(*sample)
            if lhs:
                candidates.append({
                    'lhs': lhs, 'effect': effect, 'action': action,
                    'support': len(states), 'temperature': 0.0,
                    'has_velocity': False
                })
        
        # Velocity rules (need higher support)
        for (action, effect), states in by_ae_vel.items():
            if len(states) < self.vel_min_support: continue
            sample = random.sample(states, min(10, len(states)))
            lhs = frozenset.intersection(*sample)
            if lhs and any(t.startswith('V') for t in lhs):
                candidates.append({
                    'lhs': lhs, 'effect': effect, 'action': action,
                    'support': len(states), 'temperature': 0.0,
                    'has_velocity': True
                })
        
        return candidates
    
    def _anneal(self, candidates):
        random.shuffle(self.observations)
        n = len(self.observations)
        for rnd in range(3):
            test = self.observations[rnd*n//5:(rnd+1)*n//5]
            for rule in candidates:
                tp, fp, fn = 0, 0, 0
                for obs in test:
                    if rule['has_velocity']:
                        extended = obs['state'] | obs['delta']
                    else:
                        extended = obs['state']
                    state = self._abstract(extended)
                    actual = self._abstract(obs['effect'])
                    if rule['action'] == obs['action'] and rule['lhs'] <= state:
                        for e in rule['effect']:
                            if e in actual: tp += 1
                            else: fp += 1
                        for e in actual:
                            if e not in rule['effect']: fn += 1
                if tp+fp+fn > 0:
                    f1 = 2*tp/(2*tp+fp+fn)
                else:
                    f1 = 0.5
                rule['temperature'] = rule['temperature']*0.5 + (1-f1)*0.5
    
    def build(self):
        self._build_classes()
        candidates = self._generate_candidates()
        self._anneal(candidates)
        self.rules = [r for r in candidates if r['temperature'] < self.cold_threshold]
        # Sort: velocity rules first (more specific), then by LHS size
        self.rules.sort(key=lambda r: (-r['has_velocity'], -len(r['lhs']), -r['support']))
    
    def predict(self, state: Set, action: int, delta: Set = None) -> Set:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        delta = frozenset(delta or set())
        
        # Try velocity rules first
        extended = frozenset(state) | delta
        abstract_ext = self._abstract(extended)
        
        for r in self.rules:
            if r['action'] == action:
                if r['has_velocity']:
                    if r['lhs'] <= abstract_ext:
                        return set(r['effect'])
                else:
                    if r['lhs'] <= self._abstract(frozenset(state)):
                        return set(r['effect'])
        return set()

print("UnifiedTemporalV2 loaded!")
