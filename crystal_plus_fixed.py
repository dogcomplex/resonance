"""
CRYSTAL PLUS FIXED - Exact copy of Crystal with optional extras
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List

class CrystalPlusFixed:
    def __init__(self, coherence=0.95, cold_threshold=0.5, min_energy=5,
                 n_anneal_rounds=3, use_universal=False):
        self.coherence = coherence
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        self.n_anneal_rounds = n_anneal_rounds
        self.use_universal = use_universal
        
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        self.classes = {}
        self.rules = []
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after - before)} |
                          {f"-{t}" for t in (before - after)})
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({'before': before, 'action': action, 'effect': effect})
        
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
    
    def _interference(self, t1, t2):
        w1, w2 = self.waves[t1], self.waves[t2]
        e1, e2 = self.energy[t1], self.energy[t2]
        if e1 == 0 or e2 == 0: return 0.0
        all_e = set(w1.keys()) | set(w2.keys())
        dot = sum((w1.get(e,0)/e1)*(w2.get(e,0)/e2) for e in all_e)
        n1 = sum((w1.get(e,0)/e1)**2 for e in all_e)**0.5
        n2 = sum((w2.get(e,0)/e2)**2 for e in all_e)**0.5
        return dot/(n1*n2) if n1 and n2 else 0.0
    
    def _structural_match(self, t1, t2):
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
    
    def _abstract(self, s):
        return frozenset(self.classes.get(t, t) for t in s)
    
    def _generate_candidates(self):
        by_ae = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
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
            
            # Exactly 3 partial intersections like Crystal
            if len(states) >= 4:
                for _ in range(3):
                    subset = random.sample(states, len(states)//2)
                    lhs_p = frozenset.intersection(*subset)
                    if lhs_p and lhs_p != lhs_full:
                        candidates.append({
                            'lhs': lhs_p, 'effect': effect, 'action': action,
                            'support': len(subset), 'temperature': 0.3  # Match Crystal
                        })
        return candidates
    
    def _anneal(self, candidates):
        random.shuffle(self.observations)
        n_test = len(self.observations) // 5
        
        for rnd in range(self.n_anneal_rounds):
            test_start = (rnd * n_test) % len(self.observations)
            test = self.observations[test_start:test_start + n_test]
            
            for rule in candidates:
                tp, fp, fn = 0, 0, 0
                for obs in test:
                    state = self._abstract(obs['before'])
                    actual = self._abstract(obs['effect'])
                    if rule['action'] == obs['action'] and rule['lhs'] <= state:
                        for e in rule['effect']:
                            if e in actual: tp += 1
                            else: fp += 1
                        for e in actual:
                            if e not in rule['effect']: fn += 1
                
                if tp+fp+fn > 0:
                    prec = tp/(tp+fp) if tp+fp else 0
                    rec = tp/(tp+fn) if tp+fn else 0
                    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
                else:
                    f1 = 0.5
                
                rule['temperature'] = rule['temperature']*0.5 + (1-f1)*0.5
    
    def _sieve_universal(self, candidates, n_buckets=3):
        buckets = [random.sample(self.observations, len(self.observations)//3) 
                   for _ in range(n_buckets)]
        
        for rule in candidates:
            scores = []
            for bucket in buckets:
                tp, fp, fn = 0, 0, 0
                for obs in bucket[:50]:
                    state = self._abstract(obs['before'])
                    actual = self._abstract(obs['effect'])
                    if rule['action'] == obs['action'] and rule['lhs'] <= state:
                        for e in rule['effect']:
                            if e in actual: tp += 1
                            else: fp += 1
                        for e in actual:
                            if e not in rule['effect']: fn += 1
                if tp+fp+fn > 0:
                    scores.append(2*tp/(2*tp+fp+fn))
            
            if len(scores) >= 2:
                consistency = min(scores) / (max(scores) + 0.01)
                if consistency > 0.8:
                    rule['temperature'] *= 0.95
                elif consistency < 0.5:
                    rule['temperature'] = min(1.0, rule['temperature'] + 0.05)
    
    def build(self):
        self._build_classes()
        candidates = self._generate_candidates()
        self._anneal(candidates)
        
        if self.use_universal:
            self._sieve_universal(candidates)
        
        self.rules = [c for c in candidates if c['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
    
    def predict(self, state: Set, action: int) -> Set:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        abstract_state = self._abstract(frozenset(state))
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        return set()

print("CrystalPlusFixed loaded!")
