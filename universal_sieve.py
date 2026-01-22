"""
UNIVERSAL SIEVE - Cross-universe validation

Rules that work across different seeds have higher "universality".
These are the true laws of the game, not artifacts of specific initial conditions.

Approach:
1. Train on multiple "universes" (different seeds)
2. Generate candidate rules from combined observations
3. Score each rule by how many universes it works in
4. "Universality" = cross-seed consistency
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List

class UniversalSieve:
    def __init__(self, coherence=0.95, cold_threshold=0.5, min_energy=5,
                 n_universes=3):
        self.coherence = coherence
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        self.n_universes = n_universes
        
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        
        # Per-universe observations
        self.universe_obs = defaultdict(list)
        self.current_universe = 0
        
        self.exact = defaultdict(lambda: defaultdict(int))
        self.classes = {}
        self.rules = []
    
    def set_universe(self, universe_id: int):
        self.current_universe = universe_id
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | 
                          {f"-{t}" for t in (before-after)})
        
        self.exact[(before, action)][effect] += 1
        self.universe_obs[self.current_universe].append({
            'state': before, 'action': action, 'effect': effect
        })
        
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
    
    def _interference(self, t1, t2):
        w1, w2 = self.waves[t1], self.waves[t2]
        e1, e2 = self.energy[t1], self.energy[t2]
        if e1 == 0 or e2 == 0: return 0.0
        all_e = set(w1) | set(w2)
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
        # Combine all universes for candidate generation
        all_obs = []
        for u_obs in self.universe_obs.values():
            all_obs.extend(u_obs)
        
        by_ae = defaultdict(list)
        for obs in all_obs:
            if obs['effect']:
                key = (obs['action'], self._abstract(obs['effect']))
                by_ae[key].append(self._abstract(obs['state']))
        
        candidates = []
        for (action, effect), states in by_ae.items():
            if len(states) < 2: continue
            sample = random.sample(states, min(10, len(states)))
            lhs = frozenset.intersection(*sample)
            if lhs:
                candidates.append({
                    'lhs': lhs, 'effect': effect, 'action': action,
                    'support': len(states), 'temperature': 0.0,
                    'universe_scores': {}, 'universality': 0.0
                })
            if len(states) >= 4:
                for _ in range(2):
                    subset = random.sample(states, len(states)//2)
                    lhs_p = frozenset.intersection(*subset)
                    if lhs_p and lhs_p != lhs:
                        candidates.append({
                            'lhs': lhs_p, 'effect': effect, 'action': action,
                            'support': len(subset), 'temperature': 0.2,
                            'universe_scores': {}, 'universality': 0.0
                        })
        return candidates
    
    def _score_in_universe(self, rule, universe_id):
        """Score rule in a specific universe."""
        tp, fp, fn = 0, 0, 0
        for obs in self.universe_obs[universe_id]:
            state = self._abstract(obs['state'])
            actual = self._abstract(obs['effect'])
            if rule['action'] == obs['action'] and rule['lhs'] <= state:
                for e in rule['effect']:
                    if e in actual: tp += 1
                    else: fp += 1
                for e in actual:
                    if e not in rule['effect']: fn += 1
        if tp+fp+fn == 0: return 0.5
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        return 2*prec*rec/(prec+rec) if prec+rec else 0
    
    def build(self):
        self._build_classes()
        candidates = self._generate_candidates()
        
        # Score each rule across all universes
        for rule in candidates:
            scores = []
            for u_id in self.universe_obs.keys():
                s = self._score_in_universe(rule, u_id)
                rule['universe_scores'][u_id] = s
                scores.append(s)
            
            if scores:
                # Universality = fraction of universes where rule works well
                rule['universality'] = sum(s > 0.5 for s in scores) / len(scores)
                rule['temperature'] = 1 - (sum(scores) / len(scores))
        
        self.rules = [r for r in candidates if r['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (-r['universality'], -len(r['lhs']), -r['support']))
    
    def predict(self, state: Set, action: int) -> Set:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        abstract_state = self._abstract(frozenset(state))
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        return set()

print("UniversalSieve loaded!")
