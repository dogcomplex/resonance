"""
RESONANCE SIEVE - Flexible cascaded sieve framework

Core idea: Rules are "resonance patterns" that either amplify or decay
based on how they interact with different observation buckets.

Sieves can be applied in any order, multiple times, with random bucketing.
The "temperature" is really "inverse resonance strength".

Each sieve:
- Takes candidates + a bucket of observations
- Scores each candidate's resonance in that bucket
- Updates temperature (decays if resonant, increases if not)

Reservoir computing parallel:
- Observations are input signals
- Rules are reservoir states
- Sieves are the nonlinear transformation
- Surviving rules are the readout
"""

import random
import math
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List, Callable

class ResonanceSieve:
    def __init__(self, 
                 coherence=0.95,
                 min_energy=5,
                 decay_rate=0.3,      # How much temperature drops on resonance
                 excite_rate=0.2,     # How much temperature rises on dissonance
                 cold_threshold=0.4):
        
        self.coherence = coherence
        self.min_energy = min_energy
        self.decay_rate = decay_rate
        self.excite_rate = excite_rate
        self.cold_threshold = cold_threshold
        
        # Wave data for fidelity sieve
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        
        # Raw observations
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        
        # Token classes (from fidelity sieve)
        self.classes = {}
        
        # Candidate rules with temperatures
        self.candidates = []
        self.rules = []  # Final cold rules
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | 
                          {f"-{t}" for t in (before-after)})
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'state': before, 'action': action, 'effect': effect,
            'idx': len(self.observations)
        })
        
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
    
    # ============ SIEVE FUNCTIONS ============
    
    def sieve_probability(self, candidates: List[Dict], bucket: List[Dict]) -> List[Dict]:
        """
        Probability sieve: Score by F1 on bucket.
        Rules that predict correctly resonate (cool down).
        """
        for rule in candidates:
            tp, fp, fn = 0, 0, 0
            for obs in bucket:
                state = self._abstract(obs['state'])
                actual = self._abstract(obs['effect'])
                if rule['action'] == obs['action'] and rule['lhs'] <= state:
                    for e in rule['effect']:
                        if e in actual: tp += 1
                        else: fp += 1
                    for e in actual:
                        if e not in rule['effect']: fn += 1
            
            if tp + fp + fn > 0:
                f1 = 2*tp / (2*tp + fp + fn)
                if f1 > 0.6:  # Resonance
                    rule['temperature'] *= (1 - self.decay_rate)
                else:  # Dissonance
                    rule['temperature'] += self.excite_rate * (1 - f1)
            
            rule['temperature'] = min(1.0, rule['temperature'])
        
        return candidates
    
    def sieve_fidelity(self, candidates: List[Dict], bucket: List[Dict]) -> List[Dict]:
        """
        Fidelity sieve: Try abstracting rules.
        If abstract version resonates better, promote it.
        """
        new_candidates = []
        seen_lhs = set()
        
        for rule in candidates:
            # Keep original
            new_candidates.append(rule)
            seen_lhs.add((rule['lhs'], rule['action'], rule['effect']))
            
            # Try abstracting LHS further
            abstract_lhs = self._abstract(rule['lhs'])
            key = (abstract_lhs, rule['action'], rule['effect'])
            
            if abstract_lhs != rule['lhs'] and key not in seen_lhs:
                new_rule = {
                    'lhs': abstract_lhs,
                    'effect': rule['effect'],
                    'action': rule['action'],
                    'temperature': rule['temperature'] + 0.1,  # Start slightly warmer
                    'support': rule['support'],
                    'parent': id(rule)
                }
                new_candidates.append(new_rule)
                seen_lhs.add(key)
        
        return new_candidates
    
    def sieve_temporal(self, candidates: List[Dict], bucket: List[Dict], 
                       time_offsets: List[int] = [0, 1, 2, 4]) -> List[Dict]:
        """
        Temporal sieve: Score by consistency across time offsets.
        Rules that work at multiple time distances are more universal.
        """
        # Group bucket by temporal position
        by_offset = defaultdict(list)
        for i, obs in enumerate(bucket):
            for offset in time_offsets:
                if i >= offset:
                    by_offset[offset].append(obs)
        
        for rule in candidates:
            scores = []
            for offset, obs_list in by_offset.items():
                if not obs_list:
                    continue
                tp, fp, fn = 0, 0, 0
                for obs in obs_list[:50]:  # Sample
                    state = self._abstract(obs['state'])
                    actual = self._abstract(obs['effect'])
                    if rule['action'] == obs['action'] and rule['lhs'] <= state:
                        for e in rule['effect']:
                            if e in actual: tp += 1
                            else: fp += 1
                        for e in actual:
                            if e not in rule['effect']: fn += 1
                if tp + fp + fn > 0:
                    scores.append(2*tp / (2*tp + fp + fn))
            
            if scores:
                consistency = min(scores) / (max(scores) + 0.01)  # How consistent?
                avg_score = sum(scores) / len(scores)
                
                if consistency > 0.8 and avg_score > 0.5:  # Consistent resonance
                    rule['temperature'] *= (1 - self.decay_rate)
                elif consistency < 0.5:  # Inconsistent
                    rule['temperature'] += self.excite_rate
        
        return candidates
    
    def sieve_universal(self, candidates: List[Dict], 
                        buckets: List[List[Dict]]) -> List[Dict]:
        """
        Universal sieve: Score by consistency across different buckets.
        (Could be different seeds, different observation subsets, etc.)
        """
        for rule in candidates:
            bucket_scores = []
            for bucket in buckets:
                tp, fp, fn = 0, 0, 0
                for obs in bucket[:30]:  # Sample
                    state = self._abstract(obs['state'])
                    actual = self._abstract(obs['effect'])
                    if rule['action'] == obs['action'] and rule['lhs'] <= state:
                        for e in rule['effect']:
                            if e in actual: tp += 1
                            else: fp += 1
                        for e in actual:
                            if e not in rule['effect']: fn += 1
                if tp + fp + fn > 0:
                    bucket_scores.append(2*tp / (2*tp + fp + fn))
            
            if bucket_scores:
                # Universal = works in most buckets
                good_buckets = sum(1 for s in bucket_scores if s > 0.5)
                universality = good_buckets / len(bucket_scores)
                
                if universality > 0.7:
                    rule['temperature'] *= (1 - self.decay_rate)
                elif universality < 0.3:
                    rule['temperature'] += self.excite_rate
        
        return candidates
    
    # ============ HELPER METHODS ============
    
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
        """Generate initial candidates from observations."""
        by_ae = defaultdict(list)
        for obs in self.observations:
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
                    'support': len(states), 'temperature': 0.5  # Start warm
                })
        return candidates
    
    def _random_bucket(self, size=None):
        """Get a random bucket of observations."""
        if size is None:
            size = len(self.observations) // 5
        return random.sample(self.observations, min(size, len(self.observations)))
    
    # ============ MAIN BUILD ============
    
    def build(self, sieve_schedule=None):
        """
        Build rules using sieve schedule.
        Default: [probability, fidelity, probability, temporal, probability]
        """
        self._build_classes()
        self.candidates = self._generate_candidates()
        
        if sieve_schedule is None:
            # Default schedule: interleaved sieves with random buckets
            sieve_schedule = [
                ('probability', {}),
                ('fidelity', {}),
                ('probability', {}),
                ('temporal', {'time_offsets': [0, 1, 2]}),
                ('probability', {}),
                ('probability', {}),  # Extra passes
                ('probability', {}),
            ]
        
        for sieve_name, kwargs in sieve_schedule:
            bucket = self._random_bucket()
            
            if sieve_name == 'probability':
                self.candidates = self.sieve_probability(self.candidates, bucket)
            elif sieve_name == 'fidelity':
                self.candidates = self.sieve_fidelity(self.candidates, bucket)
            elif sieve_name == 'temporal':
                self.candidates = self.sieve_temporal(self.candidates, bucket, **kwargs)
            elif sieve_name == 'universal':
                buckets = [self._random_bucket() for _ in range(3)]
                self.candidates = self.sieve_universal(self.candidates, buckets)
            
            # Prune very hot candidates
            self.candidates = [c for c in self.candidates if c['temperature'] < 0.9]
        
        # Final: keep cold rules
        self.rules = [c for c in self.candidates if c['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (r['temperature'], -len(r['lhs'])))
    
    def predict(self, state: Set, action: int) -> Set:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        abstract_state = self._abstract(frozenset(state))
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        return set()

print("ResonanceSieve loaded!")
