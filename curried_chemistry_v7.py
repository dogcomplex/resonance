"""
CURRIED CHEMISTRY V7 - Match baseline rule generation

Add partial intersections back to match baseline candidate count.
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set

class CurriedChemistryV7:
    def __init__(self, coherence=0.85, min_energy=3, cold_threshold=0.5):
        self.coherence = coherence
        self.min_energy = min_energy
        self.cold_threshold = cold_threshold
        
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        
        self.token_waves = defaultdict(lambda: defaultdict(float))
        self.token_energy = defaultdict(float)
        self.token_classes = {}
        
        self.rules = []
        self.shared_intermediates = []
    
    def observe(self, state: Set[str], action: int, next_state: Set[str]):
        before = frozenset(state)
        effect = frozenset(
            {f"+{t}" for t in (frozenset(next_state) - before)} |
            {f"-{t}" for t in (before - frozenset(next_state))}
        )
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({'before': before, 'effect': effect, 'action': action})
        
        for token in before:
            self.token_energy[token] += 1
            for e in effect:
                self.token_waves[token][e] += 1.0
    
    def _build_token_classes(self):
        def interference(t1, t2):
            w1, w2 = self.token_waves[t1], self.token_waves[t2]
            e1, e2 = self.token_energy[t1], self.token_energy[t2]
            if e1 == 0 or e2 == 0: return 0.0
            all_e = set(w1.keys()) | set(w2.keys())
            dot = sum((w1.get(e,0)/e1) * (w2.get(e,0)/e2) for e in all_e)
            n1 = sum((w1.get(e,0)/e1)**2 for e in all_e) ** 0.5
            n2 = sum((w2.get(e,0)/e2)**2 for e in all_e) ** 0.5
            return dot / (n1 * n2) if n1 and n2 else 0.0
        
        def structural_match(t1, t2):
            if '_' not in t1 or '_' not in t2: return True
            p1, p2 = t1.split('_'), t2.split('_')
            return len(p1) == len(p2) and sum(1 for a, b in zip(p1, p2) if a != b) <= 1
        
        tokens = [t for t in self.token_waves if self.token_energy[t] >= self.min_energy]
        self.token_classes = {t: t for t in self.token_waves}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.token_classes[t2] != t2: continue
                if not structural_match(t1, t2): continue
                if interference(t1, t2) > self.coherence:
                    self.token_classes[t2] = self.token_classes[t1]
    
    def _abstract(self, s):
        return frozenset(self.token_classes.get(t, t) for t in s)
    
    def _discover_rules(self):
        """Discover rules - matching baseline approach."""
        by_effect = defaultdict(list)
        for obs in self.observations:
            abstract_before = self._abstract(obs['before'])
            abstract_effect = self._abstract(obs['effect'])
            if abstract_effect:
                by_effect[(obs['action'], abstract_effect)].append(abstract_before)
        
        candidates = []
        token_set_to_effects = defaultdict(set)
        
        for (action, effect), contexts in by_effect.items():
            if len(contexts) < 2:
                continue
            
            # Full intersection
            sample = random.sample(contexts, min(10, len(contexts)))
            common = frozenset.intersection(*sample)
            
            if common:
                candidates.append({
                    'lhs': common,
                    'effect': effect,
                    'action': action,
                    'support': len(contexts),
                    'temperature': 0.0
                })
                token_set_to_effects[common].add(effect)
            
            # Partial intersections (like baseline)
            if len(contexts) >= 4:
                for _ in range(3):
                    subset = random.sample(contexts, len(contexts) // 2)
                    partial = frozenset.intersection(*subset)
                    if partial and partial != common:
                        candidates.append({
                            'lhs': partial,
                            'effect': effect,
                            'action': action,
                            'support': len(subset),
                            'temperature': 0.3
                        })
                        token_set_to_effects[partial].add(effect)
        
        # Find shared intermediates
        shared = []
        for token_set, effects in token_set_to_effects.items():
            if len(effects) > 1:
                shared.append({
                    'tokens': token_set,
                    'effects': effects,
                    'n_effects': len(effects)
                })
        
        return candidates, shared
    
    def _anneal(self, candidates):
        """Anneal using fold-based validation (like baseline)."""
        random.shuffle(self.observations)
        n = len(self.observations)
        fold_size = n // 5
        
        for fold_idx in range(5):
            start = (fold_idx * fold_size) % n
            fold = self.observations[start:start + fold_size]
            
            for rule in candidates:
                tp, fp, fn = 0, 0, 0
                
                for obs in fold:
                    if rule['action'] != obs['action']:
                        continue
                    
                    state = self._abstract(obs['before'])
                    actual = self._abstract(obs['effect'])
                    
                    if rule['lhs'] <= state:
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
        print("Building curried chemistry v7...")
        
        self._build_token_classes()
        print(f"  Token classes: {len(set(self.token_classes.values()))}")
        
        candidates, shared = self._discover_rules()
        self.shared_intermediates = shared
        print(f"  Candidates: {len(candidates)}")
        print(f"  Shared intermediates: {len(shared)}")
        
        self._anneal(candidates)
        
        self.rules = [c for c in candidates if c['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
        print(f"  After annealing: {len(self.rules)}")
    
    def predict(self, state: Set[str], action: int) -> Set[str]:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        abstract_state = self._abstract(frozenset(state))
        
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        
        return set()


print("CurriedChemistryV7 loaded!")
