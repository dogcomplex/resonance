"""
UNIFIED RESONANCE V4 - Merge equivalent rules during BUILD

The insight: If rules R1 and R2 have the same wave signature (resonate),
combine their observations to build a STRONGER rule.

This is the same operation as token merging, applied to rules:
- Tokens with same wave → merge into class
- Rules with same wave → merge observations → build unified rule
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List

class UnifiedResonanceV4:
    def __init__(self,
                 coherence=0.85,
                 cold_threshold=0.5,
                 min_energy=5,
                 n_folds=5):
        
        self.coherence = coherence
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        self.n_folds = n_folds
        
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        
        self.token_classes = {}
        self.rules = []
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after - before)} |
                          {f"-{t}" for t in (before - after)})
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'before': before, 'action': action, 'effect': effect
        })
        
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
    
    def _interference(self, waves, energy, t1, t2):
        w1, w2 = waves[t1], waves[t2]
        e1, e2 = energy[t1], energy[t2]
        if e1 == 0 or e2 == 0: return 0.0
        all_e = set(w1.keys()) | set(w2.keys())
        dot = sum((w1.get(e,0)/e1) * (w2.get(e,0)/e2) for e in all_e)
        n1 = sum((w1.get(e,0)/e1)**2 for e in all_e) ** 0.5
        n2 = sum((w2.get(e,0)/e2)**2 for e in all_e) ** 0.5
        return dot / (n1 * n2) if n1 and n2 else 0.0
    
    def _structural_match(self, t1, t2):
        if '_' not in t1 or '_' not in t2: return True
        p1, p2 = t1.split('_'), t2.split('_')
        if len(p1) != len(p2): return False
        return sum(1 for a, b in zip(p1, p2) if a != b) <= 1
    
    def _build_token_classes(self):
        tokens = [t for t in self.waves if self.energy[t] >= self.min_energy]
        self.token_classes = {t: t for t in self.waves}
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.token_classes[t2] != t2: continue
                if not self._structural_match(t1, t2): continue
                if self._interference(self.waves, self.energy, t1, t2) > self.coherence:
                    self.token_classes[t2] = self.token_classes[t1]
    
    def _abstract(self, s: FrozenSet) -> FrozenSet:
        return frozenset(self.token_classes.get(t, t) for t in s)
    
    def _generate_rules_with_meta(self):
        """
        Generate rules with meta-resonance folded in.
        
        Process:
        1. Group observations by (action, effect)
        2. For each group, generate candidate rules
        3. Build wave signatures for rules
        4. MERGE observations from equivalent rules
        5. Re-generate rules from merged observations
        """
        # Group by (action, abstracted_effect)
        by_ae = defaultdict(list)
        for obs in self.observations:
            if not obs['effect']: continue
            key = (obs['action'], self._abstract(obs['effect']))
            by_ae[key].append(self._abstract(obs['before']))
        
        # First pass: Generate initial rules and their wave signatures
        initial_rules = []
        rule_waves = defaultdict(lambda: defaultdict(float))
        rule_energy = defaultdict(float)
        
        for (action, effect), states in by_ae.items():
            if len(states) < 2: continue
            
            sample = random.sample(states, min(10, len(states)))
            lhs = frozenset.intersection(*sample)
            
            if lhs:
                r_id = len(initial_rules)
                initial_rules.append({
                    'lhs': lhs, 'effect': effect, 'action': action,
                    'states': states, 'id': r_id
                })
                
                # Build wave signature from matching observations
                for state in states:
                    if lhs <= state:
                        rule_energy[r_id] += 1
                        for e in effect:
                            rule_waves[r_id][e] += 1.0
        
        # Find equivalent rules (same action, similar wave)
        rule_classes = {r['id']: r['id'] for r in initial_rules}
        rule_ids = [r['id'] for r in initial_rules if rule_energy[r['id']] >= 3]
        
        for i, r1 in enumerate(rule_ids):
            for r2 in rule_ids[i+1:]:
                if rule_classes[r2] != r2: continue
                rule1 = initial_rules[r1]
                rule2 = initial_rules[r2]
                if rule1['action'] != rule2['action']: continue
                if self._interference(rule_waves, rule_energy, r1, r2) > self.coherence:
                    rule_classes[r2] = rule_classes[r1]
        
        # Merge observations from equivalent rules
        class_to_rules = defaultdict(list)
        for r in initial_rules:
            cls = rule_classes.get(r['id'], r['id'])
            class_to_rules[cls].append(r)
        
        # Generate final rules from merged observations
        candidates = []
        for cls, group in class_to_rules.items():
            if not group: continue
            
            action = group[0]['action']
            effect = group[0]['effect']
            
            # Combine all states from equivalent rules
            all_states = []
            for r in group:
                all_states.extend(r['states'])
            
            # Generate rule from combined states
            if len(all_states) >= 2:
                sample = random.sample(all_states, min(15, len(all_states)))
                lhs = frozenset.intersection(*sample)
                
                if lhs:
                    candidates.append({
                        'lhs': lhs, 'effect': effect, 'action': action,
                        'support': len(all_states), 'temperature': 0.0,
                        'merged_from': len(group)
                    })
                
                # Partial intersections
                if len(all_states) >= 4:
                    for _ in range(3):
                        subset = random.sample(all_states, len(all_states)//2)
                        lhs_p = frozenset.intersection(*subset)
                        if lhs_p and lhs_p != lhs:
                            candidates.append({
                                'lhs': lhs_p, 'effect': effect, 'action': action,
                                'support': len(subset), 'temperature': 0.3,
                                'merged_from': len(group)
                            })
        
        return candidates, len(class_to_rules), sum(1 for c, g in class_to_rules.items() if len(g) > 1)
    
    def _anneal(self, candidates):
        random.shuffle(self.observations)
        n = len(self.observations)
        fold_size = n // self.n_folds
        
        for fold_idx in range(self.n_folds):
            start = (fold_idx * fold_size) % n
            fold = self.observations[start:start + fold_size]
            
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
                    prec = tp/(tp+fp) if tp+fp else 0
                    rec = tp/(tp+fn) if tp+fn else 0
                    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
                else:
                    f1 = 0.5
                rule['temperature'] = rule['temperature'] * 0.5 + (1-f1) * 0.5
    
    def build(self):
        print("Building unified resonance v4 (with meta-merge)...")
        
        self._build_token_classes()
        n_token_classes = len(set(self.token_classes.values()))
        print(f"  Token classes: {n_token_classes}")
        
        candidates, n_groups, n_merged = self._generate_rules_with_meta()
        print(f"  Initial rule groups: {n_groups} ({n_merged} merged)")
        
        self._anneal(candidates)
        self.rules = [c for c in candidates if c['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
        
        merged_rules = sum(1 for r in self.rules if r.get('merged_from', 1) > 1)
        print(f"  Final rules: {len(self.rules)} ({merged_rules} from merged groups)")
    
    def predict(self, state: Set, action: int) -> Set:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        abstract_state = self._abstract(frozenset(state))
        for rule in self.rules:
            if rule['action'] == action and rule['lhs'] <= abstract_state:
                return set(rule['effect'])
        return set()

print("UnifiedResonanceV4 loaded!")
