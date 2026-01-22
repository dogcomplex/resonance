"""
TOKEN CHEMISTRY V3 - Fixed meta-binding usage

The key insight: Meta-bindings mean "these rules are the same".
If ANY rule in a meta-group matches, use its prediction.

Prediction flow:
1. Look for exact match in ANY binding
2. Group equivalent bindings together
3. Use first matching binding from any group
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set

class TokenChemistryV3:
    def __init__(self, coherence=0.85, cold_threshold=0.5, min_energy=5):
        self.coherence = coherence
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        
        self.classes = {}
        self.bindings = []
        self.binding_groups = []  # Groups of equivalent bindings for prediction
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state) | {f"ACTION_{action}"}
        after = frozenset(next_state)
        delta = frozenset(
            {f"+{t}" for t in (after - frozenset(state))} |
            {f"-{t}" for t in (frozenset(state) - after)}
        )
        
        self.exact[(frozenset(state), action)][delta] += 1
        self.observations.append({'complex': before, 'delta': delta, 'action': action})
        
        for token in before:
            self.energy[token] += 1
            for d in delta:
                self.waves[token][d] += 1.0
    
    def _interference(self, waves, energy, t1, t2):
        w1, w2 = waves[t1], waves[t2]
        e1, e2 = energy[t1], energy[t2]
        if e1 == 0 or e2 == 0: return 0.0
        all_d = set(w1.keys()) | set(w2.keys())
        dot = sum((w1.get(d,0)/e1) * (w2.get(d,0)/e2) for d in all_d)
        n1 = sum((w1.get(d,0)/e1)**2 for d in all_d) ** 0.5
        n2 = sum((w2.get(d,0)/e2)**2 for d in all_d) ** 0.5
        return dot / (n1 * n2) if n1 and n2 else 0.0
    
    def _structural_match(self, t1, t2):
        if t1.startswith("ACTION_") or t2.startswith("ACTION_"):
            return t1 == t2
        if '_' not in t1 or '_' not in t2:
            return True
        p1, p2 = t1.split('_'), t2.split('_')
        return len(p1) == len(p2) and sum(1 for a, b in zip(p1, p2) if a != b) <= 1
    
    def _abstract(self, s: FrozenSet) -> FrozenSet:
        return frozenset(self.classes.get(t, t) for t in s)
    
    def build(self):
        print("Building token chemistry v3...")
        
        # 1. Token resonance
        tokens = [t for t in self.waves if self.energy[t] >= self.min_energy]
        self.classes = {t: t for t in self.waves}
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2: continue
                if not self._structural_match(t1, t2): continue
                if self._interference(self.waves, self.energy, t1, t2) > self.coherence:
                    self.classes[t2] = self.classes[t1]
        print(f"  Token classes: {len(set(self.classes.values()))}")
        
        # 2. Discover bindings
        by_delta = defaultdict(list)
        for obs in self.observations:
            abstract_delta = self._abstract(obs['delta'])
            if abstract_delta:
                by_delta[(obs['action'], abstract_delta)].append(self._abstract(obs['complex']))
        
        bind_waves = defaultdict(lambda: defaultdict(float))
        bind_energy = defaultdict(float)
        candidates = []
        
        for (action, delta), complexes in by_delta.items():
            if len(complexes) < 2: continue
            sample = random.sample(complexes, min(10, len(complexes)))
            lhs = frozenset.intersection(*sample)
            if lhs:
                b_id = len(candidates)
                candidates.append({
                    'id': b_id, 'inputs': lhs, 'outputs': delta, 'action': action,
                    'support': len(complexes), 'temperature': 0.0
                })
                for cplx in complexes:
                    if lhs <= cplx:
                        bind_energy[b_id] += 1
                        for d in delta:
                            bind_waves[b_id][d] += 1.0
        
        # Anneal
        random.shuffle(self.observations)
        fold_size = len(self.observations) // 5
        for fold_idx in range(5):
            start = (fold_idx * fold_size) % len(self.observations)
            fold = self.observations[start:start + fold_size]
            for b in candidates:
                tp, fp, fn = 0, 0, 0
                for obs in fold:
                    if obs['action'] != b['action']: continue
                    cplx = self._abstract(obs['complex'])
                    actual = self._abstract(obs['delta'])
                    if b['inputs'] <= cplx:
                        for d in b['outputs']:
                            if d in actual: tp += 1
                            else: fp += 1
                        for d in actual:
                            if d not in b['outputs']: fn += 1
                if tp + fp + fn > 0:
                    prec = tp/(tp+fp) if tp+fp else 0
                    rec = tp/(tp+fn) if tp+fn else 0
                    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
                else:
                    f1 = 0.5
                b['temperature'] = b['temperature'] * 0.5 + (1-f1) * 0.5
        
        self.bindings = [c for c in candidates if c['temperature'] < self.cold_threshold]
        print(f"  Bindings: {len(self.bindings)}")
        
        # 3. Build binding groups via resonance
        bind_ids = [b['id'] for b in self.bindings if bind_energy[b['id']] >= 3]
        bind_class = {b['id']: b['id'] for b in self.bindings}
        
        for i, b1 in enumerate(bind_ids):
            for b2 in bind_ids[i+1:]:
                if bind_class[b2] != b2: continue
                binding1 = next(b for b in self.bindings if b['id'] == b1)
                binding2 = next(b for b in self.bindings if b['id'] == b2)
                if binding1['action'] != binding2['action']: continue
                if self._interference(bind_waves, bind_energy, b1, b2) > self.coherence:
                    bind_class[b2] = bind_class[b1]
        
        # Create binding groups for prediction
        # Each group: {action, outputs, all_inputs: [list of alternative LHS]}
        class_to_bindings = defaultdict(list)
        for b in self.bindings:
            cls = bind_class.get(b['id'], b['id'])
            class_to_bindings[cls].append(b)
        
        self.binding_groups = []
        for cls, binds in class_to_bindings.items():
            # All bindings in a group should have same action and outputs
            action = binds[0]['action']
            outputs = binds[0]['outputs']
            all_inputs = [b['inputs'] for b in binds]
            # Sort by LHS size (smaller = more general = try first)
            all_inputs.sort(key=len)
            
            self.binding_groups.append({
                'action': action,
                'outputs': outputs,
                'all_inputs': all_inputs,
                'n_alternatives': len(all_inputs)
            })
        
        # Sort groups by specificity (smallest min LHS first = most general)
        self.binding_groups.sort(key=lambda g: (min(len(i) for i in g['all_inputs']), -g['n_alternatives']))
        
        n_multi = sum(1 for g in self.binding_groups if g['n_alternatives'] > 1)
        print(f"  Binding groups: {len(self.binding_groups)} ({n_multi} with alternatives)")
    
    def predict(self, state: Set, action: int) -> Set:
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        complex_tokens = self._abstract(frozenset(state) | {f"ACTION_{action}"})
        
        # Check binding groups - try each alternative LHS
        for group in self.binding_groups:
            if group['action'] != action:
                continue
            
            # Try each alternative LHS (sorted by generality)
            for lhs in group['all_inputs']:
                if lhs <= complex_tokens:
                    return set(group['outputs'])
        
        return set()

print("TokenChemistryV3 loaded!")
