"""
DEEP RESONANCE - Recursive layers until convergence

Each layer:
1. Takes tokens (could be original tokens, rules, or meta-tokens from previous layer)
2. Finds equivalences via wave interference
3. Generates rules
4. Passes BOTH equivalent classes AND rules to next layer as tokens

Continues until no new interesting structure emerges.
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List, Tuple, Any

class DeepResonance:
    def __init__(self,
                 coherence=0.85,
                 cold_threshold=0.5,
                 min_energy=3,
                 max_layers=5,
                 min_new_structure=5):  # Stop if fewer than this new classes/rules
        
        self.coherence = coherence
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        self.max_layers = max_layers
        self.min_new_structure = min_new_structure
        
        # Layer 0: Raw observations
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        
        # Each layer's results
        self.layers = []  # List of {tokens, classes, rules, waves, energy}
        
    def observe(self, state: Set, action: int, next_state: Set):
        """Accumulate raw observations."""
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after - before)} |
                          {f"-{t}" for t in (before - after)})
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'before': before, 'action': action, 'effect': effect,
            'after': after
        })
    
    def _interference(self, waves, energy, t1, t2):
        """Compute wave interference between two tokens."""
        w1, w2 = waves[t1], waves[t2]
        e1, e2 = energy[t1], energy[t2]
        if e1 == 0 or e2 == 0: return 0.0
        
        all_e = set(w1.keys()) | set(w2.keys())
        dot = sum((w1.get(e,0)/e1) * (w2.get(e,0)/e2) for e in all_e)
        n1 = sum((w1.get(e,0)/e1)**2 for e in all_e) ** 0.5
        n2 = sum((w2.get(e,0)/e2)**2 for e in all_e) ** 0.5
        
        return dot / (n1 * n2) if n1 and n2 else 0.0
    
    def _build_classes(self, waves, energy):
        """Build equivalence classes via interference."""
        tokens = [t for t in waves if energy[t] >= self.min_energy]
        classes = {t: t for t in waves}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if classes[t2] != t2: continue
                if self._interference(waves, energy, t1, t2) > self.coherence:
                    classes[t2] = classes[t1]
        
        return classes
    
    def _abstract(self, s: FrozenSet, classes: Dict) -> FrozenSet:
        """Map tokens through equivalence classes."""
        return frozenset(classes.get(t, t) for t in s)
    
    def _generate_rules(self, observations, classes, n_folds=3):
        """Generate and anneal rules."""
        # Group by (action, abstracted_effect)
        by_ae = defaultdict(list)
        for obs in observations:
            if not obs['effect']: continue
            key = (obs['action'], self._abstract(obs['effect'], classes))
            by_ae[key].append(self._abstract(obs['before'], classes))
        
        candidates = []
        for (action, effect), states in by_ae.items():
            if len(states) < 2: continue
            
            sample = random.sample(states, min(10, len(states)))
            lhs = frozenset.intersection(*sample)
            
            if lhs:
                candidates.append({
                    'lhs': lhs, 'effect': effect, 'action': action,
                    'support': len(states), 'temperature': 0.0
                })
            
            if len(states) >= 4:
                for _ in range(2):
                    subset = random.sample(states, len(states)//2)
                    lhs_p = frozenset.intersection(*subset)
                    if lhs_p and lhs_p != lhs:
                        candidates.append({
                            'lhs': lhs_p, 'effect': effect, 'action': action,
                            'support': len(subset), 'temperature': 0.3
                        })
        
        # Anneal
        random.shuffle(observations)
        n = len(observations)
        fold_size = max(1, n // 5)
        
        for fold_idx in range(n_folds):
            start = (fold_idx * fold_size) % n
            fold = observations[start:start + fold_size]
            
            for rule in candidates:
                tp, fp, fn = 0, 0, 0
                for obs in fold:
                    state = self._abstract(obs['before'], classes)
                    actual = self._abstract(obs['effect'], classes)
                    
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
        
        rules = [c for c in candidates if c['temperature'] < self.cold_threshold]
        rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
        return rules
    
    def _rule_to_token(self, layer_idx: int, rule_idx: int) -> str:
        """Convert a rule to a token for the next layer."""
        return f"L{layer_idx}_R{rule_idx}"
    
    def _class_to_token(self, layer_idx: int, class_rep: str) -> str:
        """Convert a class representative to a token for the next layer."""
        return f"L{layer_idx}_C_{class_rep}"
    
    def _build_layer(self, layer_idx: int, prev_observations: List, 
                     prev_classes: Dict = None) -> Dict:
        """Build a single layer."""
        
        # Build wave signatures
        waves = defaultdict(lambda: defaultdict(float))
        energy = defaultdict(float)
        
        for obs in prev_observations:
            before = obs['before'] if prev_classes is None else self._abstract(obs['before'], prev_classes)
            effect = obs['effect'] if prev_classes is None else self._abstract(obs['effect'], prev_classes)
            
            for token in before:
                energy[token] += 1
                for e in effect:
                    waves[token][e] += 1.0
        
        # Build classes
        classes = self._build_classes(waves, energy)
        
        # Generate rules
        rules = self._generate_rules(prev_observations, classes)
        
        return {
            'waves': waves,
            'energy': energy,
            'classes': classes,
            'rules': rules,
            'n_unique_classes': len(set(classes.values())),
            'n_rules': len(rules)
        }
    
    def _create_next_layer_observations(self, layer_idx: int) -> List[Dict]:
        """
        Create observations for the next layer.
        
        Tokens for next layer:
        1. Class representatives from current layer
        2. Rules from current layer (as tokens)
        
        An "observation" at Layer N+1 is:
        - before: which L_N classes and rules are "active" given the original state
        - after: which L_N classes and rules are "active" given the original next_state
        """
        current = self.layers[layer_idx]
        classes = current['classes']
        rules = current['rules']
        
        new_observations = []
        
        for obs in self.observations:
            # Map original state through all layers up to current
            state = obs['before']
            next_state = obs['after']
            
            for i in range(layer_idx + 1):
                state = self._abstract(state, self.layers[i]['classes'])
                next_state = self._abstract(next_state, self.layers[i]['classes'])
            
            # Build token set for this state
            before_tokens = set()
            after_tokens = set()
            
            # Add class representatives as tokens
            for t in state:
                before_tokens.add(self._class_to_token(layer_idx, classes.get(t, t)))
            for t in next_state:
                after_tokens.add(self._class_to_token(layer_idx, classes.get(t, t)))
            
            # Add rules that match as tokens
            for r_idx, rule in enumerate(rules):
                if rule['lhs'] <= state:
                    before_tokens.add(self._rule_to_token(layer_idx, r_idx))
                if rule['lhs'] <= next_state:
                    after_tokens.add(self._rule_to_token(layer_idx, r_idx))
            
            # Create effect (what changed in token space)
            effect = frozenset(
                {f"+{t}" for t in (after_tokens - before_tokens)} |
                {f"-{t}" for t in (before_tokens - after_tokens)}
            )
            
            new_observations.append({
                'before': frozenset(before_tokens),
                'after': frozenset(after_tokens),
                'action': obs['action'],
                'effect': effect
            })
        
        return new_observations
    
    def build(self):
        """Build all layers until convergence."""
        print(f"Building deep resonance (max {self.max_layers} layers)...")
        
        # Layer 0: Direct from observations
        layer0 = self._build_layer(0, self.observations)
        self.layers.append(layer0)
        print(f"  Layer 0: {layer0['n_unique_classes']} classes, {layer0['n_rules']} rules")
        
        # Build subsequent layers
        for layer_idx in range(1, self.max_layers):
            # Create observations for this layer
            layer_obs = self._create_next_layer_observations(layer_idx - 1)
            
            # Build the layer
            layer = self._build_layer(layer_idx, layer_obs)
            self.layers.append(layer)
            
            print(f"  Layer {layer_idx}: {layer['n_unique_classes']} classes, {layer['n_rules']} rules")
            
            # Check for convergence
            if layer['n_rules'] < self.min_new_structure and layer['n_unique_classes'] < self.min_new_structure:
                print(f"  Converged at layer {layer_idx} (insufficient new structure)")
                break
            
            # Check if we're just getting the same thing
            if layer_idx > 0:
                prev = self.layers[layer_idx - 1]
                if (layer['n_rules'] >= prev['n_rules'] * 0.95 and 
                    layer['n_unique_classes'] >= prev['n_unique_classes'] * 0.95):
                    print(f"  Converged at layer {layer_idx} (no compression)")
                    break
        
        print(f"Built {len(self.layers)} layers total")
    
    def predict(self, state: Set, action: int) -> Set:
        """Predict using Layer 0 (could extend to use higher layers)."""
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        if not self.layers:
            return set()
        
        abstract_state = self._abstract(frozenset(state), self.layers[0]['classes'])
        for rule in self.layers[0]['rules']:
            if rule['action'] == action and rule['lhs'] <= abstract_state:
                return set(rule['effect'])
        
        return set()
    
    def get_layer_summary(self):
        """Get summary of what each layer discovered."""
        summary = []
        for i, layer in enumerate(self.layers):
            # Count merged classes
            class_counts = defaultdict(int)
            for t, c in layer['classes'].items():
                class_counts[c] += 1
            merged = sum(1 for c, n in class_counts.items() if n > 1)
            
            summary.append({
                'layer': i,
                'unique_classes': layer['n_unique_classes'],
                'merged_classes': merged,
                'rules': layer['n_rules']
            })
        return summary

print("DeepResonance loaded!")
