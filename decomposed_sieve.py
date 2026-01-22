"""
DECOMPOSED SIEVE - Learn independent effect components separately

Key insight from analysis:
- Paddle moves are action-dependent, position-independent
- Ball moves are physics-dependent (position + velocity)
- These are INDEPENDENT phenomena that combine

New approach:
1. DECOMPOSE effects into independent components
2. Learn SEPARATE rules for each component type
3. COMPOSE predictions by combining component predictions

This is like physics: separate forces that superimpose.
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List

class DecomposedSieve:
    """
    Sieve that learns decomposed effect components.
    
    Separates effects into:
    - Paddle effects (d_ tokens) - action-dependent
    - Ball effects (g_ tokens) - physics-dependent
    """
    
    def __init__(self, coherence=0.95, cold_threshold=0.5, min_energy=5):
        self.coherence = coherence
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        
        self.prev_state = None
        
        self.classes = {}
        
        # Separate rule sets
        self.paddle_rules = []  # Rules for paddle effects
        self.ball_rules = []    # Rules for ball effects (with velocity)
    
    def observe(self, state: Set, action: int, next_state: Set):
        before = frozenset(state)
        after = frozenset(next_state)
        
        effect = frozenset({f"+{t}" for t in (after-before)} | 
                          {f"-{t}" for t in (before-after)})
        
        # Decompose effect
        paddle_effect = frozenset(e for e in effect if 'd_' in e)
        ball_effect = frozenset(e for e in effect if 'g_' in e)
        
        # Compute delta (velocity)
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
            'effect': effect,
            'paddle_effect': paddle_effect,
            'ball_effect': ball_effect
        })
        
        # Build waves
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
        
        self.prev_state = before
    
    def reset_episode(self):
        self.prev_state = None
    
    def _interference(self, t1: str, t2: str) -> float:
        w1, w2 = self.waves[t1], self.waves[t2]
        e1, e2 = self.energy[t1], self.energy[t2]
        
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
        tokens = [t for t in self.waves if self.energy[t] >= self.min_energy]
        self.classes = {t: t for t in self.waves}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2:
                    continue
                if not self._structural_match(t1, t2):
                    continue
                if self._interference(t1, t2) > self.coherence:
                    self.classes[t2] = self.classes[t1]
    
    def _abstract(self, s: FrozenSet) -> FrozenSet:
        result = set()
        for t in s:
            if t.startswith('D+'):
                base = t[2:]
                result.add(f"D+{self.classes.get(base, base)}")
            elif t.startswith('D-'):
                base = t[2:]
                result.add(f"D-{self.classes.get(base, base)}")
            else:
                result.add(self.classes.get(t, t))
        return frozenset(result)
    
    def _generate_paddle_rules(self) -> List[Dict]:
        """
        Generate rules for paddle effects.
        
        Paddle movement depends mainly on ACTION, less on state.
        But state determines WHERE paddle is (what moves).
        """
        by_ae = defaultdict(list)
        
        for obs in self.observations:
            if obs['paddle_effect']:
                key = (obs['action'], self._abstract(obs['paddle_effect']))
                # Paddle rules only need paddle-related state tokens
                paddle_state = frozenset(t for t in obs['state'] if 'd_' in t)
                by_ae[key].append(self._abstract(paddle_state))
        
        candidates = []
        for (action, effect), states in by_ae.items():
            if len(states) < 2:
                continue
            
            sample = random.sample(states, min(10, len(states)))
            lhs = frozenset.intersection(*sample)
            
            if lhs:
                candidates.append({
                    'lhs': lhs,
                    'effect': effect,
                    'action': action,
                    'support': len(states),
                    'temperature': 0.0,
                    'type': 'paddle'
                })
        
        return candidates
    
    def _generate_ball_rules(self) -> List[Dict]:
        """
        Generate rules for ball effects.
        
        Ball movement depends on position AND velocity.
        """
        by_ae = defaultdict(list)
        
        for obs in self.observations:
            if obs['ball_effect']:
                key = (obs['action'], self._abstract(obs['ball_effect']))
                # Ball rules need position + velocity
                extended = obs['state'] | obs['delta']
                by_ae[key].append(self._abstract(extended))
        
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
                    'type': 'ball',
                    'has_velocity': has_vel
                })
            
            # Partial intersections
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
                            'type': 'ball',
                            'has_velocity': has_vel
                        })
        
        return candidates
    
    def _anneal(self, candidates: List[Dict], use_velocity: bool = False):
        """Anneal candidates with validation."""
        random.shuffle(self.observations)
        n_test = len(self.observations) // 5
        
        for round_i in range(3):
            test_start = (round_i * n_test) % len(self.observations)
            test_obs = self.observations[test_start:test_start + n_test]
            
            for rule in candidates:
                tp, fp, fn = 0, 0, 0
                
                for obs in test_obs:
                    if rule['type'] == 'paddle':
                        paddle_state = frozenset(t for t in obs['state'] if 'd_' in t)
                        state = self._abstract(paddle_state)
                        actual = self._abstract(obs['paddle_effect'])
                    else:  # ball
                        extended = obs['state'] | obs['delta'] if use_velocity else obs['state']
                        state = self._abstract(extended)
                        actual = self._abstract(obs['ball_effect'])
                    
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
        
        # Generate and anneal paddle rules
        paddle_candidates = self._generate_paddle_rules()
        self._anneal(paddle_candidates)
        self.paddle_rules = [r for r in paddle_candidates if r['temperature'] < self.cold_threshold]
        self.paddle_rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
        
        # Generate and anneal ball rules (with velocity)
        ball_candidates = self._generate_ball_rules()
        self._anneal(ball_candidates, use_velocity=True)
        self.ball_rules = [r for r in ball_candidates if r['temperature'] < self.cold_threshold]
        self.ball_rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
    
    @property
    def rules(self):
        return self.paddle_rules + self.ball_rules
    
    def predict(self, state: Set, action: int, delta: Set = None) -> Set:
        """Predict by combining paddle and ball predictions."""
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        result = set()
        
        # Paddle prediction
        paddle_state = frozenset(t for t in state if 'd_' in t)
        abstract_paddle = self._abstract(paddle_state)
        
        for r in self.paddle_rules:
            if r['action'] == action and r['lhs'] <= abstract_paddle:
                result.update(r['effect'])
                break
        
        # Ball prediction (with velocity)
        if delta is None:
            delta = set()
        extended = frozenset(state) | frozenset(delta)
        abstract_ball = self._abstract(extended)
        
        for r in self.ball_rules:
            if r['action'] == action and r['lhs'] <= abstract_ball:
                result.update(r['effect'])
                break
        
        return result


print("DecomposedSieve loaded!")
