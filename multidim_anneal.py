"""
MULTI-DIMENSIONAL ANNEALING with Cross-Universe Validation

Three dimensions of annealing:
1. TIME SERIES: Rules validated across different time windows
2. FIDELITY: Rules validated at different abstraction levels
3. PROBABILITY: Rules validated across confidence thresholds

Plus: CROSS-UNIVERSE validation
- Run multiple simulations with different seeds
- Rules that work across universes are more fundamental
- "Non-local" connections emerge when patterns match across universes
"""

import random
from collections import defaultdict
from typing import FrozenSet, Set, Dict, List, Tuple
import math

class MultiDimAnneal:
    """
    Multi-dimensional annealing with cross-universe validation.
    
    Anneals along three axes simultaneously:
    1. Temporal: Does rule work at different time points?
    2. Fidelity: Does rule work at different abstraction levels?
    3. Probabilistic: Does rule have consistent confidence?
    
    Plus cross-validation across different "universes" (seed variations).
    """
    
    def __init__(self,
                 base_coherence=0.95,
                 fidelity_levels=[0.99, 0.95, 0.90],  # Abstraction thresholds
                 time_windows=3,                       # Temporal splits
                 n_universes=3,                        # Cross-universe validation
                 cold_threshold=0.4,                   # Final acceptance
                 min_energy=5):
        
        self.base_coherence = base_coherence
        self.fidelity_levels = sorted(fidelity_levels, reverse=True)
        self.time_windows = time_windows
        self.n_universes = n_universes
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        
        # Wave state
        self.waves = defaultdict(lambda: defaultdict(float))
        self.energy = defaultdict(float)
        
        # Temporal tracking
        self.observations = []
        self.obs_time = []  # When each observation occurred
        self.current_time = 0
        
        self.exact = defaultdict(lambda: defaultdict(int))
        
        # Multi-fidelity classes
        self.class_levels = {}  # fidelity -> classes
        
        # Results
        self.classes = {}
        self.rules = []
        
        # Cross-universe state
        self.universe_rules = {}  # universe_id -> rules
    
    def observe(self, state: Set, action: int, next_state: Set):
        """Accumulate observations with temporal tracking."""
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | 
                          {f"-{t}" for t in (before-after)})
        
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'before': before,
            'effect': effect,
            'action': action
        })
        self.obs_time.append(self.current_time)
        self.current_time += 1
        
        for token in before:
            self.energy[token] += 1
            for e in effect:
                self.waves[token][e] += 1.0
    
    def _interference(self, t1: str, t2: str) -> float:
        """Compute wave interference."""
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
    
    def _build_classes_at_fidelity(self, fidelity: float) -> Dict[str, str]:
        """Build classes at a specific fidelity (coherence) level."""
        tokens = [t for t in self.waves if self.energy[t] >= self.min_energy]
        classes = {t: t for t in self.waves}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if classes[t2] != t2:
                    continue
                if not self._structural_match(t1, t2):
                    continue
                if self._interference(t1, t2) > fidelity:
                    classes[t2] = classes[t1]
        
        return classes
    
    def _generate_candidates_at_fidelity(self, classes: Dict[str, str], 
                                          obs_subset: List[Dict]) -> List[Dict]:
        """Generate candidate rules at a specific fidelity."""
        def abstract(s):
            return frozenset(classes.get(t, t) for t in s)
        
        by_ae = defaultdict(list)
        for obs in obs_subset:
            if obs['effect']:
                key = (obs['action'], abstract(obs['effect']))
                by_ae[key].append(abstract(obs['before']))
        
        candidates = []
        
        for (action, effect), states in by_ae.items():
            if len(states) < 2:
                continue
            
            # Full intersection
            sample = random.sample(states, min(10, len(states)))
            lhs = frozenset.intersection(*sample)
            if lhs:
                candidates.append({
                    'lhs': lhs,
                    'effect': effect,
                    'action': action,
                    'support': len(states),
                    'temperature': 0.0,
                    'temporal_score': 0.0,
                    'fidelity_score': 0.0,
                    'prob_score': 0.0
                })
            
            # Partial intersections
            if len(states) >= 4:
                for _ in range(2):
                    subset = random.sample(states, len(states) // 2)
                    lhs_partial = frozenset.intersection(*subset)
                    if lhs_partial and lhs_partial != lhs:
                        candidates.append({
                            'lhs': lhs_partial,
                            'effect': effect,
                            'action': action,
                            'support': len(subset),
                            'temperature': 0.2,
                            'temporal_score': 0.0,
                            'fidelity_score': 0.0,
                            'prob_score': 0.0
                        })
        
        return candidates
    
    def _score_rule(self, rule: Dict, test_obs: List[Dict], 
                    classes: Dict[str, str]) -> float:
        """Score a rule on test observations."""
        def abstract(s):
            return frozenset(classes.get(t, t) for t in s)
        
        tp, fp, fn = 0, 0, 0
        
        for obs in test_obs:
            state = abstract(obs['before'])
            actual = abstract(obs['effect'])
            
            if rule['action'] == obs['action'] and rule['lhs'] <= state:
                predicted = rule['effect']
                for e in predicted:
                    if e in actual: tp += 1
                    else: fp += 1
                for e in actual:
                    if e not in predicted: fn += 1
        
        if tp + fp + fn == 0:
            return 0.5
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    def _temporal_anneal(self, candidates: List[Dict], classes: Dict[str, str]):
        """Anneal along temporal dimension."""
        if not self.observations:
            return
        
        # Split into time windows
        n = len(self.observations)
        window_size = n // self.time_windows
        
        for rule in candidates:
            scores = []
            for w in range(self.time_windows):
                start = w * window_size
                end = start + window_size if w < self.time_windows - 1 else n
                window_obs = self.observations[start:end]
                
                score = self._score_rule(rule, window_obs, classes)
                scores.append(score)
            
            # Temporal consistency = low variance across windows
            if scores:
                mean_score = sum(scores) / len(scores)
                variance = sum((s - mean_score)**2 for s in scores) / len(scores)
                rule['temporal_score'] = mean_score * (1 - variance)
    
    def _fidelity_anneal(self, candidates: List[Dict]):
        """Anneal along fidelity dimension."""
        for rule in candidates:
            fidelity_scores = []
            
            for fidelity in self.fidelity_levels:
                classes = self.class_levels[fidelity]
                score = self._score_rule(rule, self.observations[-len(self.observations)//5:], classes)
                fidelity_scores.append(score)
            
            # Fidelity consistency = works at multiple abstraction levels
            if fidelity_scores:
                rule['fidelity_score'] = sum(fidelity_scores) / len(fidelity_scores)
    
    def _probability_anneal(self, candidates: List[Dict], classes: Dict[str, str]):
        """Anneal along probability dimension."""
        def abstract(s):
            return frozenset(classes.get(t, t) for t in s)
        
        for rule in candidates:
            # Count how often rule fires and is correct
            fires = 0
            correct = 0
            
            for obs in self.observations:
                state = abstract(obs['before'])
                if rule['action'] == obs['action'] and rule['lhs'] <= state:
                    fires += 1
                    actual = abstract(obs['effect'])
                    if rule['effect'] == actual:
                        correct += 1
            
            # Probabilistic score = confidence when rule fires
            rule['prob_score'] = correct / fires if fires > 0 else 0.0
    
    def _compute_final_temperature(self, rule: Dict) -> float:
        """Combine all dimensions into final temperature."""
        # Weight the three dimensions
        w_temporal = 0.3
        w_fidelity = 0.3
        w_prob = 0.4
        
        combined_score = (
            w_temporal * rule['temporal_score'] +
            w_fidelity * rule['fidelity_score'] +
            w_prob * rule['prob_score']
        )
        
        # Temperature = 1 - score (cold = good)
        return 1.0 - combined_score
    
    def build(self):
        """Build with multi-dimensional annealing."""
        # Build class systems at all fidelity levels
        for fidelity in self.fidelity_levels:
            self.class_levels[fidelity] = self._build_classes_at_fidelity(fidelity)
        
        # Use base fidelity for main classes
        self.classes = self.class_levels[self.base_coherence]
        
        # Generate candidates
        candidates = self._generate_candidates_at_fidelity(
            self.classes, self.observations
        )
        
        # Multi-dimensional annealing
        self._temporal_anneal(candidates, self.classes)
        self._fidelity_anneal(candidates)
        self._probability_anneal(candidates, self.classes)
        
        # Compute final temperatures
        for rule in candidates:
            rule['temperature'] = self._compute_final_temperature(rule)
        
        # Filter cold rules
        self.rules = [r for r in candidates if r['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
    
    def predict(self, state: Set, action: int) -> Set:
        key = (frozenset(state), action)
        
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        abstract_state = frozenset(self.classes.get(t, t) for t in state)
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        
        return set()


class CrossUniverseAnneal:
    """
    Cross-universe annealing: Train on multiple "universes" (seeds),
    then validate rules across universes.
    
    Rules that work in all universes are more fundamental/universal.
    This is the "non-local" connection - patterns that transcend
    the specific initial conditions.
    """
    
    def __init__(self, 
                 n_universes=3,
                 base_coherence=0.95,
                 cold_threshold=0.4,
                 min_energy=5):
        
        self.n_universes = n_universes
        self.base_coherence = base_coherence
        self.cold_threshold = cold_threshold
        self.min_energy = min_energy
        
        # Per-universe state
        self.universes = []  # List of (waves, energy, observations)
        
        # Global state (merged across universes)
        self.global_waves = defaultdict(lambda: defaultdict(float))
        self.global_energy = defaultdict(float)
        self.all_observations = []
        
        self.exact = defaultdict(lambda: defaultdict(int))
        
        self.classes = {}
        self.rules = []
        
        # Current universe
        self.current_universe = 0
    
    def start_universe(self, universe_id: int):
        """Start a new universe for observation."""
        self.current_universe = universe_id
        while len(self.universes) <= universe_id:
            self.universes.append({
                'waves': defaultdict(lambda: defaultdict(float)),
                'energy': defaultdict(float),
                'observations': []
            })
    
    def observe(self, state: Set, action: int, next_state: Set):
        """Observe in current universe."""
        before = frozenset(state)
        after = frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | 
                          {f"-{t}" for t in (before-after)})
        
        self.exact[(before, action)][effect] += 1
        
        obs = {'before': before, 'effect': effect, 'action': action,
               'universe': self.current_universe}
        
        # Add to current universe
        u = self.universes[self.current_universe]
        u['observations'].append(obs)
        for token in before:
            u['energy'][token] += 1
            for e in effect:
                u['waves'][token][e] += 1.0
        
        # Add to global
        self.all_observations.append(obs)
        for token in before:
            self.global_energy[token] += 1
            for e in effect:
                self.global_waves[token][e] += 1.0
    
    def _interference(self, t1: str, t2: str, waves, energy) -> float:
        w1, w2 = waves[t1], waves[t2]
        e1, e2 = energy[t1], energy[t2]
        
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
    
    def _build_classes(self, waves, energy) -> Dict[str, str]:
        tokens = [t for t in waves if energy[t] >= self.min_energy]
        classes = {t: t for t in waves}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if classes[t2] != t2:
                    continue
                if not self._structural_match(t1, t2):
                    continue
                if self._interference(t1, t2, waves, energy) > self.base_coherence:
                    classes[t2] = classes[t1]
        
        return classes
    
    def _generate_candidates(self, classes, observations) -> List[Dict]:
        def abstract(s):
            return frozenset(classes.get(t, t) for t in s)
        
        by_ae = defaultdict(list)
        for obs in observations:
            if obs['effect']:
                key = (obs['action'], abstract(obs['effect']))
                by_ae[key].append(abstract(obs['before']))
        
        candidates = []
        for (action, effect), states in by_ae.items():
            if len(states) < 2:
                continue
            
            sample = random.sample(states, min(10, len(states)))
            lhs = frozenset.intersection(*sample)
            if lhs:
                candidates.append({
                    'lhs': lhs, 'effect': effect, 'action': action,
                    'support': len(states), 'temperature': 0.0,
                    'cross_universe_score': 0.0
                })
            
            if len(states) >= 4:
                for _ in range(2):
                    subset = random.sample(states, len(states) // 2)
                    lhs_partial = frozenset.intersection(*subset)
                    if lhs_partial and lhs_partial != lhs:
                        candidates.append({
                            'lhs': lhs_partial, 'effect': effect, 'action': action,
                            'support': len(subset), 'temperature': 0.2,
                            'cross_universe_score': 0.0
                        })
        
        return candidates
    
    def _score_in_universe(self, rule, universe_idx, classes) -> float:
        """Score rule in a specific universe."""
        def abstract(s):
            return frozenset(classes.get(t, t) for t in s)
        
        u = self.universes[universe_idx]
        tp, fp, fn = 0, 0, 0
        
        for obs in u['observations']:
            state = abstract(obs['before'])
            actual = abstract(obs['effect'])
            
            if rule['action'] == obs['action'] and rule['lhs'] <= state:
                predicted = rule['effect']
                for e in predicted:
                    if e in actual: tp += 1
                    else: fp += 1
                for e in actual:
                    if e not in predicted: fn += 1
        
        if tp + fp + fn == 0:
            return 0.5
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    def _cross_universe_anneal(self, candidates, classes):
        """Score rules across all universes."""
        for rule in candidates:
            universe_scores = []
            for u_idx in range(len(self.universes)):
                score = self._score_in_universe(rule, u_idx, classes)
                universe_scores.append(score)
            
            if universe_scores:
                # Cross-universe score = minimum across universes
                # (rule must work in ALL universes)
                rule['cross_universe_score'] = min(universe_scores)
                
                # Also track consistency
                mean = sum(universe_scores) / len(universe_scores)
                variance = sum((s - mean)**2 for s in universe_scores) / len(universe_scores)
                rule['universe_consistency'] = 1 - variance
    
    def build(self):
        """Build with cross-universe validation."""
        # Build global classes
        self.classes = self._build_classes(self.global_waves, self.global_energy)
        
        # Generate candidates from global observations
        candidates = self._generate_candidates(self.classes, self.all_observations)
        
        # Cross-universe annealing
        self._cross_universe_anneal(candidates, self.classes)
        
        # Final temperature combines cross-universe score
        for rule in candidates:
            rule['temperature'] = 1.0 - rule['cross_universe_score']
        
        # Filter
        self.rules = [r for r in candidates if r['temperature'] < self.cold_threshold]
        self.rules.sort(key=lambda r: (-len(r['lhs']), -r['support']))
    
    def predict(self, state: Set, action: int) -> Set:
        key = (frozenset(state), action)
        
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        
        abstract_state = frozenset(self.classes.get(t, t) for t in state)
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        
        return set()


print("MultiDimAnneal and CrossUniverseAnneal loaded!")
