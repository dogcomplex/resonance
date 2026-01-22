"""
V9c: For unseen states, predict effects where most-specific rule has 
     probability > average probability across all matching effects.
     
This is adaptive thresholding - if rules are confident, predict confidently.
If all rules are ~30%, don't predict anything (uncertain).
"""

from collections import defaultdict
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Rule:
    pattern: FrozenSet[str]
    action: int
    effect: str
    successes: int = 0
    failures: int = 0
    
    @property
    def probability(self) -> float:
        n = self.successes + self.failures
        return self.successes / n if n > 0 else 0.0
    
    @property
    def observations(self) -> int:
        return self.successes + self.failures
    
    @property
    def specificity(self) -> int:
        return len(self.pattern)
    
    def record(self, occurred: bool):
        if occurred:
            self.successes += 1
        else:
            self.failures += 1


class HierarchicalLearner:
    def __init__(self, n_actions: int = 7,
                 position_prefixes: List[str] = None,
                 prototype_prefixes: List[str] = None,
                 min_confidence: float = 0.5):  # Minimum probability to predict
        self.n_actions = n_actions
        self.position_prefixes = position_prefixes or ['pos_']
        self.prototype_prefixes = prototype_prefixes or [
            'front_', 'left_', 'right_', 'dir_', 'carry_', 'ctx_'
        ]
        self.min_confidence = min_confidence
        
        self.exact_observations: Dict[Tuple[FrozenSet, int], Dict[FrozenSet, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: Dict[Tuple[str, int], List[Rule]] = defaultdict(list)
        self.rel_rules: Dict[Tuple[str, int], List[Rule]] = defaultdict(list)
        self.effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        self.rel_effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
        self.delta_effects: Dict[Tuple[FrozenSet, str, int], Dict[Tuple[int,int], int]] = defaultdict(lambda: defaultdict(int))
        self.delta_counts: Dict[Tuple[FrozenSet, str, int], int] = defaultdict(int)
        
        self.total_observations = 0
    
    def _is_position_token(self, token: str) -> bool:
        return any(token.startswith(p) for p in self.position_prefixes)
    
    def _is_position_effect(self, effect: str) -> bool:
        return self._is_position_token(effect[1:])
    
    def _get_relative_state(self, state: FrozenSet) -> FrozenSet:
        return frozenset(t for t in state if not self._is_position_token(t))
    
    def _get_prototype(self, state: FrozenSet) -> FrozenSet:
        return frozenset(t for t in state if any(p in t for p in self.prototype_prefixes))
    
    def _get_direction(self, state: FrozenSet) -> Optional[str]:
        for t in state:
            if t.startswith('dir_'):
                return t
        return None
    
    def _get_position_token(self, state: FrozenSet) -> Optional[str]:
        for t in state:
            if self._is_position_token(t):
                return t
        return None
    
    def _get_position(self, state: FrozenSet) -> Optional[Tuple[int, int]]:
        for t in state:
            if t.startswith('pos_'):
                parts = t.split('_')
                if len(parts) == 3:
                    try:
                        return (int(parts[1]), int(parts[2]))
                    except ValueError:
                        pass
        return None
    
    def observe(self, before: Set[str], action: int, after: Set[str]) -> Dict:
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        self.total_observations += 1
        
        added = after_fs - before_fs
        removed = before_fs - after_fs
        actual_effects = frozenset({f"+{t}" for t in added} | {f"-{t}" for t in removed})
        
        self.exact_observations[(before_fs, action)][actual_effects] += 1
        
        pos_effects = {e for e in actual_effects if self._is_position_effect(e)}
        rel_effects = frozenset(actual_effects) - pos_effects
        before_rel = self._get_relative_state(before_fs)
        
        for effect in pos_effects:
            self._ensure_rules(effect, action, before_fs, self.rules, self.effect_patterns)
        for effect in rel_effects:
            self._ensure_rules(effect, action, before_rel, self.rel_rules, self.rel_effect_patterns)
        
        self._update_rules(self.rules, before_fs, action, actual_effects)
        self._update_rules(self.rel_rules, before_rel, action, rel_effects)
        
        direction = self._get_direction(before_fs)
        if direction:
            bp, ap = self._get_position(before_fs), self._get_position(after_fs)
            if bp and ap:
                delta = (ap[0] - bp[0], ap[1] - bp[1])
                proto = self._get_prototype(before_fs)
                key = (proto, direction, action)
                self.delta_effects[key][delta] += 1
                self.delta_counts[key] += 1
        
        return {'effects': set(actual_effects)}
    
    def _update_rules(self, rule_dict, context, action, actual_effects):
        for (effect, act), rule_list in rule_dict.items():
            if act != action:
                continue
            for rule in rule_list:
                if rule.pattern.issubset(context):
                    rule.record(effect in actual_effects)
    
    def _ensure_rules(self, effect, action, context, rule_dict, pattern_dict):
        key = (effect, action)
        patterns = pattern_dict[key]
        rules = rule_dict[key]
        existing = {r.pattern for r in rules}
        
        patterns.append(context)
        if context not in existing:
            rules.append(Rule(pattern=context, action=action, effect=effect))
        
        if len(patterns) >= 2:
            common = frozenset(set(patterns[0]).intersection(*patterns[1:]))
            if common and common not in existing:
                rules.append(Rule(pattern=common, action=action, effect=effect))
        
        for token in context:
            single = frozenset([token])
            if single not in existing:
                rules.append(Rule(pattern=single, action=action, effect=effect))
    
    def _get_best_rule(self, rules: List[Rule], context: FrozenSet,
                        require_position: bool = False) -> Optional[Rule]:
        matching = [r for r in rules if r.pattern.issubset(context) and r.successes > 0]
        
        if require_position:
            pos_token = self._get_position_token(context)
            if pos_token:
                matching = [r for r in matching if pos_token in r.pattern]
        
        if not matching:
            return None
        
        matching.sort(key=lambda r: (-r.specificity, -r.observations))
        return matching[0]
    
    def predict(self, state: Set[str], action: int, threshold: float = None) -> Set[str]:
        if threshold is None:
            threshold = self.min_confidence
            
        state_fs = frozenset(state)
        
        # EXACT MATCH
        key = (state_fs, action)
        if key in self.exact_observations:
            obs = self.exact_observations[key]
            if obs:
                if len(obs) == 1:
                    return set(list(obs.keys())[0])
                total = sum(obs.values())
                effect_counts = defaultdict(int)
                for effects, cnt in obs.items():
                    for e in effects:
                        effect_counts[e] += cnt
                return {e for e, c in effect_counts.items() if c / total >= threshold}
        
        # UNSEEN: Get probabilities from best rules
        state_rel = self._get_relative_state(state_fs)
        effect_probs = {}
        
        for (effect, act) in self.rel_rules.keys():
            if act != action:
                continue
            best = self._get_best_rule(self.rel_rules[(effect, action)], state_rel)
            if best:
                effect_probs[effect] = (best.probability, best.specificity, best.observations)
        
        for (effect, act) in self.rules.keys():
            if act != action or not self._is_position_effect(effect):
                continue
            best = self._get_best_rule(self.rules[(effect, action)], state_fs, require_position=True)
            if best:
                effect_probs[effect] = (best.probability, best.specificity, best.observations)
        
        if not effect_probs:
            # Delta fallback
            predictions = set()
            direction = self._get_direction(state_fs)
            current_pos = self._get_position(state_fs)
            if direction and current_pos:
                proto = self._get_prototype(state_fs)
                delta_key = (proto, direction, action)
                if delta_key in self.delta_counts and self.delta_counts[delta_key] > 0:
                    total = self.delta_counts[delta_key]
                    best_delta = max(self.delta_effects[delta_key].items(), key=lambda x: x[1])
                    delta, count = best_delta
                    if count / total >= threshold and delta != (0, 0):
                        new_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
                        predictions.add(f"-pos_{current_pos[0]}_{current_pos[1]}")
                        predictions.add(f"+pos_{new_pos[0]}_{new_pos[1]}")
            return predictions
        
        # Predict effects where prob >= threshold AND specificity is high enough
        # Adaptive: only predict if rule is more specific than average
        avg_spec = sum(s for _, s, _ in effect_probs.values()) / len(effect_probs)
        
        predictions = set()
        for effect, (prob, spec, n) in effect_probs.items():
            # Predict if: probability meets threshold AND rule is specific
            if prob >= threshold and spec >= avg_spec:
                predictions.add(effect)
        
        return predictions
    
    def predict_probs(self, state: Set[str], action: int) -> Dict[str, float]:
        """Return probability distribution over effects from best matching rules."""
        state_fs = frozenset(state)
        
        # Exact match - return observed distribution
        key = (state_fs, action)
        if key in self.exact_observations:
            obs = self.exact_observations[key]
            if obs:
                total = sum(obs.values())
                effect_counts = defaultdict(int)
                for effects, cnt in obs.items():
                    for e in effects:
                        effect_counts[e] += cnt
                return {e: c/total for e, c in effect_counts.items()}
        
        # Fallback - return probabilities from best rules
        state_rel = self._get_relative_state(state_fs)
        probs = {}
        
        for (effect, act) in self.rel_rules.keys():
            if act != action:
                continue
            best = self._get_best_rule(self.rel_rules[(effect, action)], state_rel)
            if best:
                probs[effect] = best.probability
        
        for (effect, act) in self.rules.keys():
            if act != action or not self._is_position_effect(effect):
                continue
            best = self._get_best_rule(self.rules[(effect, action)], state_fs, require_position=True)
            if best:
                probs[effect] = best.probability
        
        return probs
    
    def stats(self) -> Dict:
        return {
            'observations': self.total_observations,
            'exact_states': len(self.exact_observations),
            'full_rules': sum(len(rl) for rl in self.rules.values()),
            'rel_rules': sum(len(rl) for rl in self.rel_rules.values()),
        }
