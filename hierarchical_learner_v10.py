"""
V10b: Smarter adaptive confidence.

Key insight: Only blend when there's EVIDENCE of non-determinism.
- If we've seen this state multiple times with same outcome → deterministic, trust exact
- If we've seen it once → check if SIMILAR states are deterministic
- If similar states show variance → this might be aliased, blend
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
                 prototype_prefixes: List[str] = None):
        self.n_actions = n_actions
        self.position_prefixes = position_prefixes or ['pos_']
        self.prototype_prefixes = prototype_prefixes or [
            'front_', 'left_', 'right_', 'dir_', 'carry_', 'ctx_'
        ]
        
        self.exact_observations: Dict[Tuple[FrozenSet, int], Dict[FrozenSet, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: Dict[Tuple[str, int], List[Rule]] = defaultdict(list)
        self.rel_rules: Dict[Tuple[str, int], List[Rule]] = defaultdict(list)
        self.effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        self.rel_effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
        self.delta_effects: Dict[Tuple[FrozenSet, str, int], Dict[Tuple[int,int], int]] = defaultdict(lambda: defaultdict(int))
        self.delta_counts: Dict[Tuple[FrozenSet, str, int], int] = defaultdict(int)
        
        # Track overall determinism of the environment
        self.total_observations = 0
        self.total_multi_outcome_states = 0  # States with >1 outcome
    
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
        
        key = (before_fs, action)
        was_single = len(self.exact_observations[key]) <= 1
        self.exact_observations[key][actual_effects] += 1
        is_multi = len(self.exact_observations[key]) > 1
        
        # Track if this state became multi-outcome
        if was_single and is_multi:
            self.total_multi_outcome_states += 1
        
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
                delta_key = (proto, direction, action)
                self.delta_effects[delta_key][delta] += 1
                self.delta_counts[delta_key] += 1
        
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
    
    def _environment_seems_deterministic(self) -> bool:
        """Does the environment appear to be deterministic?"""
        if len(self.exact_observations) < 10:
            return True  # Assume deterministic until proven otherwise
        return self.total_multi_outcome_states / len(self.exact_observations) < 0.1
    
    def predict_probs(self, state: Set[str], action: int) -> Dict[str, float]:
        state_fs = frozenset(state)
        key = (state_fs, action)
        
        # Check exact observations
        if key in self.exact_observations:
            obs = self.exact_observations[key]
            total = sum(obs.values())
            n_outcomes = len(obs)
            
            if n_outcomes == 1:
                # Single outcome - this is our best knowledge
                effects = list(obs.keys())[0]
                return {e: 1.0 for e in effects}
            else:
                # Multiple outcomes - return distribution
                effect_counts = defaultdict(int)
                for effects, count in obs.items():
                    for e in effects:
                        effect_counts[e] += count
                return {e: c/total for e, c in effect_counts.items()}
        
        # No exact match - use general rules
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
        
        # Delta fallback
        if not any(self._is_position_effect(e) for e in probs):
            direction = self._get_direction(state_fs)
            current_pos = self._get_position(state_fs)
            if direction and current_pos:
                proto = self._get_prototype(state_fs)
                delta_key = (proto, direction, action)
                if delta_key in self.delta_counts and self.delta_counts[delta_key] > 0:
                    total = self.delta_counts[delta_key]
                    best_delta = max(self.delta_effects[delta_key].items(), key=lambda x: x[1])
                    delta, count = best_delta
                    prob = count / total
                    if delta != (0, 0):
                        new_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
                        probs[f"-pos_{current_pos[0]}_{current_pos[1]}"] = prob
                        probs[f"+pos_{new_pos[0]}_{new_pos[1]}"] = prob
        
        return probs
    
    def predict(self, state: Set[str], action: int, threshold: float = 0.5) -> Set[str]:
        """Predict effects above threshold."""
        state_fs = frozenset(state)
        key = (state_fs, action)
        
        # EXACT MATCH - always trust if we have it
        if key in self.exact_observations:
            obs = self.exact_observations[key]
            if obs:
                n_outcomes = len(obs)
                if n_outcomes == 1:
                    # Deterministic - return exactly what we saw
                    return set(list(obs.keys())[0])
                else:
                    # Probabilistic - return effects above threshold
                    total = sum(obs.values())
                    effect_counts = defaultdict(int)
                    for effects, count in obs.items():
                        for e in effects:
                            effect_counts[e] += count
                    return {e for e, c in effect_counts.items() if c/total >= threshold}
        
        # NO EXACT MATCH - use general rules with adaptive threshold
        probs = self.predict_probs(state, action)
        
        # If environment is deterministic, be more aggressive
        if self._environment_seems_deterministic():
            # In deterministic env, the most likely outcome is probably correct
            # Lower threshold for effects that are clearly predicted
            return {e for e, p in probs.items() if p >= threshold * 0.8}
        else:
            # In probabilistic env, stick with threshold
            return {e for e, p in probs.items() if p >= threshold}
    
    def stats(self) -> Dict:
        return {
            'observations': self.total_observations,
            'exact_states': len(self.exact_observations),
            'multi_outcome_states': self.total_multi_outcome_states,
            'determinism': 1 - (self.total_multi_outcome_states / max(1, len(self.exact_observations))),
            'rel_rules': sum(len(rl) for rl in self.rel_rules.values()),
        }
