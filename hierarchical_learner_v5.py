"""
HIERARCHICAL PROBABILISTIC LEARNER V5 (FINAL)

Adds Derived Tokens (on top of V4):
- Automatically discovers useful numerical comparisons
- Adds derived tokens for relationships that predict outcomes
- Configurable via numeric_token_patterns
"""

from collections import defaultdict
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math
import re


@dataclass
class HierarchicalRule:
    pattern: FrozenSet[str]
    action: int
    effect: str
    successes: int = 0
    failures: int = 0
    probability_history: List[float] = field(default_factory=list)
    
    @property
    def probability(self) -> float:
        n = self.successes + self.failures
        return self.successes / n if n > 0 else 0.5
    
    @property
    def observations(self) -> int:
        return self.successes + self.failures
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        n = self.observations
        if n == 0:
            return (0.0, 1.0)
        p = self.probability
        z = 1.96
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        return (max(0, center - spread), min(1, center + spread))
    
    @property
    def is_deterministic(self) -> bool:
        ci = self.confidence_interval
        return self.observations >= 10 and (ci[0] > 0.95 or ci[1] < 0.05)
    
    def record(self, occurred: bool):
        if occurred:
            self.successes += 1
        else:
            self.failures += 1
        self.probability_history.append(self.probability)


class HierarchicalLearner:
    """
    V5 (Final): All extensions integrated.
    - V2: Context-aware (position vs relative)
    - V3: Prototype fallback
    - V4: Temporal compression
    - V5: Derived tokens for numeric comparisons
    """
    
    def __init__(self, n_actions: int = 7, 
                 position_prefixes: List[str] = None,
                 prototype_prefixes: List[str] = None,
                 transition_markers: List[str] = None,
                 derive_comparisons: bool = True):
        self.n_actions = n_actions
        self.position_prefixes = position_prefixes or ['pos_']
        self.prototype_prefixes = prototype_prefixes or [
            'front_', 'left_', 'right_', 'dir_', 'carry_', 'ctx_', 
            'terrain_', '_type_', '_hp_'
        ]
        self.transition_markers = transition_markers or [
            'animating', 'ctx_animation', 'anim_frame', 'transitional'
        ]
        self.derive_comparisons = derive_comparisons
        
        self.observations = []
        self.vocabulary: Set[str] = set()
        
        # Full state rules
        self.rules: Dict[Tuple[str, int], List[HierarchicalRule]] = defaultdict(list)
        self.effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
        # Relative state rules
        self.rel_rules: Dict[Tuple[str, int], List[HierarchicalRule]] = defaultdict(list)
        self.rel_effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
        # Prototype rules
        self.proto_effects: Dict[Tuple[FrozenSet, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.proto_counts: Dict[Tuple[FrozenSet, int], int] = defaultdict(int)
        
        # Temporal compression
        self.pending_stable: Optional[Tuple[FrozenSet, int]] = None
        self.temporal_compressions = 0
        
        # Derived tokens (V5 NEW)
        self.derived_tokens_added = 0
        
        # Stats
        self.proto_fallbacks = 0
        self.exact_hits = 0
    
    def _is_position_token(self, token: str) -> bool:
        return any(token.startswith(p) for p in self.position_prefixes)
    
    def _is_position_effect(self, effect: str) -> bool:
        return self._is_position_token(effect[1:])
    
    def _get_relative_state(self, state: FrozenSet) -> FrozenSet:
        return frozenset(t for t in state if not self._is_position_token(t))
    
    def _get_prototype(self, state: FrozenSet) -> FrozenSet:
        proto = set()
        for t in state:
            if any(p in t for p in self.prototype_prefixes):
                proto.add(t)
        return frozenset(proto)
    
    def _is_transitional(self, state: FrozenSet) -> bool:
        return any(marker in t for t in state for marker in self.transition_markers)
    
    def _derive_tokens(self, state: Set[str]) -> Set[str]:
        """V5 NEW: Add derived comparison tokens."""
        if not self.derive_comparisons:
            return state
        
        derived = set(state)
        
        # Extract numeric values from tokens
        numerics = {}
        for t in state:
            # Pattern: prefix_value (e.g., player_atk_52, enemy_def_43)
            match = re.match(r'(.+?)_(\d+)$', t)
            if match:
                prefix, value = match.groups()
                numerics[prefix] = int(value)
        
        # Generate comparisons for known pairs
        comparison_pairs = [
            ('player_atk', 'enemy_def', 'atk_vs_def'),
            ('player_spd', 'enemy_spd', 'speed_cmp'),
            ('player_def', 'enemy_atk', 'def_vs_atk'),
        ]
        
        for key1, key2, name in comparison_pairs:
            if key1 in numerics and key2 in numerics:
                if numerics[key1] > numerics[key2]:
                    derived.add(f"derived_{name}_advantage")
                else:
                    derived.add(f"derived_{name}_disadvantage")
                self.derived_tokens_added += 1
        
        return derived
    
    def observe(self, before: Set[str], action: int, after: Set[str]) -> Dict:
        # Add derived tokens
        before_derived = self._derive_tokens(before)
        after_derived = self._derive_tokens(after)
        
        before_fs = frozenset(before_derived)
        after_fs = frozenset(after_derived)
        
        self.observations.append((before_fs, action, after_fs))
        self.vocabulary.update(before_derived)
        self.vocabulary.update(after_derived)
        
        # Temporal compression
        before_transitional = self._is_transitional(before_fs)
        after_transitional = self._is_transitional(after_fs)
        
        if not before_transitional and after_transitional:
            self.pending_stable = (before_fs, action)
        elif before_transitional and not after_transitional:
            if self.pending_stable:
                orig_state, orig_action = self.pending_stable
                self._do_observe(orig_state, orig_action, after_fs)
                self.temporal_compressions += 1
                self.pending_stable = None
            self._do_observe(before_fs, action, after_fs)
        elif before_transitional and after_transitional:
            pass
        else:
            self._do_observe(before_fs, action, after_fs)
        
        added = after_derived - before_derived
        removed = before_derived - after_derived
        return {'effects': {f"+{t}" for t in added} | {f"-{t}" for t in removed}}
    
    def _do_observe(self, before_fs: FrozenSet, action: int, after_fs: FrozenSet):
        added = after_fs - before_fs
        removed = before_fs - after_fs
        actual_effects = {f"+{t}" for t in added} | {f"-{t}" for t in removed}
        
        # Filter out derived token effects (they're computed, not learned)
        actual_effects = {e for e in actual_effects if 'derived_' not in e}
        
        pos_effects = {e for e in actual_effects if self._is_position_effect(e)}
        rel_effects = {e for e in actual_effects if not self._is_position_effect(e)}
        
        before_rel = self._get_relative_state(before_fs)
        
        for effect_key, rule_list in self.rules.items():
            effect, act = effect_key
            if act != action:
                continue
            for rule in rule_list:
                if rule.pattern.issubset(before_fs):
                    rule.record(effect in actual_effects)
        
        for effect_key, rule_list in self.rel_rules.items():
            effect, act = effect_key
            if act != action:
                continue
            for rule in rule_list:
                if rule.pattern.issubset(before_rel):
                    rule.record(effect in rel_effects)
        
        for effect in pos_effects:
            self._ensure_rules(effect, action, before_fs, use_relative=False)
        for effect in rel_effects:
            self._ensure_rules(effect, action, before_rel, use_relative=True)
        
        proto_key = self._get_prototype(before_fs)
        for e in rel_effects:
            self.proto_effects[(proto_key, action)][e] += 1
        self.proto_counts[(proto_key, action)] += 1
    
    def _ensure_rules(self, effect: str, action: int, context: FrozenSet, use_relative: bool):
        effect_key = (effect, action)
        if use_relative:
            patterns = self.rel_effect_patterns[effect_key]
            rule_list = self.rel_rules[effect_key]
        else:
            patterns = self.effect_patterns[effect_key]
            rule_list = self.rules[effect_key]
        
        patterns.append(context)
        existing_patterns = {r.pattern for r in rule_list}
        
        if context not in existing_patterns:
            rule_list.append(HierarchicalRule(pattern=context, action=action, effect=effect))
        
        if len(patterns) >= 2:
            common = set(patterns[0])
            for p in patterns[1:]:
                common &= p
            common_fs = frozenset(common)
            if common_fs and common_fs not in existing_patterns:
                rule_list.append(HierarchicalRule(pattern=common_fs, action=action, effect=effect))
        
        for token in context:
            single = frozenset([token])
            if single not in existing_patterns:
                rule_list.append(HierarchicalRule(pattern=single, action=action, effect=effect))
    
    def get_stable(self, min_obs: int = 5) -> List[HierarchicalRule]:
        all_rules = []
        for rule_list in self.rules.values():
            for r in rule_list:
                if r.observations >= min_obs:
                    all_rules.append(r)
        for rule_list in self.rel_rules.values():
            for r in rule_list:
                if r.observations >= min_obs:
                    all_rules.append(r)
        return all_rules
    
    def predict(self, state: Set[str], action: int, prob_threshold: float = 0.5) -> Set[str]:
        state_derived = self._derive_tokens(state)
        state_fs = frozenset(state_derived)
        state_rel = self._get_relative_state(state_fs)
        predictions = set()
        used_fallback = False
        
        rel_predicted = set()
        for (effect, act), rule_list in self.rel_rules.items():
            if act != action:
                continue
            matching = [r for r in rule_list if r.pattern.issubset(state_rel)]
            if matching:
                best = max(matching, key=lambda r: (len(r.pattern), r.observations))
                if best.probability >= prob_threshold and best.observations >= 3:
                    rel_predicted.add(effect)
        
        if not rel_predicted:
            proto_key = self._get_prototype(state_fs)
            key = (proto_key, action)
            if key in self.proto_counts and self.proto_counts[key] >= 3:
                total = self.proto_counts[key]
                for effect, count in self.proto_effects[key].items():
                    if count / total >= prob_threshold:
                        rel_predicted.add(effect)
                        used_fallback = True
        
        predictions.update(rel_predicted)
        
        for (effect, act), rule_list in self.rules.items():
            if act != action or not self._is_position_effect(effect):
                continue
            matching = [r for r in rule_list if r.pattern.issubset(state_fs)]
            if matching:
                best = max(matching, key=lambda r: (len(r.pattern), r.observations))
                if best.probability >= prob_threshold and best.observations >= 3:
                    predictions.add(effect)
        
        if used_fallback:
            self.proto_fallbacks += 1
        else:
            self.exact_hits += 1
        
        return predictions
    
    def stats(self) -> Dict:
        return {
            'observations': len(self.observations),
            'full_state_rules': sum(len(rl) for rl in self.rules.values()),
            'relative_state_rules': sum(len(rl) for rl in self.rel_rules.values()),
            'prototype_states': len(self.proto_counts),
            'temporal_compressions': self.temporal_compressions,
            'derived_tokens_added': self.derived_tokens_added,
            'proto_fallbacks': self.proto_fallbacks,
        }


if __name__ == "__main__":
    learner = HierarchicalLearner(n_actions=7)
    print("V5 initialized:", learner.stats())
