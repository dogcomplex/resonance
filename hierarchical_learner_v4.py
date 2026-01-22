"""
HIERARCHICAL PROBABILISTIC LEARNER V4

Adds Temporal Compression (on top of V3):
- Detects transitional/animation states
- Learns stable→stable transitions, skipping intermediates
- Configurable via transition_markers
"""

from collections import defaultdict
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class HierarchicalRule:
    """A rule with probability and specificity."""
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
    V4: Adds temporal compression for animation/transitional states.
    """
    
    def __init__(self, n_actions: int = 7, 
                 position_prefixes: List[str] = None,
                 prototype_prefixes: List[str] = None,
                 transition_markers: List[str] = None):
        self.n_actions = n_actions
        self.position_prefixes = position_prefixes or ['pos_']
        self.prototype_prefixes = prototype_prefixes or [
            'front_', 'left_', 'right_', 'dir_', 'carry_', 'ctx_', 
            'terrain_', '_type_', '_hp_'
        ]
        # Tokens that indicate transitional/animation states (V4 NEW)
        self.transition_markers = transition_markers or [
            'animating', 'ctx_animation', 'anim_frame', 'transitional'
        ]
        
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
        
        # Temporal compression state (V4 NEW)
        self.pending_stable: Optional[Tuple[FrozenSet, int]] = None
        self.temporal_compressions = 0
        
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
        """Check if state is transitional/animation (V4 NEW)."""
        return any(marker in t for t in state for marker in self.transition_markers)
    
    def observe(self, before: Set[str], action: int, after: Set[str]) -> Dict:
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        self.observations.append((before_fs, action, after_fs))
        self.vocabulary.update(before)
        self.vocabulary.update(after)
        
        # V4 NEW: Temporal compression logic
        before_transitional = self._is_transitional(before_fs)
        after_transitional = self._is_transitional(after_fs)
        
        if not before_transitional and after_transitional:
            # Entering transitional state - save stable state
            self.pending_stable = (before_fs, action)
        elif before_transitional and not after_transitional:
            # Exiting transitional state - learn compressed transition
            if self.pending_stable:
                orig_state, orig_action = self.pending_stable
                self._do_observe(orig_state, orig_action, after_fs)
                self.temporal_compressions += 1
                self.pending_stable = None
            # Also learn the exit transition normally
            self._do_observe(before_fs, action, after_fs)
        elif before_transitional and after_transitional:
            # Still in transitional - just track, don't learn
            pass
        else:
            # Normal stable→stable
            self._do_observe(before_fs, action, after_fs)
        
        added = after - before
        removed = before - after
        return {'effects': {f"+{t}" for t in added} | {f"-{t}" for t in removed}}
    
    def _do_observe(self, before_fs: FrozenSet, action: int, after_fs: FrozenSet):
        """Internal observe without temporal logic."""
        added = after_fs - before_fs
        removed = before_fs - after_fs
        actual_effects = {f"+{t}" for t in added} | {f"-{t}" for t in removed}
        
        pos_effects = {e for e in actual_effects if self._is_position_effect(e)}
        rel_effects = {e for e in actual_effects if not self._is_position_effect(e)}
        
        before_rel = self._get_relative_state(before_fs)
        
        # Update full state rules
        for effect_key, rule_list in self.rules.items():
            effect, act = effect_key
            if act != action:
                continue
            for rule in rule_list:
                if rule.pattern.issubset(before_fs):
                    rule.record(effect in actual_effects)
        
        # Update relative state rules
        for effect_key, rule_list in self.rel_rules.items():
            effect, act = effect_key
            if act != action:
                continue
            for rule in rule_list:
                if rule.pattern.issubset(before_rel):
                    rule.record(effect in rel_effects)
        
        # Create rules
        for effect in pos_effects:
            self._ensure_rules(effect, action, before_fs, use_relative=False)
        for effect in rel_effects:
            self._ensure_rules(effect, action, before_rel, use_relative=True)
        
        # Prototype tracking
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
            for rule in rule_list:
                if rule.observations >= min_obs:
                    all_rules.append(rule)
        for rule_list in self.rel_rules.values():
            for rule in rule_list:
                if rule.observations >= min_obs:
                    all_rules.append(rule)
        return all_rules
    
    def predict(self, state: Set[str], action: int, prob_threshold: float = 0.5) -> Set[str]:
        state_fs = frozenset(state)
        state_rel = self._get_relative_state(state_fs)
        predictions = set()
        used_fallback = False
        
        # 1. Relative rules
        rel_predicted = set()
        for (effect, act), rule_list in self.rel_rules.items():
            if act != action:
                continue
            matching = [r for r in rule_list if r.pattern.issubset(state_rel)]
            if matching:
                best = max(matching, key=lambda r: (len(r.pattern), r.observations))
                if best.probability >= prob_threshold and best.observations >= 3:
                    rel_predicted.add(effect)
        
        # 2. Prototype fallback
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
        
        # 3. Position effects
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
            'proto_fallbacks': self.proto_fallbacks,
        }


if __name__ == "__main__":
    learner = HierarchicalLearner(n_actions=4)
    print("V4 initialized:", learner.stats())
