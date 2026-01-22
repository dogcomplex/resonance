"""
V6b: Position prototype fallback with DELTA effects, not absolute positions
"""

from collections import defaultdict
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import re


@dataclass
class HierarchicalRule:
    pattern: FrozenSet[str]
    action: int
    effect: str
    successes: int = 0
    failures: int = 0
    
    @property
    def probability(self) -> float:
        n = self.successes + self.failures
        if n == 0:
            return 0.5
        return self.successes / n
    
    @property
    def observations(self) -> int:
        return self.successes + self.failures
    
    def record(self, occurred: bool):
        if occurred:
            self.successes += 1
        else:
            self.failures += 1


class HierarchicalLearner:
    def __init__(self, n_actions: int = 7, 
                 position_prefixes: List[str] = None,
                 prototype_prefixes: List[str] = None,
                 transition_markers: List[str] = None,
                 derive_comparisons: bool = True):
        self.n_actions = n_actions
        self.position_prefixes = position_prefixes or ['pos_']
        self.prototype_prefixes = prototype_prefixes or [
            'front_', 'left_', 'right_', 'dir_', 'carry_', 'ctx_'
        ]
        self.transition_markers = transition_markers or [
            'animating', 'ctx_animation', 'anim_frame', 'transitional'
        ]
        self.derive_comparisons = derive_comparisons
        
        self.observations = []
        self.vocabulary: Set[str] = set()
        
        # Rules
        self.rules: Dict[Tuple[str, int], List[HierarchicalRule]] = defaultdict(list)
        self.effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        self.rel_rules: Dict[Tuple[str, int], List[HierarchicalRule]] = defaultdict(list)
        self.rel_effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
        # Prototypes
        self.proto_effects: Dict[Tuple[FrozenSet, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.proto_counts: Dict[Tuple[FrozenSet, int], int] = defaultdict(int)
        
        # V6b: Position DELTA prototypes (key: proto, direction, action -> delta pattern)
        self.delta_proto_effects: Dict[Tuple[FrozenSet, str, int], Dict[Tuple[int,int], int]] = defaultdict(lambda: defaultdict(int))
        self.delta_proto_counts: Dict[Tuple[FrozenSet, str, int], int] = defaultdict(int)
        
        # Temporal
        self.pending_stable: Optional[Tuple[FrozenSet, int]] = None
        self.temporal_compressions = 0
        
        # Stats
        self.proto_fallbacks = 0
        self.delta_fallbacks = 0
    
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
    
    def _get_direction(self, state: FrozenSet) -> Optional[str]:
        for t in state:
            if t.startswith('dir_'):
                return t
        return None
    
    def _get_position(self, state: FrozenSet) -> Optional[Tuple[int, int]]:
        """Extract position as (x, y) tuple."""
        for t in state:
            if t.startswith('pos_'):
                parts = t.split('_')
                # Handle pos_X_Y format
                if len(parts) == 3:
                    try:
                        return (int(parts[1]), int(parts[2]))
                    except ValueError:
                        pass
                # Handle pos_x_N_y_M format
                if 'x' in t and 'y' in t:
                    try:
                        x_idx = parts.index('x')
                        y_idx = parts.index('y')
                        return (int(parts[x_idx+1]), int(parts[y_idx+1]))
                    except (ValueError, IndexError):
                        pass
        return None
    
    def _is_transitional(self, state: FrozenSet) -> bool:
        return any(marker in t for t in state for marker in self.transition_markers)
    
    def observe(self, before: Set[str], action: int, after: Set[str]) -> Dict:
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        self.observations.append((before_fs, action, after_fs))
        self.vocabulary.update(before)
        self.vocabulary.update(after)
        
        # Temporal
        before_trans = self._is_transitional(before_fs)
        after_trans = self._is_transitional(after_fs)
        
        if not before_trans and after_trans:
            self.pending_stable = (before_fs, action)
        elif before_trans and not after_trans:
            if self.pending_stable:
                self._do_observe(self.pending_stable[0], self.pending_stable[1], after_fs)
                self.temporal_compressions += 1
                self.pending_stable = None
            self._do_observe(before_fs, action, after_fs)
        elif not before_trans and not after_trans:
            self._do_observe(before_fs, action, after_fs)
        
        added = after - before
        removed = before - after
        return {'effects': {f"+{t}" for t in added} | {f"-{t}" for t in removed}}
    
    def _do_observe(self, before_fs: FrozenSet, action: int, after_fs: FrozenSet):
        added = after_fs - before_fs
        removed = before_fs - after_fs
        actual_effects = {f"+{t}" for t in added} | {f"-{t}" for t in removed}
        
        pos_effects = {e for e in actual_effects if self._is_position_effect(e)}
        rel_effects = {e for e in actual_effects if not self._is_position_effect(e)}
        
        before_rel = self._get_relative_state(before_fs)
        proto_key = self._get_prototype(before_fs)
        direction = self._get_direction(before_fs)
        
        # Update rules
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
        
        # Create rules
        for effect in pos_effects:
            self._ensure_rules(effect, action, before_fs, use_relative=False)
        for effect in rel_effects:
            self._ensure_rules(effect, action, before_rel, use_relative=True)
        
        # Relative prototypes
        for e in rel_effects:
            self.proto_effects[(proto_key, action)][e] += 1
        self.proto_counts[(proto_key, action)] += 1
        
        # V6b: Delta prototypes
        if direction:
            before_pos = self._get_position(before_fs)
            after_pos = self._get_position(after_fs)
            if before_pos and after_pos:
                delta = (after_pos[0] - before_pos[0], after_pos[1] - before_pos[1])
                self.delta_proto_effects[(proto_key, direction, action)][delta] += 1
                self.delta_proto_counts[(proto_key, direction, action)] += 1
    
    def _ensure_rules(self, effect: str, action: int, context: FrozenSet, use_relative: bool):
        effect_key = (effect, action)
        if use_relative:
            patterns = self.rel_effect_patterns[effect_key]
            rule_list = self.rel_rules[effect_key]
        else:
            patterns = self.effect_patterns[effect_key]
            rule_list = self.rules[effect_key]
        
        patterns.append(context)
        existing = {r.pattern for r in rule_list}
        
        if context not in existing:
            rule_list.append(HierarchicalRule(pattern=context, action=action, effect=effect))
        
        if len(patterns) >= 2:
            common = frozenset(set(patterns[0]).intersection(*patterns[1:]))
            if common and common not in existing:
                rule_list.append(HierarchicalRule(pattern=common, action=action, effect=effect))
        
        for token in context:
            single = frozenset([token])
            if single not in existing:
                rule_list.append(HierarchicalRule(pattern=single, action=action, effect=effect))
    
    def predict(self, state: Set[str], action: int, prob_threshold: float = 0.5) -> Set[str]:
        state_fs = frozenset(state)
        state_rel = self._get_relative_state(state_fs)
        proto_key = self._get_prototype(state_fs)
        direction = self._get_direction(state_fs)
        current_pos = self._get_position(state_fs)
        
        predictions = set()
        
        # 1. Relative effects
        rel_predicted = set()
        for (effect, act), rule_list in self.rel_rules.items():
            if act != action:
                continue
            matching = [r for r in rule_list if r.pattern.issubset(state_rel)]
            if matching:
                best = max(matching, key=lambda r: (len(r.pattern), r.observations))
                if best.probability >= prob_threshold and best.observations >= 3:
                    rel_predicted.add(effect)
        
        # Relative prototype fallback
        if not rel_predicted:
            key = (proto_key, action)
            if key in self.proto_counts and self.proto_counts[key] >= 3:
                total = self.proto_counts[key]
                for effect, count in self.proto_effects[key].items():
                    if count / total >= prob_threshold:
                        rel_predicted.add(effect)
                        self.proto_fallbacks += 1
        
        predictions.update(rel_predicted)
        
        # 2. Position effects from full rules
        pos_predicted = set()
        for (effect, act), rule_list in self.rules.items():
            if act != action or not self._is_position_effect(effect):
                continue
            matching = [r for r in rule_list if r.pattern.issubset(state_fs)]
            if matching:
                best = max(matching, key=lambda r: (len(r.pattern), r.observations))
                if best.probability >= prob_threshold and best.observations >= 3:
                    pos_predicted.add(effect)
        
        # V6b: Delta prototype fallback
        if not pos_predicted and direction and current_pos:
            key = (proto_key, direction, action)
            if key in self.delta_proto_counts and self.delta_proto_counts[key] >= 3:
                total = self.delta_proto_counts[key]
                for delta, count in self.delta_proto_effects[key].items():
                    if count / total >= prob_threshold:
                        # Apply delta to current position
                        new_x = current_pos[0] + delta[0]
                        new_y = current_pos[1] + delta[1]
                        if delta != (0, 0):
                            pos_predicted.add(f"-pos_{current_pos[0]}_{current_pos[1]}")
                            pos_predicted.add(f"+pos_{new_x}_{new_y}")
                        self.delta_fallbacks += 1
        
        predictions.update(pos_predicted)
        
        return predictions
    
    def stats(self) -> Dict:
        return {
            'observations': len(self.observations),
            'full_rules': sum(len(rl) for rl in self.rules.values()),
            'rel_rules': sum(len(rl) for rl in self.rel_rules.values()),
            'rel_prototypes': len(self.proto_counts),
            'delta_prototypes': len(self.delta_proto_counts),
            'proto_fallbacks': self.proto_fallbacks,
            'delta_fallbacks': self.delta_fallbacks,
        }


if __name__ == "__main__":
    learner = HierarchicalLearner()
    print("V6b initialized:", learner.stats())
