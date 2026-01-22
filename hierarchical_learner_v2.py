"""
HIERARCHICAL PROBABILISTIC LEARNER V2

Adds Context-Aware Learning:
- Separates position-dependent from position-independent effects
- Uses relative state for view/dir effects (better generalization)
- Uses full state for position effects (position-specific)

This is the first extension integration.
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
    def specificity(self) -> int:
        return len(self.pattern)
    
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
    def is_stable(self) -> bool:
        if len(self.probability_history) < 10:
            return False
        recent = self.probability_history[-10:]
        mean = sum(recent) / len(recent)
        variance = sum((p - mean)**2 for p in recent) / len(recent)
        return variance < 0.01
    
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
    
    def __str__(self):
        pattern_str = ', '.join(sorted(self.pattern)) if self.pattern else '∅'
        ci = self.confidence_interval
        tags = []
        if self.is_deterministic:
            tags.append("DET")
        if self.is_stable:
            tags.append("STABLE")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return (f"{{{pattern_str}}} + a{self.action} → {self.effect} "
                f"({self.probability:.0%} [{ci[0]:.0%}-{ci[1]:.0%}], "
                f"n={self.observations}){tag_str}")


class HierarchicalLearner:
    """
    Learns hierarchical probabilistic rules with context awareness.
    
    V2 adds:
    - Automatic detection of position tokens (configurable)
    - Separate tracking for position-dependent vs position-independent effects
    - Better generalization for view/direction effects
    """
    
    def __init__(self, n_actions: int = 7, position_prefixes: List[str] = None):
        self.n_actions = n_actions
        
        # Configurable position token detection
        # Default: tokens starting with 'pos_' are position-specific
        self.position_prefixes = position_prefixes or ['pos_']
        
        self.observations: List[Tuple[FrozenSet, int, FrozenSet]] = []
        self.vocabulary: Set[str] = set()
        
        # Full state rules (for position effects)
        self.rules: Dict[Tuple[str, int], List[HierarchicalRule]] = defaultdict(list)
        self.effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
        # Relative state rules (for position-independent effects) - NEW in V2
        self.rel_rules: Dict[Tuple[str, int], List[HierarchicalRule]] = defaultdict(list)
        self.rel_effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
        # Stats
        self.context_aware_enabled = True
    
    def _is_position_token(self, token: str) -> bool:
        """Check if token is position-related."""
        return any(token.startswith(p) for p in self.position_prefixes)
    
    def _is_position_effect(self, effect: str) -> bool:
        """Check if effect involves position change."""
        # Effect is like "+pos_3_4" or "-pos_3_4"
        token = effect[1:]  # Remove +/- prefix
        return self._is_position_token(token)
    
    def _get_relative_state(self, state: FrozenSet) -> FrozenSet:
        """Remove position tokens from state."""
        return frozenset(t for t in state if not self._is_position_token(t))
    
    def observe(self, before: Set[str], action: int, after: Set[str]) -> Dict:
        """Record observation and update rules."""
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        self.observations.append((before_fs, action, after_fs))
        self.vocabulary.update(before)
        self.vocabulary.update(after)
        
        # Compute effects
        added = after - before
        removed = before - after
        actual_effects = {f"+{t}" for t in added} | {f"-{t}" for t in removed}
        
        # Separate position vs relative effects
        pos_effects = {e for e in actual_effects if self._is_position_effect(e)}
        rel_effects = {e for e in actual_effects if not self._is_position_effect(e)}
        
        # Get relative state
        before_rel = self._get_relative_state(before_fs)
        
        # Update FULL state rules (for position effects)
        for effect_key, rule_list in self.rules.items():
            effect, act = effect_key
            if act != action:
                continue
            for rule in rule_list:
                if rule.pattern.issubset(before_fs):
                    rule.record(effect in actual_effects)
        
        # Update RELATIVE state rules (for relative effects) - NEW in V2
        if self.context_aware_enabled:
            for effect_key, rule_list in self.rel_rules.items():
                effect, act = effect_key
                if act != action:
                    continue
                for rule in rule_list:
                    if rule.pattern.issubset(before_rel):
                        rule.record(effect in rel_effects)
        
        # Create/update rules for position effects (full state)
        for effect in pos_effects:
            self._ensure_rules(effect, action, before_fs, use_relative=False)
        
        # Create/update rules for relative effects (relative state) - NEW in V2
        if self.context_aware_enabled:
            for effect in rel_effects:
                self._ensure_rules(effect, action, before_rel, use_relative=True)
        else:
            # Fallback: use full state for all effects
            for effect in rel_effects:
                self._ensure_rules(effect, action, before_fs, use_relative=False)
        
        return {'effects': actual_effects, 'pos_effects': pos_effects, 'rel_effects': rel_effects}
    
    def _ensure_rules(self, effect: str, action: int, context: FrozenSet, use_relative: bool = False):
        """Ensure rules exist at various specificity levels."""
        effect_key = (effect, action)
        
        if use_relative:
            patterns = self.rel_effect_patterns[effect_key]
            rule_list = self.rel_rules[effect_key]
        else:
            patterns = self.effect_patterns[effect_key]
            rule_list = self.rules[effect_key]
        
        patterns.append(context)
        existing_patterns = {r.pattern for r in rule_list}
        
        # 1. Most specific: full context
        if context not in existing_patterns:
            rule_list.append(HierarchicalRule(
                pattern=context, action=action, effect=effect
            ))
        
        # 2. Common intersection (most general)
        if len(patterns) >= 2:
            common = set(patterns[0])
            for p in patterns[1:]:
                common &= p
            common_fs = frozenset(common)
            
            if common_fs and common_fs not in existing_patterns:
                rule_list.append(HierarchicalRule(
                    pattern=common_fs, action=action, effect=effect
                ))
        
        # 3. Single-token patterns
        for token in context:
            single = frozenset([token])
            if single not in existing_patterns:
                rule_list.append(HierarchicalRule(
                    pattern=single, action=action, effect=effect
                ))
    
    def get_stable(self, min_obs: int = 5) -> List[HierarchicalRule]:
        """Get all stable rules with minimum observations."""
        all_rules = []
        
        # Full state rules
        for rule_list in self.rules.values():
            for rule in rule_list:
                if rule.observations >= min_obs:
                    all_rules.append(rule)
        
        # Relative state rules
        for rule_list in self.rel_rules.values():
            for rule in rule_list:
                if rule.observations >= min_obs:
                    all_rules.append(rule)
        
        return all_rules
    
    def predict(self, state: Set[str], action: int, prob_threshold: float = 0.5) -> Set[str]:
        """
        Predict effects for state-action pair.
        Uses relative rules for relative effects, full rules for position effects.
        """
        state_fs = frozenset(state)
        state_rel = self._get_relative_state(state_fs)
        predictions = set()
        
        # Predict relative effects using relative rules
        if self.context_aware_enabled:
            for (effect, act), rule_list in self.rel_rules.items():
                if act != action:
                    continue
                
                matching = [r for r in rule_list if r.pattern.issubset(state_rel)]
                if matching:
                    best = max(matching, key=lambda r: (len(r.pattern), r.observations))
                    if best.probability >= prob_threshold and best.observations >= 3:
                        predictions.add(effect)
        
        # Predict position effects using full rules
        for (effect, act), rule_list in self.rules.items():
            if act != action:
                continue
            
            # Skip relative effects if context_aware (handled above)
            if self.context_aware_enabled and not self._is_position_effect(effect):
                continue
            
            matching = [r for r in rule_list if r.pattern.issubset(state_fs)]
            if matching:
                best = max(matching, key=lambda r: (len(r.pattern), r.observations))
                if best.probability >= prob_threshold and best.observations >= 3:
                    predictions.add(effect)
        
        return predictions
    
    def stats(self) -> Dict:
        """Return learner statistics."""
        full_rules = sum(len(rl) for rl in self.rules.values())
        rel_rules = sum(len(rl) for rl in self.rel_rules.values())
        
        return {
            'observations': len(self.observations),
            'vocabulary_size': len(self.vocabulary),
            'full_state_rules': full_rules,
            'relative_state_rules': rel_rules,
            'total_rules': full_rules + rel_rules,
            'context_aware_enabled': self.context_aware_enabled,
        }


# Quick test
if __name__ == "__main__":
    learner = HierarchicalLearner(n_actions=4)
    
    # Simulate some observations
    states = [
        {'pos_1_1', 'dir_0', 'front_t1'},
        {'pos_1_2', 'dir_0', 'front_t1'},
        {'pos_2_1', 'dir_0', 'front_t2'},
    ]
    
    for i, s in enumerate(states[:-1]):
        learner.observe(s, 2, states[i+1])
    
    print("Stats:", learner.stats())
    print(f"Rules: {len(learner.get_stable(min_obs=1))}")
