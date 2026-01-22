"""
HIERARCHICAL PROBABILISTIC LEARNER

Combines:
1. Optimistic crystallization (fast learning)
2. Probabilistic rules (handles stochasticity)
3. Hierarchical specificity (general + specific rules coexist)
4. Raw tokenization (no domain knowledge)

Key features:
- General rules give "rule of thumb" predictions
- Specific rules override when they apply
- Probability distributions stabilize over time
- Both deterministic and probabilistic rules are useful
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
    
    @property
    def is_probabilistic(self) -> bool:
        return self.observations >= 10 and 0.1 < self.probability < 0.9
    
    def record(self, occurred: bool):
        if occurred:
            self.successes += 1
        else:
            self.failures += 1
        self.probability_history.append(self.probability)
        if len(self.probability_history) > 100:
            self.probability_history = self.probability_history[-50:]
    
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
    Learns hierarchical probabilistic rules.
    
    General rules (small patterns) coexist with specific rules (large patterns).
    When predicting, most specific matching rule is used.
    """
    
    def __init__(self, n_actions: int = 7):
        self.n_actions = n_actions
        
        self.observations: List[Tuple[FrozenSet, int, FrozenSet]] = []
        self.vocabulary: Set[str] = set()
        
        # Rules indexed by (effect, action)
        # Each effect can have multiple rules at different specificity levels
        self.rules: Dict[Tuple[str, int], List[HierarchicalRule]] = defaultdict(list)
        
        # Track patterns where each effect occurred
        self.effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
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
        
        # Update all matching rules
        for effect_key, rule_list in self.rules.items():
            effect, act = effect_key
            if act != action:
                continue
            
            for rule in rule_list:
                if rule.pattern.issubset(before_fs):
                    rule.record(effect in actual_effects)
        
        # Create/update rules for observed effects
        for effect in actual_effects:
            self._ensure_rules(effect, action, before_fs)
        
        return {'effects': actual_effects}
    
    def _ensure_rules(self, effect: str, action: int, context: FrozenSet):
        """Ensure rules exist at various specificity levels."""
        effect_key = (effect, action)
        self.effect_patterns[effect_key].append(context)
        
        patterns = self.effect_patterns[effect_key]
        rule_list = self.rules[effect_key]
        existing_patterns = {r.pattern for r in rule_list}
        
        # 1. Most specific: full context
        if context not in existing_patterns:
            rule_list.append(HierarchicalRule(
                pattern=context, action=action, effect=effect
            ))
        
        # 2. Common intersection of all patterns (most general)
        if len(patterns) >= 2:
            common = set(patterns[0])
            for p in patterns[1:]:
                common &= p
            common_fs = frozenset(common)
            
            if common_fs and common_fs not in existing_patterns:
                rule_list.append(HierarchicalRule(
                    pattern=common_fs, action=action, effect=effect
                ))
        
        # 3. Try single-token patterns (most general rules)
        for token in context:
            single = frozenset([token])
            if single not in existing_patterns:
                # Only add if this token appears in most patterns
                count = sum(1 for p in patterns if token in p)
                if count >= len(patterns) * 0.5:  # At least 50%
                    rule_list.append(HierarchicalRule(
                        pattern=single, action=action, effect=effect
                    ))
    
    def predict(self, before: Set[str], action: int) -> Dict[str, Tuple[float, HierarchicalRule]]:
        """
        Predict effects with probabilities.
        
        Uses most specific matching rule for each effect.
        Returns dict of effect → (probability, rule).
        """
        before_fs = frozenset(before)
        predictions = {}
        
        for effect_key, rule_list in self.rules.items():
            effect, act = effect_key
            if act != action:
                continue
            
            # Find most specific matching rule with enough observations
            best_rule = None
            best_specificity = -1
            
            for rule in rule_list:
                if rule.observations < 3:
                    continue
                if not rule.pattern.issubset(before_fs):
                    continue
                if rule.specificity > best_specificity:
                    best_specificity = rule.specificity
                    best_rule = rule
            
            if best_rule:
                predictions[effect] = (best_rule.probability, best_rule)
        
        return predictions
    
    def get_rules(self, min_obs: int = 5) -> List[HierarchicalRule]:
        """Get all rules with minimum observations."""
        all_rules = []
        for rule_list in self.rules.values():
            for rule in rule_list:
                if rule.observations >= min_obs:
                    all_rules.append(rule)
        return sorted(all_rules, key=lambda r: (r.action, r.effect, -r.specificity))
    
    def get_deterministic(self, min_obs: int = 10) -> List[HierarchicalRule]:
        """Get deterministic rules (>95% or <5% probability)."""
        return [r for r in self.get_rules(min_obs) if r.is_deterministic]
    
    def get_probabilistic(self, min_obs: int = 10) -> List[HierarchicalRule]:
        """Get probabilistic rules (10-90% probability)."""
        return [r for r in self.get_rules(min_obs) if r.is_probabilistic]
    
    def get_stable(self, min_obs: int = 10) -> List[HierarchicalRule]:
        """Get rules with stable probability distributions."""
        return [r for r in self.get_rules(min_obs) if r.is_stable]
    
    def get_minimal_deterministic(self, min_obs: int = 10) -> List[HierarchicalRule]:
        """
        Get minimal deterministic rules.
        
        For each effect, returns the smallest pattern that is deterministic.
        """
        det_rules = self.get_deterministic(min_obs)
        
        # Group by (effect, action)
        by_effect = defaultdict(list)
        for rule in det_rules:
            by_effect[(rule.effect, rule.action)].append(rule)
        
        minimal = []
        for effect_key, rules in by_effect.items():
            # Find smallest deterministic pattern
            rules.sort(key=lambda r: r.specificity)
            if rules:
                minimal.append(rules[0])
        
        return sorted(minimal, key=lambda r: (r.action, r.effect))
    
    def stats(self) -> Dict:
        """Get statistics."""
        all_rules = self.get_rules(min_obs=1)
        return {
            'observations': len(self.observations),
            'vocabulary': len(self.vocabulary),
            'total_rules': len(all_rules),
            'deterministic': len(self.get_deterministic()),
            'probabilistic': len(self.get_probabilistic()),
            'stable': len(self.get_stable()),
            'effects_tracked': len(self.rules)
        }


def raw_tokenize(obs, carrying=None) -> FrozenSet[str]:
    """Raw tokenizer - no domain knowledge."""
    import numpy as np
    tokens = set()
    
    # Front cell
    front = obs[5, 3]
    tokens.add(f"front_t{int(front[0])}")
    if front[0] >= 4:
        tokens.add(f"front_c{int(front[1])}")
        tokens.add(f"front_s{int(front[2])}")
    
    # Left/right
    tokens.add(f"left_t{int(obs[5, 2, 0])}")
    tokens.add(f"right_t{int(obs[5, 4, 0])}")
    
    # Carrying
    if carrying and carrying[0] != 0:
        tokens.add(f"carry_t{int(carrying[0])}")
        tokens.add(f"carry_c{int(carrying[1])}")
    else:
        tokens.add("carry_none")
    
    # Notable objects in view
    for r in range(7):
        for c in range(7):
            obj_type = int(obs[r, c, 0])
            if obj_type >= 4:
                dist = abs(r - 6) + abs(c - 3)
                loc = "adj" if dist <= 1 else "near" if dist <= 3 else "far"
                tokens.add(f"see_t{obj_type}_{loc}")
                if obj_type == 4:
                    tokens.add(f"door_s{int(obs[r, c, 2])}")
    
    return frozenset(tokens)


if __name__ == "__main__":
    print("HierarchicalLearner loaded successfully")
    print("\nFeatures:")
    print("  - Probabilistic rules with confidence intervals")
    print("  - Hierarchical specificity (general → specific)")
    print("  - Most specific rule used for prediction")
    print("  - Deterministic and probabilistic rules coexist")
    print("  - Raw tokenization (no domain knowledge)")
