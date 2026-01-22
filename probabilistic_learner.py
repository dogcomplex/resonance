"""
PROBABILISTIC OPTIMISTIC LEARNER

Key innovations:
1. Rules have probability distributions, not just confidence
2. General rules coexist with specific rules
3. Specific rules override general ones when they apply
4. Probability distributions stabilize → crystallization
5. Shattering becomes "probability update" not "rule deletion"

Example:
  General rule: {front_t4} + toggle → +door_s0 (70%)
  Specific rule: {front_t4, front_s2, carry_t5} + toggle → +door_s0 (100%)
  
  The specific rule (locked door + carrying key) always works.
  The general rule is probabilistic because sometimes door is already open.
"""

from collections import defaultdict
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import math
import random


@dataclass
class ProbabilisticRule:
    """A rule with probability distribution over outcomes."""
    pattern: FrozenSet[str]
    action: int
    effect: str
    
    # Outcome tracking
    successes: int = 0  # Times effect occurred when pattern matched
    failures: int = 0   # Times effect didn't occur when pattern matched
    
    # Confidence tracking
    last_probability: float = 0.5
    probability_history: List[float] = field(default_factory=list)
    
    @property
    def probability(self) -> float:
        total = self.successes + self.failures
        if total == 0:
            return 0.5  # Prior
        return self.successes / total
    
    @property
    def observations(self) -> int:
        return self.successes + self.failures
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """Wilson score interval for probability."""
        n = self.observations
        if n == 0:
            return (0.0, 1.0)
        
        p = self.probability
        z = 1.96  # 95% confidence
        
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        
        return (max(0, center - spread), min(1, center + spread))
    
    @property
    def is_stable(self) -> bool:
        """Has probability stabilized?"""
        if len(self.probability_history) < 10:
            return False
        
        recent = self.probability_history[-10:]
        variance = sum((p - self.probability)**2 for p in recent) / len(recent)
        return variance < 0.01  # Low variance = stable
    
    @property
    def is_deterministic(self) -> bool:
        """Is this rule effectively deterministic?"""
        ci = self.confidence_interval
        return ci[0] > 0.95 or ci[1] < 0.05
    
    def record(self, occurred: bool):
        """Record an outcome."""
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
        det = " [DET]" if self.is_deterministic else ""
        stable = " [STABLE]" if self.is_stable else ""
        return (f"{{{pattern_str}}} + a{self.action} → {self.effect} "
                f"({self.probability:.0%} [{ci[0]:.0%}-{ci[1]:.0%}], n={self.observations}){det}{stable}")


class ProbabilisticLearner:
    """
    Learns probabilistic rules with hierarchical specificity.
    """
    
    def __init__(self, n_actions: int = 7, min_obs_to_track: int = 1):
        self.n_actions = n_actions
        self.min_obs = min_obs_to_track
        
        # All observations
        self.observations: List[Tuple[FrozenSet, int, FrozenSet]] = []
        self.vocabulary: Set[str] = set()
        
        # Rules indexed by (effect, action)
        # Multiple rules can exist for same effect (different specificity)
        self.rules: Dict[Tuple[str, int], List[ProbabilisticRule]] = defaultdict(list)
        
        # Track what patterns we've seen for each effect
        self.effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
    def observe(self, before: Set[str], action: int, after: Set[str]) -> Dict:
        """
        Record observation and update all matching rules.
        
        Returns prediction accuracy info.
        """
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        self.observations.append((before_fs, action, after_fs))
        self.vocabulary.update(before)
        self.vocabulary.update(after)
        
        # Compute actual effects
        added = after - before
        removed = before - after
        actual_effects = set()
        for t in added:
            actual_effects.add(f"+{t}")
        for t in removed:
            actual_effects.add(f"-{t}")
        
        # Update existing rules
        predictions = {}
        for effect_key, rule_list in self.rules.items():
            effect, act = effect_key
            if act != action:
                continue
            
            for rule in rule_list:
                if not rule.pattern.issubset(before_fs):
                    continue
                
                # This rule fires
                occurred = effect in actual_effects
                rule.record(occurred)
                
                # Track for prediction accuracy
                if effect not in predictions:
                    predictions[effect] = []
                predictions[effect].append((rule, occurred))
        
        # Create/update rules for observed effects
        for effect in actual_effects:
            self._update_rules(effect, action, before_fs, True)
        
        # Track non-effects for patterns that matched but effect didn't occur
        # This is important for learning that some patterns DON'T cause effects
        for effect_key, rule_list in self.rules.items():
            effect, act = effect_key
            if act != action:
                continue
            if effect in actual_effects:
                continue  # Effect occurred, already handled
            
            for rule in rule_list:
                if rule.pattern.issubset(before_fs):
                    # Pattern matched but effect didn't occur
                    pass  # Already recorded in update loop above
        
        return {
            'actual_effects': actual_effects,
            'predictions': predictions
        }
    
    def _update_rules(self, effect: str, action: int, context: FrozenSet, occurred: bool):
        """Update or create rules for this effect."""
        effect_key = (effect, action)
        
        # Record this pattern
        self.effect_patterns[effect_key].append(context)
        patterns = self.effect_patterns[effect_key]
        
        # Try to create increasingly general rules
        # Start with full context, then try intersections
        
        # 1. Full context rule (most specific)
        self._ensure_rule(effect, action, context)
        
        # 2. If we have multiple patterns, find common subsets
        if len(patterns) >= 2:
            # Intersection of all patterns
            common = set(patterns[0])
            for p in patterns[1:]:
                common &= p
            
            if common:
                self._ensure_rule(effect, action, frozenset(common))
            
            # Also try intersection of recent patterns (for evolving rules)
            if len(patterns) >= 5:
                recent_common = set(patterns[-1])
                for p in patterns[-5:]:
                    recent_common &= p
                if recent_common and recent_common != common:
                    self._ensure_rule(effect, action, frozenset(recent_common))
    
    def _ensure_rule(self, effect: str, action: int, pattern: FrozenSet):
        """Ensure a rule exists for this pattern."""
        effect_key = (effect, action)
        rule_list = self.rules[effect_key]
        
        # Check if rule already exists
        for rule in rule_list:
            if rule.pattern == pattern:
                return  # Already exists
        
        # Create new rule
        rule_list.append(ProbabilisticRule(
            pattern=pattern,
            action=action,
            effect=effect
        ))
    
    def predict(self, before: Set[str], action: int) -> Dict[str, float]:
        """
        Predict effects with probabilities.
        
        Returns dict of effect → probability.
        Uses most specific matching rule for each effect.
        """
        before_fs = frozenset(before)
        predictions = {}
        
        for effect_key, rule_list in self.rules.items():
            effect, act = effect_key
            if act != action:
                continue
            
            # Find most specific matching rule
            best_rule = None
            best_specificity = -1
            
            for rule in rule_list:
                if not rule.pattern.issubset(before_fs):
                    continue
                
                specificity = len(rule.pattern)
                if specificity > best_specificity:
                    best_specificity = specificity
                    best_rule = rule
            
            if best_rule and best_rule.observations >= self.min_obs:
                predictions[effect] = best_rule.probability
        
        return predictions
    
    def get_rules(self, min_obs: int = 5, stable_only: bool = False) -> List[ProbabilisticRule]:
        """Get all rules, optionally filtered."""
        all_rules = []
        
        for rule_list in self.rules.values():
            for rule in rule_list:
                if rule.observations < min_obs:
                    continue
                if stable_only and not rule.is_stable:
                    continue
                all_rules.append(rule)
        
        # Sort by: action, effect, specificity (descending)
        return sorted(all_rules, key=lambda r: (r.action, r.effect, -len(r.pattern)))
    
    def get_deterministic_rules(self, min_obs: int = 10) -> List[ProbabilisticRule]:
        """Get rules that are effectively deterministic."""
        return [r for r in self.get_rules(min_obs) if r.is_deterministic]
    
    def get_probabilistic_rules(self, min_obs: int = 10) -> List[ProbabilisticRule]:
        """Get rules that are clearly probabilistic."""
        return [r for r in self.get_rules(min_obs) 
                if not r.is_deterministic and r.probability > 0.1 and r.probability < 0.9]
    
    def stats(self) -> Dict:
        """Get learner statistics."""
        all_rules = self.get_rules(min_obs=1)
        det_rules = self.get_deterministic_rules(min_obs=5)
        stable_rules = [r for r in all_rules if r.is_stable]
        
        return {
            'observations': len(self.observations),
            'vocabulary': len(self.vocabulary),
            'total_rules': len(all_rules),
            'deterministic': len(det_rules),
            'stable': len(stable_rules),
            'effects_tracked': len(self.rules)
        }


if __name__ == "__main__":
    print("ProbabilisticLearner loaded successfully")
    
    # Quick test with stochastic game
    learner = ProbabilisticLearner(n_actions=2)
    
    # Simulate: action 0 at pos_0 → 80% chance of moving to pos_1
    for i in range(200):
        before = {'pos_0'}
        action = 0
        if random.random() < 0.8:
            after = {'pos_1'}
        else:
            after = {'pos_0'}  # Stayed
        
        learner.observe(before, action, after)
    
    print(f"\nStats: {learner.stats()}")
    print("\nRules:")
    for rule in learner.get_rules(min_obs=10):
        print(f"  {rule}")
