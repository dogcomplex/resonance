"""
OPTIMISTIC CRYSTALLIZING LEARNER

A rule learning system that:
1. Aggressively crystallizes patterns from minimal observations (2-3)
2. Shatters rules when predictions fail
3. Refines patterns using counter-examples
4. Ablates to find minimal necessary conditions

This approach trades safety for speed:
- Traditional: Wait for 10+ observations, 95%+ confidence → slow but safe
- Optimistic: Crystallize after 2 observations → fast but may shatter

Usage:
    learner = OptimisticLearner(n_actions=7)
    
    for episode in range(1000):
        state = env.reset()
        for step in range(max_steps):
            action = select_action(state)
            next_state = env.step(action)
            
            surprises = learner.observe(state, action, next_state)
            if surprises:
                print(f"Learned something new! {len(surprises)} rules shattered")
            
            # Use learned rules for prediction
            predicted = learner.predict(state, action)
            
            state = next_state
    
    # Get minimal ruleset
    for rule in learner.get_rules():
        print(rule)
"""

from collections import defaultdict
from typing import Set, FrozenSet, Dict, List, Tuple
from dataclasses import dataclass
import random


@dataclass
class Rule:
    """A crystallized rule with pattern, action, and effect."""
    pattern: FrozenSet[str]
    action: int
    effect: str  # e.g., "+token" or "-token"
    confidence: float
    observations: int
    shatter_count: int = 0
    
    def __str__(self):
        pattern_str = ', '.join(sorted(self.pattern)) if self.pattern else '∅'
        shatter = f" [shattered {self.shatter_count}x]" if self.shatter_count else ""
        return f"{{{pattern_str}}} + a{self.action} → {self.effect} ({self.confidence:.0%}, n={self.observations}){shatter}"
    
    def matches(self, state: FrozenSet[str], action: int) -> bool:
        """Check if this rule applies to the given state and action."""
        return action == self.action and self.pattern.issubset(state)


class OptimisticLearner:
    """
    Learns rules through optimistic crystallization with shatter-on-surprise.
    """
    
    def __init__(self, n_actions: int = 7, 
                 min_observations: int = 2,
                 min_confidence: float = 0.90):
        """
        Initialize learner.
        
        Args:
            n_actions: Number of possible actions
            min_observations: Minimum observations before crystallizing (default: 2)
            min_confidence: Minimum precision to crystallize (default: 0.90)
        """
        self.n_actions = n_actions
        self.min_obs = min_observations
        self.min_conf = min_confidence
        
        # Data storage
        self.observations: List[Tuple[FrozenSet, int, FrozenSet]] = []
        self.vocabulary: Set[str] = set()
        
        # Effect tracking: (effect, action) → list of contexts where it occurred
        self.positive_examples: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        self.negative_examples: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
        # Active rules
        self.rules: Dict[Tuple[str, int], Rule] = {}
        
        # Statistics
        self.total_shatters = 0
        self.total_surprises = 0
    
    def observe(self, before: Set[str], action: int, after: Set[str]) -> List[Rule]:
        """
        Record an observation and update rules.
        
        Returns list of rules that were shattered (surprised).
        """
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        # Check predictions against reality
        shattered = self._check_predictions(before_fs, action, after_fs)
        
        # Record observation
        self.observations.append((before_fs, action, after_fs))
        self.vocabulary.update(before)
        self.vocabulary.update(after)
        
        # Track effects
        added = after - before
        removed = before - after
        
        effects_occurred = set()
        for token in added:
            effect = f"+{token}"
            self.positive_examples[(effect, action)].append(before_fs)
            effects_occurred.add((effect, action))
            self._try_crystallize(effect, action)
        
        for token in removed:
            effect = f"-{token}"
            self.positive_examples[(effect, action)].append(before_fs)
            effects_occurred.add((effect, action))
            self._try_crystallize(effect, action)
        
        # Track negative examples (rule matched but effect didn't happen)
        for effect_key, rule in self.rules.items():
            if effect_key[1] != action:
                continue
            if not rule.pattern.issubset(before_fs):
                continue
            if effect_key not in effects_occurred:
                self.negative_examples[effect_key].append(before_fs)
        
        return shattered
    
    def _check_predictions(self, before: FrozenSet, action: int, 
                           after: FrozenSet) -> List[Rule]:
        """Check if any active rules made wrong predictions. Shatter if so."""
        shattered = []
        
        for effect_key, rule in list(self.rules.items()):
            effect, act = effect_key
            if act != action:
                continue
            if not rule.pattern.issubset(before):
                continue
            
            # Rule fires - did effect actually happen?
            target = effect[1:]
            occurred = (effect[0] == '+' and target in after and target not in before) or \
                      (effect[0] == '-' and target in before and target not in after)
            
            if not occurred:
                # SURPRISE! Prediction failed
                shattered.append(rule)
                self._shatter(rule, before)
        
        return shattered
    
    def _shatter(self, rule: Rule, counter_example: FrozenSet):
        """Remove rule and record counter-example for refinement."""
        effect_key = (rule.effect, rule.action)
        
        self.total_shatters += 1
        self.total_surprises += 1
        
        del self.rules[effect_key]
        self.negative_examples[effect_key].append(counter_example)
        
        # Try to re-crystallize with refined pattern
        self._try_crystallize(rule.effect, rule.action, rule.shatter_count + 1)
    
    def _try_crystallize(self, effect: str, action: int, shatter_count: int = 0):
        """Attempt to crystallize a rule for this effect."""
        effect_key = (effect, action)
        
        # Already have a rule?
        if effect_key in self.rules:
            self.rules[effect_key].observations += 1
            return
        
        positives = self.positive_examples.get(effect_key, [])
        negatives = self.negative_examples.get(effect_key, [])
        
        # Need enough observations (more if previously shattered)
        min_needed = self.min_obs + shatter_count
        if len(positives) < min_needed:
            return
        
        # Find tokens common to all positive examples
        common = set(positives[0])
        for ctx in positives[1:]:
            common &= ctx
        
        # ABLATION: Find minimal necessary tokens
        # A token is necessary if removing it would match a negative example
        necessary = set()
        for token in common:
            other = common - {token}
            for neg in negatives:
                if other.issubset(neg):
                    necessary.add(token)
                    break
        
        # If no tokens proven necessary, try to find smallest sufficient set
        if not necessary:
            for token in sorted(common):
                test_pattern = necessary | {token}
                precision = self._compute_precision(test_pattern, action, effect)
                if precision >= self.min_conf:
                    necessary.add(token)
                    break
            
            if not necessary:
                necessary = common
        
        # Verify precision
        precision = self._compute_precision(necessary, action, effect)
        
        if precision < self.min_conf:
            return
        
        # Count observations with this pattern
        obs_count = sum(1 for ctx in positives if necessary.issubset(ctx))
        
        if obs_count < self.min_obs:
            return
        
        # Crystallize!
        self.rules[effect_key] = Rule(
            pattern=frozenset(necessary),
            action=action,
            effect=effect,
            confidence=precision,
            observations=obs_count,
            shatter_count=shatter_count
        )
    
    def _compute_precision(self, pattern: Set[str], action: int, effect: str) -> float:
        """Compute precision of a pattern for an effect."""
        matches = effect_count = 0
        pattern_fs = frozenset(pattern)
        
        for before, a, after in self.observations:
            if a != action:
                continue
            if not pattern_fs.issubset(before):
                continue
            
            matches += 1
            target = effect[1:]
            if effect[0] == '+':
                if target in after and target not in before:
                    effect_count += 1
            else:
                if target in before and target not in after:
                    effect_count += 1
        
        return effect_count / matches if matches > 0 else 0
    
    def predict(self, state: Set[str], action: int) -> Set[str]:
        """Predict next state using crystallized rules."""
        result = set(state)
        state_fs = frozenset(state)
        
        for (effect, act), rule in self.rules.items():
            if act != action:
                continue
            if not rule.pattern.issubset(state_fs):
                continue
            
            if effect[0] == '+':
                result.add(effect[1:])
            else:
                result.discard(effect[1:])
        
        return result
    
    def get_rules(self) -> List[Rule]:
        """Get all crystallized rules, sorted by action and pattern size."""
        return sorted(self.rules.values(), 
                     key=lambda r: (r.action, len(r.pattern), r.effect))
    
    def stats(self) -> Dict:
        """Get learner statistics."""
        return {
            'observations': len(self.observations),
            'vocabulary': len(self.vocabulary),
            'rules': len(self.rules),
            'shatters': self.total_shatters,
            'effects_seen': len(self.positive_examples)
        }
    
    def __repr__(self):
        stats = self.stats()
        return f"OptimisticLearner({stats['rules']} rules, {stats['shatters']} shatters, {stats['observations']} obs)"


if __name__ == "__main__":
    # Quick test
    print("OptimisticLearner loaded successfully")
    
    learner = OptimisticLearner(n_actions=4)
    
    # Simulate simple game
    for i in range(100):
        before = {'pos_0'} if i % 2 == 0 else {'pos_1'}
        action = 0
        after = {'pos_1'} if 'pos_0' in before else {'pos_0'}
        learner.observe(before, action, after)
    
    print(f"\n{learner}")
    print("\nRules:")
    for rule in learner.get_rules():
        print(f"  {rule}")
