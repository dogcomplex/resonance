"""
Production Rule Learner

Based on the insight that game rules can be encoded as production rules:
  LHS => RHS
  
Where:
- LHS tokens are requirements (must be present)
- RHS tokens are productions (what the rule produces)
- A token on BOTH sides acts as a catalyst/conditional (IF pattern THEN result)

For classification:
  pattern_conditions => label

NO CHEATING - patterns are discovered purely from observation.
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Token:
    """A token representing a condition or fact."""
    name: str
    value: Any
    
    def __str__(self):
        return f"{self.name}={self.value}"


@dataclass
class ProductionRule:
    """
    A production rule: IF conditions THEN production
    
    For classification:
    - LHS = set of input conditions (position has value)
    - RHS = output label
    """
    lhs: FrozenSet[Token]
    rhs: int
    support: int = 0
    confidence: float = 1.0
    alive: bool = True
    
    def matches(self, tokens: Set[Token]) -> bool:
        return self.lhs.issubset(tokens)
    
    def specificity(self) -> int:
        return len(self.lhs)
    
    def __hash__(self):
        return hash((self.lhs, self.rhs))


class ProductionRuleLearner:
    """
    Few-shot learner using production rules.
    """
    
    def __init__(self, num_outputs: int = 5, board_size: int = None,
                 label_names: List[str] = None,
                 pattern_sizes: List[int] = None,
                 **kwargs):
        self.num_outputs = num_outputs
        self.board_size = board_size
        self.label_names = label_names
        self.pattern_sizes = pattern_sizes or [3]
        
        self.rules: Dict[Tuple[FrozenSet[Token], int], ProductionRule] = {}
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        self.label_counts: Dict[int, int] = defaultdict(int)
        self.observed_values: Set[str] = set()
        
        self.stats = {'generated': 0, 'eliminated': 0, 'predictions': 0}
    
    def _board_to_tokens(self, board: str) -> Set[Token]:
        tokens = set()
        for pos, val in enumerate(board):
            tokens.add(Token(f"p{pos}", val))
        return tokens
    
    def _generate_rules(self, board: str, label: int):
        tokens = self._board_to_tokens(board)
        token_list = list(tokens)
        
        for size in self.pattern_sizes:
            if size > len(token_list):
                continue
            for combo in combinations(token_list, size):
                lhs = frozenset(combo)
                key = (lhs, label)
                if key not in self.rules:
                    self.rules[key] = ProductionRule(lhs=lhs, rhs=label, support=1)
                    self.stats['generated'] += 1
                else:
                    self.rules[key].support += 1
    
    def _eliminate_contradicted(self, board: str, label: int):
        tokens = self._board_to_tokens(board)
        eliminated = 0
        
        for key, rule in list(self.rules.items()):
            if rule.alive and rule.matches(tokens) and rule.rhs != label:
                rule.alive = False
                eliminated += 1
        
        self.stats['eliminated'] += eliminated
        if eliminated > 500:
            self.rules = {k: v for k, v in self.rules.items() if v.alive}
    
    def predict(self, observation: str) -> int:
        self.stats['predictions'] += 1
        
        if self.board_size is None:
            self.board_size = len(observation)
        for char in observation:
            self.observed_values.add(char)
        
        tokens = self._board_to_tokens(observation)
        matching = [r for r in self.rules.values() if r.alive and r.matches(tokens)]
        
        if not matching:
            return self._prior_predict()
        
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts.get(l, 0) / total for l in range(self.num_outputs)}
        
        # Group by specificity
        by_spec: Dict[int, List[ProductionRule]] = defaultdict(list)
        for rule in matching:
            by_spec[rule.specificity()].append(rule)
        
        # Use most specific
        for spec in sorted(by_spec.keys(), reverse=True):
            rules = by_spec[spec]
            votes: Dict[int, float] = defaultdict(float)
            for rule in rules:
                boost = 1.0 / (label_freq.get(rule.rhs, 0.1) + 0.01)
                votes[rule.rhs] += rule.support * boost
            
            if votes:
                best = max(votes, key=votes.get)
                best_rules = [r for r in rules if r.rhs == best]
                if best_rules and max(r.support for r in best_rules) >= 2:
                    return best
        
        # Fall back
        votes: Dict[int, float] = defaultdict(float)
        for rule in matching:
            boost = 1.0 / (label_freq.get(rule.rhs, 0.1) + 0.01)
            votes[rule.rhs] += rule.specificity() * rule.support * boost
        
        return max(votes, key=votes.get) if votes else self._prior_predict()
    
    def _prior_predict(self) -> int:
        if not self.label_counts:
            return random.randint(0, self.num_outputs - 1)
        total = sum(self.label_counts.values())
        r = random.random() * total
        cumsum = 0
        for label, count in sorted(self.label_counts.items(), key=lambda x: -x[1]):
            cumsum += count
            if r <= cumsum:
                return label
        return max(self.label_counts, key=self.label_counts.get)
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        self.history.append((observation, guess, correct_label))
        self.observations.append((observation, correct_label))
        self.label_counts[correct_label] += 1
        
        if self.board_size is None:
            self.board_size = len(observation)
        for char in observation:
            self.observed_values.add(char)
        
        self._eliminate_contradicted(observation, correct_label)
        self._generate_rules(observation, correct_label)
    
    def get_stats(self) -> Dict[str, Any]:
        alive = sum(1 for r in self.rules.values() if r.alive)
        return {**self.stats, 'rules': alive, 'observations': len(self.observations)}
    
    def describe_knowledge(self) -> str:
        lines = ["=== Production Rule Learner ===\n"]
        alive = [r for r in self.rules.values() if r.alive]
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Alive rules: {len(alive)}")
        
        lines.append("\n--- Rules by Label ---")
        by_label: Dict[int, List[ProductionRule]] = defaultdict(list)
        for rule in alive:
            by_label[rule.rhs].append(rule)
        
        for label_idx in range(self.num_outputs):
            name = self.label_names[label_idx] if self.label_names else str(label_idx)
            rules = sorted(by_label.get(label_idx, []), key=lambda r: (-r.specificity(), -r.support))
            lines.append(f"\n  {name}: {len(rules)} rules")
            for rule in rules[:5]:
                lhs_str = ' AND '.join(str(t) for t in sorted(rule.lhs, key=str))
                lines.append(f"    IF {lhs_str} (sup={rule.support})")
        
        return '\n'.join(lines)


class EnhancedProductionLearner(ProductionRuleLearner):
    """Enhanced with multi-size patterns and pure rule tracking."""
    
    def __init__(self, **kwargs):
        kwargs['pattern_sizes'] = [2, 3, 4]
        super().__init__(**kwargs)
        self.pure_rules: Dict[int, List[ProductionRule]] = defaultdict(list)
    
    def _update_pure_rules(self):
        self.pure_rules.clear()
        for rule in self.rules.values():
            if rule.alive and rule.support >= 2:
                self.pure_rules[rule.rhs].append(rule)
        for label in self.pure_rules:
            self.pure_rules[label].sort(key=lambda r: (-r.specificity(), -r.support))
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        if len(self.observations) % 10 == 0:
            self._update_pure_rules()
    
    def predict(self, observation: str) -> int:
        self.stats['predictions'] += 1
        
        if self.board_size is None:
            self.board_size = len(observation)
        for char in observation:
            self.observed_values.add(char)
        
        tokens = self._board_to_tokens(observation)
        
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts.get(l, 0) / total for l in range(self.num_outputs)}
        
        # Check pure rules first, rarest labels first
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for rule in self.pure_rules.get(label, []):
                if rule.matches(tokens):
                    return label
        
        # Fall back to weighted vote
        matching = [r for r in self.rules.values() if r.alive and r.matches(tokens)]
        if not matching:
            return self._prior_predict()
        
        votes: Dict[int, float] = defaultdict(float)
        for rule in matching:
            boost = 1.0 / (label_freq.get(rule.rhs, 0.1) + 0.01)
            votes[rule.rhs] += (rule.specificity() ** 2) * rule.support * boost
        
        return max(votes, key=votes.get) if votes else self._prior_predict()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    from test_harness import test_learner
    
    print("="*70)
    print("Testing Production Rule Learner")
    print("="*70)
    
    result = test_learner(
        ProductionRuleLearner, tictactoe, random_board, label_space,
        rounds=500, verbose=True
    )
    
    print(f"\nFinal Accuracy: {result['final_accuracy']:.1%}")
    print("\nPer-Label Accuracy:")
    for label, acc in result['per_label_accuracy'].items():
        count = result['per_label_counts'][label]
        print(f"  {label:8s}: {acc:.1%} ({count} samples)")
    
    print("\n" + result['learner'].describe_knowledge())
    
    print("\n" + "="*70)
    print("Testing Enhanced Production Learner")
    print("="*70)
    
    result2 = test_learner(
        EnhancedProductionLearner, tictactoe, random_board, label_space,
        rounds=500, verbose=True
    )
    
    print(f"\nFinal Accuracy: {result2['final_accuracy']:.1%}")
    print("\nPer-Label Accuracy:")
    for label, acc in result2['per_label_accuracy'].items():
        count = result2['per_label_counts'][label]
        print(f"  {label:8s}: {acc:.1%} ({count} samples)")
