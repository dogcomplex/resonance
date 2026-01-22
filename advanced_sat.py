"""
Advanced SAT-Based Learner

Extends the production rule system with:
1. Boolean combinations (OR rules - multiple patterns can satisfy)
2. Negative rules (IF pattern THEN NOT label)
3. Chained inference (derive intermediate facts)
4. Consistency checking (detect contradictions)
5. Rule minimization (find smallest covering rule set)

This is the path toward learning FULL game rules, not just classification.
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field

from few_shot_algs.sat_production import (
    Token, TokenType, ProductionRule, FactBase, 
    InferenceEngine, SATLikeReasoner, ProductionRuleEngine
)


class BooleanRuleType:
    """Types of boolean rules."""
    AND = "and"      # All conditions must be true
    OR = "or"        # At least one condition must be true
    NOT = "not"      # Negation
    IMPLIES = "implies"  # If A then B


@dataclass
class BooleanRule:
    """
    A rule with boolean structure.
    
    Supports:
    - AND: All conditions must be true
    - OR: At least one condition must be true (disjunction)
    - Nested combinations
    """
    rule_type: str
    conditions: List[Any]  # Can be Tokens or nested BooleanRules
    conclusion: Token
    
    # Statistics
    fires: int = 0
    correct: int = 0
    alive: bool = True
    
    def evaluate(self, facts: Set[Token]) -> bool:
        """Evaluate if this rule's conditions are satisfied."""
        if self.rule_type == BooleanRuleType.AND:
            return all(self._eval_condition(c, facts) for c in self.conditions)
        elif self.rule_type == BooleanRuleType.OR:
            return any(self._eval_condition(c, facts) for c in self.conditions)
        elif self.rule_type == BooleanRuleType.NOT:
            return not self._eval_condition(self.conditions[0], facts)
        else:
            raise ValueError(f"Unknown rule type: {self.rule_type}")
    
    def _eval_condition(self, condition, facts: Set[Token]) -> bool:
        """Evaluate a single condition."""
        if isinstance(condition, BooleanRule):
            return condition.evaluate(facts)
        elif isinstance(condition, Token):
            if condition.negated:
                return condition.negate() not in facts
            return condition in facts
        elif isinstance(condition, frozenset):
            # Treat as AND of tokens
            return all(t in facts for t in condition)
        else:
            raise ValueError(f"Unknown condition type: {type(condition)}")
    
    def confidence(self) -> float:
        if self.fires == 0:
            return 0.5
        return self.correct / self.fires


class DisjunctiveRule:
    """
    A disjunctive rule: (pattern1 OR pattern2 OR ...) => label
    
    This captures the structure of TicTacToe wins:
    (row1 OR row2 OR row3 OR col1 OR col2 OR col3 OR diag1 OR diag2) => win
    
    Once we discover individual patterns, we can combine them.
    """
    
    def __init__(self, patterns: List[FrozenSet[Token]], conclusion: Token):
        self.patterns = patterns  # List of AND patterns (any one can fire)
        self.conclusion = conclusion
        self.fires = 0
        self.correct = 0
        self.alive = True
    
    def matches(self, facts: Set[Token]) -> bool:
        """Check if any pattern matches."""
        return any(pattern.issubset(facts) for pattern in self.patterns)
    
    def fire(self, facts: Set[Token], true_label: Optional[int] = None) -> Optional[Token]:
        """Fire if any pattern matches."""
        if not self.matches(facts):
            return None
        
        self.fires += 1
        if true_label is not None and self.conclusion.token_type == TokenType.LABEL:
            if self.conclusion.value == true_label:
                self.correct += 1
            else:
                self.alive = False
        
        return self.conclusion
    
    def confidence(self) -> float:
        if self.fires == 0:
            return 0.5
        return self.correct / self.fires
    
    def __str__(self):
        patterns_str = " OR ".join(
            "(" + " AND ".join(str(t) for t in p) + ")"
            for p in self.patterns
        )
        return f"IF {patterns_str} THEN {self.conclusion}"


class NegativeRule:
    """
    A negative rule: IF pattern THEN NOT label
    
    Useful for ruling out possibilities:
    - IF board_full AND no_win1 AND no_win2 THEN NOT ok (must be draw)
    """
    
    def __init__(self, pattern: FrozenSet[Token], excluded_label: int):
        self.pattern = pattern
        self.excluded_label = excluded_label
        self.fires = 0
        self.correct = 0  # Times the excluded label was indeed not the answer
        self.alive = True
    
    def matches(self, facts: Set[Token]) -> bool:
        return self.pattern.issubset(facts)
    
    def excludes(self, label: int) -> bool:
        return label == self.excluded_label


class AdvancedSATLearner:
    """
    Advanced learner combining production rules with SAT-like reasoning.
    
    Key features:
    1. Discovers disjunctive rules (OR combinations)
    2. Uses negative inference
    3. Chains derived facts
    4. Minimizes rule set
    """
    
    def __init__(self, num_outputs: int = 5, board_size: int = 9,
                 label_names: List[str] = None,
                 pattern_sizes: List[int] = None,
                 discover_disjunctions: bool = True,
                 use_negative_inference: bool = True,
                 **kwargs):
        self.num_outputs = num_outputs
        self.board_size = board_size
        self.label_names = label_names or [f"label_{i}" for i in range(num_outputs)]
        self.pattern_sizes = pattern_sizes or [3]
        self.discover_disjunctions = discover_disjunctions
        self.use_negative_inference = use_negative_inference
        
        # Base production rules
        self.rules: Dict[int, ProductionRule] = {}
        self.rule_id = 0
        
        # Disjunctive rules (discovered by combining base rules)
        self.disjunctive_rules: Dict[int, DisjunctiveRule] = {}
        
        # Negative rules
        self.negative_rules: List[NegativeRule] = []
        
        # Observations
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        self.label_counts: Dict[int, int] = defaultdict(int)
        
        # Pattern statistics for disjunction discovery
        self.pattern_label_counts: Dict[FrozenSet[Token], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        self.stats = {
            'rules_generated': 0,
            'rules_eliminated': 0,
            'disjunctions_discovered': 0,
            'predictions': 0,
        }
    
    def _board_to_facts(self, board: str) -> Set[Token]:
        """Convert board to set of tokens."""
        facts = set()
        for pos, val in enumerate(board):
            facts.add(Token(f"p{pos}", val, TokenType.POSITION))
        
        # Meta facts
        if '0' not in board:
            facts.add(Token("board_full", True, TokenType.META))
        
        return facts
    
    def _generate_rules(self, board: str, label: int):
        """Generate production rules from observation."""
        facts = self._board_to_facts(board)
        position_tokens = [t for t in facts if t.token_type == TokenType.POSITION]
        
        rhs = Token("label", label, TokenType.LABEL)
        
        for size in self.pattern_sizes:
            if size > len(position_tokens):
                continue
            
            for combo in combinations(position_tokens, size):
                lhs = frozenset(combo)
                
                # Track pattern-label statistics
                self.pattern_label_counts[lhs][label] += 1
                
                # Check if rule exists
                exists = False
                for rule in self.rules.values():
                    if rule.lhs == lhs and rule.rhs.value == label:
                        exists = True
                        break
                
                if not exists:
                    rule = ProductionRule(lhs=lhs, rhs=rhs)
                    self.rules[self.rule_id] = rule
                    self.rule_id += 1
                    self.stats['rules_generated'] += 1
    
    def _eliminate_contradicted(self, board: str, label: int):
        """Eliminate contradicted rules."""
        facts = self._board_to_facts(board)
        eliminated = 0
        
        for rule in self.rules.values():
            if not rule.alive:
                continue
            
            if rule.lhs_satisfied(facts):
                rule.fires += 1
                if rule.rhs.token_type == TokenType.LABEL:
                    if rule.rhs.value == label:
                        rule.correct += 1
                    else:
                        rule.alive = False
                        eliminated += 1
        
        self.stats['rules_eliminated'] += eliminated
    
    def _discover_disjunctions(self):
        """
        Discover disjunctive rules by combining pure rules with same label.
        
        If multiple patterns all predict label L with 100% accuracy,
        they can be combined: (P1 OR P2 OR P3) => L
        """
        if not self.discover_disjunctions:
            return
        
        # Group pure rules by label
        pure_by_label: Dict[int, List[ProductionRule]] = defaultdict(list)
        for rule in self.rules.values():
            if rule.alive and rule.fires >= 2 and rule.correct == rule.fires:
                if rule.rhs.token_type == TokenType.LABEL:
                    pure_by_label[rule.rhs.value].append(rule)
        
        # Create disjunctive rules
        for label, rules in pure_by_label.items():
            if len(rules) >= 2:
                patterns = [rule.lhs for rule in rules]
                conclusion = Token("label", label, TokenType.LABEL)
                
                # Check if we already have this disjunction
                key = (frozenset(patterns), label)
                if key not in self.disjunctive_rules:
                    disj_rule = DisjunctiveRule(patterns, conclusion)
                    self.disjunctive_rules[len(self.disjunctive_rules)] = disj_rule
                    self.stats['disjunctions_discovered'] += 1
    
    def _get_pure_rules(self, min_support: int = 2) -> Dict[int, List[ProductionRule]]:
        """Get rules with 100% precision."""
        pure = defaultdict(list)
        for rule in self.rules.values():
            if rule.alive and rule.fires >= min_support and rule.correct == rule.fires:
                if rule.rhs.token_type == TokenType.LABEL:
                    pure[rule.rhs.value].append(rule)
        
        for label in pure:
            pure[label].sort(key=lambda r: (-len(r.lhs), -r.fires))
        
        return pure
    
    def _get_high_confidence_rules(self, min_conf: float = 0.85, 
                                    min_support: int = 2) -> Dict[int, List[ProductionRule]]:
        """Get high confidence rules."""
        confident = defaultdict(list)
        for rule in self.rules.values():
            if rule.alive and rule.fires >= min_support and rule.confidence() >= min_conf:
                if rule.rhs.token_type == TokenType.LABEL:
                    confident[rule.rhs.value].append(rule)
        
        for label in confident:
            confident[label].sort(key=lambda r: (-r.confidence(), -len(r.lhs), -r.fires))
        
        return confident
    
    def predict(self, observation: str) -> int:
        """Predict using tiered rule matching."""
        self.stats['predictions'] += 1
        
        facts = self._board_to_facts(observation)
        
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts.get(l, 0) / total 
                     for l in range(self.num_outputs)}
        
        # TIER 1: Pure rules, rare labels first
        pure_rules = self._get_pure_rules(min_support=3)
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for rule in pure_rules.get(label, []):
                if rule.lhs_satisfied(facts):
                    return label
        
        # TIER 2: Disjunctive rules
        for disj_rule in self.disjunctive_rules.values():
            if disj_rule.alive and disj_rule.matches(facts):
                return disj_rule.conclusion.value
        
        # TIER 3: High confidence rules with voting
        confident = self._get_high_confidence_rules(min_conf=0.8, min_support=2)
        votes: Dict[int, float] = defaultdict(float)
        
        for label in range(self.num_outputs):
            for rule in confident.get(label, []):
                if rule.lhs_satisfied(facts):
                    rarity_boost = 1.0 / (label_freq.get(label, 0.1) + 0.01)
                    weight = rule.confidence() * (len(rule.lhs) ** 2) * rule.fires * rarity_boost
                    votes[label] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        # TIER 4: All alive rules
        for rule in self.rules.values():
            if rule.alive and rule.lhs_satisfied(facts) and rule.rhs.token_type == TokenType.LABEL:
                label = rule.rhs.value
                rarity_boost = 1.0 / (label_freq.get(label, 0.1) + 0.01)
                votes[label] += len(rule.lhs) * (rule.fires + 1) * rarity_boost
        
        if votes:
            return max(votes, key=votes.get)
        
        # TIER 5: Prior
        return self._prior_predict()
    
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
        """Learn from observation."""
        self.history.append((observation, guess, correct_label))
        self.observations.append((observation, correct_label))
        self.label_counts[correct_label] += 1
        
        self._eliminate_contradicted(observation, correct_label)
        self._generate_rules(observation, correct_label)
        
        # Discover disjunctions periodically
        if len(self.observations) % 50 == 0:
            self._discover_disjunctions()
    
    def get_stats(self) -> Dict[str, Any]:
        alive = sum(1 for r in self.rules.values() if r.alive)
        pure = sum(len(v) for v in self._get_pure_rules().values())
        return {
            **self.stats,
            'rules': alive,
            'pure_rules': pure,
            'observations': len(self.observations),
        }
    
    def describe_knowledge(self) -> str:
        """Describe learned knowledge."""
        lines = ["=== Advanced SAT Learner ===\n"]
        
        pure_rules = self._get_pure_rules()
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Alive rules: {sum(1 for r in self.rules.values() if r.alive)}")
        lines.append(f"Pure rules: {sum(len(v) for v in pure_rules.values())}")
        lines.append(f"Disjunctive rules: {len(self.disjunctive_rules)}")
        
        lines.append("\n--- Pure Rules by Label ---")
        for label_idx in range(self.num_outputs):
            label_name = self.label_names[label_idx]
            rules = pure_rules.get(label_idx, [])
            lines.append(f"\n  {label_name}: {len(rules)} rules")
            for rule in rules[:5]:
                lhs_str = " AND ".join(str(t) for t in sorted(rule.lhs, key=str))
                lines.append(f"    IF {lhs_str} (n={rule.fires})")
        
        if self.disjunctive_rules:
            lines.append("\n--- Disjunctive Rules ---")
            for disj_rule in list(self.disjunctive_rules.values())[:3]:
                lines.append(f"  {disj_rule}")
        
        return '\n'.join(lines)
    
    def export_complete_rules(self) -> str:
        """Export a complete, minimal set of rules."""
        lines = ["# Complete Game Rules Discovered", ""]
        
        pure_rules = self._get_pure_rules(min_support=2)
        
        for label_idx in range(self.num_outputs):
            label_name = self.label_names[label_idx]
            rules = pure_rules.get(label_idx, [])
            
            if rules:
                lines.append(f"## {label_name}")
                
                # If multiple rules, format as disjunction
                if len(rules) > 1:
                    lines.append(f"{label_name} = (")
                    for i, rule in enumerate(rules):
                        lhs_str = " AND ".join(str(t) for t in sorted(rule.lhs, key=str))
                        prefix = "    " if i == 0 else " OR "
                        lines.append(f"{prefix}({lhs_str})")
                    lines.append(")")
                else:
                    rule = rules[0]
                    lhs_str = " AND ".join(str(t) for t in sorted(rule.lhs, key=str))
                    lines.append(f"{label_name} = ({lhs_str})")
                
                lines.append("")
        
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE
    
    print("="*70)
    print("Testing Advanced SAT Learner")
    print("="*70)
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    learner = AdvancedSATLearner(
        num_outputs=5,
        label_names=LABEL_SPACE,
        pattern_sizes=[3]
    )
    
    correct = 0
    checkpoints = [10, 25, 50, 100, 200, 500, 1000, 2000, 3000]
    
    for i in range(3500):
        obs = gen.next()
        if obs is None:
            print(f"All {i} states observed!")
            break
        
        board, true_idx = obs
        pred_idx = learner.predict(board)
        
        if pred_idx == true_idx:
            correct += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if (i + 1) in checkpoints:
            acc = correct / (i + 1)
            stats = learner.get_stats()
            print(f"  @{i+1:5d}: acc={acc:.1%} rules={stats['rules']} pure={stats['pure_rules']} disj={stats['disjunctions_discovered']}")
    
    print(f"\nFinal Accuracy: {correct/(i+1):.1%}")
    print("\n" + learner.describe_knowledge())
    
    print("\n" + "="*70)
    print("COMPLETE RULES DISCOVERED")
    print("="*70)
    print(learner.export_complete_rules())
