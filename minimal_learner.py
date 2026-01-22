"""
Minimal Rule Learner

Discovers the minimal set of transition rules that explain all observations.

Key insight: We want to find rules that are:
1. GENERAL: Apply to many situations (not overfit to specific board states)
2. MINIMAL: Smallest LHS that correctly predicts transition
3. CONSISTENT: Never contradicted by observations

Algorithm:
1. Observe (before, after) state pairs
2. Extract candidate rules with varying specificity
3. Eliminate rules contradicted by observations
4. Merge similar rules into general patterns
5. Output minimal covering set
"""

import random
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/claude/locus')

from few_shot_algs.transition_rules import (
    Token, TokenType, TransitionRule, GameState, 
    TransitionEngine, TicTacToeRuleGenerator
)


@dataclass
class RuleCandidate:
    """A candidate rule with statistics."""
    lhs: FrozenSet[Token]
    rhs: FrozenSet[Token]
    support: int = 0           # Times this exact transition observed
    generalizations: int = 0   # Times a more specific version observed
    contradictions: int = 0    # Times LHS matched but RHS differed
    
    def __post_init__(self):
        if not isinstance(self.lhs, frozenset):
            object.__setattr__(self, 'lhs', frozenset(self.lhs))
        if not isinstance(self.rhs, frozenset):
            object.__setattr__(self, 'rhs', frozenset(self.rhs))
    
    @property
    def consumed(self) -> FrozenSet[Token]:
        return self.lhs - self.rhs
    
    @property
    def produced(self) -> FrozenSet[Token]:
        return self.rhs - self.lhs
    
    @property
    def catalysts(self) -> FrozenSet[Token]:
        return self.lhs & self.rhs
    
    @property
    def specificity(self) -> int:
        return len(self.lhs)
    
    @property
    def is_valid(self) -> bool:
        return self.contradictions == 0 and self.support > 0
    
    def confidence(self) -> float:
        total = self.support + self.contradictions
        return self.support / total if total > 0 else 0.0
    
    def matches_lhs(self, state: Set[Token]) -> bool:
        return self.lhs.issubset(state)
    
    def is_consistent_with(self, before: Set[Token], after: Set[Token]) -> bool:
        """Check if this rule is consistent with an observed transition."""
        if not self.matches_lhs(before):
            return True  # Doesn't apply, so not inconsistent
        
        # Check that what we expect to be produced IS produced
        expected_after = (before - self.consumed) | self.produced
        
        # The observed after should contain our expected changes
        return self.produced.issubset(after) and not self.consumed.intersection(after)
    
    def __hash__(self):
        return hash((self.lhs, self.rhs))
    
    def __eq__(self, other):
        return self.lhs == other.lhs and self.rhs == other.rhs
    
    def __str__(self):
        parts = []
        if self.catalysts:
            parts.append(f"IF {{{', '.join(str(t) for t in sorted(self.catalysts, key=str))}}}")
        if self.consumed:
            parts.append(f"CONSUME {{{', '.join(str(t) for t in sorted(self.consumed, key=str))}}}")
        if self.produced:
            parts.append(f"PRODUCE {{{', '.join(str(t) for t in sorted(self.produced, key=str))}}}")
        return " ".join(parts) if parts else "NOOP"


class MinimalRuleLearner:
    """
    Learns minimal transition rules from game traces.
    
    Strategy:
    1. Start with most general rules (smallest LHS)
    2. Add specificity only when needed to avoid contradictions
    3. Merge rules that always fire together
    """
    
    def __init__(self, min_lhs_size: int = 2, max_lhs_size: int = 4):
        self.min_lhs_size = min_lhs_size
        self.max_lhs_size = max_lhs_size
        
        # Observed transitions
        self.transitions: List[Tuple[Set[Token], Set[Token]]] = []
        
        # Candidate rules
        self.candidates: Dict[Tuple[FrozenSet, FrozenSet], RuleCandidate] = {}
        
        # Confirmed minimal rules
        self.confirmed_rules: List[RuleCandidate] = []
        
        # Statistics
        self.stats = {
            'transitions_observed': 0,
            'candidates_generated': 0,
            'candidates_eliminated': 0,
        }
    
    def _extract_essential_tokens(self, before: Set[Token], after: Set[Token]) -> Tuple[Set[Token], Set[Token]]:
        """
        Extract the essential tokens that changed.
        
        Consumed: in before but not after
        Produced: in after but not before
        """
        consumed = before - after
        produced = after - before
        return consumed, produced
    
    def _generate_candidate_rules(self, before: Set[Token], after: Set[Token]):
        """
        Generate candidate rules of varying specificity.
        
        Start with minimal LHS (just consumed + turn token) and add catalysts.
        """
        consumed, produced = self._extract_essential_tokens(before, after)
        
        if not consumed and not produced:
            return  # No change, skip
        
        # Get potential catalysts (tokens in both before and after)
        potential_catalysts = before & after
        
        # Filter to relevant catalysts (cell tokens and turn tokens)
        relevant_catalysts = {t for t in potential_catalysts 
                            if t.name.startswith('c') or t.name.startswith('turn')}
        
        # Generate rules with increasing catalyst sets
        for num_catalysts in range(0, min(len(relevant_catalysts) + 1, self.max_lhs_size)):
            for catalyst_combo in combinations(relevant_catalysts, num_catalysts):
                lhs = consumed | frozenset(catalyst_combo)
                rhs = produced | frozenset(catalyst_combo)
                
                if len(lhs) < self.min_lhs_size:
                    continue
                if len(lhs) > self.max_lhs_size:
                    continue
                
                key = (frozenset(lhs), frozenset(rhs))
                
                if key not in self.candidates:
                    self.candidates[key] = RuleCandidate(lhs=lhs, rhs=rhs)
                    self.stats['candidates_generated'] += 1
                
                self.candidates[key].support += 1
    
    def _check_consistency(self, before: Set[Token], after: Set[Token]):
        """Check all candidate rules for consistency with this transition."""
        consumed, produced = self._extract_essential_tokens(before, after)
        
        for candidate in list(self.candidates.values()):
            if candidate.matches_lhs(before):
                # Rule fires - check if it's consistent
                if not candidate.is_consistent_with(before, after):
                    candidate.contradictions += 1
                    if candidate.contradictions > 0:
                        self.stats['candidates_eliminated'] += 1
    
    def observe_transition(self, before: Set[Token], after: Set[Token]):
        """Learn from an observed state transition."""
        self.transitions.append((before.copy(), after.copy()))
        self.stats['transitions_observed'] += 1
        
        # Generate new candidate rules
        self._generate_candidate_rules(before, after)
        
        # Check consistency of all candidates
        self._check_consistency(before, after)
    
    def observe_game_trace(self, states: List[Set[Token]]):
        """Learn from a game trace."""
        for i in range(len(states) - 1):
            self.observe_transition(states[i], states[i + 1])
    
    def _filter_valid_rules(self) -> List[RuleCandidate]:
        """Get rules that haven't been contradicted."""
        return [c for c in self.candidates.values() if c.is_valid]
    
    def _find_minimal_covering_set(self) -> List[RuleCandidate]:
        """
        Find minimal set of rules that cover all transitions.
        
        Greedy approach: Pick rules that cover most uncovered transitions.
        """
        valid_rules = self._filter_valid_rules()
        
        # Group by (consumed, produced) pattern
        pattern_groups: Dict[Tuple, List[RuleCandidate]] = defaultdict(list)
        for rule in valid_rules:
            pattern = (rule.consumed, rule.produced)
            pattern_groups[pattern].append(rule)
        
        minimal_set = []
        
        for pattern, rules in pattern_groups.items():
            # Sort by specificity (prefer more general) then support
            rules.sort(key=lambda r: (r.specificity, -r.support))
            
            # Find most general rule that's still valid
            for rule in rules:
                if rule.is_valid:
                    minimal_set.append(rule)
                    break
        
        return minimal_set
    
    def get_learned_rules(self) -> List[RuleCandidate]:
        """Get the learned minimal rule set."""
        if not self.confirmed_rules:
            self.confirmed_rules = self._find_minimal_covering_set()
        return self.confirmed_rules
    
    def describe_learned_rules(self) -> str:
        """Describe learned rules."""
        lines = ["=== Minimal Rule Learner ===\n"]
        
        lines.append(f"Transitions observed: {self.stats['transitions_observed']}")
        lines.append(f"Candidates generated: {self.stats['candidates_generated']}")
        lines.append(f"Candidates eliminated: {self.stats['candidates_eliminated']}")
        
        valid = self._filter_valid_rules()
        lines.append(f"Valid candidates: {len(valid)}")
        
        minimal = self.get_learned_rules()
        lines.append(f"Minimal rule set: {len(minimal)}")
        
        # Categorize rules
        move_rules = [r for r in minimal if any('empty' in str(t) for t in r.consumed)]
        win_rules = [r for r in minimal if any('result' in str(t) for t in r.produced)]
        other_rules = [r for r in minimal if r not in move_rules and r not in win_rules]
        
        lines.append(f"\n--- Move Rules ({len(move_rules)}) ---")
        for rule in sorted(move_rules, key=lambda r: str(r))[:10]:
            lines.append(f"  {rule}  (support={rule.support})")
        if len(move_rules) > 10:
            lines.append(f"  ... and {len(move_rules) - 10} more")
        
        lines.append(f"\n--- Win Detection Rules ({len(win_rules)}) ---")
        for rule in sorted(win_rules, key=lambda r: str(r))[:10]:
            lines.append(f"  {rule}  (support={rule.support})")
        if len(win_rules) > 10:
            lines.append(f"  ... and {len(win_rules) - 10} more")
        
        if other_rules:
            lines.append(f"\n--- Other Rules ({len(other_rules)}) ---")
            for rule in other_rules[:5]:
                lines.append(f"  {rule}  (support={rule.support})")
        
        return '\n'.join(lines)
    
    def export_rules(self) -> str:
        """Export rules in a clean format."""
        rules = self.get_learned_rules()
        
        lines = ["# Learned Game Rules", ""]
        
        move_rules = [r for r in rules if any('empty' in str(t) for t in r.consumed)]
        win_rules = [r for r in rules if any('result' in str(t) for t in r.produced)]
        
        lines.append("## Move Rules")
        lines.append("```")
        for rule in move_rules:
            lines.append(str(rule))
        lines.append("```")
        
        lines.append("\n## Win Detection Rules")
        lines.append("```")
        for rule in win_rules:
            lines.append(str(rule))
        lines.append("```")
        
        return '\n'.join(lines)


def test_learning():
    """Test the learner on simulated games."""
    print("="*70)
    print("Testing Minimal Rule Learner")
    print("="*70)
    
    # Generate ground truth rules
    rules = TicTacToeRuleGenerator.generate_all_rules()
    engine = TransitionEngine(rules)
    
    # Create learner
    learner = MinimalRuleLearner(min_lhs_size=2, max_lhs_size=5)
    
    # Simulate games and learn
    print("\nSimulating 200 games...")
    for game_num in range(200):
        states = engine.simulate_game()
        token_states = [s.tokens for s in states]
        learner.observe_game_trace(token_states)
        
        if (game_num + 1) % 50 == 0:
            valid = len([c for c in learner.candidates.values() if c.is_valid])
            print(f"  After {game_num + 1} games: {learner.stats['transitions_observed']} transitions, "
                  f"{learner.stats['candidates_generated']} candidates, {valid} valid")
    
    print("\n" + learner.describe_learned_rules())
    
    print("\n" + "="*70)
    print("EXPORTED RULES")
    print("="*70)
    print(learner.export_rules())


def analyze_rule_discovery():
    """Analyze what types of rules are being discovered."""
    print("\n" + "="*70)
    print("Rule Discovery Analysis")
    print("="*70)
    
    rules = TicTacToeRuleGenerator.generate_all_rules()
    engine = TransitionEngine(rules)
    
    learner = MinimalRuleLearner(min_lhs_size=2, max_lhs_size=5)
    
    # Simulate many games
    for _ in range(500):
        states = engine.simulate_game()
        token_states = [s.tokens for s in states]
        learner.observe_game_trace(token_states)
    
    valid_rules = learner._filter_valid_rules()
    
    print(f"\nTotal valid rules: {len(valid_rules)}")
    
    # Analyze patterns
    print("\n--- Pattern Analysis ---")
    
    # Count by consumed pattern
    consumed_patterns = defaultdict(int)
    for rule in valid_rules:
        pattern = frozenset(t.name.split('_')[0] for t in rule.consumed)
        consumed_patterns[str(sorted(pattern))] += 1
    
    print("\nConsumed patterns (type counts):")
    for pattern, count in sorted(consumed_patterns.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pattern}: {count}")
    
    # Count by produced pattern  
    produced_patterns = defaultdict(int)
    for rule in valid_rules:
        pattern = frozenset(t.name.split('_')[0] for t in rule.produced)
        produced_patterns[str(sorted(pattern))] += 1
    
    print("\nProduced patterns (type counts):")
    for pattern, count in sorted(produced_patterns.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pattern}: {count}")
    
    # Find most general move rules
    print("\n--- Most General Move Rules ---")
    move_rules = [r for r in valid_rules if any('empty' in str(t) for t in r.consumed)]
    move_rules.sort(key=lambda r: (r.specificity, -r.support))
    
    for rule in move_rules[:5]:
        print(f"  Specificity {rule.specificity}, Support {rule.support}: {rule}")


if __name__ == "__main__":
    test_learning()
    analyze_rule_discovery()
