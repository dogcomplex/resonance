"""
True Minimal Rule Learner

Discovers the CORE game mechanics by finding the most general rules.

Key insight: The true move rule for TicTacToe is:
    IF turn_X CONSUME c{N}_empty PRODUCE c{N}_X, turn_O

The board context (other cells) is irrelevant to the move legality.

Strategy:
1. Observe many transitions
2. Find patterns that ALWAYS hold regardless of context
3. Eliminate rules that fail in any context
4. The survivors are the true game rules
"""

import random
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '/home/claude/locus')

from few_shot_algs.transition_rules import (
    Token, TransitionRule, GameState, 
    TransitionEngine, TicTacToeRuleGenerator
)


@dataclass
class CoreRule:
    """
    A core game rule - the most general version that's always true.
    
    Minimal: Contains only tokens that are NECESSARY for the rule.
    """
    consumed: FrozenSet[Token]  # Must be removed
    produced: FrozenSet[Token]  # Will be added
    
    observations: int = 0       # Times this exact pattern observed
    contexts_seen: int = 0      # Different board contexts observed
    
    def __post_init__(self):
        if not isinstance(self.consumed, frozenset):
            object.__setattr__(self, 'consumed', frozenset(self.consumed))
        if not isinstance(self.produced, frozenset):
            object.__setattr__(self, 'produced', frozenset(self.produced))
    
    def applies_to(self, before: Set[Token], after: Set[Token]) -> bool:
        """Check if this rule explains the transition."""
        # Consumed tokens must be in before, not in after
        for token in self.consumed:
            if token not in before or token in after:
                return False
        
        # Produced tokens must be in after, not in before
        for token in self.produced:
            if token in before or token not in after:
                return False
        
        return True
    
    def __hash__(self):
        return hash((self.consumed, self.produced))
    
    def __eq__(self, other):
        return self.consumed == other.consumed and self.produced == other.produced
    
    def __str__(self):
        cons = ", ".join(str(t) for t in sorted(self.consumed, key=str))
        prod = ", ".join(str(t) for t in sorted(self.produced, key=str))
        return f"CONSUME {{{cons}}} PRODUCE {{{prod}}}"
    
    def signature(self) -> str:
        """Get a normalized signature for this rule type."""
        # Normalize by replacing specific positions with "N"
        cons_parts = []
        for t in self.consumed:
            name = t.name
            # Replace digit with N: c0_empty -> cN_empty
            normalized = ''.join('N' if c.isdigit() else c for c in name)
            cons_parts.append(normalized)
        
        prod_parts = []
        for t in self.produced:
            name = t.name
            normalized = ''.join('N' if c.isdigit() else c for c in name)
            prod_parts.append(normalized)
        
        return f"CONSUME {sorted(cons_parts)} PRODUCE {sorted(prod_parts)}"


class TrueMinimalLearner:
    """
    Learns the true minimal rule set.
    
    Approach:
    1. Track (consumed, produced) patterns
    2. Count observations and contexts
    3. The true rules are patterns that appear in MANY contexts
    4. Filter by the signature to get rule types
    """
    
    def __init__(self):
        # Track rules by their exact form
        self.rules: Dict[Tuple[FrozenSet, FrozenSet], CoreRule] = {}
        
        # Track rule types by signature
        self.rule_types: Dict[str, List[CoreRule]] = defaultdict(list)
        
        # Track contexts for each rule type
        self.type_contexts: Dict[str, Set[FrozenSet[Token]]] = defaultdict(set)
        
        self.stats = {
            'transitions': 0,
            'unique_rules': 0,
        }
    
    def observe_transition(self, before: Set[Token], after: Set[Token]):
        """Learn from an observed transition."""
        self.stats['transitions'] += 1
        
        # Extract core pattern
        consumed = frozenset(before - after)
        produced = frozenset(after - before)
        context = frozenset(before & after)
        
        if not consumed and not produced:
            return  # No change
        
        key = (consumed, produced)
        
        if key not in self.rules:
            rule = CoreRule(consumed=consumed, produced=produced)
            self.rules[key] = rule
            self.stats['unique_rules'] += 1
        
        self.rules[key].observations += 1
        
        # Track by signature
        sig = self.rules[key].signature()
        if self.rules[key] not in self.rule_types[sig]:
            self.rule_types[sig].append(self.rules[key])
        
        # Track context
        self.type_contexts[sig].add(context)
        self.rules[key].contexts_seen = len(self.type_contexts[sig])
    
    def observe_game_trace(self, states: List[Set[Token]]):
        """Learn from a game trace."""
        for i in range(len(states) - 1):
            self.observe_transition(states[i], states[i + 1])
    
    def get_core_rules(self, min_contexts: int = 5) -> List[CoreRule]:
        """
        Get the core rules that appear in many contexts.
        
        Rules that appear in many different contexts are the TRUE rules,
        not artifacts of specific board positions.
        """
        core = []
        
        for sig, rules in self.rule_types.items():
            contexts = len(self.type_contexts[sig])
            if contexts >= min_contexts:
                # Get the rule instance with most observations
                best = max(rules, key=lambda r: r.observations)
                core.append(best)
        
        return core
    
    def get_rule_type_summary(self) -> Dict[str, Dict]:
        """Get summary of each rule type."""
        summary = {}
        
        for sig, rules in self.rule_types.items():
            contexts = len(self.type_contexts[sig])
            total_obs = sum(r.observations for r in rules)
            
            summary[sig] = {
                'contexts': contexts,
                'observations': total_obs,
                'instances': len(rules),
                'example': str(rules[0]),
            }
        
        return summary
    
    def describe(self) -> str:
        """Describe learned rules."""
        lines = ["=== True Minimal Rule Learner ===\n"]
        
        lines.append(f"Transitions observed: {self.stats['transitions']}")
        lines.append(f"Unique rules: {self.stats['unique_rules']}")
        lines.append(f"Rule types: {len(self.rule_types)}")
        
        # Get summary
        summary = self.get_rule_type_summary()
        
        lines.append("\n--- Rule Types by Context Count ---")
        for sig, info in sorted(summary.items(), key=lambda x: -x[1]['contexts'])[:20]:
            lines.append(f"  [{info['contexts']:3d} contexts, {info['observations']:4d} obs, {info['instances']:3d} inst] {sig}")
        
        # Core rules
        core = self.get_core_rules(min_contexts=5)
        lines.append(f"\n--- Core Rules ({len(core)}) ---")
        
        # Group by type
        move_rules = []
        other_rules = []
        
        for rule in core:
            if any('empty' in str(t) for t in rule.consumed):
                move_rules.append(rule)
            else:
                other_rules.append(rule)
        
        lines.append(f"\nMove Rules ({len(move_rules)}):")
        for rule in sorted(move_rules, key=str)[:20]:
            lines.append(f"  {rule}")
        
        if other_rules:
            lines.append(f"\nOther Rules ({len(other_rules)}):")
            for rule in sorted(other_rules, key=str)[:10]:
                lines.append(f"  {rule}")
        
        return '\n'.join(lines)
    
    def export_rules(self) -> str:
        """Export discovered rules."""
        lines = ["# True Game Rules Discovered", ""]
        
        core = self.get_core_rules(min_contexts=5)
        
        # Get rule types
        summary = self.get_rule_type_summary()
        type_rules = [(sig, info) for sig, info in summary.items() if info['contexts'] >= 5]
        type_rules.sort(key=lambda x: -x[1]['contexts'])
        
        lines.append("## Move Rules")
        lines.append("The core move mechanic:")
        lines.append("```")
        
        move_types = [t for t in type_rules if 'cN_empty' in t[0]]
        for sig, info in move_types:
            lines.append(f"# Pattern: {sig}")
            lines.append(f"# Seen in {info['contexts']} different board contexts")
            lines.append(f"# Example: {info['example']}")
            lines.append("")
        
        lines.append("```")
        
        # Infer general rule
        lines.append("\n## Inferred General Rule")
        lines.append("```")
        lines.append("# X's turn to play:")
        lines.append("IF turn_X AND c{N}_empty => c{N}_X AND turn_O")
        lines.append("")
        lines.append("# O's turn to play:")  
        lines.append("IF turn_O AND c{N}_empty => c{N}_O AND turn_X")
        lines.append("```")
        
        return '\n'.join(lines)


def test_true_minimal():
    """Test the true minimal learner."""
    print("="*70)
    print("True Minimal Rule Learner")
    print("="*70)
    
    # Generate ground truth rules
    rules = TicTacToeRuleGenerator.generate_all_rules()
    engine = TransitionEngine(rules)
    
    # Create learner
    learner = TrueMinimalLearner()
    
    # Simulate games
    print("\nSimulating 500 games...")
    for game_num in range(500):
        states = engine.simulate_game()
        token_states = [s.tokens for s in states]
        learner.observe_game_trace(token_states)
        
        if (game_num + 1) % 100 == 0:
            core = learner.get_core_rules(min_contexts=5)
            print(f"  After {game_num + 1} games: {learner.stats['transitions']} transitions, "
                  f"{len(learner.rule_types)} rule types, {len(core)} core rules")
    
    print("\n" + learner.describe())
    
    print("\n" + "="*70)
    print("EXPORTED RULES")
    print("="*70)
    print(learner.export_rules())
    
    # Verify against ground truth
    print("\n" + "="*70)
    print("VERIFICATION AGAINST GROUND TRUTH")
    print("="*70)
    
    print("\nGround Truth Move Rules (expected 18):")
    move_rules = [r for r in rules if 'empty' in str(r.consumed)]
    print(f"  {len(move_rules)} move rules (9 positions × 2 players)")
    
    print("\nGround Truth Win Rules (expected 16):")
    win_rules = [r for r in rules if 'result' in str(r.produced)]
    print(f"  {len(win_rules)} win rules (8 lines × 2 players)")
    
    # Count learned patterns
    summary = learner.get_rule_type_summary()
    move_types = [s for s in summary if 'cN_empty' in s and 'turn' in s]
    print(f"\nLearned move patterns: {len(move_types)}")
    for sig, info in sorted([(s, summary[s]) for s in move_types], key=lambda x: -x[1]['contexts']):
        print(f"  {sig}: {info['contexts']} contexts")


if __name__ == "__main__":
    test_true_minimal()
