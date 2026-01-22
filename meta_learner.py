"""
Meta-Learning: Learner Strategies as Production Rules

The learning PROCESS itself can be expressed as production rules!

Domain tokens (the game):
  p0_0  p0_1  p0_2    # Position 0 has value 0, 1, or 2
  ...
  p8_0  p8_1  p8_2    # Position 8 has value 0, 1, or 2
  label_ok  label_win1  label_win2  label_draw  label_error

Meta tokens (the learner's state):
  pattern_X_seen       # Pattern X has been observed
  pattern_X_pure       # Pattern X has 100% precision for some label
  pattern_X_predicts_Y # Pattern X predicts label Y
  confidence_high      # Overall confidence is high
  observation_count_N  # Number of observations seen
  
Strategy rules (how to learn):
  pattern_new  observation  =>  pattern_seen  stats_updated
  pattern_seen  contradiction  =>  pattern_eliminated
  pattern_pure  rare_label  =>  pattern_priority_high
  
Prediction rules (how to decide):
  input  pattern_pure  =>  output_from_pattern
  input  no_pure_match  =>  voting_mode
  voting_mode  pattern_high_conf  =>  vote_added
  votes_tallied  =>  output_highest
"""

import random
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '/home/claude/locus')

from few_shot_algs.production import Token, Rule, State, RuleEngine


# =============================================================================
# DOMAIN: TicTacToe State Space
# =============================================================================

POSITIONS = list(range(9))
VALUES = ['0', '1', '2']
LABELS = ['ok', 'win1', 'win2', 'draw', 'error']

def board_to_tokens(board: str) -> Set[str]:
    """Convert board string to token set."""
    return {f"p{i}_{board[i]}" for i in range(len(board))}

def tokens_to_board(tokens: Set[str]) -> str:
    """Convert token set back to board string."""
    board = ['0'] * 9
    for t in tokens:
        if t.startswith('p') and '_' in t:
            pos = int(t[1])
            val = t.split('_')[1]
            board[pos] = val
    return ''.join(board)


# =============================================================================
# META-LEARNER: Learning Strategy as Production Rules
# =============================================================================

@dataclass
class PatternStats:
    """Statistics for a pattern."""
    tokens: FrozenSet[str]
    count: int = 0
    label_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def precision(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        return {l: c / self.count for l, c in self.label_counts.items()}
    
    @property
    def best_label(self) -> Tuple[str, float]:
        if not self.label_counts:
            return ('ok', 0.0)
        best = max(self.label_counts.items(), key=lambda x: x[1])
        return (best[0], best[1] / self.count if self.count > 0 else 0.0)
    
    @property
    def is_pure(self) -> bool:
        return self.count >= 2 and self.best_label[1] == 1.0


class MetaLearner:
    """
    A learner whose strategy is expressed as production rules.
    
    Meta-state tokens:
      obs_count_N          - observation count
      pattern_P_count_N    - pattern P seen N times
      pattern_P_pure       - pattern P is pure (100% precision)
      pattern_P_pred_L     - pattern P predicts label L
      phase_explore        - exploration phase
      phase_exploit        - exploitation phase
      
    Strategy rules (firing order matters):
      # Phase transitions
      obs_count_100  phase_explore  =>  obs_count_100  phase_exploit
      
      # Pattern management  
      pattern_new  =>  pattern_tracked
      pattern_tracked  contradiction  =>  pattern_eliminated
      pattern_tracked  pure_check  =>  pattern_pure  (if 100% precision)
      
      # Prediction (priority order)
      input  pattern_pure  rare_label  =>  predict_from_pure
      input  pattern_pure  =>  predict_from_pure
      input  pattern_soft  =>  add_vote
      no_match  =>  predict_prior
    """
    
    def __init__(self, pattern_sizes: List[int] = None):
        self.pattern_sizes = pattern_sizes or [3]
        
        # Pattern statistics
        self.patterns: Dict[FrozenSet[str], PatternStats] = {}
        
        # Label statistics
        self.label_counts: Dict[str, int] = defaultdict(int)
        
        # Meta-state
        self.meta_state = State({
            'phase_explore': 1,
            'obs_count': 0,
        })
        
        # Strategy rules (as production rules!)
        self.strategy_rules = self._build_strategy_rules()
        
        self.stats = {
            'observations': 0,
            'patterns': 0,
            'pure_patterns': 0,
        }
    
    def _build_strategy_rules(self) -> List[Rule]:
        """Build the meta-level strategy rules."""
        rules = []
        
        # Phase transition: explore -> exploit after 100 observations
        # (This is conceptual - actual implementation below)
        
        return rules
    
    def _extract_patterns(self, tokens: Set[str]) -> List[FrozenSet[str]]:
        """Extract patterns from token set."""
        patterns = []
        token_list = sorted(tokens)
        for size in self.pattern_sizes:
            for combo in combinations(token_list, size):
                patterns.append(frozenset(combo))
        return patterns
    
    def _update_meta_state(self):
        """Update meta-state based on current knowledge."""
        self.meta_state.tokens['obs_count'] = self.stats['observations']
        
        # Phase transition rule (expressed as logic)
        if self.stats['observations'] >= 100:
            if 'phase_explore' in self.meta_state.tokens:
                del self.meta_state.tokens['phase_explore']
            self.meta_state.tokens['phase_exploit'] = 1
        
        # Count pure patterns
        pure = sum(1 for p in self.patterns.values() if p.is_pure)
        self.stats['pure_patterns'] = pure
    
    def observe(self, board: str, label: str):
        """
        Observe (board, label) pair.
        
        Strategy rule: observation  =>  patterns_updated  stats_updated
        """
        self.stats['observations'] += 1
        self.label_counts[label] += 1
        
        tokens = board_to_tokens(board)
        patterns = self._extract_patterns(tokens)
        
        for pattern in patterns:
            # Strategy rule: pattern_new  =>  pattern_tracked
            if pattern not in self.patterns:
                self.patterns[pattern] = PatternStats(tokens=pattern)
                self.stats['patterns'] += 1
            
            # Strategy rule: observation  pattern  =>  stats_updated
            self.patterns[pattern].count += 1
            self.patterns[pattern].label_counts[label] += 1
        
        self._update_meta_state()
    
    def predict(self, board: str) -> str:
        """
        Predict label for board.
        
        Strategy rules (in priority order):
          1. pattern_pure  rare_label  =>  predict_label
          2. pattern_pure  =>  predict_label  
          3. pattern_soft  =>  add_vote
          4. no_match  =>  predict_prior
        """
        tokens = board_to_tokens(board)
        patterns = self._extract_patterns(tokens)
        
        # Calculate label priors (for rarity boost)
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts.get(l, 0) / total for l in LABELS}
        
        # RULE 1 & 2: Pure pattern matching (rare labels first)
        pure_matches = []
        for pattern in patterns:
            if pattern in self.patterns and self.patterns[pattern].is_pure:
                label, precision = self.patterns[pattern].best_label
                pure_matches.append((pattern, label, label_freq.get(label, 1.0)))
        
        if pure_matches:
            # Sort by rarity (rarest first)
            pure_matches.sort(key=lambda x: x[2])
            return pure_matches[0][1]
        
        # RULE 3: Soft voting
        votes: Dict[str, float] = defaultdict(float)
        for pattern in patterns:
            if pattern not in self.patterns:
                continue
            
            p = self.patterns[pattern]
            if p.count < 2:
                continue
            
            label, precision = p.best_label
            if precision >= 0.8:
                rarity_boost = 1.0 / (label_freq.get(label, 0.1) + 0.01)
                weight = precision * len(pattern) * p.count * rarity_boost
                votes[label] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        # RULE 4: Prior
        if self.label_counts:
            return max(self.label_counts, key=self.label_counts.get)
        return 'ok'
    
    def describe_strategy(self) -> str:
        """Describe the learning strategy as production rules."""
        lines = ["=== META-LEARNER STRATEGY ===\n"]
        
        lines.append("# Phase Management")
        lines.append("phase_explore  obs_count_100  =>  phase_exploit  obs_count_100")
        lines.append("")
        
        lines.append("# Pattern Learning")
        lines.append("observation  pattern_new  =>  pattern_tracked")
        lines.append("pattern_tracked  label_observed  =>  pattern_stats_updated")
        lines.append("pattern_stats  precision_100  support_2  =>  pattern_pure")
        lines.append("")
        
        lines.append("# Prediction (priority order)")
        lines.append("input  pattern_pure  label_rare  =>  output_label  # Priority 1")
        lines.append("input  pattern_pure  =>  output_label              # Priority 2")
        lines.append("input  pattern_soft  =>  vote_added                # Priority 3")
        lines.append("votes  =>  output_highest_vote                     # Priority 4")
        lines.append("no_match  =>  output_prior                         # Priority 5")
        lines.append("")
        
        lines.append("# Current Meta-State")
        lines.append(f"  observations: {self.stats['observations']}")
        lines.append(f"  patterns: {self.stats['patterns']}")
        lines.append(f"  pure_patterns: {self.stats['pure_patterns']}")
        phase = 'exploit' if 'phase_exploit' in self.meta_state.tokens else 'explore'
        lines.append(f"  phase: {phase}")
        
        return '\n'.join(lines)
    
    def export_learned_rules(self) -> str:
        """Export learned domain rules in production format."""
        lines = ["# LEARNED DOMAIN RULES\n"]
        
        # Group pure patterns by label
        by_label: Dict[str, List[PatternStats]] = defaultdict(list)
        for p in self.patterns.values():
            if p.is_pure:
                label, _ = p.best_label
                by_label[label].append(p)
        
        for label in LABELS:
            patterns = by_label.get(label, [])
            if not patterns:
                continue
            
            lines.append(f"\n## {label} (detected by {len(patterns)} patterns)")
            
            # Sort by count (most observed first)
            patterns.sort(key=lambda p: -p.count)
            
            for p in patterns[:10]:
                tokens_str = "  ".join(sorted(p.tokens))
                lines.append(f"{tokens_str}  =>  {label}")
        
        return '\n'.join(lines)


# =============================================================================
# STRATEGY COMPARISON: Different Learners as Rule Sets
# =============================================================================

def compare_strategies():
    """Compare different learning strategies."""
    print("="*70)
    print("LEARNING STRATEGIES AS PRODUCTION RULES")
    print("="*70)
    
    print("""
Different learner architectures can be expressed as production rule strategies:

=== BLIND LEARNER ===
# Just tracks label frequencies
observation  label  =>  label_count_updated
predict  =>  output_most_common_label

=== PATTERN LEARNER ===
# Tracks pattern-label co-occurrence
observation  pattern  label  =>  pattern_label_count_updated
pattern  precision_100  =>  pattern_pure
predict  pattern_pure  =>  output_pattern_label
predict  no_pure  =>  output_voting

=== UNIFIED LEARNER ===
# Adds soft rules and rarity boosting
observation  pattern  label  =>  pattern_stats_updated
pattern  precision_100  support_2  =>  rule_hard
pattern  precision_85  support_2  =>  rule_soft
predict  rule_hard  label_rare  =>  output_label  # Priority 1
predict  rule_hard  =>  output_label              # Priority 2
predict  rule_soft  =>  vote_weighted             # Priority 3
predict  =>  output_prior                         # Priority 4

=== HYPOTHESIS ELIMINATION ===
# Strict elimination on contradiction
observation  pattern  label  =>  hypothesis_created
hypothesis  contradiction  =>  hypothesis_eliminated
predict  hypothesis_surviving  =>  output_label

=== BUBBLE-UP ABSTRACTION ===
# Creates abstract labels for pattern groups
pattern  high_frequency  high_precision  =>  abstraction_created
abstraction  A0  =  pattern_set_X
predict  abstraction_matches  =>  output_abstraction_label
""")
    
    # Actually run the MetaLearner
    print("\n" + "="*70)
    print("META-LEARNER TEST")
    print("="*70)
    
    from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    learner = MetaLearner(pattern_sizes=[3])
    
    correct = 0
    checkpoints = [100, 500, 1000, 2000]
    
    for i in range(2500):
        obs = gen.next()
        if obs is None:
            break
        
        board, true_idx = obs
        true_label = LABEL_SPACE[true_idx]
        
        pred_label = learner.predict(board)
        
        if pred_label == true_label:
            correct += 1
        
        learner.observe(board, true_label)
        
        if (i + 1) in checkpoints:
            acc = correct / (i + 1)
            print(f"  @{i+1}: acc={acc:.1%}, patterns={learner.stats['patterns']}, pure={learner.stats['pure_patterns']}")
    
    print(f"\nFinal accuracy: {correct/2500:.1%}")
    
    print("\n" + learner.describe_strategy())
    
    print("\n" + "="*70)
    print("LEARNED DOMAIN RULES (sample)")
    print("="*70)
    print(learner.export_learned_rules()[:2000])


def reaching_100_percent():
    """What would it take to reach 100%?"""
    print("\n" + "="*70)
    print("PATH TO 100% ACCURACY")
    print("="*70)
    
    print("""
Current best: ~94% with UnifiedLearner

Errors come from:
1. win2 detection: 33% (hard - fewer examples)
2. draw detection: 0% (very hard - only 16 states)
3. ok vs win: some ambiguity

To reach 100%, we need:

=== NEGATIVE PATTERNS ===
# Detect absence of wins
NOT_win1  =  !has_row_X  !has_col_X  !has_diag_X
NOT_win2  =  !has_row_O  !has_col_O  !has_diag_O

board_full  NOT_win1  NOT_win2  =>  draw

=== PARITY REASONING ===
# X always plays first, so:
count_X  count_O  count_X_equals_count_O  =>  X_turn
count_X  count_O  count_X_equals_count_O_plus_1  =>  O_turn

# Win can only happen on the player who just moved
three_X  O_turn  =>  win1  # X just completed 3-in-row
three_O  X_turn  =>  win2  # O just completed 3-in-row

=== LARGER PATTERNS ===
# Some states need 4+ token patterns
p0_1  p1_1  p2_1  p4_0  =>  ok  # Has 3-in-row but game continues (invalid?)

=== TEMPORAL REASONING ===
# Game history matters
state_t  move  =>  state_t+1
state_t+1  three_in_row  =>  win

The key insight: TicTacToe has HIDDEN STATE (whose turn, move history)
that affects labels but isn't visible in a single board snapshot.

For true 100%, we'd need to either:
1. Include turn/parity in the observation
2. Infer it from X/O counts
3. Learn that some board states are INVALID (can't actually occur)
""")


if __name__ == "__main__":
    compare_strategies()
    reaching_100_percent()
