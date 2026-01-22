"""
Enhanced Production Rule System

Negation: !token is just an equivalence class for "everything except token"
  - !p0_1 = {p0_0, p0_2} (position 0 is not value 1)
  - Derived from observed value space, not magic

Priority: PRIORITY flag in LHS fires before non-priority rules
  - PRIORITY3 > PRIORITY2 > PRIORITY > (no priority)
  - Just ordering, no additional power

Goal: Reach 100% accuracy as early as possible!
"""

import random
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field
import re

import sys
sys.path.insert(0, '/home/claude/locus')


@dataclass(frozen=True)
class Token:
    """
    Token with optional negation, quantity, probability, and priority.
    
    Negation is just equivalence class notation:
      !p0_1 means "p0 is NOT 1" = "p0 is in {0, 2}"
    
    Priority is execution ordering:
      PRIORITY, PRIORITY2, PRIORITY3 (higher = fires first)
    """
    name: str
    qty: int = 1
    prob: Optional[float] = None
    negated: bool = False
    priority: int = 0  # 0=none, 1=PRIORITY, 2=PRIORITY2, 3=PRIORITY3
    
    def __str__(self):
        s = ""
        if self.priority > 0:
            s += f"PRIORITY{''.join(str(i) for i in range(2, self.priority+1) if i > 1)} "
        if self.negated:
            s += "!"
        s += self.name
        if self.qty != 1:
            s += str(self.qty)
        if self.prob is not None:
            s += f"%{int(self.prob * 100)}"
        return s
    
    def matches(self, state_tokens: Set[str], value_space: Dict[str, Set[str]] = None) -> bool:
        """
        Check if this token matches the state.
        
        For negated tokens, checks that the positive version is NOT present.
        This is equivalent to checking membership in the complement set.
        """
        if self.negated:
            # !p0_1 matches if p0_1 is NOT in state
            return self.name not in state_tokens
        else:
            return self.name in state_tokens
    
    @staticmethod
    def parse(s: str) -> 'Token':
        """Parse token from string."""
        s = s.strip()
        
        # Check for priority prefix
        priority = 0
        if s.startswith('PRIORITY'):
            if s.startswith('PRIORITY3'):
                priority = 3
                s = s[9:].strip()
            elif s.startswith('PRIORITY2'):
                priority = 2
                s = s[9:].strip()
            else:
                priority = 1
                s = s[8:].strip()
        
        # Check for negation
        negated = False
        if s.startswith('!'):
            negated = True
            s = s[1:]
        
        # Parse probability suffix
        prob = None
        if '%' in s:
            parts = s.rsplit('%', 1)
            s = parts[0]
            prob = int(parts[1]) / 100.0
        
        # Parse name and quantity
        match = re.match(r'^([a-zA-Z_][a-zA-Z_]*)(\d*)$', s)
        if not match:
            return Token(s, 1, prob, negated, priority)
        
        name = match.group(1)
        qty_str = match.group(2)
        qty = int(qty_str) if qty_str else 1
        
        return Token(name, qty, prob, negated, priority)


@dataclass
class Rule:
    """
    Production rule with priority support.
    
    Priority determines firing order:
      PRIORITY3 rules fire first
      PRIORITY2 rules fire second
      PRIORITY rules fire third
      Non-priority rules fire last
    """
    lhs: Tuple[Token, ...]
    rhs: Tuple[Token, ...]
    
    fires: int = 0
    correct: int = 0
    
    def __post_init__(self):
        if not isinstance(self.lhs, tuple):
            object.__setattr__(self, 'lhs', tuple(self.lhs))
        if not isinstance(self.rhs, tuple):
            object.__setattr__(self, 'rhs', tuple(self.rhs))
    
    @property
    def priority(self) -> int:
        """Highest priority among LHS tokens."""
        return max((t.priority for t in self.lhs), default=0)
    
    @property
    def catalysts(self) -> Set[str]:
        """Token names on both sides."""
        lhs_names = {t.name for t in self.lhs if not t.negated}
        rhs_names = {t.name for t in self.rhs if not t.negated}
        return lhs_names & rhs_names
    
    def matches(self, state: Set[str]) -> bool:
        """Check if LHS matches state (including negations)."""
        for token in self.lhs:
            if token.priority > 0:
                continue  # Priority tokens are flags, not conditions
            if not token.matches(state):
                return False
        return True
    
    def __str__(self):
        lhs_str = "  ".join(str(t) for t in self.lhs)
        rhs_str = "  ".join(str(t) for t in self.rhs)
        return f"{lhs_str}  =>  {rhs_str}"
    
    @staticmethod
    def parse(s: str) -> 'Rule':
        """Parse rule from string."""
        if "=>" not in s:
            raise ValueError(f"Rule must contain '=>': {s}")
        
        lhs_str, rhs_str = s.split("=>", 1)
        lhs_tokens = tuple(Token.parse(t) for t in lhs_str.split() if t)
        rhs_tokens = tuple(Token.parse(t) for t in rhs_str.split() if t)
        
        return Rule(lhs=lhs_tokens, rhs=rhs_tokens)


class ValueSpace:
    """
    Tracks the observed value space for each position.
    
    This allows negation to be a derived equivalence class:
      !p0_1 = {p0_v : v in observed_values[0] and v != 1}
    """
    
    def __init__(self):
        # position -> set of observed values
        self.values: Dict[str, Set[str]] = defaultdict(set)
    
    def observe(self, tokens: Set[str]):
        """Record observed tokens to build value space."""
        for token in tokens:
            if '_' in token:
                parts = token.rsplit('_', 1)
                position = parts[0]  # e.g., "p0"
                value = parts[1]      # e.g., "1"
                self.values[position].add(value)
    
    def get_complement(self, token: str) -> Set[str]:
        """
        Get the complement set for a negated token.
        
        !p0_1 -> {p0_0, p0_2} (if we've seen values 0, 1, 2)
        """
        if '_' not in token:
            return set()
        
        parts = token.rsplit('_', 1)
        position = parts[0]
        excluded_value = parts[1]
        
        return {f"{position}_{v}" for v in self.values[position] if v != excluded_value}


class PriorityLearner:
    """
    Learner that discovers rules with priority and negation.
    
    Strategy:
    1. Track patterns and their label correlations
    2. Discover pure patterns (100% precision)
    3. Assign priority based on label rarity
    4. Use negation for draw detection (absence of wins)
    
    Goal: 100% accuracy as early as possible!
    """
    
    def __init__(self, pattern_sizes: List[int] = None):
        self.pattern_sizes = pattern_sizes or [3]
        
        # Value space for negation
        self.value_space = ValueSpace()
        
        # Pattern statistics: pattern -> {label -> count}
        self.patterns: Dict[FrozenSet[str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pattern_counts: Dict[FrozenSet[str], int] = defaultdict(int)
        
        # Label statistics
        self.label_counts: Dict[str, int] = defaultdict(int)
        
        # Discovered rules (sorted by priority)
        self.rules: List[Rule] = []
        
        # Derived abstract tokens
        self.abstractions: Dict[str, FrozenSet[str]] = {}
        
        # Stats
        self.observations = 0
        self.correct_predictions = 0
    
    def _tokenize(self, board: str) -> Set[str]:
        """Convert board to tokens."""
        return {f"p{i}_{board[i]}" for i in range(len(board))}
    
    def _extract_patterns(self, tokens: Set[str]) -> List[FrozenSet[str]]:
        """Extract patterns of configured sizes."""
        patterns = []
        token_list = sorted(tokens)
        for size in self.pattern_sizes:
            for combo in combinations(token_list, size):
                patterns.append(frozenset(combo))
        return patterns
    
    def _check_win(self, board: str, player: str) -> bool:
        """Check if player has won (helper for derived tokens)."""
        lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        val = '1' if player == 'X' else '2'
        return any(all(board[i] == val for i in line) for line in lines)
    
    def _get_derived_tokens(self, board: str) -> Set[str]:
        """
        Compute derived tokens from board state.
        
        These are equivalence classes, not magic:
        - has_win_X = EXISTS line where all positions are 1
        - board_full = NOT EXISTS position with value 0
        """
        tokens = self._tokenize(board)
        
        # Check for wins (derived from pattern existence)
        if self._check_win(board, 'X'):
            tokens.add('has_win_X')
        else:
            tokens.add('no_win_X')  # Equivalence class: !has_win_X
        
        if self._check_win(board, 'O'):
            tokens.add('has_win_O')
        else:
            tokens.add('no_win_O')
        
        # Check board full
        if '0' not in board:
            tokens.add('board_full')
        else:
            tokens.add('has_empty')
        
        return tokens
    
    def observe(self, board: str, label: str):
        """Observe a board and its label."""
        self.observations += 1
        self.label_counts[label] += 1
        
        tokens = self._tokenize(board)
        self.value_space.observe(tokens)
        
        # Extract and track patterns
        patterns = self._extract_patterns(tokens)
        for pattern in patterns:
            self.pattern_counts[pattern] += 1
            self.patterns[pattern][label] += 1
        
        # Rebuild rules periodically
        if self.observations % 50 == 0:
            self._rebuild_rules()
    
    def _rebuild_rules(self):
        """Rebuild rule set from observed patterns."""
        self.rules = []
        
        total_obs = sum(self.label_counts.values())
        if total_obs == 0:
            return
        
        # Calculate label frequencies for rarity
        label_freq = {l: c / total_obs for l, c in self.label_counts.items()}
        
        # Find pure patterns (100% precision)
        pure_patterns = []
        for pattern, label_counts in self.patterns.items():
            total = self.pattern_counts[pattern]
            if total < 2:
                continue
            
            for label, count in label_counts.items():
                precision = count / total
                if precision == 1.0:
                    rarity = 1.0 / (label_freq.get(label, 0.01) + 0.001)
                    pure_patterns.append((pattern, label, total, rarity))
        
        # Sort by rarity (rarer labels get higher priority)
        pure_patterns.sort(key=lambda x: -x[3])
        
        # Assign priorities based on rarity ranking
        for i, (pattern, label, support, rarity) in enumerate(pure_patterns):
            # Top patterns get PRIORITY3, next get PRIORITY2, etc.
            if i < 8:  # Top 8 (one per win line roughly)
                priority = 3
            elif i < 24:
                priority = 2
            else:
                priority = 1
            
            # Build rule
            lhs_tokens = [Token.parse(f"PRIORITY{priority if priority > 1 else ''} {t}") 
                         if j == 0 else Token.parse(t)
                         for j, t in enumerate(sorted(pattern))]
            rhs_tokens = [Token.parse(t) for t in sorted(pattern)] + [Token.parse(label)]
            
            rule = Rule(lhs=tuple(lhs_tokens), rhs=tuple(rhs_tokens))
            self.rules.append(rule)
        
        # Add draw rule (uses negation - equivalence class for "no wins")
        # board_full AND !has_win_X AND !has_win_O => draw
        if self.label_counts.get('draw', 0) > 0:
            draw_rule = Rule.parse("PRIORITY3 board_full  no_win_X  no_win_O  =>  board_full  no_win_X  no_win_O  draw")
            self.rules.insert(0, draw_rule)  # High priority
        
        # Sort rules by priority (descending)
        self.rules.sort(key=lambda r: -r.priority)
    
    def predict(self, board: str) -> str:
        """
        Predict label using priority-ordered rules.
        
        Rules fire in priority order:
          PRIORITY3 rules checked first
          PRIORITY2 rules checked second
          etc.
        """
        tokens = self._get_derived_tokens(board)
        
        # Check rules in priority order
        for rule in self.rules:
            if rule.matches(tokens):
                # Extract predicted label from RHS
                for rhs_token in rule.rhs:
                    if rhs_token.name in ['ok', 'win1', 'win2', 'draw', 'error']:
                        return rhs_token.name
        
        # Default: most common label
        if self.label_counts:
            return max(self.label_counts, key=self.label_counts.get)
        return 'ok'
    
    def describe(self) -> str:
        """Describe learned rules."""
        lines = ["=== PRIORITY LEARNER ===\n"]
        
        lines.append(f"Observations: {self.observations}")
        lines.append(f"Rules: {len(self.rules)}")
        lines.append(f"Labels: {dict(self.label_counts)}")
        
        lines.append("\n--- Rules by Priority ---")
        
        by_priority = defaultdict(list)
        for rule in self.rules:
            by_priority[rule.priority].append(rule)
        
        for p in [3, 2, 1, 0]:
            if by_priority[p]:
                lines.append(f"\nPriority {p} ({len(by_priority[p])} rules):")
                for rule in by_priority[p][:5]:
                    lines.append(f"  {rule}")
                if len(by_priority[p]) > 5:
                    lines.append(f"  ... and {len(by_priority[p]) - 5} more")
        
        return '\n'.join(lines)


def test_priority_learner():
    """Test the priority learner for early 100% accuracy."""
    print("="*70)
    print("PRIORITY LEARNER - Racing to 100%")
    print("="*70)
    
    from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    learner = PriorityLearner(pattern_sizes=[3])
    
    correct = 0
    first_100 = None
    
    # Checkpoints
    checkpoints = [100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6046]
    checkpoint_results = {}
    
    all_obs = []
    while True:
        obs = gen.next()
        if obs is None:
            break
        all_obs.append(obs)
    
    random.shuffle(all_obs)  # Random order to test early convergence
    
    for i, (board, true_idx) in enumerate(all_obs):
        true_label = LABEL_SPACE[true_idx]
        
        pred_label = learner.predict(board)
        
        if pred_label == true_label:
            correct += 1
        
        learner.observe(board, true_label)
        
        n = i + 1
        acc = correct / n
        
        if n in checkpoints:
            checkpoint_results[n] = acc
            print(f"  @{n}: {acc:.1%} accuracy ({correct}/{n}), {len(learner.rules)} rules")
        
        # Check for 100%
        if first_100 is None and acc == 1.0 and n >= 100:
            first_100 = n
    
    print(f"\nFinal accuracy: {correct}/{len(all_obs)} = {correct/len(all_obs):.1%}")
    
    if first_100:
        print(f"First reached 100% at observation {first_100}")
    else:
        print("Never reached 100%")
    
    print("\n" + learner.describe())
    
    # Per-label analysis
    print("\n--- Per-Label Analysis ---")
    
    # Reset and test
    oracle.reset_seen()
    gen = UniqueObservationGenerator(oracle)
    
    by_label = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    while True:
        obs = gen.next()
        if obs is None:
            break
        board, true_idx = obs
        true_label = LABEL_SPACE[true_idx]
        pred = learner.predict(board)
        
        by_label[true_label]['total'] += 1
        if pred == true_label:
            by_label[true_label]['correct'] += 1
    
    for label in LABEL_SPACE:
        info = by_label[label]
        if info['total'] > 0:
            acc = info['correct'] / info['total']
            print(f"  {label}: {acc:.1%} ({info['correct']}/{info['total']})")


if __name__ == "__main__":
    test_priority_learner()
