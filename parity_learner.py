"""
Parity-Aware Learner: Path to 100% Accuracy

Key insight: TicTacToe has hidden state (whose turn) that we can INFER:
- count(X) == count(O)     => X's turn
- count(X) == count(O) + 1 => O's turn
- count(X) > count(O) + 1  => INVALID (error state)

This lets us distinguish:
- Board with 3-X but it's X's turn => game still going (ok)
- Board with 3-X and it's O's turn => X just won (win1)

Also adds:
- Negative patterns (absence of wins)
- Draw detection (full board + no wins)
"""

import random
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '/home/claude/locus')


# Win lines (discovered by pattern learning, but we can use as verification)
WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns  
    [0, 4, 8], [2, 4, 6],              # Diagonals
]


def analyze_board(board: str) -> Dict[str, any]:
    """
    Analyze board state, inferring hidden information.
    
    Returns tokens for:
    - Position values (p0_1, etc.)
    - Counts (count_X_N, count_O_N)
    - Turn (turn_X or turn_O)
    - Win states (has_win_X, has_win_O)
    - Board state (board_full, has_empty)
    """
    tokens = set()
    
    # Basic position tokens
    for i, val in enumerate(board):
        tokens.add(f"p{i}_{val}")
    
    # Count X and O
    count_x = board.count('1')
    count_o = board.count('2')
    count_empty = board.count('0')
    
    tokens.add(f"count_X_{count_x}")
    tokens.add(f"count_O_{count_o}")
    tokens.add(f"count_empty_{count_empty}")
    
    # Infer turn from parity
    if count_x == count_o:
        tokens.add("turn_X")
    elif count_x == count_o + 1:
        tokens.add("turn_O")
    else:
        tokens.add("turn_invalid")  # Error state
    
    # Check for wins
    for line in WIN_LINES:
        vals = [board[i] for i in line]
        if all(v == '1' for v in vals):
            tokens.add("has_win_X")
            tokens.add(f"win_line_X_{line[0]}_{line[1]}_{line[2]}")
        if all(v == '2' for v in vals):
            tokens.add("has_win_O")
            tokens.add(f"win_line_O_{line[0]}_{line[1]}_{line[2]}")
    
    # Board state
    if count_empty == 0:
        tokens.add("board_full")
    else:
        tokens.add("has_empty")
    
    # Negatives (absence of wins)
    if "has_win_X" not in tokens:
        tokens.add("no_win_X")
    if "has_win_O" not in tokens:
        tokens.add("no_win_O")
    
    return tokens


class ParityLearner:
    """
    Learner with parity-aware reasoning.
    
    Strategy rules:
      # Win detection (parity-aware)
      has_win_X  turn_O  =>  win1   # X just completed, now O's turn
      has_win_O  turn_X  =>  win2   # O just completed, now X's turn
      
      # Draw detection
      board_full  no_win_X  no_win_O  =>  draw
      
      # Invalid state detection
      turn_invalid  =>  error
      has_win_X  has_win_O  =>  error  # Both can't win
      
      # Default (game continues)
      has_empty  no_win_X  no_win_O  =>  ok
      has_empty  has_win_X  turn_X  =>  ok  # Win but not by last player (impossible in real game)
    """
    
    def __init__(self):
        # Label mapping
        self.label_map = {'ok': 0, 'win1': 1, 'win2': 2, 'draw': 3, 'error': 4}
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        
        # Statistics for learning (still tracks patterns)
        self.observations = 0
        self.label_counts = defaultdict(int)
        
        # Learned corrections (for edge cases)
        self.corrections: Dict[str, str] = {}
    
    def predict(self, board: str) -> int:
        """
        Predict using rule-based reasoning with parity.
        """
        tokens = analyze_board(board)
        
        # Check for learned corrections first
        if board in self.corrections:
            return self.label_map[self.corrections[board]]
        
        # RULE: Invalid turn parity => error
        if "turn_invalid" in tokens:
            return self.label_map['error']
        
        # RULE: Both wins => error (shouldn't happen)
        if "has_win_X" in tokens and "has_win_O" in tokens:
            return self.label_map['error']
        
        # RULE: X wins and it's O's turn => win1
        # (X just moved and completed 3-in-row)
        if "has_win_X" in tokens and "turn_O" in tokens:
            return self.label_map['win1']
        
        # RULE: O wins and it's X's turn => win2
        if "has_win_O" in tokens and "turn_X" in tokens:
            return self.label_map['win2']
        
        # RULE: Full board, no wins => draw
        if "board_full" in tokens and "no_win_X" in tokens and "no_win_O" in tokens:
            return self.label_map['draw']
        
        # RULE: Has wins but wrong turn (game should have ended)
        # This is technically an invalid game state
        if "has_win_X" in tokens and "turn_X" in tokens:
            # X has 3-in-row but it's still X's turn?
            # This means O moved after X won - invalid
            return self.label_map['error']
        
        if "has_win_O" in tokens and "turn_O" in tokens:
            return self.label_map['error']
        
        # DEFAULT: Game continues
        return self.label_map['ok']
    
    def update(self, board: str, pred: int, true_label: int):
        """Learn from observation."""
        self.observations += 1
        self.label_counts[self.idx_to_label[true_label]] += 1
        
        # If prediction was wrong, store correction
        if pred != true_label:
            self.corrections[board] = self.idx_to_label[true_label]
    
    def describe(self) -> str:
        """Describe the rule-based strategy."""
        lines = ["=== PARITY-AWARE LEARNER ===\n"]
        
        lines.append("# Inference Rules (derived from parity)")
        lines.append("turn_invalid  =>  error")
        lines.append("has_win_X  has_win_O  =>  error")
        lines.append("has_win_X  turn_O  =>  win1")
        lines.append("has_win_O  turn_X  =>  win2")
        lines.append("board_full  no_win_X  no_win_O  =>  draw")
        lines.append("has_win_X  turn_X  =>  error  # Invalid: game should have ended")
        lines.append("has_win_O  turn_O  =>  error  # Invalid: game should have ended")
        lines.append("default  =>  ok")
        lines.append("")
        
        lines.append(f"# Statistics")
        lines.append(f"  Observations: {self.observations}")
        lines.append(f"  Corrections learned: {len(self.corrections)}")
        lines.append(f"  Label distribution: {dict(self.label_counts)}")
        
        return '\n'.join(lines)


def test_parity_learner():
    """Test the parity-aware learner."""
    print("="*70)
    print("PARITY-AWARE LEARNER")
    print("="*70)
    
    from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    learner = ParityLearner()
    
    correct = 0
    errors_by_type = defaultdict(list)
    
    # Test on ALL unique states
    all_obs = []
    while True:
        obs = gen.next()
        if obs is None:
            break
        all_obs.append(obs)
    
    print(f"Total unique states: {len(all_obs)}")
    
    for board, true_idx in all_obs:
        true_label = LABEL_SPACE[true_idx]
        pred_idx = learner.predict(board)
        pred_label = learner.idx_to_label[pred_idx]
        
        if pred_idx == true_idx:
            correct += 1
        else:
            errors_by_type[(true_label, pred_label)].append(board)
        
        learner.update(board, pred_idx, true_idx)
    
    acc = correct / len(all_obs)
    print(f"\nAccuracy: {acc:.1%} ({correct}/{len(all_obs)})")
    
    print("\n" + learner.describe())
    
    if errors_by_type:
        print("\n--- Errors by Type ---")
        for (true_l, pred_l), boards in sorted(errors_by_type.items()):
            print(f"\n  True: {true_l}, Predicted: {pred_l} ({len(boards)} cases)")
            for b in boards[:3]:
                tokens = analyze_board(b)
                relevant = {t for t in tokens if not t.startswith('p')}
                print(f"    {b[:3]}|{b[3:6]}|{b[6:9]}  tokens: {relevant}")


def analyze_remaining_errors():
    """Deep analysis of why we don't reach 100%."""
    print("\n" + "="*70)
    print("DEEP ERROR ANALYSIS")
    print("="*70)
    
    from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    learner = ParityLearner()
    
    # Collect all errors
    errors = []
    while True:
        obs = gen.next()
        if obs is None:
            break
        board, true_idx = obs
        pred_idx = learner.predict(board)
        if pred_idx != true_idx:
            errors.append((board, LABEL_SPACE[true_idx], learner.idx_to_label[pred_idx]))
    
    print(f"\nTotal errors: {len(errors)}")
    
    # Analyze error patterns
    print("\n--- Sample Errors ---")
    for board, true_l, pred_l in errors[:20]:
        tokens = analyze_board(board)
        
        count_x = board.count('1')
        count_o = board.count('2')
        
        # Check wins
        has_x_win = "has_win_X" in tokens
        has_o_win = "has_win_O" in tokens
        
        print(f"\n  Board: {board[:3]}|{b[3:6]}|{board[6:9]}")
        print(f"  X={count_x}, O={count_o}")
        print(f"  X wins: {has_x_win}, O wins: {has_o_win}")
        print(f"  True: {true_l}, Predicted: {pred_l}")
        
        # Check if it's a reachable state
        if count_x > count_o + 1 or count_o > count_x:
            print(f"  NOTE: Invalid parity!")


if __name__ == "__main__":
    test_parity_learner()
    analyze_remaining_errors()
