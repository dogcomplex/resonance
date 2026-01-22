"""
Improved Game Oracle with:
1. Proper error state handling (both-win = error)
2. Multiple ruleset variations for testing
3. Stratified sampling option
4. Error states as learnable labels
"""

import random
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from itertools import product, combinations


LABELS = ['ok', 'winX', 'winO', 'draw', 'error']


def enumerate_boards(size: int = 9, symbols: str = '012') -> List[str]:
    """Generate all possible board states."""
    return [''.join(combo) for combo in product(symbols, repeat=size)]


class FlexibleOracle:
    """
    Flexible game oracle with configurable rules.
    
    Modes:
    - 'standard': Normal TicTacToe (both-win = error, excluded by default)
    - 'include_errors': Include impossible states labeled as 'error'
    - 'x_priority': Both-win labeled as winX (old behavior)
    
    Rulesets:
    - 'standard': 8 win lines (3 rows, 3 cols, 2 diags)
    - 'rows_only': Only row wins
    - 'diags_only': Only diagonal wins
    - 'corners': Win by controlling all 4 corners
    - 'custom': Provide your own win conditions
    """
    
    STANDARD_LINES = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6),              # Diagonals
    ]
    
    ROWS_ONLY = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    DIAGS_ONLY = [(0, 4, 8), (2, 4, 6)]
    CORNERS = [(0, 2, 6, 8)]  # 4-in-a-row variant
    
    def __init__(self, 
                 win_lines: List[Tuple] = None,
                 mode: str = 'standard',
                 ruleset: str = 'standard'):
        
        self.mode = mode
        
        # Set win lines based on ruleset
        if win_lines is not None:
            self.win_lines = [tuple(l) for l in win_lines]
        elif ruleset == 'standard':
            self.win_lines = self.STANDARD_LINES
        elif ruleset == 'rows_only':
            self.win_lines = self.ROWS_ONLY
        elif ruleset == 'diags_only':
            self.win_lines = self.DIAGS_ONLY
        elif ruleset == 'corners':
            self.win_lines = self.CORNERS
        else:
            self.win_lines = self.STANDARD_LINES
        
        # Cache states
        self._all_states: Dict[str, str] = {}
        self._states_by_label: Dict[str, List[str]] = {l: [] for l in LABELS}
        self._enumerate()
        
        # Sampling state
        self._unseen: Set[str] = set(self._all_states.keys())
    
    def _is_valid_parity(self, board: str) -> bool:
        """Check if X count and O count are valid (X goes first)."""
        x_count = board.count('1')
        o_count = board.count('2')
        return x_count == o_count or x_count == o_count + 1
    
    def _check_win(self, board: str, player: str) -> bool:
        """Check if player has won."""
        val = '1' if player == 'X' else '2'
        for line in self.win_lines:
            if all(board[i] == val for i in line):
                return True
        return False
    
    def _compute_label(self, board: str) -> str:
        """Compute label for a board state."""
        # Check parity
        if not self._is_valid_parity(board):
            return 'error'
        
        x_wins = self._check_win(board, 'X')
        o_wins = self._check_win(board, 'O')
        
        # Both win = impossible state
        if x_wins and o_wins:
            if self.mode == 'x_priority':
                return 'winX'  # Old behavior
            else:
                return 'error'  # Correct behavior
        
        if x_wins:
            return 'winX'
        if o_wins:
            return 'winO'
        
        # Full board = draw
        if '0' not in board:
            return 'draw'
        
        return 'ok'
    
    def _enumerate(self):
        """Enumerate all states."""
        for board in enumerate_boards(9, '012'):
            label = self._compute_label(board)
            
            # Skip errors unless including them
            if label == 'error' and self.mode not in ('include_errors', 'x_priority'):
                continue
            
            self._all_states[board] = label
            self._states_by_label[label].append(board)
    
    def label(self, board: str) -> str:
        """Get label for board."""
        return self._all_states.get(board, 'error')
    
    def label_idx(self, board: str) -> int:
        """Get label index."""
        return LABELS.index(self.label(board))
    
    def random_board(self, unique: bool = True) -> Optional[Tuple[str, str]]:
        """Get random board and its label."""
        if unique:
            if not self._unseen:
                return None
            board = random.choice(list(self._unseen))
            self._unseen.remove(board)
        else:
            board = random.choice(list(self._all_states.keys()))
        
        return (board, self._all_states[board])
    
    def stratified_sample(self) -> Optional[Tuple[str, str]]:
        """
        Sample with preference for rare labels.
        
        Ensures early exposure to all label types.
        """
        # Weight by inverse frequency
        weights = {}
        for label, states in self._states_by_label.items():
            unseen_states = [s for s in states if s in self._unseen]
            if unseen_states:
                # Inverse frequency weighting
                weights[label] = 1.0 / (len(states) + 1)
        
        if not weights:
            return None
        
        # Sample label
        total = sum(weights.values())
        r = random.random() * total
        cumsum = 0
        chosen_label = None
        for label, weight in weights.items():
            cumsum += weight
            if r <= cumsum:
                chosen_label = label
                break
        
        if chosen_label is None:
            chosen_label = list(weights.keys())[0]
        
        # Sample state from that label
        unseen_states = [s for s in self._states_by_label[chosen_label] 
                        if s in self._unseen]
        if not unseen_states:
            return self.random_board(unique=True)
        
        board = random.choice(unseen_states)
        self._unseen.remove(board)
        return (board, self._all_states[board])
    
    def reset(self):
        """Reset sampling state."""
        self._unseen = set(self._all_states.keys())
    
    def describe(self) -> str:
        """Describe oracle configuration."""
        lines = [f"=== Flexible Oracle ==="]
        lines.append(f"Mode: {self.mode}")
        lines.append(f"Win lines: {len(self.win_lines)}")
        lines.append(f"Total states: {len(self._all_states)}")
        lines.append("\nLabel distribution:")
        for label in LABELS:
            count = len(self._states_by_label[label])
            if count > 0:
                pct = count / len(self._all_states) * 100
                lines.append(f"  {label}: {count} ({pct:.1f}%)")
        return '\n'.join(lines)


def test_rulesets():
    """Test different rulesets."""
    print("="*70)
    print("TESTING DIFFERENT RULESETS")
    print("="*70)
    
    rulesets = [
        ('standard', 'Standard TicTacToe'),
        ('rows_only', 'Rows Only'),
        ('diags_only', 'Diagonals Only'),
    ]
    
    for ruleset, name in rulesets:
        print(f"\n--- {name} ---")
        
        # Test with errors included
        oracle = FlexibleOracle(ruleset=ruleset, mode='include_errors')
        print(oracle.describe())


def test_error_learning():
    """Test that learner can discover error states."""
    print("\n" + "="*70)
    print("ERROR STATE LEARNING")
    print("="*70)
    
    # Oracle that includes error states
    oracle = FlexibleOracle(mode='include_errors')
    print(oracle.describe())
    
    # Check some error states
    print("\n--- Sample Error States ---")
    error_states = oracle._states_by_label['error'][:5]
    for board in error_states:
        x_count = board.count('1')
        o_count = board.count('2')
        
        x_wins = oracle._check_win(board, 'X')
        o_wins = oracle._check_win(board, 'O')
        
        print(f"  {board[:3]}|{board[3:6]}|{board[6:9]}: X={x_count}, O={o_count}, Xwins={x_wins}, Owins={o_wins}")


def test_stratified_sampling():
    """Test stratified sampling for faster convergence."""
    print("\n" + "="*70)
    print("STRATIFIED SAMPLING")
    print("="*70)
    
    oracle = FlexibleOracle(mode='standard')
    
    # Compare random vs stratified
    for method, sampler in [('Random', oracle.random_board), 
                            ('Stratified', oracle.stratified_sample)]:
        oracle.reset()
        
        first_seen = {}
        for i in range(500):
            result = sampler()
            if result is None:
                break
            board, label = result
            if label not in first_seen:
                first_seen[label] = i + 1
        
        oracle.reset()
        
        print(f"\n{method} sampling - first appearance:")
        for label in LABELS:
            if label in first_seen:
                print(f"  {label}: @{first_seen[label]}")


if __name__ == "__main__":
    test_rulesets()
    test_error_learning()
    test_stratified_sampling()
