"""
Game Oracle Framework

Provides:
1. Game oracles that map board states to labels
2. Unique observation generators (no duplicates)
3. Full state space enumeration
4. Support for error states
5. Random rule generation for testing

Key invariants:
- Observations are deterministic (same input -> same output)
- No hidden information
- Rules are invariant (won't change during learning)
"""

import random
from typing import List, Dict, Set, Callable, Tuple, Optional
from collections import Counter
from itertools import product


# Label space (consistent across all games)
LABEL_SPACE = ['ok', 'win1', 'win2', 'draw', 'error']


def enumerate_all_boards(board_size: int = 9, symbols: str = '012') -> List[str]:
    """Generate all possible board states."""
    return [''.join(combo) for combo in product(symbols, repeat=board_size)]


def is_valid_turn_count(board: str) -> bool:
    """Check if board has valid turn count (X goes first)."""
    count_1 = board.count('1')  # X
    count_2 = board.count('2')  # O
    # X should equal O or be one more (X goes first)
    return count_1 == count_2 or count_1 == count_2 + 1


class GameOracle:
    """
    Base class for game oracles.
    
    Provides:
    - Deterministic labeling of board states
    - Full state space enumeration
    - Unique observation generation
    """
    
    def __init__(self, board_size: int = 9, symbols: str = '012',
                 include_errors: bool = False):
        self.board_size = board_size
        self.symbols = symbols
        self.include_errors = include_errors
        self.label_space = LABEL_SPACE
        
        # Cache all states and their labels
        self._all_states: Dict[str, str] = {}
        self._states_by_label: Dict[str, List[str]] = {l: [] for l in LABEL_SPACE}
        self._enumerate_states()
        
        # For unique observation generation
        self._unseen_states: Set[str] = set(self._all_states.keys())
        self._seen_states: Set[str] = set()
    
    def _enumerate_states(self):
        """Enumerate all valid states. Override in subclasses."""
        raise NotImplementedError
    
    def label(self, board: str) -> str:
        """Get label for board state."""
        return self._all_states.get(board, 'error')
    
    def label_idx(self, board: str) -> int:
        """Get label index for board state."""
        return self.label_space.index(self.label(board))
    
    def random_board(self, unique: bool = True) -> Optional[str]:
        """
        Get a random board state.
        
        Args:
            unique: If True, only return boards not yet seen.
                   Returns None if all boards have been seen.
        """
        if unique:
            if not self._unseen_states:
                return None  # All states seen!
            board = random.choice(list(self._unseen_states))
            self._unseen_states.remove(board)
            self._seen_states.add(board)
            return board
        else:
            return random.choice(list(self._all_states.keys()))
    
    def reset_seen(self):
        """Reset the seen/unseen tracking."""
        self._unseen_states = set(self._all_states.keys())
        self._seen_states = set()
    
    def coverage(self) -> float:
        """Fraction of states that have been observed."""
        total = len(self._all_states)
        return len(self._seen_states) / total if total > 0 else 0.0
    
    def remaining(self) -> int:
        """Number of unseen states."""
        return len(self._unseen_states)
    
    def total_states(self) -> int:
        """Total number of states."""
        return len(self._all_states)
    
    def states_by_label(self, label: str) -> List[str]:
        """Get all states with a given label."""
        return self._states_by_label[label]
    
    def label_distribution(self) -> Dict[str, int]:
        """Get count of states per label."""
        return {l: len(self._states_by_label[l]) for l in self.label_space}


class TicTacToeOracle(GameOracle):
    """Standard TicTacToe oracle."""
    
    def __init__(self, win_conditions: List[List[int]] = None,
                 include_errors: bool = False):
        self.win_conditions = win_conditions or [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6],              # Diagonals
        ]
        super().__init__(board_size=9, symbols='012', include_errors=include_errors)
    
    def _enumerate_states(self):
        """Enumerate all valid TicTacToe states."""
        for board in enumerate_all_boards(9, '012'):
            label = self._compute_label(board)
            
            if label == 'error' and not self.include_errors:
                continue
            
            self._all_states[board] = label
            self._states_by_label[label].append(board)
    
    def _compute_label(self, board: str) -> str:
        """Compute label for a board state."""
        # Check turn count validity
        if not is_valid_turn_count(board):
            return 'error'
        
        # Check for wins
        for condition in self.win_conditions:
            if all(board[i] == '1' for i in condition):
                return 'win1'
        
        for condition in self.win_conditions:
            if all(board[i] == '2' for i in condition):
                return 'win2'
        
        # Check for draw (full board, no wins)
        if '0' not in board:
            return 'draw'
        
        return 'ok'


class RandomRuleOracle(GameOracle):
    """
    Oracle with randomly generated win conditions.
    
    For testing learner generalization to arbitrary rules.
    """
    
    def __init__(self, num_win_conditions: int = 8, 
                 win_size: int = 3,
                 seed: int = None,
                 include_errors: bool = False):
        if seed is not None:
            random.seed(seed)
        
        # Generate random win conditions
        self.win_conditions = self._generate_random_conditions(
            num_win_conditions, win_size
        )
        
        super().__init__(board_size=9, symbols='012', include_errors=include_errors)
    
    def _generate_random_conditions(self, num: int, size: int) -> List[List[int]]:
        """Generate random win conditions."""
        conditions = set()
        attempts = 0
        max_attempts = num * 100
        
        while len(conditions) < num and attempts < max_attempts:
            attempts += 1
            # Random positions
            positions = tuple(sorted(random.sample(range(9), size)))
            conditions.add(positions)
        
        return [list(c) for c in conditions]
    
    def _enumerate_states(self):
        """Enumerate states using the random rules."""
        for board in enumerate_all_boards(9, '012'):
            label = self._compute_label(board)
            
            if label == 'error' and not self.include_errors:
                continue
            
            self._all_states[board] = label
            self._states_by_label[label].append(board)
    
    def _compute_label(self, board: str) -> str:
        """Compute label using random win conditions."""
        if not is_valid_turn_count(board):
            return 'error'
        
        for condition in self.win_conditions:
            if all(board[i] == '1' for i in condition):
                return 'win1'
        
        for condition in self.win_conditions:
            if all(board[i] == '2' for i in condition):
                return 'win2'
        
        if '0' not in board:
            return 'draw'
        
        return 'ok'


class UniqueObservationGenerator:
    """
    Generates unique observations from an oracle.
    
    Ensures:
    - No duplicate observations
    - Tracks coverage
    - Can stratify by label if needed
    """
    
    def __init__(self, oracle: GameOracle, stratified: bool = False):
        self.oracle = oracle
        self.stratified = stratified
        self.oracle.reset_seen()
    
    def next(self) -> Optional[Tuple[str, int]]:
        """Get next unique observation."""
        board = self.oracle.random_board(unique=True)
        if board is None:
            return None
        return board, self.oracle.label_idx(board)
    
    def next_batch(self, n: int) -> List[Tuple[str, int]]:
        """Get batch of unique observations."""
        batch = []
        for _ in range(n):
            obs = self.next()
            if obs is None:
                break
            batch.append(obs)
        return batch
    
    def coverage(self) -> float:
        return self.oracle.coverage()
    
    def remaining(self) -> int:
        return self.oracle.remaining()
    
    def reset(self):
        self.oracle.reset_seen()


if __name__ == "__main__":
    print("=== Game Oracle Framework ===\n")
    
    # Standard TicTacToe
    oracle = TicTacToeOracle()
    print(f"Standard TicTacToe:")
    print(f"  Total states: {oracle.total_states()}")
    print(f"  Distribution: {oracle.label_distribution()}")
    
    # With errors
    oracle_err = TicTacToeOracle(include_errors=True)
    print(f"\nWith error states:")
    print(f"  Total states: {oracle_err.total_states()}")
    print(f"  Distribution: {oracle_err.label_distribution()}")
    
    # Random rules
    print("\n--- Random Rule Games ---")
    for seed in [42, 123, 456]:
        oracle = RandomRuleOracle(num_win_conditions=8, win_size=3, seed=seed)
        dist = oracle.label_distribution()
        print(f"\nSeed {seed}:")
        print(f"  Win conditions: {oracle.win_conditions}")
        print(f"  States: {oracle.total_states()}")
        print(f"  Wins: {dist['win1']} + {dist['win2']} = {dist['win1']+dist['win2']}")
    
    # Unique observation generator
    print("\n--- Unique Observation Generator ---")
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    obs_count = 0
    while True:
        obs = gen.next()
        if obs is None:
            break
        obs_count += 1
    
    print(f"Generated {obs_count} unique observations")
    print(f"Coverage: {gen.coverage():.1%}")
