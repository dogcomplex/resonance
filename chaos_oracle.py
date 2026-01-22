"""
Chaos Oracle - Test learner against wild game variations

Supports:
- Variable pattern sizes (3, 4, 5+ in a row)
- Random win conditions
- Asymmetric rules (X and O have different win conditions!)
- Multiple win conditions required
"""

import random
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from itertools import combinations, product


LABELS = ['ok', 'winX', 'winO', 'draw', 'error']


class ChaosOracle:
    """
    Flexible oracle supporting arbitrary win conditions.
    
    Features:
    - Variable pattern sizes
    - Separate X and O win conditions (asymmetric games!)
    - Error state detection
    """
    
    def __init__(self, 
                 x_win_lines: List[Tuple] = None,
                 o_win_lines: List[Tuple] = None,
                 mode: str = 'standard'):
        """
        Args:
            x_win_lines: Win conditions for X (default: standard TicTacToe)
            o_win_lines: Win conditions for O (default: same as X)
            mode: 'standard' (exclude errors), 'include_errors' (include all)
        """
        self.mode = mode
        
        # Default to standard TicTacToe
        standard = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        
        self.x_win_lines = [tuple(l) for l in (x_win_lines or standard)]
        self.o_win_lines = [tuple(l) for l in (o_win_lines or self.x_win_lines)]
        
        self.is_symmetric = (set(self.x_win_lines) == set(self.o_win_lines))
        
        # Pattern sizes for learner hints
        self.x_pattern_sizes = set(len(l) for l in self.x_win_lines)
        self.o_pattern_sizes = set(len(l) for l in self.o_win_lines)
        
        # Enumerate states
        self._all_states: Dict[str, str] = {}
        self._states_by_label: Dict[str, List[str]] = {l: [] for l in LABELS}
        self._enumerate()
        
        self._unseen: Set[str] = set(self._all_states.keys())
    
    def _is_valid_parity(self, board: str) -> bool:
        x = board.count('1')
        o = board.count('2')
        return x == o or x == o + 1
    
    def _check_win(self, board: str, player: str, lines: List[Tuple]) -> bool:
        val = '1' if player == 'X' else '2'
        for line in lines:
            if all(board[i] == val for i in line):
                return True
        return False
    
    def _compute_label(self, board: str) -> str:
        if not self._is_valid_parity(board):
            return 'error'
        
        x_wins = self._check_win(board, 'X', self.x_win_lines)
        o_wins = self._check_win(board, 'O', self.o_win_lines)
        
        # Both win = error (for symmetric games)
        # For asymmetric, this might be valid...
        if x_wins and o_wins:
            return 'error'
        
        if x_wins:
            return 'winX'
        if o_wins:
            return 'winO'
        
        if '0' not in board:
            return 'draw'
        
        return 'ok'
    
    def _enumerate(self):
        for board in [''.join(c) for c in product('012', repeat=9)]:
            label = self._compute_label(board)
            
            if label == 'error' and self.mode == 'standard':
                continue
            
            self._all_states[board] = label
            self._states_by_label[label].append(board)
    
    def label(self, board: str) -> str:
        return self._all_states.get(board, 'error')
    
    def random_board(self, unique: bool = True) -> Optional[Tuple[str, str]]:
        if unique:
            if not self._unseen:
                return None
            board = random.choice(list(self._unseen))
            self._unseen.remove(board)
        else:
            board = random.choice(list(self._all_states.keys()))
        return (board, self._all_states[board])
    
    def reset(self):
        self._unseen = set(self._all_states.keys())
    
    def describe(self) -> str:
        lines = [f"=== Chaos Oracle ==="]
        lines.append(f"X win lines: {len(self.x_win_lines)}, sizes: {self.x_pattern_sizes}")
        lines.append(f"O win lines: {len(self.o_win_lines)}, sizes: {self.o_pattern_sizes}")
        lines.append(f"Symmetric: {self.is_symmetric}")
        lines.append(f"Total states: {len(self._all_states)}")
        lines.append("Label distribution:")
        for label in LABELS:
            count = len(self._states_by_label[label])
            if count > 0:
                pct = count / len(self._all_states) * 100
                lines.append(f"  {label}: {count} ({pct:.1f}%)")
        return '\n'.join(lines)


# =============================================================================
# CHAOS RULESET GENERATORS
# =============================================================================

ALL_TRIPLES = list(combinations(range(9), 3))
ALL_QUADS = list(combinations(range(9), 4))

ROWS = [(0,1,2), (3,4,5), (6,7,8)]
COLS = [(0,3,6), (1,4,7), (2,5,8)]
DIAGS = [(0,4,8), (2,4,6)]
STANDARD = ROWS + COLS + DIAGS

L_SHAPES = [(0,1,3), (2,1,5), (6,3,7), (8,5,7)]
KNIGHT_MOVES = [(0,5,6), (2,3,8), (0,7,2), (6,1,8)]


def generate_chaos_rulesets() -> Dict[str, Dict]:
    """Generate a variety of test rulesets."""
    rulesets = {}
    
    # === SYMMETRIC GAMES (X and O have same rules) ===
    
    # Standard variants
    rulesets['standard'] = {'x': STANDARD, 'o': None}
    rulesets['rows_only'] = {'x': ROWS, 'o': None}
    rulesets['cols_only'] = {'x': COLS, 'o': None}
    rulesets['diags_only'] = {'x': DIAGS, 'o': None}
    rulesets['rows_diags'] = {'x': ROWS + DIAGS, 'o': None}
    
    # Single line
    rulesets['single_line'] = {'x': [(0,1,2)], 'o': None}
    
    # Weird patterns
    rulesets['L_shapes'] = {'x': L_SHAPES, 'o': None}
    rulesets['knight_moves'] = {'x': KNIGHT_MOVES, 'o': None}
    
    # Random rulesets
    random.seed(42)
    for n in [2, 3, 5, 8, 12]:
        rules = random.sample(ALL_TRIPLES, n)
        rulesets[f'random_{n}'] = {'x': rules, 'o': None}
    
    # 4-position patterns
    rulesets['corners_4'] = {'x': [(0,2,6,8)], 'o': None}
    rulesets['edges_4'] = {'x': [(1,3,5,7)], 'o': None}
    
    random.seed(123)
    rulesets['random_4pos'] = {'x': random.sample(ALL_QUADS, 4), 'o': None}
    
    # Mixed sizes
    rulesets['mixed_3_4'] = {'x': [(0,1,2), (3,4,5), (0,2,6,8)], 'o': None}
    
    # === ASYMMETRIC GAMES (X and O have different rules!) ===
    
    # X wins with rows, O wins with columns
    rulesets['asym_rows_vs_cols'] = {'x': ROWS, 'o': COLS}
    
    # X wins with standard, O only with diagonals
    rulesets['asym_standard_vs_diags'] = {'x': STANDARD, 'o': DIAGS}
    
    # X has 8 ways to win, O has only 2
    rulesets['asym_8_vs_2'] = {'x': STANDARD, 'o': [(0,4,8), (2,4,6)]}
    
    # Random asymmetric
    random.seed(456)
    x_rules = random.sample(ALL_TRIPLES, 5)
    o_rules = random.sample(ALL_TRIPLES, 3)
    rulesets['asym_random'] = {'x': x_rules, 'o': o_rules}
    
    return rulesets


# =============================================================================
# ADAPTIVE LEARNER FOR CHAOS
# =============================================================================

class AdaptiveLearner:
    """
    Learner that adapts to unknown rule sizes and asymmetry.
    
    Features:
    - Detects pattern sizes automatically
    - Handles asymmetric games
    - Multiple pattern size support
    """
    
    def __init__(self, pattern_sizes: List[int] = None, min_support: int = 2):
        self.pattern_sizes = pattern_sizes or [3]  # Default to 3
        self.min_support = min_support
        
        # Track patterns by size and player
        self.pattern_counts = defaultdict(int)
        self.pattern_labels = defaultdict(lambda: defaultdict(int))
        
        # Discovered win lines (separate for X and O for asymmetric games)
        self.x_lines = set()
        self.o_lines = set()
        
        self.label_counts = defaultdict(int)
        self.observations = 0
        
        # Detect if game is asymmetric
        self.x_line_positions = set()
        self.o_line_positions = set()
    
    def _is_valid_parity(self, board: str) -> bool:
        x = board.count('1')
        o = board.count('2')
        return x == o or x == o + 1
    
    def _get_homogeneous_patterns(self, board: str, player_val: str):
        """Get all homogeneous patterns for a player."""
        for size in self.pattern_sizes:
            for positions in combinations(range(9), size):
                values = [board[p] for p in positions]
                if all(v == player_val for v in values):
                    yield positions
    
    def observe(self, board: str, label: str):
        self.observations += 1
        self.label_counts[label] += 1
        
        if label == 'error':
            return
        
        # Track patterns for both players
        for player_val, player_label, lines_set, pos_set in [
            ('1', 'winX', self.x_lines, self.x_line_positions),
            ('2', 'winO', self.o_lines, self.o_line_positions)
        ]:
            for positions in self._get_homogeneous_patterns(board, player_val):
                key = (positions, player_val)
                
                self.pattern_counts[key] += 1
                self.pattern_labels[key][label] += 1
                
                total = self.pattern_counts[key]
                
                if total >= self.min_support:
                    # Check if this is a win pattern for this player
                    if player_val == '1':
                        # X line: 100% precision for winX
                        win_count = self.pattern_labels[key].get('winX', 0)
                        if win_count == total:
                            lines_set.add(positions)
                            pos_set.add(positions)
                        else:
                            lines_set.discard(positions)
                    else:
                        # O line: 100% for (winX OR winO) - game ending
                        win_count = sum(self.pattern_labels[key].get(l, 0) 
                                       for l in ('winX', 'winO'))
                        if win_count == total:
                            lines_set.add(positions)
                            pos_set.add(positions)
                        else:
                            lines_set.discard(positions)
    
    def predict(self, board: str) -> str:
        if not self._is_valid_parity(board):
            return 'error'
        
        # Check X wins
        x_wins = False
        for positions in self.x_lines:
            if all(board[p] == '1' for p in positions):
                x_wins = True
                break
        
        # Check O wins
        o_wins = False
        for positions in self.o_lines:
            if all(board[p] == '2' for p in positions):
                o_wins = True
                break
        
        if x_wins and o_wins:
            return 'error'
        if x_wins:
            return 'winX'
        if o_wins:
            return 'winO'
        
        if '0' not in board:
            return 'draw'
        
        return 'ok'
    
    def is_asymmetric(self) -> bool:
        """Detect if the game appears asymmetric."""
        return self.x_line_positions != self.o_line_positions


def test_chaos():
    """Test learner against chaos rulesets."""
    print("="*70)
    print("CHAOS RULESET CHALLENGE")
    print("="*70)
    
    rulesets = generate_chaos_rulesets()
    
    results = []
    
    for name, config in rulesets.items():
        x_lines = config['x']
        o_lines = config['o']  # None means same as X
        
        # Determine pattern sizes
        sizes = set(len(l) for l in x_lines)
        if o_lines:
            sizes.update(len(l) for l in o_lines)
        
        # Create oracle
        oracle = ChaosOracle(x_win_lines=x_lines, o_win_lines=o_lines)
        
        # Create learner with appropriate pattern sizes
        learner = AdaptiveLearner(pattern_sizes=list(sizes), min_support=2)
        
        # Train and test
        correct = 0
        checkpoints = {}
        
        i = 0
        while True:
            result = oracle.random_board(unique=True)
            if result is None:
                break
            
            board, true_label = result
            pred = learner.predict(board)
            
            if pred == true_label:
                correct += 1
            
            learner.observe(board, true_label)
            i += 1
            
            if i in [100, 500, 1000]:
                checkpoints[i] = correct / i
        
        final_acc = correct / i if i > 0 else 0
        
        # Check line discovery
        expected_x = len(x_lines)
        expected_o = len(o_lines) if o_lines else expected_x
        found_x = len(learner.x_lines)
        found_o = len(learner.o_lines)
        
        results.append({
            'name': name,
            'states': i,
            'final': final_acc,
            '@100': checkpoints.get(100, 0),
            '@500': checkpoints.get(500, 0),
            'x_found': f"{found_x}/{expected_x}",
            'o_found': f"{found_o}/{expected_o}",
            'asym': not oracle.is_symmetric,
        })
    
    # Print results table
    print(f"\n{'Ruleset':<22} {'States':>6} {'@100':>6} {'@500':>6} {'Final':>6} {'X':>5} {'O':>5} {'Asym':>5}")
    print("-"*75)
    
    for r in results:
        asym = "Y" if r['asym'] else ""
        print(f"{r['name']:<22} {r['states']:>6} {r['@100']:>6.1%} {r['@500']:>6.1%} {r['final']:>6.1%} {r['x_found']:>5} {r['o_found']:>5} {asym:>5}")
    
    # Summary
    print("\n--- Summary ---")
    avg_final = sum(r['final'] for r in results) / len(results)
    avg_100 = sum(r['@100'] for r in results if r['@100'] > 0) / len([r for r in results if r['@100'] > 0])
    
    print(f"Average @100: {avg_100:.1%}")
    print(f"Average final: {avg_final:.1%}")
    
    perfect = [r for r in results if r['final'] >= 0.99]
    print(f"Perfect (99%+): {len(perfect)}/{len(results)}")


if __name__ == "__main__":
    test_chaos()
