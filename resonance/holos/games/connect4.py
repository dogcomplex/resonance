"""
holos/games/connect4.py - Connect-4 Game Interface for HOLOS

This implements GameInterface for Connect-4 (Layer 0).

The game:
- States: 7x6 board positions with X/O pieces + turn
- Successors: Legal column drops
- Predecessors: Positions that could lead here (lift pieces)
- Boundary: Terminal positions (win/draw) - no tablebase needed
- Values: Win(+1), Draw(0), Loss(-1)
- Propagation: Minimax (X maximizes, O minimizes)

Key insight: Connect-4 is BOUNDED for HOLOS because:
- Lower bound: Terminal positions (4-in-a-row or full board)
- Upper bound: Empty board (or configurable piece count)
- Forward: Moves add pieces
- Backward: Unmoves remove pieces

Connect-4 is a solved game: First player (X) wins with perfect play.
This implementation allows HOLOS to rediscover this through bidirectional search.
"""

import random
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
from collections import defaultdict

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from holos.holos import GameInterface


# ============================================================
# CONNECT-4 STATE
# ============================================================

class C4State:
    """
    Compact Connect-4 state representation.

    Board is stored as 7 columns, each a string of 6 characters.
    Bottom is index 0, top is index 5.
    Characters: '.' (empty), 'X' (first player), 'O' (second player)
    """
    __slots__ = ['cols', 'turn', '_hash', '_board']

    def __init__(self, cols: Tuple[str, ...] = None, turn: str = 'X'):
        if cols is None:
            cols = tuple('.' * 6 for _ in range(7))
        self.cols = cols
        self.turn = turn
        self._hash = None
        self._board = None

    def __hash__(self):
        if self._hash is None:
            # Canonical hash: min of board and its horizontal mirror
            mirror = tuple(reversed(self.cols))
            self._hash = hash((min(self.cols, mirror), self.turn))
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def height(self, col: int) -> int:
        """Height of pieces in column (0-6)"""
        return 6 - self.cols[col].count('.')

    def can_play(self, col: int) -> bool:
        """Can a piece be dropped in this column?"""
        return self.cols[col][5] == '.'

    def play(self, col: int) -> Optional['C4State']:
        """Drop a piece in column, return new state or None if illegal"""
        if not self.can_play(col):
            return None
        h = self.height(col)
        new_col = self.cols[col][:h] + self.turn + self.cols[col][h+1:]
        new_cols = list(self.cols)
        new_cols[col] = new_col
        return C4State(tuple(new_cols), 'O' if self.turn == 'X' else 'X')

    def unplay(self, col: int) -> Optional['C4State']:
        """Remove top piece from column (for predecessor generation)"""
        h = self.height(col)
        if h == 0:
            return None
        # The piece being removed must be the OPPONENT's (since they just played)
        opponent = 'O' if self.turn == 'X' else 'X'
        if self.cols[col][h-1] != opponent:
            return None
        new_col = self.cols[col][:h-1] + '.' + self.cols[col][h:]
        new_cols = list(self.cols)
        new_cols[col] = new_col
        return C4State(tuple(new_cols), opponent)

    def get(self, col: int, row: int) -> Optional[str]:
        """Get piece at position (None if empty)"""
        if 0 <= col < 7 and 0 <= row < 6:
            c = self.cols[col][row]
            return c if c != '.' else None
        return None

    def get_valid_moves(self) -> List[int]:
        """Get list of valid columns to play"""
        return [c for c in range(7) if self.can_play(c)]

    def piece_count(self) -> int:
        """Total number of pieces on board"""
        return sum(c.count('X') + c.count('O') for c in self.cols)

    def x_count(self) -> int:
        """Number of X pieces"""
        return sum(c.count('X') for c in self.cols)

    def o_count(self) -> int:
        """Number of O pieces"""
        return sum(c.count('O') for c in self.cols)

    def check_win(self) -> Optional[str]:
        """Check for winner. Returns 'X', 'O', or None"""
        for col in range(7):
            for row in range(6):
                p = self.get(col, row)
                if p is None:
                    continue
                # Check 4 directions: right, up, up-right, up-left
                # Horizontal (right)
                if col <= 3 and all(self.get(col+i, row) == p for i in range(4)):
                    return p
                # Vertical (up)
                if row <= 2 and all(self.get(col, row+i) == p for i in range(4)):
                    return p
                # Diagonal (up-right)
                if col <= 3 and row <= 2 and all(self.get(col+i, row+i) == p for i in range(4)):
                    return p
                # Diagonal (up-left)
                if col >= 3 and row <= 2 and all(self.get(col-i, row+i) == p for i in range(4)):
                    return p
        return None

    def is_terminal(self) -> bool:
        """Is game over (win or draw)?"""
        return self.check_win() is not None or self.piece_count() == 42

    def terminal_value(self) -> int:
        """Get terminal value: +1 (X wins), -1 (O wins), 0 (draw)"""
        w = self.check_win()
        if w == 'X':
            return 1
        elif w == 'O':
            return -1
        else:
            return 0  # Draw

    def display(self) -> str:
        """Return string representation of board"""
        lines = []
        lines.append(f"Turn: {self.turn}")
        lines.append("+-" + "-" * 14 + "+")
        for row in range(5, -1, -1):
            row_str = "| "
            for col in range(7):
                c = self.cols[col][row]
                row_str += c + " "
            row_str += "|"
            lines.append(row_str)
        lines.append("+-" + "-" * 14 + "+")
        lines.append("  0 1 2 3 4 5 6")
        return "\n".join(lines)

    def to_compact(self) -> Tuple[Tuple[str, ...], str]:
        """Compact representation for serialization"""
        return (self.cols, self.turn)

    @staticmethod
    def from_compact(data: Tuple[Tuple[str, ...], str]) -> 'C4State':
        """Restore from compact representation"""
        return C4State(data[0], data[1])


# ============================================================
# CONNECT-4 VALUE
# ============================================================

@dataclass(frozen=True)
class C4Value:
    """Connect-4 game-theoretic value"""
    value: int  # +1 (X wins), 0 (Draw), -1 (O wins)

    def __repr__(self):
        return {1: "X-Win", 0: "Draw", -1: "O-Win"}.get(self.value, f"Value({self.value})")


# ============================================================
# CONNECT-4 FEATURES (Equivalence Classes)
# ============================================================

@dataclass(frozen=True)
class C4Features:
    """
    Equivalence class features for Connect-4.

    Positions with same features often have same outcomes.
    Used for equivalence-based propagation.
    """
    x_count: int
    o_count: int
    x_threats: int  # Number of 3-in-a-row with open cell
    o_threats: int
    height_profile: Tuple[int, ...]  # Sorted column heights
    turn: str

    def __hash__(self):
        return hash((self.x_count, self.o_count, self.x_threats,
                     self.o_threats, self.height_profile, self.turn))


def count_threats(state: C4State, player: str) -> int:
    """Count number of winning threats (3-in-a-row with 1 empty)"""
    threats = 0
    for col in range(7):
        for row in range(6):
            # Check 4 directions
            for dc, dr in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                # Check if pattern fits in board
                end_col = col + 3 * dc
                end_row = row + 3 * dr
                if not (0 <= end_col < 7 and 0 <= end_row < 6):
                    continue
                if dr == -1 and row - 3 * abs(dr) < 0:
                    continue

                # Get the 4 cells
                window = []
                for i in range(4):
                    c = col + i * dc
                    r = row + i * dr
                    if 0 <= r < 6:
                        window.append(state.get(c, r))
                    else:
                        window.append('_')  # Out of bounds marker

                # Count as threat if 3 player pieces + 1 empty
                if window.count(player) == 3 and window.count(None) == 1:
                    threats += 1

    return threats


def extract_features(state: C4State) -> C4Features:
    """Extract equivalence features from state"""
    heights = tuple(sorted(state.height(c) for c in range(7)))

    return C4Features(
        x_count=state.x_count(),
        o_count=state.o_count(),
        x_threats=count_threats(state, 'X'),
        o_threats=count_threats(state, 'O'),
        height_profile=heights,
        turn=state.turn
    )


# ============================================================
# CONNECT-4 GAME INTERFACE
# ============================================================

class Connect4Game(GameInterface[C4State, C4Value]):
    """
    Connect-4 interface for HOLOS.

    Boundaries:
    - Lower: Terminal positions (win or draw)
    - Upper: Configurable max pieces (default: 42 = full board)

    Unlike chess, Connect-4 doesn't need external tablebases.
    Terminal positions ARE the boundary - we know their values directly.
    """

    def __init__(self, max_pieces: int = 42, min_pieces: int = 0):
        """
        Initialize Connect-4 game.

        Args:
            max_pieces: Maximum pieces to consider (default 42 = full game)
            min_pieces: Minimum pieces for "near-end" searches
        """
        self.max_pieces = max_pieces
        self.min_pieces = min_pieces

    def hash_state(self, state: C4State) -> int:
        """Hash a state for deduplication"""
        return hash(state)

    def get_successors(self, state: C4State) -> List[Tuple[C4State, int]]:
        """
        Get successor states (legal moves).
        Returns list of (child_state, move) where move is column number.
        """
        if state.piece_count() >= self.max_pieces:
            return []

        if state.is_terminal():
            return []

        successors = []
        # Center columns first (better move ordering)
        for col in [3, 2, 4, 1, 5, 0, 6]:
            if state.can_play(col):
                child = state.play(col)
                successors.append((child, col))

        return successors

    def get_predecessors(self, state: C4State) -> List[Tuple[C4State, int]]:
        """
        Get predecessor states (positions that could lead here).
        This is "unplaying" - removing the last piece placed.
        """
        if state.piece_count() <= self.min_pieces:
            return []

        predecessors = []
        for col in range(7):
            pred = state.unplay(col)
            if pred is not None:
                predecessors.append((pred, col))

        return predecessors

    def is_boundary(self, state: C4State) -> bool:
        """
        Is this a boundary state?
        For Connect-4: terminal positions are boundaries.
        """
        return state.is_terminal()

    def get_boundary_value(self, state: C4State) -> Optional[C4Value]:
        """Get value for boundary (terminal) state"""
        if not state.is_terminal():
            return None
        return C4Value(state.terminal_value())

    def is_terminal(self, state: C4State) -> Tuple[bool, Optional[C4Value]]:
        """Check if state is terminal"""
        if state.is_terminal():
            return True, C4Value(state.terminal_value())
        return False, None

    def propagate_value(self, state: C4State,
                        child_values: List[C4Value]) -> Optional[C4Value]:
        """
        Minimax propagation.
        X (first player) maximizes, O minimizes.

        If state is None (called from reverse propagation), we cannot
        determine whose turn it is, so we return None to skip.
        """
        if not child_values:
            return None

        # State required for minimax - if None, skip reverse propagation
        if state is None:
            return None

        values = [cv.value for cv in child_values]

        if state.turn == 'X':
            # X is maximizing - if any child is a win, X wins
            if 1 in values:
                return C4Value(1)
            # Can't determine draw/loss without all children
            return None
        else:
            # O is minimizing - if any child is a loss (for X), O achieves it
            if -1 in values:
                return C4Value(-1)
            return None

    def get_features(self, state: C4State) -> C4Features:
        """Extract equivalence features"""
        return extract_features(state)

    def get_signature(self, state: C4State) -> str:
        """
        Get signature for goal matching.
        Returns piece count signature like "X10O9" (10 X's, 9 O's).
        """
        return f"X{state.x_count()}O{state.o_count()}"

    def get_lightning_successors(self, state: C4State) -> List[Tuple[C4State, int]]:
        """
        For lightning mode: prioritize center columns and winning moves.
        """
        if state.is_terminal():
            return []

        successors = []
        # Center priority
        for col in [3, 2, 4, 1, 5, 0, 6]:
            if state.can_play(col):
                child = state.play(col)
                successors.append((child, col))

        # Sort by winning potential
        def score_move(item):
            child, col = item
            if child.check_win() == state.turn:
                return 1000  # Winning move
            # Prefer center
            return 10 - abs(col - 3)

        successors.sort(key=score_move, reverse=True)
        return successors[:4]  # Limit branching

    def get_lightning_predecessors(self, state: C4State) -> List[Tuple[C4State, int]]:
        """For backward lightning: same as regular predecessors"""
        return self.get_predecessors(state)

    def score_for_lightning(self, state: C4State, move: int) -> float:
        """Score a move for lightning prioritization"""
        # Prefer center columns
        return 10.0 - abs(move - 3)

    def apply_move(self, state: C4State, move: int) -> Optional[C4State]:
        """Apply a move (column drop) to get successor state"""
        return state.play(move)

    def generate_boundary_seeds(self, template: C4State, count: int = 100) -> List[C4State]:
        """
        Generate terminal positions for backward wave seeding.

        Strategy: Generate random games and collect terminal positions.
        These serve as the "boundary" for backward wave expansion.
        """
        positions = []
        seen = set()

        for _ in range(count * 20):
            if len(positions) >= count:
                break

            # Play random game until terminal
            state = C4State()
            while not state.is_terminal():
                moves = state.get_valid_moves()
                if not moves:
                    break
                col = random.choice(moves)
                state = state.play(col)

            if state.is_terminal():
                h = hash(state)
                if h not in seen:
                    seen.add(h)
                    positions.append(state)

        print(f"Generated {len(positions)} terminal positions for backward seeding")
        return positions


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_solver(max_pieces: int = 42):
    """Create a HOLOS solver for Connect-4"""
    from holos.holos import HOLOSSolver

    game = Connect4Game(max_pieces=max_pieces)
    solver = HOLOSSolver(game, name=f"connect4_{max_pieces}")
    return solver, game


def random_position(num_pieces: int) -> Optional[C4State]:
    """Generate random position with given number of pieces"""
    state = C4State()
    for i in range(num_pieces):
        moves = state.get_valid_moves()
        if not moves:
            break
        col = random.choice(moves)
        child = state.play(col)
        if child is None:
            break
        if child.is_terminal():
            break
        state = child
    return state if state.piece_count() == num_pieces else None


def play_game(state: C4State, hologram) -> List[Tuple[C4State, int, C4Value]]:
    """
    Play out a game using hologram for move selection.
    Returns list of (state, move, value) for each ply.
    """
    game = Connect4Game()
    history = []

    while not state.is_terminal():
        h = game.hash_state(state)
        value = hologram.query(h)

        if value is None:
            # Not in hologram - pick random move
            moves = state.get_valid_moves()
            if not moves:
                break
            col = random.choice(moves)
        else:
            # Find best move from hologram
            best_col = None
            best_value = None

            for child, col in game.get_successors(state):
                ch = game.hash_state(child)
                cv = hologram.query(ch)
                if cv is not None:
                    if best_value is None:
                        best_value = cv.value
                        best_col = col
                    elif state.turn == 'X' and cv.value > best_value:
                        best_value = cv.value
                        best_col = col
                    elif state.turn == 'O' and cv.value < best_value:
                        best_value = cv.value
                        best_col = col

            col = best_col if best_col is not None else random.choice(state.get_valid_moves())

        history.append((state, col, value))
        state = state.play(col)

    return history
