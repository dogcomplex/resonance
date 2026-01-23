"""
game_chess.py - Chess Game Interface for HOLOS

This implements GameInterface for chess endgames.
It defines how chess positions work as a game for HOLOS to solve.
"""

from typing import List, Tuple, Optional, Any
from dataclasses import dataclass

from holos_core import GameInterface

# Import chess primitives from fractal_holos3
from fractal_holos3 import (
    ChessState, SyzygyProbe,
    generate_moves, apply_move, generate_predecessors,
    is_terminal, in_check, extract_features,
    is_white, piece_type, PIECE_VALUES
)


@dataclass(frozen=True)
class ChessValue:
    """Chess game-theoretic value: Win(+1), Draw(0), Loss(-1)"""
    value: int  # +1, 0, -1

    def __repr__(self):
        if self.value == 1:
            return "Win"
        elif self.value == -1:
            return "Loss"
        else:
            return "Draw"


class ChessGame(GameInterface[ChessState, ChessValue]):
    """
    Chess endgame interface for HOLOS.

    Boundaries:
    - Lower: 7-piece positions (Syzygy tablebases)
    - Upper: Configurable (e.g., 8-piece, 9-piece)

    The game is bounded: positions outside the piece-count range are invalid.
    """

    def __init__(self, syzygy_path: str = "./syzygy",
                 min_pieces: int = 7, max_pieces: int = 8):
        self.syzygy = SyzygyProbe(syzygy_path)
        self.min_pieces = min_pieces
        self.max_pieces = max_pieces

    def hash_state(self, state: ChessState) -> int:
        return hash(state)

    def get_successors(self, state: ChessState) -> List[Tuple[ChessState, Any]]:
        """Legal moves from this position"""
        # Don't expand past max pieces (no reverse captures here)
        if state.piece_count() > self.max_pieces:
            return []

        moves = generate_moves(state)
        successors = []
        for move in moves:
            child = apply_move(state, move)
            # Only include if within bounds
            if self.min_pieces <= child.piece_count() <= self.max_pieces:
                successors.append((child, move))
            elif child.piece_count() < self.min_pieces:
                # Reached boundary via capture - still valid
                successors.append((child, move))

        return successors

    def get_predecessors(self, state: ChessState) -> List[Tuple[ChessState, Any]]:
        """Positions that could lead to this via legal move (uncapture)"""
        # Don't generate predecessors past max pieces
        if state.piece_count() >= self.max_pieces:
            return []

        preds = generate_predecessors(state, max_uncaptures=3)
        result = []
        for pred in preds:
            if pred.piece_count() <= self.max_pieces:
                result.append((pred, None))  # Move info not tracked for predecessors
        return result

    def is_boundary(self, state: ChessState) -> bool:
        """Is this position on the Syzygy boundary?"""
        return state.piece_count() <= self.min_pieces

    def get_boundary_value(self, state: ChessState) -> Optional[ChessValue]:
        """Query Syzygy for value"""
        if not self.is_boundary(state):
            return None

        val = self.syzygy.probe(state)
        if val is not None:
            return ChessValue(val)
        return None

    def is_terminal(self, state: ChessState) -> Tuple[bool, Optional[ChessValue]]:
        """Check for checkmate/stalemate"""
        moves = generate_moves(state)
        is_term, value = is_terminal(state, moves)
        if is_term:
            return True, ChessValue(value) if value is not None else ChessValue(0)
        return False, None

    def propagate_value(self, state: ChessState,
                        child_values: List[ChessValue]) -> Optional[ChessValue]:
        """
        Minimax propagation.

        White wants max value (+1), Black wants min value (-1).
        """
        if not child_values:
            return None

        values = [cv.value for cv in child_values]

        if state.turn == 'w':
            # White to move: picks max
            if 1 in values:
                return ChessValue(1)  # White can win
            # Can't confirm draw/loss without ALL children
            return None
        else:
            # Black to move: picks min
            if -1 in values:
                return ChessValue(-1)  # Black can win
            return None

    def get_features(self, state: ChessState) -> Any:
        """Extract equivalence class features"""
        return extract_features(state)

    def get_lightning_successors(self, state: ChessState) -> List[Tuple[ChessState, Any]]:
        """For lightning: only captures (reduce piece count toward boundary)"""
        moves = generate_moves(state)
        captures = [m for m in moves if m[2] is not None]

        successors = []
        for move in captures:
            child = apply_move(state, move)
            successors.append((child, move))

        return successors

    def score_for_lightning(self, state: ChessState, move: Any) -> float:
        """Score captures by MVV-LVA (Most Valuable Victim - Least Valuable Attacker)"""
        if move[2] is None:
            return 0.0
        return PIECE_VALUES.get(move[2], 0)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_chess_solver(syzygy_path: str = "./syzygy",
                        min_pieces: int = 7, max_pieces: int = 8):
    """Create a HOLOS solver configured for chess"""
    from holos_core import HOLOSSolver

    game = ChessGame(syzygy_path, min_pieces, max_pieces)
    solver = HOLOSSolver(game, name=f"chess_{min_pieces}to{max_pieces}")
    return solver, game


def demo_chess():
    """Demo chess solving with HOLOS"""
    from holos_core import SeedPoint, SearchMode
    from fractal_holos3 import random_position

    print("Creating chess HOLOS solver...")
    solver, game = create_chess_solver()

    # Generate some 8-piece positions to solve
    print("Generating 8-piece positions...")
    forward_states = []
    for _ in range(50):
        state = random_position("KQRRvKQRR")
        if state:
            forward_states.append(state)

    print(f"Generated {len(forward_states)} positions")

    # Generate boundary positions
    print("Generating 7-piece boundary positions...")
    backward_states = []
    for _ in range(100):
        state = random_position("KQRRvKQR")
        if state and game.syzygy.probe(state) is not None:
            backward_states.append(state)

    print(f"Generated {len(backward_states)} boundary positions")

    # Create seed points
    forward_seeds = [SeedPoint(s, SearchMode.WAVE, 1) for s in forward_states]
    backward_seeds = [SeedPoint(s, SearchMode.WAVE, 1) for s in backward_states]

    # Solve!
    hologram = solver.solve(forward_seeds, backward_seeds, max_iterations=20)

    print(f"\nSolved {len(hologram.solved)} positions")
    print(f"Connections: {len(solver.connections)}")
    print(f"Stats: {solver.stats}")


if __name__ == "__main__":
    demo_chess()
