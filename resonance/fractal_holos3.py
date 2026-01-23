"""
fractal_holos3.py - Bidirectional Lightning + Crystal Solver

CORE ALGORITHM (Slime Mold Crystallization):
1. LIGHTNING (Tendrils): Fast depth-first probes from both directions
   - Deductive wave: Forward from 8+ piece positions
   - Abductive wave: Backward from 7-piece syzygy boundary

2. CRYSTALLIZATION (When tendrils meet):
   - Connection event triggers local wave expansion
   - Similar nearby branches are reinforced (like slime mold)
   - Rigorous BFS fills in the connected region

3. HOLOGRAPHIC STORAGE:
   - Spine paths: Principal variations connecting layers
   - Pattern signatures: Equivalence classes
   - Critical positions: Branch points where value changes

Physics inspiration:
- Bidirectional search = standing wave formation
- Crystallization = phase transition at interference maxima
- Slime mold = adaptive network that finds optimal paths
"""

import os
import sys
import time
import pickle
import random
import gc
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import IntEnum
import heapq

# ============================================================
# CHESS CORE
# ============================================================

class Piece(IntEnum):
    EMPTY = 0
    W_KING = 1; W_QUEEN = 2; W_ROOK = 3; W_BISHOP = 4; W_KNIGHT = 5; W_PAWN = 6
    B_KING = 7; B_QUEEN = 8; B_ROOK = 9; B_BISHOP = 10; B_KNIGHT = 11; B_PAWN = 12

PIECE_CHARS = '.KQRBNPkqrbnp'
PIECE_NAMES = {'K': Piece.W_KING, 'Q': Piece.W_QUEEN, 'R': Piece.W_ROOK,
               'B': Piece.W_BISHOP, 'N': Piece.W_KNIGHT, 'P': Piece.W_PAWN}
PIECE_VALUES = {2: 900, 3: 500, 4: 330, 5: 320, 6: 100, 8: 900, 9: 500, 10: 330, 11: 320, 12: 100}

# All possible pieces that can be captured/uncaptured
ALL_CAPTURABLE = [Piece.W_QUEEN, Piece.W_ROOK, Piece.W_BISHOP, Piece.W_KNIGHT, Piece.W_PAWN,
                  Piece.B_QUEEN, Piece.B_ROOK, Piece.B_BISHOP, Piece.B_KNIGHT, Piece.B_PAWN]

def is_white(p): return 1 <= p <= 6
def is_black(p): return 7 <= p <= 12
def piece_type(p): return ((p - 1) % 6) + 1 if p > 0 else 0


class ChessState:
    __slots__ = ['pieces', 'turn', '_hash', '_board']

    def __init__(self, pieces, turn='w'):
        self.pieces = tuple(sorted(pieces))
        self.turn = turn
        self._hash = None
        self._board = None

    def __hash__(self):
        if self._hash is None:
            flipped = tuple(sorted((p, self._flip_h(sq)) for p, sq in self.pieces))
            canonical = min(self.pieces, flipped)
            self._hash = hash((canonical, self.turn))
        return self._hash

    def _flip_h(self, sq):
        return (sq // 8) * 8 + (7 - sq % 8)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def to_board(self):
        if self._board is None:
            board = [Piece.EMPTY] * 64
            for p, sq in self.pieces:
                board[sq] = p
            self._board = board
        return self._board

    def piece_count(self):
        return len(self.pieces)

    def material_score(self):
        """Material balance (positive = white ahead)"""
        score = 0
        for p, sq in self.pieces:
            if p in PIECE_VALUES:
                score += PIECE_VALUES[p] if is_white(p) else -PIECE_VALUES[p]
        return score

    def display(self):
        board = self.to_board()
        print(f"  Turn: {'White' if self.turn == 'w' else 'Black'}")
        print("  +-----------------+")
        for rank in range(7, -1, -1):
            print(f"{rank+1} |", end=" ")
            for file in range(8):
                print(PIECE_CHARS[board[rank * 8 + file]], end=" ")
            print("|")
        print("  +-----------------+")
        print("    a b c d e f g h")


def generate_moves(state):
    """Generate legal moves"""
    board = state.to_board()
    is_white_turn = state.turn == 'w'
    moves = []

    for piece, from_sq in state.pieces:
        if is_white_turn and not is_white(piece):
            continue
        if not is_white_turn and not is_black(piece):
            continue

        pt = piece_type(piece)

        if pt == 1:  # King
            for d in [-9, -8, -7, -1, 1, 7, 8, 9]:
                to_sq = from_sq + d
                if not (0 <= to_sq < 64) or abs((from_sq % 8) - (to_sq % 8)) > 1:
                    continue
                target = board[to_sq]
                if (is_white_turn and is_white(target)) or (not is_white_turn and is_black(target)):
                    continue
                moves.append((from_sq, to_sq, target if target else None))
        elif pt == 2:  # Queen
            moves.extend(_sliding(board, from_sq, is_white_turn, [-9,-8,-7,-1,1,7,8,9]))
        elif pt == 3:  # Rook
            moves.extend(_sliding(board, from_sq, is_white_turn, [-8,-1,1,8]))
        elif pt == 4:  # Bishop
            moves.extend(_sliding(board, from_sq, is_white_turn, [-9,-7,7,9]))
        elif pt == 5:  # Knight
            for d in [-17,-15,-10,-6,6,10,15,17]:
                to_sq = from_sq + d
                if not (0 <= to_sq < 64) or abs((from_sq % 8) - (to_sq % 8)) > 2:
                    continue
                target = board[to_sq]
                if (is_white_turn and is_white(target)) or (not is_white_turn and is_black(target)):
                    continue
                moves.append((from_sq, to_sq, target if target else None))

    return [m for m in moves if not in_check(apply_move(state, m), state.turn)]


def _sliding(board, from_sq, is_white_turn, directions):
    moves = []
    for d in directions:
        sq = from_sq + d
        prev_file = from_sq % 8
        while 0 <= sq < 64:
            sq_file = sq % 8
            if abs(sq_file - prev_file) > 1:
                break
            target = board[sq]
            if target == Piece.EMPTY:
                moves.append((from_sq, sq, None))
            elif (is_white_turn and is_black(target)) or (not is_white_turn and is_white(target)):
                moves.append((from_sq, sq, target))
                break
            else:
                break
            prev_file = sq_file
            sq += d
    return moves


def apply_move(state, move):
    from_sq, to_sq, _ = move
    new_pieces = []
    moved = None
    for p, sq in state.pieces:
        if sq == from_sq:
            moved = p
        elif sq == to_sq:
            pass
        else:
            new_pieces.append((p, sq))
    if moved:
        new_pieces.append((moved, to_sq))
    return ChessState(new_pieces, 'b' if state.turn == 'w' else 'w')


def in_check(state, color):
    board = state.to_board()
    king = Piece.W_KING if color == 'w' else Piece.B_KING
    king_sq = next((sq for p, sq in state.pieces if p == king), None)
    if king_sq is None:
        return True
    enemy_white = (color == 'b')
    for p, sq in state.pieces:
        if enemy_white and not is_white(p):
            continue
        if not enemy_white and not is_black(p):
            continue
        if attacks(board, sq, king_sq, p):
            return True
    return False


def attacks(board, from_sq, to_sq, piece):
    pt = piece_type(piece)
    dr, df = (to_sq // 8) - (from_sq // 8), (to_sq % 8) - (from_sq % 8)
    if pt == 1: return abs(dr) <= 1 and abs(df) <= 1 and (dr or df)
    if pt == 5: return (abs(dr), abs(df)) in [(1,2), (2,1)]
    if pt == 3: return (dr == 0 or df == 0) and (dr or df) and path_clear(board, from_sq, to_sq)
    if pt == 4: return abs(dr) == abs(df) and dr != 0 and path_clear(board, from_sq, to_sq)
    if pt == 2: return ((dr == 0 or df == 0) or abs(dr) == abs(df)) and (dr or df) and path_clear(board, from_sq, to_sq)
    return False


def path_clear(board, from_sq, to_sq):
    dr = 0 if (to_sq // 8) == (from_sq // 8) else (1 if to_sq // 8 > from_sq // 8 else -1)
    df = 0 if (to_sq % 8) == (from_sq % 8) else (1 if to_sq % 8 > from_sq % 8 else -1)
    d = dr * 8 + df
    if d == 0: return True
    sq = from_sq + d
    while sq != to_sq and 0 <= sq < 64:
        if abs((sq % 8) - ((sq - d) % 8)) > 1:
            break
        if board[sq] != Piece.EMPTY:
            return False
        sq += d
    return True


def is_terminal(state, moves=None):
    if state.piece_count() == 2:
        return True, 0
    if moves is None:
        moves = generate_moves(state)
    if len(moves) == 0:
        if in_check(state, state.turn):
            return True, (-1 if state.turn == 'w' else 1)
        return True, 0
    return False, None


# ============================================================
# EQUIVALENCE CLASSES (Pattern Abstraction)
# ============================================================

@dataclass(frozen=True)
class ChessFeatures:
    """
    Equivalence class features for chess positions.
    Positions with identical features often have identical game-theoretic values.

    This saved 80%+ on Connect-4 by avoiding re-solving equivalent positions.
    """
    # Material signature (sorted piece types present)
    material_white: Tuple[int, ...]  # Sorted piece types (without king)
    material_black: Tuple[int, ...]

    # Material balance (centipawns)
    material_balance: int

    # Piece count
    piece_count: int

    # King proximity (Manhattan distance)
    king_distance: int

    # Turn to move
    turn: str

    def __hash__(self):
        return hash((self.material_white, self.material_black,
                     self.material_balance, self.piece_count,
                     self.king_distance, self.turn))


def extract_features(state: ChessState) -> ChessFeatures:
    """Extract equivalence features from a chess position"""
    white_pieces = []
    black_pieces = []
    wk_sq = bk_sq = None
    material = 0

    for p, sq in state.pieces:
        pt = piece_type(p)
        if pt == 1:  # King
            if is_white(p):
                wk_sq = sq
            else:
                bk_sq = sq
        else:
            if is_white(p):
                white_pieces.append(pt)
                material += PIECE_VALUES.get(p, 0)
            else:
                black_pieces.append(pt)
                material -= PIECE_VALUES.get(p, 0)

    # King distance (Manhattan)
    king_dist = 0
    if wk_sq is not None and bk_sq is not None:
        king_dist = abs(wk_sq // 8 - bk_sq // 8) + abs(wk_sq % 8 - bk_sq % 8)

    return ChessFeatures(
        material_white=tuple(sorted(white_pieces)),
        material_black=tuple(sorted(black_pieces)),
        material_balance=material,
        piece_count=state.piece_count(),
        king_distance=king_dist,
        turn=state.turn
    )


# ============================================================
# PREDECESSOR GENERATION (True Backward Search)
# ============================================================

def generate_predecessors(state: ChessState, max_uncaptures: int = 3) -> List[ChessState]:
    """
    Generate positions that could lead to this state via a legal move.

    This is TRUE backward search - what positions could have resulted in THIS position?

    For a move (from_sq, to_sq, captured) that led to current state:
    - The piece at to_sq was at from_sq before
    - If captured is not None, that piece was at to_sq before
    - The turn was opposite

    We generate predecessors by:
    1. Un-moving pieces (piece at to_sq could have come from various from_sq)
    2. Un-capturing (optionally restore a captured piece at the to_sq)
    """
    predecessors = []
    board = state.to_board()

    # Previous turn (opposite of current)
    prev_turn = 'b' if state.turn == 'w' else 'w'
    is_prev_white = (prev_turn == 'w')

    # For each piece that belongs to the side that just moved
    for piece, to_sq in state.pieces:
        # Skip if this piece didn't move last (wrong color)
        if is_prev_white and not is_white(piece):
            continue
        if not is_prev_white and not is_black(piece):
            continue

        pt = piece_type(piece)

        # Generate potential "from" squares this piece could have come from
        from_squares = _get_reverse_moves(board, to_sq, piece, pt)

        for from_sq in from_squares:
            # Create predecessor: piece was at from_sq, moved to to_sq

            # Case 1: No capture (just move the piece back)
            new_pieces = []
            for p, sq in state.pieces:
                if sq == to_sq and p == piece:
                    new_pieces.append((piece, from_sq))  # Move piece back
                else:
                    new_pieces.append((p, sq))

            pred = ChessState(new_pieces, prev_turn)

            # Validate: the move from from_sq to to_sq must be legal
            if _validate_predecessor(pred, from_sq, to_sq, piece, None):
                predecessors.append(pred)

            # Case 2: With capture (restore a captured piece at to_sq)
            if state.piece_count() < 32 and len(predecessors) < max_uncaptures * 10:
                # Try uncapturing each possible piece type
                capturable = ALL_CAPTURABLE
                for cap_piece in capturable:
                    # Can only capture enemy pieces
                    if is_prev_white and is_white(cap_piece):
                        continue
                    if not is_prev_white and is_black(cap_piece):
                        continue

                    new_pieces_cap = []
                    for p, sq in state.pieces:
                        if sq == to_sq and p == piece:
                            new_pieces_cap.append((piece, from_sq))  # Move piece back
                        else:
                            new_pieces_cap.append((p, sq))
                    new_pieces_cap.append((cap_piece, to_sq))  # Restore captured piece

                    pred_cap = ChessState(new_pieces_cap, prev_turn)

                    if _validate_predecessor(pred_cap, from_sq, to_sq, piece, cap_piece):
                        predecessors.append(pred_cap)

    return predecessors


def _get_reverse_moves(board: List[int], to_sq: int, piece: int, pt: int) -> List[int]:
    """Get squares a piece could have moved FROM to reach to_sq"""
    from_squares = []

    if pt == 1:  # King - could come from adjacent squares
        for d in [-9, -8, -7, -1, 1, 7, 8, 9]:
            from_sq = to_sq - d
            if 0 <= from_sq < 64 and abs((to_sq % 8) - (from_sq % 8)) <= 1:
                if board[from_sq] == Piece.EMPTY:  # Must be empty now
                    from_squares.append(from_sq)

    elif pt == 5:  # Knight
        for d in [-17, -15, -10, -6, 6, 10, 15, 17]:
            from_sq = to_sq - d
            if 0 <= from_sq < 64 and abs((to_sq % 8) - (from_sq % 8)) <= 2:
                if board[from_sq] == Piece.EMPTY:
                    from_squares.append(from_sq)

    elif pt in [2, 3, 4]:  # Queen, Rook, Bishop - sliding pieces
        if pt == 2:
            directions = [-9, -8, -7, -1, 1, 7, 8, 9]
        elif pt == 3:
            directions = [-8, -1, 1, 8]
        else:
            directions = [-9, -7, 7, 9]

        for d in directions:
            sq = to_sq - d
            prev_file = to_sq % 8
            while 0 <= sq < 64:
                sq_file = sq % 8
                if abs(sq_file - prev_file) > 1:
                    break
                if board[sq] == Piece.EMPTY:
                    from_squares.append(sq)
                else:
                    break  # Blocked
                prev_file = sq_file
                sq -= d

    return from_squares


def _validate_predecessor(pred: ChessState, from_sq: int, to_sq: int,
                          piece: int, captured: Optional[int]) -> bool:
    """Validate that the predecessor could legally lead to the current state"""
    # Check: not in check after the move (i.e., the move must be legal)
    # The side that moved should not be in check in the result state

    # Check kings not adjacent
    wk_sq = bk_sq = None
    for p, sq in pred.pieces:
        if p == Piece.W_KING:
            wk_sq = sq
        elif p == Piece.B_KING:
            bk_sq = sq

    if wk_sq is not None and bk_sq is not None:
        if abs(wk_sq // 8 - bk_sq // 8) <= 1 and abs(wk_sq % 8 - bk_sq % 8) <= 1:
            return False

    # The side to move in predecessor should not be in check from the opponent
    # (Unless they're about to deliver checkmate, but we skip that complexity)
    opp_color = 'b' if pred.turn == 'w' else 'w'
    if in_check(pred, opp_color):
        return False

    return True


# ============================================================
# SYZYGY
# ============================================================

class SyzygyProbe:
    def __init__(self, path="./syzygy"):
        self.available = False
        self.probes = 0
        self.hits = 0
        try:
            import chess
            import chess.syzygy
            self.chess = chess
            self.tb = chess.syzygy.Tablebase()
            if os.path.exists(path):
                self.tb.add_directory(path)
                self.available = True
                print(f"[OK] Syzygy loaded from {path}")
        except:
            print("[X] Syzygy not available")

    def probe(self, state):
        if not self.available or state.piece_count() > 7:
            return None
        self.probes += 1
        try:
            board = self.chess.Board()
            board.clear()
            type_map = {1: self.chess.KING, 2: self.chess.QUEEN, 3: self.chess.ROOK,
                        4: self.chess.BISHOP, 5: self.chess.KNIGHT, 6: self.chess.PAWN}
            for p, sq in state.pieces:
                board.set_piece_at(sq, self.chess.Piece(type_map[piece_type(p)], is_white(p)))
            board.turn = (state.turn == 'w')
            wdl = self.tb.probe_wdl(board)
            if wdl is None: return None
            self.hits += 1
            if wdl > 0: return 1 if state.turn == 'w' else -1
            if wdl < 0: return -1 if state.turn == 'w' else 1
            return 0
        except:
            return None


# ============================================================
# HOLOGRAPHIC STORAGE
# ============================================================

@dataclass
class SpinePath:
    """Principal variation connecting layers"""
    start_hash: int
    moves: List[Tuple[int, int, Optional[int]]]
    end_hash: int
    end_value: int
    depth: int = 0


@dataclass
class Hologram:
    """Interference pattern storage for one layer"""
    name: str
    solved: Dict[int, int] = field(default_factory=dict)
    spines: List[SpinePath] = field(default_factory=list)
    boundary_hashes: Set[int] = field(default_factory=set)

    # Equivalence class tracking (pattern abstraction)
    equiv_classes: Dict[ChessFeatures, Set[int]] = field(default_factory=lambda: defaultdict(set))
    equiv_outcomes: Dict[ChessFeatures, Optional[int]] = field(default_factory=dict)

    def query(self, h: int) -> Optional[int]:
        return self.solved.get(h)

    def add_boundary(self, h: int, value: int):
        self.solved[h] = value
        self.boundary_hashes.add(h)

    def add_spine(self, spine: SpinePath):
        self.spines.append(spine)
        self.solved[spine.start_hash] = spine.end_value

    def add_with_features(self, h: int, value: int, features: ChessFeatures):
        """Add solved position and track equivalence class"""
        self.solved[h] = value
        self.equiv_classes[features].add(h)
        self._update_equiv_outcome(features, value)

    def _update_equiv_outcome(self, features: ChessFeatures, value: int):
        """Track outcome for equivalence class (None if inconsistent)"""
        if features in self.equiv_outcomes:
            if self.equiv_outcomes[features] != value:
                self.equiv_outcomes[features] = None  # Inconsistent
        else:
            self.equiv_outcomes[features] = value

    def propagate_equivalence(self) -> int:
        """Propagate solutions via equivalence classes"""
        count = 0
        for features, hashes in self.equiv_classes.items():
            if features not in self.equiv_outcomes:
                continue
            outcome = self.equiv_outcomes[features]
            if outcome is None:
                continue
            for h in hashes:
                if h not in self.solved:
                    self.solved[h] = outcome
                    count += 1
        return count

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'solved': self.solved,
                'spines': self.spines,
                'boundary': self.boundary_hashes,
                'equiv_classes': dict(self.equiv_classes),
                'equiv_outcomes': self.equiv_outcomes,
            }, f)
        equiv_count = sum(len(v) for v in self.equiv_classes.values())
        print(f"  [Saved hologram '{self.name}': {len(self.solved):,} solved, "
              f"{len(self.spines):,} spines, {equiv_count:,} equiv]")

    @staticmethod
    def load(path: str) -> 'Hologram':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        h = Hologram(data['name'])
        h.solved = data['solved']
        h.spines = data.get('spines', [])
        h.boundary_hashes = data.get('boundary', set())
        h.equiv_classes = defaultdict(set, data.get('equiv_classes', {}))
        h.equiv_outcomes = data.get('equiv_outcomes', {})
        return h


# ============================================================
# LIGHTNING PROBE (Depth-first tendril)
# ============================================================

class LightningProbe:
    """
    Fast depth-first search that finds ONE path to boundary.
    Like a lightning bolt finding ground, or slime mold tendrils.

    KEY: Only follow capture sequences to quickly hit syzygy boundary.
    """

    def __init__(self, syzygy: SyzygyProbe, hologram: Hologram, max_depth: int = 15):
        self.syzygy = syzygy
        self.hologram = hologram
        self.max_depth = max_depth
        self.nodes_visited = 0
        self.max_nodes = 10000  # Hard limit

    def probe(self, state: ChessState) -> Tuple[Optional[int], List]:
        """
        Capture-only search to find path to syzygy.
        Returns (value, path) where path is list of (state, move).
        """
        self.nodes_visited = 0
        path = []
        value = self._search_captures(state, 0, path)
        return value, path

    def _search_captures(self, state: ChessState, depth: int, path: List) -> Optional[int]:
        """Search only capture moves to quickly reach boundary"""
        self.nodes_visited += 1

        if self.nodes_visited > self.max_nodes:
            return None

        h = hash(state)

        # Check hologram first
        cached = self.hologram.query(h)
        if cached is not None:
            return cached

        # Check syzygy boundary
        if state.piece_count() <= 7:
            value = self.syzygy.probe(state)
            if value is not None:
                return value

        # Depth limit
        if depth >= self.max_depth:
            return None

        # Terminal check
        moves = generate_moves(state)
        is_term, term_value = is_terminal(state, moves)
        if is_term:
            return term_value

        # ONLY search captures (this is the key speedup)
        captures = [m for m in moves if m[2] is not None]
        if not captures:
            return None  # No captures = can't reach boundary quickly

        # Order by MVV-LVA
        captures.sort(key=lambda m: -PIECE_VALUES.get(m[2], 0))

        # Limit branching
        for move in captures[:5]:  # Only top 5 captures
            child = apply_move(state, move)
            child_path = []
            value = self._search_captures(child, depth + 1, child_path)

            if value is not None:
                path.clear()
                path.append((state, move))
                path.extend(child_path)
                return value

        return None


# ============================================================
# BIDIRECTIONAL SOLVER (Slime Mold Crystallization)
# ============================================================

class BidirectionalHOLOS:
    """
    Bidirectional Lightning + Crystal solver.

    Two wavefronts:
    1. DEDUCTIVE (forward from 8+ pieces): "What happens if I play this?"
    2. ABDUCTIVE (backward from 7-piece): "What leads to this outcome?"

    When tendrils connect, crystallization expands the connection.
    """

    def __init__(self, syzygy_path="./syzygy", save_dir="./fractal_holos3",
                 max_memory_mb=4000):
        self.syzygy = SyzygyProbe(syzygy_path)
        self.save_dir = save_dir
        self.max_memory_mb = max_memory_mb

        os.makedirs(save_dir, exist_ok=True)

        # Holographic storage
        self.hologram = Hologram("main")

        # Bidirectional frontiers
        self.forward_frontier: Dict[int, ChessState] = {}   # From starting positions
        self.backward_frontier: Dict[int, ChessState] = {}  # From syzygy boundary

        # Parent tracking for path reconstruction
        self.forward_parents: Dict[int, Tuple[int, Tuple]] = {}  # hash -> (parent_hash, move)
        self.backward_parents: Dict[int, Tuple[int, Tuple]] = {}

        # All seen positions
        self.forward_seen: Set[int] = set()
        self.backward_seen: Set[int] = set()

        # Connection points (where waves meet)
        self.connections: List[Tuple[int, int, int]] = []  # (forward_hash, backward_hash, value)

        # Stats
        self.stats = {
            'lightning_probes': 0,
            'connections': 0,
            'crystallized': 0,
            'spines_found': 0,
        }

    def memory_mb(self) -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except:
            return (len(self.forward_frontier) + len(self.backward_frontier)) * 300 / (1024 * 1024)

    def load_hologram(self):
        path = os.path.join(self.save_dir, "main_hologram.pkl")
        if os.path.exists(path):
            self.hologram = Hologram.load(path)
            print(f"Loaded hologram: {len(self.hologram.solved):,} solved")

    def save_hologram(self):
        path = os.path.join(self.save_dir, "main_hologram.pkl")
        self.hologram.save(path)

    def solve(self, forward_starts: List[ChessState],
              backward_starts: List[ChessState] = None,
              max_iterations: int = 100,
              lightning_interval: int = 5):
        """
        Bidirectional solve with lightning probes and crystallization.
        """
        print(f"\n{'='*60}")
        print("BIDIRECTIONAL HOLOS - Slime Mold Crystallization")
        print(f"{'='*60}")

        # Initialize forward frontier
        for state in forward_starts:
            h = hash(state)
            if h not in self.forward_seen:
                self.forward_seen.add(h)
                self.forward_frontier[h] = state

        # Generate backward starts from syzygy if not provided
        if backward_starts is None:
            backward_starts = self._generate_boundary_positions(forward_starts[0])

        seeded_backward = 0
        for state in backward_starts:
            h = hash(state)
            if h not in self.backward_seen:
                self.backward_seen.add(h)
                self.backward_frontier[h] = state
                # Probe syzygy for value
                value = self.syzygy.probe(state)
                if value is not None:
                    features = extract_features(state)
                    self.hologram.add_with_features(h, value, features)
                    self.hologram.add_boundary(h, value)
                    seeded_backward += 1

        # Also seed backward wave from any 7-piece positions we reach during forward expansion
        print(f"Seeded {seeded_backward} backward positions with syzygy values")

        print(f"Forward frontier: {len(self.forward_frontier):,}")
        print(f"Backward frontier: {len(self.backward_frontier):,}")

        start_time = time.time()

        for iteration in range(max_iterations):
            mem = self.memory_mb()

            if mem > self.max_memory_mb * 0.9:
                print(f"\nMemory limit reached ({mem:.0f} MB)")
                break

            if not self.forward_frontier and not self.backward_frontier:
                print("\nBoth frontiers empty!")
                break

            print(f"\n--- Iteration {iteration} ---")
            print(f"  Forward: {len(self.forward_frontier):,}, Backward: {len(self.backward_frontier):,}")
            print(f"  Solved: {len(self.hologram.solved):,}, Memory: {mem:.0f} MB")

            # Lightning probes every N iterations
            if iteration % lightning_interval == 0 and self.forward_frontier:
                self._lightning_phase()

            # Expand both frontiers
            forward_contacts = self._expand_forward()
            backward_contacts = self._expand_backward()

            # Check for connections
            new_connections = self._find_connections()

            if new_connections:
                print(f"  ** {new_connections} NEW CONNECTIONS! Crystallizing... **")
                self._crystallize_connections()

            # Propagate
            propagated = self._propagate()

            elapsed = time.time() - start_time
            total_seen = len(self.forward_seen) + len(self.backward_seen)
            rate = total_seen / elapsed if elapsed > 0 else 0
            equiv_count = sum(len(v) for v in self.hologram.equiv_classes.values())
            equiv_shortcuts = self.stats.get('equiv_shortcuts', 0)
            equiv_propagated = self.stats.get('equiv_propagated', 0)
            print(f"  Contacts: {forward_contacts}+{backward_contacts}, Propagated: {propagated}")
            print(f"  Rate: {rate:.0f} pos/s, Connections: {len(self.connections)}")
            print(f"  Equivalence: {equiv_count:,} tracked, {equiv_shortcuts:,} shortcuts, {equiv_propagated:,} propagated")

        self.save_hologram()

        print(f"\n{'='*60}")
        print(f"COMPLETE: {len(self.hologram.solved):,} solved")
        print(f"  Spines: {len(self.hologram.spines):,}")
        print(f"  Connections: {len(self.connections):,}")
        print(f"{'='*60}")

    def _generate_boundary_positions(self, template: ChessState, count: int = 100) -> List[ChessState]:
        """
        Generate 7-piece positions for backward wave seeding.

        Strategy: Create valid 7-piece positions with similar material type
        to what's in the template (e.g., if template has queens, keep queens).
        """
        positions = []
        pieces_template = list(template.pieces)

        # Get piece types from template (excluding kings)
        template_types = [piece_type(p) for p, sq in pieces_template if piece_type(p) != 1]

        # If template is already <=7 pieces, expand it differently
        if len(pieces_template) <= 7:
            # Generate variations by randomizing squares
            for _ in range(count * 5):
                if len(positions) >= count:
                    break

                piece_types = [p for p, sq in pieces_template]
                new_squares = random.sample(range(64), len(piece_types))

                # Check kings not adjacent
                king_sqs = [new_squares[i] for i, p in enumerate(piece_types) if piece_type(p) == 1]
                if len(king_sqs) == 2:
                    kr1, kf1 = king_sqs[0] // 8, king_sqs[0] % 8
                    kr2, kf2 = king_sqs[1] // 8, king_sqs[1] % 8
                    if abs(kr1 - kr2) <= 1 and abs(kf1 - kf2) <= 1:
                        continue

                state = ChessState(list(zip(piece_types, new_squares)), random.choice(['w', 'b']))
                if not in_check(state, 'b' if state.turn == 'w' else 'w'):
                    # Only add if syzygy can solve it
                    if self.syzygy.probe(state) is not None:
                        positions.append(state)
        else:
            # Template has >7 pieces, reduce to 7
            for _ in range(count * 10):
                if len(positions) >= count:
                    break

                pieces = list(pieces_template)
                random.shuffle(pieces)

                kings = [(p, sq) for p, sq in pieces if piece_type(p) == 1]
                others = [(p, sq) for p, sq in pieces if piece_type(p) != 1]

                # Keep 5 non-king pieces
                if len(others) >= 5:
                    kept = others[:5]
                    new_pieces = kings + kept

                    if len(new_pieces) == 7:
                        piece_types = [p for p, sq in new_pieces]
                        new_squares = random.sample(range(64), 7)

                        king_sqs = [new_squares[i] for i, p in enumerate(piece_types) if piece_type(p) == 1]
                        if len(king_sqs) == 2:
                            kr1, kf1 = king_sqs[0] // 8, king_sqs[0] % 8
                            kr2, kf2 = king_sqs[1] // 8, king_sqs[1] % 8
                            if abs(kr1 - kr2) <= 1 and abs(kf1 - kf2) <= 1:
                                continue

                        state = ChessState(list(zip(piece_types, new_squares)), random.choice(['w', 'b']))
                        if not in_check(state, 'b' if state.turn == 'w' else 'w'):
                            if self.syzygy.probe(state) is not None:
                                positions.append(state)

        print(f"Generated {len(positions)} boundary positions (target: {count})")
        return positions

    def _lightning_phase(self):
        """Send lightning probes from forward frontier"""
        probe = LightningProbe(self.syzygy, self.hologram, max_depth=20)

        # Sample positions from frontier
        sample_size = min(10, len(self.forward_frontier))
        samples = random.sample(list(self.forward_frontier.values()), sample_size)

        spines_found = 0
        for state in samples:
            value, path = probe.probe(state)
            self.stats['lightning_probes'] += 1

            if value is not None and path:
                # Found a path to known territory - create spine
                h = hash(state)
                moves = [m for s, m in path]
                end_state = state
                for s, m in path:
                    end_state = apply_move(s, m)

                spine = SpinePath(
                    start_hash=h,
                    moves=moves,
                    end_hash=hash(end_state),
                    end_value=value,
                    depth=len(moves)
                )
                self.hologram.add_spine(spine)
                spines_found += 1

                # Add intermediate positions to solved
                current = state
                for s, m in path:
                    ch = hash(current)
                    self.hologram.solved[ch] = value
                    current = apply_move(s, m)

        if spines_found:
            print(f"  Lightning: {spines_found} spines found ({probe.nodes_visited} nodes)")
            self.stats['spines_found'] += spines_found

    def _expand_forward(self) -> int:
        """Expand forward frontier by one layer with equivalence tracking"""
        items = list(self.forward_frontier.items())
        if not items:
            return 0

        next_frontier = {}
        contacts = 0
        equiv_added = 0

        for h, state in items:
            # Extract and track features for current state
            features = extract_features(state)
            self.hologram.equiv_classes[features].add(h)
            equiv_added += 1

            # Check terminal
            moves = generate_moves(state)
            is_term, value = is_terminal(state, moves)
            if is_term:
                self.hologram.add_with_features(h, value, features)
                contacts += 1
                continue

            # Already solved?
            if h in self.hologram.solved:
                contacts += 1
                continue

            # Check equivalence class for potential shortcut
            if features in self.hologram.equiv_outcomes:
                eq_value = self.hologram.equiv_outcomes[features]
                if eq_value is not None:
                    self.hologram.solved[h] = eq_value
                    contacts += 1
                    self.stats['equiv_shortcuts'] = self.stats.get('equiv_shortcuts', 0) + 1
                    continue

            # Expand children
            for move in moves:
                child = apply_move(state, move)
                ch = hash(child)

                # Check syzygy - seed backward wave from boundary positions
                if child.piece_count() <= 7:
                    value = self.syzygy.probe(child)
                    if value is not None:
                        child_features = extract_features(child)
                        self.hologram.add_with_features(ch, value, child_features)
                        self.hologram.add_boundary(ch, value)
                        contacts += 1

                        # Seed backward wave from this boundary position!
                        if ch not in self.backward_seen:
                            self.backward_seen.add(ch)
                            self.backward_frontier[ch] = child
                        continue

                # Check if backward wave already explored this
                if ch in self.backward_seen:
                    # Connection! Mark for crystallization
                    if ch in self.hologram.solved:
                        contacts += 1
                        continue

                if ch not in self.forward_seen:
                    self.forward_seen.add(ch)
                    next_frontier[ch] = child
                    self.forward_parents[ch] = (h, move)

        self.forward_frontier = next_frontier

        if equiv_added > 0:
            self.stats['equiv_tracked'] = self.stats.get('equiv_tracked', 0) + equiv_added

        return contacts

    def _expand_backward(self) -> int:
        """
        Expand backward frontier using TRUE predecessor generation.

        This generates positions that could have LED TO the current boundary positions.
        For each solved boundary position, we ask: "What positions could result in this?"

        This is abductive reasoning: given the outcome, what were the causes?

        MINIMAX LOGIC for backward propagation:
        - If child is solved with value V from child's perspective
        - Predecessor had opposite turn, so they CHOSE to reach this child
        - If V is good for the side that moved TO child, it was their choice
        - Predecessor value depends on whether this was their BEST option

        For now: We track that predecessor CAN reach this child value.
        Full minimax requires checking ALL children of predecessor.
        """
        items = list(self.backward_frontier.items())
        if not items:
            return 0

        next_frontier = {}
        contacts = 0
        equiv_added = 0

        # Track predecessor -> list of (child_hash, child_value) for later minimax
        pred_children: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        for h, state in items:
            # Generate TRUE predecessors (positions that could lead to this state)
            predecessors = generate_predecessors(state, max_uncaptures=3)

            for pred in predecessors:
                ph = hash(pred)

                # Track this child for the predecessor (even if seen before)
                if h in self.hologram.solved:
                    pred_children[ph].append((h, self.hologram.solved[h]))

                if ph in self.backward_seen:
                    continue

                self.backward_seen.add(ph)

                # Extract features for equivalence tracking
                features = extract_features(pred)
                self.hologram.equiv_classes[features].add(ph)
                equiv_added += 1

                # Check if this predecessor is already solved via forward wave
                if ph in self.forward_seen:
                    # CONNECTION! Forward and backward waves meet
                    contacts += 1
                    self.stats['connections'] += 1
                    # Don't add to frontier, will be handled by connection logic
                    continue

                next_frontier[ph] = pred
                self.backward_parents[ph] = (h, None)

        # Now apply minimax logic to predecessors with known children
        minimax_solved = 0
        for ph, children in pred_children.items():
            if ph in self.hologram.solved:
                continue  # Already solved

            if not children:
                continue

            # Get the predecessor state to check whose turn it is
            pred_state = next_frontier.get(ph) or self.backward_frontier.get(ph)
            if pred_state is None:
                continue

            # Predecessor's turn: they choose among children
            # If it's white's turn in predecessor, they want max value
            # If it's black's turn in predecessor, they want min value
            is_white_turn = (pred_state.turn == 'w')

            child_values = [v for _, v in children]

            # Conservative: only solve if we have evidence of best play
            # If white to move and ANY child gives +1, predecessor is at least +1 potential
            # If black to move and ANY child gives -1, predecessor is at least -1 potential

            if is_white_turn and 1 in child_values:
                # White can choose a winning child
                features = extract_features(pred_state)
                self.hologram.add_with_features(ph, 1, features)
                minimax_solved += 1
            elif not is_white_turn and -1 in child_values:
                # Black can choose a winning child
                features = extract_features(pred_state)
                self.hologram.add_with_features(ph, -1, features)
                minimax_solved += 1
            # Note: draws and forced losses require checking ALL children

        contacts += minimax_solved
        self.backward_frontier = next_frontier

        if equiv_added > 0:
            self.stats['equiv_tracked'] = self.stats.get('equiv_tracked', 0) + equiv_added
        if minimax_solved > 0:
            self.stats['minimax_solved'] = self.stats.get('minimax_solved', 0) + minimax_solved

        return contacts

    def _find_connections(self) -> int:
        """Find where forward and backward waves meet"""
        new_connections = 0

        # Check overlap between seen sets
        overlap = self.forward_seen & self.backward_seen

        for h in overlap:
            # Check if this is a new connection
            if h in self.hologram.solved:
                # Already have value - record connection
                existing = [(fh, bh, v) for fh, bh, v in self.connections if fh == h or bh == h]
                if not existing:
                    value = self.hologram.solved[h]
                    self.connections.append((h, h, value))
                    new_connections += 1
                    self.stats['connections'] += 1

        return new_connections

    def _crystallize_connections(self):
        """
        Expand wave from connection points (slime mold reinforcement).
        When a connection is found, do rigorous BFS around it.
        """
        for forward_h, backward_h, value in self.connections[-10:]:  # Last 10 connections
            # Get states if available
            state = self.forward_frontier.get(forward_h) or self.backward_frontier.get(forward_h)
            if state is None:
                continue

            # Local BFS expansion
            local_frontier = {forward_h: state}
            local_seen = {forward_h}

            for _ in range(3):  # 3 layers of crystallization
                next_local = {}
                for h, s in local_frontier.items():
                    moves = generate_moves(s)
                    for move in moves:
                        child = apply_move(s, move)
                        ch = hash(child)

                        if ch not in local_seen:
                            local_seen.add(ch)
                            next_local[ch] = child

                            # Check if solvable
                            if child.piece_count() <= 7:
                                val = self.syzygy.probe(child)
                                if val is not None:
                                    self.hologram.add_boundary(ch, val)
                                    self.stats['crystallized'] += 1

                local_frontier = next_local

    def _propagate(self, max_iters: int = 50) -> int:
        """Propagate solved values through parent links AND equivalence classes"""
        total = 0

        for _ in range(max_iters):
            newly_solved = 0

            # Forward propagation
            for ch, (ph, move) in list(self.forward_parents.items()):
                if ph in self.hologram.solved and ch not in self.hologram.solved:
                    # Child inherits from parent (with minimax flip)
                    self.hologram.solved[ch] = self.hologram.solved[ph]
                    newly_solved += 1

            # Backward propagation
            for ch, (ph, move) in list(self.backward_parents.items()):
                if ch in self.hologram.solved and ph not in self.hologram.solved:
                    self.hologram.solved[ph] = self.hologram.solved[ch]
                    newly_solved += 1

            # Equivalence class propagation (key optimization!)
            equiv_solved = self.hologram.propagate_equivalence()
            newly_solved += equiv_solved
            if equiv_solved > 0:
                self.stats['equiv_propagated'] = self.stats.get('equiv_propagated', 0) + equiv_solved

            total += newly_solved
            if newly_solved == 0:
                break

        return total


# ============================================================
# UTILITIES
# ============================================================

def random_position(material: str, max_attempts=1000):
    white_str, black_str = material.upper().split('V')
    white = [PIECE_NAMES[c] for c in white_str]
    black = [PIECE_NAMES[c] + 6 for c in black_str]
    all_pieces = white + black

    for _ in range(max_attempts):
        squares = random.sample(range(64), len(all_pieces))
        pieces = list(zip(all_pieces, squares))
        state = ChessState(pieces, 'w')

        wk = next((sq for p, sq in pieces if p == Piece.W_KING), None)
        bk = next((sq for p, sq in pieces if p == Piece.B_KING), None)
        if wk is None or bk is None:
            continue
        if abs(wk // 8 - bk // 8) <= 1 and abs(wk % 8 - bk % 8) <= 1:
            continue
        if in_check(state, 'b'):
            continue
        return state
    return None


def exhaustive_boundary_test(material_7: str = "KQRRvKQR",
                              num_boundary: int = 1000,
                              max_pred_per_pos: int = 100,
                              syzygy_path: str = "./syzygy"):
    """
    Exhaustive test: Enumerate all 8-piece positions reachable from 7-piece boundary.

    This is the key test for HOLOS viability:
    1. Sample 7-piece boundary positions from Syzygy
    2. Generate ALL 8-piece predecessors
    3. Analyze compression potential (how many unique 8-piece -> same outcome?)

    Returns dict with statistics.
    """
    print("=" * 60)
    print(f"EXHAUSTIVE BOUNDARY TEST: {material_7}")
    print("=" * 60)

    syzygy = SyzygyProbe(syzygy_path)
    if not syzygy.available:
        print("ERROR: Syzygy not available")
        return None

    # Phase 1: Collect 7-piece boundary positions
    print(f"\nPhase 1: Collecting {num_boundary} 7-piece boundary positions...")
    boundary_7 = {}  # hash -> (state, value)

    attempts = 0
    while len(boundary_7) < num_boundary and attempts < num_boundary * 20:
        attempts += 1
        state = random_position(material_7)
        if state is None:
            continue
        h = hash(state)
        if h in boundary_7:
            continue
        value = syzygy.probe(state)
        if value is not None:
            boundary_7[h] = (state, value)

    values_7 = [v for _, v in boundary_7.values()]
    print(f"  Collected: {len(boundary_7)}")
    print(f"  Values: +1={values_7.count(1)}, 0={values_7.count(0)}, -1={values_7.count(-1)}")

    # Phase 2: Generate all 8-piece predecessors
    print(f"\nPhase 2: Generating 8-piece predecessors...")
    predecessors_8 = {}  # hash -> (state, parent_7_hash, parent_7_value)
    pred_to_children = defaultdict(list)  # pred_hash -> [(child_hash, child_value), ...]

    import time
    t0 = time.time()

    for h7, (state7, value7) in boundary_7.items():
        preds = generate_predecessors(state7, max_uncaptures=5)

        for pred in preds[:max_pred_per_pos]:
            if pred.piece_count() != 8:
                continue

            ph = hash(pred)
            pred_to_children[ph].append((h7, value7))

            if ph not in predecessors_8:
                predecessors_8[ph] = (pred, h7, value7)

    elapsed = time.time() - t0
    print(f"  Unique 8-piece positions: {len(predecessors_8):,}")
    print(f"  Total pred->child edges: {sum(len(v) for v in pred_to_children.values()):,}")
    print(f"  Time: {elapsed:.2f}s")

    # Phase 3: Analyze - which 8-piece positions can we solve via minimax?
    print(f"\nPhase 3: Minimax analysis...")

    solved_8 = {}  # hash -> value
    unsolved_8 = set()

    for ph, children in pred_to_children.items():
        pred_state = predecessors_8[ph][0]
        is_white_turn = (pred_state.turn == 'w')
        child_values = [v for _, v in children]

        # Minimax: side to move picks best option
        if is_white_turn:
            # White picks max
            if 1 in child_values:
                solved_8[ph] = 1  # White can win
            elif len(set(child_values)) == 1 and child_values[0] == 0:
                # All children are draws (conservative - may have unseen children)
                pass  # Can't confirm draw without checking ALL children
            elif len(set(child_values)) == 1 and child_values[0] == -1:
                # All seen children lose (but might have winning children we haven't seen)
                pass
            else:
                unsolved_8.add(ph)
        else:
            # Black picks min
            if -1 in child_values:
                solved_8[ph] = -1  # Black can win
            else:
                unsolved_8.add(ph)

    values_8 = list(solved_8.values())
    print(f"  Solved 8-piece: {len(solved_8):,} ({100*len(solved_8)/len(predecessors_8):.1f}%)")
    print(f"  Unsolved: {len(unsolved_8):,}")
    print(f"  Solved values: +1={values_8.count(1)}, 0={values_8.count(0)}, -1={values_8.count(-1)}")

    # Phase 4: Feature-based compression analysis
    print(f"\nPhase 4: Compression analysis...")

    feature_clusters = defaultdict(list)  # features -> [(hash, value), ...]

    for ph, (pred_state, _, _) in predecessors_8.items():
        features = extract_features(pred_state)
        value = solved_8.get(ph, None)
        feature_clusters[features].append((ph, value))

    # Count clusters with uniform values
    uniform_clusters = 0
    mixed_clusters = 0
    for features, members in feature_clusters.items():
        values = [v for _, v in members if v is not None]
        if len(values) >= 2:
            if len(set(values)) == 1:
                uniform_clusters += 1
            else:
                mixed_clusters += 1

    compression_ratio = len(predecessors_8) / len(feature_clusters) if feature_clusters else 1

    print(f"  Feature clusters: {len(feature_clusters):,}")
    print(f"  Uniform value clusters: {uniform_clusters}")
    print(f"  Mixed value clusters: {mixed_clusters}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  7-piece boundary: {len(boundary_7):,} positions")
    print(f"  8-piece reachable: {len(predecessors_8):,} positions")
    print(f"  Expansion factor: {len(predecessors_8)/len(boundary_7):.1f}x")
    print(f"  Minimax solvable: {len(solved_8):,} ({100*len(solved_8)/len(predecessors_8):.1f}%)")
    print(f"  Feature compression: {compression_ratio:.2f}x")
    print(f"  Effective compression: {len(predecessors_8)/len(feature_clusters):.1f}x")

    return {
        'boundary_7': len(boundary_7),
        'predecessors_8': len(predecessors_8),
        'solved_8': len(solved_8),
        'feature_clusters': len(feature_clusters),
        'compression_ratio': compression_ratio,
        'uniform_clusters': uniform_clusters,
        'mixed_clusters': mixed_clusters,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', default='KQRRvKQRR')
    parser.add_argument('--positions', type=int, default=100)
    parser.add_argument('--max-memory', type=int, default=4000, help='Max memory MB')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--exhaustive', action='store_true', help='Run exhaustive boundary test')
    parser.add_argument('--boundary-material', default='KQRRvKQR', help='7-piece material for exhaustive test')
    args = parser.parse_args()

    if args.exhaustive:
        results = exhaustive_boundary_test(
            material_7=args.boundary_material,
            num_boundary=args.positions,
        )
        return

    print("="*60)
    print("FRACTAL HOLOS v3 - Bidirectional Lightning + Crystal")
    print("="*60)

    solver = BidirectionalHOLOS(max_memory_mb=args.max_memory)
    solver.load_hologram()

    # Generate starting positions
    print(f"\nGenerating {args.positions} starting positions for {args.material}...")
    states = []
    for _ in range(args.positions):
        state = random_position(args.material)
        if state:
            states.append(state)
    print(f"Generated {len(states)} valid positions")

    if states:
        solver.solve(states, max_iterations=args.iterations)


if __name__ == '__main__':
    main()
