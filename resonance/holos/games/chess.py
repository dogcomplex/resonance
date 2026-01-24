"""
holos/games/chess.py - Chess Game Interface for HOLOS

This implements GameInterface for chess endgames (Layer 0).

The game:
- States: Chess positions (piece placements + turn)
- Successors: Legal moves
- Predecessors: Positions that could lead here (unmoves/uncaptures)
- Boundary: 7-piece positions (Syzygy tablebases)
- Values: Win(+1), Draw(0), Loss(-1)
- Propagation: Minimax (White max, Black min)

Key insight: Chess is a BOUNDED game for HOLOS because:
- Lower bound: 7-piece Syzygy (known values)
- Upper bound: Configurable piece count
- Forward: Moves reduce pieces (captures)
- Backward: Uncaptures increase pieces
"""

import os
import random
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
from enum import IntEnum

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from holos.holos import GameInterface


# ============================================================
# CHESS PRIMITIVES
# ============================================================

class Piece(IntEnum):
    EMPTY = 0
    W_KING = 1; W_QUEEN = 2; W_ROOK = 3; W_BISHOP = 4; W_KNIGHT = 5; W_PAWN = 6
    B_KING = 7; B_QUEEN = 8; B_ROOK = 9; B_BISHOP = 10; B_KNIGHT = 11; B_PAWN = 12


PIECE_CHARS = '.KQRBNPkqrbnp'
PIECE_NAMES = {'K': Piece.W_KING, 'Q': Piece.W_QUEEN, 'R': Piece.W_ROOK,
               'B': Piece.W_BISHOP, 'N': Piece.W_KNIGHT, 'P': Piece.W_PAWN}
PIECE_VALUES = {2: 900, 3: 500, 4: 330, 5: 320, 6: 100,
                8: 900, 9: 500, 10: 330, 11: 320, 12: 100}

ALL_CAPTURABLE = [Piece.W_QUEEN, Piece.W_ROOK, Piece.W_BISHOP, Piece.W_KNIGHT, Piece.W_PAWN,
                  Piece.B_QUEEN, Piece.B_ROOK, Piece.B_BISHOP, Piece.B_KNIGHT, Piece.B_PAWN]


def is_white(p): return 1 <= p <= 6
def is_black(p): return 7 <= p <= 12
def piece_type(p): return ((p - 1) % 6) + 1 if p > 0 else 0


class ChessState:
    """Chess position state"""
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


# ============================================================
# CHESS VALUE
# ============================================================

@dataclass(frozen=True)
class ChessValue:
    """Chess game-theoretic value"""
    value: int  # +1 (White wins), 0 (Draw), -1 (Black wins)

    def __repr__(self):
        return {1: "Win", 0: "Draw", -1: "Loss"}.get(self.value, f"Value({self.value})")


# ============================================================
# CHESS FEATURES (Equivalence Classes)
# ============================================================

@dataclass(frozen=True)
class ChessFeatures:
    """Equivalence class features"""
    material_white: Tuple[int, ...]
    material_black: Tuple[int, ...]
    material_balance: int
    piece_count: int
    king_distance: int
    turn: str


def extract_features(state: ChessState) -> ChessFeatures:
    """Extract equivalence features"""
    white_pieces = []
    black_pieces = []
    wk_sq = bk_sq = None
    material = 0

    for p, sq in state.pieces:
        pt = piece_type(p)
        if pt == 1:
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
# MOVE GENERATION
# ============================================================

def generate_moves(state: ChessState) -> List[Tuple[int, int, Optional[int]]]:
    """Generate legal moves as (from_sq, to_sq, captured_piece)"""
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


def apply_move(state: ChessState, move: Tuple) -> ChessState:
    """Apply move to state"""
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


def in_check(state: ChessState, color: str) -> bool:
    """Is the given color in check?"""
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


def is_terminal(state: ChessState, moves=None) -> Tuple[bool, Optional[int]]:
    """Check for game end"""
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
# PREDECESSOR GENERATION
# ============================================================

def generate_predecessors(state: ChessState, max_uncaptures: int = 3) -> List[ChessState]:
    """Generate positions that could lead to this state"""
    predecessors = []
    board = state.to_board()
    prev_turn = 'b' if state.turn == 'w' else 'w'
    is_prev_white = (prev_turn == 'w')

    for piece, to_sq in state.pieces:
        if is_prev_white and not is_white(piece):
            continue
        if not is_prev_white and not is_black(piece):
            continue

        pt = piece_type(piece)
        from_squares = _get_reverse_moves(board, to_sq, piece, pt)

        for from_sq in from_squares:
            # No capture
            new_pieces = []
            for p, sq in state.pieces:
                if sq == to_sq and p == piece:
                    new_pieces.append((piece, from_sq))
                else:
                    new_pieces.append((p, sq))

            pred = ChessState(new_pieces, prev_turn)
            if _validate_predecessor(pred):
                predecessors.append(pred)

            # With capture
            if state.piece_count() < 32 and len(predecessors) < max_uncaptures * 10:
                for cap_piece in ALL_CAPTURABLE:
                    if is_prev_white and is_white(cap_piece):
                        continue
                    if not is_prev_white and is_black(cap_piece):
                        continue

                    new_pieces_cap = []
                    for p, sq in state.pieces:
                        if sq == to_sq and p == piece:
                            new_pieces_cap.append((piece, from_sq))
                        else:
                            new_pieces_cap.append((p, sq))
                    new_pieces_cap.append((cap_piece, to_sq))

                    pred_cap = ChessState(new_pieces_cap, prev_turn)
                    if _validate_predecessor(pred_cap):
                        predecessors.append(pred_cap)

    return predecessors


def _get_reverse_moves(board, to_sq, piece, pt):
    """Get squares a piece could have come from"""
    from_squares = []

    if pt == 1:  # King
        for d in [-9, -8, -7, -1, 1, 7, 8, 9]:
            from_sq = to_sq - d
            if 0 <= from_sq < 64 and abs((to_sq % 8) - (from_sq % 8)) <= 1:
                if board[from_sq] == Piece.EMPTY:
                    from_squares.append(from_sq)
    elif pt == 5:  # Knight
        for d in [-17, -15, -10, -6, 6, 10, 15, 17]:
            from_sq = to_sq - d
            if 0 <= from_sq < 64 and abs((to_sq % 8) - (from_sq % 8)) <= 2:
                if board[from_sq] == Piece.EMPTY:
                    from_squares.append(from_sq)
    elif pt in [2, 3, 4]:  # Sliding pieces
        directions = {2: [-9,-8,-7,-1,1,7,8,9], 3: [-8,-1,1,8], 4: [-9,-7,7,9]}[pt]
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
                    break
                prev_file = sq_file
                sq -= d

    return from_squares


def _validate_predecessor(pred: ChessState) -> bool:
    """Validate predecessor position"""
    wk_sq = bk_sq = None
    for p, sq in pred.pieces:
        if p == Piece.W_KING:
            wk_sq = sq
        elif p == Piece.B_KING:
            bk_sq = sq

    if wk_sq is not None and bk_sq is not None:
        if abs(wk_sq // 8 - bk_sq // 8) <= 1 and abs(wk_sq % 8 - bk_sq % 8) <= 1:
            return False

    opp_color = 'b' if pred.turn == 'w' else 'w'
    if in_check(pred, opp_color):
        return False

    return True


# ============================================================
# SYZYGY PROBE
# ============================================================

class SyzygyProbe:
    """Interface to Syzygy tablebases"""

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

    def probe(self, state: ChessState) -> Optional[int]:
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
            if wdl is None:
                return None
            self.hits += 1
            if wdl > 0:
                return 1 if state.turn == 'w' else -1
            if wdl < 0:
                return -1 if state.turn == 'w' else 1
            return 0
        except:
            return None


# ============================================================
# CHESS GAME INTERFACE
# ============================================================

class ChessGame(GameInterface[ChessState, ChessValue]):
    """
    Chess endgame interface for HOLOS.

    Boundaries:
    - Lower: min_pieces positions (typically 7 for Syzygy tablebases)
    - Upper: max_pieces (configurable)

    Optional Targeting:
    - target_material: Specific material signature to reach (e.g., "KQRRvKQR")
    - source_materials: Valid source materials to search from (auto-generated if None)

    When targeting is enabled:
    - is_boundary() only accepts target material positions
    - get_successors() filters out captures to wrong materials
    - get_predecessors() only allows valid source materials
    """

    def __init__(self, syzygy_path: str = "./syzygy",
                 min_pieces: int = 7, max_pieces: int = 8,
                 target_material: str = None,
                 source_materials: List[str] = None):
        """
        Initialize chess game interface.

        Args:
            syzygy_path: Path to Syzygy tablebases
            min_pieces: Minimum pieces (boundary), typically 7 for Syzygy
            max_pieces: Maximum pieces to consider
            target_material: Optional target material signature (e.g., "KQRRvKQR")
                            When set, only this material counts as boundary.
            source_materials: Valid 8-piece source materials (auto-generated if None)
        """
        self.syzygy = SyzygyProbe(syzygy_path)
        self.min_pieces = min_pieces
        self.max_pieces = max_pieces

        # Targeting configuration
        self.target_material = target_material.upper() if target_material else None
        self.source_materials = None
        self._target_signature = None
        self._source_signatures = None

        if self.target_material:
            self._target_signature = self._parse_material_signature(self.target_material)
            if source_materials is None:
                source_materials = get_parent_materials(self.target_material)
            self.source_materials = [m.upper() for m in source_materials]
            self._source_signatures = {self._parse_material_signature(m) for m in self.source_materials}

            print(f"ChessGame targeting enabled:")
            print(f"  Target: {self.target_material} ({self.min_pieces} pieces)")
            print(f"  Sources: {len(self.source_materials)} configurations")

        # Filter statistics (for targeted mode)
        self.filter_stats = {
            'wrong_material_filtered': 0,
            'target_material_found': 0,
            'source_positions_generated': 0,
        }

    def _parse_material_signature(self, material: str) -> frozenset:
        """Parse material string into a hashable signature"""
        white_str, black_str = material.upper().split('V')
        pieces = []
        for c in white_str:
            pieces.append(('w', c))
        for c in black_str:
            pieces.append(('b', c))
        return frozenset(pieces)

    def _get_state_material_signature(self, state: ChessState) -> frozenset:
        """Get material signature from state"""
        piece_chars = {
            Piece.W_KING: 'K', Piece.W_QUEEN: 'Q', Piece.W_ROOK: 'R',
            Piece.W_BISHOP: 'B', Piece.W_KNIGHT: 'N', Piece.W_PAWN: 'P',
            Piece.B_KING: 'K', Piece.B_QUEEN: 'Q', Piece.B_ROOK: 'R',
            Piece.B_BISHOP: 'B', Piece.B_KNIGHT: 'N', Piece.B_PAWN: 'P',
        }
        pieces = []
        for p, sq in state.pieces:
            color = 'w' if p <= 6 else 'b'
            char = piece_chars.get(p, '?')
            pieces.append((color, char))
        return frozenset(pieces)

    def is_target_material(self, state: ChessState) -> bool:
        """Check if state has target material configuration"""
        if not self.target_material:
            return False
        return self._get_state_material_signature(state) == self._target_signature

    def is_source_material(self, state: ChessState) -> bool:
        """Check if state has a valid source material configuration"""
        if not self.source_materials:
            return False
        return self._get_state_material_signature(state) in self._source_signatures

    def hash_state(self, state: ChessState) -> int:
        return hash(state)

    def get_successors(self, state: ChessState) -> List[Tuple[ChessState, Any]]:
        """
        Legal moves from this position.

        Without targeting: All legal moves within piece count bounds.
        With targeting: Filters out captures that reach wrong material.
        """
        if state.piece_count() > self.max_pieces:
            return []

        moves = generate_moves(state)
        successors = []

        for move in moves:
            child = apply_move(state, move)
            child_count = child.piece_count()

            # Targeting mode: filter captures to wrong material
            if self.target_material and move[2] is not None:  # Is a capture
                if child_count == self.min_pieces:
                    # Captured down to target piece count - must be target material
                    if self.is_target_material(child):
                        successors.append((child, move))
                    else:
                        self.filter_stats['wrong_material_filtered'] += 1
                elif child_count > self.min_pieces:
                    # Still above target - allow
                    successors.append((child, move))
            else:
                # Non-capture move or no targeting
                if self.min_pieces <= child_count <= self.max_pieces:
                    successors.append((child, move))
                elif child_count < self.min_pieces:
                    successors.append((child, move))

        return successors

    def get_predecessors(self, state: ChessState) -> List[Tuple[ChessState, Any]]:
        """
        Positions that could lead to this.

        Without targeting: All valid predecessors within piece count bounds.
        With targeting: Only predecessors from valid source materials.
        """
        if state.piece_count() >= self.max_pieces:
            return []

        preds = generate_predecessors(state, max_uncaptures=3)
        valid_preds = []

        for pred in preds:
            if pred.piece_count() > self.max_pieces:
                continue

            # Targeting mode: check if predecessor is valid source material
            if self.target_material and pred.piece_count() == self.max_pieces:
                if self.is_source_material(pred):
                    valid_preds.append((pred, None))
                else:
                    self.filter_stats['wrong_material_filtered'] += 1
            else:
                valid_preds.append((pred, None))

        return valid_preds

    def is_boundary(self, state: ChessState) -> bool:
        """
        Is this on the boundary?

        Without targeting: Any position at min_pieces is boundary.
        With targeting: Only target material positions at min_pieces.
        """
        if state.piece_count() > self.min_pieces:
            return False

        # If targeting enabled, only target material counts as boundary
        if self.target_material:
            if state.piece_count() != self.min_pieces:
                return False
            return self.is_target_material(state)

        return state.piece_count() <= self.min_pieces

    def get_boundary_value(self, state: ChessState) -> Optional[ChessValue]:
        """Query Syzygy for value"""
        if not self.is_boundary(state):
            return None
        val = self.syzygy.probe(state)
        if val is not None:
            if self.target_material:
                self.filter_stats['target_material_found'] += 1
            return ChessValue(val)
        return None

    def is_terminal(self, state: ChessState) -> Tuple[bool, Optional[ChessValue]]:
        """Check for checkmate/stalemate"""
        moves = generate_moves(state)
        is_term, value = is_terminal(state, moves)
        if is_term:
            return True, ChessValue(value if value is not None else 0)
        return False, None

    def propagate_value(self, state: ChessState,
                        child_values: List[ChessValue]) -> Optional[ChessValue]:
        """
        Minimax propagation.

        For minimax, we need to know whose turn it is:
        - White's turn: propagate if any child is +1 (white win)
        - Black's turn: propagate if any child is -1 (black win)

        If state is None (called from reverse propagation), we cannot
        determine whose turn it is, so we return None to skip.
        """
        if not child_values:
            return None

        # State required for minimax - if None, skip reverse propagation
        if state is None:
            return None

        values = [cv.value for cv in child_values]

        if state.turn == 'w':
            if 1 in values:
                return ChessValue(1)
            return None
        else:
            if -1 in values:
                return ChessValue(-1)
            return None

    def get_features(self, state: ChessState) -> Any:
        """Extract equivalence features"""
        return extract_features(state)

    def get_lightning_successors(self, state: ChessState) -> List[Tuple[ChessState, Any]]:
        """For lightning: captures only"""
        moves = generate_moves(state)
        captures = [m for m in moves if m[2] is not None]
        return [(apply_move(state, m), m) for m in captures]

    def get_lightning_predecessors(self, state: ChessState) -> List[Tuple[ChessState, Any]]:
        """For backward lightning: uncaptures (add pieces)"""
        preds = generate_predecessors(state, max_uncaptures=5)
        # Filter to those that ADD a piece (uncaptures)
        return [(p, None) for p in preds if p.piece_count() > state.piece_count()]

    def score_for_lightning(self, state: ChessState, move: Any) -> float:
        """Score captures by MVV-LVA"""
        if move is None or move[2] is None:
            return 0.0
        return PIECE_VALUES.get(move[2], 0)

    def apply_move(self, state: ChessState, move: Any) -> ChessState:
        """Apply a move to get successor state (for spine tracking)"""
        return apply_move(state, move)

    def get_signature(self, state: ChessState) -> str:
        """
        Get material signature for goal matching.

        Returns string like 'KQRRvKQR' for use with GoalCondition.
        This is a Layer 0 capability used by Layer 1/2 for goal filtering.
        """
        return get_material_string(state)

    def enumerate_positions(self, material: str, count: int = 100) -> List[ChessState]:
        """
        Generate positions with given material signature.

        This is a Layer 0 capability used by Layer 1/2 for seed generation.
        """
        return enumerate_material_positions(material, self.syzygy, count)

    def get_parent_signatures(self, target_signature: str) -> List[str]:
        """
        Get signatures of states that can transition to target.

        For chess, returns materials that can capture down to target.
        This is a Layer 0 capability used by Layer 1/2 for seed planning.
        """
        return get_parent_materials(target_signature)

    def generate_boundary_seeds(self, template: ChessState, count: int = 100) -> List[ChessState]:
        """
        Generate boundary positions for backward wave seeding.

        Strategy: Create valid positions at self.min_pieces with similar material type
        to what's in the template (e.g., if template has queens, keep queens).

        The boundary is defined by self.min_pieces (typically 7 for syzygy tables,
        but can be lower for simpler testing).

        This matches fractal_holos3.py's _generate_boundary_positions() method.
        """
        positions = []
        pieces_template = list(template.pieces)
        target_pieces = self.min_pieces  # Use configured boundary

        # If template is already at or below boundary, randomize squares
        if len(pieces_template) <= target_pieces:
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
            # Template has more pieces than target, reduce to target_pieces
            non_king_target = target_pieces - 2  # Account for 2 kings
            for _ in range(count * 10):
                if len(positions) >= count:
                    break

                pieces = list(pieces_template)
                random.shuffle(pieces)

                kings = [(p, sq) for p, sq in pieces if piece_type(p) == 1]
                others = [(p, sq) for p, sq in pieces if piece_type(p) != 1]

                # Keep non_king_target non-king pieces
                if len(others) >= non_king_target:
                    kept = others[:non_king_target]
                    new_pieces = kings + kept

                    if len(new_pieces) == target_pieces:
                        piece_types = [p for p, sq in new_pieces]
                        new_squares = random.sample(range(64), target_pieces)

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

    def generate_target_boundary_seeds(self, count: int = 1000) -> List[ChessState]:
        """
        Generate random positions with target material for backward seeding.

        Only works when target_material is set. Returns positions with the
        exact target material configuration that can be solved by Syzygy.

        This method enables targeted bidirectional search by seeding the
        backward wave with positions we're trying to reach.
        """
        if not self.target_material:
            raise ValueError("generate_target_boundary_seeds requires target_material to be set")

        white_pieces, black_pieces = _parse_material_string(self.target_material)
        all_pieces = white_pieces + black_pieces
        positions = []

        for _ in range(count * 10):
            if len(positions) >= count:
                break

            squares = random.sample(range(64), len(all_pieces))
            pieces = list(zip(all_pieces, squares))
            state = ChessState(pieces, random.choice(['w', 'b']))

            # Validate king positions
            wk = next((sq for p, sq in pieces if p == Piece.W_KING), None)
            bk = next((sq for p, sq in pieces if p == Piece.B_KING), None)
            if wk is None or bk is None:
                continue
            if abs(wk // 8 - bk // 8) <= 1 and abs(wk % 8 - bk % 8) <= 1:
                continue
            if in_check(state, 'b' if state.turn == 'w' else 'w'):
                continue

            # Must be solvable by syzygy
            if self.syzygy.probe(state) is not None:
                positions.append(state)

        print(f"Generated {len(positions)} target boundary positions ({self.target_material})")
        return positions

    def generate_source_positions(self, count_per_material: int = 100) -> List[ChessState]:
        """
        Generate positions from all valid source materials.

        Only works when target_material is set. Returns 8-piece positions
        that could potentially capture down to the target 7-piece material.

        This method enables targeted forward search from valid source positions.
        """
        if not self.target_material or not self.source_materials:
            raise ValueError("generate_source_positions requires target_material to be set")

        all_positions = []

        for material in self.source_materials:
            white_pieces, black_pieces = _parse_material_string(material)
            all_pieces = white_pieces + black_pieces
            positions = []

            for _ in range(count_per_material * 10):
                if len(positions) >= count_per_material:
                    break

                squares = random.sample(range(64), len(all_pieces))
                pieces = list(zip(all_pieces, squares))
                state = ChessState(pieces, random.choice(['w', 'b']))

                # Validate
                wk = next((sq for p, sq in pieces if p == Piece.W_KING), None)
                bk = next((sq for p, sq in pieces if p == Piece.B_KING), None)
                if wk is None or bk is None:
                    continue
                if abs(wk // 8 - bk // 8) <= 1 and abs(wk % 8 - bk % 8) <= 1:
                    continue
                if in_check(state, 'b' if state.turn == 'w' else 'w'):
                    continue

                positions.append(state)

            all_positions.extend(positions)
            self.filter_stats['source_positions_generated'] += len(positions)

        print(f"Generated {len(all_positions)} source positions from {len(self.source_materials)} materials")
        return all_positions

    def get_filter_summary(self) -> str:
        """Get summary of filter statistics (for targeted mode)"""
        if not self.target_material:
            return "No targeting enabled"
        return (f"ChessGame({self.target_material}):\n"
                f"  Target material found: {self.filter_stats['target_material_found']}\n"
                f"  Wrong material filtered: {self.filter_stats['wrong_material_filtered']}\n"
                f"  Source positions generated: {self.filter_stats['source_positions_generated']}")


def _parse_material_string(material: str) -> Tuple[List[int], List[int]]:
    """Parse 'KQRRvKQR' into piece lists"""
    material = material.upper()
    white_str, black_str = material.split('V')

    white_pieces = []
    for c in white_str:
        if c in PIECE_NAMES:
            white_pieces.append(PIECE_NAMES[c])

    black_pieces = []
    for c in black_str:
        if c in PIECE_NAMES:
            black_pieces.append(PIECE_NAMES[c] + 6)  # Black pieces are +6

    return white_pieces, black_pieces


# ============================================================
# MATERIAL UTILITIES (for Layer 1/2 goal setting)
# ============================================================

def get_material_string(state: ChessState) -> str:
    """
    Get material signature as string like 'KQRRvKQR'.

    This is a Layer 0 capability used by Layer 1/2 for goal setting.
    """
    piece_chars = {
        Piece.W_KING: 'K', Piece.W_QUEEN: 'Q', Piece.W_ROOK: 'R',
        Piece.W_BISHOP: 'B', Piece.W_KNIGHT: 'N', Piece.W_PAWN: 'P',
        Piece.B_KING: 'K', Piece.B_QUEEN: 'Q', Piece.B_ROOK: 'R',
        Piece.B_BISHOP: 'B', Piece.B_KNIGHT: 'N', Piece.B_PAWN: 'P',
    }
    white = ''.join(sorted(
        [piece_chars.get(p, '?') for p, sq in state.pieces if p <= 6],
        key=lambda c: 'KQRBNP'.index(c) if c in 'KQRBNP' else 99
    ))
    black = ''.join(sorted(
        [piece_chars.get(p, '?') for p, sq in state.pieces if p > 6],
        key=lambda c: 'KQRBNP'.index(c) if c in 'KQRBNP' else 99
    ))
    return f"{white}v{black}"


def get_capture_result_material(state: ChessState, move: Tuple) -> Optional[str]:
    """
    If move is a capture, return the resulting material string.
    Returns None if not a capture.

    This is a Layer 0 capability used by Layer 1/2 for goal filtering.
    """
    if move[2] is None:  # Not a capture
        return None
    child = apply_move(state, move)
    return get_material_string(child)


def get_parent_materials(target_material: str) -> List[str]:
    """
    Get all material configurations that can capture down to target.

    Example: 'KQRRvKQR' can come from KQRRvKQRR, KQRRvKQRQ, etc.

    This is a Layer 0 capability used by Layer 1/2 for seed generation.
    """
    white_str, black_str = target_material.upper().split('V')
    parents = []

    # Add piece to black (white will capture it)
    for piece in ['Q', 'R', 'B', 'N', 'P']:
        parents.append(f"{white_str}v{black_str}{piece}")

    # Add piece to white (black will capture it)
    for piece in ['Q', 'R', 'B', 'N', 'P']:
        parents.append(f"{white_str}{piece}v{black_str}")

    return parents


def enumerate_material_positions(material: str, syzygy: 'SyzygyProbe',
                                  count: int = 100, max_attempts: int = None) -> List[ChessState]:
    """
    Generate random valid positions with given material.

    If syzygy is provided, only returns positions that syzygy can solve.
    This is a Layer 0 capability used by Layer 1/2 for seed generation.
    """
    if max_attempts is None:
        max_attempts = count * 10

    white_str, black_str = material.upper().split('V')
    white = [PIECE_NAMES[c] for c in white_str]
    black = [PIECE_NAMES[c] + 6 for c in black_str]
    all_pieces = white + black

    positions = []
    for _ in range(max_attempts):
        if len(positions) >= count:
            break

        squares = random.sample(range(64), len(all_pieces))
        pieces = list(zip(all_pieces, squares))
        state = ChessState(pieces, random.choice(['w', 'b']))

        # Validate
        wk = next((sq for p, sq in pieces if p == Piece.W_KING), None)
        bk = next((sq for p, sq in pieces if p == Piece.B_KING), None)
        if wk is None or bk is None:
            continue
        if abs(wk // 8 - bk // 8) <= 1 and abs(wk % 8 - bk % 8) <= 1:
            continue
        if in_check(state, 'b' if state.turn == 'w' else 'w'):
            continue

        # Check syzygy if available
        if syzygy is not None and syzygy.probe(state) is None:
            continue

        positions.append(state)

    return positions


# ============================================================
# GENERAL UTILITIES
# ============================================================

def random_position(material: str, max_attempts=1000) -> Optional[ChessState]:
    """Generate random position with given material"""
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


def create_chess_solver(syzygy_path: str = "./syzygy",
                        min_pieces: int = 7, max_pieces: int = 8):
    """Create a HOLOS solver for chess"""
    from holos.holos import HOLOSSolver

    game = ChessGame(syzygy_path, min_pieces, max_pieces)
    solver = HOLOSSolver(game, name=f"chess_{min_pieces}to{max_pieces}")
    return solver, game


def create_targeted_solver(target_material: str = "KQRRvKQR",
                           syzygy_path: str = "./syzygy",
                           max_memory_mb: int = 4000):
    """
    Create a HOLOS solver for targeted material search.

    Returns solver and game configured to find all 8-piece positions
    leading to the target 7-piece material.

    This replaces the deprecated chess_targeted.py module.
    """
    from holos.holos import HOLOSSolver

    target_pieces = len(target_material.replace('V', '').replace('v', ''))
    game = ChessGame(
        syzygy_path,
        min_pieces=target_pieces,
        max_pieces=target_pieces + 1,
        target_material=target_material
    )
    solver = HOLOSSolver(game, name=f"targeted_{target_material}", max_memory_mb=max_memory_mb)

    return solver, game


# ============================================================
# BACKWARDS COMPATIBILITY
# ============================================================

# Alias for backwards compatibility with chess_targeted.py
# Users should migrate to ChessGame(target_material=...) instead
class TargetedChessGame(ChessGame):
    """
    DEPRECATED: Use ChessGame with target_material parameter instead.

    This class exists for backwards compatibility with code that imported
    TargetedChessGame from chess_targeted.py. New code should use:

        game = ChessGame(syzygy_path, target_material="KQRRvKQR")

    Instead of:

        game = TargetedChessGame(syzygy_path, "KQRRvKQR")
    """

    def __init__(self, syzygy_path: str = "./syzygy",
                 target_material: str = "KQRRvKQR",
                 source_materials: List[str] = None):
        import warnings
        warnings.warn(
            "TargetedChessGame is deprecated. Use ChessGame(target_material=...) instead.",
            DeprecationWarning,
            stacklevel=2
        )

        target_pieces = len(target_material.replace('V', '').replace('v', ''))
        super().__init__(
            syzygy_path=syzygy_path,
            min_pieces=target_pieces,
            max_pieces=target_pieces + 1,
            target_material=target_material,
            source_materials=source_materials
        )

    def summary(self) -> str:
        """Alias for get_filter_summary() for backwards compatibility"""
        return self.get_filter_summary()


# Helper functions that were in chess_targeted.py
def get_8piece_variants(target_7piece: str) -> List[str]:
    """
    Alias for get_parent_materials for backwards compatibility.

    Get all 8-piece material configs that could capture down to target 7-piece.
    """
    return get_parent_materials(target_7piece)


def material_string(state: ChessState) -> str:
    """Alias for get_material_string for backwards compatibility."""
    return get_material_string(state)


def parse_material_string(material: str) -> Tuple[List[int], List[int]]:
    """Public alias for _parse_material_string for backwards compatibility."""
    return _parse_material_string(material)
