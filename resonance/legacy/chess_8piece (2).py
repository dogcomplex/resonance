"""
8-Piece Chess Endgame Solver
=============================

Using Crystalline Wave Propagation

GOAL: Solve 8-piece endgames by growing from known 7-piece tablebases.

KEY INSIGHT: We don't solve from scratch. We treat existing 7-piece 
tablebases as "crystallized boundaries" and grow the 8th piece INTO them.

PHYSICS MODEL:
- 7-piece solutions = frozen boundary (like ground in lightning)
- 8-piece positions = unknown region  
- Captures that lead to 7-piece = contact points
- Solution crystallizes from contact points backward

EQUIVALENCE CLASSES FOR CHESS:
- Board symmetry (8 symmetries for centerless positions)
- Piece permutation (which specific rook doesn't matter)
- Pawn structure patterns
- Material signature (what pieces exist)

IMPLEMENTATION PHASES:
1. Load/connect to 7-piece Syzygy tablebases
2. Generate 8-piece positions (specific material combo)
3. Identify "contact points" (captures leading to 7-piece)
4. Lightning: find paths from 8-piece to contact points
5. Crystal: grow solutions backward from contact points

STARTING SIMPLE:
- KQRvKQR (King+Queen+Rook vs King+Queen+Rook)
- This has rich tactics but limited piece count
- ~billions of positions, not trillions
"""

import os
import sys
import time
import pickle
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
from enum import IntEnum
import struct

# ============================================================
# CHESS PRIMITIVES
# ============================================================

class Piece(IntEnum):
    EMPTY = 0
    W_KING = 1
    W_QUEEN = 2
    W_ROOK = 3
    W_BISHOP = 4
    W_KNIGHT = 5
    W_PAWN = 6
    B_KING = 7
    B_QUEEN = 8
    B_ROOK = 9
    B_BISHOP = 10
    B_KNIGHT = 11
    B_PAWN = 12

PIECE_CHARS = '.KQRBNPkqrbnp'
PIECE_VALUES = {
    Piece.W_QUEEN: 9, Piece.W_ROOK: 5, Piece.W_BISHOP: 3, 
    Piece.W_KNIGHT: 3, Piece.W_PAWN: 1,
    Piece.B_QUEEN: 9, Piece.B_ROOK: 5, Piece.B_BISHOP: 3,
    Piece.B_KNIGHT: 3, Piece.B_PAWN: 1,
}

def is_white(piece):
    return 1 <= piece <= 6

def is_black(piece):
    return 7 <= piece <= 12

def piece_color(piece):
    if is_white(piece): return 'w'
    if is_black(piece): return 'b'
    return None


@dataclass
class ChessState:
    """
    Compact chess endgame state.
    
    For 8-piece endgames, we use piece-list representation:
    - List of (piece_type, square) tuples
    - Square: 0-63 (a1=0, h8=63)
    - Side to move
    - No castling/en-passant (endgame)
    """
    pieces: Tuple[Tuple[int, int], ...]  # ((piece, square), ...)
    turn: str  # 'w' or 'b'
    _hash: int = None
    
    def __hash__(self):
        if self._hash is None:
            # Canonical form: sorted pieces, with symmetry reduction
            canonical = self._canonical_form()
            self._hash = hash((canonical, self.turn))
        return self._hash
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def _canonical_form(self):
        """
        Reduce by board symmetry.
        For pawnless endgames: 8 symmetries (D4 group)
        With pawns: only 2 (vertical flip)
        """
        # Check if any pawns
        has_pawns = any(p in (Piece.W_PAWN, Piece.B_PAWN) for p, _ in self.pieces)
        
        if has_pawns:
            # Only horizontal symmetry
            forms = [self.pieces, self._flip_horizontal()]
        else:
            # Full D4 symmetry (8 transformations)
            forms = [
                self.pieces,
                self._flip_horizontal(),
                self._flip_vertical(),
                self._flip_diagonal(),
                self._flip_antidiagonal(),
                self._rotate_90(),
                self._rotate_180(),
                self._rotate_270(),
            ]
        
        # Also sort pieces within each form
        sorted_forms = [tuple(sorted(f)) for f in forms]
        return min(sorted_forms)
    
    def _transform_square(self, sq, transform):
        """Apply transformation to square"""
        rank, file = sq // 8, sq % 8
        
        if transform == 'h':  # Horizontal flip
            return rank * 8 + (7 - file)
        elif transform == 'v':  # Vertical flip
            return (7 - rank) * 8 + file
        elif transform == 'd':  # Diagonal flip (transpose)
            return file * 8 + rank
        elif transform == 'a':  # Anti-diagonal flip
            return (7 - file) * 8 + (7 - rank)
        elif transform == 'r90':  # Rotate 90 clockwise
            return file * 8 + (7 - rank)
        elif transform == 'r180':  # Rotate 180
            return (7 - rank) * 8 + (7 - file)
        elif transform == 'r270':  # Rotate 270 clockwise
            return (7 - file) * 8 + rank
        return sq
    
    def _flip_horizontal(self):
        return tuple((p, self._transform_square(sq, 'h')) for p, sq in self.pieces)
    
    def _flip_vertical(self):
        return tuple((p, self._transform_square(sq, 'v')) for p, sq in self.pieces)
    
    def _flip_diagonal(self):
        return tuple((p, self._transform_square(sq, 'd')) for p, sq in self.pieces)
    
    def _flip_antidiagonal(self):
        return tuple((p, self._transform_square(sq, 'a')) for p, sq in self.pieces)
    
    def _rotate_90(self):
        return tuple((p, self._transform_square(sq, 'r90')) for p, sq in self.pieces)
    
    def _rotate_180(self):
        return tuple((p, self._transform_square(sq, 'r180')) for p, sq in self.pieces)
    
    def _rotate_270(self):
        return tuple((p, self._transform_square(sq, 'r270')) for p, sq in self.pieces)
    
    def to_board(self):
        """Convert to 64-element board array"""
        board = [Piece.EMPTY] * 64
        for piece, sq in self.pieces:
            board[sq] = piece
        return board
    
    @staticmethod
    def from_board(board, turn):
        """Create from 64-element board array"""
        pieces = []
        for sq, piece in enumerate(board):
            if piece != Piece.EMPTY:
                pieces.append((piece, sq))
        return ChessState(tuple(pieces), turn)
    
    def material_key(self):
        """Material signature for equivalence classes"""
        white = sorted(p for p, _ in self.pieces if is_white(p))
        black = sorted(p for p, _ in self.pieces if is_black(p))
        return (tuple(white), tuple(black))
    
    def piece_count(self):
        return len(self.pieces)
    
    def display(self):
        """Pretty print the board"""
        board = self.to_board()
        print(f"\n  Turn: {'White' if self.turn == 'w' else 'Black'}")
        print("  +-----------------+")
        for rank in range(7, -1, -1):
            print(f"{rank+1} |", end=" ")
            for file in range(8):
                sq = rank * 8 + file
                piece = board[sq]
                print(PIECE_CHARS[piece], end=" ")
            print("|")
        print("  +-----------------+")
        print("    a b c d e f g h")


# ============================================================
# MOVE GENERATION
# ============================================================

def generate_moves(state: ChessState) -> List[Tuple[int, int, Optional[int]]]:
    """
    Generate all legal moves.
    Returns list of (from_sq, to_sq, captured_piece or None)
    """
    board = state.to_board()
    moves = []
    
    is_white_turn = state.turn == 'w'
    
    for piece, from_sq in state.pieces:
        # Skip opponent's pieces
        if is_white_turn and is_black(piece):
            continue
        if not is_white_turn and is_white(piece):
            continue
        
        # Generate moves based on piece type
        piece_type = piece if is_white(piece) else piece - 6
        
        if piece_type == 1:  # King
            moves.extend(_king_moves(board, from_sq, is_white_turn))
        elif piece_type == 2:  # Queen
            moves.extend(_queen_moves(board, from_sq, is_white_turn))
        elif piece_type == 3:  # Rook
            moves.extend(_rook_moves(board, from_sq, is_white_turn))
        elif piece_type == 4:  # Bishop
            moves.extend(_bishop_moves(board, from_sq, is_white_turn))
        elif piece_type == 5:  # Knight
            moves.extend(_knight_moves(board, from_sq, is_white_turn))
        elif piece_type == 6:  # Pawn
            moves.extend(_pawn_moves(board, from_sq, is_white_turn))
    
    # Filter out moves that leave king in check
    legal_moves = []
    for move in moves:
        new_state = apply_move(state, move)
        if not is_in_check(new_state, state.turn):
            legal_moves.append(move)
    
    return legal_moves


def _king_moves(board, from_sq, is_white_turn):
    moves = []
    rank, file = from_sq // 8, from_sq % 8
    
    for dr, df in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
        new_rank, new_file = rank + dr, file + df
        if 0 <= new_rank < 8 and 0 <= new_file < 8:
            to_sq = new_rank * 8 + new_file
            target = board[to_sq]
            
            # Can't capture own pieces
            if is_white_turn and is_white(target):
                continue
            if not is_white_turn and is_black(target):
                continue
            
            captured = target if target != Piece.EMPTY else None
            moves.append((from_sq, to_sq, captured))
    
    return moves


def _sliding_moves(board, from_sq, is_white_turn, directions):
    moves = []
    rank, file = from_sq // 8, from_sq % 8
    
    for dr, df in directions:
        for dist in range(1, 8):
            new_rank, new_file = rank + dr * dist, file + df * dist
            if not (0 <= new_rank < 8 and 0 <= new_file < 8):
                break
            
            to_sq = new_rank * 8 + new_file
            target = board[to_sq]
            
            if target == Piece.EMPTY:
                moves.append((from_sq, to_sq, None))
            elif (is_white_turn and is_black(target)) or (not is_white_turn and is_white(target)):
                moves.append((from_sq, to_sq, target))
                break
            else:
                break
    
    return moves


def _rook_moves(board, from_sq, is_white_turn):
    return _sliding_moves(board, from_sq, is_white_turn, [(0,1), (0,-1), (1,0), (-1,0)])


def _bishop_moves(board, from_sq, is_white_turn):
    return _sliding_moves(board, from_sq, is_white_turn, [(1,1), (1,-1), (-1,1), (-1,-1)])


def _queen_moves(board, from_sq, is_white_turn):
    return _rook_moves(board, from_sq, is_white_turn) + _bishop_moves(board, from_sq, is_white_turn)


def _knight_moves(board, from_sq, is_white_turn):
    moves = []
    rank, file = from_sq // 8, from_sq % 8
    
    for dr, df in [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]:
        new_rank, new_file = rank + dr, file + df
        if 0 <= new_rank < 8 and 0 <= new_file < 8:
            to_sq = new_rank * 8 + new_file
            target = board[to_sq]
            
            if is_white_turn and is_white(target):
                continue
            if not is_white_turn and is_black(target):
                continue
            
            captured = target if target != Piece.EMPTY else None
            moves.append((from_sq, to_sq, captured))
    
    return moves


def _pawn_moves(board, from_sq, is_white_turn):
    moves = []
    rank, file = from_sq // 8, from_sq % 8
    
    direction = 1 if is_white_turn else -1
    start_rank = 1 if is_white_turn else 6
    promo_rank = 7 if is_white_turn else 0
    
    # Forward move
    new_rank = rank + direction
    if 0 <= new_rank < 8:
        to_sq = new_rank * 8 + file
        if board[to_sq] == Piece.EMPTY:
            moves.append((from_sq, to_sq, None))
            
            # Double push from start
            if rank == start_rank:
                to_sq2 = (rank + 2 * direction) * 8 + file
                if board[to_sq2] == Piece.EMPTY:
                    moves.append((from_sq, to_sq2, None))
    
    # Captures
    for df in [-1, 1]:
        new_file = file + df
        if 0 <= new_file < 8 and 0 <= new_rank < 8:
            to_sq = new_rank * 8 + new_file
            target = board[to_sq]
            if (is_white_turn and is_black(target)) or (not is_white_turn and is_white(target)):
                moves.append((from_sq, to_sq, target))
    
    return moves


def apply_move(state: ChessState, move: Tuple[int, int, Optional[int]]) -> ChessState:
    """Apply a move and return new state"""
    from_sq, to_sq, captured = move
    
    new_pieces = []
    moved_piece = None
    
    for piece, sq in state.pieces:
        if sq == from_sq:
            moved_piece = piece
        elif sq == to_sq:
            continue  # Captured
        else:
            new_pieces.append((piece, sq))
    
    if moved_piece is not None:
        # Handle pawn promotion
        if moved_piece in (Piece.W_PAWN, Piece.B_PAWN):
            promo_rank = 7 if moved_piece == Piece.W_PAWN else 0
            if to_sq // 8 == promo_rank:
                moved_piece = Piece.W_QUEEN if is_white(moved_piece) else Piece.B_QUEEN
        
        new_pieces.append((moved_piece, to_sq))
    
    new_turn = 'b' if state.turn == 'w' else 'w'
    return ChessState(tuple(new_pieces), new_turn)


def is_in_check(state: ChessState, color: str) -> bool:
    """Check if the given color's king is in check"""
    board = state.to_board()
    
    # Find king
    king_piece = Piece.W_KING if color == 'w' else Piece.B_KING
    king_sq = None
    for piece, sq in state.pieces:
        if piece == king_piece:
            king_sq = sq
            break
    
    if king_sq is None:
        return True  # No king = definitely in check (checkmate)
    
    # Check if any opponent piece attacks the king
    opponent_is_white = color == 'b'
    
    for piece, sq in state.pieces:
        if opponent_is_white and not is_white(piece):
            continue
        if not opponent_is_white and not is_black(piece):
            continue
        
        # Check if this piece attacks the king square
        if attacks_square(board, sq, king_sq, piece):
            return True
    
    return False


def attacks_square(board, from_sq, target_sq, piece) -> bool:
    """Check if piece at from_sq attacks target_sq"""
    piece_type = piece if is_white(piece) else piece - 6
    
    from_rank, from_file = from_sq // 8, from_sq % 8
    to_rank, to_file = target_sq // 8, target_sq % 8
    dr, df = to_rank - from_rank, to_file - from_file
    
    if piece_type == 1:  # King
        return abs(dr) <= 1 and abs(df) <= 1 and (dr != 0 or df != 0)
    
    elif piece_type == 2:  # Queen
        return attacks_square(board, from_sq, target_sq, Piece.W_ROOK) or \
               attacks_square(board, from_sq, target_sq, Piece.W_BISHOP)
    
    elif piece_type == 3:  # Rook
        if dr != 0 and df != 0:
            return False
        return _path_clear(board, from_sq, target_sq)
    
    elif piece_type == 4:  # Bishop
        if abs(dr) != abs(df):
            return False
        return _path_clear(board, from_sq, target_sq)
    
    elif piece_type == 5:  # Knight
        return (abs(dr), abs(df)) in [(1, 2), (2, 1)]
    
    elif piece_type == 6:  # Pawn
        direction = 1 if is_white(piece) else -1
        return dr == direction and abs(df) == 1
    
    return False


def _path_clear(board, from_sq, to_sq) -> bool:
    """Check if path between squares is clear (for sliding pieces)"""
    from_rank, from_file = from_sq // 8, from_sq % 8
    to_rank, to_file = to_sq // 8, to_sq % 8
    
    dr = 0 if to_rank == from_rank else (1 if to_rank > from_rank else -1)
    df = 0 if to_file == from_file else (1 if to_file > from_file else -1)
    
    rank, file = from_rank + dr, from_file + df
    while (rank, file) != (to_rank, to_file):
        if board[rank * 8 + file] != Piece.EMPTY:
            return False
        rank += dr
        file += df
    
    return True


def is_checkmate(state: ChessState) -> bool:
    """Check if current side is checkmated"""
    if not is_in_check(state, state.turn):
        return False
    return len(generate_moves(state)) == 0


def is_stalemate(state: ChessState) -> bool:
    """Check if current side is stalemated"""
    if is_in_check(state, state.turn):
        return False
    return len(generate_moves(state)) == 0


def is_terminal(state: ChessState) -> Tuple[bool, Optional[int]]:
    """
    Check if position is terminal.
    Returns (is_terminal, value) where value is:
    - +1 for white wins
    - -1 for black wins
    - 0 for draw
    """
    if is_checkmate(state):
        # Current side is checkmated, opponent wins
        return True, -1 if state.turn == 'w' else 1
    
    if is_stalemate(state):
        return True, 0
    
    # Check for insufficient material
    # (simplified - just K vs K)
    if len(state.pieces) == 2:
        return True, 0
    
    return False, None


# ============================================================
# SYZYGY TABLEBASE INTERFACE (STUB)
# ============================================================

class SyzygyProbe:
    """
    Interface to 7-piece Syzygy tablebases.
    
    In real implementation, would use python-chess or direct DTZ probe.
    For now, this is a stub that returns None (unknown).
    """
    
    def __init__(self, tablebase_path=None):
        self.path = tablebase_path
        self.available = False
        
        # Try to load python-chess if available
        try:
            import chess
            import chess.syzygy
            if tablebase_path and os.path.exists(tablebase_path):
                self.tablebase = chess.syzygy.Tablebase()
                self.tablebase.add_directory(tablebase_path)
                self.available = True
                print(f"Loaded Syzygy tablebases from {tablebase_path}")
        except ImportError:
            print("python-chess not available, Syzygy probe disabled")
        except Exception as e:
            print(f"Failed to load Syzygy: {e}")
    
    def probe(self, state: ChessState) -> Optional[int]:
        """
        Probe tablebase for position value.
        Returns +1 (white wins), -1 (black wins), 0 (draw), or None (unknown).
        """
        if not self.available:
            return None
        
        if state.piece_count() > 7:
            return None
        
        # Convert to python-chess board and probe
        try:
            import chess
            board = chess.Board()
            board.clear()
            
            for piece, sq in state.pieces:
                chess_piece = chess.Piece(
                    piece_type=(piece - 1) % 6 + 1,
                    color=is_white(piece)
                )
                board.set_piece_at(sq, chess_piece)
            
            board.turn = state.turn == 'w'
            
            wdl = self.tablebase.probe_wdl(board)
            if wdl > 0:
                return 1 if state.turn == 'w' else -1
            elif wdl < 0:
                return -1 if state.turn == 'w' else 1
            else:
                return 0
        except:
            return None


# ============================================================
# CRYSTALLINE ENDGAME SOLVER
# ============================================================

SAVE_DIR = "./chess_8piece_state"

class ChessEndgameSolver:
    """
    8-piece chess endgame solver using crystalline wave propagation.
    
    Uses 7-piece Syzygy tablebases as the "frozen boundary" and
    grows solutions backward from captures that reach 7 pieces.
    """
    
    def __init__(self, tablebase_path=None):
        self.syzygy = SyzygyProbe(tablebase_path)
        
        self.solved: Dict[int, int] = {}
        self.frontier: Dict[int, ChessState] = {}
        self.all_seen: Set[int] = set()
        
        self.children: Dict[int, List[int]] = {}
        self.parents: Dict[int, List[int]] = defaultdict(list)
        self.state_turns: Dict[int, str] = {}
        
        self.crystal_spine: Set[int] = set()
        self.crystal_front: Set[int] = set()
        
        self.equiv_classes: Dict[Tuple, Set[int]] = defaultdict(set)
        self.equiv_outcomes: Dict[Tuple, Optional[int]] = {}
        
        self.phase = "generating"
        self.material_config = None
        
        self.metrics = {
            'total_time': 0.0,
            'positions_generated': 0,
            'syzygy_probes': 0,
            'syzygy_hits': 0,
        }
    
    def generate_positions(self, material: str):
        """
        Generate all legal positions for given material configuration.
        
        Material format: "KQRvKQR" (white pieces v black pieces)
        """
        self.material_config = material
        
        # Parse material - map characters to piece types
        piece_map = {
            'K': 'KING', 'Q': 'QUEEN', 'R': 'ROOK', 
            'B': 'BISHOP', 'N': 'KNIGHT', 'P': 'PAWN'
        }
        
        white_str, black_str = material.upper().split('V')
        white_pieces = [getattr(Piece, f'W_{piece_map[c]}') for c in white_str]
        black_pieces = [getattr(Piece, f'B_{piece_map[c]}') for c in black_str]
        
        all_pieces = white_pieces + black_pieces
        
        print(f"\nGenerating positions for {material}")
        print(f"  White: {[PIECE_CHARS[p] for p in white_pieces]}")
        print(f"  Black: {[PIECE_CHARS[p] for p in black_pieces]}")
        print(f"  Total pieces: {len(all_pieces)}")
        
        # Generate all placements
        # This is combinatorial: 64 choose n * arrangements
        # For 8 pieces: roughly 64^8 / 8! ~ 4 billion (with symmetry ~ 500M)
        
        # For efficiency, generate incrementally
        self._gen_count = 0
        self._gen_start = time.time()
        
        print(f"  Generating white-to-move positions...")
        self._generate_recursive(all_pieces, [], 0, 'w')
        
        print(f"  Generating black-to-move positions...")
        self._generate_recursive(all_pieces, [], 0, 'b')
        
        print(f"  Generated: {len(self.frontier):,} unique legal positions")
        print(f"  Time: {time.time() - self._gen_start:.1f}s")
        self.metrics['positions_generated'] = len(self.frontier)
    
    def _generate_recursive(self, pieces_to_place, placed, min_sq, turn):
        """Recursively generate positions"""
        if not pieces_to_place:
            # All pieces placed, create state
            self._gen_count += 1
            if self._gen_count % 100000 == 0:
                print(f"    Checked {self._gen_count:,} placements, "
                      f"found {len(self.frontier):,} legal...")
            
            state = ChessState(tuple(placed), turn)
            h = hash(state)
            
            if h not in self.all_seen:
                # Check legality (kings not adjacent, side to move not giving check)
                if self._is_legal_position(state):
                    self.all_seen.add(h)
                    self.frontier[h] = state
                    self.state_turns[h] = turn
                    
                    # Equivalence class by material
                    key = state.material_key()
                    self.equiv_classes[key].add(h)
            return
        
        piece = pieces_to_place[0]
        remaining = pieces_to_place[1:]
        
        # Place this piece on each valid square
        for sq in range(min_sq, 64):
            # Check if square already occupied
            if any(s == sq for _, s in placed):
                continue
            
            new_placed = placed + [(piece, sq)]
            
            # For identical pieces, enforce ordering to avoid duplicates
            if remaining and remaining[0] == piece:
                self._generate_recursive(remaining, new_placed, sq + 1, turn)
            else:
                self._generate_recursive(remaining, new_placed, 0, turn)
    
    def _is_legal_position(self, state: ChessState) -> bool:
        """Check if position is legal"""
        board = state.to_board()
        
        # Find kings
        wk_sq = bk_sq = None
        for piece, sq in state.pieces:
            if piece == Piece.W_KING:
                wk_sq = sq
            elif piece == Piece.B_KING:
                bk_sq = sq
        
        if wk_sq is None or bk_sq is None:
            return False
        
        # Kings not adjacent
        wk_rank, wk_file = wk_sq // 8, wk_sq % 8
        bk_rank, bk_file = bk_sq // 8, bk_sq % 8
        if abs(wk_rank - bk_rank) <= 1 and abs(wk_file - bk_file) <= 1:
            return False
        
        # Side not to move must not be in check
        opponent = 'b' if state.turn == 'w' else 'w'
        if is_in_check(state, opponent):
            return False
        
        return True
    
    def solve(self):
        """Main solving loop"""
        print("="*60)
        print("♔ 8-PIECE CHESS ENDGAME SOLVER ♔")
        print("="*60)
        
        if not self.frontier:
            print("No positions generated! Call generate_positions() first.")
            return
        
        start_time = time.time()
        
        # Phase 1: Probe Syzygy for captures leading to 7-piece
        print(f"\n--- PHASE 1: Contact Points (7-piece boundary) ---")
        contact_points = self._find_contact_points()
        print(f"  Found {contact_points:,} contact points")
        
        # Phase 2: Propagate solutions backward
        print(f"\n--- PHASE 2: Crystallization ---")
        self._crystallize()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"Total positions: {len(self.all_seen):,}")
        print(f"Solved: {len(self.solved):,} ({100*len(self.solved)/len(self.all_seen):.1f}%)")
        print(f"Time: {total_time:.1f}s")
        
        # Outcome distribution
        wins = sum(1 for v in self.solved.values() if v == 1)
        losses = sum(1 for v in self.solved.values() if v == -1)
        draws = sum(1 for v in self.solved.values() if v == 0)
        print(f"\nOutcomes:")
        print(f"  White wins: {wins:,}")
        print(f"  Black wins: {losses:,}")
        print(f"  Draws: {draws:,}")
    
    def _find_contact_points(self):
        """
        Find positions where a capture leads to a 7-piece position
        that can be probed in Syzygy.
        """
        contact = 0
        
        for h, state in list(self.frontier.items()):
            if h in self.solved:
                continue
            
            # Check terminal
            term, value = is_terminal(state)
            if term:
                self.solved[h] = value
                self.crystal_front.add(h)
                contact += 1
                continue
            
            # Generate moves
            moves = generate_moves(state)
            child_hashes = []
            
            for move in moves:
                child = apply_move(state, move)
                ch = hash(child)
                child_hashes.append(ch)
                self.parents[ch].append(h)
                
                # Is this a capture leading to 7-piece?
                if move[2] is not None and child.piece_count() == 7:
                    # Probe Syzygy
                    self.metrics['syzygy_probes'] += 1
                    value = self.syzygy.probe(child)
                    
                    if value is not None:
                        self.metrics['syzygy_hits'] += 1
                        self.solved[ch] = value
                        self.crystal_front.add(ch)
                        contact += 1
            
            self.children[h] = child_hashes
        
        return contact
    
    def _crystallize(self):
        """Propagate solutions backward from contact points"""
        iteration = 0
        
        while True:
            iteration += 1
            newly_solved = 0
            
            for h in list(self.children.keys()):
                if h in self.solved:
                    continue
                
                child_list = self.children[h]
                child_values = []
                unknown = 0
                
                for ch in child_list:
                    if ch in self.solved:
                        child_values.append(self.solved[ch])
                    else:
                        unknown += 1
                
                if not child_values:
                    continue
                
                turn = self.state_turns.get(h, 'w')
                
                # Early termination
                if turn == 'w' and 1 in child_values:
                    self.solved[h] = 1
                    newly_solved += 1
                elif turn == 'b' and -1 in child_values:
                    self.solved[h] = -1
                    newly_solved += 1
                elif unknown == 0:
                    value = max(child_values) if turn == 'w' else min(child_values)
                    self.solved[h] = value
                    newly_solved += 1
            
            if newly_solved == 0:
                break
            
            print(f"  Iteration {iteration}: +{newly_solved:,}, total={len(self.solved):,}")
        
        return len(self.solved)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Path to Syzygy tablebases
    syzygy_path = "./syzygy"
    
    solver = ChessEndgameSolver(tablebase_path=syzygy_path)
    
    # Test with material specified on command line, or default
    if len(sys.argv) > 1:
        material = sys.argv[1]
    else:
        # Default: KQRRvKQRR (8-piece!) - we have the 7-piece KQRRvKQR table!
        material = "KQRRvKQRR"
    
    print(f"\n{'='*60}")
    print(f"ATTACKING {material} ({sum(1 for c in material if c.isupper())} pieces)")
    print(f"{'='*60}")
    print(f"\nThis is {'an 8-piece endgame!' if sum(1 for c in material if c.isupper()) == 8 else 'a ' + str(sum(1 for c in material if c.isupper())) + '-piece endgame'}")
    print(f"Using 7-piece Syzygy tables as boundary condition...")
    
    solver.generate_positions(material)
    solver.solve()
