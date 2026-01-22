"""
chess_core.py - Core chess primitives for endgame solving

Shared module for:
- Piece representation
- Board state with symmetry reduction
- Move generation
- Check/checkmate detection
- Syzygy interface
"""

import os
from enum import IntEnum
from typing import List, Tuple, Optional, Set

# ============================================================
# PIECE REPRESENTATION
# ============================================================

class Piece(IntEnum):
    EMPTY = 0
    W_KING = 1; W_QUEEN = 2; W_ROOK = 3; W_BISHOP = 4; W_KNIGHT = 5; W_PAWN = 6
    B_KING = 7; B_QUEEN = 8; B_ROOK = 9; B_BISHOP = 10; B_KNIGHT = 11; B_PAWN = 12

PIECE_CHARS = '.KQRBNPkqrbnp'
PIECE_NAMES = {
    'K': Piece.W_KING, 'Q': Piece.W_QUEEN, 'R': Piece.W_ROOK,
    'B': Piece.W_BISHOP, 'N': Piece.W_KNIGHT, 'P': Piece.W_PAWN
}

def is_white(p): return 1 <= p <= 6
def is_black(p): return 7 <= p <= 12
def piece_type(p): return ((p - 1) % 6) + 1 if p > 0 else 0


# ============================================================
# CHESS STATE
# ============================================================

class ChessState:
    """
    Minimal chess state for endgames.
    Uses piece-list representation with symmetry reduction.
    """
    __slots__ = ['pieces', 'turn', '_hash']
    
    def __init__(self, pieces, turn='w'):
        self.pieces = tuple(sorted(pieces))
        self.turn = turn
        self._hash = None
    
    def __hash__(self):
        if self._hash is None:
            canonical = self._canonical()
            self._hash = hash((canonical, self.turn))
        return self._hash
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def _canonical(self):
        """Reduce by board symmetry (horizontal flip for pawnless)"""
        flipped = tuple(sorted((p, self._flip_h(sq)) for p, sq in self.pieces))
        return min(self.pieces, flipped)
    
    def _flip_h(self, sq):
        rank, file = sq // 8, sq % 8
        return rank * 8 + (7 - file)
    
    def to_board(self):
        board = [Piece.EMPTY] * 64
        for p, sq in self.pieces:
            board[sq] = p
        return board
    
    def piece_count(self):
        return len(self.pieces)
    
    def material_key(self):
        """Material signature for equivalence classes"""
        white = tuple(sorted(p for p, _ in self.pieces if is_white(p)))
        black = tuple(sorted(p for p, _ in self.pieces if is_black(p)))
        return (white, black)
    
    def material_string(self):
        """Human readable material string like 'KQRRvKQRR'"""
        mat = self.material_key()
        w_str = ''.join(PIECE_CHARS[p] for p in mat[0])
        b_str = ''.join(PIECE_CHARS[p].lower() for p in mat[1])
        return f"{w_str}v{b_str}"
    
    def to_fen(self):
        """Convert to FEN string"""
        board = self.to_board()
        rows = []
        for rank in range(7, -1, -1):
            row = ""
            empty = 0
            for file in range(8):
                p = board[rank * 8 + file]
                if p == Piece.EMPTY:
                    empty += 1
                else:
                    if empty > 0:
                        row += str(empty)
                        empty = 0
                    row += PIECE_CHARS[p]
            if empty > 0:
                row += str(empty)
            rows.append(row)
        return f"{'/'.join(rows)} {self.turn} - - 0 1"
    
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
    
    def compact(self):
        """Compact representation for storage"""
        # Each piece: 4 bits type + 6 bits square = 10 bits
        # 8 pieces = 80 bits = 10 bytes + 1 bit turn
        # We'll use a simpler format: bytes
        data = bytearray()
        data.append(len(self.pieces))
        data.append(0 if self.turn == 'w' else 1)
        for p, sq in self.pieces:
            data.append(p)
            data.append(sq)
        return bytes(data)
    
    @staticmethod
    def from_compact(data):
        n = data[0]
        turn = 'w' if data[1] == 0 else 'b'
        pieces = []
        for i in range(n):
            p = data[2 + i*2]
            sq = data[3 + i*2]
            pieces.append((p, sq))
        return ChessState(pieces, turn)


# ============================================================
# MOVE GENERATION
# ============================================================

def generate_moves(state: ChessState) -> List[Tuple[int, int, Optional[int]]]:
    """
    Generate all legal moves.
    Returns list of (from_sq, to_sq, captured_piece or None)
    """
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
                if not (0 <= to_sq < 64):
                    continue
                if abs((from_sq % 8) - (to_sq % 8)) > 1:
                    continue
                target = board[to_sq]
                if is_white_turn and is_white(target):
                    continue
                if not is_white_turn and is_black(target):
                    continue
                moves.append((from_sq, to_sq, target if target else None))
        
        elif pt == 2:  # Queen
            moves.extend(_sliding(board, from_sq, is_white_turn, 
                                  [-9, -8, -7, -1, 1, 7, 8, 9]))
        
        elif pt == 3:  # Rook
            moves.extend(_sliding(board, from_sq, is_white_turn, [-8, -1, 1, 8]))
        
        elif pt == 4:  # Bishop
            moves.extend(_sliding(board, from_sq, is_white_turn, [-9, -7, 7, 9]))
        
        elif pt == 5:  # Knight
            for d in [-17, -15, -10, -6, 6, 10, 15, 17]:
                to_sq = from_sq + d
                if not (0 <= to_sq < 64):
                    continue
                if abs((from_sq % 8) - (to_sq % 8)) > 2:
                    continue
                target = board[to_sq]
                if is_white_turn and is_white(target):
                    continue
                if not is_white_turn and is_black(target):
                    continue
                moves.append((from_sq, to_sq, target if target else None))
    
    # Filter for legality
    legal = []
    for move in moves:
        new_state = apply_move(state, move)
        if not in_check(new_state, state.turn):
            legal.append(move)
    
    return legal


def _sliding(board, from_sq, is_white_turn, directions):
    moves = []
    from_file = from_sq % 8
    
    for d in directions:
        sq = from_sq + d
        prev_file = from_file
        
        while 0 <= sq < 64:
            sq_file = sq % 8
            # Check wraparound
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


def apply_move(state: ChessState, move: Tuple[int, int, Optional[int]]) -> ChessState:
    """Apply a move and return new state"""
    from_sq, to_sq, captured = move
    new_pieces = []
    moved_piece = None
    
    for p, sq in state.pieces:
        if sq == from_sq:
            moved_piece = p
        elif sq == to_sq:
            pass  # Captured
        else:
            new_pieces.append((p, sq))
    
    if moved_piece:
        new_pieces.append((moved_piece, to_sq))
    
    return ChessState(new_pieces, 'b' if state.turn == 'w' else 'w')


def in_check(state: ChessState, color: str) -> bool:
    """Is the given color's king in check?"""
    board = state.to_board()
    king = Piece.W_KING if color == 'w' else Piece.B_KING
    king_sq = None
    
    for p, sq in state.pieces:
        if p == king:
            king_sq = sq
            break
    
    if king_sq is None:
        return True
    
    enemy_is_white = (color == 'b')
    
    for p, sq in state.pieces:
        if enemy_is_white and not is_white(p):
            continue
        if not enemy_is_white and not is_black(p):
            continue
        
        if attacks(board, sq, king_sq, p):
            return True
    
    return False


def attacks(board, from_sq, to_sq, piece) -> bool:
    """Does piece at from_sq attack to_sq?"""
    pt = piece_type(piece)
    dr = (to_sq // 8) - (from_sq // 8)
    df = (to_sq % 8) - (from_sq % 8)
    
    if pt == 1:  # King
        return abs(dr) <= 1 and abs(df) <= 1 and (dr or df)
    elif pt == 5:  # Knight
        return (abs(dr), abs(df)) in [(1,2), (2,1)]
    elif pt == 3:  # Rook
        if dr != 0 and df != 0:
            return False
        return path_clear(board, from_sq, to_sq)
    elif pt == 4:  # Bishop
        if abs(dr) != abs(df):
            return False
        return path_clear(board, from_sq, to_sq)
    elif pt == 2:  # Queen
        if dr == 0 or df == 0 or abs(dr) == abs(df):
            return path_clear(board, from_sq, to_sq)
    return False


def path_clear(board, from_sq, to_sq) -> bool:
    from_rank, from_file = from_sq // 8, from_sq % 8
    to_rank, to_file = to_sq // 8, to_sq % 8
    
    dr = 0 if to_rank == from_rank else (1 if to_rank > from_rank else -1)
    df = 0 if to_file == from_file else (1 if to_file > from_file else -1)
    
    d = dr * 8 + df
    if d == 0:
        return True
    
    sq = from_sq + d
    while sq != to_sq and 0 <= sq < 64:
        sq_file = sq % 8
        prev_file = (sq - d) % 8
        if abs(sq_file - prev_file) > 1:
            break
        if board[sq] != Piece.EMPTY:
            return False
        sq += d
    return True


def is_checkmate(state: ChessState) -> bool:
    if not in_check(state, state.turn):
        return False
    return len(generate_moves(state)) == 0


def is_stalemate(state: ChessState) -> bool:
    if in_check(state, state.turn):
        return False
    return len(generate_moves(state)) == 0


def is_terminal(state: ChessState) -> Tuple[bool, Optional[int]]:
    """
    Returns (is_terminal, value) where value is:
    +1 for white wins, -1 for black wins, 0 for draw
    """
    if is_checkmate(state):
        return True, (-1 if state.turn == 'w' else 1)
    if is_stalemate(state):
        return True, 0
    if len(state.pieces) == 2:  # KvK
        return True, 0
    return False, None


# ============================================================
# SYZYGY INTERFACE
# ============================================================

class SyzygyProbe:
    """Interface to Syzygy tablebases via python-chess"""
    
    def __init__(self, path="./syzygy"):
        self.path = path
        self.tb = None
        self.available = False
        
        try:
            import chess
            import chess.syzygy
            self.chess = chess
            self.tb = chess.syzygy.Tablebase()
            if os.path.exists(path):
                self.tb.add_directory(path)
                self.available = True
                print(f"✓ Syzygy loaded from {path}")
        except ImportError:
            print("✗ python-chess not available")
        except Exception as e:
            print(f"✗ Syzygy error: {e}")
    
    def probe(self, state: ChessState) -> Optional[int]:
        """
        Probe tablebase for position value.
        Returns +1 (white wins), -1 (black wins), 0 (draw), or None
        """
        if not self.available or state.piece_count() > 7:
            return None
        
        try:
            board = self.chess.Board()
            board.clear()
            
            # Map our piece types to python-chess
            our_to_chess = {
                1: self.chess.KING, 2: self.chess.QUEEN, 3: self.chess.ROOK,
                4: self.chess.BISHOP, 5: self.chess.KNIGHT, 6: self.chess.PAWN,
            }
            
            for p, sq in state.pieces:
                pt = piece_type(p)
                chess_type = our_to_chess[pt]
                color = is_white(p)
                board.set_piece_at(sq, self.chess.Piece(chess_type, color))
            
            board.turn = (state.turn == 'w')
            
            wdl = self.tb.probe_wdl(board)
            
            if wdl is None:
                return None
            if wdl > 0:
                return 1 if state.turn == 'w' else -1
            elif wdl < 0:
                return -1 if state.turn == 'w' else 1
            else:
                return 0
        except:
            return None


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def parse_material(material: str):
    """Parse material string like 'KQRRvKQRR' into piece lists"""
    white_str, black_str = material.upper().split('V')
    white_pieces = [PIECE_NAMES[c] for c in white_str]
    black_pieces = [PIECE_NAMES[c] + 6 for c in black_str]
    return white_pieces, black_pieces


def random_position(material: str, max_attempts=1000) -> Optional[ChessState]:
    """Generate a random legal position with given material"""
    import random
    
    white_pieces, black_pieces = parse_material(material)
    all_pieces = white_pieces + black_pieces
    
    for _ in range(max_attempts):
        squares = random.sample(range(64), len(all_pieces))
        pieces = list(zip(all_pieces, squares))
        state = ChessState(pieces, 'w')
        
        # Find kings
        wk = bk = None
        for p, sq in pieces:
            if p == Piece.W_KING: wk = sq
            if p == Piece.B_KING: bk = sq
        
        if wk is None or bk is None:
            continue
        
        # Kings not adjacent
        if abs(wk // 8 - bk // 8) <= 1 and abs(wk % 8 - bk % 8) <= 1:
            continue
        
        # Side not to move not in check
        if in_check(state, 'b'):
            continue
        
        return state
    
    return None
