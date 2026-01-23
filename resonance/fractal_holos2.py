"""
fractal_holos2.py - Memory-Safe Fractal Holographic Solver

MEMORY ARCHITECTURE:
- HOT TIER (RAM): 10-60GB configurable, actively processed
- COLD TIER (Disk): 200GB+ for overflow and long-term storage

KEY FIXES FROM PREVIOUS VERSIONS:
1. Proactive memory gating - check BEFORE allocating, not after
2. Compact state encoding - 16 bytes instead of 350 bytes per state
3. No dual edge storage - only parent edges for retrograde
4. Chunked frontier processing with automatic spill
5. Batched disk I/O with LMDB for speed
6. Deque for O(1) queue operations

MEMORY SAFETY:
- Hard limit enforcement with configurable buffer
- Automatic spill to disk when approaching limit
- Graceful degradation, never crash from OOM
"""

import os
import sys
import time
import pickle
import random
import gc
import struct
import mmap
from collections import deque
from typing import Dict, Set, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
import threading

# Try to import psutil for accurate memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed. Memory tracking will be estimated.")

# Try to import lmdb for fast disk storage
try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False
    print("WARNING: lmdb not installed. Falling back to sqlite3.")
    import sqlite3


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class MemoryConfig:
    """Memory management configuration"""
    hot_ram_min_gb: float = 10.0      # Minimum RAM to use
    hot_ram_max_gb: float = 60.0      # Maximum RAM before spill
    hot_ram_target_gb: float = 40.0   # Target working set
    cold_disk_max_gb: float = 200.0   # Maximum disk storage

    safety_buffer_gb: float = 5.0     # Keep this much RAM free
    spill_threshold: float = 0.85     # Spill at 85% of max
    batch_size: int = 100_000         # Positions per batch

    checkpoint_interval: int = 300    # Seconds between checkpoints

    def __post_init__(self):
        self.hot_ram_min_bytes = int(self.hot_ram_min_gb * 1024**3)
        self.hot_ram_max_bytes = int(self.hot_ram_max_gb * 1024**3)
        self.hot_ram_target_bytes = int(self.hot_ram_target_gb * 1024**3)
        self.cold_disk_max_bytes = int(self.cold_disk_max_gb * 1024**3)
        self.safety_buffer_bytes = int(self.safety_buffer_gb * 1024**3)


# ============================================================
# MEMORY MONITOR
# ============================================================

class MemoryMonitor:
    """Tracks RAM usage and enforces limits"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._process = psutil.Process() if HAS_PSUTIL else None
        self._last_check = 0
        self._cached_usage = 0
        self._check_interval = 0.1  # seconds

    def get_used_bytes(self) -> int:
        """Get current process memory usage"""
        now = time.time()
        if now - self._last_check < self._check_interval:
            return self._cached_usage

        if self._process:
            self._cached_usage = self._process.memory_info().rss
        else:
            # Rough estimate based on Python object counts
            import gc
            self._cached_usage = sum(sys.getsizeof(obj) for obj in gc.get_objects()[:10000]) * 100

        self._last_check = now
        return self._cached_usage

    def get_available_bytes(self) -> int:
        """Get available RAM for allocation"""
        if HAS_PSUTIL:
            return psutil.virtual_memory().available
        return self.config.hot_ram_max_bytes - self.get_used_bytes()

    def get_used_gb(self) -> float:
        return self.get_used_bytes() / (1024**3)

    def get_available_gb(self) -> float:
        return self.get_available_bytes() / (1024**3)

    def should_spill(self) -> bool:
        """Check if we should spill to disk"""
        used = self.get_used_bytes()
        threshold = self.config.hot_ram_max_bytes * self.config.spill_threshold
        return used > threshold

    def can_allocate(self, bytes_needed: int) -> bool:
        """Check if we can safely allocate this much"""
        available = self.get_available_bytes()
        return available > bytes_needed + self.config.safety_buffer_bytes

    def estimate_batch_cost(self, batch_size: int) -> int:
        """Estimate memory cost for expanding a batch"""
        # Per position: ~100 bytes for hash + state encoding + overhead
        # Per child: ~50 bytes for edge + hash
        # Assume 35 children average for chess
        per_position = 100 + (35 * 50)
        return batch_size * per_position


# ============================================================
# COMPACT STATE ENCODING
# ============================================================

class Piece(IntEnum):
    EMPTY = 0
    W_KING = 1; W_QUEEN = 2; W_ROOK = 3; W_BISHOP = 4; W_KNIGHT = 5; W_PAWN = 6
    B_KING = 7; B_QUEEN = 8; B_ROOK = 9; B_BISHOP = 10; B_KNIGHT = 11; B_PAWN = 12

PIECE_CHARS = '.KQRBNPkqrbnp'
PIECE_NAMES = {'K': Piece.W_KING, 'Q': Piece.W_QUEEN, 'R': Piece.W_ROOK,
               'B': Piece.W_BISHOP, 'N': Piece.W_KNIGHT, 'P': Piece.W_PAWN}

def is_white(p): return 1 <= p <= 6
def is_black(p): return 7 <= p <= 12
def piece_type(p): return ((p - 1) % 6) + 1 if p > 0 else 0


class CompactState:
    """
    Memory-efficient chess state encoding.

    Encoding: 16 bytes total
    - Bytes 0-13: Up to 14 pieces, each as (piece_type:4bits, square:6bits) = 10 bits
      Packed: 14 pieces * 10 bits = 140 bits = 17.5 bytes, but we use 14 bytes
      Actually: piece(4) + sq(6) = 10 bits, pack 8 pieces in 10 bytes
    - Byte 14: piece count + turn bit
    - Byte 15: reserved/checksum

    Simplified: Just use struct for 8-piece positions (most common)
    8 pieces * 2 bytes (piece:1, square:1) = 16 bytes + turn = 17 bytes
    """

    __slots__ = ['_data', '_hash', '_pieces_cache', '_board_cache']

    # Format: 8 pieces as (piece, square) pairs + turn byte
    STRUCT_FMT = '<16BH'  # 16 bytes for pieces + 2 bytes for turn/hash seed
    STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

    def __init__(self, pieces: List[Tuple[int, int]], turn: str):
        # Sort pieces for canonical form
        pieces = sorted(pieces)

        # Encode pieces
        data = [0] * 16
        for i, (piece, square) in enumerate(pieces[:8]):
            data[i*2] = piece
            data[i*2 + 1] = square

        # Turn in high byte
        turn_bit = 1 if turn == 'w' else 0
        count = len(pieces)
        meta = (turn_bit << 8) | count

        self._data = struct.pack(self.STRUCT_FMT, *data, meta)
        self._hash = None
        self._pieces_cache = None
        self._board_cache = None

    @classmethod
    def from_bytes(cls, data: bytes) -> 'CompactState':
        """Reconstruct from bytes"""
        obj = cls.__new__(cls)
        obj._data = data
        obj._hash = None
        obj._pieces_cache = None
        obj._board_cache = None
        return obj

    def to_bytes(self) -> bytes:
        return self._data

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._data)
        return self._hash

    def __eq__(self, other):
        return self._data == other._data

    @property
    def turn(self) -> str:
        meta = struct.unpack('<H', self._data[16:18])[0]
        return 'w' if (meta >> 8) & 1 else 'b'

    @property
    def pieces(self) -> List[Tuple[int, int]]:
        if self._pieces_cache is not None:
            return self._pieces_cache
        data = struct.unpack('<16B', self._data[:16])
        meta = struct.unpack('<H', self._data[16:18])[0]
        count = meta & 0xFF
        result = []
        for i in range(min(count, 8)):
            piece = data[i*2]
            square = data[i*2 + 1]
            if piece > 0:
                result.append((piece, square))
        self._pieces_cache = result
        return result

    def piece_count(self) -> int:
        meta = struct.unpack('<H', self._data[16:18])[0]
        return meta & 0xFF

    def to_board(self) -> List[int]:
        if self._board_cache is not None:
            return self._board_cache
        board = [Piece.EMPTY] * 64
        for p, sq in self.pieces:
            board[sq] = p
        self._board_cache = board
        return board

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
# CHESS LOGIC (Optimized)
# ============================================================

def generate_moves(state: CompactState) -> List[Tuple[int, int, Optional[int]]]:
    """Generate legal moves. Returns list of (from_sq, to_sq, captured_piece)"""
    board = state.to_board()
    is_white_turn = state.turn == 'w'
    moves = []

    for piece, from_sq in state.pieces:
        if is_white_turn and not is_white(piece): continue
        if not is_white_turn and not is_black(piece): continue

        pt = piece_type(piece)
        if pt == 1:  # King
            for d in [-9, -8, -7, -1, 1, 7, 8, 9]:
                to_sq = from_sq + d
                if not (0 <= to_sq < 64): continue
                if abs((from_sq % 8) - (to_sq % 8)) > 1: continue
                target = board[to_sq]
                if (is_white_turn and is_white(target)) or (not is_white_turn and is_black(target)): continue
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
                if not (0 <= to_sq < 64): continue
                if abs((from_sq % 8) - (to_sq % 8)) > 2: continue
                target = board[to_sq]
                if (is_white_turn and is_white(target)) or (not is_white_turn and is_black(target)): continue
                moves.append((from_sq, to_sq, target if target else None))

    # Filter illegal moves (leaving king in check)
    return [m for m in moves if not _leaves_in_check(state, m)]


def _sliding(board, from_sq, is_white_turn, directions):
    moves = []
    for d in directions:
        sq = from_sq + d
        prev_file = from_sq % 8
        while 0 <= sq < 64:
            sq_file = sq % 8
            if abs(sq_file - prev_file) > 1: break
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


def apply_move(state: CompactState, move: Tuple[int, int, Optional[int]]) -> CompactState:
    """Apply move and return new state"""
    from_sq, to_sq, _ = move
    new_pieces = []
    moved = None

    for p, sq in state.pieces:
        if sq == from_sq:
            moved = p
        elif sq == to_sq:
            pass  # Captured
        else:
            new_pieces.append((p, sq))

    if moved:
        new_pieces.append((moved, to_sq))

    new_turn = 'b' if state.turn == 'w' else 'w'
    return CompactState(new_pieces, new_turn)


def _leaves_in_check(state: CompactState, move: Tuple[int, int, Optional[int]]) -> bool:
    """Check if move leaves own king in check"""
    new_state = apply_move(state, move)
    return in_check(new_state, state.turn)


def in_check(state: CompactState, color: str) -> bool:
    """Check if the given color's king is in check"""
    board = state.to_board()
    king = Piece.W_KING if color == 'w' else Piece.B_KING
    king_sq = None

    for p, sq in state.pieces:
        if p == king:
            king_sq = sq
            break

    if king_sq is None:
        return True  # King captured = in check

    # Check if any enemy piece attacks the king
    enemy_is_white = (color == 'b')
    for p, sq in state.pieces:
        if enemy_is_white and not is_white(p): continue
        if not enemy_is_white and not is_black(p): continue
        if attacks(board, sq, king_sq, p):
            return True
    return False


def attacks(board, from_sq, to_sq, piece) -> bool:
    """Check if piece at from_sq attacks to_sq"""
    pt = piece_type(piece)
    dr = (to_sq // 8) - (from_sq // 8)
    df = (to_sq % 8) - (from_sq % 8)

    if pt == 1:  # King
        return abs(dr) <= 1 and abs(df) <= 1 and (dr != 0 or df != 0)
    if pt == 5:  # Knight
        return (abs(dr), abs(df)) in [(1,2), (2,1)]
    if pt == 3:  # Rook
        return (dr == 0 or df == 0) and dr != df and _path_clear(board, from_sq, to_sq)
    if pt == 4:  # Bishop
        return abs(dr) == abs(df) and dr != 0 and _path_clear(board, from_sq, to_sq)
    if pt == 2:  # Queen
        return ((dr == 0 or df == 0) or abs(dr) == abs(df)) and (dr != 0 or df != 0) and _path_clear(board, from_sq, to_sq)
    return False


def _path_clear(board, from_sq, to_sq) -> bool:
    """Check if path between squares is clear"""
    dr = 0 if (to_sq // 8) == (from_sq // 8) else (1 if to_sq // 8 > from_sq // 8 else -1)
    df = 0 if (to_sq % 8) == (from_sq % 8) else (1 if to_sq % 8 > from_sq % 8 else -1)
    d = dr * 8 + df
    if d == 0: return True

    sq = from_sq + d
    while sq != to_sq and 0 <= sq < 64:
        if abs((sq % 8) - ((sq - d) % 8)) > 1: break
        if board[sq] != Piece.EMPTY: return False
        sq += d
    return True


def is_terminal(state: CompactState, moves: Optional[List] = None) -> Tuple[bool, Optional[int]]:
    """Check if position is terminal. Returns (is_terminal, value)
    Pass moves if already computed to avoid regenerating."""
    if moves is None:
        moves = generate_moves(state)
    if not moves:
        if in_check(state, state.turn):
            # Checkmate
            return True, (-1 if state.turn == 'w' else 1)
        else:
            # Stalemate
            return True, 0

    # Insufficient material (just kings)
    if state.piece_count() == 2:
        return True, 0

    return False, None


# ============================================================
# DISK STORAGE (LMDB or SQLite fallback)
# ============================================================

class DiskStore:
    """
    Disk-backed storage for cold data.
    Uses LMDB for speed if available, SQLite as fallback.
    """

    def __init__(self, path: str, max_size_gb: float = 200.0):
        self.path = path
        self.max_size = int(max_size_gb * 1024**3)
        os.makedirs(path, exist_ok=True)

        if HAS_LMDB:
            self._init_lmdb()
        else:
            self._init_sqlite()

        self._write_buffer = []
        self._buffer_size = 50000

    def _init_lmdb(self):
        self.backend = 'lmdb'
        self.env = lmdb.open(
            self.path,
            map_size=self.max_size,
            max_dbs=5,
            writemap=True,
            metasync=False,
            sync=False
        )
        # Databases
        self.db_frontier = self.env.open_db(b'frontier')
        self.db_solved = self.env.open_db(b'solved')
        self.db_parents = self.env.open_db(b'parents', dupsort=True)
        self.db_states = self.env.open_db(b'states')  # hash -> compact state bytes

    def _init_sqlite(self):
        self.backend = 'sqlite'
        db_path = os.path.join(self.path, 'store.db')
        self.conn = sqlite3.connect(db_path, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA cache_size=-100000")  # 100MB cache

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS frontier (
                hash INTEGER PRIMARY KEY,
                state BLOB
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS solved (
                hash INTEGER PRIMARY KEY,
                value INTEGER,
                turn INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS parents (
                child INTEGER,
                parent INTEGER,
                PRIMARY KEY (child, parent)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_parents_child ON parents(child)")

    # ---- Frontier Operations ----

    def add_to_frontier(self, h: int, state_bytes: bytes):
        """Add position to cold frontier"""
        self._write_buffer.append(('frontier', h, state_bytes))
        if len(self._write_buffer) >= self._buffer_size:
            self.flush()

    def get_frontier_batch(self, batch_size: int) -> List[Tuple[int, bytes]]:
        """Get and remove a batch from cold frontier"""
        if self.backend == 'lmdb':
            self.flush()
            results = []
            with self.env.begin(write=True, db=self.db_frontier) as txn:
                cursor = txn.cursor()
                count = 0
                to_delete = []
                for key, value in cursor:
                    h = struct.unpack('<q', key)[0]
                    results.append((h, value))
                    to_delete.append(key)
                    count += 1
                    if count >= batch_size:
                        break
                for key in to_delete:
                    txn.delete(key)
            return results
        else:
            cursor = self.conn.execute(
                "SELECT hash, state FROM frontier LIMIT ?", (batch_size,)
            )
            results = cursor.fetchall()
            if results:
                hashes = [r[0] for r in results]
                placeholders = ','.join('?' * len(hashes))
                self.conn.execute(f"DELETE FROM frontier WHERE hash IN ({placeholders})", hashes)
            return results

    def frontier_size(self) -> int:
        """Get number of positions in cold frontier"""
        if self.backend == 'lmdb':
            with self.env.begin(db=self.db_frontier) as txn:
                return txn.stat()['entries']
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM frontier")
            return cursor.fetchone()[0]

    # ---- Solved Operations ----

    def add_solved(self, h: int, value: int, turn_is_white: bool):
        """Add solved position"""
        self._write_buffer.append(('solved', h, (value, turn_is_white)))
        if len(self._write_buffer) >= self._buffer_size:
            self.flush()

    def get_solved(self, h: int) -> Optional[Tuple[int, bool]]:
        """Get solved value and turn"""
        if self.backend == 'lmdb':
            self.flush()
            with self.env.begin(db=self.db_solved) as txn:
                key = struct.pack('<q', h)
                data = txn.get(key)
                if data:
                    value, turn = struct.unpack('<ib', data)
                    return value, bool(turn)
        else:
            cursor = self.conn.execute("SELECT value, turn FROM solved WHERE hash=?", (h,))
            row = cursor.fetchone()
            if row:
                return row[0], bool(row[1])
        return None

    def get_solved_batch(self, hashes: List[int]) -> Dict[int, Tuple[int, bool]]:
        """Get multiple solved values"""
        results = {}
        if self.backend == 'lmdb':
            self.flush()
            with self.env.begin(db=self.db_solved) as txn:
                for h in hashes:
                    key = struct.pack('<q', h)
                    data = txn.get(key)
                    if data:
                        value, turn = struct.unpack('<ib', data)
                        results[h] = (value, bool(turn))
        else:
            placeholders = ','.join('?' * len(hashes))
            cursor = self.conn.execute(
                f"SELECT hash, value, turn FROM solved WHERE hash IN ({placeholders})",
                hashes
            )
            for row in cursor:
                results[row[0]] = (row[1], bool(row[2]))
        return results

    def solved_count(self) -> int:
        """Get number of solved positions"""
        if self.backend == 'lmdb':
            with self.env.begin(db=self.db_solved) as txn:
                return txn.stat()['entries']
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM solved")
            return cursor.fetchone()[0]

    # ---- Parent Edge Operations ----

    def add_parent(self, child_h: int, parent_h: int):
        """Add parent edge for retrograde analysis"""
        self._write_buffer.append(('parent', child_h, parent_h))
        if len(self._write_buffer) >= self._buffer_size:
            self.flush()

    def get_parents(self, child_h: int) -> List[int]:
        """Get all parents of a position"""
        if self.backend == 'lmdb':
            self.flush()
            results = []
            with self.env.begin(db=self.db_parents) as txn:
                cursor = txn.cursor()
                key = struct.pack('<q', child_h)
                if cursor.set_key(key):
                    for _, value in cursor.iternext_dup():
                        results.append(struct.unpack('<q', value)[0])
            return results
        else:
            cursor = self.conn.execute(
                "SELECT parent FROM parents WHERE child=?", (child_h,)
            )
            return [row[0] for row in cursor]

    def get_parents_batch(self, child_hashes: List[int]) -> Dict[int, List[int]]:
        """Get parents for multiple children"""
        results = {h: [] for h in child_hashes}
        if self.backend == 'lmdb':
            self.flush()
            with self.env.begin(db=self.db_parents) as txn:
                for child_h in child_hashes:
                    cursor = txn.cursor()
                    key = struct.pack('<q', child_h)
                    if cursor.set_key(key):
                        for _, value in cursor.iternext_dup():
                            results[child_h].append(struct.unpack('<q', value)[0])
        else:
            placeholders = ','.join('?' * len(child_hashes))
            cursor = self.conn.execute(
                f"SELECT child, parent FROM parents WHERE child IN ({placeholders})",
                child_hashes
            )
            for row in cursor:
                results[row[0]].append(row[1])
        return results

    # ---- Flush and Close ----

    def flush(self):
        """Flush write buffer to disk"""
        if not self._write_buffer:
            return

        if self.backend == 'lmdb':
            with self.env.begin(write=True) as txn:
                for op, key, value in self._write_buffer:
                    key_bytes = struct.pack('<q', key)
                    if op == 'frontier':
                        txn.put(key_bytes, value, db=self.db_frontier)
                    elif op == 'solved':
                        val, turn = value
                        txn.put(key_bytes, struct.pack('<ib', val, turn), db=self.db_solved)
                    elif op == 'parent':
                        # value is parent_h
                        txn.put(key_bytes, struct.pack('<q', value), db=self.db_parents, dupdata=True)
            self._write_buffer.clear()
        else:
            # Group by operation type for executemany (much faster)
            frontier_batch = []
            solved_batch = []
            parent_batch = []

            for op, key, value in self._write_buffer:
                if op == 'frontier':
                    frontier_batch.append((key, value))
                elif op == 'solved':
                    val, turn = value
                    solved_batch.append((key, val, 1 if turn else 0))
                elif op == 'parent':
                    parent_batch.append((key, value))

            self.conn.execute("BEGIN")
            if frontier_batch:
                self.conn.executemany("INSERT OR IGNORE INTO frontier VALUES (?, ?)", frontier_batch)
            if solved_batch:
                self.conn.executemany("INSERT OR REPLACE INTO solved VALUES (?, ?, ?)", solved_batch)
            if parent_batch:
                self.conn.executemany("INSERT OR IGNORE INTO parents VALUES (?, ?)", parent_batch)
            self.conn.execute("COMMIT")
            self._write_buffer.clear()

    def close(self):
        """Close storage"""
        self.flush()
        if self.backend == 'lmdb':
            self.env.close()
        else:
            self.conn.close()


# ============================================================
# SYZYGY INTERFACE
# ============================================================

class SyzygyProbe:
    """Interface to Syzygy tablebases for 7-piece positions"""

    def __init__(self, path: str = "./syzygy"):
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
                print(f"[Syzygy] Loaded from {path}")
        except Exception as e:
            print(f"[Syzygy] Not available: {e}")

    def probe(self, state: CompactState) -> Optional[int]:
        """Probe tablebase. Returns 1 (white wins), -1 (black wins), 0 (draw), or None"""
        if not self.available or state.piece_count() > 7:
            return None

        self.probes += 1
        try:
            board = self.chess.Board()
            board.clear()

            type_map = {
                1: self.chess.KING, 2: self.chess.QUEEN, 3: self.chess.ROOK,
                4: self.chess.BISHOP, 5: self.chess.KNIGHT, 6: self.chess.PAWN
            }

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
# FRACTAL REGION
# ============================================================

@dataclass
class FractalRegion:
    """A solved region that can be used as boundary"""
    name: str
    piece_count: int
    solved_count: int
    boundary_count: int

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'piece_count': self.piece_count,
                'solved_count': self.solved_count,
                'boundary_count': self.boundary_count,
            }, f)

    @staticmethod
    def load(path: str) -> 'FractalRegion':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return FractalRegion(**data)


# ============================================================
# MAIN SOLVER
# ============================================================

class FractalHOLOS2:
    """
    Memory-safe fractal holographic solver.

    Architecture:
    - HOT (RAM): frontier_hot, solved_hot, unsolved_counts
    - COLD (Disk): frontier_cold, solved_cold, parent edges
    """

    def __init__(self,
                 config: MemoryConfig = None,
                 syzygy_path: str = "./syzygy",
                 save_dir: str = "./fractal_holos2"):

        self.config = config or MemoryConfig()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Memory monitor
        self.monitor = MemoryMonitor(self.config)

        # Syzygy
        self.syzygy = SyzygyProbe(syzygy_path)

        # Disk storage
        self.disk = DiskStore(os.path.join(save_dir, "cold"), self.config.cold_disk_max_gb)

        # HOT tier (RAM)
        self.frontier_hot: Dict[int, bytes] = {}  # hash -> compact state bytes
        self.solved_hot: Dict[int, Tuple[int, bool]] = {}  # hash -> (value, turn_is_white)
        self.unsolved_counts: Dict[int, Tuple[int, bool]] = {}  # hash -> (count, turn_is_white)
        self.seen: Set[int] = set()  # All seen hashes (for dedup)

        # Loaded regions
        self.regions: Dict[str, FractalRegion] = {}

        # Metrics
        self.stats = {
            'expanded': 0,
            'solved': 0,
            'spills': 0,
            'boundary_hits': 0,
        }

        # Checkpointing
        self._last_checkpoint = time.time()
        self._current_region = None
        self._current_depth = 0

    def load_regions(self):
        """Load previously solved regions"""
        for filename in os.listdir(self.save_dir):
            if filename.endswith('_region.pkl'):
                try:
                    path = os.path.join(self.save_dir, filename)
                    region = FractalRegion.load(path)
                    self.regions[region.name] = region
                    print(f"  [Loaded region '{region.name}': {region.solved_count:,} positions]")
                except Exception as e:
                    print(f"  [Failed to load {filename}: {e}]")

    def _spill_to_cold(self, fraction: float = 0.5):
        """Spill fraction of hot frontier to cold disk"""
        if not self.frontier_hot:
            return

        spill_count = int(len(self.frontier_hot) * fraction)
        if spill_count == 0:
            return

        print(f"  [Spilling {spill_count:,} positions to disk...]")

        # Take oldest items (FIFO-ish by using popitem)
        for _ in range(spill_count):
            if not self.frontier_hot:
                break
            h, state_bytes = self.frontier_hot.popitem()
            self.disk.add_to_frontier(h, state_bytes)

        self.disk.flush()
        self.stats['spills'] += 1
        gc.collect()

        print(f"  [Spill complete. Hot: {len(self.frontier_hot):,}, Cold: {self.disk.frontier_size():,}]")

    def _reload_from_cold(self, batch_size: int = None):
        """Reload positions from cold storage to hot"""
        if batch_size is None:
            batch_size = self.config.batch_size

        # Check if we have room
        cost = self.monitor.estimate_batch_cost(batch_size)
        if not self.monitor.can_allocate(cost):
            batch_size = batch_size // 2
            if batch_size < 1000:
                return 0

        batch = self.disk.get_frontier_batch(batch_size)
        for h, state_bytes in batch:
            self.frontier_hot[h] = state_bytes

        return len(batch)

    def _check_memory_and_spill(self):
        """Check memory and spill if needed"""
        if self.monitor.should_spill():
            mem_gb = self.monitor.get_used_gb()
            print(f"\n  [Memory warning: {mem_gb:.1f} GB used, spilling...]")
            self._spill_to_cold(0.5)
            gc.collect()

    def _checkpoint(self):
        """Save checkpoint"""
        checkpoint = {
            'region': self._current_region,
            'depth': self._current_depth,
            'stats': self.stats,
            'frontier_hot_size': len(self.frontier_hot),
            'seen_size': len(self.seen),
            'timestamp': time.time(),
        }

        path = os.path.join(self.save_dir, 'checkpoint.pkl')
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

        self.disk.flush()
        print(f"  [Checkpoint saved: depth={self._current_depth}, solved={self.stats['solved']:,}]")

    def solve_region(self, name: str, starting_states: List[CompactState], max_depth: int = 100):
        """
        Solve a region using memory-safe fractal approach.
        """
        print(f"\n{'='*60}")
        print(f"SOLVING REGION: {name}")
        print(f"Memory config: {self.config.hot_ram_min_gb}-{self.config.hot_ram_max_gb} GB RAM")
        print(f"{'='*60}")

        self._current_region = name
        self._current_depth = 0

        # Clear hot state
        self.frontier_hot.clear()
        self.solved_hot.clear()
        self.unsolved_counts.clear()
        self.seen.clear()
        gc.collect()

        # Initialize frontier
        for state in starting_states:
            h = hash(state)
            if h not in self.seen:
                self.seen.add(h)
                self.frontier_hot[h] = state.to_bytes()

        print(f"Starting with {len(self.frontier_hot):,} positions")

        start_time = time.time()
        piece_count = starting_states[0].piece_count() if starting_states else 0

        for depth in range(max_depth):
            self._current_depth = depth

            # Memory check
            mem_gb = self.monitor.get_used_gb()
            avail_gb = self.monitor.get_available_gb()
            hot_count = len(self.frontier_hot)
            cold_count = self.disk.frontier_size()

            print(f"\n--- Depth {depth} ---")
            print(f"  Frontier: {hot_count:,} hot + {cold_count:,} cold = {hot_count + cold_count:,} total")
            print(f"  Memory: {mem_gb:.1f} GB used, {avail_gb:.1f} GB available")
            print(f"  Solved: {self.stats['solved']:,}")

            # Check if done
            if hot_count == 0 and cold_count == 0:
                print("  Frontier exhausted!")
                break

            # Memory management
            self._check_memory_and_spill()

            # Reload from cold if hot is empty
            if hot_count == 0 and cold_count > 0:
                reloaded = self._reload_from_cold()
                print(f"  [Reloaded {reloaded:,} from cold]")
                if reloaded == 0:
                    break

            # Expand one batch
            batch_size = min(len(self.frontier_hot), self.config.batch_size)
            original_batch_size = batch_size

            # Proactive memory check - reduce batch if memory constrained
            cost = self.monitor.estimate_batch_cost(batch_size)
            min_batch = min(100, batch_size)  # Allow smaller batches for small datasets

            while not self.monitor.can_allocate(cost) and batch_size > min_batch:
                batch_size = batch_size // 2
                cost = self.monitor.estimate_batch_cost(batch_size)

            # Only spill if we truly can't fit even a small batch
            if batch_size < min_batch and not self.monitor.can_allocate(self.monitor.estimate_batch_cost(min_batch)):
                print(f"  [Memory too constrained for expansion (need {cost/1e6:.1f}MB), spilling...]")
                self._spill_to_cold(0.7)
                gc.collect()
                continue

            # Use whatever batch size we have
            batch_size = max(batch_size, min_batch)

            print(f"  Expanding batch of {batch_size:,}...", flush=True)
            expand_start = time.time()

            # Expand
            newly_solved = self._expand_batch(batch_size)

            expand_time = time.time() - expand_start
            print(f"  Expand took {expand_time:.1f}s", flush=True)

            # Propagate
            propagated = self._propagate(newly_solved)

            elapsed = time.time() - start_time
            rate = self.stats['expanded'] / elapsed if elapsed > 0 else 0
            print(f"  Expanded: {batch_size:,}, New solved: {len(newly_solved):,}, Propagated: {propagated:,}")
            print(f"  Time: {elapsed:.0f}s, Rate: {rate:.0f} pos/s")

            # Checkpoint
            if time.time() - self._last_checkpoint > self.config.checkpoint_interval:
                self._checkpoint()
                self._last_checkpoint = time.time()

        # Finalize region
        self.disk.flush()

        total_solved = self.stats['solved'] + self.disk.solved_count()
        region = FractalRegion(
            name=name,
            piece_count=piece_count,
            solved_count=total_solved,
            boundary_count=self.stats['boundary_hits']
        )

        region_path = os.path.join(self.save_dir, f"{name}_region.pkl")
        region.save(region_path)
        self.regions[name] = region

        print(f"\n{'='*60}")
        print(f"Region '{name}' complete: {total_solved:,} positions solved")
        print(f"{'='*60}")

        return region

    def _expand_batch(self, batch_size: int) -> List[int]:
        """Expand a batch of positions from hot frontier"""
        newly_solved = []
        next_frontier = {}

        # Take batch from hot frontier
        batch = []
        for _ in range(min(batch_size, len(self.frontier_hot))):
            if not self.frontier_hot:
                break
            h, state_bytes = self.frontier_hot.popitem()
            batch.append((h, state_bytes))

        last_report = time.time()

        for i, (h, state_bytes) in enumerate(batch):
            # Progress report every 30 seconds or every 50000 positions
            now = time.time()
            if i > 0 and (i % 50000 == 0 or now - last_report > 30.0):
                print(f"    Expanding: {i:,}/{len(batch):,} ({i*100//len(batch)}%)", flush=True)
                last_report = now
                self._check_memory_and_spill()

            # Reconstruct state
            state = CompactState.from_bytes(state_bytes)

            # Generate moves ONCE (used for both terminal check and expansion)
            moves = generate_moves(state)

            # Check terminal using pre-computed moves
            is_term, value = is_terminal(state, moves)
            if is_term:
                self.solved_hot[h] = (value, state.turn == 'w')
                self.disk.add_solved(h, value, state.turn == 'w')
                newly_solved.append(h)
                self.stats['solved'] += 1
                continue

            child_hashes = []

            for move in moves:
                child = apply_move(state, move)
                ch = hash(child)
                child_hashes.append(ch)

                # Skip if already seen (most common case - huge speedup)
                if ch in self.seen:
                    continue

                # Check boundaries only for NEW positions
                value = None

                # Syzygy - only probe on captures that reduce to <=6 pieces
                if move[2] is not None and child.piece_count() <= 6:
                    value = self.syzygy.probe(child)
                    if value is not None:
                        self.stats['boundary_hits'] += 1

                # Previously solved in this session
                if value is None and ch in self.solved_hot:
                    value, _ = self.solved_hot[ch]

                if value is not None:
                    self.seen.add(ch)
                    if ch not in self.solved_hot:
                        self.solved_hot[ch] = (value, child.turn == 'w')
                        self.disk.add_solved(ch, value, child.turn == 'w')
                        newly_solved.append(ch)
                        self.stats['solved'] += 1
                    continue

                # Add to frontier (new unsolved position)
                self.seen.add(ch)
                child_bytes = child.to_bytes()

                # Add to hot or cold based on memory
                if self.monitor.can_allocate(len(child_bytes) + 100):
                    next_frontier[ch] = child_bytes
                else:
                    self.disk.add_to_frontier(ch, child_bytes)

            # Record unsolved children count
            solved_children = sum(1 for ch in child_hashes if ch in self.solved_hot)
            unsolved_count = len(child_hashes) - solved_children

            if unsolved_count > 0:
                self.unsolved_counts[h] = (unsolved_count, state.turn == 'w')
            else:
                # All children solved, can determine value now
                child_values = [self.solved_hot[ch][0] for ch in child_hashes if ch in self.solved_hot]
                if child_values:
                    # Minimax
                    if state.turn == 'w':
                        value = max(child_values)  # White maximizes
                    else:
                        value = min(child_values)  # Black minimizes

                    self.solved_hot[h] = (value, state.turn == 'w')
                    self.disk.add_solved(h, value, state.turn == 'w')
                    newly_solved.append(h)
                    self.stats['solved'] += 1

            self.stats['expanded'] += 1

        # Flush disk buffer periodically
        if len(self.disk._write_buffer) > 10000:
            self.disk.flush()

        # Merge next frontier into hot
        self.frontier_hot.update(next_frontier)

        return newly_solved

    def _propagate(self, seeds: List[int], max_iters: int = 100) -> int:
        """Propagate solved values backward"""
        if not seeds:
            return 0

        queue = deque(seeds)
        processed = 0
        seen_in_prop = set(seeds)

        for _ in range(max_iters):
            if not queue:
                break

            batch = []
            while queue and len(batch) < 10000:
                batch.append(queue.popleft())

            if not batch:
                break

            # Get parents for batch
            parent_map = self.disk.get_parents_batch(batch)

            for child_h in batch:
                if child_h not in self.solved_hot:
                    continue

                child_val, _ = self.solved_hot[child_h]

                for parent_h in parent_map.get(child_h, []):
                    if parent_h in self.solved_hot:
                        continue

                    if parent_h not in self.unsolved_counts:
                        continue

                    count, parent_is_white = self.unsolved_counts[parent_h]

                    # Check for immediate win
                    resolved = False
                    if parent_is_white and child_val == 1:
                        # White found winning move
                        self.solved_hot[parent_h] = (1, True)
                        self.disk.add_solved(parent_h, 1, True)
                        resolved = True
                    elif not parent_is_white and child_val == -1:
                        # Black found winning move
                        self.solved_hot[parent_h] = (-1, False)
                        self.disk.add_solved(parent_h, -1, False)
                        resolved = True
                    else:
                        # Decrement count
                        count -= 1
                        if count <= 0:
                            # All moves explored, none winning
                            value = -1 if parent_is_white else 1
                            self.solved_hot[parent_h] = (value, parent_is_white)
                            self.disk.add_solved(parent_h, value, parent_is_white)
                            resolved = True
                        else:
                            self.unsolved_counts[parent_h] = (count, parent_is_white)

                    if resolved:
                        del self.unsolved_counts[parent_h]
                        self.stats['solved'] += 1
                        processed += 1

                        if parent_h not in seen_in_prop:
                            seen_in_prop.add(parent_h)
                            queue.append(parent_h)

        return processed

    def close(self):
        """Clean up resources"""
        self.disk.close()


# ============================================================
# UTILITIES
# ============================================================

def random_position(material: str, max_attempts: int = 1000) -> Optional[CompactState]:
    """Generate random legal position with given material"""
    white_str, black_str = material.upper().split('V')
    white = [PIECE_NAMES[c] for c in white_str]
    black = [PIECE_NAMES[c] + 6 for c in black_str]
    all_pieces = white + black

    for _ in range(max_attempts):
        squares = random.sample(range(64), len(all_pieces))
        pieces = list(zip(all_pieces, squares))
        state = CompactState(pieces, 'w')

        # Validate
        wk_sq = next((sq for p, sq in pieces if p == Piece.W_KING), None)
        bk_sq = next((sq for p, sq in pieces if p == Piece.B_KING), None)

        if wk_sq is None or bk_sq is None:
            continue

        # Kings not adjacent
        if abs(wk_sq // 8 - bk_sq // 8) <= 1 and abs(wk_sq % 8 - bk_sq % 8) <= 1:
            continue

        # Not leaving opponent in check
        if in_check(state, 'b'):
            continue

        return state

    return None


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fractal HOLOS v2 - Memory-Safe Solver")
    parser.add_argument('--material', default='KQRvKQR', help='Material balance (e.g., KQRvKQR)')
    parser.add_argument('--positions', type=int, default=1000, help='Starting positions')
    parser.add_argument('--ram-min', type=float, default=10.0, help='Minimum RAM in GB')
    parser.add_argument('--ram-max', type=float, default=60.0, help='Maximum RAM in GB')
    parser.add_argument('--ram-target', type=float, default=40.0, help='Target RAM in GB')
    parser.add_argument('--disk-max', type=float, default=200.0, help='Maximum disk in GB')
    parser.add_argument('--regions', type=int, default=5, help='Number of regions to solve')
    parser.add_argument('--batch-size', type=int, default=100000, help='Batch size')
    args = parser.parse_args()

    print("="*60)
    print("FRACTAL HOLOS v2 - Memory-Safe Solver")
    print("="*60)
    print(f"Material: {args.material}")
    print(f"RAM: {args.ram_min}-{args.ram_max} GB (target: {args.ram_target} GB)")
    print(f"Disk: up to {args.disk_max} GB")
    print(f"Backend: {'LMDB' if HAS_LMDB else 'SQLite'}")
    print()

    config = MemoryConfig(
        hot_ram_min_gb=args.ram_min,
        hot_ram_max_gb=args.ram_max,
        hot_ram_target_gb=args.ram_target,
        cold_disk_max_gb=args.disk_max,
        batch_size=args.batch_size,
    )

    solver = FractalHOLOS2(config=config)
    solver.load_regions()

    try:
        for i in range(args.regions):
            region_name = f"{args.material}_region_{i}"

            if region_name in solver.regions:
                print(f"\nRegion '{region_name}' already solved, skipping...")
                continue

            print(f"\nGenerating {args.positions} starting positions...")
            starts = []
            for _ in range(args.positions):
                state = random_position(args.material)
                if state:
                    starts.append(state)

            if not starts:
                print("Failed to generate starting positions!")
                continue

            print(f"Generated {len(starts)} valid positions")

            if len(starts) >= 3:
                print("\nSample positions:")
                for s in starts[:3]:
                    s.display()
                    print()

            solver.solve_region(region_name, starts)

        # Summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Regions solved: {len(solver.regions)}")
        print(f"Total expanded: {solver.stats['expanded']:,}")
        print(f"Total solved: {solver.stats['solved']:,}")
        print(f"Boundary hits: {solver.stats['boundary_hits']:,}")
        print(f"Memory spills: {solver.stats['spills']}")

    finally:
        solver.close()


if __name__ == "__main__":
    main()
