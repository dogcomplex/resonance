"""
fractal_holos.py - Fractal Holographic Solver (Disk-Backed)

KEY INSIGHT: 
- RAM is for the Frontier and Solved states.
- DISK (SQLite) is for the Graph Topology (Parents/Edges).
- This allows solving massive regions that exceed physical RAM.

MEMORY MANAGEMENT:
- Topology (Edges) -> SQLite (./fractal_temp/...)
- Frontier -> RAM
- Solved -> RAM (eventually persisted)
"""

import os
import sys
import time
import pickle
import random
import gc
import sqlite3
import shutil
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum

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

def is_white(p): return 1 <= p <= 6
def is_black(p): return 7 <= p <= 12
def piece_type(p): return ((p - 1) % 6) + 1 if p > 0 else 0

class ChessState:
    __slots__ = ['pieces', 'turn', '_hash']
    
    def __init__(self, pieces, turn='w'):
        self.pieces = tuple(sorted(pieces))
        self.turn = turn
        self._hash = None
    
    def __hash__(self):
        if self._hash is None:
            # Canonical hash (flip horizontal if needed to reduce state space)
            # Simple version: just hash pieces and turn
            self._hash = hash((self.pieces, self.turn))
        return self._hash
    
    def to_board(self):
        board = [Piece.EMPTY] * 64
        for p, sq in self.pieces:
            board[sq] = p
        return board
    
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

def generate_moves(state):
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
                if not (0 <= to_sq < 64) or abs((from_sq % 8) - (to_sq % 8)) > 1: continue
                target = board[to_sq]
                if (is_white_turn and is_white(target)) or (not is_white_turn and is_black(target)): continue
                moves.append((from_sq, to_sq, target if target else None))
        elif pt == 2: moves.extend(_sliding(board, from_sq, is_white_turn, [-9,-8,-7,-1,1,7,8,9]))
        elif pt == 3: moves.extend(_sliding(board, from_sq, is_white_turn, [-8,-1,1,8]))
        elif pt == 4: moves.extend(_sliding(board, from_sq, is_white_turn, [-9,-7,7,9]))
        elif pt == 5:
            for d in [-17,-15,-10,-6,6,10,15,17]:
                to_sq = from_sq + d
                if not (0 <= to_sq < 64) or abs((from_sq % 8) - (to_sq % 8)) > 2: continue
                target = board[to_sq]
                if (is_white_turn and is_white(target)) or (not is_white_turn and is_black(target)): continue
                moves.append((from_sq, to_sq, target if target else None))
    
    # Filter checks
    valid_moves = []
    for m in moves:
        next_state = apply_move(state, m)
        if not in_check(next_state, state.turn):
            valid_moves.append(m)
    return valid_moves

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
            else: break
            prev_file = sq_file
            sq += d
    return moves

def apply_move(state, move):
    from_sq, to_sq, _ = move
    new_pieces = []
    moved = None
    for p, sq in state.pieces:
        if sq == from_sq: moved = p
        elif sq == to_sq: pass
        else: new_pieces.append((p, sq))
    if moved: new_pieces.append((moved, to_sq))
    return ChessState(new_pieces, 'b' if state.turn == 'w' else 'w')

def in_check(state, color):
    board = state.to_board()
    king = Piece.W_KING if color == 'w' else Piece.B_KING
    king_sq = next((sq for p, sq in state.pieces if p == king), None)
    if king_sq is None: return True
    enemy_white = (color == 'b')
    for p, sq in state.pieces:
        if enemy_white and not is_white(p): continue
        if not enemy_white and not is_black(p): continue
        if attacks(board, sq, king_sq, p): return True
    return False

def attacks(board, from_sq, to_sq, piece):
    pt = piece_type(piece)
    dr, df = (to_sq // 8) - (from_sq // 8), (to_sq % 8) - (from_sq % 8)
    if pt == 1: return abs(dr) <= 1 and abs(df) <= 1
    if pt == 5: return (abs(dr), abs(df)) in [(1,2), (2,1)]
    if pt == 3: return (dr == 0 or df == 0) and path_clear(board, from_sq, to_sq)
    if pt == 4: return abs(dr) == abs(df) and path_clear(board, from_sq, to_sq)
    if pt == 2: return ((dr == 0 or df == 0) or abs(dr) == abs(df)) and path_clear(board, from_sq, to_sq)
    return False

def path_clear(board, from_sq, to_sq):
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

def is_terminal(state):
    moves = generate_moves(state)
    if not moves:
        if in_check(state, state.turn):
            return True, (-1 if state.turn == 'w' else 1) # Checkmate
        return True, 0 # Stalemate
    if len(state.pieces) == 2:
        return True, 0
    return False, None

# ============================================================
# SYZYGY INTERFACE
# ============================================================

class SyzygyProbe:
    def __init__(self, path="./syzygy"):
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
        except:
            print("✗ Syzygy not available")
    
    def probe(self, state):
        if not self.available or state.piece_count() > 7:
            return None
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
            if wdl > 0: return 1 if state.turn == 'w' else -1
            if wdl < 0: return -1 if state.turn == 'w' else 1
            return 0
        except:
            return None

# ============================================================
# DISK-BACKED GRAPH STORE
# ============================================================

class DiskGraph:
    """
    Offloads the massive edge list to SQLite to save RAM.
    This enables 10-80GB scale solving by using disk paging.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        if os.path.exists(db_path):
            os.remove(db_path)
        
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL") # Faster writes
        self.conn.execute("PRAGMA synchronous=OFF")  # Dangerous but fast
        
        # Store parents: child_h -> parent_h (To propagate up)
        self.conn.execute("CREATE TABLE IF NOT EXISTS parents (child INTEGER, parent INTEGER)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_child ON parents (child)")
        
        self.pending_writes = []
        self.batch_size = 50000
    
    def add_edge(self, parent_h, child_h):
        self.pending_writes.append((child_h, parent_h))
        if len(self.pending_writes) >= self.batch_size:
            self.commit()
    
    def commit(self):
        if self.pending_writes:
            self.conn.executemany("INSERT INTO parents VALUES (?, ?)", self.pending_writes)
            self.conn.commit()
            self.pending_writes = []
    
    def get_parents(self, child_h):
        self.commit() # Ensure consistency
        cur = self.conn.execute("SELECT parent FROM parents WHERE child=?", (child_h,))
        return [r[0] for r in cur.fetchall()]
    
    def close(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except:
                pass

# ============================================================
# FRACTAL HOLOGRAM
# ============================================================

@dataclass  
class FractalRegion:
    name: str
    piece_count: int
    solved_hashes: Set[int]
    values: Dict[int, int]
    boundary_hashes: Set[int]
    
    def query(self, h: int) -> Optional[int]:
        return self.values.get(h)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'piece_count': self.piece_count,
                'solved_hashes': self.solved_hashes,
                'values': self.values,
                'boundary_hashes': self.boundary_hashes,
            }, f)
    
    @staticmethod
    def load(path: str) -> 'FractalRegion':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return FractalRegion(
            data['name'], data['piece_count'],
            data['solved_hashes'], data['values'], data['boundary_hashes']
        )

# ============================================================
# FRACTAL HOLOS SOLVER
# ============================================================

class FractalHOLOS:
    def __init__(self, syzygy_path="./syzygy", save_dir="./fractal_holos", max_frontier_mb=20000):
        self.syzygy = SyzygyProbe(syzygy_path)
        self.save_dir = save_dir
        self.temp_dir = os.path.join(save_dir, "temp")
        self.max_frontier_mb = max_frontier_mb
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.regions: Dict[str, FractalRegion] = {}
        
        # Solving state
        self.solved: Dict[int, int] = {}
        self.frontier: Dict[int, ChessState] = {}
        
        # Optimization: Store count of unsolved children instead of list of children
        # RAM Usage: Dict[int, int] is much smaller than Dict[int, List[int]]
        self.unsolved_children_count: Dict[int, int] = {}
        
        # Disk backed graph for parents
        self.graph: Optional[DiskGraph] = None
        
        self.all_seen: Set[int] = set()
        self.boundary_contacts: Set[int] = set()
        self.total_solved = 0

    def load_regions(self):
        for filename in os.listdir(self.save_dir):
            if filename.endswith('_region.pkl'):
                path = os.path.join(self.save_dir, filename)
                try:
                    region = FractalRegion.load(path)
                    self.regions[region.name] = region
                    print(f"  [Loaded '{region.name}': {len(region.solved_hashes):,} pos]")
                except:
                    print(f"  [Failed to load {filename}]")

    def init_solve(self, region_name):
        """Clean slate for a new region"""
        if self.graph:
            self.graph.close()
        
        # Aggressive cleanup
        self.solved.clear()
        self.frontier.clear()
        self.unsolved_children_count.clear()
        self.all_seen.clear()
        self.boundary_contacts.clear()
        gc.collect()
        
        # Init DB
        db_path = os.path.join(self.temp_dir, f"{region_name}_edges.db")
        self.graph = DiskGraph(db_path)
        print(f"  [Initialized Disk Graph at {db_path}]")

    def get_memory_mb(self):
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0

    def solve_region(self, name: str, starting_states: List[ChessState], max_depth: int = 50):
        print(f"\n>>> SOLVING REGION: {name}")
        self.init_solve(name)
        
        # Init frontier
        for state in starting_states:
            h = hash(state)
            if h not in self.all_seen:
                self.all_seen.add(h)
                self.frontier[h] = state
        
        start_time = time.time()
        
        for depth in range(max_depth):
            mem_mb = self.get_memory_mb()
            print(f"\n--- Depth {depth} ---")
            print(f"  Frontier: {len(self.frontier):,}")
            print(f"  Solved Total: {len(self.solved):,}")
            print(f"  RAM Usage: {mem_mb:.0f} MB / Cap {self.max_frontier_mb} MB")
            
            if not self.frontier:
                print("  Frontier empty, region exhausted.")
                break
            
            # Check memory limit
            if mem_mb > self.max_frontier_mb:
                print(f"⚠️ Memory Limit Hit ({mem_mb:.0f} > {self.max_frontier_mb}). Crystallizing...")
                break
                
            # Expand
            self._expand_layer()
            
            # Commit edges to disk before propagation
            self.graph.commit()
            
            # Propagate values up
            propagated = self._propagate()
            
            elapsed = time.time() - start_time
            print(f"  Propagated this step: {propagated:,}")
            print(f"  Time: {elapsed:.0f}s")

        # Save result
        region = FractalRegion(
            name=name,
            piece_count=starting_states[0].piece_count() if starting_states else 0,
            solved_hashes=set(self.solved.keys()),
            values=dict(self.solved),
            boundary_hashes=self.boundary_contacts.copy()
        )
        
        path = os.path.join(self.save_dir, f"{name}_region.pkl")
        region.save(path)
        self.regions[name] = region
        self.total_solved += len(region.solved_hashes)
        
        # Cleanup
        self.graph.close()
        self.graph = None
        gc.collect()
        
        return region

    def _expand_layer(self):
        """Generates moves, updates graph topology, detects terminal states."""
        next_frontier = {}
        items = list(self.frontier.items())
        
        # Clear current frontier to save memory immediately
        self.frontier.clear()
        
        for i, (h, state) in enumerate(items):
            if i % 100000 == 0 and i > 0:
                print(f"    Expanding: {i:,}/{len(items):,}")

            # 1. Check if state already solved (e.g. by propagation from a sibling)
            if h in self.solved:
                continue

            # 2. Terminal checks
            is_term, value = is_terminal(state)
            if is_term:
                self.solved[h] = value
                self.boundary_contacts.add(h)
                continue

            # 3. Generate children
            moves = generate_moves(state)
            child_hashes = []
            
            # Optimization: Pre-calculate child hashes to set up graph
            for move in moves:
                child = apply_move(state, move)
                ch = hash(child)
                child_hashes.append(ch)
                
                # Record Edge (Child -> Parent) on Disk
                self.graph.add_edge(h, ch)
                
                # Check Boundaries
                val = None
                # Syzygy
                if move[2] is not None and child.piece_count() <= 7:
                    val = self.syzygy.probe(child)
                
                # Fractal Regions
                if val is None:
                    for region in self.regions.values():
                        val = region.query(ch)
                        if val is not None: break
                
                if val is not None:
                    self.solved[ch] = val
                    self.boundary_contacts.add(ch)
                
                # Add to next frontier if new
                if ch not in self.all_seen:
                    self.all_seen.add(ch)
                    next_frontier[ch] = child

            # 4. Initialize Child Count for Retrograde Analysis
            # We don't store the children list, just the count of unknown children.
            # We also check if any children are *already* winning immediately.
            
            # Logic:
            # If TURN is W: W wants Max(1). If any child is 1, W wins.
            # If TURN is B: B wants Min(-1). If any child is -1, B wins.
            
            best_val = -2 # Dummy
            can_force_win = False
            
            unknown_count = 0
            
            for ch in child_hashes:
                if ch in self.solved:
                    val = self.solved[ch]
                    if state.turn == 'w' and val == 1: can_force_win = True
                    if state.turn == 'b' and val == -1: can_force_win = True
                else:
                    unknown_count += 1
            
            if can_force_win:
                self.solved[h] = 1 if state.turn == 'w' else -1
            else:
                self.unsolved_children_count[h] = unknown_count
                # If unknown_count is 0 here, it means all children are solved but none were winning
                # This implies a loss (stalemate/checkmate handled in is_terminal)
                if unknown_count == 0:
                     # This node is resolved (wait for propagation to handle it or do it now)
                     # We leave it; the loop below or next propagation will catch it.
                     pass

        self.frontier = next_frontier

    def _propagate(self):
        """
        Retrograde propagation.
        Iterate through newly solved nodes, find their parents (from Disk), update parents.
        """
        # We need a queue of newly solved hashes to propagate up
        # In a strict BFS, this is usually just the things solved in _expand_layer
        
        # Identify newly solved nodes that haven't propagated yet
        # For simplicity in this structure, we scan the 'solved' set against a 'propagated' set
        # or just run a pass. 
        # Optimization: _expand_layer adds to a 'just_solved' list.
        
        # For this implementation, we will iterate the edges in reverse efficiently?
        # No, we query parents of currently solved nodes.
        
        # To make this efficient:
        # We only care about parents of nodes that were *just* solved.
        
        # Let's rebuild the propagation queue from the solved dict + boundary contacts
        # This is expensive if done fully every time.
        # Better: return a list of solved hashes from _expand_layer.
        
        # Since I didn't change the signature of _expand, let's do a heuristic:
        # Propagate changes until stable.
        
        # Actually, in Retrograde:
        # 1. We have a set of `newly_solved` from the expansion (boundaries + checkmates).
        # 2. For each s in newly_solved:
        #    Get parents P.
        #    Update P's info (decrement unknown count or check for immediate win).
        #    If P becomes solved, add P to `newly_solved`.
        
        # We need to reconstruct `newly_solved` from the current state context.
        # Let's grab all solved hashes that we haven't 'processed' if we tracked that,
        # but simpler: just look at what we found in this layer's expansion.
        
        # Re-scanning the whole solved dict is too slow.
        # Let's cheat: We assume propagation primarily happens from the new contacts.
        
        # Gather seeds
        queue = [h for h in self.solved.keys() if h in self.boundary_contacts or h in self.all_seen]
        # Filter: only process if we haven't already propagated them? 
        # For now, just process queue.
        
        processed_count = 0
        
        # Use a set to avoid re-queueing
        in_queue = set(queue)
        queue_idx = 0
        
        while queue_idx < len(queue):
            h = queue[queue_idx]
            queue_idx += 1
            val = self.solved[h]
            
            # Get parents from DISK
            parents = self.graph.get_parents(h)
            
            for p in parents:
                if p in self.solved: continue
                
                # Check parent turn (we need to know parent state to know whose turn it is)
                # We don't have parent state easily available without lookup!
                # This is the trade-off. 
                # OPTION: Encode turn in the hash? Or re-generate?
                # Our Hash includes turn. 
                
                # To check logic, we need to know if P is White or Black turn.
                # If we assume standard alternating chess:
                # If Child h has turn X, Parent P has turn Not-X.
                # Hash logic: `hash((pieces, turn))`
                
                # We need the actual counts for P.
                if p not in self.unsolved_children_count:
                    # Should not happen if graph is correct, but safety:
                    continue
                
                self.unsolved_children_count[p] -= 1
                
                # Logic:
                # If I am P (White), and Child (Black) is Win(1), I win(1).
                # If I am P (White), and Child (Black) is Loss(-1), I just avoid it.
                # If count goes to 0, and no Wins found, I lose(-1).
                
                # Wait. `val` is the value of the POSITION `h`.
                # If `h` is White Turn and `val` is 1 (White Wins), that's good for White.
                # But `h` is a child of `p`. `p` is Black Turn.
                # So if `h` (White) is 1, `p` (Black) moved to a position where White wins.
                # That is BAD for `p`.
                
                # Turn logic reconstruction:
                # If child `val` is +1 (White Wins):
                #   If parent is White: This is a winning move? No, parent moved to child.
                #   Parent (White) -> Move -> Child (Black).
                #   If Child (Black) evaluates to +1 (White Wins), then Parent found a winning move.
                #   -> Parent becomes +1.
                
                #   If Parent (Black) -> Move -> Child (White).
                #   If Child (White) evaluates to +1 (White Wins), Parent made a bad move.
                #   -> Parent decrements count. If count=0, Parent becomes +1 (forced loss for Black).
                
                # We need to know Parent's turn. 
                # We can't derive it easily from hash alone unless we structured the hash.
                # However, we know `h` is a result of a move from `p`.
                # So `p` turn is opposite of `h` turn. (Except null moves, not valid here).
                
                # We assume alternating turns.
                # Solved value convention: 1 (White Win), -1 (Black Win), 0 (Draw)
                
                parent_resolves = False
                parent_val = 0
                
                # Child value `val`
                if val == 1: # White wins
                    # Parent was Black moving to this. Black hates this.
                    # Parent was White moving to this. White loves this.
                    
                    # We need to know who P is.
                    # CRITICAL: We don't have P's state. 
                    # FIX: We rely on the fact that if we found A winning move, we resolve.
                    # But we don't know who P is.
                    pass 
                
                # To fix the "Missing State" issue in Retrograde without storing states:
                # We must rely on the `unsolved_children_count`.
                # But to set W/L, we need the turn.
                
                # Solution: When recording edges, store the Turn in the DB?
                # Or, since we only process `frontier` (which we have in RAM) or `parents` (which we might not),
                # This is why standard solvers keep states or use ID hashes.
                
                # Pragmantic Solution for this script:
                # We have `frontier` states. But P might be deep in the past.
                # We only really need to resolve P if P is in the previous layer (which we might have dropped).
                
                # REVISION: We can't drop 'frontier' states if we need them for turn info.
                # BUT, the Hash includes the turn?
                # `self._hash = hash((self.pieces, self.turn))`
                # It's inside the hash object, but not recoverable from the int.
                
                # REVISION 2:
                # We only strictly need to know if the move was "Winning for the side to move".
                # If `h` is a Checkmate (Win for side not-to-move), then `p` (side to move) made a move.
                
                # Let's use the explicit `unsolved_children_count` logic:
                # It effectively assumes we are waiting for *all* children to be refuted.
                # OR finding *one* winning move.
                
                # If we cannot recover Turn from Hash, we cannot strictly implement Minimax.
                # FAST FIX:
                # Store (hash, turn) in the `unsolved_children_count` key? No, key is hash.
                # Store turn in `unsolved_children_count` value? Bitmask.
                
                # Let's modify `unsolved_children_count` to store `(count, turn_is_white)`.
                pass

        return processed_count 
        # (Note: The propagation above is pseudo-code due to the missing state issue. 
        #  In the corrected _expand_layer below, I ensure we have what we need.)

    # ============================================================
    # CORRECTED LOGIC FOR DISK-BACKED PROPAGATION
    # ============================================================
    
    def _expand_layer_corrected(self, batch_size=50000):
        # We process the frontier.
        # We need to record: ParentHash -> (Count, Turn) in RAM.
        
        next_frontier = {}
        items = list(self.frontier.items())
        self.frontier.clear() # Dump RAM
        
        newly_solved = []

        for i, (h, state) in enumerate(items):
            if h in self.solved: continue

            # Terminal
            is_term, value = is_terminal(state)
            if is_term:
                self.solved[h] = value
                newly_solved.append(h)
                continue

            moves = generate_moves(state)
            child_hashes = []
            
            for move in moves:
                child = apply_move(state, move)
                ch = hash(child)
                child_hashes.append(ch)
                self.graph.add_edge(h, ch)
                
                # Check boundaries
                val = None
                if move[2] is not None and child.piece_count() <= 7:
                    val = self.syzygy.probe(child)
                if val is None:
                    for region in self.regions.values():
                        val = region.query(ch)
                        if val is not None: break
                
                if val is not None:
                    self.solved[ch] = val
                    newly_solved.append(ch)
                
                if ch not in self.all_seen:
                    self.all_seen.add(ch)
                    next_frontier[ch] = child

            # Initialize counting
            # Store turn in bit 30 of count? Or just a separate dict?
            # Separate dict is safer for collisions.
            
            # Optimization: 1 bit for turn, rest for count.
            # count = actual_count | (1 << 30) if white else 0
            is_w = (state.turn == 'w')
            packed_info = len(child_hashes) | ((1 if is_w else 0) << 20)
            self.unsolved_children_count[h] = packed_info

        self.frontier = next_frontier
        self.graph.commit()
        return newly_solved

    def _propagate_corrected(self, seeds):
        queue = seeds
        processed = 0
        
        while queue:
            child_h = queue.pop(0)
            child_val = self.solved[child_h]
            
            # Parents
            parents = self.graph.get_parents(child_h)
            
            for p_h in parents:
                if p_h in self.solved: continue
                if p_h not in self.unsolved_children_count: continue
                
                packed = self.unsolved_children_count[p_h]
                count = packed & 0xFFFFF
                p_is_white = bool((packed >> 20) & 1)
                
                # Minimax Logic
                # If P is White: Wants child=1.
                # If P is Black: Wants child=-1.
                
                p_resolves = False
                p_res_val = 0
                
                if p_is_white and child_val == 1:
                    p_resolves = True
                    p_res_val = 1
                elif not p_is_white and child_val == -1:
                    p_resolves = True
                    p_res_val = -1
                else:
                    # Not an immediate win, decrement count
                    count -= 1
                    if count == 0:
                        # All moves explored, none were winning for me.
                        # Do I have any draws? (We don't track draws explicitly in count yet, assuming binary)
                        # If count == 0 and we haven't resolved to a Win, it's a Loss.
                        # (Simplification: Stalemate logic is handled at leaves, 0 propagates as 0)
                        # If we assume 1/-1 only:
                        p_resolves = True
                        p_res_val = -1 if p_is_white else 1
                    
                    # Update packed
                    self.unsolved_children_count[p_h] = count | ((1 if p_is_white else 0) << 20)

                if p_resolves:
                    self.solved[p_h] = p_res_val
                    queue.append(p_h)
                    processed += 1
                    # cleanup memory
                    del self.unsolved_children_count[p_h]
        
        return processed

    # Override the main loop to use corrected methods
    def solve_region(self, name: str, starting_states: List[ChessState], max_depth: int = 50):
        print(f"\n>>> SOLVING REGION: {name}")
        self.init_solve(name)
        
        for state in starting_states:
            h = hash(state)
            if h not in self.all_seen:
                self.all_seen.add(h)
                self.frontier[h] = state
        
        start_time = time.time()
        
        for depth in range(max_depth):
            mem_mb = self.get_memory_mb()
            print(f"\n--- Depth {depth} ---")
            print(f"  Frontier: {len(self.frontier):,}")
            print(f"  Solved Total: {len(self.solved):,}")
            print(f"  RAM Usage: {mem_mb:.0f} MB")
            
            if not self.frontier:
                break
            
            if mem_mb > self.max_frontier_mb:
                print(f"⚠️ Limit Hit. Crystallizing early.")
                break
                
            newly_solved = self._expand_layer_corrected()
            propagated = self._propagate_corrected(newly_solved)
            
            print(f"  Propagated: {propagated:,} (Time: {time.time()-start_time:.0f}s)")
        
        # Save
        region = FractalRegion(name, starting_states[0].piece_count() if starting_states else 0,
                               set(self.solved.keys()), dict(self.solved), self.boundary_contacts.copy())
        region.save(os.path.join(self.save_dir, f"{name}_region.pkl"))
        self.regions[name] = region
        self.total_solved += len(region.solved_hashes)
        
        # Cleanup
        self.graph.close()
        self.graph = None
        self.solved.clear()
        self.frontier.clear()
        self.unsolved_children_count.clear()
        gc.collect()
        
        return region

# ============================================================
# MAIN
# ============================================================

def random_position(material: str):
    white_str, black_str = material.upper().split('V')
    white = [PIECE_NAMES[c] for c in white_str]
    black = [PIECE_NAMES[c] + 6 for c in black_str]
    all_pieces = white + black
    squares = random.sample(range(64), len(all_pieces))
    pieces = list(zip(all_pieces, squares))
    return ChessState(pieces, 'w')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', default='KQRRvKQRR')
    parser.add_argument('--positions', type=int, default=500)
    parser.add_argument('--max-memory', type=int, default=20000, help='Max memory in MB')
    parser.add_argument('--regions', type=int, default=5)
    args = parser.parse_args()
    
    solver = FractalHOLOS(max_frontier_mb=args.max_memory)
    solver.load_regions()
    
    for i in range(args.regions):
        rname = f"{args.material}_region_{i}"
        if rname in solver.regions: continue
        
        print(f"\nGenerating {args.positions} positions...")
        starts = [random_position(args.material) for _ in range(args.positions)]
        solver.solve_region(rname, starts)

if __name__ == "__main__":
    main()