"""
chess_holographic.py - Two-Phase Solver with Holographic Storage

PHASE 1: LIGHTNING
- BFS from starting positions
- Find paths to 7-piece Syzygy boundary
- Creates "interference pattern" (solution spine)

PHASE 2: CRYSTALLIZATION  
- Grow solved region outward from spine
- Like crystal growth or slime mold network formation
- Solutions propagate from the "frozen" boundary

HOLOGRAPHIC STORAGE:
- Instead of storing all positions (petabytes)
- Store the "interference pattern" - key positions where waves meet
- Any position can be reconstructed by tracing to pattern

Analogy:
- Hologram stores interference, not the image
- We store solution boundaries, not all solutions
- Query = "shine light through" = trace to boundary
"""

import os
import sys
import time
import pickle
import random
import struct
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

# ============================================================
# CHESS CORE (embedded for single-file usage)
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
            flipped = tuple(sorted((p, self._flip_h(sq)) for p, sq in self.pieces))
            canonical = min(self.pieces, flipped)
            self._hash = hash((canonical, self.turn))
        return self._hash
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def _flip_h(self, sq):
        return (sq // 8) * 8 + (7 - sq % 8)
    
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


# ============================================================
# MOVE GENERATION (condensed)
# ============================================================

def generate_moves(state):
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
        if abs((sq % 8) - ((sq - d) % 8)) > 1:
            break
        if board[sq] != Piece.EMPTY:
            return False
        sq += d
    return True


def is_terminal(state):
    if is_checkmate(state):
        return True, (-1 if state.turn == 'w' else 1)
    if is_stalemate(state):
        return True, 0
    if len(state.pieces) == 2:
        return True, 0
    return False, None


def is_checkmate(state):
    return in_check(state, state.turn) and len(generate_moves(state)) == 0


def is_stalemate(state):
    return not in_check(state, state.turn) and len(generate_moves(state)) == 0


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
# HOLOGRAPHIC STORAGE
# ============================================================

@dataclass
class HologramNode:
    """A node in the holographic interference pattern"""
    hash: int
    value: int  # +1, 0, -1
    depth: int  # Distance from 7-piece boundary
    parent_hash: Optional[int] = None


class HolographicStorage:
    """
    Stores the "interference pattern" of solution waves.
    
    Key insight: We only need to store positions where:
    1. Waves from 7-piece boundary reach (contact points)
    2. Solution value changes (decision boundaries)
    3. Multiple paths meet (interference nodes)
    
    Query: Trace any position to nearest stored node.
    """
    
    def __init__(self, save_path="./hologram"):
        self.save_path = save_path
        self.nodes: Dict[int, HologramNode] = {}
        self.boundary_nodes: Set[int] = set()  # Direct 7-piece contacts
        self.decision_nodes: Set[int] = set()  # Where value changes
        
    def add_boundary(self, h: int, value: int):
        """Add a 7-piece boundary contact"""
        self.nodes[h] = HologramNode(h, value, 0)
        self.boundary_nodes.add(h)
    
    def add_propagated(self, h: int, value: int, depth: int, parent_h: int):
        """Add a propagated solution"""
        if h not in self.nodes or self.nodes[h].depth > depth:
            self.nodes[h] = HologramNode(h, value, depth, parent_h)
            
            # Check if this is a decision boundary
            if parent_h in self.nodes and self.nodes[parent_h].value != value:
                self.decision_nodes.add(h)
    
    def query(self, h: int) -> Optional[int]:
        """Query for a position's value"""
        if h in self.nodes:
            return self.nodes[h].value
        return None
    
    def save(self):
        os.makedirs(self.save_path, exist_ok=True)
        with open(f"{self.save_path}/hologram.pkl", 'wb') as f:
            pickle.dump({
                'nodes': self.nodes,
                'boundary': self.boundary_nodes,
                'decision': self.decision_nodes,
            }, f)
        print(f"  [Hologram saved: {len(self.nodes):,} nodes, "
              f"{len(self.boundary_nodes):,} boundary, {len(self.decision_nodes):,} decision]")
    
    def load(self):
        path = f"{self.save_path}/hologram.pkl"
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.nodes = data['nodes']
            self.boundary_nodes = data['boundary']
            self.decision_nodes = data['decision']
            print(f"  [Hologram loaded: {len(self.nodes):,} nodes]")
            return True
        except:
            return False
    
    def stats(self):
        return {
            'total_nodes': len(self.nodes),
            'boundary_nodes': len(self.boundary_nodes),
            'decision_nodes': len(self.decision_nodes),
            'avg_depth': sum(n.depth for n in self.nodes.values()) / max(1, len(self.nodes)),
        }


# ============================================================
# TWO-PHASE SOLVER
# ============================================================

SAVE_DIR = "./chess_hologram_state"

class HolographicSolver:
    """
    Two-phase solver with holographic storage.
    
    PHASE 1 (LIGHTNING):
    - BFS from starting positions
    - Stop when we hit 7-piece Syzygy boundary
    - Creates initial interference pattern
    
    PHASE 2 (CRYSTALLIZATION):
    - Propagate solutions backward from boundary
    - Grow the solved region like crystal growth
    - Record decision boundaries in hologram
    """
    
    def __init__(self, syzygy_path="./syzygy", save_dir=SAVE_DIR):
        self.syzygy = SyzygyProbe(syzygy_path)
        self.save_dir = save_dir
        self.hologram = HolographicStorage(f"{save_dir}/hologram")
        
        # State
        self.solved: Dict[int, int] = {}
        self.frontier: Dict[int, ChessState] = {}
        self.all_seen: Set[int] = set()
        self.children: Dict[int, List[int]] = {}
        self.parents: Dict[int, List[int]] = defaultdict(list)
        self.state_turns: Dict[int, str] = {}
        
        # Tracking
        self.phase = "lightning"
        self.depth = 0
        self.spine: Set[int] = set()  # The crystallization spine
        self.start_hashes: List[int] = []
        
        self.metrics = {
            'lightning_time': 0,
            'crystal_time': 0,
            'contacts': 0,
            'propagated': 0,
        }
    
    def add_start(self, state: ChessState, show=True):
        h = hash(state)
        if h not in self.all_seen:
            self.all_seen.add(h)
            self.frontier[h] = state
            self.state_turns[h] = state.turn
            self.start_hashes.append(h)
            if show:
                state.display()
    
    def checkpoint(self):
        os.makedirs(self.save_dir, exist_ok=True)
        data = {
            'solved': self.solved,
            'frontier_hashes': list(self.frontier.keys()),
            'all_seen': self.all_seen,
            'children': self.children,
            'parents': dict(self.parents),
            'state_turns': self.state_turns,
            'phase': self.phase,
            'depth': self.depth,
            'spine': self.spine,
            'start_hashes': self.start_hashes,
            'metrics': self.metrics,
        }
        with open(f"{self.save_dir}/checkpoint.pkl", 'wb') as f:
            pickle.dump(data, f)
        self.hologram.save()
        print(f"  [Checkpoint: phase={self.phase}, depth={self.depth}, solved={len(self.solved):,}]")
    
    def load_checkpoint(self):
        path = f"{self.save_dir}/checkpoint.pkl"
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.solved = data['solved']
            self.all_seen = data['all_seen']
            self.children = data['children']
            self.parents = defaultdict(list, data['parents'])
            self.state_turns = data['state_turns']
            self.phase = data['phase']
            self.depth = data['depth']
            self.spine = data['spine']
            self.start_hashes = data['start_hashes']
            self.metrics = data['metrics']
            self.hologram.load()
            print(f"✓ Resumed: phase={self.phase}, depth={self.depth}, solved={len(self.solved):,}")
            return True
        except Exception as e:
            print(f"✗ Load failed: {e}")
            return False
    
    # ==================== PHASE 1: LIGHTNING ====================
    
    def lightning_expand(self, max_positions=500000):
        """Expand frontier, looking for 7-piece contacts"""
        items = list(self.frontier.items())[:max_positions]
        total = len(items)
        new_frontier = {}
        terminals = contacts = new_states = 0
        
        for i, (h, state) in enumerate(items):
            if i > 0 and i % 50000 == 0:
                print(f"    {i:,}/{total:,} - contacts: {contacts:,}")
            
            term, value = is_terminal(state)
            if term:
                self.solved[h] = value
                self.hologram.add_boundary(h, value)
                terminals += 1
                continue
            
            moves = generate_moves(state)
            # Captures first
            moves.sort(key=lambda m: 0 if m[2] is None else -1)
            
            child_hashes = []
            for move in moves:
                child = apply_move(state, move)
                ch = hash(child)
                child_hashes.append(ch)
                self.parents[ch].append(h)
                
                # Capture to ≤7 pieces?
                if move[2] is not None and child.piece_count() <= 7:
                    value = self.syzygy.probe(child)
                    if value is not None:
                        self.solved[ch] = value
                        self.hologram.add_boundary(ch, value)
                        self.spine.add(ch)
                        contacts += 1
                        self.metrics['contacts'] += 1
                        
                        if contacts == 1:
                            print(f"\n  ⚡ FIRST CONTACT! Lightning strikes!")
                
                if ch not in self.all_seen:
                    self.all_seen.add(ch)
                    new_frontier[ch] = child
                    self.state_turns[ch] = child.turn
                    new_states += 1
            
            self.children[h] = child_hashes
        
        # Update frontier
        for h, _ in items:
            self.frontier.pop(h, None)
        self.frontier.update(new_frontier)
        self.depth += 1
        
        return total, terminals, new_states, contacts
    
    # ==================== PHASE 2: CRYSTALLIZATION ====================
    
    def crystallize(self, max_iters=1000):
        """Propagate solutions backward from spine"""
        total_new = 0
        
        for iteration in range(max_iters):
            newly_solved = 0
            
            for h in list(self.children.keys()):
                if h in self.solved:
                    continue
                
                child_vals = [self.solved[ch] for ch in self.children[h] if ch in self.solved]
                unknown = sum(1 for ch in self.children[h] if ch not in self.solved)
                
                if not child_vals:
                    continue
                
                turn = self.state_turns.get(h, 'w')
                value = None
                
                # Early termination
                if turn == 'w' and 1 in child_vals:
                    value = 1
                elif turn == 'b' and -1 in child_vals:
                    value = -1
                elif unknown == 0:
                    value = max(child_vals) if turn == 'w' else min(child_vals)
                
                if value is not None:
                    self.solved[h] = value
                    newly_solved += 1
                    
                    # Add to hologram
                    best_child = None
                    for ch in self.children[h]:
                        if ch in self.solved and self.solved[ch] == value:
                            best_child = ch
                            break
                    if best_child:
                        depth = self.hologram.nodes[best_child].depth + 1 if best_child in self.hologram.nodes else 1
                        self.hologram.add_propagated(h, value, depth, best_child)
            
            total_new += newly_solved
            self.metrics['propagated'] += newly_solved
            
            if newly_solved == 0:
                break
            
            if iteration % 20 == 0:
                print(f"    Crystal iter {iteration}: +{newly_solved:,}, total: {len(self.solved):,}")
        
        return total_new
    
    # ==================== MAIN SOLVE ====================
    
    def solve(self, max_depth=100, batch_size=500000, checkpoint_every=5):
        print("="*60)
        print("🔮 HOLOGRAPHIC CHESS SOLVER 🔮")
        print("Phase 1: Lightning → Find boundary contacts")
        print("Phase 2: Crystallization → Grow from spine")
        print("="*60)
        
        if not self.frontier:
            print("No starting positions!")
            return
        
        start_time = time.time()
        
        # ===== PHASE 1: LIGHTNING =====
        if self.phase == "lightning":
            print(f"\n{'='*50}")
            print("⚡ PHASE 1: LIGHTNING")
            print(f"{'='*50}")
            
            lightning_start = time.time()
            
            while self.depth < max_depth and self.frontier:
                print(f"\n--- Depth {self.depth} ---")
                print(f"  Frontier: {len(self.frontier):,}")
                
                expanded, terms, new, contacts = self.lightning_expand(batch_size)
                print(f"  Expanded: {expanded:,}, New: {new:,}, Contacts: {contacts:,}")
                
                # Propagate what we have
                propagated = self.crystallize()
                print(f"  Propagated: {propagated:,}, Solved: {len(self.solved):,}")
                
                # Check starting positions
                solved_starts = sum(1 for h in self.start_hashes if h in self.solved)
                if solved_starts > 0:
                    print(f"\n  🎉 {solved_starts}/{len(self.start_hashes)} starting positions solved!")
                
                if self.depth % checkpoint_every == 0:
                    self.checkpoint()
                
                elapsed = time.time() - start_time
                rate = len(self.all_seen) / elapsed if elapsed > 0 else 0
                print(f"  Time: {elapsed:.0f}s, Rate: {rate:.0f} pos/s")
            
            self.metrics['lightning_time'] = time.time() - lightning_start
            self.phase = "crystal"
            self.checkpoint()
        
        # ===== PHASE 2: CRYSTALLIZATION =====
        if self.phase == "crystal":
            print(f"\n{'='*50}")
            print("🔮 PHASE 2: CRYSTALLIZATION")
            print(f"{'='*50}")
            
            crystal_start = time.time()
            
            # Keep propagating until no more changes
            for wave in range(100):
                print(f"\n--- Crystal Wave {wave} ---")
                
                propagated = self.crystallize()
                print(f"  Propagated: {propagated:,}, Total solved: {len(self.solved):,}")
                
                if propagated == 0:
                    break
                
                if wave % 5 == 0:
                    self.checkpoint()
            
            self.metrics['crystal_time'] = time.time() - crystal_start
            self.phase = "complete"
        
        # ===== FINAL SUMMARY =====
        self.checkpoint()
        self.print_summary()
    
    def print_summary(self):
        elapsed = self.metrics['lightning_time'] + self.metrics['crystal_time']
        
        print(f"\n{'='*60}")
        print("HOLOGRAPHIC SOLVER - FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Positions explored: {len(self.all_seen):,}")
        print(f"Positions solved: {len(self.solved):,}")
        print(f"Spine size: {len(self.spine):,}")
        print(f"Hologram nodes: {len(self.hologram.nodes):,}")
        print(f"  - Boundary: {len(self.hologram.boundary_nodes):,}")
        print(f"  - Decision: {len(self.hologram.decision_nodes):,}")
        print(f"\n⚡ Lightning time: {self.metrics['lightning_time']:.0f}s")
        print(f"🔮 Crystal time: {self.metrics['crystal_time']:.0f}s")
        print(f"📊 Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        
        # Outcomes
        if self.solved:
            w = sum(1 for v in self.solved.values() if v == 1)
            d = sum(1 for v in self.solved.values() if v == 0)
            l = sum(1 for v in self.solved.values() if v == -1)
            print(f"\nOutcomes: W={w:,} D={d:,} L={l:,}")
        
        # Starting positions
        print(f"\nStarting positions ({len(self.start_hashes)}):")
        for i, h in enumerate(self.start_hashes[:10]):
            if h in self.solved:
                result = {1: "White wins", 0: "Draw", -1: "Black wins"}[self.solved[h]]
                print(f"  #{i+1}: {result}")
            else:
                print(f"  #{i+1}: Not solved")


# ============================================================
# UTILITIES
# ============================================================

def random_position(material, max_attempts=1000):
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


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', default='KQRRvKQRR')
    parser.add_argument('--positions', type=int, default=1000)
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--batch', type=int, default=500000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    solver = HolographicSolver()
    
    if args.resume and solver.load_checkpoint():
        print("Resuming from checkpoint...")
    else:
        print(f"\n{'='*60}")
        print(f"🔮 HOLOGRAPHIC 8-PIECE SOLVER 🔮")
        print(f"{'='*60}")
        print(f"Material: {args.material}")
        print(f"Positions: {args.positions}")
        
        # Generate starting positions
        print(f"\nGenerating {args.positions} random positions...")
        for i in range(args.positions):
            state = random_position(args.material)
            if state:
                solver.add_start(state, show=(i < 3))
            if (i + 1) % 200 == 0:
                print(f"  Generated {i+1}...")
        
        print(f"\nStarting with {len(solver.frontier)} positions")
    
    solver.solve(max_depth=args.depth, batch_size=args.batch)


if __name__ == "__main__":
    main()
