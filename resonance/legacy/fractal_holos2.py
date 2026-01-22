"""
fractal_holos.py - Fractal Holographic Solver

KEY INSIGHT: Don't keep everything in memory.
Solve regions, crystallize them, abstract them away.

FRACTAL STRUCTURE:
- Same algorithm at every scale
- Solve a region → compress to hologram → use as "solid ground"
- Next region builds on previous holograms
- Self-similar all the way down (and up)

MEMORY MANAGEMENT:
- Cap frontier at N positions
- When full: crystallize current region, save hologram, clear memory
- Resume with hologram as "known boundary"

This is how you actually solve chess:
1. Solve 7-piece (exists: Syzygy)
2. Solve 8-piece regions, using 7-piece as ground
3. Solve 9-piece regions, using 8-piece holograms as ground
4. ... continue up to opening position
5. Each "region" is a fractal piece of the whole
"""

import os
import sys
import time
import pickle
import random
import gc
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum

# ============================================================
# CHESS CORE (minimal, embedded)
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


# Minimal move generation (same as before, condensed)
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
                print(f"✓ Syzygy loaded from {path}")
        except:
            print("✗ Syzygy not available")
    
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
# FRACTAL HOLOGRAM
# ============================================================

@dataclass  
class FractalRegion:
    """A solved region that can be used as 'solid ground'"""
    name: str
    piece_count: int
    solved_hashes: Set[int]  # Hashes of solved positions
    values: Dict[int, int]   # hash -> value
    boundary_hashes: Set[int]  # Hashes at the edge (contacts)
    
    def query(self, h: int) -> Optional[int]:
        return self.values.get(h)
    
    def is_boundary(self, h: int) -> bool:
        return h in self.boundary_hashes
    
    def size_mb(self) -> float:
        # Rough estimate: 20 bytes per solved hash
        return len(self.solved_hashes) * 20 / (1024 * 1024)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'piece_count': self.piece_count,
                'solved_hashes': self.solved_hashes,
                'values': self.values,
                'boundary_hashes': self.boundary_hashes,
            }, f)
        print(f"  [Saved region '{self.name}': {len(self.solved_hashes):,} positions, {self.size_mb():.1f} MB]")
    
    @staticmethod
    def load(path: str) -> 'FractalRegion':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        region = FractalRegion(
            data['name'], data['piece_count'],
            data['solved_hashes'], data['values'], data['boundary_hashes']
        )
        print(f"  [Loaded region '{region.name}': {len(region.solved_hashes):,} positions]")
        return region


# ============================================================
# FRACTAL HOLOS SOLVER
# ============================================================

class FractalHOLOS:
    """
    Fractal Holographic Solver with memory management.
    
    Key features:
    1. Memory-bounded: crystallize and clear when full
    2. Region-based: each crystallized region becomes "solid ground"
    3. Fractal: same algorithm at every scale
    4. Resumable: saves regions to disk
    """
    
    def __init__(self, syzygy_path="./syzygy", save_dir="./fractal_holos",
                 max_frontier_mb=1000):  # Cap at 1GB
        self.syzygy = SyzygyProbe(syzygy_path)
        self.save_dir = save_dir
        self.max_frontier_mb = max_frontier_mb
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Completed regions (loaded from disk or built)
        self.regions: Dict[str, FractalRegion] = {}
        
        # Current solving state
        self.solved: Dict[int, int] = {}
        self.frontier: Dict[int, ChessState] = {}
        self.children: Dict[int, List[int]] = {}
        self.parents: Dict[int, List[int]] = defaultdict(list)
        self.all_seen: Set[int] = set()
        self.boundary_contacts: Set[int] = set()
        
        # Metrics
        self.current_region_name = None
        self.depth = 0
        self.total_solved = 0
    
    def load_regions(self):
        """Load all saved regions"""
        for filename in os.listdir(self.save_dir):
            if filename.endswith('_region.pkl'):
                path = os.path.join(self.save_dir, filename)
                region = FractalRegion.load(path)
                self.regions[region.name] = region
        print(f"Loaded {len(self.regions)} regions")
    
    def memory_usage_mb(self) -> float:
        """Estimate current memory usage more accurately"""
        import sys
        
        # More accurate estimation based on actual object sizes
        # Dictionary overhead is ~200-300 bytes per entry
        # Plus the actual data
        
        frontier_mb = len(self.frontier) * 250 / (1024 * 1024)  # State objects are bigger
        solved_mb = len(self.solved) * 50 / (1024 * 1024)
        children_mb = len(self.children) * 100 / (1024 * 1024)  # Dict + list overhead
        children_mb += sum(len(c) for c in self.children.values()) * 8 / (1024 * 1024)
        parents_mb = len(self.parents) * 100 / (1024 * 1024)
        parents_mb += sum(len(p) for p in self.parents.values()) * 8 / (1024 * 1024)
        all_seen_mb = len(self.all_seen) * 50 / (1024 * 1024)  # Set overhead
        
        total = frontier_mb + solved_mb + children_mb + parents_mb + all_seen_mb
        return total
    
    def checkpoint_progress(self):
        """Save current solving progress (mid-region checkpoint)"""
        checkpoint = {
            'region_name': self.current_region_name,
            'depth': self.depth,
            'solved': self.solved,
            'frontier_hashes': set(self.frontier.keys()),
            'all_seen': self.all_seen,
            'children': self.children,
            'parents': dict(self.parents),
            'boundary_contacts': self.boundary_contacts,
        }
        path = os.path.join(self.save_dir, 'checkpoint.pkl')
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  [Checkpoint saved: depth={self.depth}, solved={len(self.solved):,}]")
    
    def load_checkpoint(self) -> bool:
        """Load mid-region checkpoint if exists"""
        path = os.path.join(self.save_dir, 'checkpoint.pkl')
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
            self.current_region_name = checkpoint['region_name']
            self.depth = checkpoint['depth']
            self.solved = checkpoint['solved']
            self.all_seen = checkpoint['all_seen']
            self.children = checkpoint['children']
            self.parents = defaultdict(list, checkpoint['parents'])
            self.boundary_contacts = checkpoint['boundary_contacts']
            # Note: frontier states need to be regenerated or stored separately
            print(f"✓ Checkpoint loaded: region={self.current_region_name}, depth={self.depth}, solved={len(self.solved):,}")
            return True
        except Exception as e:
            print(f"✗ Checkpoint load failed: {e}")
            return False
    
    def clear_checkpoint(self):
        """Remove checkpoint after region complete"""
        path = os.path.join(self.save_dir, 'checkpoint.pkl')
        if os.path.exists(path):
            os.remove(path)
    
    def clear_memory(self):
        """Clear solving state but keep regions"""
        self.solved.clear()
        self.frontier.clear()
        self.children.clear()
        self.parents.clear()
        self.all_seen.clear()
        self.boundary_contacts.clear()
        gc.collect()
    
    def crystallize_region(self, name: str, piece_count: int):
        """Convert current solved state into a region and save"""
        region = FractalRegion(
            name=name,
            piece_count=piece_count,
            solved_hashes=set(self.solved.keys()),
            values=dict(self.solved),
            boundary_hashes=self.boundary_contacts.copy()
        )
        
        # Save to disk
        path = os.path.join(self.save_dir, f"{name}_region.pkl")
        region.save(path)
        
        # Add to regions
        self.regions[name] = region
        
        return region
    
    def query_regions(self, h: int) -> Optional[int]:
        """Query all loaded regions for a position"""
        for region in self.regions.values():
            value = region.query(h)
            if value is not None:
                return value
        return None
    
    def solve_region(self, name: str, starting_states: List[ChessState],
                     max_depth: int = 50, batch_size: int = 500000):
        """
        Solve a region using fractal HOLOS.
        
        Uses:
        - Syzygy for 7-piece boundary
        - Previously solved regions as additional boundaries
        """
        
        print(f"\n{'='*60}")
        print(f"SOLVING REGION: {name}")
        print(f"{'='*60}")
        
        self.clear_memory()
        self.current_region_name = name
        self.depth = 0
        
        piece_count = starting_states[0].piece_count() if starting_states else 8
        
        # Initialize frontier
        for state in starting_states:
            h = hash(state)
            if h not in self.all_seen:
                self.all_seen.add(h)
                self.frontier[h] = state
        
        print(f"Starting positions: {len(self.frontier)}")
        
        start_time = time.time()
        
        for depth in range(max_depth):
            self.depth = depth
            
            # Check memory - with buffer
            mem_mb = self.memory_usage_mb()
            mem_limit_soft = self.max_frontier_mb * 0.7  # Start warning at 70%
            mem_limit_hard = self.max_frontier_mb * 0.9  # Force crystallize at 90%
            
            if mem_mb > mem_limit_hard:
                print(f"\n⚠️ Hard memory limit reached ({mem_mb:.0f} MB > {mem_limit_hard:.0f} MB)")
                print("Crystallizing current progress...")
                break
            elif mem_mb > mem_limit_soft and depth > 3:
                print(f"\n⚠️ Approaching memory limit ({mem_mb:.0f} MB)")
                # Try to free some memory by clearing non-essential data
                self.parents.clear()
                gc.collect()
                mem_mb = self.memory_usage_mb()
                print(f"  After cleanup: {mem_mb:.0f} MB")
                if mem_mb > mem_limit_hard:
                    print("  Still too high, crystallizing...")
                    break
            
            if not self.frontier:
                print("Frontier empty!")
                break
            
            print(f"\n--- Depth {depth} ---")
            print(f"  Frontier: {len(self.frontier):,}")
            print(f"  Memory: {mem_mb:.0f} MB")
            
            # Expand
            contacts = self._expand_layer(batch_size)
            
            # Propagate
            propagated = self._propagate()
            
            print(f"  Contacts: {contacts:,}, Propagated: {propagated:,}")
            print(f"  Solved: {len(self.solved):,}")
            
            elapsed = time.time() - start_time
            rate = len(self.all_seen) / elapsed if elapsed > 0 else 0
            print(f"  Time: {elapsed:.0f}s, Rate: {rate:.0f} pos/s")
            
            # Checkpoint every 5 depths
            if depth > 0 and depth % 5 == 0:
                self.checkpoint_progress()
        
        # Crystallize
        region = self.crystallize_region(name, piece_count)
        self.total_solved += len(region.solved_hashes)
        self.clear_checkpoint()  # Remove checkpoint after successful crystallization
        
        print(f"\nRegion '{name}' complete: {len(region.solved_hashes):,} positions")
        
        return region
    
    def _expand_layer(self, batch_size: int) -> int:
        """Expand frontier, checking all boundaries"""
        items = list(self.frontier.items())[:batch_size]
        next_frontier = {}
        contacts = 0
        
        for i, (h, state) in enumerate(items):
            if i > 0 and i % 50000 == 0:
                print(f"    {i:,}/{len(items):,} - contacts: {contacts:,}")
            
            # Check terminal
            is_term, value = is_terminal(state)
            if is_term:
                self.solved[h] = value
                self.boundary_contacts.add(h)
                continue
            
            # Generate moves (captures first for faster boundary contact)
            moves = generate_moves(state)
            moves.sort(key=lambda m: 0 if m[2] is None else -1)
            
            child_hashes = []
            
            for move in moves:
                child = apply_move(state, move)
                ch = hash(child)
                child_hashes.append(ch)
                self.parents[ch].append(h)
                
                # Check all boundaries:
                
                # 1. Syzygy (7-piece)
                if move[2] is not None and child.piece_count() <= 7:
                    value = self.syzygy.probe(child)
                    if value is not None:
                        self.solved[ch] = value
                        self.boundary_contacts.add(ch)
                        contacts += 1
                        continue
                
                # 2. Previously solved regions
                region_value = self.query_regions(ch)
                if region_value is not None:
                    self.solved[ch] = region_value
                    contacts += 1
                    continue
                
                # Add to frontier if new
                if ch not in self.all_seen:
                    self.all_seen.add(ch)
                    next_frontier[ch] = child
            
            self.children[h] = child_hashes
        
        # Update frontier
        for h, _ in items:
            self.frontier.pop(h, None)
        self.frontier.update(next_frontier)
        
        return contacts
    
    def _propagate(self, max_iters: int = 200) -> int:
        """Propagate solutions backward"""
        total = 0
        
        for _ in range(max_iters):
            newly_solved = 0
            
            for h in list(self.children.keys()):
                if h in self.solved:
                    continue
                
                child_vals = [self.solved[ch] for ch in self.children[h] if ch in self.solved]
                unknown = sum(1 for ch in self.children[h] if ch not in self.solved)
                
                if not child_vals:
                    continue
                
                # Minimax (assuming White maximizes, Black minimizes)
                # This is simplified - real implementation needs turn tracking
                if 1 in child_vals:
                    self.solved[h] = 1
                    newly_solved += 1
                elif unknown == 0:
                    self.solved[h] = max(child_vals)
                    newly_solved += 1
            
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


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', default='KQRRvKQRR')
    parser.add_argument('--positions', type=int, default=500)
    parser.add_argument('--max-memory', type=int, default=2000, help='Max memory in MB')
    parser.add_argument('--regions', type=int, default=5, help='Number of regions to solve')
    args = parser.parse_args()
    
    print("="*60)
    print("FRACTAL HOLOS - Memory-Bounded Solver")
    print("="*60)
    
    solver = FractalHOLOS(max_frontier_mb=args.max_memory)
    solver.load_regions()
    
    # Solve multiple regions
    for region_idx in range(args.regions):
        region_name = f"{args.material}_region_{region_idx}"
        
        # Check if already solved
        if region_name in solver.regions:
            print(f"\nRegion '{region_name}' already solved, skipping...")
            continue
        
        # Generate starting positions
        print(f"\nGenerating {args.positions} positions for region {region_idx}...")
        starts = []
        for _ in range(args.positions):
            state = random_position(args.material)
            if state:
                starts.append(state)
        
        if len(starts) < 3:
            print("  First 3 positions:")
            for s in starts[:3]:
                s.display()
        
        # Solve region
        solver.solve_region(region_name, starts)
    
    # Summary
    print(f"\n{'='*60}")
    print("FRACTAL HOLOS SUMMARY")
    print(f"{'='*60}")
    print(f"Total regions: {len(solver.regions)}")
    print(f"Total positions solved: {solver.total_solved:,}")
    
    total_mb = sum(r.size_mb() for r in solver.regions.values())
    print(f"Total storage: {total_mb:.1f} MB")
    
    print("\nRegions:")
    for name, region in solver.regions.items():
        print(f"  {name}: {len(region.solved_hashes):,} positions ({region.size_mb():.1f} MB)")


if __name__ == "__main__":
    main()
