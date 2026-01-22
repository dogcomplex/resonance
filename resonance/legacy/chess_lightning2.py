"""
8-Piece Chess Lightning Solver
==============================

NO UPFRONT GENERATION!

Instead:
1. Start from a specific 8-piece position (or generate random starting positions)
2. Expand forward via BFS/DFS
3. When we hit a 7-piece position (via capture), probe Syzygy
4. Lightning strikes backward!

This is the wave approach applied properly - we don't need ALL positions,
just the ones reachable from interesting starting points.
"""

import os
import sys
import time
import pickle
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque
from enum import IntEnum
import random

# ============================================================
# CHESS PRIMITIVES (same as before, condensed)
# ============================================================

class Piece(IntEnum):
    EMPTY = 0
    W_KING = 1; W_QUEEN = 2; W_ROOK = 3; W_BISHOP = 4; W_KNIGHT = 5; W_PAWN = 6
    B_KING = 7; B_QUEEN = 8; B_ROOK = 9; B_BISHOP = 10; B_KNIGHT = 11; B_PAWN = 12

PIECE_CHARS = '.KQRBNPkqrbnp'

def is_white(p): return 1 <= p <= 6
def is_black(p): return 7 <= p <= 12

class ChessState:
    """Minimal chess state for endgames"""
    __slots__ = ['pieces', 'turn', '_hash']
    
    def __init__(self, pieces, turn='w'):
        # pieces: tuple of (piece_type, square)
        self.pieces = tuple(sorted(pieces))
        self.turn = turn
        self._hash = None
    
    def __hash__(self):
        if self._hash is None:
            # Symmetry reduction for pawnless
            canonical = self._canonical()
            self._hash = hash((canonical, self.turn))
        return self._hash
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def _canonical(self):
        """Reduce by board symmetry (8-fold for pawnless)"""
        forms = [self.pieces]
        # Add horizontal flip
        forms.append(tuple(sorted((p, self._flip_h(sq)) for p, sq in self.pieces)))
        return min(forms)
    
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
        white = tuple(sorted(p for p, _ in self.pieces if is_white(p)))
        black = tuple(sorted(p for p, _ in self.pieces if is_black(p)))
        return (white, black)
    
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
# FAST MOVE GENERATION
# ============================================================

def generate_moves(state):
    """Generate all legal moves as (from_sq, to_sq, captured_piece_or_None)"""
    board = state.to_board()
    is_white_turn = state.turn == 'w'
    moves = []
    
    for piece, from_sq in state.pieces:
        if is_white_turn and not is_white(piece):
            continue
        if not is_white_turn and not is_black(piece):
            continue
        
        piece_type = piece if is_white(piece) else piece - 6
        
        if piece_type == 1:  # King
            for d in [-9, -8, -7, -1, 1, 7, 8, 9]:
                to_sq = from_sq + d
                if not (0 <= to_sq < 64):
                    continue
                # Prevent wraparound
                if abs((from_sq % 8) - (to_sq % 8)) > 1:
                    continue
                target = board[to_sq]
                if is_white_turn and is_white(target):
                    continue
                if not is_white_turn and is_black(target):
                    continue
                moves.append((from_sq, to_sq, target if target else None))
        
        elif piece_type == 2:  # Queen
            moves.extend(_sliding(board, from_sq, is_white_turn, 
                                  [-9, -8, -7, -1, 1, 7, 8, 9]))
        
        elif piece_type == 3:  # Rook
            moves.extend(_sliding(board, from_sq, is_white_turn, [-8, -1, 1, 8]))
        
        elif piece_type == 4:  # Bishop
            moves.extend(_sliding(board, from_sq, is_white_turn, [-9, -7, 7, 9]))
        
        elif piece_type == 5:  # Knight
            for d in [-17, -15, -10, -6, 6, 10, 15, 17]:
                to_sq = from_sq + d
                if not (0 <= to_sq < 64):
                    continue
                # Prevent wraparound
                from_file, to_file = from_sq % 8, to_sq % 8
                if abs(from_file - to_file) > 2:
                    continue
                target = board[to_sq]
                if is_white_turn and is_white(target):
                    continue
                if not is_white_turn and is_black(target):
                    continue
                moves.append((from_sq, to_sq, target if target else None))
    
    # Filter for legality (king not in check after move)
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
        while 0 <= sq < 64:
            # Check wraparound
            sq_file = sq % 8
            if d in [-1, 7, -9] and sq_file == 7:
                break
            if d in [1, -7, 9] and sq_file == 0:
                break
            
            target = board[sq]
            if target == Piece.EMPTY:
                moves.append((from_sq, sq, None))
            elif (is_white_turn and is_black(target)) or (not is_white_turn and is_white(target)):
                moves.append((from_sq, sq, target))
                break
            else:
                break
            sq += d
    
    return moves


def apply_move(state, move):
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


def in_check(state, color):
    """Is the given color's king in check?"""
    board = state.to_board()
    king = Piece.W_KING if color == 'w' else Piece.B_KING
    king_sq = None
    
    for p, sq in state.pieces:
        if p == king:
            king_sq = sq
            break
    
    if king_sq is None:
        return True  # No king = bad
    
    # Check if any enemy piece attacks king_sq
    enemy_is_white = (color == 'b')
    
    for p, sq in state.pieces:
        if enemy_is_white and not is_white(p):
            continue
        if not enemy_is_white and not is_black(p):
            continue
        
        if attacks(board, sq, king_sq, p):
            return True
    
    return False


def attacks(board, from_sq, to_sq, piece):
    """Does piece at from_sq attack to_sq?"""
    pt = piece if is_white(piece) else piece - 6
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


def path_clear(board, from_sq, to_sq):
    from_rank, from_file = from_sq // 8, from_sq % 8
    to_rank, to_file = to_sq // 8, to_sq % 8
    
    dr = 0 if to_rank == from_rank else (1 if to_rank > from_rank else -1)
    df = 0 if to_file == from_file else (1 if to_file > from_file else -1)
    
    d = dr * 8 + df
    if d == 0:
        return True  # Same square?
    
    sq = from_sq + d
    while sq != to_sq and 0 <= sq < 64:
        # Check for file wraparound
        sq_file = sq % 8
        prev_file = (sq - d) % 8
        if abs(sq_file - prev_file) > 1:
            break
        if board[sq] != Piece.EMPTY:
            return False
        sq += d
    return True


def is_checkmate(state):
    if not in_check(state, state.turn):
        return False
    return len(generate_moves(state)) == 0


def is_stalemate(state):
    if in_check(state, state.turn):
        return False
    return len(generate_moves(state)) == 0


def is_terminal(state):
    """Returns (is_terminal, value)"""
    if is_checkmate(state):
        return True, (-1 if state.turn == 'w' else 1)
    if is_stalemate(state):
        return True, 0
    if len(state.pieces) == 2:  # KvK
        return True, 0
    return False, None


# ============================================================
# SYZYGY PROBE
# ============================================================

class SyzygyProbe:
    def __init__(self, path="./syzygy"):
        self.path = path
        self.tb = None
        self.available = False
        
        try:
            import chess
            import chess.syzygy
            self.tb = chess.syzygy.Tablebase()
            if os.path.exists(path):
                self.tb.add_directory(path)
                self.available = True
                print(f"✓ Syzygy loaded from {path}")
        except ImportError:
            print("✗ python-chess not available")
        except Exception as e:
            print(f"✗ Syzygy error: {e}")
    
    def probe(self, state):
        """Returns +1 (white wins), -1 (black wins), 0 (draw), or None"""
        if not self.available or state.piece_count() > 7:
            return None
        
        try:
            import chess
            board = chess.Board()
            board.clear()
            
            for p, sq in state.pieces:
                pt = ((p - 1) % 6) + 1
                color = is_white(p)
                board.set_piece_at(sq, chess.Piece(pt, color))
            
            board.turn = (state.turn == 'w')
            
            wdl = self.tb.probe_wdl(board)
            if wdl > 0:
                return 1 if state.turn == 'w' else -1
            elif wdl < 0:
                return -1 if state.turn == 'w' else 1
            else:
                return 0
        except Exception as e:
            return None


# ============================================================
# LIGHTNING SOLVER
# ============================================================

SAVE_DIR = "./chess_lightning_state"

class LightningSolver:
    """
    Lightning-first solver for 8-piece endgames.
    
    NO upfront generation!
    Start from specific positions, expand via BFS,
    probe Syzygy when we hit 7-piece captures.
    """
    
    def __init__(self, syzygy_path="./syzygy"):
        self.syzygy = SyzygyProbe(syzygy_path)
        
        self.solved: Dict[int, int] = {}
        self.frontier: Dict[int, ChessState] = {}
        self.all_seen: Set[int] = set()
        self.children: Dict[int, List[int]] = {}
        self.parents: Dict[int, List[int]] = defaultdict(list)
        self.state_turns: Dict[int, str] = {}
        
        self.depth = 0
        self.contact_points = 0
        
        self.metrics = {
            'positions_explored': 0,
            'syzygy_probes': 0,
            'syzygy_hits': 0,
            'contact_time': None,
            'solve_time': None,
        }
    
    def add_starting_position(self, state: ChessState):
        """Add a starting position to explore from"""
        h = hash(state)
        if h not in self.all_seen:
            self.all_seen.add(h)
            self.frontier[h] = state
            self.state_turns[h] = state.turn
            print(f"Added starting position (hash={h})")
            state.display()
    
    def generate_random_start(self, material="KQRRvKQRR"):
        """Generate a random legal starting position"""
        piece_map = {'K': Piece.W_KING, 'Q': Piece.W_QUEEN, 'R': Piece.W_ROOK,
                     'B': Piece.W_BISHOP, 'N': Piece.W_KNIGHT, 'P': Piece.W_PAWN}
        
        white_str, black_str = material.upper().split('V')
        white_pieces = [piece_map[c] for c in white_str]
        black_pieces = [piece_map[c] + 6 for c in black_str]  # +6 for black
        
        all_pieces = white_pieces + black_pieces
        
        # Random placement
        for attempt in range(1000):
            squares = random.sample(range(64), len(all_pieces))
            pieces = list(zip(all_pieces, squares))
            state = ChessState(pieces, 'w')
            
            # Check legality
            board = state.to_board()
            
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
    
    def expand_layer(self, max_positions=100000):
        """Expand frontier by one layer, probing Syzygy on 7-piece captures"""
        items = list(self.frontier.items())[:max_positions]
        total = len(items)
        terminals = 0
        new_states = 0
        contacts = 0
        captures_to_7 = 0  # Track how many captures reduce to ≤7 pieces
        next_frontier = {}
        
        start_time = time.time()
        
        for i, (h, state) in enumerate(items):
            if i > 0 and i % 10000 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"    {i:,}/{total:,} ({100*i/total:.0f}%) - "
                      f"{rate:.0f} pos/s - contacts: {contacts}")
            
            # Check terminal
            term, value = is_terminal(state)
            if term:
                self.solved[h] = value
                terminals += 1
                continue
            
            # Generate moves - CAPTURES FIRST for faster contact
            moves = generate_moves(state)
            # Sort: captures first (captured piece value), then quiet moves
            def move_priority(m):
                if m[2] is not None:  # Capture
                    return -100  # Captures first
                return 0
            moves.sort(key=move_priority)
            
            child_hashes = []
            
            for move in moves:
                child = apply_move(state, move)
                ch = hash(child)
                child_hashes.append(ch)
                self.parents[ch].append(h)
                
                # Is this a capture leading to 7 pieces or fewer?
                if move[2] is not None:
                    child_pieces = child.piece_count()
                    if child_pieces <= 7:
                        captures_to_7 += 1
                        self.metrics['syzygy_probes'] += 1
                        value = self.syzygy.probe(child)
                        
                        if value is not None:
                            self.metrics['syzygy_hits'] += 1
                            self.solved[ch] = value
                            contacts += 1
                            
                            if contacts <= 3:  # Show first few contacts
                                print(f"\n⚡ CONTACT #{contacts}! {child_pieces}-piece position")
                                print(f"   Syzygy says: {'+' if value > 0 else ('=' if value == 0 else '-')}")
                                child.display()
                            
                            if self.metrics['contact_time'] is None:
                                self.metrics['contact_time'] = time.time()
                
                # Add to frontier if not seen
                if ch not in self.all_seen:
                    self.all_seen.add(ch)
                    next_frontier[ch] = child
                    self.state_turns[ch] = child.turn
                    new_states += 1
            
            self.children[h] = child_hashes
        
        # Remove expanded from frontier, add new
        for h, _ in items:
            if h in self.frontier:
                del self.frontier[h]
        
        self.frontier.update(next_frontier)
        self.depth += 1
        self.contact_points += contacts
        self.metrics['positions_explored'] = len(self.all_seen)
        
        return total, terminals, new_states, contacts, captures_to_7
    
    def propagate_solutions(self, max_iters=500):
        """Propagate solutions backward"""
        total_solved = 0
        
        for iteration in range(max_iters):
            newly_solved = 0
            
            for h in list(self.children.keys()):
                if h in self.solved:
                    continue
                
                child_values = []
                unknown = 0
                
                for ch in self.children[h]:
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
            
            total_solved += newly_solved
            
            if newly_solved == 0:
                break
            
            if iteration % 20 == 0 and newly_solved > 0:
                print(f"    Propagation iter {iteration}: +{newly_solved}, total solved: {len(self.solved):,}")
        
        return total_solved
    
    def solve(self, max_depth=50, positions_per_layer=50000):
        """Main solving loop"""
        print("="*60)
        print("⚡ 8-PIECE LIGHTNING SOLVER ⚡")
        print("="*60)
        
        if not self.frontier:
            print("No starting positions! Call add_starting_position() first.")
            return
        
        solve_start = time.time()
        
        for depth in range(max_depth):
            print(f"\n--- DEPTH {depth} ---")
            print(f"  Frontier: {len(self.frontier):,}")
            
            if not self.frontier:
                print("  Frontier empty!")
                break
            
            # Expand
            expanded, terminals, new_states, contacts, captures_to_7 = self.expand_layer(positions_per_layer)
            print(f"  Expanded: {expanded:,}, Terminals: {terminals:,}")
            print(f"  New states: {new_states:,}, Captures→7: {captures_to_7:,}, Contacts: {contacts:,}")
            
            # Propagate
            propagated = self.propagate_solutions()
            print(f"  Propagated: {propagated:,}")
            print(f"  Total solved: {len(self.solved):,} / {len(self.all_seen):,}")
            
            # Time estimate
            elapsed = time.time() - solve_start
            rate = len(self.all_seen) / elapsed if elapsed > 0 else 0
            print(f"  Time: {elapsed:.1f}s, Rate: {rate:.0f} pos/s")
            
            # Check if any starting position is solved
            for h in list(self.frontier.keys())[:10]:
                if h in self.solved:
                    print(f"\n🎉 STARTING POSITION SOLVED! Value: {self.solved[h]}")
        
        # Final stats
        self.metrics['solve_time'] = time.time() - solve_start
        
        print(f"\n{'='*60}")
        print("FINAL STATS")
        print(f"{'='*60}")
        print(f"Positions explored: {len(self.all_seen):,}")
        print(f"Positions solved: {len(self.solved):,}")
        print(f"Syzygy probes: {self.metrics['syzygy_probes']:,}")
        print(f"Syzygy hits: {self.metrics['syzygy_hits']:,}")
        print(f"Total time: {self.metrics['solve_time']:.1f}s")
        
        # Outcome distribution
        if self.solved:
            wins = sum(1 for v in self.solved.values() if v == 1)
            losses = sum(1 for v in self.solved.values() if v == -1)
            draws = sum(1 for v in self.solved.values() if v == 0)
            print(f"\nOutcomes: W={wins:,} D={draws:,} L={losses:,}")


def main():
    solver = LightningSolver("./syzygy")
    
    # Generate random 8-piece starting positions
    print("\nGenerating random KQRRvKQRR starting positions...")
    
    for i in range(10):  # Start with 10 random positions
        state = solver.generate_random_start("KQRRvKQRR")
        if state:
            solver.add_starting_position(state)
    
    print(f"\nStarting with {len(solver.frontier)} positions")
    
    # Solve!
    solver.solve(max_depth=30, positions_per_layer=100000)


if __name__ == "__main__":
    main()
