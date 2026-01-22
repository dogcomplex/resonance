"""
Connect-4 Crystalline Wave Solver

Inspired by:
- River deltas (main channel → branches)  
- Lichtenberg figures (trunk → fractal branches)
- Crystal growth (seed → propagating front)
- Viscous fingering (first path → pressure source)

Key insight: The first solution becomes a NEW BOUNDARY CONDITION.
Subsequent exploration propagates FROM the crystallized spine,
not just from the original endpoints.

Two phases:
1. LIGHTNING: Find a solution path (creates crystallized spine)
2. CRYSTALLIZATION: Grow the solved region outward from the spine

This is how nature "completes" a solution - it doesn't re-solve
from scratch, it uses the existing structure as scaffolding.
"""

import pickle
import json
import os
import time
import gc
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, Optional, List
from dataclasses import dataclass

SAVE_DIR = "./c4_crystal_state"
BATCH_SIZE = 500000

@dataclass(frozen=True)
class C4Features:
    """Equivalence class features"""
    x_count: int
    o_count: int
    x_threats: int
    o_threats: int
    height_profile: Tuple[int, ...]
    turn: str
    
    def __hash__(self):
        return hash((self.x_count, self.o_count, self.x_threats, 
                     self.o_threats, self.height_profile, self.turn))


class C4State:
    """Compact Connect-4 state"""
    __slots__ = ['cols', 'turn', '_hash']
    
    def __init__(self, cols=None, turn='X'):
        if cols is None:
            cols = tuple('.' * 6 for _ in range(7))
        self.cols = cols
        self.turn = turn
        self._hash = None
    
    def __hash__(self):
        if self._hash is None:
            mirror = tuple(reversed(self.cols))
            self._hash = hash((min(self.cols, mirror), self.turn))
        return self._hash
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def height(self, col):
        return 6 - self.cols[col].count('.')
    
    def can_play(self, col):
        return self.cols[col][5] == '.'
    
    def play(self, col):
        if not self.can_play(col):
            return None
        h = self.height(col)
        new_col = self.cols[col][:h] + self.turn + self.cols[col][h+1:]
        new_cols = list(self.cols)
        new_cols[col] = new_col
        return C4State(tuple(new_cols), 'O' if self.turn == 'X' else 'X')
    
    def get(self, col, row):
        if 0 <= col < 7 and 0 <= row < 6:
            c = self.cols[col][row]
            return c if c != '.' else None
        return None
    
    def get_valid_moves(self):
        return [c for c in range(7) if self.can_play(c)]
    
    def piece_count(self):
        return sum(c.count('X') + c.count('O') for c in self.cols)
    
    def check_win(self):
        for col in range(7):
            for row in range(6):
                p = self.get(col, row)
                if p is None:
                    continue
                if col <= 3 and all(self.get(col+i, row) == p for i in range(4)):
                    return p
                if row <= 2 and all(self.get(col, row+i) == p for i in range(4)):
                    return p
                if col <= 3 and row <= 2 and all(self.get(col+i, row+i) == p for i in range(4)):
                    return p
                if col <= 3 and row >= 3 and all(self.get(col+i, row-i) == p for i in range(4)):
                    return p
        return None
    
    def is_terminal(self):
        return self.check_win() is not None or self.piece_count() == 42
    
    def terminal_value(self):
        w = self.check_win()
        if w == 'X': return 1
        elif w == 'O': return -1
        else: return 0
    
    def get_features(self):
        heights = tuple(sorted(self.height(c) for c in range(7)))
        x_count = sum(c.count('X') for c in self.cols)
        o_count = sum(c.count('O') for c in self.cols)
        
        def count_threats(player):
            threats = 0
            for col in range(7):
                for row in range(6):
                    for dc, dr in [(1,0), (0,1), (1,1), (1,-1)]:
                        if col + 3*dc > 6 or row + 3*dr > 5 or row + 3*dr < 0:
                            continue
                        window = [self.get(col+i*dc, row+i*dr) for i in range(4)]
                        if window.count(player) == 3 and window.count(None) == 1:
                            threats += 1
            return threats
        
        return C4Features(x_count, o_count, count_threats('X'), 
                         count_threats('O'), heights, self.turn)
    
    def to_compact(self):
        return (self.cols, self.turn)
    
    @staticmethod
    def from_compact(data):
        return C4State(data[0], data[1])


class CrystallineSolver:
    """
    Two-phase crystalline solver.
    
    Phase 1 (LIGHTNING): Bidirectional search to find first solution
    - Creates a "crystallized spine" from start to terminals
    
    Phase 2 (CRYSTALLIZATION): Grow solved region from the spine
    - The spine acts as a new boundary condition
    - Waves propagate outward from BOTH original boundaries AND the spine
    - Like crystal growth or river delta branching
    """
    
    def __init__(self):
        # State storage
        self.solved: Dict[int, int] = {}
        self.frontier: Dict[int, tuple] = {}  # Current exploration edge
        self.all_seen: Set[int] = set()
        
        # Graph structure  
        self.children: Dict[int, List[int]] = {}
        self.parents: Dict[int, List[int]] = defaultdict(list)
        self.state_turns: Dict[int, str] = {}
        
        # Crystallization tracking
        self.crystal_spine: Set[int] = set()  # The "main channel" - first solution path
        self.crystal_front: Set[int] = set()  # Current crystallization boundary
        
        # Equivalence
        self.equiv_classes: Dict[C4Features, Set[int]] = defaultdict(set)
        self.equiv_outcomes: Dict[C4Features, Optional[int]] = {}
        
        # Phasing
        self.phase = "lightning"  # "lightning", "crystallizing", "complete"
        self.depth = 0
        self.lightning_time = None
        self.lightning_solution = None
        
        self.metrics = {
            'total_time': 0.0,
            'lightning_time': 0.0,
            'crystal_time': 0.0,
            'lightning_states': 0,
            'crystal_states': 0,
        }
    
    def initialize(self):
        start = C4State()
        h = hash(start)
        self.frontier[h] = start.to_compact()
        self.all_seen.add(h)
        self.state_turns[h] = start.turn
        self.equiv_classes[start.get_features()].add(h)
        print(f"Initialized at START position")
    
    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        data = {
            'solved': self.solved,
            'frontier': self.frontier,
            'all_seen': self.all_seen,
            'children': self.children,
            'parents': dict(self.parents),
            'state_turns': self.state_turns,
            'crystal_spine': self.crystal_spine,
            'crystal_front': self.crystal_front,
            'equiv_classes': self.equiv_classes,
            'equiv_outcomes': self.equiv_outcomes,
            'phase': self.phase,
            'depth': self.depth,
            'lightning_time': self.lightning_time,
            'lightning_solution': self.lightning_solution,
            'metrics': self.metrics,
        }
        with open(f"{SAVE_DIR}/state.pkl", 'wb') as f:
            pickle.dump(data, f)
        print(f"  [Saved: phase={self.phase}, depth={self.depth}, "
              f"solved={len(self.solved):,}, spine={len(self.crystal_spine):,}]")
    
    def load(self):
        if not os.path.exists(f"{SAVE_DIR}/state.pkl"):
            return False
        try:
            with open(f"{SAVE_DIR}/state.pkl", 'rb') as f:
                data = pickle.load(f)
            self.solved = data['solved']
            self.frontier = data['frontier']
            self.all_seen = data['all_seen']
            self.children = data['children']
            self.parents = defaultdict(list, data['parents'])
            self.state_turns = data['state_turns']
            self.crystal_spine = data['crystal_spine']
            self.crystal_front = data['crystal_front']
            self.equiv_classes = data['equiv_classes']
            self.equiv_outcomes = data['equiv_outcomes']
            self.phase = data['phase']
            self.depth = data['depth']
            self.lightning_time = data['lightning_time']
            self.lightning_solution = data['lightning_solution']
            self.metrics = data['metrics']
            print(f"Loaded: phase={self.phase}, depth={self.depth}, "
                  f"solved={len(self.solved):,}, spine={len(self.crystal_spine):,}")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False
    
    def _update_equiv_outcome(self, features, value):
        if features in self.equiv_outcomes:
            if self.equiv_outcomes[features] != value:
                self.equiv_outcomes[features] = None
        else:
            self.equiv_outcomes[features] = value
    
    def _get_turn(self, h):
        if h in self.state_turns:
            return self.state_turns[h]
        return 'X'
    
    # ========== LIGHTNING PHASE ==========
    
    def expand_frontier(self):
        """Expand frontier by one depth layer"""
        items = list(self.frontier.items())
        total = len(items)
        terminals = 0
        new_states = 0
        next_frontier = {}
        
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = items[batch_start:batch_end]
            
            for h, compact in batch:
                state = C4State.from_compact(compact)
                self.state_turns[h] = state.turn
                
                if state.is_terminal():
                    value = state.terminal_value()
                    self.solved[h] = value
                    self.crystal_front.add(h)  # Terminals are initial crystal front
                    terminals += 1
                    features = state.get_features()
                    self._update_equiv_outcome(features, value)
                    continue
                
                moves = state.get_valid_moves()
                moves.sort(key=lambda c: abs(c - 3))  # Center first
                
                child_hashes = []
                for move in moves:
                    child = state.play(move)
                    ch = hash(child)
                    child_hashes.append(ch)
                    self.parents[ch].append(h)
                    
                    if ch not in self.all_seen:
                        self.all_seen.add(ch)
                        next_frontier[ch] = child.to_compact()
                        self.state_turns[ch] = child.turn
                        new_states += 1
                        features = child.get_features()
                        self.equiv_classes[features].add(ch)
                
                self.children[h] = child_hashes
            
            if batch_end < total and batch_end % BATCH_SIZE == 0:
                print(f"    Batch {batch_end}/{total}")
        
        self.frontier = next_frontier
        self.depth += 1
        return total, terminals, new_states
    
    def propagate_solutions(self):
        """Propagate solutions with early termination"""
        newly_solved = 0
        changed = True
        iterations = 0
        
        while changed and iterations < 500:
            changed = False
            iterations += 1
            
            for ph in list(self.children.keys()):
                if ph in self.solved:
                    continue
                
                child_list = self.children[ph]
                child_values = []
                unknown = 0
                
                for ch in child_list:
                    if ch in self.solved:
                        child_values.append(self.solved[ch])
                    else:
                        unknown += 1
                
                if not child_values:
                    continue
                
                turn = self._get_turn(ph)
                
                # Early termination
                if turn == 'X' and 1 in child_values:
                    self.solved[ph] = 1
                    self.crystal_front.add(ph)
                    newly_solved += 1
                    changed = True
                elif turn == 'O' and -1 in child_values:
                    self.solved[ph] = -1
                    self.crystal_front.add(ph)
                    newly_solved += 1
                    changed = True
                elif unknown == 0:
                    value = max(child_values) if turn == 'X' else min(child_values)
                    self.solved[ph] = value
                    self.crystal_front.add(ph)
                    newly_solved += 1
                    changed = True
        
        return newly_solved
    
    def propagate_equivalence(self):
        """Propagate via equivalence"""
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
                    self.crystal_front.add(h)
                    count += 1
        return count
    
    def extract_spine(self):
        """
        Extract the principal variation (main solution path).
        This becomes the crystallized spine.
        """
        spine = []
        state = C4State()
        h = hash(state)
        
        while h in self.solved and h in self.children:
            self.crystal_spine.add(h)
            spine.append(h)
            
            if state.is_terminal():
                break
            
            # Find best child
            best_child = None
            best_value = None
            turn = self._get_turn(h)
            
            for ch in self.children[h]:
                if ch not in self.solved:
                    continue
                cv = self.solved[ch]
                if best_value is None:
                    best_value = cv
                    best_child = ch
                elif turn == 'X' and cv > best_value:
                    best_value = cv
                    best_child = ch
                elif turn == 'O' and cv < best_value:
                    best_value = cv
                    best_child = ch
            
            if best_child is None:
                break
            
            # Find which move leads to best_child
            for move in state.get_valid_moves():
                child = state.play(move)
                if hash(child) == best_child:
                    state = child
                    h = best_child
                    break
            else:
                break
        
        return spine
    
    def check_start_solved(self):
        start_h = hash(C4State())
        if start_h in self.solved:
            return True, self.solved[start_h]
        return False, None
    
    # ========== CRYSTALLIZATION PHASE ==========
    
    def crystallize_from_spine(self):
        """
        Grow the solved region outward from the crystallized spine.
        
        The spine acts as a new boundary - we propagate solutions
        FROM spine states to their siblings (other branches).
        
        This is like:
        - River delta branching from main channel
        - Crystal growth from seed
        - Lichtenberg figure branching from trunk
        """
        newly_solved = 0
        wave_iteration = 0
        
        # Initialize: solved states adjacent to spine are the growth front
        growth_front = set()
        for spine_h in self.crystal_spine:
            # Add siblings of spine (other children of spine's parents)
            for parent_h in self.parents.get(spine_h, []):
                if parent_h in self.children:
                    for sibling_h in self.children[parent_h]:
                        if sibling_h not in self.solved and sibling_h in self.all_seen:
                            growth_front.add(sibling_h)
            
            # Add children of spine that aren't solved
            if spine_h in self.children:
                for child_h in self.children[spine_h]:
                    if child_h not in self.solved and child_h in self.all_seen:
                        growth_front.add(child_h)
        
        print(f"\n  Initial growth front: {len(growth_front):,} states adjacent to spine")
        
        while growth_front or self.frontier:
            wave_iteration += 1
            wave_solved = 0
            
            # Expand frontier if we still have unexplored states
            if self.frontier:
                expanded, terminals, new_states = self.expand_frontier()
                print(f"  Wave {wave_iteration}: expanded {expanded:,}, "
                      f"terminals {terminals:,}, new {new_states:,}")
            
            # Propagate solutions (this will solve states near the spine/solved region)
            prop = self.propagate_solutions()
            equiv = self.propagate_equivalence()
            wave_solved = prop + equiv
            newly_solved += wave_solved
            
            # Update growth front: unsolved states adjacent to newly solved
            new_front = set()
            for h in self.crystal_front:
                if h in self.children:
                    for ch in self.children[h]:
                        if ch not in self.solved and ch in self.all_seen:
                            new_front.add(ch)
                for ph in self.parents.get(h, []):
                    if ph not in self.solved:
                        new_front.add(ph)
            
            growth_front = new_front
            self.crystal_front.clear()  # Reset for next wave
            
            if wave_solved > 0:
                print(f"    Crystallized: {wave_solved:,} (prop={prop:,}, equiv={equiv:,})")
                print(f"    Total solved: {len(self.solved):,}, growth front: {len(growth_front):,}")
            
            # Check completion
            if not self.frontier and wave_solved == 0:
                break
            
            # Periodic save
            if wave_iteration % 5 == 0:
                self.save()
        
        return newly_solved
    
    # ========== MAIN SOLVE ==========
    
    def solve(self, max_depth=42):
        """
        Main solving loop.
        
        Phase 1: Lightning - find first solution
        Phase 2: Crystallization - grow from spine to solve everything
        """
        print("="*60)
        print("🔮 CRYSTALLINE WAVE SOLVER 🔮")
        print("Phase 1: Lightning → Find solution spine")
        print("Phase 2: Crystallization → Grow from spine")
        print("="*60)
        
        start_time = time.time()
        
        # ===== PHASE 1: LIGHTNING =====
        if self.phase == "lightning":
            print(f"\n{'='*60}")
            print("⚡ PHASE 1: LIGHTNING")
            print(f"{'='*60}")
            
            while self.depth <= max_depth and self.frontier:
                iter_start = time.time()
                
                print(f"\n--- Depth {self.depth} ---")
                
                expanded, terminals, new_states = self.expand_frontier()
                print(f"  Expanded: {expanded:,}, Terminals: {terminals:,}, New: {new_states:,}")
                
                prop = self.propagate_solutions()
                equiv = self.propagate_equivalence()
                print(f"  Propagated: {prop:,}, Equiv: {equiv:,}")
                
                solved, value = self.check_start_solved()
                
                iter_time = time.time() - iter_start
                elapsed = time.time() - start_time
                print(f"  Solved: {len(self.solved):,}, Time: {iter_time:.1f}s, "
                      f"Elapsed: {elapsed:.1f}s")
                
                if solved:
                    self.lightning_time = elapsed
                    self.lightning_solution = value
                    self.metrics['lightning_time'] = elapsed
                    self.metrics['lightning_states'] = len(self.all_seen)
                    
                    outcome = {1: "X WINS", 0: "DRAW", -1: "O WINS"}[value]
                    print(f"\n{'='*60}")
                    print(f"⚡ LIGHTNING STRIKE: {outcome} ⚡")
                    print(f"  Time: {elapsed:.1f}s")
                    print(f"  States explored: {len(self.all_seen):,}")
                    print(f"{'='*60}")
                    
                    # Extract spine
                    spine = self.extract_spine()
                    print(f"\n  Crystallized spine: {len(spine)} states")
                    
                    self.phase = "crystallizing"
                    self.save()
                    break
                
                self.save()
        
        # ===== PHASE 2: CRYSTALLIZATION =====
        if self.phase == "crystallizing":
            print(f"\n{'='*60}")
            print("🔮 PHASE 2: CRYSTALLIZATION")
            print("Growing solved region from spine...")
            print(f"{'='*60}")
            
            crystal_start = time.time()
            
            crystal_solved = self.crystallize_from_spine()
            
            crystal_time = time.time() - crystal_start
            self.metrics['crystal_time'] = crystal_time
            self.metrics['crystal_states'] = len(self.solved) - self.metrics.get('lightning_states', 0)
            
            self.phase = "complete"
        
        # ===== FINAL STATUS =====
        total_time = time.time() - start_time
        self.metrics['total_time'] = total_time
        
        print(f"\n{'='*60}")
        print("CRYSTALLIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total states explored: {len(self.all_seen):,}")
        print(f"Total states solved: {len(self.solved):,}")
        print(f"Spine size: {len(self.crystal_spine):,}")
        print(f"\n⚡ Lightning time: {self.lightning_time:.1f}s")
        print(f"🔮 Crystal time: {self.metrics.get('crystal_time', 0):.1f}s")
        print(f"📊 Total time: {total_time:.1f}s")
        
        solved, value = self.check_start_solved()
        if solved:
            outcome = {1: "FIRST PLAYER (X) WINS", 0: "DRAW", -1: "SECOND PLAYER (O) WINS"}[value]
            print(f"\n🎉 CONNECT-4: {outcome} 🎉")
        
        # Show spine (principal variation)
        print(f"\n📍 Principal Variation (optimal play):")
        state = C4State()
        moves = []
        for i in range(20):  # Show up to 20 moves
            h = hash(state)
            if h not in self.children or h not in self.solved:
                break
            if state.is_terminal():
                break
            
            turn = self._get_turn(h)
            best_move = None
            best_value = None
            
            for ch in self.children[h]:
                if ch not in self.solved:
                    continue
                cv = self.solved[ch]
                if best_value is None or \
                   (turn == 'X' and cv > best_value) or \
                   (turn == 'O' and cv < best_value):
                    best_value = cv
                    # Find which column
                    for col in state.get_valid_moves():
                        if hash(state.play(col)) == ch:
                            best_move = col
                            break
            
            if best_move is None:
                break
            
            moves.append(best_move)
            state = state.play(best_move)
        
        print(f"  Moves: {moves}")
        
        self.save()
        return solved


if __name__ == "__main__":
    solver = CrystallineSolver()
    
    if solver.load():
        print(f"\nResuming from checkpoint...")
    else:
        print("\nStarting fresh...")
        solver.initialize()
    
    solver.solve(max_depth=42)
