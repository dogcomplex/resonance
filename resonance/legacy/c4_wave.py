"""
Connect-4 Wave Solver - Iterative Deepening Bidirectional

This mimics natural wave physics:
- Forward wave expands one depth at a time (like a ripple)
- Backward wave propagates after each layer (reflection/interference)
- Standing wave emerges where they meet

Memory-bounded via batching within each layer.
"""

import pickle
import json
import os
import time
import gc
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass

SAVE_DIR = "./c4_wave_state"
BATCH_SIZE = 500000  # Process this many states at a time

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


class WaveSolver:
    """
    Iterative deepening bidirectional solver.
    Mimics wave physics: expand one layer, then propagate backward.
    """
    
    def __init__(self):
        self.solved: Dict[int, int] = {}
        self.current_layer: Dict[int, tuple] = {}  # Current depth frontier
        self.next_layer: Dict[int, tuple] = {}     # Next depth (children)
        self.all_seen: Set[int] = set()            # All visited hashes
        self.children: Dict[int, list] = {}
        self.parents: Dict[int, list] = defaultdict(list)  # child -> [parent hashes]
        self.equiv_classes: Dict[C4Features, Set[int]] = defaultdict(set)
        self.equiv_outcomes: Dict[C4Features, Optional[int]] = {}
        self.depth = 0
        self.metrics = {
            'total_time': 0.0,
            'states_by_depth': {},
            'solved_by_depth': {},
        }
    
    def initialize(self):
        start = C4State()
        h = hash(start)
        self.current_layer[h] = start.to_compact()
        self.all_seen.add(h)
        self.equiv_classes[start.get_features()].add(h)
        print(f"Initialized with start position (depth 0)")
    
    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(f"{SAVE_DIR}/solved.pkl", 'wb') as f:
            pickle.dump(self.solved, f)
        with open(f"{SAVE_DIR}/current.pkl", 'wb') as f:
            pickle.dump(self.current_layer, f)
        with open(f"{SAVE_DIR}/next.pkl", 'wb') as f:
            pickle.dump(self.next_layer, f)
        with open(f"{SAVE_DIR}/seen.pkl", 'wb') as f:
            pickle.dump(self.all_seen, f)
        with open(f"{SAVE_DIR}/children.pkl", 'wb') as f:
            pickle.dump(self.children, f)
        with open(f"{SAVE_DIR}/parents.pkl", 'wb') as f:
            pickle.dump(dict(self.parents), f)
        with open(f"{SAVE_DIR}/equiv.pkl", 'wb') as f:
            pickle.dump(self.equiv_classes, f)
        with open(f"{SAVE_DIR}/equiv_out.pkl", 'wb') as f:
            pickle.dump(self.equiv_outcomes, f)
        with open(f"{SAVE_DIR}/depth.txt", 'w') as f:
            f.write(str(self.depth))
        with open(f"{SAVE_DIR}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"  [Saved: depth={self.depth}, solved={len(self.solved):,}, seen={len(self.all_seen):,}]")
    
    def load(self):
        if not os.path.exists(f"{SAVE_DIR}/solved.pkl"):
            return False
        try:
            with open(f"{SAVE_DIR}/solved.pkl", 'rb') as f:
                self.solved = pickle.load(f)
            with open(f"{SAVE_DIR}/current.pkl", 'rb') as f:
                self.current_layer = pickle.load(f)
            with open(f"{SAVE_DIR}/next.pkl", 'rb') as f:
                self.next_layer = pickle.load(f)
            with open(f"{SAVE_DIR}/seen.pkl", 'rb') as f:
                self.all_seen = pickle.load(f)
            with open(f"{SAVE_DIR}/children.pkl", 'rb') as f:
                self.children = pickle.load(f)
            with open(f"{SAVE_DIR}/parents.pkl", 'rb') as f:
                self.parents = defaultdict(list, pickle.load(f))
            with open(f"{SAVE_DIR}/equiv.pkl", 'rb') as f:
                self.equiv_classes = pickle.load(f)
            with open(f"{SAVE_DIR}/equiv_out.pkl", 'rb') as f:
                self.equiv_outcomes = pickle.load(f)
            with open(f"{SAVE_DIR}/depth.txt") as f:
                self.depth = int(f.read().strip())
            with open(f"{SAVE_DIR}/metrics.json") as f:
                self.metrics = json.load(f)
            print(f"Loaded: depth={self.depth}, solved={len(self.solved):,}, seen={len(self.all_seen):,}")
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
    
    def expand_one_layer(self):
        """
        Expand current layer to next layer (one depth increment).
        Processes in batches to limit memory.
        """
        print(f"\n--- DEPTH {self.depth} -> {self.depth + 1} ---")
        
        items = list(self.current_layer.items())
        total = len(items)
        terminals = 0
        new_children = 0
        
        # Process in batches
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = items[batch_start:batch_end]
            
            for h, compact in batch:
                state = C4State.from_compact(compact)
                
                # Check terminal
                if state.is_terminal():
                    value = state.terminal_value()
                    self.solved[h] = value
                    terminals += 1
                    features = state.get_features()
                    self._update_equiv_outcome(features, value)
                    continue
                
                # Generate children
                child_hashes = []
                for move in state.get_valid_moves():
                    child = state.play(move)
                    ch = hash(child)
                    child_hashes.append(ch)
                    
                    # Track parent relationship
                    self.parents[ch].append(h)
                    
                    if ch not in self.all_seen:
                        self.all_seen.add(ch)
                        self.next_layer[ch] = child.to_compact()
                        new_children += 1
                        
                        features = child.get_features()
                        self.equiv_classes[features].add(ch)
                
                self.children[h] = child_hashes
            
            if batch_end < total:
                print(f"  Batch {batch_end}/{total} ({100*batch_end/total:.0f}%)")
        
        # Update metrics
        self.metrics['states_by_depth'][str(self.depth)] = total
        
        print(f"  Layer {self.depth}: {total:,} states, {terminals:,} terminals, {new_children:,} new children")
        
        # Move to next layer
        self.current_layer = self.next_layer
        self.next_layer = {}
        self.depth += 1
        
        gc.collect()
        
        return total, terminals, new_children
    
    def propagate_backward_one_pass(self):
        """
        One pass of backward propagation through the game tree.
        Returns number of newly solved states.
        """
        newly_solved = 0
        
        # Process states that have children
        for h, child_list in self.children.items():
            if h in self.solved:
                continue
            
            # Check if all children are solved
            child_values = []
            all_solved = True
            for ch in child_list:
                if ch in self.solved:
                    child_values.append(self.solved[ch])
                else:
                    all_solved = False
                    break
            
            if not all_solved:
                continue
            
            # Determine turn at state h
            # Look up the compact state
            compact = None
            for layer in [self.current_layer]:
                if h in layer:
                    compact = layer[h]
                    break
            
            # If not in current layer, we need to track it differently
            # The turn alternates: even depth = X, odd depth = O (approximately)
            # Better: check parent count vs child turn
            if compact:
                state = C4State.from_compact(compact)
                turn = state.turn
            else:
                # Infer from children - if children have turn O, parent had turn X
                # This is a bit hacky, try to find any child's compact
                turn = 'X'  # Default, will be overwritten
                for ch in child_list:
                    if ch in self.current_layer:
                        child_state = C4State.from_compact(self.current_layer[ch])
                        turn = 'O' if child_state.turn == 'X' else 'X'
                        break
            
            # Minimax
            if turn == 'X':
                value = max(child_values)
            else:
                value = min(child_values)
            
            self.solved[h] = value
            newly_solved += 1
        
        return newly_solved
    
    def propagate_equivalence(self):
        """Propagate via equivalence classes"""
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
    
    def backward_until_stable(self, max_iters=100):
        """Run backward propagation until no more progress"""
        print(f"\n  Backward propagation...")
        total = 0
        for i in range(max_iters):
            prop = self.propagate_backward_one_pass()
            equiv = self.propagate_equivalence()
            total += prop + equiv
            if prop == 0 and equiv == 0:
                break
            if (i + 1) % 10 == 0:
                print(f"    Iter {i+1}: propagated {prop}, equiv {equiv}, total_solved {len(self.solved):,}")
        
        print(f"  Backward done: {total:,} newly solved, total {len(self.solved):,}")
        return total
    
    def check_start_solved(self):
        """Check if start position is solved"""
        start_h = hash(C4State())
        if start_h in self.solved:
            return True, self.solved[start_h]
        return False, None
    
    def solve(self, max_depth=42):
        """Main solving loop - iterative deepening with backward propagation"""
        print("="*60)
        print("CONNECT-4 WAVE SOLVER")
        print("Iterative deepening with backward interference")
        print("="*60)
        
        start_time = time.time()
        
        while self.depth <= max_depth and self.current_layer:
            layer_start = time.time()
            
            # Expand one depth layer
            total, terminals, new_children = self.expand_one_layer()
            
            # Propagate backward (let the waves interfere)
            self.backward_until_stable()
            
            # Check if start is solved
            solved, value = self.check_start_solved()
            
            layer_time = time.time() - layer_start
            self.metrics['total_time'] += layer_time
            self.metrics['solved_by_depth'][str(self.depth - 1)] = len(self.solved)
            
            print(f"  Time: {layer_time:.1f}s, Total solved: {len(self.solved):,}")
            
            if solved:
                outcome = {1: "X WINS", 0: "DRAW", -1: "O WINS"}[value]
                print(f"\n{'='*60}")
                print(f"🎉 SOLVED AT DEPTH {self.depth}: {outcome} 🎉")
                print(f"{'='*60}")
                self.save()
                return True
            
            # Save after each depth
            self.save()
            
            # Progress report
            elapsed = time.time() - start_time
            print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"  Next layer size: {len(self.current_layer):,}")
        
        print(f"\nReached max depth or empty frontier")
        print(f"Total solved: {len(self.solved):,}")
        
        # Final backward propagation
        print("\nFinal backward propagation...")
        for _ in range(10):
            prop = self.backward_until_stable()
            if prop == 0:
                break
        
        solved, value = self.check_start_solved()
        if solved:
            outcome = {1: "X WINS", 0: "DRAW", -1: "O WINS"}[value]
            print(f"\n🎉 SOLVED: {outcome} 🎉")
        else:
            print(f"\nStart position not solved :(")
        
        self.save()
        return solved


if __name__ == "__main__":
    solver = WaveSolver()
    
    if solver.load():
        print("\nResuming from saved state...")
    else:
        print("\nStarting fresh...")
        solver.initialize()
    
    solver.solve(max_depth=42)
