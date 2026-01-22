"""
Connect-4 Uncapped Bidirectional Solver
Full forward exploration - no limits!

This version removes the 150K cap on forward expansion,
allowing us to explore all reachable states from the start.
"""

import pickle
import json
import os
import time
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass

SAVE_DIR = "./c4_uncapped_state"

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
    
    def unplay(self, col):
        h = self.height(col)
        if h == 0:
            return None
        piece = self.cols[col][h-1]
        new_col = self.cols[col][:h-1] + '.' + self.cols[col][h:]
        new_cols = list(self.cols)
        new_cols[col] = new_col
        return C4State(tuple(new_cols), piece)
    
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
        """Returns 'X', 'O', or None"""
        for col in range(7):
            for row in range(6):
                p = self.get(col, row)
                if p is None:
                    continue
                # Horizontal
                if col <= 3:
                    if all(self.get(col+i, row) == p for i in range(4)):
                        return p
                # Vertical
                if row <= 2:
                    if all(self.get(col, row+i) == p for i in range(4)):
                        return p
                # Diagonal up
                if col <= 3 and row <= 2:
                    if all(self.get(col+i, row+i) == p for i in range(4)):
                        return p
                # Diagonal down
                if col <= 3 and row >= 3:
                    if all(self.get(col+i, row-i) == p for i in range(4)):
                        return p
        return None
    
    def is_terminal(self):
        return self.check_win() is not None or self.piece_count() == 42
    
    def terminal_value(self):
        w = self.check_win()
        if w == 'X':
            return 1
        elif w == 'O':
            return -1
        else:
            return 0
    
    def get_features(self):
        heights = tuple(sorted(self.height(c) for c in range(7)))
        x_count = sum(c.count('X') for c in self.cols)
        o_count = sum(c.count('O') for c in self.cols)
        
        def count_threats(player):
            threats = 0
            for col in range(7):
                for row in range(6):
                    # Horizontal
                    if col <= 3:
                        window = [self.get(col+i, row) for i in range(4)]
                        if window.count(player) == 3 and window.count(None) == 1:
                            threats += 1
                    # Vertical
                    if row <= 2:
                        window = [self.get(col, row+i) for i in range(4)]
                        if window.count(player) == 3 and window.count(None) == 1:
                            threats += 1
                    # Diagonal up
                    if col <= 3 and row <= 2:
                        window = [self.get(col+i, row+i) for i in range(4)]
                        if window.count(player) == 3 and window.count(None) == 1:
                            threats += 1
                    # Diagonal down
                    if col <= 3 and row >= 3:
                        window = [self.get(col+i, row-i) for i in range(4)]
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


class UncappedSolver:
    """
    Bidirectional solver with NO CAPS on forward expansion.
    Explores ALL reachable positions.
    """
    
    def __init__(self):
        self.solved: Dict[int, int] = {}  # hash -> value
        self.forward_frontier: Dict[int, tuple] = {}  # hash -> compact state
        self.backward_frontier: Dict[int, tuple] = {}
        self.children: Dict[int, list] = {}  # hash -> [child hashes]
        self.equiv_classes: Dict[C4Features, Set[int]] = defaultdict(set)
        self.equiv_outcomes: Dict[C4Features, Optional[int]] = {}
        self.metrics = {
            'iterations': 0,
            'total_time': 0.0,
            'forward_expanded': 0,
            'backward_expanded': 0,
        }
    
    def initialize(self):
        """Start fresh from empty board"""
        start = C4State()
        h = hash(start)
        self.forward_frontier[h] = start.to_compact()
        self.equiv_classes[start.get_features()].add(h)
        print(f"Initialized with start position")
    
    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(f"{SAVE_DIR}/solved.pkl", 'wb') as f:
            pickle.dump(self.solved, f)
        with open(f"{SAVE_DIR}/forward.pkl", 'wb') as f:
            pickle.dump(self.forward_frontier, f)
        with open(f"{SAVE_DIR}/backward.pkl", 'wb') as f:
            pickle.dump(self.backward_frontier, f)
        with open(f"{SAVE_DIR}/children.pkl", 'wb') as f:
            pickle.dump(self.children, f)
        with open(f"{SAVE_DIR}/equiv.pkl", 'wb') as f:
            pickle.dump(self.equiv_classes, f)
        with open(f"{SAVE_DIR}/equiv_out.pkl", 'wb') as f:
            pickle.dump(self.equiv_outcomes, f)
        with open(f"{SAVE_DIR}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved: {len(self.solved)} solved, {len(self.forward_frontier)} fwd, {len(self.backward_frontier)} bwd")
    
    def load(self):
        if not os.path.exists(f"{SAVE_DIR}/solved.pkl"):
            return False
        try:
            with open(f"{SAVE_DIR}/solved.pkl", 'rb') as f:
                self.solved = pickle.load(f)
            with open(f"{SAVE_DIR}/forward.pkl", 'rb') as f:
                self.forward_frontier = pickle.load(f)
            with open(f"{SAVE_DIR}/backward.pkl", 'rb') as f:
                self.backward_frontier = pickle.load(f)
            with open(f"{SAVE_DIR}/children.pkl", 'rb') as f:
                self.children = pickle.load(f)
            with open(f"{SAVE_DIR}/equiv.pkl", 'rb') as f:
                self.equiv_classes = pickle.load(f)
            with open(f"{SAVE_DIR}/equiv_out.pkl", 'rb') as f:
                self.equiv_outcomes = pickle.load(f)
            with open(f"{SAVE_DIR}/metrics.json") as f:
                self.metrics = json.load(f)
            print(f"Loaded: {len(self.solved)} solved, {len(self.forward_frontier)} fwd, {len(self.backward_frontier)} bwd")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False
    
    def expand_forward_fully(self):
        """
        Expand forward frontier COMPLETELY - no limits!
        Returns when all reachable non-terminal states are explored.
        """
        iteration = 0
        total_new = 0
        total_terminals = 0
        
        while self.forward_frontier:
            iteration += 1
            new_states = {}
            terminals_found = 0
            to_remove = []
            
            batch = list(self.forward_frontier.items())
            
            for h, compact in batch:
                state = C4State.from_compact(compact)
                
                # Check terminal
                if state.is_terminal():
                    value = state.terminal_value()
                    self.solved[h] = value
                    terminals_found += 1
                    to_remove.append(h)
                    
                    features = state.get_features()
                    self._update_equiv_outcome(features, value)
                    
                    # Add to backward frontier for backprop
                    self.backward_frontier[h] = compact
                    continue
                
                # Generate children
                child_hashes = []
                for move in state.get_valid_moves():
                    child = state.play(move)
                    ch = hash(child)
                    child_hashes.append(ch)
                    
                    if ch not in self.solved and ch not in self.forward_frontier and ch not in new_states:
                        new_states[ch] = child.to_compact()
                        features = child.get_features()
                        self.equiv_classes[features].add(ch)
                
                self.children[h] = child_hashes
                to_remove.append(h)
            
            for h in to_remove:
                if h in self.forward_frontier:
                    del self.forward_frontier[h]
            
            self.forward_frontier.update(new_states)
            total_new += len(new_states)
            total_terminals += terminals_found
            
            print(f"  Forward iter {iteration}: expanded {len(to_remove)}, "
                  f"new {len(new_states)}, terminals {terminals_found}, "
                  f"frontier {len(self.forward_frontier)}, "
                  f"total_solved {len(self.solved)}")
            
            # Save periodically
            if iteration % 5 == 0:
                self.metrics['forward_expanded'] = total_new
                self.save()
        
        print(f"\nForward expansion COMPLETE!")
        print(f"  Total new states: {total_new}")
        print(f"  Total terminals: {total_terminals}")
        print(f"  Children tracked: {len(self.children)}")
        return total_new, total_terminals
    
    def _update_equiv_outcome(self, features, value):
        if features in self.equiv_outcomes:
            if self.equiv_outcomes[features] != value:
                self.equiv_outcomes[features] = None  # Mixed
        else:
            self.equiv_outcomes[features] = value
    
    def propagate_backward(self):
        """
        Propagate solved values backward through the game tree.
        Uses minimax: X maximizes, O minimizes.
        """
        iteration = 0
        total_propagated = 0
        
        changed = True
        while changed:
            iteration += 1
            changed = False
            propagated = 0
            
            # For each state with children, check if we can solve it
            for h, child_list in list(self.children.items()):
                if h in self.solved:
                    continue
                
                # Get values of all children
                child_values = []
                for ch in child_list:
                    if ch in self.solved:
                        child_values.append(self.solved[ch])
                
                if len(child_values) != len(child_list):
                    continue  # Not all children solved yet
                
                # Minimax
                # Determine whose turn it was at state h
                # The children are positions AFTER a move, so their turn tells us who moved
                # If children have turn 'O', then X just moved, so at h it was X's turn
                sample_child_compact = None
                for ch in child_list:
                    if ch in self.forward_frontier:
                        sample_child_compact = self.forward_frontier[ch]
                        break
                    elif ch in self.backward_frontier:
                        sample_child_compact = self.backward_frontier[ch]
                        break
                
                # Fallback: reconstruct from parent
                parent_compact = self.forward_frontier.get(h) or self.backward_frontier.get(h)
                if parent_compact:
                    parent = C4State.from_compact(parent_compact)
                    if parent.turn == 'X':
                        value = max(child_values)
                    else:
                        value = min(child_values)
                    
                    self.solved[h] = value
                    propagated += 1
                    changed = True
                    
                    features = parent.get_features()
                    self._update_equiv_outcome(features, value)
            
            # Also propagate via equivalence
            equiv_solved = self.propagate_equivalence()
            
            total_propagated += propagated
            
            if propagated > 0 or equiv_solved > 0:
                print(f"  Backward iter {iteration}: propagated {propagated}, "
                      f"equiv {equiv_solved}, total_solved {len(self.solved)}")
            
            # Check if start is solved
            start_h = hash(C4State())
            if start_h in self.solved:
                print(f"\n*** START POSITION SOLVED! ***")
                break
            
            # Save periodically
            if iteration % 10 == 0:
                self.save()
        
        return total_propagated
    
    def propagate_equivalence(self):
        """Propagate solutions via equivalence classes"""
        solved_count = 0
        
        for features, hashes in self.equiv_classes.items():
            if features not in self.equiv_outcomes:
                continue
            outcome = self.equiv_outcomes[features]
            if outcome is None:
                continue
            
            for h in hashes:
                if h not in self.solved:
                    self.solved[h] = outcome
                    solved_count += 1
        
        return solved_count
    
    def solve(self):
        """Main solving loop"""
        print("="*60)
        print("UNCAPPED CONNECT-4 SOLVER")
        print("="*60)
        
        start_time = time.time()
        
        # Phase 1: Full forward expansion
        print("\n" + "="*60)
        print("PHASE 1: Forward Expansion (no limits)")
        print("="*60)
        self.expand_forward_fully()
        
        # Phase 2: Backward propagation
        print("\n" + "="*60)
        print("PHASE 2: Backward Propagation")
        print("="*60)
        self.propagate_backward()
        
        elapsed = time.time() - start_time
        
        # Final status
        print("\n" + "="*60)
        print("FINAL STATUS")
        print("="*60)
        print(f"Total solved: {len(self.solved):,}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        start_h = hash(C4State())
        if start_h in self.solved:
            outcome = {1: "FIRST PLAYER (X) WINS", 0: "DRAW", -1: "SECOND PLAYER (O) WINS"}
            print(f"\n{'='*60}")
            print(f"🎉 CONNECT-4 SOLVED: {outcome[self.solved[start_h]]} 🎉")
            print(f"{'='*60}")
        else:
            print("\nStart position not yet solved!")
        
        self.save()
        return start_h in self.solved


if __name__ == "__main__":
    solver = UncappedSolver()
    
    if solver.load():
        print("\nResuming from saved state...")
    else:
        print("\nStarting fresh...")
        solver.initialize()
    
    solver.solve()
