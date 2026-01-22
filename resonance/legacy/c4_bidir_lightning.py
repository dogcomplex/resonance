"""
Connect-4 Bidirectional Lightning Solver

Inspired by:
- Slime molds solving mazes (grow from both food sources)
- Bidirectional lightning (upward + downward leaders)
- Axon guidance (growth cone meets target gradient)
- Quantum transactional interpretation (offer + confirmation waves)

The key insight: Nature doesn't just search from one end.
When there's "pressure" at BOTH boundaries, waves propagate
from both and MEET IN THE MIDDLE.

For game trees:
- Forward leader: explores from START (unknown → known)
- Backward leader: explores from TERMINALS (known → unknown)  
- CONTACT: when any forward state has all children solved
- This triggers cascading "crystallization" back to start

This should be MUCH faster than unidirectional search!
"""

import pickle
import json
import os
import time
import gc
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, Optional, List
from dataclasses import dataclass

SAVE_DIR = "./c4_bidir_lightning_state"
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
    
    def unplay(self, col):
        """Remove top piece (for backward exploration)"""
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


class BidirectionalLightningSolver:
    """
    Bidirectional wave solver.
    
    Two leaders explore simultaneously:
    - Forward leader: from START, expanding into unknown
    - Backward leader: from TERMINALS, propagating known values
    
    CONTACT occurs when forward-explored state becomes fully solvable
    (all its children are in the backward-solved set)
    
    This triggers crystallization cascade back to start.
    """
    
    def __init__(self):
        # Forward wave (from start)
        self.forward_frontier: Dict[int, tuple] = {}  # Current forward edge
        self.forward_seen: Set[int] = set()  # All forward-visited
        self.forward_depth = 0
        
        # Backward wave (from terminals)  
        self.backward_frontier: Dict[int, tuple] = {}  # Current backward edge
        self.backward_seen: Set[int] = set()  # All backward-visited
        self.backward_depth = 0
        
        # Solved states (value known)
        self.solved: Dict[int, int] = {}
        
        # Graph structure
        self.children: Dict[int, List[int]] = {}  # parent -> [children]
        self.parents: Dict[int, List[int]] = defaultdict(list)  # child -> [parents]
        self.state_turns: Dict[int, str] = {}  # hash -> turn ('X' or 'O')
        
        # Equivalence
        self.equiv_classes: Dict[C4Features, Set[int]] = defaultdict(set)
        self.equiv_outcomes: Dict[C4Features, Optional[int]] = {}
        
        # State
        self.phase = "exploring"  # "exploring", "crystallizing", "complete"
        self.contact_depth = None  # Depth where waves met
        self.contact_time = None
        
        self.metrics = {
            'total_time': 0.0,
            'contact_time': 0.0,
            'forward_states': 0,
            'backward_states': 0,
        }
    
    def initialize(self):
        """Set up initial conditions at both ends"""
        # Forward: start from empty board
        start = C4State()
        h = hash(start)
        self.forward_frontier[h] = start.to_compact()
        self.forward_seen.add(h)
        self.equiv_classes[start.get_features()].add(h)
        
        print(f"Initialized forward leader at START")
        print(f"Backward leader will grow from terminals as they're discovered")
    
    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        data = {
            'forward_frontier': self.forward_frontier,
            'forward_seen': self.forward_seen,
            'forward_depth': self.forward_depth,
            'backward_frontier': self.backward_frontier,
            'backward_seen': self.backward_seen,
            'backward_depth': self.backward_depth,
            'solved': self.solved,
            'children': self.children,
            'parents': dict(self.parents),
            'state_turns': self.state_turns,
            'equiv_classes': self.equiv_classes,
            'equiv_outcomes': self.equiv_outcomes,
            'phase': self.phase,
            'contact_depth': self.contact_depth,
            'contact_time': self.contact_time,
            'metrics': self.metrics,
        }
        
        with open(f"{SAVE_DIR}/state.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  [Saved: fwd_d={self.forward_depth}, bwd_d={self.backward_depth}, "
              f"solved={len(self.solved):,}, phase={self.phase}]")
    
    def load(self):
        if not os.path.exists(f"{SAVE_DIR}/state.pkl"):
            return False
        try:
            with open(f"{SAVE_DIR}/state.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.forward_frontier = data['forward_frontier']
            self.forward_seen = data['forward_seen']
            self.forward_depth = data['forward_depth']
            self.backward_frontier = data['backward_frontier']
            self.backward_seen = data['backward_seen']
            self.backward_depth = data['backward_depth']
            self.solved = data['solved']
            self.children = data['children']
            self.parents = defaultdict(list, data['parents'])
            self.state_turns = data.get('state_turns', {})
            self.equiv_classes = data['equiv_classes']
            self.equiv_outcomes = data['equiv_outcomes']
            self.phase = data['phase']
            self.contact_depth = data['contact_depth']
            self.contact_time = data['contact_time']
            self.metrics = data['metrics']
            
            print(f"Loaded: fwd_d={self.forward_depth}, bwd_d={self.backward_depth}, "
                  f"solved={len(self.solved):,}, phase={self.phase}")
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
    
    def expand_forward_layer(self):
        """Expand forward frontier by one depth layer"""
        items = list(self.forward_frontier.items())
        total = len(items)
        terminals = 0
        new_states = 0
        next_frontier = {}
        
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = items[batch_start:batch_end]
            
            for h, compact in batch:
                state = C4State.from_compact(compact)
                self.state_turns[h] = state.turn  # Store turn info
                
                # Terminal? Add to backward wave origin
                if state.is_terminal():
                    value = state.terminal_value()
                    self.solved[h] = value
                    self.backward_frontier[h] = compact
                    self.backward_seen.add(h)
                    terminals += 1
                    
                    features = state.get_features()
                    self._update_equiv_outcome(features, value)
                    continue
                
                # Generate children
                child_hashes = []
                moves = state.get_valid_moves()
                # Priority: center first
                moves.sort(key=lambda c: abs(c - 3))
                
                for move in moves:
                    child = state.play(move)
                    ch = hash(child)
                    child_hashes.append(ch)
                    self.parents[ch].append(h)
                    
                    if ch not in self.forward_seen:
                        self.forward_seen.add(ch)
                        next_frontier[ch] = child.to_compact()
                        self.state_turns[ch] = child.turn  # Store turn
                        new_states += 1
                        
                        features = child.get_features()
                        self.equiv_classes[features].add(ch)
                
                self.children[h] = child_hashes
            
            if batch_end < total and batch_end % BATCH_SIZE == 0:
                print(f"    Forward batch {batch_end}/{total}")
        
        self.forward_frontier = next_frontier
        self.forward_depth += 1
        self.metrics['forward_states'] = len(self.forward_seen)
        
        return total, terminals, new_states
    
    def expand_backward_layer(self):
        """
        Expand backward frontier by propagating solutions.
        
        Two mechanisms (like slime mold):
        1. PROPAGATION: Parent solved when ALL children solved (minimax)
        2. EARLY TERMINATION: Parent solved when ANY child guarantees outcome
           - X's turn + any child=+1 → parent=+1 (X found a win)
           - O's turn + any child=-1 → parent=-1 (O found a win)
        
        Runs until no more progress (full cascade).
        """
        newly_solved = 0
        total_iterations = 0
        
        # Keep propagating until stable
        changed = True
        while changed and total_iterations < 500:
            changed = False
            total_iterations += 1
            round_solved = 0
            
            # Check ALL states with children (not just backward frontier)
            for ph in list(self.children.keys()):
                if ph in self.solved:
                    continue
                
                child_list = self.children[ph]
                
                # Gather what we know about children
                child_values = []
                unknown_children = 0
                for ch in child_list:
                    if ch in self.solved:
                        child_values.append(self.solved[ch])
                    else:
                        unknown_children += 1
                
                if not child_values:
                    continue  # No info yet
                
                # Determine parent's turn
                turn = self._get_turn_for_state(ph)
                
                # === EARLY TERMINATION (alpha-beta style) ===
                if turn == 'X':
                    # X is maximizing - if ANY child is +1, X wins
                    if 1 in child_values:
                        self.solved[ph] = 1
                        self.backward_seen.add(ph)
                        newly_solved += 1
                        round_solved += 1
                        changed = True
                        continue
                    # If all children known and no +1, take max
                    if unknown_children == 0:
                        self.solved[ph] = max(child_values)
                        self.backward_seen.add(ph)
                        newly_solved += 1
                        round_solved += 1
                        changed = True
                else:
                    # O is minimizing - if ANY child is -1, O wins
                    if -1 in child_values:
                        self.solved[ph] = -1
                        self.backward_seen.add(ph)
                        newly_solved += 1
                        round_solved += 1
                        changed = True
                        continue
                    # If all children known and no -1, take min
                    if unknown_children == 0:
                        self.solved[ph] = min(child_values)
                        self.backward_seen.add(ph)
                        newly_solved += 1
                        round_solved += 1
                        changed = True
            
            if total_iterations % 50 == 0 and round_solved > 0:
                print(f"    Backward cascade iter {total_iterations}: +{round_solved} this round")
        
        if total_iterations > 1:
            print(f"    Backward cascade: {total_iterations} iterations, {newly_solved} total solved")
        
        self.backward_depth += 1
        self.metrics['backward_states'] = len(self.backward_seen)
        
        return newly_solved
    
    def _get_turn_for_state(self, h):
        """Determine whose turn it is at state h"""
        # Check stored turn info first
        if h in self.state_turns:
            return self.state_turns[h]
        
        # Try to find compact state
        for source in [self.forward_frontier, self.backward_frontier]:
            if h in source:
                state = C4State.from_compact(source[h])
                self.state_turns[h] = state.turn  # Cache it
                return state.turn
        
        # Infer from children's turn (children have opposite turn)
        if h in self.children:
            for ch in self.children[h]:
                if ch in self.state_turns:
                    child_turn = self.state_turns[ch]
                    parent_turn = 'O' if child_turn == 'X' else 'X'
                    self.state_turns[h] = parent_turn  # Cache it
                    return parent_turn
        
        return 'X'  # Default fallback
    
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
    
    def check_start_solved(self):
        start_h = hash(C4State())
        if start_h in self.solved:
            return True, self.solved[start_h]
        return False, None
    
    def check_contact(self):
        """Check if forward and backward waves have made contact"""
        # Contact = any forward-explored state is now fully solved
        # This happens when backward wave reaches forward frontier
        contact_count = len(self.forward_seen & self.backward_seen)
        return contact_count
    
    def solve(self, max_depth=42):
        """
        Main solving loop - bidirectional wave expansion.
        
        Each iteration:
        1. Expand forward frontier (discover new states)
        2. Expand backward frontier (propagate solutions)
        3. Check for contact (waves meeting)
        4. Check if start is solved
        """
        print("="*60)
        print("⚡⚡ BIDIRECTIONAL LIGHTNING SOLVER ⚡⚡")
        print("Forward leader: START → unknown")
        print("Backward leader: TERMINALS → unknown")
        print("Contact: when waves meet!")
        print("="*60)
        
        start_time = time.time()
        
        while self.forward_depth <= max_depth:
            iter_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"ITERATION: Forward depth {self.forward_depth}, Backward depth {self.backward_depth}")
            print(f"{'='*60}")
            
            # === FORWARD EXPANSION ===
            print(f"\n⚡ Forward leader expanding...")
            fwd_total, fwd_term, fwd_new = self.expand_forward_layer()
            print(f"  Expanded {fwd_total:,}, found {fwd_term:,} terminals, {fwd_new:,} new states")
            print(f"  Forward frontier: {len(self.forward_frontier):,}")
            
            # === BACKWARD EXPANSION ===
            print(f"\n⚡ Backward leader propagating...")
            bwd_solved = self.expand_backward_layer()
            print(f"  Propagated to {bwd_solved:,} new solved states")
            
            # === EQUIVALENCE ===
            equiv = self.propagate_equivalence()
            if equiv > 0:
                print(f"  Equivalence: +{equiv:,} solved")
            
            # === CONTACT CHECK ===
            contact = self.check_contact()
            if contact > 0 and self.contact_time is None:
                self.contact_time = time.time() - start_time
                self.contact_depth = (self.forward_depth, self.backward_depth)
                print(f"\n💥 CONTACT! Waves meeting at {contact:,} states!")
                print(f"  Time to contact: {self.contact_time:.1f}s")
            
            # === START CHECK ===
            solved, value = self.check_start_solved()
            
            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            
            print(f"\n📊 Status:")
            print(f"  Forward seen: {len(self.forward_seen):,}")
            print(f"  Backward seen: {len(self.backward_seen):,}")
            print(f"  Total solved: {len(self.solved):,}")
            print(f"  Iteration time: {iter_time:.1f}s")
            print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            
            if solved:
                self.phase = "complete"
                outcome = {1: "X WINS", 0: "DRAW", -1: "O WINS"}[value]
                
                print(f"\n{'='*60}")
                print(f"💥⚡ CRYSTALLIZATION COMPLETE: {outcome} ⚡💥")
                print(f"{'='*60}")
                print(f"  Contact time: {self.contact_time:.1f}s")
                print(f"  Total time: {elapsed:.1f}s")
                print(f"  Forward depth reached: {self.forward_depth}")
                print(f"  States explored: {len(self.forward_seen):,}")
                print(f"  States solved: {len(self.solved):,}")
                
                self.metrics['total_time'] = elapsed
                self.metrics['contact_time'] = self.contact_time
                self.save()
                return True
            
            # Check if stuck
            if len(self.forward_frontier) == 0:
                print("\nForward frontier exhausted!")
                break
            
            # Save checkpoint
            self.save()
        
        # Final status
        print(f"\n{'='*60}")
        print("EXPLORATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total states: {len(self.forward_seen):,}")
        print(f"Total solved: {len(self.solved):,}")
        
        solved, value = self.check_start_solved()
        if solved:
            outcome = {1: "X WINS", 0: "DRAW", -1: "O WINS"}[value]
            print(f"Result: {outcome}")
        else:
            print("Start position NOT solved")
        
        self.save()
        return solved


if __name__ == "__main__":
    solver = BidirectionalLightningSolver()
    
    if solver.load():
        print("\nResuming from checkpoint...")
    else:
        print("\nStarting fresh...")
        solver.initialize()
    
    solver.solve(max_depth=42)
