"""
Bidirectional Wave Solver with Equivalence Classes

Key innovations:
1. TRUE bidirectional: forward from start AND backward from wins
2. Equivalence classes: reduce state space by grouping similar positions
3. Memory efficiency: discard interior states, keep only frontiers
4. Standing wave: where waves meet = solved region = new gravity

Connect-4 equivalence features:
- Material: X count, O count
- Column heights profile
- Threats: 3-in-a-row with empty 4th
- Connectivity patterns
"""

import pickle
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

import sys

# Cross-platform save directory
if sys.platform == "win32":
    SAVE_DIR = "./c4_bidir_state"
else:
    SAVE_DIR = "/home/claude/c4_bidir_state"

@dataclass(frozen=True)
class C4Features:
    """Equivalence class features for Connect-4"""
    x_count: int
    o_count: int
    x_threats: int  # 3-in-a-row with empty
    o_threats: int
    height_profile: Tuple[int, ...]  # Sorted column heights
    turn: str
    
    def __hash__(self):
        return hash((self.x_count, self.o_count, self.x_threats, 
                     self.o_threats, self.height_profile, self.turn))


class C4State:
    """Compact Connect-4 state"""
    
    __slots__ = ['cols', 'turn', '_hash', '_features']
    
    def __init__(self, cols: Tuple[str, ...] = None, turn: str = 'X'):
        if cols is None:
            cols = tuple('.' * 6 for _ in range(7))
        self.cols = cols
        self.turn = turn
        self._hash = None
        self._features = None
    
    def __hash__(self):
        if self._hash is None:
            mirror = tuple(reversed(self.cols))
            self._hash = hash((min(self.cols, mirror), self.turn))
        return self._hash
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def height(self, col: int) -> int:
        return 6 - self.cols[col].count('.')
    
    def can_play(self, col: int) -> bool:
        return self.cols[col][5] == '.'
    
    def play(self, col: int) -> Optional['C4State']:
        if not self.can_play(col):
            return None
        h = self.height(col)
        new_col = self.cols[col][:h] + self.turn + self.cols[col][h+1:]
        new_cols = list(self.cols)
        new_cols[col] = new_col
        return C4State(tuple(new_cols), 'O' if self.turn == 'X' else 'X')
    
    def unplay(self, col: int) -> Optional['C4State']:
        """Remove top piece from column (for backward expansion)"""
        h = self.height(col)
        if h == 0:
            return None
        
        piece = self.cols[col][h-1]
        new_col = self.cols[col][:h-1] + '.' + self.cols[col][h:]
        new_cols = list(self.cols)
        new_cols[col] = new_col
        
        # Turn becomes the piece we removed
        return C4State(tuple(new_cols), piece)
    
    def get(self, col: int, row: int) -> Optional[str]:
        if 0 <= col < 7 and 0 <= row < 6:
            return self.cols[col][row]
        return None
    
    def check_winner(self) -> Optional[str]:
        for col in range(7):
            for row in range(6):
                piece = self.get(col, row)
                if piece == '.':
                    continue
                for dc, dr in [(1,0), (0,1), (1,1), (1,-1)]:
                    if all(self.get(col+dc*i, row+dr*i) == piece for i in range(4)):
                        return piece
        return None
    
    def is_full(self) -> bool:
        return all(self.cols[c][5] != '.' for c in range(7))
    
    def get_terminal(self) -> Optional[str]:
        winner = self.check_winner()
        if winner == 'X': return 'W'
        if winner == 'O': return 'L'
        if self.is_full(): return 'T'
        return None
    
    def get_valid_moves(self) -> List[int]:
        return [c for c in range(7) if self.can_play(c)]
    
    def piece_count(self) -> int:
        return sum(self.height(c) for c in range(7))
    
    def get_features(self) -> C4Features:
        """Extract equivalence class features"""
        if self._features:
            return self._features
        
        x_count = sum(c.count('X') for c in self.cols)
        o_count = sum(c.count('O') for c in self.cols)
        
        # Count threats
        x_threats = o_threats = 0
        for col in range(7):
            for row in range(6):
                for dc, dr in [(1,0), (0,1), (1,1), (1,-1)]:
                    line = [self.get(col+dc*i, row+dr*i) for i in range(4)]
                    if None in line:
                        continue
                    if line.count('X') == 3 and line.count('.') == 1:
                        x_threats += 1
                    if line.count('O') == 3 and line.count('.') == 1:
                        o_threats += 1
        
        # Height profile (sorted for symmetry)
        heights = tuple(sorted(self.height(c) for c in range(7)))
        
        self._features = C4Features(x_count, o_count, x_threats, o_threats, heights, self.turn)
        return self._features
    
    def to_compact(self) -> bytes:
        """Compact serialization"""
        return (self.cols, self.turn).__repr__().encode()
    
    @classmethod
    def from_compact(cls, data: bytes) -> 'C4State':
        cols, turn = eval(data.decode())
        return cls(cols, turn)
    
    def display(self) -> str:
        lines = []
        for row in range(5, -1, -1):
            line = '|' + '|'.join(self.cols[col][row] for col in range(7)) + '|'
            lines.append(line)
        lines.append('+' + '-' * 13 + '+')
        return '\n'.join(lines)


class BidirectionalSolver:
    """
    True bidirectional wave solver.
    
    Forward wave: expands from start
    Backward wave: expands from terminal wins
    Standing wave: where they meet
    
    Memory efficient: only keep frontiers, discard interiors
    """
    
    def __init__(self):
        # Solved states: hash -> value (1=X wins, 0=tie, -1=O wins)
        self.solved: Dict[int, int] = {}
        
        # Forward frontier: states reachable from start, not yet fully expanded
        self.forward_frontier: Dict[int, bytes] = {}  # hash -> compact state
        
        # Backward frontier: states that can reach a terminal
        self.backward_frontier: Dict[int, bytes] = {}
        
        # Standing wave: states in BOTH frontiers (the meeting point)
        self.standing_wave: Set[int] = set()
        
        # Equivalence classes: features -> {hashes with same features}
        self.equiv_classes: Dict[C4Features, Set[int]] = defaultdict(set)
        
        # Equivalence outcomes: features -> outcome (if all states in class have same outcome)
        self.equiv_outcomes: Dict[C4Features, int] = {}
        
        # Children map (sparse - only for frontier states)
        self.children: Dict[int, List[int]] = {}
        
        # Metrics
        self.metrics = {
            'forward_expanded': 0,
            'backward_expanded': 0,
            'standing_wave_size': 0,
            'equiv_classes_with_outcome': 0,
            'solved_by_equivalence': 0,
            'total_time': 0,
            'iterations': 0
        }
    
    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        with open(f"{SAVE_DIR}/solved.pkl", 'wb') as f:
            pickle.dump(self.solved, f)
        with open(f"{SAVE_DIR}/forward.pkl", 'wb') as f:
            pickle.dump(self.forward_frontier, f)
        with open(f"{SAVE_DIR}/backward.pkl", 'wb') as f:
            pickle.dump(self.backward_frontier, f)
        with open(f"{SAVE_DIR}/standing.pkl", 'wb') as f:
            pickle.dump(self.standing_wave, f)
        with open(f"{SAVE_DIR}/equiv.pkl", 'wb') as f:
            pickle.dump(dict(self.equiv_classes), f)
        with open(f"{SAVE_DIR}/equiv_out.pkl", 'wb') as f:
            pickle.dump(self.equiv_outcomes, f)
        with open(f"{SAVE_DIR}/children.pkl", 'wb') as f:
            pickle.dump(self.children, f)
        with open(f"{SAVE_DIR}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Saved: {len(self.solved)} solved, {len(self.forward_frontier)} fwd, "
              f"{len(self.backward_frontier)} bwd, {len(self.standing_wave)} standing")
    
    def load(self) -> bool:
        if not os.path.exists(f"{SAVE_DIR}/solved.pkl"):
            return False
        try:
            with open(f"{SAVE_DIR}/solved.pkl", 'rb') as f:
                self.solved = pickle.load(f)
            with open(f"{SAVE_DIR}/forward.pkl", 'rb') as f:
                self.forward_frontier = pickle.load(f)
            with open(f"{SAVE_DIR}/backward.pkl", 'rb') as f:
                self.backward_frontier = pickle.load(f)
            with open(f"{SAVE_DIR}/standing.pkl", 'rb') as f:
                self.standing_wave = pickle.load(f)
            with open(f"{SAVE_DIR}/equiv.pkl", 'rb') as f:
                self.equiv_classes = defaultdict(set, pickle.load(f))
            with open(f"{SAVE_DIR}/equiv_out.pkl", 'rb') as f:
                self.equiv_outcomes = pickle.load(f)
            with open(f"{SAVE_DIR}/children.pkl", 'rb') as f:
                self.children = pickle.load(f)
            with open(f"{SAVE_DIR}/metrics.json", 'r') as f:
                self.metrics = json.load(f)
            print(f"Loaded: {len(self.solved)} solved, {len(self.forward_frontier)} fwd, "
                  f"{len(self.backward_frontier)} bwd, {len(self.standing_wave)} standing")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False
    
    def initialize(self):
        start = C4State()
        h = hash(start)
        self.forward_frontier[h] = start.to_compact()
        
        features = start.get_features()
        self.equiv_classes[features].add(h)
        
        print("Initialized with empty board in forward frontier")
    
    def expand_forward(self, max_new: int = 100000) -> dict:
        """Expand forward frontier by one layer"""
        new_states = {}
        terminals_found = 0
        
        to_remove = []
        
        for h, compact in list(self.forward_frontier.items()):
            if h in self.solved:
                to_remove.append(h)
                continue
            
            state = C4State.from_compact(compact)
            
            # Check terminal
            terminal = state.get_terminal()
            if terminal:
                value = {'W': 1, 'T': 0, 'L': -1}[terminal]
                self.solved[h] = value
                self.backward_frontier[h] = compact  # Terminals start backward wave
                terminals_found += 1
                to_remove.append(h)
                
                # Update equivalence
                features = state.get_features()
                self._update_equiv_outcome(features, value)
                continue
            
            # Generate children
            child_hashes = []
            for move in state.get_valid_moves():
                child = state.play(move)
                ch = hash(child)
                child_hashes.append(ch)
                
                if ch not in self.solved and ch not in self.forward_frontier and ch not in new_states:
                    if len(new_states) < max_new:
                        new_states[ch] = child.to_compact()
                        
                        # Track equivalence class
                        features = child.get_features()
                        self.equiv_classes[features].add(ch)
            
            self.children[h] = child_hashes
            to_remove.append(h)  # Fully expanded, remove from frontier
        
        # Remove expanded states from frontier
        for h in to_remove:
            if h in self.forward_frontier:
                del self.forward_frontier[h]
        
        # Add new states to frontier
        self.forward_frontier.update(new_states)
        
        self.metrics['forward_expanded'] += len(to_remove)
        
        return {
            'expanded': len(to_remove),
            'terminals': terminals_found,
            'new_states': len(new_states),
            'frontier_size': len(self.forward_frontier)
        }
    
    def expand_backward(self, max_new: int = 100000) -> dict:
        """
        Expand backward frontier.
        For each state in backward frontier, find its PARENTS (states that lead to it).
        """
        new_states = {}
        propagated = 0
        
        to_process = list(self.backward_frontier.items())
        
        for h, compact in to_process:
            if h not in self.solved:
                continue
            
            state = C4State.from_compact(compact)
            child_value = self.solved[h]
            
            # Find parent states by "unplaying" each column
            for col in range(7):
                parent = state.unplay(col)
                if parent is None:
                    continue
                
                ph = hash(parent)
                
                # If parent already solved, skip
                if ph in self.solved:
                    continue
                
                # Add to backward frontier
                if ph not in self.backward_frontier and ph not in new_states:
                    if len(new_states) < max_new:
                        new_states[ph] = parent.to_compact()
                        
                        features = parent.get_features()
                        self.equiv_classes[features].add(ph)
                
                # Try to solve parent via minimax
                if ph in self.children:
                    child_values = []
                    for ch in self.children[ph]:
                        if ch in self.solved:
                            child_values.append(self.solved[ch])
                    
                    if len(child_values) == len(self.children[ph]) and child_values:
                        # All children solved - can solve parent
                        parent_state = C4State.from_compact(
                            self.forward_frontier.get(ph) or new_states.get(ph) or 
                            self.backward_frontier.get(ph, parent.to_compact())
                        )
                        if parent_state.turn == 'X':
                            value = max(child_values)
                        else:
                            value = min(child_values)
                        
                        self.solved[ph] = value
                        propagated += 1
                        
                        features = parent_state.get_features()
                        self._update_equiv_outcome(features, value)
        
        self.backward_frontier.update(new_states)
        self.metrics['backward_expanded'] += len(new_states)
        
        return {
            'new_states': len(new_states),
            'propagated': propagated,
            'frontier_size': len(self.backward_frontier)
        }
    
    def compute_standing_wave(self) -> int:
        """Find states in BOTH forward and backward frontiers"""
        fwd_set = set(self.forward_frontier.keys())
        bwd_set = set(self.backward_frontier.keys())
        
        # Also include states that have been fully processed (in children map)
        processed = set(self.children.keys())
        
        self.standing_wave = (fwd_set | processed) & bwd_set
        self.metrics['standing_wave_size'] = len(self.standing_wave)
        
        return len(self.standing_wave)
    
    def _update_equiv_outcome(self, features: C4Features, value: int):
        """Update equivalence class outcome tracking"""
        if features in self.equiv_outcomes:
            if self.equiv_outcomes[features] != value:
                # Mixed outcomes - not a pure equivalence class
                self.equiv_outcomes[features] = None
        else:
            self.equiv_outcomes[features] = value
    
    def propagate_equivalence(self) -> int:
        """Use equivalence classes to solve more states"""
        solved_by_equiv = 0
        
        for features, hashes in self.equiv_classes.items():
            if features not in self.equiv_outcomes:
                continue
            
            outcome = self.equiv_outcomes[features]
            if outcome is None:
                continue  # Mixed class
            
            for h in hashes:
                if h not in self.solved:
                    self.solved[h] = outcome
                    solved_by_equiv += 1
        
        self.metrics['solved_by_equivalence'] += solved_by_equiv
        self.metrics['equiv_classes_with_outcome'] = sum(
            1 for v in self.equiv_outcomes.values() if v is not None
        )
        
        return solved_by_equiv
    
    def prune_interior(self) -> int:
        """
        Remove states that are fully interior (not on any frontier).
        Keep only: solved value and hash, discard full state representation.
        """
        pruned = 0
        
        # States to keep full representation for:
        # - Forward frontier
        # - Backward frontier
        # - Standing wave
        
        keep_full = set(self.forward_frontier.keys()) | set(self.backward_frontier.keys()) | self.standing_wave
        
        # Prune children map for fully solved branches
        to_prune = []
        for h in self.children:
            if h in self.solved and h not in keep_full:
                # Check if all children are solved
                if all(ch in self.solved for ch in self.children[h]):
                    to_prune.append(h)
        
        for h in to_prune:
            del self.children[h]
            pruned += 1
        
        return pruned
    
    def run_one_iteration(self, max_states: int = 100000) -> dict:
        """Run one complete iteration of bidirectional solving"""
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"ITERATION {self.metrics['iterations'] + 1}")
        print(f"{'='*60}")
        
        # 1. Expand forward
        print("\n1. Forward expansion...")
        fwd_result = self.expand_forward(max_states)
        print(f"   Expanded: {fwd_result['expanded']}, Terminals: {fwd_result['terminals']}")
        print(f"   New states: {fwd_result['new_states']}, Frontier: {fwd_result['frontier_size']}")
        
        # 2. Expand backward
        print("\n2. Backward expansion...")
        bwd_result = self.expand_backward(max_states)
        print(f"   New states: {bwd_result['new_states']}, Propagated: {bwd_result['propagated']}")
        print(f"   Frontier: {bwd_result['frontier_size']}")
        
        # 3. Compute standing wave
        print("\n3. Standing wave...")
        standing = self.compute_standing_wave()
        print(f"   Standing wave size: {standing}")
        
        # 4. Propagate via equivalence
        print("\n4. Equivalence propagation...")
        equiv_solved = self.propagate_equivalence()
        print(f"   Solved by equivalence: {equiv_solved}")
        print(f"   Pure equiv classes: {self.metrics['equiv_classes_with_outcome']}")
        
        # 5. Prune interior states
        print("\n5. Pruning interior states...")
        pruned = self.prune_interior()
        print(f"   Pruned: {pruned}")
        
        elapsed = time.time() - start_time
        self.metrics['total_time'] += elapsed
        self.metrics['iterations'] += 1
        
        # Check start position
        start_h = hash(C4State())
        start_solved = start_h in self.solved
        
        print(f"\n6. Status:")
        print(f"   Total solved: {len(self.solved)}")
        print(f"   Iteration time: {elapsed:.2f}s")
        print(f"   Total time: {self.metrics['total_time']:.2f}s")
        
        if start_solved:
            outcome = {1: 'X WINS', 0: 'TIE', -1: 'O WINS'}[self.solved[start_h]]
            print(f"\n*** GAME SOLVED: {outcome} ***")
        
        # Save
        print("\n7. Saving...")
        self.save()
        
        return {
            'forward': fwd_result,
            'backward': bwd_result,
            'standing_wave': standing,
            'equiv_solved': equiv_solved,
            'pruned': pruned,
            'total_solved': len(self.solved),
            'start_solved': start_solved,
            'elapsed': elapsed
        }


def main():
    solver = BidirectionalSolver()
    
    if solver.load():
        print("\nResuming from saved state...")
    else:
        print("\nStarting fresh...")
        solver.initialize()
    
    result = solver.run_one_iteration(max_states=150000)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Iterations: {solver.metrics['iterations']}")
    print(f"Total solved: {len(solver.solved)}")
    print(f"Forward frontier: {len(solver.forward_frontier)}")
    print(f"Backward frontier: {len(solver.backward_frontier)}")
    print(f"Standing wave: {len(solver.standing_wave)}")
    print(f"Equiv classes with outcome: {solver.metrics['equiv_classes_with_outcome']}")
    print(f"Total time: {solver.metrics['total_time']:.2f}s")
    
    if result['start_solved']:
        print(f"\n*** Connect-4 SOLVED! ***")
    else:
        print(f"\nRun again to continue...")
    
    return result


def run_until_solved(max_iterations=200, max_states_per_iter=150000):
    """
    Run solver until Connect-4 is solved or max iterations reached.
    Includes memory monitoring and periodic saves.
    """
    import gc
    
    solver = BidirectionalSolver()
    
    if solver.load():
        print("\nResuming from saved state...")
    else:
        print("\nStarting fresh...")
        solver.initialize()
    
    start_iter = solver.metrics['iterations']
    
    for i in range(max_iterations):
        current_iter = solver.metrics['iterations'] + 1
        print(f"\n{'#'*60}")
        print(f"# ITERATION {current_iter} (run {i+1}/{max_iterations})")
        print(f"{'#'*60}")
        
        result = solver.run_one_iteration(max_states=max_states_per_iter)
        
        # Memory management: force garbage collection every 5 iterations
        if i % 5 == 4:
            gc.collect()
            print("   [Garbage collection performed]")
        
        # Check if solved
        if result['start_solved']:
            print("\n" + "=" * 60)
            print("🎉 CONNECT-4 SOLVED! 🎉")
            print("=" * 60)
            start_h = hash(C4State())
            outcome = {1: 'FIRST PLAYER (X) WINS', 0: 'TIE', -1: 'SECOND PLAYER (O) WINS'}
            print(f"Result: {outcome[solver.solved[start_h]]}")
            print(f"Total iterations: {solver.metrics['iterations']}")
            print(f"Total time: {solver.metrics['total_time']:.2f}s ({solver.metrics['total_time']/60:.1f} min)")
            print(f"Total states solved: {len(solver.solved):,}")
            return solver
        
        # Progress report every 10 iterations
        if current_iter % 10 == 0:
            print(f"\n--- Progress Report (iter {current_iter}) ---")
            print(f"    Solved: {len(solver.solved):,}")
            print(f"    Forward frontier: {len(solver.forward_frontier):,}")
            print(f"    Backward frontier: {len(solver.backward_frontier):,}")
            print(f"    Time: {solver.metrics['total_time']:.1f}s")
    
    print(f"\nMax iterations ({max_iterations}) reached without solving.")
    print(f"Final state: {len(solver.solved):,} solved")
    return solver


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--solve":
        # Run until solved
        max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        run_until_solved(max_iterations=max_iter)
    else:
        # Single iteration (default)
        main()
