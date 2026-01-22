"""
Connect-4 Amplitude Wave Solver

Physically realistic wave dynamics:
- Forward wave carries "possibility amplitude" (splits among branches)
- Backward wave carries "determination amplitude" (accumulates from solutions)
- Standing wave = interference pattern = where to focus computation

This naturally prioritizes high-confidence paths without artificial heuristics.
The amplitude IS the heuristic, emergent from wave dynamics.
"""

import pickle
import json
import os
import time
import gc
import heapq
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass, field

SAVE_DIR = "./c4_amplitude_state"
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


@dataclass
class WaveState:
    """State with wave amplitudes"""
    compact: tuple
    forward_amp: float = 0.0   # Possibility amplitude (from start)
    backward_amp: float = 0.0  # Determination amplitude (from terminals)
    depth: int = 0
    
    @property
    def standing_amp(self):
        """Interference pattern - product of forward and backward"""
        return self.forward_amp * self.backward_amp
    
    @property
    def total_amp(self):
        """Total wave energy at this point"""
        return self.forward_amp + self.backward_amp


class AmplitudeWaveSolver:
    """
    Wave solver with amplitude tracking.
    
    Physical model:
    - Forward amplitude = probability of reaching this state from start
    - Backward amplitude = "determination pressure" from known outcomes
    - Standing amplitude = confidence in this region
    
    High standing amplitude = prioritize for solving
    """
    
    def __init__(self):
        self.solved: Dict[int, int] = {}  # hash -> value
        self.wave_states: Dict[int, WaveState] = {}  # hash -> WaveState
        self.children: Dict[int, list] = {}  # hash -> [child hashes]
        self.parents: Dict[int, Set[int]] = defaultdict(set)  # child -> {parent hashes}
        
        self.equiv_classes: Dict[C4Features, Set[int]] = defaultdict(set)
        self.equiv_outcomes: Dict[C4Features, Optional[int]] = {}
        
        # Priority queue for expansion (by standing amplitude)
        self.forward_frontier: list = []  # heap of (-amp, hash)
        self.backward_frontier: list = []  # heap of (-amp, hash)
        
        # Tracking
        self.forward_expanded: Set[int] = set()
        self.backward_expanded: Set[int] = set()
        
        self.depth = 0
        self.phase = "forward"  # or "backward" or "interference"
        
        self.metrics = {
            'total_time': 0.0,
            'forward_expansions': 0,
            'backward_propagations': 0,
            'interference_solves': 0,
        }
    
    def initialize(self):
        """Start with empty board at full amplitude"""
        start = C4State()
        h = hash(start)
        
        ws = WaveState(
            compact=start.to_compact(),
            forward_amp=1.0,  # Full possibility amplitude
            backward_amp=0.0,
            depth=0
        )
        self.wave_states[h] = ws
        heapq.heappush(self.forward_frontier, (-ws.forward_amp, h))
        
        self.equiv_classes[start.get_features()].add(h)
        print(f"Initialized with start position (amplitude=1.0)")
    
    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Convert heaps to lists for pickling
        with open(f"{SAVE_DIR}/solved.pkl", 'wb') as f:
            pickle.dump(self.solved, f)
        with open(f"{SAVE_DIR}/wave_states.pkl", 'wb') as f:
            pickle.dump(self.wave_states, f)
        with open(f"{SAVE_DIR}/children.pkl", 'wb') as f:
            pickle.dump(self.children, f)
        with open(f"{SAVE_DIR}/parents.pkl", 'wb') as f:
            pickle.dump(dict(self.parents), f)
        with open(f"{SAVE_DIR}/equiv.pkl", 'wb') as f:
            pickle.dump(self.equiv_classes, f)
        with open(f"{SAVE_DIR}/equiv_out.pkl", 'wb') as f:
            pickle.dump(self.equiv_outcomes, f)
        with open(f"{SAVE_DIR}/fwd_frontier.pkl", 'wb') as f:
            pickle.dump(self.forward_frontier, f)
        with open(f"{SAVE_DIR}/bwd_frontier.pkl", 'wb') as f:
            pickle.dump(self.backward_frontier, f)
        with open(f"{SAVE_DIR}/fwd_expanded.pkl", 'wb') as f:
            pickle.dump(self.forward_expanded, f)
        with open(f"{SAVE_DIR}/bwd_expanded.pkl", 'wb') as f:
            pickle.dump(self.backward_expanded, f)
        with open(f"{SAVE_DIR}/state.json", 'w') as f:
            json.dump({'depth': self.depth, 'phase': self.phase}, f)
        with open(f"{SAVE_DIR}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"  [Saved: solved={len(self.solved):,}, wave_states={len(self.wave_states):,}]")
    
    def load(self):
        if not os.path.exists(f"{SAVE_DIR}/solved.pkl"):
            return False
        try:
            with open(f"{SAVE_DIR}/solved.pkl", 'rb') as f:
                self.solved = pickle.load(f)
            with open(f"{SAVE_DIR}/wave_states.pkl", 'rb') as f:
                self.wave_states = pickle.load(f)
            with open(f"{SAVE_DIR}/children.pkl", 'rb') as f:
                self.children = pickle.load(f)
            with open(f"{SAVE_DIR}/parents.pkl", 'rb') as f:
                self.parents = defaultdict(set, {k: set(v) for k, v in pickle.load(f).items()})
            with open(f"{SAVE_DIR}/equiv.pkl", 'rb') as f:
                self.equiv_classes = pickle.load(f)
            with open(f"{SAVE_DIR}/equiv_out.pkl", 'rb') as f:
                self.equiv_outcomes = pickle.load(f)
            with open(f"{SAVE_DIR}/fwd_frontier.pkl", 'rb') as f:
                self.forward_frontier = pickle.load(f)
            with open(f"{SAVE_DIR}/bwd_frontier.pkl", 'rb') as f:
                self.backward_frontier = pickle.load(f)
            with open(f"{SAVE_DIR}/fwd_expanded.pkl", 'rb') as f:
                self.forward_expanded = pickle.load(f)
            with open(f"{SAVE_DIR}/bwd_expanded.pkl", 'rb') as f:
                self.backward_expanded = pickle.load(f)
            with open(f"{SAVE_DIR}/state.json") as f:
                state = json.load(f)
                self.depth = state['depth']
                self.phase = state['phase']
            with open(f"{SAVE_DIR}/metrics.json") as f:
                self.metrics = json.load(f)
            
            print(f"Loaded: solved={len(self.solved):,}, wave_states={len(self.wave_states):,}")
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
    
    def expand_forward_batch(self, batch_size=10000):
        """
        Expand highest-amplitude forward frontier states.
        Amplitude splits among children (conservation of probability).
        """
        expanded = 0
        terminals = 0
        new_states = 0
        
        while self.forward_frontier and expanded < batch_size:
            neg_amp, h = heapq.heappop(self.forward_frontier)
            
            if h in self.forward_expanded:
                continue
            if h not in self.wave_states:
                continue
            
            ws = self.wave_states[h]
            state = C4State.from_compact(ws.compact)
            
            self.forward_expanded.add(h)
            expanded += 1
            
            # Terminal state
            if state.is_terminal():
                value = state.terminal_value()
                self.solved[h] = value
                terminals += 1
                
                # Terminal gets full backward amplitude
                ws.backward_amp = 1.0
                
                # Add to backward frontier
                heapq.heappush(self.backward_frontier, (-ws.backward_amp, h))
                
                features = state.get_features()
                self._update_equiv_outcome(features, value)
                continue
            
            # Generate children - amplitude splits
            moves = state.get_valid_moves()
            child_amp = ws.forward_amp / len(moves)  # Conservation!
            
            child_hashes = []
            for move in moves:
                child = state.play(move)
                ch = hash(child)
                child_hashes.append(ch)
                self.parents[ch].add(h)
                
                if ch not in self.wave_states:
                    # New state
                    child_ws = WaveState(
                        compact=child.to_compact(),
                        forward_amp=child_amp,
                        backward_amp=0.0,
                        depth=ws.depth + 1
                    )
                    self.wave_states[ch] = child_ws
                    new_states += 1
                    
                    features = child.get_features()
                    self.equiv_classes[features].add(ch)
                else:
                    # Existing state - add amplitude (constructive interference!)
                    self.wave_states[ch].forward_amp += child_amp
                
                # Add to frontier if not expanded
                if ch not in self.forward_expanded:
                    heapq.heappush(self.forward_frontier, 
                                   (-self.wave_states[ch].forward_amp, ch))
            
            self.children[h] = child_hashes
        
        self.metrics['forward_expansions'] += expanded
        return expanded, terminals, new_states
    
    def propagate_backward_batch(self, batch_size=10000):
        """
        Propagate backward amplitude from solved states to parents.
        Uses minimax: parent gets amplitude weighted by outcome certainty.
        """
        propagated = 0
        newly_solved = 0
        
        while self.backward_frontier and propagated < batch_size:
            neg_amp, h = heapq.heappop(self.backward_frontier)
            
            if h in self.backward_expanded:
                continue
            if h not in self.solved:
                continue
            
            self.backward_expanded.add(h)
            propagated += 1
            
            child_value = self.solved[h]
            ws = self.wave_states.get(h)
            if not ws:
                continue
            
            # Propagate to parents
            for ph in self.parents.get(h, set()):
                if ph in self.solved:
                    continue
                
                parent_ws = self.wave_states.get(ph)
                if not parent_ws:
                    continue
                
                # Add backward amplitude (wave propagating back)
                # Weight by child's backward amp
                parent_ws.backward_amp += ws.backward_amp / len(self.parents[h])
                
                # Try to solve parent via minimax
                if ph in self.children:
                    child_values = []
                    all_solved = True
                    for ch in self.children[ph]:
                        if ch in self.solved:
                            child_values.append(self.solved[ch])
                        else:
                            all_solved = False
                    
                    if all_solved and child_values:
                        parent = C4State.from_compact(parent_ws.compact)
                        if parent.turn == 'X':
                            value = max(child_values)
                        else:
                            value = min(child_values)
                        
                        self.solved[ph] = value
                        parent_ws.backward_amp = 1.0  # Solved = full determination
                        newly_solved += 1
                        
                        features = parent.get_features()
                        self._update_equiv_outcome(features, value)
                
                # Add parent to backward frontier
                if ph not in self.backward_expanded:
                    heapq.heappush(self.backward_frontier, 
                                   (-parent_ws.backward_amp, ph))
        
        self.metrics['backward_propagations'] += propagated
        return propagated, newly_solved
    
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
                    if h in self.wave_states:
                        self.wave_states[h].backward_amp = 1.0
                    count += 1
        return count
    
    def check_start_solved(self):
        start_h = hash(C4State())
        if start_h in self.solved:
            return True, self.solved[start_h]
        return False, None
    
    def get_amplitude_stats(self):
        """Get statistics about wave amplitudes"""
        if not self.wave_states:
            return {}
        
        fwd_amps = [ws.forward_amp for ws in self.wave_states.values()]
        bwd_amps = [ws.backward_amp for ws in self.wave_states.values()]
        standing = [ws.standing_amp for ws in self.wave_states.values()]
        
        return {
            'max_forward': max(fwd_amps),
            'max_backward': max(bwd_amps),
            'max_standing': max(standing),
            'nonzero_standing': sum(1 for s in standing if s > 0),
        }
    
    def solve(self, max_iterations=10000):
        """
        Main solving loop - alternating wave expansion.
        
        Physical model:
        1. Forward wave expands (possibilities propagate)
        2. Backward wave propagates (determinations flow back)
        3. Interference creates standing wave (high-confidence regions)
        4. Repeat until standing wave reaches start
        """
        print("="*60)
        print("AMPLITUDE WAVE SOLVER")
        print("Forward ←→ Backward Interference")
        print("="*60)
        
        start_time = time.time()
        iteration = 0
        
        # Phase tracking for output
        last_phase = None
        phase_start = time.time()
        
        while iteration < max_iterations:
            iteration += 1
            
            # === FORWARD EXPANSION ===
            fwd_exp, fwd_term, fwd_new = self.expand_forward_batch(10000)
            
            # === BACKWARD PROPAGATION ===
            bwd_prop, bwd_solved = self.propagate_backward_batch(10000)
            
            # === EQUIVALENCE ===
            equiv_solved = self.propagate_equivalence()
            
            # === CHECK SOLUTION ===
            solved, value = self.check_start_solved()
            
            # Progress output
            if iteration % 10 == 0 or solved:
                elapsed = time.time() - start_time
                stats = self.get_amplitude_stats()
                
                print(f"\nIteration {iteration} ({elapsed:.1f}s)")
                print(f"  Forward:  expanded={fwd_exp}, terminals={fwd_term}, new={fwd_new}")
                print(f"  Backward: propagated={bwd_prop}, solved={bwd_solved}")
                print(f"  Equiv:    {equiv_solved}")
                print(f"  Total:    states={len(self.wave_states):,}, solved={len(self.solved):,}")
                print(f"  Frontier: fwd={len(self.forward_frontier):,}, bwd={len(self.backward_frontier):,}")
                if stats:
                    print(f"  Amplitude: max_standing={stats['max_standing']:.6f}, "
                          f"nonzero={stats['nonzero_standing']:,}")
            
            if solved:
                outcome = {1: "X WINS", 0: "DRAW", -1: "O WINS"}[value]
                print(f"\n{'='*60}")
                print(f"🎉 SOLVED: {outcome} 🎉")
                print(f"{'='*60}")
                self.save()
                return True
            
            # Save periodically
            if iteration % 100 == 0:
                self.save()
            
            # Check if stuck
            if fwd_exp == 0 and bwd_prop == 0 and equiv_solved == 0:
                print("\nNo progress - checking remaining states...")
                break
        
        # Final status
        print(f"\nFinal: {len(self.solved):,} solved out of {len(self.wave_states):,} states")
        self.save()
        
        solved, value = self.check_start_solved()
        return solved


def analyze_waves(solver):
    """Analyze the wave interference pattern"""
    print("\n" + "="*60)
    print("WAVE ANALYSIS")
    print("="*60)
    
    # Find highest standing amplitude states
    standing = [(h, ws.standing_amp, ws.forward_amp, ws.backward_amp) 
                for h, ws in solver.wave_states.items()]
    standing.sort(key=lambda x: -x[1])
    
    print("\nTop 10 standing wave amplitudes:")
    for h, sa, fa, ba in standing[:10]:
        solved_str = f"SOLVED={solver.solved[h]}" if h in solver.solved else "unsolved"
        print(f"  {h}: standing={sa:.6f} (fwd={fa:.6f}, bwd={ba:.6f}) [{solved_str}]")
    
    # Amplitude distribution by depth
    depth_amps = defaultdict(list)
    for h, ws in solver.wave_states.items():
        depth_amps[ws.depth].append(ws.standing_amp)
    
    print("\nStanding amplitude by depth:")
    for d in sorted(depth_amps.keys())[:15]:
        amps = depth_amps[d]
        nonzero = sum(1 for a in amps if a > 0)
        max_amp = max(amps) if amps else 0
        print(f"  Depth {d:2d}: {len(amps):>8,} states, {nonzero:>6,} with standing wave, max={max_amp:.6f}")


if __name__ == "__main__":
    solver = AmplitudeWaveSolver()
    
    if solver.load():
        print("\nResuming from saved state...")
    else:
        print("\nStarting fresh...")
        solver.initialize()
    
    solver.solve(max_iterations=50000)
    
    analyze_waves(solver)
