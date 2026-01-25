"""
quantum_fewshot.py - Quantum-Inspired Few-Shot Solving

The question: What's the FASTEST way to find ANY solution path?

Physics intuition from the new paradigm:

1. OSMOSIS (old): Follow concentration gradient
   - Good for: balanced bilateral exploration
   - Problem: Still explores "width" not just "depth"

2. LIGHTNING (old): Direct soliton path
   - Good for: forced sequences
   - Problem: Only works when path is obvious

3. QUANTUM-GUIDED (new): Amplitude-weighted depth-first
   - Insight: In quantum search (Grover), amplitude concentrates on solutions
   - If we track amplitudes through the search, HIGH amplitude = likely path
   - Combine with constraint propagation for CSPs

The key insight from quantum mechanics:
- Amplitude flows through the graph like probability
- Interference CONCENTRATES amplitude on good paths
- Measurement collapses to highest-amplitude states

For few-shot solving, we want:
- DEPTH over breadth (find ONE path fast)
- GUIDED by where amplitude naturally flows
- COLLAPSE immediately when solution found

This is like "beam search" but with quantum-inspired scoring.
"""

import math
import time
import heapq
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from resonance.holos.games.sudoku import SudokuState, SudokuGame, SudokuValue
from resonance.holos.games.connect4 import C4State, Connect4Game, C4Value


@dataclass
class QuantumState:
    """State with quantum-like amplitude tracking."""
    state: Any
    amplitude: float = 1.0  # Probability amplitude
    depth: int = 0          # Steps from start
    path: List[Any] = field(default_factory=list)  # Moves taken

    def __lt__(self, other):
        # For heapq: higher amplitude = higher priority (use negative)
        return -self.amplitude < -other.amplitude


class QuantumFewShotSolver:
    """
    Quantum-inspired few-shot solver.

    Key differences from traditional search:

    1. AMPLITUDE TRACKING: Each state has an amplitude that flows from parent
       - Amplitude = parent_amplitude / sqrt(branching_factor)
       - This naturally penalizes wide branches

    2. INTERFERENCE: When paths meet, amplitudes combine
       - Constructive: Same direction = amplify
       - Destructive: Opposite direction = cancel
       - This focuses search on "likely" paths

    3. COLLAPSE: When we find a solution, we're done
       - No need to explore further
       - The amplitude told us this was a good path

    4. CONSTRAINT BOOST: For CSPs, forced moves get amplitude boost
       - If only 1 candidate, amplitude *= 2
       - This is like the quantum oracle marking solutions
    """

    def __init__(self, game: Any, beam_width: int = 5):
        self.game = game
        self.beam_width = beam_width  # How many paths to track simultaneously

        # State tracking
        self.visited: Set[int] = set()
        self.amplitudes: Dict[int, float] = {}

        # Stats
        self.stats = {
            'states_explored': 0,
            'max_depth': 0,
            'amplitude_boosts': 0,
            'path_length': 0,
        }

    def solve(self, start: Any, max_steps: int = 10000,
              verbose: bool = True) -> Optional[List[Any]]:
        """
        Find ANY solution path as quickly as possible.

        Returns the solution path (list of moves) or None.
        """
        t0 = time.time()

        # Initialize with start state
        h = self.game.hash_state(start)
        initial = QuantumState(state=start, amplitude=1.0, depth=0, path=[])

        # Priority queue (max-heap by amplitude)
        beam = [initial]
        heapq.heapify(beam)

        self.visited.add(h)
        self.amplitudes[h] = 1.0

        solution_path = None

        while beam and self.stats['states_explored'] < max_steps:
            # Get highest-amplitude state
            current = heapq.heappop(beam)
            self.stats['states_explored'] += 1
            self.stats['max_depth'] = max(self.stats['max_depth'], current.depth)

            # Check if solved
            is_term, value = self.game.is_terminal(current.state)
            if is_term and value is not None:
                if self._is_winning(value):
                    solution_path = current.path
                    self.stats['path_length'] = len(solution_path)
                    break
                else:
                    # Dead end - don't expand
                    continue

            # Expand successors
            successors = list(self.game.get_successors(current.state))
            if not successors:
                continue

            # Calculate amplitude for children
            # Key insight: amplitude splits among children
            branching = len(successors)
            base_amplitude = current.amplitude / math.sqrt(branching)

            for child, move in successors:
                ch = self.game.hash_state(child)

                if ch in self.visited:
                    # Interference! Combine amplitudes
                    old_amp = self.amplitudes[ch]
                    new_amp = old_amp + base_amplitude  # Constructive
                    self.amplitudes[ch] = new_amp
                    continue

                self.visited.add(ch)

                # Apply constraint boost for CSPs
                boosted_amplitude = self._apply_constraint_boost(
                    child, base_amplitude
                )
                self.amplitudes[ch] = boosted_amplitude

                # Create new quantum state
                new_state = QuantumState(
                    state=child,
                    amplitude=boosted_amplitude,
                    depth=current.depth + 1,
                    path=current.path + [move]
                )

                # Add to beam (keep only top beam_width)
                heapq.heappush(beam, new_state)

            # Prune beam to beam_width
            if len(beam) > self.beam_width * 2:
                beam = heapq.nsmallest(self.beam_width, beam)
                heapq.heapify(beam)

        elapsed = time.time() - t0

        if verbose:
            print(f"  States explored: {self.stats['states_explored']}")
            print(f"  Max depth: {self.stats['max_depth']}")
            print(f"  Amplitude boosts: {self.stats['amplitude_boosts']}")
            print(f"  Time: {elapsed*1000:.1f}ms")
            if solution_path:
                print(f"  Solution path length: {len(solution_path)}")

        return solution_path

    def _is_winning(self, value: Any) -> bool:
        """Check if value represents a winning/solved state."""
        if hasattr(value, 'solved'):
            return value.solved
        if hasattr(value, 'winner'):
            return value.winner is not None
        return value is not None

    def _apply_constraint_boost(self, state: Any, base_amplitude: float) -> float:
        """
        Apply amplitude boost based on constraint analysis.

        For CSPs: fewer candidates = higher amplitude (more constrained = more likely correct)
        For games: winning threats = higher amplitude
        """
        boost = 1.0

        # Sudoku-specific: boost for forced cells and constraint tightness
        if isinstance(state, SudokuState):
            min_candidates = 10
            total_candidates = 0
            empty_count = 0
            forced_count = 0

            for r, c in state.empty_cells():
                cands = len(state.get_candidates(r, c))
                if cands == 0:
                    return 0.0  # Dead end - kill this path
                total_candidates += cands
                empty_count += 1
                if cands < min_candidates:
                    min_candidates = cands
                if cands == 1:
                    forced_count += 1

            # Boost based on constraint tightness
            avg_candidates = total_candidates / max(1, empty_count)

            if forced_count > 0:
                # Has forced moves - very constrained, very good!
                boost = 2.0 + forced_count * 0.5
                self.stats['amplitude_boosts'] += 1
            elif min_candidates == 2:
                boost = 1.5
            elif avg_candidates < 3:
                boost = 1.2  # Fairly constrained
            elif avg_candidates > 5:
                boost = 0.5  # Very unconstrained - penalize heavily

        # Connect4-specific: boost for winning threats
        elif isinstance(state, C4State):
            # Check if we can win next move
            for col in range(7):
                if state.can_play(col):
                    test = state.play(col)
                    if test and test.check_win(state.turn):
                        boost = 5.0  # Winning move available!
                        self.stats['amplitude_boosts'] += 1
                        break

        return base_amplitude * boost


class QuantumConstraintSolver:
    """
    Combined quantum amplitude + constraint propagation solver.

    This is the "best of both worlds":
    1. Constraint propagation handles forced deductions (collapse)
    2. Quantum amplitude guides branch ORDERING (not pruning)
    3. Full backtracking ensures completeness
    """

    def __init__(self, beam_width: int = 3):
        self.beam_width = beam_width  # Only affects scoring, not pruning
        self.stats = {
            'propagations': 0,
            'branches': 0,
            'states_explored': 0,
            'backtracks': 0,
        }

    def solve_sudoku(self, puzzle: SudokuState, verbose: bool = True) -> Optional[SudokuState]:
        """Solve Sudoku using quantum-guided constraint propagation."""
        t0 = time.time()

        # Start with constraint propagation
        current, forced = self._propagate(puzzle)
        self.stats['propagations'] += forced

        if current.is_solved():
            if verbose:
                print(f"  Solved by propagation alone! ({forced} forced)")
            return current

        # Check for dead end after propagation
        if self._is_dead_end(current):
            return None

        # Need to branch - use quantum-guided DFS with full backtracking
        solution = self._quantum_dfs(current)

        elapsed = time.time() - t0

        if verbose:
            print(f"  Total time: {elapsed*1000:.1f}ms")
            print(f"  Stats: {self.stats}")

        return solution

    def _propagate(self, state: SudokuState) -> Tuple[SudokuState, int]:
        """Apply constraint propagation until stuck or dead."""
        forced = 0

        while True:
            progress = False

            for r in range(9):
                for c in range(9):
                    if state.get(r, c) == 0:
                        cands = state.get_candidates(r, c)
                        if len(cands) == 1:
                            digit = list(cands)[0]
                            state = state.set(r, c, digit)
                            forced += 1
                            progress = True
                        elif len(cands) == 0:
                            # Dead end
                            return state, forced

            if not progress:
                break

        return state, forced

    def _is_dead_end(self, state: SudokuState) -> bool:
        """Check if state is a dead end (any cell with 0 candidates)."""
        for r, c in state.empty_cells():
            if len(state.get_candidates(r, c)) == 0:
                return True
        return False

    def _quantum_dfs(self, start: SudokuState) -> Optional[SudokuState]:
        """
        Quantum-amplitude-ordered DFS with full backtracking.

        Key insight: We use amplitude to ORDER branches, not to PRUNE them.
        This gives us the speed of best-first with the completeness of DFS.
        """
        # Find cell with minimum candidates (MRV heuristic)
        best_cell = None
        best_count = 10

        for r in range(9):
            for c in range(9):
                if start.get(r, c) == 0:
                    cands = start.get_candidates(r, c)
                    if len(cands) == 0:
                        self.stats['backtracks'] += 1
                        return None  # Dead end
                    if len(cands) < best_count:
                        best_count = len(cands)
                        best_cell = (r, c, cands)

        if best_cell is None:
            return start if start.is_solved() else None

        r, c, candidates = best_cell
        self.stats['branches'] += 1

        # Score each candidate by "quantum amplitude" = propagation power
        scored_candidates = []
        for digit in candidates:
            test = start.set(r, c, digit)
            if test.has_conflicts():
                continue

            # Propagate and score
            propagated, forced = self._propagate(test)
            self.stats['propagations'] += forced

            if propagated.is_solved():
                # Immediate win!
                return propagated

            if self._is_dead_end(propagated):
                # Dead end - skip
                continue

            # Amplitude score: more forced = more constrained = better
            # Also consider remaining empty cells
            remaining = 81 - propagated.filled_count()
            score = forced * 10 + (81 - remaining)  # Higher = better
            scored_candidates.append((score, digit, propagated))

        # Sort by amplitude (descending) - best paths first
        scored_candidates.sort(reverse=True)

        # Try ALL candidates in amplitude order (full DFS, no pruning)
        for score, digit, propagated in scored_candidates:
            self.stats['states_explored'] += 1

            # Recurse
            result = self._quantum_dfs(propagated)
            if result is not None:
                return result

        self.stats['backtracks'] += 1
        return None


def benchmark_solvers():
    """Compare different solving approaches."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Few-Shot Solving Approaches")
    print("=" * 70)

    from resonance.holos.games.sudoku import get_sample_puzzles
    from resonance.holos.constraint_wave import ConstraintWaveSolver

    puzzles = get_sample_puzzles()

    for difficulty, puzzle in puzzles.items():
        print(f"\n--- {difficulty.upper()} ({puzzle.filled_count()} clues) ---")

        # 1. Pure constraint propagation
        print("\n1. Constraint Wave (pure propagation + backtrack):")
        t0 = time.time()
        cw_solver = ConstraintWaveSolver()
        cw_solutions = cw_solver.solve(puzzle, verbose=False)
        cw_time = time.time() - t0
        print(f"   Solved: {len(cw_solutions) > 0}")
        print(f"   Time: {cw_time*1000:.1f}ms")
        print(f"   Branches: {cw_solver.stats['branches']}")
        print(f"   Propagations: {cw_solver.stats['propagations']}")

        # 2. Quantum few-shot (with wider beam for hard puzzles)
        beam = 10 if difficulty == 'hard' else 5
        print(f"\n2. Quantum Few-Shot (beam={beam}):")
        t0 = time.time()
        qfs_solver = QuantumFewShotSolver(SudokuGame(), beam_width=beam)
        qfs_path = qfs_solver.solve(puzzle, max_steps=100000, verbose=False)
        qfs_time = time.time() - t0
        print(f"   Solved: {qfs_path is not None}")
        print(f"   Time: {qfs_time*1000:.1f}ms")
        print(f"   States: {qfs_solver.stats['states_explored']}")
        print(f"   Path length: {qfs_solver.stats['path_length']}")

        # 3. Quantum + Constraint combined
        print("\n3. Quantum-Constraint Hybrid:")
        t0 = time.time()
        qc_solver = QuantumConstraintSolver(beam_width=3)
        qc_solution = qc_solver.solve_sudoku(puzzle, verbose=False)
        qc_time = time.time() - t0
        print(f"   Solved: {qc_solution is not None}")
        print(f"   Time: {qc_time*1000:.1f}ms")
        print(f"   Branches: {qc_solver.stats['branches']}")
        print(f"   States: {qc_solver.stats['states_explored']}")

        print(f"\n   Winner: ", end="")
        times = [
            ("Constraint Wave", cw_time),
            ("Quantum Few-Shot", qfs_time),
            ("Quantum-Constraint", qc_time),
        ]
        winner = min(times, key=lambda x: x[1])
        print(f"{winner[0]} ({winner[1]*1000:.1f}ms)")


def analyze_physics():
    """Analyze the physics of different approaches."""
    print("\n" + "=" * 70)
    print("PHYSICS ANALYSIS: Osmosis vs Quantum-Guided")
    print("=" * 70)

    print("""
    OSMOSIS (diffusion-based):
    - Physics: Particles flow from high to low concentration
    - Search analog: Expand where frontier is "denser"
    - Pros: Natural bilateral balance
    - Cons: Still explores width, not optimized for ANY path

    QUANTUM-GUIDED (amplitude-based):
    - Physics: Probability amplitude concentrates on likely outcomes
    - Search analog: Expand where amplitude is highest
    - Pros: Naturally depth-first on promising paths
    - Cons: Can miss if heuristic is wrong

    THE KEY DIFFERENCE:

    Osmosis asks: "Where is the concentration gradient?"
    Quantum asks: "Where is the amplitude concentrated?"

    For few-shot solving, amplitude is better because:
    1. Amplitude naturally flows TOWARD solutions (via boosts)
    2. Amplitude penalizes wide branching (splits by sqrt(n))
    3. Amplitude interference reinforces good paths

    HYBRID APPROACH (best of both):

    1. Use CONSTRAINT PROPAGATION first (the "collapse")
       - This is like quantum measurement - forces values
       - Eliminates branches before they're explored

    2. Use AMPLITUDE-GUIDED SEARCH when stuck
       - Score candidates by how many propagations they enable
       - This is like the quantum oracle marking good states

    3. Use BEAM SEARCH to limit memory
       - Keep only top-k amplitude states
       - This is like decoherence - weak paths die out

    The result is something like:
    - Grover's search (amplitude amplification)
    - Combined with constraint propagation
    - In a classical beam search framework

    This is the "poor man's quantum" applied to search!
    """)


if __name__ == "__main__":
    benchmark_solvers()
    analyze_physics()
