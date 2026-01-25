"""
constraint_wave.py - Constraint-Wave System for Sudoku

Adapting the closure/wave paradigm for constraint satisfaction problems.

The key insight: Sudoku isn't adversarial, but it IS bidirectional:
- FORWARD wave: "What can I place here?" (possibilities)
- BACKWARD wave: "What must be true for this to be solved?" (constraints)

Closure occurs when:
- Forward possibility meets backward constraint
- Only ONE possibility remains (forced cell)
- A path from puzzle to solution is proven

This is analogous to:
- Pressure = number of remaining candidates
- Permeability = how freely constraints propagate
- Interior = region where all cells are forced or solved
- Crystallization = constraint propagation locking in values
"""

import math
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from resonance.holos.games.sudoku import SudokuState, SudokuGame, SudokuValue


@dataclass
class ConstraintState:
    """State of a cell in the constraint wave system."""
    # Possibilities (forward wave)
    candidates: Set[int] = field(default_factory=lambda: set(range(1, 10)))

    # Constraints (backward wave) - digits that MUST NOT be here
    excluded: Set[int] = field(default_factory=set)

    # Pressure = remaining possibilities after exclusions
    @property
    def pressure(self) -> int:
        return len(self.candidates - self.excluded)

    # Is this a closure? (forced cell)
    @property
    def is_forced(self) -> bool:
        remaining = self.candidates - self.excluded
        return len(remaining) == 1

    # The forced value, if any
    @property
    def forced_value(self) -> Optional[int]:
        remaining = self.candidates - self.excluded
        if len(remaining) == 1:
            return list(remaining)[0]
        return None


@dataclass
class WaveFront:
    """Track the wavefront of constraint propagation."""
    # Cells that changed this iteration
    changed_cells: Set[Tuple[int, int]] = field(default_factory=set)

    # Newly forced cells (closures)
    forced_cells: List[Tuple[int, int, int]] = field(default_factory=list)

    # Energy remaining
    energy: float = 100.0


class ConstraintWaveSystem:
    """
    Wave system for Sudoku using constraint propagation physics.

    Instead of forward/backward waves meeting in state space,
    we have:
    - Forward wave: Place digits, expand possibilities
    - Constraint wave: Propagate exclusions, reduce possibilities

    Closure = cell where |candidates - excluded| = 1
    """

    def __init__(self, game: SudokuGame = None):
        self.game = game or SudokuGame()

        # The puzzle state (evolves as we solve)
        self.puzzle: Optional[SudokuState] = None

        # Constraint tracking per cell
        self.cells: Dict[Tuple[int, int], ConstraintState] = {}

        # Wave fronts
        self.forward_front: WaveFront = WaveFront()
        self.constraint_front: WaveFront = WaveFront()

        # Solution path
        self.solution_path: List[Tuple[int, int, int]] = []

        # Stats
        self.iteration = 0
        self.stats = {
            'propagations': 0,
            'forced_cells': 0,
            'backtracks': 0,
            'interiors_formed': 0,
        }

    def initialize(self, puzzle: SudokuState):
        """Initialize from a puzzle."""
        self.puzzle = puzzle

        # Initialize all cells
        for r in range(9):
            for c in range(9):
                self.cells[(r, c)] = ConstraintState()

        # Set initial constraints from given digits
        for r, c, digit in puzzle.filled_cells():
            self._place_digit(r, c, digit, propagate=False)

        # Initial constraint propagation
        self._propagate_all_constraints()

    def _place_digit(self, r: int, c: int, digit: int, propagate: bool = True):
        """Place a digit and optionally propagate constraints."""
        # Update puzzle state
        self.puzzle = self.puzzle.set(r, c, digit)

        # Mark cell as having exactly this candidate
        cell = self.cells[(r, c)]
        cell.candidates = {digit}
        cell.excluded = set(range(1, 10)) - {digit}

        # Record in solution path
        self.solution_path.append((r, c, digit))

        if propagate:
            # Add affected cells to constraint front
            for affected in self._get_peers(r, c):
                self.constraint_front.changed_cells.add(affected)
            self._propagate_constraints_from(r, c, digit)

    def _get_peers(self, r: int, c: int) -> Set[Tuple[int, int]]:
        """Get all cells that share a constraint with (r, c)."""
        peers = set()

        # Same row
        for col in range(9):
            if col != c:
                peers.add((r, col))

        # Same column
        for row in range(9):
            if row != r:
                peers.add((row, c))

        # Same box
        box_r, box_c = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                nr, nc = box_r + dr, box_c + dc
                if (nr, nc) != (r, c):
                    peers.add((nr, nc))

        return peers

    def _propagate_constraints_from(self, r: int, c: int, digit: int):
        """Propagate constraint from placed digit to peers."""
        for pr, pc in self._get_peers(r, c):
            cell = self.cells[(pr, pc)]

            # Add digit to excluded set
            if digit not in cell.excluded and digit in cell.candidates:
                cell.excluded.add(digit)
                self.stats['propagations'] += 1

                # Check if this creates a closure (forced cell)
                if cell.is_forced and self.puzzle.get(pr, pc) == 0:
                    forced = cell.forced_value
                    self.constraint_front.forced_cells.append((pr, pc, forced))

                # Add to wavefront for further propagation
                self.constraint_front.changed_cells.add((pr, pc))

    def _propagate_all_constraints(self):
        """Full constraint propagation pass."""
        changed = True
        while changed:
            changed = False

            for r in range(9):
                for c in range(9):
                    if self.puzzle.get(r, c) == 0:
                        # Cell is empty, compute its constraints
                        cell = self.cells[(r, c)]
                        old_excluded = len(cell.excluded)

                        # Get digits used in peers
                        for digit in self.puzzle.get_row(r):
                            if digit > 0:
                                cell.excluded.add(digit)
                        for digit in self.puzzle.get_col(c):
                            if digit > 0:
                                cell.excluded.add(digit)
                        for digit in self.puzzle.get_box(r, c):
                            if digit > 0:
                                cell.excluded.add(digit)

                        if len(cell.excluded) > old_excluded:
                            changed = True
                            self.stats['propagations'] += 1

    def step(self) -> Dict[str, Any]:
        """One step of constraint wave propagation."""
        self.iteration += 1

        # Process forced cells (closures)
        placed = 0
        new_forced = []

        # Find all currently forced cells
        for r in range(9):
            for c in range(9):
                if self.puzzle.get(r, c) == 0:
                    cell = self.cells[(r, c)]
                    if cell.is_forced:
                        digit = cell.forced_value
                        new_forced.append((r, c, digit))

        # Place forced digits
        for r, c, digit in new_forced:
            if self.puzzle.get(r, c) == 0:  # Still empty
                self._place_digit(r, c, digit)
                placed += 1
                self.stats['forced_cells'] += 1

        # Check for dead ends (cell with no candidates)
        dead_end = False
        for r in range(9):
            for c in range(9):
                if self.puzzle.get(r, c) == 0:
                    cell = self.cells[(r, c)]
                    if cell.pressure == 0:
                        dead_end = True
                        break

        # Calculate metrics
        total_pressure = sum(
            self.cells[(r, c)].pressure
            for r in range(9) for c in range(9)
            if self.puzzle.get(r, c) == 0
        )

        filled = self.puzzle.filled_count()
        remaining = 81 - filled

        return {
            'iteration': self.iteration,
            'placed': placed,
            'filled': filled,
            'remaining': remaining,
            'total_pressure': total_pressure,
            'avg_pressure': total_pressure / max(1, remaining),
            'dead_end': dead_end,
            'is_solved': self.puzzle.is_solved(),
        }

    def find_branch_point(self) -> Optional[Tuple[int, int, Set[int]]]:
        """
        Find best cell to branch on when stuck.

        Uses MRV (Minimum Remaining Values) heuristic.
        Returns (row, col, candidates) or None if solved/dead.
        """
        best = None
        best_count = 10

        for r in range(9):
            for c in range(9):
                if self.puzzle.get(r, c) == 0:
                    cell = self.cells[(r, c)]
                    remaining = cell.candidates - cell.excluded

                    if len(remaining) == 0:
                        return None  # Dead end

                    if len(remaining) < best_count:
                        best_count = len(remaining)
                        best = (r, c, remaining)

        return best

    def run(self, max_iterations: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """Run constraint wave until solved or stuck."""
        t0 = time.time()

        if verbose:
            print(f"\n  Starting puzzle: {self.puzzle.filled_count()} clues")
            print("  Iter | Filled | Remain | Pressure | Placed")
            print("  " + "-" * 50)

        for i in range(max_iterations):
            result = self.step()

            if verbose and i % 5 == 0:
                print(f"  {result['iteration']:4d} | {result['filled']:6d} | "
                      f"{result['remaining']:6d} | {result['total_pressure']:8d} | "
                      f"{result['placed']:6d}")

            if result['is_solved']:
                if verbose:
                    print(f"\n  SOLVED in {i+1} iterations!")
                break

            if result['dead_end']:
                if verbose:
                    print(f"\n  Dead end at iteration {i+1}")
                break

            if result['placed'] == 0:
                # No progress - need to branch
                if verbose:
                    print(f"\n  Stuck at iteration {i+1} - need branching")
                break

        elapsed = time.time() - t0

        return {
            'iterations': self.iteration,
            'elapsed': elapsed,
            'solved': self.puzzle.is_solved(),
            'filled': self.puzzle.filled_count(),
            'solution_path_length': len(self.solution_path),
            'stats': self.stats,
        }


class ConstraintWaveSolver:
    """
    Full solver using constraint waves with backtracking.

    Combines:
    - Constraint wave propagation (the "fast" part)
    - Branching with backtracking (when waves can't progress)
    """

    def __init__(self, game: SudokuGame = None):
        self.game = game or SudokuGame()
        self.solutions: List[SudokuState] = []
        self.stats = {
            'branches': 0,
            'backtracks': 0,
            'propagations': 0,
        }

    def solve(self, puzzle: SudokuState, find_all: bool = False,
              verbose: bool = True) -> List[SudokuState]:
        """
        Solve using constraint wave + backtracking.

        Args:
            puzzle: Starting puzzle
            find_all: If True, find all solutions
            verbose: Print progress

        Returns:
            List of solutions found
        """
        self.solutions = []
        self._solve_recursive(puzzle, verbose=verbose, depth=0, find_all=find_all)
        return self.solutions

    def _solve_recursive(self, puzzle: SudokuState, verbose: bool, depth: int,
                         find_all: bool = False):
        """Recursive solve with constraint propagation."""
        # Run constraint wave
        system = ConstraintWaveSystem(self.game)
        system.initialize(puzzle)
        result = system.run(max_iterations=100, verbose=False)

        self.stats['propagations'] += system.stats['propagations']

        if result['solved']:
            self.solutions.append(system.puzzle)
            if verbose:
                print(f"    Found solution #{len(self.solutions)} at depth {depth}")
            return

        # Check for dead end (no candidates for some cell)
        if result.get('dead_end'):
            self.stats['backtracks'] += 1
            return

        # Need to branch - find cell with fewest candidates
        branch = system.find_branch_point()

        if branch is None:
            # Dead end - no valid candidates
            self.stats['backtracks'] += 1
            return

        r, c, candidates = branch
        self.stats['branches'] += 1

        if verbose and depth < 3:
            print(f"    Branching at ({r},{c}) with {len(candidates)} options, depth {depth}")

        for digit in sorted(candidates):
            # Try placing this digit
            new_puzzle = system.puzzle.set(r, c, digit)
            if not new_puzzle.has_conflicts():
                self._solve_recursive(new_puzzle, verbose, depth + 1, find_all)

                if self.solutions and not find_all:
                    return  # Found one, stop


def run_sudoku_constraint_wave():
    """Test constraint wave system on Sudoku puzzles."""
    print("\n" + "=" * 70)
    print("CONSTRAINT WAVE SUDOKU SOLVER")
    print("=" * 70)

    from resonance.holos.games.sudoku import get_sample_puzzles

    puzzles = get_sample_puzzles()

    for difficulty, puzzle in puzzles.items():
        print(f"\n--- {difficulty.upper()} PUZZLE ({puzzle.filled_count()} clues) ---")
        print(puzzle.display())

        solver = ConstraintWaveSolver()
        solutions = solver.solve(puzzle, verbose=True)

        print(f"\n  Results:")
        print(f"    Solutions found: {len(solutions)}")
        print(f"    Branches: {solver.stats['branches']}")
        print(f"    Backtracks: {solver.stats['backtracks']}")
        print(f"    Propagations: {solver.stats['propagations']}")

        if solutions:
            print(f"\n  Solution:")
            print(solutions[0].display())


def compare_approaches():
    """Compare constraint wave vs fast daemon on Sudoku."""
    print("\n" + "=" * 70)
    print("COMPARISON: Constraint Wave vs Fast Daemon")
    print("=" * 70)

    from resonance.holos.games.sudoku import get_sample_puzzles
    from resonance.holos.fast_daemon import run_fast_daemon_search

    puzzles = get_sample_puzzles()
    puzzle = puzzles['easy']

    print(f"\nPuzzle: {puzzle.filled_count()} clues")

    # Constraint Wave approach
    print("\n1. CONSTRAINT WAVE (physics-based):")
    t0 = time.time()
    solver = ConstraintWaveSolver()
    solutions = solver.solve(puzzle, verbose=False)
    t1 = time.time()

    print(f"   Solved: {len(solutions) > 0}")
    print(f"   Time: {(t1-t0)*1000:.1f}ms")
    print(f"   Propagations: {solver.stats['propagations']}")

    # Fast Daemon approach (wave meeting)
    print("\n2. FAST DAEMON (state-space waves):")
    game = SudokuGame()
    result = run_fast_daemon_search(
        game,
        start_states=[puzzle],
        max_iterations=200,
        daemon_frequency=10,
        verbose=False
    )

    print(f"   States explored: {result['states']}")
    print(f"   Closures: {result['closures']}")
    print(f"   Values: {result['values']}")
    print(f"   Time: {result['elapsed']*1000:.1f}ms")

    print("""
    ANALYSIS:

    Constraint Wave (suited for Sudoku):
    - Works with the STRUCTURE of constraints
    - Propagation is natural (like waves in a constrained medium)
    - "Closure" = cell with one candidate
    - Very efficient for constraint problems

    Fast Daemon (suited for adversarial games):
    - Works with STATE SPACE exploration
    - Forward/backward waves in game tree
    - "Closure" = waves meeting at same state
    - Better for minimax problems

    The physics is the same:
    - Waves propagate through a medium
    - Constraints create impedance
    - Closure occurs at equilibrium points
    - Fast observation speeds up convergence

    The difference is the MEDIUM:
    - Sudoku: constraint graph (cells connected by rules)
    - Connect4: game tree (states connected by moves)
    """)


if __name__ == "__main__":
    run_sudoku_constraint_wave()
    compare_approaches()
