"""
holos/games/sudoku.py - Sudoku Puzzle Interface for HOLOS

This implements GameInterface for Sudoku puzzles (Layer 0).

The puzzle:
- States: 9x9 grids with digits 1-9 or empty (0)
- Successors: Place a valid digit in an empty cell (respecting constraints)
- Predecessors: Remove a digit from a filled cell (for backward search)
- Boundary: Completed valid grids (all 81 cells filled, no conflicts)
- Values: Solved(1) or Unsolved(0)
- Propagation: If any child is solved, parent can reach a solution

Key insight: Sudoku is BOUNDED for HOLOS because:
- Lower bound: Completed grids (81 filled cells) = known solved
- Upper bound: Starting puzzle (given clues)
- Forward: Place digits (increase filled count)
- Backward: Remove digits (decrease filled count)

HOLOS approach to Sudoku:
1. Forward wave: Expand from puzzle by placing valid digits
2. Backward wave: Generate solved grids, remove digits toward puzzle
3. Connection: When waves meet, we have a solution path

This is different from traditional Sudoku solvers (constraint propagation,
backtracking) - HOLOS explores the full state space bidirectionally.
"""

import random
from typing import List, Tuple, Optional, Any, Dict, Set
from dataclasses import dataclass
from collections import defaultdict

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from holos.holos import GameInterface


# ============================================================
# SUDOKU STATE
# ============================================================

class SudokuState:
    """
    Compact Sudoku state representation.

    Grid is stored as a tuple of 81 integers (0-9).
    0 = empty cell, 1-9 = placed digit.
    Index: row * 9 + col (row-major order).
    """
    __slots__ = ['grid', '_hash', '_conflicts']

    def __init__(self, grid: Tuple[int, ...] = None):
        if grid is None:
            grid = tuple([0] * 81)
        self.grid = grid
        self._hash = None
        self._conflicts = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.grid)
        return self._hash

    def __eq__(self, other):
        return self.grid == other.grid

    def get(self, row: int, col: int) -> int:
        """Get digit at position (0 = empty)"""
        return self.grid[row * 9 + col]

    def set(self, row: int, col: int, digit: int) -> 'SudokuState':
        """Return new state with digit placed at position"""
        idx = row * 9 + col
        new_grid = list(self.grid)
        new_grid[idx] = digit
        return SudokuState(tuple(new_grid))

    def filled_count(self) -> int:
        """Number of filled cells"""
        return sum(1 for d in self.grid if d > 0)

    def empty_cells(self) -> List[Tuple[int, int]]:
        """List of (row, col) for empty cells"""
        return [(i // 9, i % 9) for i, d in enumerate(self.grid) if d == 0]

    def filled_cells(self) -> List[Tuple[int, int, int]]:
        """List of (row, col, digit) for filled cells"""
        return [(i // 9, i % 9, d) for i, d in enumerate(self.grid) if d > 0]

    def get_row(self, row: int) -> List[int]:
        """Get all digits in a row"""
        return [self.grid[row * 9 + c] for c in range(9)]

    def get_col(self, col: int) -> List[int]:
        """Get all digits in a column"""
        return [self.grid[r * 9 + col] for r in range(9)]

    def get_box(self, row: int, col: int) -> List[int]:
        """Get all digits in the 3x3 box containing (row, col)"""
        box_r, box_c = (row // 3) * 3, (col // 3) * 3
        return [self.grid[(box_r + r) * 9 + (box_c + c)]
                for r in range(3) for c in range(3)]

    def get_candidates(self, row: int, col: int) -> Set[int]:
        """Get valid digits that can be placed at (row, col)"""
        if self.get(row, col) != 0:
            return set()

        used = set(self.get_row(row)) | set(self.get_col(col)) | set(self.get_box(row, col))
        return set(range(1, 10)) - used

    def has_conflicts(self) -> bool:
        """Check if grid has any constraint violations"""
        if self._conflicts is not None:
            return self._conflicts

        # Check rows
        for r in range(9):
            row = [d for d in self.get_row(r) if d > 0]
            if len(row) != len(set(row)):
                self._conflicts = True
                return True

        # Check columns
        for c in range(9):
            col = [d for d in self.get_col(c) if d > 0]
            if len(col) != len(set(col)):
                self._conflicts = True
                return True

        # Check boxes
        for br in range(3):
            for bc in range(3):
                box = [d for d in self.get_box(br * 3, bc * 3) if d > 0]
                if len(box) != len(set(box)):
                    self._conflicts = True
                    return True

        self._conflicts = False
        return False

    def is_complete(self) -> bool:
        """Is the grid completely filled?"""
        return self.filled_count() == 81

    def is_solved(self) -> bool:
        """Is the grid completely and correctly filled?"""
        return self.is_complete() and not self.has_conflicts()

    def display(self) -> str:
        """Return string representation of grid"""
        lines = []
        for r in range(9):
            if r % 3 == 0:
                lines.append("+-------+-------+-------+")
            row_str = "| "
            for c in range(9):
                d = self.get(r, c)
                row_str += str(d) if d > 0 else "."
                row_str += " "
                if c % 3 == 2:
                    row_str += "| "
            lines.append(row_str)
        lines.append("+-------+-------+-------+")
        return "\n".join(lines)

    def to_string(self) -> str:
        """Convert to 81-character string"""
        return ''.join(str(d) if d > 0 else '.' for d in self.grid)

    @staticmethod
    def from_string(s: str) -> 'SudokuState':
        """Create from 81-character string (. or 0 for empty)"""
        grid = []
        for c in s:
            if c == '.' or c == '0':
                grid.append(0)
            elif c.isdigit():
                grid.append(int(c))
        if len(grid) != 81:
            raise ValueError(f"Expected 81 characters, got {len(grid)}")
        return SudokuState(tuple(grid))


# ============================================================
# SUDOKU VALUE
# ============================================================

@dataclass(frozen=True)
class SudokuValue:
    """Sudoku solution value"""
    solved: bool  # True if this state leads to/is a solution

    def __repr__(self):
        return "Solved" if self.solved else "Unsolved"


# ============================================================
# SUDOKU FEATURES (Equivalence Classes)
# ============================================================

@dataclass(frozen=True)
class SudokuFeatures:
    """
    Equivalence class features for Sudoku.

    Positions with same features may have similar solving properties.
    """
    filled_count: int
    row_fill_profile: Tuple[int, ...]  # Sorted fill counts per row
    col_fill_profile: Tuple[int, ...]  # Sorted fill counts per col
    min_candidates: int  # Minimum candidates for any empty cell
    has_naked_single: bool  # Any cell with exactly 1 candidate?

    def __hash__(self):
        return hash((self.filled_count, self.row_fill_profile,
                     self.col_fill_profile, self.min_candidates,
                     self.has_naked_single))


def extract_features(state: SudokuState) -> SudokuFeatures:
    """Extract equivalence features from state"""
    # Row and column fill profiles
    row_fills = tuple(sorted(sum(1 for d in state.get_row(r) if d > 0) for r in range(9)))
    col_fills = tuple(sorted(sum(1 for d in state.get_col(c) if d > 0) for c in range(9)))

    # Candidate analysis
    min_cands = 10
    has_naked = False
    for r, c in state.empty_cells():
        cands = len(state.get_candidates(r, c))
        if cands < min_cands:
            min_cands = cands
        if cands == 1:
            has_naked = True

    if min_cands == 10:
        min_cands = 0  # No empty cells

    return SudokuFeatures(
        filled_count=state.filled_count(),
        row_fill_profile=row_fills,
        col_fill_profile=col_fills,
        min_candidates=min_cands,
        has_naked_single=has_naked
    )


# ============================================================
# SUDOKU GAME INTERFACE
# ============================================================

class SudokuGame(GameInterface[SudokuState, SudokuValue]):
    """
    Sudoku interface for HOLOS.

    Boundaries:
    - Lower: Completed valid grids (81 cells filled, no conflicts)
    - Upper: Starting puzzle (or any partial state)

    Unlike chess (Syzygy) or Connect-4 (terminal wins), Sudoku's boundary
    is simply "completely filled and valid".
    """

    def __init__(self, min_filled: int = 0, max_filled: int = 81,
                 full_expansion: bool = False, max_successors: int = 20):
        """
        Initialize Sudoku game.

        Args:
            min_filled: Minimum filled cells to consider (for bounded search)
            max_filled: Maximum filled cells (usually 81)
            full_expansion: If True, return successors for ALL empty cells (not just MRV)
            max_successors: Maximum successors to return (for full_expansion mode)
        """
        self.min_filled = min_filled
        self.max_filled = max_filled
        self.full_expansion = full_expansion
        self.max_successors = max_successors

    def hash_state(self, state: SudokuState) -> int:
        """Hash a state for deduplication"""
        return hash(state)

    def get_successors(self, state: SudokuState) -> List[Tuple[SudokuState, Tuple[int, int, int]]]:
        """
        Get successor states by placing valid digits.
        Move is (row, col, digit).

        If full_expansion=False (default): Uses MRV (Minimum Remaining Values)
        heuristic - fill cells with fewest candidates first.

        If full_expansion=True: Returns successors for ALL empty cells,
        up to max_successors. Better for HOLOS wave exploration.
        """
        if state.filled_count() >= self.max_filled:
            return []

        if state.is_complete() or state.has_conflicts():
            return []

        # Find empty cells with their candidate counts
        cells_with_cands = []
        for r, c in state.empty_cells():
            cands = state.get_candidates(r, c)
            if len(cands) == 0:
                return []  # Dead end - no valid placements
            cells_with_cands.append((len(cands), r, c, cands))

        if not cells_with_cands:
            return []

        # MRV: Sort by candidate count
        cells_with_cands.sort()

        successors = []

        if self.full_expansion:
            # Full expansion: try all cells (up to limit)
            for _, r, c, candidates in cells_with_cands:
                for digit in sorted(candidates):
                    child = state.set(r, c, digit)
                    successors.append((child, (r, c, digit)))
                    if len(successors) >= self.max_successors:
                        return successors
        else:
            # MRV: Pick only cell with fewest candidates
            _, r, c, candidates = cells_with_cands[0]
            for digit in sorted(candidates):
                child = state.set(r, c, digit)
                successors.append((child, (r, c, digit)))

        return successors

    def get_predecessors(self, state: SudokuState) -> List[Tuple[SudokuState, Tuple[int, int, int]]]:
        """
        Get predecessor states by removing digits.

        For backward search: given a more-filled state, what states
        could have led to it by placing one digit?
        """
        if state.filled_count() <= self.min_filled:
            return []

        predecessors = []
        for r, c, digit in state.filled_cells():
            # Remove this digit
            pred = state.set(r, c, 0)

            # Verify the digit could have been legally placed
            # (it should be a valid candidate in the predecessor)
            if digit in pred.get_candidates(r, c):
                predecessors.append((pred, (r, c, digit)))

        return predecessors

    def is_boundary(self, state: SudokuState) -> bool:
        """
        Is this a boundary state?
        For Sudoku: completely filled valid grids are boundaries.
        """
        return state.is_solved()

    def get_boundary_value(self, state: SudokuState) -> Optional[SudokuValue]:
        """Get value for boundary (solved) state"""
        if not state.is_solved():
            return None
        return SudokuValue(solved=True)

    def is_terminal(self, state: SudokuState) -> Tuple[bool, Optional[SudokuValue]]:
        """
        Check if state is terminal.

        Terminal states:
        - Solved grid (success)
        - Grid with conflicts (failure)
        - Grid with empty cell but no candidates (failure)
        """
        if state.is_solved():
            return True, SudokuValue(solved=True)

        if state.has_conflicts():
            return True, SudokuValue(solved=False)

        # Check for dead ends (empty cell with no candidates)
        for r, c in state.empty_cells():
            if len(state.get_candidates(r, c)) == 0:
                return True, SudokuValue(solved=False)

        return False, None

    def propagate_value(self, state: SudokuState,
                        child_values: List[SudokuValue]) -> Optional[SudokuValue]:
        """
        Propagate values from children to parent.

        For Sudoku: if ANY child leads to a solution, parent can reach solution.
        (Unlike minimax games, Sudoku is single-player - we just need one path)
        """
        if not child_values:
            return None

        # If any child is solved, parent can reach a solution
        if any(cv.solved for cv in child_values):
            return SudokuValue(solved=True)

        return None

    def get_features(self, state: SudokuState) -> SudokuFeatures:
        """Extract equivalence features"""
        return extract_features(state)

    def get_signature(self, state: SudokuState) -> str:
        """
        Get signature for goal matching.
        Returns filled count signature like "F45" (45 filled cells).
        """
        return f"F{state.filled_count()}"

    def get_lightning_successors(self, state: SudokuState) -> List[Tuple[SudokuState, Tuple[int, int, int]]]:
        """
        For lightning mode: only consider "forced" moves (naked singles).

        Naked single = cell with exactly one candidate.
        """
        if state.is_complete() or state.has_conflicts():
            return []

        successors = []
        for r, c in state.empty_cells():
            cands = state.get_candidates(r, c)
            if len(cands) == 1:
                digit = list(cands)[0]
                child = state.set(r, c, digit)
                successors.append((child, (r, c, digit)))

        # If no naked singles, return regular successors (limited)
        if not successors:
            return self.get_successors(state)[:3]

        return successors

    def get_lightning_predecessors(self, state: SudokuState) -> List[Tuple[SudokuState, Tuple[int, int, int]]]:
        """For backward lightning: same as regular predecessors (limited)"""
        return self.get_predecessors(state)[:5]

    def score_for_lightning(self, state: SudokuState, move: Tuple[int, int, int]) -> float:
        """Score a move for lightning prioritization"""
        r, c, digit = move
        # Prefer moves that create more naked singles
        child = state.set(r, c, digit)
        naked_count = sum(1 for er, ec in child.empty_cells()
                         if len(child.get_candidates(er, ec)) == 1)
        return naked_count

    def apply_move(self, state: SudokuState, move: Tuple[int, int, int]) -> SudokuState:
        """Apply a move (place digit) to get successor state"""
        r, c, digit = move
        return state.set(r, c, digit)

    def generate_boundary_seeds(self, template: SudokuState, count: int = 100) -> List[SudokuState]:
        """
        Generate solved Sudoku grids for backward wave seeding.

        Strategy: Generate random valid completed Sudoku grids.
        These serve as the "boundary" for backward wave expansion.
        """
        solutions = []
        seen = set()

        for _ in range(count * 20):
            if len(solutions) >= count:
                break

            solution = self._generate_random_solution()
            if solution is not None:
                h = hash(solution)
                if h not in seen:
                    seen.add(h)
                    solutions.append(solution)

        print(f"Generated {len(solutions)} solved grids for backward seeding")
        return solutions

    def _generate_random_solution(self) -> Optional[SudokuState]:
        """Generate a random valid solved Sudoku grid"""
        state = SudokuState()

        # Fill grid using backtracking with random choices
        def fill(s: SudokuState) -> Optional[SudokuState]:
            empty = s.empty_cells()
            if not empty:
                return s if s.is_solved() else None

            # Pick cell with fewest candidates (MRV)
            cells = [(len(s.get_candidates(r, c)), r, c) for r, c in empty]
            cells.sort()
            _, r, c = cells[0]

            candidates = list(s.get_candidates(r, c))
            if not candidates:
                return None

            random.shuffle(candidates)
            for digit in candidates:
                child = s.set(r, c, digit)
                result = fill(child)
                if result is not None:
                    return result

            return None

        return fill(state)


# ============================================================
# PUZZLE GENERATION AND UTILITIES
# ============================================================

def generate_puzzle(num_clues: int = 30, max_attempts: int = 100) -> Optional[SudokuState]:
    """
    Generate a Sudoku puzzle with given number of clues.

    Strategy:
    1. Generate a solved grid
    2. Remove digits while maintaining unique solution
    """
    game = SudokuGame()

    for _ in range(max_attempts):
        # Generate solved grid
        solution = game._generate_random_solution()
        if solution is None:
            continue

        # Remove digits to create puzzle
        puzzle = solution
        cells_to_remove = 81 - num_clues
        removed = 0

        # Get all filled positions
        positions = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(positions)

        for r, c in positions:
            if removed >= cells_to_remove:
                break

            digit = puzzle.get(r, c)
            if digit == 0:
                continue

            # Remove digit
            test_puzzle = puzzle.set(r, c, 0)

            # Check if still has unique solution (simplified check)
            # Full uniqueness check is expensive - we just verify solvability
            if _is_solvable(test_puzzle):
                puzzle = test_puzzle
                removed += 1

        if puzzle.filled_count() == num_clues:
            return puzzle

    return None


def _is_solvable(state: SudokuState) -> bool:
    """Quick check if puzzle is solvable (simple backtracking)"""
    def solve(s: SudokuState) -> bool:
        if s.is_solved():
            return True

        empty = s.empty_cells()
        if not empty:
            return False

        # MRV
        cells = [(len(s.get_candidates(r, c)), r, c) for r, c in empty]
        cells.sort()
        _, r, c = cells[0]

        for digit in s.get_candidates(r, c):
            child = s.set(r, c, digit)
            if solve(child):
                return True

        return False

    return solve(state)


def solve_sudoku(puzzle: SudokuState) -> Optional[SudokuState]:
    """Solve a Sudoku puzzle using simple backtracking (for verification)"""
    def solve(s: SudokuState) -> Optional[SudokuState]:
        if s.is_solved():
            return s

        empty = s.empty_cells()
        if not empty:
            return None

        # MRV
        cells = [(len(s.get_candidates(r, c)), r, c) for r, c in empty]
        cells.sort()
        _, r, c = cells[0]

        for digit in s.get_candidates(r, c):
            child = s.set(r, c, digit)
            result = solve(child)
            if result is not None:
                return result

        return None

    return solve(puzzle)


# ============================================================
# SAMPLE PUZZLES
# ============================================================

# Easy puzzle (38 clues)
EASY_PUZZLE = """
530070000
600195000
098000060
800060003
400803001
700020006
060000280
000419005
000080079
""".replace('\n', '')

# Medium puzzle (28 clues)
MEDIUM_PUZZLE = """
000260701
680070090
190004500
820100040
004602900
050003028
009300074
040050036
703018000
""".replace('\n', '')

# Hard puzzle (24 clues)
HARD_PUZZLE = """
000000000
000003085
001020000
000507000
004000100
090000000
500000073
002010000
000040009
""".replace('\n', '')


def get_sample_puzzles() -> Dict[str, SudokuState]:
    """Get sample puzzles of varying difficulty"""
    return {
        'easy': SudokuState.from_string(EASY_PUZZLE),
        'medium': SudokuState.from_string(MEDIUM_PUZZLE),
        'hard': SudokuState.from_string(HARD_PUZZLE),
    }
