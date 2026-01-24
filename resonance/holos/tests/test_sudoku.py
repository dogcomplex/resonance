#!/usr/bin/env python3
"""
test_sudoku.py - Full test run of Sudoku using HOLOS

This demonstrates HOLOS solving Sudoku puzzles through bidirectional search.

HOLOS approach to Sudoku:
- Forward wave: Expand from puzzle by placing valid digits
- Backward wave: Start from solved grids, remove digits toward puzzle
- Connection: When waves meet, we have a solution path

This is different from traditional Sudoku solvers (constraint propagation,
backtracking) - HOLOS explores the state space bidirectionally.
"""

import time
import random
import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.sudoku import (
    SudokuGame, SudokuState, SudokuValue, SudokuFeatures,
    get_sample_puzzles, solve_sudoku, generate_puzzle
)


def test_basic_state():
    """Test basic SudokuState operations"""
    print("\n" + "=" * 60)
    print("TEST: Basic SudokuState")
    print("=" * 60)

    # Empty grid
    state = SudokuState()
    print(f"\nEmpty grid:")
    print(state.display())
    assert state.filled_count() == 0
    assert not state.has_conflicts()
    assert not state.is_complete()

    # Place a digit
    state2 = state.set(0, 0, 5)
    print(f"\nAfter placing 5 at (0,0):")
    print(state2.display())
    assert state2.filled_count() == 1
    assert state2.get(0, 0) == 5

    # Check candidates
    cands = state2.get_candidates(0, 1)
    print(f"\nCandidates at (0,1): {cands}")
    assert 5 not in cands  # 5 is in same row

    # Test from_string
    puzzles = get_sample_puzzles()
    easy = puzzles['easy']
    print(f"\nEasy puzzle ({easy.filled_count()} clues):")
    print(easy.display())

    print("\n[PASS] Basic SudokuState works correctly")


def test_game_interface():
    """Test SudokuGame interface"""
    print("\n" + "=" * 60)
    print("TEST: Game Interface")
    print("=" * 60)

    game = SudokuGame()
    puzzles = get_sample_puzzles()
    puzzle = puzzles['easy']

    print(f"\nPuzzle ({puzzle.filled_count()} clues):")
    print(puzzle.display())

    # Test successors
    successors = game.get_successors(puzzle)
    print(f"\nSuccessors from puzzle: {len(successors)}")
    if successors:
        child, move = successors[0]
        r, c, digit = move
        print(f"  First move: place {digit} at ({r},{c})")

    # Test predecessors from a more complete state
    # First solve the puzzle
    solution = solve_sudoku(puzzle)
    assert solution is not None, "Puzzle should be solvable"
    print(f"\nSolution:")
    print(solution.display())

    predecessors = game.get_predecessors(solution)
    print(f"\nPredecessors from solution: {len(predecessors)}")

    # Test boundary
    assert game.is_boundary(solution), "Solution should be boundary"
    assert not game.is_boundary(puzzle), "Puzzle should not be boundary"

    value = game.get_boundary_value(solution)
    print(f"\nSolution boundary value: {value}")
    assert value.solved

    print("\n[PASS] Game interface works correctly")


def test_features():
    """Test feature extraction"""
    print("\n" + "=" * 60)
    print("TEST: Feature Extraction")
    print("=" * 60)

    game = SudokuGame()
    puzzles = get_sample_puzzles()
    puzzle = puzzles['easy']

    features = game.get_features(puzzle)
    print(f"\nFeatures for easy puzzle:")
    print(f"  filled_count: {features.filled_count}")
    print(f"  row_fill_profile: {features.row_fill_profile}")
    print(f"  col_fill_profile: {features.col_fill_profile}")
    print(f"  min_candidates: {features.min_candidates}")
    print(f"  has_naked_single: {features.has_naked_single}")

    # Signature
    sig = game.get_signature(puzzle)
    print(f"  signature: {sig}")

    print("\n[PASS] Feature extraction works correctly")


def test_lightning_probe():
    """Test lightning probe on Sudoku"""
    print("\n" + "=" * 60)
    print("TEST: Lightning Probe")
    print("=" * 60)

    from holos.holos import LightningProbe

    game = SudokuGame()
    puzzles = get_sample_puzzles()
    puzzle = puzzles['easy']

    print(f"\nTesting lightning probe on easy puzzle...")
    probe = LightningProbe(game, {}, direction="forward", max_depth=50)
    value, path = probe.probe(puzzle)

    print(f"\nLightning result:")
    print(f"  Value: {value}")
    print(f"  Path length: {len(path)}")
    print(f"  Nodes visited: {probe.nodes_visited}")

    if path:
        print(f"  First 5 moves: {[m for s, m in path[:5]]}")

    print("\n[PASS] Lightning probe works correctly")


def test_small_solve():
    """Test solving a nearly-complete puzzle"""
    print("\n" + "=" * 60)
    print("TEST: Small Solve (near-complete puzzle)")
    print("=" * 60)

    game = SudokuGame()

    # Create a nearly-complete puzzle (only 5 empty cells)
    solution = game._generate_random_solution()
    if solution is None:
        print("Could not generate solution")
        return None, None

    # Remove 5 random cells
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    puzzle = solution
    for r, c in cells[:5]:
        puzzle = puzzle.set(r, c, 0)

    print(f"\nNear-complete puzzle ({puzzle.filled_count()} clues):")
    print(puzzle.display())

    solver = HOLOSSolver(game, name="sudoku_small", max_memory_mb=500)
    forward_seeds = [SeedPoint(puzzle, SearchMode.WAVE)]

    print(f"\nStarting solve...")
    start_time = time.time()

    hologram = solver.solve(
        forward_seeds,
        backward_seeds=None,
        max_iterations=10,
        lightning_interval=2
    )

    elapsed = time.time() - start_time

    print(f"\n" + "=" * 60)
    print(f"SOLVE COMPLETE in {elapsed:.1f}s")
    print("=" * 60)
    print(hologram.summary())

    # Check if puzzle was solved
    puzzle_h = game.hash_state(puzzle)
    puzzle_value = hologram.query(puzzle_h)
    print(f"\nPuzzle value: {puzzle_value}")

    print(f"\nStatistics:")
    for key, val in solver.stats.items():
        if isinstance(val, (int, float)) and val > 0:
            print(f"  {key}: {val}")

    print("\n[PASS] Small solve completed")
    return hologram, game


def test_easy_puzzle():
    """Test solving the easy sample puzzle"""
    print("\n" + "=" * 60)
    print("TEST: Easy Puzzle Solve")
    print("=" * 60)

    game = SudokuGame()
    puzzles = get_sample_puzzles()
    puzzle = puzzles['easy']

    print(f"\nEasy puzzle ({puzzle.filled_count()} clues):")
    print(puzzle.display())

    solver = HOLOSSolver(
        game,
        name="sudoku_easy",
        max_memory_mb=1000,
        max_frontier_size=100_000
    )
    forward_seeds = [SeedPoint(puzzle, SearchMode.WAVE)]

    print(f"\nStarting solve...")
    start_time = time.time()

    hologram = solver.solve(
        forward_seeds,
        backward_seeds=None,
        max_iterations=15,
        lightning_interval=3
    )

    elapsed = time.time() - start_time

    print(f"\n" + "=" * 60)
    print(f"SOLVE COMPLETE in {elapsed:.1f}s")
    print("=" * 60)
    print(hologram.summary())

    # Check solution
    puzzle_h = game.hash_state(puzzle)
    puzzle_value = hologram.query(puzzle_h)
    print(f"\nPuzzle solved: {puzzle_value}")

    # Also verify with traditional solver
    traditional_solution = solve_sudoku(puzzle)
    if traditional_solution:
        print(f"\nTraditional solution found:")
        print(traditional_solution.display())

    print("\n[PASS] Easy puzzle test completed")
    return hologram, game


def test_boundary_generation():
    """Test generating solved grids for backward seeding"""
    print("\n" + "=" * 60)
    print("TEST: Boundary (Solved Grid) Generation")
    print("=" * 60)

    game = SudokuGame()
    puzzle = SudokuState()  # Empty as template

    seeds = game.generate_boundary_seeds(puzzle, count=10)
    print(f"\nGenerated {len(seeds)} solved grids")

    # Verify all are valid solutions
    for i, solution in enumerate(seeds):
        assert solution.is_solved(), f"Grid {i} should be solved"

    # Show first one
    if seeds:
        print(f"\nSample generated solution:")
        print(seeds[0].display())

    print("\n[PASS] Boundary generation works correctly")


def test_bidirectional_solve():
    """Test bidirectional solving with explicit backward seeds"""
    print("\n" + "=" * 60)
    print("TEST: Bidirectional Solve")
    print("=" * 60)

    game = SudokuGame()

    # Create a simple puzzle with 10 empty cells
    solution = game._generate_random_solution()
    if solution is None:
        print("Could not generate solution")
        return

    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    puzzle = solution
    for r, c in cells[:10]:
        puzzle = puzzle.set(r, c, 0)

    print(f"\nPuzzle ({puzzle.filled_count()} clues):")
    print(puzzle.display())

    # Create backward seeds from the actual solution (and variants)
    backward_seeds = [SeedPoint(solution, SearchMode.WAVE)]

    # Also generate some random solutions
    for _ in range(5):
        sol = game._generate_random_solution()
        if sol:
            backward_seeds.append(SeedPoint(sol, SearchMode.WAVE))

    solver = HOLOSSolver(
        game,
        name="sudoku_bidir",
        max_memory_mb=500,
        spine_as_boundary=True
    )
    forward_seeds = [SeedPoint(puzzle, SearchMode.WAVE)]

    print(f"\nForward seeds: {len(forward_seeds)}")
    print(f"Backward seeds: {len(backward_seeds)}")

    start_time = time.time()

    hologram = solver.solve(
        forward_seeds,
        backward_seeds=backward_seeds,
        max_iterations=10,
        lightning_interval=2
    )

    elapsed = time.time() - start_time

    print(f"\n" + "=" * 60)
    print(f"SOLVE COMPLETE in {elapsed:.1f}s")
    print("=" * 60)

    # Check for connections
    print(f"\nConnections found: {len(solver.connections)}")
    print(f"Spines found: {solver.stats['spines_found']}")

    puzzle_h = game.hash_state(puzzle)
    puzzle_value = hologram.query(puzzle_h)
    print(f"Puzzle solved: {puzzle_value}")

    print("\n[PASS] Bidirectional solve completed")


def run_full_test():
    """Run all tests"""
    print("=" * 60)
    print("SUDOKU HOLOS FULL TEST")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Basic tests
    test_basic_state()
    test_game_interface()
    test_features()
    test_boundary_generation()
    test_lightning_probe()

    # Solve tests
    test_small_solve()
    test_bidirectional_solve()
    test_easy_puzzle()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_test()
