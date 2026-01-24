#!/usr/bin/env python3
"""
test_osmosis.py - Test the Osmosis mode for HOLOS

Osmosis mode implements "careful bilateral exploration":
- Expands ONE frontier state at a time
- Always picks the state with highest certainty/information
- Naturally balances forward/backward based on gradient
- Like osmosis in biology - flow driven by concentration difference

Physical analogy to other modes:
- Lightning: electrical discharge (fast, direct path)
- Wave: water waves (uniform BFS expansion)
- Crystal: crystallization (grows from nucleation points)
- Osmosis: diffusion (selective, gradient-driven flow)

This test compares osmosis against regular wave mode on:
1. Sudoku puzzles
2. Connect4 positions
"""

import time
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.sudoku import SudokuGame, SudokuState, get_sample_puzzles
from holos.games.connect4 import Connect4Game, C4State


def test_osmosis_sudoku_near_complete():
    """Test osmosis on a nearly-complete Sudoku puzzle"""
    print("\n" + "=" * 60)
    print("TEST: Osmosis on Near-Complete Sudoku")
    print("=" * 60)

    game = SudokuGame()

    # Generate a nearly-complete puzzle (5 empty cells)
    solution = game._generate_random_solution()
    if solution is None:
        print("Could not generate solution")
        return None

    # Remove 5 random cells
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    puzzle = solution
    for r, c in cells[:5]:
        puzzle = puzzle.set(r, c, 0)

    print(f"\nPuzzle ({puzzle.filled_count()} clues):")
    print(puzzle.display())

    solver = HOLOSSolver(game, name="sudoku_osmosis", max_memory_mb=500)
    forward_seeds = [SeedPoint(puzzle, SearchMode.OSMOSIS)]

    print("\nStarting OSMOSIS solve...")
    start_time = time.time()

    hologram = solver.solve_osmosis(
        forward_seeds,
        backward_seeds=None,
        max_steps=1000,
        verbose=True
    )

    elapsed = time.time() - start_time

    # Check result
    puzzle_h = game.hash_state(puzzle)
    puzzle_value = hologram.query(puzzle_h)
    print(f"\nPuzzle hash: {puzzle_h}")
    print(f"Puzzle in solved: {puzzle_h in hologram.solved}")
    print(f"Puzzle value: {puzzle_value}")
    print(f"Total solved: {len(hologram.solved)}")
    if hologram.solved:
        sample_values = list(hologram.solved.values())[:5]
        print(f"Sample values: {sample_values}")
    print(f"Time: {elapsed:.2f}s")

    print("\n[PASS] Osmosis Sudoku test completed")
    return hologram


def test_osmosis_sudoku_easy():
    """Test osmosis on easy sample puzzle"""
    print("\n" + "=" * 60)
    print("TEST: Osmosis on Easy Sudoku")
    print("=" * 60)

    game = SudokuGame()
    puzzles = get_sample_puzzles()
    puzzle = puzzles['easy']

    print(f"\nEasy puzzle ({puzzle.filled_count()} clues):")
    print(puzzle.display())

    solver = HOLOSSolver(game, name="sudoku_easy_osmosis", max_memory_mb=1000)
    forward_seeds = [SeedPoint(puzzle, SearchMode.OSMOSIS)]

    print("\nStarting OSMOSIS solve...")
    start_time = time.time()

    hologram = solver.solve_osmosis(
        forward_seeds,
        backward_seeds=None,
        max_steps=5000,
        verbose=True
    )

    elapsed = time.time() - start_time

    puzzle_h = game.hash_state(puzzle)
    puzzle_value = hologram.query(puzzle_h)
    print(f"\nPuzzle solved: {puzzle_value}")
    print(f"Time: {elapsed:.2f}s")

    print("\n[PASS] Easy Sudoku osmosis test completed")
    return hologram


def test_osmosis_connect4_simple():
    """Test osmosis on simple Connect4 position"""
    print("\n" + "=" * 60)
    print("TEST: Osmosis on Simple Connect4")
    print("=" * 60)

    game = Connect4Game()

    # Create a position close to terminal
    # X plays first, we'll create a position with 6 moves made
    state = C4State()

    # Play some moves: X at 3, O at 2, X at 3, O at 2, X at 3 (X threatens win)
    moves = [3, 2, 3, 2, 3, 2]  # Alternating columns
    for col in moves:
        state = state.play(col)

    print(f"\nConnect4 position after {len(moves)} moves:")
    print(state.display())
    print(f"Turn: {state.turn}")

    solver = HOLOSSolver(game, name="c4_osmosis", max_memory_mb=500)
    forward_seeds = [SeedPoint(state, SearchMode.OSMOSIS)]

    print("\nStarting OSMOSIS solve...")
    start_time = time.time()

    hologram = solver.solve_osmosis(
        forward_seeds,
        backward_seeds=None,
        max_steps=2000,
        verbose=True
    )

    elapsed = time.time() - start_time

    state_h = game.hash_state(state)
    state_value = hologram.query(state_h)
    print(f"\nPosition value: {state_value}")
    print(f"Time: {elapsed:.2f}s")

    print("\n[PASS] Connect4 osmosis test completed")
    return hologram


def compare_osmosis_vs_wave():
    """Compare osmosis against regular wave mode"""
    print("\n" + "=" * 60)
    print("COMPARISON: Osmosis vs Wave Mode")
    print("=" * 60)

    game = SudokuGame()

    # Generate test puzzle
    solution = game._generate_random_solution()
    if solution is None:
        print("Could not generate solution")
        return

    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    puzzle = solution
    for r, c in cells[:8]:  # 8 empty cells
        puzzle = puzzle.set(r, c, 0)

    print(f"\nTest puzzle ({puzzle.filled_count()} clues):")
    print(puzzle.display())

    # Test WAVE mode
    print("\n--- WAVE MODE ---")
    solver_wave = HOLOSSolver(game, name="wave_test", max_memory_mb=500)
    forward_seeds = [SeedPoint(puzzle, SearchMode.WAVE)]

    start_wave = time.time()
    hologram_wave = solver_wave.solve(
        forward_seeds,
        backward_seeds=None,
        max_iterations=20,
        lightning_interval=5
    )
    wave_time = time.time() - start_wave

    puzzle_h = game.hash_state(puzzle)
    wave_solved = puzzle_h in hologram_wave.solved
    wave_expanded = solver_wave.stats['forward_expanded'] + solver_wave.stats['backward_expanded']

    print(f"Wave: Solved={wave_solved}, Time={wave_time:.2f}s, Expanded={wave_expanded:,}")

    # Test OSMOSIS mode
    print("\n--- OSMOSIS MODE ---")
    solver_osmosis = HOLOSSolver(game, name="osmosis_test", max_memory_mb=500)
    forward_seeds = [SeedPoint(puzzle, SearchMode.OSMOSIS)]

    start_osmosis = time.time()
    hologram_osmosis = solver_osmosis.solve_osmosis(
        forward_seeds,
        backward_seeds=None,
        max_steps=5000,
        verbose=False
    )
    osmosis_time = time.time() - start_osmosis

    osmosis_solved = puzzle_h in hologram_osmosis.solved
    osmosis_stats = hologram_osmosis.stats.get('osmosis', {})
    osmosis_steps = osmosis_stats.get('forward_steps', 0) + osmosis_stats.get('backward_steps', 0)

    print(f"Osmosis: Solved={osmosis_solved}, Time={osmosis_time:.2f}s, Steps={osmosis_steps:,}")

    # Summary
    print("\n--- SUMMARY ---")
    print(f"Wave:    {'SOLVED' if wave_solved else 'UNSOLVED'} in {wave_time:.2f}s ({wave_expanded:,} expansions)")
    print(f"Osmosis: {'SOLVED' if osmosis_solved else 'UNSOLVED'} in {osmosis_time:.2f}s ({osmosis_steps:,} steps)")

    if wave_solved and osmosis_solved:
        efficiency_wave = wave_expanded / wave_time if wave_time > 0 else 0
        efficiency_osmosis = osmosis_steps / osmosis_time if osmosis_time > 0 else 0
        print(f"\nEfficiency (operations/second):")
        print(f"  Wave: {efficiency_wave:.0f}")
        print(f"  Osmosis: {efficiency_osmosis:.0f}")
        print(f"\nOsmosis uses fewer operations but may take longer per step")
        print(f"Wave is breadth-first, osmosis is 'best-first bilateral'")

    print("\n[PASS] Comparison completed")


def run_all_tests():
    """Run all osmosis tests"""
    print("=" * 60)
    print("OSMOSIS MODE FULL TEST SUITE")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run tests
    test_osmosis_sudoku_near_complete()
    test_osmosis_connect4_simple()
    compare_osmosis_vs_wave()

    # Optional: harder tests (comment out if too slow)
    # test_osmosis_sudoku_easy()

    print("\n" + "=" * 60)
    print("ALL OSMOSIS TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "sudoku":
            test_osmosis_sudoku_near_complete()
        elif test_name == "sudoku_easy":
            test_osmosis_sudoku_easy()
        elif test_name == "connect4":
            test_osmosis_connect4_simple()
        elif test_name == "compare":
            compare_osmosis_vs_wave()
        else:
            print(f"Unknown test: {test_name}")
            print("Available: sudoku, sudoku_easy, connect4, compare")
    else:
        run_all_tests()
