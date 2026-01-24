"""
holos/test_sudoku_seeds.py - Seed-Based Sudoku Solver Testing

This tests the full seed-based approach for Sudoku:
1. Solve puzzles and capture solution paths as seeds
2. Store seeds (not full solution space)
3. Reconstruct solutions from seeds
4. Measure compression ratio and speedup

Goal: Demonstrate that storing seeds + reconstruction algorithm
is more efficient than storing all positions.
"""

import sys
import os
import time
import pickle
from typing import List, Tuple, Dict, Set, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.sudoku import (
    SudokuGame, SudokuState, SudokuValue,
    get_sample_puzzles, solve_sudoku, generate_puzzle
)
from holos.games.seeds import (
    TacticalSeed, TacticalValue, TacticalSeedGame, SeedDirection
)


# ============================================================
# SEED-BASED SOLUTION STORAGE
# ============================================================

@dataclass
class SudokuSeedSolution:
    """
    A compressed representation of a Sudoku solution.

    Instead of storing all intermediate states, store:
    - The initial puzzle (seed)
    - The move sequence (path)
    - Key checkpoints for fast lookup
    """
    puzzle_hash: int
    puzzle_string: str  # 81-char representation
    moves: List[Tuple[int, int, int]]  # List of (row, col, digit)
    checkpoints: Dict[int, int]  # depth -> hash at that depth
    solution_hash: int

    def storage_size(self) -> int:
        """Estimate storage size in bytes"""
        # puzzle_string: 81 bytes
        # moves: ~4 bytes per move (row, col, digit packed)
        # checkpoints: ~8 bytes per checkpoint
        return 81 + len(self.moves) * 4 + len(self.checkpoints) * 8 + 16

    def __repr__(self):
        return f"SeedSolution(moves={len(self.moves)}, checkpoints={len(self.checkpoints)})"


def solve_and_capture_path(game: SudokuGame, puzzle: SudokuState) -> Tuple[List[Tuple[SudokuState, Any]], bool]:
    """
    Solve puzzle and capture the complete solution path.

    Uses traditional backtracking but records the path.
    """
    path = []

    def solve(state: SudokuState) -> bool:
        if state.is_solved():
            return True

        empty = state.empty_cells()
        if not empty:
            return False

        # MRV heuristic
        cells = [(len(state.get_candidates(r, c)), r, c) for r, c in empty]
        cells.sort()
        _, r, c = cells[0]

        for digit in state.get_candidates(r, c):
            child = state.set(r, c, digit)
            move = (r, c, digit)
            path.append((child, move))

            if solve(child):
                return True

            path.pop()  # Backtrack

        return False

    success = solve(puzzle)
    return path, success


def create_seed_solution(game: SudokuGame, puzzle: SudokuState) -> SudokuSeedSolution:
    """Create a compressed seed solution from puzzle"""
    path, success = solve_and_capture_path(game, puzzle)

    if not success:
        raise ValueError("Puzzle is unsolvable")

    # Extract moves and create checkpoints
    moves = [move for state, move in path]

    # Checkpoint every 10 moves
    checkpoints = {}
    current = puzzle
    for i, (state, move) in enumerate(path):
        if i % 10 == 0:
            checkpoints[i] = game.hash_state(current)
        current = state

    solution_hash = game.hash_state(path[-1][0]) if path else game.hash_state(puzzle)

    return SudokuSeedSolution(
        puzzle_hash=game.hash_state(puzzle),
        puzzle_string=puzzle.to_string(),
        moves=moves,
        checkpoints=checkpoints,
        solution_hash=solution_hash
    )


def reconstruct_from_seed(seed_solution: SudokuSeedSolution) -> Tuple[List[int], int]:
    """
    Reconstruct all position hashes from a seed solution.

    Returns: (list of hashes, count)
    """
    puzzle = SudokuState.from_string(seed_solution.puzzle_string)
    game = SudokuGame()

    hashes = [game.hash_state(puzzle)]
    current = puzzle

    for r, c, digit in seed_solution.moves:
        current = current.set(r, c, digit)
        hashes.append(game.hash_state(current))

    return hashes, len(hashes)


# ============================================================
# COMPREHENSIVE SUDOKU SOLVE TEST
# ============================================================

def test_sudoku_solve_all_methods():
    """
    Test solving Sudoku with multiple methods:
    1. Traditional backtracking
    2. HOLOS osmosis (best for Sudoku)
    3. Seed-based reconstruction
    """
    print("=" * 70)
    print("SUDOKU SOLVE: ALL METHODS COMPARISON")
    print("=" * 70)

    game = SudokuGame()
    puzzles = get_sample_puzzles()

    results = {}

    for name, puzzle in puzzles.items():
        print(f"\n{'='*60}")
        print(f"PUZZLE: {name.upper()} ({puzzle.filled_count()} clues)")
        print("=" * 60)
        print(puzzle.display())

        puzzle_results = {'clues': puzzle.filled_count()}

        # Method 1: Traditional backtracking
        print("\n--- Traditional Backtracking ---")
        start = time.time()
        solution = solve_sudoku(puzzle)
        elapsed = time.time() - start
        print(f"Time: {elapsed:.4f}s")
        print(f"Solved: {solution is not None}")
        puzzle_results['traditional'] = {
            'time': elapsed,
            'solved': solution is not None
        }

        # Method 2: Capture path as seed
        print("\n--- Seed Capture ---")
        start = time.time()
        seed_sol = create_seed_solution(game, puzzle)
        elapsed = time.time() - start
        print(f"Time: {elapsed:.4f}s")
        print(f"Path length: {len(seed_sol.moves)} moves")
        print(f"Checkpoints: {len(seed_sol.checkpoints)}")
        print(f"Storage: {seed_sol.storage_size()} bytes")
        puzzle_results['seed_capture'] = {
            'time': elapsed,
            'path_length': len(seed_sol.moves),
            'storage_bytes': seed_sol.storage_size()
        }

        # Method 3: Reconstruct from seed
        print("\n--- Seed Reconstruction ---")
        start = time.time()
        hashes, count = reconstruct_from_seed(seed_sol)
        elapsed = time.time() - start
        print(f"Time: {elapsed:.6f}s")
        print(f"Positions reconstructed: {count}")
        full_storage = count * 16
        print(f"Full storage would be: {full_storage} bytes")
        print(f"Compression ratio: {full_storage / seed_sol.storage_size():.1f}x")
        puzzle_results['seed_reconstruct'] = {
            'time': elapsed,
            'positions': count,
            'compression': full_storage / seed_sol.storage_size()
        }

        # Method 4: HOLOS Osmosis (best for Sudoku)
        print("\n--- HOLOS Osmosis ---")
        solver = HOLOSSolver(game, name=f"sudoku_{name}", max_memory_mb=200)
        forward_seeds = [SeedPoint(puzzle, SearchMode.OSMOSIS)]

        start = time.time()
        hologram = solver.solve_osmosis(forward_seeds, max_steps=2000, verbose=False)
        elapsed = time.time() - start

        puzzle_h = game.hash_state(puzzle)
        value = hologram.query(puzzle_h)
        solved = value is not None and hasattr(value, 'solved') and value.solved
        print(f"Time: {elapsed:.3f}s")
        print(f"Solved positions: {len(hologram.solved)}")
        print(f"Puzzle solved: {value}")
        puzzle_results['holos_osmosis'] = {
            'time': elapsed,
            'solved_positions': len(hologram.solved),
            'puzzle_solved': solved
        }

        results[name] = puzzle_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, res in results.items():
        print(f"\n{name.upper()} ({res['clues']} clues):")
        print(f"  Traditional: {res['traditional']['time']:.4f}s")
        print(f"  Seed path: {res['seed_capture']['path_length']} moves, {res['seed_capture']['storage_bytes']} bytes")
        print(f"  Compression: {res['seed_reconstruct']['compression']:.1f}x")
        osmosis = res.get('holos_osmosis', {})
        print(f"  HOLOS Osmosis: {osmosis.get('time', 0):.3f}s, {osmosis.get('solved_positions', 0)} positions, solved={osmosis.get('puzzle_solved', False)}")

    return results


def test_multiple_puzzles_seed_storage():
    """
    Test solving MANY puzzles and storing as seeds.

    Demonstrates that seed storage scales well.
    """
    print("\n" + "=" * 70)
    print("MULTIPLE PUZZLES: SEED STORAGE TEST")
    print("=" * 70)

    game = SudokuGame()

    # Generate multiple puzzles of varying difficulty
    print("\nGenerating puzzles...")
    puzzles = []

    # Easy (35-40 clues)
    for _ in range(5):
        p = generate_puzzle(num_clues=35, max_attempts=10)
        if p:
            puzzles.append(('easy', p))

    # Medium (28-32 clues)
    for _ in range(5):
        p = generate_puzzle(num_clues=30, max_attempts=10)
        if p:
            puzzles.append(('medium', p))

    # Harder (25-27 clues)
    for _ in range(3):
        p = generate_puzzle(num_clues=26, max_attempts=10)
        if p:
            puzzles.append(('hard', p))

    print(f"Generated {len(puzzles)} puzzles")

    # Solve all and capture as seeds
    print("\nSolving and capturing seeds...")
    seed_solutions = []
    total_solve_time = 0
    total_positions = 0
    total_storage = 0

    for i, (difficulty, puzzle) in enumerate(puzzles):
        start = time.time()
        try:
            seed_sol = create_seed_solution(game, puzzle)
            elapsed = time.time() - start

            hashes, count = reconstruct_from_seed(seed_sol)

            seed_solutions.append(seed_sol)
            total_solve_time += elapsed
            total_positions += count
            total_storage += seed_sol.storage_size()

            if (i + 1) % 5 == 0:
                print(f"  Solved {i+1}/{len(puzzles)}")
        except Exception as e:
            print(f"  Puzzle {i+1} failed: {e}")

    # Calculate storage comparison
    full_storage = total_positions * 16  # hash + value

    print(f"\n" + "-" * 60)
    print("STORAGE COMPARISON")
    print("-" * 60)
    print(f"Puzzles solved: {len(seed_solutions)}")
    print(f"Total solve time: {total_solve_time:.2f}s")
    print(f"Total positions covered: {total_positions:,}")
    print(f"Seed storage: {total_storage:,} bytes ({total_storage/1024:.1f} KB)")
    print(f"Full storage: {full_storage:,} bytes ({full_storage/1024:.1f} KB)")
    print(f"Compression ratio: {full_storage/total_storage:.1f}x")

    return seed_solutions


def test_layer1_sudoku_seeds():
    """
    Test Layer 1 tactical optimization for Sudoku seeds.

    Evaluates different seed configurations (depth, mode, direction).
    """
    print("\n" + "=" * 70)
    print("LAYER 1: SUDOKU SEED OPTIMIZATION")
    print("=" * 70)

    game = SudokuGame()
    puzzles = get_sample_puzzles()

    # Use easy puzzle as seed position
    puzzle = puzzles['easy']
    print(f"\nUsing easy puzzle as seed ({puzzle.filled_count()} clues)")

    # Create seed pool
    puzzle_hash = game.hash_state(puzzle)
    seed_pool = [(puzzle_hash, puzzle)]

    # Also add some solved grids to seed pool
    print("Generating solved grids for seed pool...")
    for _ in range(5):
        solution = game._generate_random_solution()
        if solution:
            seed_pool.append((game.hash_state(solution), solution))

    print(f"Seed pool: {len(seed_pool)} positions")

    # Create Layer 1 game
    layer1_game = TacticalSeedGame(
        underlying_game=game,
        seed_pool=seed_pool,
        max_depth=4
    )

    # Evaluate different configurations for the puzzle
    print("\n--- Evaluating Puzzle Seed Configurations ---")
    results = []

    for depth in [1, 2, 3]:
        for mode in [SearchMode.LIGHTNING, SearchMode.WAVE]:
            for direction in [SeedDirection.FORWARD, SeedDirection.BILATERAL]:
                seed = TacticalSeed(
                    puzzle_hash, depth, mode, direction, puzzle
                )
                value = layer1_game.evaluate(seed)
                results.append((seed, value))
                print(f"  {seed.signature():40} -> fwd={value.forward_coverage:4}, eff={value.efficiency:.1f}")

    # Best configuration
    best = max(results, key=lambda x: x[1].efficiency)
    print(f"\nBest configuration: {best[0].signature()}")
    print(f"Best value: {best[1]}")

    # Also evaluate solved grids (backward expansion)
    if len(seed_pool) > 1:
        print("\n--- Evaluating Solved Grid Configurations ---")
        sol_hash, solution = seed_pool[1]

        for depth in [1, 2]:
            for direction in [SeedDirection.BACKWARD, SeedDirection.BILATERAL]:
                seed = TacticalSeed(
                    sol_hash, depth, SearchMode.WAVE, direction, solution
                )
                value = layer1_game.evaluate(seed)
                print(f"  {seed.signature():40} -> bwd={value.backward_coverage:4}, eff={value.efficiency:.1f}")

    print(f"\n{layer1_game.summary()}")

    return results


def test_seed_database():
    """
    Build a seed database from many solved puzzles.

    Demonstrates:
    1. Solving many puzzles
    2. Storing solutions as seeds
    3. Reconstructing on demand
    4. Database query performance
    """
    print("\n" + "=" * 70)
    print("SEED DATABASE: BUILD AND QUERY")
    print("=" * 70)

    game = SudokuGame()

    # Generate many puzzles
    print("\nGenerating puzzles of varying difficulty...")
    puzzles = []

    # Generate puzzles at different clue counts
    for num_clues in [40, 35, 30, 28]:
        for _ in range(5):
            p = generate_puzzle(num_clues=num_clues, max_attempts=20)
            if p:
                puzzles.append((num_clues, p))

    print(f"Generated {len(puzzles)} puzzles")

    # Solve all and build seed database
    print("\nBuilding seed database...")
    seed_db = {}  # puzzle_hash -> SeedSolution
    solve_times = []
    storage_total = 0

    start_total = time.time()
    for i, (clues, puzzle) in enumerate(puzzles):
        puzzle_h = game.hash_state(puzzle)

        start = time.time()
        try:
            seed_sol = create_seed_solution(game, puzzle)
            elapsed = time.time() - start

            seed_db[puzzle_h] = seed_sol
            solve_times.append(elapsed)
            storage_total += seed_sol.storage_size()

        except Exception as e:
            print(f"  Puzzle {i+1} failed: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(puzzles)}")

    total_time = time.time() - start_total

    print(f"\n--- Database Statistics ---")
    print(f"Puzzles in database: {len(seed_db)}")
    print(f"Total build time: {total_time:.2f}s")
    print(f"Average solve time: {sum(solve_times)/len(solve_times)*1000:.1f}ms")
    print(f"Total storage: {storage_total:,} bytes ({storage_total/1024:.1f} KB)")

    # Test reconstruction performance
    print("\n--- Reconstruction Performance ---")
    recon_times = []
    total_positions = 0

    for puzzle_h, seed_sol in seed_db.items():
        start = time.time()
        hashes, count = reconstruct_from_seed(seed_sol)
        elapsed = time.time() - start
        recon_times.append(elapsed)
        total_positions += count

    avg_recon = sum(recon_times) / len(recon_times) * 1000  # ms
    print(f"Average reconstruction time: {avg_recon:.3f}ms")
    print(f"Total positions reconstructable: {total_positions:,}")

    # Compare to full storage
    full_storage = total_positions * 16
    compression = full_storage / storage_total

    print(f"\n--- Storage Comparison ---")
    print(f"Seed storage: {storage_total:,} bytes")
    print(f"Full storage: {full_storage:,} bytes")
    print(f"Compression ratio: {compression:.1f}x")

    # Demonstrate query capability
    print("\n--- Query Example ---")
    if seed_db:
        # Take first puzzle
        puzzle_h = list(seed_db.keys())[0]
        seed_sol = seed_db[puzzle_h]

        # Reconstruct and find solution
        hashes, _ = reconstruct_from_seed(seed_sol)
        print(f"Puzzle hash: {puzzle_h}")
        print(f"Solution hash: {seed_sol.solution_hash}")
        print(f"Path length: {len(seed_sol.moves)}")
        print(f"First 5 moves: {seed_sol.moves[:5]}")

    return seed_db


def test_holos_with_seed_caching():
    """
    Test HOLOS solving with seed-based caching.

    The idea: After solving with HOLOS, extract key positions as seeds.
    When re-solving, use seeds to quickly reconstruct the solution space.
    """
    print("\n" + "=" * 70)
    print("HOLOS WITH SEED CACHING")
    print("=" * 70)

    game = SudokuGame(full_expansion=True, max_successors=15)
    puzzles = get_sample_puzzles()
    puzzle = puzzles['easy']

    print(f"\nPuzzle: {puzzle.filled_count()} clues")

    # First solve: capture seeds from spines
    print("\n--- First Solve (cold) ---")
    solver = HOLOSSolver(game, name="sudoku_cache", max_memory_mb=300, max_frontier_size=20_000)

    # Generate backward seeds
    solution = solve_sudoku(puzzle)
    backward_seeds = [SeedPoint(solution, SearchMode.WAVE)]
    for _ in range(5):
        sol = game._generate_random_solution()
        if sol:
            backward_seeds.append(SeedPoint(sol, SearchMode.WAVE))

    forward_seeds = [SeedPoint(puzzle, SearchMode.WAVE)]

    start = time.time()
    hologram = solver.solve(forward_seeds, backward_seeds, max_iterations=5, lightning_interval=1)
    cold_time = time.time() - start

    print(f"Cold solve time: {cold_time:.2f}s")
    print(f"Positions solved: {len(hologram.solved):,}")
    print(f"Spines found: {len(hologram.spines)}")
    print(f"Connections: {len(hologram.connections)}")

    # Extract seeds from spines
    spine_seeds = []
    for spine in hologram.spines:
        spine_seeds.append(spine.start_hash)
        spine_seeds.append(spine.end_hash)
        for h, _ in spine.checkpoints:
            spine_seeds.append(h)

    print(f"\nExtracted {len(spine_seeds)} seed hashes from spines")

    # Simulate "warm" solve: assume we have the seed solutions cached
    print("\n--- Simulated Warm Solve ---")
    # In a real system, we'd load seeds from disk and reconstruct
    # For now, show the potential speedup

    cached_solved = set(hologram.solved.keys())
    print(f"Cached positions available: {len(cached_solved):,}")

    # Re-solve with "cached" knowledge
    # This simulates having the solved positions pre-loaded
    print(f"Re-query time: instant (positions already in memory)")

    return hologram, spine_seeds


# ============================================================
# MAIN
# ============================================================

def main():
    """Run all Sudoku seed tests"""
    print("=" * 70)
    print("SUDOKU SEED-BASED SOLVER TEST SUITE")
    print("=" * 70)
    print("""
This tests the seed-based approach for Sudoku:
1. Solve puzzles and capture solution paths as "seeds"
2. Store seeds instead of all intermediate positions
3. Reconstruct positions on-demand from seeds
4. Measure compression ratio and speedup

Key insight: For Sudoku, the solution PATH is compact (50-60 moves),
while the explored SPACE can be large. Storing paths = compression.
    """)

    # Test 1: Compare all solving methods
    print("\n" + "#" * 70)
    print("# TEST 1: ALL METHODS COMPARISON")
    print("#" * 70)
    test_sudoku_solve_all_methods()

    # Test 2: Multiple puzzles with seed storage
    print("\n" + "#" * 70)
    print("# TEST 2: MULTIPLE PUZZLES SEED STORAGE")
    print("#" * 70)
    test_multiple_puzzles_seed_storage()

    # Test 3: Seed database
    print("\n" + "#" * 70)
    print("# TEST 3: SEED DATABASE")
    print("#" * 70)
    test_seed_database()

    # Test 4: Layer 1 optimization
    print("\n" + "#" * 70)
    print("# TEST 4: LAYER 1 SEED OPTIMIZATION")
    print("#" * 70)
    test_layer1_sudoku_seeds()

    # Test 5: HOLOS with seed caching
    print("\n" + "#" * 70)
    print("# TEST 5: HOLOS WITH SEED CACHING")
    print("#" * 70)
    test_holos_with_seed_caching()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
