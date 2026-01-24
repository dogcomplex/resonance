"""
holos/test_seed_compression.py - Test seed-based storage compression

This tests the core HOLOS compression hypothesis:
    Instead of storing N solved positions, store K seeds that can regenerate them.
    Compression ratio: N/K (potentially 1000x-8000x)

The test:
1. Run a HOLOS search and capture the solved positions
2. Identify which seeds contributed to which solved positions
3. Measure: Can we reconstruct the solved set from just the seeds?
4. Calculate actual compression ratio

This validates whether we can move from:
    Current: Store all 10M solved positions (10GB)
    Future:  Store 1K seeds (10KB) + reconstruction algorithm
"""

import sys
import os
import time
import pickle
from typing import Dict, Set, List, Tuple, Any
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.storage import Hologram, SeedFrontierMapping


def deterministic_expand(game, seed_state, depth: int, mode: str = "wave") -> Set[int]:
    """
    Deterministically expand from a seed to get all reachable positions.

    This is the reconstruction function - given a seed and depth,
    regenerate the exact set of positions that would be explored.
    """
    seen = set()
    frontier = {game.hash_state(seed_state): seed_state}
    seen.add(game.hash_state(seed_state))

    for d in range(depth):
        next_frontier = {}
        for h, state in frontier.items():
            for child, move in game.get_successors(state):
                ch = game.hash_state(child)
                if ch not in seen:
                    seen.add(ch)
                    next_frontier[ch] = child
        frontier = next_frontier
        if not frontier:
            break

    return seen


def analyze_seed_coverage(game, solver, hologram, forward_seeds: List, backward_seeds: List):
    """
    Analyze how well seeds cover the solved positions.

    Returns dict with:
    - total_solved: Total positions in hologram
    - seeds_used: Number of seeds
    - reconstructable: Positions that can be reconstructed from seeds
    - coverage_ratio: reconstructable / total_solved
    - compression_ratio: total_solved / seeds_used
    """
    print("\n" + "=" * 60)
    print("SEED COVERAGE ANALYSIS")
    print("=" * 60)

    total_solved = len(hologram.solved)
    print(f"Total solved positions: {total_solved:,}")

    # Track which positions each seed can reach
    seed_coverage = {}
    all_reconstructable = set()

    # Analyze forward seeds
    print(f"\nAnalyzing {len(forward_seeds)} forward seeds...")
    for i, sp in enumerate(forward_seeds):
        seed_state = sp.state
        seed_hash = game.hash_state(seed_state)

        # Try different depths
        for depth in [1, 2, 3, 4, 5]:
            reachable = deterministic_expand(game, seed_state, depth)
            solved_reachable = reachable & set(hologram.solved.keys())

            if len(solved_reachable) > 0:
                key = (seed_hash, depth, "forward")
                seed_coverage[key] = {
                    'reachable': len(reachable),
                    'solved_reachable': len(solved_reachable),
                    'efficiency': len(solved_reachable) / (depth + 1),
                }
                all_reconstructable |= solved_reachable

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(forward_seeds)} forward seeds")

    # Analyze backward seeds
    print(f"\nAnalyzing {len(backward_seeds)} backward seeds...")
    for i, sp in enumerate(backward_seeds):
        seed_state = sp.state
        seed_hash = game.hash_state(seed_state)

        # Backward seeds use predecessors
        for depth in [1, 2, 3]:
            # For backward, we'd use get_predecessors - simplified here
            reachable = deterministic_expand(game, seed_state, depth)
            solved_reachable = reachable & set(hologram.solved.keys())

            if len(solved_reachable) > 0:
                key = (seed_hash, depth, "backward")
                seed_coverage[key] = {
                    'reachable': len(reachable),
                    'solved_reachable': len(solved_reachable),
                    'efficiency': len(solved_reachable) / (depth + 1),
                }
                all_reconstructable |= solved_reachable

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(backward_seeds)} backward seeds")

    # Calculate metrics
    coverage_ratio = len(all_reconstructable) / total_solved if total_solved > 0 else 0
    seeds_used = len(forward_seeds) + len(backward_seeds)
    compression_ratio = total_solved / seeds_used if seeds_used > 0 else 0

    print(f"\n--- Results ---")
    print(f"Seeds used: {seeds_used}")
    print(f"Positions reconstructable from seeds: {len(all_reconstructable):,}")
    print(f"Coverage ratio: {coverage_ratio:.1%}")
    print(f"Compression ratio: {compression_ratio:.0f}x")

    # Find optimal seeds (greedy set cover)
    print(f"\n--- Optimal Seed Selection (Greedy) ---")
    remaining = set(hologram.solved.keys())
    selected_seeds = []

    while remaining and seed_coverage:
        # Find seed that covers most remaining positions
        best_key = None
        best_coverage = 0

        for key, data in seed_coverage.items():
            seed_hash, depth, direction = key
            # Recalculate coverage of remaining
            if direction == "forward":
                # Would need to re-expand... simplified
                pass

        # Simplified: just report the coverage we found
        break

    # Summary statistics by depth
    print(f"\n--- Coverage by Depth ---")
    by_depth = defaultdict(lambda: {'count': 0, 'total_solved': 0})
    for (seed_hash, depth, direction), data in seed_coverage.items():
        by_depth[depth]['count'] += 1
        by_depth[depth]['total_solved'] += data['solved_reachable']

    for depth in sorted(by_depth.keys()):
        d = by_depth[depth]
        avg = d['total_solved'] / d['count'] if d['count'] > 0 else 0
        print(f"  Depth {depth}: {d['count']} seeds, avg {avg:.0f} solved/seed")

    return {
        'total_solved': total_solved,
        'seeds_used': seeds_used,
        'reconstructable': len(all_reconstructable),
        'coverage_ratio': coverage_ratio,
        'compression_ratio': compression_ratio,
        'seed_coverage': seed_coverage,
    }


def test_seed_compression():
    """
    Run a small search and analyze seed compression potential.
    """
    from holos.games.chess import ChessGame

    print("=" * 70)
    print("SEED COMPRESSION TEST")
    print("=" * 70)

    # Create game with targeting
    game = ChessGame(
        "./syzygy",
        min_pieces=7,
        max_pieces=8,
        target_material="KQRRvKQR"
    )

    # Generate seeds
    print("\nGenerating seeds...")
    backward_seeds = game.generate_target_boundary_seeds(100)
    forward_seeds = game.generate_source_positions(20)

    print(f"Forward seeds: {len(forward_seeds)}")
    print(f"Backward seeds: {len(backward_seeds)}")

    # Create solver and run
    solver = HOLOSSolver(
        game,
        name="compression_test",
        max_memory_mb=1000,
        max_frontier_size=500_000,
    )

    forward_sp = [SeedPoint(p, SearchMode.WAVE) for p in forward_seeds]
    backward_sp = [SeedPoint(p, SearchMode.WAVE) for p in backward_seeds]

    print("\nRunning HOLOS search...")
    start = time.time()
    hologram = solver.solve(forward_sp, backward_sp, max_iterations=5)
    elapsed = time.time() - start

    print(f"Search complete: {len(hologram.solved):,} solved in {elapsed:.1f}s")

    # Analyze seed coverage
    analysis = analyze_seed_coverage(game, solver, hologram, forward_sp, backward_sp)

    # Theoretical vs actual
    print("\n" + "=" * 60)
    print("COMPRESSION POTENTIAL")
    print("=" * 60)

    # If we stored just seeds + reconstruction params
    seed_storage = (len(forward_seeds) + len(backward_seeds)) * 100  # ~100 bytes per seed
    full_storage = len(hologram.solved) * 16  # ~16 bytes per (hash, value) pair

    print(f"Full storage (all positions): {full_storage / 1024:.1f} KB")
    print(f"Seed storage (seeds only): {seed_storage / 1024:.1f} KB")
    print(f"Potential compression: {full_storage / seed_storage:.0f}x")

    print(f"\nCoverage achieved: {analysis['coverage_ratio']:.1%}")
    print(f"Missing positions: {analysis['total_solved'] - analysis['reconstructable']:,}")

    # The key question
    print("\n" + "=" * 60)
    print("KEY FINDING")
    print("=" * 60)

    if analysis['coverage_ratio'] > 0.95:
        print("EXCELLENT: >95% of positions reconstructable from seeds!")
        print("Seed-based storage is highly viable.")
    elif analysis['coverage_ratio'] > 0.80:
        print("GOOD: >80% coverage. Most positions reconstructable.")
        print("Need to track ~20% of positions separately.")
    elif analysis['coverage_ratio'] > 0.50:
        print("MODERATE: 50-80% coverage.")
        print("Hybrid approach needed: seeds + explicit storage for rest.")
    else:
        print("LIMITED: <50% coverage.")
        print("Positions arise from wave interactions, not just seed expansion.")
        print("Need to identify 'interaction seeds' or store explicitly.")

    return analysis


def test_connection_seeds():
    """
    Test whether CONNECTION POINTS make good seeds for reconstruction.

    Hypothesis: The positions where forward and backward waves meet
    are optimal seeds because they encode maximum information.
    """
    from holos.games.chess import ChessGame

    print("\n" + "=" * 70)
    print("CONNECTION POINT SEED TEST")
    print("=" * 70)

    game = ChessGame(
        "./syzygy",
        min_pieces=7,
        max_pieces=8,
        target_material="KQRRvKQR"
    )

    # Generate seeds
    backward_seeds = game.generate_target_boundary_seeds(200)
    forward_seeds = game.generate_source_positions(30)

    solver = HOLOSSolver(game, name="connection_test", max_memory_mb=1500)

    forward_sp = [SeedPoint(p, SearchMode.WAVE) for p in forward_seeds]
    backward_sp = [SeedPoint(p, SearchMode.WAVE) for p in backward_seeds]

    hologram = solver.solve(forward_sp, backward_sp, max_iterations=6)

    print(f"Solved: {len(hologram.solved):,}")
    print(f"Connections: {len(hologram.connections)}")
    print(f"Spines: {len(hologram.spines)}")

    # Analyze connection points as seeds
    if hologram.connections:
        print(f"\nAnalyzing {len(hologram.connections)} connection points as seeds...")

        connection_coverage = set()
        for fh, bh, value in hologram.connections:
            # Each connection point could be a seed
            # The forward hash and backward hash both contribute
            connection_coverage.add(fh)
            connection_coverage.add(bh)

        # How many unique connection points?
        print(f"Unique connection hashes: {len(connection_coverage)}")

        # Are these in solved?
        in_solved = connection_coverage & set(hologram.solved.keys())
        print(f"Connections in solved set: {len(in_solved)}")

        # Theoretical: if connections are optimal seeds
        # We could regenerate the solved region from them
        compression = len(hologram.solved) / len(connection_coverage) if connection_coverage else 0
        print(f"Potential compression via connections: {compression:.0f}x")

    return hologram


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full analysis")
    parser.add_argument("--connections", action="store_true", help="Test connection seeds")
    args = parser.parse_args()

    if args.connections:
        test_connection_seeds()
    else:
        test_seed_compression()
