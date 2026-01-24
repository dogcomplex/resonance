"""
holos/test_layer1_seeds.py - Test Layer 1 Tactical Seed Optimization

This tests the Layer 1 architecture where:
- State: A single TacticalSeed (position, depth, mode, direction)
- Value: TacticalValue (forward_coverage, backward_coverage, efficiency)
- Game: TacticalSeedGame wraps any Layer 0 game

The test demonstrates:
1. Creating a seed pool from Layer 0 positions
2. Evaluating seeds with dual coverage (forward + backward)
3. Finding optimal seed configurations
4. Progress display matching chess output format

Run: python -m holos.test_layer1_seeds
"""

import sys
import os
import time
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.seeds import (
    TacticalSeed, TacticalValue, TacticalSeedGame,
    SeedDirection, optimize_single_seed, create_tactical_solver
)


def test_connect4_seeds():
    """
    Test Layer 1 with Connect4 as underlying game.

    Connect4 is simpler than chess - good for testing the Layer 1 architecture.
    """
    from holos.games.connect4 import Connect4Game

    print("=" * 70)
    print("LAYER 1 TEST: Connect4 Seed Optimization")
    print("=" * 70)

    # Create Layer 0 game
    layer0_game = Connect4Game()

    # Create some candidate seed positions from different stages
    print("\nGenerating seed pool from Connect4 positions...")
    seed_pool = []

    # Start with empty board
    from holos.games.connect4 import C4State
    empty = C4State()
    seed_pool.append((layer0_game.hash_state(empty), empty))

    # Generate some positions by playing a few moves
    positions = [empty]
    seen_hashes = {layer0_game.hash_state(empty)}

    # BFS to get positions at different depths
    for depth in range(3):
        next_positions = []
        for pos in positions:
            for child, move in layer0_game.get_successors(pos):
                h = layer0_game.hash_state(child)
                if h not in seen_hashes and len(seed_pool) < 20:
                    seen_hashes.add(h)
                    seed_pool.append((h, child))
                    next_positions.append(child)
        positions = next_positions[:10]  # Limit branching

    print(f"Seed pool: {len(seed_pool)} positions")

    # Create Layer 1 game
    layer1_game = TacticalSeedGame(
        underlying_game=layer0_game,
        seed_pool=seed_pool,
        max_depth=4
    )

    # Evaluate different configurations for each seed
    print("\n" + "-" * 60)
    print("EVALUATING SEED CONFIGURATIONS")
    print("-" * 60)

    results = []

    for i, (pos_hash, pos_state) in enumerate(seed_pool[:5]):  # Test first 5 seeds
        print(f"\n--- Seed {i+1}/{min(5, len(seed_pool))} (hash={pos_hash}) ---")

        best_config = None
        best_value = None

        # Test different configurations
        for depth in [1, 2, 3]:
            for mode in [SearchMode.LIGHTNING, SearchMode.WAVE]:
                for direction in [SeedDirection.FORWARD, SeedDirection.BACKWARD, SeedDirection.BILATERAL]:
                    seed = TacticalSeed(pos_hash, depth, mode, direction, pos_state)
                    value = layer1_game.evaluate(seed, verbose=False)

                    if best_value is None or value.efficiency > best_value.efficiency:
                        best_config = seed
                        best_value = value

        print(f"  Best: {best_config.signature()}")
        print(f"  Value: {best_value}")
        results.append((best_config, best_value))

    # Summary
    print("\n" + "=" * 60)
    print("LAYER 1 RESULTS")
    print("=" * 60)

    print(f"\nBest configurations found:")
    for i, (config, value) in enumerate(sorted(results, key=lambda x: -x[1].efficiency)[:5]):
        print(f"  {i+1}. {config.signature()}")
        print(f"     Forward: {value.forward_coverage:,}, Backward: {value.backward_coverage:,}")
        print(f"     Efficiency: {value.efficiency:.1f}")

    print(f"\n{layer1_game.summary()}")

    return results


def test_chess_seeds():
    """
    Test Layer 1 with Chess as underlying game.

    Uses the targeted chess game to focus on KQRRvKQR material.
    """
    try:
        from holos.games.chess import ChessGame
    except ImportError:
        print("Chess game not available (python-chess or syzygy not installed)")
        return None

    print("\n" + "=" * 70)
    print("LAYER 1 TEST: Chess Seed Optimization (KQRRvKQR)")
    print("=" * 70)

    # Create Layer 0 game
    try:
        layer0_game = ChessGame(
            "./syzygy",
            min_pieces=7,
            max_pieces=8,
            target_material="KQRRvKQR"
        )
    except Exception as e:
        print(f"Could not create ChessGame: {e}")
        return None

    # Generate seed pool from boundary positions
    print("\nGenerating seed pool from chess positions...")

    # Get some boundary positions (from syzygy)
    boundary_seeds = layer0_game.generate_target_boundary_seeds(10)

    # Get some source positions (8-piece positions near boundary)
    source_seeds = layer0_game.generate_source_positions(5)

    # Build seed pool
    seed_pool = []
    for pos in boundary_seeds + source_seeds:
        h = layer0_game.hash_state(pos)
        seed_pool.append((h, pos))

    print(f"Seed pool: {len(seed_pool)} positions ({len(boundary_seeds)} boundary, {len(source_seeds)} source)")

    # Create Layer 1 game
    layer1_game = TacticalSeedGame(
        underlying_game=layer0_game,
        seed_pool=seed_pool,
        max_depth=4
    )

    # Evaluate different configurations
    print("\n" + "-" * 60)
    print("EVALUATING CHESS SEED CONFIGURATIONS")
    print("-" * 60)

    start_time = time.time()
    results = []

    # Test boundary seeds (backward direction preferred)
    print("\nEvaluating boundary seeds (prefer backward expansion)...")
    for i, (pos_hash, pos_state) in enumerate(seed_pool[:len(boundary_seeds)]):
        if i >= 5:  # Limit for testing
            break

        best_config = None
        best_value = None

        for depth in [2, 3, 4]:
            for mode in [SearchMode.LIGHTNING, SearchMode.WAVE]:
                # Boundary seeds work best with backward expansion
                for direction in [SeedDirection.BACKWARD, SeedDirection.BILATERAL]:
                    seed = TacticalSeed(pos_hash, depth, mode, direction, pos_state)
                    value = layer1_game.evaluate(seed, verbose=False)

                    if best_value is None or value.efficiency > best_value.efficiency:
                        best_config = seed
                        best_value = value

        if best_config:
            print(f"  Seed {i+1}: {best_config.signature()} -> eff={best_value.efficiency:.1f}")
            results.append((best_config, best_value, "boundary"))

    # Test source seeds (forward direction preferred)
    print("\nEvaluating source seeds (prefer forward expansion)...")
    source_start = len(boundary_seeds)
    for i, (pos_hash, pos_state) in enumerate(seed_pool[source_start:]):
        if i >= 5:  # Limit for testing
            break

        best_config = None
        best_value = None

        for depth in [2, 3, 4]:
            for mode in [SearchMode.LIGHTNING, SearchMode.WAVE]:
                # Source seeds work best with forward expansion
                for direction in [SeedDirection.FORWARD, SeedDirection.BILATERAL]:
                    seed = TacticalSeed(pos_hash, depth, mode, direction, pos_state)
                    value = layer1_game.evaluate(seed, verbose=False)

                    if best_value is None or value.efficiency > best_value.efficiency:
                        best_config = seed
                        best_value = value

        if best_config:
            print(f"  Seed {i+1}: {best_config.signature()} -> eff={best_value.efficiency:.1f}")
            results.append((best_config, best_value, "source"))

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("CHESS SEED OPTIMIZATION RESULTS")
    print("=" * 60)

    # Separate by type
    boundary_results = [(c, v) for c, v, t in results if t == "boundary"]
    source_results = [(c, v) for c, v, t in results if t == "source"]

    print(f"\nBoundary seeds (backward expansion):")
    for config, value in sorted(boundary_results, key=lambda x: -x[1].efficiency)[:3]:
        print(f"  {config.signature()}")
        print(f"    Forward: {value.forward_coverage:,}, Backward: {value.backward_coverage:,}")
        print(f"    Efficiency: {value.efficiency:.1f}")

    print(f"\nSource seeds (forward expansion):")
    for config, value in sorted(source_results, key=lambda x: -x[1].efficiency)[:3]:
        print(f"  {config.signature()}")
        print(f"    Forward: {value.forward_coverage:,}, Backward: {value.backward_coverage:,}")
        print(f"    Efficiency: {value.efficiency:.1f}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n{layer1_game.summary()}")

    return results


def test_holos_on_seeds():
    """
    Run HOLOSSolver on TacticalSeedGame.

    This demonstrates using HOLOS bidirectional search at Layer 1.
    The forward/backward frontiers explore the seed configuration space.
    """
    from holos.games.connect4 import Connect4Game, C4State

    print("\n" + "=" * 70)
    print("HOLOS SOLVER ON LAYER 1 (SEED OPTIMIZATION)")
    print("=" * 70)

    # Create Layer 0 game
    layer0_game = Connect4Game()

    # Build seed pool
    empty = C4State()
    seed_pool = [(layer0_game.hash_state(empty), empty)]

    # Add some positions
    positions = [empty]
    seen = {layer0_game.hash_state(empty)}
    for _ in range(2):
        next_pos = []
        for pos in positions:
            for child, _ in layer0_game.get_successors(pos):
                h = layer0_game.hash_state(child)
                if h not in seen and len(seed_pool) < 15:
                    seen.add(h)
                    seed_pool.append((h, child))
                    next_pos.append(child)
        positions = next_pos[:5]

    print(f"Seed pool: {len(seed_pool)} positions")

    # Create Layer 1 game
    layer1_game = TacticalSeedGame(
        underlying_game=layer0_game,
        seed_pool=seed_pool,
        max_depth=4
    )

    # Pre-evaluate some seeds to serve as "boundary"
    print("\nPre-evaluating boundary seeds...")
    evaluated_seeds = []
    for pos_hash, pos_state in seed_pool[:5]:
        # Evaluate at max depth (these become boundary)
        seed = TacticalSeed(pos_hash, 4, SearchMode.WAVE, SeedDirection.BILATERAL, pos_state)
        value = layer1_game.evaluate(seed, verbose=True)
        evaluated_seeds.append(seed)

    # Create seeds for HOLOS search
    # Forward seeds: low-depth configurations (start of search)
    forward_seeds = []
    for pos_hash, pos_state in seed_pool[:3]:
        seed = TacticalSeed(pos_hash, 1, SearchMode.LIGHTNING, SeedDirection.FORWARD, pos_state)
        forward_seeds.append(SeedPoint(seed, SearchMode.WAVE))

    # Backward seeds: high-depth evaluated configurations (boundary)
    backward_seeds = []
    for seed in evaluated_seeds[:3]:
        backward_seeds.append(SeedPoint(seed, SearchMode.WAVE))

    print(f"\nForward seeds (low config): {len(forward_seeds)}")
    print(f"Backward seeds (high config, evaluated): {len(backward_seeds)}")

    # Create solver
    solver = HOLOSSolver(
        layer1_game,
        name="layer1_seeds",
        max_memory_mb=100,
        max_frontier_size=1000
    )

    print("\n" + "=" * 60)
    print("HOLOS Search on TacticalSeedGame")
    print("=" * 60)

    # Run search
    hologram = solver.solve(forward_seeds, backward_seeds, max_iterations=3)

    # Analyze results
    print("\n" + "=" * 60)
    print("LAYER 1 HOLOS RESULTS")
    print("=" * 60)

    print(f"Solved configurations: {len(hologram.solved):,}")
    print(f"Connections: {len(hologram.connections)}")
    print(f"Spines: {len(hologram.spines)}")

    # Show some solved configurations
    print("\nSample solved configurations:")
    for i, (h, value) in enumerate(list(hologram.solved.items())[:5]):
        print(f"  {i+1}. hash={h}, value={value}")

    print(f"\n{layer1_game.summary()}")

    return hologram


def main():
    """Run all Layer 1 tests"""
    print("=" * 70)
    print("LAYER 1 (TACTICS) TEST SUITE")
    print("=" * 70)
    print("""
Layer 1 Architecture:
  - State: TacticalSeed (position_hash, depth, mode, direction)
  - Value: TacticalValue (forward_coverage, backward_coverage, efficiency)
  - Game: TacticalSeedGame wraps any Layer 0 game

Key insight: Each seed needs DUAL coverage for bidirectional search.
  - Boundary seeds work best with BACKWARD expansion
  - Source seeds work best with FORWARD expansion
  - Bilateral is expensive but maximizes coverage
    """)

    # Test with Connect4 (simpler, always available)
    print("\n" + "#" * 70)
    print("# TEST 1: Connect4 Seeds")
    print("#" * 70)
    test_connect4_seeds()

    # Test with Chess (if available)
    print("\n" + "#" * 70)
    print("# TEST 2: Chess Seeds")
    print("#" * 70)
    test_chess_seeds()

    # Test HOLOS on seed game
    print("\n" + "#" * 70)
    print("# TEST 3: HOLOS Solver on Layer 1")
    print("#" * 70)
    test_holos_on_seeds()

    print("\n" + "=" * 70)
    print("LAYER 1 TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
