"""
Test script for tandem layer execution.

Tests the TandemOrchestrator with Connect4 game.
"""

import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from holos.games.connect4 import Connect4Game, C4State
from holos.tandem import (
    TandemOrchestrator, TandemSeedGame, create_tandem_solver,
    compare_recommendation_modes
)
from holos.holos import HOLOSSolver, SeedPoint, SearchMode


def test_basic_tandem():
    """Basic test of tandem execution"""
    print("\n" + "="*60)
    print("TEST: Basic Tandem Execution with Connect4")
    print("="*60)

    game = Connect4Game()

    # Start from empty board and a few early positions
    start = C4State()
    positions = [start]

    # Add some positions with a few moves using game.get_successors
    for child, col in game.get_successors(start):
        if col in [2, 3, 4]:  # Center-ish columns
            positions.append(child)
            # Add grandchildren too
            for grandchild, col2 in game.get_successors(child):
                if col2 in [1, 2, 3, 4, 5]:
                    positions.append(grandchild)

    print(f"Initial positions: {len(positions)}")

    # Create tandem orchestrator
    orchestrator = create_tandem_solver(
        game,
        initial_positions=positions,
        max_depth=4,
        mode="osmosis"
    )

    # Run a few iterations
    results = orchestrator.run_session(
        max_iterations=5,
        seeds_per_iteration=3,
        layer0_iterations=2,
        verbose=True
    )

    print("\nTest complete!")
    return orchestrator


def test_mode_comparison():
    """Compare different recommendation modes"""
    print("\n" + "="*60)
    print("TEST: Compare Recommendation Modes")
    print("="*60)

    game = Connect4Game()
    start = C4State()

    # Build initial position pool using get_successors
    positions = [start]
    frontier = [start]
    seen = {game.hash_state(start)}

    for depth in range(3):
        next_frontier = []
        for pos in frontier:
            for child, col in game.get_successors(pos):
                h = game.hash_state(child)
                if h not in seen:
                    seen.add(h)
                    positions.append(child)
                    next_frontier.append(child)
        frontier = next_frontier[:50]  # Limit growth

    print(f"Initial positions: {len(positions)}")

    # Compare modes
    results = compare_recommendation_modes(
        game,
        positions[:100],  # Use subset
        iterations=5,
        seeds_per_iter=3
    )

    print("\n" + "="*60)
    print("COMPARISON RESULTS:")
    print("="*60)
    for mode, data in results.items():
        print(f"\n{mode}:")
        for k, v in data.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")


def test_osmosis_scoring():
    """Test the osmosis scoring mechanism"""
    print("\n" + "="*60)
    print("TEST: Osmosis Scoring")
    print("="*60)

    game = Connect4Game()
    start = C4State()

    # Create tandem game
    positions = [(game.hash_state(start), start)]
    tandem = TandemSeedGame(game, max_depth=4, initial_positions=positions)

    # Generate some candidate configs
    from holos.games.seeds import TacticalSeed, TacticalValue, SeedDirection

    configs = []
    for depth in range(1, 5):
        for mode in [SearchMode.LIGHTNING, SearchMode.WAVE]:
            config = TacticalSeed(
                positions[0][0], depth, mode, SeedDirection.FORWARD, start
            )
            configs.append(config)

    print(f"Generated {len(configs)} configs")

    # Score them all (no evaluations yet)
    print("\nScores before any evaluation:")
    for config in configs[:5]:
        score = tandem.score_for_osmosis(config)
        print(f"  {config.signature()}: score={score:.1f}")

    # Simulate some evaluations
    eval_config = configs[0]
    eval_value = TacticalValue(
        forward_coverage=100,
        backward_coverage=50,
        overlap_potential=0.0,
        cost=1,
        efficiency=150.0
    )
    tandem.record_evaluation(eval_config, eval_value)

    print(f"\nRecorded evaluation: {eval_config.signature()} -> eff={eval_value.efficiency}")

    # Re-score - neighbors should score higher now
    print("\nScores after evaluation:")
    for config in configs[:5]:
        score = tandem.score_for_osmosis(config)
        h = hash(config)
        status = "(evaluated)" if h in tandem.eval_cache else ""
        print(f"  {config.signature()}: score={score:.1f} {status}")


def test_frontier_growth():
    """Test that Layer 0 frontiers feed into Layer 1 pool"""
    print("\n" + "="*60)
    print("TEST: Frontier Growth")
    print("="*60)

    game = Connect4Game()
    start = C4State()

    orchestrator = create_tandem_solver(
        game,
        initial_positions=[start],
        max_depth=3,
        mode="osmosis"
    )

    print(f"Initial pool size: {len(orchestrator.layer1_game.seed_pool)}")

    # Run iterations and watch pool grow
    for i in range(3):
        result = orchestrator.run_iteration(
            seeds_per_iteration=2,
            layer0_iterations=3,
            verbose=False
        )
        pool_size = len(orchestrator.layer1_game.seed_pool)
        frontier_size = (len(orchestrator.layer0_solver.forward_frontier) +
                        len(orchestrator.layer0_solver.backward_frontier))
        print(f"Iteration {i+1}: pool={pool_size}, L0 frontier={frontier_size}, "
              f"solved={len(orchestrator.layer0_solver.solved)}")


def test_lightning_mode():
    """Test lightning recommendation mode"""
    print("\n" + "="*60)
    print("TEST: Lightning Mode")
    print("="*60)

    game = Connect4Game()
    start = C4State()

    # Build a small pool using get_successors
    positions = [start]
    for child, col in game.get_successors(start):
        if col in [2, 3, 4]:
            positions.append(child)

    orchestrator = create_tandem_solver(
        game,
        initial_positions=positions,
        max_depth=5,
        mode="lightning"
    )

    # Run and observe DFS-like behavior
    results = orchestrator.run_session(
        max_iterations=4,
        seeds_per_iteration=3,
        layer0_iterations=2,
        verbose=True
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test tandem layer execution")
    parser.add_argument("--test", choices=["basic", "compare", "scoring", "frontier", "lightning", "all"],
                       default="basic", help="Which test to run")

    args = parser.parse_args()

    if args.test == "basic" or args.test == "all":
        test_basic_tandem()

    if args.test == "scoring" or args.test == "all":
        test_osmosis_scoring()

    if args.test == "frontier" or args.test == "all":
        test_frontier_growth()

    if args.test == "lightning" or args.test == "all":
        test_lightning_mode()

    if args.test == "compare" or args.test == "all":
        test_mode_comparison()

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)
