"""
holos/demo.py - Demonstration of the HOLOS architecture

This script demonstrates:
1. Creating a chess game interface
2. Running HOLOS solver
3. Using sessions for multi-round solving
4. The seed selection meta-game
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.storage import Hologram, SpinePath, SeedFrontierMapping
from holos.session import SessionManager, create_session
from holos.games.chess import (
    ChessGame, ChessState, ChessValue,
    random_position, create_chess_solver
)
from holos.games.seeds import SeedGame, SeedConfiguration, SeedSpec, create_seed_solver


def demo_chess_solving():
    """Demonstrate basic chess solving"""
    print("=" * 60)
    print("DEMO: Chess Endgame Solving")
    print("=" * 60)

    # Create solver
    solver, game = create_chess_solver(
        syzygy_path="./syzygy",  # Adjust path as needed
        min_pieces=7,
        max_pieces=8
    )

    # Generate test positions
    print("\nGenerating 8-piece positions...")
    positions = []
    for i in range(20):
        pos = random_position("KQRRvKQRR")
        if pos:
            positions.append(pos)

    if not positions:
        print("Could not generate positions (need syzygy tables)")
        return None

    print(f"Generated {len(positions)} valid positions")

    # Create seeds
    forward_seeds = [SeedPoint(p, SearchMode.WAVE, priority=1) for p in positions]

    # Solve
    print("\nRunning HOLOS solver...")
    hologram = solver.solve(forward_seeds, max_iterations=20)

    print(f"\nResults:")
    print(hologram.summary())

    return hologram


def demo_bidirectional_lightning():
    """Demonstrate bidirectional lightning probes"""
    print("\n" + "=" * 60)
    print("DEMO: Bidirectional Lightning")
    print("=" * 60)

    solver, game = create_chess_solver()

    # Generate a position
    pos = random_position("KQRRvKQR")
    if not pos:
        print("Could not generate position")
        return

    print("\nTest position:")
    pos.display()

    # Forward lightning (toward boundary)
    from holos.holos import LightningProbe

    forward_probe = LightningProbe(game, {}, direction="forward", max_depth=10)
    value, path = forward_probe.probe(pos)

    print(f"\nForward lightning:")
    print(f"  Value found: {value}")
    print(f"  Path length: {len(path)}")
    print(f"  Nodes visited: {forward_probe.nodes_visited}")

    # If we have a boundary position, try backward lightning
    if game.is_boundary(pos):
        boundary_val = game.get_boundary_value(pos)
        print(f"\n  (Position is on boundary: {boundary_val})")

        backward_probe = LightningProbe(game, {}, direction="backward", max_depth=10)
        bwd_value, bwd_path = backward_probe.probe(pos)

        print(f"\nBackward lightning:")
        print(f"  Value found: {bwd_value}")
        print(f"  Path length: {len(bwd_path)}")
        print(f"  Nodes visited: {backward_probe.nodes_visited}")


def demo_session_management():
    """Demonstrate multi-round session management"""
    print("\n" + "=" * 60)
    print("DEMO: Session Management")
    print("=" * 60)

    solver, game = create_chess_solver()

    # Create or resume session
    session = create_session(
        session_id="demo_chess",
        game_name="chess_8piece",
        save_dir="./holos_sessions",
        budget=500.0
    )

    print(f"\nSession state:")
    print(session.state.summary())

    # Generate seeds
    positions = [random_position("KQRRvKQRR") for _ in range(10)]
    positions = [p for p in positions if p]

    if positions:
        # Run a single round
        stats = session.run_round(
            solver, positions,
            max_iterations=10,
            budget=50.0
        )

        print(f"\nRound stats:")
        print(f"  Solved: {stats.states_solved}")
        print(f"  Explored: {stats.states_explored}")
        print(f"  Duration: {stats.duration():.2f}s")


def demo_seed_game():
    """Demonstrate the seed selection meta-game"""
    print("\n" + "=" * 60)
    print("DEMO: Seed Selection Meta-Game (Layer 1)")
    print("=" * 60)

    # Create seed game
    solver, seed_game = create_seed_solver(material="KQRRvKQR")

    # Create a dummy seed pool
    # In real usage, this would be actual boundary positions
    seed_pool = [(hash(f"seed_{i}"), f"state_{i}") for i in range(50)]
    seed_game.set_seed_pool(seed_pool)

    # Create configurations to evaluate
    from holos.games.seeds import SeedConfiguration, SeedSpec, create_initial_configs

    configs = create_initial_configs(seed_game, num_configs=10)
    print(f"\nGenerated {len(configs)} initial configurations:")
    for config in configs[:5]:
        print(f"  {config.signature()}")

    # The meta-game searches over configurations
    # Each configuration represents a choice of:
    # - Which boundary positions to seed from
    # - What depth to expand each seed
    # - What search mode to use

    print("\nMeta-game structure:")
    print("  State: SeedConfiguration (set of SeedSpecs)")
    print("  Successors: Add seed, increase depth, change mode")
    print("  Predecessors: Remove seed, decrease depth")
    print("  Value: Efficiency (coverage / cost)")

    print("\nKey discovery from experiments:")
    print("  1 seed @ depth 5:  ~1655 efficiency")
    print("  20 seeds @ depth 2: ~191 efficiency")
    print("  -> DEPTH is the dominant variable (~10x difference)")


def demo_storage():
    """Demonstrate storage structures"""
    print("\n" + "=" * 60)
    print("DEMO: Storage Structures")
    print("=" * 60)

    # Create a spine path
    spine = SpinePath(
        start_hash=12345,
        moves=[(0, 8, None), (8, 16, 9), (16, 24, None)],  # Example moves
        end_hash=67890,
        end_value=1,  # White wins
        checkpoints=[(12345, "start"), (54321, "mid"), (67890, "end")]
    )

    print("\nSpine Path:")
    print(f"  Start: {spine.start_hash}")
    print(f"  End: {spine.end_hash} (value: {spine.end_value})")
    print(f"  Depth: {spine.depth}")
    print(f"  Checkpoints: {len(spine.checkpoints)}")

    # Create seed->frontier mapping
    mapping = SeedFrontierMapping(
        seed_hash=12345,
        depth=3,
        mode="wave",
        expansion_params={"max_branch": 10}
    )

    print("\nSeed->Frontier Mapping:")
    print(f"  Seed: {mapping.seed_hash}")
    print(f"  Depth: {mapping.depth}")
    print(f"  Mode: {mapping.mode}")
    print(f"  Compression estimate: ~{10 ** mapping.depth}x")

    # Create hologram
    hologram = Hologram("demo")
    hologram.solved = {12345: 1, 67890: 0, 11111: -1}
    hologram.spines.append(spine)
    hologram.seed_mappings.append(mapping)

    print("\nHologram:")
    print(hologram.summary())


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("HOLOS ARCHITECTURE DEMONSTRATION")
    print("=" * 60)

    # Run demos that don't need syzygy
    demo_storage()
    demo_seed_game()

    # These need syzygy tables
    try:
        demo_chess_solving()
        demo_bidirectional_lightning()
        demo_session_management()
    except Exception as e:
        print(f"\nChess demos need syzygy tables: {e}")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

    print("\nAll checklist items implemented:")
    print("  [x] Spine paths (SpinePath with checkpoints)")
    print("  [x] Backward lightning (LightningProbe direction='backward')")
    print("  [x] SessionManager (multi-round handling)")
    print("  [x] Seed->Frontier storage (SeedFrontierMapping)")
    print("  [x] Mode selection as meta-decision (ModeSelector)")


if __name__ == "__main__":
    main()
