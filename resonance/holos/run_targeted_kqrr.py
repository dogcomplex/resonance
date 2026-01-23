"""
holos/run_targeted_kqrr.py - Exhaustive KQRRvKQR Targeted Search

Goal: Find ALL 8-piece positions that lead to KQRRvKQR 7-piece solutions.

This demonstrates:
1. Material-filtered boundary conditions
2. Early termination on wrong captures
3. Seed-based reproducibility
4. Batched solving for memory management

The search should be reproducible from just the seed list.
"""

import sys
import os
import pickle
import time
import gc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.storage import Hologram
from holos.games.chess_targeted import (
    TargetedChessGame, create_targeted_solver,
    get_8piece_variants, material_string
)


def run_targeted_search(
    target_material: str = "KQRRvKQR",
    syzygy_path: str = "./syzygy",
    save_dir: str = "./holos_targeted",
    # Search parameters
    backward_seeds_count: int = 500,
    forward_seeds_per_material: int = 50,
    max_iterations: int = 20,
    max_memory_mb: int = 3000,
    # Batching for memory management
    batch_size: int = 100,
    max_batches: int = None,  # None = unlimited
):
    """
    Run targeted search for positions leading to specific material.

    Args:
        target_material: 7-piece material to target (e.g., "KQRRvKQR")
        syzygy_path: Path to syzygy tablebases
        save_dir: Directory for results
        backward_seeds_count: Number of 7-piece positions to seed backward from
        forward_seeds_per_material: Number of 8-piece positions per source material
        max_iterations: Max iterations per batch
        max_memory_mb: Memory limit per batch
        batch_size: Seeds per batch for memory management
        max_batches: Maximum batches (None = unlimited)
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"TARGETED HOLOS SEARCH: {target_material}")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Save directory: {save_dir}")
    print()

    # Create game and solver
    game = TargetedChessGame(syzygy_path, target_material)

    # Generate backward seeds (target 7-piece positions)
    print("\n" + "-" * 40)
    print("PHASE 1: Generate Backward Seeds (Target Material)")
    print("-" * 40)
    backward_positions = game.generate_target_boundary_seeds(backward_seeds_count)

    if not backward_positions:
        print("ERROR: Could not generate backward seeds. Check syzygy path.")
        return None

    # Save backward seeds
    backward_seeds_file = os.path.join(save_dir, "backward_seeds.pkl")
    with open(backward_seeds_file, 'wb') as f:
        pickle.dump(backward_positions, f)
    print(f"Saved {len(backward_positions)} backward seeds to {backward_seeds_file}")

    # Generate forward seeds (8-piece source positions)
    print("\n" + "-" * 40)
    print("PHASE 2: Generate Forward Seeds (Source Materials)")
    print("-" * 40)
    forward_positions = game.generate_source_positions(forward_seeds_per_material)

    if not forward_positions:
        print("ERROR: Could not generate forward seeds.")
        return None

    # Save forward seeds
    forward_seeds_file = os.path.join(save_dir, "forward_seeds.pkl")
    with open(forward_seeds_file, 'wb') as f:
        pickle.dump(forward_positions, f)
    print(f"Saved {len(forward_positions)} forward seeds to {forward_seeds_file}")

    # Batched solving
    print("\n" + "-" * 40)
    print("PHASE 3: Batched Bidirectional Search")
    print("-" * 40)

    combined_hologram = Hologram(f"targeted_{target_material}")
    total_solved = 0
    total_connections = 0
    batch_num = 0

    # Process forward seeds in batches
    for i in range(0, len(forward_positions), batch_size):
        if max_batches is not None and batch_num >= max_batches:
            print(f"\nReached max batches ({max_batches}), stopping.")
            break

        batch_forward = forward_positions[i:i + batch_size]
        batch_num += 1

        print(f"\n=== Batch {batch_num} ({len(batch_forward)} forward seeds) ===")

        # Create fresh solver for this batch
        solver = HOLOSSolver(game, name=f"batch_{batch_num}", max_memory_mb=max_memory_mb)

        # Convert to SeedPoints
        forward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in batch_forward]

        # Use subset of backward seeds
        backward_sample = backward_positions[:min(len(backward_positions), backward_seeds_count)]
        backward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in backward_sample]

        # Run solver
        start_time = time.time()
        hologram = solver.solve(forward_seeds, backward_seeds, max_iterations=max_iterations)
        elapsed = time.time() - start_time

        # Merge results
        combined_hologram = combined_hologram.merge(hologram)

        batch_solved = len(hologram.solved)
        batch_connections = solver.stats['connections']
        total_solved += batch_solved
        total_connections += batch_connections

        print(f"Batch {batch_num} complete:")
        print(f"  Solved: {batch_solved:,} ({elapsed:.1f}s)")
        print(f"  Connections: {batch_connections}")
        print(f"  Running total: {len(combined_hologram.solved):,} solved")

        # Force garbage collection to free memory between batches
        del solver
        del hologram
        gc.collect()

        # Save intermediate results
        if batch_num % 5 == 0:
            checkpoint_file = os.path.join(save_dir, f"checkpoint_batch{batch_num}.pkl")
            combined_hologram.save(checkpoint_file)
            print(f"  Checkpoint saved: {checkpoint_file}")

    # Save final results
    print("\n" + "-" * 40)
    print("PHASE 4: Save Final Results")
    print("-" * 40)

    final_file = os.path.join(save_dir, "final_hologram.pkl")
    combined_hologram.save(final_file)

    # Save metadata
    metadata = {
        'target_material': target_material,
        'source_materials': game.source_materials,
        'backward_seeds_count': len(backward_positions),
        'forward_seeds_count': len(forward_positions),
        'total_solved': len(combined_hologram.solved),
        'total_connections': total_connections,
        'total_spines': len(combined_hologram.spines),
        'batches': batch_num,
        'completed': datetime.now().isoformat(),
        'filter_stats': game.filter_stats,
    }

    metadata_file = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nFinal Results:")
    print(f"  Total solved: {len(combined_hologram.solved):,}")
    print(f"  Total connections: {total_connections}")
    print(f"  Total spines: {len(combined_hologram.spines)}")
    print(f"  Batches completed: {batch_num}")
    print(f"\nFilter statistics:")
    print(f"  Target material found: {game.filter_stats['target_material_found']}")
    print(f"  Wrong material filtered: {game.filter_stats['wrong_material_filtered']}")
    print(f"\nSaved to: {save_dir}")
    print(f"  - final_hologram.pkl")
    print(f"  - backward_seeds.pkl")
    print(f"  - forward_seeds.pkl")
    print(f"  - metadata.pkl")

    return combined_hologram, metadata


def resume_from_seeds(
    save_dir: str = "./holos_targeted",
    max_iterations: int = 20,
    max_memory_mb: int = 3000,
):
    """
    Resume search from saved seeds - demonstrates reproducibility.
    """
    # Load seeds
    backward_seeds_file = os.path.join(save_dir, "backward_seeds.pkl")
    forward_seeds_file = os.path.join(save_dir, "forward_seeds.pkl")
    metadata_file = os.path.join(save_dir, "metadata.pkl")

    with open(backward_seeds_file, 'rb') as f:
        backward_positions = pickle.load(f)
    with open(forward_seeds_file, 'rb') as f:
        forward_positions = pickle.load(f)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    target_material = metadata['target_material']

    print("=" * 70)
    print(f"RESUMING FROM SEEDS: {target_material}")
    print("=" * 70)
    print(f"Backward seeds: {len(backward_positions)}")
    print(f"Forward seeds: {len(forward_positions)}")

    # This would continue the search using saved seeds
    # For now, just verify reproducibility
    return metadata


def run_single_batch(
    target_material: str,
    batch_num: int,
    forward_seeds_file: str,
    backward_seeds_file: str,
    batch_start: int,
    batch_size: int,
    max_iterations: int,
    max_memory_mb: int,
    output_file: str,
    syzygy_path: str = "./syzygy",
):
    """
    Run a single batch - designed to be called in subprocess for memory isolation.
    """
    # Load seeds
    with open(forward_seeds_file, 'rb') as f:
        forward_positions = pickle.load(f)
    with open(backward_seeds_file, 'rb') as f:
        backward_positions = pickle.load(f)

    # Get batch slice
    batch_forward = forward_positions[batch_start:batch_start + batch_size]

    if not batch_forward:
        return None

    # Create fresh game and solver
    game = TargetedChessGame(syzygy_path, target_material)
    solver = HOLOSSolver(game, name=f"batch_{batch_num}", max_memory_mb=max_memory_mb)

    # Convert to SeedPoints
    forward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in batch_forward]
    backward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in backward_positions]

    # Run solver
    start_time = time.time()
    hologram = solver.solve(forward_seeds, backward_seeds, max_iterations=max_iterations)
    elapsed = time.time() - start_time

    # Save batch result
    batch_result = {
        'hologram': hologram,
        'stats': solver.stats,
        'filter_stats': game.filter_stats,
        'elapsed': elapsed,
        'batch_num': batch_num,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(batch_result, f)

    return batch_result


def quick_test(target_material: str = "KQRRvKQR"):
    """Quick test with minimal parameters"""
    print("=" * 70)
    print("QUICK TEST")
    print("=" * 70)

    return run_targeted_search(
        target_material=target_material,
        backward_seeds_count=50,
        forward_seeds_per_material=10,
        max_iterations=5,
        max_memory_mb=500,
        batch_size=20,
        max_batches=2,
        save_dir="./holos_targeted_test"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Targeted HOLOS Search")
    parser.add_argument("--target", default="KQRRvKQR", help="Target material")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--full", action="store_true", help="Full search mode")
    parser.add_argument("--backward-seeds", type=int, default=500, help="Backward seeds count")
    parser.add_argument("--forward-seeds", type=int, default=50, help="Forward seeds per material")
    parser.add_argument("--iterations", type=int, default=20, help="Max iterations per batch")
    parser.add_argument("--memory", type=int, default=3000, help="Max memory MB")
    parser.add_argument("--batches", type=int, default=None, help="Max batches")

    args = parser.parse_args()

    if args.quick:
        quick_test(args.target)
    elif args.full:
        run_targeted_search(
            target_material=args.target,
            backward_seeds_count=args.backward_seeds,
            forward_seeds_per_material=args.forward_seeds,
            max_iterations=args.iterations,
            max_memory_mb=args.memory,
            max_batches=args.batches,
        )
    else:
        # Default: quick test
        quick_test(args.target)
