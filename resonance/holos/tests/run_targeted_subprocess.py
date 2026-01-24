"""
holos/run_targeted_subprocess.py - Subprocess-isolated batch runner

Runs each batch in a fresh subprocess to truly isolate memory.
This is the recommended approach for large-scale searches.
"""

import sys
import os
import pickle
import time
import subprocess
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def run_batch_subprocess(
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
    """Run a batch in a subprocess for memory isolation"""

    # Use raw strings for paths to avoid escape issues
    fwd_file = forward_seeds_file.replace('\\', '/')
    bwd_file = backward_seeds_file.replace('\\', '/')
    out_file = output_file.replace('\\', '/')
    syzygy = syzygy_path.replace('\\', '/')

    # Create a script to run in subprocess
    script = f'''
import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.chess_targeted import TargetedChessGame

# Load seeds
with open("{fwd_file}", 'rb') as f:
    forward_positions = pickle.load(f)
with open("{bwd_file}", 'rb') as f:
    backward_positions = pickle.load(f)

# Get batch slice
batch_forward = forward_positions[{batch_start}:{batch_start + batch_size}]

if not batch_forward:
    print("No positions in batch")
    sys.exit(0)

# Create fresh game and solver
game = TargetedChessGame("{syzygy}", "{target_material}")
solver = HOLOSSolver(game, name="batch_{batch_num}", max_memory_mb={max_memory_mb})

# Convert to SeedPoints
forward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in batch_forward]
backward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in backward_positions]

# Run solver
import time
start_time = time.time()
hologram = solver.solve(forward_seeds, backward_seeds, max_iterations={max_iterations})
elapsed = time.time() - start_time

# Save batch result
batch_result = {{
    'hologram': hologram,
    'stats': solver.stats,
    'filter_stats': game.filter_stats,
    'elapsed': elapsed,
    'batch_num': {batch_num},
}}

with open("{out_file}", 'wb') as f:
    pickle.dump(batch_result, f)

print(f"Batch {batch_num} complete: {{len(hologram.solved):,}} solved in {{elapsed:.1f}}s")
'''

    # Write script to temp file
    script_file = f"_batch_{batch_num}_script.py"
    with open(script_file, 'w') as f:
        f.write(script)

    try:
        # Run in subprocess
        result = subprocess.run(
            [sys.executable, script_file],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per batch
        )

        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"Batch {batch_num} timed out!")
        return False
    finally:
        # Clean up script
        if os.path.exists(script_file):
            os.remove(script_file)


def run_targeted_subprocess(
    target_material: str = "KQRRvKQR",
    syzygy_path: str = "./syzygy",
    save_dir: str = "./holos_targeted_sp",
    backward_seeds_count: int = 500,
    forward_seeds_per_material: int = 50,
    max_iterations: int = 10,
    max_memory_mb: int = 2000,
    batch_size: int = 50,
    max_batches: int = None,
):
    """
    Run targeted search with subprocess isolation between batches.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"TARGETED HOLOS SEARCH (SUBPROCESS MODE): {target_material}")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    # Phase 1: Generate seeds (in main process - only once)
    print("\n--- PHASE 1: Generate Seeds ---")

    from holos.games.chess_targeted import TargetedChessGame

    game = TargetedChessGame(syzygy_path, target_material)

    backward_positions = game.generate_target_boundary_seeds(backward_seeds_count)
    backward_seeds_file = os.path.join(save_dir, "backward_seeds.pkl")
    with open(backward_seeds_file, 'wb') as f:
        pickle.dump(backward_positions, f)
    print(f"Saved {len(backward_positions)} backward seeds")

    forward_positions = game.generate_source_positions(forward_seeds_per_material)
    forward_seeds_file = os.path.join(save_dir, "forward_seeds.pkl")
    with open(forward_seeds_file, 'wb') as f:
        pickle.dump(forward_positions, f)
    print(f"Saved {len(forward_positions)} forward seeds")

    # Clean up game to free memory before batches
    del game

    # Phase 2: Run batches in subprocesses
    print("\n--- PHASE 2: Subprocess Batches ---")

    from holos.storage import Hologram

    combined_hologram = Hologram(f"targeted_{target_material}")
    total_connections = 0
    batch_num = 0
    total_filter_stats = {'target_material_found': 0, 'wrong_material_filtered': 0}

    for i in range(0, len(forward_positions), batch_size):
        if max_batches is not None and batch_num >= max_batches:
            print(f"Reached max batches ({max_batches})")
            break

        batch_num += 1
        output_file = os.path.join(save_dir, f"batch_{batch_num}_result.pkl")

        print(f"\n=== Batch {batch_num} (positions {i}-{i+batch_size}) ===")

        success = run_batch_subprocess(
            target_material=target_material,
            batch_num=batch_num,
            forward_seeds_file=forward_seeds_file,
            backward_seeds_file=backward_seeds_file,
            batch_start=i,
            batch_size=batch_size,
            max_iterations=max_iterations,
            max_memory_mb=max_memory_mb,
            output_file=output_file,
            syzygy_path=syzygy_path,
        )

        if success and os.path.exists(output_file):
            # Load and merge result
            with open(output_file, 'rb') as f:
                batch_result = pickle.load(f)

            combined_hologram = combined_hologram.merge(batch_result['hologram'])
            total_connections += batch_result['stats']['connections']

            for k, v in batch_result['filter_stats'].items():
                total_filter_stats[k] = total_filter_stats.get(k, 0) + v

            print(f"  Merged: {len(combined_hologram.solved):,} total solved")

            # Clean up batch file
            os.remove(output_file)
        else:
            print(f"  Batch {batch_num} failed or no output")

    # Phase 3: Save final results
    print("\n--- PHASE 3: Save Results ---")

    final_file = os.path.join(save_dir, "final_hologram.pkl")
    combined_hologram.save(final_file)

    metadata = {
        'target_material': target_material,
        'backward_seeds_count': len(backward_positions),
        'forward_seeds_count': len(forward_positions),
        'total_solved': len(combined_hologram.solved),
        'total_connections': total_connections,
        'total_spines': len(combined_hologram.spines),
        'batches': batch_num,
        'completed': datetime.now().isoformat(),
        'filter_stats': total_filter_stats,
    }

    metadata_file = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nFinal Results:")
    print(f"  Total solved: {len(combined_hologram.solved):,}")
    print(f"  Total connections: {total_connections}")
    print(f"  Batches completed: {batch_num}")
    print(f"  Filter stats: {total_filter_stats}")
    print(f"\nSaved to: {save_dir}")

    return combined_hologram, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Targeted HOLOS (Subprocess Mode)")
    parser.add_argument("--target", default="KQRRvKQR", help="Target material")
    parser.add_argument("--backward-seeds", type=int, default=200, help="Backward seeds")
    parser.add_argument("--forward-seeds", type=int, default=30, help="Forward seeds per material")
    parser.add_argument("--iterations", type=int, default=10, help="Max iterations")
    parser.add_argument("--memory", type=int, default=2000, help="Max memory MB per batch")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--batches", type=int, default=None, help="Max batches")

    args = parser.parse_args()

    run_targeted_subprocess(
        target_material=args.target,
        backward_seeds_count=args.backward_seeds,
        forward_seeds_per_material=args.forward_seeds,
        max_iterations=args.iterations,
        max_memory_mb=args.memory,
        batch_size=args.batch_size,
        max_batches=args.batches,
    )
