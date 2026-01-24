"""
holos/full_search.py - Full Exhaustive Search Extension

This module EXTENDS the existing HOLOS infrastructure for large-scale searches:

1. EXTENDS storage.py:
   - DiskBackedHologram: Hologram subclass that writes to disk incrementally
   - Avoids RAM exhaustion for billions of positions

2. EXTENDS session.py:
   - FullSearchSession: SessionManager subclass for exhaustive searches
   - Subprocess isolation for memory safety
   - Disk space monitoring

3. INTEGRATES with holos.py:
   - Uses HOLOSSolver directly (no parallel implementation)
   - Uses GoalCondition for targeting
   - Respects all existing parameters (max_memory_mb, max_frontier_size, etc.)

4. INTEGRATES with games/:
   - Uses chess.py with target_material for material filtering
   - Uses chess.py material utilities (get_parent_materials, etc.)

Usage:
    # Initialize and run
    python holos/full_search.py --target KQRRvKQR --init
    python holos/full_search.py --target KQRRvKQR --run

    # Resume after crash
    python holos/full_search.py --target KQRRvKQR --run

    # Check status
    python holos/full_search.py --target KQRRvKQR --status
"""

import os
import sys
import time
import json
import pickle
import shutil
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.storage import Hologram, SpinePath
from holos.session import SessionManager, SessionState, SessionPhase, RoundStats
from holos.holos import HOLOSSolver, SeedPoint, SearchMode, GoalCondition


# ============================================================
# DISK-BACKED HOLOGRAM EXTENSION
# ============================================================

class DiskBackedHologram(Hologram):
    """
    Hologram subclass that incrementally writes to disk.

    For very large searches (billions of positions), keeping everything
    in RAM is impossible. This class:

    1. Writes solved positions to disk in chunks
    2. Keeps only recent positions in RAM for propagation
    3. Can query disk for any position
    4. Maintains full Hologram compatibility

    Disk format:
        {name}_solved/
            chunk_0000.pkl  # Dict[int, value] for positions 0-999999
            chunk_0001.pkl
            ...
        {name}_spines.pkl
        {name}_meta.pkl
    """

    def __init__(self, name: str, disk_dir: str, ram_limit: int = 1_000_000):
        """
        Args:
            name: Hologram name
            disk_dir: Directory for disk storage
            ram_limit: Max positions to keep in RAM before flushing
        """
        super().__init__(name)
        self.disk_dir = Path(disk_dir)
        self.disk_dir.mkdir(parents=True, exist_ok=True)

        self.ram_limit = ram_limit
        self.chunk_size = 500_000  # Positions per chunk file
        self.current_chunk = 0
        self.disk_count = 0  # Positions written to disk

        # Track what's on disk
        self.chunk_dir = self.disk_dir / f"{name}_solved"
        self.chunk_dir.mkdir(exist_ok=True)

        # Load existing state if resuming
        self._load_meta()

    def _meta_path(self) -> Path:
        return self.disk_dir / f"{self.name}_meta.pkl"

    def _spines_path(self) -> Path:
        return self.disk_dir / f"{self.name}_spines.pkl"

    def _chunk_path(self, chunk_id: int) -> Path:
        return self.chunk_dir / f"chunk_{chunk_id:04d}.pkl"

    def _load_meta(self):
        """Load metadata if resuming"""
        if self._meta_path().exists():
            with open(self._meta_path(), 'rb') as f:
                meta = pickle.load(f)
            self.disk_count = meta.get('disk_count', 0)
            self.current_chunk = meta.get('current_chunk', 0)
            self.boundary_hashes = meta.get('boundary_hashes', set())
            self.equiv_outcomes = meta.get('equiv_outcomes', {})
            self.stats = meta.get('stats', {})
            print(f"Resumed DiskBackedHologram: {self.disk_count:,} on disk")

        if self._spines_path().exists():
            with open(self._spines_path(), 'rb') as f:
                self.spines = pickle.load(f)

    def _save_meta(self):
        """Save metadata"""
        meta = {
            'disk_count': self.disk_count,
            'current_chunk': self.current_chunk,
            'boundary_hashes': self.boundary_hashes,
            'equiv_outcomes': self.equiv_outcomes,
            'stats': self.stats,
        }
        with open(self._meta_path(), 'wb') as f:
            pickle.dump(meta, f)

        with open(self._spines_path(), 'wb') as f:
            pickle.dump(self.spines, f)

    def flush_to_disk(self, force: bool = False):
        """
        Flush RAM solved positions to disk if over limit.

        Args:
            force: Flush even if under limit
        """
        if len(self.solved) < self.ram_limit and not force:
            return

        if not self.solved:
            return

        # Write current solved dict to chunk file
        chunk_path = self._chunk_path(self.current_chunk)

        # If chunk exists, merge with it
        if chunk_path.exists():
            with open(chunk_path, 'rb') as f:
                existing = pickle.load(f)
            existing.update(self.solved)
            to_write = existing
        else:
            to_write = dict(self.solved)

        # Check if chunk is full
        if len(to_write) >= self.chunk_size:
            # Write full chunk
            with open(chunk_path, 'wb') as f:
                pickle.dump(dict(list(to_write.items())[:self.chunk_size]), f)

            # Start new chunk with overflow
            self.current_chunk += 1
            overflow = dict(list(to_write.items())[self.chunk_size:])
            if overflow:
                with open(self._chunk_path(self.current_chunk), 'wb') as f:
                    pickle.dump(overflow, f)
        else:
            with open(chunk_path, 'wb') as f:
                pickle.dump(to_write, f)

        self.disk_count += len(self.solved)
        self.solved.clear()
        self._save_meta()

    def query(self, h: int) -> Optional[Any]:
        """Query value, checking RAM then disk"""
        # Check RAM first
        if h in self.solved:
            return self.solved[h]

        # Check disk chunks
        for chunk_id in range(self.current_chunk + 1):
            chunk_path = self._chunk_path(chunk_id)
            if chunk_path.exists():
                with open(chunk_path, 'rb') as f:
                    chunk = pickle.load(f)
                if h in chunk:
                    return chunk[h]

        return None

    def total_count(self) -> int:
        """Total positions (RAM + disk)"""
        return len(self.solved) + self.disk_count

    def merge_hologram(self, other: Hologram):
        """
        Merge another hologram into this one.

        Unlike parent merge(), this writes to disk incrementally.
        """
        # Add solved positions
        for h, v in other.solved.items():
            if h not in self.solved:
                self.solved[h] = v

        # Flush if needed
        self.flush_to_disk()

        # Merge spines (dedupe)
        existing_starts = {s.start_hash for s in self.spines}
        for spine in other.spines:
            if spine.start_hash not in existing_starts:
                self.spines.append(spine)

        # Merge other attributes
        self.boundary_hashes |= other.boundary_hashes

        for features, outcome in other.equiv_outcomes.items():
            if features not in self.equiv_outcomes:
                self.equiv_outcomes[features] = outcome

    def finalize(self):
        """Finalize: flush remaining RAM to disk"""
        self.flush_to_disk(force=True)
        self._save_meta()
        print(f"Finalized: {self.total_count():,} total positions")

    def summary(self) -> str:
        """Extended summary with disk stats"""
        base = super().summary()
        disk_size = sum(
            f.stat().st_size for f in self.chunk_dir.glob("*.pkl")
        ) / (1024**2) if self.chunk_dir.exists() else 0

        return (f"{base}\n"
                f"  Disk positions: {self.disk_count:,}\n"
                f"  RAM positions: {len(self.solved):,}\n"
                f"  Disk size: {disk_size:.1f} MB\n"
                f"  Chunks: {self.current_chunk + 1}")


# ============================================================
# FULL SEARCH SESSION EXTENSION
# ============================================================

@dataclass
class FullSearchState(SessionState):
    """Extended session state for full exhaustive searches"""

    # Target material info
    target_material: str = ""
    source_materials: List[str] = field(default_factory=list)

    # Seed tracking
    backward_seeds_file: str = ""
    forward_seeds_file: str = ""
    total_forward_seeds: int = 0

    # Batch tracking
    batch_size: int = 100
    batches_completed: int = 0
    batches_total: int = 0

    # Subprocess settings
    subprocess_memory_mb: int = 3000
    subprocess_iterations: int = 12

    # Resource tracking
    peak_memory_mb: float = 0
    total_time_hours: float = 0

    def summary(self) -> str:
        base = super().summary()
        return (f"{base}\n"
                f"  Target: {self.target_material}\n"
                f"  Batches: {self.batches_completed}/{self.batches_total}\n"
                f"  Time: {self.total_time_hours:.2f} hours")


class FullSearchSession(SessionManager):
    """
    SessionManager extension for full exhaustive searches.

    Key differences from base SessionManager:
    1. Uses subprocess isolation for memory safety
    2. Uses DiskBackedHologram for large result sets
    3. Systematic seed enumeration (not incremental)
    4. Resource monitoring (disk space, memory)
    """

    def __init__(self,
                 target_material: str,
                 save_dir: str = None,
                 syzygy_path: str = "./syzygy",
                 # Batch parameters
                 batch_size: int = 100,
                 subprocess_memory_mb: int = 3000,
                 subprocess_iterations: int = 12,
                 # Resource limits
                 min_disk_gb: float = 10.0):

        self.target_material = target_material
        self.syzygy_path = syzygy_path
        self.batch_size = batch_size
        self.subprocess_memory_mb = subprocess_memory_mb
        self.subprocess_iterations = subprocess_iterations
        self.min_disk_gb = min_disk_gb

        # Default save dir
        if save_dir is None:
            save_dir = f"./search_{target_material}"

        # Initialize parent (creates session_id from target)
        session_id = f"full_{target_material}"
        super().__init__(session_id, "chess_targeted", save_dir)

        # Override state with extended version
        self._upgrade_state()

        # Use disk-backed hologram
        self.hologram = DiskBackedHologram(
            session_id,
            disk_dir=save_dir,
            ram_limit=500_000
        )

    def _upgrade_state(self):
        """Upgrade SessionState to FullSearchState if needed"""
        if not isinstance(self.state, FullSearchState):
            # Convert existing state
            old = self.state
            self.state = FullSearchState(
                session_id=old.session_id,
                game_name=old.game_name,
                created_at=old.created_at,
                phase=old.phase,
                current_round=old.current_round,
                total_solved=old.total_solved,
                total_explored=old.total_explored,
                total_connections=old.total_connections,
                rounds=old.rounds,
                pending_seeds=old.pending_seeds,
                explored_seeds=old.explored_seeds,
                total_budget=old.total_budget,
                budget_used=old.budget_used,
                # Extended fields
                target_material=self.target_material,
                batch_size=self.batch_size,
                subprocess_memory_mb=self.subprocess_memory_mb,
                subprocess_iterations=self.subprocess_iterations,
            )

    def initialize_seeds(self,
                        backward_count: int = 5000,
                        forward_per_material: int = 200):
        """
        Generate and save seeds for the full search.

        Uses ChessGame with target_material to generate:
        - backward_seeds: Target material (7-piece) positions
        - forward_seeds: Source material (8-piece) positions
        """
        from holos.games.chess import ChessGame

        print("=" * 70)
        print(f"INITIALIZING SEEDS: {self.target_material}")
        print("=" * 70)

        target_pieces = len(self.target_material.replace('V', '').replace('v', ''))
        game = ChessGame(
            self.syzygy_path,
            min_pieces=target_pieces,
            max_pieces=target_pieces + 1,
            target_material=self.target_material
        )

        # Generate backward seeds
        print(f"Generating {backward_count} backward seeds...")
        backward_seeds = game.generate_target_boundary_seeds(backward_count)
        backward_file = os.path.join(self.save_dir, "backward_seeds.pkl")
        with open(backward_file, 'wb') as f:
            pickle.dump(backward_seeds, f)

        # Generate forward seeds
        print(f"Generating forward seeds ({forward_per_material} per material)...")
        forward_seeds = game.generate_source_positions(forward_per_material)
        forward_file = os.path.join(self.save_dir, "forward_seeds.pkl")
        with open(forward_file, 'wb') as f:
            pickle.dump(forward_seeds, f)

        # Update state
        self.state.backward_seeds_file = backward_file
        self.state.forward_seeds_file = forward_file
        self.state.total_forward_seeds = len(forward_seeds)
        self.state.source_materials = game.source_materials
        self.state.batches_total = (len(forward_seeds) + self.batch_size - 1) // self.batch_size

        self.save()

        print(f"\nSeeds generated:")
        print(f"  Backward: {len(backward_seeds)}")
        print(f"  Forward: {len(forward_seeds)}")
        print(f"  Batches: {self.state.batches_total}")

    def check_resources(self) -> Tuple[bool, str]:
        """Check if resources are sufficient to continue"""
        # Check disk space
        total, used, free = shutil.disk_usage(self.save_dir)
        free_gb = free / (1024**3)
        if free_gb < self.min_disk_gb:
            return False, f"Disk space low: {free_gb:.1f} GB"

        return True, f"OK (disk: {free_gb:.1f} GB free)"

    def run_batch_subprocess(self, batch_num: int,
                             batch_start: int) -> Tuple[bool, Optional[Dict]]:
        """
        Run a single batch in a subprocess.

        This provides memory isolation - each batch starts fresh.
        """
        output_file = os.path.join(self.save_dir, f"_batch_{batch_num}_result.pkl")

        # Build script
        script = self._build_subprocess_script(
            batch_num, batch_start, output_file
        )

        script_file = os.path.join(self.save_dir, f"_batch_{batch_num}_script.py")
        with open(script_file, 'w') as f:
            f.write(script)

        try:
            result = subprocess.run(
                [sys.executable, script_file],
                capture_output=True,
                text=True,
                timeout=1200  # 20 min timeout
            )

            # Print output
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"  {line}")

            if result.returncode != 0:
                print(f"  STDERR: {result.stderr[:500]}")
                return False, None

            # Load result
            if os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    batch_result = pickle.load(f)
                os.remove(output_file)
                return True, batch_result

            return False, None

        except subprocess.TimeoutExpired:
            print(f"  Batch {batch_num} timed out")
            return False, None
        finally:
            if os.path.exists(script_file):
                os.remove(script_file)

    def _build_subprocess_script(self, batch_num: int, batch_start: int,
                                  output_file: str) -> str:
        """Build subprocess script that uses HOLOSSolver properly"""

        # Escape paths
        fwd = self.state.forward_seeds_file.replace('\\', '/')
        bwd = self.state.backward_seeds_file.replace('\\', '/')
        out = output_file.replace('\\', '/')
        syzygy = self.syzygy_path.replace('\\', '/')

        return f'''
import sys
import os
import pickle
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.chess import ChessGame

# Load seeds
with open("{fwd}", 'rb') as f:
    forward_positions = pickle.load(f)
with open("{bwd}", 'rb') as f:
    backward_positions = pickle.load(f)

# Get batch slice
batch_forward = forward_positions[{batch_start}:{batch_start + self.batch_size}]

if not batch_forward:
    print("No positions in batch")
    sys.exit(0)

# Create targeted chess game
target_material = "{self.target_material}"
target_pieces = len(target_material.replace('V', '').replace('v', ''))
game = ChessGame(
    "{syzygy}",
    min_pieces=target_pieces,
    max_pieces=target_pieces + 1,
    target_material=target_material
)
solver = HOLOSSolver(
    game,
    name="batch_{batch_num}",
    max_memory_mb={self.subprocess_memory_mb},
    max_frontier_size=2_000_000,
)

# Create SeedPoints (standard HOLOS interface)
forward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in batch_forward]
backward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in backward_positions]

# Run solver (standard HOLOS solve)
start = time.time()
hologram = solver.solve(forward_seeds, backward_seeds, max_iterations={self.subprocess_iterations})
elapsed = time.time() - start

# Package result
result = {{
    'hologram': hologram,
    'stats': solver.stats,
    'filter_stats': game.filter_stats,
    'elapsed': elapsed,
    'batch_num': {batch_num},
}}

with open("{out}", 'wb') as f:
    pickle.dump(result, f)

print(f"Solved: {{len(hologram.solved):,}} in {{elapsed:.1f}}s")
print(f"Connections: {{solver.stats.get('connections', 0)}}")
'''

    def run(self, max_batches: int = None):
        """
        Run the full search.

        Processes batches of forward seeds, using subprocess isolation.
        Results are merged into DiskBackedHologram.
        """
        print("=" * 70)
        print(f"FULL SEARCH: {self.target_material}")
        print("=" * 70)
        print(self.state.summary())

        # Load seeds to get counts
        with open(self.state.forward_seeds_file, 'rb') as f:
            forward_seeds = pickle.load(f)

        total_batches = self.state.batches_total
        start_batch = self.state.batches_completed

        if max_batches:
            total_batches = min(total_batches, start_batch + max_batches)

        print(f"\nBatches: {start_batch + 1} to {total_batches}")

        start_time = time.time()

        for batch_num in range(start_batch + 1, total_batches + 1):
            # Check resources
            ok, msg = self.check_resources()
            if not ok:
                print(f"\n!!! STOPPING: {msg} !!!")
                break

            batch_start = (batch_num - 1) * self.batch_size
            print(f"\n=== Batch {batch_num}/{total_batches} (seeds {batch_start}-{batch_start + self.batch_size}) ===")

            success, result = self.run_batch_subprocess(batch_num, batch_start)

            if success and result:
                # Merge into disk-backed hologram
                self.hologram.merge_hologram(result['hologram'])

                # Update state
                self.state.batches_completed = batch_num
                self.state.total_solved = self.hologram.total_count()
                self.state.total_connections += result['stats'].get('connections', 0)

                elapsed_hours = (time.time() - start_time) / 3600
                self.state.total_time_hours += elapsed_hours

                self.save()

                print(f"  Total: {self.hologram.total_count():,} solved")
            else:
                print(f"  Batch {batch_num} failed")

        # Finalize
        self.hologram.finalize()
        self.save()

        print("\n" + "=" * 70)
        print("SEARCH COMPLETE")
        print("=" * 70)
        print(self.state.summary())
        print(self.hologram.summary())


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full exhaustive HOLOS search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize (generate seeds)
  python full_search.py --target KQRRvKQR --init

  # Run (or resume)
  python full_search.py --target KQRRvKQR --run

  # Run limited batches
  python full_search.py --target KQRRvKQR --run --max-batches 50

  # Check status
  python full_search.py --target KQRRvKQR --status
"""
    )

    parser.add_argument("--target", default="KQRRvKQR",
                        help="Target material")
    parser.add_argument("--search-dir", default=None,
                        help="Search directory")
    parser.add_argument("--syzygy", default="./syzygy",
                        help="Syzygy path")

    # Actions
    parser.add_argument("--init", action="store_true",
                        help="Initialize seeds")
    parser.add_argument("--run", action="store_true",
                        help="Run search")
    parser.add_argument("--status", action="store_true",
                        help="Show status")

    # Parameters
    parser.add_argument("--backward-seeds", type=int, default=5000)
    parser.add_argument("--forward-seeds", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--memory", type=int, default=3000)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--min-disk-gb", type=float, default=10.0)

    args = parser.parse_args()

    session = FullSearchSession(
        target_material=args.target,
        save_dir=args.search_dir,
        syzygy_path=args.syzygy,
        batch_size=args.batch_size,
        subprocess_memory_mb=args.memory,
        subprocess_iterations=args.iterations,
        min_disk_gb=args.min_disk_gb,
    )

    if args.init:
        session.initialize_seeds(
            backward_count=args.backward_seeds,
            forward_per_material=args.forward_seeds,
        )
        print("\nInitialized. Run with --run to start.")

    elif args.run:
        if not session.state.forward_seeds_file:
            print("Not initialized. Run with --init first.")
            return
        session.run(max_batches=args.max_batches)

    elif args.status:
        print(session.state.summary())
        print(session.hologram.summary())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
