"""
holos/batched_solver.py - Memory-Bounded Batched Solving

This module wraps HOLOSSolver to handle arbitrarily large searches by:
1. Splitting work into batches that fit in memory
2. Spilling frontiers to disk when memory is tight
3. Processing batches sequentially
4. Merging all results together

KEY PRINCIPLE: No logical limits. The search is IDENTICAL to unbounded search,
just executed in pieces that fit available memory.

The work is the same. Only the scheduling differs.
"""

import os
import pickle
import gc
import time
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from holos.holos import HOLOSSolver, SeedPoint, SearchMode, GoalCondition, GameInterface
from holos.storage import Hologram


@dataclass
class BatchState:
    """State of a batch waiting to be processed"""
    batch_id: int
    forward_hashes: Set[int]  # Hashes of frontier positions
    backward_hashes: Set[int]
    forward_file: str  # Path to pickled positions
    backward_file: str
    depth: int  # How many iterations this batch has been through
    parent_batch: Optional[int] = None  # Which batch spawned this one


@dataclass
class BatchedSolverConfig:
    """Configuration for batched solving"""
    work_dir: str = "./holos_batched"  # Directory for spillover files
    max_memory_mb: int = 8000  # Memory budget per batch
    target_batch_size: int = 500_000  # Target positions per batch
    checkpoint_interval: int = 5  # Save checkpoint every N batches
    max_iterations_per_batch: int = 10  # Iterations before re-batching
    max_total_batches: int = None  # Optional limit for testing (None = unlimited)


class BatchedSolver:
    """
    Solver that handles arbitrarily large searches via batching.

    The key insight: When frontier exceeds memory, we don't LIMIT it.
    We SPLIT it into multiple batches and process them sequentially.

    Each batch:
    1. Loads its frontier from disk
    2. Runs N iterations of HOLOS
    3. Saves solved positions to merged hologram
    4. If frontier still too large, splits into sub-batches
    5. Spills sub-batch frontiers to disk

    This is equivalent to running the full search, just in pieces.
    """

    def __init__(self, game: GameInterface, config: BatchedSolverConfig = None):
        self.game = game
        self.config = config or BatchedSolverConfig()

        os.makedirs(self.config.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.work_dir, "frontiers"), exist_ok=True)
        os.makedirs(os.path.join(self.config.work_dir, "checkpoints"), exist_ok=True)

        # Merged results
        self.hologram = Hologram("batched_solve")

        # Global seen sets (persisted to disk)
        self.forward_seen: Set[int] = set()
        self.backward_seen: Set[int] = set()

        # Queue of batches to process
        self.pending_batches: List[BatchState] = []
        self.completed_batches: List[int] = []

        # Stats
        self.stats = {
            'batches_processed': 0,
            'batches_spawned': 0,
            'total_solved': 0,
            'total_connections': 0,
            'frontier_spills': 0,
            'positions_spilled': 0,
        }

        self._next_batch_id = 0

    def _new_batch_id(self) -> int:
        bid = self._next_batch_id
        self._next_batch_id += 1
        return bid

    def _frontier_path(self, batch_id: int, direction: str) -> str:
        return os.path.join(
            self.config.work_dir, "frontiers",
            f"batch_{batch_id}_{direction}.pkl"
        )

    def _save_frontier(self, positions: Dict[int, Any], batch_id: int, direction: str):
        """Save frontier positions to disk"""
        path = self._frontier_path(batch_id, direction)
        with open(path, 'wb') as f:
            pickle.dump(positions, f)
        self.stats['frontier_spills'] += 1
        self.stats['positions_spilled'] += len(positions)

    def _load_frontier(self, batch_id: int, direction: str) -> Dict[int, Any]:
        """Load frontier positions from disk"""
        path = self._frontier_path(batch_id, direction)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _cleanup_frontier(self, batch_id: int, direction: str):
        """Remove frontier file after processing"""
        path = self._frontier_path(batch_id, direction)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                # Windows file locking - try again after GC
                gc.collect()
                try:
                    os.remove(path)
                except PermissionError:
                    # Still locked, leave it for later cleanup
                    pass

    def _estimate_memory_for_batch(self, fwd_size: int, bwd_size: int) -> float:
        """Estimate memory in MB for a batch of given size"""
        # ~400 bytes per position including all tracking structures
        bytes_per_pos = 400
        total_bytes = (fwd_size + bwd_size) * bytes_per_pos
        return total_bytes / (1024 * 1024)

    def _split_frontier(self, frontier: Dict[int, Any], num_splits: int) -> List[Dict[int, Any]]:
        """Split a frontier dict into N roughly equal parts"""
        items = list(frontier.items())
        splits = []
        chunk_size = (len(items) + num_splits - 1) // num_splits

        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            splits.append(chunk)

        return splits

    def solve(self, forward_seeds: List[SeedPoint],
              backward_seeds: List[SeedPoint] = None,
              goal: GoalCondition = None) -> Hologram:
        """
        Main solve method - handles arbitrarily large searches.

        This is equivalent to HOLOSSolver.solve() but batches work
        to fit in memory.
        """
        print(f"\n{'='*70}")
        print(f"BATCHED HOLOS SOLVER")
        print(f"{'='*70}")
        print(f"Work directory: {self.config.work_dir}")
        print(f"Memory budget: {self.config.max_memory_mb} MB per batch")
        print(f"Target batch size: {self.config.target_batch_size:,} positions")

        start_time = time.time()

        # Initialize first batch with all seeds
        initial_forward = {}
        for seed in forward_seeds:
            h = self.game.hash_state(seed.state)
            if h not in self.forward_seen:
                self.forward_seen.add(h)
                initial_forward[h] = seed.state

        initial_backward = {}
        if backward_seeds:
            for seed in backward_seeds:
                h = self.game.hash_state(seed.state)
                if h not in self.backward_seen:
                    self.backward_seen.add(h)
                    initial_backward[h] = seed.state

                    # Get boundary value
                    if self.game.is_boundary(seed.state):
                        value = self.game.get_boundary_value(seed.state)
                        if value is not None:
                            self.hologram.solved[h] = value

        print(f"Initial forward: {len(initial_forward):,}")
        print(f"Initial backward: {len(initial_backward):,}")
        print(f"Initial boundary values: {len(self.hologram.solved):,}")

        # Create initial batch(es)
        self._create_batches_from_frontiers(initial_forward, initial_backward, depth=0)

        # Process all batches
        while self.pending_batches:
            # Check batch limit (for testing)
            if (self.config.max_total_batches is not None and
                self.stats['batches_processed'] >= self.config.max_total_batches):
                print(f"\nReached batch limit ({self.config.max_total_batches}), stopping.")
                print(f"Remaining batches in queue: {len(self.pending_batches)}")
                break

            batch = self.pending_batches.pop(0)
            self._process_batch(batch, goal)

            # Checkpoint periodically
            if self.stats['batches_processed'] % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            # Force garbage collection between batches
            gc.collect()

        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"BATCHED SOLVE COMPLETE")
        print(f"{'='*70}")
        print(f"Total solved: {len(self.hologram.solved):,}")
        print(f"Batches processed: {self.stats['batches_processed']}")
        print(f"Total connections: {self.stats['total_connections']}")
        print(f"Frontier spills: {self.stats['frontier_spills']}")
        print(f"Positions spilled: {self.stats['positions_spilled']:,}")
        print(f"Elapsed: {elapsed:.1f}s")

        # Final save
        self.hologram.save(os.path.join(self.config.work_dir, "final_hologram.pkl"))

        return self.hologram

    def _create_batches_from_frontiers(self, forward: Dict[int, Any],
                                        backward: Dict[int, Any],
                                        depth: int,
                                        parent_batch: int = None):
        """Create batch(es) from frontiers, splitting if too large"""

        total_size = len(forward) + len(backward)
        est_memory = self._estimate_memory_for_batch(len(forward), len(backward))

        if est_memory <= self.config.max_memory_mb * 0.7:
            # Fits in memory - single batch
            batch_id = self._new_batch_id()
            self._save_frontier(forward, batch_id, "forward")
            self._save_frontier(backward, batch_id, "backward")

            batch = BatchState(
                batch_id=batch_id,
                forward_hashes=set(forward.keys()),
                backward_hashes=set(backward.keys()),
                forward_file=self._frontier_path(batch_id, "forward"),
                backward_file=self._frontier_path(batch_id, "backward"),
                depth=depth,
                parent_batch=parent_batch,
            )
            self.pending_batches.append(batch)
            self.stats['batches_spawned'] += 1
            print(f"  Created batch {batch_id}: {len(forward):,} fwd, {len(backward):,} bwd")
        else:
            # Too large - need to split intelligently
            # The challenge: backward frontier represents all target positions,
            # and all forward positions need to potentially connect to all of them.
            #
            # Strategy:
            # - If backward is the bottleneck, we need to process backward in chunks
            #   and let solved positions accumulate in the hologram
            # - If forward is the bottleneck, split forward and process chunks
            #   against the full backward

            bwd_est = self._estimate_memory_for_batch(0, len(backward))
            fwd_est = self._estimate_memory_for_batch(len(forward), 0)

            if bwd_est > self.config.max_memory_mb * 0.5:
                # Backward is too big - split backward and process in phases
                # Each phase solves some backward positions, which go into hologram
                num_bwd_splits = max(2, int(bwd_est / (self.config.max_memory_mb * 0.3)))
                print(f"  Backward too large ({len(backward):,}), splitting into {num_bwd_splits} phases")

                bwd_splits = self._split_frontier(backward, num_bwd_splits)

                # First phase: process with first chunk of backward
                # Solved positions go into hologram and can be used by later phases
                for i, bwd_chunk in enumerate(bwd_splits):
                    batch_id = self._new_batch_id()
                    self._save_frontier(forward, batch_id, "forward")
                    self._save_frontier(bwd_chunk, batch_id, "backward")

                    batch = BatchState(
                        batch_id=batch_id,
                        forward_hashes=set(forward.keys()),
                        backward_hashes=set(bwd_chunk.keys()),
                        forward_file=self._frontier_path(batch_id, "forward"),
                        backward_file=self._frontier_path(batch_id, "backward"),
                        depth=depth,
                        parent_batch=parent_batch,
                    )
                    self.pending_batches.append(batch)
                    self.stats['batches_spawned'] += 1
                    print(f"  Created phase-batch {batch_id}: {len(forward):,} fwd, {len(bwd_chunk):,} bwd")

            else:
                # Forward is the bottleneck - split forward
                num_splits = max(2, int(est_memory / (self.config.max_memory_mb * 0.5)))
                print(f"  Splitting forward into {num_splits} batches (est {est_memory:.0f} MB)")

                fwd_splits = self._split_frontier(forward, num_splits)

                for i, fwd_chunk in enumerate(fwd_splits):
                    batch_id = self._new_batch_id()
                    self._save_frontier(fwd_chunk, batch_id, "forward")
                    self._save_frontier(backward, batch_id, "backward")

                    batch = BatchState(
                        batch_id=batch_id,
                        forward_hashes=set(fwd_chunk.keys()),
                        backward_hashes=set(backward.keys()),
                        forward_file=self._frontier_path(batch_id, "forward"),
                        backward_file=self._frontier_path(batch_id, "backward"),
                        depth=depth,
                        parent_batch=parent_batch,
                    )
                    self.pending_batches.append(batch)
                    self.stats['batches_spawned'] += 1
                    print(f"  Created sub-batch {batch_id}: {len(fwd_chunk):,} fwd")

    def _process_batch(self, batch: BatchState, goal: GoalCondition = None):
        """Process a single batch"""
        print(f"\n--- Processing Batch {batch.batch_id} (depth {batch.depth}) ---")

        # Load frontiers
        forward = self._load_frontier(batch.batch_id, "forward")
        backward = self._load_frontier(batch.batch_id, "backward")

        print(f"  Loaded: {len(forward):,} forward, {len(backward):,} backward")

        # Create solver for this batch
        # NO artificial limits - we've already sized the batch to fit
        solver = HOLOSSolver(
            self.game,
            name=f"batch_{batch.batch_id}",
            max_memory_mb=self.config.max_memory_mb,
            max_frontier_size=None,  # No limit - batch is pre-sized
            max_backward_depth=None,  # No limit - full search
        )

        # Transfer global seen sets to solver
        solver.forward_seen = set(self.forward_seen)
        solver.backward_seen = set(self.backward_seen)
        solver.solved = dict(self.hologram.solved)

        # Set up frontiers
        solver.forward_frontier = forward
        solver.backward_frontier = backward

        # Initialize backward depths (all at batch.depth from boundary)
        for h in backward:
            solver.backward_depth[h] = batch.depth

        # Run iterations
        batch_start_time = time.time()
        for iteration in range(self.config.max_iterations_per_batch):
            mem = solver.memory_mb()

            if mem > self.config.max_memory_mb * 0.85:
                print(f"  Memory pressure at iteration {iteration}, stopping batch")
                break

            if not solver.forward_frontier and not solver.backward_frontier:
                print(f"  Frontiers exhausted at iteration {iteration}")
                break

            # Run one iteration
            fwd_contacts = solver._expand_forward()
            bwd_contacts = solver._expand_backward()

            new_conns = solver._find_connections()
            if new_conns:
                solver._crystallize()

            solver._propagate()

            print(f"  Iter {iteration}: fwd={len(solver.forward_frontier):,}, "
                  f"bwd={len(solver.backward_frontier):,}, "
                  f"solved={len(solver.solved):,}, mem={mem:.0f}MB")

        batch_elapsed = time.time() - batch_start_time

        # Merge results into global hologram
        new_solved = 0
        for h, v in solver.solved.items():
            if h not in self.hologram.solved:
                self.hologram.solved[h] = v
                new_solved += 1

        # Update global seen sets
        self.forward_seen.update(solver.forward_seen)
        self.backward_seen.update(solver.backward_seen)

        # Merge connections
        self.stats['total_connections'] += solver.stats['connections']

        print(f"  Batch {batch.batch_id} done: +{new_solved:,} solved in {batch_elapsed:.1f}s")

        # If frontiers remain, create new batches for them
        if solver.forward_frontier or solver.backward_frontier:
            print(f"  Remaining frontier: {len(solver.forward_frontier):,} fwd, "
                  f"{len(solver.backward_frontier):,} bwd")
            self._create_batches_from_frontiers(
                solver.forward_frontier,
                solver.backward_frontier,
                depth=batch.depth + 1,
                parent_batch=batch.batch_id,
            )

        # Cleanup
        self._cleanup_frontier(batch.batch_id, "forward")
        self._cleanup_frontier(batch.batch_id, "backward")
        self.completed_batches.append(batch.batch_id)
        self.stats['batches_processed'] += 1
        self.stats['total_solved'] = len(self.hologram.solved)

        # Free memory
        del solver
        del forward
        del backward

    def _save_checkpoint(self):
        """Save current state for resume capability"""
        checkpoint = {
            'hologram': self.hologram,
            'forward_seen': self.forward_seen,
            'backward_seen': self.backward_seen,
            'pending_batches': self.pending_batches,
            'completed_batches': self.completed_batches,
            'stats': self.stats,
            'next_batch_id': self._next_batch_id,
        }
        path = os.path.join(
            self.config.work_dir, "checkpoints",
            f"checkpoint_{self.stats['batches_processed']}.pkl"
        )
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  Checkpoint saved: {path}")

    @classmethod
    def resume(cls, work_dir: str, game: GameInterface,
               goal: GoalCondition = None) -> 'BatchedSolver':
        """Resume from latest checkpoint"""
        # Find latest checkpoint
        checkpoint_dir = os.path.join(work_dir, "checkpoints")
        checkpoints = sorted([
            f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")
        ])

        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")

        latest = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"Resuming from: {latest}")

        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)

        # Reconstruct solver
        config = BatchedSolverConfig(work_dir=work_dir)
        solver = cls(game, config)
        solver.hologram = checkpoint['hologram']
        solver.forward_seen = checkpoint['forward_seen']
        solver.backward_seen = checkpoint['backward_seen']
        solver.pending_batches = checkpoint['pending_batches']
        solver.completed_batches = checkpoint['completed_batches']
        solver.stats = checkpoint['stats']
        solver._next_batch_id = checkpoint['next_batch_id']

        print(f"Resumed: {len(solver.hologram.solved):,} solved, "
              f"{len(solver.pending_batches)} batches pending")

        # Continue processing
        return solver


def solve_batched(game: GameInterface,
                  forward_seeds: List[SeedPoint],
                  backward_seeds: List[SeedPoint] = None,
                  goal: GoalCondition = None,
                  work_dir: str = "./holos_batched",
                  max_memory_mb: int = 8000) -> Hologram:
    """
    Convenience function for batched solving.

    This is the recommended way to run large searches.
    """
    config = BatchedSolverConfig(
        work_dir=work_dir,
        max_memory_mb=max_memory_mb,
    )
    solver = BatchedSolver(game, config)
    return solver.solve(forward_seeds, backward_seeds, goal)
