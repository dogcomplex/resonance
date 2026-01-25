"""
holos/seed_meta_game.py - Layer 2 Meta-Game Over Seeds

The Key Insight:
Seeds themselves have structure. If seed B is reachable from seed A
via a single move, we can store A + the move instead of B.

This creates a RECURSIVE compression:
  Layer 0: 4.5T positions → 337B seeds (13x)
  Layer 1: 337B seeds → ?M meta-seeds (potential 10-100x more)

The SeedMetaGame treats seeds as positions in a meta-game:
  - State: A seed (or set of seeds)
  - Move: "Generate seed B from seed A via move M"
  - Boundary: Seeds that cannot be generated from others (true meta-seeds)
  - Value: Coverage (how many seeds does this meta-seed generate?)

This is a FULL HOLOS search over seed space!
"""

import os
import sys
import time
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import GameInterface, HOLOSSolver, SeedPoint, SearchMode
from holos.games.connect4 import Connect4Game, C4State


# ============================================================
# SEED RELATIONSHIP STRUCTURES
# ============================================================

@dataclass(frozen=True)
class SeedRef:
    """
    Reference to a seed position.

    Can be either:
    1. Direct: Store the full seed
    2. Derived: Store (parent_seed, move) pair
    """
    seed_hash: int
    is_direct: bool = True
    parent_hash: Optional[int] = None
    move: Optional[Any] = None

    def storage_size(self) -> int:
        if self.is_direct:
            return 45  # Full seed storage
        else:
            return 12  # parent_hash (8) + move (4)


@dataclass
class MetaSeed:
    """
    A meta-seed: generates multiple seeds via expansion.

    Like a seed generates positions, a meta-seed generates seeds.
    """
    root_hash: int           # The root seed hash
    root_state: Any          # The actual seed state
    derived_count: int = 0   # How many seeds can be derived from this
    depth: int = 0           # Expansion depth

    def __hash__(self):
        return hash((self.root_hash, self.depth))

    def __eq__(self, other):
        if not isinstance(other, MetaSeed):
            return False
        return self.root_hash == other.root_hash and self.depth == other.depth


@dataclass
class MetaSeedValue:
    """
    Value of a meta-seed: how much compression does it provide?
    """
    coverage: int           # Seeds generated from this meta-seed
    direct_seeds: int       # Seeds that must be stored directly
    compression_ratio: float
    depth: int

    def __lt__(self, other):
        return self.coverage < other.coverage


# ============================================================
# SEED META-GAME
# ============================================================

class SeedMetaGame(GameInterface):
    """
    A meta-game where positions are SEEDS and moves are seed derivations.

    This enables HOLOS to search the space of seed compressions!

    State: MetaSeed (a root seed + expansion depth)
    Move: Increase depth or switch to different root
    Value: MetaSeedValue (coverage metrics)
    Boundary: Meta-seeds that provide good coverage
    """

    def __init__(self,
                 seed_set: Dict[int, Any],  # hash -> seed state
                 underlying_game: GameInterface,
                 max_depth: int = 3):
        """
        Args:
            seed_set: Dictionary of seed hash -> seed state
            underlying_game: The game these seeds come from
            max_depth: Maximum derivation depth to explore
        """
        self.seed_set = seed_set
        self.underlying_game = underlying_game
        self.max_depth = max_depth

        # Build reverse index: which seeds are reachable from which?
        self._build_derivation_graph()

        # Cache evaluations
        self.eval_cache: Dict[int, MetaSeedValue] = {}

    def _build_derivation_graph(self):
        """
        Build the derivation graph: seed_hash -> list of (child_hash, move)
        """
        print("Building seed derivation graph...")
        start = time.time()

        # Forward: parent -> children
        self.derivable_from: Dict[int, List[Tuple[int, Any]]] = defaultdict(list)

        # Reverse: child -> parents
        self.derived_from: Dict[int, List[int]] = defaultdict(list)

        seed_hashes = set(self.seed_set.keys())

        for seed_hash, seed_state in self.seed_set.items():
            # Get successors in underlying game
            for child, move in self.underlying_game.get_successors(seed_state):
                child_hash = self.underlying_game.hash_state(child)
                if child_hash in seed_hashes:
                    self.derivable_from[seed_hash].append((child_hash, move))
                    self.derived_from[child_hash].append(seed_hash)

        # Count statistics
        can_derive = sum(1 for h in seed_hashes if self.derivable_from[h])
        is_derived = sum(1 for h in seed_hashes if self.derived_from[h])

        elapsed = time.time() - start
        print(f"  Built in {elapsed:.2f}s")
        print(f"  Seeds that can derive others: {can_derive:,}")
        print(f"  Seeds derivable from others: {is_derived:,}")
        print(f"  True roots (not derivable): {len(seed_hashes) - is_derived:,}")

    def hash_state(self, state: MetaSeed) -> int:
        return hash(state)

    def get_successors(self, state: MetaSeed) -> List[Tuple[MetaSeed, Any]]:
        """
        Successors = increase depth OR switch to a derivable seed as new root
        """
        successors = []

        # Option 1: Increase depth (if not at max)
        if state.depth < self.max_depth:
            new_state = MetaSeed(
                root_hash=state.root_hash,
                root_state=state.root_state,
                depth=state.depth + 1
            )
            successors.append((new_state, ("depth", state.depth + 1)))

        # Option 2: Switch to a derived seed as new root
        # This explores different "meta-seed" choices
        for child_hash, move in self.derivable_from.get(state.root_hash, [])[:5]:
            child_state = self.seed_set[child_hash]
            new_state = MetaSeed(
                root_hash=child_hash,
                root_state=child_state,
                depth=0
            )
            successors.append((new_state, ("switch", child_hash, move)))

        return successors

    def get_predecessors(self, state: MetaSeed) -> List[Tuple[MetaSeed, Any]]:
        """
        Predecessors = decrease depth OR find parent seeds
        """
        predecessors = []

        # Option 1: Decrease depth
        if state.depth > 0:
            new_state = MetaSeed(
                root_hash=state.root_hash,
                root_state=state.root_state,
                depth=state.depth - 1
            )
            predecessors.append((new_state, ("depth", state.depth - 1)))

        # Option 2: This seed could be derived from a parent seed
        for parent_hash in self.derived_from.get(state.root_hash, [])[:5]:
            parent_state = self.seed_set[parent_hash]
            new_state = MetaSeed(
                root_hash=parent_hash,
                root_state=parent_state,
                depth=1  # Parent at depth 1 reaches this seed
            )
            predecessors.append((new_state, ("parent", parent_hash)))

        return predecessors

    def is_boundary(self, state: MetaSeed) -> bool:
        """
        Boundary = evaluated meta-seeds with good coverage
        """
        # Seeds that can't derive others are "terminals"
        if state.root_hash not in self.derivable_from:
            return True
        if not self.derivable_from[state.root_hash]:
            return True
        return False

    def get_boundary_value(self, state: MetaSeed) -> MetaSeedValue:
        """Get value of a boundary meta-seed"""
        return self.evaluate(state)

    def is_terminal(self, state: MetaSeed) -> Tuple[bool, Optional[MetaSeedValue]]:
        """Terminal if at max depth or no derivations possible"""
        if state.depth >= self.max_depth:
            return True, self.evaluate(state)
        if not self.derivable_from.get(state.root_hash):
            return True, self.evaluate(state)
        return False, None

    def evaluate(self, state: MetaSeed) -> MetaSeedValue:
        """
        Evaluate a meta-seed: how many seeds can it generate?
        """
        h = self.hash_state(state)
        if h in self.eval_cache:
            return self.eval_cache[h]

        # BFS expansion to count reachable seeds
        reachable = {state.root_hash}
        frontier = {state.root_hash}

        for d in range(state.depth):
            next_frontier = set()
            for seed_hash in frontier:
                for child_hash, move in self.derivable_from.get(seed_hash, []):
                    if child_hash not in reachable:
                        reachable.add(child_hash)
                        next_frontier.add(child_hash)
            frontier = next_frontier

        coverage = len(reachable)
        direct = len(self.seed_set) - coverage + 1  # 1 for this meta-seed
        compression = len(self.seed_set) / max(direct, 1)

        value = MetaSeedValue(
            coverage=coverage,
            direct_seeds=direct,
            compression_ratio=compression,
            depth=state.depth
        )

        self.eval_cache[h] = value
        return value

    def propagate_value(self, state: MetaSeed,
                       child_values: List[MetaSeedValue]) -> MetaSeedValue:
        """Best coverage among children"""
        if not child_values:
            return self.evaluate(state)
        return max(child_values, key=lambda v: v.coverage)

    def get_features(self, state: MetaSeed) -> Any:
        """Features for equivalence classes"""
        return (state.depth, len(self.derivable_from.get(state.root_hash, [])))


# ============================================================
# META-SEED OPTIMIZER
# ============================================================

class MetaSeedOptimizer:
    """
    Find the minimal set of meta-seeds that cover all seeds.

    This is like the classic "set cover" problem, but we use HOLOS
    to explore the space of possible covers.
    """

    def __init__(self, seed_set: Dict[int, Any], underlying_game: GameInterface):
        self.seed_set = seed_set
        self.underlying_game = underlying_game
        self.meta_game = SeedMetaGame(seed_set, underlying_game, max_depth=4)

    def find_root_seeds(self) -> Set[int]:
        """
        Find seeds that cannot be derived from any other seed.
        These MUST be stored as meta-seeds.
        """
        all_seeds = set(self.seed_set.keys())
        derived = set()

        for children in self.meta_game.derivable_from.values():
            for child_hash, _ in children:
                derived.add(child_hash)

        return all_seeds - derived

    def greedy_cover(self, max_meta_seeds: int = 1000) -> List[MetaSeed]:
        """
        Greedy set cover: repeatedly pick the meta-seed with most uncovered seeds.
        """
        print("\nRunning greedy meta-seed selection...")

        uncovered = set(self.seed_set.keys())
        meta_seeds = []

        # Start with root seeds (must be included)
        roots = self.find_root_seeds()
        print(f"Root seeds (not derivable): {len(roots):,}")

        for root_hash in roots:
            if root_hash not in uncovered:
                continue

            root_state = self.seed_set[root_hash]
            meta_seed = MetaSeed(root_hash, root_state, depth=0)

            # Find best depth for this root
            best_depth = 0
            best_coverage = 1

            for d in range(1, 5):
                test_meta = MetaSeed(root_hash, root_state, depth=d)
                value = self.meta_game.evaluate(test_meta)
                if value.coverage > best_coverage:
                    best_coverage = value.coverage
                    best_depth = d

            meta_seed = MetaSeed(root_hash, root_state, depth=best_depth)
            meta_seeds.append(meta_seed)

            # Mark as covered
            covered = self._expand_meta_seed(meta_seed)
            uncovered -= covered

            if len(meta_seeds) >= max_meta_seeds:
                break

        print(f"After roots: {len(meta_seeds)} meta-seeds, {len(uncovered)} uncovered")

        # Greedy add more meta-seeds
        while uncovered and len(meta_seeds) < max_meta_seeds:
            best_meta = None
            best_newly_covered = 0

            # Try each uncovered seed as potential meta-seed
            for seed_hash in list(uncovered)[:100]:  # Sample for speed
                seed_state = self.seed_set[seed_hash]

                for d in range(5):
                    test_meta = MetaSeed(seed_hash, seed_state, depth=d)
                    covered = self._expand_meta_seed(test_meta)
                    newly_covered = len(covered & uncovered)

                    if newly_covered > best_newly_covered:
                        best_newly_covered = newly_covered
                        best_meta = test_meta

            if best_meta is None or best_newly_covered == 0:
                # Add remaining as direct seeds
                for seed_hash in uncovered:
                    seed_state = self.seed_set[seed_hash]
                    meta_seeds.append(MetaSeed(seed_hash, seed_state, depth=0))
                break

            meta_seeds.append(best_meta)
            covered = self._expand_meta_seed(best_meta)
            uncovered -= covered

            if len(meta_seeds) % 100 == 0:
                print(f"  {len(meta_seeds)} meta-seeds, {len(uncovered)} uncovered")

        return meta_seeds

    def _expand_meta_seed(self, meta_seed: MetaSeed) -> Set[int]:
        """Get all seeds covered by a meta-seed at given depth"""
        reachable = {meta_seed.root_hash}
        frontier = {meta_seed.root_hash}

        for _ in range(meta_seed.depth):
            next_frontier = set()
            for seed_hash in frontier:
                for child_hash, _ in self.meta_game.derivable_from.get(seed_hash, []):
                    if child_hash not in reachable:
                        reachable.add(child_hash)
                        next_frontier.add(child_hash)
            frontier = next_frontier

        return reachable

    def holos_search(self, max_iterations: int = 20) -> Dict[str, Any]:
        """
        Use HOLOS to search the meta-seed space!

        This is the key innovation: treating meta-seed selection as
        itself a search problem.
        """
        print("\nRunning HOLOS search over meta-seed space...")

        solver = HOLOSSolver(self.meta_game, name="meta_seed_search",
                            max_memory_mb=500)

        # Generate initial meta-seeds from roots
        roots = self.find_root_seeds()
        initial_seeds = []
        for root_hash in list(roots)[:50]:  # Start with 50 roots
            root_state = self.seed_set[root_hash]
            meta_seed = MetaSeed(root_hash, root_state, depth=1)
            initial_seeds.append(SeedPoint(meta_seed, SearchMode.WAVE))

        print(f"Starting with {len(initial_seeds)} initial meta-seeds")

        # Run HOLOS
        hologram = solver.solve(initial_seeds, max_iterations=max_iterations)

        print(f"\nHOLOS results:")
        print(f"  Solved meta-seeds: {len(hologram.solved)}")
        print(f"  Connections: {len(hologram.connections)}")
        print(f"  Spines: {len(hologram.spines)}")

        # Analyze best meta-seeds found
        best_values = []
        for h, value in hologram.solved.items():
            if isinstance(value, MetaSeedValue):
                best_values.append(value)

        if best_values:
            best_values.sort(key=lambda v: -v.coverage)
            print(f"\nTop 5 meta-seeds by coverage:")
            for v in best_values[:5]:
                print(f"  Coverage: {v.coverage}, Depth: {v.depth}, Ratio: {v.compression_ratio:.1f}x")

        return {
            'hologram': hologram,
            'solver_stats': solver.stats,
            'best_values': best_values
        }


# ============================================================
# MAIN
# ============================================================

def analyze_meta_compression():
    """
    Full analysis of meta-seed compression for Connect4.
    """
    print("=" * 70)
    print("META-SEED COMPRESSION ANALYSIS")
    print("=" * 70)
    print()
    print("The idea: Seeds have structure. Can we compress seeds with meta-seeds?")
    print()

    # Build seed set
    from holos.connect4_full_solve import SeedDatabase, LayerSolver

    db = SeedDatabase(':memory:')
    solver = LayerSolver(db)
    solver.solve_all(start_piece_count=10, end_piece_count=0, verbose=False)

    # Extract seed set
    seed_set = {}
    for layer in db.layers.values():
        for seed in layer.seeds.values():
            h = seed.hash()
            seed_set[h] = seed.to_state()

    print(f"Total seeds: {len(seed_set):,}")
    print()

    # Create optimizer
    game = Connect4Game()
    optimizer = MetaSeedOptimizer(seed_set, game)

    # Find roots
    roots = optimizer.find_root_seeds()
    print(f"Root seeds (cannot be derived): {len(roots):,}")
    print(f"Derivable seeds: {len(seed_set) - len(roots):,}")
    print()

    # Greedy cover
    print("=" * 70)
    print("GREEDY META-SEED SELECTION")
    print("=" * 70)
    meta_seeds = optimizer.greedy_cover(max_meta_seeds=5000)

    print(f"\nFinal meta-seed count: {len(meta_seeds):,}")

    # Calculate compression
    original_storage = len(seed_set) * 45  # 45 bytes per seed

    # Meta-seed storage: root seeds (45 bytes) + derived refs (12 bytes)
    meta_storage = 0
    for ms in meta_seeds:
        covered = optimizer._expand_meta_seed(ms)
        meta_storage += 45  # The meta-seed itself
        meta_storage += (len(covered) - 1) * 12  # Derived references

    print(f"\n--- Storage Comparison ---")
    print(f"Original seed storage: {original_storage:,} bytes ({original_storage/1024:.1f} KB)")
    print(f"Meta-seed storage: {meta_storage:,} bytes ({meta_storage/1024:.1f} KB)")
    print(f"Meta-compression ratio: {original_storage/meta_storage:.2f}x")

    # Combined compression
    total_positions = sum(l.positions_solved for l in db.layers.values())
    original_full = total_positions * 1  # 1 byte per position

    print(f"\n--- Combined Compression ---")
    print(f"Full position storage: {original_full:,} bytes")
    print(f"Seed storage: {original_storage:,} bytes ({original_full/original_storage:.1f}x)")
    print(f"Meta-seed storage: {meta_storage:,} bytes ({original_full/meta_storage:.1f}x)")

    # HOLOS search
    print()
    print("=" * 70)
    print("HOLOS SEARCH OVER META-SEED SPACE")
    print("=" * 70)
    results = optimizer.holos_search(max_iterations=10)

    return {
        'seed_set': seed_set,
        'meta_seeds': meta_seeds,
        'roots': roots,
        'original_storage': original_storage,
        'meta_storage': meta_storage,
        'holos_results': results
    }


if __name__ == "__main__":
    analyze_meta_compression()
