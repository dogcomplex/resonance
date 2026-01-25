"""
holos/layer2_covers.py - Cover Scale (Layer 2)

Layer 2 operates at the COVER scale:
- State: A cover (SEQUENCE of Layer 1 paths, ordered by priority)
- Value: Cover quality (total coverage, efficiency, completeness)
- Moves: Add/remove/reorder paths in the cover
- Boundary: Complete covers (those achieving target coverage)

Key insight: A cover is a SEQUENCE of paths, not a set.
The order determines exploration priority - which paths to pursue first.

The two waves at Layer 2:
- Forward wave: Partial covers being built up (adding paths)
- Backward wave: Complete covers being refined (removing redundancy)
- Closure: When a cover achieves target coverage efficiently

This differs from the original strategy.py which searched over budget ALLOCATIONS.
Here we search over path SEQUENCES, treating order as fundamental.

The HOLOS pattern at Layer 2:
- State space: All possible orderings of available paths
- Bidirectional search finds optimal path orderings
- Closures indicate efficient complete covers
"""

from typing import List, Tuple, Optional, Any, Dict, Set, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict

from holos.holos import GameInterface
from holos.storage import SpinePath
from holos.closure import (
    ClosureDetector, ClosureEvent, ClosureType,
    PhaseAlignment, WaveOrigin
)
from holos.layer1_paths import PartialPath, PathValue, PathGame


# ============================================================
# LAYER 2 STATE: COVER (Sequence of Paths)
# ============================================================

@dataclass
class PathCover:
    """
    A cover: an ordered sequence of paths.

    This is the STATE at Layer 2 - we're searching over path orderings.

    The ORDER matters:
    - Earlier paths have higher priority
    - Exploration proceeds in sequence order
    - Redundant paths can be pruned from the end

    A cover is "complete" when it achieves target coverage.
    """
    # Ordered sequence of paths (by priority)
    paths: Tuple[int, ...]  # Tuple of path hashes (immutable for hashing)

    # Coverage tracking
    total_coverage: int = 0
    unique_coverage: int = 0  # States not covered by earlier paths

    # Metadata
    origin: WaveOrigin = WaveOrigin.FORWARD

    # Cache of path objects (not hashed)
    _path_objects: Dict[int, Any] = field(default_factory=dict, hash=False, compare=False)

    def __hash__(self):
        return hash(self.paths)

    def __len__(self):
        return len(self.paths)

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    def add_path(self, path_hash: int, path_obj: Any = None) -> 'PathCover':
        """Create new cover with path added at end (lowest priority)"""
        new_paths = self.paths + (path_hash,)
        new_objs = dict(self._path_objects)
        if path_obj is not None:
            new_objs[path_hash] = path_obj
        return PathCover(
            paths=new_paths,
            origin=self.origin,
            _path_objects=new_objs
        )

    def prepend_path(self, path_hash: int, path_obj: Any = None) -> 'PathCover':
        """Create new cover with path added at start (highest priority)"""
        new_paths = (path_hash,) + self.paths
        new_objs = dict(self._path_objects)
        if path_obj is not None:
            new_objs[path_hash] = path_obj
        return PathCover(
            paths=new_paths,
            origin=self.origin,
            _path_objects=new_objs
        )

    def remove_path(self, index: int) -> 'PathCover':
        """Create new cover with path at index removed"""
        if index < 0 or index >= len(self.paths):
            return self
        new_paths = self.paths[:index] + self.paths[index+1:]
        return PathCover(
            paths=new_paths,
            origin=self.origin,
            _path_objects=self._path_objects
        )

    def reorder(self, new_order: Tuple[int, ...]) -> 'PathCover':
        """Create new cover with paths reordered"""
        # new_order is indices into current paths
        new_paths = tuple(self.paths[i] for i in new_order if i < len(self.paths))
        return PathCover(
            paths=new_paths,
            origin=self.origin,
            _path_objects=self._path_objects
        )

    def swap(self, i: int, j: int) -> 'PathCover':
        """Create new cover with paths i and j swapped"""
        if i < 0 or j < 0 or i >= len(self.paths) or j >= len(self.paths):
            return self
        path_list = list(self.paths)
        path_list[i], path_list[j] = path_list[j], path_list[i]
        return PathCover(
            paths=tuple(path_list),
            origin=self.origin,
            _path_objects=self._path_objects
        )

    def signature(self) -> str:
        """Human-readable signature"""
        return f"Cover({self.origin.value}, {self.num_paths} paths, cov={self.total_coverage})"


# ============================================================
# LAYER 2 VALUE: COVER QUALITY
# ============================================================

@dataclass(frozen=True)
class CoverValue:
    """
    Value of a cover at Layer 2.

    Measures cover quality:
    - coverage: Total states covered by all paths
    - unique_coverage: States covered (accounting for overlap)
    - efficiency: unique_coverage / num_paths
    - completeness: fraction of target coverage achieved
    - priority: Combined priority for extending this cover
    """
    coverage: int           # Total coverage (with overlap)
    unique_coverage: int    # Coverage without double-counting
    efficiency: float       # unique_coverage / num_paths
    completeness: float     # unique_coverage / target
    priority: float         # Combined priority score

    # For complete covers
    is_complete: bool = False

    def __repr__(self):
        status = "COMPLETE" if self.is_complete else "partial"
        return f"CoverValue({status}, cov={self.unique_coverage}, eff={self.efficiency:.1f}, complete={self.completeness:.1%})"

    def __lt__(self, other):
        # Primary: completeness, Secondary: efficiency
        if self.completeness != other.completeness:
            return self.completeness < other.completeness
        return self.efficiency < other.efficiency


# ============================================================
# LAYER 2 GAME INTERFACE
# ============================================================

class CoverGame(GameInterface[PathCover, CoverValue]):
    """
    Layer 2 game: Search over covers (path orderings).

    State space: All orderings of available paths
    Successors: Add path to cover, or reorder paths
    Predecessors: Remove path from cover
    Boundary: Complete covers (achieve target coverage)
    Value: Cover quality (coverage, efficiency, completeness)
    """

    def __init__(self,
                 available_paths: List[Tuple[int, Any]],  # (hash, path_obj)
                 target_coverage: int = 1000,
                 closure_detector: ClosureDetector = None):
        """
        Args:
            available_paths: Pool of (hash, path_object) pairs
            target_coverage: Target coverage to achieve
            closure_detector: Shared closure detector
        """
        self.available_paths = {h: obj for h, obj in available_paths}
        self.path_hashes = list(self.available_paths.keys())
        self.target_coverage = target_coverage
        self.closure_detector = closure_detector or ClosureDetector()

        # Coverage cache: path_hash -> set of states covered
        self.path_coverage: Dict[int, Set[int]] = {}

        # Statistics
        self.stats = {
            'covers_created': 0,
            'covers_extended': 0,
            'complete_covers': 0,
        }

    def set_path_coverage(self, path_hash: int, covered_states: Set[int]):
        """Set the coverage for a path (called from Layer 1)"""
        self.path_coverage[path_hash] = covered_states

    def hash_state(self, cover: PathCover) -> int:
        return hash(cover)

    def get_successors(self, cover: PathCover) -> List[Tuple[PathCover, Any]]:
        """
        Extend cover by adding paths or reordering.

        Successors:
        - Add any available path not in cover
        - Swap adjacent paths (local reordering)
        """
        successors = []
        current_paths = set(cover.paths)

        # Add paths not yet in cover
        for path_hash in self.path_hashes:
            if path_hash not in current_paths:
                path_obj = self.available_paths.get(path_hash)
                new_cover = cover.add_path(path_hash, path_obj)
                successors.append((new_cover, ('add', path_hash)))
                self.stats['covers_extended'] += 1

        # Swap adjacent paths (reordering moves)
        for i in range(len(cover.paths) - 1):
            swapped = cover.swap(i, i + 1)
            successors.append((swapped, ('swap', i, i + 1)))

        return successors

    def get_predecessors(self, cover: PathCover) -> List[Tuple[PathCover, Any]]:
        """
        Simplify cover by removing paths.

        Predecessors:
        - Remove last path (lowest priority)
        - Remove any redundant path
        """
        predecessors = []

        # Remove each path
        for i in range(len(cover.paths)):
            reduced = cover.remove_path(i)
            if len(reduced.paths) > 0:  # Don't create empty covers
                predecessors.append((reduced, ('remove', i)))

        return predecessors

    def is_boundary(self, cover: PathCover) -> bool:
        """Cover is at boundary if it achieves target coverage"""
        unique_cov = self._compute_unique_coverage(cover)
        return unique_cov >= self.target_coverage

    def get_boundary_value(self, cover: PathCover) -> Optional[CoverValue]:
        """Get value for complete cover"""
        if not self.is_boundary(cover):
            return None

        unique_cov = self._compute_unique_coverage(cover)
        total_cov = sum(len(self.path_coverage.get(h, set())) for h in cover.paths)

        return CoverValue(
            coverage=total_cov,
            unique_coverage=unique_cov,
            efficiency=unique_cov / max(1, cover.num_paths),
            completeness=1.0,
            priority=float('inf'),
            is_complete=True
        )

    def is_terminal(self, cover: PathCover) -> Tuple[bool, Optional[CoverValue]]:
        """Cover is terminal if complete with no redundancy"""
        if not self.is_boundary(cover):
            return False, None

        # Check for redundancy: can we remove any path and still be complete?
        for i in range(len(cover.paths)):
            reduced = cover.remove_path(i)
            if len(reduced.paths) > 0 and self.is_boundary(reduced):
                # Redundant path exists, not terminal
                return False, None

        # No redundancy - this is an irreducible complete cover
        return True, self.get_boundary_value(cover)

    def propagate_value(self, cover: PathCover, child_values: List[CoverValue]) -> Optional[CoverValue]:
        """Propagate value from child covers to parent"""
        if not child_values:
            return None

        # Best child by completeness then efficiency
        best = max(child_values, key=lambda v: (v.completeness, v.efficiency))

        # Parent is one path smaller
        return CoverValue(
            coverage=max(0, best.coverage - 100),  # Estimate
            unique_coverage=max(0, best.unique_coverage - 100),
            efficiency=best.efficiency * 0.95,
            completeness=best.completeness * 0.9,
            priority=best.priority * 0.9
        )

    def get_features(self, cover: PathCover) -> Any:
        """Feature extraction for equivalence"""
        return (cover.num_paths, self._compute_unique_coverage(cover) // 100)

    def _compute_unique_coverage(self, cover: PathCover) -> int:
        """Compute unique states covered (no double counting)"""
        covered = set()
        for path_hash in cover.paths:
            if path_hash in self.path_coverage:
                covered.update(self.path_coverage[path_hash])
        return len(covered)

    # ==================== Cover-Specific Methods ====================

    def create_empty_cover(self) -> PathCover:
        """Create an empty cover as starting point"""
        self.stats['covers_created'] += 1
        return PathCover(paths=(), origin=WaveOrigin.FORWARD)

    def create_full_cover(self) -> PathCover:
        """Create a cover containing all available paths"""
        self.stats['covers_created'] += 1
        return PathCover(
            paths=tuple(self.path_hashes),
            origin=WaveOrigin.BACKWARD,
            _path_objects=self.available_paths
        )

    def evaluate_cover(self, cover: PathCover) -> CoverValue:
        """Evaluate a cover's quality"""
        unique_cov = self._compute_unique_coverage(cover)
        total_cov = sum(len(self.path_coverage.get(h, set())) for h in cover.paths)

        efficiency = unique_cov / max(1, cover.num_paths)
        completeness = unique_cov / self.target_coverage if self.target_coverage > 0 else 0

        # Priority: completeness weighted by efficiency
        priority = completeness * efficiency * 100

        return CoverValue(
            coverage=total_cov,
            unique_coverage=unique_cov,
            efficiency=efficiency,
            completeness=min(1.0, completeness),
            priority=priority,
            is_complete=completeness >= 1.0
        )

    def find_minimal_cover(self, cover: PathCover) -> PathCover:
        """Greedily remove redundant paths to find minimal cover"""
        if not self.is_boundary(cover):
            return cover

        current = cover
        for i in range(len(cover.paths) - 1, -1, -1):
            reduced = current.remove_path(i)
            if len(reduced.paths) > 0 and self.is_boundary(reduced):
                current = reduced

        return current


# ============================================================
# LAYER 2 SOLVER
# ============================================================

class CoverLayerSolver:
    """
    Solver for Layer 2 (cover scale).

    Uses bidirectional search over cover space:
    - Forward: Build up covers by adding paths
    - Backward: Refine covers by removing redundancy
    - Closure: Efficient complete covers

    This implements the same HOLOS pattern as Layers 0 and 1.
    """

    def __init__(self,
                 available_paths: List[Tuple[int, Any]],
                 target_coverage: int = 1000,
                 closure_detector: ClosureDetector = None):
        """
        Args:
            available_paths: Pool of (hash, path_object) pairs
            target_coverage: Target coverage to achieve
            closure_detector: Shared closure detector
        """
        self.closure_detector = closure_detector or ClosureDetector()
        self.cover_game = CoverGame(available_paths, target_coverage, self.closure_detector)

        # Cover fronts
        self.forward_front: Dict[int, PathCover] = {}
        self.backward_front: Dict[int, PathCover] = {}

        # Priority queues
        self.forward_queue: List[Tuple[float, PathCover]] = []
        self.backward_queue: List[Tuple[float, PathCover]] = []

        # Results
        self.best_covers: List[PathCover] = []

        # Statistics
        self.stats = {
            'iterations': 0,
            'forward_extensions': 0,
            'backward_extensions': 0,
            'connections': 0,
            'complete_covers': 0,
        }

    def set_path_coverage(self, path_hash: int, covered_states: Set[int]):
        """Set coverage for a path (pass-through to game)"""
        self.cover_game.set_path_coverage(path_hash, covered_states)

    def initialize(self):
        """Initialize forward and backward fronts"""
        # Forward: start from empty cover
        empty = self.cover_game.create_empty_cover()
        value = self.cover_game.evaluate_cover(empty)
        h = hash(empty)
        self.forward_front[h] = empty
        self.forward_queue.append((-value.priority, empty))

        # Backward: start from full cover (all paths)
        full = self.cover_game.create_full_cover()
        value = self.cover_game.evaluate_cover(full)
        h = hash(full)
        self.backward_front[h] = full
        self.backward_queue.append((-value.priority, full))

    def step(self, mode: str = "balanced") -> Dict:
        """Perform one step of cover search"""
        self.stats['iterations'] += 1
        result = {'extended': 0, 'complete': 0}

        if mode == "forward":
            self._extend_forward()
            result['extended'] = 1
        elif mode == "backward":
            self._extend_backward()
            result['extended'] = 1
        else:  # balanced
            self._extend_forward()
            self._extend_backward()
            result['extended'] = 2

        # Check for complete covers
        result['complete'] = self._check_completions()

        return result

    def _extend_forward(self):
        """Extend best forward cover (add paths)"""
        if not self.forward_queue:
            return

        _, cover = self.forward_queue.pop(0)

        for new_cover, move in self.cover_game.get_successors(cover):
            value = self.cover_game.evaluate_cover(new_cover)
            h = hash(new_cover)
            if h not in self.forward_front:
                self.forward_front[h] = new_cover
                self.forward_queue.append((-value.priority, new_cover))
                self.stats['forward_extensions'] += 1

                # Check if complete
                if value.is_complete:
                    self.best_covers.append(new_cover)
                    self.stats['complete_covers'] += 1

        self.forward_queue.sort(key=lambda x: x[0])

    def _extend_backward(self):
        """Extend best backward cover (remove redundancy)"""
        if not self.backward_queue:
            return

        _, cover = self.backward_queue.pop(0)

        for new_cover, move in self.cover_game.get_predecessors(cover):
            value = self.cover_game.evaluate_cover(new_cover)
            h = hash(new_cover)
            if h not in self.backward_front:
                self.backward_front[h] = new_cover
                self.backward_queue.append((-value.priority, new_cover))
                self.stats['backward_extensions'] += 1

                # Track if still complete (minimal cover search)
                if value.is_complete:
                    self.best_covers.append(new_cover)

        self.backward_queue.sort(key=lambda x: x[0])

    def _check_completions(self) -> int:
        """Check for connections between forward and backward fronts"""
        completions = 0

        # Look for covers that appear in both fronts
        for h, fwd_cover in self.forward_front.items():
            if h in self.backward_front:
                # Connection found
                self.stats['connections'] += 1
                completions += 1

                # Record closure
                value = self.cover_game.evaluate_cover(fwd_cover)
                self.closure_detector.check_closure(
                    state_hash=h,
                    forward_value=value.unique_coverage,
                    backward_value=value.efficiency,
                    layer=2
                )

        return completions

    def solve(self,
              max_iterations: int = 100,
              mode: str = "balanced",
              verbose: bool = True) -> List[PathCover]:
        """
        Solve: find optimal path covers.

        Args:
            max_iterations: Maximum iterations
            mode: Search mode ("forward", "backward", "balanced")
            verbose: Print progress

        Returns:
            List of complete covers found
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Layer 2 Cover Solver")
            print(f"  Available paths: {len(self.cover_game.available_paths)}")
            print(f"  Target coverage: {self.cover_game.target_coverage}")
            print(f"{'='*60}")

        self.initialize()

        for i in range(max_iterations):
            result = self.step(mode=mode)

            if verbose and (i % 10 == 0 or result['complete'] > 0):
                print(f"  Iter {i}: fwd={len(self.forward_front)}, bwd={len(self.backward_front)}, "
                      f"complete={len(self.best_covers)}")

            if not self.forward_queue and not self.backward_queue:
                if verbose:
                    print("  Fronts exhausted")
                break

        # Find minimal covers
        minimal_covers = []
        for cover in self.best_covers:
            minimal = self.cover_game.find_minimal_cover(cover)
            if minimal not in minimal_covers:
                minimal_covers.append(minimal)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Complete: {len(self.best_covers)} covers found")
            print(f"  Minimal covers: {len(minimal_covers)}")
            print(f"  Closures: {self.closure_detector.stats['total_closures']}")
            print(f"{'='*60}")

        return minimal_covers


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_cover_solver(paths: List[Tuple[int, Any]],
                        target: int = 1000) -> CoverLayerSolver:
    """Create a Layer 2 cover solver"""
    return CoverLayerSolver(paths, target)


def find_optimal_cover(paths: List[Tuple[int, Any]],
                       path_coverages: Dict[int, Set[int]],
                       target: int = 1000,
                       max_iterations: int = 100) -> Optional[PathCover]:
    """
    Find an optimal cover for the given paths.

    Args:
        paths: List of (hash, path_object) pairs
        path_coverages: Dict mapping path_hash -> set of covered states
        target: Target coverage
        max_iterations: Maximum iterations

    Returns:
        Optimal PathCover or None
    """
    solver = create_cover_solver(paths, target)

    # Set coverages
    for path_hash, covered in path_coverages.items():
        solver.set_path_coverage(path_hash, covered)

    covers = solver.solve(max_iterations, verbose=False)
    return covers[0] if covers else None
