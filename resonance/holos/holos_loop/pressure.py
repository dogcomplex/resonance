"""
holos/pressure.py - Pressure-based continuous wave expansion

Instead of stopping at first closure, this system:
1. Measures "pressure" at each frontier point
2. Continues expanding where pressure remains high
3. Closures reduce local pressure but don't halt search
4. Tracks interior formation and "decryption"

The goal: Map the FULL solution space, not just find first path.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import time

from holos.holos import GameInterface
from holos.closure import (
    ClosureDetector, ClosureEvent, ClosureType,
    PhaseAlignment, ModeEmergence
)
from holos.storage import SpinePath


# ============================================================
# PRESSURE COMPUTATION
# ============================================================

@dataclass
class PressureState:
    """Pressure metrics for a frontier position."""
    hash: int
    unexplored_successors: int    # How many children not yet visited
    unexplored_predecessors: int  # How many parents not yet visited
    distance_to_closure: float    # Min distance to any closure point
    local_branching: float        # Branching factor in neighborhood
    closure_density: float        # Fraction of neighbors that are closures

    @property
    def forward_pressure(self) -> float:
        """Pressure driving forward expansion."""
        return self.unexplored_successors * (1 - self.closure_density)

    @property
    def backward_pressure(self) -> float:
        """Pressure driving backward expansion."""
        return self.unexplored_predecessors * (1 - self.closure_density)

    @property
    def total_pressure(self) -> float:
        """Combined pressure."""
        return self.forward_pressure + self.backward_pressure


class Permeability(Enum):
    """How easily waves flow through a region."""
    HIGH = "high"        # Many paths, few closures
    MEDIUM = "medium"    # Moderate constraint
    LOW = "low"          # Highly constrained
    SEALED = "sealed"    # Fully closed interior


@dataclass
class RegionPermeability:
    """Permeability metrics for a region."""
    center_hash: int
    radius: int
    avg_branching: float
    closure_density: float
    boundary_fraction: float  # What fraction of region is boundary

    @property
    def permeability(self) -> Permeability:
        if self.closure_density > 0.8:
            return Permeability.SEALED
        elif self.closure_density > 0.5:
            return Permeability.LOW
        elif self.avg_branching < 3:
            return Permeability.MEDIUM
        return Permeability.HIGH


# ============================================================
# INTERIOR DETECTION
# ============================================================

@dataclass
class Interior:
    """A closed region where all values are determined."""
    center_hash: int
    boundary: Set[int]      # Closure points forming the boundary
    interior: Set[int]      # Points inside the boundary
    values: Dict[int, Any]  # Propagated values
    creation_time: float
    decrypted: bool = False

    @property
    def size(self) -> int:
        return len(self.interior)

    @property
    def boundary_size(self) -> int:
        return len(self.boundary)

    def contains(self, h: int) -> bool:
        return h in self.interior or h in self.boundary


class InteriorDetector:
    """Detects and tracks closed interiors."""

    def __init__(self, game: GameInterface):
        self.game = game
        self.interiors: List[Interior] = []
        self.point_to_interior: Dict[int, Interior] = {}

    def detect_interior(
        self,
        center_hash: int,
        states: Dict[int, Any],
        closures: Set[int],
        values: Dict[int, Any],
        radius: int = 5
    ) -> Optional[Interior]:
        """
        Check if a region around center forms a closed interior.

        A region is interior if:
        1. BFS from center hits only closures or boundaries within radius
        2. All boundary points have known values
        """
        boundary = set()
        interior = set()
        queue = [(center_hash, 0)]
        visited = set()

        while queue:
            h, depth = queue.pop(0)
            if h in visited:
                continue
            visited.add(h)

            # Is this a boundary point?
            if h in closures or h in values:
                boundary.add(h)
                continue

            # Too far from center?
            if depth >= radius:
                boundary.add(h)
                continue

            # Not closed yet
            interior.add(h)

            # Explore neighbors
            if h in states:
                state = states[h]
                for child, _ in self.game.get_successors(state):
                    ch = self.game.hash_state(child)
                    if ch not in visited:
                        queue.append((ch, depth + 1))

        # Check if boundary is fully closed (all have values)
        boundary_solved = all(h in values for h in boundary)

        if boundary_solved and interior:
            return Interior(
                center_hash=center_hash,
                boundary=boundary,
                interior=interior,
                values={h: values[h] for h in boundary if h in values},
                creation_time=time.time(),
                decrypted=False
            )
        return None

    def decrypt_interior(
        self,
        interior: Interior,
        states: Dict[int, Any],
        values: Dict[int, Any]
    ) -> Dict[int, Any]:
        """
        Propagate values from boundary into interior.
        Returns newly computed values.
        """
        new_values = {}
        queue = list(interior.boundary)
        in_queue = set(queue)

        while queue:
            h = queue.pop(0)
            if h in values:
                continue

            state = states.get(h)
            if state is None:
                continue

            # Get children's values
            child_values = []
            all_children_known = True
            for child, _ in self.game.get_successors(state):
                ch = self.game.hash_state(child)
                if ch in values:
                    child_values.append(values[ch])
                elif ch in new_values:
                    child_values.append(new_values[ch])
                else:
                    all_children_known = False

            # Can we propagate?
            if child_values and all_children_known:
                propagated = self.game.propagate_value(state, child_values)
                if propagated is not None:
                    new_values[h] = propagated
                    values[h] = propagated

                    # Parents may now be solvable
                    for parent, _ in self.game.get_predecessors(state):
                        ph = self.game.hash_state(parent)
                        if ph in interior.interior and ph not in in_queue:
                            queue.append(ph)
                            in_queue.add(ph)

        interior.decrypted = True
        interior.values.update(new_values)
        return new_values


# ============================================================
# PRESSURE WAVE SYSTEM
# ============================================================

@dataclass
class PressureConfig:
    """Configuration for pressure-based search."""
    initial_pressure: float = 100.0
    closure_damping: float = 0.3      # How much closure reduces pressure
    distance_decay: float = 0.9       # Pressure decay with distance
    saturation_threshold: float = 0.1 # Minimum pressure to continue
    interior_radius: int = 5          # Radius for interior detection


class PressureWaveSystem:
    """
    Continuous wave expansion driven by pressure differentials.

    Unlike basic closure system that stops at first spine,
    this continues until pressure equilibrates (full mapping).
    """

    def __init__(
        self,
        game: GameInterface,
        config: PressureConfig = None
    ):
        self.game = game
        self.config = config or PressureConfig()

        # State storage
        self.states: Dict[int, Any] = {}
        self.values: Dict[int, Any] = {}

        # Frontiers (active expansion edge)
        self.forward_frontier: Dict[int, Any] = {}
        self.backward_frontier: Dict[int, Any] = {}

        # Wave tracking (all states touched by each wave)
        self.forward_reached: Set[int] = set()
        self.backward_reached: Set[int] = set()

        # Pressure tracking
        self.pressure: Dict[int, float] = {}

        # Closure tracking
        self.closure_detector = ClosureDetector()
        self.closures: Set[int] = set()
        self.spines: List[SpinePath] = []

        # Interior tracking
        self.interior_detector = InteriorDetector(game)
        self.interiors: List[Interior] = []

        # Statistics
        self.stats = {
            'iterations': 0,
            'total_expanded': 0,
            'closures_found': 0,
            'interiors_found': 0,
            'interiors_decrypted': 0,
            'values_propagated': 0,
        }

    def setup(self, forward_seeds: List[Any], backward_seeds: List[Any] = None):
        """Initialize with seed positions."""
        # Forward seeds
        for state in forward_seeds:
            h = self.game.hash_state(state)
            self.forward_frontier[h] = state
            self.forward_reached.add(h)
            self.states[h] = state
            self.pressure[h] = self.config.initial_pressure

        # Backward seeds (or generate from boundary)
        if backward_seeds:
            for state in backward_seeds:
                h = self.game.hash_state(state)
                self.backward_frontier[h] = state
                self.backward_reached.add(h)
                self.states[h] = state
                if self.game.is_boundary(state):
                    self.values[h] = self.game.get_boundary_value(state)
        else:
            self._generate_backward_seeds(forward_seeds)

    def _generate_backward_seeds(self, templates: List[Any], count: int = 100):
        """Generate backward seeds from boundary."""
        seeds = self.game.generate_boundary_seeds(templates[0], count)
        for state in seeds[:count]:
            h = self.game.hash_state(state)
            self.backward_frontier[h] = state
            self.backward_reached.add(h)
            self.states[h] = state
            if self.game.is_boundary(state):
                self.values[h] = self.game.get_boundary_value(state)

    def compute_pressure(self, h: int) -> PressureState:
        """Compute pressure metrics for a position."""
        state = self.states.get(h)
        if state is None:
            return PressureState(h, 0, 0, float('inf'), 0, 0)

        # Count unexplored successors
        unexplored_succ = 0
        total_succ = 0
        for child, _ in self.game.get_successors(state):
            ch = self.game.hash_state(child)
            total_succ += 1
            if ch not in self.states:
                unexplored_succ += 1

        # Count unexplored predecessors
        unexplored_pred = 0
        total_pred = 0
        closure_neighbors = 0
        for parent, _ in self.game.get_predecessors(state):
            ph = self.game.hash_state(parent)
            total_pred += 1
            if ph not in self.states:
                unexplored_pred += 1
            if ph in self.closures:
                closure_neighbors += 1

        # Closure density
        total_neighbors = total_succ + total_pred
        closure_density = closure_neighbors / max(1, total_neighbors)

        # Distance to nearest closure (simplified: 0 if in closures, inf otherwise)
        distance = 0 if h in self.closures else float('inf')

        return PressureState(
            hash=h,
            unexplored_successors=unexplored_succ,
            unexplored_predecessors=unexplored_pred,
            distance_to_closure=distance,
            local_branching=total_succ,
            closure_density=closure_density
        )

    def step(self) -> Dict[str, Any]:
        """
        One step of pressure-driven expansion.

        Returns metrics about what happened.
        """
        self.stats['iterations'] += 1

        # 1. Update pressure for all frontier points
        for h in list(self.forward_frontier.keys()) + list(self.backward_frontier.keys()):
            ps = self.compute_pressure(h)
            self.pressure[h] = ps.total_pressure * (1 - ps.closure_density)

        # 2. Sort by pressure (highest first)
        forward_by_pressure = sorted(
            self.forward_frontier.keys(),
            key=lambda h: self.pressure.get(h, 0),
            reverse=True
        )
        backward_by_pressure = sorted(
            self.backward_frontier.keys(),
            key=lambda h: self.pressure.get(h, 0),
            reverse=True
        )

        # 3. Expand highest pressure points
        expanded = 0
        new_closures = []

        # Forward expansion
        for h in forward_by_pressure[:10]:  # Expand top 10
            if self.pressure.get(h, 0) < self.config.saturation_threshold:
                break

            state = self.forward_frontier[h]
            for child, move in self.game.get_successors(state):
                ch = self.game.hash_state(child)
                if ch not in self.forward_reached:
                    self.states[ch] = child
                    self.forward_frontier[ch] = child
                    self.forward_reached.add(ch)
                    self.pressure[ch] = self.pressure[h] * self.config.distance_decay
                    expanded += 1

                    # Check for closure - both waves have reached this state
                    if ch in self.backward_reached:
                        event = self._check_closure(ch)
                        if event:
                            new_closures.append(event)

                    # Check if boundary
                    if self.game.is_boundary(child):
                        self.values[ch] = self.game.get_boundary_value(child)

            # Remove from frontier after expansion
            del self.forward_frontier[h]

        # Backward expansion
        for h in backward_by_pressure[:10]:
            if self.pressure.get(h, 0) < self.config.saturation_threshold:
                break

            state = self.backward_frontier[h]
            for parent, move in self.game.get_predecessors(state):
                ph = self.game.hash_state(parent)
                if ph not in self.backward_reached:
                    self.states[ph] = parent
                    self.backward_frontier[ph] = parent
                    self.backward_reached.add(ph)
                    self.pressure[ph] = self.pressure[h] * self.config.distance_decay
                    expanded += 1

                    # Check for closure - both waves have reached this state
                    if ph in self.forward_reached:
                        event = self._check_closure(ph)
                        if event:
                            new_closures.append(event)

            del self.backward_frontier[h]

        # 4. Process closures - reduce local pressure
        for event in new_closures:
            h = event.state_hash
            self.closures.add(h)
            self.pressure[h] *= self.config.closure_damping
            self.stats['closures_found'] += 1

            # Try to detect interior around closure
            interior = self.interior_detector.detect_interior(
                h, self.states, self.closures, self.values,
                radius=self.config.interior_radius
            )
            if interior:
                self.interiors.append(interior)
                self.stats['interiors_found'] += 1

                # Decrypt interior
                new_vals = self.interior_detector.decrypt_interior(
                    interior, self.states, self.values
                )
                self.stats['values_propagated'] += len(new_vals)
                self.stats['interiors_decrypted'] += 1

        self.stats['total_expanded'] += expanded

        # 5. Compute total remaining pressure
        total_pressure = sum(self.pressure.values())

        return {
            'expanded': expanded,
            'new_closures': len(new_closures),
            'total_pressure': total_pressure,
            'frontier_size': len(self.forward_frontier) + len(self.backward_frontier),
            'values_known': len(self.values),
            'interiors': len(self.interiors),
        }

    def _check_closure(self, h: int) -> Optional[ClosureEvent]:
        """Check if position is a closure point."""
        # Both waves must have reached this state
        fwd_reached = h in self.forward_reached
        bwd_reached = h in self.backward_reached

        if not (fwd_reached and bwd_reached):
            return None

        # Determine closure type based on whether we have a value
        if h in self.values:
            # Both waves reached a valued position - full closure
            return ClosureEvent(
                state_hash=h,
                layer=0,
                closure_type=ClosureType.IRREDUCIBLE,
                phase_diff=0.0,
                forward_value=self.values.get(h),
                backward_value=self.values.get(h),
                iteration=self.stats['iterations'],
                num_contributing_paths=1
            )
        else:
            # Waves met but no value yet - partial closure
            return ClosureEvent(
                state_hash=h,
                layer=0,
                closure_type=ClosureType.REDUCIBLE,
                phase_diff=0.1,  # Placeholder
                forward_value=None,
                backward_value=None,
                iteration=self.stats['iterations'],
                num_contributing_paths=1
            )

    def run(
        self,
        max_iterations: int = 100,
        target_coverage: float = 0.99,
        min_pressure: float = None
    ) -> Dict[str, Any]:
        """
        Run until pressure equilibrium or target coverage.
        """
        min_pressure = min_pressure or self.config.saturation_threshold
        t0 = time.time()

        for i in range(max_iterations):
            result = self.step()

            if i % 10 == 0:
                coverage = len(self.values) / max(1, len(self.states))
                print(f"  Iter {i}: expanded={result['expanded']}, "
                      f"closures={len(self.closures)}, "
                      f"values={len(self.values)}, "
                      f"coverage={coverage:.1%}, "
                      f"pressure={result['total_pressure']:.1f}")

            # Stopping conditions
            if result['total_pressure'] < min_pressure:
                print(f"  Pressure equilibrium reached")
                break

            coverage = len(self.values) / max(1, len(self.states))
            if coverage >= target_coverage:
                print(f"  Target coverage reached: {coverage:.1%}")
                break

        elapsed = time.time() - t0
        coverage = len(self.values) / max(1, len(self.states))

        return {
            'iterations': self.stats['iterations'],
            'elapsed': elapsed,
            'states_explored': len(self.states),
            'values_known': len(self.values),
            'coverage': coverage,
            'closures': len(self.closures),
            'interiors': len(self.interiors),
            'stats': self.stats,
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def run_pressure_search(
    game: GameInterface,
    start_states: List[Any],
    max_iterations: int = 100,
    target_coverage: float = 0.99,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run pressure-based search to map solution space."""
    if verbose:
        print("\n" + "=" * 60)
        print("Pressure Wave Search")
        print("=" * 60)

    system = PressureWaveSystem(game)
    system.setup(start_states)

    if verbose:
        print(f"  Forward seeds: {len(start_states)}")
        print(f"  Backward seeds: {len(system.backward_frontier)}")

    result = system.run(
        max_iterations=max_iterations,
        target_coverage=target_coverage
    )

    if verbose:
        print(f"\nResults:")
        print(f"  States explored: {result['states_explored']}")
        print(f"  Values known: {result['values_known']}")
        print(f"  Coverage: {result['coverage']:.1%}")
        print(f"  Closures: {result['closures']}")
        print(f"  Interiors: {result['interiors']}")

    return result
