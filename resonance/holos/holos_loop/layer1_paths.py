"""
holos/layer1_paths.py - Path Scale (Layer 1)

Layer 1 operates at the PATH scale:
- State: A path (sequence of Layer 0 states)
- Value: Path quality (coverage, efficiency, closure potential)
- Moves: Extend path forward or backward
- Boundary: Complete paths (those that reach Layer 0 boundary)

Key insight: Paths are the fundamental objects of search.
A "spine" is just a complete path. Layer 1 searches for good paths.

The two waves at Layer 1:
- Forward wave: Partial paths extending from start toward boundary
- Backward wave: Partial paths extending from boundary toward start
- Closure: When forward and backward partial paths can connect

This differs from the original seeds.py which searched over seed PARAMETERS.
Here we search over the paths THEMSELVES, treating them as first-class objects.

Priorities at Layer 1:
- Which paths to extend first (priority queue over partial paths)
- Which direction to extend (forward vs backward)
- How much to invest in each path
"""

from typing import List, Tuple, Optional, Any, Dict, Set, Generic, TypeVar
from dataclasses import dataclass, field
from collections import defaultdict
import time

from holos.holos import GameInterface, SearchMode, HOLOSSolver, SeedPoint
from holos.storage import SpinePath
from holos.closure import (
    ClosureDetector, ClosureEvent, ClosureType,
    PhaseAlignment, ModeEmergence, WaveOrigin
)

# Type variables from underlying Layer 0 game
S = TypeVar('S')  # Layer 0 state type
V = TypeVar('V')  # Layer 0 value type


# ============================================================
# LAYER 1 STATE: PATH
# ============================================================

@dataclass
class PartialPath:
    """
    A partial path through Layer 0 state space.

    This is the STATE at Layer 1 - we're searching over paths.

    A path is a sequence of (state_hash, move) pairs representing
    a trajectory through the game tree.
    """
    # Path content
    steps: Tuple[Tuple[int, Any], ...]  # ((hash1, move1), (hash2, move2), ...)

    # Endpoints
    start_hash: int
    end_hash: int

    # Direction this path was grown from
    origin: WaveOrigin

    # Cached actual states (not hashed, for reconstruction)
    _start_state: Any = field(default=None, hash=False, compare=False)
    _end_state: Any = field(default=None, hash=False, compare=False)

    def __hash__(self):
        return hash((self.start_hash, self.end_hash, self.steps))

    def __len__(self):
        return len(self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)

    def extend_forward(self, next_hash: int, move: Any, next_state: Any = None) -> 'PartialPath':
        """Create new path with one more step forward"""
        new_steps = self.steps + ((next_hash, move),)
        return PartialPath(
            steps=new_steps,
            start_hash=self.start_hash,
            end_hash=next_hash,
            origin=self.origin,
            _start_state=self._start_state,
            _end_state=next_state
        )

    def extend_backward(self, prev_hash: int, move: Any, prev_state: Any = None) -> 'PartialPath':
        """Create new path with one more step backward"""
        new_steps = ((prev_hash, move),) + self.steps
        return PartialPath(
            steps=new_steps,
            start_hash=prev_hash,
            end_hash=self.end_hash,
            origin=self.origin,
            _start_state=prev_state,
            _end_state=self._end_state
        )

    def can_connect_to(self, other: 'PartialPath') -> bool:
        """Check if this path can connect to another"""
        # Forward path's end meets backward path's start
        if self.origin == WaveOrigin.FORWARD and other.origin == WaveOrigin.BACKWARD:
            return self.end_hash == other.start_hash
        if self.origin == WaveOrigin.BACKWARD and other.origin == WaveOrigin.FORWARD:
            return self.start_hash == other.end_hash
        return False

    def connect(self, other: 'PartialPath') -> Optional['PartialPath']:
        """Connect two paths if possible"""
        if not self.can_connect_to(other):
            return None

        if self.origin == WaveOrigin.FORWARD:
            # self is forward, other is backward
            # Connection point is self.end = other.start
            combined = self.steps + other.steps
            return PartialPath(
                steps=combined,
                start_hash=self.start_hash,
                end_hash=other.end_hash,
                origin=WaveOrigin.BOTH,
                _start_state=self._start_state,
                _end_state=other._end_state
            )
        else:
            # self is backward, other is forward
            combined = other.steps + self.steps
            return PartialPath(
                steps=combined,
                start_hash=other.start_hash,
                end_hash=self.end_hash,
                origin=WaveOrigin.BOTH,
                _start_state=other._start_state,
                _end_state=self._end_state
            )

    def to_spine(self, end_value: Any) -> SpinePath:
        """Convert completed path to SpinePath"""
        moves = [move for _, move in self.steps]
        return SpinePath(
            start_hash=self.start_hash,
            moves=moves,
            end_hash=self.end_hash,
            end_value=end_value,
            depth=len(moves)
        )

    def signature(self) -> str:
        """Human-readable signature"""
        return f"Path({self.origin.value}, len={len(self)}, {self.start_hash % 10000:04d}→{self.end_hash % 10000:04d})"


# ============================================================
# LAYER 1 VALUE: PATH QUALITY
# ============================================================

@dataclass(frozen=True)
class PathValue:
    """
    Value of a path at Layer 1.

    Measures path quality for prioritization:
    - coverage: How many Layer 0 states does this path touch?
    - efficiency: Coverage per unit path length
    - closure_potential: How likely is this path to lead to closure?
    - priority: Overall priority score for extending this path
    """
    coverage: int              # States touched by this path
    efficiency: float          # coverage / length
    closure_potential: float   # Estimated probability of reaching closure
    priority: float            # Combined priority score

    # For completed paths
    is_complete: bool = False
    end_value: Any = None      # Layer 0 value at path end

    def __repr__(self):
        status = "COMPLETE" if self.is_complete else "partial"
        return f"PathValue({status}, cov={self.coverage}, eff={self.efficiency:.1f}, pri={self.priority:.1f})"

    def __lt__(self, other):
        return self.priority < other.priority


# ============================================================
# LAYER 1 GAME INTERFACE
# ============================================================

class PathGame(GameInterface[PartialPath, PathValue]):
    """
    Layer 1 game: Search over paths through Layer 0.

    State space: All partial paths through Layer 0
    Successors: Extend path forward (add one more move)
    Predecessors: Extend path backward (add one predecessor)
    Boundary: Complete paths (reach Layer 0 boundary)
    Value: Path quality (coverage, efficiency, closure potential)

    This implements GameInterface so HOLOS can search Layer 1
    using the same bidirectional algorithm as Layer 0.
    """

    def __init__(self,
                 layer0_game: GameInterface[S, V],
                 max_path_length: int = 50,
                 closure_detector: ClosureDetector = None):
        """
        Args:
            layer0_game: The underlying Layer 0 game
            max_path_length: Maximum path length to explore
            closure_detector: Shared closure detector
        """
        self.layer0_game = layer0_game
        self.max_path_length = max_path_length
        self.closure_detector = closure_detector or ClosureDetector()

        # Path registry: hash -> path (for lookup)
        self.path_registry: Dict[int, PartialPath] = {}

        # Forward and backward path fronts (for connection detection)
        self.forward_paths: Dict[int, List[PartialPath]] = defaultdict(list)  # end_hash -> paths
        self.backward_paths: Dict[int, List[PartialPath]] = defaultdict(list)  # start_hash -> paths

        # Completed paths (closures)
        self.completed_paths: List[PartialPath] = []

        # Statistics
        self.stats = {
            'paths_created': 0,
            'paths_extended': 0,
            'closures_found': 0,
            'max_path_length_seen': 0,
        }

    def hash_state(self, path: PartialPath) -> int:
        return hash(path)

    def get_successors(self, path: PartialPath) -> List[Tuple[PartialPath, Any]]:
        """
        Extend path forward (toward Layer 0 boundary).

        For a forward path: add moves from the end state
        For a backward path: this goes "deeper" from boundary
        """
        if path.length >= self.max_path_length:
            return []

        successors = []
        end_state = path._end_state

        if end_state is None:
            return []

        # Get Layer 0 successors from path end
        for child, move in self.layer0_game.get_successors(end_state):
            child_hash = self.layer0_game.hash_state(child)
            new_path = path.extend_forward(child_hash, move, child)
            successors.append((new_path, ('forward', move)))
            self.stats['paths_extended'] += 1

        return successors

    def get_predecessors(self, path: PartialPath) -> List[Tuple[PartialPath, Any]]:
        """
        Extend path backward (from Layer 0 boundary toward start).

        For a backward path: add predecessors from the start state
        For a forward path: this goes "toward origin"
        """
        if path.length >= self.max_path_length:
            return []

        predecessors = []
        start_state = path._start_state

        if start_state is None:
            return []

        # Get Layer 0 predecessors from path start
        for parent, move in self.layer0_game.get_predecessors(start_state):
            parent_hash = self.layer0_game.hash_state(parent)
            new_path = path.extend_backward(parent_hash, move, parent)
            predecessors.append((new_path, ('backward', move)))
            self.stats['paths_extended'] += 1

        return predecessors

    def is_boundary(self, path: PartialPath) -> bool:
        """
        Path is at boundary if it reaches Layer 0 boundary.
        """
        if path._end_state is not None:
            return self.layer0_game.is_boundary(path._end_state)
        return False

    def get_boundary_value(self, path: PartialPath) -> Optional[PathValue]:
        """Get value for path that reaches Layer 0 boundary"""
        if not self.is_boundary(path):
            return None

        # Get Layer 0 boundary value
        l0_value = self.layer0_game.get_boundary_value(path._end_state)

        return PathValue(
            coverage=path.length,
            efficiency=1.0,  # Boundary paths have perfect efficiency
            closure_potential=1.0,  # Already at boundary
            priority=1000.0,  # High priority
            is_complete=True,
            end_value=l0_value
        )

    def is_terminal(self, path: PartialPath) -> Tuple[bool, Optional[PathValue]]:
        """Path is terminal if it's complete (forward meets backward)"""
        if path.origin == WaveOrigin.BOTH:
            # This is a connected path
            return True, PathValue(
                coverage=path.length,
                efficiency=path.length / max(1, path.length),
                closure_potential=1.0,
                priority=float('inf'),
                is_complete=True
            )
        return False, None

    def propagate_value(self, path: PartialPath, child_values: List[PathValue]) -> Optional[PathValue]:
        """
        Propagate value from child paths to parent.

        A path's value is the best value among its extensions,
        adjusted for the cost of extension.
        """
        if not child_values:
            return None

        # Best child by priority
        best = max(child_values, key=lambda v: v.priority)

        # Parent path is one step shorter, so slightly lower coverage
        return PathValue(
            coverage=max(0, best.coverage - 1),
            efficiency=best.efficiency * 0.95,  # Slight decay
            closure_potential=best.closure_potential * 0.9,
            priority=best.priority * 0.95,
            is_complete=False
        )

    def get_features(self, path: PartialPath) -> Any:
        """Feature extraction for equivalence"""
        return (path.length, path.origin)

    # ==================== Path-Specific Methods ====================

    def create_initial_path(self, state: S, direction: WaveOrigin) -> PartialPath:
        """Create a single-state path as starting point"""
        h = self.layer0_game.hash_state(state)
        path = PartialPath(
            steps=(),
            start_hash=h,
            end_hash=h,
            origin=direction,
            _start_state=state,
            _end_state=state
        )
        self.stats['paths_created'] += 1
        return path

    def register_path(self, path: PartialPath):
        """Register path for connection detection"""
        h = hash(path)
        self.path_registry[h] = path

        if path.origin == WaveOrigin.FORWARD:
            self.forward_paths[path.end_hash].append(path)
        elif path.origin == WaveOrigin.BACKWARD:
            self.backward_paths[path.start_hash].append(path)

        if path.length > self.stats['max_path_length_seen']:
            self.stats['max_path_length_seen'] = path.length

    def find_connections(self) -> List[Tuple[PartialPath, PartialPath]]:
        """Find forward-backward path pairs that can connect"""
        connections = []

        # For each forward path ending at hash H,
        # look for backward paths starting at hash H
        for end_hash, fwd_paths in self.forward_paths.items():
            if end_hash in self.backward_paths:
                bwd_paths = self.backward_paths[end_hash]
                for fwd in fwd_paths:
                    for bwd in bwd_paths:
                        if fwd.can_connect_to(bwd):
                            connections.append((fwd, bwd))

        return connections

    def connect_paths(self, forward: PartialPath, backward: PartialPath) -> Optional[PartialPath]:
        """Connect a forward and backward path"""
        connected = forward.connect(backward)
        if connected:
            self.completed_paths.append(connected)
            self.stats['closures_found'] += 1

            # Record closure event
            self.closure_detector.check_closure(
                state_hash=forward.end_hash,  # Connection point
                forward_value=forward.length,
                backward_value=backward.length,
                layer=1,
                iteration=0
            )

        return connected

    def evaluate_path(self, path: PartialPath) -> PathValue:
        """
        Evaluate a path's quality.

        This is called to assign priority scores.
        """
        # Base coverage
        coverage = path.length

        # Efficiency
        efficiency = coverage / max(1, path.length)

        # Closure potential: how likely to reach closure?
        # Based on whether similar paths have succeeded
        closure_potential = 0.5  # Default

        # Check if end position is near known closures
        if path.end_hash in self.closure_detector.closure_by_hash:
            closure_potential = 0.9

        # Check for connection opportunity
        if path.origin == WaveOrigin.FORWARD and path.end_hash in self.backward_paths:
            closure_potential = 1.0
        if path.origin == WaveOrigin.BACKWARD and path.start_hash in self.forward_paths:
            closure_potential = 1.0

        # Priority combines all factors
        priority = coverage * efficiency * closure_potential * 10.0

        return PathValue(
            coverage=coverage,
            efficiency=efficiency,
            closure_potential=closure_potential,
            priority=priority
        )


# ============================================================
# LAYER 1 SOLVER
# ============================================================

class PathLayerSolver:
    """
    Solver for Layer 1 (path scale).

    Uses HOLOS to search the space of paths, finding optimal
    paths through Layer 0 state space.

    The solver maintains:
    - Forward paths: Growing from start states toward boundary
    - Backward paths: Growing from boundary toward start
    - Connections: Where forward and backward paths meet
    """

    def __init__(self,
                 layer0_game: GameInterface,
                 closure_detector: ClosureDetector = None,
                 max_path_length: int = 50):
        """
        Args:
            layer0_game: The underlying Layer 0 game
            closure_detector: Shared closure detector
            max_path_length: Maximum path length
        """
        self.closure_detector = closure_detector or ClosureDetector()
        self.path_game = PathGame(layer0_game, max_path_length, self.closure_detector)
        self.layer0_game = layer0_game

        # Path fronts
        self.forward_front: Dict[int, PartialPath] = {}  # hash -> path
        self.backward_front: Dict[int, PartialPath] = {}

        # Priority queues (sorted by PathValue.priority)
        self.forward_queue: List[Tuple[float, PartialPath]] = []
        self.backward_queue: List[Tuple[float, PartialPath]] = []

        # Results
        self.completed_spines: List[SpinePath] = []

        # Statistics
        self.stats = {
            'iterations': 0,
            'forward_extensions': 0,
            'backward_extensions': 0,
            'connections': 0,
            'spines_found': 0,
        }

    def add_forward_seed(self, state: S):
        """Add a starting state for forward paths"""
        path = self.path_game.create_initial_path(state, WaveOrigin.FORWARD)
        value = self.path_game.evaluate_path(path)
        h = hash(path)
        self.forward_front[h] = path
        self.forward_queue.append((-value.priority, path))
        self.forward_queue.sort(key=lambda x: x[0])
        self.path_game.register_path(path)

    def add_backward_seed(self, state: S):
        """Add a boundary state for backward paths"""
        path = self.path_game.create_initial_path(state, WaveOrigin.BACKWARD)
        value = self.path_game.evaluate_path(path)
        h = hash(path)
        self.backward_front[h] = path
        self.backward_queue.append((-value.priority, path))
        self.backward_queue.sort(key=lambda x: x[0])
        self.path_game.register_path(path)

    def step(self, mode: str = "balanced") -> Dict:
        """
        Perform one step of path search.

        Args:
            mode: "forward", "backward", "balanced", or "osmosis"

        Returns:
            Dict with step statistics
        """
        self.stats['iterations'] += 1
        result = {
            'extended': 0,
            'connections': 0,
            'spines': 0,
        }

        if mode == "osmosis":
            # Pick single best path from either front
            return self._step_osmosis()
        elif mode == "forward":
            self._extend_forward()
            result['extended'] = 1
        elif mode == "backward":
            self._extend_backward()
            result['extended'] = 1
        else:  # balanced
            self._extend_forward()
            self._extend_backward()
            result['extended'] = 2

        # Check for connections
        connections = self.path_game.find_connections()
        for fwd, bwd in connections:
            connected = self.path_game.connect_paths(fwd, bwd)
            if connected:
                # Get Layer 0 value at endpoint
                end_value = None
                if connected._end_state is not None:
                    if self.layer0_game.is_boundary(connected._end_state):
                        end_value = self.layer0_game.get_boundary_value(connected._end_state)

                spine = connected.to_spine(end_value)
                self.completed_spines.append(spine)
                result['connections'] += 1
                result['spines'] += 1
                self.stats['connections'] += 1
                self.stats['spines_found'] += 1

        return result

    def _extend_forward(self):
        """Extend best forward path"""
        if not self.forward_queue:
            return

        _, path = self.forward_queue.pop(0)

        for new_path, move in self.path_game.get_successors(path):
            value = self.path_game.evaluate_path(new_path)
            h = hash(new_path)
            if h not in self.forward_front:
                self.forward_front[h] = new_path
                self.forward_queue.append((-value.priority, new_path))
                self.path_game.register_path(new_path)
                self.stats['forward_extensions'] += 1

        self.forward_queue.sort(key=lambda x: x[0])

    def _extend_backward(self):
        """Extend best backward path"""
        if not self.backward_queue:
            return

        _, path = self.backward_queue.pop(0)

        for new_path, move in self.path_game.get_predecessors(path):
            value = self.path_game.evaluate_path(new_path)
            h = hash(new_path)
            if h not in self.backward_front:
                self.backward_front[h] = new_path
                self.backward_queue.append((-value.priority, new_path))
                self.path_game.register_path(new_path)
                self.stats['backward_extensions'] += 1

        self.backward_queue.sort(key=lambda x: x[0])

    def _step_osmosis(self) -> Dict:
        """Osmosis: pick single highest-priority path to extend"""
        result = {'extended': 0, 'connections': 0, 'spines': 0}

        # Compare best from each front
        fwd_best = self.forward_queue[0] if self.forward_queue else None
        bwd_best = self.backward_queue[0] if self.backward_queue else None

        if fwd_best is None and bwd_best is None:
            return result

        # Pick better one (lower negative priority = better)
        if fwd_best is None:
            self._extend_backward()
        elif bwd_best is None:
            self._extend_forward()
        elif fwd_best[0] < bwd_best[0]:
            self._extend_forward()
        else:
            self._extend_backward()

        result['extended'] = 1

        # Check connections
        connections = self.path_game.find_connections()
        for fwd, bwd in connections:
            connected = self.path_game.connect_paths(fwd, bwd)
            if connected:
                end_value = None
                if connected._end_state is not None and self.layer0_game.is_boundary(connected._end_state):
                    end_value = self.layer0_game.get_boundary_value(connected._end_state)
                spine = connected.to_spine(end_value)
                self.completed_spines.append(spine)
                result['connections'] += 1
                result['spines'] += 1

        return result

    def solve(self,
              forward_seeds: List[S],
              backward_seeds: List[S] = None,
              max_iterations: int = 100,
              mode: str = "balanced",
              verbose: bool = True) -> List[SpinePath]:
        """
        Solve: find paths connecting forward seeds to backward seeds.

        Args:
            forward_seeds: Starting states
            backward_seeds: Boundary states (auto-generated if None)
            max_iterations: Maximum iterations
            mode: Search mode
            verbose: Print progress

        Returns:
            List of completed spines (paths from start to boundary)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Layer 1 Path Solver")
            print(f"  Mode: {mode}")
            print(f"  Forward seeds: {len(forward_seeds)}")
            print(f"{'='*60}")

        # Initialize forward paths
        for state in forward_seeds:
            self.add_forward_seed(state)

        # Initialize backward paths
        if backward_seeds is None:
            # Auto-generate from Layer 0
            if forward_seeds:
                backward_seeds = self.layer0_game.generate_boundary_seeds(
                    forward_seeds[0], count=100
                )

        if backward_seeds:
            for state in backward_seeds:
                self.add_backward_seed(state)

        if verbose:
            print(f"  Backward seeds: {len(backward_seeds) if backward_seeds else 0}")

        # Main loop
        for i in range(max_iterations):
            result = self.step(mode=mode)

            if verbose and (i % 10 == 0 or result['spines'] > 0):
                print(f"  Iter {i}: fwd={len(self.forward_front)}, bwd={len(self.backward_front)}, "
                      f"spines={len(self.completed_spines)}")

            if not self.forward_queue and not self.backward_queue:
                if verbose:
                    print("  Fronts exhausted")
                break

        if verbose:
            print(f"\n{'='*60}")
            print(f"Complete: {len(self.completed_spines)} spines found")
            print(f"  Closures detected: {self.closure_detector.stats['total_closures']}")
            print(f"  Irreducible: {self.closure_detector.stats['irreducible']}")
            print(f"{'='*60}")

        return self.completed_spines


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_path_solver(layer0_game: GameInterface,
                       max_path_length: int = 50) -> PathLayerSolver:
    """Create a Layer 1 path solver for the given Layer 0 game"""
    closure_detector = ClosureDetector()
    return PathLayerSolver(layer0_game, closure_detector, max_path_length)


def solve_with_paths(layer0_game: GameInterface,
                     start_states: List[Any],
                     boundary_states: List[Any] = None,
                     max_iterations: int = 100) -> List[SpinePath]:
    """
    Convenience function to solve using Layer 1 path search.

    Args:
        layer0_game: The game to solve
        start_states: Starting positions
        boundary_states: Boundary positions (optional)
        max_iterations: Maximum iterations

    Returns:
        List of solution spines
    """
    solver = create_path_solver(layer0_game)
    return solver.solve(start_states, boundary_states, max_iterations)
