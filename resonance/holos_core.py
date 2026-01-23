"""
holos_core.py - Universal HOLOS Algorithm (Layer-Agnostic)

This is the GENERIC bidirectional search algorithm that works on ANY game/lattice.
It doesn't know about chess, seeds, or any specific domain.

The algorithm:
1. LIGHTNING: DFS probes to find quick paths between frontiers
2. WAVE: BFS expansion from frontiers
3. CRYSTAL: Local search around connection points

To use HOLOS on a new domain, implement the GameInterface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Set, List, Tuple, Optional, Any, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import random
import time

# Generic type for states
S = TypeVar('S')  # State type
V = TypeVar('V')  # Value type


class SearchMode(Enum):
    """Search modes the meta-learner can choose"""
    LIGHTNING = "lightning"  # DFS - fast probes to find paths
    WAVE = "wave"            # BFS - complete frontier expansion
    CRYSTAL = "crystal"      # Local search around known-good points


# ============================================================
# GAME INTERFACE (Implement this for each domain)
# ============================================================

class GameInterface(ABC, Generic[S, V]):
    """
    Abstract interface for any game/optimization problem.

    Implement this to use HOLOS on your domain.
    """

    @abstractmethod
    def hash_state(self, state: S) -> int:
        """Return unique hash for state"""
        pass

    @abstractmethod
    def get_successors(self, state: S) -> List[Tuple[S, Any]]:
        """
        Return list of (successor_state, move) pairs.
        'move' can be any object describing the transition.
        """
        pass

    @abstractmethod
    def get_predecessors(self, state: S) -> List[Tuple[S, Any]]:
        """
        Return list of (predecessor_state, move) pairs.
        For backward wave expansion.
        """
        pass

    @abstractmethod
    def is_boundary(self, state: S) -> bool:
        """Is this state on the known boundary (solved)?"""
        pass

    @abstractmethod
    def get_boundary_value(self, state: S) -> Optional[V]:
        """Get value if state is on boundary, None otherwise"""
        pass

    @abstractmethod
    def is_terminal(self, state: S) -> Tuple[bool, Optional[V]]:
        """Is this a terminal state? If so, what's its value?"""
        pass

    @abstractmethod
    def propagate_value(self, state: S, child_values: List[V]) -> Optional[V]:
        """
        Given a state and its children's values, compute this state's value.

        For minimax: max/min depending on whose turn
        For optimization: could be max, weighted average, etc.

        Return None if value cannot be determined yet.
        """
        pass

    @abstractmethod
    def get_features(self, state: S) -> Any:
        """Extract equivalence class features for pattern matching"""
        pass

    def get_lightning_successors(self, state: S) -> List[Tuple[S, Any]]:
        """
        Successors for lightning (DFS) mode.
        Default: same as regular successors, but subclass can filter
        to only "aggressive" moves (like captures in chess).
        """
        return self.get_successors(state)

    def score_for_lightning(self, state: S, move: Any) -> float:
        """Score a move for lightning ordering. Higher = try first."""
        return 0.0


# ============================================================
# HOLOGRAM (Storage)
# ============================================================

@dataclass
class Hologram(Generic[S, V]):
    """
    Holographic storage for solved states and patterns.

    Layer-agnostic - works with any state/value types.
    """
    name: str
    solved: Dict[int, V] = field(default_factory=dict)
    boundary_hashes: Set[int] = field(default_factory=set)

    # Equivalence class tracking
    equiv_classes: Dict[Any, Set[int]] = field(default_factory=lambda: defaultdict(set))
    equiv_values: Dict[Any, Optional[V]] = field(default_factory=dict)

    # Spine paths (principal variations)
    spines: List[Tuple[int, List[Any], int, V]] = field(default_factory=list)

    def query(self, h: int) -> Optional[V]:
        return self.solved.get(h)

    def add_solved(self, h: int, value: V, features: Any = None):
        self.solved[h] = value
        if features is not None:
            self.equiv_classes[features].add(h)
            self._update_equiv_value(features, value)

    def add_boundary(self, h: int, value: V):
        self.solved[h] = value
        self.boundary_hashes.add(h)

    def _update_equiv_value(self, features: Any, value: V):
        """Track consistent value for equivalence class"""
        if features in self.equiv_values:
            if self.equiv_values[features] != value:
                self.equiv_values[features] = None  # Inconsistent
        else:
            self.equiv_values[features] = value


# ============================================================
# LIGHTNING PROBE (DFS)
# ============================================================

class LightningProbe(Generic[S, V]):
    """
    Fast DFS to find paths between frontiers.

    Like lightning finding ground - follows the path of least resistance
    (highest-scored moves) to quickly reach known territory.
    """

    def __init__(self, game: GameInterface[S, V], hologram: Hologram[S, V],
                 max_depth: int = 15, max_nodes: int = 10000):
        self.game = game
        self.hologram = hologram
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.nodes_visited = 0

    def probe(self, state: S) -> Tuple[Optional[V], List[Tuple[S, Any]]]:
        """
        Search from state to find path to known territory.
        Returns (value, path) where path is list of (state, move).
        """
        self.nodes_visited = 0
        path = []
        value = self._search(state, 0, path, set())
        return value, path

    def _search(self, state: S, depth: int, path: List, visited: Set[int]) -> Optional[V]:
        self.nodes_visited += 1

        if self.nodes_visited > self.max_nodes:
            return None

        h = self.game.hash_state(state)

        if h in visited:
            return None
        visited.add(h)

        # Check hologram
        cached = self.hologram.query(h)
        if cached is not None:
            return cached

        # Check boundary
        if self.game.is_boundary(state):
            value = self.game.get_boundary_value(state)
            if value is not None:
                return value

        # Depth limit
        if depth >= self.max_depth:
            return None

        # Terminal check
        is_term, term_value = self.game.is_terminal(state)
        if is_term:
            return term_value

        # Get lightning successors (filtered/ordered for speed)
        successors = self.game.get_lightning_successors(state)
        if not successors:
            return None

        # Score and sort
        scored = [(self.game.score_for_lightning(state, move), child, move)
                  for child, move in successors]
        scored.sort(reverse=True, key=lambda x: x[0])

        # Try top successors
        for score, child, move in scored[:5]:
            child_value = self._search(child, depth + 1, path, visited)

            if child_value is not None:
                path.insert(0, (state, move))
                return child_value

        return None


# ============================================================
# HOLOS SOLVER (The Universal Algorithm)
# ============================================================

@dataclass
class SeedPoint(Generic[S]):
    """A seed point for wave expansion"""
    state: S
    mode: SearchMode = SearchMode.WAVE
    max_depth: int = 1  # How far this seed should expand


class HOLOSSolver(Generic[S, V]):
    """
    Universal HOLOS solver.

    Works on any domain that implements GameInterface.

    The algorithm:
    1. Initialize forward and backward frontiers
    2. Expand waves from both directions
    3. Use lightning probes to find connections
    4. Crystallize around connection points
    5. Propagate values
    """

    def __init__(self, game: GameInterface[S, V], name: str = "holos"):
        self.game = game
        self.hologram = Hologram[S, V](name)

        # Bidirectional frontiers
        self.forward_frontier: Dict[int, S] = {}
        self.backward_frontier: Dict[int, S] = {}

        self.forward_seen: Set[int] = set()
        self.backward_seen: Set[int] = set()

        # Parent tracking
        self.forward_parents: Dict[int, Tuple[int, Any]] = {}
        self.backward_parents: Dict[int, Tuple[int, Any]] = {}

        # Connections (where waves meet)
        self.connections: List[int] = []

        # Stats
        self.stats = {
            'lightning_probes': 0,
            'connections': 0,
            'crystallized': 0,
            'propagated': 0,
        }

    def seed_forward(self, states: List[S]):
        """Add states to forward frontier"""
        for state in states:
            h = self.game.hash_state(state)
            if h not in self.forward_seen:
                self.forward_seen.add(h)
                self.forward_frontier[h] = state

    def seed_backward(self, states: List[S]):
        """Add states to backward frontier (usually boundary states)"""
        for state in states:
            h = self.game.hash_state(state)
            if h not in self.backward_seen:
                self.backward_seen.add(h)
                self.backward_frontier[h] = state

                # If it's a boundary, record its value
                if self.game.is_boundary(state):
                    value = self.game.get_boundary_value(state)
                    if value is not None:
                        features = self.game.get_features(state)
                        self.hologram.add_solved(h, value, features)
                        self.hologram.add_boundary(h, value)

    def solve(self, forward_seeds: List[SeedPoint[S]] = None,
              backward_seeds: List[SeedPoint[S]] = None,
              max_iterations: int = 100,
              lightning_interval: int = 5,
              verbose: bool = True):
        """
        Run bidirectional HOLOS search.

        Args:
            forward_seeds: Starting points for forward wave
            backward_seeds: Starting points for backward wave (boundary)
            max_iterations: Maximum expansion iterations
            lightning_interval: How often to run lightning probes
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"HOLOS SOLVER - Bidirectional Search")
            print(f"{'='*60}")

        # Initialize frontiers from seeds
        if forward_seeds:
            self.seed_forward([sp.state for sp in forward_seeds])
        if backward_seeds:
            self.seed_backward([sp.state for sp in backward_seeds])

        if verbose:
            print(f"Forward frontier: {len(self.forward_frontier)}")
            print(f"Backward frontier: {len(self.backward_frontier)}")

        start_time = time.time()

        for iteration in range(max_iterations):
            if not self.forward_frontier and not self.backward_frontier:
                if verbose:
                    print("\nBoth frontiers empty!")
                break

            # Lightning probes
            if iteration % lightning_interval == 0 and self.forward_frontier:
                lightning_hits = self._lightning_phase()
                if verbose and lightning_hits:
                    print(f"  Lightning: {lightning_hits} paths found")

            # Expand frontiers
            forward_contacts = self._expand_forward()
            backward_contacts = self._expand_backward()

            # Find connections
            new_connections = self._find_connections()

            if new_connections:
                if verbose:
                    print(f"  ** {new_connections} CONNECTIONS **")
                self._crystallize()

            # Propagate values
            propagated = self._propagate()

            if verbose and iteration % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iteration}: solved={len(self.hologram.solved)}, "
                      f"fwd={len(self.forward_frontier)}, bwd={len(self.backward_frontier)}, "
                      f"conn={len(self.connections)}, time={elapsed:.1f}s")

        if verbose:
            print(f"\n{'='*60}")
            print(f"COMPLETE: {len(self.hologram.solved)} states solved")
            print(f"{'='*60}")

        return self.hologram

    def _lightning_phase(self, num_probes: int = 10) -> int:
        """Run lightning probes from forward frontier"""
        probe = LightningProbe(self.game, self.hologram)

        sample_size = min(num_probes, len(self.forward_frontier))
        samples = random.sample(list(self.forward_frontier.values()), sample_size)

        hits = 0
        for state in samples:
            value, path = probe.probe(state)
            self.stats['lightning_probes'] += 1

            if value is not None and path:
                # Found path - record spine and values
                h = self.game.hash_state(state)
                features = self.game.get_features(state)
                self.hologram.add_solved(h, value, features)
                hits += 1

                # Add intermediate states
                for s, m in path:
                    sh = self.game.hash_state(s)
                    if sh not in self.hologram.solved:
                        sf = self.game.get_features(s)
                        self.hologram.add_solved(sh, value, sf)

        return hits

    def _expand_forward(self) -> int:
        """Expand forward frontier by one layer"""
        items = list(self.forward_frontier.items())
        if not items:
            return 0

        next_frontier = {}
        contacts = 0

        for h, state in items:
            # Already solved?
            if h in self.hologram.solved:
                contacts += 1
                continue

            # Track features
            features = self.game.get_features(state)
            self.hologram.equiv_classes[features].add(h)

            # Check equivalence shortcut
            if features in self.hologram.equiv_values:
                eq_val = self.hologram.equiv_values[features]
                if eq_val is not None:
                    self.hologram.add_solved(h, eq_val, features)
                    contacts += 1
                    continue

            # Terminal?
            is_term, term_val = self.game.is_terminal(state)
            if is_term:
                self.hologram.add_solved(h, term_val, features)
                contacts += 1
                continue

            # Expand successors
            for child, move in self.game.get_successors(state):
                ch = self.game.hash_state(child)

                # Check boundary
                if self.game.is_boundary(child):
                    value = self.game.get_boundary_value(child)
                    if value is not None:
                        cf = self.game.get_features(child)
                        self.hologram.add_solved(ch, value, cf)
                        self.hologram.add_boundary(ch, value)
                        contacts += 1

                        # Seed backward from this boundary point
                        if ch not in self.backward_seen:
                            self.backward_seen.add(ch)
                            self.backward_frontier[ch] = child
                        continue

                # Check backward wave
                if ch in self.backward_seen:
                    contacts += 1
                    self.stats['connections'] += 1
                    continue

                if ch not in self.forward_seen:
                    self.forward_seen.add(ch)
                    next_frontier[ch] = child
                    self.forward_parents[ch] = (h, move)

        self.forward_frontier = next_frontier
        return contacts

    def _expand_backward(self) -> int:
        """Expand backward frontier using predecessors"""
        items = list(self.backward_frontier.items())
        if not items:
            return 0

        next_frontier = {}
        contacts = 0

        # Track predecessors and their children for value propagation
        pred_children: Dict[int, List[Tuple[int, V]]] = defaultdict(list)

        for h, state in items:
            for pred, move in self.game.get_predecessors(state):
                ph = self.game.hash_state(pred)

                # Track child value for propagation
                if h in self.hologram.solved:
                    pred_children[ph].append((h, self.hologram.solved[h]))

                if ph in self.backward_seen:
                    continue

                self.backward_seen.add(ph)

                # Track features
                features = self.game.get_features(pred)
                self.hologram.equiv_classes[features].add(ph)

                # Check forward wave
                if ph in self.forward_seen:
                    contacts += 1
                    self.stats['connections'] += 1
                    self.connections.append(ph)
                    continue

                next_frontier[ph] = pred
                self.backward_parents[ph] = (h, move)

        # Value propagation via game rules
        for ph, children in pred_children.items():
            if ph in self.hologram.solved:
                continue

            child_values = [v for _, v in children]
            pred_state = next_frontier.get(ph) or self.backward_frontier.get(ph)

            if pred_state is not None:
                value = self.game.propagate_value(pred_state, child_values)
                if value is not None:
                    features = self.game.get_features(pred_state)
                    self.hologram.add_solved(ph, value, features)
                    contacts += 1

        self.backward_frontier = next_frontier
        return contacts

    def _find_connections(self) -> int:
        """Find where forward and backward waves meet"""
        overlap = self.forward_seen & self.backward_seen
        new_conn = 0

        for h in overlap:
            if h not in self.connections:
                self.connections.append(h)
                new_conn += 1

        return new_conn

    def _crystallize(self, depth: int = 3):
        """Local BFS around connection points"""
        for h in self.connections[-10:]:  # Recent connections
            state = self.forward_frontier.get(h) or self.backward_frontier.get(h)
            if state is None:
                continue

            local_frontier = {h: state}
            local_seen = {h}

            for _ in range(depth):
                next_local = {}
                for lh, ls in local_frontier.items():
                    for child, move in self.game.get_successors(ls):
                        ch = self.game.hash_state(child)
                        if ch in local_seen:
                            continue
                        local_seen.add(ch)
                        next_local[ch] = child

                        if self.game.is_boundary(child):
                            value = self.game.get_boundary_value(child)
                            if value is not None:
                                self.hologram.add_boundary(ch, value)
                                self.stats['crystallized'] += 1

                local_frontier = next_local

    def _propagate(self, max_iters: int = 50) -> int:
        """Propagate values through parent links and equivalence"""
        total = 0

        for _ in range(max_iters):
            newly_solved = 0

            # Equivalence propagation
            for features, hashes in self.hologram.equiv_classes.items():
                if features not in self.hologram.equiv_values:
                    continue
                eq_val = self.hologram.equiv_values[features]
                if eq_val is None:
                    continue
                for h in hashes:
                    if h not in self.hologram.solved:
                        self.hologram.solved[h] = eq_val
                        newly_solved += 1

            total += newly_solved
            if newly_solved == 0:
                break

        self.stats['propagated'] += total
        return total
