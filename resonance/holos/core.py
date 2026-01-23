"""
holos/core.py - Universal HOLOS Algorithm

This module contains the game-agnostic HOLOS solver. It implements:
- GameInterface: Abstract base class for any game/lattice
- SearchMode: Lightning, Wave, Crystal modes
- LightningProbe: Fast DFS to find paths (forward AND backward)
- HOLOSSolver: Main solver with bidirectional search

The key insight: Forward and Backward are MIRRORS.
- Forward lightning: DFS from start toward boundary
- Backward lightning: DFS from boundary toward start (using predecessors)

Both use the same algorithm, just with different direction functions.

CONSISTENCY NOTE: This module is designed to be functionally equivalent to
fractal_holos3.py's BidirectionalHOLOS class, but game-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Tuple, Optional, Any, Dict, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import random
import time

# Type variables for game state and value
S = TypeVar('S')  # State type
V = TypeVar('V')  # Value type


class SearchMode(Enum):
    """Search modes for seed expansion"""
    LIGHTNING = "lightning"  # DFS probe for fast paths
    WAVE = "wave"            # BFS for breadth coverage
    CRYSTAL = "crystal"      # Local search around connections


@dataclass
class SeedPoint(Generic[S]):
    """A seed point for bidirectional search"""
    state: S
    mode: SearchMode
    priority: int = 1
    depth: int = 1  # How deep to expand from this seed

    def __hash__(self):
        return hash((id(self.state), self.mode, self.priority, self.depth))


class GameInterface(ABC, Generic[S, V]):
    """
    Abstract interface for any game that HOLOS can solve.

    Games are defined by:
    - States (positions in the game space)
    - Successors (forward moves: state -> children)
    - Predecessors (backward moves: state -> parents)
    - Boundary (states with known values)
    - Value propagation (how to combine child values)

    The interface is symmetric: forward and backward use the same structure.
    """

    @abstractmethod
    def hash_state(self, state: S) -> int:
        """Hash a state for deduplication"""
        pass

    @abstractmethod
    def get_successors(self, state: S) -> List[Tuple[S, Any]]:
        """Get successor states (forward direction)"""
        pass

    @abstractmethod
    def get_predecessors(self, state: S) -> List[Tuple[S, Any]]:
        """Get predecessor states (backward direction)"""
        pass

    @abstractmethod
    def is_boundary(self, state: S) -> bool:
        """Is this state on the boundary (has known value)?"""
        pass

    @abstractmethod
    def get_boundary_value(self, state: S) -> Optional[V]:
        """Get the known value for a boundary state"""
        pass

    @abstractmethod
    def is_terminal(self, state: S) -> Tuple[bool, Optional[V]]:
        """Check if state is terminal (game over)"""
        pass

    @abstractmethod
    def propagate_value(self, state: S, child_values: List[V]) -> Optional[V]:
        """
        Propagate values from children to parent.

        THIS IS GAME-SPECIFIC:
        - Chess: minimax (white max, black min)
        - Optimization: max efficiency
        - Path finding: min cost
        """
        pass

    def get_features(self, state: S) -> Any:
        """Extract equivalence class features (optional)"""
        return None

    # Optional methods for mode-specific behavior
    def get_lightning_successors(self, state: S) -> List[Tuple[S, Any]]:
        """Successors for lightning mode (e.g., captures only in chess)"""
        return self.get_successors(state)

    def get_lightning_predecessors(self, state: S) -> List[Tuple[S, Any]]:
        """Predecessors for backward lightning (e.g., uncaptures only)"""
        return self.get_predecessors(state)

    def score_for_lightning(self, state: S, move: Any) -> float:
        """Score a move for lightning prioritization"""
        return 0.0

    def generate_boundary_seeds(self, template_state: S, count: int = 100) -> List[S]:
        """
        Generate boundary positions for backward wave seeding.
        Override in game-specific implementations.
        Returns empty list by default (must be overridden for auto-generation).
        """
        return []

    def apply_move(self, state: S, move: Any) -> S:
        """Apply a move to get the resulting state. Used for spine reconstruction."""
        # Default: try to find it in successors
        for child, m in self.get_successors(state):
            if m == move:
                return child
        return None


# ============================================================
# LIGHTNING PROBE (Bidirectional DFS)
# ============================================================

class LightningProbe(Generic[S, V]):
    """
    Fast depth-first probe that finds ONE path to boundary.

    Can run in either direction:
    - Forward: Start from position, find path to boundary
    - Backward: Start from boundary, find path toward start

    The direction is determined by which game methods are used.

    Matches fractal_holos3.py behavior but uses game interface.
    """

    def __init__(self, game: GameInterface[S, V],
                 solved: Dict[int, V],
                 direction: str = "forward",
                 max_depth: int = 15,
                 max_nodes: int = 10000):
        self.game = game
        self.solved = solved
        self.direction = direction
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.nodes_visited = 0

    def probe(self, state: S) -> Tuple[Optional[V], List[Tuple[S, Any]]]:
        """
        Find path from state to boundary (or from boundary to start).
        Returns (value, path) where path is list of (state, move).
        """
        self.nodes_visited = 0
        path = []
        value = self._search(state, 0, path, set())
        return value, path

    def _search(self, state: S, depth: int, path: List, visited: Set[int]) -> Optional[V]:
        """Recursive DFS search"""
        self.nodes_visited += 1

        if self.nodes_visited > self.max_nodes:
            return None

        h = self.game.hash_state(state)

        # Cycle detection
        if h in visited:
            return None
        visited.add(h)

        # Check if already solved
        if h in self.solved:
            return self.solved[h]

        # Check boundary
        if self.game.is_boundary(state):
            value = self.game.get_boundary_value(state)
            if value is not None:
                return value

        # Terminal check
        is_term, term_value = self.game.is_terminal(state)
        if is_term:
            return term_value

        # Depth limit
        if depth >= self.max_depth:
            return None

        # Get next states based on direction
        if self.direction == "forward":
            next_states = self.game.get_lightning_successors(state)
        else:  # backward
            next_states = self.game.get_lightning_predecessors(state)

        if not next_states:
            return None

        # Sort by score (prioritize promising moves)
        scored = [(self.game.score_for_lightning(state, move), child, move)
                  for child, move in next_states]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Try each path (limit branching like fractal_holos3)
        for score, child, move in scored[:5]:
            child_path = []
            value = self._search(child, depth + 1, child_path, visited.copy())

            if value is not None:
                path.clear()
                path.append((state, move))
                path.extend(child_path)
                return value

        return None


# ============================================================
# MODE SELECTION AS META-DECISION
# ============================================================

@dataclass
class ModeDecision:
    """
    Tracks mode selection decisions for meta-learning.

    The idea: mode selection (lightning vs wave vs crystal) is itself
    a decision that can be optimized. Track outcomes to learn which
    mode works best in which situations.
    """
    state_hash: int
    features: Any  # State features at decision point
    mode_chosen: SearchMode
    outcome: Optional[V] = None  # Did this mode find a solution?
    nodes_used: int = 0
    path_length: int = 0
    success: bool = False


class ModeSelector:
    """
    Selects search mode based on state features and history.

    This can be used at Layer 1 or Layer 2 of the meta-game:
    - Layer 1: Choose mode per seed position
    - Layer 2: Choose mode allocation strategy
    """

    def __init__(self):
        self.history: List[ModeDecision] = []
        self.feature_outcomes: Dict[Any, Dict[SearchMode, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def select_mode(self, state: S, game: GameInterface[S, V],
                    default: SearchMode = SearchMode.WAVE) -> SearchMode:
        """
        Select mode based on state features and history.

        Simple strategy:
        - If features match a pattern with known good mode, use that
        - Otherwise use default (wave for safety, lightning for speed)
        """
        features = game.get_features(state)
        if features is None:
            return default

        # Check if we have history for these features
        if features in self.feature_outcomes:
            outcomes = self.feature_outcomes[features]
            best_mode = default
            best_success_rate = 0.0

            for mode, results in outcomes.items():
                if results:
                    rate = sum(results) / len(results)
                    if rate > best_success_rate:
                        best_success_rate = rate
                        best_mode = mode

            return best_mode

        return default

    def record_outcome(self, decision: ModeDecision):
        """Record the outcome of a mode decision"""
        self.history.append(decision)
        if decision.features is not None:
            self.feature_outcomes[decision.features][decision.mode_chosen].append(
                decision.success
            )


# ============================================================
# HOLOS SOLVER
# ============================================================

class HOLOSSolver(Generic[S, V]):
    """
    Universal HOLOS solver using bidirectional search.

    The solver maintains:
    - Forward frontier: States to expand toward boundary
    - Backward frontier: States to expand from boundary toward start
    - Solved states: Map of hash -> value
    - Connections: Where forward and backward waves meet

    Key methods:
    - solve(): Main entry point
    - expand_forward(): One step of forward wave
    - expand_backward(): One step of backward wave
    - lightning_forward(): DFS probe from forward frontier
    - lightning_backward(): DFS probe from boundary toward start

    CONSISTENCY: This class is designed to match fractal_holos3.py's
    BidirectionalHOLOS behavior while being game-agnostic.
    """

    def __init__(self, game: GameInterface[S, V], name: str = "holos",
                 max_memory_mb: int = 4000):
        self.game = game
        self.name = name
        self.max_memory_mb = max_memory_mb

        # State tracking
        self.forward_frontier: Dict[int, S] = {}
        self.backward_frontier: Dict[int, S] = {}
        self.forward_seen: Set[int] = set()
        self.backward_seen: Set[int] = set()
        self.solved: Dict[int, V] = {}

        # Parent tracking for path reconstruction
        self.forward_parents: Dict[int, Tuple[int, Any]] = {}
        self.backward_parents: Dict[int, Tuple[int, Any]] = {}

        # Connections and spines
        self.connections: List[Tuple[int, int, V]] = []
        self.spines: List['SpinePath'] = []  # Added for fractal_holos3 compatibility

        # Mode selection
        self.mode_selector = ModeSelector()

        # Equivalence tracking
        self.equiv_classes: Dict[Any, Set[int]] = defaultdict(set)
        self.equiv_outcomes: Dict[Any, Optional[V]] = {}

        # Stats (matching fractal_holos3.py)
        self.stats = {
            'lightning_probes': 0,
            'connections': 0,
            'crystallized': 0,
            'spines_found': 0,
            'forward_expanded': 0,
            'backward_expanded': 0,
            'equiv_shortcuts': 0,
            'equiv_tracked': 0,
            'equiv_propagated': 0,
            'minimax_solved': 0,
        }

    def memory_mb(self) -> float:
        """Get current memory usage in MB (matches fractal_holos3.py)"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except:
            # Fallback estimate
            return (len(self.forward_frontier) + len(self.backward_frontier)) * 300 / (1024 * 1024)

    def solve(self, forward_seeds: List[SeedPoint[S]],
              backward_seeds: List[SeedPoint[S]] = None,
              max_iterations: int = 100,
              lightning_interval: int = 5) -> 'Hologram':
        """
        Main solve method (matches fractal_holos3.py signature).

        Args:
            forward_seeds: Starting positions to solve
            backward_seeds: Boundary positions to expand from (auto-generated if None)
            max_iterations: Maximum expansion iterations
            lightning_interval: Run lightning probes every N iterations

        Returns:
            Hologram with solved states and spines
        """
        from .storage import Hologram, SpinePath

        print(f"\n{'='*60}")
        print(f"HOLOS Solver: {self.name}")
        print(f"{'='*60}")

        # Initialize forward frontier
        for seed in forward_seeds:
            h = self.game.hash_state(seed.state)
            if h not in self.forward_seen:
                self.forward_seen.add(h)
                self.forward_frontier[h] = seed.state

        # Auto-generate backward seeds if not provided (like fractal_holos3)
        if backward_seeds is None and forward_seeds:
            generated = self.game.generate_boundary_seeds(forward_seeds[0].state, count=100)
            backward_seeds = [SeedPoint(s, SearchMode.WAVE) for s in generated]
            print(f"Auto-generated {len(backward_seeds)} backward seeds")

        # Initialize backward frontier
        seeded_backward = 0
        if backward_seeds:
            for seed in backward_seeds:
                h = self.game.hash_state(seed.state)
                if h not in self.backward_seen:
                    self.backward_seen.add(h)
                    self.backward_frontier[h] = seed.state
                    # Get boundary value
                    if self.game.is_boundary(seed.state):
                        value = self.game.get_boundary_value(seed.state)
                        if value is not None:
                            self.solved[h] = value
                            # Track features
                            features = self.game.get_features(seed.state)
                            if features is not None:
                                self.equiv_classes[features].add(h)
                                self._update_equiv_outcome(features, value)
                            seeded_backward += 1

        print(f"Seeded {seeded_backward} backward positions with boundary values")
        print(f"Forward frontier: {len(self.forward_frontier):,}")
        print(f"Backward frontier: {len(self.backward_frontier):,}")

        start_time = time.time()

        for iteration in range(max_iterations):
            mem = self.memory_mb()

            # Memory check (like fractal_holos3)
            if mem > self.max_memory_mb * 0.9:
                print(f"\nMemory limit reached ({mem:.0f} MB)")
                break

            if not self.forward_frontier and not self.backward_frontier:
                print("\nBoth frontiers empty!")
                break

            print(f"\n--- Iteration {iteration} ---")
            print(f"  Forward: {len(self.forward_frontier):,}, Backward: {len(self.backward_frontier):,}")
            print(f"  Solved: {len(self.solved):,}, Memory: {mem:.0f} MB")

            # Lightning probes every N iterations (like fractal_holos3)
            if iteration % lightning_interval == 0:
                self._lightning_phase()

            # Expand both directions
            fwd_contacts = self._expand_forward()
            bwd_contacts = self._expand_backward()

            # Check for connections
            new_conns = self._find_connections()
            if new_conns:
                print(f"  ** {new_conns} NEW CONNECTIONS! Crystallizing... **")
                self._crystallize()

            # Propagate values
            propagated = self._propagate()

            elapsed = time.time() - start_time
            total_seen = len(self.forward_seen) + len(self.backward_seen)
            rate = total_seen / elapsed if elapsed > 0 else 0
            equiv_count = sum(len(v) for v in self.equiv_classes.values())

            print(f"  Contacts: {fwd_contacts}+{bwd_contacts}, Propagated: {propagated}")
            print(f"  Rate: {rate:.0f} pos/s, Connections: {len(self.connections)}")
            print(f"  Equivalence: {equiv_count:,} tracked, {self.stats['equiv_shortcuts']:,} shortcuts, {self.stats['equiv_propagated']:,} propagated")

        # Build hologram
        hologram = Hologram(self.name)
        hologram.solved = dict(self.solved)
        hologram.connections = list(self.connections)
        hologram.spines = list(self.spines)
        hologram.equiv_classes = dict(self.equiv_classes)
        hologram.equiv_outcomes = dict(self.equiv_outcomes)
        hologram.stats = dict(self.stats)

        print(f"\n{'='*60}")
        print(f"COMPLETE: {len(self.solved):,} solved")
        print(f"  Spines: {len(self.spines):,}")
        print(f"  Connections: {len(self.connections):,}")
        print(f"{'='*60}")

        return hologram

    def _update_equiv_outcome(self, features: Any, value: V):
        """Track outcome for equivalence class (None if inconsistent)"""
        if features in self.equiv_outcomes:
            if self.equiv_outcomes[features] != value:
                self.equiv_outcomes[features] = None  # Inconsistent
        else:
            self.equiv_outcomes[features] = value

    def _lightning_phase(self):
        """Run lightning probes from both frontiers (matches fractal_holos3)"""
        from .storage import SpinePath

        # Forward lightning (like fractal_holos3)
        if self.forward_frontier:
            sample_size = min(10, len(self.forward_frontier))
            samples = random.sample(list(self.forward_frontier.values()), sample_size)

            spines_found = 0
            total_nodes = 0

            for state in samples:
                probe = LightningProbe(
                    self.game, self.solved,
                    direction="forward",
                    max_depth=20
                )
                value, path = probe.probe(state)
                self.stats['lightning_probes'] += 1
                total_nodes += probe.nodes_visited

                if value is not None and path:
                    # Create spine (like fractal_holos3)
                    h = self.game.hash_state(state)
                    moves = [m for s, m in path]

                    # Calculate end state
                    end_state = state
                    for s, m in path:
                        next_state = self.game.apply_move(s, m)
                        if next_state is not None:
                            end_state = next_state

                    spine = SpinePath(
                        start_hash=h,
                        moves=moves,
                        end_hash=self.game.hash_state(end_state),
                        end_value=value,
                        depth=len(moves)
                    )
                    self.spines.append(spine)
                    spines_found += 1

                    # Add intermediate positions to solved (like fractal_holos3)
                    self.solved[h] = value
                    for s, m in path:
                        sh = self.game.hash_state(s)
                        if sh not in self.solved:
                            self.solved[sh] = value

            if spines_found:
                print(f"  Lightning (fwd): {spines_found} spines found ({total_nodes} nodes)")
                self.stats['spines_found'] += spines_found

        # BACKWARD LIGHTNING - DFS from boundary toward start (ENHANCEMENT over fractal_holos3)
        if self.backward_frontier:
            sample_size = min(5, len(self.backward_frontier))
            samples = random.sample(list(self.backward_frontier.values()), sample_size)

            for state in samples:
                probe = LightningProbe(
                    self.game, self.solved,
                    direction="backward",
                    max_depth=15
                )
                value, path = probe.probe(state)
                self.stats['lightning_probes'] += 1

                if value is not None and path:
                    h = self.game.hash_state(state)
                    self.solved[h] = value
                    for s, m in path:
                        sh = self.game.hash_state(s)
                        if sh not in self.solved:
                            self.solved[sh] = value

    def _expand_forward(self) -> int:
        """Expand forward frontier by one layer (matches fractal_holos3)"""
        if not self.forward_frontier:
            return 0

        items = list(self.forward_frontier.items())
        next_frontier = {}
        contacts = 0
        equiv_added = 0

        for h, state in items:
            # Track equivalence (like fractal_holos3)
            features = self.game.get_features(state)
            if features is not None:
                self.equiv_classes[features].add(h)
                equiv_added += 1

            # Check terminal
            is_term, value = self.game.is_terminal(state)
            if is_term:
                self.solved[h] = value
                if features is not None:
                    self._update_equiv_outcome(features, value)
                contacts += 1
                continue

            # Already solved?
            if h in self.solved:
                contacts += 1
                continue

            # Equivalence shortcut (like fractal_holos3)
            if features is not None and features in self.equiv_outcomes:
                eq_value = self.equiv_outcomes[features]
                if eq_value is not None:
                    self.solved[h] = eq_value
                    self.stats['equiv_shortcuts'] += 1
                    contacts += 1
                    continue

            # Expand children
            for child, move in self.game.get_successors(state):
                ch = self.game.hash_state(child)

                # Check boundary - seed backward wave (like fractal_holos3)
                if self.game.is_boundary(child):
                    value = self.game.get_boundary_value(child)
                    if value is not None:
                        self.solved[ch] = value
                        child_features = self.game.get_features(child)
                        if child_features is not None:
                            self.equiv_classes[child_features].add(ch)
                            self._update_equiv_outcome(child_features, value)
                        contacts += 1

                        # Seed backward wave from this boundary position!
                        if ch not in self.backward_seen:
                            self.backward_seen.add(ch)
                            self.backward_frontier[ch] = child
                        continue

                # Check if backward wave already explored this
                if ch in self.backward_seen:
                    if ch in self.solved:
                        contacts += 1
                        continue

                if ch not in self.forward_seen:
                    self.forward_seen.add(ch)
                    next_frontier[ch] = child
                    self.forward_parents[ch] = (h, move)

        self.forward_frontier = next_frontier
        self.stats['forward_expanded'] += len(items)

        if equiv_added > 0:
            self.stats['equiv_tracked'] += equiv_added

        return contacts

    def _expand_backward(self) -> int:
        """Expand backward frontier using predecessors (matches fractal_holos3)"""
        if not self.backward_frontier:
            return 0

        items = list(self.backward_frontier.items())
        next_frontier = {}
        contacts = 0
        equiv_added = 0

        # Track pred -> children for minimax (like fractal_holos3)
        pred_children: Dict[int, List[Tuple[int, V]]] = defaultdict(list)

        for h, state in items:
            # Generate predecessors
            for pred, move in self.game.get_predecessors(state):
                ph = self.game.hash_state(pred)

                # Track child relationship for minimax
                if h in self.solved:
                    pred_children[ph].append((h, self.solved[h]))

                if ph in self.backward_seen:
                    continue

                self.backward_seen.add(ph)

                # Track features (like fractal_holos3)
                features = self.game.get_features(pred)
                if features is not None:
                    self.equiv_classes[features].add(ph)
                    equiv_added += 1

                # Check if forward wave reached this - CONNECTION!
                if ph in self.forward_seen:
                    self.stats['connections'] += 1
                    contacts += 1
                    continue

                next_frontier[ph] = pred
                self.backward_parents[ph] = (h, move)

        # Apply minimax to predecessors (like fractal_holos3)
        minimax_solved = 0
        for ph, children in pred_children.items():
            if ph in self.solved:
                continue

            if not children:
                continue

            pred_state = next_frontier.get(ph) or self.backward_frontier.get(ph)
            if pred_state is None:
                continue

            child_values = [v for _, v in children]
            value = self.game.propagate_value(pred_state, child_values)
            if value is not None:
                self.solved[ph] = value
                features = self.game.get_features(pred_state)
                if features is not None:
                    self._update_equiv_outcome(features, value)
                minimax_solved += 1

        contacts += minimax_solved
        self.backward_frontier = next_frontier
        self.stats['backward_expanded'] += len(items)

        if equiv_added > 0:
            self.stats['equiv_tracked'] += equiv_added
        if minimax_solved > 0:
            self.stats['minimax_solved'] += minimax_solved

        return contacts

    def _find_connections(self) -> int:
        """Find where forward and backward waves meet (matches fractal_holos3)"""
        overlap = self.forward_seen & self.backward_seen
        new_conns = 0

        for h in overlap:
            if h in self.solved:
                existing = [c for c in self.connections if c[0] == h or c[1] == h]
                if not existing:
                    self.connections.append((h, h, self.solved[h]))
                    new_conns += 1
                    self.stats['connections'] += 1

        return new_conns

    def _crystallize(self):
        """Expand around connection points (matches fractal_holos3)"""
        for fh, bh, value in self.connections[-10:]:
            state = self.forward_frontier.get(fh) or self.backward_frontier.get(fh)
            if state is None:
                continue

            # Local BFS (like fractal_holos3)
            local = {fh: state}
            local_seen = {fh}

            for _ in range(3):  # 3 layers
                next_local = {}
                for h, s in local.items():
                    for child, move in self.game.get_successors(s):
                        ch = self.game.hash_state(child)
                        if ch not in local_seen:
                            local_seen.add(ch)
                            next_local[ch] = child

                            if self.game.is_boundary(child):
                                val = self.game.get_boundary_value(child)
                                if val is not None:
                                    self.solved[ch] = val
                                    self.stats['crystallized'] += 1

                local = next_local

    def _propagate(self) -> int:
        """Propagate values through parent links and equivalence (matches fractal_holos3)"""
        total = 0

        for _ in range(50):
            newly_solved = 0

            # Forward propagation
            for ch, (ph, move) in list(self.forward_parents.items()):
                if ph in self.solved and ch not in self.solved:
                    self.solved[ch] = self.solved[ph]
                    newly_solved += 1

            # Backward propagation
            for ch, (ph, move) in list(self.backward_parents.items()):
                if ch in self.solved and ph not in self.solved:
                    self.solved[ph] = self.solved[ch]
                    newly_solved += 1

            # Equivalence propagation (like fractal_holos3)
            equiv_solved = 0
            for features, hashes in self.equiv_classes.items():
                if features not in self.equiv_outcomes:
                    for h in hashes:
                        if h in self.solved:
                            self.equiv_outcomes[features] = self.solved[h]
                            break

                if features in self.equiv_outcomes:
                    value = self.equiv_outcomes[features]
                    if value is not None:
                        for h in hashes:
                            if h not in self.solved:
                                self.solved[h] = value
                                equiv_solved += 1

            newly_solved += equiv_solved
            if equiv_solved > 0:
                self.stats['equiv_propagated'] += equiv_solved

            total += newly_solved
            if newly_solved == 0:
                break

        return total
