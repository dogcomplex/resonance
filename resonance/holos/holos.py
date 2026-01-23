"""
holos/holos.py - Universal HOLOS Algorithm (THE ENGINE)

This module contains the game-agnostic HOLOS solver. It implements:
- GameInterface: Abstract base class for any game/lattice
- SearchMode: Lightning, Wave, Crystal modes
- LightningProbe: Fast DFS to find paths (forward AND backward)
- HOLOSSolver: Main solver with bidirectional search

The key insight: Forward and Backward are MIRRORS.
- Forward lightning: DFS from start toward boundary
- Backward lightning: DFS from boundary toward start (using predecessors)

Both use the same algorithm, just with different direction functions.

This is THE ENGINE that all layers use:
- Layer 0 (chess.py): HOLOSSolver(ChessGame) searches chess positions
- Layer 1 (seeds.py): HOLOSSolver(SeedGame) searches seed configurations
- Layer 2 (strategy.py): HOLOSSolver(StrategyGame) searches goal allocations

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


@dataclass
class GoalCondition:
    """
    Defines what counts as 'success' for a targeted search.

    This is a Layer 1/2 concept - the STRATEGY of what to search for.
    Layer 0 (game) provides the capabilities, Layer 1/2 decides the goals.

    Attributes:
        target_signatures: Set of state signatures that count as goal states.
                          For chess, this might be material strings like {'KQRRvKQR'}.
        early_terminate_misses: If True, stop expanding paths that can't reach goals.
        name: Human-readable name for this goal condition.

    The game must implement get_signature(state) for this to work.
    """
    target_signatures: Set[str]
    early_terminate_misses: bool = True
    name: str = "unnamed_goal"

    def matches(self, signature: str) -> bool:
        """Check if a state signature matches this goal"""
        return signature in self.target_signatures

    def __hash__(self):
        return hash((frozenset(self.target_signatures), self.early_terminate_misses, self.name))


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

    def get_signature(self, state: S) -> Optional[str]:
        """
        Get a signature for goal matching.

        For chess, this returns the material string like 'KQRRvKQR'.
        For other games, it could be any string that identifies a state category.

        Returns None if signatures are not supported.

        This is used by Layer 1/2 goal conditions to filter states.
        """
        return None  # Default: no signature support

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
# NOTE: ModeDecision and ModeSelector moved to games/seeds.py (Layer 1)
# They are Layer 1 concerns - tactical decisions about HOW to search.
# ============================================================


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
                 max_memory_mb: int = 4000,
                 max_frontier_size: int = 2_000_000,
                 max_backward_depth: int = None):
        """
        Initialize HOLOS solver with memory management.

        Args:
            game: Game interface to solve
            name: Solver name for logging
            max_memory_mb: Memory limit in MB (process will stop if exceeded)
            max_frontier_size: Hard cap on frontier size (prevents memory explosion)
            max_backward_depth: Limit backward expansion depth (None = unlimited)
                               For targeted search, use 1-2 to prevent explosion.
        """
        self.game = game
        self.name = name
        self.max_memory_mb = max_memory_mb
        self.max_frontier_size = max_frontier_size
        self.max_backward_depth = max_backward_depth

        # State tracking
        self.forward_frontier: Dict[int, S] = {}
        self.backward_frontier: Dict[int, S] = {}
        self.forward_seen: Set[int] = set()
        self.backward_seen: Set[int] = set()
        self.solved: Dict[int, V] = {}

        # Depth tracking for backward expansion
        self.backward_depth: Dict[int, int] = {}  # hash -> depth from boundary

        # Parent tracking for path reconstruction
        self.forward_parents: Dict[int, Tuple[int, Any]] = {}
        self.backward_parents: Dict[int, Tuple[int, Any]] = {}

        # Connections and spines
        self.connections: List[Tuple[int, int, V]] = []
        self.spines: List['SpinePath'] = []  # Added for fractal_holos3 compatibility

        # Mode selection moved to Layer 1 (games/seeds.py)
        # Use ModeSelector from there if needed

        # Equivalence tracking (with size limit per class)
        self.equiv_classes: Dict[Any, Set[int]] = defaultdict(set)
        self.equiv_outcomes: Dict[Any, Optional[V]] = {}
        self.max_equiv_class_size = 10_000  # Prevent memory explosion

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
            'goal_filtered': 0,  # States filtered by goal condition
            'frontier_capped': 0,  # Times frontier hit size cap
            'depth_limited': 0,  # Positions skipped due to depth limit
        }

        # Goal condition for targeted search (set during solve())
        self.current_goal: Optional[GoalCondition] = None

    def memory_mb(self) -> float:
        """Get current memory usage in MB (matches fractal_holos3.py)"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except:
            # Fallback estimate
            return (len(self.forward_frontier) + len(self.backward_frontier)) * 300 / (1024 * 1024)

    def reset(self):
        """
        Reset solver state for a fresh start.

        Use this when you want to solve a new independent problem
        without creating a new solver instance. For incremental solving
        across related problems, don't call this - solver state accumulation
        is intentional.
        """
        self.forward_frontier.clear()
        self.backward_frontier.clear()
        self.forward_seen.clear()
        self.backward_seen.clear()
        self.solved.clear()
        self.backward_depth.clear()
        self.forward_parents.clear()
        self.backward_parents.clear()
        self.connections.clear()
        self.spines.clear()
        self.equiv_classes.clear()
        self.equiv_outcomes.clear()
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
            'goal_filtered': 0,
            'frontier_capped': 0,
            'depth_limited': 0,
        }

    def solve(self, forward_seeds: List[SeedPoint[S]],
              backward_seeds: List[SeedPoint[S]] = None,
              max_iterations: int = 100,
              lightning_interval: int = 5,
              goal: GoalCondition = None) -> 'Hologram':
        """
        Main solve method (matches fractal_holos3.py signature).

        Optional goal parameter enables targeted search:
        - Only states matching goal.target_signatures count as boundaries
        - If goal.early_terminate_misses, filter out paths to non-goal states

        Args:
            forward_seeds: Starting positions to solve
            backward_seeds: Boundary positions to expand from (auto-generated if None)
            max_iterations: Maximum expansion iterations
            lightning_interval: Run lightning probes every N iterations

        Returns:
            Hologram with solved states and spines
        """
        from .storage import Hologram, SpinePath

        # Store goal for use in expansion methods
        self.current_goal = goal

        print(f"\n{'='*60}")
        print(f"HOLOS Solver: {self.name}")
        if goal:
            print(f"Goal: {goal.name} ({len(goal.target_signatures)} targets)")
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
                    self.backward_depth[h] = 0  # Boundary seeds are depth 0
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
        if self.max_backward_depth is not None:
            print(f"Backward depth limit: {self.max_backward_depth}")

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

    def _expand_forward(self, max_frontier_size: int = 500000) -> int:
        """Expand forward frontier by one layer (matches fractal_holos3)"""
        if not self.forward_frontier:
            return 0

        # Sample if frontier is too large to prevent memory explosion
        items = list(self.forward_frontier.items())
        if len(items) > max_frontier_size:
            items = random.sample(items, max_frontier_size)

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
                    # Goal filtering: if we have a goal, check if boundary matches
                    if self.current_goal is not None:
                        sig = self.game.get_signature(child)
                        if sig is not None and not self.current_goal.matches(sig):
                            # Boundary doesn't match goal - early terminate this path
                            self.stats['goal_filtered'] += 1
                            if self.current_goal.early_terminate_misses:
                                continue  # Don't add to frontier or solved

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

    def _expand_backward(self, max_input_size: int = 500_000) -> int:
        """
        Expand backward frontier using predecessors.

        Memory safety features:
        1. Input sampling: If frontier too large, sample
        2. Output cap: Hard limit on next_frontier size
        3. Depth limiting: Stop at max_backward_depth
        4. Equiv class cap: Don't track huge equivalence classes
        """
        if not self.backward_frontier:
            return 0

        # Sample if frontier is too large to prevent memory explosion
        items = list(self.backward_frontier.items())
        if len(items) > max_input_size:
            items = random.sample(items, max_input_size)

        next_frontier = {}
        contacts = 0
        equiv_added = 0
        frontier_capped = False
        depth_limited_count = 0

        # Track pred -> children for minimax (like fractal_holos3)
        pred_children: Dict[int, List[Tuple[int, V]]] = defaultdict(list)

        for h, state in items:
            # Check depth limit - get current depth of this state
            current_depth = self.backward_depth.get(h, 0)

            # If we've reached depth limit, don't expand further (only if limit is set)
            if self.max_backward_depth is not None and current_depth >= self.max_backward_depth:
                depth_limited_count += 1
                continue

            # Hard cap on output frontier size (only if limit is set)
            if self.max_frontier_size is not None and len(next_frontier) >= self.max_frontier_size:
                if not frontier_capped:
                    frontier_capped = True
                    self.stats['frontier_capped'] += 1
                continue  # Stop adding, but continue processing for minimax

            # Generate predecessors
            for pred, move in self.game.get_predecessors(state):
                ph = self.game.hash_state(pred)

                # Track child relationship for minimax
                if h in self.solved:
                    pred_children[ph].append((h, self.solved[h]))

                if ph in self.backward_seen:
                    continue

                self.backward_seen.add(ph)

                # Track depth of predecessor (one more than current)
                self.backward_depth[ph] = current_depth + 1

                # Track features with size limit
                features = self.game.get_features(pred)
                if features is not None:
                    if len(self.equiv_classes[features]) < self.max_equiv_class_size:
                        self.equiv_classes[features].add(ph)
                        equiv_added += 1

                # Check if forward wave reached this - CONNECTION!
                if ph in self.forward_seen:
                    self.stats['connections'] += 1
                    contacts += 1
                    continue

                # Only add to frontier if we haven't hit the cap (or no cap set)
                if self.max_frontier_size is None or len(next_frontier) < self.max_frontier_size:
                    next_frontier[ph] = pred
                    self.backward_parents[ph] = (h, move)

        if depth_limited_count > 0:
            self.stats['depth_limited'] += depth_limited_count

        if frontier_capped and self.max_frontier_size is not None:
            print(f"  Frontier capped at {self.max_frontier_size:,}")

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

        # Memory cleanup: clear parent tracking when solved count is very large
        # Parent tracking is only needed for spine reconstruction
        if len(self.solved) > 5_000_000:
            cleared_forward = len(self.forward_parents)
            cleared_backward = len(self.backward_parents)
            self.forward_parents.clear()
            self.backward_parents.clear()
            print(f"  Cleared {cleared_forward + cleared_backward:,} parent links to save memory")

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
