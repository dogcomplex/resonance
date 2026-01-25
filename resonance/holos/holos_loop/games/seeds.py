"""
holos/games/seeds.py - Seed Tactics Meta-Game (Layer 1)

LAYER 1: TACTICS - Optimizing a SINGLE seed's parameters

This implements GameInterface for the tactical optimization of individual seeds.
Layer 2 (Strategy) handles coordination of MULTIPLE seeds.

The game:
- State: A single seed specification (position, depth, mode, direction)
- Moves: Adjust depth, mode, or direction
- Value: TacticalValue or CompressionAwareSeedValue
- Boundary: Evaluated seeds (cached results from Layer 0)

KEY INSIGHT: A seed's value depends on DUAL coverage:
- Forward coverage: positions reachable BY expanding from this seed
- Backward coverage: positions that CAN REACH this seed
Both matter for bidirectional search effectiveness.

COMPRESSION INSIGHT (Update 16):
- Seeds must compress better than direct position storage
- net_savings = direct_storage - seed_storage
- If net_savings <= 0, the seed has NEGATIVE efficiency
- Use CompressionAwareSeedValue for compression-aware optimization
- Index-based encoding compresses better than object storage

The tactical game optimizes a seed's parameters to maximize efficiency.
The strategic game (Layer 2) selects WHICH seeds to use and their ordering.

Architecture:
    Layer 2 (Strategy): Multi-seed coordination, ordering, budget allocation
        | selects seeds
    Layer 1 (Tactics): Single seed optimization (THIS FILE)
        | configures seeds
    Layer 0 (Chess): Position-level search
        | queries
    Boundary: Syzygy Tablebases
"""

import time
from typing import List, Tuple, Optional, Any, Set, Dict, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum

from holos.holos import GameInterface, SearchMode

# Type variables for the underlying game
S = TypeVar('S')  # State type of underlying game
V = TypeVar('V')  # Value type of underlying game


# ============================================================
# SEED STATE (Layer 1 State)
# ============================================================

class SeedDirection(Enum):
    """Direction of seed expansion"""
    FORWARD = "forward"    # Expand toward boundary (deduction)
    BACKWARD = "backward"  # Expand from boundary (abduction)
    BILATERAL = "bilateral"  # Both directions (most expensive but most coverage)


@dataclass(frozen=True)
class TacticalSeed:
    """
    A single seed with its tactical parameters.

    This is the STATE in Layer 1 - we're optimizing these parameters.

    Unlike the old SeedConfiguration (which was a SET of seeds),
    TacticalSeed is a SINGLE seed. Layer 2 handles multiple seeds.
    """
    position_hash: int          # Which position to seed from
    depth: int                  # How many expansion steps
    mode: SearchMode            # Lightning, Wave, Crystal, Osmosis
    direction: SeedDirection    # Forward, Backward, or Bilateral

    # Optional: position state for reconstruction
    # (Not part of hash - just for convenience)
    _state: Any = field(default=None, hash=False, compare=False)

    def cost(self) -> int:
        """
        Compute cost of this seed configuration.

        Cost model:
        - depth: linear cost (each level ~10x positions)
        - bilateral: 2x cost (both directions)
        - lightning: 0.5x cost (selective expansion)
        """
        base_cost = self.depth

        if self.direction == SeedDirection.BILATERAL:
            base_cost *= 2

        if self.mode == SearchMode.LIGHTNING:
            base_cost = max(1, base_cost // 2)

        return base_cost

    def signature(self) -> str:
        return f"Seed(d={self.depth}, {self.mode.value}, {self.direction.value})"


# ============================================================
# TACTICAL VALUE (Layer 1 Value)
# ============================================================

@dataclass(frozen=True)
class TacticalValue:
    """
    Value of a seed's tactical configuration.

    Measures DUAL coverage because effective bidirectional search
    needs both forward and backward waves to meet.

    Metrics:
    - forward_coverage: positions reachable expanding FROM this seed
    - backward_coverage: positions that can REACH this seed (predecessors)
    - overlap_potential: estimated overlap with other seeds (for Layer 2)
    - cost: computational cost of this configuration
    - efficiency: combined coverage / cost
    """
    forward_coverage: int       # Positions reachable forward
    backward_coverage: int      # Positions reachable backward
    overlap_potential: float    # How much this might overlap with neighbors
    cost: int                   # Computational cost
    efficiency: float           # (forward + backward) / cost

    # For few-shot optimization (Layer 2 will use these)
    time_to_first_solve: float = 0.0  # How quickly did we find solutions?
    solves_per_second: float = 0.0    # Rate of finding solutions

    def total_coverage(self) -> int:
        return self.forward_coverage + self.backward_coverage

    def __repr__(self):
        return (f"TacticalValue(fwd={self.forward_coverage}, bwd={self.backward_coverage}, "
                f"cost={self.cost}, eff={self.efficiency:.1f})")

    def __lt__(self, other):
        # Primary: efficiency, Secondary: total coverage
        if self.efficiency != other.efficiency:
            return self.efficiency < other.efficiency
        return self.total_coverage() < other.total_coverage()

    def __eq__(self, other):
        if not isinstance(other, TacticalValue):
            return False
        return (self.efficiency == other.efficiency and
                self.total_coverage() == other.total_coverage())


# ============================================================
# TACTICAL SEED GAME (Layer 1 Game Interface)
# ============================================================

class TacticalSeedGame(GameInterface[TacticalSeed, TacticalValue]):
    """
    The tactical game of optimizing a SINGLE seed's parameters.

    This is a game-agnostic Layer 1 implementation that can wrap
    ANY underlying Layer 0 game.

    State Space:
    - Position (from seed pool) × Depth (1-max) × Mode (4 options) × Direction (3 options)
    - Typically ~10,000 states per seed position

    Moves:
    - Forward (increase cost): increase depth, switch to bilateral
    - Backward (decrease cost): decrease depth, switch to single direction
    - Sideways: change mode

    Boundary:
    - Evaluated seeds (we've measured their actual coverage)
    - Depth 0 (no expansion - trivial)

    Value Propagation:
    - Maximize efficiency (coverage/cost)
    - Prefer configurations with balanced forward/backward coverage
    """

    def __init__(self,
                 underlying_game: GameInterface,
                 seed_pool: List[Tuple[int, Any]] = None,
                 max_depth: int = 6,
                 target_coverage: float = 0.9):
        """
        Args:
            underlying_game: The Layer 0 game (e.g., ChessGame)
            seed_pool: List of (hash, state) for candidate seed positions
            max_depth: Maximum expansion depth to consider
            target_coverage: Target coverage ratio (for terminal detection)
        """
        self.underlying_game = underlying_game
        self.seed_pool = seed_pool or []
        self.max_depth = max_depth
        self.target_coverage = target_coverage

        # Evaluation cache: hash -> TacticalValue
        self.eval_cache: Dict[int, TacticalValue] = {}

        # Statistics
        self.stats = {
            'evaluations': 0,
            'cache_hits': 0,
            'total_positions_expanded': 0,
        }

    def set_seed_pool(self, pool: List[Tuple[int, Any]]):
        """Set the pool of candidate seed positions"""
        self.seed_pool = pool

    # ==================== GameInterface Implementation ====================

    def hash_state(self, state: TacticalSeed) -> int:
        return hash(state)

    def get_successors(self, state: TacticalSeed) -> List[Tuple[TacticalSeed, Any]]:
        """
        Generate more expensive configurations (higher coverage potential).

        Forward moves in the tactical lattice:
        - Increase depth (more expansion)
        - Switch to bilateral (if not already)
        - Switch to more thorough mode (lightning → wave → crystal)
        """
        successors = []

        # INCREASE DEPTH
        if state.depth < self.max_depth:
            new_state = TacticalSeed(
                state.position_hash,
                state.depth + 1,
                state.mode,
                state.direction,
                state._state
            )
            successors.append((new_state, ('deepen', state.depth + 1)))

        # SWITCH TO BILATERAL
        if state.direction != SeedDirection.BILATERAL:
            new_state = TacticalSeed(
                state.position_hash,
                state.depth,
                state.mode,
                SeedDirection.BILATERAL,
                state._state
            )
            successors.append((new_state, ('bilateral',)))

        # UPGRADE MODE (lightning → wave → crystal)
        mode_order = [SearchMode.LIGHTNING, SearchMode.WAVE, SearchMode.CRYSTAL]
        if state.mode in mode_order:
            idx = mode_order.index(state.mode)
            if idx < len(mode_order) - 1:
                new_mode = mode_order[idx + 1]
                new_state = TacticalSeed(
                    state.position_hash,
                    state.depth,
                    new_mode,
                    state.direction,
                    state._state
                )
                successors.append((new_state, ('upgrade_mode', new_mode)))

        return successors

    def get_predecessors(self, state: TacticalSeed) -> List[Tuple[TacticalSeed, Any]]:
        """
        Generate cheaper configurations (lower cost).

        Backward moves in the tactical lattice:
        - Decrease depth
        - Switch from bilateral to single direction
        - Downgrade mode
        """
        predecessors = []

        # DECREASE DEPTH
        if state.depth > 1:
            new_state = TacticalSeed(
                state.position_hash,
                state.depth - 1,
                state.mode,
                state.direction,
                state._state
            )
            predecessors.append((new_state, ('shallow', state.depth - 1)))

        # SWITCH FROM BILATERAL
        if state.direction == SeedDirection.BILATERAL:
            for direction in [SeedDirection.FORWARD, SeedDirection.BACKWARD]:
                new_state = TacticalSeed(
                    state.position_hash,
                    state.depth,
                    state.mode,
                    direction,
                    state._state
                )
                predecessors.append((new_state, ('single_direction', direction)))

        # DOWNGRADE MODE
        mode_order = [SearchMode.LIGHTNING, SearchMode.WAVE, SearchMode.CRYSTAL]
        if state.mode in mode_order:
            idx = mode_order.index(state.mode)
            if idx > 0:
                new_mode = mode_order[idx - 1]
                new_state = TacticalSeed(
                    state.position_hash,
                    state.depth,
                    new_mode,
                    state.direction,
                    state._state
                )
                predecessors.append((new_state, ('downgrade_mode', new_mode)))

        return predecessors

    def is_boundary(self, state: TacticalSeed) -> bool:
        """
        Boundary conditions:
        - Depth 0 (trivial - no expansion)
        - Already evaluated (cached)
        """
        if state.depth == 0:
            return True
        return hash(state) in self.eval_cache

    def get_boundary_value(self, state: TacticalSeed) -> Optional[TacticalValue]:
        """Get value for boundary state"""
        h = hash(state)

        if h in self.eval_cache:
            self.stats['cache_hits'] += 1
            return self.eval_cache[h]

        if state.depth == 0:
            return TacticalValue(0, 0, 0.0, 0, 0.0)

        return None

    def is_terminal(self, state: TacticalSeed) -> Tuple[bool, Optional[TacticalValue]]:
        """Terminal when we've achieved target coverage (optional)"""
        # For now, no terminal - let the search explore fully
        return False, None

    def propagate_value(self, state: TacticalSeed,
                        child_values: List[TacticalValue]) -> Optional[TacticalValue]:
        """
        Propagate value from children (more expensive configs) to parents.

        Key insight: A cheaper config's POTENTIAL efficiency is bounded by
        what more expensive configs actually achieve.

        We propagate the best efficiency seen, adjusted for cost difference.
        """
        if not child_values:
            return None

        # Best child by efficiency
        best = max(child_values, key=lambda v: v.efficiency)

        # The parent's potential is at least as good (it costs less)
        # But we don't know its actual coverage without evaluation
        return best

    def get_features(self, state: TacticalSeed) -> Any:
        """
        Equivalence features for the tactical seed.

        Seeds with same (depth, mode, direction) have similar behavior
        regardless of position, so we can use this for equivalence.
        """
        return (state.depth, state.mode, state.direction)

    # ==================== Lightning Methods ====================

    def get_lightning_successors(self, state: TacticalSeed) -> List[Tuple[TacticalSeed, Any]]:
        """For lightning: prioritize depth increases"""
        successors = []

        if state.depth < self.max_depth:
            new_state = TacticalSeed(
                state.position_hash,
                state.depth + 1,
                state.mode,
                state.direction,
                state._state
            )
            successors.append((new_state, ('deepen', state.depth + 1)))

        return successors

    def get_lightning_predecessors(self, state: TacticalSeed) -> List[Tuple[TacticalSeed, Any]]:
        """For backward lightning: prioritize mode changes"""
        return self.get_predecessors(state)[:5]

    def score_for_lightning(self, state: TacticalSeed, move: Any) -> float:
        """Score moves for lightning prioritization"""
        if move is None:
            return 0.0

        action = move[0]
        if action == 'deepen':
            return 10.0  # Depth increases are most impactful
        elif action == 'bilateral':
            return 5.0
        elif action == 'upgrade_mode':
            return 3.0
        return 1.0

    # ==================== Evaluation (calls Layer 0) ====================

    def evaluate(self, state: TacticalSeed, verbose: bool = False) -> TacticalValue:
        """
        Actually evaluate a seed configuration by running Layer 0 expansion.

        This is EXPENSIVE - it runs the underlying game's expansion.
        Results are cached for reuse.
        """
        h = hash(state)
        if h in self.eval_cache:
            return self.eval_cache[h]

        if state.depth == 0:
            value = TacticalValue(0, 0, 0.0, 0, 0.0)
            self.eval_cache[h] = value
            return value

        self.stats['evaluations'] += 1

        # Get the actual position state
        position_state = state._state
        if position_state is None:
            # Try to find it in seed pool
            for ph, ps in self.seed_pool:
                if ph == state.position_hash:
                    position_state = ps
                    break

        if position_state is None:
            # Can't evaluate without position
            return TacticalValue(0, 0, 0.0, state.cost(), 0.0)

        start_time = time.time()

        # Measure forward coverage
        forward_coverage = 0
        if state.direction in [SeedDirection.FORWARD, SeedDirection.BILATERAL]:
            forward_positions = self._expand_forward(position_state, state.depth, state.mode)
            forward_coverage = len(forward_positions)
            self.stats['total_positions_expanded'] += forward_coverage

        # Measure backward coverage
        backward_coverage = 0
        if state.direction in [SeedDirection.BACKWARD, SeedDirection.BILATERAL]:
            backward_positions = self._expand_backward(position_state, state.depth, state.mode)
            backward_coverage = len(backward_positions)
            self.stats['total_positions_expanded'] += backward_coverage

        elapsed = time.time() - start_time

        # Calculate efficiency
        cost = state.cost()
        total_coverage = forward_coverage + backward_coverage
        efficiency = total_coverage / cost if cost > 0 else 0.0

        # Few-shot metrics
        solves_per_second = total_coverage / elapsed if elapsed > 0 else 0.0

        value = TacticalValue(
            forward_coverage=forward_coverage,
            backward_coverage=backward_coverage,
            overlap_potential=0.0,  # Would need neighbor info for this
            cost=cost,
            efficiency=efficiency,
            time_to_first_solve=elapsed,  # Simplified
            solves_per_second=solves_per_second,
        )

        self.eval_cache[h] = value

        if verbose:
            print(f"  Evaluated {state.signature()}: {value}")

        return value

    def _expand_forward(self, state: Any, depth: int, mode: SearchMode) -> Set[int]:
        """Expand forward from state using underlying game"""
        seen = set()
        frontier = {self.underlying_game.hash_state(state): state}
        seen.add(self.underlying_game.hash_state(state))

        for d in range(depth):
            next_frontier = {}
            for h, s in frontier.items():
                # Use mode-appropriate expansion
                if mode == SearchMode.LIGHTNING:
                    successors = self.underlying_game.get_lightning_successors(s)
                else:
                    successors = self.underlying_game.get_successors(s)

                for child, move in successors:
                    ch = self.underlying_game.hash_state(child)
                    if ch not in seen:
                        seen.add(ch)
                        next_frontier[ch] = child

            frontier = next_frontier
            if not frontier:
                break

        return seen

    def _expand_backward(self, state: Any, depth: int, mode: SearchMode) -> Set[int]:
        """Expand backward from state using underlying game"""
        seen = set()
        frontier = {self.underlying_game.hash_state(state): state}
        seen.add(self.underlying_game.hash_state(state))

        for d in range(depth):
            next_frontier = {}
            for h, s in frontier.items():
                # Use mode-appropriate expansion
                if mode == SearchMode.LIGHTNING:
                    predecessors = self.underlying_game.get_lightning_predecessors(s)
                else:
                    predecessors = self.underlying_game.get_predecessors(s)

                for pred, move in predecessors:
                    ph = self.underlying_game.hash_state(pred)
                    if ph not in seen:
                        seen.add(ph)
                        next_frontier[ph] = pred

            frontier = next_frontier
            if not frontier:
                break

        return seen

    # ==================== Utility Methods ====================

    def create_seed(self, position_hash: int, position_state: Any,
                    depth: int = 3, mode: SearchMode = SearchMode.WAVE,
                    direction: SeedDirection = SeedDirection.FORWARD) -> TacticalSeed:
        """Create a TacticalSeed with the given parameters"""
        return TacticalSeed(position_hash, depth, mode, direction, position_state)

    def get_signature(self, state: TacticalSeed) -> str:
        """Get string signature for display"""
        return state.signature()

    def summary(self) -> str:
        """Get summary of evaluation statistics"""
        return (f"TacticalSeedGame:\n"
                f"  Seed pool: {len(self.seed_pool)} positions\n"
                f"  Evaluations: {self.stats['evaluations']}\n"
                f"  Cache hits: {self.stats['cache_hits']}\n"
                f"  Positions expanded: {self.stats['total_positions_expanded']:,}")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_tactical_solver(underlying_game: GameInterface,
                           seed_pool: List[Tuple[int, Any]] = None,
                           max_depth: int = 6):
    """
    Create a HOLOS solver for tactical seed optimization.

    Args:
        underlying_game: The Layer 0 game
        seed_pool: Candidate seed positions
        max_depth: Maximum depth to explore

    Returns:
        (solver, game) tuple
    """
    from holos.holos import HOLOSSolver

    game = TacticalSeedGame(underlying_game, seed_pool, max_depth)
    solver = HOLOSSolver(game, name="tactical_seeds")
    return solver, game


def optimize_single_seed(underlying_game: GameInterface,
                         seed_state: Any,
                         max_depth: int = 5,
                         verbose: bool = True) -> Tuple[TacticalSeed, TacticalValue]:
    """
    Find optimal parameters for a single seed position.

    This is a convenience function that runs the tactical optimization
    for one seed and returns the best configuration found.

    Args:
        underlying_game: The Layer 0 game
        seed_state: The position to optimize as a seed
        max_depth: Maximum depth to consider
        verbose: Print progress

    Returns:
        (best_seed, best_value) tuple
    """
    seed_hash = underlying_game.hash_state(seed_state)
    seed_pool = [(seed_hash, seed_state)]

    game = TacticalSeedGame(underlying_game, seed_pool, max_depth)

    if verbose:
        print(f"Optimizing seed: hash={seed_hash}")
        print(f"Testing depths 1-{max_depth}, modes, directions...")

    best_seed = None
    best_value = None

    # Evaluate all configurations for this single seed
    for depth in range(1, max_depth + 1):
        for mode in [SearchMode.LIGHTNING, SearchMode.WAVE]:
            for direction in [SeedDirection.FORWARD, SeedDirection.BACKWARD, SeedDirection.BILATERAL]:
                seed = TacticalSeed(seed_hash, depth, mode, direction, seed_state)
                value = game.evaluate(seed, verbose=verbose)

                if best_value is None or value.efficiency > best_value.efficiency:
                    best_seed = seed
                    best_value = value

    if verbose:
        print(f"\nBest configuration: {best_seed.signature()}")
        print(f"Value: {best_value}")
        print(game.summary())

    return best_seed, best_value


# ============================================================
# LEGACY COMPATIBILITY
# ============================================================

# Keep old names for backward compatibility
SeedSpec = TacticalSeed
SeedValue = TacticalValue
SeedConfiguration = TacticalSeed  # Single seed is now the state
SeedGame = TacticalSeedGame


# ============================================================
# MODE SELECTION (Layer 1 Responsibility)
# ============================================================

@dataclass
class ModeDecision:
    """
    Tracks mode selection decisions for meta-learning.

    The idea: mode selection (lightning vs wave vs crystal) is itself
    a decision that can be optimized. Track outcomes to learn which
    mode works best in which situations.

    This is a Layer 1 concern - tactical decisions about HOW to search.
    """
    state_hash: int
    features: Any  # State features at decision point
    mode_chosen: SearchMode
    outcome: Any = None  # Did this mode find a solution?
    nodes_used: int = 0
    path_length: int = 0
    success: bool = False


class ModeSelector:
    """
    Selects search mode based on state features and history.

    This is Layer 1's job - tactical mode selection per seed position.
    Layer 2 (strategy) decides overall budget allocation.
    """

    def __init__(self):
        from collections import defaultdict
        self.history: List[ModeDecision] = []
        self.feature_outcomes: Dict[Any, Dict[SearchMode, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def select_mode(self, state, game,
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
