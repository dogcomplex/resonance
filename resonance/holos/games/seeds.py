"""
holos/games/seeds.py - Seed Selection Meta-Game (Layer 1)

This implements GameInterface for the META-GAME of choosing seeds.

The game:
- State: A configuration of seeds (which positions, what mode, what depth)
- Moves: Add/remove/modify seeds
- Value: Efficiency score (coverage / cost)
- Boundary: Known-optimal configurations

This is the first meta-level: HOLOS searching for how to run HOLOS.

Key Discovery from experiments:
- DEPTH is the dominant variable (not seed selection)
- 1 seed @ depth 5: 1655 efficiency (coverage/cost)
- 20 seeds @ depth 2: 191 efficiency
- This is a ~10x difference!

The meta-game searches the LATTICE of configurations:
- Moving "up": Adding seeds (more coverage, more cost)
- Moving "down": Removing seeds (less coverage, less cost)
- Moving "sideways": Changing mode/depth

The key insight: We're optimizing EFFICIENCY, not just coverage.
"""

import random
from typing import List, Tuple, Optional, Any, Set, FrozenSet, Dict
from dataclasses import dataclass, field
from collections import defaultdict

from holos.core import GameInterface, SearchMode


# ============================================================
# SEED CONFIGURATION STATE
# ============================================================

@dataclass(frozen=True)
class SeedSpec:
    """Specification for a single seed"""
    position_hash: int      # Hash of the boundary position
    mode: SearchMode        # Lightning, Wave, or Crystal
    depth: int              # How far to expand from this seed

    def __hash__(self):
        return hash((self.position_hash, self.mode, self.depth))


@dataclass(frozen=True)
class SeedConfiguration:
    """
    A complete seed configuration - the STATE in the meta-game.

    This is a point in the lattice of all possible seed configurations.
    """
    seeds: FrozenSet[SeedSpec]
    material: str  # What material config (e.g., "KQRRvKQR")

    _hash: int = field(default=None, hash=False, compare=False)

    def __hash__(self):
        if self._hash is None:
            object.__setattr__(self, '_hash', hash((self.seeds, self.material)))
        return self._hash

    def __len__(self):
        return len(self.seeds)

    def total_cost(self) -> int:
        """Total compute cost = sum of depths"""
        return sum(s.depth for s in self.seeds)

    def signature(self) -> str:
        modes = defaultdict(int)
        for s in self.seeds:
            modes[s.mode.value] += 1
        return f"Config({len(self.seeds)} seeds, cost={self.total_cost()})"


# ============================================================
# SEED VALUE (Efficiency Metrics)
# ============================================================

@dataclass(frozen=True)
class SeedValue:
    """Value of a seed configuration: efficiency metrics"""
    coverage: int           # Number of positions covered
    cost: int               # Total depth cost
    efficiency: float       # coverage / cost

    def __repr__(self):
        return f"Value(cov={self.coverage}, cost={self.cost}, eff={self.efficiency:.1f})"

    def __lt__(self, other):
        return self.efficiency < other.efficiency

    def __eq__(self, other):
        if not isinstance(other, SeedValue):
            return False
        return self.efficiency == other.efficiency


# ============================================================
# SEED GAME INTERFACE
# ============================================================

class SeedGame(GameInterface[SeedConfiguration, SeedValue]):
    """
    The meta-game of seed selection.

    This is a LATTICE game:
    - Moving "up" = adding seeds (more coverage, more cost)
    - Moving "down" = removing seeds (less coverage, less cost)
    - Moving "sideways" = swapping seeds, changing modes/depths

    Boundaries:
    - Empty configuration: coverage=0, cost=0 (known)
    - Full coverage configurations (empirically measured)

    The game searches for EFFICIENT configurations.
    """

    def __init__(self, chess_game=None, material: str = "KQRRvKQR",
                 target_coverage: float = 0.9):
        self.chess_game = chess_game  # The Layer 0 game
        self.material = material
        self.target_coverage = target_coverage

        # Cache of evaluated configurations
        self.eval_cache: Dict[int, SeedValue] = {}

        # Pool of candidate seed positions
        self.seed_pool: List[Tuple[int, Any]] = []  # (hash, state)

    def set_seed_pool(self, pool: List[Tuple[int, Any]]):
        """Set the pool of candidate seeds"""
        self.seed_pool = pool

    def hash_state(self, state: SeedConfiguration) -> int:
        return hash(state)

    def get_successors(self, state: SeedConfiguration) -> List[Tuple[SeedConfiguration, Any]]:
        """
        Generate successor configurations (larger/more expensive).

        Moves:
        - Add a seed from pool
        - Increase depth of existing seed
        - Change mode of existing seed
        """
        successors = []
        seeds = set(state.seeds)
        current_hashes = {s.position_hash for s in seeds}

        # ADD: Add a new seed from pool
        for ph, pos in self.seed_pool[:50]:  # Limit branching
            if ph not in current_hashes:
                for mode in [SearchMode.WAVE, SearchMode.LIGHTNING]:
                    for depth in [1, 2, 3]:
                        new_seed = SeedSpec(ph, mode, depth)
                        new_config = SeedConfiguration(
                            frozenset(seeds | {new_seed}),
                            state.material
                        )
                        successors.append((new_config, ('add', new_seed)))

        # INCREASE DEPTH
        for seed in seeds:
            if seed.depth < 5:
                new_seed = SeedSpec(seed.position_hash, seed.mode, seed.depth + 1)
                new_seeds = (seeds - {seed}) | {new_seed}
                new_config = SeedConfiguration(frozenset(new_seeds), state.material)
                successors.append((new_config, ('deepen', seed)))

        # CHANGE MODE
        for seed in seeds:
            for new_mode in SearchMode:
                if new_mode != seed.mode:
                    new_seed = SeedSpec(seed.position_hash, new_mode, seed.depth)
                    new_seeds = (seeds - {seed}) | {new_seed}
                    new_config = SeedConfiguration(frozenset(new_seeds), state.material)
                    successors.append((new_config, ('mode', seed, new_mode)))

        return successors[:100]  # Cap total

    def get_predecessors(self, state: SeedConfiguration) -> List[Tuple[SeedConfiguration, Any]]:
        """
        Generate predecessor configurations (smaller/cheaper).

        Moves:
        - Remove a seed
        - Decrease depth
        """
        predecessors = []
        seeds = set(state.seeds)

        # REMOVE
        for seed in seeds:
            if len(seeds) > 1:
                new_config = SeedConfiguration(
                    frozenset(seeds - {seed}),
                    state.material
                )
                predecessors.append((new_config, ('remove', seed)))

        # DECREASE DEPTH
        for seed in seeds:
            if seed.depth > 1:
                new_seed = SeedSpec(seed.position_hash, seed.mode, seed.depth - 1)
                new_seeds = (seeds - {seed}) | {new_seed}
                new_config = SeedConfiguration(frozenset(new_seeds), state.material)
                predecessors.append((new_config, ('shallow', seed)))

        return predecessors

    def is_boundary(self, state: SeedConfiguration) -> bool:
        """Boundary: empty or previously evaluated"""
        if len(state.seeds) == 0:
            return True
        return hash(state) in self.eval_cache

    def get_boundary_value(self, state: SeedConfiguration) -> Optional[SeedValue]:
        """Get cached value or compute for boundary"""
        h = hash(state)

        if h in self.eval_cache:
            return self.eval_cache[h]

        if len(state.seeds) == 0:
            return SeedValue(0, 0, 0.0)

        return None

    def is_terminal(self, state: SeedConfiguration) -> Tuple[bool, Optional[SeedValue]]:
        """Terminal if full coverage achieved"""
        return False, None

    def propagate_value(self, state: SeedConfiguration,
                        child_values: List[SeedValue]) -> Optional[SeedValue]:
        """
        Value propagation in the lattice.

        Unlike minimax, we're optimizing efficiency.
        A configuration's potential is bounded by its children.
        """
        if not child_values:
            return None

        # Best efficiency among children
        best_child = max(child_values, key=lambda v: v.efficiency)
        return best_child

    def get_features(self, state: SeedConfiguration) -> Any:
        """Equivalence features"""
        num_seeds = len(state.seeds)
        total_cost = state.total_cost()
        modes = tuple(sorted(s.mode.value for s in state.seeds))
        return (num_seeds, total_cost, modes)

    def evaluate(self, state: SeedConfiguration,
                 expand_fn=None) -> SeedValue:
        """
        Actually evaluate a configuration by measuring coverage.

        This is EXPENSIVE - runs the lower layer.

        Args:
            expand_fn: Function to expand seeds into coverage set
                       (seed_hash, depth, mode) -> Set[int]
        """
        h = hash(state)
        if h in self.eval_cache:
            return self.eval_cache[h]

        if len(state.seeds) == 0:
            value = SeedValue(0, 0, 0.0)
            self.eval_cache[h] = value
            return value

        # Compute coverage
        covered = set()
        total_cost = 0

        for seed_spec in state.seeds:
            if expand_fn:
                seed_covered = expand_fn(
                    seed_spec.position_hash,
                    seed_spec.depth,
                    seed_spec.mode
                )
                covered |= seed_covered
            total_cost += seed_spec.depth

        coverage = len(covered)
        efficiency = coverage / total_cost if total_cost > 0 else 0.0

        value = SeedValue(coverage, total_cost, efficiency)
        self.eval_cache[h] = value
        return value

    # Lightning-specific methods
    def get_lightning_successors(self, state: SeedConfiguration) -> List[Tuple[SeedConfiguration, Any]]:
        """For lightning: only depth increases (move toward coverage)"""
        successors = []
        seeds = set(state.seeds)

        for seed in seeds:
            if seed.depth < 5:
                new_seed = SeedSpec(seed.position_hash, seed.mode, seed.depth + 1)
                new_seeds = (seeds - {seed}) | {new_seed}
                new_config = SeedConfiguration(frozenset(new_seeds), state.material)
                successors.append((new_config, ('deepen', seed)))

        return successors

    def get_lightning_predecessors(self, state: SeedConfiguration) -> List[Tuple[SeedConfiguration, Any]]:
        """For backward lightning: add seeds (move toward complex configs)"""
        successors = []
        seeds = set(state.seeds)
        current_hashes = {s.position_hash for s in seeds}

        for ph, pos in self.seed_pool[:20]:
            if ph not in current_hashes:
                new_seed = SeedSpec(ph, SearchMode.WAVE, 2)
                new_config = SeedConfiguration(
                    frozenset(seeds | {new_seed}),
                    state.material
                )
                successors.append((new_config, ('add', new_seed)))

        return successors

    def score_for_lightning(self, state: SeedConfiguration, move: Any) -> float:
        """Score moves for lightning prioritization"""
        if move is None:
            return 0.0

        action = move[0]
        if action == 'deepen':
            return 10.0  # Prioritize depth increases
        elif action == 'add':
            return 5.0
        return 1.0


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_seed_solver(chess_game=None, material: str = "KQRRvKQR"):
    """Create a HOLOS solver for the seed selection game"""
    from holos.core import HOLOSSolver

    game = SeedGame(chess_game, material)
    solver = HOLOSSolver(game, name=f"seeds_{material}")
    return solver, game


def create_initial_configs(game: SeedGame, num_configs: int = 20) -> List[SeedConfiguration]:
    """Generate diverse initial configurations"""
    configs = []

    # Empty config (boundary)
    configs.append(SeedConfiguration(frozenset(), game.material))

    # Single seed configs at various depths
    for ph, pos in game.seed_pool[:10]:
        for depth in [1, 2, 3, 5]:
            seed = SeedSpec(ph, SearchMode.WAVE, depth)
            config = SeedConfiguration(frozenset([seed]), game.material)
            configs.append(config)

    # Multi-seed configs
    for num_seeds in [3, 5, 10]:
        seeds = set()
        for ph, pos in random.sample(game.seed_pool, min(num_seeds, len(game.seed_pool))):
            seeds.add(SeedSpec(ph, SearchMode.WAVE, 2))
        config = SeedConfiguration(frozenset(seeds), game.material)
        configs.append(config)

    return configs[:num_configs]
