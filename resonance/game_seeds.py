"""
game_seeds.py - Seed Selection Game Interface for HOLOS

This implements GameInterface for the META-GAME of choosing seeds.

The game:
- State: A configuration of seeds (which positions, what mode, what depth)
- Moves: Add/remove/modify seeds
- Value: Efficiency score from running the lower layer
- Boundary: Known-optimal configurations (full coverage, empty set, etc.)

This is the first meta-level: HOLOS searching for how to run HOLOS.
"""

from typing import List, Tuple, Optional, Any, Set, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
import random
import hashlib

from holos_core import GameInterface, SearchMode, SeedPoint

# Import from chess layer
from fractal_holos3 import (
    ChessState, SyzygyProbe, random_position, generate_predecessors
)


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

    # Cached metrics (computed lazily)
    _hash: int = field(default=None, hash=False, compare=False)

    def __hash__(self):
        if self._hash is None:
            object.__setattr__(self, '_hash',
                               hash((self.seeds, self.material)))
        return self._hash

    def __len__(self):
        return len(self.seeds)

    def total_cost(self) -> int:
        """Total compute cost = sum of depths"""
        return sum(s.depth for s in self.seeds)

    def signature(self) -> str:
        modes = defaultdict(int)
        depths = defaultdict(int)
        for s in self.seeds:
            modes[s.mode.value] += 1
            depths[s.depth] += 1
        return f"Config({len(self.seeds)} seeds, cost={self.total_cost()})"


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

    The game searches for EFFICIENT configurations (high coverage/cost ratio).
    """

    def __init__(self, syzygy_path: str = "./syzygy", material: str = "KQRRvKQR",
                 target_coverage: float = 0.9):
        self.syzygy = SyzygyProbe(syzygy_path)
        self.material = material
        self.target_coverage = target_coverage

        # Cache of evaluated configurations
        self.eval_cache: dict[int, SeedValue] = {}

        # Pool of candidate seed positions (boundary positions)
        self.seed_pool: List[Tuple[int, ChessState]] = []
        self._build_seed_pool()

        # Universe of 8-piece positions (for coverage measurement)
        self.universe: Set[int] = set()
        self._build_universe()

    def _build_seed_pool(self, pool_size: int = 200):
        """Build pool of candidate boundary positions"""
        print(f"Building seed pool for {self.material}...")
        for _ in range(pool_size * 3):
            pos = random_position(self.material)
            if pos and self.syzygy.probe(pos) is not None:
                h = hash(pos)
                if not any(ph == h for ph, _ in self.seed_pool):
                    self.seed_pool.append((h, pos))
            if len(self.seed_pool) >= pool_size:
                break
        print(f"  Seed pool: {len(self.seed_pool)} candidates")

    def _build_universe(self, sample_size: int = 100):
        """Build universe of 8-piece positions to measure coverage against"""
        print("Building coverage universe...")
        for ph, pos in self.seed_pool[:sample_size]:
            preds = generate_predecessors(pos, max_uncaptures=3)
            for pred in preds:
                if pred.piece_count() == 8:
                    self.universe.add(hash(pred))
        print(f"  Universe: {len(self.universe)} 8-piece positions")

    def hash_state(self, state: SeedConfiguration) -> int:
        return hash(state)

    def get_successors(self, state: SeedConfiguration) -> List[Tuple[SeedConfiguration, Any]]:
        """
        Generate successor configurations.

        Moves:
        - Add a seed from pool
        - Increase depth of existing seed
        - Change mode of existing seed
        """
        successors = []
        seeds = set(state.seeds)

        # Current seed hashes
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

        # INCREASE DEPTH: Deepen existing seeds
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

        return successors[:100]  # Cap total successors

    def get_predecessors(self, state: SeedConfiguration) -> List[Tuple[SeedConfiguration, Any]]:
        """
        Generate predecessor configurations (smaller/simpler configs).

        Moves:
        - Remove a seed
        - Decrease depth
        """
        predecessors = []
        seeds = set(state.seeds)

        # REMOVE: Remove a seed
        for seed in seeds:
            if len(seeds) > 1:  # Keep at least one seed
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
        """
        Boundary conditions:
        - Empty config (0 seeds)
        - Previously evaluated configs
        """
        if len(state.seeds) == 0:
            return True
        return hash(state) in self.eval_cache

    def get_boundary_value(self, state: SeedConfiguration) -> Optional[SeedValue]:
        """Get cached value or compute for boundary"""
        h = hash(state)

        # Check cache
        if h in self.eval_cache:
            return self.eval_cache[h]

        # Empty config
        if len(state.seeds) == 0:
            return SeedValue(0, 0, 0.0)

        return None

    def is_terminal(self, state: SeedConfiguration) -> Tuple[bool, Optional[SeedValue]]:
        """Terminal if full coverage achieved"""
        # Can't be terminal without evaluation
        return False, None

    def propagate_value(self, state: SeedConfiguration,
                        child_values: List[SeedValue]) -> Optional[SeedValue]:
        """
        Value propagation in the lattice.

        Unlike minimax, we're optimizing efficiency.
        A configuration's potential is bounded by its children.

        For now: if we have child values, this state is at least as good as
        the best reachable child (we could choose to move there).
        """
        if not child_values:
            return None

        # Best efficiency among children
        best_child = max(child_values, key=lambda v: v.efficiency)
        return best_child

    def get_features(self, state: SeedConfiguration) -> Any:
        """Equivalence features for configurations"""
        # Group by: number of seeds, total cost, mode distribution
        num_seeds = len(state.seeds)
        total_cost = state.total_cost()
        modes = tuple(sorted(s.mode.value for s in state.seeds))
        return (num_seeds, total_cost, modes)

    def evaluate(self, state: SeedConfiguration) -> SeedValue:
        """
        Actually evaluate a configuration by measuring coverage.

        This is EXPENSIVE - runs the lower layer.
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
            # Find the position
            pos = None
            for ph, p in self.seed_pool:
                if ph == seed_spec.position_hash:
                    pos = p
                    break

            if pos is None:
                continue

            # Expand based on mode and depth
            seed_covered = self._expand_seed(pos, seed_spec.mode, seed_spec.depth)
            covered |= seed_covered
            total_cost += seed_spec.depth

        # Use total coverage, not universe intersection
        coverage = len(covered)
        efficiency = coverage / total_cost if total_cost > 0 else 0.0

        value = SeedValue(coverage, total_cost, efficiency)
        self.eval_cache[h] = value
        return value

    def _expand_seed(self, pos: ChessState, mode: SearchMode, depth: int) -> Set[int]:
        """Expand from a seed position"""
        covered = set()
        frontier = [pos]

        for d in range(depth):
            next_frontier = []
            for p in frontier:
                preds = generate_predecessors(p, max_uncaptures=3)
                for pred in preds:
                    if pred.piece_count() == 8:
                        covered.add(hash(pred))
                    elif pred.piece_count() == 7:
                        if mode != SearchMode.LIGHTNING:  # Lightning doesn't expand 7-piece
                            next_frontier.append(pred)

                # For wave/crystal, limit frontier growth
                if mode == SearchMode.WAVE:
                    next_frontier = next_frontier[:100]
                elif mode == SearchMode.CRYSTAL:
                    next_frontier = next_frontier[:50]

            frontier = next_frontier

        return covered


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_seed_solver(syzygy_path: str = "./syzygy", material: str = "KQRRvKQR"):
    """Create a HOLOS solver for the seed selection game"""
    from holos_core import HOLOSSolver

    game = SeedGame(syzygy_path, material)
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


def demo_seed_game():
    """Demo the seed selection meta-game"""
    from holos_core import SeedPoint, SearchMode

    print("="*60)
    print("SEED SELECTION META-GAME DEMO")
    print("="*60)

    # Create the game
    solver, game = create_seed_solver()

    # Generate initial configurations
    initial_configs = create_initial_configs(game)
    print(f"Generated {len(initial_configs)} initial configurations")

    # Evaluate a few to seed the boundary
    print("\nEvaluating boundary configurations...")
    for config in initial_configs[:5]:
        value = game.evaluate(config)
        print(f"  {config.signature()}: {value}")

    # Create seed points for HOLOS
    forward_seeds = [SeedPoint(c, SearchMode.WAVE, 1) for c in initial_configs]

    # Known-good configurations as backward seeds
    backward_configs = [c for c in initial_configs if hash(c) in game.eval_cache]
    backward_seeds = [SeedPoint(c, SearchMode.WAVE, 1) for c in backward_configs]

    print(f"\nForward seeds: {len(forward_seeds)}")
    print(f"Backward seeds: {len(backward_seeds)}")

    # Run HOLOS on the meta-game
    print("\nRunning HOLOS on seed selection game...")
    hologram = solver.solve(forward_seeds, backward_seeds, max_iterations=10)

    # Find best configuration
    best_config = None
    best_value = None
    for h, value in game.eval_cache.items():
        if best_value is None or value.efficiency > best_value.efficiency:
            best_value = value
            # Find config with this hash
            for config in initial_configs:
                if hash(config) == h:
                    best_config = config
                    break

    print(f"\nBest configuration found:")
    if best_config:
        print(f"  {best_config.signature()}")
        print(f"  {best_value}")
        print(f"  Seeds: {len(best_config.seeds)}")
        for seed in best_config.seeds:
            print(f"    - mode={seed.mode.value}, depth={seed.depth}")


if __name__ == "__main__":
    demo_seed_game()
