"""
holos/tandem.py - Tandem Layer Execution

Runs Layer 0 and Layer 1 HOLOS searches in tandem, where each layer
informs the other.

KEY INSIGHT: Layer 1's state space is different from Layer 0.
- Layer 0: Bidirectional search between start and boundary
- Layer 1: Single-wave search from a GROWING boundary (evaluated seeds)

The full config space is already known (positions × depths × modes × directions).
What we don't know is which configs are GOOD. The boundary grows as Layer 0
evaluates seeds and feeds results back.

This creates an interesting HOLOS variant:
- No "forward wave" in the traditional sense (state space is pre-known)
- Backward wave propagates FROM evaluated seeds
- Osmosis/Lightning pick which config to test next
- Each "test" = run Layer 0 with that seed

The layers inform each other:
- Layer 1 recommends seeds based on pattern of evaluated results
- Layer 0 tests seeds and returns coverage/efficiency
- Layer 0 results become Layer 1 boundary values
"""

import time
from typing import List, Tuple, Optional, Any, Set, Dict
from dataclasses import dataclass, field
from collections import defaultdict

from holos.holos import GameInterface, SearchMode, SeedPoint, HOLOSSolver
from holos.games.seeds import (
    TacticalSeed, TacticalValue, TacticalSeedGame,
    SeedDirection
)


# ============================================================
# TANDEM LAYER 1 GAME
# ============================================================

class TandemSeedGame(TacticalSeedGame):
    """
    Layer 1 game designed for tandem execution with Layer 0.

    Key differences from TacticalSeedGame:
    1. Boundary grows dynamically as Layer 0 provides results
    2. Seed pool comes from Layer 0 frontiers (not fixed)
    3. Osmosis scoring based on neighbor patterns

    The state space shape:
    - All configs are "reachable" (no forward expansion needed)
    - Boundary = evaluated configs (grows over time)
    - Search = pick highest-confidence unevaluated config
    """

    def __init__(self,
                 underlying_game: GameInterface,
                 max_depth: int = 6,
                 initial_positions: List[Tuple[int, Any]] = None):
        """
        Args:
            underlying_game: Layer 0 game
            max_depth: Maximum seed depth to consider
            initial_positions: Starting position pool (can grow dynamically)
        """
        super().__init__(underlying_game, initial_positions or [], max_depth)

        # Track which configs we've recommended but not yet evaluated
        self.pending_evaluation: Set[int] = set()

        # Track pattern statistics for osmosis scoring
        self.feature_stats: Dict[Any, List[TacticalValue]] = defaultdict(list)

        # Track iteration history for learning
        self.iteration_history: List[Dict] = []

    def add_positions(self, positions: List[Tuple[int, Any]]):
        """Add new positions to the seed pool (from Layer 0 frontiers)"""
        existing_hashes = {h for h, _ in self.seed_pool}
        for h, state in positions:
            if h not in existing_hashes:
                self.seed_pool.append((h, state))
                existing_hashes.add(h)

    def record_evaluation(self, seed: TacticalSeed, value: TacticalValue):
        """Record a Layer 0 evaluation result (expands our boundary)"""
        h = hash(seed)
        self.eval_cache[h] = value
        self.pending_evaluation.discard(h)

        # Track feature statistics for pattern learning
        features = self.get_features(seed)
        self.feature_stats[features].append(value)

        # Also track by position features if underlying game supports it
        if seed._state is not None:
            try:
                pos_features = self.underlying_game.get_features(seed._state)
                if pos_features is not None:
                    self.feature_stats[('pos', pos_features)].append(value)
            except:
                pass

    # ==================== Osmosis Support ====================

    def score_for_osmosis(self, state: TacticalSeed) -> float:
        """
        Score a seed config for osmosis-style selection.

        Higher score = more confident about this config's value.

        Factors:
        1. Evaluated neighbors: configs adjacent in (depth, mode, direction) space
        2. Feature pattern: similar configs had consistent values
        3. Position quality: underlying position features match good patterns
        """
        score = 0.0

        # 1. Count evaluated neighbors
        neighbors = self._get_config_neighbors(state)
        evaluated_neighbors = []
        for neighbor in neighbors:
            nh = hash(neighbor)
            if nh in self.eval_cache:
                evaluated_neighbors.append(self.eval_cache[nh])

        # More evaluated neighbors = higher confidence
        score += len(evaluated_neighbors) * 10.0

        # 2. Neighbor value consistency (low variance = high confidence)
        if len(evaluated_neighbors) >= 2:
            efficiencies = [v.efficiency for v in evaluated_neighbors]
            mean_eff = sum(efficiencies) / len(efficiencies)
            variance = sum((e - mean_eff) ** 2 for e in efficiencies) / len(efficiencies)
            # Low variance = consistent = confident
            score += 50.0 / (1.0 + variance)

            # Bonus if neighbors are good
            if mean_eff > 100:
                score += mean_eff * 0.1

        # 3. Feature pattern matching
        features = self.get_features(state)
        if features in self.feature_stats:
            past_values = self.feature_stats[features]
            if past_values:
                avg_efficiency = sum(v.efficiency for v in past_values) / len(past_values)
                # Bonus for configs with feature patterns that worked well
                score += avg_efficiency * 0.5

        # 4. Position quality (if we have position features)
        if state._state is not None:
            try:
                pos_features = self.underlying_game.get_features(state._state)
                if pos_features and ('pos', pos_features) in self.feature_stats:
                    past = self.feature_stats[('pos', pos_features)]
                    if past:
                        avg = sum(v.efficiency for v in past) / len(past)
                        score += avg * 0.3
            except:
                pass

        # 5. Prefer unexplored over pending
        h = hash(state)
        if h in self.pending_evaluation:
            score -= 100.0  # Deprioritize already-pending configs

        return score

    def _get_config_neighbors(self, state: TacticalSeed) -> List[TacticalSeed]:
        """Get configs adjacent in parameter space"""
        neighbors = []

        # Depth neighbors
        if state.depth > 1:
            neighbors.append(TacticalSeed(
                state.position_hash, state.depth - 1,
                state.mode, state.direction, state._state
            ))
        if state.depth < self.max_depth:
            neighbors.append(TacticalSeed(
                state.position_hash, state.depth + 1,
                state.mode, state.direction, state._state
            ))

        # Mode neighbors
        modes = [SearchMode.LIGHTNING, SearchMode.WAVE, SearchMode.CRYSTAL]
        if state.mode in modes:
            idx = modes.index(state.mode)
            if idx > 0:
                neighbors.append(TacticalSeed(
                    state.position_hash, state.depth,
                    modes[idx-1], state.direction, state._state
                ))
            if idx < len(modes) - 1:
                neighbors.append(TacticalSeed(
                    state.position_hash, state.depth,
                    modes[idx+1], state.direction, state._state
                ))

        # Direction neighbors
        directions = [SeedDirection.FORWARD, SeedDirection.BACKWARD, SeedDirection.BILATERAL]
        if state.direction in directions:
            idx = directions.index(state.direction)
            for i, d in enumerate(directions):
                if i != idx:
                    neighbors.append(TacticalSeed(
                        state.position_hash, state.depth,
                        state.mode, d, state._state
                    ))

        return neighbors

    # ==================== Recommendation Generation ====================

    def get_next_recommendations(self,
                                  count: int = 5,
                                  mode: str = "osmosis") -> List[TacticalSeed]:
        """
        Get the next seed configs to test in Layer 0.

        Args:
            count: Number of recommendations
            mode: Selection strategy
                - "osmosis": Pick highest-confidence configs
                - "lightning": DFS from best known seeds toward better configs
                - "random": Random unexplored configs
                - "diverse": Spread across different positions/settings

        Returns:
            List of TacticalSeed configs to test
        """
        if mode == "osmosis":
            return self._recommend_osmosis(count)
        elif mode == "lightning":
            return self._recommend_lightning(count)
        elif mode == "random":
            return self._recommend_random(count)
        elif mode == "diverse":
            return self._recommend_diverse(count)
        else:
            return self._recommend_osmosis(count)

    def _recommend_osmosis(self, count: int) -> List[TacticalSeed]:
        """Pick configs with highest osmosis scores"""
        candidates = self._generate_candidate_configs()

        # Score all candidates
        scored = []
        for config in candidates:
            h = hash(config)
            if h not in self.eval_cache and h not in self.pending_evaluation:
                score = self.score_for_osmosis(config)
                scored.append((score, config))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Take top N
        result = []
        for score, config in scored[:count]:
            result.append(config)
            self.pending_evaluation.add(hash(config))

        return result

    def _recommend_lightning(self, count: int) -> List[TacticalSeed]:
        """DFS from best seeds toward better configs"""
        result = []

        # Find best evaluated seeds
        if not self.eval_cache:
            # No evaluations yet - start with cheap configs
            return self._recommend_initial(count)

        best_seeds = sorted(
            self.eval_cache.items(),
            key=lambda x: x[1].efficiency,
            reverse=True
        )[:10]

        # For each best seed, try "upgrading" it
        for seed_hash, value in best_seeds:
            if len(result) >= count:
                break

            # Find the seed config (need to reconstruct from hash)
            # This is a limitation - we should store configs, not just hashes
            # For now, try successors of seeds we know
            for ph, ps in self.seed_pool:
                for depth in range(1, self.max_depth + 1):
                    for mode in [SearchMode.LIGHTNING, SearchMode.WAVE]:
                        for direction in SeedDirection:
                            config = TacticalSeed(ph, depth, mode, direction, ps)
                            if hash(config) == seed_hash:
                                # Found it! Now try successors
                                for succ, _ in self.get_successors(config):
                                    sh = hash(succ)
                                    if sh not in self.eval_cache and sh not in self.pending_evaluation:
                                        result.append(succ)
                                        self.pending_evaluation.add(sh)
                                        if len(result) >= count:
                                            break
                                break

        # Fill with osmosis if not enough
        if len(result) < count:
            result.extend(self._recommend_osmosis(count - len(result)))

        return result[:count]

    def _recommend_random(self, count: int) -> List[TacticalSeed]:
        """Random unexplored configs"""
        import random

        candidates = self._generate_candidate_configs()
        unexplored = [c for c in candidates
                      if hash(c) not in self.eval_cache
                      and hash(c) not in self.pending_evaluation]

        selected = random.sample(unexplored, min(count, len(unexplored)))
        for config in selected:
            self.pending_evaluation.add(hash(config))

        return selected

    def _recommend_diverse(self, count: int) -> List[TacticalSeed]:
        """Spread across different positions and settings"""
        result = []
        used_positions = set()
        used_settings = set()

        candidates = self._generate_candidate_configs()

        # Score by osmosis but enforce diversity
        scored = []
        for config in candidates:
            h = hash(config)
            if h not in self.eval_cache and h not in self.pending_evaluation:
                score = self.score_for_osmosis(config)
                scored.append((score, config))

        scored.sort(key=lambda x: -x[0])

        for score, config in scored:
            if len(result) >= count:
                break

            # Check diversity
            setting = (config.depth, config.mode, config.direction)
            if config.position_hash in used_positions and setting in used_settings:
                continue

            result.append(config)
            self.pending_evaluation.add(hash(config))
            used_positions.add(config.position_hash)
            used_settings.add(setting)

        return result

    def _recommend_initial(self, count: int) -> List[TacticalSeed]:
        """Initial recommendations when no evaluations exist"""
        result = []

        # Start with depth=1, lightning, forward - cheapest configs
        for ph, ps in self.seed_pool[:count]:
            config = TacticalSeed(
                ph, 1, SearchMode.LIGHTNING, SeedDirection.FORWARD, ps
            )
            result.append(config)
            self.pending_evaluation.add(hash(config))

        return result

    def _generate_candidate_configs(self) -> List[TacticalSeed]:
        """Generate all candidate configs from current pool"""
        candidates = []

        for ph, ps in self.seed_pool:
            for depth in range(1, self.max_depth + 1):
                for mode in [SearchMode.LIGHTNING, SearchMode.WAVE]:
                    for direction in SeedDirection:
                        candidates.append(TacticalSeed(ph, depth, mode, direction, ps))

        return candidates

    # ==================== Statistics ====================

    def get_statistics(self) -> Dict:
        """Get current state statistics"""
        return {
            'pool_size': len(self.seed_pool),
            'evaluated': len(self.eval_cache),
            'pending': len(self.pending_evaluation),
            'feature_patterns': len(self.feature_stats),
            'best_efficiency': max((v.efficiency for v in self.eval_cache.values()), default=0),
            'avg_efficiency': (sum(v.efficiency for v in self.eval_cache.values()) /
                              len(self.eval_cache) if self.eval_cache else 0),
        }


# ============================================================
# TANDEM ORCHESTRATOR
# ============================================================

@dataclass
class TandemIterationResult:
    """Result of one tandem iteration"""
    iteration: int
    seeds_tested: int
    layer0_positions_solved: int
    layer0_connections: int
    best_seed_efficiency: float
    avg_seed_efficiency: float
    elapsed_seconds: float

    def __repr__(self):
        return (f"Iteration {self.iteration}: tested={self.seeds_tested}, "
                f"solved={self.layer0_positions_solved}, conn={self.layer0_connections}, "
                f"best_eff={self.best_seed_efficiency:.1f}, time={self.elapsed_seconds:.2f}s")


class TandemOrchestrator:
    """
    Orchestrates Layer 0 and Layer 1 running in tandem.

    Each iteration:
    1. Layer 1 recommends seeds based on current knowledge
    2. Layer 0 tests those seeds (limited iterations)
    3. Results feed back to Layer 1 as boundary values
    4. Repeat

    The layers learn from each other:
    - Layer 1 discovers which seed patterns work well
    - Layer 0 gets better seeds, solves more efficiently
    """

    def __init__(self,
                 layer0_game: GameInterface,
                 initial_positions: List[Tuple[int, Any]] = None,
                 max_depth: int = 6,
                 recommendation_mode: str = "osmosis"):
        """
        Args:
            layer0_game: The underlying game (chess, connect4, etc.)
            initial_positions: Starting positions for seeding
            max_depth: Maximum seed depth for Layer 1
            recommendation_mode: How Layer 1 picks seeds
        """
        self.layer0_game = layer0_game
        self.layer0_solver = HOLOSSolver(layer0_game, name="layer0_tandem")

        self.layer1_game = TandemSeedGame(
            layer0_game,
            max_depth=max_depth,
            initial_positions=initial_positions
        )

        self.recommendation_mode = recommendation_mode
        self.iteration = 0
        self.results: List[TandemIterationResult] = []

    def add_positions(self, positions: List[Tuple[int, Any]]):
        """Add positions to the seed pool"""
        self.layer1_game.add_positions(positions)

    def run_iteration(self,
                      seeds_per_iteration: int = 5,
                      layer0_iterations: int = 3,
                      verbose: bool = True) -> TandemIterationResult:
        """
        Run one tandem iteration.

        Args:
            seeds_per_iteration: How many seeds Layer 1 recommends
            layer0_iterations: How many iterations Layer 0 runs per seed
            verbose: Print progress

        Returns:
            TandemIterationResult
        """
        start_time = time.time()
        self.iteration += 1

        if verbose:
            print(f"\n=== Tandem Iteration {self.iteration} ===")
            stats = self.layer1_game.get_statistics()
            print(f"Layer 1 state: {stats}")

        # 1. Layer 1 recommends seeds
        recommended = self.layer1_game.get_next_recommendations(
            count=seeds_per_iteration,
            mode=self.recommendation_mode
        )

        if verbose:
            print(f"Layer 1 recommends {len(recommended)} seeds:")
            for seed in recommended:
                print(f"  {seed.signature()}")

        if not recommended:
            if verbose:
                print("No more seeds to recommend!")
            return TandemIterationResult(
                iteration=self.iteration,
                seeds_tested=0,
                layer0_positions_solved=0,
                layer0_connections=0,
                best_seed_efficiency=0,
                avg_seed_efficiency=0,
                elapsed_seconds=time.time() - start_time
            )

        # 2. Test each seed in Layer 0
        seed_results = []
        total_solved = 0
        total_connections = 0

        for seed in recommended:
            # Create Layer 0 seed point
            if seed._state is None:
                continue

            seed_point = SeedPoint(
                seed._state,
                seed.mode,
                priority=1,
                depth=seed.depth
            )

            # Record solver state before
            solved_before = len(self.layer0_solver.solved)
            conn_before = len(self.layer0_solver.connections)

            # Run Layer 0
            try:
                self.layer0_solver.solve(
                    [seed_point],
                    max_iterations=layer0_iterations
                )
            except Exception as e:
                if verbose:
                    print(f"  Error testing {seed.signature()}: {e}")
                continue

            # Measure coverage
            solved_after = len(self.layer0_solver.solved)
            conn_after = len(self.layer0_solver.connections)

            forward_coverage = solved_after - solved_before
            connections = conn_after - conn_before

            total_solved += forward_coverage
            total_connections += connections

            # Create TacticalValue from results
            value = TacticalValue(
                forward_coverage=forward_coverage,
                backward_coverage=0,  # Would need separate measurement
                overlap_potential=0.0,
                cost=seed.cost(),
                efficiency=forward_coverage / seed.cost() if seed.cost() > 0 else 0,
                time_to_first_solve=0.0,
                solves_per_second=0.0
            )

            seed_results.append((seed, value))

            # 3. Feed result back to Layer 1
            self.layer1_game.record_evaluation(seed, value)

            if verbose:
                print(f"  {seed.signature()}: coverage={forward_coverage}, eff={value.efficiency:.1f}")

        # 4. Update Layer 1 with new positions from Layer 0 frontiers
        new_positions = []
        for h, state in list(self.layer0_solver.forward_frontier.items())[:100]:
            new_positions.append((h, state))
        for h, state in list(self.layer0_solver.backward_frontier.items())[:100]:
            new_positions.append((h, state))

        self.layer1_game.add_positions(new_positions)

        # Compute summary
        best_eff = max((v.efficiency for _, v in seed_results), default=0)
        avg_eff = (sum(v.efficiency for _, v in seed_results) / len(seed_results)
                   if seed_results else 0)

        result = TandemIterationResult(
            iteration=self.iteration,
            seeds_tested=len(seed_results),
            layer0_positions_solved=total_solved,
            layer0_connections=total_connections,
            best_seed_efficiency=best_eff,
            avg_seed_efficiency=avg_eff,
            elapsed_seconds=time.time() - start_time
        )

        self.results.append(result)

        if verbose:
            print(f"Result: {result}")

        return result

    def run_session(self,
                    max_iterations: int = 20,
                    seeds_per_iteration: int = 5,
                    layer0_iterations: int = 3,
                    target_solved: int = None,
                    verbose: bool = True) -> List[TandemIterationResult]:
        """
        Run a full tandem session.

        Args:
            max_iterations: Maximum tandem iterations
            seeds_per_iteration: Seeds per iteration
            layer0_iterations: Layer 0 iterations per seed test
            target_solved: Stop when this many positions are solved
            verbose: Print progress

        Returns:
            List of iteration results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Tandem Session")
            print(f"  Mode: {self.recommendation_mode}")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Seeds per iteration: {seeds_per_iteration}")
            print(f"  Initial pool size: {len(self.layer1_game.seed_pool)}")
            print(f"{'='*60}")

        for i in range(max_iterations):
            result = self.run_iteration(
                seeds_per_iteration=seeds_per_iteration,
                layer0_iterations=layer0_iterations,
                verbose=verbose
            )

            # Check stopping conditions
            if target_solved and len(self.layer0_solver.solved) >= target_solved:
                if verbose:
                    print(f"\nTarget of {target_solved} solved positions reached!")
                break

            if result.seeds_tested == 0:
                if verbose:
                    print("\nNo more seeds to test, stopping.")
                break

        if verbose:
            self.print_summary()

        return self.results

    def print_summary(self):
        """Print session summary"""
        print(f"\n{'='*60}")
        print("Tandem Session Summary")
        print(f"{'='*60}")

        total_seeds = sum(r.seeds_tested for r in self.results)
        total_solved = len(self.layer0_solver.solved)
        total_time = sum(r.elapsed_seconds for r in self.results)

        print(f"Iterations: {len(self.results)}")
        print(f"Seeds tested: {total_seeds}")
        print(f"Positions solved: {total_solved}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Solve rate: {total_solved / total_time:.1f} positions/sec" if total_time > 0 else "N/A")

        print(f"\nLayer 1 Statistics:")
        stats = self.layer1_game.get_statistics()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print(f"\nLayer 0 Statistics:")
        for k, v in self.layer0_solver.stats.items():
            print(f"  {k}: {v}")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_tandem_solver(game: GameInterface,
                         initial_positions: List[Any] = None,
                         max_depth: int = 6,
                         mode: str = "osmosis") -> TandemOrchestrator:
    """
    Create a tandem orchestrator for the given game.

    Args:
        game: Layer 0 game
        initial_positions: Starting positions (will extract hashes)
        max_depth: Maximum seed depth
        mode: Recommendation mode ("osmosis", "lightning", "random", "diverse")

    Returns:
        TandemOrchestrator ready to run
    """
    # Convert positions to (hash, state) pairs
    if initial_positions:
        pool = [(game.hash_state(pos), pos) for pos in initial_positions]
    else:
        pool = []

    return TandemOrchestrator(
        layer0_game=game,
        initial_positions=pool,
        max_depth=max_depth,
        recommendation_mode=mode
    )


def compare_recommendation_modes(game: GameInterface,
                                  initial_positions: List[Any],
                                  iterations: int = 10,
                                  seeds_per_iter: int = 5) -> Dict[str, Dict]:
    """
    Compare different recommendation modes on the same game.

    Returns dict of mode -> results summary
    """
    results = {}

    for mode in ["osmosis", "lightning", "random", "diverse"]:
        print(f"\n{'='*60}")
        print(f"Testing mode: {mode}")
        print(f"{'='*60}")

        orchestrator = create_tandem_solver(
            game, initial_positions, max_depth=5, mode=mode
        )

        orchestrator.run_session(
            max_iterations=iterations,
            seeds_per_iteration=seeds_per_iter,
            verbose=False
        )

        results[mode] = {
            'total_solved': len(orchestrator.layer0_solver.solved),
            'seeds_tested': sum(r.seeds_tested for r in orchestrator.results),
            'avg_efficiency': orchestrator.layer1_game.get_statistics()['avg_efficiency'],
            'time': sum(r.elapsed_seconds for r in orchestrator.results),
        }

        print(f"  Solved: {results[mode]['total_solved']}")
        print(f"  Seeds tested: {results[mode]['seeds_tested']}")
        print(f"  Avg efficiency: {results[mode]['avg_efficiency']:.1f}")

    return results
