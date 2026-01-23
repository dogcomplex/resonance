"""
holos_meta_game.py - HOLOS searching for HOLOS strategies

The meta-game: Use the SAME bidirectional lightning+crystal algorithm
to search the space of seeding strategies.

ANALOGY:
  Base Game (Chess)              Meta-Game (Strategy Search)
  -----------------              ---------------------------
  State = Board position         State = Strategy configuration
  Move = Legal chess move        Move = Modify strategy parameters
  Value = Win/Draw/Loss          Value = Efficiency (coverage/cost)
  Syzygy = 7-piece oracle        Meta-oracle = Empirical measurement
  Lightning = Capture chains     Lightning = High-impact parameter changes
  Crystal = Local BFS            Crystal = Local parameter tuning

KEY INSIGHT: "Captures" in meta-game = decisions that quickly reduce
uncertainty about strategy quality. Just as chess captures reduce piece
count toward the solvable boundary, meta-captures reduce parameter space
toward the measurable region.
"""

import os
import sys
import time
import random
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# Import base HOLOS components
from fractal_holos3 import (
    SyzygyProbe, ChessState, random_position, generate_predecessors,
    extract_features, generate_moves, apply_move, is_terminal
)


# ============================================================
# META-GAME STATE REPRESENTATION
# ============================================================

class StrategyType(Enum):
    """Types of seeding strategies"""
    RANDOM = "random"
    HIGH_CONNECTIVITY = "high_connectivity"
    GREEDY_COVERAGE = "greedy_coverage"
    CLUSTER_CENTERS = "cluster_centers"
    BOUNDARY_DENSITY = "boundary_density"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class MetaState:
    """
    State in the meta-game = a strategy configuration.

    Like a chess position, but describes HOW to search rather than WHAT to search.
    """
    # Strategy type (analogous to material composition)
    strategy_type: StrategyType

    # Parameters (analogous to piece positions)
    num_seeds: int           # How many seeds to use
    connectivity_weight: float  # 0-1, how much to weight connectivity
    coverage_weight: float      # 0-1, how much to weight coverage
    exploration_rate: float     # 0-1, random exploration vs exploitation

    # Material context (what piece configuration are we optimizing for)
    material_7: str

    def __hash__(self):
        return hash((
            self.strategy_type,
            self.num_seeds,
            round(self.connectivity_weight, 2),
            round(self.coverage_weight, 2),
            round(self.exploration_rate, 2),
            self.material_7
        ))

    def to_dict(self) -> dict:
        return {
            'strategy_type': self.strategy_type.value,
            'num_seeds': self.num_seeds,
            'connectivity_weight': self.connectivity_weight,
            'coverage_weight': self.coverage_weight,
            'exploration_rate': self.exploration_rate,
            'material_7': self.material_7
        }


@dataclass
class MetaFeatures:
    """
    Equivalence class features for meta-states.
    Two strategies with same features might have same efficiency.
    """
    strategy_family: str      # e.g., "greedy_family", "random_family"
    seed_count_bucket: int    # Bucketed seed count (10, 20, 50, 100, etc.)
    balance: float            # connectivity_weight / coverage_weight ratio

    def __hash__(self):
        return hash((self.strategy_family, self.seed_count_bucket, round(self.balance, 1)))


def extract_meta_features(state: MetaState) -> MetaFeatures:
    """Extract equivalence features from a meta-state"""
    # Strategy family
    if state.strategy_type in [StrategyType.RANDOM]:
        family = "random_family"
    elif state.strategy_type in [StrategyType.GREEDY_COVERAGE, StrategyType.HIGH_CONNECTIVITY]:
        family = "greedy_family"
    else:
        family = "hybrid_family"

    # Bucket seed count
    if state.num_seeds <= 10:
        bucket = 10
    elif state.num_seeds <= 25:
        bucket = 25
    elif state.num_seeds <= 50:
        bucket = 50
    else:
        bucket = 100

    # Balance ratio
    if state.coverage_weight > 0:
        balance = state.connectivity_weight / state.coverage_weight
    else:
        balance = float('inf')

    return MetaFeatures(family, bucket, balance)


# ============================================================
# META-GAME MOVES
# ============================================================

def generate_meta_moves(state: MetaState) -> List[MetaState]:
    """
    Generate legal moves in the meta-game.

    Analogous to generate_moves() in chess, but for strategy space.
    """
    moves = []

    # Type changes (like piece movement)
    for new_type in StrategyType:
        if new_type != state.strategy_type:
            moves.append(MetaState(
                strategy_type=new_type,
                num_seeds=state.num_seeds,
                connectivity_weight=state.connectivity_weight,
                coverage_weight=state.coverage_weight,
                exploration_rate=state.exploration_rate,
                material_7=state.material_7
            ))

    # Parameter adjustments (like moving pieces to adjacent squares)
    deltas = [-0.2, -0.1, 0.1, 0.2]

    for d in deltas:
        # Connectivity weight
        new_cw = max(0, min(1, state.connectivity_weight + d))
        if new_cw != state.connectivity_weight:
            moves.append(MetaState(
                strategy_type=state.strategy_type,
                num_seeds=state.num_seeds,
                connectivity_weight=new_cw,
                coverage_weight=state.coverage_weight,
                exploration_rate=state.exploration_rate,
                material_7=state.material_7
            ))

        # Coverage weight
        new_cvw = max(0, min(1, state.coverage_weight + d))
        if new_cvw != state.coverage_weight:
            moves.append(MetaState(
                strategy_type=state.strategy_type,
                num_seeds=state.num_seeds,
                connectivity_weight=state.connectivity_weight,
                coverage_weight=new_cvw,
                exploration_rate=state.exploration_rate,
                material_7=state.material_7
            ))

        # Exploration rate
        new_er = max(0, min(1, state.exploration_rate + d))
        if new_er != state.exploration_rate:
            moves.append(MetaState(
                strategy_type=state.strategy_type,
                num_seeds=state.num_seeds,
                connectivity_weight=state.connectivity_weight,
                coverage_weight=state.coverage_weight,
                exploration_rate=new_er,
                material_7=state.material_7
            ))

    # Seed count changes (like captures - significant state changes!)
    for factor in [0.5, 0.75, 1.25, 1.5, 2.0]:
        new_seeds = max(5, min(200, int(state.num_seeds * factor)))
        if new_seeds != state.num_seeds:
            moves.append(MetaState(
                strategy_type=state.strategy_type,
                num_seeds=new_seeds,
                connectivity_weight=state.connectivity_weight,
                coverage_weight=state.coverage_weight,
                exploration_rate=state.exploration_rate,
                material_7=state.material_7
            ))

    return moves


def generate_meta_captures(state: MetaState) -> List[MetaState]:
    """
    Generate "capture" moves in meta-game - high-impact changes.

    In chess, captures reduce piece count toward boundary.
    In meta-game, "captures" are moves that significantly change strategy
    and quickly reveal whether it's good or bad.
    """
    captures = []

    # Major strategy type changes (like capturing a queen)
    major_types = [StrategyType.RANDOM, StrategyType.GREEDY_COVERAGE]
    for new_type in major_types:
        if new_type != state.strategy_type:
            captures.append(MetaState(
                strategy_type=new_type,
                num_seeds=state.num_seeds,
                connectivity_weight=state.connectivity_weight,
                coverage_weight=state.coverage_weight,
                exploration_rate=state.exploration_rate,
                material_7=state.material_7
            ))

    # Drastic parameter changes (like capturing pieces)
    extremes = [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5)]
    for cw, cvw in extremes:
        if cw != state.connectivity_weight or cvw != state.coverage_weight:
            captures.append(MetaState(
                strategy_type=state.strategy_type,
                num_seeds=state.num_seeds,
                connectivity_weight=cw,
                coverage_weight=cvw,
                exploration_rate=state.exploration_rate,
                material_7=state.material_7
            ))

    return captures


# ============================================================
# META-ORACLE (Empirical Strategy Evaluation)
# ============================================================

class MetaOracle:
    """
    The "Syzygy" of the meta-game: empirical measurement of strategy quality.

    Like Syzygy gives exact values for 7-piece positions, MetaOracle gives
    measured efficiency scores for strategies that have been evaluated.
    """

    def __init__(self, syzygy: SyzygyProbe, cache_path: str = "./meta_oracle_cache.pkl"):
        self.syzygy = syzygy
        self.cache_path = cache_path
        self.cache: Dict[int, float] = {}  # state_hash -> efficiency
        self.evaluations = 0
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            print(f"[MetaOracle] Loaded {len(self.cache)} cached evaluations")

    def _save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def probe(self, state: MetaState) -> Optional[float]:
        """
        Probe the meta-oracle for a strategy's efficiency.

        Returns cached value if available, None if evaluation needed.
        Like Syzygy, this is exact for evaluated strategies.
        """
        h = hash(state)
        return self.cache.get(h)

    def evaluate(self, state: MetaState, num_samples: int = 100) -> float:
        """
        Actually evaluate a strategy by running it.

        This is expensive! Like computing a tablebase entry.
        """
        h = hash(state)
        if h in self.cache:
            return self.cache[h]

        self.evaluations += 1

        # Generate seeds using the strategy
        seeds = self._generate_seeds_with_strategy(state, num_samples)

        # Measure coverage
        coverage = self._measure_coverage(seeds, state.material_7)

        # Efficiency = coverage / seeds used
        efficiency = coverage / max(1, len(seeds))

        # Cache the result
        self.cache[h] = efficiency

        # Periodically save
        if self.evaluations % 10 == 0:
            self._save_cache()

        return efficiency

    def _generate_seeds_with_strategy(self, state: MetaState,
                                       num_samples: int) -> List[ChessState]:
        """Generate seed positions using the specified strategy"""
        seeds = []

        # Generate candidate positions
        candidates = []
        for _ in range(num_samples * 3):
            pos = random_position(state.material_7)
            if pos and self.syzygy.probe(pos) is not None:
                candidates.append(pos)
            if len(candidates) >= num_samples * 2:
                break

        if not candidates:
            return []

        if state.strategy_type == StrategyType.RANDOM:
            # Random selection
            return random.sample(candidates, min(state.num_seeds, len(candidates)))

        elif state.strategy_type == StrategyType.HIGH_CONNECTIVITY:
            # Sort by connectivity (predecessor count)
            scored = []
            for pos in candidates:
                preds = generate_predecessors(pos, max_uncaptures=2)
                score = len(preds)
                scored.append((score, pos))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [pos for _, pos in scored[:state.num_seeds]]

        elif state.strategy_type == StrategyType.GREEDY_COVERAGE:
            # Greedy set cover
            selected = []
            covered = set()

            for pos in candidates:
                preds = generate_predecessors(pos, max_uncaptures=2)
                pred_hashes = {hash(p) for p in preds}
                new_coverage = pred_hashes - covered

                if new_coverage:
                    selected.append(pos)
                    covered |= pred_hashes

                if len(selected) >= state.num_seeds:
                    break

            return selected

        else:  # HYBRID - weighted combination
            scored = []
            covered = set()

            for pos in candidates:
                preds = generate_predecessors(pos, max_uncaptures=2)
                pred_hashes = {hash(p) for p in preds}

                connectivity = len(preds)
                new_coverage = len(pred_hashes - covered)

                # Weighted score
                score = (state.connectivity_weight * connectivity +
                        state.coverage_weight * new_coverage +
                        state.exploration_rate * random.random())

                scored.append((score, pos, pred_hashes))

            scored.sort(key=lambda x: x[0], reverse=True)

            selected = []
            for score, pos, pred_hashes in scored:
                if len(selected) >= state.num_seeds:
                    break
                selected.append(pos)
                covered |= pred_hashes

            return selected

    def _measure_coverage(self, seeds: List[ChessState], material_7: str) -> int:
        """Measure how many 8-piece positions the seeds can reach"""
        covered = set()

        for seed in seeds:
            preds = generate_predecessors(seed, max_uncaptures=3)
            for pred in preds:
                if pred.piece_count() == 8:
                    covered.add(hash(pred))

        return len(covered)


# ============================================================
# META-GAME LIGHTNING PROBE
# ============================================================

class MetaLightningProbe:
    """
    Lightning probe for meta-game: quickly find good strategies.

    Like chess lightning that follows captures to reach boundary,
    meta-lightning follows "capture" moves (high-impact changes)
    to quickly reach evaluated strategies.
    """

    def __init__(self, oracle: MetaOracle, max_depth: int = 5):
        self.oracle = oracle
        self.max_depth = max_depth
        self.nodes_visited = 0

    def probe(self, state: MetaState) -> Tuple[Optional[float], List[MetaState]]:
        """
        Search for path to evaluated strategy.
        Returns (efficiency, path) or (None, []) if no path found.
        """
        self.nodes_visited = 0
        path = []
        value = self._search_captures(state, 0, path)
        return value, path

    def _search_captures(self, state: MetaState, depth: int,
                         path: List[MetaState]) -> Optional[float]:
        self.nodes_visited += 1

        # Check oracle (like checking syzygy)
        cached = self.oracle.probe(state)
        if cached is not None:
            return cached

        if depth >= self.max_depth:
            return None

        # Only follow "captures" (high-impact moves)
        captures = generate_meta_captures(state)

        for next_state in captures[:3]:  # Top 3 captures
            child_path = []
            value = self._search_captures(next_state, depth + 1, child_path)

            if value is not None:
                path.clear()
                path.append(state)
                path.extend(child_path)
                return value

        return None


# ============================================================
# BIDIRECTIONAL META-HOLOS SOLVER
# ============================================================

class BidirectionalMetaHOLOS:
    """
    HOLOS applied to search for optimal HOLOS strategies.

    This is the recursive meta-game: using bidirectional search
    to find the best way to do bidirectional search!
    """

    def __init__(self, syzygy_path: str = "./syzygy"):
        self.syzygy = SyzygyProbe(syzygy_path)
        self.oracle = MetaOracle(self.syzygy)

        # Holographic storage for meta-game
        self.solved: Dict[int, float] = {}  # state_hash -> efficiency
        self.all_states: Dict[int, MetaState] = {}  # Keep all states for lookup

        # Equivalence classes
        self.equiv_classes: Dict[MetaFeatures, Set[int]] = defaultdict(set)
        self.equiv_outcomes: Dict[MetaFeatures, Optional[float]] = {}

        # Frontiers
        self.forward_frontier: Dict[int, MetaState] = {}
        self.backward_frontier: Dict[int, MetaState] = {}

        self.forward_seen: Set[int] = set()
        self.backward_seen: Set[int] = set()

        # Stats
        self.stats = {
            'lightning_probes': 0,
            'connections': 0,
            'evaluations': 0,
        }

    def solve(self, material_7: str = "KQRRvKQR",
              starting_strategies: List[MetaState] = None,
              max_iterations: int = 20,
              samples_per_eval: int = 50):
        """
        Find optimal seeding strategy for given material.
        """
        print("=" * 60)
        print("META-HOLOS: Searching for Optimal Strategy")
        print("=" * 60)
        print(f"Material: {material_7}")

        # Initialize forward frontier with starting strategies
        if starting_strategies is None:
            starting_strategies = self._generate_starting_strategies(material_7)

        for state in starting_strategies:
            h = hash(state)
            if h not in self.forward_seen:
                self.forward_seen.add(h)
                self.forward_frontier[h] = state
                self.all_states[h] = state

        # Initialize backward frontier with "known good" strategies
        # (Like seeding from syzygy boundary)
        known_good = [
            MetaState(StrategyType.GREEDY_COVERAGE, 50, 0.3, 0.7, 0.0, material_7),
            MetaState(StrategyType.HIGH_CONNECTIVITY, 50, 0.7, 0.3, 0.0, material_7),
        ]

        for state in known_good:
            h = hash(state)
            if h not in self.backward_seen:
                self.backward_seen.add(h)
                self.backward_frontier[h] = state
                self.all_states[h] = state
                # Evaluate and seed
                efficiency = self.oracle.evaluate(state, samples_per_eval)
                self.solved[h] = efficiency
                self.stats['evaluations'] += 1

        print(f"Forward frontier: {len(self.forward_frontier)}")
        print(f"Backward frontier: {len(self.backward_frontier)}")

        best_state = None
        best_efficiency = 0

        for iteration in range(max_iterations):
            print(f"\n--- Meta-Iteration {iteration} ---")

            # Lightning phase: probe from forward frontier
            if iteration % 3 == 0:
                self._lightning_phase(samples_per_eval)

            # Expand frontiers
            forward_contacts = self._expand_forward(samples_per_eval)
            backward_contacts = self._expand_backward()

            # Find connections
            connections = self._find_connections()

            if connections:
                print(f"  ** {connections} META-CONNECTIONS **")

            # Track best
            for h, eff in self.solved.items():
                if eff > best_efficiency:
                    best_efficiency = eff
                    best_state = self.all_states.get(h)

            # Propagate via equivalence
            propagated = self._propagate_equivalence()

            print(f"  Solved: {len(self.solved)}, Best efficiency: {best_efficiency:.2f}")
            print(f"  Forward: {len(self.forward_frontier)}, Backward: {len(self.backward_frontier)}")
            print(f"  Oracle evaluations: {self.oracle.evaluations}")

            # Regenerate frontiers if empty (exploration)
            if not self.forward_frontier and iteration < max_iterations - 1:
                print("  Regenerating frontiers for more exploration...")
                new_starts = self._generate_random_strategies(material_7, 20)
                for state in new_starts:
                    h = hash(state)
                    if h not in self.forward_seen:
                        self.forward_seen.add(h)
                        self.forward_frontier[h] = state
                        self.all_states[h] = state

        # Final summary
        print("\n" + "=" * 60)
        print("META-HOLOS COMPLETE")
        print("=" * 60)

        # Find best strategy
        if best_state:
            print(f"\nBest Strategy Found:")
            print(f"  Type: {best_state.strategy_type.value}")
            print(f"  Seeds: {best_state.num_seeds}")
            print(f"  Connectivity weight: {best_state.connectivity_weight:.2f}")
            print(f"  Coverage weight: {best_state.coverage_weight:.2f}")
            print(f"  Efficiency: {best_efficiency:.2f}")

        return best_state, best_efficiency

    def _generate_starting_strategies(self, material_7: str) -> List[MetaState]:
        """Generate diverse starting strategies"""
        strategies = []

        for strategy_type in StrategyType:
            for num_seeds in [10, 20, 50, 100]:
                for cw in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    cvw = 1.0 - cw
                    for er in [0.0, 0.1, 0.3]:
                        strategies.append(MetaState(
                            strategy_type=strategy_type,
                            num_seeds=num_seeds,
                            connectivity_weight=cw,
                            coverage_weight=cvw,
                            exploration_rate=er,
                            material_7=material_7
                        ))

        return strategies

    def _generate_random_strategies(self, material_7: str, count: int) -> List[MetaState]:
        """Generate random strategies for exploration"""
        strategies = []
        for _ in range(count):
            strategies.append(MetaState(
                strategy_type=random.choice(list(StrategyType)),
                num_seeds=random.choice([5, 10, 15, 20, 30, 50, 75, 100, 150]),
                connectivity_weight=random.random(),
                coverage_weight=random.random(),
                exploration_rate=random.random() * 0.5,
                material_7=material_7
            ))
        return strategies

    def _lightning_phase(self, samples_per_eval: int):
        """Meta-lightning: quick probes to find evaluated strategies"""
        probe = MetaLightningProbe(self.oracle)

        sample_size = min(5, len(self.forward_frontier))
        samples = random.sample(list(self.forward_frontier.values()), sample_size)

        found = 0
        for state in samples:
            value, path = probe.probe(state)
            self.stats['lightning_probes'] += 1

            if value is not None:
                # Found path to known strategy
                h = hash(state)
                self.solved[h] = value
                found += 1

        if found:
            print(f"  Meta-lightning: found {found} paths to evaluated strategies")

    def _expand_forward(self, samples_per_eval: int) -> int:
        """Expand forward frontier in meta-game"""
        items = list(self.forward_frontier.items())
        if not items:
            return 0

        next_frontier = {}
        contacts = 0

        for h, state in items[:20]:  # Limit expansion
            # Check if already solved
            if h in self.solved:
                contacts += 1
                continue

            # Track equivalence
            features = extract_meta_features(state)
            self.equiv_classes[features].add(h)

            # Check equivalence shortcut
            if features in self.equiv_outcomes:
                eq_val = self.equiv_outcomes[features]
                if eq_val is not None:
                    self.solved[h] = eq_val
                    contacts += 1
                    continue

            # EVALUATE some frontier positions directly (like touching syzygy boundary)
            # This is crucial - we need to actually measure strategies to learn
            if random.random() < 0.5:  # 50% chance to evaluate
                efficiency = self.oracle.evaluate(state, samples_per_eval)
                self.solved[h] = efficiency
                self.stats['evaluations'] += 1
                contacts += 1
                # Add to backward frontier for propagation
                self.backward_frontier[h] = state
                self.backward_seen.add(h)
                continue

            # Generate moves
            moves = generate_meta_moves(state)

            for next_state in moves[:10]:  # Limit branching
                nh = hash(next_state)

                if nh in self.forward_seen:
                    continue

                self.forward_seen.add(nh)
                self.all_states[nh] = next_state

                # Check if backward wave knows this
                if nh in self.backward_seen and nh in self.solved:
                    contacts += 1
                    continue

                next_frontier[nh] = next_state

        self.forward_frontier = next_frontier
        return contacts

    def _expand_backward(self) -> int:
        """Expand backward from evaluated strategies"""
        items = list(self.backward_frontier.items())
        if not items:
            return 0

        next_frontier = {}
        contacts = 0

        for h, state in items[:10]:
            # Generate predecessor strategies (what could lead to this?)
            # In meta-game, these are strategies that are "one step away"
            moves = generate_meta_moves(state)

            for pred_state in moves[:5]:
                ph = hash(pred_state)

                if ph in self.backward_seen:
                    continue

                self.backward_seen.add(ph)

                # Check forward wave
                if ph in self.forward_seen:
                    contacts += 1
                    self.stats['connections'] += 1

                next_frontier[ph] = pred_state

        self.backward_frontier = next_frontier
        return contacts

    def _find_connections(self) -> int:
        """Find where forward and backward meta-waves meet"""
        overlap = self.forward_seen & self.backward_seen

        new_connections = 0
        for h in overlap:
            if h in self.solved:
                continue

            # Need to evaluate this connection point
            state = self.forward_frontier.get(h) or self.backward_frontier.get(h)
            if state:
                # Actually evaluate it
                efficiency = self.oracle.evaluate(state, 50)
                self.solved[h] = efficiency
                self.stats['evaluations'] += 1
                new_connections += 1

        return new_connections

    def _propagate_equivalence(self) -> int:
        """Propagate solutions via equivalence classes"""
        total = 0

        # Update equivalence outcomes
        for features, hashes in self.equiv_classes.items():
            values = [self.solved[h] for h in hashes if h in self.solved]
            if values:
                # Use average for continuous efficiency values
                avg = sum(values) / len(values)
                std = (sum((v - avg)**2 for v in values) / len(values)) ** 0.5

                # Only propagate if consistent (low variance)
                if std < 0.1 * avg:
                    self.equiv_outcomes[features] = avg

        # Propagate to unsolved
        for features, outcome in self.equiv_outcomes.items():
            if outcome is None:
                continue
            for h in self.equiv_classes[features]:
                if h not in self.solved:
                    self.solved[h] = outcome
                    total += 1

        return total


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HOLOS Meta-Game: Search for optimal strategies")
    parser.add_argument('--material', default='KQRRvKQR', help='7-piece material')
    parser.add_argument('--iterations', type=int, default=15, help='Meta-search iterations')
    parser.add_argument('--samples', type=int, default=50, help='Samples per evaluation')
    args = parser.parse_args()

    solver = BidirectionalMetaHOLOS()
    best_state, best_efficiency = solver.solve(
        material_7=args.material,
        max_iterations=args.iterations,
        samples_per_eval=args.samples
    )

    if best_state:
        print(f"\n{'='*60}")
        print("RECOMMENDED STRATEGY")
        print(f"{'='*60}")
        print(f"Strategy type: {best_state.strategy_type.value}")
        print(f"Number of seeds: {best_state.num_seeds}")
        print(f"Connectivity weight: {best_state.connectivity_weight:.2f}")
        print(f"Coverage weight: {best_state.coverage_weight:.2f}")
        print(f"Efficiency score: {best_efficiency:.2f}")


if __name__ == '__main__':
    main()
