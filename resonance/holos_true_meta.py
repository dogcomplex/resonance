"""
holos_true_meta.py - TRUE Meta-Game: Raw Point Cloud Optimization

The REAL meta-game is not "which strategy type?" but rather:
- Which SPECIFIC boundary positions to seed?
- How far should each seed search?
- What's the optimal point cloud distribution?

This is a PACKING PROBLEM:
- We have a fixed budget (compute/storage)
- We want to maximize coverage of the 8-piece space
- Seeds compete for coverage (overlap is waste)
- Different seeds have different "reach" (connectivity)

NO STRATEGY LABELS - just raw optimization of:
1. Seed selection (which 7-piece positions)
2. Search depth per seed
3. Total seed count vs depth tradeoff

The meta-game state IS the seed configuration itself.
"""

import os
import random
import time
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
import hashlib

from fractal_holos3 import (
    SyzygyProbe, ChessState, random_position, generate_predecessors,
    generate_moves, apply_move, extract_features
)


# ============================================================
# TRUE META-STATE: A Point Cloud of Seeds
# ============================================================

@dataclass
class SeedConfig:
    """
    A single seed in the point cloud.

    This is a specific 7-piece position with its search parameters.
    """
    position: ChessState      # The actual 7-piece boundary position
    search_depth: int         # How many predecessor layers to explore (1-5)

    # Measured properties (filled in after evaluation)
    coverage: int = 0         # How many 8-piece positions this seed reaches
    unique_coverage: int = 0  # Coverage not overlapping with other seeds

    def __hash__(self):
        return hash((hash(self.position), self.search_depth))


@dataclass
class PointCloud:
    """
    The TRUE meta-game state: a specific configuration of seeds.

    This is NOT "greedy" or "random" - it's a specific set of positions
    with specific search depths. The meta-game searches over these.
    """
    seeds: Tuple[SeedConfig, ...]  # Immutable tuple of seeds
    material: str                   # What material config (e.g., "KQRRvKQR")

    # Computed metrics
    total_coverage: int = 0
    total_cost: int = 0  # Compute cost (sum of depths)
    efficiency: float = 0.0  # coverage / cost

    def __hash__(self):
        # Hash based on seed positions and depths
        seed_hashes = tuple(sorted(hash(s) for s in self.seeds))
        return hash((seed_hashes, self.material))

    def signature(self) -> str:
        """Compact representation for debugging"""
        return f"Cloud({len(self.seeds)} seeds, depth_sum={sum(s.search_depth for s in self.seeds)})"


# ============================================================
# POINT CLOUD MOVES (How to modify a cloud)
# ============================================================

def mutate_cloud(cloud: PointCloud, syzygy: SyzygyProbe,
                 mutation_type: str = None) -> Optional[PointCloud]:
    """
    Generate a neighboring point cloud by mutation.

    Mutations:
    - add_seed: Add a new random seed
    - remove_seed: Remove a seed
    - swap_seed: Replace one seed with another
    - adjust_depth: Change a seed's search depth
    - split_seed: Replace one deep seed with two shallow ones
    - merge_seeds: Replace two shallow seeds with one deep one
    """
    if mutation_type is None:
        mutation_type = random.choice([
            'add_seed', 'remove_seed', 'swap_seed',
            'adjust_depth', 'split_seed', 'merge_seeds'
        ])

    seeds = list(cloud.seeds)

    if mutation_type == 'add_seed' and len(seeds) < 100:
        # Add a new random seed
        new_pos = random_position(cloud.material)
        if new_pos and syzygy.probe(new_pos) is not None:
            depth = random.randint(1, 3)
            seeds.append(SeedConfig(new_pos, depth))

    elif mutation_type == 'remove_seed' and len(seeds) > 1:
        # Remove a random seed (prefer low-coverage ones)
        idx = random.randint(0, len(seeds) - 1)
        seeds.pop(idx)

    elif mutation_type == 'swap_seed' and len(seeds) > 0:
        # Replace one seed with a new random one
        idx = random.randint(0, len(seeds) - 1)
        new_pos = random_position(cloud.material)
        if new_pos and syzygy.probe(new_pos) is not None:
            seeds[idx] = SeedConfig(new_pos, seeds[idx].search_depth)

    elif mutation_type == 'adjust_depth' and len(seeds) > 0:
        # Change a seed's search depth
        idx = random.randint(0, len(seeds) - 1)
        delta = random.choice([-1, 1])
        new_depth = max(1, min(5, seeds[idx].search_depth + delta))
        seeds[idx] = SeedConfig(seeds[idx].position, new_depth)

    elif mutation_type == 'split_seed' and len(seeds) > 0:
        # Replace one deep seed with two shallow ones
        idx = random.randint(0, len(seeds) - 1)
        if seeds[idx].search_depth >= 2:
            old_seed = seeds.pop(idx)
            new_depth = max(1, old_seed.search_depth - 1)
            seeds.append(SeedConfig(old_seed.position, new_depth))

            new_pos = random_position(cloud.material)
            if new_pos and syzygy.probe(new_pos) is not None:
                seeds.append(SeedConfig(new_pos, new_depth))

    elif mutation_type == 'merge_seeds' and len(seeds) >= 2:
        # Merge two shallow seeds into one deeper one
        idx1 = random.randint(0, len(seeds) - 1)
        idx2 = random.randint(0, len(seeds) - 1)
        if idx1 != idx2:
            s1 = seeds[min(idx1, idx2)]
            # Remove both, keep the one with higher coverage
            seeds = [s for i, s in enumerate(seeds) if i not in (idx1, idx2)]
            new_depth = min(5, s1.search_depth + 1)
            seeds.append(SeedConfig(s1.position, new_depth))

    if not seeds:
        return None

    return PointCloud(tuple(seeds), cloud.material)


def crossover_clouds(cloud1: PointCloud, cloud2: PointCloud) -> PointCloud:
    """
    Combine two point clouds (genetic crossover).
    Take seeds from both parents.
    """
    all_seeds = list(cloud1.seeds) + list(cloud2.seeds)

    # Take a random subset
    n = random.randint(
        min(len(cloud1.seeds), len(cloud2.seeds)),
        len(cloud1.seeds) + len(cloud2.seeds)
    )
    n = min(n, 50)  # Cap at 50 seeds

    selected = random.sample(all_seeds, min(n, len(all_seeds)))
    return PointCloud(tuple(selected), cloud1.material)


# ============================================================
# POINT CLOUD EVALUATION
# ============================================================

class CloudEvaluator:
    """
    Evaluate a point cloud's coverage and efficiency.

    This is the "oracle" of the true meta-game.
    """

    def __init__(self, syzygy: SyzygyProbe, cache_path: str = "./cloud_eval_cache.pkl"):
        self.syzygy = syzygy
        self.cache_path = cache_path
        self.cache: Dict[int, Tuple[int, int, float]] = {}  # hash -> (coverage, cost, efficiency)
        self.evaluations = 0
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"[CloudEvaluator] Loaded {len(self.cache)} cached evaluations")
            except:
                self.cache = {}

    def _save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def evaluate(self, cloud: PointCloud) -> Tuple[int, int, float]:
        """
        Evaluate a point cloud's coverage.

        Returns: (total_coverage, total_cost, efficiency)
        """
        h = hash(cloud)
        if h in self.cache:
            return self.cache[h]

        self.evaluations += 1

        # Compute coverage for each seed
        all_covered: Set[int] = set()
        total_cost = 0

        for seed in cloud.seeds:
            seed_covered = self._expand_seed(seed)
            seed.coverage = len(seed_covered)
            seed.unique_coverage = len(seed_covered - all_covered)
            all_covered |= seed_covered
            total_cost += seed.search_depth

        total_coverage = len(all_covered)
        efficiency = total_coverage / total_cost if total_cost > 0 else 0

        # Update cloud metrics
        cloud.total_coverage = total_coverage
        cloud.total_cost = total_cost
        cloud.efficiency = efficiency

        # Cache result
        result = (total_coverage, total_cost, efficiency)
        self.cache[h] = result

        if self.evaluations % 50 == 0:
            self._save_cache()

        return result

    def _expand_seed(self, seed: SeedConfig) -> Set[int]:
        """
        Expand from a seed position to find reachable 8-piece positions.

        search_depth controls how many layers of predecessors to explore:
        - depth 1: direct predecessors only
        - depth 2: predecessors of predecessors
        - etc.
        """
        covered = set()
        frontier = [seed.position]

        for depth in range(seed.search_depth):
            next_frontier = []
            for pos in frontier:
                preds = generate_predecessors(pos, max_uncaptures=3)
                for pred in preds:
                    if pred.piece_count() == 8:
                        covered.add(hash(pred))
                    elif pred.piece_count() == 7:
                        # Can continue expanding from 7-piece positions
                        next_frontier.append(pred)
            frontier = next_frontier[:100]  # Limit frontier size

        return covered


# ============================================================
# EVOLUTIONARY SEARCH OVER POINT CLOUDS
# ============================================================

class PointCloudEvolver:
    """
    Evolutionary search to find optimal point cloud configurations.

    NO predefined strategies - just raw mutation and selection.
    """

    def __init__(self, syzygy_path: str = "./syzygy"):
        self.syzygy = SyzygyProbe(syzygy_path)
        self.evaluator = CloudEvaluator(self.syzygy)

        # Population
        self.population: List[PointCloud] = []
        self.best_cloud: Optional[PointCloud] = None
        self.best_efficiency: float = 0

        # History for analysis
        self.history: List[Tuple[int, float, int, int]] = []  # (gen, best_eff, best_cov, best_cost)

    def initialize_population(self, material: str, pop_size: int = 20,
                              seeds_per_cloud: int = 10):
        """Create initial random population"""
        print(f"Initializing population of {pop_size} clouds...")

        for i in range(pop_size):
            seeds = []
            for _ in range(seeds_per_cloud):
                pos = random_position(material)
                if pos and self.syzygy.probe(pos) is not None:
                    depth = random.randint(1, 3)
                    seeds.append(SeedConfig(pos, depth))

            if seeds:
                cloud = PointCloud(tuple(seeds), material)
                self.population.append(cloud)

        print(f"Created {len(self.population)} initial clouds")

    def evolve(self, generations: int = 50, pop_size: int = 20,
               material: str = "KQRRvKQR",
               elite_count: int = 3,
               mutation_rate: float = 0.7):
        """
        Run evolutionary search.

        This is HOLOS applied to point cloud search:
        - Lightning: Random mutations explore new regions
        - Crystal: Best configurations get refined
        - Propagation: Good genes spread through crossover
        """
        print("=" * 60)
        print("POINT CLOUD EVOLUTION - True Meta-Game")
        print("=" * 60)
        print(f"Material: {material}")
        print(f"Population: {pop_size}, Generations: {generations}")
        print()

        if not self.population:
            self.initialize_population(material, pop_size)

        for gen in range(generations):
            # Evaluate all clouds
            scored = []
            for cloud in self.population:
                coverage, cost, efficiency = self.evaluator.evaluate(cloud)
                scored.append((efficiency, coverage, cost, cloud))

            # Sort by efficiency (could also optimize for coverage or balanced)
            scored.sort(reverse=True, key=lambda x: x[0])

            # Track best
            best_eff, best_cov, best_cost, best_cloud = scored[0]
            if best_eff > self.best_efficiency:
                self.best_efficiency = best_eff
                self.best_cloud = best_cloud

            self.history.append((gen, best_eff, best_cov, best_cost))

            # Report
            avg_eff = sum(s[0] for s in scored) / len(scored)
            avg_seeds = sum(len(s[3].seeds) for s in scored) / len(scored)
            print(f"Gen {gen:3d}: best_eff={best_eff:6.2f}, avg_eff={avg_eff:5.2f}, "
                  f"best_cov={best_cov:5d}, cost={best_cost:3d}, avg_seeds={avg_seeds:.1f}")

            # Selection: keep elites
            elites = [cloud for _, _, _, cloud in scored[:elite_count]]

            # Generate new population
            new_population = list(elites)

            while len(new_population) < pop_size:
                if random.random() < mutation_rate:
                    # Mutation: take a good cloud and mutate it
                    parent_idx = random.randint(0, min(len(scored)-1, pop_size//2))
                    parent = scored[parent_idx][3]
                    child = mutate_cloud(parent, self.syzygy)
                    if child:
                        new_population.append(child)
                else:
                    # Crossover: combine two good clouds
                    p1_idx = random.randint(0, min(len(scored)-1, pop_size//2))
                    p2_idx = random.randint(0, min(len(scored)-1, pop_size//2))
                    child = crossover_clouds(scored[p1_idx][3], scored[p2_idx][3])
                    new_population.append(child)

            self.population = new_population[:pop_size]

        # Final report
        print()
        print("=" * 60)
        print("EVOLUTION COMPLETE")
        print("=" * 60)

        if self.best_cloud:
            print(f"\nBest Point Cloud Found:")
            print(f"  Seeds: {len(self.best_cloud.seeds)}")
            print(f"  Total coverage: {self.best_cloud.total_coverage}")
            print(f"  Total cost: {self.best_cloud.total_cost}")
            print(f"  Efficiency: {self.best_efficiency:.2f}")

            # Analyze depth distribution
            depths = [s.search_depth for s in self.best_cloud.seeds]
            print(f"\n  Depth distribution:")
            for d in sorted(set(depths)):
                count = depths.count(d)
                print(f"    Depth {d}: {count} seeds ({100*count/len(depths):.0f}%)")

            # Show improvement curve
            print(f"\n  Evolution curve (efficiency):")
            for i in range(0, len(self.history), max(1, len(self.history)//10)):
                gen, eff, cov, cost = self.history[i]
                bar = '#' * int(eff)
                print(f"    Gen {gen:3d}: {eff:6.2f} {bar}")

        return self.best_cloud, self.best_efficiency


# ============================================================
# COMPARISON WITH PREDEFINED STRATEGIES
# ============================================================

def compare_evolved_vs_predefined(material: str = "KQRRvKQR"):
    """
    Compare evolved point clouds against predefined strategies.

    This tests whether raw evolution finds something better than
    "greedy" or "random" - which were premature abstractions.
    """
    print("=" * 60)
    print("COMPARISON: Evolved Point Clouds vs Predefined Strategies")
    print("=" * 60)

    syzygy = SyzygyProbe("./syzygy")
    evaluator = CloudEvaluator(syzygy)

    # 1. Create predefined "greedy coverage" cloud
    print("\n1. Building 'Greedy Coverage' cloud...")
    greedy_seeds = []
    covered = set()
    candidates = []

    for _ in range(500):
        pos = random_position(material)
        if pos and syzygy.probe(pos) is not None:
            preds = generate_predecessors(pos, max_uncaptures=3)
            pred_hashes = {hash(p) for p in preds if p.piece_count() == 8}
            candidates.append((len(pred_hashes - covered), pos, pred_hashes))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for score, pos, pred_hashes in candidates[:20]:
        greedy_seeds.append(SeedConfig(pos, search_depth=2))
        covered |= pred_hashes

    greedy_cloud = PointCloud(tuple(greedy_seeds), material)
    greedy_cov, greedy_cost, greedy_eff = evaluator.evaluate(greedy_cloud)
    print(f"   Greedy: coverage={greedy_cov}, cost={greedy_cost}, efficiency={greedy_eff:.2f}")

    # 2. Create random cloud
    print("\n2. Building 'Random' cloud...")
    random_seeds = []
    for _ in range(20):
        pos = random_position(material)
        if pos and syzygy.probe(pos) is not None:
            random_seeds.append(SeedConfig(pos, search_depth=2))

    random_cloud = PointCloud(tuple(random_seeds), material)
    random_cov, random_cost, random_eff = evaluator.evaluate(random_cloud)
    print(f"   Random: coverage={random_cov}, cost={random_cost}, efficiency={random_eff:.2f}")

    # 3. Evolve a cloud
    print("\n3. Evolving point cloud...")
    evolver = PointCloudEvolver()
    evolved_cloud, evolved_eff = evolver.evolve(
        generations=30,
        pop_size=15,
        material=material
    )

    if evolved_cloud:
        evolved_cov = evolved_cloud.total_coverage
        evolved_cost = evolved_cloud.total_cost
    else:
        evolved_cov = evolved_cost = 0

    # 4. Summary
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"\n{'Strategy':<15} {'Coverage':>10} {'Cost':>8} {'Efficiency':>12}")
    print("-" * 50)
    print(f"{'Greedy':<15} {greedy_cov:>10} {greedy_cost:>8} {greedy_eff:>12.2f}")
    print(f"{'Random':<15} {random_cov:>10} {random_cost:>8} {random_eff:>12.2f}")
    print(f"{'Evolved':<15} {evolved_cov:>10} {evolved_cost:>8} {evolved_eff:>12.2f}")

    # Improvement
    if greedy_eff > 0:
        improvement = (evolved_eff - greedy_eff) / greedy_eff * 100
        print(f"\nEvolved vs Greedy: {improvement:+.1f}% efficiency")

    return evolved_cloud


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="True Meta-Game: Point Cloud Evolution")
    parser.add_argument('--material', default='KQRRvKQR', help='7-piece material')
    parser.add_argument('--generations', type=int, default=50, help='Evolution generations')
    parser.add_argument('--pop-size', type=int, default=20, help='Population size')
    parser.add_argument('--compare', action='store_true', help='Compare evolved vs predefined')
    args = parser.parse_args()

    if args.compare:
        compare_evolved_vs_predefined(args.material)
    else:
        evolver = PointCloudEvolver()
        best_cloud, best_eff = evolver.evolve(
            generations=args.generations,
            pop_size=args.pop_size,
            material=args.material
        )


if __name__ == '__main__':
    main()
