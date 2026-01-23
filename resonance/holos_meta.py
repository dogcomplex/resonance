"""
holos_meta.py - The Meta-Game Layer for HOLOS

This implements the SET COVER optimization:
- Which boundary positions to seed for maximum coverage?
- How to minimize overlap between seed expansions?
- The meta-game of choosing WHERE to search

The insight: We're not compressing the SOLUTION, we're compressing the SEARCH.
"""

import os
import sys
import time
import pickle
import random
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass, field

from fractal_holos3 import (
    ChessState, Piece, SyzygyProbe,
    generate_predecessors, generate_moves, apply_move,
    random_position, extract_features, is_white, piece_type,
    PIECE_CHARS
)


@dataclass
class SeedAnalysis:
    """Analysis of a boundary seed position"""
    state: ChessState
    value: int
    predecessor_count: int
    predecessor_hashes: Set[int]

    # Connectivity metrics
    unique_materials: Set[Tuple]  # Material configs of predecessors
    avg_king_distance: float


@dataclass
class CoverageResult:
    """Result of coverage analysis"""
    seeds_used: int
    positions_covered: int
    coverage_per_seed: float
    overlap_ratio: float  # How much overlap between seeds
    material_diversity: int


def analyze_seed(state: ChessState, value: int, max_preds: int = 100) -> SeedAnalysis:
    """Analyze a single boundary seed for its coverage potential"""
    preds = generate_predecessors(state, max_uncaptures=5)

    pred_hashes = set()
    materials = set()
    king_dists = []

    for pred in preds[:max_preds]:
        if pred.piece_count() != 8:
            continue

        pred_hashes.add(hash(pred))

        # Extract material signature
        white_mat = tuple(sorted(piece_type(p) for p, sq in pred.pieces
                                  if is_white(p) and piece_type(p) != 1))
        materials.add(white_mat)

        # King distance
        wk = bk = None
        for p, sq in pred.pieces:
            if piece_type(p) == 1:
                if is_white(p): wk = sq
                else: bk = sq
        if wk and bk:
            kd = abs(wk // 8 - bk // 8) + abs(wk % 8 - bk % 8)
            king_dists.append(kd)

    return SeedAnalysis(
        state=state,
        value=value,
        predecessor_count=len(pred_hashes),
        predecessor_hashes=pred_hashes,
        unique_materials=materials,
        avg_king_distance=sum(king_dists) / len(king_dists) if king_dists else 0
    )


def greedy_set_cover(seeds: List[SeedAnalysis], target_coverage: int = None) -> List[SeedAnalysis]:
    """
    Greedy SET COVER: Select seeds that maximize NEW coverage.

    This is the meta-game: choosing boundary positions to minimize
    total seeds needed for a given coverage.
    """
    covered = set()
    selected = []
    remaining = list(seeds)

    while remaining:
        # Find seed that covers most NEW positions
        best_seed = None
        best_new_coverage = 0

        for seed in remaining:
            new_coverage = len(seed.predecessor_hashes - covered)
            if new_coverage > best_new_coverage:
                best_new_coverage = new_coverage
                best_seed = seed

        if best_seed is None or best_new_coverage == 0:
            break

        selected.append(best_seed)
        covered.update(best_seed.predecessor_hashes)
        remaining.remove(best_seed)

        if target_coverage and len(covered) >= target_coverage:
            break

    return selected


def random_selection(seeds: List[SeedAnalysis], count: int) -> List[SeedAnalysis]:
    """Random baseline for comparison"""
    return random.sample(seeds, min(count, len(seeds)))


def analyze_coverage_strategy(seeds: List[SeedAnalysis],
                              strategy: str = "greedy",
                              num_seeds: int = 50) -> CoverageResult:
    """Compare coverage strategies"""

    if strategy == "greedy":
        selected = greedy_set_cover(seeds)[:num_seeds]
    elif strategy == "random":
        selected = random_selection(seeds, num_seeds)
    elif strategy == "high_connectivity":
        # Sort by predecessor count, take top
        sorted_seeds = sorted(seeds, key=lambda s: -s.predecessor_count)
        selected = sorted_seeds[:num_seeds]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Calculate coverage
    all_covered = set()
    overlaps = 0

    for seed in selected:
        overlap = len(seed.predecessor_hashes & all_covered)
        overlaps += overlap
        all_covered.update(seed.predecessor_hashes)

    total_from_seeds = sum(s.predecessor_count for s in selected)
    overlap_ratio = overlaps / total_from_seeds if total_from_seeds > 0 else 0

    # Material diversity
    all_materials = set()
    for seed in selected:
        all_materials.update(seed.unique_materials)

    return CoverageResult(
        seeds_used=len(selected),
        positions_covered=len(all_covered),
        coverage_per_seed=len(all_covered) / len(selected) if selected else 0,
        overlap_ratio=overlap_ratio,
        material_diversity=len(all_materials)
    )


def run_meta_game_analysis(material_7: str = "KQRRvKQR",
                           num_boundary: int = 500,
                           syzygy_path: str = "./syzygy"):
    """
    Run the meta-game analysis: compare seeding strategies.
    """
    print("=" * 60)
    print("META-GAME ANALYSIS: Optimizing Boundary Seeding")
    print("=" * 60)

    syzygy = SyzygyProbe(syzygy_path)
    if not syzygy.available:
        print("ERROR: Syzygy not available")
        return

    # Collect boundary positions
    print(f"\nPhase 1: Collecting {num_boundary} boundary seeds...")
    boundary = {}
    attempts = 0
    while len(boundary) < num_boundary and attempts < num_boundary * 20:
        attempts += 1
        state = random_position(material_7)
        if state is None:
            continue
        h = hash(state)
        if h in boundary:
            continue
        value = syzygy.probe(state)
        if value is not None:
            boundary[h] = (state, value)

    print(f"  Collected: {len(boundary)}")

    # Analyze each seed
    print(f"\nPhase 2: Analyzing seed connectivity...")
    seeds = []
    for i, (h, (state, value)) in enumerate(boundary.items()):
        seed = analyze_seed(state, value)
        seeds.append(seed)
        if (i + 1) % 100 == 0:
            print(f"  Analyzed {i + 1}/{len(boundary)}")

    # Stats
    avg_preds = sum(s.predecessor_count for s in seeds) / len(seeds)
    max_preds = max(s.predecessor_count for s in seeds)
    min_preds = min(s.predecessor_count for s in seeds)

    print(f"\nSeed Statistics:")
    print(f"  Avg predecessors: {avg_preds:.1f}")
    print(f"  Min: {min_preds}, Max: {max_preds}")

    # Compare strategies
    print(f"\nPhase 3: Comparing seeding strategies...")
    print("-" * 60)

    for num_seeds in [10, 25, 50, 100]:
        if num_seeds > len(seeds):
            continue

        print(f"\nUsing {num_seeds} seeds:")

        for strategy in ["random", "high_connectivity", "greedy"]:
            # Run multiple times for random
            if strategy == "random":
                results = []
                for _ in range(10):
                    r = analyze_coverage_strategy(seeds, strategy, num_seeds)
                    results.append(r)
                avg_coverage = sum(r.positions_covered for r in results) / len(results)
                avg_per_seed = sum(r.coverage_per_seed for r in results) / len(results)
                avg_overlap = sum(r.overlap_ratio for r in results) / len(results)
                print(f"  {strategy:20} | Coverage: {avg_coverage:6.0f} | Per seed: {avg_per_seed:5.1f} | Overlap: {avg_overlap:.1%}")
            else:
                r = analyze_coverage_strategy(seeds, strategy, num_seeds)
                print(f"  {strategy:20} | Coverage: {r.positions_covered:6.0f} | Per seed: {r.coverage_per_seed:5.1f} | Overlap: {r.overlap_ratio:.1%}")

    # The meta-insight
    print("\n" + "=" * 60)
    print("META-GAME INSIGHT")
    print("=" * 60)
    print("""
The greedy SET COVER strategy should outperform random selection.

If coverage_per_seed(greedy) >> coverage_per_seed(random):
  -> The meta-game has STRUCTURE we can exploit
  -> Solving the meta-game compresses the search

If they're similar:
  -> The space is uniformly connected
  -> No meta-game advantage (but still works)

The FRACTAL insight: This same analysis applies at EVERY level.
- Which 8-piece positions to seed for 9-piece coverage?
- Which 9-piece for 10-piece?
- ...all the way up
""")

    return seeds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', default='KQRRvKQR')
    parser.add_argument('--seeds', type=int, default=500)
    args = parser.parse_args()

    run_meta_game_analysis(args.material, args.seeds)
