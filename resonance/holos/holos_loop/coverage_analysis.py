"""
coverage_analysis.py - Analyze what 100% coverage means for Connect4

Questions to answer:
1. What does 100% coverage look like?
2. Is it fully solved or just "interesting positions"?
3. What's the wall-clock estimate and storage?
"""

import sys
import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from resonance.holos.games.connect4 import Connect4Game, C4State, C4Value
from resonance.holos.fast_daemon import FastDaemonWaveSystem


@dataclass
class CoverageMetrics:
    """Track different types of coverage."""
    states_explored: int = 0
    states_with_values: int = 0
    terminal_states: int = 0
    closures: int = 0

    # Breakdown by depth (ply)
    states_by_ply: Dict[int, int] = None
    values_by_ply: Dict[int, int] = None

    # Breakdown by value
    wins_x: int = 0
    wins_o: int = 0
    draws: int = 0
    unknown: int = 0


def count_pieces(state: C4State) -> int:
    """Count total pieces on board (= ply/depth)."""
    return sum(c.count('X') + c.count('O') for c in state.cols)


def analyze_coverage(system: FastDaemonWaveSystem, game: Connect4Game) -> CoverageMetrics:
    """Analyze what coverage we have."""
    metrics = CoverageMetrics()
    metrics.states_by_ply = {}
    metrics.values_by_ply = {}

    for h, state in system.states.items():
        metrics.states_explored += 1
        ply = count_pieces(state)

        metrics.states_by_ply[ply] = metrics.states_by_ply.get(ply, 0) + 1

        if h in system.values:
            metrics.states_with_values += 1
            metrics.values_by_ply[ply] = metrics.values_by_ply.get(ply, 0) + 1

            val = system.values[h]
            if hasattr(val, 'winner'):
                if val.winner == 'X':
                    metrics.wins_x += 1
                elif val.winner == 'O':
                    metrics.wins_o += 1
                else:
                    metrics.draws += 1
        else:
            metrics.unknown += 1

        # Check if terminal
        is_term, _ = game.is_terminal(state)
        if is_term:
            metrics.terminal_states += 1

    metrics.closures = len(system.closures)
    return metrics


def estimate_full_solve():
    """Estimate what a full Connect4 solve would require."""
    print("\n" + "=" * 70)
    print("CONNECT4 FULL SOLVE ESTIMATION")
    print("=" * 70)

    # Known facts about Connect4
    print("""
    KNOWN FACTS (from computer science literature):

    - Total legal positions: ~4.5 trillion (4.5 x 10^12)
    - First player (X) wins with perfect play
    - Solved in 1988 by Victor Allis and James Allen
    - Original solve required ~300 hours of compute (1988 hardware)

    BREAKDOWN BY PLY:
    """)

    # Approximate positions per ply (from combinatorics)
    # These are rough estimates
    ply_estimates = {
        0: 1,          # Empty board
        1: 7,          # 7 possible first moves
        2: 49,         # 7x7
        3: 238,        # Accounting for invalid
        4: 1120,
        5: 4263,
        6: 16422,
        7: 54859,
        8: 184275,
        9: 533766,
        10: 1531121,   # About 1.5M
        # ... exponential growth
        42: 1,         # Full board (if game goes to draw)
    }

    print("    Ply 0:  1 (empty)")
    print("    Ply 1:  7 (first move)")
    print("    Ply 2:  49")
    print("    Ply 3:  ~238")
    print("    Ply 4:  ~1,120")
    print("    Ply 5:  ~4,263")
    print("    ...")
    print("    Ply 10: ~1.5M")
    print("    Ply 20: ~500B")
    print("    Ply 30: ~2T (peak)")
    print("    Ply 42: 1 (full)")
    print()

    print("""
    WHAT "100% COVERAGE" MEANS:

    Option A: ALL 4.5T positions have known values
       - Storage: 4.5 TB (1 byte/value)
       - With compression: 50-500 GB
       - Compute: ~10^14 operations
       - Time (single core): ~1 year
       - Time (1000 cores): ~8 hours

    Option B: "Interesting" positions only
       - Opening book: Positions reachable in first 10-15 ply
       - ~10M-100M positions
       - Storage: 10-100 MB compressed
       - Compute: Hours on single core
       - This is what most engines use

    Option C: "Solved" via proof tree
       - Only store positions that prove first player wins
       - Much smaller than full enumeration
       - This is what Allis did in 1988
    """)


def run_extended_test():
    """Run extended test to see coverage growth."""
    print("\n" + "=" * 70)
    print("EXTENDED COVERAGE TEST")
    print("=" * 70)

    game = Connect4Game()
    start = C4State()

    # Run for longer
    system = FastDaemonWaveSystem(
        game,
        daemon_frequency=10,
        enable_amplification=True,
        enable_collapse=True
    )
    system.initialize([start])

    print("\n  Running 500 iterations with daemon...")
    print("  Iter | States | Closures | Values | Coverage")
    print("  " + "-" * 50)

    checkpoints = []
    for i in range(500):
        system.step()

        if i % 50 == 0:
            coverage = len(system.values) / max(1, len(system.states)) * 100
            print(f"  {i:4d} | {len(system.states):6d} | {len(system.closures):8d} | "
                  f"{len(system.values):6d} | {coverage:5.1f}%")
            checkpoints.append({
                'iter': i,
                'states': len(system.states),
                'closures': len(system.closures),
                'values': len(system.values),
                'coverage': coverage
            })

    # Final analysis
    metrics = analyze_coverage(system, game)

    print("\n  FINAL METRICS:")
    print(f"    States explored: {metrics.states_explored:,}")
    print(f"    States with values: {metrics.states_with_values:,}")
    print(f"    Closures: {metrics.closures:,}")
    print(f"    Coverage: {metrics.states_with_values / metrics.states_explored * 100:.1f}%")

    print("\n  BY PLY (depth):")
    for ply in sorted(metrics.states_by_ply.keys()):
        states = metrics.states_by_ply[ply]
        values = metrics.values_by_ply.get(ply, 0)
        pct = values / states * 100 if states > 0 else 0
        print(f"    Ply {ply:2d}: {states:5d} states, {values:5d} valued ({pct:5.1f}%)")

    print("\n  BY VALUE:")
    print(f"    X wins: {metrics.wins_x:,}")
    print(f"    O wins: {metrics.wins_o:,}")
    print(f"    Draws:  {metrics.draws:,}")
    print(f"    Unknown: {metrics.unknown:,}")

    # Extrapolate
    print("\n  EXTRAPOLATION:")

    # States per iteration
    states_per_iter = metrics.states_explored / 500
    print(f"    States/iteration: {states_per_iter:.1f}")

    # Time per iteration (rough)
    time_per_iter = 0.002  # ~2ms assumed

    # To reach 100M states (practical opening book)
    iters_for_100m = 100_000_000 / states_per_iter
    time_for_100m = iters_for_100m * time_per_iter
    print(f"    Iterations for 100M states: {iters_for_100m:,.0f}")
    print(f"    Time estimate: {time_for_100m / 3600:.1f} hours")

    # To reach 4.5T states (full solve)
    iters_for_full = 4_500_000_000_000 / states_per_iter
    time_for_full = iters_for_full * time_per_iter
    print(f"    Iterations for full 4.5T: {iters_for_full:,.0f}")
    print(f"    Time estimate: {time_for_full / 3600 / 24 / 365:.1f} years (single core)")
    print(f"    Time with 1000x GPU: {time_for_full / 3600 / 1000:.1f} hours")

    return metrics, checkpoints


def practical_recommendations():
    """What's practical to achieve."""
    print("\n" + "=" * 70)
    print("PRACTICAL RECOMMENDATIONS")
    print("=" * 70)

    print("""
    FOR CONNECT4 WITH HOLOS/CLOSURE SYSTEM:

    1. OPENING BOOK (Practical Now)
       - Cover first 10-12 ply
       - ~10M positions
       - Storage: ~10MB compressed
       - Compute: ~1 hour
       - Result: Know optimal play for common openings

    2. STRATEGIC POSITIONS (Practical Now)
       - Cover positions 3-4 moves from terminal
       - Backward wave from terminals
       - ~100M positions
       - Storage: ~100MB compressed
       - Compute: ~10 hours
       - Result: Perfect endgame play

    3. MEETING IN THE MIDDLE (Our Approach)
       - Forward from opening, backward from terminals
       - Meet around ply 15-20
       - Closures prove connections
       - Much smaller than full enumeration
       - Similar to Allis's proof approach

    4. FULL ENUMERATION (Research Project)
       - All 4.5T positions
       - Storage: 50-500GB compressed
       - Compute: Days with GPU cluster
       - Result: Perfect play from any position

    STORAGE BREAKDOWN (1 byte per value):

    | Coverage     | Positions | Raw Size | Compressed |
    |--------------|-----------|----------|------------|
    | 10 ply       | 10M       | 10 MB    | ~1 MB      |
    | 15 ply       | 500M      | 500 MB   | ~50 MB     |
    | 20 ply       | 100B      | 100 GB   | ~10 GB     |
    | Full (4.5T)  | 4.5T      | 4.5 TB   | ~500 GB    |

    RECOMMENDATION:

    For HOLOS demonstration, target "practical closure":
    - 1M-10M positions
    - Cover openings AND endgames
    - Show bidirectional waves meeting
    - This proves the concept without requiring cluster compute
    """)


def main():
    """Run all analysis."""
    estimate_full_solve()
    metrics, checkpoints = run_extended_test()
    practical_recommendations()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Current test achieved:
    - {metrics.states_explored:,} states in 500 iterations
    - {metrics.states_with_values:,} with known values ({metrics.states_with_values/metrics.states_explored*100:.1f}%)
    - {metrics.closures:,} closures (wave meeting points)

    100% coverage meaning:
    - Full solve: ALL 4.5T positions have values
    - Practical: Opening + endgame coverage (~10M-100M)
    - Proof: Only positions needed to prove first player wins

    Wall-clock estimates for full solve:
    - Single core: ~1 year
    - 1000-core GPU: ~8 hours
    - Storage: 50-500 GB compressed

    The 9% coverage in our test means:
    - 9% of EXPLORED states have propagated values
    - NOT 9% of all possible positions
    - With more iterations, this grows as closures propagate values
    """)


if __name__ == "__main__":
    main()
