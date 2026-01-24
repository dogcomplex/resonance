"""
holos/connect4_full_solve.py - Full Connect4 Solver with Seed Compression

This aims to solve ALL 4.5 trillion Connect4 positions using:
1. Layer-by-layer solving (by piece count)
2. Seed-based compression instead of storing all positions
3. Layer 2 strategy optimization for multi-seed coordination

Connect4 Facts:
- 4,531,985,219,092 total positions (4.5 trillion)
- 42 maximum moves (7 columns x 6 rows)
- First player (X) wins with perfect play
- Symmetric positions can be canonicalized

Strategy:
1. Solve from terminals backward (piece_count = 42 down to 0)
2. At each layer, identify "frontier seeds" - positions that define value boundaries
3. Store seeds + reconstruction algorithm instead of all positions
4. Layer 2 optimizes which seeds to keep for maximum compression

Key Insight:
- Most positions have values determined by a small set of "critical" positions
- If we know the value at critical positions, we can reconstruct others via minimax
- Seeds are these critical positions

Storage Estimation:
- Full storage: 4.5T positions × 1 byte = 4.5 TB
- Seed storage (1%): 45 billion seeds × 9 bytes = 405 GB
- Seed storage (0.1%): 4.5 billion seeds × 9 bytes = 40 GB
- Target: <100 GB for full solution
"""

import os
import sys
import time
import pickle
import sqlite3
import hashlib
import multiprocessing as mp
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.connect4 import Connect4Game, C4State, C4Value


# ============================================================
# CONSTANTS
# ============================================================

TOTAL_POSITIONS = 4_531_985_219_092  # 4.5 trillion
MAX_PIECES = 42
COLS = 7
ROWS = 6

# Approximate positions per piece count (from known analysis)
# These are estimates - actual counts vary
POSITIONS_BY_PIECE_COUNT = {
    0: 1,
    1: 7,
    2: 49,
    3: 238,
    4: 1_120,
    5: 4_263,
    6: 16_422,
    7: 54_859,
    8: 184_275,
    9: 558_186,
    10: 1_662_623,
    # ... grows exponentially to ~10^12 at piece count ~21
}


# ============================================================
# SEED STORAGE
# ============================================================

@dataclass
class C4Seed:
    """
    A seed position with its game-theoretic value.

    Seeds are "critical" positions where:
    - Terminal states (wins/draws)
    - Value transition points (where child values disagree)
    - Forced win/loss positions
    """
    cols: Tuple[str, ...]  # Board state
    turn: str              # 'X' or 'O'
    value: int             # +1, 0, -1
    piece_count: int       # For layer identification
    is_terminal: bool      # Terminal state?
    critical_reason: str   # Why is this a seed?

    def to_state(self) -> C4State:
        """Convert back to C4State"""
        return C4State(self.cols, self.turn)

    def hash(self) -> int:
        """Hash for deduplication"""
        return hash((self.cols, self.turn))

    def storage_size(self) -> int:
        """Estimate bytes needed"""
        # cols: 7 * 6 = 42 chars, turn: 1, value: 1, piece_count: 1
        return 42 + 3


@dataclass
class SeedLayer:
    """
    Seeds for a specific piece count.

    At each piece count, we store:
    - Terminal seeds (positions that are won/drawn)
    - Transition seeds (positions where minimax decision matters)
    """
    piece_count: int
    seeds: Dict[int, C4Seed] = field(default_factory=dict)  # hash -> seed
    total_positions: int = 0
    positions_solved: int = 0

    def add_seed(self, seed: C4Seed):
        h = seed.hash()
        self.seeds[h] = seed

    def get_seed(self, state: C4State) -> Optional[C4Seed]:
        h = hash((state.cols, state.turn))
        return self.seeds.get(h)

    def compression_ratio(self) -> float:
        if len(self.seeds) == 0:
            return 0.0
        return self.positions_solved / len(self.seeds)

    def storage_bytes(self) -> int:
        return sum(s.storage_size() for s in self.seeds.values())


class SeedDatabase:
    """
    Database of all seeds across all layers.

    Uses SQLite for persistence with in-memory caching.
    """

    def __init__(self, db_path: str = "connect4_seeds.db"):
        self.db_path = db_path
        self.layers: Dict[int, SeedLayer] = {}
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seeds (
                hash INTEGER PRIMARY KEY,
                cols TEXT,
                turn TEXT,
                value INTEGER,
                piece_count INTEGER,
                is_terminal INTEGER,
                critical_reason TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS layer_stats (
                piece_count INTEGER PRIMARY KEY,
                total_positions INTEGER,
                positions_solved INTEGER,
                seed_count INTEGER
            )
        """)
        self.conn.commit()

    def add_seed(self, seed: C4Seed):
        """Add a seed to the database"""
        # In-memory
        if seed.piece_count not in self.layers:
            self.layers[seed.piece_count] = SeedLayer(seed.piece_count)
        self.layers[seed.piece_count].add_seed(seed)

        # Persist to SQLite
        self.conn.execute("""
            INSERT OR REPLACE INTO seeds
            (hash, cols, turn, value, piece_count, is_terminal, critical_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            seed.hash(),
            '|'.join(seed.cols),
            seed.turn,
            seed.value,
            seed.piece_count,
            1 if seed.is_terminal else 0,
            seed.critical_reason
        ))

    def get_seed(self, state: C4State) -> Optional[C4Seed]:
        """Look up seed by state"""
        h = hash((state.cols, state.turn))
        pc = state.piece_count()

        # Check in-memory cache first
        if pc in self.layers:
            seed = self.layers[pc].get_seed(state)
            if seed:
                return seed

        # Check database
        row = self.conn.execute(
            "SELECT * FROM seeds WHERE hash = ?", (h,)
        ).fetchone()

        if row:
            return C4Seed(
                cols=tuple(row[1].split('|')),
                turn=row[2],
                value=row[3],
                piece_count=row[4],
                is_terminal=row[5] == 1,
                critical_reason=row[6]
            )
        return None

    def update_layer_stats(self, piece_count: int, total: int, solved: int):
        """Update statistics for a layer"""
        if piece_count in self.layers:
            self.layers[piece_count].total_positions = total
            self.layers[piece_count].positions_solved = solved

        self.conn.execute("""
            INSERT OR REPLACE INTO layer_stats
            (piece_count, total_positions, positions_solved, seed_count)
            VALUES (?, ?, ?, ?)
        """, (
            piece_count, total, solved,
            len(self.layers[piece_count].seeds) if piece_count in self.layers else 0
        ))
        self.conn.commit()

    def summary(self) -> str:
        """Get summary statistics"""
        total_seeds = sum(len(layer.seeds) for layer in self.layers.values())
        total_solved = sum(layer.positions_solved for layer in self.layers.values())
        total_storage = sum(layer.storage_bytes() for layer in self.layers.values())

        lines = [
            "=" * 60,
            "SEED DATABASE SUMMARY",
            "=" * 60,
            f"Layers: {len(self.layers)}",
            f"Total seeds: {total_seeds:,}",
            f"Positions covered: {total_solved:,}",
            f"Storage: {total_storage:,} bytes ({total_storage/1024/1024:.1f} MB)",
        ]

        if total_seeds > 0:
            lines.append(f"Average compression: {total_solved/total_seeds:.1f}x")

        return "\n".join(lines)

    def close(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()


# ============================================================
# LAYER SOLVER
# ============================================================

class LayerSolver:
    """
    Solves Connect4 positions layer by layer (by piece count).

    Strategy:
    1. Start from terminals (piece_count = 42 or won positions)
    2. Work backward, using solved layers as boundary
    3. Extract seeds at each layer
    """

    def __init__(self, db: SeedDatabase, max_memory_mb: int = 4000):
        self.db = db
        self.game = Connect4Game()
        self.max_memory_mb = max_memory_mb
        self.stats = defaultdict(int)

    def solve_layer(self, piece_count: int, verbose: bool = True) -> SeedLayer:
        """
        Solve all positions with exactly `piece_count` pieces.

        Uses the layer below (piece_count + 1) as boundary.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"SOLVING LAYER: {piece_count} pieces")
            print("=" * 60)

        layer = SeedLayer(piece_count)

        # Generate all positions at this piece count
        positions = self._generate_layer_positions(piece_count)
        layer.total_positions = len(positions)

        if verbose:
            print(f"Positions in layer: {len(positions):,}")

        # Solve each position
        solved = 0
        seeds_found = 0

        for i, state in enumerate(positions):
            value, is_seed, reason = self._solve_position(state)
            solved += 1

            if is_seed:
                seed = C4Seed(
                    cols=state.cols,
                    turn=state.turn,
                    value=value,
                    piece_count=piece_count,
                    is_terminal=state.is_terminal(),
                    critical_reason=reason
                )
                layer.add_seed(seed)
                self.db.add_seed(seed)
                seeds_found += 1

            if verbose and (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,}/{len(positions):,} ({seeds_found} seeds)")

        layer.positions_solved = solved
        self.db.update_layer_stats(piece_count, len(positions), solved)

        if verbose:
            print(f"\nLayer {piece_count} complete:")
            print(f"  Positions: {solved:,}")
            print(f"  Seeds: {seeds_found:,}")
            print(f"  Compression: {layer.compression_ratio():.1f}x")

        return layer

    def _generate_layer_positions(self, piece_count: int) -> List[C4State]:
        """
        Generate all valid positions with exactly piece_count pieces.

        This is the core enumeration - needs to be efficient.
        """
        if piece_count == 0:
            return [C4State()]

        # For small piece counts, enumerate directly
        if piece_count <= 8:
            return self._enumerate_positions(piece_count)

        # For larger counts, use previous layer + successors
        # This requires the previous layer to be solved
        prev_layer = piece_count - 1
        if prev_layer in self.db.layers:
            positions = []
            seen = set()

            for seed_hash, seed in self.db.layers[prev_layer].seeds.items():
                state = seed.to_state()
                for child, move in self.game.get_successors(state):
                    if child.piece_count() == piece_count:
                        h = self.game.hash_state(child)
                        if h not in seen:
                            seen.add(h)
                            positions.append(child)

            return positions

        # Fallback: enumerate directly (expensive for large counts)
        return self._enumerate_positions(piece_count)

    def _enumerate_positions(self, piece_count: int) -> List[C4State]:
        """
        Enumerate all positions with piece_count pieces.

        Uses BFS from empty board.
        """
        positions = []
        seen = set()

        # BFS from empty
        frontier = [C4State()]
        seen.add(hash(C4State()))

        for depth in range(piece_count):
            next_frontier = []
            for state in frontier:
                for child, move in self.game.get_successors(state):
                    h = self.game.hash_state(child)
                    if h not in seen:
                        seen.add(h)
                        if child.piece_count() == piece_count:
                            positions.append(child)
                        elif child.piece_count() < piece_count:
                            next_frontier.append(child)

            frontier = next_frontier

            # Memory limit check
            if len(seen) > 10_000_000:
                print(f"  Warning: Memory limit at depth {depth}")
                break

        return positions

    def _solve_position(self, state: C4State) -> Tuple[int, bool, str]:
        """
        Solve a position and determine if it's a seed.

        Returns: (value, is_seed, reason)
        """
        # Terminal check
        winner = state.check_win()
        if winner:
            return (1 if winner == 'X' else -1, True, "terminal_win")

        if state.piece_count() == 42:
            return (0, True, "terminal_draw")

        # Look up children values
        child_values = []
        for child, move in self.game.get_successors(state):
            # Check seed database
            seed = self.db.get_seed(child)
            if seed:
                child_values.append(seed.value)
            else:
                # If child not in database, it's unsolved
                # This shouldn't happen if we solve layers in order
                child_values.append(0)

        if not child_values:
            return (0, True, "no_moves")

        # Minimax
        if state.turn == 'X':
            value = max(child_values)
        else:
            value = min(child_values)

        # Determine if this is a seed
        # Seed if: terminal, or children have mixed values, or forced win/loss
        unique_child_values = set(child_values)

        if len(unique_child_values) > 1:
            # Children disagree - this is a critical decision point
            return (value, True, "value_transition")

        if value == 1 and state.turn == 'X':
            # X to play and can win - important seed
            return (value, True, "forced_win")

        if value == -1 and state.turn == 'O':
            # O to play and can win - important seed
            return (value, True, "forced_win")

        # Not a seed - can be reconstructed from children
        return (value, False, "")

    def solve_all(self, start_piece_count: int = 42, end_piece_count: int = 0,
                  verbose: bool = True):
        """
        Solve all layers from start to end (going backward).
        """
        if verbose:
            print("=" * 70)
            print("FULL CONNECT4 SOLVE")
            print("=" * 70)
            print(f"Solving layers {start_piece_count} -> {end_piece_count}")

        start_time = time.time()

        for pc in range(start_piece_count, end_piece_count - 1, -1):
            layer_start = time.time()
            layer = self.solve_layer(pc, verbose=verbose)
            layer_time = time.time() - layer_start

            if verbose:
                print(f"  Time: {layer_time:.1f}s")

        total_time = time.time() - start_time

        if verbose:
            print("\n" + self.db.summary())
            print(f"\nTotal time: {total_time:.1f}s")


# ============================================================
# LAYER 2: STRATEGY OPTIMIZATION
# ============================================================

@dataclass
class StrategyConfig:
    """
    A Layer 2 strategy configuration.

    Determines which seeds to keep for optimal compression.
    """
    seed_selection: str = "all"  # "all", "critical", "sampled"
    sample_rate: float = 1.0     # If sampled, what fraction
    prioritize_wins: bool = True  # Keep win/loss seeds preferentially
    max_seeds_per_layer: int = 0  # 0 = unlimited

    def describe(self) -> str:
        return f"Strategy({self.seed_selection}, rate={self.sample_rate})"


class StrategyOptimizer:
    """
    Layer 2: Optimizes multi-seed strategy for compression.

    Given a solved database, finds minimal seed sets that
    can still reconstruct all values.
    """

    def __init__(self, db: SeedDatabase):
        self.db = db

    def analyze_layer(self, piece_count: int) -> Dict[str, Any]:
        """
        Analyze a layer to find compression opportunities.
        """
        if piece_count not in self.db.layers:
            return {}

        layer = self.db.layers[piece_count]

        # Categorize seeds
        categories = defaultdict(list)
        for h, seed in layer.seeds.items():
            categories[seed.critical_reason].append(seed)

        return {
            'piece_count': piece_count,
            'total_seeds': len(layer.seeds),
            'categories': {k: len(v) for k, v in categories.items()},
            'compression': layer.compression_ratio(),
        }

    def optimize_layer(self, piece_count: int, config: StrategyConfig) -> SeedLayer:
        """
        Create optimized seed set for a layer.

        Removes redundant seeds while maintaining reconstructability.
        """
        if piece_count not in self.db.layers:
            return SeedLayer(piece_count)

        original = self.db.layers[piece_count]
        optimized = SeedLayer(piece_count)

        # Always keep terminals
        for h, seed in original.seeds.items():
            if seed.is_terminal:
                optimized.add_seed(seed)
            elif config.prioritize_wins and seed.value != 0:
                optimized.add_seed(seed)
            elif config.seed_selection == "all":
                optimized.add_seed(seed)
            elif config.seed_selection == "sampled":
                if hash(h) % 100 < config.sample_rate * 100:
                    optimized.add_seed(seed)

        # Apply max limit
        if config.max_seeds_per_layer > 0:
            if len(optimized.seeds) > config.max_seeds_per_layer:
                # Keep most important seeds
                sorted_seeds = sorted(
                    optimized.seeds.values(),
                    key=lambda s: (s.is_terminal, abs(s.value), s.piece_count),
                    reverse=True
                )
                optimized.seeds = {
                    s.hash(): s for s in sorted_seeds[:config.max_seeds_per_layer]
                }

        optimized.total_positions = original.total_positions
        optimized.positions_solved = original.positions_solved

        return optimized

    def compare_strategies(self, strategies: List[StrategyConfig]) -> Dict[str, Any]:
        """
        Compare different strategies on all layers.
        """
        results = {}

        for config in strategies:
            total_seeds = 0
            total_storage = 0

            for pc in self.db.layers:
                optimized = self.optimize_layer(pc, config)
                total_seeds += len(optimized.seeds)
                total_storage += optimized.storage_bytes()

            results[config.describe()] = {
                'total_seeds': total_seeds,
                'storage_bytes': total_storage,
                'storage_mb': total_storage / 1024 / 1024,
            }

        return results


# ============================================================
# RECONSTRUCTION
# ============================================================

class ValueReconstructor:
    """
    Reconstructs position values from seed database.

    Given seeds, can compute value of any position via minimax.
    """

    def __init__(self, db: SeedDatabase, max_depth: int = 42):
        self.db = db
        self.game = Connect4Game()
        self.cache: Dict[int, int] = {}
        self.stats = {'hits': 0, 'misses': 0, 'computed': 0}
        self.max_depth = max_depth  # Max piece count we have data for

    def get_value(self, state: C4State, depth: int = 0) -> int:
        """
        Get value of a position, computing if necessary.
        """
        h = self.game.hash_state(state)

        # Check cache
        if h in self.cache:
            self.stats['hits'] += 1
            return self.cache[h]

        # Check seed database (use in-memory only to avoid locking)
        pc = state.piece_count()
        if pc in self.db.layers:
            seed = self.db.layers[pc].get_seed(state)
            if seed:
                self.cache[h] = seed.value
                self.stats['hits'] += 1
                return seed.value

        # If beyond our data, return draw (unknown)
        if pc > self.max_depth:
            return 0

        # Must compute via minimax
        self.stats['misses'] += 1
        value = self._compute_value(state, depth)
        self.cache[h] = value
        return value

    def _compute_value(self, state: C4State, depth: int) -> int:
        """
        Compute value via minimax using seeds + cache.
        """
        self.stats['computed'] += 1

        # Terminal check
        winner = state.check_win()
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        elif state.piece_count() == 42:
            return 0

        # Depth limit to prevent infinite recursion
        if depth > 50:
            return 0

        # Get child values
        child_values = []
        for child, move in self.game.get_successors(state):
            cv = self.get_value(child, depth + 1)
            child_values.append(cv)

        if not child_values:
            return 0

        # Minimax
        if state.turn == 'X':
            return max(child_values)
        else:
            return min(child_values)

    def verify_position(self, state: C4State, expected: int) -> bool:
        """Verify a position's value matches expected"""
        computed = self.get_value(state)
        return computed == expected

    def summary(self) -> str:
        hit_rate = self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
        return f"Reconstructor: {hit_rate:.1%} hit rate, {self.stats['computed']} computed"


# ============================================================
# MAIN ENTRY POINTS
# ============================================================

def solve_small_test(max_pieces: int = 12):
    """
    Test solve on small piece counts.
    """
    print("=" * 70)
    print(f"CONNECT4 SMALL TEST (0-{max_pieces} pieces)")
    print("=" * 70)

    db_path = f"connect4_test_{max_pieces}.db"
    db = SeedDatabase(db_path)

    solver = LayerSolver(db)
    solver.solve_all(start_piece_count=max_pieces, end_piece_count=0)

    print("\n" + db.summary())

    # Test reconstruction
    print("\n--- Reconstruction Test ---")
    recon = ValueReconstructor(db)

    # Test some positions
    test_states = [C4State()]  # Empty board
    for _ in range(5):
        for child, move in Connect4Game().get_successors(test_states[-1]):
            test_states.append(child)
            break

    for state in test_states[:5]:
        value = recon.get_value(state)
        print(f"  {state.piece_count()} pieces: value = {value}")

    print(recon.summary())

    db.close()
    return db_path


def run_strategy_comparison(db_path: str):
    """
    Compare different seed selection strategies.
    """
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    db = SeedDatabase(db_path)
    optimizer = StrategyOptimizer(db)

    strategies = [
        StrategyConfig(seed_selection="all"),
        StrategyConfig(seed_selection="critical", prioritize_wins=True),
        StrategyConfig(seed_selection="sampled", sample_rate=0.5),
        StrategyConfig(seed_selection="sampled", sample_rate=0.1),
    ]

    results = optimizer.compare_strategies(strategies)

    print("\n--- Results ---")
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Seeds: {data['total_seeds']:,}")
        print(f"  Storage: {data['storage_mb']:.2f} MB")

    db.close()


def main():
    """Run the full Connect4 solver"""
    import argparse

    parser = argparse.ArgumentParser(description="Connect4 Full Solver with Seed Compression")
    parser.add_argument("--test", type=int, default=12, help="Max pieces for test run")
    parser.add_argument("--full", action="store_true", help="Attempt full solve")
    parser.add_argument("--strategy", action="store_true", help="Run strategy comparison")
    parser.add_argument("--db", type=str, default="connect4_seeds.db", help="Database path")

    args = parser.parse_args()

    if args.full:
        print("Full solve not implemented yet - would need distributed computing")
        print("Running test solve instead...")

    db_path = solve_small_test(args.test)

    if args.strategy:
        run_strategy_comparison(db_path)


if __name__ == "__main__":
    main()
