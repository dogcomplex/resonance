"""
holos/connect4_1pct_test.py - Connect4 1% Coverage Test

Goal: Solve ~45 million positions (1% of 4.5 trillion) to gather statistics
for extrapolating feasibility of full solve.

Key Metrics to Collect:
1. Positions per second
2. Memory usage per position
3. Seed extraction ratio (seeds / positions)
4. Compression ratio with different strategies:
   - Full object storage + gzip
   - Indexed storage + gzip
   - Meta-seed storage + gzip
5. Time and storage projections for full 4.5T

Connect4 Position Counts by Piece Count:
  0:          1
  1:          7
  2:         49
  3:        238
  4:      1,120
  5:      4,263
  6:     16,422
  7:     54,859
  8:    184,275
  9:    558,186
 10:  1,662,623
 11:  4,568,485
 12: 12,236,135  (cumulative ~19M by here)
 13: 29,834,180  (cumulative ~49M)
 ...
 21: ~69 billion (peak)

Target: Layers 0-12 gives us ~19M positions, or layers 0-13 for ~49M.
"""

import os
import sys
import time
import gzip
import struct
import pickle
import psutil
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.games.connect4 import Connect4Game, C4State, C4Value
from holos.compression import (
    IndexedStateEncoder, CompressionAwareSeedValue, StateDimension,
    DimensionType, StateRepresentationComparer, estimate_metaseed_compression
)


# ============================================================
# CONSTANTS
# ============================================================

TOTAL_POSITIONS = 4_531_985_219_092  # 4.5 trillion
TARGET_COVERAGE = 0.01  # 1%
TARGET_POSITIONS = int(TOTAL_POSITIONS * TARGET_COVERAGE)  # ~45 billion... too many

# More realistic: solve layers 0-12 for ~19M positions
# That's 19M / 4.5T = 0.00042% but gives good statistics
POSITIONS_BY_LAYER = {
    0: 1, 1: 7, 2: 49, 3: 238, 4: 1_120, 5: 4_263,
    6: 16_422, 7: 54_859, 8: 184_275, 9: 558_186,
    10: 1_662_623, 11: 4_568_485, 12: 12_236_135,
    13: 29_834_180, 14: 68_081_095, 15: 145_000_000,
}

# Actually compute more realistic numbers
# Layers 0-10: ~2.48M positions (fast test)
# Layers 0-12: ~19M positions (medium test)
# Layers 0-14: ~130M positions (1% test)


# ============================================================
# SEED STORAGE (Minimal for stats collection)
# ============================================================

@dataclass
class SeedStats:
    """Statistics for seed compression"""
    layer: int
    positions_total: int = 0
    positions_solved: int = 0
    seeds_extracted: int = 0
    terminal_seeds: int = 0
    transition_seeds: int = 0
    forced_win_seeds: int = 0
    solve_time_s: float = 0.0
    memory_mb: float = 0.0

    @property
    def compression_ratio(self) -> float:
        if self.seeds_extracted == 0:
            return 0.0
        return self.positions_solved / self.seeds_extracted

    @property
    def seed_rate(self) -> float:
        """Fraction of positions that are seeds"""
        if self.positions_solved == 0:
            return 0.0
        return self.seeds_extracted / self.positions_solved


@dataclass
class C4SeedCompact:
    """Compact seed representation for storage comparison"""
    cols_packed: bytes  # 7 columns, each up to 6 chars, packed
    turn: int           # 0=X, 1=O
    value: int          # -1, 0, 1
    layer: int          # Piece count


# ============================================================
# LAYER ENUMERATION - MEMORY EFFICIENT
# ============================================================

def enumerate_layer_from_previous(game: Connect4Game,
                                   piece_count: int,
                                   previous_layer_hashes: Set[int],
                                   previous_layer_states: Dict[int, C4State]) -> List[C4State]:
    """
    Enumerate layer by expanding from previous layer.

    Much more memory efficient than BFS from empty.
    """
    if piece_count == 0:
        return [C4State()]

    positions = []
    seen = set()

    for h, state in previous_layer_states.items():
        if state.check_win():
            continue

        for child, move in game.get_successors(state):
            ch = game.hash_state(child)
            if ch not in seen:
                seen.add(ch)
                if child.piece_count() == piece_count:
                    positions.append(child)

    return positions


def enumerate_layer_streaming(game: Connect4Game, piece_count: int,
                               max_memory_mb: int = 2000):
    """
    Memory-efficient layer enumeration using streaming.

    Yields positions one at a time instead of storing all in memory.
    Uses iterative deepening with hash-based deduplication.
    """
    if piece_count == 0:
        yield C4State()
        return

    # For small layers, use simple enumeration
    if piece_count <= 8:
        seen = set()
        frontier = [C4State()]
        seen.add(game.hash_state(C4State()))

        for depth in range(piece_count):
            next_frontier = []
            for state in frontier:
                if state.check_win():
                    continue

                for child, move in game.get_successors(state):
                    h = game.hash_state(child)
                    if h not in seen:
                        seen.add(h)
                        if child.piece_count() == piece_count:
                            yield child
                        elif child.piece_count() < piece_count and not child.check_win():
                            next_frontier.append(child)

            frontier = next_frontier
            if not frontier:
                break
        return

    # For larger layers, use hash-only deduplication
    # Store only hashes, reconstruct states as needed
    seen_hashes = set()
    current_layer_states = {game.hash_state(C4State()): C4State()}

    for depth in range(piece_count):
        next_layer = {}
        positions_at_target = []

        for h, state in current_layer_states.items():
            if state.check_win():
                continue

            for child, move in game.get_successors(state):
                ch = game.hash_state(child)
                if ch not in seen_hashes:
                    seen_hashes.add(ch)
                    if child.piece_count() == piece_count:
                        positions_at_target.append(child)
                    elif child.piece_count() < piece_count and not child.check_win():
                        next_layer[ch] = child

        # Yield positions at target depth
        for pos in positions_at_target:
            yield pos

        current_layer_states = next_layer

        # Memory check
        mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if mem_mb > max_memory_mb:
            print(f"    Warning: Memory limit ({mem_mb:.0f}MB) at depth {depth}")
            break

        if not current_layer_states:
            break


def enumerate_layer(game: Connect4Game, piece_count: int,
                    max_positions: int = 0) -> List[C4State]:
    """
    Enumerate all positions with exactly piece_count pieces.

    Uses streaming enumeration with memory limits.
    """
    positions = []
    for pos in enumerate_layer_streaming(game, piece_count, max_memory_mb=3000):
        positions.append(pos)
        if max_positions > 0 and len(positions) >= max_positions:
            break
    return positions


def solve_layer_incremental(game: Connect4Game,
                             piece_count: int,
                             previous_states: Dict[int, C4State],
                             solved_values: Dict[int, int],
                             verbose: bool = True) -> Tuple[SeedStats, Dict[int, int], Dict[int, C4State]]:
    """
    Solve a layer incrementally by expanding from previous layer.

    Memory efficient - only stores current layer + values.

    Args:
        game: Connect4 game interface
        piece_count: Layer to solve
        previous_states: States from previous layer (piece_count - 1)
        solved_values: Dict of hash -> value from higher layers
        verbose: Print progress

    Returns:
        (stats, new_values, current_layer_states)
    """
    stats = SeedStats(layer=piece_count)
    start_time = time.time()

    new_values = {}
    current_states = {}
    seeds_by_reason = defaultdict(int)
    seen = set()

    # Special case: layer 0
    if piece_count == 0:
        start = C4State()
        h = game.hash_state(start)
        current_states[h] = start
        stats.positions_total = 1
        stats.positions_solved = 1
        # Layer 0 needs children solved first - handle separately
        return stats, new_values, current_states

    # Expand from previous layer
    if verbose:
        print(f"  Expanding from {len(previous_states):,} parent positions...", end=" ", flush=True)

    positions_count = 0
    for ph, parent in previous_states.items():
        if parent.check_win():
            continue

        for child, move in game.get_successors(parent):
            ch = game.hash_state(child)
            if ch in seen:
                continue
            seen.add(ch)

            if child.piece_count() != piece_count:
                continue

            positions_count += 1
            current_states[ch] = child

            # Solve immediately
            winner = child.check_win()
            if winner:
                value = 1 if winner == 'X' else -1
                new_values[ch] = value
                stats.seeds_extracted += 1
                stats.terminal_seeds += 1
                seeds_by_reason['terminal_win'] += 1
                continue

            if child.piece_count() == 42:
                new_values[ch] = 0
                stats.seeds_extracted += 1
                stats.terminal_seeds += 1
                seeds_by_reason['terminal_draw'] += 1
                continue

            # Look up children
            child_values = []
            for grandchild, _ in game.get_successors(child):
                gh = game.hash_state(grandchild)
                if gh in solved_values:
                    child_values.append(solved_values[gh])
                elif gh in new_values:
                    child_values.append(new_values[gh])

            if not child_values:
                new_values[ch] = 0
                stats.seeds_extracted += 1
                seeds_by_reason['no_moves'] += 1
                continue

            # Minimax
            if child.turn == 'X':
                value = max(child_values)
            else:
                value = min(child_values)

            new_values[ch] = value

            # Determine if seed
            unique_values = set(child_values)
            if len(unique_values) > 1:
                stats.seeds_extracted += 1
                stats.transition_seeds += 1
                seeds_by_reason['value_transition'] += 1
            elif value == 1 and child.turn == 'X':
                stats.seeds_extracted += 1
                stats.forced_win_seeds += 1
                seeds_by_reason['forced_win_X'] += 1
            elif value == -1 and child.turn == 'O':
                stats.seeds_extracted += 1
                stats.forced_win_seeds += 1
                seeds_by_reason['forced_win_O'] += 1

    stats.positions_total = positions_count
    stats.positions_solved = positions_count
    stats.solve_time_s = time.time() - start_time

    if verbose:
        print(f"{positions_count:,} positions")
        print(f"  Layer {piece_count}: {stats.positions_solved:,} positions, "
              f"{stats.seeds_extracted:,} seeds ({stats.compression_ratio:.1f}x), "
              f"{stats.solve_time_s:.1f}s")
        for reason, count in sorted(seeds_by_reason.items()):
            print(f"    {reason}: {count:,}")

    return stats, new_values, current_states


def solve_layer_with_stats(game: Connect4Game,
                           piece_count: int,
                           solved_values: Dict[int, int],
                           verbose: bool = True) -> Tuple[SeedStats, Dict[int, int]]:
    """
    Solve a layer and collect statistics.

    Args:
        game: Connect4 game interface
        piece_count: Layer to solve
        solved_values: Dict of hash -> value from higher layers
        verbose: Print progress

    Returns:
        (stats, new_solved_values)
    """
    stats = SeedStats(layer=piece_count)
    start_time = time.time()
    start_mem = psutil.Process().memory_info().rss / 1024 / 1024

    # Enumerate positions - use max to avoid memory issues
    max_positions = 2_000_000 if piece_count > 10 else 0

    if verbose:
        print(f"  Enumerating layer {piece_count}...", end=" ", flush=True)

    positions = enumerate_layer(game, piece_count, max_positions=max_positions)
    stats.positions_total = len(positions)
    stats.positions_solved = len(positions)

    if verbose:
        print(f"{len(positions):,} positions")

    # Solve each position
    new_values = {}
    seeds_by_reason = defaultdict(int)

    for i, state in enumerate(positions):
        h = game.hash_state(state)

        # Terminal check
        winner = state.check_win()
        if winner:
            value = 1 if winner == 'X' else -1
            new_values[h] = value
            stats.seeds_extracted += 1
            stats.terminal_seeds += 1
            seeds_by_reason['terminal_win'] += 1
            continue

        if state.piece_count() == 42:
            new_values[h] = 0
            stats.seeds_extracted += 1
            stats.terminal_seeds += 1
            seeds_by_reason['terminal_draw'] += 1
            continue

        # Look up children
        child_values = []
        for child, move in game.get_successors(state):
            ch = game.hash_state(child)
            if ch in solved_values:
                child_values.append(solved_values[ch])
            elif ch in new_values:
                child_values.append(new_values[ch])
            else:
                # Child not solved - shouldn't happen if we solve in order
                child_values.append(0)

        if not child_values:
            new_values[h] = 0
            stats.seeds_extracted += 1
            seeds_by_reason['no_moves'] += 1
            continue

        # Minimax
        if state.turn == 'X':
            value = max(child_values)
        else:
            value = min(child_values)

        new_values[h] = value

        # Determine if seed
        unique_values = set(child_values)

        if len(unique_values) > 1:
            stats.seeds_extracted += 1
            stats.transition_seeds += 1
            seeds_by_reason['value_transition'] += 1
        elif value == 1 and state.turn == 'X':
            stats.seeds_extracted += 1
            stats.forced_win_seeds += 1
            seeds_by_reason['forced_win_X'] += 1
        elif value == -1 and state.turn == 'O':
            stats.seeds_extracted += 1
            stats.forced_win_seeds += 1
            seeds_by_reason['forced_win_O'] += 1

        # Progress
        if verbose and (i + 1) % 100000 == 0:
            print(f"    {i+1:,}/{len(positions):,} ({stats.seeds_extracted:,} seeds)")

    stats.solve_time_s = time.time() - start_time
    stats.memory_mb = psutil.Process().memory_info().rss / 1024 / 1024 - start_mem

    if verbose:
        print(f"  Layer {piece_count}: {stats.positions_solved:,} positions, "
              f"{stats.seeds_extracted:,} seeds ({stats.compression_ratio:.1f}x), "
              f"{stats.solve_time_s:.1f}s")
        for reason, count in sorted(seeds_by_reason.items()):
            print(f"    {reason}: {count:,}")

    return stats, new_values


# ============================================================
# COMPRESSION COMPARISON
# ============================================================

def compare_storage_formats(positions: List[C4State],
                            values: Dict[int, int],
                            game: Connect4Game,
                            verbose: bool = True) -> Dict[str, Any]:
    """
    Compare different storage formats for the solved positions.

    Formats:
    1. Full pickle + gzip
    2. Hash + value only + gzip
    3. Indexed position + gzip
    4. Seeds only + gzip
    """
    results = {}

    # Format 1: Full pickle
    start = time.time()
    full_data = pickle.dumps([(pos.cols, pos.turn, values.get(game.hash_state(pos), 0))
                              for pos in positions])
    full_gzip = gzip.compress(full_data)
    encode_time = time.time() - start

    results['full_pickle_gzip'] = {
        'raw_bytes': len(full_data),
        'compressed_bytes': len(full_gzip),
        'ratio': len(full_data) / len(full_gzip) if full_gzip else 0,
        'encode_time': encode_time,
    }

    if verbose:
        print(f"  Full pickle + gzip: {len(full_gzip):,} bytes "
              f"({len(full_data)/len(full_gzip):.1f}x compression)")

    # Format 2: Hash + value only (8 bytes hash + 1 byte value)
    start = time.time()
    hash_value_data = b''.join(
        struct.pack('qb', game.hash_state(pos), values.get(game.hash_state(pos), 0))
        for pos in positions
    )
    hash_value_gzip = gzip.compress(hash_value_data)
    encode_time = time.time() - start

    results['hash_value_gzip'] = {
        'raw_bytes': len(hash_value_data),
        'compressed_bytes': len(hash_value_gzip),
        'ratio': len(hash_value_data) / len(hash_value_gzip) if hash_value_gzip else 0,
        'encode_time': encode_time,
    }

    if verbose:
        print(f"  Hash + value + gzip: {len(hash_value_gzip):,} bytes "
              f"({len(hash_value_data)/len(hash_value_gzip):.1f}x compression)")

    # Format 3: Compact position encoding
    # Each position: 7 columns * 1 byte each (heights) + 1 byte turn + 1 byte value = 9 bytes
    start = time.time()
    compact_data = b''
    for pos in positions:
        # Encode column heights (0-6 each)
        heights = bytes([len(col) for col in pos.cols])
        # Encode column contents as bits (X=1, O=0)
        content_bits = []
        for col in pos.cols:
            bits = 0
            for i, c in enumerate(col):
                if c == 'X':
                    bits |= (1 << i)
            content_bits.append(bits)
        content = bytes(content_bits)
        turn_val = struct.pack('bb', 0 if pos.turn == 'X' else 1,
                               values.get(game.hash_state(pos), 0))
        compact_data += heights + content + turn_val

    compact_gzip = gzip.compress(compact_data)
    encode_time = time.time() - start

    results['compact_gzip'] = {
        'raw_bytes': len(compact_data),
        'compressed_bytes': len(compact_gzip),
        'ratio': len(compact_data) / len(compact_gzip) if compact_gzip else 0,
        'encode_time': encode_time,
        'bytes_per_position': len(compact_data) / len(positions) if positions else 0,
    }

    if verbose:
        print(f"  Compact + gzip: {len(compact_gzip):,} bytes "
              f"({len(compact_data)/len(compact_gzip):.1f}x compression)")

    # Format 4: Values only (assume positions enumerable)
    # Just store the value for each canonical position index
    start = time.time()
    values_only = bytes([values.get(game.hash_state(pos), 0) + 1 for pos in positions])  # +1 to make 0-2
    values_gzip = gzip.compress(values_only)
    encode_time = time.time() - start

    results['values_only_gzip'] = {
        'raw_bytes': len(values_only),
        'compressed_bytes': len(values_gzip),
        'ratio': len(values_only) / len(values_gzip) if values_gzip else 0,
        'encode_time': encode_time,
    }

    if verbose:
        print(f"  Values only + gzip: {len(values_gzip):,} bytes "
              f"({len(values_only)/len(values_gzip):.1f}x compression)")

    return results


# ============================================================
# MAIN TEST
# ============================================================

def run_1pct_test_incremental(max_layer: int = 12, verbose: bool = True):
    """
    Run 1% coverage test using memory-efficient incremental approach.

    Solves layers from 0 up to max_layer (building forward), then
    propagates values backwards (from max_layer down to 0).

    This is more memory-efficient because we only need to keep:
    1. Current layer states
    2. All solved values (just hashes)
    """
    print("=" * 70)
    print("CONNECT4 1% COVERAGE TEST (INCREMENTAL)")
    print("=" * 70)
    print(f"Target: Layers 0-{max_layer}")

    # Estimate total positions
    estimated_positions = sum(
        POSITIONS_BY_LAYER.get(i, 0) for i in range(max_layer + 1)
    )
    print(f"Estimated positions: {estimated_positions:,}")
    print(f"This is {estimated_positions / TOTAL_POSITIONS * 100:.6f}% of full game")
    print()

    game = Connect4Game()
    all_stats: List[SeedStats] = []
    all_layers: Dict[int, Dict[int, C4State]] = {}  # layer -> {hash: state}

    total_start = time.time()

    # PHASE 1: Enumerate all layers (0 to max_layer)
    print("PHASE 1: ENUMERATION")
    print("-" * 50)

    # Start with layer 0
    current_states = {game.hash_state(C4State()): C4State()}
    all_layers[0] = current_states.copy()
    print(f"Layer 0: 1 position")

    # Expand each layer
    for layer in range(1, max_layer + 1):
        next_states = {}
        seen = set()

        for h, state in current_states.items():
            if state.check_win():
                continue

            for child, move in game.get_successors(state):
                ch = game.hash_state(child)
                if ch not in seen and child.piece_count() == layer:
                    seen.add(ch)
                    next_states[ch] = child

        all_layers[layer] = next_states
        current_states = next_states
        print(f"Layer {layer}: {len(next_states):,} positions")

        # Memory check
        mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if mem_mb > 6000:
            print(f"  Warning: Memory at {mem_mb:.0f}MB")

    enum_time = time.time() - total_start
    total_positions = sum(len(layer) for layer in all_layers.values())
    print(f"\nEnumeration complete: {total_positions:,} positions in {enum_time:.1f}s")

    # PHASE 2: Solve backwards (from max_layer down to 0)
    print("\nPHASE 2: BACKWARD SOLVE")
    print("-" * 50)

    solved_values: Dict[int, int] = {}

    for layer in range(max_layer, -1, -1):
        layer_states = all_layers[layer]
        stats = SeedStats(layer=layer)
        stats.positions_total = len(layer_states)
        stats.positions_solved = len(layer_states)
        layer_start = time.time()

        new_values = {}
        seeds_by_reason = defaultdict(int)

        for h, state in layer_states.items():
            # Terminal check
            winner = state.check_win()
            if winner:
                value = 1 if winner == 'X' else -1
                new_values[h] = value
                stats.seeds_extracted += 1
                stats.terminal_seeds += 1
                seeds_by_reason['terminal_win'] += 1
                continue

            if state.piece_count() == 42:
                new_values[h] = 0
                stats.seeds_extracted += 1
                stats.terminal_seeds += 1
                seeds_by_reason['terminal_draw'] += 1
                continue

            # Look up children values
            child_values = []
            for child, move in game.get_successors(state):
                ch = game.hash_state(child)
                if ch in solved_values:
                    child_values.append(solved_values[ch])

            if not child_values:
                new_values[h] = 0
                stats.seeds_extracted += 1
                seeds_by_reason['no_moves'] += 1
                continue

            # Minimax
            if state.turn == 'X':
                value = max(child_values)
            else:
                value = min(child_values)

            new_values[h] = value

            # Determine if seed
            unique_values = set(child_values)
            if len(unique_values) > 1:
                stats.seeds_extracted += 1
                stats.transition_seeds += 1
                seeds_by_reason['value_transition'] += 1
            elif value == 1 and state.turn == 'X':
                stats.seeds_extracted += 1
                stats.forced_win_seeds += 1
                seeds_by_reason['forced_win_X'] += 1
            elif value == -1 and state.turn == 'O':
                stats.seeds_extracted += 1
                stats.forced_win_seeds += 1
                seeds_by_reason['forced_win_O'] += 1

        solved_values.update(new_values)
        stats.solve_time_s = time.time() - layer_start
        all_stats.append(stats)

        ratio_str = f"{stats.compression_ratio:.1f}x" if stats.seeds_extracted > 0 else "N/A"
        print(f"Layer {layer}: {stats.positions_solved:,} positions, "
              f"{stats.seeds_extracted:,} seeds ({ratio_str}), {stats.solve_time_s:.1f}s")
        for reason, count in sorted(seeds_by_reason.items()):
            if count > 0:
                print(f"    {reason}: {count:,}")

        # Free layer memory after solving
        del all_layers[layer]

    total_time = time.time() - total_start

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total_positions = sum(s.positions_solved for s in all_stats)
    total_seeds = sum(s.seeds_extracted for s in all_stats)
    total_terminal = sum(s.terminal_seeds for s in all_stats)
    total_transition = sum(s.transition_seeds for s in all_stats)
    total_forced = sum(s.forced_win_seeds for s in all_stats)

    print(f"\nPositions solved: {total_positions:,}")
    print(f"Seeds extracted: {total_seeds:,}")
    if total_seeds > 0:
        print(f"Overall compression: {total_positions / total_seeds:.1f}x")
        print(f"\nSeed breakdown:")
        print(f"  Terminal: {total_terminal:,} ({total_terminal/total_seeds*100:.1f}%)")
        print(f"  Transition: {total_transition:,} ({total_transition/total_seeds*100:.1f}%)")
        print(f"  Forced win: {total_forced:,} ({total_forced/total_seeds*100:.1f}%)")

    print(f"\nTime: {total_time:.1f}s")
    print(f"Rate: {total_positions / total_time:.0f} positions/second")

    # Per-layer breakdown
    print("\nPer-layer statistics:")
    print(f"{'Layer':>5} {'Positions':>12} {'Seeds':>10} {'Ratio':>8} {'Time':>8}")
    print("-" * 50)
    for s in sorted(all_stats, key=lambda x: x.layer):
        ratio_str = f"{s.compression_ratio:.1f}x" if s.seeds_extracted > 0 else "N/A"
        print(f"{s.layer:>5} {s.positions_solved:>12,} {s.seeds_extracted:>10,} "
              f"{ratio_str:>8} {s.solve_time_s:>7.1f}s")

    # Extrapolation to full game
    print("\n" + "=" * 70)
    print("EXTRAPOLATION TO FULL 4.5T GAME")
    print("=" * 70)

    # Time estimate
    positions_per_sec = total_positions / total_time
    full_time_sec = TOTAL_POSITIONS / positions_per_sec
    full_time_days = full_time_sec / 86400

    print(f"\nAt {positions_per_sec:.0f} positions/second:")
    print(f"  Full solve time: {full_time_days:.0f} days ({full_time_sec/3600:.0f} hours)")

    # Storage estimate
    if total_seeds > 0:
        avg_compression = total_positions / total_seeds
        estimated_seeds = TOTAL_POSITIONS / avg_compression

        print(f"\nWith {avg_compression:.1f}x seed compression:")
        print(f"  Estimated seeds: {estimated_seeds:,.0f}")

        for name, bytes_per_seed in [("9-byte compact", 9), ("4-byte indexed", 4), ("2-byte indexed", 2)]:
            raw_storage = estimated_seeds * bytes_per_seed
            compressed_storage = raw_storage / 3
            print(f"  {name}: {raw_storage/1e12:.2f} TB raw, {compressed_storage/1e12:.2f} TB compressed")

    # Values-only storage
    values_storage = TOTAL_POSITIONS
    values_compressed = values_storage / 8
    print(f"\n  Values-only (enumerable): {values_storage/1e12:.1f} TB raw, "
          f"{values_compressed/1e12:.1f} TB compressed")

    # Hardware requirements
    print("\n" + "=" * 70)
    print("HARDWARE REQUIREMENTS")
    print("=" * 70)

    print(f"\nFor full 4.5T solve:")
    print(f"  Estimated time: {full_time_days:.0f} days single-threaded")
    print(f"  With 100 cores: {full_time_days/100:.1f} days")
    print(f"  With 1000 cores: {full_time_days/1000:.2f} days ({full_time_days/1000*24:.1f} hours)")

    if total_seeds > 0:
        best_storage_tb = min(
            estimated_seeds * 4 / 3 / 1e12,
            values_compressed / 1e12,
        )
        print(f"  Minimum storage: {best_storage_tb:.1f} TB")

    return {
        'total_positions': total_positions,
        'total_seeds': total_seeds,
        'compression_ratio': total_positions / total_seeds if total_seeds > 0 else 0,
        'time_seconds': total_time,
        'positions_per_second': positions_per_sec,
        'stats': all_stats,
    }


def run_1pct_test(max_layer: int = 12, verbose: bool = True):
    """
    Run 1% coverage test on Connect4.

    Solves layers 0 through max_layer and collects statistics.
    For layers > 10, uses incremental approach to save memory.
    """
    if max_layer > 10:
        return run_1pct_test_incremental(max_layer, verbose)

    print("=" * 70)
    print("CONNECT4 1% COVERAGE TEST")
    print("=" * 70)
    print(f"Target: Layers 0-{max_layer}")

    # Estimate total positions
    estimated_positions = sum(
        POSITIONS_BY_LAYER.get(i, 0) for i in range(max_layer + 1)
    )
    print(f"Estimated positions: {estimated_positions:,}")
    print(f"This is {estimated_positions / TOTAL_POSITIONS * 100:.6f}% of full game")
    print()

    game = Connect4Game()
    all_stats: List[SeedStats] = []
    solved_values: Dict[int, int] = {}
    all_positions: List[C4State] = []

    total_start = time.time()

    # Solve from high piece counts down
    for layer in range(max_layer, -1, -1):
        print(f"\nLayer {layer}:")

        stats, new_values = solve_layer_with_stats(
            game, layer, solved_values, verbose=verbose
        )
        all_stats.append(stats)

        # Merge values
        solved_values.update(new_values)

        # Keep positions for compression comparison (only if small enough)
        if stats.positions_total < 100000:
            layer_positions = enumerate_layer(game, layer)
            all_positions.extend(layer_positions)

    total_time = time.time() - total_start

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total_positions = sum(s.positions_solved for s in all_stats)
    total_seeds = sum(s.seeds_extracted for s in all_stats)
    total_terminal = sum(s.terminal_seeds for s in all_stats)
    total_transition = sum(s.transition_seeds for s in all_stats)
    total_forced = sum(s.forced_win_seeds for s in all_stats)

    print(f"\nPositions solved: {total_positions:,}")
    print(f"Seeds extracted: {total_seeds:,}")
    print(f"Overall compression: {total_positions / total_seeds:.1f}x")
    print(f"\nSeed breakdown:")
    print(f"  Terminal: {total_terminal:,} ({total_terminal/total_seeds*100:.1f}%)")
    print(f"  Transition: {total_transition:,} ({total_transition/total_seeds*100:.1f}%)")
    print(f"  Forced win: {total_forced:,} ({total_forced/total_seeds*100:.1f}%)")

    print(f"\nTime: {total_time:.1f}s")
    print(f"Rate: {total_positions / total_time:.0f} positions/second")

    # Per-layer breakdown
    print("\nPer-layer statistics:")
    print(f"{'Layer':>5} {'Positions':>12} {'Seeds':>10} {'Ratio':>8} {'Time':>8}")
    print("-" * 50)
    for s in sorted(all_stats, key=lambda x: x.layer):
        print(f"{s.layer:>5} {s.positions_solved:>12,} {s.seeds_extracted:>10,} "
              f"{s.compression_ratio:>8.1f}x {s.solve_time_s:>7.1f}s")

    # Compression comparison
    if len(all_positions) > 0 and len(all_positions) < 500000:
        print("\n" + "=" * 70)
        print("STORAGE FORMAT COMPARISON")
        print("=" * 70)
        print(f"Testing on {len(all_positions):,} positions from smaller layers\n")

        storage_results = compare_storage_formats(
            all_positions, solved_values, game, verbose=True
        )

    # Extrapolation to full game
    print("\n" + "=" * 70)
    print("EXTRAPOLATION TO FULL 4.5T GAME")
    print("=" * 70)

    # Time estimate
    positions_per_sec = total_positions / total_time
    full_time_sec = TOTAL_POSITIONS / positions_per_sec
    full_time_days = full_time_sec / 86400

    print(f"\nAt {positions_per_sec:.0f} positions/second:")
    print(f"  Full solve time: {full_time_days:.0f} days ({full_time_sec/3600:.0f} hours)")

    # Storage estimate
    avg_compression = total_positions / total_seeds
    estimated_seeds = TOTAL_POSITIONS / avg_compression

    # With different storage formats
    print(f"\nWith {avg_compression:.1f}x seed compression:")
    print(f"  Estimated seeds: {estimated_seeds:,.0f}")

    for name, bytes_per_seed in [("9-byte compact", 9), ("4-byte indexed", 4), ("2-byte indexed", 2)]:
        raw_storage = estimated_seeds * bytes_per_seed
        # Assume ~3x gzip compression based on our tests
        compressed_storage = raw_storage / 3
        print(f"  {name}: {raw_storage/1e12:.2f} TB raw, {compressed_storage/1e12:.2f} TB compressed")

    # Values-only storage (if positions enumerable)
    values_storage = TOTAL_POSITIONS  # 1 byte per position
    values_compressed = values_storage / 8  # Assume 8x compression for value bits
    print(f"\n  Values-only (enumerable): {values_storage/1e12:.1f} TB raw, "
          f"{values_compressed/1e12:.1f} TB compressed")

    # Hardware requirements
    print("\n" + "=" * 70)
    print("HARDWARE REQUIREMENTS")
    print("=" * 70)

    print(f"\nFor full 4.5T solve:")
    print(f"  Estimated time: {full_time_days:.0f} days single-threaded")
    print(f"  With 100 cores: {full_time_days/100:.1f} days")
    print(f"  With 1000 cores: {full_time_days/1000:.2f} days ({full_time_days/1000*24:.1f} hours)")

    best_storage_tb = min(
        estimated_seeds * 4 / 3 / 1e12,  # Indexed + gzip
        values_compressed / 1e12,         # Values-only
    )
    print(f"  Minimum storage: {best_storage_tb:.1f} TB")

    return {
        'total_positions': total_positions,
        'total_seeds': total_seeds,
        'compression_ratio': avg_compression,
        'time_seconds': total_time,
        'positions_per_second': positions_per_sec,
        'stats': all_stats,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Connect4 1% coverage test")
    parser.add_argument("--max-layer", type=int, default=12,
                       help="Maximum layer (piece count) to solve")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")

    args = parser.parse_args()

    results = run_1pct_test(max_layer=args.max_layer, verbose=not args.quiet)
