"""
holos/test_compression.py - Test compression-aware state representation

Tests:
1. IndexedStateEncoder - encode/decode correctness
2. CompressionAwareSeedValue - efficiency calculations
3. StateRepresentationComparer - find best representation
4. Connect4 seed compression comparison
"""

import gzip
import struct
import time
from typing import List, Tuple

from holos.compression import (
    StateDimension, DimensionType, IndexedStateEncoder,
    CompressionAwareSeedValue, ValueBucketer, StateRepresentationComparer,
    create_seed_encoder, create_coverage_bucketer,
    estimate_seed_storage, estimate_metaseed_compression
)
from holos.holos import SearchMode
from holos.games.seeds import TacticalSeed, SeedDirection


def test_dimension_encoding():
    """Test individual dimension encoding"""
    print("\n=== Testing Dimension Encoding ===\n")

    # Test DISCRETE
    mode_dim = StateDimension(
        name="mode",
        dim_type=DimensionType.DISCRETE,
        discrete_values=[SearchMode.LIGHTNING, SearchMode.WAVE,
                        SearchMode.CRYSTAL, SearchMode.OSMOSIS]
    )

    assert mode_dim.cardinality == 4
    assert mode_dim.encode_value(SearchMode.LIGHTNING) == 0
    assert mode_dim.encode_value(SearchMode.WAVE) == 1
    assert mode_dim.decode_value(2) == SearchMode.CRYSTAL
    print(f"  DISCRETE mode: {mode_dim.cardinality} values - OK")

    # Test BOUNDED_INT
    depth_dim = StateDimension(
        name="depth",
        dim_type=DimensionType.BOUNDED_INT,
        min_value=1,
        max_value=6
    )

    assert depth_dim.cardinality == 6
    assert depth_dim.encode_value(1) == 0
    assert depth_dim.encode_value(6) == 5
    assert depth_dim.decode_value(3) == 4
    print(f"  BOUNDED_INT depth: {depth_dim.cardinality} values - OK")

    # Test BUCKETED
    coverage_dim = StateDimension(
        name="coverage",
        dim_type=DimensionType.BUCKETED,
        bucket_boundaries=[100, 1000, 10000, 100000, 1000000]
    )

    assert coverage_dim.cardinality == 6  # 5 boundaries = 6 buckets
    assert coverage_dim.encode_value(50) == 0
    assert coverage_dim.encode_value(500) == 1
    assert coverage_dim.encode_value(5000) == 2
    assert coverage_dim.encode_value(5000000) == 5
    print(f"  BUCKETED coverage: {coverage_dim.cardinality} buckets - OK")


def test_indexed_encoder():
    """Test full IndexedStateEncoder"""
    print("\n=== Testing IndexedStateEncoder ===\n")

    # Create encoder for TacticalSeed-like states
    dimensions = [
        StateDimension("depth", DimensionType.BOUNDED_INT, min_value=1, max_value=6),
        StateDimension("mode", DimensionType.DISCRETE,
                      discrete_values=[SearchMode.LIGHTNING, SearchMode.WAVE,
                                      SearchMode.CRYSTAL, SearchMode.OSMOSIS]),
        StateDimension("direction", DimensionType.DISCRETE,
                      discrete_values=[SeedDirection.FORWARD, SeedDirection.BACKWARD,
                                      SeedDirection.BILATERAL]),
        StateDimension("position", DimensionType.BOUNDED_INT, min_value=0, max_value=999),
    ]

    def extract(seed: TacticalSeed) -> Tuple:
        return (seed.depth, seed.mode, seed.direction, seed.position_hash % 1000)

    def construct(values: Tuple) -> TacticalSeed:
        depth, mode, direction, pos = values
        return TacticalSeed(pos, depth, mode, direction)

    encoder = IndexedStateEncoder(dimensions, extract, construct)

    print(f"  Total state space: {encoder.total_states:,}")
    print(f"  Bits per index: {encoder.bits_per_index}")
    print(f"  Bytes per index: {encoder.bytes_per_index}")

    # Expected: 6 * 4 * 3 * 1000 = 72,000 states
    assert encoder.total_states == 72000
    assert encoder.bits_per_index == 17  # ceil(log2(72000))
    assert encoder.bytes_per_index == 4  # Needs 32-bit int

    # Test encode/decode roundtrip
    test_seeds = [
        TacticalSeed(0, 1, SearchMode.LIGHTNING, SeedDirection.FORWARD),
        TacticalSeed(500, 3, SearchMode.WAVE, SeedDirection.BACKWARD),
        TacticalSeed(999, 6, SearchMode.OSMOSIS, SeedDirection.BILATERAL),
    ]

    for seed in test_seeds:
        idx = encoder.encode(seed)
        decoded = encoder.decode(idx)
        assert decoded.depth == seed.depth
        assert decoded.mode == seed.mode
        assert decoded.direction == seed.direction
        # Position may be truncated to 0-999
        print(f"  Encode/decode: {seed.signature()} -> idx={idx} -> OK")

    # Test batch encoding
    encoded_batch = encoder.encode_batch(test_seeds)
    decoded_batch = encoder.decode_batch(encoded_batch)
    assert len(decoded_batch) == len(test_seeds)
    print(f"  Batch encode: {len(test_seeds)} seeds -> {len(encoded_batch)} bytes - OK")


def test_compression_aware_value():
    """Test CompressionAwareSeedValue calculations"""
    print("\n=== Testing CompressionAwareSeedValue ===\n")

    # Case 1: Good seed (worth storing)
    good_seed = CompressionAwareSeedValue(
        forward_coverage=10000,
        backward_coverage=5000,
        compute_cost=100,
        seed_storage_bytes=11,  # Seed config
        frontier_storage_bytes=0,  # Derived from parent
        derivation_bytes=1,
        bytes_per_position=4,
    )

    print(f"  Good seed: {good_seed}")
    print(f"    Direct storage: {good_seed.direct_storage:,} bytes")
    print(f"    Seed storage: {good_seed.seed_storage} bytes")
    print(f"    Net savings: {good_seed.net_savings:,} bytes")
    print(f"    Compression ratio: {good_seed.compression_ratio:.1f}x")
    print(f"    Efficiency: {good_seed.efficiency:.1f}")
    print(f"    Worth storing: {good_seed.is_worth_storing}")

    assert good_seed.is_worth_storing
    assert good_seed.net_savings > 0
    assert good_seed.efficiency > 0

    # Case 2: Bad seed (not worth storing)
    bad_seed = CompressionAwareSeedValue(
        forward_coverage=2,
        backward_coverage=1,
        compute_cost=100,
        seed_storage_bytes=50,  # Overhead exceeds coverage
        frontier_storage_bytes=100,
        derivation_bytes=0,
        bytes_per_position=4,
    )

    print(f"\n  Bad seed: {bad_seed}")
    print(f"    Direct storage: {bad_seed.direct_storage} bytes")
    print(f"    Seed storage: {bad_seed.seed_storage} bytes")
    print(f"    Net savings: {bad_seed.net_savings} bytes")
    print(f"    Efficiency: {bad_seed.efficiency}")
    print(f"    Worth storing: {bad_seed.is_worth_storing}")

    assert not bad_seed.is_worth_storing
    assert bad_seed.net_savings < 0
    assert bad_seed.efficiency == -1.0

    # Comparison
    assert good_seed > bad_seed  # Good seed has higher efficiency


def test_value_bucketer():
    """Test ValueBucketer for coverage values"""
    print("\n=== Testing ValueBucketer ===\n")

    bucketer = create_coverage_bucketer(max_coverage=10_000_000)

    print(f"  Number of buckets: {bucketer.num_buckets}")

    # Test various coverage values
    test_values = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]

    print("  Bucketing test values:")
    for val in test_values:
        bucket = bucketer.bucket(val)
        rep = bucketer.representative(bucket)
        print(f"    {val:>10,} -> bucket {bucket:2d} -> representative {rep:,.0f}")

    # Create dimension from bucketer
    dim = bucketer.create_dimension("coverage")
    assert dim.dim_type == DimensionType.BUCKETED
    print(f"\n  Created dimension with {dim.cardinality} values - OK")


def test_representation_comparison():
    """Compare indexed vs object representation"""
    print("\n=== Testing Representation Comparison ===\n")

    # Generate test seeds
    num_seeds = 1000
    test_seeds = []
    for i in range(num_seeds):
        depth = (i % 6) + 1
        mode = [SearchMode.LIGHTNING, SearchMode.WAVE,
                SearchMode.CRYSTAL, SearchMode.OSMOSIS][i % 4]
        direction = [SeedDirection.FORWARD, SeedDirection.BACKWARD,
                    SeedDirection.BILATERAL][i % 3]
        seed = TacticalSeed(i, depth, mode, direction)
        test_seeds.append(seed)

    # Create encoder
    encoder = create_seed_encoder(max_positions=num_seeds, max_depth=6)

    print(f"  Generated {num_seeds} test seeds")
    print(f"  Encoder state space: {encoder.total_states:,}")

    # Compare representations
    comparer = StateRepresentationComparer(encoder)
    results = comparer.compare_representations(test_seeds)

    print("\n  Compression results:")
    for r in results:
        print(f"    {r}")

    # Find best
    best = comparer.find_best_representation(test_seeds)
    print(f"\n  Best representation: {best.strategy_name}")
    print(f"    Size: {best.compressed_size:,} bytes")
    print(f"    Ratio: {best.compression_ratio:.1f}x")


def test_metaseed_compression_estimate():
    """Test meta-seed compression estimation"""
    print("\n=== Testing Meta-Seed Compression Estimate ===\n")

    # Simulate Connect4-like meta-seeds
    num_roots = 8599
    num_derivations = 154967

    # Create fake derivation data
    root_seeds = list(range(num_roots))
    derivation_moves = [(i % num_roots, i % 7) for i in range(num_derivations)]

    raw_bytes, compressed_bytes, ratio = estimate_metaseed_compression(
        root_seeds, derivation_moves)

    print(f"  Roots: {num_roots:,}")
    print(f"  Derivations: {num_derivations:,}")
    print(f"  Raw size: {raw_bytes:,} bytes ({raw_bytes/1024:.1f} KB)")
    print(f"  Compressed: {compressed_bytes:,} bytes ({compressed_bytes/1024:.1f} KB)")
    print(f"  Compression ratio: {ratio:.1f}x")


def test_connect4_comparison():
    """Compare representation strategies on Connect4 seeds"""
    print("\n=== Connect4 Seed Representation Comparison ===\n")

    try:
        from holos.games.connect4 import Connect4Game, C4State
    except ImportError:
        print("  [SKIP] Connect4 not available")
        return

    game = Connect4Game()

    # Generate some positions and create seeds from them
    start = C4State()
    positions = []
    frontier = {game.hash_state(start): start}

    # Expand a few layers to get positions
    for depth in range(3):
        next_frontier = {}
        for h, state in frontier.items():
            positions.append((h, state))
            for child, move in game.get_successors(state):
                ch = game.hash_state(child)
                if ch not in next_frontier and len(next_frontier) < 1000:
                    next_frontier[ch] = child
        frontier = next_frontier
        if len(positions) >= 1000:
            break

    print(f"  Generated {len(positions)} positions")

    # Create seeds from positions
    seeds = []
    for h, state in positions[:500]:
        for depth in [2, 3, 4]:
            seed = TacticalSeed(h, depth, SearchMode.WAVE, SeedDirection.FORWARD, state)
            seeds.append(seed)

    print(f"  Created {len(seeds)} seed configurations")

    # Create encoder
    encoder = create_seed_encoder(max_positions=len(positions), max_depth=6)

    # Adjust position_hash encoding (map to position index)
    position_to_idx = {h: i for i, (h, _) in enumerate(positions)}

    # Create adjusted encoder
    dimensions = [
        StateDimension("depth", DimensionType.BOUNDED_INT, min_value=1, max_value=6),
        StateDimension("mode", DimensionType.DISCRETE,
                      discrete_values=[SearchMode.LIGHTNING, SearchMode.WAVE,
                                      SearchMode.CRYSTAL, SearchMode.OSMOSIS]),
        StateDimension("direction", DimensionType.DISCRETE,
                      discrete_values=[SeedDirection.FORWARD, SeedDirection.BACKWARD,
                                      SeedDirection.BILATERAL]),
        StateDimension("position_index", DimensionType.BOUNDED_INT,
                      min_value=0, max_value=len(positions)-1),
    ]

    def extract(seed: TacticalSeed) -> Tuple:
        pos_idx = position_to_idx.get(seed.position_hash, 0)
        return (seed.depth, seed.mode, seed.direction, pos_idx)

    def construct(values: Tuple) -> TacticalSeed:
        depth, mode, direction, pos_idx = values
        h = positions[pos_idx][0] if pos_idx < len(positions) else 0
        return TacticalSeed(h, depth, mode, direction)

    encoder = IndexedStateEncoder(dimensions, extract, construct)

    # Compare
    comparer = StateRepresentationComparer(encoder)
    results = comparer.compare_representations(seeds)

    print("\n  Compression comparison:")
    for r in results:
        print(f"    {r}")

    # Show the winner
    best = results[0]
    print(f"\n  WINNER: {best.strategy_name}")
    print(f"    {len(seeds)} seeds in {best.compressed_size:,} bytes")
    print(f"    = {best.compressed_size / len(seeds):.1f} bytes/seed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("COMPRESSION-AWARE STATE REPRESENTATION TESTS")
    print("=" * 60)

    test_dimension_encoding()
    test_indexed_encoder()
    test_compression_aware_value()
    test_value_bucketer()
    test_representation_comparison()
    test_metaseed_compression_estimate()
    test_connect4_comparison()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
