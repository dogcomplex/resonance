"""
holos/compression.py - Compression-Aware State Representation

Core insight: Each layer trades compute for storage. This module provides:
1. Index-based state encoding (compact integer indices vs full objects)
2. Compression-aware efficiency metrics
3. Automatic comparison of representation strategies
4. Bucketing for continuous values

The key innovation: For well-defined games, states can be enumerated and
referenced by index. This is especially powerful for meta-layers (seeds,
strategies) which have bounded, stable state spaces.

Usage:
    encoder = IndexedStateEncoder(state_space_definition)
    idx = encoder.encode(state)  # Object -> int
    state = encoder.decode(idx)  # int -> Object

    # Compare compression strategies
    comparer = StateRepresentationComparer(encoder)
    best = comparer.find_best_representation(states)
"""

import gzip
import struct
import math
from typing import (
    List, Tuple, Optional, Any, Set, Dict, Generic, TypeVar,
    Callable, Iterator, Union
)
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from abc import ABC, abstractmethod

# Type variables
S = TypeVar('S')  # State type
V = TypeVar('V')  # Value type


# ============================================================
# INDEX-BASED STATE ENCODING
# ============================================================

class DimensionType(IntEnum):
    """Types of dimensions in state space"""
    DISCRETE = 0      # Finite set of values (enum-like)
    BOUNDED_INT = 1   # Integer in [min, max]
    BUCKETED = 2      # Continuous value bucketed to discrete


@dataclass
class StateDimension:
    """
    Defines one dimension of the state space.

    For seeds:
    - depth: BOUNDED_INT(1, 6) -> 6 values
    - mode: DISCRETE(4 values) -> 4 values
    - direction: DISCRETE(3 values) -> 3 values
    - position_index: BOUNDED_INT(0, N-1) -> N values

    Total space: 6 * 4 * 3 * N = 72N
    """
    name: str
    dim_type: DimensionType

    # For DISCRETE: list of valid values
    discrete_values: List[Any] = field(default_factory=list)

    # For BOUNDED_INT: range
    min_value: int = 0
    max_value: int = 0

    # For BUCKETED: bucket boundaries
    bucket_boundaries: List[float] = field(default_factory=list)

    @property
    def cardinality(self) -> int:
        """Number of possible values in this dimension"""
        if self.dim_type == DimensionType.DISCRETE:
            return len(self.discrete_values)
        elif self.dim_type == DimensionType.BOUNDED_INT:
            return self.max_value - self.min_value + 1
        elif self.dim_type == DimensionType.BUCKETED:
            return len(self.bucket_boundaries) + 1
        return 0

    def encode_value(self, value: Any) -> int:
        """Convert value to index within this dimension"""
        if self.dim_type == DimensionType.DISCRETE:
            try:
                return self.discrete_values.index(value)
            except ValueError:
                raise ValueError(f"Value {value} not in discrete values: {self.discrete_values}")

        elif self.dim_type == DimensionType.BOUNDED_INT:
            if not (self.min_value <= value <= self.max_value):
                raise ValueError(f"Value {value} not in range [{self.min_value}, {self.max_value}]")
            return value - self.min_value

        elif self.dim_type == DimensionType.BUCKETED:
            # Find which bucket this value falls into
            for i, boundary in enumerate(self.bucket_boundaries):
                if value < boundary:
                    return i
            return len(self.bucket_boundaries)

        return 0

    def decode_value(self, index: int) -> Any:
        """Convert index back to value"""
        if self.dim_type == DimensionType.DISCRETE:
            return self.discrete_values[index]

        elif self.dim_type == DimensionType.BOUNDED_INT:
            return index + self.min_value

        elif self.dim_type == DimensionType.BUCKETED:
            # Return bucket midpoint or representative value
            if index == 0:
                if self.bucket_boundaries:
                    return self.bucket_boundaries[0] / 2
                return 0
            elif index >= len(self.bucket_boundaries):
                return self.bucket_boundaries[-1] * 1.5 if self.bucket_boundaries else 0
            else:
                return (self.bucket_boundaries[index-1] + self.bucket_boundaries[index]) / 2

        return None


class IndexedStateEncoder(Generic[S]):
    """
    Encodes states as compact integer indices.

    The state space is defined by a list of dimensions, and states are
    encoded as a single integer index into the product space.

    Example for TacticalSeed:
        dimensions = [
            StateDimension("depth", DimensionType.BOUNDED_INT, min_value=1, max_value=6),
            StateDimension("mode", DimensionType.DISCRETE,
                          discrete_values=[LIGHTNING, WAVE, CRYSTAL, OSMOSIS]),
            StateDimension("direction", DimensionType.DISCRETE,
                          discrete_values=[FORWARD, BACKWARD, BILATERAL]),
            StateDimension("position", DimensionType.BOUNDED_INT, min_value=0, max_value=N-1),
        ]
        encoder = IndexedStateEncoder(dimensions, extract_fn, construct_fn)
    """

    def __init__(self,
                 dimensions: List[StateDimension],
                 extract_fn: Callable[[S], Tuple[Any, ...]],
                 construct_fn: Callable[[Tuple[Any, ...]], S]):
        """
        Args:
            dimensions: List of dimension definitions
            extract_fn: Function to extract dimension values from state
            construct_fn: Function to construct state from dimension values
        """
        self.dimensions = dimensions
        self.extract_fn = extract_fn
        self.construct_fn = construct_fn

        # Precompute strides for mixed-radix encoding
        self._strides = []
        stride = 1
        for dim in reversed(dimensions):
            self._strides.insert(0, stride)
            stride *= dim.cardinality

        self._total_states = stride

    @property
    def total_states(self) -> int:
        """Total number of possible states in this space"""
        return self._total_states

    @property
    def bits_per_index(self) -> int:
        """Bits needed to encode any index"""
        if self._total_states == 0:
            return 0
        return math.ceil(math.log2(self._total_states))

    @property
    def bytes_per_index(self) -> int:
        """Bytes needed to encode any index (packed)"""
        bits = self.bits_per_index
        if bits <= 8:
            return 1
        elif bits <= 16:
            return 2
        elif bits <= 32:
            return 4
        else:
            return 8

    def encode(self, state: S) -> int:
        """Convert state to compact integer index"""
        values = self.extract_fn(state)

        if len(values) != len(self.dimensions):
            raise ValueError(f"Expected {len(self.dimensions)} values, got {len(values)}")

        index = 0
        for dim, value, stride in zip(self.dimensions, values, self._strides):
            dim_index = dim.encode_value(value)
            index += dim_index * stride

        return index

    def decode(self, index: int) -> S:
        """Convert integer index back to state"""
        if not (0 <= index < self._total_states):
            raise ValueError(f"Index {index} out of range [0, {self._total_states})")

        values = []
        remaining = index

        for dim, stride in zip(self.dimensions, self._strides):
            dim_index = remaining // stride
            remaining = remaining % stride
            values.append(dim.decode_value(dim_index))

        return self.construct_fn(tuple(values))

    def encode_batch(self, states: List[S]) -> bytes:
        """Encode multiple states to compact bytes"""
        indices = [self.encode(s) for s in states]

        # Pack based on bits needed
        bytes_per = self.bytes_per_index

        if bytes_per == 1:
            return struct.pack(f'{len(indices)}B', *indices)
        elif bytes_per == 2:
            return struct.pack(f'{len(indices)}H', *indices)
        elif bytes_per == 4:
            return struct.pack(f'{len(indices)}I', *indices)
        else:
            return struct.pack(f'{len(indices)}Q', *indices)

    def decode_batch(self, data: bytes) -> List[S]:
        """Decode bytes back to states"""
        bytes_per = self.bytes_per_index

        if bytes_per == 1:
            indices = struct.unpack(f'{len(data)}B', data)
        elif bytes_per == 2:
            indices = struct.unpack(f'{len(data)//2}H', data)
        elif bytes_per == 4:
            indices = struct.unpack(f'{len(data)//4}I', data)
        else:
            indices = struct.unpack(f'{len(data)//8}Q', data)

        return [self.decode(i) for i in indices]


# ============================================================
# COMPRESSION-AWARE SEED VALUE
# ============================================================

@dataclass(frozen=True)
class CompressionAwareSeedValue:
    """
    Value of a seed accounting for compression efficiency.

    The key insight: a seed is only useful if storing it (plus reconstruction
    metadata) is smaller than storing the covered positions directly.

    net_savings = direct_storage - seed_storage

    If net_savings <= 0, the seed has NEGATIVE efficiency and should be
    discarded in favor of direct position storage.
    """
    # Raw coverage metrics
    forward_coverage: int       # Positions reachable forward
    backward_coverage: int      # Positions reachable backward
    compute_cost: int           # FLOPs or time to expand

    # Storage metrics
    seed_storage_bytes: int     # Bytes to store this seed
    frontier_storage_bytes: int # Bytes for initial frontier (if not derived)
    derivation_bytes: int       # Bytes if derived from parent seed

    # Compression context
    bytes_per_position: int = 4  # Default: 4 bytes per position hash

    # Legacy compatibility
    overlap_potential: float = 0.0
    time_to_first_solve: float = 0.0
    solves_per_second: float = 0.0

    @property
    def total_coverage(self) -> int:
        return self.forward_coverage + self.backward_coverage

    @property
    def direct_storage(self) -> int:
        """Storage if we just stored covered positions directly"""
        return self.total_coverage * self.bytes_per_position

    @property
    def seed_storage(self) -> int:
        """Storage for seed-based representation"""
        return self.seed_storage_bytes + self.frontier_storage_bytes

    @property
    def net_savings(self) -> int:
        """Bytes saved by using seed vs direct storage"""
        return self.direct_storage - self.seed_storage

    @property
    def is_worth_storing(self) -> bool:
        """Is this seed better than direct storage?"""
        return self.net_savings > 0

    @property
    def cost(self) -> int:
        """Legacy compatibility: computational cost"""
        return self.compute_cost

    @property
    def efficiency(self) -> float:
        """
        Compression-aware efficiency.

        Returns bytes saved per unit compute cost.
        Negative efficiency means the seed is worse than direct storage.
        """
        if not self.is_worth_storing:
            return -1.0  # Negative = worse than direct storage
        return self.net_savings / max(1, self.compute_cost)

    @property
    def compression_ratio(self) -> float:
        """How much smaller is seed storage vs direct storage?"""
        if self.seed_storage == 0:
            return float('inf')
        return self.direct_storage / self.seed_storage

    def __repr__(self):
        worth = "WORTH" if self.is_worth_storing else "NOT WORTH"
        return (f"CompressionAwareSeedValue(coverage={self.total_coverage}, "
                f"savings={self.net_savings}B, eff={self.efficiency:.1f}, {worth})")

    def __lt__(self, other):
        # Primary: efficiency (compression-aware)
        # Secondary: total coverage
        if self.efficiency != other.efficiency:
            return self.efficiency < other.efficiency
        return self.total_coverage < other.total_coverage

    def __eq__(self, other):
        if not isinstance(other, CompressionAwareSeedValue):
            return False
        return (self.efficiency == other.efficiency and
                self.total_coverage == other.total_coverage)


# ============================================================
# BUCKETING FOR CONTINUOUS VALUES
# ============================================================

class ValueBucketer:
    """
    Buckets continuous values into discrete bins for indexed storage.

    Useful for coverage values, efficiency metrics, etc. where
    approximate equality is sufficient for seed comparison.

    The insight: if a crude seed (bucketed to lower precision) is nearly
    as effective as a precise one, use the crude one - it compresses better.
    """

    def __init__(self,
                 boundaries: List[float] = None,
                 scale: str = "log",  # "linear", "log", or "custom"
                 num_buckets: int = 16,
                 min_value: float = 0,
                 max_value: float = 1e9):
        """
        Args:
            boundaries: Custom bucket boundaries (if scale="custom")
            scale: How to distribute buckets
            num_buckets: Number of buckets for auto-generation
            min_value, max_value: Range for auto-generation
        """
        if boundaries:
            self.boundaries = sorted(boundaries)
        elif scale == "linear":
            step = (max_value - min_value) / num_buckets
            self.boundaries = [min_value + step * i for i in range(1, num_buckets)]
        elif scale == "log":
            # Logarithmic scale for wide-ranging values
            if min_value <= 0:
                min_value = 1
            log_min = math.log10(min_value)
            log_max = math.log10(max_value)
            log_step = (log_max - log_min) / num_buckets
            self.boundaries = [10 ** (log_min + log_step * i) for i in range(1, num_buckets)]
        else:
            self.boundaries = []

    @property
    def num_buckets(self) -> int:
        return len(self.boundaries) + 1

    def bucket(self, value: float) -> int:
        """Return bucket index for value"""
        for i, boundary in enumerate(self.boundaries):
            if value < boundary:
                return i
        return len(self.boundaries)

    def representative(self, bucket_index: int) -> float:
        """Return representative value for bucket"""
        if bucket_index == 0:
            if self.boundaries:
                return self.boundaries[0] / 2
            return 0
        elif bucket_index >= len(self.boundaries):
            if self.boundaries:
                return self.boundaries[-1] * 1.5
            return 0
        else:
            return (self.boundaries[bucket_index-1] + self.boundaries[bucket_index]) / 2

    def create_dimension(self, name: str) -> StateDimension:
        """Create a StateDimension for this bucketer"""
        return StateDimension(
            name=name,
            dim_type=DimensionType.BUCKETED,
            bucket_boundaries=self.boundaries.copy()
        )


# ============================================================
# REPRESENTATION COMPARISON
# ============================================================

@dataclass
class CompressionResult:
    """Result of comparing compression strategies"""
    strategy_name: str
    raw_size: int           # Bytes before compression
    compressed_size: int    # Bytes after gzip
    compression_ratio: float
    encode_time_ms: float
    decode_time_ms: float

    # Quality metrics (for lossy strategies like bucketing)
    exact_reconstruction: bool = True
    max_error: float = 0.0  # For bucketed values

    def __repr__(self):
        lossy = "" if self.exact_reconstruction else f" (lossy, max_err={self.max_error:.2f})"
        return (f"{self.strategy_name}: {self.compressed_size:,} bytes "
                f"({self.compression_ratio:.1f}x){lossy}")


class StateRepresentationComparer:
    """
    Compares different state representation strategies.

    Strategies compared:
    1. Full object (pickle + gzip)
    2. Index-based (integer indices + gzip)
    3. Bucketed index (coarser indices + gzip)
    4. Delta-encoded (store differences from previous)

    The winner is chosen based on compressed size, with quality
    constraints for lossy strategies.
    """

    def __init__(self, encoder: IndexedStateEncoder = None):
        """
        Args:
            encoder: Optional index encoder for indexed strategies
        """
        self.encoder = encoder
        self.results: List[CompressionResult] = []

    def compare_representations(self,
                                states: List[Any],
                                serializer: Callable[[Any], bytes] = None) -> List[CompressionResult]:
        """
        Compare all representation strategies on the given states.

        Args:
            states: List of states to encode
            serializer: Custom serializer for full-object strategy

        Returns:
            List of CompressionResult, sorted by compressed size
        """
        import pickle
        import time

        self.results = []

        if not states:
            return self.results

        # Strategy 1: Full object (pickle)
        if serializer is None:
            serializer = pickle.dumps

        start = time.time()
        raw_data = b''.join(serializer(s) for s in states)
        encode_time = (time.time() - start) * 1000

        compressed = gzip.compress(raw_data)

        start = time.time()
        # Decode test (just decompress, don't unpickle all)
        gzip.decompress(compressed)
        decode_time = (time.time() - start) * 1000

        self.results.append(CompressionResult(
            strategy_name="full_object_pickle",
            raw_size=len(raw_data),
            compressed_size=len(compressed),
            compression_ratio=len(raw_data) / len(compressed) if compressed else 0,
            encode_time_ms=encode_time,
            decode_time_ms=decode_time,
        ))

        # Strategy 2: Index-based (if encoder available)
        if self.encoder:
            start = time.time()
            raw_indices = self.encoder.encode_batch(states)
            encode_time = (time.time() - start) * 1000

            compressed = gzip.compress(raw_indices)

            start = time.time()
            decoded = self.encoder.decode_batch(gzip.decompress(compressed))
            decode_time = (time.time() - start) * 1000

            self.results.append(CompressionResult(
                strategy_name="indexed",
                raw_size=len(raw_indices),
                compressed_size=len(compressed),
                compression_ratio=len(raw_indices) / len(compressed) if compressed else 0,
                encode_time_ms=encode_time,
                decode_time_ms=decode_time,
            ))

        # Strategy 3: Raw indices (no gzip, for comparison)
        if self.encoder:
            raw_indices = self.encoder.encode_batch(states)
            self.results.append(CompressionResult(
                strategy_name="indexed_raw",
                raw_size=len(raw_indices),
                compressed_size=len(raw_indices),  # No compression
                compression_ratio=1.0,
                encode_time_ms=0,
                decode_time_ms=0,
            ))

        # Sort by compressed size
        self.results.sort(key=lambda r: r.compressed_size)

        return self.results

    def find_best_representation(self,
                                  states: List[Any],
                                  max_error: float = 0.0) -> CompressionResult:
        """
        Find the best representation strategy.

        Args:
            states: States to encode
            max_error: Maximum acceptable error for lossy strategies

        Returns:
            Best CompressionResult
        """
        results = self.compare_representations(states)

        # Filter by quality constraints
        acceptable = [r for r in results
                     if r.exact_reconstruction or r.max_error <= max_error]

        if not acceptable:
            # Fall back to first exact result
            exact = [r for r in results if r.exact_reconstruction]
            return exact[0] if exact else results[0]

        return acceptable[0]  # Already sorted by size


# ============================================================
# SEED-SPECIFIC ENCODER
# ============================================================

def create_seed_encoder(max_positions: int,
                        max_depth: int = 6) -> IndexedStateEncoder:
    """
    Create an encoder for TacticalSeed states.

    Args:
        max_positions: Maximum number of positions in seed pool
        max_depth: Maximum expansion depth

    Returns:
        IndexedStateEncoder for TacticalSeed
    """
    from holos.holos import SearchMode
    from holos.games.seeds import TacticalSeed, SeedDirection

    dimensions = [
        StateDimension(
            name="depth",
            dim_type=DimensionType.BOUNDED_INT,
            min_value=1,
            max_value=max_depth
        ),
        StateDimension(
            name="mode",
            dim_type=DimensionType.DISCRETE,
            discrete_values=[SearchMode.LIGHTNING, SearchMode.WAVE,
                           SearchMode.CRYSTAL, SearchMode.OSMOSIS]
        ),
        StateDimension(
            name="direction",
            dim_type=DimensionType.DISCRETE,
            discrete_values=[SeedDirection.FORWARD, SeedDirection.BACKWARD,
                           SeedDirection.BILATERAL]
        ),
        StateDimension(
            name="position_index",
            dim_type=DimensionType.BOUNDED_INT,
            min_value=0,
            max_value=max_positions - 1
        ),
    ]

    def extract(seed: TacticalSeed) -> Tuple[Any, ...]:
        return (seed.depth, seed.mode, seed.direction, seed.position_hash)

    def construct(values: Tuple[Any, ...]) -> TacticalSeed:
        depth, mode, direction, position_hash = values
        return TacticalSeed(position_hash, depth, mode, direction)

    return IndexedStateEncoder(dimensions, extract, construct)


def create_coverage_bucketer(max_coverage: int = 10_000_000) -> ValueBucketer:
    """
    Create a bucketer for coverage values.

    Uses log scale since coverage spans many orders of magnitude.
    """
    return ValueBucketer(
        scale="log",
        num_buckets=32,  # 32 buckets covers ~10 orders of magnitude well
        min_value=1,
        max_value=max_coverage
    )


# ============================================================
# COMPRESSION-AWARE EVALUATION
# ============================================================

def estimate_seed_storage(seed_config: Any,
                          forward_coverage: int,
                          backward_coverage: int,
                          encoder: IndexedStateEncoder = None,
                          include_frontier: bool = True) -> Tuple[int, int]:
    """
    Estimate storage for a seed configuration.

    Args:
        seed_config: The seed configuration
        forward_coverage: Forward coverage count
        backward_coverage: Backward coverage count
        encoder: Optional encoder for indexed storage
        include_frontier: Whether to include frontier storage estimate

    Returns:
        (seed_storage_bytes, frontier_storage_bytes)
    """
    # Seed storage
    if encoder:
        seed_bytes = encoder.bytes_per_index
    else:
        # Estimate: position_hash (8) + depth (1) + mode (1) + direction (1) = 11 bytes
        seed_bytes = 11

    # Frontier storage (if not derived)
    if include_frontier:
        # Estimate: positions need to be stored to reconstruct
        # But with derivation, only root positions needed
        frontier_bytes = 0  # Assume derived
    else:
        frontier_bytes = 0

    return seed_bytes, frontier_bytes


def evaluate_seed_with_compression(
    seed_config: Any,
    underlying_game: Any,
    encoder: IndexedStateEncoder = None,
    bytes_per_position: int = 4
) -> CompressionAwareSeedValue:
    """
    Evaluate a seed with compression-aware metrics.

    This is the main evaluation function that replaces the old evaluate()
    method with compression awareness.
    """
    from holos.games.seeds import TacticalSeedGame, SeedDirection

    # Create temporary evaluator
    game = TacticalSeedGame(underlying_game, max_depth=6)

    # Get coverage (expensive - runs Layer 0 expansion)
    position_state = seed_config._state
    if position_state is None:
        return CompressionAwareSeedValue(
            forward_coverage=0,
            backward_coverage=0,
            compute_cost=seed_config.cost(),
            seed_storage_bytes=0,
            frontier_storage_bytes=0,
            derivation_bytes=0,
            bytes_per_position=bytes_per_position,
        )

    # Measure coverage
    forward_coverage = 0
    backward_coverage = 0

    if seed_config.direction in [SeedDirection.FORWARD, SeedDirection.BILATERAL]:
        forward_positions = game._expand_forward(
            position_state, seed_config.depth, seed_config.mode)
        forward_coverage = len(forward_positions)

    if seed_config.direction in [SeedDirection.BACKWARD, SeedDirection.BILATERAL]:
        backward_positions = game._expand_backward(
            position_state, seed_config.depth, seed_config.mode)
        backward_coverage = len(backward_positions)

    # Estimate storage
    seed_bytes, frontier_bytes = estimate_seed_storage(
        seed_config, forward_coverage, backward_coverage, encoder)

    return CompressionAwareSeedValue(
        forward_coverage=forward_coverage,
        backward_coverage=backward_coverage,
        compute_cost=seed_config.cost(),
        seed_storage_bytes=seed_bytes,
        frontier_storage_bytes=frontier_bytes,
        derivation_bytes=1,  # If derived from parent, just 1 move byte
        bytes_per_position=bytes_per_position,
    )


# ============================================================
# META-SEED COMPRESSION HELPERS
# ============================================================

def estimate_metaseed_compression(
    root_seeds: List[int],
    derivation_moves: List[Tuple[int, int]],  # (parent_idx, move)
    seed_encoder: IndexedStateEncoder = None
) -> Tuple[int, int, float]:
    """
    Estimate compression for meta-seed representation.

    Args:
        root_seeds: List of root seed indices
        derivation_moves: List of (parent_index, move) pairs
        seed_encoder: Encoder for seed indices

    Returns:
        (raw_bytes, compressed_bytes, ratio)
    """
    # Raw bytes
    bytes_per_root = seed_encoder.bytes_per_index if seed_encoder else 4
    bytes_per_move = 1  # Moves 0-6 in Connect4 fit in 1 byte

    raw_bytes = (len(root_seeds) * bytes_per_root +
                 len(derivation_moves) * (bytes_per_root + bytes_per_move))

    # Simulate compression
    # Pack roots
    root_data = struct.pack(f'{len(root_seeds)}I', *root_seeds)

    # Pack derivation moves (parent_idx, move)
    move_data = b''
    for parent_idx, move in derivation_moves:
        move_data += struct.pack('IB', parent_idx, move)

    full_data = root_data + move_data
    compressed = gzip.compress(full_data)

    ratio = len(full_data) / len(compressed) if compressed else 1.0

    return len(full_data), len(compressed), ratio
