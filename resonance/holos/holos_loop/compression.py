"""
holos_loop/compression.py - Compression-Aware Value Metrics

Imported and adapted from original compression.py.

Core insight: Each layer trades compute for storage. Seeds are only useful
if storing them (plus reconstruction metadata) is smaller than storing
the covered positions directly.

Physics interpretation:
    Compression is about INFORMATION DENSITY.
    - High compression = information is STRUCTURED (patterns, regularities)
    - Low compression = information is RANDOM (no exploitable patterns)

    The wave system naturally discovers structure through closures.
    Closures represent places where forward and backward information AGREE,
    which is exactly the kind of regularity that enables compression.

    Equivalence classes are another form of discovered structure -
    states with the same features can be compressed to a single representative.
"""

import math
from typing import List, Tuple, Optional, Any, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from enum import IntEnum

S = TypeVar('S')  # State type


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

    Physics interpretation:
        This is like measuring the FREE ENERGY of a configuration.
        Positive net_savings = thermodynamically favorable
        Negative net_savings = unfavorable, will be selected against
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

    # Physics metrics (new)
    closure_count: int = 0       # Closures discovered during expansion
    interior_count: int = 0      # Interiors sealed
    energy_cost: float = 0.0     # Energy consumed during expansion

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
            return -1.0
        return self.net_savings / max(1, self.compute_cost)

    @property
    def compression_ratio(self) -> float:
        """How much smaller is seed storage vs direct storage?"""
        if self.seed_storage == 0:
            return float('inf')
        return self.direct_storage / self.seed_storage

    @property
    def closure_density(self) -> float:
        """Closures per position explored"""
        if self.total_coverage == 0:
            return 0.0
        return self.closure_count / self.total_coverage

    @property
    def energy_efficiency(self) -> float:
        """Coverage per unit energy"""
        if self.energy_cost <= 0:
            return float('inf')
        return self.total_coverage / self.energy_cost

    def __repr__(self):
        worth = "WORTH" if self.is_worth_storing else "NOT WORTH"
        return (f"CompressionAwareSeedValue(coverage={self.total_coverage}, "
                f"savings={self.net_savings}B, eff={self.efficiency:.1f}, "
                f"closures={self.closure_count}, {worth})")

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

    Physics interpretation:
        Each dimension is a DEGREE OF FREEDOM in the configuration space.
        The total state space is the PRODUCT of all dimensions.
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

    Physics interpretation:
        This is like assigning QUANTUM NUMBERS to states.
        Each dimension is a quantum number, and the full state is specified
        by the tuple of all quantum numbers.

        The encoding is a bijection between states and integers,
        which enables efficient storage and lookup.
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


# ============================================================
# BUCKETING UTILITIES
# ============================================================

class ValueBucketer:
    """
    Buckets continuous values into discrete bins for indexed storage.

    Useful for coverage values, efficiency metrics, etc. where
    approximate equality is sufficient for seed comparison.

    Physics interpretation:
        Bucketing is a form of COARSE-GRAINING.
        Fine details are lost, but the essential structure is preserved.
        This is analogous to renormalization in physics.
    """

    def __init__(self,
                 boundaries: List[float] = None,
                 scale: str = "log",
                 num_buckets: int = 16,
                 min_value: float = 0,
                 max_value: float = 1e9):
        if boundaries:
            self.boundaries = sorted(boundaries)
        elif scale == "linear":
            step = (max_value - min_value) / num_buckets
            self.boundaries = [min_value + step * i for i in range(1, num_buckets)]
        elif scale == "log":
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
# CONVENIENCE FUNCTIONS
# ============================================================

def estimate_seed_value(
    forward_coverage: int,
    backward_coverage: int,
    compute_cost: int,
    seed_bytes: int = 32,
    bytes_per_position: int = 4,
    closures: int = 0,
    energy: float = 0.0
) -> CompressionAwareSeedValue:
    """
    Quick estimation of seed value with reasonable defaults.

    Args:
        forward_coverage: Positions reachable forward
        backward_coverage: Positions reachable backward
        compute_cost: Computational cost (iterations, FLOPs, etc.)
        seed_bytes: Bytes to store the seed itself
        bytes_per_position: Bytes per position hash
        closures: Number of closures discovered
        energy: Energy consumed

    Returns:
        CompressionAwareSeedValue with all metrics computed
    """
    return CompressionAwareSeedValue(
        forward_coverage=forward_coverage,
        backward_coverage=backward_coverage,
        compute_cost=compute_cost,
        seed_storage_bytes=seed_bytes,
        frontier_storage_bytes=0,
        derivation_bytes=0,
        bytes_per_position=bytes_per_position,
        closure_count=closures,
        energy_cost=energy
    )


def compare_seeds(seeds: List[CompressionAwareSeedValue]) -> List[Tuple[int, CompressionAwareSeedValue]]:
    """
    Compare seeds by efficiency, returning sorted (rank, seed) pairs.

    Seeds with negative efficiency (worse than direct storage) are ranked last.
    """
    # Split into worth-storing and not-worth-storing
    worth = [(i, s) for i, s in enumerate(seeds) if s.is_worth_storing]
    not_worth = [(i, s) for i, s in enumerate(seeds) if not s.is_worth_storing]

    # Sort by efficiency (descending)
    worth.sort(key=lambda x: -x[1].efficiency)
    not_worth.sort(key=lambda x: x[1].efficiency)  # Less negative is better

    return worth + not_worth
