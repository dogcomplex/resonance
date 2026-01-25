"""
HOLOS - Hierarchical Omniscient Learning and Optimization System

A universal bidirectional search framework for game-theoretic and optimization problems.

Key Concepts:
- GameInterface: Abstract interface for any game/lattice structure
- HOLOSSolver: Universal solver using bidirectional search
- Hologram: Compressed storage of solved states
- SessionManager: Multi-round handling for incremental solving
- CompressionAwareSeedValue: Compression-aware efficiency for seeds

The insight: HOLOS maps ALL paths in a game space. Winning/losing are just
properties we care about - the algorithm finds ALL reachable states and their
relationships, then filters for whatever objective matters.

Layers (with compression-aware efficiency):
- Layer 0: The base game (e.g., chess endgames) - game positions
- Layer 1: Seed tactics - single seed optimization with compression metrics
- Layer 2: Strategy - multi-seed coordination + compression schemes
- Layer 3: Balance - compute/storage policy decisions

Each layer uses the same bidirectional principle:
- Forward: Expand from initial states toward boundary
- Backward: Expand from boundary toward initial states
- Meeting: Where waves connect, we have solved paths

Compression hierarchy (multiplicative):
- Game rules: ~13x (positions to seeds)
- Derivation structure: ~6x (meta-seeds)
- Standard compression: ~8x (gzip)
"""

from .holos import (
    GameInterface,
    SearchMode,
    SeedPoint,
    GoalCondition,
    LightningProbe,
    HOLOSSolver,
)

from .storage import (
    Hologram,
    SpinePath,
    SeedFrontierMapping,
)

from .session import (
    SessionManager,
    SessionState,
)

from .full_search import (
    DiskBackedHologram,
    FullSearchSession,
    FullSearchState,
)

from .compression import (
    IndexedStateEncoder,
    CompressionAwareSeedValue,
    StateDimension,
    DimensionType,
    ValueBucketer,
    StateRepresentationComparer,
    create_seed_encoder,
    create_coverage_bucketer,
)

__version__ = "0.1.2"
__all__ = [
    # Core algorithm
    "GameInterface",
    "SearchMode",
    "SeedPoint",
    "GoalCondition",
    "LightningProbe",
    "HOLOSSolver",
    # Storage
    "Hologram",
    "SpinePath",
    "SeedFrontierMapping",
    "DiskBackedHologram",
    # Session management
    "SessionManager",
    "SessionState",
    # Full search (large-scale)
    "FullSearchSession",
    "FullSearchState",
    # Compression (Update 16)
    "IndexedStateEncoder",
    "CompressionAwareSeedValue",
    "StateDimension",
    "DimensionType",
    "ValueBucketer",
    "StateRepresentationComparer",
    "create_seed_encoder",
    "create_coverage_bucketer",
]
