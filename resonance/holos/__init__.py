"""
HOLOS - Hierarchical Omniscient Learning and Optimization System

A universal bidirectional search framework for game-theoretic and optimization problems.

Key Concepts:
- GameInterface: Abstract interface for any game/lattice structure
- HOLOSSolver: Universal solver using bidirectional search
- Hologram: Compressed storage of solved states
- SessionManager: Multi-round handling for incremental solving

The insight: HOLOS maps ALL paths in a game space. Winning/losing are just
properties we care about - the algorithm finds ALL reachable states and their
relationships, then filters for whatever objective matters.

Layers:
- Layer 0: The base game (e.g., chess endgames)
- Layer 1: Seed selection (which positions to expand from, at what depth)
- Layer 2+: Meta-strategy (how to allocate compute, which materials to prioritize)

Each layer uses the same bidirectional principle:
- Forward: Expand from initial states toward boundary
- Backward: Expand from boundary toward initial states
- Meeting: Where waves connect, we have solved paths
"""

from .core import (
    GameInterface,
    SearchMode,
    SeedPoint,
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

__version__ = "0.1.0"
__all__ = [
    # Core algorithm
    "GameInterface",
    "SearchMode",
    "SeedPoint",
    "LightningProbe",
    "HOLOSSolver",
    # Storage
    "Hologram",
    "SpinePath",
    "SeedFrontierMapping",
    # Session management
    "SessionManager",
    "SessionState",
]
