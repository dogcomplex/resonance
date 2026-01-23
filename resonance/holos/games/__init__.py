"""
holos/games - Game-specific interfaces for HOLOS

This package contains GameInterface implementations for various games:

- chess: Chess endgames (Layer 0)
  - Uses Syzygy tablebases as boundary
  - Minimax value propagation
  - Captures as lightning moves

- seeds: Seed selection meta-game (Layer 1)
  - Searches over seed configurations
  - Efficiency-based value propagation
  - Meta-game that controls Layer 0

Future games could include:
- Go endgames
- Optimization problems (TSP, scheduling)
- Combinatorial games (Connect-4, etc.)

Each game implements the GameInterface from holos.core:
- hash_state: Hash for deduplication
- get_successors: Forward moves
- get_predecessors: Backward moves
- is_boundary: Check for known value
- propagate_value: Combine child values
"""

from .chess import ChessGame, ChessValue, ChessState
from .seeds import SeedGame, SeedConfiguration, SeedValue, SeedSpec

__all__ = [
    # Chess
    "ChessGame",
    "ChessValue",
    "ChessState",
    # Seeds
    "SeedGame",
    "SeedConfiguration",
    "SeedValue",
    "SeedSpec",
]
