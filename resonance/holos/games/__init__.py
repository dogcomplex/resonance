"""
holos/games - Game-specific interfaces for HOLOS

This package contains GameInterface implementations for various games:

- chess: Chess endgames (Layer 0)
  - Uses Syzygy tablebases as boundary
  - Minimax value propagation
  - Captures as lightning moves

- connect4: Connect-4 full game (Layer 0)
  - Terminal positions as boundary (no external tablebase needed)
  - Minimax value propagation
  - First player (X) wins with perfect play

- seeds: Seed selection meta-game (Layer 1)
  - Searches over seed configurations
  - Efficiency-based value propagation
  - Meta-game that controls Layer 0

- strategy: Goal/budget allocation meta-game (Layer 2)
  - Searches over goal configurations
  - Completeness-based value propagation
  - Meta-meta-game that controls Layer 1

Each game implements the GameInterface from holos.holos:
- hash_state: Hash for deduplication
- get_successors: Forward moves
- get_predecessors: Backward moves
- is_boundary: Check for known value
- propagate_value: Combine child values
"""

from .chess import ChessGame, ChessValue, ChessState
from .connect4 import Connect4Game, C4State, C4Value, C4Features
from .seeds import SeedGame, SeedConfiguration, SeedValue, SeedSpec, ModeDecision, ModeSelector
from .strategy import GoalCondition, StrategyGame, StrategyState, StrategyValue
from .chess_targeted import TargetedChessGame, create_targeted_solver

__all__ = [
    # Chess (Layer 0)
    "ChessGame",
    "ChessValue",
    "ChessState",
    # Connect-4 (Layer 0)
    "Connect4Game",
    "C4State",
    "C4Value",
    "C4Features",
    # Targeted Chess (deprecated - use GoalCondition)
    "TargetedChessGame",
    "create_targeted_solver",
    # Seeds (Layer 1)
    "SeedGame",
    "SeedConfiguration",
    "SeedValue",
    "SeedSpec",
    "ModeDecision",
    "ModeSelector",
    # Strategy (Layer 2)
    "GoalCondition",
    "StrategyGame",
    "StrategyState",
    "StrategyValue",
]
