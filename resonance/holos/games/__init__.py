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

- sudoku: Sudoku puzzle solver (Layer 0)
  - Completed valid grids as boundary
  - Single-player (any child solved = parent solvable)
  - MRV heuristic for move ordering

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

from .chess import (
    ChessGame, ChessValue, ChessState,
    # Targeting support (consolidated from chess_targeted.py)
    TargetedChessGame, create_targeted_solver,
    # Material utilities
    get_material_string, get_parent_materials, enumerate_material_positions,
    get_8piece_variants, material_string, parse_material_string,
    random_position, create_chess_solver
)
from .connect4 import Connect4Game, C4State, C4Value, C4Features
from .sudoku import SudokuGame, SudokuState, SudokuValue, SudokuFeatures
from .seeds import SeedGame, SeedConfiguration, SeedValue, SeedSpec, ModeDecision, ModeSelector
from .strategy import GoalCondition, StrategyGame, StrategyState, StrategyValue

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
    # Sudoku (Layer 0)
    "SudokuGame",
    "SudokuState",
    "SudokuValue",
    "SudokuFeatures",
    # Targeted Chess (backwards compatibility - use ChessGame with target_material)
    "TargetedChessGame",  # Deprecated wrapper
    "create_targeted_solver",
    # Material utilities
    "get_material_string",
    "get_parent_materials",
    "enumerate_material_positions",
    "get_8piece_variants",  # Alias for get_parent_materials
    "material_string",  # Alias for get_material_string
    "parse_material_string",
    "random_position",
    "create_chess_solver",
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
