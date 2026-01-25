"""
holos_loop - The Sieve Paradigm

Evolution: Waves → Closures → Sieves

The sieve is the universal primitive. Everything else is parameterization:
- Games are rule sets
- Layers are frequency bands
- Modes are resonance patterns
- Solutions are stable interference patterns

The Sieve:
    - sieve.py: Pattern, Amplitude, Rule, Sieve (THE primitive)
    - analog_sieve.py: Continuous field version
    - meta_sieve.py: Stacked sieves, rule learning
    - compile_game.py: GameInterface → Rules compiler

Legacy Engine (being absorbed):
    - holos.py: GameInterface, HOLOSSolver (the old engine)
    - storage.py: Hologram, SpinePath (output structures)
    - closure.py: ClosureType, ClosureDetector, ModeEmergence
    - unified_closure.py: Full integration system

Layer Architecture (now frequency bands):
    - layer1_paths.py: Paths as first-class objects
    - layer2_covers.py: Ordered path covers
    - layer3_policy.py: Resource-aware policies
    - wave_system.py: Multi-layer orchestration

Physics Extensions:
    - pressure.py: Pressure-based expansion
    - quantum_closure.py: Complex amplitude tracking

Utilities:
    - goals.py: GoalCondition for targeted search
    - session_state.py: Persistence and checkpointing
    - compression.py: Compression-aware value metrics
    - memory.py: Memory management

Games:
    - games/connect4.py: Connect4Game
    - games/sudoku.py: SudokuGame
    - games/chess.py: ChessGame
    - games/seeds.py: TacticalSeed utilities
"""

__version__ = "2.0.0"

# THE SIEVE (Universal Primitive)
from .sieve import (
    Pattern,
    Amplitude,
    Rule,
    FlexibleRule,
    MatchMode,
    Sieve,
    solve as sieve_solve,
    value_to_phase,
    phase_to_value,
)

from .compile_game import (
    GameSieve,
    LazyRuleGenerator,
    state_to_pattern,
    pattern_to_state,
    compile_game_to_rules,
    solve_game_on_sieve,
)

from .meta_sieve import (
    Observation,
    MetaSieve,
    HierarchicalSieve,
    learn_rules,
    infer_rules_from_traces,
)

from .analog_sieve import (
    Manifold,
    GridManifold,
    EmbeddedManifold,
    AnalogSieve,
    discretize_field,
    continuize_patterns,
    create_game_potential,
)

# Legacy engine (from copied dependencies)
from .holos import (
    GameInterface,
    SearchMode,
    SeedPoint,
    HOLOSSolver,
    LightningProbe,
)

# Storage structures
from .storage import (
    Hologram,
    SpinePath,
    SeedFrontierMapping,
)

# Closure system
from .closure import (
    ClosureType,
    ClosureEvent,
    WaveOrigin,
    PhaseAlignment,
    ClosureDetector,
    ModeEmergence,
)

# Unified system
from .unified_closure import (
    UnifiedConfig,
    UnifiedState,
    UnifiedClosureSystem,
    run_unified_search,
)

# Layer architecture
from .layer1_paths import (
    PartialPath,
    PathLayerSolver,
)

from .layer2_covers import (
    PathCover,
    CoverLayerSolver,
)

from .layer3_policy import (
    CoverPolicy,
    PolicyLayerSolver,
)

# Wave system
from .wave_system import (
    LayerMedium,
    create_wave_system,
    run_wave_search,
)

# Goals and targeting
from .goals import (
    GoalCondition,
    GoalAllocation,
    create_material_goal,
    create_pattern_goal,
)

# Session management
from .session_state import (
    SessionPhase,
    RoundStats,
    SessionState,
    create_session,
    load_or_create_session,
)

# Compression utilities
from .compression import (
    CompressionAwareSeedValue,
    StateDimension,
    DimensionType,
    IndexedStateEncoder,
    ValueBucketer,
    estimate_seed_value,
)

# Memory management
from .memory import (
    MemoryConfig,
    MemoryTracker,
    MemoryBudget,
    get_memory_tracker,
    memory_mb,
    memory_ok,
)

# Convenience re-exports
__all__ = [
    # THE SIEVE (Primary)
    "Pattern",
    "Amplitude",
    "Rule",
    "Sieve",
    "sieve_solve",
    "GameSieve",
    "solve_game_on_sieve",
    "MetaSieve",
    "HierarchicalSieve",
    "learn_rules",
    "AnalogSieve",
    "GridManifold",
    # Legacy Core
    "GameInterface",
    "SearchMode",
    "SeedPoint",
    "HOLOSSolver",
    "LightningProbe",
    # Storage
    "Hologram",
    "SpinePath",
    "SeedFrontierMapping",
    # Closure
    "ClosureType",
    "ClosureEvent",
    "WaveOrigin",
    "PhaseAlignment",
    "ClosureDetector",
    "ModeEmergence",
    # Unified
    "UnifiedConfig",
    "UnifiedState",
    "UnifiedClosureSystem",
    "run_unified_search",
    # Layers
    "PartialPath",
    "PathLayerSolver",
    "PathCover",
    "CoverLayerSolver",
    "CoverPolicy",
    "PolicyLayerSolver",
    # Wave
    "LayerMedium",
    "create_wave_system",
    "run_wave_search",
    # Goals
    "GoalCondition",
    "GoalAllocation",
    # Session
    "SessionPhase",
    "SessionState",
    "create_session",
    # Compression
    "CompressionAwareSeedValue",
    "estimate_seed_value",
    # Memory
    "MemoryConfig",
    "MemoryTracker",
    "memory_mb",
    "memory_ok",
]
