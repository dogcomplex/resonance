# LOCUS.md - HOLOS Architecture Source of Truth

**Last Updated**: 2026-01-22
**Version**: 0.1.0
**Status**: Functionally equivalent to fractal_holos3.py with enhancements
**Purpose**: Compressed context for AI understanding of this codebase

---

## Executive Summary

HOLOS (Hierarchical Omniscient Learning and Optimization System) is a **bidirectional search framework** for game-theoretic problems. The core insight: HOLOS maps ALL paths in a game space. Winning/losing are just properties we filter for—the algorithm finds ALL reachable states and their relationships.

The modular `holos/` package is a **game-agnostic refactor** of the original `fractal_holos3.py` monolithic implementation. It preserves functional equivalence while enabling application to any game implementing the `GameInterface`.

---

## Complete File Manifest

```
holos/                              [ROOT PACKAGE]
│
├── __init__.py                     [~65 lines]
│   Purpose: Package exports and documentation
│   Exports: GameInterface, SearchMode, SeedPoint, GoalCondition,
│            LightningProbe, HOLOSSolver, Hologram, SpinePath,
│            SeedFrontierMapping, SessionManager, SessionState
│   Version: 0.1.0
│
├── holos.py                        [~880 lines] (renamed from core.py)
│   Purpose: THE ENGINE - Universal HOLOS algorithm (game-agnostic)
│   Classes: GameInterface, SearchMode, SeedPoint, GoalCondition,
│            LightningProbe, HOLOSSolver
│   Key Design: Forward/backward symmetry, bidirectional search,
│               goal-filtered solving. This is used by ALL layers.
│
├── storage.py                      [407 lines]
│   Purpose: Holographic storage structures
│   Classes: SpinePath, SeedFrontierMapping, Hologram
│   Key Design: Spine compression, seed-based frontier reconstruction,
│               equivalence class storage
│
├── session.py                      [460 lines]
│   Purpose: Multi-round session management
│   Classes: SessionPhase, RoundStats, SessionState, SessionManager
│   Key Design: Phase transitions, budget allocation, progress tracking
│
├── demo.py                         [260 lines]
│   Purpose: Demonstration script showing all features
│
├── test_*.py                       [Various test files]
│   Purpose: Verify functional equivalence and correctness
│
├── README.md                       [249 lines]
├── COMPARISON_WITH_FRACTAL_HOLOS3.md [259 lines]
├── LOCUS.md                        [THIS FILE]
│
└── games/                          [GAME IMPLEMENTATIONS - All layers]
    │
    ├── __init__.py                 [~55 lines]
    │   Purpose: Game module exports
    │   Exports: ChessGame, SeedGame, StrategyGame, GoalCondition,
    │            ModeDecision, ModeSelector, etc.
    │
    ├── chess.py                    [~800 lines] - LAYER 0
    │   Purpose: Chess endgame implementation (capabilities)
    │   Classes: Piece, ChessState, ChessValue, ChessFeatures,
    │            SyzygyProbe, ChessGame
    │   Functions: get_material_string, get_parent_materials,
    │              enumerate_material_positions
    │   Key Design: Provides CAPABILITIES for Layer 1/2 to use
    │
    ├── seeds.py                    [~460 lines] - LAYER 1
    │   Purpose: Seed selection meta-game (tactics)
    │   Classes: SeedSpec, SeedConfiguration, SeedValue, SeedGame,
    │            ModeDecision, ModeSelector
    │   Key Design: Lattice search, efficiency optimization,
    │               mode selection learning
    │
    ├── strategy.py                 [~310 lines] - LAYER 2 (NEW)
    │   Purpose: Goal/budget allocation meta-game (strategy)
    │   Classes: GoalCondition, GoalAllocation, StrategyState,
    │            StrategyValue, StrategyGame
    │   Key Design: Resource allocation across goals, completeness
    │               optimization
    │
    └── chess_targeted.py           [DEPRECATED - ~350 lines]
        Note: Use GoalCondition from holos.py instead

holos/
├── run_targeted_kqrr.py            [~200 lines]
│   Purpose: In-process batched targeted search
│
└── run_targeted_subprocess.py      [~180 lines]
    Purpose: Subprocess-isolated targeted search (recommended)
```

**Total Lines**: ~4,500 (excluding LOCUS.md)

---

## Core Philosophy

### The Bidirectional Insight

```
Forward Wave (Deduction)      Backward Wave (Abduction)
         ↓                              ↑
    [Start] ───────────→ ←─────── [Boundary]
                    ↕
              Meeting = Induction
```

**Forward and Backward are MIRRORS.** Backward is just forward on the reversed game.

- **Forward**: Expand from start positions toward boundary (known values)
- **Backward**: Expand from boundary toward start using predecessors
- **Connection**: When waves meet, we have solved paths

### The Three Search Modes

```python
class SearchMode(Enum):
    LIGHTNING = "lightning"  # DFS probe for fast paths (captures only)
    WAVE = "wave"            # BFS for breadth coverage
    CRYSTAL = "crystal"      # Local search around connection points
```

Mode selection is itself a **meta-decision** that can be optimized (Layer 1/2).

### The Layer Architecture

```
holos.py (THE ENGINE) - Used by ALL layers
    │
    ├─→ Layer 2 (strategy.py): Goal/budget allocation
    │       State = StrategyState (goal allocations)
    │       Value = StrategyValue (completeness, efficiency)
    │       HOLOSSolver(StrategyGame) searches strategy space
    │
    ├─→ Layer 1 (seeds.py): Seed selection + mode learning
    │       State = SeedConfiguration (which seeds, depths, modes)
    │       Value = SeedValue (coverage, cost, efficiency)
    │       HOLOSSolver(SeedGame) searches seed space
    │
    └─→ Layer 0 (chess.py): Chess positions (capabilities)
            State = ChessState (position)
            Value = ChessValue (win/lose/draw)
            HOLOSSolver(ChessGame) searches position space
            ↓ queries
        Boundary: Syzygy Tablebases (7-piece endgames)
```

**Key Insight**: The SAME algorithm (holos.py) searches DIFFERENT state spaces at each layer.
- Layer 0 searches POSITIONS
- Layer 1 searches SEED CONFIGURATIONS
- Layer 2 searches GOAL ALLOCATIONS

Each layer uses the same bidirectional search, but with different:
- State types
- Value propagation
- Boundary conditions

---

## Module Deep Dive

### holos.py - THE ENGINE

**Purpose**: Game-agnostic HOLOS algorithm. This is the core search engine used by ALL layers.

#### Classes

| Class | Lines | Responsibility |
|-------|-------|----------------|
| `GameInterface[S, V]` | 80-180 | Abstract interface any game must implement |
| `SearchMode` | 33-38 | Enum: LIGHTNING, WAVE, CRYSTAL |
| `SeedPoint[S]` | 40-50 | Seed with state, mode, priority, depth |
| `GoalCondition` | 53-78 | Targeting (what counts as success) |
| `LightningProbe[S, V]` | 187-280 | Bidirectional DFS to find paths |
| `HOLOSSolver[S, V]` | ~300-880 | Main solver with bidirectional search |

Note: `ModeDecision` and `ModeSelector` moved to `games/seeds.py` (Layer 1 concerns).

#### GoalCondition - Layer 1/2 Targeting

```python
@dataclass
class GoalCondition:
    """Defines what counts as 'success' for a targeted search."""
    target_signatures: Set[str]  # e.g., {'KQRRvKQR'}
    early_terminate_misses: bool = True  # Stop paths to non-goals
    name: str = "unnamed_goal"

    def matches(self, signature: str) -> bool:
        return signature in self.target_signatures
```

**Key Architectural Insight**: Targeting is a STRATEGY decision (Layer 1/2), not a game rule (Layer 0).

- **Layer 0 (chess.py)**: Provides CAPABILITIES (material utilities)
  - `get_material_string(state)` → "KQRRvKQR"
  - `get_parent_materials("KQRRvKQR")` → ["KQRRvKQRQ", "KQRRvKQRR", ...]
  - `enumerate_material_positions("KQRRvKQR", syzygy, 100)` → positions

- **Layer 1/2 (GoalCondition)**: Uses capabilities to define GOALS
  - "Only solve positions reaching KQRRvKQR"
  - "Early-terminate paths to other 7-piece materials"

- **Solver**: Respects goals during expansion
  - Checks `game.get_signature(state)` against `goal.target_signatures`
  - Filters out boundary states that don't match goal

**Why This Separation Matters**:
1. Results are valid building blocks toward full chess solution
2. Filtered states are still CORRECT - just not our current target
3. Can combine partial results from different goal searches later

#### GameInterface Contract

```python
class GameInterface(ABC, Generic[S, V]):
    # REQUIRED METHODS (must implement)
    def hash_state(state: S) -> int           # Deduplication
    def get_successors(state: S) -> List      # Forward moves
    def get_predecessors(state: S) -> List    # Backward moves (unmoves)
    def is_boundary(state: S) -> bool         # Has known value?
    def get_boundary_value(state: S) -> V     # Get known value
    def is_terminal(state: S) -> (bool, V)    # Game over check
    def propagate_value(state: S, children: List[V]) -> V  # GAME-SPECIFIC

    # OPTIONAL METHODS (have defaults)
    def get_features(state: S) -> Any         # Equivalence features (default: None)
    def get_signature(state: S) -> str        # Goal matching (default: None) **NEW**
    def get_lightning_successors(state: S)    # Captures only (default: get_successors)
    def get_lightning_predecessors(state: S)  # Uncaptures only (default: get_predecessors)
    def score_for_lightning(state: S, move)   # MVV-LVA scoring (default: 0.0)
    def generate_boundary_seeds(template, count)  # Auto-generate seeds (default: [])
    def apply_move(state: S, move: Any) -> S  # For spine reconstruction
```

**NEW: `get_signature()` for Goal Matching**

The `get_signature()` method enables Layer 1/2 goal targeting:
- Returns a string identifying the state's "category"
- For chess: material string like "KQRRvKQR"
- Solver checks signatures against `GoalCondition.target_signatures`

**Value propagation is GAME-SPECIFIC**:
- Chess: minimax (White maximizes, Black minimizes)
- Optimization: max efficiency
- Path finding: min cost

#### HOLOSSolver Memory Management (NEW)

**Memory-safe initialization**:
```python
solver = HOLOSSolver(
    game,
    name="my_solver",
    max_memory_mb=4000,         # Process memory limit
    max_frontier_size=2_000_000, # Hard cap on frontier size
    max_backward_depth=2,        # Limit backward expansion (prevents explosion)
)
```

**Key parameters**:
- `max_memory_mb`: When process memory exceeds 90% of this, solving stops
- `max_frontier_size`: Hard cap on next_frontier dict size (default 2M)
- `max_backward_depth`: How many steps backward from boundary (None = unlimited)

**For targeted KQRRvKQR search**:
- Use `max_backward_depth=1` (only go 7→8 piece, not 7→8→9→...)
- Use `max_frontier_size=500_000` to keep memory bounded

#### HOLOSSolver Algorithm

```python
def solve(forward_seeds, backward_seeds=None, max_iterations=100,
          lightning_interval=5, goal: GoalCondition = None):  # NEW: goal param
    # 1. Store goal for expansion filtering (self.current_goal = goal)
    # 2. Initialize frontiers from seeds
    # 3. Auto-generate backward seeds if not provided (via game.generate_boundary_seeds)
    # 4. For each iteration:
    #    a. Check memory limits (stops if > 90% of max_memory_mb)
    #    b. Run lightning probes every N iterations (forward + backward)
    #    c. Expand forward frontier (BFS layer)
    #       - Filter boundary states by goal.target_signatures
    #       - Track goal_filtered stat
    #    d. Expand backward frontier (BFS using predecessors)
    #       - DEPTH LIMITED: Stop at max_backward_depth
    #       - FRONTIER CAPPED: Stop adding when frontier hits max_frontier_size
    #    e. Find connections (where waves overlap)
    #    f. Crystallize around connections
    #       - Clear parent tracking when > 5M solved (memory savings)
    #    g. Propagate values through parent links and equivalence
    # 5. Build and return Hologram
```

**Goal Filtering in Forward Expansion**:
```python
# In _expand_forward(), when reaching boundary:
if self.game.is_boundary(child):
    if self.current_goal is not None:
        sig = self.game.get_signature(child)
        if sig is not None and not self.current_goal.matches(sig):
            self.stats['goal_filtered'] += 1
            if self.current_goal.early_terminate_misses:
                continue  # Don't explore this path
```

#### Statistics Tracked

```python
self.stats = {
    'lightning_probes': 0,    # Number of DFS probes run
    'connections': 0,         # Forward/backward wave meetings
    'crystallized': 0,        # Positions found via crystallization
    'spines_found': 0,        # Spine paths discovered
    'forward_expanded': 0,    # Positions expanded forward
    'backward_expanded': 0,   # Positions expanded backward
    'equiv_shortcuts': 0,     # Solved via equivalence lookup
    'equiv_tracked': 0,       # Positions with features tracked
    'equiv_propagated': 0,    # Positions solved via equivalence propagation
    'minimax_solved': 0,      # Positions solved via minimax
    'goal_filtered': 0,       # Boundary states filtered by goal
    'frontier_capped': 0,     # Times frontier hit size cap (NEW)
    'depth_limited': 0,       # Positions skipped due to depth limit (NEW)
}
```

#### Crystallization Algorithm (core.py lines 769-795)

When forward and backward waves **connect**, crystallization intensifies the search around connection points. This is a form of "local BFS" that mines the most valuable regions.

```python
def _crystallize(self):
    """
    Expand around connection points (matches fractal_holos3).

    For each of the last 10 connections:
      1. Find the state at the connection point
      2. Do 3 layers of local BFS expansion
      3. For each position found, check if it hits boundary
      4. If boundary hit, get value and add to solved set
    """
    for fh, bh, value in self.connections[-10:]:
        state = self.forward_frontier.get(fh) or self.backward_frontier.get(fh)
        if state is None:
            continue

        # Local BFS (like fractal_holos3)
        local = {fh: state}
        local_seen = {fh}

        for _ in range(3):  # 3 layers
            next_local = {}
            for h, s in local.items():
                for child, move in self.game.get_successors(s):
                    ch = self.game.hash_state(child)
                    if ch not in local_seen:
                        local_seen.add(ch)
                        next_local[ch] = child
                        if self.game.is_boundary(child):
                            val = self.game.get_boundary_value(child)
                            if val is not None:
                                self.solved[ch] = val
                                self.stats['crystallized'] += 1
            local = next_local
```

**Key Design Decisions**:
- **Last 10 connections**: Only crystallize around recent connections to avoid redundant work
- **3 layers of BFS**: Balances depth vs. computational cost
- **Boundary check**: Only adds to solved set if position is on boundary (has known value)
- **Consistent with fractal_holos3.py**: Same algorithm, verified identical

**Why Crystallize?**
Connection points are where we have **maximum information density**. Both forward (what's reachable from start) and backward (what leads to known values) waves agree. Expanding here efficiently mines the "richest ore" in the search space.

---

### storage.py - Holographic Storage

**Purpose**: Compressed storage structures for solved states.

#### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `SpinePath` | 36-87 | Compressed principal variation (start → boundary) |
| `SeedFrontierMapping` | 89-154 | Seed + depth → deterministic frontier reconstruction |
| `Hologram` | 156-360 | Main storage: solved states, spines, equivalence |

#### SpinePath - The PV Compression

```python
@dataclass
class SpinePath:
    start_hash: int           # Starting position hash
    moves: List[Any]          # Just the moves, not states
    end_hash: int             # Terminal position hash
    end_value: Any            # Value at end
    depth: int = 0            # Path length (auto-computed)
    checkpoints: List[Tuple]  # (hash, features) at key points
```

**Design Decisions**:
- Store moves only, not intermediate states
- Checkpoints allow O(1) move lookup at key positions
- Supports truncation for partial path queries
- Spines serve TWO purposes:
  1. **Analysis**: Show how a position is solved (PV line)
  2. **Decision Making**: Quick lookup of best move from any position on spine

#### SeedFrontierMapping - THE KEY COMPRESSION INSIGHT

```python
@dataclass
class SeedFrontierMapping:
    seed_hash: int                    # Single seed position
    depth: int                        # Expansion depth
    mode: str                         # "wave", "lightning", etc.
    expansion_params: Dict[str, Any]  # Algorithm parameters
    _frontier: Optional[Set[int]]     # Lazy cache (not serialized)
```

Instead of storing frontier positions, store:
- 1 seed position hash
- depth parameter
- expansion algorithm = deterministic

**Compression Arithmetic**:
```
Depth 1: ~25 positions    (25x compression)
Depth 2: ~275 positions   (275x compression)
Depth 3: ~2775 positions  (2775x compression)
Depth 5: ~8000 positions  (8000x compression)
```

The frontier is a **DERIVED quantity**, not stored.

#### Hologram - The Output Structure

```python
@dataclass
class Hologram:
    name: str
    solved: Dict[int, Any]                    # Hash → value mapping
    spines: List[SpinePath]                   # Compressed PV lines
    boundary_hashes: Set[int]                 # Known boundary states
    connections: List[Tuple[int, int, Any]]   # Where waves met
    equiv_classes: Dict[Any, Set[int]]        # Feature → hashes
    equiv_outcomes: Dict[Any, Optional[Any]]  # Feature → consistent value
    seed_mappings: List[SeedFrontierMapping]  # Compressed frontiers
    stats: Dict[str, int]                     # Solver statistics
```

**Key Methods**:
- `query(h)`: Get value for hash
- `add_spine(spine)`: Add and index spine
- `add_with_features(h, value, features)`: Track equivalence
- `propagate_equivalence()`: Propagate values via feature classes
- `get_spine_for(h)`: Find spine containing position
- `get_best_move(h)`: Get move from spine lookup
- `save(path)` / `load(path)`: Pickle serialization
- `merge(other)`: Combine two holograms

---

### session.py - Multi-Round Management

**Purpose**: Handle incremental solving across multiple rounds.

#### Key Question: Is SessionManager Layer 1 or Layer 2?

**Answer**: BOTH.
- Layer 1 (Seed Selection): Chooses seeds for THIS round
- Layer 2 (Meta-Strategy): Allocates compute across MULTIPLE rounds

#### Classes

| Class | Lines | Responsibility |
|-------|-------|----------------|
| `SessionPhase` | 36-43 | Enum: INIT, LIGHTNING, WAVE, CRYSTAL, COMPLETE |
| `RoundStats` | 45-71 | Statistics for a single solving round |
| `SessionState` | 73-136 | Persistent state across rounds |
| `SessionManager` | 138-441 | Orchestrates multi-round solving |

#### Phase Transitions

```
INIT → LIGHTNING: Always start with fast probing
LIGHTNING → WAVE: When lightning probes saturate (avg spines < 2)
WAVE → CRYSTAL: When wave expansion slows (growth rate < 100/round)
CRYSTAL → COMPLETE: When no more progress (avg solve < 5)
```

#### SessionState Persistence

```python
@dataclass
class SessionState:
    session_id: str
    game_name: str
    phase: SessionPhase
    current_round: int
    total_solved: int
    total_explored: int
    total_connections: int
    rounds: List[RoundStats]          # Round history
    pending_seeds: List[int]          # Seeds to explore
    explored_seeds: Set[int]          # Already explored
    total_budget: float
    budget_used: float
    feature_success: Dict[Any, float] # Meta-learning data
    mode_success: Dict[str, float]    # Mode success rates
```

#### SessionManager Flow

```python
def run_session(solver, initial_seeds, rounds_per_phase=3, iterations_per_round=50):
    while should_continue():
        # 1. Consider phase transition every N rounds
        if current_round % rounds_per_phase == 0:
            advance_phase()

        # 2. Get budget for this round
        budget = min(100.0, budget_remaining())

        # 3. Run round with current seeds
        stats = run_round(solver, seeds, max_iterations, budget)

        # 4. Select seeds for next round based on history
        seeds = select_next_seeds(game)
```

#### Multi-Round Flow Details

**CRITICAL: Solver State Persistence**

The `HOLOSSolver` is designed for **incremental solving**. Calling `solve()` multiple times on the same solver instance **accumulates state**:

```python
# First call: frontiers start empty
solver.solve(seeds1, max_iterations=10)  # Forward: 1000, Backward: 5000

# Second call: frontiers carry over!
solver.solve(seeds2, max_iterations=10)  # Forward: 1000+new, Backward: 5000+new
```

This is **intentional** for multi-round sessions where we want to incrementally build on previous work. However, it means:

1. **For fresh starts**: Create a new solver instance
2. **For incremental**: Reuse the same solver across `run_round()` calls

**Hologram Merging (storage.py lines 300-334)**

When rounds complete, their holograms are merged:

```python
def merge(self, other: 'Hologram') -> 'Hologram':
    """Combine two holograms with deduplication"""
    result = Hologram(f"{self.name}+{other.name}")

    # Union of solved positions
    result.solved = {**self.solved, **other.solved}

    # Union of boundary hashes
    result.boundary_hashes = self.boundary_hashes | other.boundary_hashes

    # Deduplicate spines (same start_hash)
    spine_map = {s.start_hash: s for s in self.spines}
    for s in other.spines:
        if s.start_hash not in spine_map:
            spine_map[s.start_hash] = s
    result.spines = list(spine_map.values())

    # Merge equivalence classes
    for features, hashes in other.equiv_classes.items():
        result.equiv_classes[features] |= hashes

    # Merge connections
    existing = set(self.connections)
    for conn in other.connections:
        if conn not in existing:
            result.connections.append(conn)

    return result
```

**Memory Considerations**

Backward wave expansion can cause memory explosion due to uncaptures:
- Forward move: 1 position → ~25 legal moves
- Backward unmove: 1 position → ~25 unmoves × 3 uncapture options = ~75 positions

The backward frontier can grow **3x faster** than forward. The solver has a memory limit:
```python
if mem > self.max_memory_mb * 0.9:
    print(f"\nMemory limit reached ({mem:.0f} MB)")
    break
```

**Known Issues / Placeholder Code**

- `feature_success` in SessionState is declared but never populated (placeholder for future meta-learning)

---

### games/chess.py - Chess Endgame Implementation

**Purpose**: Layer 0 implementation for chess endgames.

#### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `Piece` | 37-40 | IntEnum for chess pieces (1-12) |
| `ChessState` | 58-103 | Position: pieces list + turn |
| `ChessValue` | 109-116 | Game-theoretic value (+1, 0, -1) |
| `ChessFeatures` | 122-131 | Equivalence class features |
| `SyzygyProbe` | 426-467 | Interface to 7-piece tablebases |
| `ChessGame` | 474-648 | GameInterface implementation |

#### ChessState Design

```python
class ChessState:
    __slots__ = ['pieces', 'turn', '_hash', '_board']

    def __init__(self, pieces, turn='w'):
        self.pieces = tuple(sorted(pieces))  # Canonical ordering
        self.turn = turn
        self._hash = None   # Lazy hash
        self._board = None  # Lazy board array
```

**Hash is CANONICAL** - horizontal flip is folded in for deduplication:
```python
def __hash__(self):
    flipped = tuple(sorted((p, self._flip_h(sq)) for p, sq in self.pieces))
    canonical = min(self.pieces, flipped)  # Smaller is canonical
    return hash((canonical, self.turn))
```

#### Boundary Definition

- **Lower**: 7-piece positions (Syzygy tablebases)
- **Upper**: Configurable max piece count (default 8)

```python
def is_boundary(self, state: ChessState) -> bool:
    return state.piece_count() <= self.min_pieces
```

#### Predecessor Generation (for backward wave)

```python
def generate_predecessors(state, max_uncaptures=3):
    # For each piece that could have moved TO its current square:
    #   1. Calculate FROM squares (reverse move directions)
    #   2. Generate position with piece at FROM
    #   3. Optionally ADD a captured piece at TO (uncapture)
    # Validation:
    #   - Kings not adjacent
    #   - Opponent not in check (they just moved)
```

#### ChessGame Key Methods

| Method | Purpose |
|--------|---------|
| `get_successors()` | Legal moves from position |
| `get_predecessors()` | Unmoves/uncaptures |
| `get_lightning_successors()` | Captures only |
| `get_lightning_predecessors()` | Uncaptures only (piece count increases) |
| `score_for_lightning()` | MVV-LVA scoring (captured piece value) |
| `generate_boundary_seeds()` | Create 7-piece positions from template |
| `apply_move()` | Apply move for spine reconstruction |
| `propagate_value()` | Minimax (white max, black min) |
| `get_signature()` | Material string for goal matching **NEW** |
| `enumerate_positions()` | Generate positions with given material **NEW** |
| `get_parent_signatures()` | Materials that capture down to target **NEW** |

#### Material Utilities (Layer 0 Capabilities)

These standalone functions provide CAPABILITIES that Layer 1/2 can use:

```python
# Get material signature from position
get_material_string(state) -> str  # "KQRRvKQR"

# Get all 8-piece materials that can capture to target
get_parent_materials("KQRRvKQR") -> List[str]
# Returns: ["KQRRvKQRQ", "KQRRvKQRR", "KQRRvKQRB", ...]
# (Each adds one piece to white or black side)

# Generate random valid positions with given material
enumerate_material_positions("KQRRvKQR", syzygy, count=100) -> List[ChessState]
```

**Design Principle**: Layer 0 provides WHAT IS POSSIBLE, Layer 1/2 decides WHAT TO DO.

#### ChessFeatures - Equivalence Class

```python
@dataclass(frozen=True)
class ChessFeatures:
    material_white: Tuple[int, ...]   # Sorted piece types
    material_black: Tuple[int, ...]
    material_balance: int             # Centipawn difference
    piece_count: int
    king_distance: int                # Manhattan distance
    turn: str
```

---

### Goal-Targeted Solving (Refactored Architecture)

**Purpose**: Find paths to specific material configurations using Layer 0/1/2 separation.

#### The Architectural Insight

**BEFORE** (chess_targeted.py - now deprecated):
- Targeting logic baked into game class
- Modified `is_boundary()` and `get_successors()`
- Tightly coupled, hard to compose

**AFTER** (GoalCondition + material utilities):
- Layer 0 (chess.py): Provides CAPABILITIES (material utilities)
- Layer 1/2 (GoalCondition): Defines WHAT TO SEARCH FOR
- Solver: Respects goals during expansion

#### Use Case: Find All KQRRvKQR Solutions

```python
from holos import HOLOSSolver, SeedPoint, SearchMode, GoalCondition
from holos.games.chess import (
    ChessGame, get_parent_materials, enumerate_material_positions
)

# 1. Setup game with material utilities (Layer 0)
game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=8)

# 2. Define goal (Layer 1/2)
goal = GoalCondition(
    target_signatures={"KQRRvKQR"},
    early_terminate_misses=True,  # Filter paths to other materials
    name="KQRRvKQR_only"
)

# 3. Generate seeds using material utilities
parent_materials = get_parent_materials("KQRRvKQR")  # 8-piece sources
forward_positions = []
for mat in parent_materials:
    forward_positions.extend(enumerate_material_positions(mat, game.syzygy, 50))

# 4. Solve with goal filtering
solver = HOLOSSolver(game, name="targeted_search")
seeds = [SeedPoint(p, SearchMode.WAVE) for p in forward_positions]
hologram = solver.solve(seeds, max_iterations=10, goal=goal)

# 5. Check results
print(f"Solved: {len(hologram.solved)}")
print(f"Goal filtered: {solver.stats['goal_filtered']}")
```

#### Why This Design is Better

1. **Composability**: Combine goals (`target_signatures = {"KQRRvKQR", "KQRBvKQR"}`)
2. **Reusability**: Same game instance for different goals
3. **Correctness**: Filtered states are VALID chess positions, just not our target
4. **Incremental**: Can merge results from different goal searches

#### Legacy: Running Subprocess Search

The subprocess runner still works but uses deprecated `chess_targeted.py`:
```bash
python holos/run_targeted_subprocess.py --target KQRRvKQR
```

**Recommended**: Use GoalCondition API directly for new searches.

---

### games/seeds.py - Seed Selection Meta-Game

**Purpose**: Layer 1 implementation - HOLOS searching for how to run HOLOS.

#### The Meta-Game Structure

```
State: SeedConfiguration (set of SeedSpecs)
Moves:
  - Add seed from pool
  - Remove seed
  - Increase/decrease depth
  - Change mode (wave/lightning/crystal)
Value: SeedValue (coverage, cost, efficiency)
Boundary: Empty configuration (coverage=0, cost=0)
```

#### Key Discovery from Experiments

```
1 seed @ depth 5:  8,275 coverage, 1,655 efficiency
20 seeds @ depth 2: 6,100 coverage, 191 efficiency
```

**DEPTH is the dominant variable** (~10x efficiency gain).

The meta-game discovered we were optimizing the WRONG dimension.

#### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `SeedSpec` | 40-48 | Single seed: position_hash, mode, depth |
| `SeedConfiguration` | 51-79 | Frozen set of SeedSpecs + material |
| `SeedValue` | 86-102 | Value tuple: coverage, cost, efficiency |
| `SeedGame` | 109-346 | GameInterface for the meta-game |

#### Lattice Structure

```
Moving "up":    Adding seeds (more coverage, more cost)
Moving "down":  Removing seeds (less coverage, less cost)
Moving "sideways": Changing mode/depth
```

#### SeedGame Value Propagation

Unlike minimax, the meta-game optimizes **efficiency**:
```python
def propagate_value(self, state, child_values):
    if not child_values:
        return None
    # Best efficiency among children
    return max(child_values, key=lambda v: v.efficiency)
```

---

## Key Design Decisions

### 1. Value Type Abstraction

| Implementation | Value Type |
|----------------|------------|
| fractal_holos3.py | Raw `int` (-1, 0, 1) |
| holos/games/chess.py | `ChessValue` wrapper |
| holos/games/seeds.py | `SeedValue(coverage, cost, efficiency)` |

The solver is generic over value type V. Each game defines its own.

### 2. Forward/Backward Symmetry

The insight: `backward = forward(reversed_game)`

For chess:
- Forward successor: Apply legal move
- Backward predecessor: Undo move (potentially uncapturing)

Both use the same algorithm structure, just different game methods.

### 3. Equivalence Classes

**Both rule-based and discovered**:
- **Rule-based**: Symmetry (horizontal flip) built into hash
- **Discovered**: Feature clustering finds positions with same outcome

If all positions with same features have same outcome, we can propagate.

### 4. Spine Paths for Decision Making

Spines aren't just for analysis—they enable O(1) move lookup:
```python
def get_best_move(self, h: int) -> Optional[Any]:
    spine = self.get_spine_for(h)
    if spine:
        return spine.get_move_at(h)
    return None
```

### 5. Memory-Efficient Storage

Key techniques:
1. **Seed compression**: Store 1 seed instead of 8000 frontier positions
2. **Spine compression**: Store moves only, reconstruct states
3. **Equivalence propagation**: Solve many positions at once
4. **Lazy evaluation**: _frontier cache in SeedFrontierMapping

### 6. Phase-Based Solving

```python
class SessionPhase(Enum):
    INIT = "init"           # Just started
    LIGHTNING = "lightning"  # Fast DFS probing
    WAVE = "wave"           # Broad BFS expansion
    CRYSTAL = "crystal"     # Focused deepening
    COMPLETE = "complete"   # Session finished
```

Different modes for different solving stages.

### 7. Mode Selection as Layer 1 Dimension

The choice of search mode (Lightning vs Wave vs Crystal) is **exposed as a dimension of the Layer 1 seed configuration space**, not hardcoded at Layer 0.

**The insight**: `SeedSpec` includes mode as a parameter:
```python
@dataclass(frozen=True)
class SeedSpec:
    position_hash: int      # Which boundary position
    mode: SearchMode        # Lightning, Wave, or Crystal
    depth: int              # How far to expand
```

This means Layer 1 can discover optimal mode allocation through search:
- "Use Lightning for positions with queen vs rook"
- "Use Wave for balanced material"
- "Crystallize around complex endgames"

**Learning mode selection is Layer 1's job**, not hardcoded heuristics. The `ModeSelector` class in `core.py` tracks outcomes to enable this learning.

---

## Consistency with fractal_holos3.py

All items verified by `test_equivalence.py`:

| Feature | Status |
|---------|--------|
| Spine creation in lightning phase | FIXED ✓ |
| Auto-generation of backward seeds | FIXED ✓ |
| Memory tracking with psutil | FIXED ✓ |
| All 10 stats keys present | FIXED ✓ |
| Equivalence class transfer to hologram | FIXED ✓ |
| `apply_move()` for spine tracking | FIXED ✓ |
| `generate_boundary_seeds()` in ChessGame | FIXED ✓ |

**Enhancements over original**:
1. Backward lightning probes (direction="backward")
2. Game-agnostic design (any GameInterface implementation)
3. Mode selection tracking (ModeSelector class)
4. Session management for multi-round solving

---

## Test Coverage

`test_equivalence.py` verifies:

| Test | What It Checks |
|------|----------------|
| ChessGame Interface | All interface methods work correctly |
| Boundary Seed Generation | generate_boundary_seeds() creates valid 7-piece positions |
| Solver Stats Keys | All 10 required stats keys present |
| Memory Tracking | memory_mb() returns valid value |
| Spine Structure | SpinePath creation and attributes |
| Lightning Probe | Both forward and backward directions |
| apply_move Method | Matches get_successors output |

Run with: `python holos/test_equivalence.py`

---

## Usage Examples

### Basic Chess Solving

```python
from holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.chess import ChessGame, random_position

game = ChessGame(syzygy_path="./syzygy", min_pieces=7, max_pieces=8)
solver = HOLOSSolver(game, name="chess_8piece")

positions = [random_position("KQRRvKQRR") for _ in range(100)]
seeds = [SeedPoint(p, SearchMode.WAVE, depth=2) for p in positions if p]

hologram = solver.solve(seeds, max_iterations=50)
```

### Goal-Targeted Solving (NEW)

```python
from holos import HOLOSSolver, SeedPoint, SearchMode, GoalCondition
from holos.games.chess import ChessGame, get_parent_materials, enumerate_material_positions

# Setup
game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=8)
solver = HOLOSSolver(game, name="targeted")

# Define goal: only KQRRvKQR solutions count
goal = GoalCondition(
    target_signatures={"KQRRvKQR"},
    early_terminate_misses=True,
    name="KQRRvKQR_only"
)

# Generate seeds from parent materials
parents = get_parent_materials("KQRRvKQR")
positions = []
for mat in parents[:3]:  # First 3 parent materials
    positions.extend(enumerate_material_positions(mat, game.syzygy, 20))

seeds = [SeedPoint(p, SearchMode.WAVE) for p in positions]
hologram = solver.solve(seeds, max_iterations=10, goal=goal)

print(f"Goal filtered: {solver.stats['goal_filtered']}")  # Non-target paths
```

### Multi-Round Session

```python
from holos.session import create_session

session = create_session("my_session", "chess_8piece", budget=500.0)
session.run_session(solver, initial_seeds, rounds_per_phase=3)
```

### Querying Results

```python
h = game.hash_state(position)
value = hologram.query(h)
best_move = hologram.get_best_move(h)
spine = hologram.get_spine_for(h)
```

### Fresh Start vs Incremental Solving

```python
# Option 1: Fresh solver for each independent problem
solver1 = HOLOSSolver(game, name="problem1")
h1 = solver1.solve(seeds1)

solver2 = HOLOSSolver(game, name="problem2")  # Fresh instance
h2 = solver2.solve(seeds2)

# Option 2: Reset same solver between problems
solver = HOLOSSolver(game, name="multi")
h1 = solver.solve(seeds1)
solver.reset()  # Clear all state
h2 = solver.solve(seeds2)

# Option 3: Incremental (state accumulates - useful for related problems)
solver = HOLOSSolver(game, name="incremental")
h1 = solver.solve(seeds1)  # Solve first batch
h2 = solver.solve(seeds2)  # Builds on previous state (no reset)
```

### Layer 1 Meta-Game

```python
from holos.games.seeds import SeedGame, create_initial_configs

seed_game = SeedGame(material="KQRRvKQR")
seed_game.set_seed_pool(boundary_positions)
configs = create_initial_configs(seed_game, num_configs=20)
# Run HOLOS on config space to find optimal seed selection
```

---

## Future Extensions

1. **Layer 2+ games**: Budget allocation, material priority optimization
2. **More game interfaces**: Go endgames, combinatorial optimization
3. **Distributed solving**: Multiple workers with hologram merging
4. **Neural guidance**: Learn move ordering from spine paths

---

## Changelog

### 2026-01-22 (Update 7) - MEMORY MANAGEMENT & CACHING ANALYSIS
- **CRITICAL FIX**: Backward frontier explosion (12.7M → 38GB crash)
- **Added** `max_frontier_size` parameter to HOLOSSolver
  - Hard cap on next_frontier size (default 2M)
  - Prevents unbounded memory growth
- **Added** `max_backward_depth` parameter to HOLOSSolver
  - Limits backward expansion depth from boundary
  - For KQRRvKQR targeting: use depth=1 (7→8 piece only)
- **Added** `backward_depth` tracking dict in solver
  - Tracks depth of each backward position from boundary
- **Added** `max_equiv_class_size` (default 10K)
  - Prevents memory explosion from huge equivalence classes
- **Added** memory cleanup in crystallization
  - Clears parent tracking when solved > 5M (saves memory)
- **Added** new stats: `frontier_capped`, `depth_limited`
- **Created** `MEMORY_AND_CACHING_ANALYSIS.md` with:
  - Root cause analysis of memory explosion
  - Design for disk-backed frontiers
  - Design for persistent incremental cache
  - Conservative caching principles
- **Tested**: 838K positions solved with ~1.4GB memory (was 38GB crash)

**Recommended parameters for targeted KQRRvKQR**:
```python
solver = HOLOSSolver(
    game, max_memory_mb=8000,
    max_frontier_size=500_000,
    max_backward_depth=1  # Only 7→8 piece
)
```

### 2026-01-22 (Update 6) - CLEAN LAYER SEPARATION
- **MAJOR**: Renamed `core.py` to `holos.py` (THE ENGINE)
  - Clarifies that this is the algorithm itself, used by ALL layers
  - Not a "game" but the search engine that games use
- **MOVED** `ModeDecision` and `ModeSelector` to `games/seeds.py`
  - These are Layer 1 concerns (tactical mode selection)
- **CREATED** `games/strategy.py` for Layer 2
  - `GoalCondition` (re-exported from holos.py)
  - `StrategyGame`: Meta-game for goal/budget allocation
  - `StrategyState`: Budget allocation across goals
  - `StrategyValue`: Completeness and efficiency metrics
- **Updated** all imports from `holos.core` to `holos.holos`
- **Key Architectural Clarity**:
  - holos.py = THE ENGINE (bidirectional search algorithm)
  - games/chess.py = Layer 0 (CAPABILITIES - what moves exist)
  - games/seeds.py = Layer 1 (TACTICS - which seeds to expand, what mode)
  - games/strategy.py = Layer 2 (STRATEGY - which goals to pursue)
  - All layers use HOLOSSolver, just with different GameInterface implementations

### 2026-01-22 (Update 5) - ARCHITECTURAL REFACTOR
- **MAJOR**: Refactored targeting into proper Layer 0/1/2 separation
  - Layer 0 (chess.py): Material CAPABILITIES (get_material_string, get_parent_materials, enumerate_material_positions)
  - Layer 1/2 (core.py): GoalCondition for targeting STRATEGY
  - Solver: goal parameter for filtered solving
- **Added** `GoalCondition` dataclass to core.py
  - `target_signatures`: Set of valid boundary signatures
  - `early_terminate_misses`: Whether to prune non-goal paths
  - Used by solver.solve(goal=goal)
- **Added** material utilities to chess.py (Layer 0):
  - `get_material_string(state)` → "KQRRvKQR"
  - `get_parent_materials(target)` → list of 8-piece parents
  - `enumerate_material_positions(material, syzygy, count)` → positions
- **Added** `get_signature()` method to GameInterface
  - ChessGame.get_signature() returns material string
  - Enables goal matching in solver
- **Added** `goal_filtered` stat to track filtered boundary states
- **Added** `test_goal_targeting.py` with 4 tests (all passing)
- **DEPRECATED** `chess_targeted.py` - superseded by GoalCondition API
  - Still functional for backward compatibility
  - New code should use GoalCondition pattern

### 2026-01-22 (Update 4)
- **Added** `games/chess_targeted.py` - Targeted material search
  - `TargetedChessGame`: Filters for specific material configurations
  - Early termination on captures to wrong material
  - Auto-generates valid 8-piece source materials
- **Added** `run_targeted_kqrr.py` - In-process batched search
- **Added** `run_targeted_subprocess.py` - Subprocess-isolated search
  - Memory isolation between batches (recommended for large searches)
  - Reproducible from saved seeds
- **Tested** KQRRvKQR targeted search:
  - 3.2M positions solved across 3 batches
  - 483 target material positions found
  - 340 wrong material positions filtered

### 2026-01-22 (Update 3)
- **Fixed** `generate_boundary_seeds()` to respect `min_pieces` setting
  - Was hardcoded to generate 7-piece positions
  - Now uses `self.min_pieces` for configurable boundary
- **Added** `solver.reset()` method for fresh starts
  - Clears all state (frontiers, seen sets, solved, connections, stats)
  - Use when solving independent problems with same solver instance
- **Added** frontier size sampling to prevent memory explosion
  - `_expand_forward()` and `_expand_backward()` now sample if frontier > 500k
  - Graceful degradation instead of hard memory crash
- **Added** `test_integration.py` with 5 comprehensive tests:
  - Boundary seed generation
  - Solver reset
  - Hologram merge with deduplication
  - Connection detection
  - Crystallization

### 2026-01-22 (Update 2)
- Added detailed Crystallization Algorithm documentation (consistent with fractal_holos3.py)
- Added Multi-Round Flow Details section explaining:
  - Solver state persistence (incremental design)
  - Hologram merging with deduplication
  - Memory considerations for backward wave explosion
- Clarified mode selection as Layer 1 dimension (SeedSpec includes mode)
- Documented known placeholder: `feature_success` (declared but unused)
- Verified spine checkpoints: NOT cruft - used for O(1) move lookup

### 2026-01-22 (Initial)
- Created comprehensive LOCUS.md with complete manifest
- All 7 fixes from COMPARISON_WITH_FRACTAL_HOLOS3.md applied
- All equivalence tests passing
- Added `generate_boundary_seeds()` to ChessGame
- Added `apply_move()` to GameInterface and ChessGame
- Full file scan with design decision extraction

---

## Quick Reference

**To add a new game**:
1. Create `games/newgame.py`
2. Implement `GameInterface[YourState, YourValue]`
3. Define boundary conditions and value propagation
4. Use with `HOLOSSolver(your_game)`

**To run tests**:
```bash
cd resonance && python holos/test_equivalence.py   # Consistency tests
cd resonance && python holos/test_integration.py   # Integration tests
```

**To run demo**:
```bash
cd resonance && python holos/demo.py
```

**To debug**:
- Check `solver.stats` for detailed metrics
- Use `hologram.summary()` for overview
- Examine `hologram.spines` for solved paths

---

*This document is the authoritative reference for the HOLOS modular architecture. Update it whenever architectural decisions are made or significant changes occur.*
