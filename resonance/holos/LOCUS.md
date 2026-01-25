# LOCUS.md - HOLOS Architecture Source of Truth

**Last Updated**: 2026-01-24
**Version**: 0.1.4
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
├── __init__.py                     [~75 lines]
│   Purpose: Package exports and documentation
│   Exports: GameInterface, SearchMode, SeedPoint, GoalCondition,
│            LightningProbe, HOLOSSolver, Hologram, SpinePath,
│            SeedFrontierMapping, SessionManager, SessionState,
│            DiskBackedHologram, FullSearchSession, FullSearchState
│   Version: 0.1.1
│
├── holos.py                        [~1400 lines] (renamed from core.py)
│   Purpose: THE ENGINE - Universal HOLOS algorithm (game-agnostic)
│   Classes: GameInterface, SearchMode, SeedPoint, GoalCondition,
│            LightningProbe, HOLOSSolver
│   Key Design: Forward/backward symmetry, bidirectional search,
│               goal-filtered solving, osmosis mode. Used by ALL layers.
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
├── LAYER_ARCHITECTURE.md           [286 lines] (NEW - Update 16)
│   Purpose: Compute vs Storage tradeoffs, compression-aware efficiency
│   Key: Each layer trades compute for storage
├── LOCUS.md                        [THIS FILE]
│
├── seed_meta_game.py               [~400 lines] (NEW - Update 15)
│   Purpose: Meta-seed compression via HOLOS search over seed space
│   Classes: SeedRef, MetaSeed, SeedMetaGame, MetaSeedOptimizer
│   Key: Treats seeds as positions, derivation moves as game moves
│
├── compression.py                  [~500 lines] (NEW - Update 16)
│   Purpose: Compression-aware state representation and efficiency
│   Classes:
│     - IndexedStateEncoder: Encode states as compact integer indices
│     - CompressionAwareSeedValue: Seed value with net_savings metric
│     - StateDimension, DimensionType: Define state space dimensions
│     - ValueBucketer: Bucket continuous values for indexed storage
│     - StateRepresentationComparer: Compare object vs indexed compression
│   Functions: create_seed_encoder(), create_coverage_bucketer()
│   Key Insight: Index-based encoding + gzip beats pickle + gzip
│
├── test_compression.py             [~350 lines] (NEW - Update 16)
│   Purpose: Test compression module
│   Tests: Dimension encoding, encoder roundtrip, compression comparison
│
└── games/                          [GAME IMPLEMENTATIONS - All layers]
    │
    ├── __init__.py                 [~80 lines]
    │   Purpose: Game module exports
    │   Exports: ChessGame, Connect4Game, SudokuGame, SeedGame,
    │            StrategyGame, GoalCondition, ModeDecision, etc.
    │
    ├── chess.py                    [~1000 lines] - LAYER 0
    │   Purpose: Chess endgame implementation (capabilities + targeting)
    │   Classes: Piece, ChessState, ChessValue, ChessFeatures,
    │            SyzygyProbe, ChessGame, TargetedChessGame (deprecated wrapper)
    │   Functions: get_material_string, get_parent_materials,
    │              enumerate_material_positions, create_targeted_solver
    │   Key Design: Provides CAPABILITIES for Layer 1/2 to use
    │   Targeting: Optional target_material param enables material filtering
    │              (consolidated from chess_targeted.py)
    │
    ├── connect4.py                 [~450 lines] - LAYER 0
    │   Purpose: Connect-4 full game implementation
    │   Classes: C4State, C4Value, C4Features, Connect4Game
    │   Key Design: Terminal positions as boundary (no tablebase),
    │               canonical hashing (mirrors fold), center-first ordering
    │   Result: X-Win confirmed (first player wins)
    │
    ├── sudoku.py                   [~550 lines] - LAYER 0 (NEW)
    │   Purpose: Sudoku puzzle solver implementation
    │   Classes: SudokuState, SudokuValue, SudokuFeatures, SudokuGame
    │   Key Design: Solved grids as boundary, MRV heuristic,
    │               single-player (any path to solution counts)
    │   Result: Successfully solves easy/medium/hard puzzles
    │
    ├── seeds.py                    [~720 lines] - LAYER 1 (REFACTORED)
    │   Purpose: Single seed optimization (tactics)
    │   Classes: SeedDirection, TacticalSeed, TacticalValue,
    │            TacticalSeedGame, ModeDecision, ModeSelector
    │   Key Design: SINGLE seed optimization (not multi-seed),
    │               Dual coverage measurement (forward + backward),
    │               Game-agnostic wrapper for any Layer 0 game
    │   Update 16: Compression-aware efficiency (use CompressionAwareSeedValue)
    │   Legacy aliases: SeedSpec, SeedValue, SeedGame (backward compat)
    │
    ├── strategy.py                 [~310 lines] - LAYER 2
    │   Purpose: Goal/budget allocation meta-game (strategy)
    │   Classes: GoalCondition, GoalAllocation, StrategyState,
    │            StrategyValue, StrategyGame
    │   Key Design: Resource allocation across goals, completeness
    │               optimization
    │
    └── chess_targeted.py           [DEPRECATED - DELETE]
        Status: CONSOLIDATED INTO chess.py (Update 13)
        Reason: Functionality now in ChessGame(target_material=...)
        Migration: Import from chess.py instead:
                   from holos.games.chess import TargetedChessGame  # deprecated wrapper
                   from holos.games.chess import ChessGame  # use this instead

holos/
├── __main__.py                     [~10 lines]
│   Purpose: Enable `python -m holos` entry point
│
├── cli.py                          [~200 lines]
│   Purpose: Unified CLI (test/run/status/demo)
│   Commands: test, run, status, demo
│
├── full_search.py                  [~400 lines]
│   Purpose: Production-scale exhaustive search
│   Classes: DiskBackedHologram, FullSearchSession, FullSearchState
│   Key Design: Extends storage.py and session.py for large searches,
│               subprocess isolation, disk-backed storage, resume capability
│
├── test_seed_compression.py        [~300 lines]
│   Purpose: Test seed-based storage compression hypothesis
│   Tests: Seed coverage analysis, connection point seeds,
│          compression ratio calculation
│
├── test_layer1_seeds.py            [~280 lines] (NEW)
│   Purpose: Test Layer 1 tactical seed optimization
│   Tests: Connect4 seed evaluation, Chess seed evaluation,
│          HOLOS solver on TacticalSeedGame, dual coverage metrics
│
└── tests/                          [TEST SUITE DIRECTORY]
    ├── demo.py                     [~260 lines] - Demo script
    ├── run_targeted_kqrr.py        [~200 lines] - Batched targeted search
    ├── run_targeted_subprocess.py  [~180 lines] - Subprocess targeted search
    ├── test_connect4.py            [~350 lines] - Connect-4 tests (all pass)
    ├── test_connections.py         [~150 lines] - Connection detection tests
    ├── test_equivalence.py         [~200 lines] - Equivalence class tests
    ├── test_full_solve.py          [~100 lines] - Full solve tests
    ├── test_goal_targeting.py      [~150 lines] - Goal targeting tests
    ├── test_integration.py         [~200 lines] - Integration tests
    ├── test_osmosis.py             [~280 lines] - Osmosis mode tests
    └── test_sudoku.py              [~300 lines] - Sudoku tests
```

**Total Lines**: ~6,150 (excluding LOCUS.md)

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

### The Four Search Modes

```python
class SearchMode(Enum):
    LIGHTNING = "lightning"  # DFS probe for fast paths (captures only)
    WAVE = "wave"            # BFS for breadth coverage
    CRYSTAL = "crystal"      # Local search around connection points
    OSMOSIS = "osmosis"      # Careful bilateral: best single step from either frontier
```

Mode selection is itself a **meta-decision** that can be optimized (Layer 1/2).

### Osmosis Mode - Maximally Careful Bilateral Exploration

**NEW in Update 11**: Osmosis mode implements the most careful possible approach to bidirectional search.

#### Physical Analogy

Like osmosis in biology - movement driven by concentration (information) gradients:
- **Lightning** = electrical discharge (fast, direct path)
- **Wave** = water waves (uniform BFS expansion)
- **Crystal** = crystallization (grows from nucleation points)
- **Osmosis** = diffusion through membrane (selective, gradient-driven)

#### Algorithm

```python
while not solved:
    1. Score ALL frontier states (forward + backward)
    2. Pick the SINGLE state with highest "certainty pressure"
    3. Expand ONLY that one state
    4. Propagate values
    5. Repeat
```

#### Scoring Function

States are scored based on:
- **Solved neighbors**: Each solved child/parent adds certainty (+10)
- **Boundary proximity**: Near-boundary positions score high (+50)
- **Forced moves**: Only one option = very certain (+100)
- **Connection potential**: About to meet other wave (+100)
- **Equivalence class**: Known outcome for features (+75)
- **Balance factor**: Smaller frontier gets boost to maintain bidirectional progress

#### Why Osmosis Works

1. **Never expands unnecessarily**: Only the "most certain" state advances
2. **Self-balancing**: Smaller frontier naturally gets priority (pressure differential)
3. **Gradient following**: Information flows from known (boundary) to unknown (start)
4. **Forced move exploitation**: Naked singles in Sudoku, forced captures in chess

#### Performance Comparison

On a Sudoku puzzle (73 clues, 8 empty cells):
```
Wave mode:    22.55s, 8,202 expansions
Osmosis mode: 0.72s,  9 steps

30x faster, 900x fewer operations
```

Osmosis excels when:
- There are forced moves (low branching paths)
- Constraint propagation narrows options
- Information gradient is clear (some regions more certain)

Wave excels when:
- No forced moves exist
- Uniform exploration needed
- Coverage matters more than speed

#### Usage

```python
from holos import HOLOSSolver, SeedPoint, SearchMode

solver = HOLOSSolver(game, name="osmosis_test")
forward_seeds = [SeedPoint(puzzle, SearchMode.OSMOSIS)]

# Use solve_osmosis() for osmosis-specific solver
hologram = solver.solve_osmosis(
    forward_seeds,
    backward_seeds=None,
    max_steps=5000,
    verbose=True
)
```

### The Layer Architecture (Refined with Compression-Aware Efficiency)

**Core Insight**: Each layer trades compute for storage. Compression efficiency must be part of the value function at every layer.

See `LAYER_ARCHITECTURE.md` for detailed design documentation.

```
holos.py (THE ENGINE) - Used by ALL layers
    |
    +-> Layer 3: BALANCE (Compute/Storage Policy)
    |       "Given hardware constraints, what's the optimal tradeoff?"
    |       State = ComputeStoragePolicy (memory limit, CPU budget, compression target)
    |       Value = Pareto efficiency (coverage per byte per FLOP)
    |       Output: Layer 2 configuration
    |
    +-> Layer 2: STRATEGY (Multi-Seed Coordination + Compression)
    |       "Which seeds together, with what compression scheme?"
    |       State = SeedSet + CompressionScheme (direct, meta-seeds, delta-encoded)
    |       Value = Combined coverage + compressed_size + reconstruction_cost
    |       Files: games/strategy.py, seed_meta_game.py
    |       KEY: Meta-seeds (roots + moves) compress 6x better than raw seeds
    |
    +-> Layer 1: TACTICS (Single Seed Optimization)
    |       "How to expand this seed efficiently?"
    |       State = TacticalSeed (position_hash, depth, mode, direction)
    |       Value = Coverage + compressed_size (NOT raw storage!)
    |       File: games/seeds.py
    |       KEY: Seeds must compress better than direct storage or they're useless
    |
    +-> Layer 0: EXECUTION (Position Search)
            "What are the game-theoretic values?"
            State = GamePosition (ChessState/C4State/SudokuState)
            Value = Win/Loss/Draw (or problem-specific value)
            Files: games/chess.py, games/connect4.py, games/sudoku.py
            Output: Solved positions -> compressed to seeds
            |
            v queries
        Boundary: Syzygy Tablebases / Terminal States / Solved Grids
```

**Key Insight**: The SAME algorithm (holos.py) searches DIFFERENT state spaces at each layer.
- Layer 0 searches POSITIONS
- Layer 1 searches SINGLE SEED PARAMETERS (compression-aware)
- Layer 2 searches MULTI-SEED CONFIGURATIONS + COMPRESSION SCHEMES
- Layer 3 searches COMPUTE/STORAGE POLICIES

Each layer uses the same bidirectional search, but with different:
- State types
- Value propagation (now includes compression metrics)
- Boundary conditions

### Compression-Aware Efficiency (Key Innovation)

**The Problem**: Old efficiency metric ignored storage overhead:
```python
# WRONG: A seed covering 100 positions but requiring 1000 bytes
# is WORSE than storing the 100 positions at 1 byte each!
efficiency = coverage / cost  # Doesn't account for storage!
```

**The Solution**: Include compression in efficiency calculation:
```python
# Storage if we just stored positions directly
direct_storage = coverage * BYTES_PER_POSITION

# Storage for this seed + reconstruction metadata
seed_storage = seed_size + frontier_overhead + compressed_derivations

# Only count as "efficient" if seed storage < direct storage
net_savings = direct_storage - seed_storage

# Efficiency must account for both compute AND storage
efficiency = net_savings / (compute_cost + reconstruction_cost)
```

**Implication**: Small seeds with negative efficiency should be discarded.
Seeds are only useful when `net_savings > 0`.

### The Compression Hierarchy (Validated)

From experiments on Connect4:
```
Level 1: Game Rules (positions -> seeds)     = 13.4x compression
Level 2: Derivation Structure (meta-seeds)   = 6x additional
Level 3: Standard Compression (gzip)         = 8x additional

Combined potential: 13x * 6x * 8x = 600x+
```

**Key Finding**: Meta-seeds (roots + derivation moves) compress to 89KB
vs full seeds at 532KB = 83% smaller after gzip.

The STRUCTURE helps standard compression algorithms:
- Derivation moves (0-6 in Connect4) are highly repetitive
- gzip finds patterns in the derivation graph
- Don't reinvent compression - create compressible data structures

### The Compute/Storage Spectrum

```
<- More Compute                                    More Storage ->

Single bidirectional     Meta-seeds      Full seeds      All positions
seed from start          (roots+moves)   (every seed)    (no compression)

Compression: infinity    Compression: 6x  Compression: 1x  Compression: 0x
Compute: Maximum         Compute: Medium  Compute: Low     Compute: Zero
```

Layer 3 decides WHERE on this spectrum to operate given hardware constraints.

### Layer 1: Single Seed Tactics (REFINED)

**Critical Insight**: Layer 1 optimizes a SINGLE seed's parameters, not multiple seeds.

```python
@dataclass(frozen=True)
class TacticalSeed:
    position_hash: int      # Which position to seed from
    depth: int              # How many expansion steps (1-6)
    mode: SearchMode        # Lightning, Wave, Crystal, Osmosis
    direction: SeedDirection  # Forward, Backward, Bilateral
    _state: Any = None      # Position state (not hashed)
```

**Why Single Seed?**
- DEPTH is the dominant variable (~10x efficiency per level)
- Single seed optimization is tractable (depth × modes × directions ≈ 72 configs)
- Multi-seed coordination belongs in Layer 2 (combinatorial explosion)

**Dual Coverage Measurement**:

```python
@dataclass(frozen=True)
class TacticalValue:
    forward_coverage: int    # Positions reachable expanding FROM seed
    backward_coverage: int   # Positions that can REACH seed (predecessors)
    overlap_potential: float # Estimated overlap with other seeds
    cost: int               # Computational cost
    efficiency: float       # (forward + backward) / cost
```

Both forward and backward coverage matter for bidirectional effectiveness:
- **Boundary seeds** work best with BACKWARD expansion (from known to unknown)
- **Source seeds** work best with FORWARD expansion (toward boundary)
- **Bilateral** is expensive but maximizes coverage for connection potential

**TacticalSeedGame as GameInterface**:

```python
class TacticalSeedGame(GameInterface[TacticalSeed, TacticalValue]):
    """Game-agnostic Layer 1 that wraps ANY underlying game."""

    def __init__(self, underlying_game: GameInterface, seed_pool, max_depth=6):
        self.underlying_game = underlying_game
        self.seed_pool = seed_pool
        self.max_depth = max_depth
        self.eval_cache = {}  # Cache evaluations

    def get_successors(self, state: TacticalSeed):
        # More expensive configs: increase depth, switch to bilateral, upgrade mode

    def get_predecessors(self, state: TacticalSeed):
        # Cheaper configs: decrease depth, single direction, downgrade mode

    def evaluate(self, state: TacticalSeed) -> TacticalValue:
        # Actually run expansion on underlying game, measure dual coverage
```

### Layer 2: Multi-Seed Strategy (Design)

**Layer 2 coordinates MULTIPLE seeds for few-shot performance.**

```python
@dataclass
class StrategyState:
    selected_seeds: List[TacticalSeed]  # Which seeds to use
    budget: float                        # Total compute budget
    ordering: List[int]                  # Expansion order
    phase_allocation: Dict[str, float]  # Lightning vs Wave vs Crystal time
```

**Key Metrics for Layer 2**:
- **Few-shot performance**: How much coverage with N seeds? (N = 5-20)
- **Time to first solve**: How quickly does ANY connection form?
- **Coverage overlap**: Are seeds redundant or complementary?
- **Budget efficiency**: Coverage per compute unit

**Why Few-Shot Focus?**
- In production, we can't afford infinite seeds
- 87.4% coverage from 300 seeds = good, but can we get 80% from 20?
- Greedy set cover finds minimal seed sets

### Layer 3: Meta-Strategy (Speculation)

**Layer 3 would search the space of STRATEGIES themselves.**

Potential State Space:
- Which Layer 2 strategies work for which game types?
- Can patterns transfer between games (chess → Go → abstract games)?
- What meta-features predict strategy success?

Potential Boundary:
- Known successful strategies on solved games
- Empirical results from Layer 2 experiments

This remains speculative - current focus is Layer 1/2 implementation.

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

### games/seeds.py - Single Seed Tactics (Layer 1)

**Purpose**: Layer 1 implementation - Optimize a SINGLE seed's parameters.

#### Architecture Insight (Updated 2026-01-23)

**BEFORE** (old design):
```
State: SeedConfiguration (SET of seeds) ← Too complex for Layer 1
```

**AFTER** (new design):
```
State: TacticalSeed (SINGLE seed) ← Layer 1 optimizes ONE seed
Multi-seed coordination → Layer 2 responsibility
```

#### The Tactical Game Structure

```
State: TacticalSeed (position_hash, depth, mode, direction)
Moves:
  - Increase depth (more coverage, more cost)
  - Switch to bilateral (both directions)
  - Upgrade mode (lightning → wave → crystal)
Value: TacticalValue (forward_coverage, backward_coverage, efficiency)
Boundary: Evaluated seeds (cached results from Layer 0)
```

#### Key Discovery from Experiments

```
1 seed @ depth 5:  8,275 coverage, 1,655 efficiency
20 seeds @ depth 2: 6,100 coverage, 191 efficiency
```

**DEPTH is the dominant variable** (~10x efficiency gain).

This led to the insight: Layer 1 should optimize SINGLE seed depth/mode/direction.
Multi-seed coordination is a Layer 2 concern.

#### Classes (Updated)

| Class | Lines | Purpose |
|-------|-------|---------|
| `SeedDirection` | 49-53 | Enum: FORWARD, BACKWARD, BILATERAL |
| `TacticalSeed` | 56-96 | Single seed: position_hash, depth, mode, direction |
| `TacticalValue` | 102-145 | Dual coverage: forward, backward, efficiency |
| `TacticalSeedGame` | 151-555 | GameInterface for single seed optimization |
| `ModeDecision` | 647-665 | Tracks mode selection outcomes |
| `ModeSelector` | 667-718 | Learns optimal mode selection |

Legacy aliases for backward compatibility:
- `SeedSpec = TacticalSeed`
- `SeedValue = TacticalValue`
- `SeedGame = TacticalSeedGame`

#### Dual Coverage Measurement

**Key Insight**: Seeds need BOTH forward and backward coverage for bidirectional search.

```python
@dataclass(frozen=True)
class TacticalValue:
    forward_coverage: int    # Positions reachable FROM seed
    backward_coverage: int   # Positions that can REACH seed
    overlap_potential: float # For Layer 2 coordination
    cost: int               # Computational cost
    efficiency: float       # (forward + backward) / cost
```

- **Boundary seeds**: High backward coverage (expand FROM boundary toward unknown)
- **Source seeds**: High forward coverage (expand TOWARD boundary)
- **Bilateral seeds**: Both, but 2x cost

#### TacticalSeedGame Value Propagation

Unlike minimax, the tactical game optimizes **efficiency**:
```python
def propagate_value(self, state, child_values):
    if not child_values:
        return None
    # Best efficiency among children
    return max(child_values, key=lambda v: v.efficiency)
```

#### Game-Agnostic Design

TacticalSeedGame wraps ANY underlying Layer 0 game:

```python
# Works with Chess
chess_game = ChessGame("./syzygy")
tactical = TacticalSeedGame(chess_game, seed_pool=positions)

# Works with Connect4
c4_game = Connect4Game()
tactical = TacticalSeedGame(c4_game, seed_pool=positions)

# Works with Sudoku
sudoku_game = SudokuGame()
tactical = TacticalSeedGame(sudoku_game, seed_pool=positions)
```

#### Convenience Functions

```python
# Optimize a single seed position
best_seed, best_value = optimize_single_seed(
    underlying_game=chess_game,
    seed_state=position,
    max_depth=5,
    verbose=True
)

# Create solver for tactical optimization
solver, game = create_tactical_solver(
    underlying_game=chess_game,
    seed_pool=positions,
    max_depth=6
)
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

## games/connect4.py - Connect-4 Implementation

**Purpose**: Layer 0 implementation for Connect-4 (solved game).

Connect-4 demonstrates HOLOS solving a complete game without external tablebases.
Terminal positions ARE the boundary - we know their values directly.

#### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `C4State` | 25-170 | Board state: 7 columns × 6 rows, turn tracking |
| `C4Value` | 180-190 | Game-theoretic value (+1 X, 0 draw, -1 O) |
| `C4Features` | 200-230 | Equivalence features (counts, threats, heights) |
| `Connect4Game` | 270-450 | GameInterface implementation |

#### Key Design Decisions

1. **Canonical Hashing**: Horizontal mirror positions hash identically
   ```python
   mirror = tuple(reversed(self.cols))
   self._hash = hash((min(self.cols, mirror), self.turn))
   ```

2. **Terminal as Boundary**: Unlike chess (Syzygy), Connect-4 uses terminal positions
   ```python
   def is_boundary(self, state): return state.is_terminal()
   ```

3. **Center-First Move Ordering**: Improves search efficiency
   ```python
   for col in [3, 2, 4, 1, 5, 0, 6]:  # Center columns first
   ```

4. **Threat Counting for Features**: Track 3-in-a-row positions for equivalence

#### Usage

```python
from holos.games.connect4 import Connect4Game, C4State
from holos import HOLOSSolver, SeedPoint, SearchMode

game = Connect4Game()
solver = HOLOSSolver(game, name="connect4")

# Start from empty board
start = C4State()
seeds = [SeedPoint(start, SearchMode.WAVE)]
hologram = solver.solve(seeds, max_iterations=20)

# Result: X-Win (first player wins with perfect play)
print(hologram.query(game.hash_state(start)))  # C4Value(1)
```

#### Test Results

```
Small solve (5 iterations):  8,615 positions in 3.7s
Medium solve (20 iterations): 1,094,767 positions in 255s
Result: X-Win confirmed (first player wins)
```

---

## games/sudoku.py - Sudoku Solver Implementation

**Purpose**: Layer 0 implementation for Sudoku puzzle solving.

Sudoku demonstrates HOLOS solving a constraint satisfaction problem.
Unlike games (minimax), Sudoku is single-player - any path to solution counts.

#### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `SudokuState` | 45-180 | 9x9 grid state, constraint checking |
| `SudokuValue` | 190-200 | Solved/Unsolved value wrapper |
| `SudokuFeatures` | 210-240 | Equivalence features (fill profiles, candidates) |
| `SudokuGame` | 260-450 | GameInterface implementation |

#### Key Design Decisions

1. **Solved Grids as Boundary**: Complete valid grids are boundaries
   ```python
   def is_boundary(self, state): return state.is_solved()
   ```

2. **MRV Heuristic**: Minimum Remaining Values for move ordering
   ```python
   # Pick cell with fewest candidates first
   cells_with_cands.sort()  # Sort by candidate count
   _, r, c, candidates = cells_with_cands[0]
   ```

3. **Single-Player Propagation**: Any solved child means parent is solvable
   ```python
   def propagate_value(self, state, child_values):
       if any(cv.solved for cv in child_values):
           return SudokuValue(solved=True)
   ```

4. **Lightning = Naked Singles**: Forced moves (cells with 1 candidate)
   ```python
   def get_lightning_successors(self, state):
       # Only return cells with exactly 1 candidate
       for r, c in state.empty_cells():
           cands = state.get_candidates(r, c)
           if len(cands) == 1:
               ...
   ```

#### Usage

```python
from holos.games.sudoku import SudokuGame, SudokuState, get_sample_puzzles
from holos import HOLOSSolver, SeedPoint, SearchMode

game = SudokuGame()
puzzles = get_sample_puzzles()
puzzle = puzzles['easy']  # 30 clues

solver = HOLOSSolver(game, name="sudoku")
seeds = [SeedPoint(puzzle, SearchMode.WAVE)]
hologram = solver.solve(seeds, max_iterations=10)

# Check if solved
print(hologram.query(game.hash_state(puzzle)))  # SudokuValue(solved=True)
```

#### Sample Puzzles

```python
puzzles = get_sample_puzzles()
# Returns dict with 'easy' (30 clues), 'medium' (28 clues), 'hard' (24 clues)
```

#### Test Results

```
Near-complete (78 clues): Solved via lightning spine in 1 iteration
Easy puzzle (30 clues): Solved via forward expansion + lightning
Backward wave: 81 predecessors per solved grid (high branching)
```

#### Notes on Sudoku + HOLOS

Sudoku has interesting properties for HOLOS:

1. **Asymmetric branching**: Forward ~1-9 successors (constrained),
   backward has 81 predecessors (any cell can be emptied)

2. **Lightning is powerful**: Naked singles chain often solves directly

3. **Backward wave explodes**: Each solved grid has 81 predecessors,
   so backward frontier grows ~81x per iteration

4. **Equivalence limited**: Sudoku positions are highly specific,
   equivalence classes don't compress as well as games

---

## Insights from c4_crystal.py

The older `resonance/c4_crystal.py` contained experimental algorithms.
Key insights integrated into HOLOS:

### 1. Crystallization Phase Separation

**Insight**: The first solution becomes a NEW BOUNDARY CONDITION.

```
Phase 1: LIGHTNING - Find first solution path (spine)
Phase 2: CRYSTALLIZATION - Grow solved region FROM the spine
```

This is modeled on natural phenomena:
- River deltas (main channel → branches)
- Lichtenberg figures (trunk → fractal branches)
- Crystal growth (seed → propagating front)

**Integration**: HOLOS `_crystallize()` expands around connection points.
Enhancement: Spines could act as additional backward wave seeds.

### 2. Phase Timing Metrics

**Insight**: Track time spent in each algorithm phase separately.

```python
self.phase_timing = {
    'lightning_time': 0.0,
    'wave_time': 0.0,
    'crystal_time': 0.0,
    'propagation_time': 0.0,
}
```

**Integration**: Added to `HOLOSSolver.stats` for analysis.

### 3. Crystal Front Tracking

**Insight**: Track which solved positions are at the "growth front".

```python
self.crystal_front: Set[int] = set()  # Expanding boundary of solved region
```

Positions on the front have unsolved neighbors - prioritize expanding there.

**Potential Enhancement**: Solution density heuristics for seed prioritization.

### 4. Spine-as-Boundary Mode

**Insight**: Once a spine is found, treat it as an additional boundary.

The spine connects start → terminal. Any position touching the spine
can propagate values bidirectionally.

**Implementation**: `enable_spine_seeding` parameter (optional).

---

## Changelog

Full changelog has been extracted to **[CHANGELOG.md](CHANGELOG.md)** to keep this document focused on architecture.

### Recent Updates (Latest First)

- **Update 16** (2026-01-24): Compression-aware layer architecture, index-based encoding
- **Update 15** (2026-01-24): Full Connect4 solver, meta-seed compression
- **Update 14** (2026-01-23): Layer 1 refined architecture (single-seed focus)
- **Update 13** (2026-01-23): Chess consolidation, test directory reorganization
- **Update 12** (2026-01-23): Full search infrastructure, seed compression hypothesis
- **Update 11** (2026-01-23): Osmosis mode (30x faster on constrained problems)
- **Update 10** (2026-01-23): Sudoku solver implementation
- **Update 9** (2026-01-23): Forward expansion fix
- **Update 8** (2026-01-22): Connect-4 implementation, crystal insights
- **Update 7** (2026-01-22): Memory management fixes
- **Update 6** (2026-01-22): Clean layer separation (core.py → holos.py)

See [CHANGELOG.md](CHANGELOG.md) for full details on each update.

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
