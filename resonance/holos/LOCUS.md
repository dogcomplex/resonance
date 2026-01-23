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
├── __init__.py                     [61 lines]
│   Purpose: Package exports and documentation
│   Exports: GameInterface, SearchMode, SeedPoint, LightningProbe,
│            HOLOSSolver, Hologram, SpinePath, SeedFrontierMapping,
│            SessionManager, SessionState
│   Version: 0.1.0
│
├── core.py                         [841 lines]
│   Purpose: Universal HOLOS algorithm (game-agnostic)
│   Classes: GameInterface, SearchMode, SeedPoint, LightningProbe,
│            ModeDecision, ModeSelector, HOLOSSolver
│   Key Design: Forward/backward symmetry, bidirectional search,
│               mode selection as meta-decision
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
│   Functions: demo_chess_solving, demo_bidirectional_lightning,
│              demo_session_management, demo_seed_game, demo_storage
│
├── test_equivalence.py             [332 lines]
│   Purpose: Verify functional equivalence with fractal_holos3.py
│   Tests: ChessGame interface, boundary seed generation, solver stats,
│          memory tracking, spine structure, lightning probe, apply_move
│
├── README.md                       [249 lines]
│   Purpose: User-facing documentation
│
├── COMPARISON_WITH_FRACTAL_HOLOS3.md [259 lines]
│   Purpose: Detailed comparison with original implementation
│
├── LOCUS.md                        [THIS FILE]
│   Purpose: Source of truth for AI context
│
└── games/                          [GAME IMPLEMENTATIONS]
    │
    ├── __init__.py                 [42 lines]
    │   Purpose: Game module exports
    │   Exports: ChessGame, ChessValue, ChessState,
    │            SeedGame, SeedConfiguration, SeedValue, SeedSpec
    │
    ├── chess.py                    [686 lines]
    │   Purpose: Layer 0 - Chess endgame implementation
    │   Classes: Piece, ChessState, ChessValue, ChessFeatures,
    │            SyzygyProbe, ChessGame
    │   Key Design: Syzygy boundary, minimax propagation,
    │               canonical hashing, predecessor generation
    │
    └── seeds.py                    [383 lines]
        Purpose: Layer 1 - Seed selection meta-game
        Classes: SeedSpec, SeedConfiguration, SeedValue, SeedGame
        Key Design: Lattice search, efficiency optimization,
                    configuration space exploration
```

**Total Lines**: ~3,472 (excluding LOCUS.md)

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
Layer 2+: Meta-strategy (budget allocation, material priority)
    ↓ controls
Layer 1: Seed Selection (which seeds, what depth, which mode)
    ↓ controls
Layer 0: Base Game (chess positions, minimax values)
    ↓ queries
Boundary: Syzygy Tablebases (7-piece endgames)
```

Each layer uses HOLOS principles but may use different value propagation.

---

## Module Deep Dive

### core.py - The Universal Solver

**Purpose**: Game-agnostic HOLOS algorithm implementation.

#### Classes

| Class | Lines | Responsibility |
|-------|-------|----------------|
| `GameInterface[S, V]` | 52-140 | Abstract interface any game must implement |
| `SearchMode` | 33-38 | Enum: LIGHTNING, WAVE, CRYSTAL |
| `SeedPoint[S]` | 40-50 | Seed with state, mode, priority, depth |
| `LightningProbe[S, V]` | 146-239 | Bidirectional DFS to find paths |
| `ModeDecision` | 247-261 | Tracks single mode decision for learning |
| `ModeSelector` | 264-316 | Tracks mode success for meta-learning |
| `HOLOSSolver[S, V]` | 322-842 | Main solver with bidirectional search |

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
    def get_lightning_successors(state: S)    # Captures only (default: get_successors)
    def get_lightning_predecessors(state: S)  # Uncaptures only (default: get_predecessors)
    def score_for_lightning(state: S, move)   # MVV-LVA scoring (default: 0.0)
    def generate_boundary_seeds(template, count)  # Auto-generate seeds (default: [])
    def apply_move(state: S, move: Any) -> S  # For spine reconstruction
```

**Value propagation is GAME-SPECIFIC**:
- Chess: minimax (White maximizes, Black minimizes)
- Optimization: max efficiency
- Path finding: min cost

#### HOLOSSolver Algorithm

```python
def solve(forward_seeds, backward_seeds=None, max_iterations=100, lightning_interval=5):
    # 1. Initialize frontiers from seeds
    # 2. Auto-generate backward seeds if not provided (via game.generate_boundary_seeds)
    # 3. For each iteration:
    #    a. Check memory limits
    #    b. Run lightning probes every N iterations (forward + backward)
    #    c. Expand forward frontier (BFS layer)
    #    d. Expand backward frontier (BFS using predecessors)
    #    e. Find connections (where waves overlap)
    #    f. Crystallize around connections
    #    g. Propagate values through parent links and equivalence
    # 4. Build and return Hologram
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
cd resonance && python holos/test_equivalence.py
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
