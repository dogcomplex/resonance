# HOLOS - Hierarchical Omniscient Learning and Optimization System

A universal bidirectional search framework for game-theoretic and optimization problems.

## Core Insight

**HOLOS maps ALL paths in a game space.** Winning/losing are just properties we care about - the algorithm finds ALL reachable states and their relationships, then filters for whatever objective matters.

## Architecture Overview

```
holos/
├── __init__.py      # Package exports
├── core.py          # Universal HOLOS algorithm
├── storage.py       # Holographic storage (spines, seeds)
├── session.py       # Multi-round management
├── games/
│   ├── __init__.py
│   ├── chess.py     # Layer 0: Chess endgames
│   └── seeds.py     # Layer 1: Seed selection meta-game
└── README.md        # This file
```

## Key Concepts

### 1. GameInterface (core.py)

Abstract interface for any game/lattice. A game is defined by:

```python
class GameInterface(ABC, Generic[S, V]):
    def hash_state(state) -> int           # Deduplication
    def get_successors(state) -> List      # Forward moves
    def get_predecessors(state) -> List    # Backward moves
    def is_boundary(state) -> bool         # Known value?
    def get_boundary_value(state) -> V     # Get known value
    def propagate_value(state, children) -> V  # Combine child values
```

**Value propagation is GAME-SPECIFIC:**
- Chess: minimax (White max, Black min)
- Optimization: max efficiency
- Path finding: min cost

### 2. Bidirectional Search

The core algorithm uses two wavefronts:

```
Forward Wave (Deduction)      Backward Wave (Abduction)
     ↓                              ↑
[Start] ───────→ ←─────── [Boundary]

         Meeting = Induction
```

**Forward and Backward are MIRRORS.** Backward is just forward on the reversed game.

### 3. Search Modes

```python
class SearchMode(Enum):
    LIGHTNING = "lightning"  # DFS probe for fast paths
    WAVE = "wave"            # BFS for breadth coverage
    CRYSTAL = "crystal"      # Local search around connections
```

**Mode selection is a meta-decision** that can itself be optimized (Layer 1/2).

### 4. Lightning Probes (Bidirectional)

```python
class LightningProbe:
    def probe(state, direction="forward"):
        # DFS from state toward boundary (forward)
        # OR from boundary toward state (backward)
```

**Backward lightning** is DFS from boundary toward start using `get_predecessors()`.

### 5. Spine Paths (storage.py)

Compressed principal variations connecting start to boundary:

```python
@dataclass
class SpinePath:
    start_hash: int      # Starting position
    moves: List[Any]     # Just the moves, not states
    end_hash: int        # Terminal position
    end_value: Any       # Value at end
    checkpoints: List    # (hash, features) at key points
```

**Spines serve two purposes:**
1. **Analysis**: Show how a position is solved (PV line)
2. **Decision Making**: Quick lookup of best move from any position on spine

### 6. Seed→Frontier Mapping (storage.py)

**THE KEY COMPRESSION INSIGHT:**

Instead of storing frontier positions, store:
- 1 seed position
- depth parameter
- expansion algorithm = deterministic

```
Depth 1: ~25 positions    (25x compression)
Depth 2: ~275 positions   (275x compression)
Depth 3: ~2775 positions  (2775x compression)
Depth 5: ~8000 positions  (8000x compression)
```

The frontier is a DERIVED quantity, not stored.

### 7. SessionManager (session.py)

Handles multi-round solving:

```python
class SessionManager:
    def run_round(solver, seeds, budget)     # Single round
    def select_next_seeds(game, num)         # Layer 1 decision
    def advance_phase()                      # Phase transitions
    def run_session(solver, seeds, budget)   # Full session
```

**Is SessionManager Layer 1 or Layer 2?**

Both:
- Layer 1 (Seed Selection): Chooses seeds for THIS round
- Layer 2 (Meta-Strategy): Allocates compute across MULTIPLE rounds

## Layer Architecture

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

## Key Discovery: Depth > Seed Selection

Experiments revealed:

```
1 seed @ depth 5:  8,275 coverage, 1,655 efficiency
20 seeds @ depth 2: 6,100 coverage, 191 efficiency
```

**Depth is the dominant variable** (~10x efficiency gain).

The meta-game discovered that we were optimizing the WRONG dimension.

## Usage Example

```python
from holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games import ChessGame, random_position

# Create chess game interface
game = ChessGame(syzygy_path="./syzygy", min_pieces=7, max_pieces=8)
solver = HOLOSSolver(game, name="chess_8piece")

# Generate starting positions
positions = [random_position("KQRRvKQRR") for _ in range(100)]
seeds = [SeedPoint(p, SearchMode.WAVE, depth=2) for p in positions if p]

# Solve
hologram = solver.solve(seeds, max_iterations=50)

# Query results
for seed in seeds[:5]:
    h = game.hash_state(seed.state)
    value = hologram.query(h)
    print(f"Position {h}: {value}")
```

## Design Decisions

### Equivalence Classes

**Both rule-based and discovered:**
- Rule-based: Symmetry (horizontal flip) built into hash
- Discovered: Feature clustering finds positions with same outcome

```python
@dataclass(frozen=True)
class ChessFeatures:
    material_white: Tuple[int, ...]
    material_black: Tuple[int, ...]
    material_balance: int
    piece_count: int
    king_distance: int
    turn: str
```

### Forward/Backward Symmetry

The insight: `backward = forward(reversed_game)`

For chess:
- Forward successor: Apply legal move
- Backward predecessor: Undo move (potentially uncapturing)

Both use the same algorithm structure.

### Multi-Round Handling

The session tracks:
- Phase (LIGHTNING → WAVE → CRYSTAL → COMPLETE)
- Budget used/remaining
- Seeds explored/pending
- Feature success rates (for meta-learning)

Phase transitions happen automatically based on progress metrics.

## Files Reference

| File | Purpose |
|------|---------|
| `core.py` | GameInterface, HOLOSSolver, LightningProbe, ModeSelector |
| `storage.py` | Hologram, SpinePath, SeedFrontierMapping |
| `session.py` | SessionManager, SessionState, RoundStats |
| `games/chess.py` | ChessGame, ChessState, ChessValue, SyzygyProbe |
| `games/seeds.py` | SeedGame, SeedConfiguration, SeedValue |

## Checklist Items Addressed

- [x] **Spine paths**: SpinePath class with checkpoints for analysis AND decision making
- [x] **Backward lightning**: LightningProbe with direction="backward" using get_lightning_predecessors
- [x] **SessionManager**: Full multi-round handling with phase management
- [x] **Seed→Frontier storage**: SeedFrontierMapping with lazy reconstruction
- [x] **Mode selection as meta-decision**: ModeSelector class tracking outcomes by features

## Future Extensions

1. **Layer 2+ games**: Budget allocation, material priority optimization
2. **More game interfaces**: Go endgames, combinatorial optimization
3. **Distributed solving**: Multiple workers with hologram merging
4. **Neural guidance**: Learn move ordering from spine paths
