# HOLOS Changelog

This changelog documents significant updates to the HOLOS architecture and implementation.

See [LOCUS.md](LOCUS.md) for the full architecture documentation.

---

## 2026-01-24 (Update 16) - COMPRESSION-AWARE LAYER ARCHITECTURE & INDEX ENCODING

**Major architectural refinement: compression is part of efficiency at every layer.**
**Implementation: Index-based state encoding for compact storage.**

### New Files
- **`LAYER_ARCHITECTURE.md`** (~286 lines): Comprehensive design document
  - Describes all four layers with compression-aware value functions
  - `CompressionAwareSeedValue`: Includes `net_savings`, `is_worth_storing`
  - `StrategyWithCompression`: Multi-seed + compression scheme selection
  - `ComputeStoragePolicy`: Layer 3 compute/storage balance decisions
  - File consolidation plan for future reorganization

- **`compression.py`** (~500 lines): Index-based state encoding
  - `IndexedStateEncoder`: Encode states as compact integer indices
    - States defined by dimensions (DISCRETE, BOUNDED_INT, BUCKETED)
    - Encode/decode roundtrip preserves state
    - Batch encoding to bytes for compression
  - `CompressionAwareSeedValue`: Seed value with storage metrics
    - `net_savings = direct_storage - seed_storage`
    - `is_worth_storing`: True only if net_savings > 0
    - `efficiency`: bytes saved per compute cost
  - `ValueBucketer`: Bucket continuous values (coverage, etc.)
    - If coarse seed is nearly as effective, use it (compresses better)
  - `StateRepresentationComparer`: Compare indexed vs object storage
  - Factory functions: `create_seed_encoder()`, `create_coverage_bucketer()`

- **`test_compression.py`** (~350 lines): Validation tests

### Key Insights Documented

1. **Compression is part of efficiency**: Seeds that don't compress better than
   direct storage are useless (negative efficiency)

2. **Index-based encoding beats object storage**:
   - 1000 seeds: indexed=2,032 bytes vs pickle=3,834 bytes (47% smaller)
   - After gzip: indexed still wins
   - Reason: Structured integers compress better than serialized objects

3. **The hierarchy is multiplicative**:
   - Game rules: 13x compression (positions to seeds)
   - Derivation structure: 6x additional (meta-seeds)
   - Standard compression: 8x additional (gzip)
   - Combined: 600x+ potential

4. **Don't reinvent compression**: Use game structure to create data that
   standard algorithms (gzip) can compress effectively

5. **Bucketing for continuous values**: If a cruder seed (bucketed coverage)
   is nearly as effective as precise one, use it - compresses better

6. **Layer 3 decides the tradeoff**: Given hardware constraints, choose where
   on the compute/storage spectrum to operate

### Layer Architecture Refinement

- **Layer 0**: Unchanged - game positions and values
- **Layer 1**: Now includes `compressed_size` in value calculation
- **Layer 2**: Now includes compression scheme selection (direct, meta-seeds, delta)
- **Layer 3**: New - decides compute/storage balance given constraints

### Experimental Results

**Index Encoding Test (1000 seeds)**:
```
Representation comparison:
  indexed:           2,032 bytes (2.0x compression)
  full_object_pickle: 3,834 bytes (50.2x compression)
  indexed_raw:        4,000 bytes (1.0x)

Winner: indexed (47% smaller than pickle after gzip)
```

**Connect4 Seed Comparison (90 seeds)**:
```
  indexed:    168 bytes = 1.9 bytes/seed
  pickle:   1,220 bytes = 13.6 bytes/seed

Winner: indexed (7x smaller)
```

**Meta-seeds (from Update 15)**:
```
Meta-seeds vs Full Seeds (after gzip):
  Full seeds:     532 KB
  Meta-seeds:     89 KB  (83% smaller!)

The STRUCTURE helps gzip:
  - 8,599 root seeds stored directly
  - 154,967 derivation moves (0-6, highly repetitive)
  - gzip finds patterns in move sequences
```

### Usage Example

```python
from holos.compression import (
    create_seed_encoder,
    CompressionAwareSeedValue,
    StateRepresentationComparer
)

# Create encoder for seed space
encoder = create_seed_encoder(max_positions=10000, max_depth=6)

# Encode seeds to compact indices
indices = encoder.encode_batch(seeds)  # -> bytes

# Evaluate with compression awareness
value = CompressionAwareSeedValue(
    forward_coverage=10000,
    backward_coverage=5000,
    compute_cost=100,
    seed_storage_bytes=11,
    frontier_storage_bytes=0,
    derivation_bytes=1,
)
if value.is_worth_storing:
    print(f"Net savings: {value.net_savings} bytes")
```

---

## 2026-01-24 (Update 15) - FULL CONNECT4 SOLVER & META-SEED COMPRESSION

**Complete Connect4 solver with layer-by-layer seed extraction:**

### New Files
- **`connect4_full_solve.py`** (~800 lines): Full Connect4 solver
  - `C4Seed`: Seed dataclass with position, value, reason
  - `SeedLayer`: Layer management with compression stats
  - `SeedDatabase`: SQLite persistence with in-memory caching
  - `LayerSolver`: Layer-by-layer minimax solving
  - `StrategyOptimizer`: Layer 2 strategy comparison
  - `ValueReconstructor`: On-demand value reconstruction from seeds

- **`seed_meta_game.py`** (~400 lines): Meta-seed compression via HOLOS
  - `SeedRef`: Reference to seed (direct or derived)
  - `MetaSeed`: Meta-seed that generates multiple seeds
  - `SeedMetaGame(GameInterface)`: HOLOS search over seed space
  - `MetaSeedOptimizer`: Greedy set cover + HOLOS search

### Key Results

**Connect4 Full Solver (10 pieces)**:
```
Positions: 1,240,914
Seeds: 92,464 (7.5%)
Compression: 13.4x average

Per-layer:
  Layer 10: 37.6x (all terminal wins)
  Layer 9:  7.1x (value transitions)
  Layer 8:  97x (few terminals)
```

**Meta-Seed Compression Analysis**:
```
Seeds derivable from others: 33.2%
Root seeds (must store): 66.8%

Storage Format Impact:
  45 bytes/seed: 0.30x (WORSE than raw)
  9 bytes/seed:  1.49x compression
  Meta-seeds:    1.83x compression

Key Insight: Storage format > meta-seeds!
```

**Full Game Projection (4.5T positions)**:
```
Full positions:    4.5 TB
Seeds (9 bytes):   3.0 TB (1.5x)
Meta-seeds (best): 2.5 TB (1.8x)
```

### Architectural Insights

1. **Seed compression works** but storage format dominates
2. **Layer structure exploitable**: 33% of seeds derivable from layer above
3. **HOLOS on seeds**: SeedMetaGame enables searching seed space itself
4. **Recursive compression limited**: Most gains from format optimization

---

## 2026-01-23 (Update 14) - LAYER 1 REFINED ARCHITECTURE

**Major refactoring of Layer 1 to single-seed optimization:**

### Architecture Changes
- **REFACTORED** seeds.py from multi-seed to single-seed focus
  - Old: `SeedConfiguration` = set of seeds (too complex for Layer 1)
  - New: `TacticalSeed` = single seed with (position, depth, mode, direction)
  - Multi-seed coordination moves to Layer 2

### New Classes
- `SeedDirection` enum: FORWARD, BACKWARD, BILATERAL
- `TacticalSeed`: Single seed state (position_hash, depth, mode, direction)
- `TacticalValue`: Dual coverage metrics (forward_coverage, backward_coverage, efficiency)
- `TacticalSeedGame(GameInterface)`: Game-agnostic Layer 1 implementation

### Key Insights
- **Dual coverage measurement**: Both forward AND backward expansion matter
  - Boundary seeds → prefer BACKWARD expansion
  - Source seeds → prefer FORWARD expansion
  - Bilateral → expensive but maximizes connection potential
- **DEPTH is dominant variable**: ~10x efficiency per depth level
- **Game-agnostic design**: TacticalSeedGame wraps ANY underlying GameInterface

### New Files
- `test_layer1_seeds.py`: Test suite for Layer 1 tactical optimization
  - Connect4 seed evaluation tests
  - Chess seed evaluation tests (if syzygy available)
  - HOLOS solver on TacticalSeedGame demonstration

### Updated Documentation
- LOCUS.md Layer Architecture section refined
- Layer 1, 2, 3 separation clarified
- seeds.py documentation updated with new classes
- Legacy compatibility aliases documented

### Seed Compression Results (Validated Across Multiple Games)

**Chess (test_seed_compression.py)**:
```
Coverage achieved: 87.4%
Compression ratio: 7,453x
Missing positions: 282,333 (interaction positions)
```

**Chess Layer 1 (test_layer1_seeds.py)**:
```
Boundary seeds (backward): 4M positions per seed, efficiency 2M
Source seeds (bilateral): 2.9M positions per seed, efficiency 725K
Total expanded: 47 million positions in 1586s
```

**Sudoku (test_sudoku_seeds.py)**:
```
Compression ratio: 2.4x (path storage vs full positions)
Easy puzzle: 51 moves = 349 bytes, solved in 1.8s via HOLOS osmosis
Database: 20 puzzles = 6.6 KB (reconstruction < 0.001ms per puzzle)
Layer 1: Backward from solution = 3,322 positions at depth 2
```

**Connect4 (test_connect4_seeds.py)**:
```
Solved: 9.2 million positions in 238s (12-depth minimax)
Full storage: 80,965 KB
Seed storage (10%): 8,097 KB
Compression ratio: 10x
Layer 1: Depth 4 wave = efficiency 353
```

**Key Insight**: Compression ratio varies by game structure:
- Chess: 7,453x (many symmetries, tablebases provide dense boundary)
- Connect4: 10x (minimax requires storing more decision points)
- Sudoku: 2.4x (single-path solutions, minimal branching)

This validates the seed-based storage hypothesis - positions
can be reconstructed from seeds + paths, enabling massive compression.

---

## 2026-01-23 (Update 13) - CHESS CONSOLIDATION & TEST DIRECTORY

- **CONSOLIDATED** `chess_targeted.py` into `chess.py`
  - `ChessGame` now accepts optional `target_material` parameter
  - When targeting enabled: `is_boundary()`, `get_successors()`, `get_predecessors()` filter by material
  - Added `generate_target_boundary_seeds()` and `generate_source_positions()` methods
  - Added `is_target_material()` and `is_source_material()` helpers
  - Filter statistics tracking: `wrong_material_filtered`, `target_material_found`, etc.
- **DEPRECATED** `chess_targeted.py` - marked for deletion
  - `TargetedChessGame` now a deprecated wrapper class in chess.py
  - Shows deprecation warning when instantiated
  - All imports should migrate to `from holos.games.chess import ChessGame`
- **UPDATED** dependent files:
  - `full_search.py` - imports from chess.py, uses `ChessGame(target_material=...)`
  - `test_seed_compression.py` - imports from chess.py
  - `games/__init__.py` - exports consolidated from chess.py
- **MOVED** test files to `tests/` directory:
  - `demo.py`, `run_targeted_kqrr.py`, `run_targeted_subprocess.py`
  - `test_connect4.py`, `test_connections.py`, `test_equivalence.py`
  - `test_full_solve.py`, `test_goal_targeting.py`, `test_integration.py`
  - `test_osmosis.py`, `test_sudoku.py`
- **FILES FOR DELETION**:
  - `games/chess_targeted.py` - consolidated into chess.py

---

## 2026-01-23 (Update 12) - Full Search Infrastructure & Seed Compression

**Major additions for production-scale exhaustive searches:**

### New Files

- **`full_search.py`** (~400 lines): Extends session.py and storage.py for large-scale searches
  - `DiskBackedHologram(Hologram)`: Writes to disk in chunks, prevents RAM exhaustion
  - `FullSearchSession(SessionManager)`: Subprocess isolation, resume capability
  - `FullSearchState(SessionState)`: Extended state for batch tracking

- **`cli.py`** (~200 lines): Unified command-line interface
  - `python -m holos.cli test` - Quick verification run
  - `python -m holos.cli run` - Full production run
  - `python -m holos.cli status` - Check progress

- **`__main__.py`**: Enables `python -m holos`

- **`test_seed_compression.py`** (~300 lines): Tests seed-based storage compression

### Architecture Integration

```
full_search.py EXTENDS (not replaces):
├── DiskBackedHologram extends Hologram (storage.py)
│   - Inherits: query(), add_spine(), merge()
│   - Adds: flush_to_disk(), total_count(), chunk management
│
├── FullSearchSession extends SessionManager (session.py)
│   - Inherits: save(), run_round(), phase management
│   - Adds: subprocess isolation, disk monitoring, batch tracking
│
└── Uses HOLOSSolver directly (holos.py)
    - No parallel implementation
    - Standard SeedPoint, SearchMode interface
```

### Seed Compression Hypothesis

**The Core Insight** (from SeedFrontierMapping):
```
Instead of storing N solved positions, store K seeds that regenerate them.
Compression: N/K (potentially 1000x-8000x)
```

**Current Status**:
- `SeedFrontierMapping` defined in storage.py but NOT actively used
- `full_search.py` stores all positions (DiskBackedHologram)
- `test_seed_compression.py` created to validate the hypothesis

**Key Questions to Test**:
1. What % of solved positions are reconstructable from seeds alone?
2. Are connection points (where waves meet) optimal seeds?
3. What's the actual compression ratio for KQRRvKQR?

**Test Command**:
```bash
python holos/test_seed_compression.py
python holos/test_seed_compression.py --connections
```

### CLI Commands

**Test Run** (~5 minutes):
```bash
python -m holos.cli test --target KQRRvKQR --batches 5 --memory 2500
```

**Full Run** (hours/days):
```bash
python -m holos.cli run --target KQRRvKQR --backward-seeds 5000 --forward-seeds 200 --memory 3500
```

**Status Check**:
```bash
python -m holos.cli status --target KQRRvKQR
```

### Documentation Corrections

- **chess_targeted.py is NOT deprecated** - actively used by full_search.py, run_targeted_*.py
- Updated __init__.py to export: DiskBackedHologram, FullSearchSession, FullSearchState
- Version bumped to 0.1.1

---

## 2026-01-23 (Update 11) - OSMOSIS MODE

- **Added** Osmosis mode - maximally careful bilateral exploration
  - New `SearchMode.OSMOSIS` enum value
  - New `solve_osmosis()` method in HOLOSSolver
  - `_score_state_for_osmosis()` for intelligent state selection
  - `_osmosis_expand_forward_single()` and `_osmosis_expand_backward_single()`
- **Key Innovation**: Expands ONE state at a time from either frontier
  - Scores based on: solved neighbors, forced moves, connection potential
  - Balance factor maintains bidirectional progress (like pressure differential)
  - Naturally follows "information gradient" from known to unknown
- **Performance**: 30x faster than wave mode on constrained problems
  - Sudoku (8 empty cells): 0.72s vs 22.55s (wave)
  - Osmosis: 9 steps vs 8,202 expansions
- **BUGFIX**: Added reverse forward propagation for single-player games
  - If child is solved, parent should also be marked solved
  - Required for Sudoku where any path to solution counts
- **Added** `test_osmosis.py` - Comprehensive osmosis test suite
  - Sudoku near-complete and easy tests
  - Connect4 position test
  - Wave vs Osmosis comparison
- **Physical Analogy**: Like osmosis/diffusion - flow driven by concentration gradient
  - Lightning = electrical discharge (fast, direct)
  - Wave = water waves (uniform expansion)
  - Crystal = crystallization (nucleation growth)
  - Osmosis = diffusion (selective, gradient-driven)

---

## 2026-01-23 (Update 10) - SUDOKU SOLVER

- **Added** `games/sudoku.py` - Full Sudoku puzzle solver
  - `SudokuState`: 9x9 grid with constraint checking
  - `SudokuValue`: Solved/Unsolved value wrapper
  - `SudokuFeatures`: Fill profiles, candidate analysis
  - `SudokuGame`: Full GameInterface implementation
  - Solved grids as boundary (no external tablebase)
  - MRV heuristic for move ordering
  - Lightning = naked singles (forced moves)
- **Added** `test_sudoku.py` - Comprehensive test suite
  - Basic state operations (grid, candidates, conflicts)
  - Game interface tests (successors, predecessors, boundary)
  - Feature extraction tests
  - Lightning probe tests
  - Bidirectional solve tests
- **Updated** `games/__init__.py` with Sudoku exports
- **Key insight**: Sudoku has asymmetric branching
  - Forward: ~1-9 constrained successors
  - Backward: 81 predecessors (remove any digit)
  - Lightning (naked singles) often solves directly

---

## 2026-01-23 (Update 9) - FORWARD EXPANSION FIX

- **BUGFIX**: Forward frontier was collapsing to 0 after lightning solved start
  - Problem: `_expand_forward()` was skipping children of solved positions
  - Solution: Always expand children, even for already-solved positions
  - Now forward frontier grows properly: 1 → 4 → 25 → 121 → 568
- **Result**: Bidirectional search now creates connections (49 in test)
  - Previously: 0 connections (forward wave died immediately)
  - Now: Proper bidirectional meeting of waves

---

## 2026-01-22 (Update 8) - CONNECT-4 & CRYSTAL INSIGHTS

- **Added** `games/connect4.py` - Full Connect-4 implementation
  - `C4State`: Compact state with canonical (mirror-folded) hashing
  - `C4Value`: Game-theoretic value wrapper
  - `C4Features`: Equivalence features (counts, threats, heights)
  - `Connect4Game`: Full GameInterface implementation
  - Terminal positions as boundary (no external tablebase needed)
  - Center-first move ordering for efficiency
- **Added** `test_connect4.py` - Comprehensive test suite
  - Basic interface tests (moves, hashing, mirrors)
  - Successor/predecessor generation tests
  - Features and equivalence tests
  - Lightning probe tests
  - Small solve: 8,615 positions in 3.7s
  - Medium solve: 1,094,767 positions in 255s
  - **Confirms**: X-Win (first player wins with perfect play)
- **Updated** `games/__init__.py` with Connect4 exports
- **Added** Phase timing metrics to HOLOSSolver
  - `lightning_time`, `wave_time`, `crystal_time`, `propagation_time`
- **Added** `spine_as_boundary` option to solver
  - When enabled, solved spines seed the backward wave
- **Documented** insights from `c4_crystal.py`:
  - Crystallization phase separation
  - Crystal front tracking concept
  - Natural metaphors (river deltas, Lichtenberg figures)
  - Spine-as-boundary enhancement

---

## 2026-01-22 (Update 7) - MEMORY MANAGEMENT & CACHING ANALYSIS

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

---

## 2026-01-22 (Update 6) - CLEAN LAYER SEPARATION

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

---

## 2026-01-22 (Update 5) - ARCHITECTURAL REFACTOR

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

---

## 2026-01-22 (Update 4)

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

---

## 2026-01-22 (Update 3)

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

---

## 2026-01-22 (Update 2)

- Added detailed Crystallization Algorithm documentation (consistent with fractal_holos3.py)
- Added Multi-Round Flow Details section explaining:
  - Solver state persistence (incremental design)
  - Hologram merging with deduplication
  - Memory considerations for backward wave explosion
- Clarified mode selection as Layer 1 dimension (SeedSpec includes mode)
- Documented known placeholder: `feature_success` (declared but unused)
- Verified spine checkpoints: NOT cruft - used for O(1) move lookup

---

## 2026-01-22 (Initial)

- Created comprehensive LOCUS.md with complete manifest
- All 7 fixes from COMPARISON_WITH_FRACTAL_HOLOS3.md applied
- All equivalence tests passing
- Added `generate_boundary_seeds()` to ChessGame
- Added `apply_move()` to GameInterface and ChessGame
- Full file scan with design decision extraction
