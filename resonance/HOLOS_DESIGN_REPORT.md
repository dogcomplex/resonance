# HOLOS: Hierarchical Omniscient Learning and Optimization System

## Design Report - January 2026

---

## Executive Summary

HOLOS is a fractal game-solving algorithm that combines:
1. **Bidirectional search** (forward from positions, backward from solved boundaries)
2. **Holographic storage** (store paths/spines, not all positions)
3. **Recursive compression** (each abstraction level is itself a "game" solved by the same algorithm)

This report assesses the viability of using HOLOS to solve 8-piece chess endgames by expanding backward from the 7-piece Syzygy tablebase boundary, with implications for larger piece counts.

---

## 1. Core Algorithm

### 1.1 Bidirectional Search (Lightning + Crystal)

```
DEDUCTIVE (Forward):  Start positions → expand via legal moves → hit boundary
ABDUCTIVE (Backward): Boundary positions → expand via predecessors → hit starts

When waves MEET → Crystallize (rigorous local expansion)
```

### 1.2 Predecessor Generation (Backward Search)

Given a position P, generate all positions P' such that P' →(legal move)→ P

This involves:
- **Un-moving**: For each piece at `to_sq`, find valid `from_sq` it could have come from
- **Un-capturing**: Optionally restore a captured piece at `to_sq`

Test results (KQRRvKQR):
- 50 boundary positions → 2,085 unique 8-piece predecessors
- Generation rate: ~82,000 predecessors/second
- Verification: All predecessors correctly lead to original position

### 1.3 Holographic Storage

Instead of storing all position→value pairs:
- **Spines**: Principal variations connecting layers
- **Decision points**: Where optimal play diverges
- **Boundary contacts**: Direct connections to Syzygy

Query: Trace any position to nearest stored node → follow path to boundary

---

## 2. Compression Theory

### 2.1 The α-Compression Model

If a game has self-similar structure at multiple scales:

```
N₀ = raw positions
N₁ = N₀^α (first abstraction level)
N₂ = N₁^α = N₀^(α²)
...
Nₖ = N₀^(αᵏ)

Total storage = Σᵢ Nᵢ ≈ N₀^α / (1-α)  [geometric series if α < 1]
```

### 2.2 Does α Need to Be 0.1?

**NO.** Any α < 1 gives compression. The question is diminishing returns:

| α | Compression per level | Levels to reach ~10³ from 10⁴⁴ |
|---|----------------------|-------------------------------|
| 0.9 | 10% reduction | ~400 levels (impractical) |
| 0.5 | Square root | ~15 levels |
| 0.1 | 10th root | ~4 levels |
| 0.01 | 100th root | ~2 levels |

**Key insight**: Even α = 0.5 (factor of 2 compression) is useful if we can stack levels:

```
Level 0: 10^9 positions (8-piece)
Level 1: 10^4.5 ≈ 30,000 (if α=0.5)
Level 2: 10^2.25 ≈ 180
Level 3: ~13
Level 4: ~4

Total: ~30,200 stored vs 10^9 = 33,000x compression
```

**The magic is recursive application.** Even weak compression compounds exponentially across levels.

### 2.3 Theoretical Foundation

This draws from several fields:

1. **Sparse Coding (Neuroscience)**
   - Brain stores basis vectors, not raw data
   - Reconstruction via linear combination
   - Chess analogy: tactical motifs as basis vectors

2. **Kolmogorov Complexity**
   - Shortest program that generates the data
   - If chess has structure, it's compressible
   - HOLOS finds the "program" (hierarchy of games)

3. **Holographic Principle (Physics)**
   - 3D information encoded on 2D boundary
   - HOLOS: N-piece information encoded via (N-1)-piece boundary

4. **Fractal Compression (Image Processing)**
   - Self-similar patterns at multiple scales
   - Store transformations, not pixels
   - HOLOS: Store path patterns, not positions

---

## 3. Coverage Requirements

### 3.1 Do We Need Full Coverage?

**For guaranteed correctness**: Yes, we need all 8-to-7 paths to be reachable from our stored structure.

**For practical utility**: No. Partial coverage still valuable:

| Coverage | Use Case |
|----------|----------|
| 100% | Tablebase-quality (never wrong) |
| 90% | Strong heuristic (occasionally falls back to search) |
| 50% | Significant speedup (guide search in half of positions) |

### 3.2 One Region vs Full Coverage

**Option A: One region, extrapolate pattern**
- Solve KQRRvKQR exhaustively
- Identify compression patterns
- Hypothesize these patterns generalize
- Test on other 7-piece boundaries

**Option B: Multiple regions, verify generalization**
- Solve several different material configurations
- Compare compression patterns
- Only trust patterns that appear consistently

**Recommended approach**: Start with Option A (one region), but design for Option B. The algorithm should be material-agnostic.

### 3.3 What Counts as "Pattern"?

Patterns we might discover:
- **Structural**: "Positions with king distance > 4 tend to be draws"
- **Material**: "Extra rook usually wins in this configuration"
- **Tactical**: "Fork positions cluster in outcome"
- **Topological**: "Positions N moves from boundary form predictable value distribution"

The hierarchy doesn't need to be semantic (tactics/strategy). It can be purely **structural** based on position features.

---

## 4. Implementation Plan

### Phase 1: Exhaustive 8↔7 Mapping (Current)

```
Input: KQRRvKQR.rtbw (7-piece tablebase)
Output: All 8-piece positions reachable from this boundary

Method:
1. Enumerate 7-piece positions from Syzygy
2. Generate predecessors (8-piece)
3. For each 8-piece, verify it can reach 7-piece boundary
4. Store paths, not just positions
```

### Phase 2: Compression Analysis

```
Input: Complete 8↔7 mapping
Output: Identified compression opportunities

Analysis:
1. Cluster positions by features (material, king distance, etc.)
2. Identify clusters with uniform values
3. Measure compression ratio: clusters/positions
4. Identify minimal distinguishing features
```

### Phase 3: Hierarchical Storage

```
Input: Compression analysis results
Output: Holographic storage structure

Structure:
- Level 0: Raw 8-piece positions (reference only)
- Level 1: Feature clusters with values
- Level 2: Meta-patterns across clusters
- Query: Descend hierarchy to find value
```

### Phase 4: Validate and Extend

```
Test: Random 8-piece positions, compare hologram query vs Syzygy
Extend: Apply same method to 9-piece, 10-piece, ...
```

---

## 5. Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Chess has low compressibility | Medium | High | Even 2x helps; fall back to direct storage |
| Predecessor generation misses positions | Low | High | Formal verification of algorithm |
| Memory explosion during enumeration | High | Medium | Chunked processing, disk caching |
| Patterns don't generalize across materials | Medium | Medium | Design material-agnostic features |

---

## 6. Philosophical Implications

The HOLOS approach suggests:

1. **Compression = Understanding**
   - If we can compress chess, we understand its structure
   - The hierarchy IS the understanding

2. **Self-Similar Games**
   - Each abstraction level is itself a game
   - Solved by the same algorithm recursively
   - "Games all the way down"

3. **Boundary-Centric Knowledge**
   - You don't need to know everything
   - You need to know paths to what you know
   - Knowledge is relational, not absolute

---

## 7. Next Steps

1. **Implement robust backward expansion** in fractal_holos3.py
2. **Run exhaustive KQRRvKQR test** - enumerate all 8-piece positions reachable from boundary
3. **Analyze compression potential** - cluster by features, measure redundancy
4. **Design Level 1 abstraction** - based on discovered patterns
5. **Iterate** - apply same analysis to Level 1, discover Level 2

---

## Appendix: Test Results

### Predecessor Generation Test (2026-01-22)

```
Input: 100 unique 7-piece KQRRvKQR positions
Output: 2,085 unique 8-piece predecessors (from 50 positions)
Rate: 82,128 predecessors/second
Verification: 100% of predecessors correctly lead to original position

Material distribution of predecessors:
  KQQRRvKqr: 20
  KQRRRvKqr: 20
  KQRRBvKqr: 20
  KQRRNvKqr: 20
  KQRRPvKqr: 20
```

### Syzygy Boundary Values

```
From 100 random KQRRvKQR positions:
  White wins (+1): 86
  Draw (0): 8
  Black wins (-1): 6

Strong white advantage in this material configuration (as expected with extra rook).
```

---

## Appendix B: Exhaustive Boundary Test Results (2026-01-22)

### Test Configuration
```
Material: KQRRvKQR (7-piece) → 8-piece predecessors
Boundary positions: 2,000
Max uncaptures: 5 per position
```

### Results

| Metric | Value |
|--------|-------|
| 7-piece boundary | 2,000 positions |
| 8-piece predecessors | 81,535 positions |
| Expansion factor | 40.8x |
| Minimax solvable | 5,465 (6.7%) |
| Feature clusters | 65 |
| **Compression ratio** | **1,254x** |

### Feature Space Analysis

The compression comes from limited feature dimensionality:

| Feature | Unique Values | Compression |
|---------|---------------|-------------|
| Material (white) | 5 (QQRR, QRRR, QRRB, QRRN, QRRP) | 8,232x |
| Material (black) | 1 (QR always) | 41,160x |
| King distance | 13 (2-14) | 3,166x |
| Turn | 1 (always black after uncapture) | 41,160x |
| **Combined** | **65** | **633x** |

### Key Insight: α ≈ 0.14 for this configuration

```
N_positions = 81,535
N_features = 65
α = log(N_features) / log(N_positions) = 1.81 / 4.91 ≈ 0.37

But this is FIRST LEVEL only!
If pattern continues: 65 → ~8 → ~3 → 1
That's 4 levels with geometric compression.
```

### Implication for Full Chess

If this pattern holds across material configurations:
- Each piece-count layer compresses ~1000x via features
- 8-piece to 7-piece: ~1000x
- 9-piece to 8-piece: ~1000x (speculative)
- ...
- 32-piece to 31-piece: ~1000x

**Total potential compression: 1000^25 ≈ 10^75**

This would make full chess theoretically tractable if:
1. The compression pattern generalizes
2. The hierarchy can be computed efficiently
3. Query time remains reasonable

---

## Appendix C: Critical Finding - Features Don't Predict Value (2026-01-22)

### The Reality Check

After deeper analysis, the initial compression results were **misleading**:

| Metric | Initial Claim | After Verification |
|--------|---------------|-------------------|
| Feature clusters | 65 | 65 (correct) |
| Compression | 12,540x | 12,540x (correct) |
| **Value consistency** | 100% | **~20%** |
| **Predictive accuracy** | ? | **84.5% (= majority baseline)** |

### The Problem

The features (material + king distance) **do not predict game-theoretic value**:

| Outcome | % of Positions | Avg King Distance | King on Edge |
|---------|----------------|-------------------|--------------|
| Win (+1) | 83.3% | 5.48 | 43% |
| Draw (0) | 8.2% | 5.55 | 46% |
| Loss (-1) | 8.4% | 5.53 | 37% |

**Differences are statistically insignificant** (< 0.1 squares difference).

### What This Means

1. **Compression without prediction is useless**
   - Grouping positions by features doesn't help if values vary within groups
   - The "compression" is real but the groups don't share outcomes

2. **Chess value is highly positional**
   - The exact placement of pieces matters, not just material and rough distances
   - Two positions with identical features can have opposite values

3. **Simple features won't work**
   - Need features that capture tactical motifs (forks, pins, mating threats)
   - These are combinatorially complex - may not compress well

### Implications for HOLOS

The hierarchical compression approach **may still work** but requires:

1. **Better features** that actually predict value (hard problem)
2. **Learned representations** (neural embeddings that capture tactical patterns)
3. **Accept lower compression** (~2-5x with high accuracy vs 1000x with no accuracy)

### The Honest Assessment

| Approach | Compression | Accuracy | Verdict |
|----------|-------------|----------|---------|
| Simple features | 1000x+ | = baseline | ❌ Useless |
| Enhanced features | ~2x | ~90% | 🟡 Marginal |
| Position-specific | 1x | 100% | ✅ But no compression |

**The fundamental tension**: Features that actually predict value don't compress much. Features that compress well don't predict value.

This is consistent with why chess tablebases are so large - the game-theoretic value really does depend on exact piece placement in complex ways.

---

## Appendix D: Reframing - Compressing SEARCH, not SOLUTION (2026-01-22)

### The Correct Question

The previous analysis asked: "Can features predict position VALUE?"
The correct question is: "Can we optimize the SEARCH to cover the space efficiently?"

These are fundamentally different:
- **Value prediction**: Compress the solution (fails for chess)
- **Search optimization**: Compress the computation (this is HOLOS)

### Key Measurements

**Connectivity from 7-piece to 8-piece:**
```
For 100 7-piece positions:
  Avg 8-piece predecessors: 39.8
  Min: 0, Max: 45

Each 7-piece position fans out to ~40 8-piece predecessors.
```

**8-piece to 7-piece reachability:**
```
For 200 random 8-piece positions:
  Can reach 7-piece in 1 capture: 74.5%
  Need quiet moves first: 25.5%
```

### The Meta-Game

The higher-order game is **SET COVER**:

```
Given: The 8-piece position space (~10^9 positions)
Goal: Cover it with minimum 7-piece boundary seeds
Constraint: Each 7-piece seed covers ~40 8-piece via backward expansion

Optimal strategy = solve SET COVER on the connectivity graph
```

Features that matter for this meta-game:
- **Connectivity**: Which 7-piece positions have MANY predecessors?
- **Overlap**: Which 7-piece positions have NON-OVERLAPPING predecessor sets?
- **Material coverage**: Do we need seeds from each material configuration?

### The Fractal Structure

```
Level 0: Solve 8-piece via backward expansion from 7-piece
         Meta-game: Which 7-piece to seed? (SET COVER)

Level 1: Solve 9-piece via backward expansion from 8-piece
         Meta-game: Which 8-piece to seed?

Level 2: Solve the meta-game itself
         Which regions of the meta-game to solve first?
```

Each level is:
1. A position game (chess at that piece count)
2. A meta-game (SET COVER for seeding)
3. A meta-meta-game (which parts of SET COVER to solve)

**This is where the fractal compression comes from** - not from predicting values, but from solving the meta-games efficiently.

### Implications

1. **Simple features don't predict value** - CORRECT, and that's OK
2. **Simple features might predict CONNECTIVITY** - This is what matters
3. **The hierarchy is about SEARCH STRATEGY, not VALUE PREDICTION**

### Next Steps

1. Measure connectivity patterns in 7-piece space
2. Identify high-coverage 7-piece seed positions
3. Test if connectivity features generalize across materials
4. Implement SET COVER heuristic for seeding

---

## Appendix E: Modular Architecture Analysis (2026-01-22)

### Answering Key Design Questions

#### Q1: Does this provide compounding solution storage gains?

**YES**, but the storage model needs clarification:

**What we store:**
- **Seed** → deterministic expansion → **Frontier**
- Given a seed (small), we can reconstruct the frontier (large)
- Each frontier is cached to compute the next frontier
- But we can always re-derive it from the seed chain

**The compression:**
```
Full frontier storage: O(|frontier|) states
Seed-based storage: O(|seeds|) where |seeds| << |frontier|

If seeds expand at factor F per depth:
  Depth 5 from 1 seed = 8000+ positions
  Storage: 1 seed vs 8000 positions = 8000x compression
```

**Chain structure:**
```
Seed_0 → expand → Frontier_0 → (Layer1 picks) → Seed_1 → expand → Frontier_1 → ...
```

We only NEED to store the seeds. Frontiers are cached for efficiency but reconstructible.

---

#### Q2: On minimax vs max-eff propagation - are these game-specific or universal?

**They are GAME-SPECIFIC, not for HOLOS to worry about.**

**Why propagation differs:**

1. **Chess (adversarial game):**
   - Two players with opposite goals
   - Minimax: "What's best assuming opponent plays optimally"
   - Value propagation: parent = max(children) if my turn, min(children) if opponent's turn

2. **Seed Selection (optimization):**
   - Single agent maximizing efficiency
   - Max-eff: "What configuration gives best coverage/cost"
   - Value propagation: parent's potential ≥ best reachable child

3. **General case:**
   - Could be cooperative (sum values)
   - Could be probabilistic (expected value)
   - Could be multi-objective (Pareto)

**HOLOS doesn't care** - it just calls `game.propagate_value(state, child_values)` and lets the game decide. The game module defines what "value" means and how it flows.

**For lightning heuristics:**
- The game provides `score_for_lightning(state, move)`
- HOLOS uses this to prioritize DFS branches
- Chess: prioritize captures (MVV-LVA)
- Seeds: prioritize high-connectivity additions

---

#### Q3: On equivalence classes / reflections - discovered procedurally or from rules?

**Both, and they're different:**

1. **Rule-based symmetries (game module):**
   - Board reflections (horizontal flip in chess)
   - Color swap (swap all white/black pieces)
   - These are KNOWN from game rules
   - ChessState.__hash__() already does horizontal canonicalization

2. **Discovered equivalences (HOLOS):**
   - "Positions with same material + king distance have same value"
   - Discovered by observing that feature clusters have consistent outcomes
   - This is what `equiv_classes` and `equiv_outcomes` track

**Current implementation does both:**
- Rule-based: In `ChessState.__hash__()` (line 69-74)
- Discovered: In `Hologram.add_with_features()` and `propagate_equivalence()`

---

#### Q4: Forward and backward as mirrors - can we reduce to single direction?

**YES, conceptually they're symmetric:**

```
Forward: start → ... → boundary
Backward: boundary → ... → start

If game has:
  get_successors(state) → children
  get_predecessors(state) → parents

Then backward search IS forward search on the "reversed game":
  reversed_game.get_successors = original_game.get_predecessors
```

**Implementation insight:**
```python
class ReversedGame(GameInterface):
    def __init__(self, original):
        self.original = original

    def get_successors(self, state):
        return self.original.get_predecessors(state)

    def get_predecessors(self, state):
        return self.original.get_successors(state)
```

---

#### Q5: Multi-round session management

**Current architecture is missing proper multi-round handling. Proposed:**

```python
class HOLOSSession:
    """Manages multi-round frontier evolution"""

    def __init__(self, game, meta_game):
        self.game = game           # Layer 0: chess
        self.meta_game = meta_game # Layer 1: seed selection

        self.frontier_history = []  # List of (seed, frontier) pairs
        self.current_frontier = None

    def run_round(self):
        # 1. Layer 1 picks seeds for this round
        seeds = self.meta_game.select_seeds(self.current_frontier)

        # 2. Layer 0 expands from seeds
        new_frontier = self.game.expand(seeds)

        # 3. Save the mapping
        self.frontier_history.append((seeds, self.current_frontier))
        self.current_frontier = new_frontier

        # 4. Layer 1 learns from results
        self.meta_game.update(seeds, new_frontier)
```

---

#### Q6: Should we bound to one seed? Expand both frontiers simultaneously?

**These are Layer 1 (meta-game) decisions!**

The meta-game should choose:
- **Number of seeds** per expansion (1? 10? 100?)
- **Which frontier** to expand (forward? backward? both?)
- **What mode** (lightning? wave? crystal?)
- **How deep** each seed should go

Suggested: Make all these parameters part of the expansion plan:
```python
@dataclass
class ExpansionPlan:
    forward_seeds: List[SeedSpec]   # Seeds for forward wave
    backward_seeds: List[SeedSpec]  # Seeds for backward wave
    mode: SearchMode                # Lightning, Wave, or Crystal
    sync: bool                      # Expand both simultaneously?
```

---

### The Inductive Step

HOLOS is fundamentally about the **inductive step**:

**Deduction** (forward): Given rules and state, derive next states
**Abduction** (backward): Given outcome, hypothesize what led there
**Induction** (connection): When forward meets backward, we've PROVEN a path

```
Forward wave: "These states are reachable from start"
Backward wave: "These states lead to known outcomes"
Intersection: "These states are BOTH reachable AND have known outcomes"
             = Complete solution for that region
```

**Lightning IS induction** - it's hypothesizing which paths matter most and testing them quickly.

**Crystal IS local induction** - once we find a connection, we inductively fill in the neighborhood.

---

### Nodes as Game States, Transitions as Rules

This is the key generalization:

```
HOLOS doesn't see "chess positions" - it sees:
  - Nodes (hashable states)
  - Edges (transitions via rules)
  - Values (outcomes)

Layer 0: nodes=chess positions, edges=moves, values=win/draw/loss
Layer 1: nodes=seed configs, edges=add/remove seed, values=efficiency
Layer N: nodes=meta-configs, edges=parameter changes, values=meta-efficiency
```

The recursion is clean because each layer just sees "a game" - it doesn't know or care that the game happens to be about optimizing another game.

---

### Action Items

1. **Add SessionManager** for multi-round frontier evolution
2. **Add backward lightning** (DFS from boundary)
3. **Store spine paths** for compressed reconstruction
4. **Make mode selection** a real meta-decision
5. **Add seed→frontier deterministic mapping** with storage
6. **Add symmetry discovery** to the core (not just game-specific)

---

*Report updated with modular architecture analysis*
*2026-01-22*
