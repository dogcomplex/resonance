# ResonanceCrystal Backtest Results

## Summary Table

| Environment | F1/Accuracy | Rules | Classes | Notes |
|-------------|-------------|-------|---------|-------|
| **SimpleStateMachine** | **100%** | 8 | 4 | Perfect on deterministic |
| **StochasticStateMachine** | **76.5%** | 13 | 4 | Above theoretical optimal (~75%) |
| **GridWorld** | **80.5%** | ~363 | ~48 | Good spatial abstraction |
| **Pong** | **69.8%** | ~99 | ~65 | Continuous physics |
| **Minesweeper** | **49.7%** | ~173 | ~75 | Logic deduction hard |
| **2048** | **35.0%** | ~368 | ~59 | Position+value coupling |

## Key Findings

### ✅ Works Well On

1. **Deterministic state machines** - 100% perfect learning
2. **Stochastic transitions** - Learns most likely outcomes, temperature reflects probability
3. **Spatial abstraction** - Merges equivalent positions (GridWorld doors, Pong ball positions)
4. **Physics-like dynamics** - Pong ball movement patterns

### ⚠️ Struggles With

1. **Logic deduction** (Minesweeper) - Rules depend on hidden information
2. **Numeric combinations** (2048) - Position+value tokenization prevents value-based rules

### 🔍 Insights

1. **Temperature = inverse probability**: Low temp rules are high-confidence (0.02 for 95% transitions, 0.21 for 50% transitions)

2. **Class merging IS fidelity**: Objects at different positions with same effects merge automatically
   - `door_closed_1_2`, `door_closed_3_2` → same class
   - `ball_at_3`, `ball_at_5` → same class

3. **Tokenization matters more than the sieve**: 2048 fails because tokenization couples position+value

4. **Single sieve works**: No need for separate fidelity/probability/temporal sieves - fold-based annealing handles all

## Recommendations

1. **For new domains**: Design tokenization to separate independent aspects (position vs value, object type vs location)

2. **For logic**: May need explicit uncertainty/partial-observability handling

3. **For Pong ceiling**: Separate ball and paddle into independent prediction channels

## The Unified Principle

> **Resonance = persistence across samples of reality**

Everything collapses to this. What survives entropy is what we call "rules."

---

## Hierarchical Resonance Experiments

### The Idea
Layer 1: State tokens → Rules (what changes together)
Layer 2: Rule tokens → Meta-rules (which rules are equivalent)

Treat Layer 1 rules as tokens for a second resonance pass. Rules that "do the same thing" 
at different positions should have similar activation patterns in Layer 2.

### Results on 2048

| Approach | F1 | Notes |
|----------|-----|-------|
| Baseline (position+value) | 35.0% | |
| Hierarchical L2 on rule activations | 30.0% | Rules grouped by ACTIVATION, not VALUE |
| Value-based rule grouping | 31.7% | 50 rule groups found, but effects still position-specific |
| Value-only tokenization | 31.8% | Loses position info needed for sliding |

### Key Finding

2048 is fundamentally **position-dependent** for a different reason than Pong:
- Pong: Position doesn't matter (ball at x=3 behaves like ball at x=5)
- 2048: Position DOES matter (sliding RIGHT affects columns, DOWN affects rows)

The Layer 2 approach found 50-70 rule equivalence classes, but:
1. Rules grouped by activation pattern, not value semantics
2. Even when grouped by value, EFFECTS still have position-specific tokens
3. The "same" operation at different positions predicts different position-specific tokens

### What Would Work

For 2048, would need to discover:
- "Column equivalence" (cells in same column behave similarly for LEFT/RIGHT)
- "Row equivalence" (cells in same row behave similarly for UP/DOWN)
- "Merge patterns" (4+4=8 regardless of position)

This is a STRUCTURAL abstraction, not just wave interference.

### The Hierarchical Insight is Still Valid

The two-layer approach DOES find rule equivalences. On environments where position genuinely 
doesn't matter (GridWorld, Pong), it should help more. The issue is 2048's position-dependence 
is about STRUCTURE (rows/columns), not just arbitrary positions.

---

## Deep Resonance - Recursive Layers

### The Idea
Keep stacking layers until no new structure emerges:
- Layer 0: Raw tokens → classes + rules
- Layer 1: L0 classes + L0 rules as tokens → meta-classes + meta-rules
- Layer N: L(N-1) outputs → further abstraction
- Stop when no compression

### Implementation
Each layer treats BOTH equivalent classes AND rules from the previous layer as tokens.
Rules that behave similarly (fire together, cause similar effects) get merged.

### Results

| Environment | Baseline | Deep Resonance | Improvement |
|-------------|----------|----------------|-------------|
| **GridWorld** | 80.5% | **87.4%** | +6.9% |
| Pong | 69.8% | 65.9% | -3.9% |
| 2048 | 35.0% | 0.0% | Collapsed |

### Key Findings

1. **GridWorld improved significantly** (87.4% vs 80.5%)
   - 293 rules found equivalences
   - Rules for "move right from any position" get grouped
   - Enables matching via equivalent rules when direct match fails

2. **Pong slightly worse** - equivalences too loose, wrong rules matched

3. **2048 collapsed** - Layer 0 merges too aggressively, all cells look similar

### The Core Insight

The recursive layer idea WORKS for finding structural patterns:
- "All right-movement rules are equivalent"
- "All key-pickup rules are equivalent"
- etc.

But needs careful tuning:
- Too loose coherence → everything collapses
- Too tight → no useful equivalences found

### Future Work

1. **Separate coherence per layer** - tighter at L0, looser at higher layers
2. **Action-aware equivalences** - only group rules with same action
3. **Effect-aware equivalences** - only group rules with similar effect structure
