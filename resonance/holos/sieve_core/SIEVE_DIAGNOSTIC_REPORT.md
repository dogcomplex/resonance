# Sieve Diagnostic Report: Learning Failure Analysis

## Update: Survival Task Testing

### Findings from Minimal State Test (DodgeBall)

With minimal state encoding (relative threat position + edge awareness), the sieve showed:

1. **Correct association learning**: When given proper state-action pairs, the sieve learns correct associations
   - `rel_0_L: L=-10.68, R=+0.75` (at left edge, threat above → RIGHT good, LEFT bad)
   - `rel_1_L: L=+1.22` (at left edge, threat one right → LEFT good)

2. **Action collapse problem**: Despite correct learning, the sieve collapses to one dominant action
   - Started at 50% exploration, ended at 92% LEFT
   - Once one action dominates, it doesn't explore alternatives

3. **Exploration-exploitation failure**: The anthropic principle alone isn't sufficient
   - Survival credit accumulates on whatever actions are tried
   - Once one action leads early, it self-reinforces

### Key Insight: Bootstrap IS Required

Your intuition was correct: early directed learning is essential. The pure anthropic principle has a cold-start problem:

```
PURE ANTHROPIC:
- All actions start equal
- Random exploration finds one that works
- That action gets survival credit
- Other actions never get tried
- Self-reinforcing loop to suboptimal strategy
```

```
WITH BOOTSTRAP:
- Explicitly emphasize promising directions
- Force exploration of state-action space
- Build associations BEFORE exploitation
- Anthropic selection refines, not discovers
```

### What Works

1. **State-level hashing** + **gradient tracking** (multi-scale)
2. **Death signal** (explicit negative interference)
3. **Survival credit** for dangerous situations
4. **Edge-aware state encoding**

### What's Missing

1. **Forced state-space coverage** - explore ALL state-action pairs
2. **Balanced exploration** - don't let one action dominate
3. **Contrastive within-state** - when in state S, penalize bad actions AND reward good ones

---

## Test Results Summary

| Task | Spiral Sieve | Wave Sieve | Random Baseline |
|------|--------------|------------|-----------------|
| TicTacToe Win Rate | 10.8% | - | ~30% |
| Pattern Matching | 24.6% | 23.2% | 25% |
| Sequence Prediction | 34.2% | 25.0% | 33% |
| XOR Learning | 51.1% | - | 50% |

**Verdict**: Neither sieve demonstrates meaningful learning on simple tasks.

## Root Cause Analysis

### Problem 1: Diffuse Pattern Space

Both sieves create individual tokens/embeddings for each pixel position. For a 4x4 pattern:
- Wave sieve creates ~16 pixel tokens per frame
- Spiral sieve creates ~1 combined embedding from 16 pixels

The issue: **patterns differ only in which pixels are active**, but both approaches spread the "identity" of a pattern across many weak signals rather than treating the whole pattern as one entity.

### Problem 2: Weak Credit Assignment

The sieves use "game length" as the primary reward signal:
```python
sieve.signal_game_end(game_length)
```

For simple classification tasks (pattern → action), game length is meaningless:
- Correct prediction → game_length = 10
- Wrong prediction → game_length = 1

This is a weak signal that gets diluted across all the couplings built during that trial.

### Problem 3: No Contrastive Learning

The sieves only observe what happens, never what didn't happen:
- Pattern A appears with Action 2 → build coupling A↔2
- But Pattern A should NOT couple with Actions 0,1,3
- Without negative examples, couplings spread equally to all actions

### Problem 4: Temporal Coupling Assumes Sequences

The wave sieve builds `prev_action → curr_pixels` couplings, assuming temporal continuity matters. But for classification:
- Pattern matching: no temporal structure
- XOR: no temporal structure
- TicTacToe: has temporal structure, but sieve can't learn illegal moves

## Why Pong Worked (Partially)

Pong had special properties that masked these issues:

1. **Spatial consistency**: Ball-paddle relationship is spatially correlated
2. **Continuous dynamics**: Ball position changes gradually, creating natural sequences
3. **Action-state coupling**: Paddle movement directly affects future states

The 45% hit rate (vs 33% random) came from building weak but consistent ball-paddle couplings over 50k frames.

## The Fundamental Insight

The sieve architecture is designed for:
> **Patterns that survive through time should be reinforced**

But our tests measure:
> **Patterns that predict correct actions should be reinforced**

These are different objectives. The sieve implements **anthropic selection** (survival → reinforcement), not **supervised learning** (correct → reinforcement).

## Recommendations

### Option A: Keep Anthropic Selection, Change Task

Test on tasks where survival = success:
- Continuous control (keep ball in play, keep agent alive)
- Long-horizon games where survival is the objective
- Environments with intrinsic entropy pressure

### Option B: Add Contrastive Signal

When an action is chosen wrong:
1. Identify what couplings led to that choice
2. Explicitly weaken those couplings
3. This turns weak anthropic selection into stronger supervised learning

### Option C: State-Level Hashing

Instead of pixel-level tokens, hash entire states:
```python
state_id = hash(frame.tobytes())  # One token per unique state
```

This would give 4 tokens for pattern matching (one per pattern), not thousands.

### Option D: Reconsider the Embedding Problem

The user's intuition was right:
> "We should either be looking at the problem generally and critiquing our own general encoding approach or simply utilizing existing solutions."

The embedding problem (how to represent states) may need to be separated from the sieve dynamics (how to select patterns). Using a pretrained encoder to produce meaningful state representations, then applying sieve selection on those, might work better.

## Next Steps

1. **Test Option C first** - simplest fix, most informative
2. **If still failing**, try Option B with explicit negative feedback
3. **Consider hybrid approach** - external encoder + sieve selection

The spiral geometry and physics principles are still valid, but the sieve needs meaningful state representations to select *between*, not raw pixel positions.

---

## Update: Natural Exploration Mechanisms

Tested four physics-aligned exploration mechanisms to solve the action collapse problem:

### Results on Simple Mapping + DodgeBall

| Mechanism | Simple Accuracy | Dodge Survival | Physical Basis |
|-----------|----------------|----------------|----------------|
| **Ergodic** (visit-counting) | 41.4% | 13.5 steps | Thermodynamic ergodicity |
| **Thermal Floor** (T > 0) | **99.1%** | 43.0 steps | Zero-point energy |
| **Symmetry** (preserve until broken) | 97.6% | 30.1 steps | Symmetry breaking |
| **Superposition** (quantum collapse) | 98.5% | **180.9 steps** | Wave function |
| Random baseline | 32.6% | 23.7 steps | - |

### Key Finding: Thermal Floor + Superposition are the SAME Physics

From different perspectives:
- **Thermal floor**: Energy minimum per mode (E = hbar*omega/2)
- **Superposition**: All states exist until collapse (wave function)

These are dual descriptions of quantum vacuum fluctuations. Zero-point energy IS the superposition of vacuum modes.

### Unified Quantum Sieve Implementation

Combined thermal floor + superposition into single mechanism:

```python
# Core principle: Zero-point amplitude prevents action collapse
ZERO_POINT_AMPLITUDE = 0.1  # sqrt(E_0)

# Actions exist in superposition until evidence collapses
# P(action) = |amplitude|^2 (Born rule)
# All amplitudes >= ZERO_POINT (never zero probability)
```

### Pong Results

| Sieve | Hit Rate | Game Length Improvement |
|-------|----------|------------------------|
| Random | 33% | - |
| Pure Wave | 45.2% | +42.9% |
| **Unified Quantum** | 43.0% | **+57.9%** |

The unified quantum sieve achieves:
- Comparable accuracy to pure wave sieve
- **Better generalization** (larger game length improvement)
- **Stable amplitudes** (no action collapse)
- **Physically grounded** exploration mechanism

### The Consolidation Insight

The four exploration mechanisms collapse to TWO fundamental principles:

1. **Zero-point energy** (thermal floor) = prevent amplitude collapse
2. **Superposition** = maintain all possibilities until evidence

And these two are actually ONE principle viewed from complementary perspectives:
- Zero-point energy is the ENERGY of maintaining superposition
- Superposition is the STATE that requires zero-point energy

The unified quantum sieve implements both through a single mechanism:
- All actions start in equal superposition
- Zero-point floor prevents any action from reaching zero probability
- Evidence (survival/death) modifies amplitudes through interference
- Born rule converts amplitudes to probabilities

This is exactly how quantum mechanics works: the vacuum is never empty, it's a superposition of all possible fluctuations.

---

## Update: Discrete Games Testing

Tested the physics-pure quantum sieve on discrete games to assess learning capabilities:

### Results

| Task | Result | Baseline | Status |
|------|--------|----------|--------|
| Pattern Matching | **90.9%** | 25% | PASS |
| Sequence Memory | **94.4%** | 33% | PASS |
| TicTacToe | **62.3% wins** | ~30% | PASS |
| Mini Sudoku | **0.0%** | ~0.02% | FAIL |

### Analysis: What the Sieve CAN Learn

The sieve excels at tasks with:

1. **Recurring states**: Pattern matching has only 4 unique patterns. The sieve sees each many times and builds strong associations.

2. **Spatial/temporal structure**: TicTacToe has spatial relationships that generalize. Sequence memory has temporal patterns.

3. **Direct state-action mapping**: When seeing state S predicts action A should follow, the sieve learns S -> A.

### What the Sieve CANNOT Learn (Yet)

Sudoku failed because:

1. **Every state is unique**: 500 puzzles = 500 different board configurations. The sieve never sees the same state twice.

2. **Requires abstract reasoning**: Sudoku needs understanding of *constraints* (row/column/box uniqueness). The sieve learns associations, not logic.

3. **No feature generalization**: The sieve hashes entire states. It can't extract "this row has 1,2,3 so needs 4" from the state.

### The Fundamental Insight

The quantum sieve learns: **"When I see pattern X, action Y tends to lead to survival"**

Sudoku requires: **"Given constraints C1, C2, C3, the valid actions are..."**

These are fundamentally different:
- **Association learning** (sieve can do this)
- **Constraint satisfaction** (sieve cannot do this yet)

### Possible Solutions

1. **Feature extraction**: Instead of hashing raw state, extract features like "row constraints satisfied", "column needs which digits", etc. This adds domain knowledge.

2. **Relational encoding**: Build couplings between features (digit X in position Y) rather than whole states. More physics-like - local interactions.

3. **Hierarchical states**: Hash sub-patterns (rows, columns, boxes) separately, then combine. Multi-scale physics.

4. **Accept the limitation**: The sieve may be fundamentally suited for sensorimotor learning (like biological neurons) rather than abstract reasoning (like symbolic AI).

### Physics Audit

Magic numbers remaining in the implementation:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| PSI_0 = 0.1 | ✅ | Zero-point amplitude (physics) |
| tau = 3.0 | ⚠️ | Decoherence time (should derive from mode density) |
| reduction = 0.5 | ❌ | Death penalty (magic) |
| heat_bath * 0.01 | ❌ | Learning rate (magic) |
| heat_bath * 0.1 | ❌ | Success boost (magic) |

These magic numbers work, but ideally should be derived from energy conservation and mode counting.
