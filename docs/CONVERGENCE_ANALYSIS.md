# Comprehensive Convergence Analysis

## Metrics Used

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Raw Accuracy** | correct / total | Inflated by majority class (75% 'ok') |
| **Balanced Accuracy** | mean of per-class accuracies | Fair across all classes |
| **Weighted Accuracy** | inverse-frequency weighted | Emphasizes rare classes |

### Example: "Always predict 'ok'"
- Raw: 75% ✓ (looks good!)
- Balanced: 25% ✗ (reveals the problem)
- Weighted: 2% ✗ (penalizes missing rare classes)

## TicTacToe Convergence Timeline

### Dataset
- Total states: 5,890
- ok: 76.7%, winX: 16.0%, winO: 7.0%, draw: 0.3%

### Convergence Curve

| Observations | Raw | Balanced | Weighted | Δ vs Baseline |
|-------------|-----|----------|----------|---------------|
| 10 | 70% | 35% | 10% | +10% |
| 50 | 61% | 35% | 14% | +10% |
| 100 | 62% | 39% | 17% | +14% |
| 200 | 69% | 51% | 36% | +26% |
| 500 | 85% | 73% | 63% | +48% |
| 1000 | 92% | 88% | 88% | +63% |
| 2000 | 96% | 94% | 93% | +69% |
| 5890 | 99% | 98% | 96% | +73% |

### Per-Class Accuracy Evolution

| Class | @100 | @500 | @1000 | Final |
|-------|------|------|-------|-------|
| ok | 70% | 93% | 92% | 99% |
| winX | 29% | 80% | 91% | 99% |
| winO | 0% | 58% | 73% | 97% |
| draw | 0% | 100% | 80% | 100% |

## Pattern-Based Random Oracles

Testing on truly random patterns (not TicTacToe structure):

| Oracle | Patterns | Final Balanced | Final Weighted |
|--------|----------|----------------|----------------|
| Pattern10 | 10 | 84% | 72% |
| Pattern25 | 25 | 69% | 61% |
| Pattern50 | 50 | 54% | 57% |
| TrueRandom | 0 | 20% | 3% |

### Key Finding: Complexity vs Learnability
- 10 patterns: Highly learnable (84% balanced)
- 25 patterns: Moderately learnable (69% balanced)
- 50 patterns: Difficult (54% balanced)
- True random: Unlearnable (20% = random guessing)

## Learner vs Baselines

### TicTacToe
| Learner | @100 | @500 | @1000 | Final |
|---------|------|------|-------|-------|
| Majority (always 'ok') | 33% | 33% | 25% | 25% |
| Random (sample dist) | 28% | 31% | 23% | 25% |
| **Pattern Learner** | **49%** | **69%** | **79%** | **96%** |

**Improvement: +71% balanced accuracy over majority baseline**

### Random10 Patterns
| Learner | @100 | @500 | @1000 | Final |
|---------|------|------|-------|-------|
| Majority | 20% | 20% | 20% | 20% |
| **Pattern Learner** | **29%** | **68%** | **74%** | **84%** |

**Improvement: +64% balanced accuracy**

## Key Insights

### 1. The "ok default" is NOT cheating when measured properly
- Raw accuracy rewards predicting majority class
- Balanced accuracy reveals true learning
- Weighted accuracy emphasizes rare class handling

### 2. Convergence Phases
1. **Cold start (0-100)**: ~35% balanced, few rules, mostly defaulting
2. **Learning (100-500)**: 40-70% balanced, rules crystallizing
3. **Refinement (500-2000)**: 70-95% balanced, edge cases resolving
4. **Converged (2000+)**: 95%+ balanced, all rules stable

### 3. Structured vs Random
- TicTacToe (8 structured patterns): 98% balanced final
- Random 10 patterns: 84% balanced final
- Random 50 patterns: 54% balanced final
- True random: 20% balanced (unlearnable)

### 4. Pattern Discovery
- TicTacToe: Discovers exactly 16 rules (8 X + 8 O)
- Random patterns: Discovers ~150-220 rules (includes spurious)
- More patterns = more hypothesis collision = harder learning

## Recommendations

1. **Use balanced accuracy** as primary metric
2. **Track per-class accuracy** for rare class debugging
3. **Use weighted accuracy** when rare classes are critical
4. **Expect ~500 observations** for decent convergence on 10-20 pattern systems
5. **Expect ~2000 observations** for near-complete convergence

## Improvement Strategies

### 1. Prior-Based Learning ✓ IMPLEMENTED
**Idea**: Pre-load known game patterns as hypotheses

**Results**:
- Known game (TicTacToe): **100% from observation 1!**
- Variant game (RowsOnly): **100% from observation 1!**
- Unknown game: Falls back to pattern learning

**Tradeoff**: Only helps if game is in library

### 2. Meta-Learning / Game Identification ✓ IMPLEMENTED
**Idea**: Identify WHICH game is being played, use its rules

**Results**:
| Game | @1 | @5 | @10 | @50 |
|------|----|----|-----|-----|
| TicTacToe (Baseline) | 100% | 50% | 33% | 52% |
| TicTacToe (Meta) | 100% | 100% | 100% | 100% |
| RowsOnly (Baseline) | 0% | 33% | 33% | 33% |
| RowsOnly (Meta) | 100% | 100% | 100% | 99% |

**Key Insight**: Meta-learner eliminates wrong games within 5-10 observations

### 3. Active Learning (NOT YET IMPLEMENTED)
**Idea**: Request specific observations to test hypotheses

**Potential**: Could converge in ~8-16 observations (one per pattern)

### 4. Hierarchical Patterns (NOT YET IMPLEMENTED)
**Idea**: Compose patterns from primitives (row, col, diag)

**Benefit**: More compact, generalizable representation

### 5. Negative Learning (PARTIAL)
**Current**: Draw = !anyWin AND boardFull

**Could extend**: Error = invalidParity OR bothWin

### 6. Transfer Learning (NOT YET IMPLEMENTED)
**Idea**: Adapt known game rules to new variants

**Example**: TicTacToe → "TicTacToe but only rows"

### 7. Structural Priors (TESTED - HURT ACCURACY)
**Lesson**: Generic priors (prefer rows/cols/diags) hurt early accuracy by causing false positives. Priors must be SPECIFIC (known games) not GENERAL.

## Recommended Hybrid Approach

```
1. Check known games library (meta-learning)
   ↓ If match found → 100% accuracy immediately
   
2. Eliminate contradicted games quickly (5-10 obs)
   ↓ If single game remains → Use its rules
   
3. Fall back to pattern learning
   ↓ Expect ~500 obs for decent convergence
   
4. Use confirmed patterns to refine game identification
   ↓ "This looks like TicTacToe minus diagonals"
```

## Early Convergence Comparison

| Approach | @1 | @10 | @50 | @100 | Notes |
|----------|----|----|-----|------|-------|
| Baseline | 100% | 33% | 52% | 39% | Random luck early |
| Meta (known game) | 100% | 100% | 100% | 100% | Perfect from start |
| Meta (unknown game) | 100% | 90% | 49% | ~60% | Falls back gracefully |

## Not Worth Pursuing

1. **General structural priors** - Cause more false positives than they help
2. **True random oracles** - By definition unlearnable
3. **50+ pattern random systems** - Too much hypothesis collision

---

## Active Learning Results

### Key Finding: 16 Queries → 100% Accuracy

Active learning dramatically accelerates rule discovery:

| Mode | Observations/Queries | Final Accuracy |
|------|---------------------|----------------|
| **Active** | **16 queries** | **100%** |
| Passive | 500 observations | 77% |
| Passive | 1000 observations | 78% |

### How Active Learning Works

Instead of passively observing random states, the learner **queries** for specific patterns:

```python
# For each candidate line (0,1,2), (3,4,5), etc:
result = oracle.query_pattern(positions=(0,1,2), value='1')
# Returns all boards where X is in top row

# Check: Is it always winX?
if all(label in ('winX', 'winO') for _, label in result):
    confirm_rule: (0,1,2) + X → winX
```

**Total queries needed**: 16 (8 patterns × 2 players)

### Transition Prediction

Learner can predict game outcomes from state + action:

```
board + action → next_state

Terminal mappings:
- winX  → 111111111 (X fills board)
- winO  → 222222222 (O fills board)  
- draw  → 000000000 (reset)
- error → 000000000 (reset)
```

**Accuracy**: 100% on valid game sequences

### Implementation

```python
# Active learning
learner.active_learn(oracle)  # 16 queries

# Passive learning
while True:
    board, label = oracle.random_board()
    learner.observe(board, label)

# Prediction
label = learner.predict(board)

# Transition prediction
next_type = learner.predict_transition(board, action)  # winX/winO/reset/continue
```

### Summary

| Capability | Status | Performance |
|------------|--------|-------------|
| Passive learning | ✓ | ~500 obs for 75%+ |
| Active learning | ✓ | 16 queries for 100% |
| Meta-learning | ✓ | 100% if game known |
| Transition prediction | ✓ | 100% on valid games |
| Negative rules | ✓ | Prunes hypothesis space |
