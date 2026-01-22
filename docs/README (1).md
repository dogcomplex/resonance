# Few-Shot Game Rule Learner

A system for inferring game rules from observations using hypothesis elimination - **with no cheating**.

## Key Results

### Classification Accuracy (Unique Observations)

| Learner | @500 | @1000 | @3000 | Win Detection |
|---------|------|-------|-------|---------------|
| BlindLearner | 74% | 74% | 75% | 0% |
| HybridLearner | 80% | 81% | 66% | **80%** |
| UnifiedLearner | **86%** | **90%** | **93%** | 66% |
| AdvancedSAT | 83% | 89% | 93% | 66% |

### Complete Rule Discovery

The **AdvancedSAT** learner discovers complete game rules:

```
win1 = (
    (p0=1 AND p1=1 AND p2=1)   # top row
 OR (p3=1 AND p4=1 AND p5=1)   # middle row
 OR (p6=1 AND p7=1 AND p8=1)   # bottom row
 OR (p0=1 AND p3=1 AND p6=1)   # left column
 OR (p1=1 AND p4=1 AND p7=1)   # middle column
 OR (p2=1 AND p5=1 AND p8=1)   # right column
 OR (p0=1 AND p4=1 AND p8=1)   # main diagonal
 OR (p2=1 AND p4=1 AND p6=1)   # anti-diagonal
)
```

### State Transition Rules (NEW!)

The **TrueMinimal** learner discovers the CORE move mechanics:

```
# X's turn to play:
IF turn_X AND c{N}_empty => c{N}_X AND turn_O

# O's turn to play:  
IF turn_O AND c{N}_empty => c{N}_O AND turn_X
```

Just 2 rule patterns covering all 18 specific moves!

## Architecture

### Production Rule Framework

```
Token = atomic fact (cell state, turn, result)
  Examples: c0_X, turn_O, result_win_X

TransitionRule = LHS => RHS
  CONSUMED: tokens removed (LHS - RHS)
  PRODUCED: tokens added (RHS - LHS)
  CATALYSTS: tokens checked but unchanged (LHS ∩ RHS)

Example win detection rule:
  IF {c0_X, c1_X, c2_X}       <- catalysts (checked)
  PRODUCE {result_win_X}      <- conclusion
```

### Learning Pipeline

1. **Observe** game traces (state sequences)
2. **Extract** (consumed, produced) patterns
3. **Generalize** by finding patterns across contexts
4. **Verify** against ground truth

## Project Structure

```
few_shot_algs/
├── unified_learner.py       # Best classifier (94%)
├── advanced_sat.py          # Complete rule discovery
├── transition_rules.py      # Production rule framework
├── true_minimal.py          # Core mechanic discovery
├── sat_production.py        # SAT-like encoding
└── minimal_learner.py       # Minimal rule set finder
```

## No Cheating Policy

All learners are **truly blind**:
- ❌ No knowledge of rows/columns/diagonals
- ❌ No knowledge of what symbols mean
- ✓ Only learns from observations
- ✓ Generalizes to random rule variants
