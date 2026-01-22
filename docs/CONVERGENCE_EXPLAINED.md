# Why Accuracy Doesn't Reach 100% (and how to fix it)

## Root Causes Identified

### 1. Incomplete State Representation (MiniGrid)
**Problem**: Original tokenization missed agent position/direction
- Same relative view (front_t2, left_t2, right_t2) occurs at different grid positions
- World is deterministic but our tokens aren't

**Fix**: Add `pos_X_Y` and `dir_D` tokens
- Result: **98.6% F1** (up from 70%)

### 2. Wrong Prediction Logic (TicTacToe)
**Problem**: Using `is_deterministic` flag for prediction
- Rules with 2% probability marked "deterministic" (< 5%) → false positives
- Rules with 84% probability NOT marked deterministic → false negatives

**Fix**: Use probability threshold (70%) instead of deterministic flag
- Result: **95% F1** (up from 61%)

### 3. State Space Coverage
**Problem**: We haven't seen all possible states
- TicTacToe: ~2500 state-action pairs seen, but many more possible
- Unseen states can't be predicted exactly

**Solution**: Hybrid approach
1. **Exact match**: If we've seen this exact state → 100% accurate
2. **Hierarchical fallback**: If unseen → use most specific matching pattern

Result: **96.2% F1**

## The Full Picture

```
                        PREDICTION ACCURACY
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    SEEN STATES           UNSEEN STATES        INCOMPLETE
    (exact match)         (generalization)      OBSERVATION
         │                     │                     │
       100%                  ~90%                    0%
    accurate             (hierarchical)         (impossible)
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                               │
                     OVERALL: ~96% F1
```

## Key Insight: The Hierarchy Serves Generalization

The hierarchical rules (size 1 → size N) aren't redundant - they enable prediction on **unseen states**:

| Situation | Method | Accuracy |
|-----------|--------|----------|
| Exact state seen | Direct lookup | 100% |
| Similar state seen | Most specific matching pattern | ~90% |
| Never seen anything like it | Most general matching pattern | ~70% |
| No matching pattern | Can't predict | 0% |

## Final Results

| Game | Before Fix | After Fix | Method |
|------|------------|-----------|--------|
| TicTacToe | 61% F1 | **96% F1** | Hybrid exact + hierarchical |
| MiniGrid Empty | 70% F1 | **98% F1** | Full state tokenization |
| MiniGrid DoorKey | 65% F1 | **~95% F1** | Full state tokenization |

## Summary

1. **Make state representation complete** - capture all relevant features
2. **Use probability threshold for prediction** - not the deterministic flag
3. **Prefer exact matches when available** - 100% accurate
4. **Fall back to most specific pattern** - enables generalization

The world IS deterministic. If we observe enough state and predict correctly, we converge to 100%.
