# Hierarchical Learner V7 - Pure Specificity

## Key Changes from V6b

1. **Pure specificity-based rule selection**: Most specific matching rule wins, ties broken by observation count
2. **Returns probabilities**: `predict_probs()` returns `{effect: probability}` 
3. **Configurable threshold**: `predict(state, action, threshold=0.5)`
4. **Cleaner architecture**: Simpler code, same concepts

## API

```python
learner = HierarchicalLearner(n_actions=7, min_observations=3)

# Learn
learner.observe(state, action, next_state)

# Get probabilities for all effects
probs = learner.predict_probs(state, action)
# Returns: {'+cell_4_1': 0.95, '+done_True': 0.05, ...}

# Get binary predictions with threshold
effects = learner.predict(state, action, threshold=0.5)
# Returns: {'+cell_4_1', '+player_2', '-cell_4_0', ...}
```

## Benchmark Results

| Environment | V6b | V7 | Notes |
|-------------|-----|-----|-------|
| FourRooms | **83%** | 75% | V6b's prototype fallback helps |
| TicTacToe | 80% | **87%** | Pure specificity better for discrete states |
| Pure Chaos | 43% | 45% | Neither can learn chaos |

## Threshold Sensitivity (FourRooms)

| Threshold | F1 | Precision | Recall |
|-----------|-----|-----------|--------|
| 0.1 | 36% | 23% | 81% |
| 0.3 | 75% | 77% | 73% |
| **0.5** | **73%** | 91% | 61% |
| 0.7 | 69% | 99% | 53% |
| 0.9 | 67% | 100% | 51% |

## When to Use V7 vs V6b

- **V7**: Better for discrete state spaces (board games, turn-based)
- **V6b**: Better for continuous/position-heavy spaces (navigation)

## Philosophy

The threshold isn't "rejecting" low-probability rules - those rules are still stored and used. The threshold answers: **"Is this effect more likely to happen than not?"**

A rule with 30% probability is valid information. But for prediction purposes, we default to predicting effects that are *expected* to happen (>50%).

For different use cases:
- Planning: Use low threshold (10%) to consider all possibilities
- Execution: Use high threshold (70%) to act on confident predictions
