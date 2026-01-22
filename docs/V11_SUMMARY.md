# V11 Implementation Summary

## Key Changes from V9

| Feature | V9 | V11 |
|---------|----|----|
| Exact memory | In-memory dict | SQLite + cache |
| Trajectory | Manual | Auto-derived |
| Rule patterns | Full hierarchical | Single + 2-token |
| Event discovery | None | Auto-discovered |
| Scalability | ~100K states | 1M+ states |

## Experimental Results

### Simulated Pong (bucketed positions)

| Metric | V9 | V11 |
|--------|----|----|
| Precision | 60.0% | 53.8% |
| Recall | 44.0% | 48.2% |
| **F1** | **50.8%** | **50.8%** |
| Train time | 10.2s | 12.5s |

**Result**: Equivalent performance, V11 has better recall due to adaptive threshold.

### V11 Discoveries

**Trajectory tokens work**: Adding `ball_x_delta_pos/neg` nearly doubles F1 on bucketed states (from 38% to 73% in isolation test).

**Event discovery works**: Auto-discovered that:
- `ball_y_boundary` occurs when `ball_y_0`, `ball_y_1`
- `ball_x_boundary` occurs when `ball_x_0`, `ball_x_1`
- These ARE the wall-bounce rules!

## Architecture

```
State tokens
    │
    ▼
Trajectory tracker ──► Add delta tokens (ball_x_delta_pos, etc.)
    │
    ▼
SQLite/cache lookup ──► If found: return exact observation
    │
    ▼ (not found)
General rules ──► Single-token + 2-token (delta+position) patterns
    │
    ▼
Adaptive threshold ──► Exact: 50%, Rules: 15%
```

## When to Use V11 vs V9

**Use V11 when:**
- State space is large (>100K states)
- Need persistence across sessions
- Environment has dynamics (velocity, momentum)
- Want event discovery/explanation

**Use V9 when:**
- State space is small
- In-memory is sufficient
- Need maximum speed
- Simple spatial relationships

## Files

- `hierarchical_learner_v11.py` - Main implementation
- `hierarchical_learner_v9.py` - Previous version (still valid)

## Next Steps

1. Improve rule specificity (more 2-token and 3-token patterns)
2. Better event detection (velocity reversals, collisions)
3. Add slice-finder integration for pixel inputs
4. Qualitative rule extraction for human-readable explanations
