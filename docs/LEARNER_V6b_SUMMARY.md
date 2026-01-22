# Hierarchical Learner V6b - Final Version

## Key Improvements Over V5

### 1. Delta-Based Position Fallback
- When exact position rules don't exist, predicts position **deltas** (e.g., move +1 in x)
- Applies delta to current position to get predicted next position
- **Result: +13.8% F1 on FourRooms** (71.1% → 84.9%)

### 2. Removed Hardcoded Domain Knowledge
- No hardcoded comparison pairs
- All numeric comparisons auto-discovered from token format (prefix_NUMBER)
- Configurable prefixes for all token types

### 3. Fixed Fallback Gaps
- Position effects now have prototype fallback
- Both relative and position predictions have fallback paths

## Benchmark Results

| Environment | F1 | Precision | Recall | Delta Fallbacks |
|-------------|-----|-----------|--------|-----------------|
| Empty-8x8 | 96.1% | 93.1% | 99.4% | 653 |
| Empty-16x16 | 95.0% | 94.5% | 95.5% | 633 |
| FourRooms | 82.9% | 86.7% | 79.4% | 750 |
| DoorKey-6x6 | 94.9% | 90.9% | 99.3% | 676 |
| **Average** | **92.2%** | | | |

## Architecture

```
observe(state, action, next_state)
    │
    ├─> Full state rules (position effects)
    ├─> Relative state rules (view/dir/carry effects)
    ├─> Relative prototypes (behavioral clustering)
    └─> Delta prototypes (position movement patterns)

predict(state, action)
    │
    ├─> Try relative rules → fallback to relative proto
    └─> Try position rules → fallback to delta proto
```

## Configuration

```python
learner = HierarchicalLearner(
    n_actions=7,
    position_prefixes=['pos_'],      # Tokens for position effects
    prototype_prefixes=['front_', 'dir_', 'carry_'],  # Behavioral tokens
    transition_markers=['animating'],  # Animation state markers
)
```

## Key Insight: Front-Only Tokenization

Left/right view tokens hurt performance because they change unpredictably
when turning. Using **front-only** tokenization improves generalization.
