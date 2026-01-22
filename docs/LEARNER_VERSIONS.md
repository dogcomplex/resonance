# Hierarchical Learner Version History

## V1 (Original)
- Hierarchical probabilistic rules
- Optimistic crystallization
- Single-token to full-context patterns

## V2 (+Context Awareness)
- Separates position-dependent from position-independent effects
- Uses relative state for view/dir/carry predictions
- **Improvement**: +3.5% F1 on FourRooms

## V3 (+Prototype Fallback)
- Clusters states by behavioral tokens
- Falls back to prototype when exact match fails
- **Improvement**: +3.2% F1 on sparse training

## V4 (+Temporal Compression)
- Detects transitional/animation states
- Learns stable→stable transitions
- Configurable via `transition_markers`
- **Note**: Environment-dependent benefit

## V5 (+Derived Tokens) - FINAL
- Automatically derives comparison tokens
- E.g., `player_atk_52` + `enemy_def_43` → `derived_atk_vs_def_advantage`
- Configurable via `derive_comparisons` flag

## Usage
```python
from hierarchical_learner_v5 import HierarchicalLearner

learner = HierarchicalLearner(
    n_actions=7,
    position_prefixes=['pos_'],           # Tokens for position effects
    prototype_prefixes=['front_', 'dir_'], # Tokens for prototypes
    transition_markers=['animating'],      # Transitional state markers
    derive_comparisons=True                # Enable derived tokens
)

learner.observe(state, action, next_state)
predictions = learner.predict(state, action, prob_threshold=0.7)
```

## Test Results Summary
| Version | EmptyEnv | FourRooms | Notes |
|---------|----------|-----------|-------|
| V1 | ~95% | ~60% | Baseline |
| V2 | ~96% | ~63% | +Context |
| V3 | ~96% | ~64% | +Prototype |
| V4 | ~92% | ~65% | +Temporal |
| V5 | ~96% | ~66% | +Derived |
