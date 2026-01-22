# Hierarchical Learner - Design Notes

## V9 → V10 Evolution

### Original Hypothesis
Blend exact matches with general rules based on confidence:
- Few observations → trust general rules more
- Many observations → trust exact match more

### What We Learned
**Blending hurts when exact data exists.** If we've seen a state even once:
- Deterministic: that single observation IS the truth
- Probabilistic: we're building a distribution, don't dilute it

### Final V10 Design
```python
def predict(state, action):
    if exact_match_exists:
        if single_outcome:
            return exact_outcome  # Deterministic
        else:
            return distribution_over_outcomes  # Probabilistic
    else:
        return general_rule_prediction
```

This is essentially V9 but with:
1. **Determinism tracking**: Know if environment is deterministic
2. **Adaptive threshold**: Slightly lower threshold for deterministic envs

## Handling Large State Spaces

### When to Care
Track: `new_states_per_observation = unique_states / total_observations`
- High (>0.5): State space is exploding
- Low (<0.1): Good coverage, can trust exact matches

### Strategies
1. **Memory management**: Stop storing exact obs, just update rules
2. **User warning**: "State space too large for exact matching"
3. **Focus on rules**: Rely more heavily on general patterns

### NOT a Blending Problem
Large state spaces don't need blending - they need:
- Better generalization (relative features)
- Smarter memory (LRU cache, importance sampling)
- Or acceptance that unseen states use general rules

## Handling State Aliasing

### Detection
If single state has multiple outcomes AND similar states are deterministic:
→ Likely aliasing, not true stochasticity

### Solutions
1. **Accept as probabilistic**: Works, loses precision
2. **Expand tokenization**: Add missing state features
3. **Add history**: Recent actions/states to context

### V10 Approach
Track `multi_outcome_states / total_states` ratio:
- Low (<10%): Environment is deterministic, aliasing is rare
- High (>30%): Either probabilistic or severe aliasing

## Summary of Key Insights

1. **Exact match is gold** - Never dilute it with general rules
2. **General rules are for unseen states only**
3. **Probabilistic ≠ needs blending** - Just track the distribution
4. **Large state spaces** - Need better generalization, not blending
5. **Aliasing** - Detect and warn, or expand representation

## File Reference

- `hierarchical_learner_v9.py` - Original: exact first, then general rules
- `hierarchical_learner_v10.py` - Final: same logic + determinism tracking
