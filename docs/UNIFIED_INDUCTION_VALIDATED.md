# Unified Rule Induction - VALIDATED

## Test Results

### Isolated Rule Test (no interactions)

| Rule | True Probability | Found | Error |
|------|-----------------|-------|-------|
| A ∧ B → +X, -A | 100% | 100.0% | 0.0% |
| C ∧ D → +Y, -C | 100% | 100.0% | 0.0% |
| E ∧ F → +Z, -E | 80% | 82.4% | 2.4% |
| G ∧ H → +W, -G | 60% | 57.4% | 2.6% |
| I ∧ J ∧ K → +V, -I | 40% | 41.0% | 1.0% |

**All rules recovered. All probabilities within ~3% of ground truth!**

## The Algorithm (Validated)

```python
def unified_induction(observations):
    by_effect = group_by_effect(observations)
    
    for effect, positives in by_effect.items():
        # Stage 1: Candidate generation via sampling
        for _ in range(NUM_SAMPLES):
            subset = random_sample(positives, ratio=0.7)
            lhs = intersection(subset)  # AND operation
            candidates.add(lhs)
        
        # Stage 2: Probability counting
        for lhs in candidates:
            lhs_present = [obs for obs in ALL_OBS if lhs <= obs.before]
            hits = sum(1 for obs in lhs_present if obs.effect == effect)
            prob = hits / len(lhs_present)
```

## Complexity
- O(num_samples × N) for candidate generation
- O(num_candidates × N) for probability counting
- **Total: O(N)** with bounded samples

## Key Properties

1. **Deterministic rules** found at exactly 100%
2. **Probabilistic rules** found with ~2-3% error
3. **Multi-token LHS** (3 tokens) discovered correctly
4. **No false positives** - only 5 rules found, all correct

## What's Needed for Complex Domains (e.g., Farm)

The Farm game test showed challenges:
- Rule interactions create compound effects
- Large states make intersection sparse
- Need to handle rule decomposition

Solutions to explore:
1. **Effect decomposition** - split compound effects
2. **Incremental refinement** - start simple, add conditions
3. **Negative evidence** - use no-change observations

## Conclusion

The algorithm works exactly as designed for isolated rules.
For interacting rules, additional machinery needed.

**The core insight is validated: intersection + counting = O(N) induction!**

## The Full Picture

```
           OBSERVATIONS
                │
                ▼
    ┌───────────────────────┐
    │  CANDIDATE GENERATION │  O(samples × N)
    │  (intersection sieve) │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  PROBABILITY COUNTING │  O(candidates × N)
    │  P(effect | LHS)      │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  BUCKETING            │
    │  100% / 80%+ / 50%+   │
    └───────────────────────┘
                │
                ▼
         DISCOVERED RULES
```

**This is a discrete analog to gradient descent.**
- Intersection = finding common patterns (like weight averaging)
- Counting = computing loss (just measuring, not optimizing)
- Bucketing = discretizing confidence (like quantization)

No backprop. No optimization. Just set operations and counting.
