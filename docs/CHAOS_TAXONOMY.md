# Chaos Taxonomy for Testing Rule Learning

## Four Types of Chaos

| Type | Rules | Learnable? | Expected F1 |
|------|-------|------------|-------------|
| 1. Seeded Deterministic | Fixed at init, stable | YES | ~90%+ on seen states |
| 2. True Chaos | Change every step | NO | ~45-50% |
| 3. Seeded Probabilistic | Fixed distributions, stable | YES | ~60-70% |
| 4. True Probabilistic | Shifting distributions | NO | ~55-60% |

## Benchmark Results

```
Variant                          F1     Precision   Recall   Learnable?
----------------------------------------------------------------------
Standard TicTacToe            87.5%       82.3%    93.5%    ✓ YES
1. Seeded Deterministic       74.0%       75.3%    72.8%    ✓ YES
3. Seeded Probabilistic       60.7%       66.0%    56.3%    ✓ YES
2. True Chaos                 49.3%       63.9%    40.1%    ✗ NO
4. True Probabilistic Chaos   58.2%       60.9%    55.7%    ✗ NO
```

## Key Insight: State Coverage

Seeded Deterministic Chaos reaches **90% F1 on seen states**, proving it IS fully learnable. The 74% overall F1 is due to 37% of test transitions being from unseen state-action pairs.

```
SEEN states:   F1=90.0% (perfect recall, some FP from general rules)
UNSEEN states: F1=56.2% (can't predict unseen deterministic transitions)
```

## Why This Matters

1. **Chaos detection works**: The learner correctly identifies unlearnable environments
2. **State coverage is key**: Even "random" rules are learnable if they're stable
3. **Seeded randomness = hidden structure**: Any stable rule system can be learned
4. **True chaos baseline**: ~45-50% F1 represents the "no signal" floor

## Implementation Notes

- Seeded variants use separate RNG for rule generation vs gameplay
- This ensures same rules throughout training and testing
- True chaos variants resample rules on every transition
