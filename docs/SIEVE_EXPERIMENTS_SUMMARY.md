# Sieve Experiments Summary

## Baseline: CrystalSieve at 69.8%

This is the high-water mark. All sieve additions either match or hurt performance.

## What We Tried

### 1. Temporal/Velocity Sieves
- Added V+/V- tokens for frame-to-frame changes
- Velocity rules were too specific (7.0 avg LHS vs 5.3)
- When they fired, same accuracy as position rules (43% vs 42%)
- **Result: No improvement**

### 2. Universal (Cross-Bucket) Sieve
- Score rules by consistency across random observation buckets
- Tried: temperature adjustment, promotion-only, sort-only
- All variants hurt performance (54-65%)
- **Result: Made things worse**

### 3. Cascade/Layered Sieves
- Multiple sieves in sequence: prob → fid → prob → univ → prob
- Different orderings tested
- Best cascade still worse than single Crystal pass
- **Result: No improvement**

### 4. Fidelity (Abstraction) Sieve
- Generate more abstract rules by removing tokens
- Created rules with smaller LHS (4-5 tokens vs 6-7)
- More general rules over-matched → worse accuracy
- **Result: Made things worse**

## Key Insights

### Why Extra Sieves Hurt

1. **Random bucketing adds noise** - scoring on random subsets gives noisy temperature updates

2. **The annealing IS the sieve** - Crystal's fold-based annealing already does cross-validation

3. **Specificity matters** - more specific rules (larger LHS) should match first; making rules more general hurts

4. **Don't mess with ordering** - sorting by universality put wrong rules first

### What Crystal Gets Right

1. **Conservative token merging** (coherence=0.95)
2. **Multiple candidate rules** (full + partial intersections)
3. **Fold-based annealing** (3 rounds, EMA temperature update)
4. **Specificity-first sorting** (larger LHS matches first)

## The 70% Ceiling

The bottleneck is **incomplete effect prediction**, not rule selection:
- We predict paddle OR ball movement, not both
- Main error type is "missing" (14) not "wrong" (3)

This isn't a sieve problem - it's a tokenization/representation problem.

## Files

- `crystal_plus_fixed.py` - Clean Crystal implementation with optional universal sieve
- `resonance_sieve.py` - Flexible sieve framework (experimental)
- `crystal_sieve_final.py` - Original best implementation

## Next Steps to Try

1. **Multi-channel prediction** - Separate paddle and ball into independent prediction problems
2. **Effect completion** - If predicting paddle, also predict likely ball
3. **Better tokenization** - Higher resolution, separate channels
4. **More training data** - 50 episodes may not be enough
