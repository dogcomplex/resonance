# Resonance Crystal - Unified Sieve

## The Algorithm

```
1. OBSERVE
   - Collect (state, delta, action, effect) tuples
   - Delta = changes since previous frame (velocity/phase)
   - Build wave signatures: token → {effect: count}

2. RESONATE
   - Tokens with similar wave signatures merge
   - This IS fidelity - emerges from interference, not a separate sieve
   - threshold: coherence=0.95

3. CRYSTALLIZE
   - Generate candidate rules: LHS → effect
   - Full intersection + partial intersections
   - Each starts with temperature (0.0 or 0.3)

4. ANNEAL
   - For each fold of observations:
     - Score each rule (F1)
     - Temperature = temperature * 0.5 + (1 - F1) * 0.5
   - THIS IS THE ONLY SIEVE - checking persistence across reality samples

5. SURVIVE
   - Rules with temperature < 0.5 survive
   - Most specific (largest LHS) matches first
```

## Results

| Configuration | F1 | Rules |
|--------------|-----|-------|
| CrystalSieve baseline | 69.8% | ~99 |
| ResonanceCrystal (no delta) | **69.8%** | ~99 |
| ResonanceCrystal (with delta) | 68.8% | ~143 |

**Key finding:** Delta/velocity tokens don't help - they add specificity without improving accuracy (both at 40% when firing).

## The Insight

> There's only ONE sieve operation - checking persistence across samples of reality.
> Fidelity, probability, temporal, universal - all collapse into this.

The annealing IS the sieve. Temperature is inverse resonance. Patterns that persist (predict correctly across folds) cool down and survive.

## Why Delta Doesn't Help

1. Delta rules are more specific (8.2 avg LHS vs 6.0)
2. Same accuracy when firing (40% for both)
3. The ball velocity isn't captured well after token abstraction

The phase-shift idea is elegant but the representation loses the directional information after abstraction. Would need directional-aware token merging.

## Code

`resonance_crystal.py` - Clean unified implementation
- `include_delta=False` matches CrystalSieve exactly
- `include_delta=True` adds D+/D- tokens (doesn't help yet)
