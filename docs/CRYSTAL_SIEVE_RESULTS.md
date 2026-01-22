# Crystal Sieve: Rule Annealing for Generalization

## Key Discovery

**Anneal RULES, not CLASSES.**

The original approach tried to anneal token equivalence classes (which atoms bond with which). This didn't help because:
- Token equivalence is already well-determined by cosine similarity
- The baseline was already near-optimal for class discovery

The breakthrough: Anneal the RULES (crystal structure), not the atoms (token classes).

## Results

| Environment | WaveSieve (baseline) | CrystalSieve | Improvement |
|-------------|---------------------|--------------|-------------|
| Pong        | 39.2%               | **64.3%**    | +64% relative |
| Breakout    | 44.9%               | **64.1%**    | +43% relative |

## Algorithm

```
NUCLEATION → GROWTH → ANNEAL → CRYSTAL
```

### Phase 1: NUCLEATION
Build token equivalence classes via wave interference (cosine similarity > 0.95)

### Phase 2: GROWTH
Generate **multiple** candidate rules per (action, effect):
- Full intersection (most specific)
- Partial intersections (more general)

### Phase 3: ANNEAL
For each candidate rule, test on held-out data:
- Rules that predict well get "cold" (confident)
- Rules that fail get "hot" (rejected)

Only keep rules with temperature < 0.5

### Phase 4: CRYSTAL
Final rule set = validated rules that generalize

## Physical Analogy

| Metallurgy | Sieve Algorithm |
|------------|-----------------|
| Atoms | Tokens |
| Bonds | Equivalence classes |
| Crystal structure | Rules |
| Defects | Overfitting rules |
| Annealing | Validation filtering |
| Pure crystal | Generalizing rule set |

## Why It Works

The baseline creates ONE rule per (action, effect) via intersection.
This rule might overfit to training data.

CrystalSieve creates MULTIPLE candidate rules and selects the ones that:
1. Are consistent across different subsets of training data
2. Generalize to held-out validation data

This is essentially: **ENSEMBLE + VALIDATION**

The "annealing" metaphor is apt because:
- High temperature = speculative rules (low confidence)
- Low temperature = validated rules (high confidence)
- Cooling process = iterative validation
- Final crystal = pure, generalizing rule set

## Code

```python
from crystal_sieve_final import CrystalSieve

sieve = CrystalSieve(coherence=0.95)

for state, action, next_state in observations:
    sieve.observe(state, action, next_state)

sieve.build()

prediction = sieve.predict(new_state, action)
```
