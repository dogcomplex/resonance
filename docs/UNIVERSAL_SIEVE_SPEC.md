# Universal Sieve Specification

## Core Insight

The same algorithm applied to three dimensions:

```
INPUT:  Set of observations
SPLIT:  Divide into N buckets
SIEVE:  Find patterns via intersection within buckets  
COUNT:  Score by buckets where pattern held
OUTPUT: Rules scored on the dimension
```

## Three Dimensions

| Dimension | Split By | Intersect On | Score Meaning |
|-----------|----------|--------------|---------------|
| **Probability** | Random sampling | (LHS, effect) pairs | How often rule holds |
| **Fidelity** | Token groupings | Effect patterns (ignoring specific tokens) | How abstract/general |
| **Invariance** | Time windows | Rules across windows | How stable over time |

## Bucket Sizing

Different dimensions may benefit from different bucket distributions:

### Time Buckets (Invariance)
```
Logarithmic: [1, 2, 4, 7, 12, 20, 33, 55, 90, 148]
           Recent ←————————————————————→ Ancient

Rationale: Recent observations matter more, but
           patterns that persist over long time are highly informative.
           Matches biological memory (recency + significance).
```

### Fidelity Buckets  
```
Start small, grow carefully:
[2, 3, 5, 8, 13, ...] tokens that can merge

Rationale: Merging 2 tokens is easy to verify.
           Merging many tokens risks over-generalization.
           Fibonacci-like growth balances exploration.
```

### Probability Buckets
```
Uniform: Equal-sized random samples

Rationale: We want unbiased estimates.
           Each bucket is an independent trial.
```

## Emergent Fidelity (No Domain Knowledge)

Tokens are **interchangeable** when:
- They appear in SOME but not ALL positives for a rule
- Swapping them doesn't change the rule's validity

```python
# Tokens in ALL positives = core LHS (required)
# Tokens in SOME positives = variable (can be abstracted)

core_lhs = {t for t in all_tokens if count[t] == len(positives)}
variable = {t for t in all_tokens if 0 < count[t] < len(positives)}

fidelity = len(variable) / (len(core_lhs) + len(variable))
```

This is **emergent abstraction** - the algorithm discovers which tokens
can be grouped without any domain knowledge.

## Recursive Application

Rules from one dimension become inputs to another:

```
Level 0: Raw observations
Level 1: Probability rules (from raw)
Level 2: Fidelity rules (treating L1 rules as observations)  
Level 3: Invariance rules (treating L2 rules as observations)
...
```

The depth question: How many levels before diminishing returns?

Hypothesis: Related to e or log(N) - the same constants that appear
in optimal information compression.

## Benchmark Results

| Environment | V9 | Unified | Universal Sieve |
|-------------|-----|---------|-----------------|
| Empty-8x8 | 100% | 100% | 100% |
| DoorKey-5x5 | 96.6% | 96.5% | 96.5% |
| FourRooms | 97.9% | 97.9% | 97.9% |
| Dynamic-5x5 | 90.4% | 89.8% | 89.8% |

**Universal Sieve maintains accuracy while adding fidelity/invariance dimensions.**

## Future Directions

1. **Optimal bucket sizing**: Test Fibonacci vs log vs exponential
2. **Recursive depth**: Test 2, 3, 4+ levels of dimension stacking
3. **Cross-dimension interaction**: Do high-fidelity rules have different invariance?
4. **Pixel domains**: Can fidelity sieve discover "objects" from raw pixels?
