# Resonance Crystal - Theoretical Framework

## The Physics Interpretation

If correct, this says:
- **All alternative realities and all of time are spatial geometries being explored**
- The resonant wave bounces through them and back
- **Resonates with itself** to boost probabilities
- Creates generalized perceivable "objects" from resonant overlaps

## Dimensions of Reality Sampling

| Dimension | Physical Analog | What it tests |
|-----------|-----------------|---------------|
| **Fidelity** | Spatial dimensions | Does pattern hold at different positions? |
| **Probability** | Solid/gas state | Does pattern hold with different densities of evidence? |
| **Time** | Temporal dimension | Does pattern hold at different time offsets? |
| **Universal** | Many-worlds / unobservable time | Does pattern hold in parallel realities? |

## Missing Dimensions?

- **Scale** - Does pattern hold at different resolutions/granularities?
- **Energy** - Does pattern hold at different "temperatures" of the system?
- **Spin/Phase** - Does pattern hold with different phase offsets?

## Complex Wave Representation

From poor_mans_quantum.txt:
- Real parts at even timesteps
- Imaginary parts at odd timesteps
- This gives us complex amplitudes for each token

Current: `token → {effect: count}` (real amplitude only)

Proposed: `token → {effect: (real, imaginary)}` where:
- `real` = effect strength at even timesteps
- `imaginary` = effect strength at odd timesteps

This captures **phase information** - whether effects are in-phase or out-of-phase.

## Interference Revisited

Currently: `interference = cosine_similarity(wave1, wave2)`

With complex waves:
```
interference = |complex_dot(wave1, wave2)| / (|wave1| * |wave2|)

where complex_dot includes phase alignment
```

Two tokens that cause the SAME effects but at DIFFERENT phases would have low interference 
(out of phase = destructive) even though their magnitudes are similar.

## Boosting vs Canceling

- **Constructive interference**: waves in phase → boost → resonates → survives
- **Destructive interference**: waves out of phase → cancel → doesn't resonate → dies

The intersection operation (LHS generation) is a form of constructive interference -
only tokens that are consistently present across observations survive.

## The Single Sieve

All of this collapses into one operation:

> **Does this pattern resonate (persist + amplify) when sampled from different angles?**

The "angles" can be:
- Different spatial positions (fidelity)
- Different time offsets (temporal)  
- Different parallel realities (universal)
- Different phases (complex extension)

**Temperature** = inverse resonance strength
**Low temperature** = strong resonance = survives
**High temperature** = weak resonance = decays to entropy
