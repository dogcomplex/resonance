# Resonance Crystal - Final Theory & Implementation

## The Single Sieve Principle

> **Does this pattern resonate when sampled from different angles of reality?**

This one operation subsumes all others:
- **Fidelity** → spatial angles
- **Probability** → density angles  
- **Temporal** → time angles
- **Universal** → parallel-reality angles
- **Phase** → wave timing angles

## The Algorithm

```
OBSERVE    → Accumulate (state, action, effect, timestep)
           → Build wave signatures: token → {effect: amplitude}
           
RESONATE   → Tokens with similar wave signatures merge
           → Interference = complex_cosine(wave1, wave2)
           → This IS fidelity (emerges from physics)
           
CRYSTALLIZE → Generate candidate rules via set intersection
            → Each rule starts with temperature
            
ANNEAL     → For each fold (sample of reality):
              - Score rule (does it predict correctly?)
              - High resonance → temperature drops
              - Low resonance → temperature rises
              
SURVIVE    → Rules below cold_threshold survive
           → Most specific matches first
```

## Physical Interpretation

If this maps to physics:
- All realities (parallel universes, times) are spatial geometries
- A resonant wave bounces through all of them
- Where it interferes constructively with itself → high probability
- These resonant overlaps become perceivable "objects"

**Temperature** = inverse resonance strength
- Cold = strong resonance = survives = becomes real
- Hot = weak resonance = decays to entropy = noise

## Dimensions Explored

| Dimension | What it tests | Physical analog |
|-----------|---------------|-----------------|
| Fidelity | Same pattern at different positions? | Spatial dimensions |
| Probability | Pattern consistent with different evidence? | Solid/gas state |
| Temporal | Pattern holds at t, t+1, t+2? | Time dimension |
| Universal | Pattern holds in parallel runs? | Many-worlds |
| Phase | Pattern holds with different timing? | Wave phase |

## Complex Waves

From poor_mans_quantum.txt:
- Real parts at even timesteps
- Imaginary parts at odd timesteps
- Gives complex amplitudes capturing phase

**Result on Pong:** No improvement because ball moves every frame
(no phase differentiation). Would help on periodic systems.

## Current Performance

| Configuration | F1 | Notes |
|--------------|-----|-------|
| ResonanceCrystal | **69.8%** | Matches original CrystalSieve |
| + Complex waves | 69.8% | Same (no phase structure in Pong) |
| + Delta tokens | 68.8% | Slightly worse (too specific) |

## The 70% Ceiling

The ceiling is **representation**, not the sieve:
- We predict paddle OR ball, not both
- Need separate channels for independent phenomena
- Or effect completion mechanism

## Next Steps

1. **Better test domain** - Need a problem where:
   - Fidelity matters (position abstraction)
   - Temporal matters (delayed effects)
   - Phase matters (periodic behavior)
   - Universal matters (stochastic outcomes)

2. **Multi-channel prediction** - Separate paddle/ball predictions

3. **Effect resonance** - Apply same sieve to effect combinations

## The Elegance

One operation. Everything else emerges:
- Token merging from wave interference
- Rule filtering from fold-based annealing
- Generalization from set intersection
- Specificity from sort order

**Resonance creates structure. Entropy destroys it. What survives is real.**
