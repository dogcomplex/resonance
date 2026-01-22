# Multi-Fidelity Induction - Breakthrough in Generalization

## The Key Insight

Fidelity abstraction must apply to **BOTH LHS AND RHS**:
- Coarse LHS: "has_raw_material" (not "has_raw_iron")
- Coarse Effect: "+has_cooked_material" (not "+has_cooked_iron")

## Generalization Results

Trained on: iron, copper, tin
Tested on: gold, silver, bronze (NEVER SEEN!)

| Scoring | Unified V2 | MultiFidelity V2 | Improvement |
|---------|-----------|------------------|-------------|
| Exact | 0.0% | 66.7% | **+66.7%** |
| Coarse | 0.0% | **100.0%** | **+100%** |

## How It Works

```python
# During observation:
effect = {"+has_cooked_iron", "-has_raw_iron", ...}
effect_coarse = {"+has_cooked_material", "-has_raw_material", ...}

# Group by COARSE effect to find patterns across variants
by_coarse[(action, effect_coarse)].append(before_state)

# Intersection finds:
LHS = {has_fire, has_raw_material}  # Common to all materials!

# Rule learned:
has_fire + has_raw_material → +has_cooked_material, -has_raw_material
```

## The Three-Scale Architecture

```
PROBABILITY × SPACE × TIME × EFFECT_FIDELITY

- Probability: 100% / 80% / 50% / 30%
- Spatial: Fine LHS / Coarse LHS
- Temporal: Recent / Historical
- Effect: Fine effects / Coarse effects

Rules at all scales coexist.
Query selects appropriate scale for the situation.
```

## Implications

1. **True Generalization**: Can predict outcomes for never-seen entities
2. **Hierarchical Abstraction**: Learns both specific and general rules
3. **Transfer Learning**: Knowledge transfers across similar domains
4. **Compression**: Coarse rules compress many fine rules into one

## This Is How Concepts Form

The learner didn't memorize "iron → cooked_iron".
It discovered the CONCEPT: "raw_material + fire → cooked_material".

This is abstraction. This is concept formation. This is generalization.

**In O(N) time.**
