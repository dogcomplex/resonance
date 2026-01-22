# Rule Learning Trace: Step-by-Step Walkthrough

## Overview

This document traces exactly how observations become rules and how those rules fire during prediction.

## Phase 1: Observation → Pattern Storage

### Example Observation

```
Battle: FIRE (ATK=80) vs GRASS (DEF=50)
Move: fire (strong)
Actual damage: 144 HP → bucket: massive
```

### Step 1: Abstraction (Raw → Tokens)

Raw numeric state gets converted to abstract tokens:

```
Raw:                          Abstract Token:
─────────────────────────────────────────────
player_type = fire        →   player.type.fire
enemy_type = grass        →   enemy.type.grass
move_type = fire          →   move.type.fire
move_power = 90           →   move.power.strong
player_hp = 100%          →   player.hp.full
enemy_hp = 100%           →   enemy.hp.full

DERIVED (computed, not observed):
fire vs grass = 2.0x      →   matchup.super_effective
fire == player_type       →   move.stab
```

Final token set (8 tokens):
```
{enemy.hp.full, enemy.type.grass, matchup.super_effective,
 move.power.strong, move.stab, move.type.fire,
 player.hp.full, player.type.fire}
```

### Step 2: Pattern Enumeration

For each subset of tokens (size 1, 2, 3), record co-occurrence with effect:

```
Size 1 patterns:
  {matchup.super_effective}     + damage.massive → count: 1/1
  {move.power.strong}           + damage.massive → count: 1/1
  {move.stab}                   + damage.massive → count: 1/1
  {player.type.fire}            + damage.massive → count: 1/1
  ...

Size 2 patterns:
  {matchup.super_effective, move.power.strong} + damage.massive → count: 1/1
  {matchup.super_effective, move.stab}         + damage.massive → count: 1/1
  {move.power.strong, move.stab}               + damage.massive → count: 1/1
  ...

Size 3 patterns:
  {matchup.super_effective, move.power.strong, move.stab} + damage.massive → count: 1/1
  ...
```

### After Multiple Observations

After 10 training battles, pattern counts accumulate:

```
Pattern                                          | Effect         | Count
─────────────────────────────────────────────────|────────────────|──────
{matchup.super_effective, move.power.strong}     | damage.massive | 3/3 (100%)
{matchup.resisted, move.power.strong}            | damage.medium  | 2/2 (100%)
{matchup.neutral}                                | damage.low     | 2/2 (100%)
{matchup.immune}                                 | damage.none    | 1/1 (100%)
{matchup.super_effective}                        | damage.massive | 3/5 (60%)
{matchup.super_effective}                        | damage.high    | 2/5 (40%)
```

## Phase 2: Rule Extraction

Rules are extracted where:
- **Support** ≥ minimum (e.g., 2 observations)
- **Confidence** ≥ threshold (e.g., 70%)

```
EXTRACTED RULES:
────────────────────────────────────────────────────────────────────

damage.massive:
  {matchup.super_effective, move.power.strong, move.stab} → 100% (n=3)
  {matchup.super_effective, move.power.strong}            → 100% (n=3)

damage.medium:
  {matchup.resisted, move.power.strong}                   → 100% (n=2)
  {matchup.resisted}                                      → 100% (n=2)

damage.low:
  {matchup.neutral, move.power.medium}                    → 100% (n=2)
  {matchup.neutral}                                       → 100% (n=2)

damage.none:
  {matchup.immune}                                        → 100% (n=1)
```

## Phase 3: Prediction (Rule Firing)

### New Battle (Never Seen Before!)

```
Battle: ICE vs GRASS
Move: ice (strong, STAB)
```

### Step 1: Abstract the New State

```
Tokens:
  • player.type.ice        ← NEVER SEEN IN TRAINING
  • enemy.type.grass
  • move.type.ice          ← NEVER SEEN IN TRAINING
  • move.power.strong
  • move.stab
  • matchup.super_effective ← DERIVED (ice vs grass = 2.0x)
```

### Step 2: Find Matching Rules

A rule fires if its pattern is a **subset** of the input tokens:

```
CHECK: {matchup.super_effective, move.power.strong, move.stab}
       ⊆ {player.type.ice, enemy.type.grass, move.type.ice,
          move.power.strong, move.stab, matchup.super_effective}
       
       matchup.super_effective ✓ present
       move.power.strong       ✓ present
       move.stab               ✓ present
       
       FIRES! → damage.massive (100%)
```

### Step 3: Rule Firing Trace

```
RULES THAT FIRE:
────────────────────────────────────────────────────────────────────

Pattern: {matchup.super_effective, move.power.strong, move.stab}
Effect:  damage.massive
Conf:    100%
Note:    ★ Pattern has NO type-specific tokens - GENERALIZES!

Pattern: {matchup.super_effective, move.power.strong}
Effect:  damage.massive
Conf:    100%
Note:    ★ Pattern has NO type-specific tokens - GENERALIZES!

Pattern: {matchup.super_effective}
Effect:  damage.massive
Conf:    60%
Note:    Lower confidence, but still fires
```

### Final Prediction

```
Best match: {matchup.super_effective, move.power.strong, move.stab}
Prediction: damage.massive (100% confidence)

Actual:     damage.massive ✓ CORRECT
```

## Why Generalization Works

### The Key Insight

The learner NEVER saw "ice" type in training, but predicts correctly because:

1. **Type effectiveness is DERIVED** before pattern matching
   ```
   (ice, grass) → lookup in type chart → matchup.super_effective
   ```

2. **Rules operate on ABSTRACT matchup**, not specific types
   ```
   {matchup.super_effective, move.power.strong} → damage.massive
   ```

3. **Any super-effective combo maps to the same rule**
   ```
   fire vs grass  → matchup.super_effective → damage.massive
   water vs fire  → matchup.super_effective → damage.massive
   ice vs grass   → matchup.super_effective → damage.massive ← NEW!
   ```

### Compression Ratio

```
WITHOUT abstraction:
  15 types × 15 types × 4 power levels = 900 specific rules needed

WITH abstraction:
  4 matchup classes × 4 power levels = 16 general rules needed

Compression: 56x fewer rules, same predictive power!
```

## Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: TYPE RULES                          │
│  {move.type.X, enemy.type.Y} → matchup.Z                        │
│                                                                  │
│  Learned from observations:                                      │
│    fire vs grass    → super_effective (100%, n=3)               │
│    water vs fire    → super_effective (100%, n=2)               │
│    electric vs ground → immune (100%, n=3)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ adds matchup.Z token
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 2: DAMAGE RULES                         │
│  {matchup.Z, move.power.W, ...} → damage.bucket                 │
│                                                                  │
│  Learned from observations:                                      │
│    super_effective + strong + stab → massive (100%, n=3)        │
│    resisted + strong → medium (100%, n=2)                       │
│    immune + anything → none (100%, n=3)                         │
└─────────────────────────────────────────────────────────────────┘
```

Both stages are pure production rules - no hardcoded logic!

## Probabilistic Reconstruction

### Abstract → Raw

```
Prediction: damage.massive

damage.massive bucket = [91, 200] HP

Sample: uniform(91, 200) → 145 HP

Actual: 144 HP
Error:  1 HP ✓
```

The bucket gives us a range; we sample within it for the raw value.
