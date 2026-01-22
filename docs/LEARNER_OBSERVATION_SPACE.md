# What The Learner Actually Sees

## Overview

The learner receives **completely anonymized observations**. It has:
- No variable names (just V0, V1, V2...)
- No value meanings (just V0_A, V0_B... for categorical, raw numbers for numeric)
- No action names (just A0, A1, A2...)
- No hints about structure or relationships

## Full Variable Space (32 variables in Pokemon-lite)

| Anon ID | Real Name (hidden) | Type | Example Values |
|---------|-------------------|------|----------------|
| V0 | screen_mode | categorical | V0_A, V0_B, V0_C |
| V1 | menu_open | boolean | 0, 1 |
| V2 | menu_cursor | numeric | 0, 1, 2, 3 |
| V3 | battle_cursor | numeric | 0, 1, 2, 3 |
| V4 | text_box_active | boolean | 0, 1 |
| V5 | player_pokemon_name | categorical | V5_A, V5_B, V5_C... |
| V6 | player_pokemon_type | categorical | V6_A, V6_B, V6_C... |
| V7 | player_pokemon_level | numeric | 5-100 |
| V8 | player_pokemon_hp | numeric | 0-100 |
| V9 | player_pokemon_max_hp | numeric | 100 |
| V10 | player_pokemon_status | categorical | V10_A, V10_B... |
| V11 | player_pokemon_atk | numeric | 40-100 |
| V12 | player_pokemon_def | numeric | 40-100 |
| V13 | player_pokemon_spd | numeric | 20-100 |
| V14 | position_x | numeric | 0-255 |
| V15 | position_y | numeric | 0-255 |
| V16 | current_map | numeric | 0-20 |
| V17 | in_battle | boolean | 0, 1 |
| V18 | turn_number | numeric | 0-50 |
| V19 | last_move | categorical | V19_A, V19_B... |
| V20 | last_damage_dealt | numeric | 0-200 |
| V21 | last_damage_taken | numeric | 0-200 |
| V22 | weather | categorical | V22_A, V22_B... |
| V23 | pokeballs_count | numeric | 0-99 |
| V24 | potions_count | numeric | 0-99 |
| V25 | money | numeric | 0-999999 |
| V26 | enemy_pokemon_name | categorical | V26_A, V26_B... |
| V27 | enemy_pokemon_type | categorical | V27_A, V27_B... |
| V28 | enemy_pokemon_level | numeric | 0-100 |
| V29 | enemy_pokemon_hp | numeric | 0-100 |
| V30 | enemy_pokemon_max_hp | numeric | 100 |
| V31 | enemy_pokemon_status | categorical | V31_A, V31_B... |

## Action Space (7 actions)

| Anon ID | Real Name (hidden) |
|---------|-------------------|
| A0 | up |
| A1 | down |
| A2 | left |
| A3 | right |
| A4 | a |
| A5 | b |
| A6 | start |

## Observation Format

Each observation is a transition:

```
{
  "before": {V0: V0_A, V1: 0, V2: 0, V3: 42, V4: 17, ...},
  "action": A4,
  "after":  {V0: V0_A, V1: 0, V2: 0, V3: 42, V4: 17, ...}
}
```

## What The Learner Must Discover

### 1. Variable Relationships
- V0 (mode) changes when action=A6 (start) - switches between V0_A/V0_B/V0_C
- V14, V15 (position) change when action=A0/A1/A2/A3 (directions)
- V29 (enemy_hp) decreases when V0=V0_C and action=A4 (battle mode + attack)

### 2. Hidden Patterns
The learner sees:
```
When V6=V6_B and V27=V27_D, V20 averages ~40
When V6=V6_A and V27=V27_E, V20 is always 0
```

It has NO IDEA this means:
```
When player_type=fire and enemy_type=grass, damage is high (super effective)
When player_type=electric and enemy_type=ground, damage is 0 (immune)
```

### 3. Evolved Groupings
Through evolution, the learner discovers:
```
Group 0: {(V6_A, V27_E), (V6_B, V27_C), (V6_C, V27_D), ...} → avg V20 = 8
Group 1: {(V6_A, V27_A), (V6_B, V27_B), (V6_C, V27_C), ...} → avg V20 = 20
Group 2: {(V6_A, V27_B), (V6_B, V27_D), (V6_C, V27_B), ...} → avg V20 = 40
```

These correspond to resisted/neutral/super-effective without knowing those concepts!

## Key Constraints

1. **Pattern size limit**: Only 3-5 variables can interact in a single rule
2. **Min support**: Need 5+ observations of a pattern to form a rule
3. **Probabilistic**: Predictions are ranges, not exact values
4. **No extrapolation**: Cannot predict unseen combinations (e.g., new type V6_F)

## Example Learned Rules (anonymized)

```
{V0=V0_C, action=A4, V3=0} → V29 decreases, V20 increases
  (In battle, pressing A with cursor=0 causes damage)

{V6=V6_B, V27=V27_D} → V20 in range [30-50]
  (Fire vs grass does high damage)

{action=A6, V0=V0_A} → V0 becomes V0_B
  (Pressing start in overworld opens menu)
```

The learner builds these rules entirely from correlation, never understanding semantics.
