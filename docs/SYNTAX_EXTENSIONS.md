# LHS => RHS Syntax Extensions

## Everything Fits!

All proposed improvements can be expressed in the existing `LHS => RHS` syntax with minimal extensions.

## Current Syntax (Already Have)

```
# Basic rule
p0_1  p1_1  p2_1  =>  winX

# Negation on LHS (equivalence class)
!p0_0  =>  ...    # means p0_1 OR p0_2

# Priority ordering
PRIORITY3  p0_1  p1_1  p2_1  =>  winX

# Quantities
gold3  =>  sword

# Probabilities
attack  =>  hit%70  miss%30

# Catalysts (present on both sides)
p0_1  p1_1  p2_1  =>  p0_1  p1_1  p2_1  winX
```

## New Extensions

### 1. Hierarchical Patterns (Macros)

```
# Definition
lines := (0,1,2)|(3,4,5)|(6,7,8)|(0,3,6)|(1,4,7)|(2,5,8)|(0,4,8)|(2,4,6)

# Usage - expands to 8 rules!
lines_1  =>  winX

# Expansion:
#   p0_1  p1_1  p2_1  =>  winX
#   p3_1  p4_1  p5_1  =>  winX
#   ... (6 more)
```

This is just **macro expansion** - no new inference power!

### 2. Negation on RHS (Negative Rules)

```
# Positive: pattern P implies label L
p0_1  p1_1  p2_1  =>  winX

# Negative: pattern P implies NOT label L
p0_1  p1_1  p3_1  =>  !winX    # L-shape doesn't win
```

Use: Prune hypotheses during learning

### 3. Constraints (Mutual Exclusion)

```
# If both present, emit error
winX  winO  =>  error

# Or equivalently with negation:
winX  =>  !winO
```

## Complete TicTacToe in Extended Syntax

```
# === DEFINITIONS ===
lines := (0,1,2)|(3,4,5)|(6,7,8)|(0,3,6)|(1,4,7)|(2,5,8)|(0,4,8)|(2,4,6)

# === ERROR DETECTION (PRIORITY4) ===
PRIORITY4  !valid_parity  =>  error
PRIORITY4  lines_1  lines_2  =>  error   # Both win

# === WIN DETECTION ===
PRIORITY3  lines_1  =>  lines_1  winX
PRIORITY2  lines_2  =>  lines_2  winO

# === DRAW ===
PRIORITY  !p0_0  !p1_0  !p2_0  !p3_0  !p4_0  !p5_0  !p6_0  !p7_0  !p8_0  =>  draw

# === DEFAULT ===
default  =>  ok
```

**Total: 6 rules (compresses to ~20 after macro expansion)**

## Transfer Learning vs Structural Priors

### Why Structural Priors FAILED:

```
Prior: "rows are likely wins"
       ↓
Prediction: winX (when row has X)
       ↓
WRONG! Row might not be full yet
```

The prior is **too vague** - it doesn't say WHEN the pattern triggers.

### Why Transfer Learning WORKS:

```
Rule from TicTacToe: "p0_1 p1_1 p2_1 => winX"
                      ↓
Tests against observations
                      ↓
Confirm or reject immediately
```

The rule is **fully specified** - it says exactly what pattern triggers what label.

**Key Difference:**
- Structural prior: "rows might matter" (vague hypothesis class)
- Transfer learning: "THIS pattern => THIS label" (testable hypothesis)

## Negative Learning Algorithm

```python
# Track pattern-label co-occurrences
pattern_seen[P] = count
pattern_label[P][L] = count

# Positive rule: P => L
if pattern_label[P][L] / pattern_seen[P] >= 0.95:
    add_rule(P => L)

# Negative rule: P => !L
if pattern_label[P][L] == 0 and pattern_seen[P] >= 10:
    add_rule(P => !L)  # Never seen L with P
```

Use negative rules to **prune hypothesis space** - if we know `P => !L`, immediately reject any hypothesis `P => L`.

## Summary

| Extension | Syntax | Power |
|-----------|--------|-------|
| Hierarchical | `lines := ... ; lines_1 => winX` | Macro expansion only |
| Negation RHS | `P => !L` | Prune hypotheses |
| Constraints | `A B => error` | Mutual exclusion |

All fit cleanly in `LHS => RHS` with no new inference mechanisms!
