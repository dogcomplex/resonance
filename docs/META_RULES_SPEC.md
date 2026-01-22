# Meta-Rule Learner Specification

## Executive Summary

The hierarchical learner can be expressed as **LHS → RHS rules** that run alongside domain rules on GPU.

**Key insight**: Learned rules ARE domain rules with `__` prefixed meta-tokens. Prediction is just rule-firing with a `__Q` trigger token.

## What the Learner Actually Does (Core Operations)

### 1. OBSERVE - Record a transition

```
Input:  before_state, action, after_state
Output: Updated memory (= new rules added)

Algorithm:
    effects = diff(before, after)         # CPU: O(|state|)
    CREATE rule: {before, __A:action, __Q} → {effects}
    For each token t in before:
        CREATE rule: {t, __A:action, __Q} → {effects}
```

### 2. PREDICT (Deduction) - Forward lookup

```
Input:  state, action
Output: predicted effects

Algorithm:
    Add __Q and __A:action to state
    Fire rules (highest priority match wins)  # GPU: native!
    Extract __E:* tokens from result
```

### 3. ABDUCE - Backward lookup

```
Input:  target_effect
Output: states/actions that produce it

Algorithm:
    Scan rules where target_effect ∈ RHS    # GPU: filter
    Return their LHS patterns
```

### 4. INDUCE_PATH - Chain search

```
Input:  start_state, end_state
Output: rule sequence connecting them

Algorithm:
    Repeated PREDICT until target reached   # CPU: loop control, GPU: per-step
```

## Token Namespaces

All meta-tokens use `__` prefix to isolate from domain:

| Token | Purpose |
|-------|---------|
| `__A:N` | Action N is being taken |
| `__Q` | Trigger prediction |
| `__O` | Trigger observation/learning |
| `__E:+X` / `__E:-X` | Effect: token X added/removed |
| `__C:H:N` | Rule H has count N |

## Learned Rules Format

When observing `state={A,B,C}, action=5, effects={+D,-B}`:

**Exact match rule** (high specificity):
```
priority=130:  {A, B, C, __A:5, __Q} → {+__E:+D, +__E:-B, -__Q}
```

**Pattern rules** (lower specificity):
```
priority=110:  {A, __A:5, __Q} → {+__E:+D, +__E:-B, -__Q}
priority=110:  {B, __A:5, __Q} → {+__E:+D, +__E:-B, -__Q}
priority=110:  {C, __A:5, __Q} → {+__E:+D, +__E:-B, -__Q}
```

**Priority formula**: `100 + |LHS|*10 + count`

Highest priority matching rule wins = most specific + most observed.

## Counter Meta-Rules

For tracking observation counts (thermometer encoding):

```
p=1000: {__O, __C:H:0} → {+__C:H:1, -__C:H:0}
p=1000: {__O, __C:H:1} → {+__C:H:2, -__C:H:1}
p=1000: {__O, __C:H:2} → {+__C:H:3, -__C:H:2}
... up to MAX_COUNT
```

## Effect Computation

**If engine supports negation (¬)**:
```
p=1200: {__O, __BEFORE:T, ¬__AFTER:T} → {+__E:-T}
p=1200: {__O, ¬__BEFORE:T, __AFTER:T} → {+__E:+T}
```

**If no negation**: CPU computes set diff (~100 ops)

## What Runs Where

| Operation | GPU | CPU |
|-----------|-----|-----|
| Prediction | ✓ (rule firing) | - |
| Count increment | ✓ (counter rules) | - |
| Argmax | ✓ (priority) | - |
| Effect computation | If ¬ available | O(|state|) |
| Rule creation | - | O(1) per rule |
| Path search | Per-step | Loop control |

## Rule Count Estimate

After 1000 observations:
- ~5000 learned rules
- ~100k counter rules (if explicit)
- Total: ~105k rules

Manageable with sparse representation on GPU.

## Minimal CPU Requirements

**Per observation**: ~100 ops (effect diff + rule creation check)

**Per prediction**: 0 ops (pure GPU rule-firing!)

**Per abduction**: 0 ops if RHS indexed, else O(|rules|) scan
