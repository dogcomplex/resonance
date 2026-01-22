# Meta-Rule Learner V2 - Engine-Native Design

## Your Engine Semantics

| Token State | Meaning |
|-------------|---------|
| -1 | Explicitly absent (negation/inhibitor) |
| 0 | Unknown / no information |
| 1+ | Present (quantity if > 1) |

- Priority via `❗N` in LHS (higher N fires first)
- To change priority: delete old rule, add new with different `❗`
- Negation check: token must be `-1`, not just `0`

## The Core Insight

**Learned rules ARE domain rules.** When you observe a transition, you create rules that live in the same rule base:

```
Observation: {A, B} + action_2 → {A, C}
Effects: {+C, -B}

Created rule:
  LHS: {A:1, B:1, __A2:1, __Q:1, ❗130}
  RHS: {+__R+C, +__R-B, -__Q}
```

Prediction = add `__Q` and `__A{n}` to state, fire rules, read `__R*` tokens.

## Minimal Rule Set

### Static Rules (2N + 2)

For each domain token T:

```
# T disappeared (before=1, after=-1)
LHS: {__O:1, __B_T:1, __A_T:-1, ❗1100}
RHS: {+__R-T}

# T appeared (before=-1, after=1)
LHS: {__O:1, __B_T:-1, __A_T:1, ❗1100}
RHS: {+__R+T}
```

Plus cleanup:
```
LHS: {__O:1, ❗900} → RHS: {-__O}
LHS: {__Q:1, ❗900} → RHS: {-__Q}
```

### Dynamic Rules (per observation)

For `observe(before, action, after)` with computed effects E:

**Exact match:**
```
LHS: {all tokens in before, __A{action}:1, __Q:1, ❗(100 + |before|*10 + count)}
RHS: {+__R{e} for e in E, -__Q}
```

**Pattern rules** (one per token in before):
```
LHS: {T:1, __A{action}:1, __Q:1, ❗(100 + 10 + count)}
RHS: {+__R{e} for e in E, -__Q}
```

## Execution Trace

### OBSERVE ({A, B}, action=2, {A, C})

1. **CPU sets markers:**
   ```
   __B_A:1, __B_B:1, __B_C:-1, __B_D:-1, ...  (before state)
   __A_A:1, __A_B:-1, __A_C:1, __A_D:-1, ...  (after state)
   __O:1, __A2:1
   ```

2. **Effect rules fire (GPU, p=1100):**
   ```
   {__O:1, __B_B:1, __A_B:-1} → {+__R-B}
   {__O:1, __B_C:-1, __A_C:1} → {+__R+C}
   ```

3. **CPU reads `__R*` tokens, creates learned rules:**
   ```
   {A:1, B:1, __A2:1, __Q:1, ❗121} → {+__R+C, +__R-B, -__Q}
   {A:1, __A2:1, __Q:1, ❗111} → {+__R+C, +__R-B, -__Q}
   {B:1, __A2:1, __Q:1, ❗111} → {+__R+C, +__R-B, -__Q}
   ```

4. **Cleanup rule fires (GPU, p=900):**
   ```
   {__O:1} → {-__O}
   ```

### PREDICT ({A, B, X}, action=2)

1. **CPU sets state:**
   ```
   {A:1, B:1, X:1, __A2:1, __Q:1}
   ```

2. **Rules checked (highest `❗` first):**
   ```
   {A:1, B:1, __A2:1, __Q:1, ❗121} ⊆ state? YES!
   ```

3. **Rule fires:**
   ```
   → {+__R+C, +__R-B, -__Q}
   ```

4. **CPU reads results:** `effects = {+C, -B}` ✓

## CPU vs GPU Split

| Operation | GPU | CPU |
|-----------|-----|-----|
| Effect computation | ✓ (marker rules) | Set markers |
| Prediction | ✓ (learned rules fire) | Read results |
| Priority resolution | ✓ (`❗` in LHS) | - |
| Rule creation | - | O(|state|) per obs |
| Rule priority bump | - | Delete/add rule |

## Rule Count Estimate

| Component | Count |
|-----------|-------|
| Static (vocab=100) | 202 |
| Learned (1000 obs, avg state=5) | ~6000 |
| **Total** | **~6200** |

## Simplifications

1. **Skip static rules**: CPU computes effects directly (often faster than 2N markers)

2. **Skip pattern rules**: Only exact-match → 1 rule per observation (loses generalization)

3. **Merge duplicates**: Same (state, action, effects) → bump `❗` instead of duplicate

4. **Cap priority**: Stop incrementing at `❗200` (confident enough)

## Token Vocabulary

| Token | Purpose |
|-------|---------|
| `__Q` | Query trigger |
| `__O` | Observe trigger |
| `__A{n}` | Action n (one-hot) |
| `__B_{T}` | Before-marker for token T |
| `__A_{T}` | After-marker for token T |
| `__R+{T}` / `__R-{T}` | Result: T appeared/disappeared |
