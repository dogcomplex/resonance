# Nested Deterministic Hierarchy

## The Discovery

YES! Deterministic rules form **nested layers** where smaller patterns subsume larger ones:

```
SIZE 1: {px} + a0 → +x0 (100%)           ← MOST GENERAL
   ↓ subsumes
SIZE 2: {e0, px} + a0 → +x0 (100%)       ← adds context (cell empty)
   ↓ subsumes  
SIZE 3: {e0, o3, px} + a0 → +x0 (100%)   ← adds more context
   ↓ subsumes
SIZE N: {...} + a0 → +x0 (100%)          ← MOST SPECIFIC
```

## Hierarchy Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FULL CONTEXT RULES                              │
│                   (Size 10+, 128 rules)                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │               CONTEXTUAL RULES                                │  │
│  │              (Size 3-9, ~100 rules)                           │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │            CONDITIONAL RULES                            │  │  │
│  │  │             (Size 2, 35 rules)                          │  │  │
│  │  │  ┌───────────────────────────────────────────────────┐  │  │  │
│  │  │  │          FUNDAMENTAL LAWS                        │  │  │  │
│  │  │  │           (Size 1, 315 rules)                    │  │  │  │
│  │  │  │                                                  │  │  │  │
│  │  │  │   {px} + a_i → +x_i   (100%)                     │  │  │  │
│  │  │  │   {po} + a_i → +o_i   (100%)                     │  │  │  │
│  │  │  │   {e_i} + a_i → -e_i  (100%)                     │  │  │  │
│  │  │  │                                                  │  │  │  │
│  │  │  └───────────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## TicTacToe Results

### Deterministic Rules by Pattern Size

| Size | Total | Independent | Subsumed |
|------|-------|-------------|----------|
| 1 | 315 | 315 | 0 |
| 2 | 35 | 0 | 35 |
| 3 | 36 | 14 | 22 |
| 4 | 23 | 8 | 15 |
| 5 | 15 | 6 | 9 |
| 10 | 128 | 18 | 110 |

**Total: 583 deterministic rules, 365 independent, 218 subsumed (37% redundant)**

### The Fundamental Laws (Size 1)

These are the **irreducible core** - single-token rules that explain everything:

```
{px} + a0 → +x0   "If it's X's turn, action 0 places X at position 0"
{px} + a1 → +x1   "If it's X's turn, action 1 places X at position 1"
...
{po} + a0 → +o0   "If it's O's turn, action 0 places O at position 0"
...
{e0} + a0 → -e0   "If position 0 is empty, action 0 removes that empty"
```

### Subsumption Chains

117 chains of length ≥3 were found. Example:

```
Effect: +x0 (action 0)

[1] {px}           → +x0 (100%)  ← CORE LAW
[2] {e0, px}       → +x0 (100%)  ← Subsumed by size 1
[3] {e0, o3, px}   → +x0 (100%)  ← Subsumed by size 2
[10] {full state}  → +x0 (100%)  ← Subsumed by all above
```

## MiniGrid Results

### DoorKey-5x5 Hierarchy

| Size | Total | Independent | Subsumed |
|------|-------|-------------|----------|
| 1 | 514 | 514 | 0 |
| 2 | 8 | 2 | 6 |
| 3 | 25 | 9 | 16 |
| 4 | 45 | 6 | 39 |
| 5-10 | 352 | 23 | 329 |

**Pattern: Size-1 rules are always independent (the core)**

### Cross-Environment Comparison

| Environment | Size-1 Rules | Total Det | Independent |
|-------------|--------------|-----------|-------------|
| TicTacToe | 315 | 583 | 365 (63%) |
| DoorKey-5x5 | 514 | 944 | 542 (57%) |
| Empty-5x5 | 156 | 288 | 211 (73%) |
| LavaGap-S5 | 64 | 166 | 125 (75%) |
| FourRooms | 121 | 169 | 133 (79%) |

## Key Insights

### 1. The Core is Always Size-1

In every environment tested, the **most general deterministic rules** are single-token patterns. These form the irreducible core that cannot be simplified further.

### 2. Larger Rules Are Often Redundant

37-43% of deterministic rules are subsumed by smaller ones. The hierarchy naturally compresses:

```
Full ruleset → Remove subsumed → Independent rules only
    583           -218              365 (37% reduction)
```

### 3. The Hierarchy Has Meaning

- **Size 1**: Fundamental laws (always apply)
- **Size 2-3**: Conditional refinements (add preconditions)
- **Size 4+**: Specific contexts (edge cases)
- **Full size**: Complete state descriptions

### 4. Independent Rules at Higher Sizes Exist

Not everything reduces to size-1. Some rules are **genuinely conditional**:

```
TicTacToe Size 3 (independent):
{e0, e1, px} + a1 → +po (97%)
"After X plays at 1, if 0 was also empty, it becomes O's turn"

This can't be derived from size-1 rules because it depends on TWO cells.
```

## Implications

### For Learning

1. Start with single-token rules - they're the foundation
2. Only add larger patterns when they're NOT subsumed
3. Use subsumption to compress the ruleset

### For Prediction

1. Check size-1 rules first (most general, always apply)
2. Check larger rules only if size-1 doesn't determine outcome
3. Most specific matching rule gives the answer

### For Understanding

The hierarchy reveals the **structure of the domain**:
- TicTacToe: Player turn determines everything
- MiniGrid: What's in front determines most actions
- Games: Rules are simpler than they appear (subsumption compresses)
