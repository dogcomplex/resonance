# Convergence and Abstraction Analysis

## Key Findings

### 1. Nested Deterministic Hierarchy

Deterministic rules form nested layers where smaller patterns subsume larger ones:

```
[Size 1] {px} + a0 → +x0 (100%)       ← MOST GENERAL
    ↓ subsumes
[Size 2] {e0, px} + a0 → +x0 (100%)   ← more specific
    ↓ subsumes
[Size 3] {e0, o3, px} + a0 → +x0      ← even more specific
```

**37-43% of deterministic rules are redundant** (subsumed by smaller rules).

### 2. Token Abstraction (Layer 0)

Above single-token rules, we can discover **abstract rules** that cover entire token classes:

```
[Size 0] {x*} + a* → placement      ← ABSTRACT (covers x0-x8)
[Size 1] {x0} + a0 → +x0            ← CONCRETE
```

**Requirements for procedural abstraction (no domain knowledge):**
1. Structural similarity (common prefix)
2. Functional similarity (same effect)
3. Statistical similarity (similar probabilities)

### 3. Convergence Metrics

| Game | Episodes | Observations | Det Rules | Independent | Abstractions |
|------|----------|--------------|-----------|-------------|--------------|
| TicTacToe | 2000 | 15,243 | 466 | 323 (69%) | 212 (52 det) |
| Connect-3 | 2000 | 18,361 | 402 | 267 (66%) | 182 (21 det) |
| Empty-5x5 | 2000 | 94,235 | 340 | 245 (72%) | 0 |
| DoorKey-5x5 | 2000 | 99,822 | 1,339 | 626 (47%) | 14 (2 det) |

**Key insight**: MiniGrid has few abstractions because token numbers represent TYPES (key=5, door=4, goal=8) not POSITIONS. Different types have different behaviors → not abstractable.

### 4. Why Not 100% Deterministic?

Rules are probabilistic when context is missing:

```
{px} + a0 → +po (varies)
├── If cell 0 empty: 100% turn change  
└── If cell 0 occupied: 0% turn change
```

The general rule sees both outcomes → probabilistic.
The specific rule `{px, e0} + a0 → +po` is 100% deterministic.

**This is WHY the hierarchy exists:**
- General rules = baseline expectations
- Specific rules = precise predictions

### 5. Complete Hierarchy Structure

```
┌────────────────────────────────────────────────────────┐
│  LAYER 4: Full Context (Size 10+)                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ LAYER 3: Contextual (Size 3-9) - Probabilistic   │  │
│  │ ┌──────────────────────────────────────────────┐ │  │
│  │ │ LAYER 2: Conditional (Size 2) - Multi-token  │ │  │
│  │ │ ┌──────────────────────────────────────────┐ │ │  │
│  │ │ │ LAYER 1: Concrete Laws (Size 1)         │ │ │  │
│  │ │ │ ┌──────────────────────────────────────┐ │ │ │  │
│  │ │ │ │ LAYER 0: Abstract Laws               │ │ │ │  │
│  │ │ │ │                                      │ │ │ │  │
│  │ │ │ │  {x*} → placement                    │ │ │ │  │
│  │ │ │ │  {e*} → -empty                       │ │ │ │  │
│  │ │ │ └──────────────────────────────────────┘ │ │ │  │
│  │ │ │  {px} + a0 → +x0                         │ │ │  │
│  │ │ │  {front_t5} + pickup → +carry            │ │ │  │
│  │ │ └──────────────────────────────────────────┘ │ │  │
│  │ └──────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 6. Convergence Rates

| Game | Time to 90% | Observations Needed | Complexity |
|------|-------------|---------------------|------------|
| TicTacToe | ~2000 ep | ~15,000 | Simple |
| Connect-3 | ~2000 ep | ~18,000 | Simple |
| Empty-5x5 | ~1000 ep | ~47,000 | Medium |
| DoorKey-5x5 | ~2000 ep | ~100,000 | Complex |

**Pattern**: More complex games need more observations but converge at similar episode counts because each episode provides more data.

## Implementation Notes

### Procedural Abstraction

```python
def discover_abstractions(rules, min_members=3, max_variance=0.02):
    # Group by (prefix, effect, action)
    by_key = defaultdict(list)
    for rule in rules:
        if len(rule.pattern) == 1:
            token = list(rule.pattern)[0]
            prefix = extract_prefix(token)  # 'x0' → 'x'
            by_key[(prefix, rule.effect, rule.action)].append(rule)
    
    abstractions = []
    for key, group in by_key.items():
        if len(group) < min_members:
            continue
        
        probs = [r.probability for r in group]
        variance = sum((p - mean(probs))**2 for p in probs) / len(probs)
        
        if variance <= max_variance:
            abstractions.append({
                'pattern': f"{key[0]}*",
                'effect': key[1],
                'mean_prob': mean(probs),
                'members': len(group)
            })
    
    return abstractions
```

### Why MiniGrid Has Fewer Abstractions

TicTacToe tokens: `x0, x1, x2, ... x8`
- All mean "X at position N"
- Can abstract to `x*` = "X anywhere"

MiniGrid tokens: `front_t5, front_t4, front_t8`
- t5 = key, t4 = door, t8 = goal
- Different types have different behaviors
- Cannot abstract (key ≠ door ≠ goal)

## Files

- `hierarchical_learner.py` - Core implementation
- `convergence_tracking.py` - Convergence analysis
- `abstraction_analysis.py` - Token abstraction discovery
