# Extension Test Results

All extensions tested using FAIR discovery methods (no domain knowledge).

---

## Summary Table

| Extension | Mechanism | FourRooms Impact | Verdict |
|-----------|-----------|------------------|---------|
| **Temporal Compression** | Skip transitional states | Minimal (95% states stable) | ❌ Not useful for MiniGrid |
| **Prototype Matching** | Cluster by behavior | +0.4% F1, 37% compression | ⚠️ Modest improvement |
| **Derived Tokens** | Discover useful comparisons | ~0% (no useful conditions found) | ❌ Thresholds too strict |
| **Context Filtering** | Remove position tokens | **+10.6% F1, 71% fewer rules** | ✅ **Significant win** |

---

## Detailed Results

### 1. Temporal Compression
**Goal**: Skip transitional/animation states, learn only between stable states.

**Results (MiniGrid Empty-8x8)**:
- 95.6% of states ARE stable
- Compression ratio: 99.3% (only 0.7% skipped)
- F1: 97.2% → 94.8% (slightly worse)

**Conclusion**: MiniGrid doesn't have animation frames. Every frame is a decision point. This extension would help more in games with animations (Pokemon, fighting games).

---

### 2. Prototype Matching  
**Goal**: Cluster states by behavioral similarity, generalize to unseen states.

**Results (FourRooms)**:
- Compressed 2880 states → 1066 prototypes (37%)
- F1: 92.0% → 92.4% (+0.4%)

**Sample prototypes discovered**:
- States with `{front_t1, left_t1, right_t1}` cluster together
- States with `{front_t2}` (wall ahead) form separate cluster

**Conclusion**: Modest improvement. Prototypes capture meaningful behavioral similarity but gains are limited in MiniGrid.

---

### 3. Derived Tokens
**Goal**: Discover useful numerical comparisons/buckets automatically.

**Results (FourRooms)**:
- 0 useful conditions found at threshold (info_gain > 0.2)
- Conditions tested: position buckets, edge detection, parity
- No improvement

**Why it failed**: 
- MiniGrid's dynamics don't depend on position buckets
- The relative view tokens already capture what matters
- Would work better in games where absolute position matters

---

### 4. Context Filtering (✅ Winner)
**Goal**: Discover which tokens are relevant for which predictions.

**Results (FourRooms)**:
- Rules: 5529 → 1628 (71% reduction!)
- F1 for relative effects: 61.3% → 71.9% (+10.6%)

**Key Discovery**: 
```
Position tokens (pos_X_Y) are position-specific → low generalization
View tokens (front_t*, left_t*) are position-independent → high generalization
```

**PMI Analysis confirmed**:
- `front_t1` ↔ `front_t2` are mutually exclusive (PMI = -10)
- `front_t1` ↔ `left_t1` are weakly associated (PMI = 0.8)
- Direction tokens are mutually exclusive

**Conclusion**: Removing position tokens for relative predictions gives:
1. 71% fewer rules (faster learning)
2. 10.6% better F1 (better generalization)
3. Rules that transfer across positions

---

## Why Context Filtering Won

The key insight: **Not all tokens are relevant for all predictions.**

In MiniGrid:
- For "will I hit a wall if I move forward?" → Only need `front_t*`
- For "what's my absolute position after moving?" → Need `pos_*`
- For "which direction am I facing?" → Only need `dir_*`

By removing position tokens:
1. States that differ only in position become identical
2. Rules learned at one position apply everywhere
3. Much faster convergence to generalizable rules

---

## Integration with Current Learner

The context filtering insight can be integrated:

```python
class ContextAwareHierarchicalLearner:
    def __init__(self):
        self.full_learner = HierarchicalLearner()  # For position-dependent effects
        self.relative_learner = HierarchicalLearner()  # For view/dir effects
    
    def observe(self, state, action, next_state):
        # Full state for position changes
        self.full_learner.observe(state, action, next_state)
        
        # Filtered state for relative changes
        rel_state = {t for t in state if not t.startswith('pos_')}
        rel_next = {t for t in next_state if not t.startswith('pos_')}
        self.relative_learner.observe(rel_state, action, rel_next)
    
    def predict(self, state, action):
        # Use relative learner for view/dir effects (more general)
        # Use full learner for position effects (position-specific)
        pass
```

---

## Next Steps

1. **Integrate context filtering** into main learner
2. **Re-test convergence** with context-aware learning
3. **Try on Pokemon** where contexts (battle/overworld/menu) are more distinct
4. **Lower thresholds** for derived token discovery and retry
