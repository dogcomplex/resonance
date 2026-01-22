# Similarity Function Comparison for Token Equivalence

## O(n) Similarity Functions Tested

| Function | Formula | Description |
|----------|---------|-------------|
| **Intersection** | A ∩ B | Elements in ALL sets (binary) |
| **Jaccard** | \|A∩B\| / \|A∪B\| | Set overlap ratio |
| **Cosine** | dot(A,B) / (\|A\|\|B\|) | Distribution similarity |
| **Dice** | 2\|A∩B\| / (\|A\|+\|B\|) | Like F1 score |

## Results on Pong (N=50 episodes)

### Threshold Sweep (Cosine)
| Threshold | UNSEEN | Tokens Merged |
|-----------|--------|---------------|
| 0.50 | 23.9% | 61 |
| 0.70 | 27.5% | 43 |
| 0.90 | 45.1% | 19 |
| **0.95** | **46.2%** | 17 |
| 1.00 | 34.5% | 0 |

**Key insight: Less aggressive merging (high threshold) = better generalization!**

### Function Comparison (at threshold=0.9)
| Function | UNSEEN |
|----------|--------|
| **Jaccard** | **46.2%** |
| Cosine | 45.1% |
| No merging | 34.5% |
| Dice | 28.2% |

## Why Jaccard Wins

Jaccard measures **set overlap** of effects:
- If token A causes effects {e1, e2, e3}
- And token B causes effects {e1, e2, e4}
- Jaccard = 2/4 = 0.5

This is simpler than cosine (no frequency weighting) but equally effective.

## Optimal Configuration

- **Similarity function**: Jaccard or Cosine
- **Threshold**: 0.9-0.95 (high, conservative)
- **Structural constraint**: Allow only 1 component to differ
- **Min observations**: 5 per token

## Complexity

All functions are O(n) where n = number of unique effects:
- Intersection: O(|A| + |B|)
- Jaccard: O(|A| + |B|) 
- Cosine: O(|A| + |B|)
- Dice: O(|A| + |B|)

Can be bucketed to constant time by quantizing effect frequencies.
