# Experimental Designs Catalog

## Design 1: Baseline Intersection Sieve
- **Algorithm**: Group by (action, effect), intersect LHS
- **Bucketing**: Random sampling, N=10 buckets
- **Scoring**: Buckets where rule held / total buckets
- **Result**: 100% on discrete games, 29-36% UNSEEN on pixels

## Design 2: Universal Sieve (3 dimensions)
- **Probability**: Random buckets → intersection → confidence
- **Fidelity**: Token groupings → intersection → abstraction  
- **Invariance**: Time windows → intersection → stability
- **Result**: Same as baseline (exact match dominates)

## Design 3: Cosine Fidelity Sieve
- **Token equivalence**: Cosine similarity of effect distributions
- **Threshold**: 0.7 (tokens with >70% similar effects merged)
- **Structural constraint**: Allow only 1 component to differ
- **Result**: UNSEEN jumped from 29% to 68% on Pong!

## Design 4: Learning Curve Analysis
- **Finding**: Optimal training point exists (25-100 episodes)
- **Beyond optimal**: UNSEEN accuracy DECREASES
- **Hypothesis**: Over-merging or overfitting to training patterns

## Key Metrics Tracked
- SEEN accuracy: Performance on states seen during training
- UNSEEN accuracy: TRUE generalization to novel states
- Gap: SEEN - UNSEEN (smaller = better generalization)
- Classes: Number of equivalence classes formed
- Merged: Number of tokens merged into classes

## Learning Curve Results

### Pong
| Episodes | SEEN | UNSEEN | Gap |
|----------|------|--------|-----|
| 5 | 70.0% | 40.7% | 29.3% |
| 25 | 76.3% | 56.9% | 19.4% |
| 100 | 76.0% | **63.2%** | 12.7% |
| 500 | 79.0% | 53.3% ↓ | 25.7% |

### Breakout
| Episodes | SEEN | UNSEEN | Gap |
|----------|------|--------|-----|
| 5 | 77.1% | 48.2% | 28.9% |
| 25 | 83.7% | **76.9%** | 6.8% |
| 100 | 81.8% | 69.0% | 12.8% |
| 500 | 85.3% | 66.4% ↓ | 18.9% |

## Ideas for Next Experiments

1. **Adaptive threshold**: Learn optimal cosine threshold per game
2. **Hierarchical merging**: tokens → classes → meta-classes
3. **Regularization**: Prevent over-merging with penalty
4. **Context-aware**: Equivalence conditional on other tokens
5. **Effect-side abstraction**: Also merge similar effects
6. **Graph clustering**: Spectral methods for communities

## Code Locations
- `/home/claude/universal_sieve.py` - Base 3D sieve
- `/home/claude/pixel_fidelity_boost.py` - Cosine fidelity
- `/home/claude/learning_curve_fixed.py` - Learning curve analysis
- `/home/claude/pixel_environments.py` - Pong/Breakout/Invaders
