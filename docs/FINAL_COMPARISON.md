# Hierarchical Learner Comparison: V6b vs V8

## Design Philosophy

**V6b**: Dual prototype fallback system
- Relative prototype: behavioral clustering for view/dir/carry effects
- Delta prototype: position movement patterns
- Probability threshold for predictions

**V8**: Pure specificity-based selection
- Most specific matching rule wins
- Single delta fallback for positions
- Returns probabilities, caller chooses threshold

## Benchmark Results (Seeded)

| Environment | V6b | V8 | Winner |
|-------------|-----|-----|--------|
| Empty-8x8 | 95.9% | 96.0% | Tie |
| FourRooms | **83.6%** | 76.7% | V6b |
| DoorKey-6x6 | 94.7% | **97.5%** | V8 |
| TicTacToe | 79.5% | **87.0%** | V8 |
| Pure Chaos | 43.4% | 43.6% | Tie |

## When to Use Which

**Use V6b for:**
- Navigation with partial observability (MiniGrid)
- Environments where behavioral prototypes matter
- Cases with many unseen position combinations

**Use V8 for:**
- Discrete state spaces (board games)
- Turn-based games with clear state transitions
- When you need probability outputs

## Key Insight

V6b's relative prototype fallback helps when:
- The exact state hasn't been seen
- But similar "behavioral context" has
- E.g., "facing wall while moving forward" pattern generalizes

V8's simpler design is cleaner but loses this generalization capability for non-position effects.

## Recommendation

For general use: **V6b** (more robust across environments)
For board games: **V8** (better discrete state handling)
