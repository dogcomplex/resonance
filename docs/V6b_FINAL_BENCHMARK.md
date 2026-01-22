# Hierarchical Learner V6b - Final Benchmark Results

## Summary

| Domain | Average F1 | Notes |
|--------|-----------|-------|
| MiniGrid | **91.3%** | Grid navigation with position deltas |
| TicTacToe | **81.9%** | Board games with contextual rules |
| Pure Chaos | 43.4% | Random transitions (baseline) |

**Overall (learnable): 86.6%**  
**Signal over noise: +43.2%**

## Detailed Results

### MiniGrid Environments
| Environment | F1 | Description |
|-------------|-----|-------------|
| Empty-8x8 | 96.1% | Simple open grid |
| FourRooms | 84.9% | Multi-room navigation |
| DoorKey-6x6 | 92.9% | Object interaction |

### TicTacToe Variants
| Variant | F1 | Description |
|---------|-----|-------------|
| Standard | 79.5% | Normal 3-in-a-row wins |
| Misere | 79.5% | 3-in-a-row loses |
| Wild | 86.9% | Can play X or O |

### Chaos Baseline
| Variant | F1 | Description |
|---------|-----|-------------|
| Pure Random | 43.4% | Random state transitions |

## Key Findings

1. **Chaos detection works**: The learner achieves ~87% on structured games but only ~43% on pure chaos, demonstrating it captures real patterns rather than memorizing.

2. **Learning curves plateau quickly**: Both structured and chaotic environments plateau by ~50-100 episodes, but at different levels (80% vs 43%).

3. **Delta-based fallback is critical**: Position predictions generalize via learned movement deltas, enabling transfer to unseen positions.

4. **TicTacToe is harder than MiniGrid**: Lower F1 (82% vs 91%) because:
   - Same action has different effects based on player (context-dependent)
   - Win detection rules depend on global board state

## Architecture Highlights

- **Position delta prototypes**: Generalize movement patterns across positions
- **Relative vs position split**: Separate rules for view changes vs movement
- **Prototype fallback**: Handle unseen states via behavioral clustering
- **No hardcoded domain knowledge**: All patterns learned from data
