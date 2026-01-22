# Backtest Summary - Honest World Model

## Results

### 1. TicTacToe Classification: 81%
| Variant | Accuracy |
|---------|----------|
| standard | 74% |
| no_diag | 79% |
| corners | 81% |
| l_shapes | 90% |

### 2. MiniGrid World Model Accuracy: 83%
| Environment | Prediction Accuracy |
|-------------|---------------------|
| Empty-5x5 | 82% |
| Empty-8x8 | 88% |
| DoorKey-5x5 | 83% |
| MultiRoom-N2-S4 | 81% |
| LavaGap-S5 | 83% |

### 3. MiniGrid Agent Performance: 54%
| Environment | Success Rate |
|-------------|--------------|
| Empty-5x5 | 100% ✓ |
| Empty-8x8 | 100% ✓ |
| DoorKey-5x5 | 0% ✗ |
| MultiRoom-N2-S4 | 0% ✗ |
| LavaGap-S5 | 69% ~ |

## Key Findings

### World Model Quality
- **83% prediction accuracy** across all environments
- **100% coverage** - rules for all (front, action) pairs
- Correctly learns:
  - `front=key + a3 → has_item` (pickup)
  - `front=door + a5 → door_open` (toggle)
  - `front=goal + a2 → on_goal` (win)

### Agent Performance Gap
- Simple environments (Empty): **100%**
- Complex environments (DoorKey): **0%**

The gap is due to **exploration**, not model accuracy:
- DoorKey requires full key→door→goal chain
- Random exploration rarely completes the chain
- Without initial successes, agent can't learn the subgoal structure

### Color Matching: 100% Discoverable
From correlation analysis:
- `carrying=blue_key + front=blue_door → 100% success`
- `carrying=red_key + front=red_door → 100% success`
- Mismatched colors → 0% success

This is an equivalence class discoverable from data.

## Fair vs Cheating

### Fair (Used)
- ✓ Token-based world model
- ✓ Graph algorithms (BFS/A*)
- ✓ Curiosity-driven exploration
- ✓ Value-based action selection
- ✓ Correlation discovery

### Cheating (Avoided)
- ✗ Hardcoded object semantics
- ✗ Subgoal chains (key→door→goal)
- ✗ Domain-specific heuristics

## Files
- `backtest.py` - Consolidated test runner
- `curiosity_agent.py` - Curiosity-driven agent v1
- `curiosity_agent_v2.py` - With backward chaining

## Next Steps for Improvement
1. **Better exploration** - intrinsic motivation for intermediate progress
2. **Hierarchical planning** - discover subgoals automatically
3. **Transfer learning** - share rules across similar environments
