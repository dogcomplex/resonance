# MiniGrid Benchmark Results

## Summary Table

| Environment | V9 | Unified | Match |
|-------------|-----|---------|-------|
| Empty-5x5 | 100.0% | 100.0% | ✓ |
| Empty-8x8 | 100.0% | 100.0% | ✓ |
| Empty-16x16 | 100.0% | 100.0% | ✓ |
| DoorKey-5x5 | 96.9% | 96.6% | ✓ |
| DoorKey-6x6 | 94.2% | 93.4% | ✓ |
| DoorKey-8x8 | 94.6% | 94.0% | ✓ |
| FourRooms | 98.7% | 98.6% | ✓ |
| LavaGap-5 | 99.2% | 98.9% | ✓ |
| LavaGap-7 | 99.5% | 99.3% | ✓ |
| **Dynamic-5x5** | 88.4% | 87.7% | ✓ |
| **Dynamic-6x6** | 86.2% | 85.6% | ✓ |

## Difficulty Ranking (Hardest First)

1. **Dynamic-6x6** (85.6%) - Moving obstacles, non-deterministic
2. **Dynamic-5x5** (87.7%) - Moving obstacles
3. **DoorKey-6x6** (93.4%) - Key-door mechanics
4. **DoorKey-8x8** (94.0%) - Key-door mechanics  
5. **DoorKey-5x5** (96.6%) - Key-door mechanics
6. **FourRooms** (98.6%) - Multi-room navigation
7. **LavaGap-5** (98.9%) - Obstacle avoidance
8. **LavaGap-7** (99.3%) - Obstacle avoidance
9. **Empty-*x*** (100.0%) - Pure navigation

## Key Findings

1. **Unified Induction matches V9** across all 11 environments (within 1%)
2. **Dynamic environments are hardest** due to non-deterministic obstacle movement
3. **DoorKey is second hardest** due to multi-step key→door→goal mechanics
4. **Empty/Lava environments are easy** - simple deterministic physics

## Why Dynamic Is Hard

The Dynamic environments have moving obstacles that change position each step.
This creates:
- Non-deterministic state transitions
- Observations that don't repeat exactly
- Rules that are probabilistic, not deterministic

This is exactly where multi-fidelity temporal coarsening could help - 
abstracting over the noise of moving obstacles to find stable patterns.

## Test Configuration

- Training: 100 episodes
- Testing: 30 episodes  
- Max steps per episode: 100
- Tokenization: Blind (no domain knowledge)
- Scoring: F1 on SEEN (state, action) pairs only
