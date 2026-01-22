# MiniGrid Benchmark - Final Report

## Improvements Implemented

1. **Memory (Past 2 Steps)**: Track previous observations and actions
2. **Probabilistic Fallback**: Score actions by (wins/total) ratio
3. **Door Toggle Fix**: Toggle CLOSED doors (not just locked!)
4. **3-Cell View**: Front + left + right observation
5. **Pattern Detection**: Detect stuck patterns (double turn)

---

## Results Comparison

| Environment | Original | +Memory | Smart Agent | Δ |
|-------------|----------|---------|-------------|---|
| Empty-5x5 | 100% | 98% | **100%** | 0% |
| Empty-8x8 | 99% | 96% | **93%** | -6% |
| DoorKey-5x5 | 93% | 93% | **97%** | +4% |
| DoorKey-6x6 | 83% | 78% | **83%** | 0% |
| **MultiRoom-N2-S4** | 0% | 97% | **100%** | **+100%** ★★★ |
| **MultiRoom-N4-S5** | 0% | 50% | **50%** | **+50%** ★★ |
| FourRooms | 47% | 37% | 30% | -17% |
| LavaGap-S5 | 100% | 100% | **93%** | -7% |

---

## Key Discovery

### The Toggle Fix ★★★

```python
# BEFORE (broken):
if front == DOOR:
    if state == LOCKED and has_key:
        action = TOGGLE  # Only toggled locked doors!

# AFTER (working):
if front == DOOR:
    if state == OPEN:
        action = FORWARD
    elif state == CLOSED:
        action = TOGGLE  # Toggle closed doors too!
    elif state == LOCKED and has_key:
        action = TOGGLE
```

**MultiRoom has CLOSED doors, not locked!** This single fix took us from 0% to 100%.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Smart Agent                         │
├─────────────────────────────────────────────────┤
│ OBSERVATION:                                     │
│   Current: front (type,color,state), left, right│
│   Memory: prev1_obs, prev1_act, prev2_obs, prev2_act │
│   Derived: carrying, on_goal, patterns          │
├─────────────────────────────────────────────────┤
│ DECISION HIERARCHY:                             │
│   1. Goal in front → FORWARD                    │
│   2. Key in front (empty handed) → PICKUP       │
│   3. Door open → FORWARD                        │
│   4. Door closed → TOGGLE                       │
│   5. Door locked + key → TOGGLE                 │
│   6. Goal on side → TURN                        │
│   7. Learned best action (prob. score)          │
│   8. Random exploration                         │
├─────────────────────────────────────────────────┤
│ LEARNING:                                       │
│   (state_key, action) → [wins, total]           │
│   Score = wins / total                          │
│   Use highest-scoring action when uncertain     │
└─────────────────────────────────────────────────┘
```

---

## Remaining Challenges

| Environment | Success | Why Hard |
|-------------|---------|----------|
| FourRooms | 30% | Large maze, aliased states |
| Memory-S7 | 0% | Needs 5+ step memory |
| LockedRoom | 0% | Multi-key color matching |
| Dynamic | 20% | Need prediction, not memory |

---

## Memory-S7 Analysis

```
Grid Layout:
WWWWWWW
W..A..W  ← Agent sees target ball here (row 1)
WWW.WWW  ← Wall blocks view
W.....W
W.....W
W.B.B.B  ← Must pick matching ball (row 5)
WWWWWWW

Minimum memory needed: ~5 steps
- See target at (3,1)
- Move through gap at (3,2)  
- Navigate to bottom (y=5)
- Target is out of view for 4-5 steps!
```

Our 2-step memory is insufficient. Would need memory ≥ 5 or adaptive deepening.

---

## Configuration Used

- **Training**: 50 episodes
- **Testing**: 30 episodes  
- **Max steps**: 300 per episode
- **Memory depth**: 2 (obs, action) pairs
- **Exploration**: ε=0.2 random actions

---

## Files

- `minigrid_full.py` - Full MiniGrid implementation
- `minigrid_memory.py` - Memory-augmented version
- `minigrid_benchmark.py` - Original benchmark
