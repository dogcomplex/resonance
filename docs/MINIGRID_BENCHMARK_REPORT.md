# MiniGrid Benchmark Report

## Configuration

**Full MiniGrid Mechanics:**
- 7 actions: turn_left, turn_right, forward, pickup, drop, toggle, done
- 11 object types: unseen, empty, wall, floor, door, key, ball, box, goal, lava, agent
- 6 colors: red, green, blue, purple, yellow, grey
- Door states: open, closed, locked
- Color-matched key-door mechanics

**Minimal Observation (Front Cell Only):**
- Only see the cell DIRECTLY in front
- Know what agent is carrying
- Know what agent is standing on
- ~4-40 unique observation tokens depending on environment

**Same UnifiedFairLearner for ALL environments - NO domain knowledge**

---

## Results Summary

| Category | Best Success | Status |
|----------|--------------|--------|
| Empty (Navigation) | 100% | ✓ Solved |
| Lava (Hazard Avoidance) | 100% | ✓ Solved |
| DoorKey (Key-Door) | 93% | ✓ Solved |
| Rooms (Multi-Room) | 47% | ~ Partial |
| Dynamic Obstacles | 28% | ~ Partial |
| Complex (Multi-Skill) | 0% | ✗ Failed |

---

## Detailed Results

### ✓ Solved (≥80% success)

| Environment | Success | Avg Steps | Training |
|-------------|---------|-----------|----------|
| Empty-5x5 | 100% | 15 | 1000 eps |
| Empty-8x8 | 99% | 74 | 500 eps |
| Empty-Random-5x5 | 100% | 13 | 1000 eps |
| LavaGap-S5 | 100% | 18 | 100 eps |
| LavaGap-S6 | 91% | 46 | 1000 eps |
| LavaCrossing-S9N1 | 81% | 150 | 100 eps |
| DoorKey-5x5 | 93% | 39 | 500 eps |
| DoorKey-6x6 | 83% | 64 | 500 eps |

### ~ Partial (30-80%)

| Environment | Success | Notes |
|-------------|---------|-------|
| DoorKey-8x8 | 75% | Larger search space |
| FourRooms | 47% | Random gap placement |

### ✗ Failed (<30%)

| Environment | Success | Why |
|-------------|---------|-----|
| Dynamic-Obstacles | 21-28% | Can't predict moving objects |
| MultiRoom | 0% | Closed doors need toggle |
| KeyCorridor | 0% | Multi-step key hunt |
| LockedRoom | 0% | Multiple color-matched keys |
| Memory | 0% | Requires past observation memory |

---

## What the Agent Learns

From minimal front-cell observation, the agent discovers:

1. **Movement semantics:**
   - "front=empty" → FORWARD works
   - "front=wall" → Turn needed
   - "front=goal" → FORWARD wins

2. **Object interactions:**
   - "front=key" + "empty_handed" → PICKUP
   - "front=door_locked" + "has_item" → TOGGLE
   - "front=door_open" → FORWARD

3. **Hazard avoidance:**
   - "front=lava" → DON'T go forward

---

## Key Insights

### What Works
- Navigation in open spaces with minimal observation
- Single key-door puzzles (one key, one door)
- Static hazard avoidance (lava)
- Learning action semantics from exploration

### What Doesn't Work
- **Memory-dependent tasks:** Only see current front cell
- **Dynamic elements:** Can't predict moving obstacles
- **Multi-step planning:** Multiple locked doors with specific key colors
- **Large state spaces:** FourRooms (19x19) needs more exploration

### Honest Assessment

The minimal observation approach works surprisingly well for:
- 8/17 environments solved (≥80%)
- 10/17 environments functional (≥30%)

But fundamentally limited by:
- No memory of past observations
- No world model or planning
- Reactive policy only

---

## Files

- `minigrid_full.py` - Full MiniGrid implementation (33 environments)
- `minigrid_benchmark.py` - Benchmark runner
- `unified_fair_learner.py` - The learning algorithm

---

## Comparison to Previous Claims

| Previous Claim | Reality |
|----------------|---------|
| "95% MiniGrid success" | On simplified 3-token observation |
| "Full MiniGrid" | Actually minimal front-cell only |
| "Discovers action semantics" | ✓ True - rotation, forward, pickup, toggle |
| "No domain knowledge" | ✓ True - same learner for all games |

This benchmark uses **full official mechanics** with **minimal observation** - 
an honest test of what the approach can actually do.
