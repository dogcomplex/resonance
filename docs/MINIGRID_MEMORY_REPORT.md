# MiniGrid Benchmark Report - With Memory

## Date: 2026-01-11

---

## Configuration

**Full MiniGrid Mechanics:**
- 7 actions: turn_left, turn_right, forward, pickup, drop, toggle, done
- 11 object types with 6 colors
- Door states: open, closed, locked
- Color-matched key-door mechanics

**Observation: Front Cell + Memory (Past 2 Steps)**
- Current: front cell type/color/state, carrying state, standing on
- Memory: prev1_front, prev1_action, prev2_front, prev2_action
- Patterns: double_turn, forward_progress, front_changed
- Regional signature for disambiguation

---

## Results Comparison

| Environment | No Memory | With Memory | Change |
|-------------|-----------|-------------|--------|
| Empty-5x5 | 100% | 98% | -2% |
| Empty-8x8 | 99% | 96% | -3% |
| DoorKey-5x5 | 93% | 93% | 0% |
| DoorKey-6x6 | 83% | 78% | -5% |
| DoorKey-8x8 | 75% | 68% | -7% |
| FourRooms | 47% | 37% | -10% |
| **MultiRoom-N2-S4** | **0%** | **97%** | **+97% ★★★** |
| **MultiRoom-N4-S5** | **0%** | **50%** | **+50% ★★** |
| LavaGap-S5 | 100% | 100% | 0% |
| LavaGap-S6 | 91% | 92% | +1% |
| Dynamic-Obstacles | 28% | 20% | -8% |
| Memory-S7 | 0% | 0% | 0% |
| LockedRoom | 0% | 0% | 0% |

---

## Key Findings

### Big Wins ★★★

**MultiRoom-N2-S4: 0% → 97%**
- Memory helps track door toggle state!
- Pattern: see closed door → toggle → see open door → forward
- Without memory, agent couldn't distinguish "door closed" from "door already opened"

**MultiRoom-N4-S5: 0% → 50%**
- Same principle, harder with 4 rooms
- Need more exploration or better heuristics

### Slight Regressions

**DoorKey-8x8, FourRooms: -5% to -10%**
- Larger state space with memory tokens
- More unique observations → sparser coverage
- Trade-off: better for sequential tasks, worse for pure exploration

### No Change

**Memory-S7, LockedRoom: Still 0%**
- Need fundamentally more than 2-step memory
- Memory-S7 requires remembering what's behind wall (longer horizon)
- LockedRoom needs multi-step key-door-room planning

---

## Memory Tokens Added

```
Current observation:
  front={type}           # What's in front
  front={color}_{type}   # With color for objects
  front_door={state}     # Door state (open/closed/locked)
  carrying={color}_{type} # What agent holds
  has_item / empty_handed
  on_goal                # If standing on goal

Memory (past 2 steps):
  prev1_front={type}     # What was in front t-1
  prev1_act={action}     # Action at t-1
  prev2_front={type}     # What was in front t-2
  prev2_act={action}     # Action at t-2

Derived patterns:
  pattern=double_turn    # Turned twice (maybe stuck)
  pattern=forward_progress # Moved forward twice
  pattern=front_changed  # Front cell changed after action
  region={N}             # Regional signature
```

---

## Summary

| Status | Count | Environments |
|--------|-------|--------------|
| ✓ Solved (≥80%) | 6 | Empty, DoorKey-5x5, MultiRoom-N2, LavaGap |
| ~ Partial (30-80%) | 4 | DoorKey-6x6/8x8, FourRooms, MultiRoom-N4 |
| ✗ Failed (<30%) | 3 | Dynamic, Memory, LockedRoom |

**Memory's Value:**
- Essential for sequential tasks (open door → go through)
- Helps detect stuck patterns (double turn)
- Enables learning action effects (toggle changes door state)

**Remaining Limitations:**
- 2-step memory insufficient for long-horizon tasks
- Dynamic obstacles need prediction, not memory
- Complex multi-key puzzles need explicit planning

---

## Files

- `minigrid_full.py` - Full MiniGrid implementation (33 environments)
- `minigrid_memory.py` - Memory-augmented benchmark
- `minigrid_benchmark.py` - Original no-memory benchmark
