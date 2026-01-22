# V11 Design: Towards General Game Understanding

## Key Insights from Experiments

### 1. Trajectory Tokens Dramatically Reduce Aliasing

| Mode | Seen F1 | Aliasing Rate |
|------|---------|---------------|
| Bucketed positions only | 38% | 58% |
| **+ Trajectory (delta) tokens** | **73%** | 35% |

Adding `ball_dx_{-1,0,+1}` and `ball_dy_{-1,0,+1}` tokens nearly doubles accuracy because the same bucketed position with different velocities IS a different state.

### 2. Auto-Discovery of Game Rules Works!

By analyzing "what tokens are present when events happen", we discovered:

**vx_reversal occurs when:**
- `ball_x_18` (53%) or `ball_x_0` (42%) → Ball hits side walls!

**vy_reversal occurs when:**
- `ball_y_19` (53%) or `ball_y_0` (39%) → Ball hits top/bottom!

These ARE the actual game physics rules - discovered with zero domain knowledge.

### 3. Slice-Finder > RAM > Bucketed Pixels

If we had object detection:
- Objects are natural game entities
- Positions can be exact (no aliasing)
- Relative features come naturally
- "Ball 2 units from paddle" is meaningful

### 4. "Understanding" vs "Exact Prediction"

We don't need to predict ball at (47, 83). We need to understand:
- Ball bounces off walls
- Ball bounces off paddle  
- Missing ball loses life

These **qualitative rules** are the real game understanding.

## V11 Architecture

```
Observation (pixels/RAM)
       ↓
Trajectory Tracking (add delta tokens)
       ↓
Exact Memory (SQLite/LRU) ─→ if found: return exact
       ↓
Quantitative Rules (hierarchical patterns)
       ↓
Trajectory Filtering (use deltas to pick consistent rules)
       ↓
Qualitative Abstraction (event detection, high-level rules)
```

## New Components

### A. Trajectory Tokens (Auto-Derived)
```python
def add_trajectory(curr, prev, tokens):
    for token in curr:
        if is_position_token(token):
            delta = get_value(token, curr) - get_value(token, prev)
            tokens.add(f"{token}_delta_{sign(delta)}")
```

### B. Event Discovery
```python
def discover_events(observations):
    # Find what tokens predict state changes
    for event_type in ['velocity_reversal', 'score_change', 'death']:
        contexts = get_contexts_where_event_happened(event_type)
        common_tokens = find_frequent_tokens(contexts)
        # These tokens are the RULES for this event!
```

### C. Qualitative Rules (Auto-Discovered)
```
vx_reversal ← ball_x_edge (wall bounce)
vx_reversal ← ball_near_paddle (paddle bounce)
vy_reversal ← ball_y_edge (ceiling/floor bounce)
score_increase ← ball_past_opponent
```

## Implementation Plan

1. **SQLite exact memory** - Handle large state spaces
2. **Trajectory tokens** - Auto-computed from consecutive frames
3. **Event discovery** - Find what predicts important state changes
4. **Qualitative rules** - High-level understanding separate from exact positions
5. **Rule explanation** - Output discovered rules in human-readable form

## Expected Results

| Component | Contribution |
|-----------|--------------|
| Exact memory | 100% on seen states |
| + Trajectory | +35% on unseen (reduces aliasing) |
| + Event rules | +20% (catches edge cases) |
| + Qualitative | Game "understanding" |

The goal: Not just predict, but EXPLAIN the game.
