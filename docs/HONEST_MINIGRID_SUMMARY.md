# Honest MiniGrid World Model - Summary

## World Model Accuracy

Measured on held-out transitions (300 train, 100 test episodes):

| Environment      | Accuracy | Coverage | Rules | Avg Conf |
|-----------------|----------|----------|-------|----------|
| Empty-5x5       | 82%      | 100%     | 18    | 83%      |
| Empty-8x8       | 89%      | 100%     | 18    | 89%      |
| DoorKey-5x5     | 82%      | 100%     | 66    | 80%      |
| DoorKey-6x6     | 81%      | 100%     | 62    | 81%      |
| MultiRoom-N2-S4 | 81%      | 100%     | 24    | 81%      |
| MultiRoom-N4-S5 | 85%      | 100%     | 24    | 78%      |
| LavaGap-S5      | 82%      | 100%     | 24    | 84%      |
| FourRooms       | 92%      | 100%     | 18    | 93%      |

**Overall: ~82-92% prediction accuracy, 100% coverage**

## Key Learned Rules (100% confidence)

### Pickup Action (action 3)
```
front=blue_key + action 3 → +[carrying=blue_key, has_item] -[empty_handed, front=key]
front=red_key + action 3 → +[carrying=red_key, has_item] -[empty_handed, front=key]
```

### Toggle Door (action 5)
```
front=red_door + action 5 → +[front_door=open] -[front_door=locked]
(Only when carrying matching color key)
```

### Movement (action 2)
```
front=empty + action 2 → front changes (moved forward)
front=goal + action 2 → +[on_goal] (SUCCESS)
front=wall + action 2 → no change (blocked)
```

### Rotation (actions 0, 1)
```
front=X + action 0/1 → front=Y (different front token)
```

## Color Matching Discovery

**100% discoverable from token correlation:**

| Carrying | Door | Success Rate |
|----------|------|--------------|
| blue_key | blue_door | 100% ✓ |
| red_key | red_door | 100% ✓ |
| green_key | green_door | 100% ✓ |
| yellow_key | yellow_door | 100% ✓ |
| any | mismatched | 0% ✗ |

**Rule discovered:** `carrying_color == door_color → toggle succeeds`

This is an equivalence class abstraction - no need for separate rules per color.

## What's Fair vs Cheating

### Fair (Learnable from Data)
- ✓ Action semantics (rotation vs movement from 2-cycles)
- ✓ Causal rules (token + action → effect)
- ✓ Color matching (from correlation)
- ✓ Success tokens (on_goal predicts success)
- ✓ A*/BFS on state graph
- ✓ Novelty-weighted exploration

### Cheating (Domain Knowledge)
- ✗ "GOAL means go there"
- ✗ "KEY means pick up"
- ✗ Hardcoded subgoal chains
- ✗ Semantic understanding of object types

## Agent Performance

**Without domain knowledge:** ~23-47%
**With domain knowledge:** ~96%

The gap is due to **exploration**, not world model accuracy.
The world model is accurate, but without initial successes
to learn from, navigation doesn't emerge.

## Fair Approaches for Improvement

1. **Curiosity-driven exploration** - visit novel states
2. **BFS toward interesting tokens** - not just goal, any rare token
3. **Try all actions on novel fronts** - systematic exploration
4. **RL on learned rules** - use causal model for planning
5. **Hierarchical subgoals** - any token can be a subgoal
