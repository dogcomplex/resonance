# V9 Hierarchical Learner - Practical Assessment

## Validated Capabilities

### ✓ Deterministic Board Games
| Game | Seen F1 | Unseen F1 | Notes |
|------|---------|-----------|-------|
| TicTacToe | 100% | 84% | Perfect on seen |
| Connect Four (4x4) | 100% | 71% | Perfect on seen |

### ✓ Puzzle Games with Generalization  
| Game | Same Layout | New Layout | Notes |
|------|-------------|------------|-------|
| Sokoban | 96.8% | **96.4%** | Relative tokens enable transfer! |

### ✓ Crafting/Inventory Systems
| Game | Seen F1 | Notes |
|------|---------|-------|
| Crafting System | 98.2% | Learns recipes perfectly |

### ✓ Probabilistic Combat
| Game | Seen F1 | Notes |
|------|---------|-------|
| Combat (damage rolls) | 81.6% | Correctly tracks distributions |

### ✓ Grid Navigation (per-seed)
| Game | Seen F1 | Notes |
|------|---------|-------|
| Empty MiniGrid | 100% | Perfect |
| FourRooms | 97.6% | Minor aliasing |
| LavaGap | 99.0% | Near-perfect |

## Key Insights

### 1. Relative Tokenization Enables Generalization
The Sokoban result is remarkable: **96.4% on completely unseen puzzle layouts** because we included `box_rel_dx_dy` tokens. The learner discovered that:
- "Push box when box is at relative position (1,0) and action=right" works universally
- No need to memorize every absolute position

### 2. Probabilistic Environments Work
The combat system correctly tracks that the same (state, action) produces multiple possible outcomes with different frequencies. The 81.6% F1 reflects the inherent uncertainty.

### 3. State Space Size Matters
- Sokoban: 3678 unique states → good coverage
- Connect Four: larger space → lower coverage, relies on general rules

## Design Patterns for New Domains

### Pattern 1: Include Relative Features
```python
# Instead of just absolute positions:
tokens.add(f"box_{x}_{y}")

# Also add relative:
tokens.add(f"box_rel_{box_x - player_x}_{box_y - player_y}")
```

### Pattern 2: Bucket Continuous Values
```python
# Instead of exact HP:
tokens.add(f"hp_{hp}")  # 0-100 = 100 different tokens

# Bucket it:
tokens.add(f"hp_{hp // 10}")  # 0-100 = 10 different tokens
```

### Pattern 3: Include Relevant History
```python
# If "has key" affects door behavior:
tokens.add(f"has_key_{has_key}")
```

## What V9 is Best For

1. **World Model for Planning**: Use with MCTS to simulate futures
2. **Rule Discovery**: "When X, then Y" extraction for game design
3. **Anomaly Detection**: "This transition shouldn't happen"
4. **Opponent Modeling**: If opponent follows consistent strategy

## Limitations

1. **No Planning**: Predicts one step, doesn't plan sequences
2. **No Function Approximation**: Can't interpolate continuous spaces
3. **Requires Tokenization**: Domain-specific feature engineering
4. **Memory Scales with State Space**: Can't handle 10^43 chess positions

## Recommended Applications

| Domain | Fit | Why |
|--------|-----|-----|
| Board game rule learning | ★★★★★ | Primary design target |
| Puzzle game mechanics | ★★★★★ | Sokoban result proves this |
| RPG combat systems | ★★★★☆ | Handles probability |
| Crafting/inventory | ★★★★★ | Perfect for recipes |
| Physics simulation | ★★☆☆☆ | Needs discretization |
| Continuous control | ★★☆☆☆ | Wrong paradigm |
