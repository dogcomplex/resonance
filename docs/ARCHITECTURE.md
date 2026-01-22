# Token-Based State Machine Learning Architecture

## Vision

Learn game rules from observations alone, progressing from simple games to complex video games like Pokemon Red.

## Core Abstraction

```
State = FrozenSet[str]  # Unordered set of tokens
Action = FrozenSet[str]  # Optional action tokens
Rule = (pattern, effect, confidence)
```

**Everything is tokens.** No position, no ordering, no special structure.

## Examples

### TicTacToe
```python
state = {"p0_X", "p1_empty", "p2_O", "p3_empty", ...}
action = {"move_p4_X"}  # or implicit from state
output = {"label_winX"} or next_state
```

### Mini Dungeon
```python
state = {"at_x1", "at_y2", "hp_8", "has_sword", "see_monster"}
action = {"action_fight"}
output = {"at_x1", "at_y2", "hp_6", "has_sword", "-see_monster"}
```

### Pokemon (hypothetical)
```python
state = {
    "hp_25", "pokemon_pikachu", "map_pallet",
    "menu_closed", "npc_north", "grass_south"
}
action = {"press_A"}
output = {..., "menu_open", "-menu_closed"}
```

## Learning Process

1. **Observe transitions**: (before, action, after)
2. **Track co-occurrence**: pattern -> effect counts
3. **Extract rules**: high-confidence associations
4. **Predict**: apply rules to new states

## Key Capabilities

| Capability | Description |
|------------|-------------|
| Pattern discovery | Find token combinations that predict effects |
| Partial observability | Predict with missing tokens |
| Minimal rules | Remove redundant patterns |
| Probabilistic | Handle stochastic transitions |

## File Structure

```
few_shot_algs/
├── clean_learner.py      # Position-based (for comparison)
├── token_learner.py      # Token-based (general)
├── active_learning.py    # Query strategies
└── adaptive_learner.py   # Pattern size adaptation
```

## Progression Plan

1. ✅ TicTacToe (deterministic, small state)
2. ✅ Mini Dungeon (multi-attribute, conditional effects)
3. 🔲 Text adventure (larger state, inventory, NPCs)
4. 🔲 Simple roguelike (procedural, randomness)
5. 🔲 Pokemon Red (visual tokens from slice-finder)

## Integration with Slice-Finder

The slice-finder module produces tokens from visual frames:
- Sprite detection → "sprite_pikachu_x45_y120"
- UI elements → "healthbar_50pct", "menu_items"
- Text OCR → "text_PIKACHU", "text_LV25"

These become the input vocabulary for the token learner.

## Next Steps

1. Add probabilistic rules (for random encounters, etc.)
2. Add temporal patterns (sequences of observations)
3. Build text adventure test environment
4. Design slice-finder → token interface

---

## Tested Environments

### 1. TicTacToe (Deterministic, Small State)
- **Tokens**: 9 position tokens (p0_X, p0_O, p0_empty, ...)
- **Rules learned**: 8 win lines for each player
- **Result**: 100% accuracy with ~1000 observations

### 2. Mini Dungeon (Conditional Effects)
- **Tokens**: Position, HP, inventory, room contents
- **Rules learned**: Movement, combat, item pickup
- **Key finding**: Conditional rules like "has_sword + fight → kill"

### 3. Mini RPG (Complex Conditional)
- **Tokens**: 61 unique tokens
- **Rules learned**: 10,000+ rules (many redundant)
- **Key finding**: Context-dependent combat outcomes

```
With sword vs rat:   -> kills (100%)
Without sword vs rat: -> trades blows, loses HP
```

## Key Results

### Conditional Effects Work!

The learner correctly discovers that outcomes depend on context:

```
{action_attack, has_sword, fighting_rat} → -fighting_rat (kill)
{action_attack, no_sword, fighting_rat} → -hp_N (take damage)
```

This is crucial for Pokemon where:
- Type advantages change damage
- Items change outcomes
- Status effects modify behavior

### Vocabulary Scales

| Environment | Tokens | Rules | Transitions |
|-------------|--------|-------|-------------|
| TicTacToe | 28 | ~100 | 5,890 |
| Mini RPG | 61 | 10,000+ | 50,000+ |
| Pokemon (est.) | 1000+ | ? | millions |

### Next Challenges

1. **Rule compression** - Many rules are redundant
2. **Minimal patterns** - Find simplest explanations
3. **Probabilistic** - Handle random encounters
4. **Temporal** - Multi-step dependencies

---

## Benchmark Results (Updated)

### Environment Comparison

| Metric | TicTacToe | MiniRPG | Roguelike |
|--------|-----------|---------|-----------|
| Vocabulary | 41 tokens | 58 tokens | 86 tokens |
| State size | ~10 tokens | ~10 tokens | ~12 tokens |
| Actions | 9 | ~12 | ~15 |
| Rules (raw) | 4,700 | 9,400 | 2,000 |
| Rules (compressed) | 360 | 700 | 760 |
| F1 @ 100 eps | 96.1% | 96.1% | 86.2% |
| F1 @ 500 eps | 97.3% | 97.3% | 86.3% |
| Precision | 98%+ | 98%+ | 95%+ |
| Recall | 96%+ | 96%+ | 78% |

### Key Findings

1. **Deterministic vs Stochastic**
   - TicTacToe/MiniRPG: 97%+ F1 (deterministic)
   - Roguelike: 86% F1 (has randomness)

2. **Precision is Robust**
   - All environments: 95%+ precision
   - Predictions are reliable for decision-making

3. **Recall Affected by Randomness**
   - Random effects (flee success, ambushes) lower recall
   - Can't learn rules for inconsistent outcomes

4. **Fast Convergence**
   - 100 episodes: already good performance
   - 500 episodes: near-optimal
   - Diminishing returns after

### Files

```
few_shot_algs/
├── token_learner.py    # Core learner
├── clean_learner.py    # Position-based (no cheating)
├── mini_rpg.py         # Simple RPG environment
├── roguelike.py        # Complex RPG environment
└── active_learning.py  # Query strategies
```

---

## Benchmark Results

### Environment Comparison

| Environment | Type | Vocabulary | F1 Score | Convergence |
|-------------|------|------------|----------|-------------|
| **TicTacToe** | Deterministic | 41 | ~97% | 500 eps |
| **MiniRPG** | Deterministic | 58 | 97.2% | 500 eps |
| **MiniRogue** | Probabilistic | 84 | 72.4% | 250 eps |

### Detailed MiniRogue Results

| Episodes | Transitions | Rules | Compressed | Precision | Recall | F1 |
|----------|-------------|-------|------------|-----------|--------|-----|
| 100 | 6,553 | 4,902 | 1,047 | 87.4% | 62.0% | 72.6% |
| 500 | 33,297 | 10,776 | 1,619 | 93.2% | 59.1% | 72.3% |
| 2000 | 135,909 | 15,520 | 1,378 | 95.7% | 58.2% | 72.4% |

### Key Insight: Probabilistic vs Deterministic

**Why MiniRogue has lower recall:**
- Random combat outcomes (15% miss chance)
- Random encounters (25% ambush rate)
- Probabilistic status effects (30% poison)
- These effects don't consistently happen → low confidence → filtered out

**This is correct behavior!**
- High precision (96%) = predictions are reliable
- Lower recall (58%) = conservative about uncertain effects
- The confidence values ARE the learned probabilities

### Rule Examples

**Deterministic (100% confidence):**
```
{action_equip_sword} → +weapon_sword, +armed
{action_move_east, at_0_0} → +at_1_0, -at_0_0
```

**Conditional (high confidence):**
```
{action_attack, armed, fighting_rat} → killed (85%)
{action_attack, unarmed, fighting_rat} → killed (40%)
```

**Probabilistic (learned probability):**
```
{action_flee} → -in_combat (70%)  # matches 70% flee success
{action_attack, fighting_spider} → +poisoned (24%)  # matches 30% * 80%
```
