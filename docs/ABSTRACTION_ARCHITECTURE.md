# Abstraction Architecture for Scalable Rule Learning

## The Problem
Raw game state has 50-100 tokens → 10^15+ possible states → unlearnable

## Solution: Multi-Layer Abstraction Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW GAME STATE                               │
│  pixel_x_120, pixel_y_80, hp_45, type_id_25, move_id_87, ...    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LAYER 1: CONTEXT DETECTION                       │
│  Detect: battle | overworld | menu | dialogue                   │
│  Filter: Keep only context-relevant tokens                      │
│  Result: 50 tokens → 10-15 relevant tokens                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LAYER 2: DERIVED TOKENS                          │
│  Compute relationships:                                         │
│    - type_matchup.{super|neutral|weak|immune}                   │
│    - speed.{faster|slower}                                      │
│    - hp_ratio.{winning|losing|tied}                            │
│    - stat_advantage.{yes|no}                                    │
│  Result: Raw values → semantic relationships                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LAYER 3: HIERARCHICAL TOKENS                     │
│  Structure: role.entity.attribute.value                         │
│    pokemon.player.active.type.electric                          │
│    pokemon.enemy.active.hp.low                                  │
│  Enables: Matching at any abstraction level                     │
│  Result: Flat tokens → structured tokens                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LAYER 4: ROLE ABSTRACTION                        │
│  Replace specific entities with roles:                          │
│    pikachu_hp_45 → active_pokemon.hp.medium                     │
│    squirtle_hp_30 → target.hp.low                              │
│  Result: 150 pokemon × 150 = 22,500 → 15 types × 15 = 225      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONTEXT-SPECIFIC RULE LEARNERS                      │
│                                                                  │
│  Battle Learner (~10 tokens/obs, ~500 rules)                    │
│    {type_matchup.super, action.attack} → damage.high            │
│                                                                  │
│  Overworld Learner (~8 tokens/obs, ~200 rules)                  │
│    {tile.grass, action.step} → encounter.possible               │
│                                                                  │
│  Menu Learner (~6 tokens/obs, ~100 rules)                       │
│    {cursor.on_item, action.select} → item.used                  │
└─────────────────────────────────────────────────────────────────┘
```

## Compression Ratios

| Stage | Before | After | Ratio |
|-------|--------|-------|-------|
| Context filter | 50 tokens | 12 tokens | 4x |
| Derived tokens | 12 raw values | 8 relations | 1.5x |
| Role abstraction | 150 pokemon | 15 types | 10x |
| **Combined** | **10^15 states** | **~10^4 states** | **10^11x** |

## Key Derivations for Pokemon

### Type Matchup (most important!)
```python
def derive_type_matchup(move_type, target_type):
    effectiveness = TYPE_CHART.get((move_type, target_type), 1.0)
    if effectiveness >= 2.0: return "super_effective"
    if effectiveness <= 0.5: return "not_very_effective"
    if effectiveness == 0: return "no_effect"
    return "neutral"
```

### HP Buckets
```python
def derive_hp_bucket(current, max_hp):
    pct = current / max_hp
    if pct > 0.7: return "high"
    if pct > 0.3: return "medium"
    if pct > 0: return "low"
    return "fainted"
```

### Battle Advantage
```python
def derive_advantage(player_stats, enemy_stats):
    tokens = set()
    if player_stats["speed"] > enemy_stats["speed"]:
        tokens.add("speed.advantage")
    if player_stats["attack"] > enemy_stats["defense"]:
        tokens.add("attack.advantage")
    return tokens
```

## Implementation Strategy

1. **Start with context detection** - biggest immediate win
2. **Add HP bucketing** - reduces numeric explosion
3. **Add type matchup derivation** - encodes the type chart
4. **Add role abstraction** - generalizes across pokemon
5. **Add hierarchical structure** - enables multi-level rules

## Expected Outcome

With full abstraction pipeline:
- ~10-15 tokens per observation (down from 50-100)
- ~1000 total rules across all contexts (down from millions)
- 95%+ coverage of observable states
- Rules generalize to unseen pokemon/moves

## Where LLM Still Needed

1. **Damage formula** - HP math is arithmetic, not pattern matching
2. **Status effect interactions** - complex multi-turn effects
3. **Ability exceptions** - Levitate ignores Ground, etc.
4. **New mechanics discovery** - things we didn't pre-encode

Pattern learner handles the regular cases.
LLM handles exceptions and formula inference.
