# Production Rule Stress Test Results

## Where the System Breaks

### 1. Grid Size Scaling (Coverage Problem)
Fixed 500 episodes, varying grid size:

| Grid | Positions | F1 Score |
|------|-----------|----------|
| 4×4 | 16 | 100% |
| 8×8 | 64 | 100% |
| 16×16 | 256 | 99.9% |
| 32×32 | 1,024 | 92.6% |
| **64×64** | 4,096 | **67.1%** |

**Breaking point:** When state space >> observations, rare states don't reach min_support.

### 2. Fixed Observation Budget
5,000 observations, varying complexity:

| Grid | Enemies | Weapons | Items | State Space | F1 |
|------|---------|---------|-------|-------------|-----|
| 4 | 3 | 2 | 3 | 768 | 95.1% |
| 8 | 10 | 6 | 8 | 122,880 | 89.4% |
| 16 | 25 | 12 | 20 | 2.4M | 82.7% |
| **20** | 40 | 15 | 30 | **7.7M** | **73.9%** |

**Breaking point:** Recall drops as state space explodes; precision stays high.

### 3. Pattern Size Limit (Conditional Depth)
Rules requiring N conditions to align:

| Conditions | Probability | Triggers/10k | Rule Found? |
|------------|-------------|--------------|-------------|
| 2 | 25% | 2,452 | ✓ YES |
| 3 | 12.5% | 1,228 | ✓ YES |
| 4 | 6.25% | 622 | ✓ YES |
| **5** | 3.1% | 312 | **✗ NO** |
| 6 | 1.6% | 163 | ✗ NO |

**Breaking point:** max_pattern_size=4 means 5+ condition rules can't be captured.

### 4. Noise Token Explosion
Adding random noise tokens:

| Noise Tokens | Rules Generated |
|--------------|-----------------|
| 0 | 3 |
| 5 | 28 |
| 10 | 78 |
| 20 | 253 |
| 30 | 528 |

**Problem:** Rule count explodes, but signal still found. Memory/compute issue, not accuracy.

### 5. Confounded Variables
When two tokens always co-occur:
```
weapon_fire + spell_fire → enemy_dead

Rules found:
  {attack} → +enemy_dead (100%)
  {has_spell_fire} → +enemy_dead (100%)  # WRONG
  {has_weapon_fire} → +enemy_dead (100%) # WRONG
```

**Problem:** Can't distinguish true cause from correlated feature.

### 6. Simpson's Paradox
Weapon is effective overall but LESS effective vs armored enemies:
```
{use_weapon} → +enemy_dead (84%)  # Looks good!
{use_weapon, enemy_armored} → +enemy_dead (40%)  # Actually bad vs armor!
```

**Problem:** Biased sampling leads to misleading aggregate rules.

### 7. State Coverage
With N binary tokens, possible states = 2^N:

| Tokens | Possible States | Seen in 10k | Coverage |
|--------|-----------------|-------------|----------|
| 5 | 32 | 32 | 100% |
| 10 | 1,024 | 1,024 | 100% |
| 15 | 32,768 | 8,518 | 26% |
| 20 | 1M | 9,941 | 0.95% |
| **25** | 33M | 10,000 | **0.03%** |

**Breaking point:** Beyond ~15 tokens, most states are never observed.

## Summary: Hard Limits

| Limit | Threshold | Effect |
|-------|-----------|--------|
| State space | >1M states | Recall drops below 80% |
| Pattern depth | >4 conditions | Rules not capturable |
| Token count | >20 per state | 99%+ states unseen |
| Confounding | Any correlated tokens | False attributions |

## Implications for Pokemon

Pokemon has:
- ~20 position tokens (sprite locations)
- ~10 HP buckets per pokemon
- ~15 type combinations
- ~150 moves
- ~6 party members
- UI state tokens

**Estimated tokens per frame: 50-100**
**Estimated state space: 2^50 = 10^15**

This is WAY beyond what pure pattern matching can handle.
Need:
1. Smart tokenization (reduce dimensionality)
2. Hierarchical tokens (compress related features)
3. LLM hypotheses (focus on likely-relevant patterns)
4. Formula extraction (handle HP math)
