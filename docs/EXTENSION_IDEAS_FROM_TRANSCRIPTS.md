# Extension Ideas from Previous Sessions

These ideas from the Pokemon abstraction sessions directly address the limitations identified in our current architecture. Each needs vetting for **domain knowledge fairness** before implementation.

---

## 1. HIERARCHICAL TOKENS (addresses: state space explosion, compositional generalization)

**From transcript: `2026-01-10-03-04-18-pokemon-abstraction-pipeline-trace.txt`**

### Idea
Instead of flat tokens, use hierarchical `path.to.value` structure:

```
RAW:                          HIERARCHICAL:
  pikachu_hp_45                 pokemon.player.0.hp.medium
  pikachu_x_120                 pokemon.player.0.pos.center
  rattata_hp_12                 pokemon.enemy.0.hp.low
```

### Key Benefit: Rules can match at ANY level
```python
{action.attack} → pokemon.enemy.*.hp.DECREASE  (general)
{action.attack, pokemon.player.type.electric, pokemon.enemy.type.water}
    → pokemon.enemy.hp.DECREASE.super  (specific)
```

### Fairness Check Required
- ⚠️ The hierarchical structure itself encodes domain knowledge (player/enemy, types)
- ✅ FAIR if: structure discovered from data (tokens with similar effects grouped)
- ❌ UNFAIR if: structure hand-coded based on game knowledge

### Implementation Direction
```python
# FAIR: Discover hierarchy through behavioral clustering
def discover_hierarchy(tokens, rules):
    # Group tokens that appear in similar rule contexts
    # Build tree structure bottom-up from behavioral similarity
    pass
```

---

## 2. ROLE-BASED ABSTRACTION (addresses: state space explosion, compositional generalization)

**From transcript: `2026-01-10-03-04-18-pokemon-abstraction-pipeline-trace.txt`**

### Idea
Track ROLES instead of specific instances:

```
RAW:                          ROLE-BASED:
  pikachu_hp_45                 active_pokemon.hp.medium
  active_slot_0                 active_pokemon.type.electric
  rattata_hp_12                 target.hp.low
```

### Key Benefit: Massive compression
- Learn TYPE interactions (15×15=225) not POKEMON interactions (150×150=22,500)

### Fairness Check Required
- ⚠️ Role assignment requires knowing what "active" and "target" mean
- ✅ FAIR if: roles discovered from causal structure (which tokens predict effects)
- ❌ UNFAIR if: roles assigned based on game semantics

### Implementation Direction
```python
# FAIR: Discover roles through causal analysis
def discover_roles(observations):
    # Tokens that are most predictive of effect type become "active"
    # Tokens that most often CHANGE become "target"
    pass
```

---

## 3. DERIVED/COMPUTED TOKENS (addresses: numerical relationships, state space)

**From transcript: `2026-01-10-03-04-18-pokemon-abstraction-pipeline-trace.txt`**

### Idea
Pre-compute relationships as tokens:

```
RAW:                          DERIVED:
  player_atk_55                 stat_advantage.attack  (if atk > def)
  enemy_def_40                  type_matchup.super_effective
  player_spd_90                 turn_order.first
  enemy_spd_45                  hp_ratio.winning
```

### Key Benefit: Encodes relationships as learnable patterns
```python
{action.attack, type_matchup.super_effective} → damage.high
# Don't need to learn fire→grass, water→fire separately
```

### Fairness Check Required
- ⚠️ Knowing WHICH comparisons matter is domain knowledge
- ✅ FAIR if: comparisons discovered from correlation with outcomes
- ❌ UNFAIR if: hand-coded type charts, stat formulas

### Implementation Direction
```python
# FAIR: Discover useful comparisons from data
def discover_derived_tokens(observations):
    # For each pair of numeric tokens, test if their comparison
    # predicts outcomes better than raw values
    # Example: if (A > B) correlates with effect E, add "A_gt_B"
    pass
```

---

## 4. CONTEXT-BASED FILTERING (addresses: state space explosion)

**From transcript: `2026-01-10-03-04-18-pokemon-abstraction-pipeline-trace.txt`**

### Idea
Separate learners per context, each seeing only relevant tokens:

```python
CONTEXTS = {
    "battle": ["pokemon.", "move.", "battle.", "damage."],
    "overworld": ["map.", "player.pos.", "npc."],
    "menu": ["menu.", "cursor.", "item."]
}
```

### Key Benefit: Massive reduction in pattern space
- Full state: 23 tokens → patterns explode
- Filtered: 8 tokens per context → manageable

### Fairness Check Required
- ⚠️ Context detection requires understanding what tokens mean
- ✅ FAIR if: contexts discovered from token co-occurrence patterns
- ❌ UNFAIR if: hand-labeled contexts

### Implementation Direction
```python
# FAIR: Discover contexts through clustering
def discover_contexts(observations):
    # States that share many tokens form a context
    # Tokens that never co-occur belong to different contexts
    pass
```

---

## 5. PATTERN VARIABLES / SCHEMA (addresses: compositional generalization)

**From transcript: `2026-01-09-05-20-19-meta-learning-syntax-extensions.txt`**

### Idea
Rules with variables that match multiple instances:

```
# Define variable over positions
@line := (0,1,2) | (3,4,5) | (6,7,8) | ...

# Rule with variable
p@line[0]_1  p@line[1]_1  p@line[2]_1  →  winX
```

### Key Benefit: One rule covers 8 concrete cases

### Fairness Check Required
- ⚠️ Knowing that "lines" exist is domain knowledge
- ✅ FAIR if: variable bindings discovered from repeated pattern structure
- ❌ UNFAIR if: line definitions hand-coded

### Implementation Direction
```python
# FAIR: Discover schemas through pattern compression
def discover_schemas(rules):
    # Find rules that differ only in token indices
    # Abstract to schema with variable binding
    # Example: if rules for (0,1,2) and (3,4,5) are identical,
    #          create schema with @row variable
    pass
```

---

## 6. TEMPORAL COMPRESSION (addresses: temporal patterns)

**From transcript: `2026-01-10-03-04-18-pokemon-abstraction-pipeline-trace.txt`**

### Idea
Track state deltas, skip intermediate frames:

```
Frame 1: selected_move
Frame 2: animation_playing  
Frame 3: damage_number_shown
Frame 4: hp_bar_decreased

Compress to: {action.attack} → {damage.dealt}
```

### Key Benefit: Skip irrelevant intermediate states

### Fairness Check Required
- ✅ FAIR: This is pure observation compression, no domain knowledge needed
- Just need to detect "stable" states vs "transitional" states

### Implementation Direction
```python
# FAIR: Detect transitional states
def detect_stable_states(observations):
    # States that persist for multiple frames are stable
    # States that appear briefly between stable states are transitional
    # Learn rules only between stable states
    pass
```

---

## 7. PROTOTYPE MATCHING (addresses: state space coverage)

**From transcript: `2026-01-10-03-04-18-pokemon-abstraction-pipeline-trace.txt`**

### Idea
Cluster similar states into prototypes:

```
Prototype "healthy_advantage": {hp.high, type.advantage, status.ok}
Prototype "risky_attack": {hp.low, speed.faster, type.neutral}

10,000 observed states → 50 prototypes
New state → nearest prototype → apply prototype's rules
```

### Key Benefit: Handles unseen states through similarity

### Fairness Check Required
- ✅ FAIR: Clustering is pure data analysis, no domain knowledge needed
- Just need a good distance metric for states

### Implementation Direction
```python
# FAIR: Cluster states by behavioral similarity
def cluster_states(observations):
    # States that lead to similar effects are similar
    # Use effect distribution as feature vector
    # Cluster to find prototypes
    pass
```

---

## Summary: What's Fair vs Unfair

| Extension | Fair Version | Unfair Version |
|-----------|--------------|----------------|
| Hierarchical tokens | Discovered from behavioral similarity | Hand-coded structure |
| Role abstraction | Discovered from causal analysis | Hand-labeled roles |
| Derived tokens | Discovered from outcome correlation | Hand-coded formulas |
| Context filtering | Discovered from co-occurrence | Hand-labeled contexts |
| Pattern variables | Discovered from rule compression | Hand-coded schemas |
| Temporal compression | Discovered from state persistence | Hand-labeled frames |
| Prototypes | Discovered from behavioral clustering | Hand-picked exemplars |

---

## Next Steps

1. **Pick one extension** to implement first
2. **Design fair discovery algorithm** (no domain knowledge)
3. **Test on current environments** (TicTacToe, MiniGrid)
4. **Verify accuracy improvement** without cheating
5. **Document what was discovered** vs what was hand-coded

Recommendation: Start with **Derived Tokens** (numerical comparison discovery) or **Context Filtering** (automatic context detection) as they're most clearly automatable.
