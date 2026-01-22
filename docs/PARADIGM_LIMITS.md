# Production Rule Paradigm: Limits and Extensions

## What We Built
A pattern-based rule learner:
```
Pattern (token set) → Effect (token changes) @ confidence
```

## What It Handles Well ✓
- Direct cause-effect (action → immediate result)
- Conditional effects (context + action → result)  
- Probabilistic transitions (confidence = learned probability)
- Multiple independent effects

## What Requires Workarounds ⚠️

| Challenge | Workaround | Cost |
|-----------|------------|------|
| Temporal patterns | History tokens (`last_action_A`) | Manual encoding |
| Counting/thresholds | Bucket tokens (`gems_enough`) | Domain knowledge |
| Relative comparisons | Comparison tokens (`stronger`) | Pre-computation |
| Delayed effects | Timer tokens (`poison_timer_3`) | Explicit state |

## What Fundamentally Breaks ✗

### 1. Arithmetic Relationships
```
WANT:   damage = weapon_power - armor
HAVE:   {weapon_5, armor_2} → damage_3
        {weapon_5, armor_3} → damage_2
        {weapon_6, armor_2} → damage_4
        ... (N×M rules vs 1 formula)
```

### 2. Variable Binding
```
WANT:   attack_X → X.hp decreases
HAVE:   attack_goblin → goblin_hp decreases
        attack_orc → orc_hp decreases
        ... (separate rule per target)
```

### 3. Negation
```
WANT:   NOT has_shield → double_damage
HAVE:   ??? (can only match presence, not absence)
```

### 4. Rule Composition
```
WANT:   fire + ice → steam (general principle)
HAVE:   {fire, ice, add_fire} → steam
        {fire, ice, add_ice} → steam
        ... (enumerate all trigger paths)
```

## The Real Insight

**All the "intelligence" is in the tokenization.**

The learner finds correlations between pre-defined features.
If the features don't exist, the rules can't be learned.

```
┌─────────────────────────────────────────────────────────────┐
│  APPEARANCE              │  REALITY                         │
├─────────────────────────────────────────────────────────────┤
│  "Learning rules"        │  Memorizing (pattern,outcome)    │
│  "Generalizing"          │  Only if tokens pre-generalize   │
│  "Understanding"         │  Correlation, not causation      │
│  "Composing"             │  Combinatorial enumeration       │
└─────────────────────────────────────────────────────────────┘
```

## Scaling Problem

| Weapons | Armors | Combinations | Rules Needed |
|---------|--------|--------------|--------------|
| 5 | 3 | 20 | 163 |
| 10 | 5 | 60 | 435 |
| 20 | 10 | 220 | 1,459 |
| 50 | 20 | 1,050 | 5,649 |

Rules scale with **state space**, not **concept complexity**.

## Levels of Abstraction Needed

### Level 1: Smart Tokenization (current)
Pre-compute derived features. Requires domain knowledge.

### Level 2: Parameterized Rules
`{weapon_X, fighting_Y} → killed_Y if damage(X) > hp(Y)`
Requires: variables, functions, constraints

### Level 3: Functional Rules  
`damage = max(1, attack - defense)`
Requires: symbolic regression, program synthesis

### Level 4: Compositional Rules
Chain rules, resolve conflicts, temporal ordering
Requires: forward inference engine

### Level 5: Meta-Rules
"Attack actions decrease target HP" (general principle)
Requires: abstraction, analogy, transfer

## Hybrid LLM + Pattern Learner

The path forward:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM (hypothesis generator)                │
│  "What features should we track?"                           │
│  "Damage might be attack - defense"                         │
│  "This looks like type effectiveness"                       │
└────────────────────────┬────────────────────────────────────┘
                         │ hypotheses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Pattern Learner (hypothesis tester)          │
│  "Does this rule hold? Confidence: 92%"                     │
│  "Counter-examples: ..."                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ verified rules + anomalies
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM (refinement)                        │
│  "Counter-examples suggest ability modifiers..."            │
│  "Add 'has_levitate' as a feature"                         │
└─────────────────────────────────────────────────────────────┘
```

**LLM provides:**
- Tokenization suggestions
- Formula hypotheses
- Abstraction/generalization
- Compositional reasoning

**Pattern learner provides:**
- Hypothesis verification
- Confidence quantification
- Anomaly detection
- Exhaustive correlation search

## For Pokemon

**Will work:**
- `{pikachu, thunderbolt, vs_water} → super_effective`
- Individual move/type combinations

**Won't work:**
- "Electric is strong against Water" (general rule)
- Damage formula without pre-computed tokens

**Need:**
- Pre-computed type effectiveness tokens
- LLM to suggest what features matter
- Hybrid approach for formula discovery
