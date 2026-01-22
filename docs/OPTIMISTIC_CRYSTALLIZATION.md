# Optimistic Crystallization: Aggressive Rule Learning

## The Core Idea

**Traditional approach**: Wait for many observations, high confidence → Safe but slow

**Optimistic approach**: Crystallize after 2 observations, shatter on surprise → Fast and adaptive

This is how humans learn: "All swans are white" until you see a black one.

## How It Works

```
1. OBSERVE: See (state, action, next_state) transition
2. CHECK: Do any crystallized rules predict wrong?
   - If yes: SHATTER the rule, record counter-example
3. CRYSTALLIZE: From just 2 observations, form a rule
4. ABLATE: Remove unnecessary tokens from pattern
5. REPEAT: Rules that survive are likely correct
```

## Key Innovation: Shatter-and-Refine

When a rule makes a wrong prediction:

```python
# Rule: {pos_1} + pickup → +has_key
# Counter-example: {pos_1, has_key} → no change

# SHATTER: Remove the rule
# RECORD: {pos_1, has_key} is a negative example
# REFINE: New rule needs to distinguish:
#   Positive: {pos_1} (no key)
#   Negative: {pos_1, has_key} (already has key)
# Result: {pos_1, NOT has_key} + pickup → +has_key
```

## Results

### Simple Game (DoorKey)
- **6 shatters** across 300 episodes
- **92% prediction accuracy**
- Clean minimal rules discovered

### MiniGrid Empty-5x5
- **46 shatters** across 500 episodes
- **23 rules** crystallized
- Geometric rules like `{left_goal} + left → +right_goal`

### MiniGrid DoorKey-5x5
- **78 shatters** across 500 episodes
- **79 rules** crystallized
- Perfect pickup rules: `{front_key} + pickup → +carrying_key`

## Comparison

| Aspect | Traditional | Optimistic |
|--------|-------------|------------|
| Min observations | 5-10 | 2 |
| Time to first rule | Slow | Instant |
| Wrong predictions | Rare | Expected |
| Adaptation speed | Slow | Fast |
| Best for | Static rules | Dynamic learning |

## When to Use

**Optimistic is better when:**
- You can afford mistakes (they're just predictions)
- Speed matters more than safety
- The environment is mostly deterministic
- You want to learn from minimal data

**Traditional is better when:**
- Mistakes are costly
- You have plenty of time
- Rules are complex/stochastic
- You need high confidence

## The Philosophy

> "It's easier to ask forgiveness than permission"

Rules that are **wrong** will be **shattered** quickly.
Rules that are **right** will **survive** and strengthen.

The system is **self-correcting** - bad rules don't last long.

## Implementation

```python
from optimistic_learner import OptimisticLearner

learner = OptimisticLearner(n_actions=7, min_observations=2)

for state, action, next_state in transitions:
    surprises = learner.observe(state, action, next_state)
    
    if surprises:
        # Learned something! Old belief was wrong.
        for rule in surprises:
            print(f"Shattered: {rule}")

# Get learned rules
for rule in learner.get_rules():
    print(rule)
```

## Files

- `optimistic_learner.py` - Clean implementation
- `abstract_tokenizer_v2.py` - MiniGrid tokenization
- `crystallizing_system.py` - Traditional (careful) crystallizer
