# Hierarchical Probabilistic Learner

## Overview

A rule learning system that combines:
- **Optimistic crystallization** - Learn fast, correct on surprise
- **Probabilistic rules** - Handle stochasticity naturally
- **Hierarchical specificity** - General and specific rules coexist
- **Raw tokenization** - No domain knowledge (no cheating)

## Key Results on MiniGrid

### DoorKey-5x5 (500 episodes)
| Metric | Value |
|--------|-------|
| Vocabulary | 31 tokens |
| Total rules | 4,602 |
| Deterministic | 1,018 |
| Probabilistic | 1,808 |
| Stable | 3,935 |

### Key Discoveries (from raw numbers only!)

**Pickup (action 3):**
```
{front_t5} + pickup → +carry_t5 (100%)
{front_t5} + pickup → +carry_c4 (100%)
```
The learner discovered that "front_t5" (type 5 = key) + pickup results in carrying!

**Drop (action 4):**
```
{carry_t5, front_t1} + drop → +carry_none (100%)
{carry_t5, front_t1} + drop → +front_c4 (100%)
```

## Hierarchical Structure

The same effect can have rules at different specificity levels:

| Specificity | Pattern | Probability | Type |
|-------------|---------|-------------|------|
| Low (general) | {front_t4} | 50% | Probabilistic |
| High (specific) | {front_t4, front_s2, carry_t5} | 100% | Deterministic |

**Why both are useful:**
- General rule: "Toggling a door (t4) often opens it (50%)"
- Specific rule: "Toggling a locked door (s2) while carrying a key (t5) always opens it (100%)"

The learner automatically maintains this hierarchy and uses the most specific matching rule for predictions.

## No Domain Knowledge

The raw tokenizer produces tokens like:
- `front_t5` (not "front_key")
- `carry_c4` (not "carrying_yellow_key")
- `door_s2` (not "door_locked")

The learner must discover all semantics from experience:
- Type 5 objects can be picked up
- Type 4 objects can be toggled
- State 2 → state 0 transition requires carrying type 5

## Files

- `hierarchical_learner.py` - Main implementation
- `raw_tokenizer.py` - Domain-agnostic tokenizer
- `probabilistic_learner.py` - Base probabilistic learner
- `optimistic_learner.py` - Fast shatter-based learner

## Usage

```python
from hierarchical_learner import HierarchicalLearner, raw_tokenize

learner = HierarchicalLearner(n_actions=7)

for episode in range(1000):
    state = env.reset()
    tokens = raw_tokenize(state)
    
    for step in range(max_steps):
        action = select_action()
        next_state = env.step(action)
        next_tokens = raw_tokenize(next_state)
        
        learner.observe(tokens, action, next_tokens)
        tokens = next_tokens

# Get rules
for rule in learner.get_minimal_deterministic():
    print(rule)

for rule in learner.get_probabilistic():
    print(rule)
```
