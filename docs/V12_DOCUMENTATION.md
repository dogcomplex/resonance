# Hierarchical Learner V12 - Dream-Ready World Model

## Epistemological Framework

V12 implements three modes of reasoning, mirroring classical epistemology:

| Mode | Direction | Question | Use Case |
|------|-----------|----------|----------|
| **DEDUCTION** | Forward | State + Action → ? | Predict outcomes, simulate |
| **ABDUCTION** | Backward | ? → Desired Effect | Goal-directed planning |
| **INDUCTION** | Middle | State_X → ? → State_Y | Explain observations, learn |

All three modes traverse the **same learned rule base** - just in different directions.

## The Three Reasoning Modes

### 1. Deduction (Forward Prediction)

```python
# Basic prediction
effects = learner.predict(state, action)

# With confidence and alternatives
prediction = learner.predict_with_confidence(state, action)
# Returns:
#   prediction.effects      - Most likely effects
#   prediction.confidence   - 0.0-1.0 confidence
#   prediction.source       - "exact", "rule", "induced", "unknown"
#   prediction.alternatives - Other possible outcomes
```

### 2. Abduction (Backward Reasoning)

```python
# "How do I get +money?"
result = learner.abduce("+💰_10")
# Returns:
#   result.enabling_actions  - Actions that produce this
#   result.required_tokens   - State needed
#   result.confidence        - Reliability

# Specific queries
learner.what_produces("💰")     # Actions that create money
learner.what_consumes("💧")     # Actions that use energy
learner.what_requires(action)   # Prerequisites for action
```

### 3. Induction (Path Inference)

**This is the key addition** - finding the "missing middle" between observed states.

```python
# Find likely rule sequence between states
paths = learner.induce_path(before_state, after_state, max_steps=10)
# Returns: [(rule_sequence, confidence), ...]
# Where each rule is (consumed_tokens, produced_tokens)

# Human-readable explanation
explanation = learner.explain_transition(before_state, after_state)
# Output:
# "Inferred path (85% confidence):
#    1. -{💧} → +{🪓🎯}
#    2. -{🌳, 🪓🎯} → +{🌳🟤, 🤎}
#    3. -{🤎} → +{💰}"

# Estimate minimum steps via currying principle
min_steps = learner.minimum_steps_estimate(before, after)
```

**How Induction Works:**

1. Compute diff: what tokens disappeared? what appeared?
2. Search for rule sequences that explain the diff
3. Handle intermediate tokens (created then consumed within path)
4. Score by: confidence × fewer_steps × known_rules_preferred
5. Use currying principle: N token changes ≈ N primitive operations

This is the **optimal packing problem** - finding which combination of known rules best explains the observed transformation.

## Dreaming & Planning Features

### Imagination Rollouts

```python
# Imagine future states
states, confidence, certain_steps = learner.imagine_trajectory(
    initial_state,
    action_sequence,
    confidence_threshold=0.3
)

# Full dream with policy
trajectory, final_conf = learner.dream_rollout(
    initial_state,
    policy_function,
    max_steps=50,
    min_confidence=0.2
)
```

### Exploration Guidance

```python
novelty = learner.novelty_score(state, action)      # 0-1, higher = more novel
exp_value = learner.exploration_value(state, action) # Novelty + uncertainty + info gain
```

### Temporal Dependencies

```python
temporal = learner.get_temporal_dependencies(action)
# temporal.prerequisites      - What usually precedes this
# temporal.min_steps_from_start - How early/late it appears
# temporal.typical_sequence   - Common path to this action
```

## Integration with RL Explorer

```python
# In dreamer-style training loop:

# 1. Collect real experience
for state, action, next_state in real_rollout:
    world_model.observe(state, action, next_state)

# 2. Dream to generate synthetic experience
dream_batch = []
for _ in range(dream_count):
    traj, conf = world_model.dream_rollout(start, policy)
    if conf > threshold:
        dream_batch.extend(traj)

# 3. Train policy on real + dreamed
policy.train(real_batch + dream_batch)

# 4. Use induction to understand what happened
for before, after in mysterious_transitions:
    explanation = world_model.explain_transition(before, after)
    # Use to improve rule confidence or discover gaps
```

## Performance (Farm Game, 380 Rules)

| Metric | Result |
|--------|--------|
| Deterministic prediction | 99.9% F1 |
| Probabilistic prediction | 77-79% F1 |
| Money-producers identified | 89/89 (100%) |
| Patterns discovered | 23 |
| Induction working | ✓ |

## Key Insight

The three reasoning modes are not separate systems - they're **different traversals of the same rule graph**:

- **Deduction**: Walk forward from current state
- **Abduction**: Walk backward from goal state  
- **Induction**: Find the path connecting two known states

This unified view means improvements to rule learning benefit all three modes simultaneously.

---

## Strategic Use of Reasoning Modes

### Summary Table

| Mode | You KNOW | You NEED | Best For |
|------|----------|----------|----------|
| **DEDUCTION** | State + Action | Outcome | Simulation, Dreaming |
| **ABDUCTION** | Goal + Rules | Triggering State | Planning, Goal pursuit |
| **INDUCTION** | Before + After | Rule sequence | Understanding, Rule discovery |

### When to Use Each Mode

**ABDUCTION** - Use when:
- You have a clear goal (e.g., "I want diamond_pickaxe")
- You're confident about the rules
- You need to find what state/action sequence achieves the goal
- Working backwards from objectives

```python
# "How do I get diamond_pickaxe?"
result = learner.abduce("+diamond_pickaxe")
# Returns: enabling_actions, required_tokens
```

**INDUCTION** - Use when:
- You observed a transition (before → after)
- You have some rules but uncertain about others
- You want to understand what happened
- You want to discover missing rules

```python
# "What rules explain this transition?"
explanation = learner.explain_transition(before, after)
# If gaps exist → you've found rule hypotheses!
```

### Rule Discovery via Induction Gaps

Key insight: **Gaps in induction explanations ARE rule hypotheses!**

```
Observed:  {A, B, D} → {E}
Explained: {A, B} → {C} (known rule)
Gap:       {C, D} → {E} (THIS IS A RULE HYPOTHESIS!)
```

The gap tells us:
- Something consumed {C, D}
- Something produced {E}
- This is evidence of an unknown rule!

### The Verification Loop

```
Observe X → Y
     ↓
INDUCTION: What rules explain X → Y?
     ↓
Found gaps? → Rule hypotheses!
     ↓
DEDUCTION: Test hypotheses - do they predict correctly?
     ↓
Update rule confidence
     ↓
ABDUCTION: Plan using confident rules
     ↓
Execute, observe results
     ↓
Back to INDUCTION
```

This is the **scientific method** applied to world modeling!

---

## Backtest Results: Induction Performance

| Game | 1-Step | Multi-Step | Cross-Episode |
|------|--------|------------|---------------|
| TicTacToe | 50/50 | 10/10 | 3/5 |
| Crafting | 50/50 | 10/10 | 0/5 |
| Sokoban | 50/50 | 10/10 | 5/5 |
| Snake | 45/45 | 8/8 | 4/5 |
| LightSwitch | 49/49 | 10/10 | 5/5 |
| 2048 | 50/50 | 9/10 | 0/5 |

Single and multi-step induction works nearly perfectly. Cross-episode is harder because states diverge significantly.

---

## Failure Analysis & Convergence

### Why Induction Fails

| Cause | Description | Solution |
|-------|-------------|----------|
| **Incomplete Coverage** | Path between states uses unobserved rules | Directed exploration |
| **Probabilistic Rules** | Few samples → poor distribution estimate | More observations |
| **Large State Spaces** | Exponential states, impossible to cover all | Hierarchical generalization |

### Cross-Episode Failures

The key insight: **different episodes explore different subgraphs**.

```
Episode 1: A → B → C (explored)
Episode 2: A → D → E (explored)

Query: B → E
Problem: No observed rules connect these subgraphs!
```

### Convergence Properties

| Rule Type | Convergence |
|-----------|-------------|
| Deterministic | 1-3 observations |
| Probabilistic | O(1/√n) error with n samples |
| Coverage | Grows linearly with directed exploration |

### Directed Exploration Results

| Strategy | Cross-Episode Success | Rules Learned |
|----------|----------------------|---------------|
| Random | 3/10 | 1064 |
| Coverage-focused | 4/10 | 1279 |

**The denser the observed transition graph, the better induction works.**

---

## Scientific Method in World Modeling

The three reasoning modes implement the scientific method:

1. **INDUCTION**: "I observed X→Y but can't explain it"
2. **HYPOTHESIS**: Gap = candidate rule (what's missing)
3. **ABDUCTION**: "What state would test this hypothesis?"
4. **DEDUCTION**: "How do I reach that state from here?"
5. **EXPERIMENT**: Execute, observe result
6. **UPDATE**: Confirm or refute hypothesis
7. **REPEAT**: Until model converges

### Metrics for Progress

- **Coverage**: % of (state, action) pairs observed
- **Induction Success**: % of transitions explainable
- **Hypothesis Count**: Unconfirmed rules (lower = better)
- **Prediction Accuracy**: Deduction F1

### When is the Model "Complete"?

Practically complete when:
- All observed transitions explainable
- All hypotheses tested
- Prediction accuracy plateaus
- Novelty scores approach zero
