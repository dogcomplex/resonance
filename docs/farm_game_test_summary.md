# Farm Game Test: What We're Actually Testing

## The Honest Answer to "How is it exploring?"

**It's NOT exploring in the RL sense.** Here's what's actually happening:

### What We Did (and Why)

1. **Direct Rule Sampling**: We generate synthetic states where each of the 380 rules can fire
2. **Oracle-based state construction**: For each rule, we construct a minimal state that satisfies the LHS requirements
3. **Observe transitions**: The learner sees (before_state, action, after_state) tuples
4. **Test prediction**: We check if the learner can predict outcomes for held-out samples

### What This Tests

✅ **Rule Learning**: Can the learner discover rules from observations?
✅ **Anonymization**: Does it work without domain knowledge?
✅ **Probabilistic Rules**: Can it handle stochastic outcomes?
✅ **Complex Rules**: 380 rules with priorities, ranges, consume-all, etc.

### What This Does NOT Test

❌ **Exploration**: The learner doesn't find these states itself
❌ **Planning**: No goal-directed search to reach new states
❌ **Credit Assignment**: No learning which paths lead to rewards
❌ **Reachability**: We don't verify these states are actually reachable in gameplay

### Results

| Type | Count | F1 Score |
|------|-------|----------|
| Deterministic | 524-552 | 99.9% |
| Probabilistic | 186-219 | 77-79% |
| **Overall** | 738-743 | **95-96%** |

### The Exploration Problem (Unsolved)

In real gameplay, reaching states like "ID_Rent6" (requires 🏦6️⃣ + 💰40000) would require:
1. Mining ore → smelting bars → crafting tools
2. Farming → harvesting → selling
3. Building machines → processing goods → more selling
4. Managing day/night cycles
5. Paying rent 5 times before reaching level 6

This is a **credit assignment + planning** problem that requires:
- Intrinsic motivation (curiosity)
- Hierarchical planning
- Goal-directed exploration

Our current test sidesteps this by directly instantiating states.

### Conclusion

The test validates that **the rule learning mechanism works** on anonymous tokens.
It does NOT validate that an agent could **discover these rules through gameplay**.

For a complete world model, we'd need:
1. ✅ Rule learning (tested here)
2. ❌ Curiosity-driven exploration (not implemented)
3. ❌ Planning with learned model (not implemented)
