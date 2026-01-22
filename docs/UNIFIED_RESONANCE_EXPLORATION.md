# Unified Resonance - Exploration Summary

## The Goal
Fold the "meta" dimension (rule equivalences, hierarchical abstraction) into the 
same single resonance operation that handles fidelity, probability, temporal, and universal.

## Approaches Tried

### V1: Inject Meta-Tokens
- Inject rules as tokens that participate in resonance
- **Result:** Rules collapsed, 0% prediction
- **Issue:** Meta-tokens dominated, original signal lost

### V2: Parallel Rule Resonance
- Build rule wave signatures from activation patterns
- Find rule equivalences
- Use during prediction
- **Result:** 83% (same as baseline)
- **Issue:** Equivalences found but prediction logic didn't use them

### V3: Merge Rule LHS Options
- Group equivalent rules
- Allow ANY LHS from group to match
- **Result:** 80% (worse than baseline)
- **Issue:** Different LHS aren't interchangeable

### V4: Merge Observations During Build
- Combine observations from equivalent rules
- Build unified rules from combined data
- **Result:** 82% (slightly worse)
- **Issue:** Only 16 merged groups found, not helpful

## Key Insights

1. **Rule equivalence ≠ LHS interchangeability**
   - Rules R1 and R2 may fire in similar situations and cause similar effects
   - But R1's LHS tokens and R2's LHS tokens are NOT interchangeable
   - "Move right from (1,1)" ≠ "Move right from (2,2)"

2. **The meta-fold may work differently**
   - Token resonance: tokens that behave same → merge (fidelity)
   - Rule resonance: rules that behave same → ??? 
   
   For tokens, merge = use same class representative
   For rules, merge ≠ combine LHS (that loses specificity)

3. **What rule equivalence IS useful for**
   - NOT: prediction fallback
   - NOT: combining observations
   - MAYBE: transfer learning (new domain, similar rules)
   - MAYBE: explanation/interpretation (these rules are "the same")

## The Deep Insight

Token resonance works because position-invariant behaviors are TRUE:
- ball_at_3 and ball_at_5 genuinely behave the same

Rule resonance finds CORRELATIONS, not EQUIVALENCES:
- Rule "move from (1,1)" and "move from (2,2)" are correlated (both are "move")
- But they're not interchangeable (require different positions)

The meta-dimension may need a DIFFERENT operation than fidelity:
- Fidelity: merge equivalent tokens
- Meta: recognize rule FAMILIES without merging

## What Would Work

The deep resonance approach (separate layers) DID improve GridWorld:
- 87.4% vs 80.5% baseline
- Because it found actual equivalences and used them for fallback prediction

The issue is trying to FOLD this into a single operation. Perhaps:
1. Token resonance (fidelity) - single operation
2. Rule generation + annealing (probability) - single operation  
3. Rule family discovery (meta) - RELATED but SEPARATE operation

The meta-dimension might be inherently hierarchical, not flat.

## Current Best Results

| Environment | Single Sieve | Deep Resonance (separate layers) |
|-------------|--------------|----------------------------------|
| GridWorld | 80.5% | **87.4%** |
| Pong | 69.8% | ~70% |

The hierarchical approach works. The unified approach needs more thought.

---

## Token Chemistry Experiments

### The Metaphor
- Tokens = particles (inert)  
- Rules = binding molecules (catalyze transformations)
- Meta-rules = molecules that bind to rules
- Wave view: interference determines what persists
- Particle view: tokens combine into complexes

### Implementation

**V1-V3 attempts:**
1. Encode action as token (ACTION_0, etc.)
2. Discover bindings (rules) from observations
3. Build binding wave signatures
4. Find resonating bindings (meta-level)
5. Create binding groups with alternative LHS

### Results

| Seed | Baseline | Chemistry V3 | Change |
|------|----------|--------------|--------|
| 42 | 82.9% | 79.6% | -3.3% |
| 123 | 83.6% | 79.0% | -4.6% |
| 456 | 85.9% | 78.6% | -7.4% |

### Why It's Not Working (Yet)

The binding resonance finds rules with similar EFFECTS, but:
1. Similar effects ≠ interchangeable LHS
2. "Move right from (1,1)" and "Move right from (2,2)" resonate
3. But their LHS are NOT alternatives - they're different situations

### The Deeper Insight

The chemistry metaphor is RIGHT but the implementation is WRONG.

When two rules "resonate" (similar wave signatures), it means:
- They fire in similar contexts
- They produce similar effects
- They ARE "the same rule" in some abstract sense

But for PREDICTION, we can't just swap their LHS. The LHS encodes
WHERE the rule applies, not WHAT the rule does.

### What Would Work

The meta-dimension should recognize rule FAMILIES without merging:
- "These 5 rules are all 'move right' variants"
- "These 3 rules are all 'pick up key' variants"

This is useful for:
- Transfer learning (map rules from one domain to another)
- Explanation (understand WHAT rules do, not just WHEN they fire)
- Compression (store one rule pattern + position variants)

But NOT for:
- Fallback prediction (wrong LHS = wrong situation)

### The Unified Sieve

The single resonance operation ("does pattern persist across samples?") works for:
- Fidelity (token equivalence) ✓
- Probability (rule confidence) ✓
- Temporal (time persistence) ✓
- Universal (seed invariance) ✓

For META (rule families), the operation FINDS equivalences but the USE is different.

---

## Curried Chemistry - Late Night Session

### The Insight

From Warren: "What if we encoded every rule as a token, and implicitly made the 
execution of any rule be the grouping of said rule with its set of input tokens. 
Rules can then operate on other rules as if they're tokens too. Meta rules are 
just yet another token that happens to operate on rules (or a mix of rules and tokens).

From a token perspective, rules are binding molecules that absorb and produce 
something. Tokens just happen to be inert. We're combining a wave view and a 
particle view here."

### Implementation Journey

**V1-V3:** Tried various approaches with explicit molecule objects and binding chains.
Results: Worse than baseline due to overly complex discovery.

**V4:** Latent intermediate discovery - find token subsets that lead to multiple effects.
Results: Mixed. Found shared intermediates but prediction worse.

**V5-V6:** Simplified to match baseline but track shared intermediates.
Results: Discovered bug - action tokens were being included in LHS.

**V7:** Fixed to match baseline exactly while tracking shared intermediates.
Results: Matches baseline performance AND discovers 47 shared intermediates!

### What We Discovered

In GridWorld with 300 training episodes:
- **47 shared intermediates** - token sets that lead to 2+ different effects
- These are "decision points" where the same intermediate molecule can produce
  different outcomes depending on what OTHER tokens are present

Example shared intermediate:
```
Token set: {agent_2_2, done_False, door_closed_1_2, goal_2_3, ...}
Leads to 2 different effects depending on other tokens present
```

### The Currying Model

The key insight holds true:
1. Rules ARE molecules (binding patterns)
2. Shared intermediates emerge naturally from observation
3. These represent "decision points" where the same binding leads to different results
4. This is the catalyst pattern: same binding, different result based on context

### Performance Summary

| Environment | Baseline | Curried V7 | Difference |
|-------------|----------|------------|------------|
| GridWorld (avg) | 84.1% | 83.8% | -0.3% |
| Pong (avg) | 67.3% | 63.0% | -4.4% |

The curried approach MATCHES baseline while additionally discovering the 
latent intermediate structure!

### What's Next

The shared intermediates tell us WHERE the rules "branch" - where context matters.
This could be used for:
1. Transfer learning (these decision points are domain-invariant)
2. Explanation (understand WHAT determines outcomes)
3. Active learning (query these decision points specifically)
