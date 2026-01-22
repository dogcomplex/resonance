# Hierarchical Learner - Integration Specification

## What the Learner Actually Is

The learner is **a hypothesis generator** - nothing more.

```
(before_state, after_state, action) → [candidate_rule₁, candidate_rule₂, ...]
```

It does NOT test rules. Testing is delegated to your existing engine.

## What You Already Have

| Component | Input | Output |
|-----------|-------|--------|
| **Deductive Engine** | Rules × States | Next_States |
| **Abductive Engine** | Rules × Target_States | Prev_States |
| **Particle Splitting** | Probabilistic rules | Multiple realities |

## What's Missing: Induction

```
Observations × ??? → Hypothesized_Rules
```

The learner fills this gap.

## The Hypothesis Generator (~50 lines)

```python
def generate_hypotheses(before, after, action):
    effects = diff(before, after)  # What changed
    candidates = []
    
    # Heuristic 1: Consumed tokens were preconditions
    for t in (before - after):  # Things that disappeared
        candidates.append(Rule(lhs={t, action}, rhs=effects, prior=0.6))
    
    # Heuristic 2: Full state match (memorization)
    candidates.append(Rule(lhs=before|{action}, rhs=effects, prior=0.3))
    
    # Heuristic 3: Unchanged tokens as catalysts
    for t in (before & after):  # Things that stayed
        candidates.append(Rule(lhs={t, action}, rhs=effects, prior=0.2))
    
    # Heuristic 4: Just action
    candidates.append(Rule(lhs={action}, rhs=effects, prior=0.1))
    
    return sorted(candidates, key=lambda r: -r.prior)
```

That's the core. Heuristics can be extended, but the structure is simple.

## Integration Architecture

```
ENVIRONMENT
    │
    │ observations: (before, after, action)
    ▼
HYPOTHESIS GENERATOR (CPU, ~50 lines)
    │
    │ candidate rules
    ▼
HYPOTHESIS POOL (priority queue)
    │
    │ batch to test
    ▼
HYPOTHESIS TESTER (YOUR GPU ENGINE, Phase 2)
    │
    │ scored rules
    ▼
THEORY (confirmed rules)
    │
    │ current best rules
    ▼
DEDUCTIVE ENGINE (YOUR GPU ENGINE, Phase 1)
```

## Bi-Phasal Execution

Your engine can alternate between two interpretations:

### Phase 1: Deduction
- **Particles** = Game states
- **Rules** = Confirmed theory
- **Output** = Predicted next states

### Phase 2: Hypothesis Testing
- **Particles** = Candidate rules (encoded as states)
- **Rules** = "Does this candidate match observation X?"
- **Output** = Surviving candidates with scores

Same engine, different semantics!

## Encoding Rules as Particles

For testing, encode a candidate rule as a "state":

```
Candidate: LHS={A, B}, Action=2, RHS={+C, -A}

Encoded as particle-state:
{__LHS:A, __LHS:B, __ACT:2, __RHS:+C, __RHS:-A}
```

Then the "testing rule" checks if this candidate matches an observation.

## Testing Logic

For each (candidate, observation) pair:

1. **LHS match?** Does candidate's LHS ⊆ observation's before?
2. **RHS match?** Does candidate's RHS == observation's actual effects?

If both: candidate passes this observation.

Score = #passed / #applicable observations

## What's Actually New

| Component | Lines | Notes |
|-----------|-------|-------|
| Hypothesis generator | ~50 | Heuristics for candidate rules |
| Hypothesis pool | ~20 | Priority queue, bounded size |
| Encoding/decoding | ~20 | Rule ↔ particle-state conversion |
| Integration glue | ~30 | Wire components together |
| **Total** | **~120** | Everything else is your existing engine |

## The Key Insight

The learner doesn't need to be smart about testing.
**It just needs to generate good candidates.**

Your engine already knows how to:
- Test rules against states (LHS ⊆ check)
- Apply rules (RHS effects)
- Handle parallelism (particles)

The learner just feeds it hypotheses to test.

## Why This Works

1. **CONSUMED tokens signal preconditions** - If B disappeared, B was probably required
2. **PRODUCED tokens signal effects** - What appeared is obviously the effect
3. **Occam's razor via testing** - Simpler rules that pass tests beat complex ones
4. **Your engine handles the hard part** - Parallel testing, probability, search

The "intelligence" is distributed:
- Generator: Propose plausible rules
- Tester: Verify against evidence
- Occam: Keep simplest survivors
