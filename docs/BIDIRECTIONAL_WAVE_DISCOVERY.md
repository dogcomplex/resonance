# Bidirectional Wave Discovery System

## Summary of Physics-Inspired Rule Learning

This document summarizes the key discoveries from applying Wheeler-Feynman style bidirectional wave analysis to rule learning and game theory.

---

## Core Insight

**Deduction + Abduction = Induction**

```
FORWARD WAVE (Deduction):   Start → ... → ?
  "Given these conditions, what can happen?"
  
BACKWARD WAVE (Abduction):  ? → ... → End
  "Given this outcome, what must have happened?"
  
STANDING WAVE (Induction):  Start ↔ End
  "What RULES connect cause to effect?"
  The interference pattern of forward and backward waves.
```

---

## Key Discoveries

### 1. The Big Crunch as Attractor

The "Big Crunch" (endpoint) isn't arbitrary - it EMERGES from the dynamics:
- In physics: Maximum entropy / ground state
- In games: Terminal states (Win/Lose/Tie/Error)
- In logic: Tautologies and contradictions
- In our system: Tokens with no outgoing rules (sinks)

**Without an attractor, there's no backward wave. Without a backward wave, there's no induction.**

### 2. Gravity = Shape of Valid State Space

"Gravity" is the constraint on what states are reachable:
- **Tight gravity** (TTT valid states): 5,478 states
- **Loose gravity** (all 3^9): 19,683 states

The "missing" states ARE the gravitational constraint. They define the playing field.

### 3. Single Run Contains All Perspectives

From ONE bidirectional wave computation, we can extract:
- X-player strategy (filter for paths through W)
- O-player strategy (filter for paths through L)
- Defensive play (filter for paths through T)
- Chaos agent (filter for paths through E)
- Position agents (filter for specific moves)

**No separate runs needed - just different filters!**

### 4. Adversarial Games = Interfering Waves

Two players wanting opposite outcomes:
```
X's wave pulls toward W (X wins)
O's wave pulls toward L (O wins)
Where they cancel: T (Tie)
```

Perfect play = perfect cancellation. This is why TTT is a draw.

### 5. Rule Discovery via Resonance

Rules that appear in BOTH forward and backward passes are "real":
```
Forward: {A, B} → can reach {C} via some rules
Backward: {C} ← needs {A, B} via some rules
Standing wave: Rules that do BOTH
```

Higher resonance (appears more often) = more fundamental rule.

### 6. Curried Rules Emerge Naturally

The standing wave finds both direct and curried paths:
```
A + B → C                    (direct)
A + B → [A•B] → C            (curried)
```

The intermediate [A•B] appears in forward expansion, gets explained by backward expansion.

---

## Implementation Components

### bidirectional_waves.py
- Basic forward/backward expansion
- Standing wave detection
- Number chain universe tests

### converging_universe.py  
- Natural attractors (ground states)
- Thermodynamic convergence
- Multiple Big Crunches

### ttt_bidirectional.py
- Tic-Tac-Toe universe
- Valid vs loose gravity
- Win condition discovery

### ttt_adversarial.py
- Adversarial wave analysis
- Move strength computation
- Perfect play derivation

### multi_perspective.py
- 116 perspectives enumerated
- Position agent analysis
- Blank/chaos agent perspectives

### bidirectional_chemistry.py
- Integration with rule learning
- Curried rule detection
- Endpoint-aware learning

---

## Key Formulas

### Wave Differential
```
differential(move) = paths_to_W(move) - paths_to_L(move)
                   ─────────────────────────────────────
                         total_paths(move)
```

### Rule Resonance
```
resonance(rule) = count(rule in standing waves)
                  ─────────────────────────────
                  total standing waves
```

### Perspective Value
```
value(move, perspective) = paths_to_goal(move)
                          ────────────────────
                          total_paths(move)
```

---

## Connections to Physics

| Concept | Our System | Physics |
|---------|------------|---------|
| Forward wave | Deduction | Retarded wave |
| Backward wave | Abduction | Advanced wave |
| Standing wave | Induction | Interference pattern |
| Big Crunch | Terminal states | Attractor/equilibrium |
| Gravity | Valid state space | Spacetime geometry |
| Multiple crunches | W/L/T/E endpoints | Many worlds |
| Adversarial | Opposite endpoints | Interference |

---

## Open Questions

1. **How do the 4 forces map?**
   - Forward wave = Strong force?
   - Backward wave = Weak force?
   - Standing wave = Electromagnetism?
   - State space shape = Gravity?

2. **Can we simulate universe bootstrap?**
   - Single wave expanding
   - Self-interaction via advanced waves
   - Symmetry breaking → force separation

3. **Sample complexity from wave analysis?**
   - Depth in standing wave = observations needed
   - Lower depth = simpler rules = fewer examples

4. **GPU-efficient bidirectional computation?**
   - Forward pass is already parallelizable
   - Backward pass as matrix transpose?
   - Standing wave as intersection kernel

---

## Usage Example

```python
from bidirectional_chemistry import BidirectionalRuleLearner

# Create learner with endpoints
learner = BidirectionalRuleLearner()
learner.add_endpoint('WIN')
learner.add_endpoint('LOSE')

# Add observations
learner.observe({'start'}, {'mid'})
learner.observe({'mid'}, {'WIN'})

# Learn rules
learner.learn_from_observations()

# Get standing waves
waves = learner.find_standing_waves({'start'}, {'WIN'})
```

---

## Conclusion

The bidirectional wave framework unifies:
- **Rule learning** (induction via standing waves)
- **Rule application** (deduction via forward waves)
- **Goal reasoning** (abduction via backward waves)
- **Multi-agent dynamics** (perspective filtering)
- **Game theory** (adversarial wave interference)

All from a single computation of the universal wave function!
