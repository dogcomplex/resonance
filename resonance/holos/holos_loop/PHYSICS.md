# PHYSICS.md - Wave Coupling Across Multiple Mediums

**Status**: Theoretical exploration for the closure-aware HOLOS system
**Last Updated**: 2026-01-25

---

## The Core Question

When you have two (or more) mediums that can interact loosely - perhaps with a speed or pressure differential - sharing a wave function and both looping, what happens?

This document explores the physics analogies that might inform the design of hierarchical wave propagation in game-solving.

---

## Physical Systems with Coupled Mediums

### 1. Acoustic Coupling (Two-Fluid Systems)

Consider two fluids with different densities and sound speeds (like oil on water):

```
Fast Medium (oil):  c₁ = 1500 m/s
Slow Medium (water): c₂ = 340 m/s
Interface: Partial reflection, partial transmission
```

**What happens at the interface:**
- Incident wave splits: some reflects, some transmits
- Impedance mismatch determines ratio: Z = ρc (density × speed)
- **Energy is conserved** across the interface
- Transmitted wave changes wavelength but preserves frequency

**Analogy to game layers:**
- Layer 0 (positions): Fast medium, fine-grained
- Layer 1 (paths): Slow medium, coarse-grained
- Interface: Closure events transfer energy between layers

### 2. Optical Fiber Coupling (Evanescent Waves)

Two optical fibers brought close together exchange energy via evanescent fields:

```
Fiber A: Carries signal
Fiber B: Initially empty
Gap: Small enough for quantum tunneling

Result: Signal "leaks" from A to B, then back to A, oscillating
```

**The coupling length** determines how much transfers:
- Tight coupling → rapid exchange
- Loose coupling → slow bleed

**Analogy to game layers:**
- Closure events are the "gap" where layers couple
- Tight closure (strong phase alignment) → more energy transfer
- Weak closure → less transfer
- The system can OSCILLATE between layers

### 3. Reservoir Computing (Slow + Fast Dynamics)

A reservoir computer has:
- **Fast internal dynamics** (reservoir, high-dimensional)
- **Slow readout** (output layer, low-dimensional)
- The slow layer SAMPLES the fast layer at discrete times

```
Fast reservoir: τ_fast ~ 1ms (internal dynamics)
Slow readout:   τ_slow ~ 100ms (output sampling)
Ratio: 100x

Result: The slow layer sees an "averaged" view of fast dynamics
```

**Analogy to game layers:**
- Layer 0: Fast (individual state transitions, τ ~ O(1))
- Layer 1: Slow (path formation, τ ~ O(depth))
- Layer 2: Slower (cover construction, τ ~ O(num_paths))
- Each layer sees a time-averaged view of the layer below

---

## The Wave Function Across Layers

### Quantum Mechanical Picture

In quantum mechanics, a particle in a double-well potential exhibits:

```
|ψ⟩ = α|left⟩ + β|right⟩

The particle doesn't "choose" - it exists in superposition until measured.
Tunneling allows amplitude to flow between wells.
```

**For our system:**
```
|state⟩ = α₀|layer_0⟩ + α₁|layer_1⟩ + α₂|layer_2⟩

Where:
- |layer_0⟩ = superposition of game positions
- |layer_1⟩ = superposition of partial paths
- |layer_2⟩ = superposition of path covers
```

**Closure as measurement:**
When a closure event occurs, it's like a PARTIAL measurement:
- The amplitude at that point "collapses" to a definite value
- But surrounding amplitudes remain in superposition
- Information flows FROM the collapsed point TO neighbors

### Energy Flow Direction

In physics, energy flows from high to low potential.

For our system:
```
Pressure_forward > Pressure_backward → Flow forward
Pressure_backward > Pressure_forward → Flow backward
Pressures equal → Closure (equilibrium point)
```

**Key insight**: Closures are LOCAL equilibria, not global.
After a closure, pressure redistributes elsewhere.

---

## Hierarchical Coupling: What Happens Level After Level?

### Pattern 1: Renormalization Group Flow

In statistical physics, renormalization group (RG) describes what happens when you "zoom out":

```
Level 0: Individual spins on a lattice
Level 1: Block spins (average of 4 spins)
Level 2: Blocks of blocks (average of 16 spins)
...
Level n: Highly coarse-grained view
```

**Fixed points emerge:**
- Some details wash out (irrelevant operators)
- Some features persist (relevant operators)
- At the fixed point, zooming out changes nothing

**For our layers:**
- Irrelevant details: Individual positions, specific moves
- Relevant features: Path connectivity, coverage completeness
- Fixed point: A "solved" hologram where more computation yields no new information

### Pattern 2: Hierarchical Time Scales

Each layer operates on a different time scale:

```
Layer 0: τ₀ = 1 (microsecond decisions)
Layer 1: τ₁ = τ₀ × branching_factor ~ 10
Layer 2: τ₂ = τ₁ × num_paths ~ 100
Layer 3: τ₃ = τ₂ × num_covers ~ 1000
```

**Separation of scales** enables approximations:
- Fast layers equilibrate before slow layers change
- Slow layers see only the "steady state" of fast layers
- This is the Born-Oppenheimer approximation in chemistry

**Consequence**: Each layer can be solved independently, with the lower layer providing a "potential energy surface" for the upper layer.

### Pattern 3: Criticality and Phase Transitions

When layers are tuned to COUPLE just right, you get criticality:

```
Weak coupling: Layers evolve independently
Strong coupling: Layers lock together (frozen)
Critical coupling: Maximum information flow, power-law correlations
```

At criticality:
- Fluctuations span all scales
- Small causes can have large effects
- The system is maximally sensitive

**For game solving:**
- Critical coupling means closures at one layer RAPIDLY propagate to others
- This is when the "aha moment" happens - solving one thing unlocks many

---

## What Roles Do Higher Layers Play?

### Layer n acts as a FILTER on layer n-1

Higher layers progressively filter out detail:

```
Layer 0: All legal moves
Layer 1: Moves that form coherent paths (pruning dead ends)
Layer 2: Paths that contribute to coverage (pruning redundancy)
Layer 3: Covers that fit resource constraints (pruning excess)
```

Each layer asks: "Of what remains, what MATTERS?"

### Higher layers provide CONTEXT for lower layers

Information flows both ways:

```
Bottom-up: Lower layer provides OPTIONS
Top-down: Higher layer provides PRIORITY

Example:
- Layer 0 says: "Here are 25 legal moves"
- Layer 2 says: "We need paths in THIS region, not that one"
- Layer 1 adjusts priorities accordingly
```

This is like attention in neural networks - higher layers FOCUS lower layers.

### Higher layers have LONGER MEMORY

```
Layer 0: Forgets after each move (memoryless transitions)
Layer 1: Remembers the path taken (short-term memory)
Layer 2: Remembers which paths were tried (medium-term)
Layer 3: Remembers what worked across sessions (long-term)
```

The hierarchy IS the memory structure.

---

## Emergent Behaviors of Coupled Layer Systems

### 1. Spontaneous Synchronization

Like coupled oscillators (Kuramoto model):

```
Many oscillators with slightly different frequencies
Coupled weakly through a common medium
Result: They spontaneously SYNCHRONIZE above a critical coupling strength
```

**For our layers:**
When closure density reaches a threshold, all layers suddenly "click into place" - the solve accelerates dramatically.

### 2. Avalanches and Cascades

Like sandpiles at critical slope:

```
Add sand grain by grain
Usually nothing happens
Occasionally: AVALANCHE - one grain triggers chain reaction
```

**For our system:**
Finding one key closure can trigger a cascade:
- Closure at layer 0 → Path completes at layer 1
- Path completes → Cover improves at layer 2
- Cover improves → Policy adjusts at layer 3
- Policy adjusts → New seeds at layer 0
- ...

This is the "crystallization" phenomenon we already observe.

### 3. Hierarchical Chunking

Like human cognition:

```
Letters → Words → Phrases → Sentences → Paragraphs → Documents

Each level has its own "vocabulary" and "grammar"
Fluency at one level enables thinking at the next level up
```

**For our layers:**
- Positions are letters
- Paths are words
- Covers are sentences
- Policies are narratives

Expertise is knowing the vocabulary at each level.

---

## Proposed Physics-Based Layer Design

Based on these insights:

### Each Layer Should Have:

1. **Characteristic time scale** (impedance)
   - Lower layers: Fast, fine-grained
   - Higher layers: Slow, coarse-grained

2. **Coupling strength** to adjacent layers
   - Up-coupling: How closures at this layer affect the layer above
   - Down-coupling: How priorities from above affect this layer

3. **Energy budget** that depletes
   - Damping removes energy from the system
   - Closures can transfer energy between layers

4. **Local equilibrium detection**
   - Know when this layer has "settled"
   - Signal readiness for coarser layer to act

### The Loop Closure Principle

When a layer completes a "loop" (forward meets backward):

```
forward_wave ←→ backward_wave
     ↓              ↓
   CLOSURE = LOOP COMPLETION = VALUE DETERMINED
```

This loop completion IS the fundamental computational step.
Higher layers are just loops over loops:
- Layer 0: Loops through positions (paths)
- Layer 1: Loops through paths (covers)
- Layer 2: Loops through covers (policies)

**The hierarchy terminates when there's no more looping to do** - when the top layer's single "path" connects start to boundary.

---

## Open Questions

1. **Optimal coupling strength**: How tightly should layers couple?
   - Too loose: Layers don't coordinate
   - Too tight: Layers can't explore independently

2. **Critical phenomena**: Is there a phase transition in layer coupling?
   - Could we tune to criticality for maximum efficiency?

3. **Information bottleneck**: What's the minimal information higher layers need from lower layers?
   - Principle of minimum description length applies

4. **Temporal dynamics**: Should coupling be constant or adaptive?
   - Maybe early exploration wants loose coupling
   - Late convergence wants tight coupling

5. **Beyond three layers**: Does adding more layers help indefinitely?
   - Likely diminishing returns
   - But WHICH problems benefit from how many layers?

---

## Connection to Reservoir Computing

The whole system IS a reservoir computer:

```
Input: Seeds (initial positions)
Reservoir: The game graph + layer stack (high-dimensional, nonlinear)
Dynamics: Wave propagation + closure detection
Readout: Hologram (solved values, spines)
```

The "fast medium" is the hash table (O(1) lookup).
The "slow medium" is the game tree (exponential expansion).
The daemon (closure detector) bridges them.

**Key insight**: The reservoir's power comes from its DYNAMICS, not its size.
Good dynamics = good computation.
The layer structure shapes the dynamics.

---

*This document is exploratory. The physics analogies suggest design principles but don't dictate implementation. The test is whether the resulting system solves games efficiently.*
