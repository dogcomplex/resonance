# Closure Physics: Pressure, Interiority, and Quantum Analogies

**Date**: 2026-01-25
**Purpose**: Explore the physical intuitions behind closure-aware search

---

## 1. Pressure and Permeability

### The Current Model

Right now, the wave system stops when it finds closures (spines connecting start to boundary). But finding the *first* winning path isn't the same as mapping the *full solution space*.

### Pressure as Driving Force

In physics, pressure differentials drive flow:
- High pressure → Low pressure = Flow
- Equal pressure = Equilibrium = No flow

For HOLOS:
```
Forward pressure = unexplored positions reachable from current frontier
Backward pressure = unknown positions that could reach boundary
Closure = pressure equalization at a point
```

**Current problem**: After closure, we stop. But closures should *relieve* pressure locally while *increasing* pressure elsewhere.

### Permeability as Medium Property

Different regions of the search space have different "permeability":
- **High permeability**: Many paths through (e.g., early game, many legal moves)
- **Low permeability**: Constrained regions (e.g., forced sequences, narrow corridors)

```python
permeability(region) = branching_factor * (1 - closure_density)
```

When a region saturates with closures, permeability drops → flow redirects to unexplored regions.

### Proposed Extension: Continuous Flow

Instead of stopping at first closure:

```python
class PressureWaveSystem:
    def step(self):
        # 1. Measure pressure at each frontier point
        for h, state in self.forward_frontier.items():
            pressure = self.compute_pressure(h, direction="forward")

        # 2. Flow proportional to pressure differential
        for h in sorted(frontier, key=lambda x: -self.pressure[x]):
            if self.pressure[h] > self.saturation_threshold:
                self.expand(h)

        # 3. Closures reduce local pressure but don't stop search
        for closure in new_closures:
            self.pressure[closure.state_hash] *= self.closure_damping
            # Pressure redistributes to neighbors

        # 4. Continue until global pressure equilibrium
        return sum(self.pressure.values()) > self.min_total_pressure
```

### For Connect4 Full Solution

To map ALL winning paths:
1. Find first spine (closure)
2. Reduce pressure at that connection point
3. Continue expanding in regions with remaining pressure
4. Iterate until pressure equilibrium (all reachable states mapped)

The "closure density" metric would tell us: what fraction of the reachable space have we closed?

---

## 2. Interiority and Unreachability

### What is an Interior?

An **interior** is a region where:
- All paths in eventually lead to closure
- No paths escape without hitting known values
- The region is "sealed" by its boundary of closures

```
        Exterior (unknown)
             |
    =====CLOSURE BOUNDARY=====
             |
        Interior (solved)
             |
    ===== GAME BOUNDARY =====
```

### Where Do Interiors Form?

In Connect4:
1. **Near terminals**: Positions 1-2 moves from win/loss are quickly interior
2. **Forced sequences**: When only one legal move exists, the path is interior
3. **Symmetric closures**: Mirror positions close simultaneously

### What Remains Unreachable?

**Unreachable positions** are those where:
- Forward wave can't reach (not valid game states)
- Backward wave can't reach (no path from any terminal)
- Disconnected components in the game graph

For Connect4, unreachable states include:
- Invalid boards (wrong piece counts, floating pieces)
- Positions where both players have 4-in-a-row (impossible in real play)

### "Decrypting" an Interior

An interior is "decrypted" when:
1. **Boundary is known**: All edges of the region connect to known values
2. **Propagation completes**: Minimax values propagate through the interior
3. **No uncertainty remains**: Every position in the interior has a definite value

The act of "decryption" is **value propagation** after closure:
```python
def decrypt_interior(self, closure_boundary: Set[int]):
    """Propagate values inward from closure boundary."""
    queue = list(closure_boundary)
    while queue:
        h = queue.pop(0)
        state = self.states[h]

        # Get children's values
        child_values = [self.values[ch] for ch, _ in self.game.get_successors(state)
                       if ch in self.values]

        # Propagate (minimax for games)
        if child_values:
            self.values[h] = self.game.propagate_value(state, child_values)

            # Parents may now be solvable
            for parent, _ in self.game.get_predecessors(state):
                ph = self.game.hash_state(parent)
                if ph not in self.values:
                    queue.append(ph)
```

### Interior Detection

```python
def detect_interior(self, center_hash: int, radius: int = 3) -> Optional[Interior]:
    """Check if a region forms a closed interior."""
    boundary = set()
    interior = set()
    queue = [(center_hash, 0)]

    while queue:
        h, depth = queue.pop(0)
        if h in interior:
            continue

        if depth >= radius or h in self.closure_points:
            boundary.add(h)
            continue

        interior.add(h)
        for child, _ in self.game.get_successors(self.states[h]):
            ch = self.game.hash_state(child)
            queue.append((ch, depth + 1))

    # Check if boundary is fully closed
    boundary_solved = all(h in self.values for h in boundary)
    if boundary_solved:
        return Interior(center=center_hash, boundary=boundary, interior=interior)
    return None
```

---

## 3. Performance Estimation: Connect4

### Current State Space

Connect4 has approximately:
- **4.5 trillion** legal positions (4.5 × 10¹²)
- **~10^13** total game states including unreachable
- First player (X) wins with perfect play

### What We're Currently Doing

From test results:
- 30 positions → 28 spines in 40 iterations
- Each iteration processes ~100-200 states
- ~1000 states examined total

**Current coverage**: ~0.00000001% of the space

### Compression Opportunities

**Equivalence classes** (not yet implemented in closure system):
1. **Horizontal symmetry**: Each position = its mirror (2x compression)
2. **Rotation invariance**: Not applicable to Connect4 (gravity)
3. **Feature classes**: Positions with same material/threat counts often have same value

From original HOLOS experiments:
- Symmetry alone: 2x compression
- Feature equivalence: 10-100x compression on some regions
- Seed-based reconstruction: 1000x+ for dense regions

### Estimated Full Solution

**Without compression**:
- 4.5 × 10¹² positions × 1 byte/value = 4.5 TB
- Compute: ~10^14 operations (several days on modern hardware)

**With compression**:
- Symmetry: 2.25 TB
- Feature equivalence: 225 GB - 2.25 TB (varies)
- Spine storage: ~45 GB (estimated 100x compression)
- Meta-seed compression: ~5-10 GB (estimated from prior experiments)

**Compute with closure system**:
The closure system doesn't reduce total compute, but:
- Better prioritizes which states to examine
- Detects when regions are "closed" (no more work needed)
- Could enable distributed solving with less coordination

### Adding Equivalence Classes

To add equivalence classes to closure system:

```python
# In closure.py
@dataclass
class ClosureDetector:
    equiv_classes: Dict[Any, Set[int]] = field(default_factory=dict)
    equiv_values: Dict[Any, Any] = field(default_factory=dict)

    def check_closure_with_equivalence(self, state_hash, forward_value,
                                        backward_value, features, layer, iteration):
        # Check if features already have known value
        if features in self.equiv_values:
            return ClosureEvent(
                state_hash=state_hash,
                closure_type=ClosureType.EQUIVALENCE,
                forward_value=self.equiv_values[features],
                ...
            )

        # Normal closure check
        event = self.check_closure(state_hash, forward_value, backward_value, layer, iteration)

        # If closed, propagate to equivalence class
        if event:
            self.equiv_classes.setdefault(features, set()).add(state_hash)
            if features not in self.equiv_values:
                self.equiv_values[features] = event.forward_value

        return event
```

---

## 4. Quantum Analogies

### The poor_mans_quantum.txt Summary

Key claims:
1. Wave superpositions can emulate ternary quantum vectors
2. Real/imaginary components time-multiplexed (even/odd steps)
3. Polynomial inputs/sensors for n "qubits"
4. But costs 2^n energy/time for full accuracy
5. Entropy filters "collapse" to readable subset
6. A Maxwell daemon could boost low-complexity terms to produce higher ones

### Connection to Closure System

The closure system has striking parallels:

| Quantum Concept | Closure Analog |
|-----------------|----------------|
| Superposition | Forward + backward waves coexisting |
| Entanglement | Paths sharing common states |
| Measurement/Collapse | Closure event (waves "agree" on value) |
| Interference | Constructive: closures reinforce; Destructive: contradictions |
| Normalization loss | Energy damping in wave system |

### Phased Waves

The closure system already uses phased waves:
- Forward wave: Phase 0
- Backward wave: Phase π (opposite direction)
- Closure: Phase alignment (0 ≈ π in value space)

To extend with complex phases (real/imaginary):

```python
class ComplexWaveSystem:
    def __init__(self):
        self.real_frontier = {}   # Even timesteps
        self.imag_frontier = {}   # Odd timesteps

    def step(self, t):
        if t % 2 == 0:
            # Real component: [A,a]*[B,-b] pattern
            self.expand_real()
        else:
            # Imaginary component: [A,a]*[b,B] pattern
            self.expand_imaginary()

    def measure(self):
        # Combine real and imaginary for "observation"
        # This is where entropy filtering happens
        observable = {}
        for h in self.real_frontier:
            if h in self.imag_frontier:
                # Both components present = stable
                observable[h] = complex(self.real_frontier[h],
                                        self.imag_frontier[h])
        return observable
```

### Reservoir Computing Connection

A **reservoir** is a dynamical system that:
1. Projects inputs into high-dimensional space
2. Evolves according to internal dynamics
3. Allows simple readout of complex computations

The closure system AS a reservoir:
- **Input**: Seed positions
- **Reservoir**: The game graph (high-dimensional state space)
- **Dynamics**: Wave propagation rules
- **Readout**: Closure events

Key insight from poor_mans_quantum.txt:
> "if the wave can be read quickly (e.g. an algorithm recording this sound wave with faster electric signaling)"

For HOLOS:
- **Fast medium**: Hash table lookups (O(1))
- **Slow medium**: Game tree expansion (O(branching^depth))
- **Daemon role**: Closure detector "reads" the fast medium to guide the slow one

### The 2^n Cost

poor_mans_quantum.txt notes the exponential cost is unavoidable. For game search:
- Connect4 has ~4.5 × 10¹² states ≈ 2^42
- Full search costs O(2^42) regardless of method
- Compression reduces STORAGE but not fundamental COMPUTE
- Closure events are like quantum measurements: they collapse uncertainty but don't reduce total information

However, the closure system offers:
1. **Early termination**: Stop when closure density saturates
2. **Selective amplification**: Focus energy on high-pressure regions
3. **Coherent subsets**: Only track states that survive entropy filtering

### Quantum-Inspired Extensions

```python
class QuantumClosureSystem:
    """
    Treat game search as quantum-like computation.
    """

    def __init__(self, game, n_qubits=10):
        self.game = game
        self.n_qubits = n_qubits  # log2 of state space complexity

        # Amplitude tracking (complex-valued)
        self.amplitudes: Dict[int, complex] = {}

        # Phase tracking
        self.phases: Dict[int, float] = {}

    def superpose(self, states: List[Any]):
        """Create equal superposition over states."""
        amp = 1.0 / len(states) ** 0.5
        for state in states:
            h = self.game.hash_state(state)
            self.amplitudes[h] = complex(amp, 0)
            self.phases[h] = 0.0

    def evolve(self):
        """Unitary evolution step."""
        new_amplitudes = {}
        for h, amp in self.amplitudes.items():
            state = self.get_state(h)
            successors = self.game.get_successors(state)

            # Distribute amplitude to successors (like branching)
            child_amp = amp / len(successors) ** 0.5
            for child, move in successors:
                ch = self.game.hash_state(child)
                new_amplitudes[ch] = new_amplitudes.get(ch, 0) + child_amp

        self.amplitudes = new_amplitudes

    def measure(self) -> List[Tuple[int, float]]:
        """Collapse superposition, return observable states."""
        # Probability = |amplitude|^2
        probs = [(h, abs(amp)**2) for h, amp in self.amplitudes.items()]

        # Normalize
        total = sum(p for _, p in probs)
        probs = [(h, p/total) for h, p in probs]

        # Return highest probability states (entropy filtering)
        return sorted(probs, key=lambda x: -x[1])[:self.n_qubits]

    def interference(self, h1: int, h2: int) -> str:
        """Check if two states interfere constructively or destructively."""
        a1, a2 = self.amplitudes.get(h1, 0), self.amplitudes.get(h2, 0)
        if abs(a1 + a2) > max(abs(a1), abs(a2)):
            return "constructive"
        elif abs(a1 + a2) < min(abs(a1), abs(a2)):
            return "destructive"
        return "neutral"
```

---

## 5. Synthesis: The Full Picture

### What We Have

1. **Closure detection**: Find where forward/backward waves agree
2. **Mode emergence**: System self-organizes into lightning/wave/crystal/osmosis
3. **Layer hierarchy**: Positions → Paths → Covers → Policies
4. **Energy conservation**: Finite budget, damping, transfer between layers

### What We Need

1. **Pressure dynamics**: Continue past first closure, map full solution
2. **Interior detection**: Know when regions are "solved"
3. **Equivalence classes**: Compress via feature matching
4. **Quantum-like amplitudes**: Track probability of paths, not just existence

### The Vision

```
                    POLICY LAYER (slow, coarse)
                         ↑↓ energy transfer on closure
                    COVER LAYER
                         ↑↓
                    PATH LAYER
                         ↑↓
                    POSITION LAYER (fast, fine)
                         ↑↓
                    GAME BOUNDARY (known values)

Each layer is a RESERVOIR that:
- Receives input from adjacent layers
- Evolves according to wave dynamics
- Produces output (closures) that feed other layers

The whole system is a QUANTUM-LIKE computer that:
- Maintains superposition (multiple paths simultaneously)
- Collapses on measurement (closure events)
- Uses interference (path combinations)
- Pays 2^n cost in energy but achieves polynomial-sensor scaling
```

### Next Steps

1. Implement pressure-based continuous flow
2. Add interior detection and "decryption"
3. Integrate equivalence classes from original HOLOS
4. Experiment with complex amplitudes for quantum-like behavior
5. Benchmark against full Connect4 solve

---

## Appendix: Estimated Connect4 Full Solve

| Metric | Without Compression | With Full Compression |
|--------|--------------------|-----------------------|
| States | 4.5 × 10¹² | 4.5 × 10¹² (same) |
| Storage | 4.5 TB | 5-50 GB |
| Compute | ~10¹⁴ ops | ~10¹⁴ ops (same) |
| Time (single core) | ~1 year | ~1 year |
| Time (1000 cores) | ~8 hours | ~8 hours |
| Energy | ~10 MWh | ~10 MWh |

The compression helps storage dramatically but doesn't reduce fundamental compute.
Closure-aware search helps PRIORITIZE compute, not reduce it.
Quantum speedup would help, but we don't have real quantum hardware.
The "poor man's quantum" approach lets us USE quantum patterns at classical cost.
