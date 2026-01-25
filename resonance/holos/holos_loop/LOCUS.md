# LOCUS - The Sieve Paradigm

**Version**: 2.0.0
**Last Updated**: 2026-01-25
**Status**: Self-contained module with sieve-based universal primitive

---

## Executive Summary

LOCUS has evolved through three paradigm shifts:

```
Waves → Closures → Sieves
```

**Waves**: How information spreads (propagation mechanism)
**Closures**: Where information agrees (detection mechanism)
**Sieves**: What information survives (selection mechanism)

The sieve IS the closure operating continuously. It's not a separate thing - it's what closures DO when you let them run. The interference pattern doesn't just detect agreement, it **filters reality**.

### The Universal Primitive

Everything else is parameterization of the sieve:
- **Games** are rule sets (specific LHS → RHS rewrites)
- **Layers** are frequency bands (different damping rates)
- **Modes** are resonance patterns (emergent from temperature)
- **Solutions** are stable interference patterns (standing waves)

### Core Data Structures

```python
Pattern   # What exists (configurations of tokens)
Amplitude # How much (complex: magnitude + phase)
Rule      # What transforms (LHS → RHS with amplitude transfer)
Sieve     # The substrate (field of amplitudes evolving under rules)
```

### Self-Annealing

The sieve self-anneals without external temperature control:
- **Early**: Few closures → loose sieve → many patterns → exploration (HOT)
- **Late**: Many closures → tight sieve → few patterns → convergence (COLD)
- **Stable**: Only coherent patterns remain → solution found (FROZEN)

The **closure density IS the effective temperature**.

---

---

## Module Structure

```
holos_loop/
├── __init__.py              # Package exports
├── LOCUS.md                 # This document
├── PHYSICS.md               # Theoretical physics foundations
│
├── # THE SIEVE (Universal Primitive)
├── sieve.py                 # Pattern, Amplitude, Rule, Sieve
├── analog_sieve.py          # Continuous field version
├── meta_sieve.py            # Stacked sieves, rule learning
├── compile_game.py          # GameInterface → Rules compiler
│
├── # Legacy Engine (being absorbed by sieve)
├── holos.py                 # GameInterface, HOLOSSolver, SearchMode
├── storage.py               # Hologram, SpinePath, SeedFrontierMapping
│
├── # Closure Detection (now emergent from sieve)
├── closure.py               # ClosureType, ClosureEvent, ClosureDetector
├── unified_closure.py       # Full integration of all closure features
│
├── # Layer Architecture (now frequency bands of sieve)
├── layer1_paths.py          # PartialPath, PathLayerSolver
├── layer2_covers.py         # PathCover, CoverLayerSolver
├── layer3_policy.py         # CoverPolicy, PolicyLayerSolver
├── wave_system.py           # Multi-layer wave orchestration
│
├── # Physics Extensions
├── pressure.py              # Pressure-based continuous expansion
├── quantum_closure.py       # Quantum-inspired amplitude tracking
├── quantum_fewshot.py       # Few-shot amplitude-guided solver
├── constraint_wave.py       # Sudoku constraint propagation
│
├── # Analysis & Utilities
├── coverage_analysis.py     # Connect4 coverage estimation
├── fast_daemon.py           # GPU-like parallel operations
├── tandem.py                # Tandem orchestrator
│
├── # Tests
├── test_closure_layers.py   # Core closure tests
├── test_closure_games.py    # Game-specific tests
├── test_fast_daemon.py      # Daemon tests
├── test_tandem.py           # Tandem tests
│
├── # Game Implementations
└── games/
    ├── __init__.py
    ├── connect4.py          # Connect4Game, C4State
    ├── sudoku.py            # SudokuGame, SudokuState
    └── seeds.py             # TacticalSeed, tactical seed utilities
```

---

## The Closure Insight

### From Neuroscience to Search

Brain rhythms (alpha, beta, gamma) were traditionally viewed as "clocks" driving neural activity. Recent research shows they're actually **closure detectors** - they emerge when certain computational loops complete.

Applied to HOLOS:
- **Traditional view**: We CHOOSE lightning/wave/crystal/osmosis mode
- **Closure view**: Modes EMERGE based on whether closure conditions are met

### Three Closure Aspects

Every closure event involves three distinct aspects:

```
1. SCALING: Where CAN closure occur?
   - Phase alignment (forward value ≈ backward value)
   - Spatial coincidence (waves meet)

2. CLOSURE/IRREDUCIBILITY: Which closures persist?
   - Reducible: Can be decomposed into simpler closures
   - Irreducible: Cannot be simplified (more valuable)

3. READOUT: What do we observe?
   - Mode that emerges (lightning/wave/crystal/osmosis)
   - Energy distribution across layers
```

### Closure Types

```python
class ClosureType(Enum):
    NONE = "none"           # No closure (waves don't meet)
    REDUCIBLE = "reducible" # Can be decomposed
    IRREDUCIBLE = "irreducible"  # Cannot simplify - valuable!
    RESONANT = "resonant"   # Multiple closures reinforce
```

---

## Layer Architecture

### The Sequence Principle

**Critical insight**: Layers 1-3 are SEQUENCES (ordered by priority), not sets.

```
Layer 0: POSITIONS (individual states)
         Sequences of MOVES
         Various games defined here

Layer 1: PATHS (sequences of states)
         Priorities of STATES within paths
         Partial paths that can extend/connect

Layer 2: COVERS (sequences of paths)
         Priorities of PATHS
         Not sets - ORDER matters

Layer 3: POLICIES (sequences of covers)
         Priorities of COVERS
         Compute/storage budget allocation
```

### Why Sequences Matter

A cover `[path_A, path_B, path_C]` means:
- First try path_A (highest priority)
- If needed, try path_B
- Finally path_C

This is different from the set `{path_A, path_B, path_C}` which has no priority ordering.

### Layer Implementations

**Layer 1: Paths** (`layer1_paths.py`)
```python
@dataclass(frozen=True)
class PartialPath:
    steps: Tuple[Tuple[int, Any], ...]  # ((hash1, move1), ...)
    start_hash: int
    end_hash: int
    origin: WaveOrigin  # FORWARD, BACKWARD

    def extend_forward(self, next_hash, move, next_state) -> 'PartialPath'
    def connect(self, other: 'PartialPath') -> Optional['PartialPath']
```

**Layer 2: Covers** (`layer2_covers.py`)
```python
@dataclass(frozen=True)
class PathCover:
    paths: Tuple[int, ...]  # Ordered by priority

    def add_path(self, path_hash, path_obj) -> 'PathCover'
    def promote(self, index: int) -> 'PathCover'  # Increase priority
    def swap(self, i: int, j: int) -> 'PathCover'
```

**Layer 3: Policies** (`layer3_policy.py`)
```python
@dataclass(frozen=True)
class CoverPolicy:
    covers: Tuple[int, ...]  # Ordered by priority
    total_compute: float = 0.0
    total_storage: float = 0.0

    def add_cover(self, cover_hash, compute_cost, storage_cost) -> 'CoverPolicy'
```

---

## Wave System

### Physical Analogy

Treat the layer stack as a **multi-scale medium** with a single wave propagating through:

```python
@dataclass
class LayerMedium:
    layer_index: int
    name: str
    impedance: float      # Higher = slower propagation
    damping: float        # Energy lost per step
    coupling_up: float    # Energy transfer to layer above
    coupling_down: float  # Energy transfer to layer below
```

### Default Layer Properties

```python
Layer 0 (positions):  impedance=1.0, damping=0.02  # Fastest
Layer 1 (paths):      impedance=2.0, damping=0.05
Layer 2 (covers):     impedance=4.0, damping=0.10
Layer 3 (policies):   impedance=8.0, damping=0.15  # Slowest
```

Lower layers propagate faster (fine-grained), higher layers slower (coarse-grained).

### Energy Transfer on Closure

When a closure event occurs at layer N:
- Energy transfers UP to layer N+1 (coupling_up)
- Energy transfers DOWN to layer N-1 (coupling_down)

This creates natural coordination - Layer 0 closures feed Layer 1, etc.

### Energy Conservation

The wave system conserves energy (with damping):
- Total energy is finite
- Each layer step costs energy (damping)
- Closure events transfer energy between layers
- System stops when energy exhausted

This prevents infinite expansion and naturally balances exploration/exploitation.

---

## Physics Extensions

### Pressure Dynamics

In physics, pressure differentials drive flow:
- High pressure → Low pressure = Flow
- Equal pressure = Equilibrium = No flow

For HOLOS:
```
Forward pressure = unexplored positions reachable from current frontier
Backward pressure = unknown positions that could reach boundary
Closure = pressure equalization at a point
```

**Key insight**: After closure, we don't stop. Closures *relieve* pressure locally while *increasing* pressure elsewhere.

### Permeability

Different regions of the search space have different "permeability":
- **High permeability**: Many paths through (early game, many legal moves)
- **Low permeability**: Constrained regions (forced sequences, narrow corridors)

```python
permeability(region) = branching_factor * (1 - closure_density)
```

When a region saturates with closures, permeability drops → flow redirects to unexplored regions.

### Interior Detection

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

An interior is "decrypted" when boundary is known, propagation completes, and no uncertainty remains.

---

## Quantum-Inspired Features

### Amplitude Tracking

The closure system has striking parallels to quantum computing:

| Quantum Concept | Closure Analog |
|-----------------|----------------|
| Superposition | Forward + backward waves coexisting |
| Entanglement | Paths sharing common states |
| Measurement/Collapse | Closure event (waves "agree" on value) |
| Interference | Constructive: closures reinforce; Destructive: contradictions |
| Normalization loss | Energy damping in wave system |

### Complex Wave System

```python
class QuantumClosureSystem:
    def __init__(self, game, n_qubits=10):
        self.amplitudes: Dict[int, complex] = {}  # Complex-valued
        self.phases: Dict[int, float] = {}

    def superpose(self, states: List[Any]):
        """Create equal superposition over states."""
        amp = 1.0 / len(states) ** 0.5
        for state in states:
            h = self.game.hash_state(state)
            self.amplitudes[h] = complex(amp, 0)

    def evolve(self):
        """Unitary evolution step - distribute amplitude to successors."""
        ...

    def measure(self) -> List[Tuple[int, float]]:
        """Collapse superposition, return observable states."""
        # Probability = |amplitude|^2
        probs = [(h, abs(amp)**2) for h, amp in self.amplitudes.items()]
        return sorted(probs, key=lambda x: -x[1])[:self.n_qubits]
```

### The 2^n Cost

The exponential cost is unavoidable. For game search:
- Connect4 has ~4.5 × 10¹² states ≈ 2^42
- Full search costs O(2^42) regardless of method
- Compression reduces STORAGE but not fundamental COMPUTE

However, the closure system offers:
1. **Early termination**: Stop when closure density saturates
2. **Selective amplification**: Focus energy on high-pressure regions
3. **Coherent subsets**: Only track states that survive entropy filtering

---

## Mode Emergence

Modes emerge from closure state, not explicit selection:

```python
class ModeEmergence:
    def get_emergent_mode(
        self,
        forward_frontier_size: int,
        backward_frontier_size: int,
        recent_closures: int,
        branching_factor: float
    ) -> str:
        # Lightning: Small frontiers, no closures yet
        if forward_frontier_size < 100 and backward_frontier_size < 100:
            if recent_closures == 0:
                return "lightning"

        # Crystal: Many closures (solidifying region)
        if recent_closures > 5:
            return "crystal"

        # Osmosis: Asymmetric frontiers (pressure differential)
        ratio = max(forward_frontier_size, backward_frontier_size) / \
                max(1, min(forward_frontier_size, backward_frontier_size))
        if ratio > 5:
            return "osmosis"

        # Wave: Default balanced expansion
        return "wave"
```

---

## Core APIs

### GameInterface

```python
class GameInterface(ABC, Generic[S, V]):
    def hash_state(self, state: S) -> int
    def get_successors(self, state: S) -> List[Tuple[S, Any]]
    def get_predecessors(self, state: S) -> List[Tuple[S, Any]]
    def is_boundary(self, state: S) -> bool
    def propagate_value(self, state: S, child_values: List[V]) -> Optional[V]
```

### Unified Search

```python
from holos_loop.unified_closure import run_unified_search, UnifiedConfig

config = UnifiedConfig(
    enable_quantum=True,
    enable_equivalence=True,
    crystal_on_closure=True,
    decrypt_on_interior=True
)

result, hologram = run_unified_search(
    game,
    start_states=positions,
    backward_states=terminals,
    max_iterations=100,
    config=config
)

print(f"Values computed: {len(hologram.solved)}")
```

### Wave System

```python
from holos_loop.wave_system import create_wave_system, run_wave_search

game = Connect4Game()
start = C4State()

result = run_wave_search(
    game,
    start_states=positions,
    max_iterations=50,
    energy=200.0
)

print(f"Spines found: {result['spines']}")
print(f"Closures: {result['closures']}")
```

### Closure Detection

```python
from holos_loop.closure import ClosureDetector, ModeEmergence

detector = ClosureDetector(phase_threshold=0.2)
emergence = ModeEmergence(detector)

event = detector.check_closure(
    state_hash=12345,
    forward_value=100.0,
    backward_value=98.0,
    layer=0,
    iteration=1
)

if event:
    print(f"Closure at {event.state_hash}: {event.closure_type}")

mode = emergence.get_emergent_mode(
    forward_frontier_size=500,
    backward_frontier_size=500,
    recent_closures=3,
    branching_factor=5.0
)
print(f"Emergent mode: {mode}")
```

---

## Test Results

### Closure Layer Tests (9 passing)

| Test | Key Metrics |
|------|-------------|
| Closure Detection | Phase alignment works correctly |
| Phase Closure | Closed/open sequences detected |
| Irreducibility | Long constrained > short many-alt |
| Mode Emergence | lightning/crystal/osmosis/wave emerge correctly |
| Layer 1 Paths | 58 spines from 4 seeds |
| Layer 2 Covers | 51 minimal covers from 5 paths |
| Layer 3 Policies | 12 complete policies |
| Wave System Basic | 4 layers initialized |
| Wave System Full | 2 spines, 18 paths found |

### Unified System (Connect4)

With 50 forward seeds and 26 backward seeds (reachable terminals):

```
Iterations: 80
States explored: 5598
Closures: 748
Values computed: 3255 (58% of explored!)
Equivalence classes: 181
Interiors formed: 725
Mode emerged: crystal (many closures)
```

---

## Performance Estimation: Connect4

### State Space

- **4.5 trillion** legal positions (4.5 × 10¹²)
- First player (X) wins with perfect play

### Compression Opportunities

1. **Horizontal symmetry**: 2x compression
2. **Feature equivalence**: 10-100x on some regions
3. **Seed-based reconstruction**: 1000x+ for dense regions

### Full Solve Estimates

| Metric | Without Compression | With Full Compression |
|--------|--------------------|-----------------------|
| States | 4.5 × 10¹² | 4.5 × 10¹² (same) |
| Storage | 4.5 TB | 5-50 GB |
| Compute | ~10¹⁴ ops | ~10¹⁴ ops (same) |
| Time (single core) | ~1 year | ~1 year |
| Time (1000 cores) | ~8 hours | ~8 hours |

Compression helps storage dramatically but doesn't reduce fundamental compute.
Closure-aware search helps PRIORITIZE compute, not reduce it.

---

## Implementation Status

### Completed Features

| Feature | Module | Status |
|---------|--------|--------|
| Pressure dynamics | `pressure.py` | ✓ Continuous expansion |
| Interior detection | `pressure.py`, `unified_closure.py` | ✓ With decryption |
| Value propagation | `quantum_closure.py`, `unified_closure.py` | ✓ Minimax through graph |
| Equivalence closure | `quantum_closure.py`, `unified_closure.py` | ✓ Feature-based |
| Quantum amplitudes | `quantum_closure.py`, `unified_closure.py` | ✓ Complex tracking |
| Mode emergence | `closure.py`, `unified_closure.py` | ✓ From closure state |
| Crystallization | `unified_closure.py` | ✓ Around closures |
| Hologram output | `unified_closure.py` | ✓ Standard format |

### Pending

- GPU parallelization (would require wave operation batching)
- Distributed solving coordination
- Full Connect4 benchmark
- Adaptive impedance (layers adjust based on problem structure)

---

## Theoretical Foundation

### Why Closure-First?

The traditional approach treats search as optimization:
- Define objective (coverage, efficiency)
- Choose mode to optimize it
- Evaluate results

The closure approach treats search as physics:
- Wave propagates through space
- Closures occur where waves meet and values align
- Modes are natural response to closure conditions

This changes what we optimize:
- Traditional: "Which mode maximizes coverage?"
- Closure: "Under what conditions do modes naturally emerge?"

### Irreducibility as Value

Not all closures are equal. Irreducible closures:
- Cannot be decomposed into simpler closures
- Represent fundamental structure in the problem
- Are worth more attention/energy

```python
def estimate_irreducibility_from_path(
    path_length: int,
    branching_factor: float,
    alternatives: int
) -> float:
    """
    Longer paths through constrained spaces = more irreducible.
    Short paths with many alternatives = reducible.
    """
    length_factor = min(1.0, path_length / 20.0)
    constraint_factor = 1.0 / (1.0 + branching_factor)
    uniqueness_factor = 1.0 / (1.0 + alternatives)

    return length_factor * constraint_factor * uniqueness_factor
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

Key insight:
- **Fast medium**: Hash table lookups (O(1))
- **Slow medium**: Game tree expansion (O(branching^depth))
- **Daemon role**: Closure detector "reads" the fast medium to guide the slow one

---

## Files Marked for Consolidation

The following files may benefit from consolidation in future refactoring:

| File | Notes |
|------|-------|
| `layer1_paths.py` / `layer1_policy.py` | Duplicate naming - paths is canonical |
| `layer2_covers.py` / `layer2_policy.py` | Duplicate naming - covers is canonical |
| `layer3_policy.py` | Policy layer - correctly named, manages compute/storage budgets |
| `LOCUS_closures.md` | Superseded by this document |
| `LOCUS_closures_physics.md` | Superseded by this document |

---

## Future Work

1. **GPU parallelization**: Batch wave operations for parallel execution
2. **Distributed solving**: Pressure-based work distribution across nodes
3. **Resonance detection**: Multiple closures reinforcing each other
4. **Adaptive impedance**: Layers that adjust based on problem structure
5. **Cross-layer learning**: Transfer knowledge between layers
6. **Visualization**: See wave propagation and closure events
7. **Benchmarking**: Compare closure vs traditional on standard problems

---

*This document consolidates LOCUS_closures.md and LOCUS_closures_physics.md into a unified reference for the holos_loop module.*
