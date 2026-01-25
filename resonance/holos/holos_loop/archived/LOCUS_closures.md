# LOCUS_closures.md - Closure-Aware Layer Architecture

**Last Updated**: 2026-01-24
**Version**: 0.1.0
**Status**: Parallel implementation alongside main HOLOS system
**Purpose**: Document the closure-aware reformulation of the layer architecture

---

## Executive Summary

This document describes an alternative layer architecture that reformulates HOLOS layers around **closure events** rather than discrete optimization targets. The key insight comes from neuroscience: modes (lightning, wave, crystal, osmosis) are **readouts** of closure conditions, not choices to be made.

The closure-aware system runs in parallel with the existing HOLOS implementation for phased integration.

---

## The Closure Insight

### From Neuroscience to Search

Brain rhythms (alpha, beta, gamma) were traditionally viewed as "clocks" driving neural activity. Recent research shows they're actually **closure detectors** - they emerge when certain computational loops complete.

Applied to HOLOS:
- **Traditional view**: We CHOOSE lightning/wave/crystal/osmosis mode
- **Closure view**: Modes EMERGE based on whether closure conditions are met

### Three Closure Layers

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

---

## Layer Reformulation

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

---

## File Manifest

```
holos/
├── closure.py           [~300 lines]
│   Purpose: Core closure detection primitives
│   Classes: ClosureType, ClosureEvent, ClosureDetector,
│            PhaseAlignment, ModeEmergence, SimpleIrreducibilityChecker
│   Functions: compute_phase_closure, estimate_irreducibility_from_path
│
├── layer1_paths.py      [~450 lines]
│   Purpose: Path scale implementation
│   Classes: PartialPath, PathValue, PathGame, PathLayerSolver
│   Key: Sequences of states, extend/connect operations
│
├── layer2_covers.py     [~400 lines]
│   Purpose: Cover scale implementation
│   Classes: PathCover, CoverValue, CoverGame, CoverLayerSolver
│   Key: SEQUENCES of paths (ordered by priority)
│
├── layer3_policy.py     [~350 lines]
│   Purpose: Policy scale implementation
│   Classes: CoverPolicy, PolicyValue, PolicyGame, PolicyLayerSolver
│   Key: Compute/storage budget constraints
│
├── wave_system.py       [~400 lines]
│   Purpose: Unified multi-layer wave orchestration
│   Classes: WaveSystem, WaveState, LayerMedium
│   Key: One wave function, multi-scale medium, energy transfer
│
└── test_closure_layers.py [~450 lines]
    Purpose: Test suite for closure-aware system
    Tests: Closure detection, phase closure, irreducibility,
           mode emergence, all layer solvers, wave system
```

---

## Core Concepts

### ClosureEvent

```python
@dataclass
class ClosureEvent:
    state_hash: int           # Where closure occurred
    layer: int                # Which layer (0-3)
    closure_type: ClosureType # REDUCIBLE, IRREDUCIBLE, RESONANT
    phase_diff: float         # Forward/backward value difference
    forward_value: Any        # Forward wave's value
    backward_value: Any       # Backward wave's value
    iteration: int            # When detected
    paths_through: int        # How many paths pass through
    local_branching: float    # Local graph complexity

    @property
    def is_irreducible(self) -> bool:
        return self.closure_type == ClosureType.IRREDUCIBLE
```

### Closure Types

```python
class ClosureType(Enum):
    NONE = "none"           # No closure (waves don't meet)
    REDUCIBLE = "reducible" # Can be decomposed
    IRREDUCIBLE = "irreducible"  # Cannot simplify - valuable!
    RESONANT = "resonant"   # Multiple closures reinforce
```

### Phase Alignment

Closure occurs when forward and backward waves **agree on value**:

```python
@dataclass
class PhaseAlignment:
    forward_value: float
    backward_value: float
    threshold: float = 0.2  # 20% tolerance

    @property
    def phase_diff(self) -> float:
        if self.forward_value == 0:
            return float('inf')
        return abs(self.forward_value - self.backward_value) / abs(self.forward_value)

    @property
    def is_aligned(self) -> bool:
        return self.phase_diff <= self.threshold
```

### Mode Emergence

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

## Layer Details

### Layer 1: Paths (layer1_paths.py)

**State**: PartialPath - a sequence of (hash, move) pairs

```python
@dataclass(frozen=True)
class PartialPath:
    steps: Tuple[Tuple[int, Any], ...]  # ((hash1, move1), ...)
    start_hash: int
    end_hash: int
    origin: WaveOrigin  # FORWARD, BACKWARD
    _start_state: Any = None
    _end_state: Any = None

    def extend_forward(self, next_hash, move, next_state) -> 'PartialPath':
        """Add step at the end (forward expansion)"""
        new_steps = self.steps + ((next_hash, move),)
        return PartialPath(
            steps=new_steps,
            start_hash=self.start_hash,
            end_hash=next_hash,
            origin=self.origin,
            _start_state=self._start_state,
            _end_state=next_state
        )

    def connect(self, other: 'PartialPath') -> Optional['PartialPath']:
        """Connect two paths if they meet"""
        if self.end_hash != other.start_hash:
            return None
        # ... merge logic
```

**Successors**: Extend path forward (add state at end)
**Predecessors**: Extend path backward (add state at start)
**Boundary**: Paths reaching Layer 0 boundary

### Layer 2: Covers (layer2_covers.py)

**State**: PathCover - an ordered sequence of path hashes

```python
@dataclass(frozen=True)
class PathCover:
    paths: Tuple[int, ...]  # Ordered by priority

    def add_path(self, path_hash, path_obj) -> 'PathCover':
        """Add path at end (lowest priority)"""
        return PathCover(paths=self.paths + (path_hash,))

    def promote(self, index: int) -> 'PathCover':
        """Move path to higher priority"""
        if index == 0:
            return self
        paths_list = list(self.paths)
        paths_list[index-1], paths_list[index] = paths_list[index], paths_list[index-1]
        return PathCover(paths=tuple(paths_list))

    def swap(self, i: int, j: int) -> 'PathCover':
        """Swap priority of two paths"""
        paths_list = list(self.paths)
        paths_list[i], paths_list[j] = paths_list[j], paths_list[i]
        return PathCover(paths=tuple(paths_list))
```

**Successors**: Add path, reorder paths
**Predecessors**: Remove path, reverse reordering
**Boundary**: Complete covers achieving target coverage

### Layer 3: Policies (layer3_policy.py)

**State**: CoverPolicy - ordered covers with resource tracking

```python
@dataclass(frozen=True)
class CoverPolicy:
    covers: Tuple[int, ...]  # Ordered by priority
    total_compute: float = 0.0
    total_storage: float = 0.0

    def add_cover(self, cover_hash, compute_cost, storage_cost) -> 'CoverPolicy':
        """Add cover if within budgets"""
        return CoverPolicy(
            covers=self.covers + (cover_hash,),
            total_compute=self.total_compute + compute_cost,
            total_storage=self.total_storage + storage_cost
        )
```

**Successors**: Add cover (within budget)
**Predecessors**: Remove cover
**Boundary**: Complete policies covering all problems

---

## Wave System (wave_system.py)

### The Physical Analogy

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

### WaveSystem API

```python
# Create system
system = create_wave_system(game, total_energy=100.0)

# Setup with seeds
system.setup(forward_seeds=positions, backward_seeds=None)

# Run until convergence
result = system.run(max_iterations=100, target_closures=50)

# Or step manually
while not done:
    step_result = system.step()
    # step_result contains closures, modes, energy distribution
```

---

## Test Results Summary

All 9 tests pass:

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

---

## Comparison with LOCUS.md (Original System)

### What's Preserved

| Concept | Original | Closure-Aware |
|---------|----------|---------------|
| Bidirectional search | Forward/backward waves | Same |
| GameInterface | Generic S, V types | Same |
| Boundary conditions | Known values at edges | Same |
| Four modes | lightning/wave/crystal/osmosis | Same names |
| Layer 0 games | Connect4, Chess, Sudoku | Reused |

### What's Different

| Aspect | Original | Closure-Aware |
|--------|----------|---------------|
| Mode selection | Explicit choice or heuristic | Emergent from closure state |
| Layer 1 state | TacticalSeed (single seed params) | PartialPath (sequence of states) |
| Layer 2 state | StrategyState (seed set) | PathCover (ordered sequence) |
| Layer 3 state | ComputeStoragePolicy | CoverPolicy (sequence of covers) |
| Layer coupling | Separate solvers | Single wave with energy transfer |
| Sequences vs sets | Seeds as sets | Explicit priority ordering |

### What's Missing (Gaps to Address)

1. **Compression metrics**: Original has detailed compression-aware efficiency (net_savings). Closure system focuses on closure quality, not storage.

2. **Goal targeting**: Original has GoalCondition for filtering. Not yet in closure system.

3. **Session management**: Original has SessionManager for multi-round solving. Closure system has WaveSystem.run() but no persistence.

4. **Spine storage**: Original has SpinePath with checkpoints. Closure system has simpler SpinePath.

5. **Equivalence classes**: Original tracks features for propagation. Not yet in closure system.

6. **Meta-seed compression**: Original explores hierarchical seed storage. Not addressed.

---

## Integration Path

### Phase 1: Parallel Operation (Current)

- Closure system runs independently
- Test on same games (Connect4, etc.)
- Compare results

### Phase 2: Closure Detection in Original

- Add ClosureDetector to HOLOSSolver
- Track closure events during normal solving
- Mode emergence as optional feature

### Phase 3: Layer Interop

- Use closure-aware Layer 1 paths with original Layer 0
- Feed results to original compression analysis
- Compare efficiency

### Phase 4: Full Integration

- Unified WaveSystem wrapping HOLOSSolver
- Energy-based layer coordination
- Preserve best of both approaches

---

## Usage Examples

### Basic Wave Search

```python
from holos.wave_system import create_wave_system, run_wave_search
from holos.games.connect4 import Connect4Game, C4State

game = Connect4Game()
start = C4State()

# Build position pool
positions = [start]
for child, col in game.get_successors(start):
    positions.append(child)

# Run wave search
result = run_wave_search(
    game,
    start_states=positions,
    max_iterations=50,
    energy=200.0
)

print(f"Spines found: {result['spines']}")
print(f"Closures: {result['closures']}")
```

### Layer 1 Path Solving

```python
from holos.layer1_paths import create_path_solver

solver = create_path_solver(game, max_path_length=20)

spines = solver.solve(
    forward_seeds=positions,
    backward_seeds=None,
    max_iterations=30,
    mode="balanced"
)

for spine in spines[:5]:
    print(f"Path: {spine.moves[:5]}... -> {spine.end_value}")
```

### Closure Detection

```python
from holos.closure import ClosureDetector, ModeEmergence

detector = ClosureDetector(phase_threshold=0.2)
emergence = ModeEmergence(detector)

# Check for closure
event = detector.check_closure(
    state_hash=12345,
    forward_value=100.0,
    backward_value=98.0,
    layer=0,
    iteration=1
)

if event:
    print(f"Closure at {event.state_hash}: {event.closure_type}")

# Get emergent mode
mode = emergence.get_emergent_mode(
    forward_frontier_size=500,
    backward_frontier_size=500,
    recent_closures=3,
    branching_factor=5.0
)
print(f"Emergent mode: {mode}")
```

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

This is more than philosophy - it changes what we optimize:
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

### Energy Conservation

The wave system conserves energy (with damping):
- Total energy is finite
- Each layer step costs energy (damping)
- Closure events transfer energy between layers
- System stops when energy exhausted

This prevents infinite expansion and naturally balances exploration/exploitation.

---

## Future Work

1. **Resonance detection**: Multiple closures reinforcing each other
2. **Adaptive impedance**: Layers that adjust based on problem structure
3. **Cross-layer learning**: Transfer knowledge between layers
4. **Visualization**: See wave propagation and closure events
5. **Benchmarking**: Compare closure vs traditional on standard problems

---

## Implementation Status (2026-01-25)

### Completed Integrations

All features from `LOCUS_closures_physics.md` are now implemented:

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

### Test Results (Connect4)

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

### New Files

```
holos/
├── closure.py           # Core closure detection (existing)
├── pressure.py          # Pressure-based continuous expansion
├── quantum_closure.py   # Quantum-inspired system
├── unified_closure.py   # Full integration of all features
├── LOCUS_closures.md    # This document
└── LOCUS_closures_physics.md  # Physics exploration
```

### Usage

```python
from holos.unified_closure import run_unified_search, UnifiedConfig

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

# Result contains stats, hologram is standard HOLOS format
print(f"Values computed: {len(hologram.solved)}")
```

---

*This document describes the closure-aware reformulation of HOLOS. See LOCUS.md for the original architecture.*
