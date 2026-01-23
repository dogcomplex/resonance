# Comparison: holos/ vs fractal_holos3.py

This document details the differences between the modular `holos/` package and the original `fractal_holos3.py`, explaining each difference and ensuring functional consistency.

## Summary of Differences

| Feature | fractal_holos3.py | holos/ | Status |
|---------|-------------------|--------|--------|
| Architecture | Monolithic | Modular (game-agnostic) | Enhanced |
| Value type | `int` (-1, 0, 1) | `ChessValue` wrapper | OK (abstraction) |
| Lightning | Forward only | Forward + Backward | Enhanced |
| Hologram | Chess-specific | Game-agnostic | Enhanced |
| Spine creation | In lightning phase | In lightning phase | FIXED ✓ |
| Memory tracking | `psutil` based | `psutil` based | FIXED ✓ |
| Auto-generate backward seeds | Yes | Yes | FIXED ✓ |
| Save/Load hologram | Integrated | Separate module | OK |
| Equivalence tracking | In Hologram class | In solver + transfer | FIXED ✓ |
| Stats tracking | Detailed | Detailed | FIXED ✓ |

## Detailed Differences

### 1. Value Type Mismatch

**fractal_holos3.py**: Uses raw `int` values (-1, 0, 1) throughout
```python
# In hologram
solved: Dict[int, int]  # hash -> value as int

# In propagate
if 1 in child_values:
    return 1
```

**holos/**: Uses `ChessValue` wrapper
```python
# In hologram
solved: Dict[int, Any]  # hash -> ChessValue

# In propagate
if 1 in values:  # values extracted from ChessValue.value
    return ChessValue(1)
```

**Issue**: The modular version wraps values, but the solver's propagate logic works with the game's `propagate_value` method which returns wrapped values. This is CORRECT for abstraction but means the stored values differ in type.

**Resolution**: The abstraction is intentional - each game defines its own value type. The ChessGame correctly extracts `.value` when needed. No fix required.

### 2. Lightning Probe Differences

**fractal_holos3.py**:
```python
class LightningProbe:
    def __init__(self, syzygy, hologram, max_depth=15):
        self.syzygy = syzygy  # Direct syzygy access
        self.hologram = hologram  # Direct hologram access

    def _search_captures(self, state, depth, path):
        # Check hologram first
        cached = self.hologram.query(h)

        # Check syzygy boundary
        if state.piece_count() <= 7:
            value = self.syzygy.probe(state)
```

**holos/core.py**:
```python
class LightningProbe:
    def __init__(self, game, solved, direction="forward", ...):
        self.game = game  # Uses GameInterface
        self.solved = solved  # Dict passed in

    def _search(self, state, depth, path, visited):
        # Check if already solved
        if h in self.solved:
            return self.solved[h]

        # Check boundary via game interface
        if self.game.is_boundary(state):
            value = self.game.get_boundary_value(state)
```

**Difference**: The modular version uses the game interface abstraction instead of direct syzygy/hologram access. This is the CORRECT approach for game-agnostic design.

**Issue**: The original also adds **cycle detection** implicitly via path tracking. The modular version has explicit `visited` set.

**Resolution**: Modular version is enhanced with explicit cycle detection. OK.

### 3. Missing Spine Creation in Modular Version

**fractal_holos3.py** creates spines properly:
```python
def _lightning_phase(self):
    if value is not None and path:
        spine = SpinePath(
            start_hash=h,
            moves=moves,
            end_hash=hash(end_state),
            end_value=value,
            depth=len(moves)
        )
        self.hologram.add_spine(spine)
```

**holos/core.py** just stores values:
```python
def _lightning_probe(self, state, direction):
    if value is not None and path:
        h = self.game.hash_state(state)
        self.solved[h] = value
        # NO SPINE CREATION!
```

**Issue**: Spines are NOT being created in the modular version!

**Resolution**: Need to add spine creation. See fix below.

### 4. Missing Auto-Generate Backward Seeds

**fractal_holos3.py**:
```python
def solve(self, forward_starts, backward_starts=None, ...):
    if backward_starts is None:
        backward_starts = self._generate_boundary_positions(forward_starts[0])
```

**holos/core.py**:
```python
def solve(self, forward_seeds, backward_seeds=None, ...):
    if backward_seeds:
        # ... initialize
    # NO AUTO-GENERATION!
```

**Issue**: Modular version requires backward seeds to be provided.

**Resolution**: Add auto-generation capability. See fix below.

### 5. Missing Memory Tracking

**fractal_holos3.py**:
```python
def memory_mb(self):
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except:
        return (len(self.forward_frontier) + len(self.backward_frontier)) * 300 / (1024 * 1024)

def solve(self, ...):
    for iteration in range(max_iterations):
        mem = self.memory_mb()
        if mem > self.max_memory_mb * 0.9:
            print(f"\nMemory limit reached ({mem:.0f} MB)")
            break
```

**holos/core.py**: No memory tracking at all.

**Resolution**: Add memory tracking to solver. See fix below.

### 6. Equivalence Tracking Location

**fractal_holos3.py**: In Hologram class
```python
class Hologram:
    equiv_classes: Dict[ChessFeatures, Set[int]]
    equiv_outcomes: Dict[ChessFeatures, Optional[int]]

    def add_with_features(self, h, value, features):
        self.solved[h] = value
        self.equiv_classes[features].add(h)
        self._update_equiv_outcome(features, value)
```

**holos/**: Split between solver and storage
```python
# In HOLOSSolver
self.equiv_classes: Dict[Any, Set[int]]
self.equiv_outcomes: Dict[Any, Optional[V]]

# In Hologram (storage.py)
equiv_classes: Dict[Any, Set[int]]
equiv_outcomes: Dict[Any, Optional[Any]]
```

**Issue**: Equivalence tracking is duplicated.

**Resolution**: The solver tracks during solving, hologram stores final results. This is actually OK - the solver builds up equivalence info, then transfers to hologram. Need to ensure proper transfer.

### 7. Stats Tracking

**fractal_holos3.py**:
```python
self.stats = {
    'lightning_probes': 0,
    'connections': 0,
    'crystallized': 0,
    'spines_found': 0,
    'equiv_shortcuts': 0,
    'equiv_tracked': 0,
    'equiv_propagated': 0,
    'minimax_solved': 0,
}
```

**holos/core.py**:
```python
self.stats = {
    'forward_expanded': 0,
    'backward_expanded': 0,
    'lightning_probes': 0,
    'connections': 0,
    'equiv_shortcuts': 0,
}
```

**Issue**: Missing several stat keys.

**Resolution**: Add missing stats. See fix below.

## Applied Fixes

All fixes have been applied and tested:

### Fix 1: Spine creation in lightning phase ✓
Added spine creation in `_lightning_phase()` in core.py

### Fix 2: Auto-generation of backward seeds ✓
Added `generate_boundary_seeds()` to GameInterface and ChessGame

### Fix 3: Memory tracking ✓
Added `memory_mb()` method with psutil support to HOLOSSolver

### Fix 4: Missing stats ✓
Added all stats keys: crystallized, spines_found, equiv_tracked, equiv_propagated, minimax_solved

### Fix 5: Transfer equivalence classes to hologram ✓
Added transfer of equiv_classes and equiv_outcomes in solve()

### Fix 6: apply_move method ✓
Added `apply_move()` to GameInterface and ChessGame for spine tracking

## Functional Equivalence Test

Run `python holos/test_equivalence.py` to verify:
- All 7 tests pass
- ChessGame interface complete
- Boundary seed generation works
- All stats keys present
- Memory tracking functional
- Spine structures correct
- Lightning probes bidirectional
- apply_move matches get_successors

The modular version adds:
1. Backward lightning probes (enhancement)
2. Game-agnostic design (enhancement)
3. Mode selection tracking (enhancement)
