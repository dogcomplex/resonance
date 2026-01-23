# HOLOS Memory and Caching Analysis

## Current State

### The Problem

The full run showed:
```
Iteration 3:
  Forward: 0, Backward: 12,766,634
  Solved: 7,705,892, Memory: 21040 MB
Memory limit reached (38117 MB)
```

**The backward frontier exploded to 12.7M positions**, consuming ~38GB of memory and halting the search.

### Root Cause Analysis

#### 1. Unbounded Frontier Growth

The current algorithm (holos.py lines 716-792) expands **all** predecessors without limit:

```python
def _expand_backward(self, max_frontier_size: int = 500000) -> int:
    # Sample if frontier is too large
    items = list(self.backward_frontier.items())
    if len(items) > max_frontier_size:
        items = random.sample(items, max_frontier_size)

    next_frontier = {}
    for h, state in items:
        for pred, move in self.game.get_predecessors(state):  # EXPLOSION
            # ... adds to next_frontier
```

**Issue**: Even with sampling, `next_frontier` grows unboundedly because each sampled state can have many predecessors. With 7-piece chess:
- Each position has ~50-200 predecessors (uncaptures)
- 500,000 positions × 100 predecessors = 50M new states per iteration

#### 2. Memory Storage Structure

The solver keeps EVERYTHING in memory:
- `forward_frontier: Dict[int, S]` - Current forward wave
- `backward_frontier: Dict[int, S]` - Current backward wave
- `forward_seen: Set[int]` - ALL hashes ever seen forward
- `backward_seen: Set[int]` - ALL hashes ever seen backward
- `solved: Dict[int, V]` - ALL solved positions
- `forward_parents: Dict[int, Tuple[int, Any]]` - Parent links
- `backward_parents: Dict[int, Tuple[int, Any]]` - Parent links
- `equiv_classes: Dict[Any, Set[int]]` - Equivalence tracking

**Memory per position** (estimated):
- ChessState: ~300 bytes
- Hash (int): 8 bytes
- Dict overhead: ~50 bytes per entry
- Total: ~360 bytes per position

**12.7M positions × 360 bytes = 4.5GB just for backward_frontier**

Plus all the tracking structures, it balloons to 38GB.

#### 3. No Persistent Caching

Current storage (storage.py) only saves at the END:
```python
def save(self, path: str):
    data = {
        'name': self.name,
        'solved': self.solved,  # ALL solved positions
        'spines': self.spines,
        ...
    }
```

**Problems**:
1. No incremental saves during solving
2. No resume from partial progress
3. No disk-based frontier management
4. No deduplication across runs

---

## Design Recommendations

### 1. Memory-Bounded Frontiers

**Current**: `max_frontier_size` caps input but not output
**Needed**: Hard cap on total frontier size with intelligent pruning

```python
class MemoryBoundedSolver:
    def __init__(self, max_frontier_positions: int = 1_000_000):
        self.max_frontier = max_frontier_positions

    def _expand_backward(self):
        next_frontier = {}

        for h, state in self.backward_frontier.items():
            for pred, move in self.game.get_predecessors(state):
                # Hard cap - stop adding when full
                if len(next_frontier) >= self.max_frontier:
                    break  # This batch is done
                ...

        # If still too big, prioritize by value/depth
        if len(next_frontier) > self.max_frontier:
            # Priority pruning
            next_frontier = self._prioritized_sample(next_frontier)
```

### 2. Depth-Limited Waves

**Key insight**: Don't expand infinitely. For targeted search:

```python
def solve(self, max_backward_depth: int = 3):
    """
    Backward wave should only go a few layers:
    - Depth 1: 7-piece → 8-piece (our target)
    - Depth 2: Might find some 8-piece via uncapture-recapture
    - Depth 3+: Diminishing returns, memory explosion
    """
    self.backward_depth_limit = max_backward_depth
```

For KQRRvKQR targeting:
- We want 8-piece positions that lead to 7-piece KQRRvKQR
- This is exactly **1 backward step**
- Going deeper finds 9-piece, 10-piece... not our goal

### 3. Seed-Based Frontier Reconstruction

The storage.py already has `SeedFrontierMapping` but it's not used during solving:

```python
@dataclass
class SeedFrontierMapping:
    """Store seed + depth, reconstruct frontier on demand"""
    seed_hash: int
    depth: int
    mode: str
    # Frontier is DERIVED, not stored
```

**Recommendation**: Use this for memory management:

```python
class DiskBackedFrontier:
    def __init__(self, cache_dir: str, max_memory_mb: int = 4000):
        self.cache_dir = cache_dir
        self.max_memory_mb = max_memory_mb
        self.in_memory: Dict[int, S] = {}
        self.on_disk: Set[int] = set()  # Just hashes of spilled positions

    def add(self, h: int, state: S):
        if self.memory_mb() > self.max_memory_mb * 0.8:
            self._spill_to_disk()
        self.in_memory[h] = state

    def _spill_to_disk(self):
        """Write oldest entries to disk"""
        # Save oldest 50% to disk
        to_spill = list(self.in_memory.items())[:len(self.in_memory) // 2]
        batch_file = f"{self.cache_dir}/frontier_batch_{uuid4()}.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(to_spill, f)
        for h, _ in to_spill:
            del self.in_memory[h]
            self.on_disk.add(h)
```

### 4. Incremental Persistent Cache

**Design**: Cache solved positions to disk, indexed by material signature.

```
holos_cache/
├── index.json              # Material → files mapping
├── solved/
│   ├── KQRRvKQR.pkl       # All solved 7-piece positions
│   ├── KQRRvKQRQ.pkl      # 8-piece positions
│   └── ...
├── spines/
│   ├── spine_001.pkl
│   └── ...
└── frontiers/
    ├── batch_001.pkl      # Frontier spillover
    └── ...
```

**Index structure**:
```python
{
    "materials": {
        "KQRRvKQR": {
            "solved_file": "solved/KQRRvKQR.pkl",
            "count": 1_234_567,
            "complete": False,  # Are ALL positions of this material solved?
            "goals_targeted": ["KQRRvKQR"],  # What goals contributed
        }
    },
    "goals": {
        "KQRRvKQR_only": {
            "target_signatures": ["KQRRvKQR"],
            "runs": [
                {"date": "2026-01-22", "solved": 13_571_583, "batch": 1}
            ]
        }
    }
}
```

### 5. Conservative Cache Loading

**Key principle**: Never claim more than we've proven.

```python
class ConservativeCache:
    def load_solved(self, material: str, goal: GoalCondition) -> Dict[int, V]:
        """
        Load solved positions, but ONLY if compatible with goal.

        - If goal is None: Load everything for this material
        - If goal targets this material: Load (we proved it leads here)
        - If goal targets different material: DON'T load (we only know
          these positions exist, not that they connect to our goal)
        """
        metadata = self._load_metadata(material)

        # Check if cached data is compatible with our goal
        if goal is not None:
            if material in goal.target_signatures:
                # This is our target - load boundary values
                return self._load_file(metadata['solved_file'])
            elif metadata.get('complete'):
                # Material fully solved, safe to use
                return self._load_file(metadata['solved_file'])
            else:
                # Partial solve for different goal - don't use
                return {}
        return self._load_file(metadata['solved_file'])
```

### 6. Resumable Solving

```python
class ResumableSolver:
    def solve(self, goal: GoalCondition, cache_dir: str,
              checkpoint_interval: int = 5):

        # Check for existing progress
        checkpoint = self._load_checkpoint(cache_dir, goal)
        if checkpoint:
            self._restore_state(checkpoint)
            print(f"Resumed from iteration {checkpoint['iteration']}")

        for iteration in range(self.current_iteration, max_iterations):
            # ... normal solving ...

            if iteration % checkpoint_interval == 0:
                self._save_checkpoint(cache_dir, goal, iteration)

    def _save_checkpoint(self, cache_dir, goal, iteration):
        checkpoint = {
            'iteration': iteration,
            'goal': goal,
            'forward_seen': self.forward_seen,
            'backward_seen': self.backward_seen,
            'solved': self.solved,
            # Frontier is reconstructible from seeds
            'forward_seeds': self._extract_seeds(self.forward_frontier),
            'backward_seeds': self._extract_seeds(self.backward_frontier),
        }
        path = f"{cache_dir}/checkpoint_{goal.name}_{iteration}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
```

---

## Immediate Fixes (Low-Hanging Fruit)

### Fix 1: Hard Frontier Cap

Add to `_expand_backward`:

```python
def _expand_backward(self, max_frontier_size: int = 500_000,
                     max_output_size: int = 2_000_000) -> int:
    next_frontier = {}

    for h, state in items:
        if len(next_frontier) >= max_output_size:
            print(f"  Frontier cap reached ({max_output_size:,})")
            break
        ...

    # Don't let it grow unboundedly
    self.backward_frontier = next_frontier
```

### Fix 2: Depth Limiting for Targeted Search

For `TargetedChessGame`, we only need depth=1 backward:

```python
def solve_targeted(self, goal, max_backward_depth: int = 1):
    """For targeted search, limit backward expansion depth."""
    self.max_backward_depth = max_backward_depth
    # In _expand_backward, track depth and stop when reached
```

### Fix 3: Clear Parent Tracking When Not Needed

The parent tracking (`forward_parents`, `backward_parents`) is only needed for spine reconstruction. After crystallization:

```python
def _crystallize(self):
    # ... existing code ...

    # Clear parent tracking to free memory
    if len(self.solved) > 5_000_000:
        self.forward_parents.clear()
        self.backward_parents.clear()
        print("  Cleared parent tracking to save memory")
```

### Fix 4: Sample Large Equivalence Classes

```python
def _update_equiv_outcome(self, features, value):
    # Don't track millions of positions in same equivalence class
    MAX_EQUIV_SIZE = 10_000
    if features in self.equiv_classes:
        if len(self.equiv_classes[features]) >= MAX_EQUIV_SIZE:
            return  # Don't track more
```

---

## Caching Evaluation

### Current State

**What's stored**:
- `Hologram.solved`: All hash → value mappings
- `Hologram.spines`: Principal variations
- `Hologram.equiv_classes`: Equivalence tracking
- `Hologram.seed_mappings`: Seeds for reconstruction

**What's NOT stored**:
- Frontier state (cannot resume)
- Goal metadata (cannot verify compatibility)
- Intermediate checkpoints
- Material-indexed lookups

### Recommendation: Multi-Level Cache

```
Level 1: Memory (hot data)
├── Current frontiers
├── Recently solved positions
└── Active equivalence classes

Level 2: Local disk (warm data)
├── Checkpoints every N iterations
├── Material-indexed solved positions
└── Completed spines

Level 3: Compressed archive (cold data)
├── Full holograms per goal
├── Spine libraries
└── Complete material solutions
```

### Safe Incremental Building

**Principle**: Each run adds to the solution set, never replaces.

```python
def merge_with_cache(new_hologram: Hologram, cache_dir: str,
                     goal: GoalCondition):
    """Safely merge new results with existing cache."""

    for material in get_materials_in_hologram(new_hologram):
        cache_file = f"{cache_dir}/solved/{material}.pkl"

        if os.path.exists(cache_file):
            existing = load_solved(cache_file)
            # MERGE: new solutions add to, never replace
            for h, v in new_hologram.solved.items():
                if get_material(h) == material:
                    if h not in existing:
                        existing[h] = v
                    # If already exists, verify consistency
                    elif existing[h] != v:
                        log_warning(f"Inconsistent value for {h}")
            save_solved(cache_file, existing)
        else:
            # New material - save directly
            save_new(cache_file, new_hologram, material)
```

---

## Recommended Implementation Priority

1. **Immediate** (fix current crash):
   - Hard frontier cap in `_expand_backward`
   - Clear parent tracking after crystallization

2. **Short-term** (this session):
   - Depth-limited backward expansion for targeted search
   - Checkpoint saving every N iterations

3. **Medium-term** (next session):
   - Disk-backed frontier management
   - Material-indexed cache structure

4. **Long-term**:
   - Full incremental caching system
   - Distributed solving support

---

## For KQRRvKQR Specifically

The goal is to find ALL 8-piece positions leading to 7-piece KQRRvKQR.

**Efficient approach**:
1. Generate 7-piece KQRRvKQR boundary seeds (done - 1000 positions)
2. Backward expand ONCE to get 8-piece parents
3. Forward expand from random 8-piece positions
4. Connect forward to backward (the 8-piece layer)
5. Repeat with more seeds until convergence

**Memory estimate**:
- 7-piece KQRRvKQR: ~500M legal positions
- 8-piece leading to it: ~50-100M positions
- With proper caching: Should fit in ~10-20GB working set

The current explosion happens because backward goes too deep (7→8→9→10→...).
