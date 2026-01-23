# Memory Analysis: fractal_holos.py vs fractal_holos_gemini.py

## Target: 10-60GB Reliable Memory Usage

---

## File 1: `fractal_holos.py` (In-Memory Solver)

### Current Architecture
```
RAM: frontier, solved, children, parents, all_seen, boundary_contacts
```

### Memory Problems Identified

#### 1. **Children Dict is MASSIVE** (Lines 354, 640)
```python
self.children: Dict[int, List[int]] = {}
# ...
self.children[h] = child_hashes
```
**Problem**: Storing full lists of child hashes. With 8-piece chess having ~30-50 moves per position and millions of positions, this explodes:
- 10M positions × 40 children × 8 bytes = **3.2GB just for child lists**
- Plus dict overhead: another ~2GB

#### 2. **Parents DefaultDict Doubles Memory** (Lines 355, 615)
```python
self.parents: Dict[int, List[int]] = defaultdict(list)
# ...
self.parents[ch].append(h)
```
**Problem**: Every edge stored TWICE (once in children, once in parents). This is redundant.

#### 3. **all_seen Set Grows Unbounded** (Line 357)
```python
self.all_seen: Set[int] = set()
```
**Problem**: Never cleared until region crystallizes. With 10M+ positions, that's ~600MB just for hashes.

#### 4. **Memory Estimation is Wrong** (Lines 373-392)
```python
def memory_usage_mb(self) -> float:
    try:
        import psutil
        # ...
    except ImportError:
        # Fallback estimates are WAY too low
        frontier_mb = len(self.frontier) * 300 / (1024 * 1024)  # Should be higher
```
**Problem**: The fallback estimates undercount by 3-5x. Python objects have significant overhead.

#### 5. **Frontier Stores Full ChessState Objects** (Line 353)
```python
self.frontier: Dict[int, ChessState] = {}
```
**Problem**: Each ChessState has:
- `pieces`: tuple of tuples ~200 bytes
- `turn`: string ~50 bytes
- `_hash`: int ~28 bytes
- Object overhead: ~56 bytes

Total: ~350 bytes per state, not 300.

### Memory Bloat Summary for fractal_holos.py
| Structure | 10M positions | Estimate |
|-----------|--------------|----------|
| frontier | 10M × 350B | 3.5 GB |
| solved | 5M × 40B | 0.2 GB |
| children | 10M × 40 × 8B | 3.2 GB |
| parents | 10M × 2 × 8B | 0.16 GB |
| all_seen | 10M × 60B | 0.6 GB |
| **TOTAL** | | **~7.7 GB** |

But Python dict overhead adds 30-50%, so actual: **~10-12GB for 10M positions**.

---

## File 2: `fractal_holos_gemini.py` (Disk-Backed)

### Current Architecture
```
RAM: frontier, solved, unsolved_children_count, all_seen
DISK: edges (SQLite)
```

### Memory Problems Identified

#### 1. **Frontier Still Stores Full States** (Line 329)
```python
self.frontier: Dict[int, ChessState] = {}
```
**Same problem as original** - states are big.

#### 2. **all_seen Still Unbounded** (Line 339)
```python
self.all_seen: Set[int] = set()
```
**Same problem** - grows forever.

#### 3. **SQLite Queries are SLOW** (Line 591)
```python
parents = self.graph.get_parents(h)
# Does: self.conn.execute("SELECT parent FROM parents WHERE child=?", (child_h,))
```
**Problem**: Every propagation step queries disk. With millions of positions, this is:
- I/O bound
- No batching
- Index lookups are O(log N) per query

#### 4. **Propagation Logic is Incomplete** (Lines 534-693)
The `_propagate` method has a massive commented section explaining it can't work without state turn info. The "corrected" version `_propagate_corrected` tries to pack turn into the count, but:
```python
packed_info = len(child_hashes) | ((1 if is_w else 0) << 20)
```
**Problem**: This limits child count to 2^20 = 1M children, which is fine, but the real issue is the propagation queue:
```python
queue = seeds
while queue:
    child_h = queue.pop(0)  # O(N) pop from front!
```
Should use `collections.deque`.

#### 5. **No Batched DB Writes** (Well, there is, but...)
```python
self.batch_size = 50000
```
Good, but the `commit()` happens too often:
```python
def add_edge(self, parent_h, child_h):
    self.pending_writes.append((child_h, parent_h))
    if len(self.pending_writes) >= self.batch_size:
        self.commit()
```
**Problem**: `get_parents` forces a `commit()` every time:
```python
def get_parents(self, child_h):
    self.commit()  # Ensures consistency
```
This kills batching benefits.

#### 6. **Memory Still Grows Because solved Dict is In-Memory**
```python
self.solved: Dict[int, int] = {}
```
At 10M solved positions: ~400MB. Acceptable, but combined with other structures...

### Memory Bloat Summary for fractal_holos_gemini.py
| Structure | 10M positions | Estimate |
|-----------|--------------|----------|
| frontier | 10M × 350B | 3.5 GB |
| solved | 10M × 40B | 0.4 GB |
| unsolved_children_count | 10M × 40B | 0.4 GB |
| all_seen | 10M × 60B | 0.6 GB |
| **TOTAL RAM** | | **~5 GB** |

Better! But the SQLite approach hits **I/O bottleneck** before memory limit.

---

## Root Causes of Both Scripts

### 1. **Storing States Instead of Just Hashes**
Both scripts keep full `ChessState` objects in frontier. For expansion, you only need the hash + ability to regenerate the state.

**Fix**: Store only hashes; regenerate states on-demand from a "seed" position + move sequence, OR use a more compact encoding.

### 2. **Dual Graph Storage (Parents + Children)**
Both track edges bidirectionally. Retrograde analysis only needs parents for propagation.

**Fix**: Store only parent edges; derive children during expansion.

### 3. **No Streaming/Chunked Processing**
Both try to hold the entire frontier in memory at once.

**Fix**: Process frontier in chunks, spill to disk, reload.

### 4. **Memory Checks are Reactive, Not Proactive**
Both check memory AFTER allocating, causing spikes.

**Fix**: Check BEFORE expanding; estimate expansion cost.

---

## Recommendations for 10-60GB Reliable Operation

### Architecture Changes

#### 1. **Hybrid Memory Model**
```
TIER 1 (RAM - Hot): Current frontier batch (1-5M positions)
TIER 2 (Memory-Mapped): Larger frontier overflow (10-50M positions)
TIER 3 (Disk - SQLite): Solved values, edges
```

#### 2. **Compact State Representation**
Instead of `ChessState` objects (~350 bytes), use:
```python
# 8-piece position fits in 64 bits for pieces + 8 bits metadata
# piece_type (4 bits) + square (6 bits) = 10 bits × 8 pieces = 80 bits
# Or use Zobrist hashing with reversible encoding
```
**Savings**: 350 bytes → 16 bytes = **22x reduction**

#### 3. **Chunked Frontier Processing**
```python
CHUNK_SIZE = 1_000_000
while frontier_chunks_remain:
    chunk = load_frontier_chunk(CHUNK_SIZE)
    expanded = expand_chunk(chunk)
    save_overflow_to_disk(expanded)
    propagate_chunk(expanded)
```

#### 4. **Memory-Mapped Solved Dict**
Use `mmap` or `lmdb` for solved values:
```python
import lmdb
env = lmdb.open('solved.lmdb', map_size=50*1024**3)  # 50GB
```

#### 5. **Proactive Memory Gating**
```python
def can_expand(self, batch_size):
    import psutil
    available_mb = psutil.virtual_memory().available / (1024**2)
    estimated_need = batch_size * 400 / (1024**2)  # 400 bytes per expansion
    return available_mb - estimated_need > SAFETY_MARGIN_MB
```

### Specific Code Fixes

#### For `fractal_holos.py`:

**Line 354**: Remove `children` dict entirely
```python
# DELETE: self.children: Dict[int, List[int]] = {}
```

**Line 355**: Remove `parents` defaultdict
```python
# DELETE: self.parents: Dict[int, List[int]] = defaultdict(list)
```

**Replace with disk-backed parent storage** (like gemini version but batched better)

**Line 640**: Don't store children, propagate immediately if possible
```python
# Instead of: self.children[h] = child_hashes
# Do: Check for immediate wins and propagate
```

#### For `fractal_holos_gemini.py`:

**Line 591**: Batch parent lookups
```python
def get_parents_batch(self, child_hashes):
    placeholders = ','.join('?' * len(child_hashes))
    cur = self.conn.execute(
        f"SELECT child, parent FROM parents WHERE child IN ({placeholders})",
        child_hashes
    )
    return {ch: [] for ch in child_hashes}  # Group results
```

**Line 762**: Use deque
```python
from collections import deque
queue = deque(seeds)
while queue:
    child_h = queue.popleft()  # O(1) instead of O(N)
```

**Line 329**: Don't store full states
```python
# Instead of: self.frontier: Dict[int, ChessState] = {}
# Use: self.frontier: Set[int] = set()
# Regenerate state when needed from hash lookup table
```

---

## Recommended Hybrid Implementation

```python
class FractalHOLOS_v3:
    def __init__(self, max_ram_gb=40, disk_path="./fractal"):
        # TIER 1: Hot RAM
        self.frontier_hot: Set[int] = set()  # Just hashes
        self.state_cache: Dict[int, bytes] = {}  # LRU cache of packed states

        # TIER 2: Memory-mapped
        self.frontier_cold = mmap_set(f"{disk_path}/frontier.mmap")

        # TIER 3: Disk
        self.db = lmdb.open(f"{disk_path}/graph.lmdb", map_size=100*GB)

        # Memory tracking
        self.max_ram_bytes = max_ram_gb * 1024**3
        self.current_ram = 0

    def expand_with_gating(self):
        available = self.get_available_ram()
        batch_size = min(len(self.frontier_hot), available // 500)  # 500 bytes per expand

        if batch_size < 10000:
            self.spill_to_cold()  # Move hot → cold
            available = self.get_available_ram()
            batch_size = min(len(self.frontier_hot), available // 500)

        # Expand batch
        ...
```

---

## Summary

| Issue | fractal_holos.py | fractal_holos_gemini.py | Fix |
|-------|-----------------|------------------------|-----|
| Full state storage | YES (3.5GB/10M) | YES (3.5GB/10M) | Use hash-only + cache |
| Dual edge storage | YES (3.2GB/10M) | NO (disk) | Remove children dict |
| Unbounded all_seen | YES | YES | Chunk and clear |
| Wrong memory estimates | YES | N/A (uses psutil) | Use psutil always |
| Slow disk I/O | N/A | YES | Batch queries |
| Queue O(N) pop | N/A | YES | Use deque |

**Target 10-60GB**:
- With current code: Can handle ~15-20M positions before overflow
- With recommended fixes: Can handle ~100-200M positions in 60GB
- Key bottleneck shifts from RAM to I/O (which is manageable with SSDs)
