# HOLOS Layer Architecture: Compute vs Storage Tradeoffs

**The Core Insight**: Each layer trades compute for storage. Compression efficiency must be part of the value function at every layer.

---

## The Four Layers

```
Layer 3: BALANCE (Compute/Storage Policy)
    "Given hardware constraints, what's the optimal tradeoff?"
    State: ComputeStoragePolicy (memory limit, CPU budget, compression target)
    Value: Pareto efficiency (coverage per byte per FLOP)
    Output: Layer 2 configuration
    │
    ▼
Layer 2: STRATEGY (Multi-Seed Coordination)
    "Which seeds together, with what compression?"
    State: SeedSet + CompressionScheme
    Value: Combined coverage + compressed size + reconstruction cost
    Output: Seed configurations for Layer 1
    │
    ▼
Layer 1: TACTICS (Single Seed Optimization)
    "How to expand this seed efficiently?"
    State: TacticalSeed (position, depth, mode, direction)
    Value: Coverage + compressed_size (NOT raw storage!)
    Output: Seed parameters for Layer 0
    │
    ▼
Layer 0: EXECUTION (Position Search)
    "What are the game-theoretic values?"
    State: GamePosition
    Value: Win/Loss/Draw (or problem-specific value)
    Output: Solved positions → compressed to seeds
```

---

## Key Innovation: Compression-Aware Efficiency

### The Old Way (Wrong)
```python
efficiency = coverage / cost
# Problem: A seed covering 100 positions but requiring 1000 bytes
# is WORSE than storing the 100 positions at 1 byte each!
```

### The New Way (Correct)
```python
# Storage if we just stored positions directly
direct_storage = coverage * BYTES_PER_POSITION

# Storage for this seed + reconstruction metadata
seed_storage = seed_size + frontier_overhead + compressed_derivations

# Only count as "efficient" if seed storage < direct storage
net_savings = direct_storage - seed_storage

# Efficiency must account for both compute AND storage
efficiency = net_savings / (compute_cost + reconstruction_cost)
```

### Implication: Small Seeds Are Often Useless

If a seed's scope is too small:
- `seed_storage > direct_storage`
- The seed has **negative** efficiency
- Better to just store the positions directly
- OR: use the seed only if it's part of a larger derivation chain

---

## Layer 1 Refinement: Seed Efficiency with Compression

```python
@dataclass
class CompressionAwareSeedValue:
    """Value of a seed accounting for compression"""

    # Raw metrics
    forward_coverage: int    # Positions reachable forward
    backward_coverage: int   # Positions reachable backward
    compute_cost: int        # FLOPs or time to expand

    # Compression metrics
    seed_storage_bytes: int  # Bytes to store this seed
    frontier_storage_bytes: int  # Bytes for initial frontier (if not derived)
    derivation_bytes: int    # Bytes if derived from parent seed

    @property
    def direct_storage(self) -> int:
        """Storage if we just stored covered positions"""
        total_coverage = self.forward_coverage + self.backward_coverage
        return total_coverage * BYTES_PER_POSITION

    @property
    def seed_storage(self) -> int:
        """Storage for seed-based representation"""
        return self.seed_storage_bytes + self.frontier_storage_bytes

    @property
    def net_savings(self) -> int:
        """Bytes saved by using seed vs direct storage"""
        return self.direct_storage - self.seed_storage

    @property
    def is_worth_storing(self) -> bool:
        """Is this seed better than direct storage?"""
        return self.net_savings > 0

    @property
    def efficiency(self) -> float:
        """Compression-aware efficiency"""
        if not self.is_worth_storing:
            return -1.0  # Negative = worse than direct storage
        return self.net_savings / max(1, self.compute_cost)
```

---

## Layer 2 Refinement: Multi-Seed Compression

The key insight from our experiments:
- **Meta-seeds** (roots + derivation moves) compress better than raw seeds
- Derivation moves (0-6 in Connect4) are highly compressible
- The STRUCTURE helps gzip find patterns

```python
@dataclass
class StrategyWithCompression:
    """Layer 2 state: multi-seed configuration + compression scheme"""

    # Which seeds to use
    seed_configs: List[TacticalSeed]

    # How to compress them
    compression_scheme: str  # "direct", "meta-seeds", "delta-encoded"

    # Derivation structure (for meta-seed scheme)
    root_seeds: List[int]        # Hashes of root seeds
    derivation_graph: Dict[int, List[Tuple[int, Any]]]  # parent -> [(child, move)]

    def compressed_size(self) -> int:
        """Estimate compressed storage size"""
        if self.compression_scheme == "direct":
            return sum(s.storage_bytes for s in self.seed_configs)
        elif self.compression_scheme == "meta-seeds":
            # Roots + moves (gzip-friendly)
            root_bytes = len(self.root_seeds) * BYTES_PER_SEED
            move_bytes = sum(len(children) for children in self.derivation_graph.values())
            # Gzip ratio ~6x on this structure
            return (root_bytes + move_bytes) // 6
        else:
            return sum(s.storage_bytes for s in self.seed_configs)
```

---

## Layer 3: Compute/Storage Balance

This is the meta-decision: given hardware constraints, what balance is optimal?

```python
@dataclass
class ComputeStoragePolicy:
    """Layer 3 state: the compute/storage tradeoff policy"""

    # Hardware constraints
    max_storage_bytes: int      # e.g., 100 GB
    max_compute_flops: int      # e.g., 10^15 FLOPs
    target_coverage: float      # e.g., 0.99 (99% of positions)

    # Policy parameters
    storage_weight: float       # How much to penalize storage
    compute_weight: float       # How much to penalize compute

    # Extremes:
    # storage_weight=0, compute_weight=1 → minimal storage (one bidirectional seed)
    # storage_weight=1, compute_weight=0 → minimal compute (store everything)

    def score_strategy(self, strategy: StrategyWithCompression) -> float:
        """Score a strategy under this policy"""
        coverage = strategy.total_coverage()
        storage = strategy.compressed_size()
        compute = strategy.reconstruction_cost()

        if coverage < self.target_coverage:
            return -float('inf')  # Must meet coverage target

        # Pareto score: lower is better
        return (self.storage_weight * storage / self.max_storage_bytes +
                self.compute_weight * compute / self.max_compute_flops)
```

### The Spectrum

```
← More Compute                                More Storage →

Single bidirectional     Meta-seeds      Full seeds      All positions
seed from start          (roots+moves)   (every seed)    (no compression)

Compression: ∞           Compression: 6x  Compression: 1x  Compression: 0x
Compute: Maximum         Compute: Medium  Compute: Low     Compute: Zero
```

---

## Fractal/Recursive Compression

Your question: Can we compress at each level AND still get benefits from the next level?

**Answer: YES, if we use the right structure.**

```
Positions  →  Seeds  →  Meta-seeds  →  Hyper-seeds
   ↓            ↓           ↓              ↓
 gzip         gzip        gzip          gzip
   ↓            ↓           ↓              ↓
Final storage at each level
```

The key insight from our experiments:
1. **Game rules** provide massive compression (positions → seeds)
2. **Derivation structure** provides additional compression (seeds → meta-seeds)
3. **Standard compression** (gzip) works at EACH level
4. The levels are **multiplicative** (13x × 6x × 8x = 600x+ potential)

### The Catch

At some point, the overhead of the meta-structure exceeds its benefit:
- Hyper-seeds might not help if meta-seeds are already highly compressed
- The "roots" at each level are incompressible (must be stored directly)
- Diminishing returns as we go up the hierarchy

**Rule of thumb**: Stop adding layers when `compressed_size(layer_N+1) >= compressed_size(layer_N)`

---

## File Consolidation Plan

### Current State
```
games/strategy.py      - Goal/budget allocation (incomplete)
seed_meta_game.py      - Meta-seed compression (experimental)
games/seeds.py         - Single seed tactics (Layer 1)
connect4_full_solve.py - Full solver with seeds (monolithic)
```

### Proposed Reorganization
```
holos/
├── layers/
│   ├── __init__.py
│   ├── layer0_execution.py    # GameInterface implementations
│   ├── layer1_tactics.py      # TacticalSeed, compression-aware values
│   ├── layer2_strategy.py     # Multi-seed + compression schemes
│   └── layer3_policy.py       # Compute/storage balance
│
├── compression/
│   ├── __init__.py
│   ├── meta_seeds.py          # Derivation graph compression
│   ├── delta_encoding.py      # Sequential seed compression
│   └── standard.py            # gzip/lz4 wrappers
│
├── games/                     # Layer 0 implementations
│   ├── chess.py
│   ├── connect4.py
│   └── sudoku.py
│
└── solvers/
    ├── full_solve.py          # Complete game solver
    └── incremental.py         # Incremental/online solving
```

---

## Summary

1. **Compression is part of efficiency** - Seeds that don't compress better than direct storage are useless
2. **Layer 2 handles multi-seed compression** - Meta-seed structure helps gzip
3. **Layer 3 decides the tradeoff** - Given constraints, what balance?
4. **Recursive compression works** - Each layer can compress, benefits multiply
5. **Standard algorithms for encoding** - Don't reinvent gzip, use game structure to create compressible data
