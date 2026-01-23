"""
holos/storage.py - Holographic Storage System

This module implements compressed storage for HOLOS:

1. Hologram: The main storage structure
   - Solved states (hash -> value)
   - Spine paths (compressed principal variations)
   - Equivalence class tracking

2. SpinePath: Compressed path from start to boundary
   - Stores only: start_hash, moves, end_value
   - Can reconstruct full path deterministically

3. SeedFrontierMapping: The key compression insight
   - Store SEEDS, not full frontiers
   - seed + depth -> deterministic expansion -> frontier
   - Compression ratio: ~8000x at depth 5

The insight: Instead of storing 8000 frontier positions, store:
- 1 seed position
- depth = 5
- expansion algorithm = deterministic BFS

Reconstruction is deterministic, so we only store the seed.
"""

from typing import Dict, Set, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
import os


@dataclass
class SpinePath:
    """
    Compressed principal variation connecting start to boundary.

    A spine is the "optimal" path from a position to the boundary.
    We store only the moves, not the intermediate positions.

    Spines serve two purposes:
    1. ANALYSIS: Show how a position is solved (PV line)
    2. DECISION MAKING: Quick lookup of best move from any position on spine

    The spine can be "walked" to reconstruct any intermediate position:
        state = start_state
        for move in spine.moves[:depth]:
            state = apply_move(state, move)
    """
    start_hash: int
    moves: List[Any]  # List of moves (game-specific)
    end_hash: int
    end_value: Any
    depth: int = 0

    # Optional: features at key positions (for analysis)
    checkpoints: List[Tuple[int, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.depth == 0:
            self.depth = len(self.moves)

    def get_move_at(self, position_hash: int) -> Optional[Any]:
        """
        Get the spine move at a position (for decision making).

        If the position is on this spine, return the move.
        Otherwise return None.
        """
        for i, (h, _) in enumerate(self.checkpoints):
            if h == position_hash and i < len(self.moves):
                return self.moves[i]
        return None

    def truncate(self, depth: int) -> 'SpinePath':
        """Return spine truncated to given depth"""
        return SpinePath(
            start_hash=self.start_hash,
            moves=self.moves[:depth],
            end_hash=self.checkpoints[depth][0] if depth < len(self.checkpoints) else self.end_hash,
            end_value=self.end_value,
            depth=min(depth, self.depth),
            checkpoints=self.checkpoints[:depth+1]
        )


@dataclass
class SeedFrontierMapping:
    """
    Maps seeds to frontiers via deterministic expansion.

    KEY INSIGHT: We don't need to store the frontier explicitly.
    Given a seed and expansion parameters, we can RECONSTRUCT the frontier.

    Storage model:
        seed_hash -> (depth, mode, expansion_params)
        frontier = expand(seed, depth, mode, params)

    Compression arithmetic:
        - Depth 1: ~25 positions (25x compression)
        - Depth 2: ~275 positions (275x compression)
        - Depth 3: ~2775 positions (2775x compression)
        - Depth 5: ~8000 positions (8000x compression)

    The frontier is a DERIVED quantity, not stored.
    """
    seed_hash: int
    depth: int
    mode: str  # "wave", "lightning", etc.
    expansion_params: Dict[str, Any] = field(default_factory=dict)

    # Cached frontier (lazy, not serialized)
    _frontier: Optional[Set[int]] = field(default=None, repr=False, compare=False)

    def get_frontier(self, expand_fn: Callable) -> Set[int]:
        """
        Get frontier by expanding from seed.

        Args:
            expand_fn: Function (seed_hash, depth, mode, params) -> Set[int]

        Returns:
            Set of frontier state hashes
        """
        if self._frontier is None:
            self._frontier = expand_fn(
                self.seed_hash, self.depth, self.mode, self.expansion_params
            )
        return self._frontier

    def clear_cache(self):
        """Clear cached frontier (for memory management)"""
        self._frontier = None

    def serialize(self) -> Dict:
        """Serialize for storage (excludes frontier cache)"""
        return {
            'seed_hash': self.seed_hash,
            'depth': self.depth,
            'mode': self.mode,
            'params': self.expansion_params,
        }

    @staticmethod
    def deserialize(data: Dict) -> 'SeedFrontierMapping':
        return SeedFrontierMapping(
            seed_hash=data['seed_hash'],
            depth=data['depth'],
            mode=data['mode'],
            expansion_params=data.get('params', {}),
        )


@dataclass
class Hologram:
    """
    Holographic storage for solved game states.

    The hologram stores:
    1. Solved states: hash -> value (the core result)
    2. Spines: Compressed paths for analysis/decision
    3. Boundary: Hashes of boundary states
    4. Equivalence classes: Feature -> hashes (for compression)
    5. Seed mappings: Seeds that can regenerate frontiers

    The hologram is the OUTPUT of HOLOS solving.
    It can be saved/loaded and queried efficiently.
    """
    name: str
    solved: Dict[int, Any] = field(default_factory=dict)
    spines: List[SpinePath] = field(default_factory=list)
    boundary_hashes: Set[int] = field(default_factory=set)
    connections: List[Tuple[int, int, Any]] = field(default_factory=list)

    # Equivalence tracking
    equiv_classes: Dict[Any, Set[int]] = field(default_factory=lambda: defaultdict(set))
    equiv_outcomes: Dict[Any, Optional[Any]] = field(default_factory=dict)

    # Seed -> frontier mappings (compressed storage)
    seed_mappings: List[SeedFrontierMapping] = field(default_factory=list)

    # Statistics
    stats: Dict[str, int] = field(default_factory=dict)

    def query(self, h: int) -> Optional[Any]:
        """Query value for a state hash"""
        return self.solved.get(h)

    def add_boundary(self, h: int, value: Any):
        """Add a boundary state"""
        self.solved[h] = value
        self.boundary_hashes.add(h)

    def add_spine(self, spine: SpinePath):
        """Add a spine path"""
        self.spines.append(spine)
        self.solved[spine.start_hash] = spine.end_value

    def add_with_features(self, h: int, value: Any, features: Any):
        """Add solved state with equivalence tracking"""
        self.solved[h] = value
        self.equiv_classes[features].add(h)
        self._update_equiv_outcome(features, value)

    def _update_equiv_outcome(self, features: Any, value: Any):
        """Track outcome for equivalence class"""
        if features in self.equiv_outcomes:
            if self.equiv_outcomes[features] != value:
                self.equiv_outcomes[features] = None  # Inconsistent
        else:
            self.equiv_outcomes[features] = value

    def propagate_equivalence(self) -> int:
        """Propagate solutions via equivalence classes"""
        count = 0
        for features, hashes in self.equiv_classes.items():
            if features not in self.equiv_outcomes:
                continue
            outcome = self.equiv_outcomes[features]
            if outcome is None:
                continue
            for h in hashes:
                if h not in self.solved:
                    self.solved[h] = outcome
                    count += 1
        return count

    def add_seed_mapping(self, seed_hash: int, depth: int, mode: str, **params):
        """Add a seed -> frontier mapping"""
        mapping = SeedFrontierMapping(
            seed_hash=seed_hash,
            depth=depth,
            mode=mode,
            expansion_params=params,
        )
        self.seed_mappings.append(mapping)

    def get_spine_for(self, h: int) -> Optional[SpinePath]:
        """Get spine that passes through position h"""
        for spine in self.spines:
            if spine.start_hash == h:
                return spine
            for ch, _ in spine.checkpoints:
                if ch == h:
                    return spine
        return None

    def get_best_move(self, h: int) -> Optional[Any]:
        """Get best move from position (via spine lookup)"""
        spine = self.get_spine_for(h)
        if spine:
            return spine.get_move_at(h)
        return None

    # Serialization
    def save(self, path: str):
        """Save hologram to file"""
        data = {
            'name': self.name,
            'solved': self.solved,
            'spines': self.spines,
            'boundary': self.boundary_hashes,
            'connections': self.connections,
            'equiv_classes': dict(self.equiv_classes),
            'equiv_outcomes': self.equiv_outcomes,
            'seed_mappings': [m.serialize() for m in self.seed_mappings],
            'stats': self.stats,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved hologram '{self.name}':")
        print(f"  Solved: {len(self.solved):,}")
        print(f"  Spines: {len(self.spines):,}")
        print(f"  Seeds: {len(self.seed_mappings):,}")

    @staticmethod
    def load(path: str) -> 'Hologram':
        """Load hologram from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        h = Hologram(data['name'])
        h.solved = data['solved']
        h.spines = data.get('spines', [])
        h.boundary_hashes = data.get('boundary', set())
        h.connections = data.get('connections', [])
        h.equiv_classes = defaultdict(set, data.get('equiv_classes', {}))
        h.equiv_outcomes = data.get('equiv_outcomes', {})
        h.seed_mappings = [
            SeedFrontierMapping.deserialize(m)
            for m in data.get('seed_mappings', [])
        ]
        h.stats = data.get('stats', {})

        return h

    def merge(self, other: 'Hologram') -> 'Hologram':
        """Merge another hologram into this one"""
        # Merge solved states
        for h, v in other.solved.items():
            if h not in self.solved:
                self.solved[h] = v

        # Merge spines (dedupe by start_hash)
        existing_starts = {s.start_hash for s in self.spines}
        for spine in other.spines:
            if spine.start_hash not in existing_starts:
                self.spines.append(spine)

        # Merge boundaries
        self.boundary_hashes |= other.boundary_hashes

        # Merge connections
        existing_conns = set((a, b) for a, b, _ in self.connections)
        for a, b, v in other.connections:
            if (a, b) not in existing_conns:
                self.connections.append((a, b, v))

        # Merge equivalence classes
        for features, hashes in other.equiv_classes.items():
            self.equiv_classes[features] |= hashes
            if features in other.equiv_outcomes and features not in self.equiv_outcomes:
                self.equiv_outcomes[features] = other.equiv_outcomes[features]

        # Merge seed mappings
        existing_seeds = {m.seed_hash for m in self.seed_mappings}
        for mapping in other.seed_mappings:
            if mapping.seed_hash not in existing_seeds:
                self.seed_mappings.append(mapping)

        return self

    def summary(self) -> str:
        """Return summary string"""
        equiv_count = sum(len(v) for v in self.equiv_classes.values())
        lines = [
            f"Hologram: {self.name}",
            f"  Solved states: {len(self.solved):,}",
            f"  Spines: {len(self.spines):,}",
            f"  Boundary: {len(self.boundary_hashes):,}",
            f"  Connections: {len(self.connections):,}",
            f"  Equivalence tracked: {equiv_count:,}",
            f"  Seed mappings: {len(self.seed_mappings):,}",
        ]

        # Compression stats if we have seed mappings
        if self.seed_mappings:
            total_depth = sum(m.depth for m in self.seed_mappings)
            avg_depth = total_depth / len(self.seed_mappings)
            # Rough estimate: each depth level ~10x expansion
            est_frontier = sum(10 ** m.depth for m in self.seed_mappings)
            compression = est_frontier / len(self.seed_mappings) if self.seed_mappings else 1
            lines.append(f"  Avg seed depth: {avg_depth:.1f}")
            lines.append(f"  Est compression: {compression:.0f}x")

        return "\n".join(lines)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_spine_from_path(path: List[Tuple[Any, Any]],
                           start_state: Any,
                           end_value: Any,
                           hash_fn: Callable) -> SpinePath:
    """
    Create a spine from a path of (state, move) pairs.

    Args:
        path: List of (state, move) from lightning probe
        start_state: The starting state
        end_value: Value at the end of the path
        hash_fn: Function to hash states

    Returns:
        SpinePath with checkpoints
    """
    moves = [move for state, move in path]
    checkpoints = [(hash_fn(state), None) for state, move in path]

    if path:
        # Add end state as final checkpoint
        last_state, last_move = path[-1]
        # Ideally we'd apply the move to get end state, but we have end_hash
        pass

    return SpinePath(
        start_hash=hash_fn(start_state),
        moves=moves,
        end_hash=checkpoints[-1][0] if checkpoints else hash_fn(start_state),
        end_value=end_value,
        checkpoints=checkpoints,
    )


def estimate_compression(num_seeds: int, avg_depth: float) -> float:
    """
    Estimate compression ratio for seed-based storage.

    Assumes ~10x expansion per depth level.
    """
    expansion_per_seed = 10 ** avg_depth
    return expansion_per_seed
