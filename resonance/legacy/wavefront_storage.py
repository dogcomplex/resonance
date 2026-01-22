"""
Wavefront Storage for Chess Endgames
====================================

Instead of storing petabytes of position→value mappings,
we store the WAVEFRONT: the boundary conditions and key paths
that allow fast reconstruction of any position's value.

Physics inspiration:
- Holograms: Store interference pattern at boundary, reconstruct 3D from it
- Slime molds: Reinforce successful paths, let failures fade
- Crystal seeds: Store nucleation points, crystal structure is implicit

What we store:
1. Syzygy 7-piece tables (boundary condition) - ALREADY HAVE
2. Spine paths (8-piece → 7-piece principal variations)
3. Pattern signatures (equivalence classes that determine outcome)
4. Critical positions (branch points where value changes)

What we DON'T store:
- Every solved position (too big)
- Positions reconstructable from spine + patterns
- Redundant symmetric positions

Reconstruction:
- Given any 8-piece position, we either:
  a) Match a pattern → instant answer
  b) Follow spine toward 7-piece → fast lookup
  c) Mini-search to hit known positions → reconstruct
"""

import pickle
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
import hashlib
import struct

WAVEFRONT_DIR = "./wavefront_data"


@dataclass
class PatternSignature:
    """
    Compressed signature of position features that determine outcome.
    
    Instead of storing individual positions, we store patterns like:
    - "White has Q+R vs lone K with K on edge" → White wins
    - "Opposite color bishops with equal pawns" → Draw
    
    For 8-piece: capture patterns, piece activity, king safety metrics
    """
    material_balance: int  # Centipawn equivalent
    piece_activity: Tuple[int, ...]  # Mobility scores
    king_safety: Tuple[int, int]  # White, Black
    pawn_structure: int  # Hash of pawn positions
    
    def __hash__(self):
        return hash((self.material_balance, self.piece_activity, 
                     self.king_safety, self.pawn_structure))
    
    def to_bytes(self) -> bytes:
        """Compact binary representation"""
        return struct.pack(
            'i8h2hI',
            self.material_balance,
            *self.piece_activity[:8],
            *self.king_safety,
            self.pawn_structure & 0xFFFFFFFF
        )
    
    @staticmethod
    def from_bytes(data: bytes) -> 'PatternSignature':
        vals = struct.unpack('i8h2hI', data)
        return PatternSignature(
            material_balance=vals[0],
            piece_activity=tuple(vals[1:9]),
            king_safety=(vals[9], vals[10]),
            pawn_structure=vals[11]
        )


@dataclass
class SpinePath:
    """
    A principal variation from 8-piece to 7-piece boundary.
    
    This is like the "trunk" of a Lichtenberg figure or
    the main channel of a river delta. Other positions
    can be evaluated relative to this path.
    """
    start_hash: int
    moves: List[Tuple[int, int]]  # (from_sq, to_sq) pairs
    end_hash: int  # 7-piece position hash
    end_value: int  # Value from Syzygy
    
    def to_bytes(self) -> bytes:
        move_bytes = b''.join(struct.pack('BB', f, t) for f, t in self.moves)
        return struct.pack(
            'qH',
            self.start_hash,
            len(self.moves)
        ) + move_bytes + struct.pack('qb', self.end_hash, self.end_value)
    
    @staticmethod
    def from_bytes(data: bytes) -> 'SpinePath':
        start_hash, num_moves = struct.unpack('qH', data[:10])
        moves = []
        offset = 10
        for _ in range(num_moves):
            f, t = struct.unpack('BB', data[offset:offset+2])
            moves.append((f, t))
            offset += 2
        end_hash, end_value = struct.unpack('qb', data[offset:offset+9])
        return SpinePath(start_hash, moves, end_hash, end_value)


@dataclass  
class CriticalPosition:
    """
    A position where the evaluation changes significantly.
    
    These are like "phase transition" points - small changes
    in the position lead to different outcomes. Worth storing
    explicitly because they're decision-critical.
    """
    position_hash: int
    position_fen: str  # For debugging/display
    best_move: Tuple[int, int]
    value: int
    alternatives: List[Tuple[Tuple[int, int], int]]  # Other moves and their values
    
    def to_dict(self) -> dict:
        return {
            'hash': self.position_hash,
            'fen': self.position_fen,
            'best': self.best_move,
            'value': self.value,
            'alts': self.alternatives
        }


class WavefrontStorage:
    """
    Holographic storage for chess endgame solutions.
    
    Core principle: Store the WAVEFRONT (boundary + spine),
    not the full solution space.
    
    Components:
    1. Pattern database: signature → outcome mapping
    2. Spine paths: principal variations to 7-piece boundary
    3. Critical positions: branch points worth explicit storage
    4. Bloom filter: fast negative lookup ("definitely not winning")
    """
    
    def __init__(self, storage_dir=WAVEFRONT_DIR):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # In-memory caches (loaded from disk)
        self.pattern_outcomes: Dict[PatternSignature, int] = {}
        self.spine_paths: List[SpinePath] = []
        self.critical_positions: Dict[int, CriticalPosition] = {}
        
        # Bloom filter for fast "not in set" checks
        self.bloom_size = 10_000_000  # 10M bits = 1.25MB
        self.bloom_filter = bytearray(self.bloom_size // 8)
        self.bloom_hashes = 7  # Number of hash functions
        
        # Statistics
        self.stats = {
            'patterns_stored': 0,
            'spines_stored': 0,
            'critical_stored': 0,
            'total_bytes': 0,
        }
    
    # ========== BLOOM FILTER ==========
    
    def _bloom_indices(self, key: int) -> List[int]:
        """Get bloom filter bit indices for a key"""
        indices = []
        h = key
        for i in range(self.bloom_hashes):
            h = hash((h, i))  # Chain hashes
            indices.append(h % self.bloom_size)
        return indices
    
    def bloom_add(self, key: int):
        """Add key to bloom filter"""
        for idx in self._bloom_indices(key):
            byte_idx, bit_idx = divmod(idx, 8)
            self.bloom_filter[byte_idx] |= (1 << bit_idx)
    
    def bloom_maybe_contains(self, key: int) -> bool:
        """Check if key MIGHT be in set (false positives possible)"""
        for idx in self._bloom_indices(key):
            byte_idx, bit_idx = divmod(idx, 8)
            if not (self.bloom_filter[byte_idx] & (1 << bit_idx)):
                return False
        return True
    
    # ========== PATTERN DATABASE ==========
    
    def add_pattern(self, signature: PatternSignature, outcome: int):
        """Record that positions matching this pattern have this outcome"""
        self.pattern_outcomes[signature] = outcome
        self.stats['patterns_stored'] = len(self.pattern_outcomes)
    
    def lookup_pattern(self, signature: PatternSignature) -> Optional[int]:
        """Check if pattern determines outcome"""
        return self.pattern_outcomes.get(signature)
    
    # ========== SPINE PATHS ==========
    
    def add_spine(self, spine: SpinePath):
        """Add a principal variation path"""
        self.spine_paths.append(spine)
        self.bloom_add(spine.start_hash)
        self.stats['spines_stored'] = len(self.spine_paths)
    
    def find_spine_from(self, start_hash: int) -> Optional[SpinePath]:
        """Find spine path starting from given position"""
        if not self.bloom_maybe_contains(start_hash):
            return None  # Definitely not in any spine
        
        for spine in self.spine_paths:
            if spine.start_hash == start_hash:
                return spine
        return None
    
    # ========== CRITICAL POSITIONS ==========
    
    def add_critical(self, critical: CriticalPosition):
        """Store a critical branch point"""
        self.critical_positions[critical.position_hash] = critical
        self.bloom_add(critical.position_hash)
        self.stats['critical_stored'] = len(self.critical_positions)
    
    def lookup_critical(self, pos_hash: int) -> Optional[CriticalPosition]:
        """Check for stored critical position"""
        if not self.bloom_maybe_contains(pos_hash):
            return None
        return self.critical_positions.get(pos_hash)
    
    # ========== RECONSTRUCTION ==========
    
    def reconstruct_value(self, pos_hash: int, signature: PatternSignature, 
                          syzygy_probe_fn=None) -> Optional[int]:
        """
        Reconstruct position value from wavefront data.
        
        Strategy:
        1. Check pattern database (O(1))
        2. Check critical positions (O(1))
        3. Check spine paths (O(1) bloom + O(n) scan)
        4. If spine found, follow to 7-piece and probe Syzygy
        
        Returns None if position requires full search.
        """
        # 1. Pattern lookup
        pattern_result = self.lookup_pattern(signature)
        if pattern_result is not None:
            return pattern_result
        
        # 2. Critical position lookup
        critical = self.lookup_critical(pos_hash)
        if critical:
            return critical.value
        
        # 3. Spine lookup
        spine = self.find_spine_from(pos_hash)
        if spine and syzygy_probe_fn:
            # We have a path to 7-piece - value is end_value
            return spine.end_value
        
        # Not in wavefront - requires search
        return None
    
    # ========== PERSISTENCE ==========
    
    def save(self):
        """Save wavefront data to disk"""
        print(f"Saving wavefront data to {self.storage_dir}...")
        
        # Save patterns
        pattern_path = os.path.join(self.storage_dir, 'patterns.pkl')
        with open(pattern_path, 'wb') as f:
            pickle.dump(self.pattern_outcomes, f)
        
        # Save spines (binary for compactness)
        spine_path = os.path.join(self.storage_dir, 'spines.bin')
        with open(spine_path, 'wb') as f:
            f.write(struct.pack('I', len(self.spine_paths)))
            for spine in self.spine_paths:
                data = spine.to_bytes()
                f.write(struct.pack('H', len(data)))
                f.write(data)
        
        # Save critical positions
        critical_path = os.path.join(self.storage_dir, 'critical.json')
        with open(critical_path, 'w') as f:
            json.dump([c.to_dict() for c in self.critical_positions.values()], f)
        
        # Save bloom filter
        bloom_path = os.path.join(self.storage_dir, 'bloom.bin')
        with open(bloom_path, 'wb') as f:
            f.write(self.bloom_filter)
        
        # Calculate total size
        total = sum(
            os.path.getsize(os.path.join(self.storage_dir, f))
            for f in os.listdir(self.storage_dir)
            if os.path.isfile(os.path.join(self.storage_dir, f))
        )
        self.stats['total_bytes'] = total
        
        print(f"  Patterns: {self.stats['patterns_stored']:,}")
        print(f"  Spines: {self.stats['spines_stored']:,}")
        print(f"  Critical: {self.stats['critical_stored']:,}")
        print(f"  Total size: {total / 1024 / 1024:.1f} MB")
    
    def load(self) -> bool:
        """Load wavefront data from disk"""
        try:
            # Load patterns
            pattern_path = os.path.join(self.storage_dir, 'patterns.pkl')
            if os.path.exists(pattern_path):
                with open(pattern_path, 'rb') as f:
                    self.pattern_outcomes = pickle.load(f)
            
            # Load spines
            spine_path = os.path.join(self.storage_dir, 'spines.bin')
            if os.path.exists(spine_path):
                with open(spine_path, 'rb') as f:
                    count = struct.unpack('I', f.read(4))[0]
                    for _ in range(count):
                        size = struct.unpack('H', f.read(2))[0]
                        data = f.read(size)
                        self.spine_paths.append(SpinePath.from_bytes(data))
            
            # Load critical
            critical_path = os.path.join(self.storage_dir, 'critical.json')
            if os.path.exists(critical_path):
                with open(critical_path, 'r') as f:
                    data = json.load(f)
                    for d in data:
                        c = CriticalPosition(
                            d['hash'], d['fen'], tuple(d['best']),
                            d['value'], [tuple(a) for a in d['alts']]
                        )
                        self.critical_positions[c.position_hash] = c
            
            # Load bloom
            bloom_path = os.path.join(self.storage_dir, 'bloom.bin')
            if os.path.exists(bloom_path):
                with open(bloom_path, 'rb') as f:
                    self.bloom_filter = bytearray(f.read())
            
            self.stats['patterns_stored'] = len(self.pattern_outcomes)
            self.stats['spines_stored'] = len(self.spine_paths)
            self.stats['critical_stored'] = len(self.critical_positions)
            
            print(f"Loaded wavefront data:")
            print(f"  Patterns: {self.stats['patterns_stored']:,}")
            print(f"  Spines: {self.stats['spines_stored']:,}")
            print(f"  Critical: {self.stats['critical_stored']:,}")
            return True
            
        except Exception as e:
            print(f"Load error: {e}")
            return False


# ============================================================
# INTEGRATION WITH LIGHTNING SOLVER
# ============================================================

class WavefrontBuilder:
    """
    Builds wavefront data from a solving run.
    
    Called during solving to extract:
    - Pattern signatures for positions with clear outcomes
    - Spine paths (principal variations)
    - Critical positions (branch points)
    """
    
    def __init__(self, storage: WavefrontStorage):
        self.storage = storage
        self.pending_spines: Dict[int, List[Tuple[int, int]]] = {}  # hash → moves so far
    
    def observe_solution(self, pos_hash: int, value: int, 
                         signature: PatternSignature,
                         from_spine: bool = False,
                         parent_hash: int = None,
                         move: Tuple[int, int] = None):
        """
        Called when a position is solved.
        
        Decides whether to add to patterns, spine, or critical.
        """
        # Add pattern if this is a common signature
        existing = self.storage.lookup_pattern(signature)
        if existing is None:
            # First time seeing this pattern with this value
            self.storage.add_pattern(signature, value)
        elif existing != value:
            # Pattern doesn't determine value - remove it
            # (This pattern leads to different outcomes)
            pass  # Could mark as "indeterminate"
        
        # Track spine paths
        if parent_hash and move:
            if parent_hash in self.pending_spines:
                self.pending_spines[pos_hash] = self.pending_spines[parent_hash] + [move]
            else:
                self.pending_spines[pos_hash] = [move]
    
    def finalize_spine(self, end_hash: int, end_value: int, fen: str = ""):
        """Called when a spine reaches 7-piece boundary"""
        if end_hash in self.pending_spines:
            moves = self.pending_spines[end_hash]
            # Find the start
            start_hash = end_hash
            for _ in range(len(moves)):
                # Walk back... (simplified - real impl would track this)
                pass
            
            # Create spine
            # spine = SpinePath(start_hash, moves, end_hash, end_value)
            # self.storage.add_spine(spine)
    
    def mark_critical(self, pos_hash: int, fen: str, best_move: Tuple[int, int],
                      value: int, alternatives: List[Tuple[Tuple[int, int], int]]):
        """Mark a position as critical (evaluation-sensitive)"""
        critical = CriticalPosition(pos_hash, fen, best_move, value, alternatives)
        self.storage.add_critical(critical)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    storage = WavefrontStorage()
    
    # Simulate adding some data
    print("Creating sample wavefront data...")
    
    # Add patterns
    for i in range(1000):
        sig = PatternSignature(
            material_balance=i % 10 - 5,
            piece_activity=(1,2,3,4,5,6,7,8),
            king_safety=(i % 3, i % 4),
            pawn_structure=hash(f"pawn_{i}")
        )
        storage.add_pattern(sig, i % 3 - 1)  # -1, 0, or 1
    
    # Add spines
    for i in range(100):
        spine = SpinePath(
            start_hash=hash(f"start_{i}"),
            moves=[(j, j+8) for j in range(5)],  # Dummy moves
            end_hash=hash(f"end_{i}"),
            end_value=i % 3 - 1
        )
        storage.add_spine(spine)
    
    # Save
    storage.save()
    
    # Reload in new instance
    storage2 = WavefrontStorage()
    storage2.load()
    
    print(f"\nReconstruction test:")
    sig = PatternSignature(0, (1,2,3,4,5,6,7,8), (0, 0), hash("pawn_0"))
    result = storage2.reconstruct_value(12345, sig)
    print(f"  Pattern lookup: {result}")
