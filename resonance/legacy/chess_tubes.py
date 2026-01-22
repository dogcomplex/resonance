"""
chess_tubes.py - Slime Mold Tube Storage

Instead of storing ALL 8-piece solutions (petabytes), we store only:
1. "Tubes" - paths from 8-piece positions to 7-piece boundaries
2. "Junctions" - key positions where tubes branch
3. "Values" - only at tube endpoints (7-piece) and junctions

Query strategy:
1. Is position ON a tube? → Return stored value
2. Is position NEAR a tube? → Trace to tube, return value
3. Otherwise → Local search to nearest tube

This mimics slime mold networks:
- Tubes form where "food" (solutions) flows frequently
- Unused paths fade away
- Key junctions are reinforced

Storage: ~1000x smaller than full tablebase
Query: O(tube_distance) instead of O(1), but still fast
"""

import os
import pickle
import struct
import mmap
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import hashlib

# Try to import core, fall back to inline if not available
try:
    from chess_core import ChessState, Piece, generate_moves, apply_move, is_terminal
except ImportError:
    print("Warning: chess_core not found, some features disabled")


@dataclass
class Tube:
    """
    A path from an 8-piece position to a 7-piece boundary.
    
    Stores:
    - start_hash: 8-piece position hash
    - moves: sequence of moves to reach 7-piece
    - end_value: result from Syzygy (+1, 0, -1)
    - frequency: how often this tube was used (for pruning)
    """
    start_hash: int
    moves: Tuple[Tuple[int, int, Optional[int]], ...]  # (from, to, captured)
    end_value: int
    frequency: int = 1
    
    def __hash__(self):
        return hash((self.start_hash, self.moves))
    
    def length(self):
        return len(self.moves)
    
    def to_bytes(self):
        """Compact serialization"""
        data = bytearray()
        # Start hash: 8 bytes
        data.extend(struct.pack('<q', self.start_hash))
        # End value: 1 byte (-1, 0, 1 → 0, 1, 2)
        data.append(self.end_value + 1)
        # Frequency: 4 bytes
        data.extend(struct.pack('<I', self.frequency))
        # Moves: 1 byte count + 3 bytes per move
        data.append(len(self.moves))
        for from_sq, to_sq, captured in self.moves:
            data.append(from_sq)
            data.append(to_sq)
            data.append(captured if captured else 255)  # 255 = no capture
        return bytes(data)
    
    @staticmethod
    def from_bytes(data, offset=0):
        start_hash = struct.unpack_from('<q', data, offset)[0]
        offset += 8
        end_value = data[offset] - 1
        offset += 1
        frequency = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        n_moves = data[offset]
        offset += 1
        moves = []
        for _ in range(n_moves):
            from_sq = data[offset]
            to_sq = data[offset + 1]
            captured = data[offset + 2]
            captured = None if captured == 255 else captured
            moves.append((from_sq, to_sq, captured))
            offset += 3
        return Tube(start_hash, tuple(moves), end_value, frequency), offset


class TubeNetwork:
    """
    Network of tubes connecting 8-piece positions to 7-piece solutions.
    
    Structure:
    - tubes: Dict[start_hash, List[Tube]] - all tubes from each position
    - junctions: Set[hash] - positions where multiple tubes meet
    - endpoint_values: Dict[hash, int] - cached values at tube endpoints
    - reverse_index: Dict[hash, List[Tube]] - tubes passing through each position
    """
    
    def __init__(self, save_path="./tube_network"):
        self.save_path = save_path
        self.tubes: Dict[int, List[Tube]] = defaultdict(list)
        self.junctions: Set[int] = set()
        self.endpoint_values: Dict[int, int] = {}
        self.reverse_index: Dict[int, List[Tube]] = defaultdict(list)
        
        self.stats = {
            'total_tubes': 0,
            'total_length': 0,
            'junctions': 0,
            'endpoints': 0,
        }
    
    def add_tube(self, tube: Tube):
        """Add a tube to the network"""
        self.tubes[tube.start_hash].append(tube)
        self.stats['total_tubes'] += 1
        self.stats['total_length'] += tube.length()
        
        # Add to reverse index (for positions along the tube)
        # This requires tracing the tube, which needs the actual state
        # For now, just index start and end
        
    def add_solution_path(self, states: List[ChessState], moves: List, value: int):
        """
        Add a solution path from 8-piece to 7-piece.
        
        states: list of states from start to end
        moves: list of moves taken
        value: final value from Syzygy
        """
        if len(states) < 2:
            return
        
        start_hash = hash(states[0])
        tube = Tube(start_hash, tuple(moves), value)
        self.add_tube(tube)
        
        # Index all positions along the path
        for i, state in enumerate(states):
            h = hash(state)
            self.reverse_index[h].append(tube)
            
            # Mark junctions (positions with multiple tubes)
            if len(self.reverse_index[h]) > 1:
                self.junctions.add(h)
        
        # Cache endpoint value
        self.endpoint_values[hash(states[-1])] = value
    
    def query(self, state: ChessState) -> Optional[int]:
        """
        Query the network for a position's value.
        
        Returns value if position is on or near a tube, None otherwise.
        """
        h = hash(state)
        
        # Direct hit - position is on a tube
        if h in self.reverse_index:
            # Use the tube with highest frequency
            tubes = self.reverse_index[h]
            best = max(tubes, key=lambda t: t.frequency)
            return best.end_value
        
        # Position has tubes starting from it
        if h in self.tubes:
            # Evaluate based on available tubes
            tubes = self.tubes[h]
            turn = state.turn
            
            values = [t.end_value for t in tubes]
            if turn == 'w':
                return max(values)  # White picks best
            else:
                return min(values)  # Black picks best
        
        return None
    
    def trace_to_tube(self, state: ChessState, max_depth=5) -> Optional[int]:
        """
        BFS to find nearest tube from a position.
        Returns value if found within max_depth, None otherwise.
        """
        from collections import deque
        
        visited = {hash(state)}
        queue = deque([(state, 0)])
        
        while queue:
            current, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            # Check if on tube
            value = self.query(current)
            if value is not None:
                return value
            
            # Expand
            for move in generate_moves(current):
                child = apply_move(current, move)
                ch = hash(child)
                if ch not in visited:
                    visited.add(ch)
                    queue.append((child, depth + 1))
        
        return None
    
    def save(self):
        """Save network to disk"""
        os.makedirs(self.save_path, exist_ok=True)
        
        # Save tubes
        with open(f"{self.save_path}/tubes.bin", 'wb') as f:
            # Header: tube count
            f.write(struct.pack('<I', len(self.tubes)))
            
            for start_hash, tube_list in self.tubes.items():
                f.write(struct.pack('<I', len(tube_list)))
                for tube in tube_list:
                    f.write(tube.to_bytes())
        
        # Save junctions
        with open(f"{self.save_path}/junctions.bin", 'wb') as f:
            f.write(struct.pack('<I', len(self.junctions)))
            for h in self.junctions:
                f.write(struct.pack('<q', h))
        
        # Save endpoint values
        with open(f"{self.save_path}/endpoints.bin", 'wb') as f:
            f.write(struct.pack('<I', len(self.endpoint_values)))
            for h, v in self.endpoint_values.items():
                f.write(struct.pack('<q', h))
                f.write(struct.pack('<b', v))
        
        # Save stats
        with open(f"{self.save_path}/stats.pkl", 'wb') as f:
            pickle.dump(self.stats, f)
        
        print(f"Saved tube network: {self.stats['total_tubes']} tubes, "
              f"{len(self.junctions)} junctions")
    
    def load(self):
        """Load network from disk"""
        if not os.path.exists(f"{self.save_path}/tubes.bin"):
            return False
        
        try:
            # Load tubes
            with open(f"{self.save_path}/tubes.bin", 'rb') as f:
                data = f.read()
            
            offset = 0
            n_starts = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            
            for _ in range(n_starts):
                n_tubes = struct.unpack_from('<I', data, offset)[0]
                offset += 4
                for _ in range(n_tubes):
                    tube, offset = Tube.from_bytes(data, offset)
                    self.tubes[tube.start_hash].append(tube)
            
            # Load junctions
            with open(f"{self.save_path}/junctions.bin", 'rb') as f:
                data = f.read()
            n = struct.unpack_from('<I', data, 0)[0]
            for i in range(n):
                h = struct.unpack_from('<q', data, 4 + i*8)[0]
                self.junctions.add(h)
            
            # Load endpoints
            with open(f"{self.save_path}/endpoints.bin", 'rb') as f:
                data = f.read()
            n = struct.unpack_from('<I', data, 0)[0]
            offset = 4
            for _ in range(n):
                h = struct.unpack_from('<q', data, offset)[0]
                v = struct.unpack_from('<b', data, offset + 8)[0]
                self.endpoint_values[h] = v
                offset += 9
            
            # Load stats
            with open(f"{self.save_path}/stats.pkl", 'rb') as f:
                self.stats = pickle.load(f)
            
            print(f"Loaded tube network: {self.stats['total_tubes']} tubes, "
                  f"{len(self.junctions)} junctions")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False
    
    def prune(self, min_frequency=2):
        """Remove rarely-used tubes to save space"""
        pruned = 0
        for start_hash in list(self.tubes.keys()):
            tubes = self.tubes[start_hash]
            kept = [t for t in tubes if t.frequency >= min_frequency]
            pruned += len(tubes) - len(kept)
            if kept:
                self.tubes[start_hash] = kept
            else:
                del self.tubes[start_hash]
        
        print(f"Pruned {pruned} low-frequency tubes")
        return pruned
    
    def estimate_size(self):
        """Estimate storage size in bytes"""
        # Each tube: ~15-50 bytes depending on length
        avg_tube_bytes = 20 + self.stats['total_length'] * 3 // max(1, self.stats['total_tubes'])
        tube_bytes = self.stats['total_tubes'] * avg_tube_bytes
        
        # Junctions: 8 bytes each
        junction_bytes = len(self.junctions) * 8
        
        # Endpoints: 9 bytes each
        endpoint_bytes = len(self.endpoint_values) * 9
        
        total = tube_bytes + junction_bytes + endpoint_bytes
        return total


class HierarchicalTubeStorage:
    """
    Multi-level tube storage for massive scale.
    
    Level 0: Hot tubes (frequently used) - in memory
    Level 1: Warm tubes - memory-mapped file
    Level 2: Cold tubes - compressed on disk
    
    Inspired by:
    - LSM trees (log-structured merge)
    - Slime mold network optimization (prune unused paths)
    - River delta formation (main channels vs tributaries)
    """
    
    def __init__(self, base_path="./tube_storage"):
        self.base_path = base_path
        self.hot = TubeNetwork(f"{base_path}/hot")
        self.warm_path = f"{base_path}/warm"
        self.cold_path = f"{base_path}/cold"
        
        os.makedirs(base_path, exist_ok=True)
    
    def add_tube(self, tube: Tube):
        """Add to hot storage"""
        self.hot.add_tube(tube)
        tube.frequency += 1  # Boost new tubes
    
    def query(self, state: ChessState) -> Optional[int]:
        """Query across all levels"""
        # Hot first (fastest)
        result = self.hot.query(state)
        if result is not None:
            return result
        
        # TODO: warm and cold lookups
        return None
    
    def compact(self):
        """Move cold tubes to disk, keep hot in memory"""
        # Prune rarely-used tubes
        self.hot.prune(min_frequency=3)
        
        # TODO: Move warm to cold storage with compression
    
    def save(self):
        self.hot.save()
    
    def load(self):
        return self.hot.load()


# ============================================================
# INTEGRATION WITH SOLVER
# ============================================================

class TubeCollector:
    """
    Collects solution paths during solving and builds tube network.
    
    Usage:
        collector = TubeCollector()
        
        # During solving, when a path to 7-piece is found:
        collector.record_path(states, moves, syzygy_value)
        
        # After solving:
        collector.build_network()
        collector.save()
    """
    
    def __init__(self, save_path="./tube_network"):
        self.paths = []  # List of (states, moves, value)
        self.network = TubeNetwork(save_path)
    
    def record_path(self, states: List[ChessState], moves: List, value: int):
        """Record a solution path"""
        self.paths.append((states, moves, value))
    
    def build_network(self):
        """Build tube network from collected paths"""
        print(f"Building tube network from {len(self.paths)} paths...")
        
        for states, moves, value in self.paths:
            self.network.add_solution_path(states, moves, value)
        
        self.network.stats['endpoints'] = len(self.network.endpoint_values)
        self.network.stats['junctions'] = len(self.network.junctions)
        
        size = self.network.estimate_size()
        print(f"Network size: {size / 1024:.1f} KB")
        print(f"  Tubes: {self.network.stats['total_tubes']}")
        print(f"  Junctions: {self.network.stats['junctions']}")
        print(f"  Endpoints: {self.network.stats['endpoints']}")
    
    def save(self):
        self.network.save()
    
    def load(self):
        return self.network.load()


if __name__ == "__main__":
    # Demo
    print("Tube Storage Demo")
    print("="*40)
    
    network = TubeNetwork("./demo_tubes")
    
    # Create some fake tubes
    for i in range(10):
        tube = Tube(
            start_hash=hash(f"position_{i}"),
            moves=((0, 8, None), (56, 48, None)),  # Fake moves
            end_value=(i % 3) - 1,  # -1, 0, 1
            frequency=i + 1
        )
        network.add_tube(tube)
    
    print(f"Created {network.stats['total_tubes']} tubes")
    print(f"Estimated size: {network.estimate_size()} bytes")
    
    # Save and reload
    network.save()
    
    network2 = TubeNetwork("./demo_tubes")
    network2.load()
    print(f"Reloaded {network2.stats['total_tubes']} tubes")
