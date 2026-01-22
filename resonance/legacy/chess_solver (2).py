"""
chess_solver.py - Lightning solver with tube collection

Features:
- Resumes from saved state on crash
- Collects solution paths as "tubes" 
- Stores tubes instead of full solutions
- Periodic checkpointing
"""

import os
import sys
import time
import pickle
import random
from collections import defaultdict, deque
from typing import Dict, Set, List, Optional

from chess_core import (
    ChessState, Piece, PIECE_CHARS,
    generate_moves, apply_move, is_terminal,
    SyzygyProbe, random_position, parse_material
)
from chess_tubes import TubeNetwork, Tube, TubeCollector


SAVE_DIR = "./solver_state"


class LightningSolver:
    """
    Lightning solver with:
    - Crash recovery via checkpoints
    - Tube collection for compact storage
    - Progress tracking
    """
    
    def __init__(self, syzygy_path="./syzygy", save_dir=SAVE_DIR):
        self.syzygy = SyzygyProbe(syzygy_path)
        self.save_dir = save_dir
        self.tube_collector = TubeCollector(f"{save_dir}/tubes")
        
        # Core state
        self.solved: Dict[int, int] = {}
        self.frontier: Dict[int, ChessState] = {}
        self.all_seen: Set[int] = set()
        self.children: Dict[int, List[int]] = {}
        self.parents: Dict[int, List[int]] = defaultdict(list)
        self.state_cache: Dict[int, ChessState] = {}  # For path reconstruction
        
        # Progress tracking
        self.depth = 0
        self.start_hashes: List[int] = []  # Original starting positions
        
        self.metrics = {
            'positions_explored': 0,
            'syzygy_probes': 0,
            'syzygy_hits': 0,
            'tubes_collected': 0,
            'checkpoint_time': None,
            'solve_start': None,
        }
    
    def add_starting_position(self, state: ChessState, show=True):
        """Add a starting position"""
        h = hash(state)
        if h not in self.all_seen:
            self.all_seen.add(h)
            self.frontier[h] = state
            self.state_cache[h] = state
            self.start_hashes.append(h)
            if show:
                print(f"Added starting position:")
                state.display()
    
    def checkpoint(self):
        """Save current state for crash recovery"""
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save core state (solved positions, frontier hashes, depth)
        checkpoint = {
            'solved': self.solved,
            'frontier_hashes': list(self.frontier.keys()),
            'all_seen': self.all_seen,
            'children': self.children,
            'parents': dict(self.parents),
            'depth': self.depth,
            'start_hashes': self.start_hashes,
            'metrics': self.metrics,
        }
        
        with open(f"{self.save_dir}/checkpoint.pkl", 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save state cache (for path reconstruction) - only recent
        # Keep cache size bounded
        if len(self.state_cache) > 1000000:
            # Keep only solved and frontier states
            important = set(self.solved.keys()) | set(self.frontier.keys()) | set(self.start_hashes)
            self.state_cache = {h: s for h, s in self.state_cache.items() if h in important}
        
        with open(f"{self.save_dir}/state_cache.pkl", 'wb') as f:
            pickle.dump(self.state_cache, f)
        
        # Save tubes
        self.tube_collector.save()
        
        self.metrics['checkpoint_time'] = time.time()
        print(f"  [Checkpoint saved: depth={self.depth}, solved={len(self.solved):,}]")
    
    def load_checkpoint(self):
        """Load from checkpoint if available"""
        checkpoint_path = f"{self.save_dir}/checkpoint.pkl"
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.solved = checkpoint['solved']
            self.all_seen = checkpoint['all_seen']
            self.children = checkpoint['children']
            self.parents = defaultdict(list, checkpoint['parents'])
            self.depth = checkpoint['depth']
            self.start_hashes = checkpoint['start_hashes']
            self.metrics = checkpoint['metrics']
            
            # Load state cache
            cache_path = f"{self.save_dir}/state_cache.pkl"
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.state_cache = pickle.load(f)
            
            # Reconstruct frontier from cached states
            frontier_hashes = checkpoint['frontier_hashes']
            for h in frontier_hashes:
                if h in self.state_cache:
                    self.frontier[h] = self.state_cache[h]
            
            # Load tubes
            self.tube_collector.load()
            
            print(f"✓ Resumed from checkpoint: depth={self.depth}, "
                  f"solved={len(self.solved):,}, frontier={len(self.frontier):,}")
            return True
        except Exception as e:
            print(f"✗ Checkpoint load failed: {e}")
            return False
    
    def collect_tube(self, start_hash: int, end_hash: int, value: int):
        """
        Collect a tube from start to end position.
        Reconstructs the path using parent pointers.
        """
        # Trace path backward from end to start
        path_hashes = [end_hash]
        current = end_hash
        
        while current != start_hash and current in self.parents:
            parent_list = self.parents[current]
            if not parent_list:
                break
            # Pick first parent (any path works)
            current = parent_list[0]
            path_hashes.append(current)
            if len(path_hashes) > 100:  # Safety limit
                break
        
        path_hashes.reverse()
        
        if path_hashes[0] != start_hash:
            return  # Couldn't trace back
        
        # Reconstruct states and moves
        states = []
        moves = []
        for i, h in enumerate(path_hashes):
            if h in self.state_cache:
                states.append(self.state_cache[h])
            else:
                break
            
            if i < len(path_hashes) - 1:
                # Find move that led to next state
                # This is approximate - we record the hash transition
                moves.append((0, 0, None))  # Placeholder
        
        if len(states) >= 2:
            self.tube_collector.record_path(states, moves, value)
            self.metrics['tubes_collected'] += 1
    
    def expand_layer(self, max_positions=100000):
        """Expand frontier by one layer"""
        items = list(self.frontier.items())[:max_positions]
        total = len(items)
        terminals = 0
        new_states = 0
        contacts = 0
        next_frontier = {}
        
        start_time = time.time()
        
        for i, (h, state) in enumerate(items):
            if i > 0 and i % 50000 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"    {i:,}/{total:,} ({100*i/total:.0f}%) - "
                      f"{rate:.0f} pos/s - contacts: {contacts:,}")
            
            # Check terminal
            term, value = is_terminal(state)
            if term:
                self.solved[h] = value
                terminals += 1
                continue
            
            # Generate moves
            moves = generate_moves(state)
            child_hashes = []
            
            for move in moves:
                child = apply_move(state, move)
                ch = hash(child)
                child_hashes.append(ch)
                self.parents[ch].append(h)
                
                # Capture leading to ≤7 pieces?
                if move[2] is not None and child.piece_count() <= 7:
                    self.metrics['syzygy_probes'] += 1
                    value = self.syzygy.probe(child)
                    
                    if value is not None:
                        self.metrics['syzygy_hits'] += 1
                        self.solved[ch] = value
                        contacts += 1
                        
                        # Collect tube from any starting position
                        for start_h in self.start_hashes[:10]:
                            if start_h in self.all_seen:
                                self.collect_tube(start_h, ch, value)
                        
                        if contacts == 1:
                            print(f"\n  ⚡ FIRST CONTACT at {child.piece_count()}-piece boundary!")
                
                # Add to frontier if new
                if ch not in self.all_seen:
                    self.all_seen.add(ch)
                    next_frontier[ch] = child
                    self.state_cache[ch] = child
                    new_states += 1
            
            self.children[h] = child_hashes
        
        # Update frontier
        for h, _ in items:
            if h in self.frontier:
                del self.frontier[h]
        self.frontier.update(next_frontier)
        self.depth += 1
        self.metrics['positions_explored'] = len(self.all_seen)
        
        return total, terminals, new_states, contacts
    
    def propagate_solutions(self, max_iters=500):
        """Propagate solutions backward"""
        total_solved = 0
        
        for iteration in range(max_iters):
            newly_solved = 0
            
            for h in list(self.children.keys()):
                if h in self.solved:
                    continue
                
                child_values = []
                unknown = 0
                
                for ch in self.children[h]:
                    if ch in self.solved:
                        child_values.append(self.solved[ch])
                    else:
                        unknown += 1
                
                if not child_values:
                    continue
                
                # Determine turn from state cache
                turn = 'w'
                if h in self.state_cache:
                    turn = self.state_cache[h].turn
                
                # Early termination
                if turn == 'w' and 1 in child_values:
                    self.solved[h] = 1
                    newly_solved += 1
                elif turn == 'b' and -1 in child_values:
                    self.solved[h] = -1
                    newly_solved += 1
                elif unknown == 0:
                    value = max(child_values) if turn == 'w' else min(child_values)
                    self.solved[h] = value
                    newly_solved += 1
            
            total_solved += newly_solved
            
            if newly_solved == 0:
                break
            
            if iteration % 20 == 0 and newly_solved > 0:
                print(f"    Propagation: +{newly_solved:,}, total: {len(self.solved):,}")
        
        return total_solved
    
    def solve(self, max_depth=100, batch_size=500000, checkpoint_interval=5):
        """Main solving loop with checkpointing"""
        print("="*60)
        print("⚡ LIGHTNING SOLVER WITH TUBE COLLECTION ⚡")
        print("="*60)
        
        if not self.frontier:
            print("No starting positions!")
            return
        
        self.metrics['solve_start'] = time.time()
        
        for depth in range(self.depth, max_depth):
            self.depth = depth
            
            print(f"\n{'='*40}")
            print(f"DEPTH {depth}")
            print(f"{'='*40}")
            print(f"  Frontier: {len(self.frontier):,}")
            print(f"  Total seen: {len(self.all_seen):,}")
            
            if not self.frontier:
                print("  Frontier empty!")
                break
            
            # Expand
            expanded, terminals, new_states, contacts = self.expand_layer(batch_size)
            print(f"  Expanded: {expanded:,}")
            print(f"  Terminals: {terminals:,}, New: {new_states:,}")
            print(f"  Contacts: {contacts:,}")
            
            # Propagate
            propagated = self.propagate_solutions()
            print(f"  Propagated: {propagated:,}")
            print(f"  Solved: {len(self.solved):,} / {len(self.all_seen):,}")
            
            # Check starting positions
            solved_starts = sum(1 for h in self.start_hashes if h in self.solved)
            if solved_starts > 0:
                print(f"\n  🎉 {solved_starts}/{len(self.start_hashes)} starting positions solved!")
            
            # Checkpoint
            if depth % checkpoint_interval == 0:
                self.checkpoint()
            
            # Time estimate
            elapsed = time.time() - self.metrics['solve_start']
            rate = len(self.all_seen) / elapsed if elapsed > 0 else 0
            print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m), Rate: {rate:.0f} pos/s")
            print(f"  Tubes collected: {self.metrics['tubes_collected']:,}")
        
        # Final checkpoint and summary
        self.checkpoint()
        self.print_summary()
    
    def print_summary(self):
        """Print final summary"""
        elapsed = time.time() - self.metrics['solve_start']
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Positions explored: {len(self.all_seen):,}")
        print(f"Positions solved: {len(self.solved):,}")
        print(f"Tubes collected: {self.metrics['tubes_collected']:,}")
        print(f"Syzygy probes: {self.metrics['syzygy_probes']:,}")
        print(f"Syzygy hits: {self.metrics['syzygy_hits']:,}")
        print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        
        # Outcome distribution
        if self.solved:
            wins = sum(1 for v in self.solved.values() if v == 1)
            draws = sum(1 for v in self.solved.values() if v == 0)
            losses = sum(1 for v in self.solved.values() if v == -1)
            print(f"\nOutcomes: W={wins:,} D={draws:,} L={losses:,}")
        
        # Starting position results
        print(f"\nStarting positions:")
        for i, h in enumerate(self.start_hashes[:10]):
            if h in self.solved:
                v = self.solved[h]
                result = {1: "White wins", 0: "Draw", -1: "Black wins"}[v]
                print(f"  #{i+1}: {result}")
            else:
                print(f"  #{i+1}: Not yet solved")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', type=str, default='KQRRvKQRR')
    parser.add_argument('--positions', type=int, default=1000)
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--batch', type=int, default=500000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    solver = LightningSolver()
    
    # Try to resume from checkpoint
    if args.resume and solver.load_checkpoint():
        print("Resuming from checkpoint...")
    else:
        print(f"\n{'='*60}")
        print(f"⚡ 8-PIECE CHESS ENDGAME SOLVER ⚡")
        print(f"{'='*60}")
        print(f"Material: {args.material}")
        print(f"Positions: {args.positions}")
        
        # Generate starting positions
        print(f"\nGenerating {args.positions} random {args.material} positions...")
        generated = 0
        while generated < args.positions:
            state = random_position(args.material)
            if state:
                show = generated < 3
                solver.add_starting_position(state, show=show)
                generated += 1
                if generated % 200 == 0:
                    print(f"  Generated {generated}...")
        
        print(f"\nStarting with {len(solver.frontier)} positions")
    
    # Solve!
    solver.solve(max_depth=args.depth, batch_size=args.batch)


if __name__ == "__main__":
    main()
