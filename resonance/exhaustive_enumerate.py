"""
exhaustive_enumerate.py - Exhaustively enumerate all 8-piece positions
reachable from KQRRvKQR.rtbw boundary.

Designed to run in background with:
- Chunked processing (memory safe)
- Periodic checkpointing (crash recovery)
- Progress logging

Usage:
  python exhaustive_enumerate.py --chunks 100 --chunk-size 1000

This will enumerate ALL 7-piece KQRRvKQR positions (via random sampling until
saturation), generate ALL their 8-piece predecessors, and save the mapping.
"""

import os
import sys
import time
import pickle
import random
import gc
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional

from fractal_holos3 import (
    ChessState, Piece, SyzygyProbe,
    generate_predecessors, generate_moves, apply_move,
    random_position, extract_features,
    PIECE_CHARS
)


@dataclass
class EnumerationState:
    """Checkpointable enumeration state"""
    # 7-piece boundary
    boundary_7: Dict[int, Tuple[ChessState, int]]  # hash -> (state, value)

    # 8-piece predecessors
    predecessors_8: Dict[int, Tuple[ChessState, int, int]]  # hash -> (state, parent_hash, parent_value)

    # Edges: 8-piece -> list of (7-piece hash, value)
    edges: Dict[int, List[Tuple[int, int]]]

    # Progress tracking
    chunks_completed: int = 0
    total_7_sampled: int = 0
    saturation_attempts: int = 0

    # Stats
    start_time: float = 0
    last_checkpoint: float = 0


def save_checkpoint(state: EnumerationState, path: str):
    """Save enumeration state to disk"""
    # Convert ChessState objects to serializable form
    boundary_serial = {}
    for h, (s, v) in state.boundary_7.items():
        boundary_serial[h] = (list(s.pieces), s.turn, v)

    pred_serial = {}
    for h, (s, ph, pv) in state.predecessors_8.items():
        pred_serial[h] = (list(s.pieces), s.turn, ph, pv)

    data = {
        'boundary_7': boundary_serial,
        'predecessors_8': pred_serial,
        'edges': dict(state.edges),
        'chunks_completed': state.chunks_completed,
        'total_7_sampled': state.total_7_sampled,
        'saturation_attempts': state.saturation_attempts,
        'start_time': state.start_time,
    }

    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(data, f)
    os.replace(tmp_path, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [Checkpoint saved: {size_mb:.1f} MB]")


def load_checkpoint(path: str) -> Optional[EnumerationState]:
    """Load enumeration state from disk"""
    if not os.path.exists(path):
        return None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Reconstruct ChessState objects
    boundary_7 = {}
    for h, (pieces, turn, v) in data['boundary_7'].items():
        state = ChessState(pieces, turn)
        boundary_7[h] = (state, v)

    predecessors_8 = {}
    for h, (pieces, turn, ph, pv) in data['predecessors_8'].items():
        state = ChessState(pieces, turn)
        predecessors_8[h] = (state, ph, pv)

    state = EnumerationState(
        boundary_7=boundary_7,
        predecessors_8=predecessors_8,
        edges=defaultdict(list, data['edges']),
        chunks_completed=data['chunks_completed'],
        total_7_sampled=data['total_7_sampled'],
        saturation_attempts=data.get('saturation_attempts', 0),
        start_time=data['start_time'],
    )

    return state


def run_exhaustive_enumeration(
    material_7: str = "KQRRvKQR",
    output_dir: str = "./exhaustive_data",
    max_chunks: int = 1000,
    chunk_size: int = 1000,
    checkpoint_interval: int = 10,
    saturation_threshold: int = 10000,  # Stop if this many attempts yield no new positions
    syzygy_path: str = "./syzygy"
):
    """
    Exhaustively enumerate 7-piece -> 8-piece mapping.

    Strategy:
    1. Sample 7-piece positions until saturation (no new positions found)
    2. For each, generate ALL 8-piece predecessors
    3. Track the full graph: which 8-piece connect to which 7-piece

    Memory management:
    - Process in chunks
    - Checkpoint periodically
    - Can resume from checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{material_7}_checkpoint.pkl")

    print("=" * 70)
    print(f"EXHAUSTIVE ENUMERATION: {material_7}")
    print("=" * 70)

    # Initialize or load state
    state = load_checkpoint(checkpoint_path)
    if state:
        print(f"Resumed from checkpoint:")
        print(f"  7-piece boundary: {len(state.boundary_7):,}")
        print(f"  8-piece predecessors: {len(state.predecessors_8):,}")
        print(f"  Chunks completed: {state.chunks_completed}")
    else:
        state = EnumerationState(
            boundary_7={},
            predecessors_8={},
            edges=defaultdict(list),
            start_time=time.time()
        )
        print("Starting fresh enumeration")

    # Initialize Syzygy
    syzygy = SyzygyProbe(syzygy_path)
    if not syzygy.available:
        print("ERROR: Syzygy not available")
        return

    # Main enumeration loop
    print(f"\nTarget: Enumerate until saturation or {max_chunks} chunks")
    print(f"Chunk size: {chunk_size} 7-piece positions per chunk")
    print("-" * 70)

    for chunk_idx in range(state.chunks_completed, max_chunks):
        chunk_start = time.time()

        # Sample 7-piece positions
        new_7 = 0
        attempts_this_chunk = 0
        positions_this_chunk = []

        while new_7 < chunk_size:
            attempts_this_chunk += 1
            state.total_7_sampled += 1

            pos = random_position(material_7)
            if pos is None:
                continue

            h = hash(pos)
            if h in state.boundary_7:
                state.saturation_attempts += 1
                # Check saturation
                if state.saturation_attempts >= saturation_threshold:
                    print(f"\n*** SATURATION REACHED after {state.saturation_attempts} duplicate attempts ***")
                    print(f"Likely enumerated most of the 7-piece space!")
                    break
                continue

            # Reset saturation counter on new position
            state.saturation_attempts = 0

            value = syzygy.probe(pos)
            if value is None:
                continue

            state.boundary_7[h] = (pos, value)
            positions_this_chunk.append((h, pos, value))
            new_7 += 1

        # Check if we hit saturation
        if state.saturation_attempts >= saturation_threshold:
            break

        # Generate predecessors for this chunk
        new_8 = 0
        new_edges = 0

        for h7, pos7, val7 in positions_this_chunk:
            preds = generate_predecessors(pos7, max_uncaptures=5)

            for pred in preds:
                if pred.piece_count() != 8:
                    continue

                ph = hash(pred)

                # Add edge
                state.edges[ph].append((h7, val7))
                new_edges += 1

                # Add predecessor if new
                if ph not in state.predecessors_8:
                    state.predecessors_8[ph] = (pred, h7, val7)
                    new_8 += 1

        state.chunks_completed += 1
        chunk_elapsed = time.time() - chunk_start
        total_elapsed = time.time() - state.start_time

        # Progress report
        print(f"Chunk {chunk_idx + 1:4d} | "
              f"7-piece: {len(state.boundary_7):,} (+{new_7}) | "
              f"8-piece: {len(state.predecessors_8):,} (+{new_8}) | "
              f"Edges: {sum(len(v) for v in state.edges.values()):,} | "
              f"Time: {chunk_elapsed:.1f}s | "
              f"Total: {total_elapsed/60:.1f}m")

        # Checkpoint
        if (chunk_idx + 1) % checkpoint_interval == 0:
            save_checkpoint(state, checkpoint_path)
            gc.collect()

    # Final save
    print("\n" + "=" * 70)
    print("ENUMERATION COMPLETE")
    print("=" * 70)

    save_checkpoint(state, checkpoint_path)

    # Save final analysis
    analysis_path = os.path.join(output_dir, f"{material_7}_analysis.txt")
    with open(analysis_path, 'w') as f:
        f.write(f"EXHAUSTIVE ENUMERATION RESULTS: {material_7}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"7-piece boundary positions: {len(state.boundary_7):,}\n")
        f.write(f"8-piece predecessor positions: {len(state.predecessors_8):,}\n")
        f.write(f"Total edges (8->7 connections): {sum(len(v) for v in state.edges.values()):,}\n")
        f.write(f"Expansion factor: {len(state.predecessors_8)/len(state.boundary_7):.2f}x\n")
        f.write(f"\nTotal sampling attempts: {state.total_7_sampled:,}\n")
        f.write(f"Saturation attempts at end: {state.saturation_attempts:,}\n")
        f.write(f"Total time: {(time.time() - state.start_time)/60:.1f} minutes\n")

        # Value distribution
        vals_7 = [v for _, v in state.boundary_7.values()]
        f.write(f"\n7-piece value distribution:\n")
        f.write(f"  +1 (white wins): {vals_7.count(1):,}\n")
        f.write(f"   0 (draw):       {vals_7.count(0):,}\n")
        f.write(f"  -1 (black wins): {vals_7.count(-1):,}\n")

        # Connectivity stats
        edge_counts = [len(v) for v in state.edges.values()]
        if edge_counts:
            f.write(f"\n8-piece connectivity (edges to 7-piece):\n")
            f.write(f"  Average: {sum(edge_counts)/len(edge_counts):.2f}\n")
            f.write(f"  Min: {min(edge_counts)}, Max: {max(edge_counts)}\n")

    print(f"\nResults saved to: {output_dir}/")
    print(f"  Checkpoint: {material_7}_checkpoint.pkl")
    print(f"  Analysis: {material_7}_analysis.txt")

    return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exhaustively enumerate 7->8 piece mapping")
    parser.add_argument('--material', default='KQRRvKQR', help='7-piece material configuration')
    parser.add_argument('--output', default='./exhaustive_data', help='Output directory')
    parser.add_argument('--chunks', type=int, default=1000, help='Maximum chunks to process')
    parser.add_argument('--chunk-size', type=int, default=1000, help='7-piece positions per chunk')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Chunks between checkpoints')
    parser.add_argument('--saturation', type=int, default=50000, help='Duplicate attempts before stopping')
    args = parser.parse_args()

    run_exhaustive_enumeration(
        material_7=args.material,
        output_dir=args.output,
        max_chunks=args.chunks,
        chunk_size=args.chunk_size,
        checkpoint_interval=args.checkpoint_interval,
        saturation_threshold=args.saturation,
    )
