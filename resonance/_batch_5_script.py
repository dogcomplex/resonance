
import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holos.core import HOLOSSolver, SeedPoint, SearchMode
from holos.games.chess_targeted import TargetedChessGame

# Load seeds
with open("./holos_targeted_sp/forward_seeds.pkl", 'rb') as f:
    forward_positions = pickle.load(f)
with open("./holos_targeted_sp/backward_seeds.pkl", 'rb') as f:
    backward_positions = pickle.load(f)

# Get batch slice
batch_forward = forward_positions[200:250]

if not batch_forward:
    print("No positions in batch")
    sys.exit(0)

# Create fresh game and solver
game = TargetedChessGame("./syzygy", "KQRRvKQR")
solver = HOLOSSolver(game, name="batch_5", max_memory_mb=2500)

# Convert to SeedPoints
forward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in batch_forward]
backward_seeds = [SeedPoint(p, SearchMode.WAVE) for p in backward_positions]

# Run solver
import time
start_time = time.time()
hologram = solver.solve(forward_seeds, backward_seeds, max_iterations=8)
elapsed = time.time() - start_time

# Save batch result
batch_result = {
    'hologram': hologram,
    'stats': solver.stats,
    'filter_stats': game.filter_stats,
    'elapsed': elapsed,
    'batch_num': 5,
}

with open("./holos_targeted_sp/batch_5_result.pkl", 'wb') as f:
    pickle.dump(batch_result, f)

print(f"Batch 5 complete: {len(hologram.solved):,} solved in {elapsed:.1f}s")
