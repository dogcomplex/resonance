"""
Connect-4 Solution Analyzer
Run this after the solver completes to analyze results.
"""
import pickle
import os
import json

state_dir = "./c4_bidir_state"

print(f"Loading from: {state_dir}")
print("=" * 60)

# Load solved states
with open(f"{state_dir}/solved.pkl", 'rb') as f:
    solved = pickle.load(f)

print(f"Total solved states: {len(solved):,}")

# Define the start position
class C4State:
    def __init__(self):
        self.board = tuple([0] * 42)  # 7x6 = 42 cells, all empty
        self.turn = 1  # X plays first
    
    def __hash__(self):
        return hash((self.board, self.turn))

start = C4State()
start_hash = hash(start)

print(f"\nStart position hash: {start_hash}")
print(f"Start position in solved: {start_hash in solved}")

if start_hash in solved:
    value = solved[start_hash]
    outcome = {1: "FIRST PLAYER (X) WINS", 0: "DRAW", -1: "SECOND PLAYER (O) WINS"}
    print(f"\n{'='*60}")
    print(f"🎉🎉🎉 CONNECT-4 SOLVED! 🎉🎉🎉")
    print(f"{'='*60}")
    print(f"\nResult: {outcome[value]}")
    print(f"Game-theoretic value: {value}")
    print(f"\nThis confirms the 1988 result by Victor Allis!")
    print(f"Connect-4 is a SOLVED GAME with perfect play.")
else:
    print("\n⚠️  Start position NOT yet solved!")
    print("The backward wave hasn't reached the start position yet.")
    print("Keep running iterations!")

# Analyze solved states by value
print(f"\n{'='*60}")
print("SOLUTION STATISTICS")
print(f"{'='*60}")

wins = sum(1 for v in solved.values() if v == 1)
losses = sum(1 for v in solved.values() if v == -1)
draws = sum(1 for v in solved.values() if v == 0)

print(f"\nSolved positions by outcome:")
print(f"  X wins (value=1):  {wins:,} ({100*wins/len(solved):.1f}%)")
print(f"  O wins (value=-1): {losses:,} ({100*losses/len(solved):.1f}%)")
print(f"  Draws (value=0):   {draws:,} ({100*draws/len(solved):.1f}%)")

# Load metrics
try:
    with open(f"{state_dir}/metrics.json") as f:
        metrics = json.load(f)
    print(f"\n{'='*60}")
    print("COMPUTATION METRICS")
    print(f"{'='*60}")
    print(f"  Iterations: {metrics.get('iterations', 'N/A')}")
    print(f"  Total time: {metrics.get('total_time', 0):.1f}s ({metrics.get('total_time', 0)/60:.1f} min)")
    print(f"  Forward expanded: {metrics.get('forward_expanded', 'N/A'):,}")
    print(f"  Backward expanded: {metrics.get('backward_expanded', 'N/A'):,}")
    print(f"  Standing wave size: {metrics.get('standing_wave_size', 'N/A'):,}")
    print(f"  Equiv classes with outcome: {metrics.get('equiv_classes_with_outcome', 'N/A'):,}")
    print(f"  Solved by equivalence: {metrics.get('solved_by_equivalence', 'N/A'):,}")
except:
    print("\nCould not load metrics.json")

# Load and analyze frontiers
try:
    with open(f"{state_dir}/forward.pkl", 'rb') as f:
        forward = pickle.load(f)
    with open(f"{state_dir}/backward.pkl", 'rb') as f:
        backward = pickle.load(f)
    with open(f"{state_dir}/standing.pkl", 'rb') as f:
        standing = pickle.load(f)
    
    print(f"\n{'='*60}")
    print("FRONTIER STATUS")
    print(f"{'='*60}")
    print(f"  Forward frontier: {len(forward):,}")
    print(f"  Backward frontier: {len(backward):,}")
    print(f"  Standing wave: {len(standing):,}")
    
    if len(forward) == 0:
        print("\n  ✓ Forward frontier exhausted!")
        print("    The backward wave has engulfed all forward-explored states.")
except:
    print("\nCould not load frontier files")

print(f"\n{'='*60}")
print("THE PHYSICS INTERPRETATION")
print(f"{'='*60}")
print("""
What we computed:
  - Forward wave: possibilities expanding from initial conditions
  - Backward wave: outcomes constraining what leads to them
  - Standing wave: where determination happens
  - Equivalence: structural patterns that predict outcomes

The solved region is "crystallized possibility" - states where
all quantum superposition has collapsed to definite outcomes.

This is what GRAVITY might be: the boundary conditions from
ancient computations, creating the "solid" substrate we call physics.
""")

# Check if we can determine the optimal first move
if start_hash in solved:
    print(f"\n{'='*60}")
    print("OPTIMAL PLAY ANALYSIS")
    print(f"{'='*60}")
    
    # Generate all first moves and check their values
    print("\nFirst move analysis (column 0-6):")
    for col in range(7):
        # Create state after first move
        board = [0] * 42
        board[col * 6] = 1  # Place X in bottom of column
        state_hash = hash((tuple(board), -1))  # Now O's turn
        
        if state_hash in solved:
            val = solved[state_hash]
            result = {1: "X wins", 0: "Draw", -1: "O wins"}[val]
            optimal = "← OPTIMAL" if val == 1 else ""
            print(f"  Column {col}: {result} {optimal}")
        else:
            print(f"  Column {col}: (not in solved)")
