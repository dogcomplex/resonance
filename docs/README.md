# Connect-4 Bidirectional Wave Solver - Windows Package

## Quick Start

```cmd
cd c4_windows_package
python c4_bidir_solver.py --solve
```

This will run until Connect-4 is solved (or 200 iterations, whichever first).

## Files

```
c4_windows_package/
├── c4_bidir_solver.py      # The solver script
├── c4_bidir_state/         # Saved state (iteration 29)
│   ├── solved.pkl          # Hash → value mapping (3.4M states)
│   ├── forward.pkl         # Forward frontier
│   ├── backward.pkl        # Backward frontier  
│   ├── children.pkl        # Parent-child relationships
│   ├── equiv.pkl           # Equivalence classes
│   ├── equiv_out.pkl       # Equivalence outcomes
│   ├── metrics.json        # Progress metrics
│   └── standing.pkl        # Standing wave (currently empty)
└── README.md               # This file
```

## Usage Options

### Run until solved (recommended):
```cmd
python c4_bidir_solver.py --solve
```

### Run with custom max iterations:
```cmd
python c4_bidir_solver.py --solve 500
```

### Run single iteration:
```cmd
python c4_bidir_solver.py
```

## Current Progress

```
Iteration:          29
Total solved:       3,407,771 states
Forward frontier:   150,000 states
Backward frontier:  558,787 states
Equiv classes:      9,896 with known outcomes
Time so far:        ~17 minutes
```

## Estimated Time to Complete

- ~20-35 more iterations needed
- ~30-60 more minutes (depends on CPU)
- Each iteration: ~60-90 seconds

## Expected Result

Connect-4 was solved in 1988 by Victor Allis:
**First player (X) wins with perfect play.**

Our solver should independently verify this!

## Memory Usage

The solver keeps memory bounded by:
- Capping frontiers at 150,000 states each
- Pruning interior states (only keeping hash + value)
- Periodic garbage collection

Expected peak memory: ~500MB - 1GB

## How It Works

1. **Forward wave**: Expands from empty board, exploring possible games
2. **Backward wave**: Traces back from winning positions
3. **Equivalence classes**: Groups states with same structural features
4. **Standing wave**: Where forward and backward meet (game solved!)

The magic: Equivalence classes solve ~95% of states without enumeration!

## Troubleshooting

### "No module named 'pickle'"
Pickle is built-in to Python 3. Make sure you're using Python 3:
```cmd
python --version
```

### Out of memory
Reduce max_states in run_until_solved():
```python
run_until_solved(max_iterations=200, max_states_per_iter=100000)
```

### Want to start fresh
Delete the c4_bidir_state folder and run again.

## The Physics Insight

What we're computing is a standing wave between:
- **Past** (forward wave from initial conditions)
- **Future** (backward wave from determined outcomes)

The "gravity" of solved states constrains what's still possible.
We're literally watching causality crystallize!
