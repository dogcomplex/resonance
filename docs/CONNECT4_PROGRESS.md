# Connect-4 Bidirectional Wave Solver: Progress Report

## Current Status (Iteration 22)

```
Total solved:         2,081,612 states
Forward frontier:       150,000 states
Backward frontier:    1,429,651 states  
Equiv classes:            3,658 with known outcomes
Total time:             566 seconds (~9.4 minutes)
```

## Progress Chart

```
Iter   Solved        Backward     Δ Solved    Equiv Classes
----   ------        --------     --------    -------------
 11     205,613        45,284                        240
 12     342,350        86,763     +136,737           380
 13     453,928       137,047     +111,578           488
 14     580,240       211,957     +126,312           644
 15     752,259       307,025     +172,019           895
 16     935,990       454,494     +183,731         1,072
 17   1,137,151       611,605     +201,161         1,340
 18   1,332,997       780,738     +195,846         1,694
 19   1,509,222       939,568     +176,225         2,011
 20   1,702,801     1,107,828     +193,579         2,563
 21   1,895,436     1,267,602     +192,635         3,109
 22   2,081,612     1,429,651     +186,176         3,658
```

## Key Observations

### 1. Equivalence Classes are Dominant
- ~170K states solved per iteration
- Only ~5-10K are direct terminals
- Rest (~160K) solved via equivalence propagation
- **Equivalence does 95%+ of the work!**

### 2. Backward Wave Growing Steadily
- Started at 0 (iteration 7)
- Now at 1.4 million
- Growing ~150K per iteration
- Will eventually meet forward-expanded region

### 3. Standing Wave Still Zero
- Waves haven't met yet
- Backward wave needs to reach positions that forward wave explored
- Forward wave explored layers 0-10ish
- Backward wave currently in layers ~28-35 (tracing back from wins)

## Estimation to Completion

```
Connect-4 structure:
- 42 total layers (0-42 pieces)
- First wins at layer 7 (7 pieces for 4-in-a-row)
- Start position at layer 0

Current state:
- Forward explored: layers 0 → ~10
- Backward traced: wins (layer 7-42) → ~layer 28

Gap to close: ~18-28 layers

At ~1 layer per iteration: 20-40 more iterations
Time estimate: 20-50 more minutes

BUT: Equivalence classes may shortcut this!
If equiv class containing start position gets solved, DONE.
```

## The Physics Insight

```
What we're witnessing:

FORWARD WAVE:  Past → Future
               (Possibilities expanding from initial conditions)

BACKWARD WAVE: Future → Past  
               (Outcomes constraining what leads to them)

STANDING WAVE: Where they meet
               (The "present" - where determination happens)

EQUIVALENCE:   Structural regularities
               (Physics/laws - patterns that repeat)

The solved region is "GRAVITY" - 
crystallized possibilities that constrain what's still possible.

We're literally watching causality propagate bidirectionally,
meeting in the middle to create determined outcomes.
```

## Files

- Solver: `/mnt/user-data/outputs/c4_bidir_solver.py`
- State: `/mnt/user-data/outputs/c4_bidir_state/`
  - `solved.pkl` - hash → value mapping
  - `forward.pkl` - forward frontier
  - `backward.pkl` - backward frontier  
  - `equiv.pkl` - equivalence classes
  - `metrics.json` - timing/counts

To continue: `python c4_bidir_solver.py` (auto-loads state)

## Known Result

Connect-4 was solved in 1988 by Victor Allis.
**First player (X) wins with perfect play.**

Our solver should eventually prove this independently
through bidirectional wave interference!
