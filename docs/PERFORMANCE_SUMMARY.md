# Production Rule Learner - Performance Summary

## Final Results

### Per-Label Accuracy (100% after training)

| Label | Accuracy | Count |
|-------|----------|-------|
| ok | 100% | 4520 |
| winX | 100% | 1098 |
| winO | 100% | 412 |
| draw | 100% | 16 |

### Early-Round Performance Comparison

| Strategy | @10 | @50 | @100 | @500 | @1000 | @2000 |
|----------|-----|-----|------|------|-------|-------|
| BlindPrior | 67% | 75% | 74% | 75% | 75% | 74% |
| RandomHypothesis | 67% | 67% | 70% | 79% | 84% | 89% |
| EleganceWeighted | 60% | 61% | 67% | 82% | 88% | 91% |
| **CompleteLearner** | **70%** | **74%** | **77%** | **91%** | **95%** | **97%** |

### Convergence

- Discovers all 8 X-win lines by observation ~100
- Discovers all 8 O-win lines by observation ~200
- Achieves 99%+ by observation ~3000
- 100% on fresh data after full training

## Rule Format

```
# Priority syntax (PRIORITY3 > PRIORITY2 > PRIORITY > none)
PRIORITY3 p0_1  p1_1  p2_1  =>  p0_1  p1_1  p2_1  winX

# Negation as equivalence class (!p0_0 = {p0_1, p0_2})
PRIORITY !p0_0  board_full  =>  draw

# Quantities (spend 3 gold -> 1 sword)
gold3  =>  sword

# Probabilistic (70% hit, 30% miss)
attack  =>  hit%70  miss%30

# Catalysts (present on both sides)
p0_1  p1_1  p2_1  =>  p0_1  p1_1  p2_1  winX
```

## Discovered Rules (18 total)

### X Wins (PRIORITY3) - 8 rules
```
PRIORITY3 p0_1  p1_1  p2_1  =>  winX  # Top row
PRIORITY3 p3_1  p4_1  p5_1  =>  winX  # Middle row
PRIORITY3 p6_1  p7_1  p8_1  =>  winX  # Bottom row
PRIORITY3 p0_1  p3_1  p6_1  =>  winX  # Left column
PRIORITY3 p1_1  p4_1  p7_1  =>  winX  # Middle column
PRIORITY3 p2_1  p5_1  p8_1  =>  winX  # Right column
PRIORITY3 p0_1  p4_1  p8_1  =>  winX  # Main diagonal
PRIORITY3 p2_1  p4_1  p6_1  =>  winX  # Anti-diagonal
```

### O Wins (PRIORITY2) - 8 rules
Same patterns with `p*_2` => `winO`

### Draw (PRIORITY1) - 1 rule
```
PRIORITY board_full  !p0_0  !p1_0  ...  !p8_0  =>  draw
```

### Default - 1 rule
```
default  =>  ok
```

## Key Insights

### O-line Discovery Challenge
- O-lines can't achieve 100% precision for `winO` alone
- When BOTH players have 3-in-row, label is `winX` (X checked first)
- **Solution**: O-lines have 100% precision for `(winX OR winO)` - game-ending

### Both-Win States
- 312 states where both X and O have 3-in-row
- These are UNREACHABLE in valid TicTacToe play
- Should be labeled `error`, not `winX`
- Current oracle labels them `winX` (X priority)

### Hypothesis Management
- Early: Many valid hypotheses (658+ at @10)
- Over time: Contradictions eliminate invalid ones
- Final: Only true win patterns remain (~10 hypotheses)
- Elegance weighting helps select among valid hypotheses

## Early Performance Analysis

### Key Finding
The baseline learner is near-optimal for early predictions because:
- 75% of states are 'ok'
- Default prediction of 'ok' achieves ~75-80% accuracy
- This IS the Bayesian optimal before evidence accumulates

### What Doesn't Help
- **Structural priors**: Predicting wins too eagerly hurts accuracy
- **Aggressive hypotheses**: More false positives than true positives early
- **Stratified sampling**: Helps see rare labels but biases distribution

### What Does Help
- **Symmetry transfer**: Use X lines to hypothesize O lines (conservative)
- **Support threshold 2+**: Avoid single-observation false rules
- **Elegance scoring**: When multiple hypotheses match, prefer simpler

### Theoretical Limit
Early accuracy is bounded by:
- @1: ~75% (prior)
- @10: ~82% (some wins seen, rules forming)
- @100: ~85% (most patterns seen once)
- @500: ~92% (rules confirmed)
- @1000: ~95% (near-complete)

### Error State Learning
The learner successfully detects error states:
- Invalid parity (wrong X/O count)
- Both-win (impossible game state)
- 100% accuracy on error detection after training

### Varied Rulesets
Tested on different game rules:
| Ruleset | Win Lines | Final Accuracy | Lines Found |
|---------|-----------|----------------|-------------|
| Standard | 8 | 99%+ | 8 X, 8 O |
| Rows Only | 3 | 99%+ | 3 X, 3 O |
| Diagonals Only | 2 | 99%+ | 2 X, 2 O |

## Chaos Gauntlet Results

### Test Suite
29 different rulesets including:
- Standard variants (rows, cols, diags, combinations)
- Weird patterns (L-shapes, knight moves)
- Random rules (2-25 random 3-position patterns)
- Large patterns (4, 5, 6 positions)
- Asymmetric games (X and O have different win conditions!)
- Extreme density (1-25 rules)

### Performance Summary

| Metric | @50 | @100 | Final |
|--------|-----|------|-------|
| Min | 58% | 68% | 78% |
| Avg | 87% | 88% | 96% |
| Max | 100% | 100% | 100% |

### Benchmarks
- Perfect (99%+): 17/29 (59%)
- Good (95%+): 24/29 (83%)
- Rule discovery: 116% avg (some over-discovery)

### Top Performers (99%+ final accuracy)
- single_line, corners_4, edges_4
- All sparse rulesets (1-2 rules)
- knight_moves, L_shapes, random_4pos
- Asymmetric rows_vs_cols

### Challenging Cases
| Ruleset | Final | Issue |
|---------|-------|-------|
| extreme_dense_2 | 78% | 18+ rules, many collisions |
| extreme_dense_0 | 80% | Too many hypotheses |
| extreme_asym | 85% | 15 X rules vs 2 O rules |

### Noise Robustness
Tested with label noise (wrong labels during training):

| Learner | 0% Noise | 5% Noise | 10% Noise |
|---------|----------|----------|-----------|
| Original (prec=1.0) | 99% | 82% | 79% |
| Robust (prec=0.95) | 95% | 97% | 96% |
| Robust (prec=0.90) | 98% | 97% | 96% |

Key insight: Soft precision threshold enables noise robustness!

### Key Findings

1. **Sparse rules are easy**: 1-2 rules converge quickly to 99%+
2. **Dense rules are hard**: 15+ rules create hypothesis collisions
3. **Asymmetry is learnable**: X and O can have different rules
4. **Large patterns work**: 4-5 position patterns discoverable
5. **6+ positions rarely fire**: Too rare to observe enough
6. **Noise tolerance**: 95% precision threshold handles 10% noise
