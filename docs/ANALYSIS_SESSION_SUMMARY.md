# Analysis Session Summary

## Key Finding: ~70% Ceiling is Due to Incomplete Effect Prediction

### Error Analysis
| Error Type | Count |
|------------|-------|
| **missing** | 14 |
| partial | 5 |
| wrong_both | 3 |
| no_rule | 3 |
| extra | 2 |

**The dominant error is "missing"** - we predict PART of the effect but miss some tokens.

Example: Predicted paddle movement but missed ball movement.

### Why Velocity Doesn't Help

Velocity rules are:
1. **Too specific** - avg 7.0 LHS tokens vs 5.3 for position
2. **Low accuracy** - 43% when they fire (same as position rules at 42%)
3. **Redundant** - most predictions same as position-only

The velocity tokens are being combined with too many position tokens, making rules that rarely match.

### Why Universal Doesn't Help

Cross-universe validation:
- Crystal avg: **69.4%**
- Universal u=3 avg: 61.8%

The "universal" rules aren't better - they're just rules that happen to be common across seeds.

### The Real Problem

Looking at paddle movement patterns:
- Action 0 (stay) causes paddle movement 31-56% of time
- This is **ball-paddle collision** mixing effects together

The game physics creates correlated effects:
- Paddle moves (action-dependent)
- Ball moves (physics-dependent)
- When ball hits paddle, BOTH move in same frame

Our rules predict one or the other, not both together correctly.

## What Would Help

1. **Effect completion** - If we predict paddle effect, also predict likely ball effect
2. **Composite rules** - Rules that output multiple independent effects
3. **Better tokenization** - Separate ball/paddle into truly independent channels

## Current Best: CrystalSieve at ~69-70%

## Files
- `crystal_sieve_final.py` - Best overall
- `universal_sieve.py` - Cross-universe (didn't help)
- `simple_velocity_sieve.py` - Velocity-only rules (didn't help)
