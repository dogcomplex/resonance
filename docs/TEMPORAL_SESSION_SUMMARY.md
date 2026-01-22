# Temporal Learning Session Summary

## What We Discovered

### 1. Original "Temporal" Scoring Was Useless
The original temporal scoring just cross-validated rules across time windows.
It wasn't learning temporal DYNAMICS at all.

### 2. Velocity Tokens (V+/V-) Capture Frame Differences
Implemented proper velocity: `V+token` = appeared, `V-token` = disappeared since last frame.

### 3. Velocity Rules Need Higher Support Threshold
| vel_min_support | F1 | Vel Rules |
|-----------------|-----|-----------|
| 3 | 59.8% | 34 |
| **5** | **69.2%** | 24 |
| 8 | 56.3% | 12 |
| 10 | 66.6% | 10 |

Low-support velocity rules overfit and hurt performance.

### 4. Velocity Rules Fire Often But Rarely Change Outcome
- Velocity rules fired: 44% of predictions
- Velocity better: 6 cases
- Velocity worse: 2 cases  
- Same: 43 cases

**Velocity helps marginally (6 vs 2) but most predictions are unchanged.**

## Current Best: ~69-70%

Both CrystalSieve and UnifiedTemporalV2 achieve similar results:
- CrystalSieve: 69.7%
- UnifiedTemporalV2 (vel_sup=5): 69.2%

## Why We're Stuck at ~70%

The bottleneck is NOT velocity. The patterns we're missing are:
1. **Long-range dependencies** - effects that depend on state far from action
2. **Rare edge cases** - insufficient support to learn reliable rules
3. **Fundamental state aliasing** - different states that look the same in our tokens

## Next Steps

1. **Cross-universe annealing** - Find rules that work across different seeds
2. **Better abstraction** - More aggressive token merging
3. **Multi-resolution** - Different tokenization granularities

## Files
- `unified_temporal_v2.py` - Best temporal sieve (matches CrystalSieve)
- `crystal_sieve_final.py` - Best non-temporal sieve
