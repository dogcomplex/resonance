# Session Summary: Annealing & Velocity Learning

## Key Discoveries

### 1. Temporal Scoring Was Just Cross-Validation
The original "temporal annealing" was NOT learning velocity - it was just checking if rules worked consistently across different time windows. This is validation, not temporal learning.

### 2. True Velocity = Frame Differences
Implemented proper velocity tokens:
- `D+token` = token appeared since last frame
- `D-token` = token disappeared since last frame

This gives us first-order temporal dynamics (velocity).

### 3. Velocity Helps... Slightly
- VelocityV2: 69.3% average (vs Crystal's 69.5%)
- Velocity rules ARE being matched (46 times in test)
- In 4 cases, velocity was BETTER than position-only
- Most predictions (151/168) are identical

### 4. Effects Are Partially Independent
- 559 paddle-only effects
- 84 ball-only effects  
- 959 combined effects

BUT decomposing hurts (58.6% vs 69.5%) because paddle/ball positions ARE correlated (paddle position affects ball bouncing).

### 5. Performance Peaks Then Declines
| Episodes | UNSEEN |
|----------|--------|
| 30 | 63.5% |
| 50 | 67.4% |
| **75** | **69.2%** |
| 100 | 66.8% |
| 200 | 65.3% |

More data = more rules = overfitting. Sweet spot around 75 episodes.

## Current Best: CrystalSieve at 69.5%

## Why Not 100%?

1. **Velocity rules have low support** - velocity patterns are rare/specific
2. **Rules are too specific** - even with abstraction, LHS conditions are complex
3. **Missing higher-order dynamics** - acceleration (delta of deltas) might help
4. **Independent effects idea needs refinement** - decomposing naively hurts

## Next Steps to Try

1. **Cross-Universe Annealing** - Rules that work across different seeds
2. **Acceleration tokens** - Second-order derivatives (D2+ prefixes)
3. **Better rule generalization** - More aggressive abstraction
4. **Ensemble methods** - Combine predictions from multiple rule sets

## Files Saved
- `crystal_sieve_final.py` - Best sieve (69.5%)
- `velocity_sieve_v2.py` - Velocity-aware version
- `decomposed_sieve.py` - Independent effects (didn't help)
- `multidim_anneal.py` - Multi-dimensional experiments
- `pixel_environments.py` - Test environments
