# Wave/Crystal Sieve Progress Summary

## Performance Evolution

| Version | UNSEEN F1 | Key Insight |
|---------|-----------|-------------|
| WaveSieve (baseline) | 46% | Cosine similarity + intersection |
| CrystalSieve | **69%** | Anneal RULES not classes |
| MultiDimAnneal | 49-62% | 3D scoring too strict |
| CrossUniverse | 67% | Cross-validation helps |
| Pruned | 6-29% | Pruning too aggressive |

## Best Configuration

**CrystalSieve** at ~75 training episodes:
- UNSEEN: **69.2%**
- SEEN: 81.0%
- Rules: 125
- Classes: 64

## Key Discoveries

### 1. Anneal Rules, Not Classes
The breakthrough was realizing that token equivalence (classes) is well-determined by cosine similarity. The annealing should happen at the RULE level:
- Generate multiple candidate rules
- Validate against held-out data
- Keep only "cold" (confident) rules

### 2. Overfitting with More Data
Performance peaks at ~75 episodes then declines:
- 30 eps: 63.5%
- 75 eps: 69.2% (peak)
- 200 eps: 65.3%

More rules = more overfitting. Need better rule selection.

### 3. Multi-Dimensional Scoring Hurts
Adding temporal/fidelity/probability scoring made things worse because:
- Combined scoring is too strict
- Rules fail one dimension but work well overall
- Simple validation is enough

### 4. The 100% Gap
Current best is ~69%. Gap to 100% likely due to:
- Pong requires VELOCITY prediction (where ball is going)
- Current tokenization is positional only
- Need temporal dynamics in the representation

## Next Steps for 100%

1. **Velocity Encoding**: Add delta-position tokens
2. **Better Rule Selection**: Don't over-prune, but rank better
3. **Temporal Rules**: Rules that span multiple timesteps
4. **Cross-Universe Distillation**: Use rules from multiple seeds to find universal patterns

## Files

- `crystal_sieve_final.py` - Best performing sieve
- `wave_sieve_final.py` - Baseline wave sieve
- `multidim_anneal.py` - Multi-dimensional experiments
- `pixel_environments.py` - Pong/Breakout environments
