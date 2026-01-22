# Physics-Inspired Sieve Analysis

## Research Summary

We explored mappings between our rule induction sieve and known physical processes:

### 1. Wave Interference / Holographic Associative Memory
- **Physical process**: Optical patterns superimposed via interference
- **Key property**: Correlation via phase conjugation = O(1) recall
- **Mapping**: Token → phase, Effect frequency → amplitude
- **Wikipedia**: "A very large number of stimulus-response patterns may be superimposed on a single neural element"

### 2. Fourier Correlation (Convolution Theorem)
- **Physical process**: Convolution in time = multiplication in frequency
- **Complexity**: O(n log n) via FFT
- **Mapping**: Token co-occurrence = convolution, Similarity = correlation
- **Key insight**: "The correlation theorem states convolution in spatial domain = multiplication in frequency domain"

### 3. Locality Sensitive Hashing (MinHash)
- **Computer science**: Hash similar items to same bucket
- **Complexity**: O(k) where k = signature size (constant!)
- **Key property**: P(h(A) = h(B)) = Jaccard(A, B)
- **Mapping**: Token effect sets → MinHash signatures → collision = equivalence

### 4. Visual Cortex Motion Detection (V1-MT)
- **Biological**: Spatiotemporal Gabor filters → direction-selective cells
- **Key property**: Population coding of velocity
- **Mapping**: Token appearance/disappearance across time → velocity estimate
- **From research**: "V1 cells tuned to different motion directions, MT pools V1 responses"

## Benchmark Results (Pong, N=50 episodes)

| Algorithm | UNSEEN Accuracy |
|-----------|-----------------|
| **Cosine (baseline)** | **45.1%** |
| MinHash/LSH | 28.2% |
| Motion Energy | 30.7% |
| Holographic | 16.0% |

## Why Cosine Still Wins

The physics-inspired approaches need tuning for this specific problem:

1. **MinHash**: Designed for set Jaccard, but we need effect DISTRIBUTION similarity
2. **Motion Energy**: Needs proper velocity space discretization for pixels
3. **Holographic**: Phase encoding loses information about frequency

## Key Insight: The Universal Pattern

All successful approaches share:
1. **Dimensionality reduction** while preserving similarity
2. **O(n) or O(n log n)** complexity
3. **Interference/correlation** for pattern matching

Our **intersection + cosine** approach IS a form of this:
- Intersection = constructive interference (matching patterns reinforce)
- Cosine similarity = correlation in effect distribution space
- Bucketing = locality-sensitive hashing of rules

## Connections to Warren's Quantum Document

From `poor_mans_quantum.txt`:
> "Only a small, coherent subset survives in readable reality, echoing the effect of quantum measurement or collapse via entropy filtering"

This is EXACTLY what our sieve does:
- Exponential state space (all possible token combinations)
- Polynomial surviving patterns (intersection finds coherent rules)
- Entropy filtering (low-probability rules discarded)

The cosine threshold (0.9) is like the "coherence time" - how similar patterns must be to constructively interfere.

## Future Directions

1. **Hybrid approach**: MinHash for candidate generation, Cosine for verification
2. **Fourier domain**: FFT on token occurrence time series
3. **Adaptive thresholds**: Learn optimal coherence threshold per domain
4. **Multi-scale**: Wavelets for temporal patterns at different scales
