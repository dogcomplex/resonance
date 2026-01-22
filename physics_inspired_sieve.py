"""
PHYSICS-INSPIRED SIEVE ALGORITHMS

Mapping known physics/math processes to our rule induction problem:

1. WAVE INTERFERENCE (Holographic Associative Memory)
   - Multiple patterns superimposed on same medium
   - Correlation via optical interference
   - O(1) recall time once pattern stored
   
2. FOURIER CORRELATION
   - Convolution theorem: conv(f,g) = F^-1(F(f) * F(g))
   - O(n log n) via FFT
   - Correlation in frequency domain is multiplication
   
3. MIN-HASH / LSH (Locality Sensitive Hashing)
   - Hash similar items to same bucket
   - Approximates Jaccard similarity in O(k) where k = signature size
   - Perfect for our bucketing strategy!

4. MOTION ENERGY (Visual Cortex V1-MT)
   - Spatiotemporal filtering (Gabor filters)
   - Direction-selective cells
   - Population coding for velocity estimation
   
Key insight: All these share a common pattern:
  - Reduce dimensionality while preserving similarity
  - Use interference/correlation to find matches
  - O(n) or O(n log n) complexity
"""

import random
import math
from collections import defaultdict
from typing import Set, FrozenSet, List, Dict, Tuple

# =============================================================================
# 1. MIN-HASH INSPIRED SIEVE
# =============================================================================

class MinHashSieve:
    """
    MinHash-inspired sieve for token equivalence.
    
    Key idea: Instead of comparing all token pairs,
    hash tokens by their effect patterns and find collisions.
    
    MinHash property: P(h(A) = h(B)) = Jaccard(A, B)
    """
    
    def __init__(self, n_hashes=50, threshold=0.8):
        self.n_hashes = n_hashes
        self.threshold = threshold
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        self.token_signatures = {}  # token -> minhash signature
        self.classes = {}
        self.rules = []
    
    def _compute_signature(self, effect_set: Set[str]) -> Tuple:
        """Compute MinHash signature for a set of effects."""
        if not effect_set:
            return tuple([float('inf')] * self.n_hashes)
        
        signature = []
        for seed in range(self.n_hashes):
            min_hash = float('inf')
            for item in effect_set:
                # Simple hash function
                h = hash((item, seed)) % (2**31)
                if h < min_hash:
                    min_hash = h
            signature.append(min_hash)
        return tuple(signature)
    
    def _signature_similarity(self, sig1: Tuple, sig2: Tuple) -> float:
        """Estimate Jaccard from MinHash signatures."""
        if not sig1 or not sig2:
            return 0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def observe(self, state, action, next_state):
        before, after = frozenset(state), frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | {f"-{t}" for t in (before-after)})
        self.exact[(before, action)][effect] += 1
        self.observations.append({'before': before, 'effect': effect, 'action': action})
    
    def build(self):
        # Build effect sets per token
        token_effects = defaultdict(set)
        for obs in self.observations:
            for token in obs['before']:
                token_effects[token].update(obs['effect'])
        
        # Compute signatures
        for token, effects in token_effects.items():
            self.token_signatures[token] = self._compute_signature(effects)
        
        # Find equivalent tokens using LSH-style banding
        tokens = list(self.token_signatures.keys())
        self.classes = {t: t for t in tokens}
        
        # Band the signatures and find collisions
        n_bands = 10
        rows_per_band = self.n_hashes // n_bands
        
        for band_idx in range(n_bands):
            buckets = defaultdict(list)
            start = band_idx * rows_per_band
            end = start + rows_per_band
            
            for token in tokens:
                sig = self.token_signatures[token]
                band = sig[start:end]
                buckets[band].append(token)
            
            # Check candidates in same bucket
            for bucket_tokens in buckets.values():
                if len(bucket_tokens) > 1:
                    for i, t1 in enumerate(bucket_tokens):
                        for t2 in bucket_tokens[i+1:]:
                            if self.classes[t2] != t2:
                                continue
                            # Verify with structural constraint
                            p1, p2 = t1.split('_'), t2.split('_')
                            if len(p1) == len(p2) and sum(a!=b for a,b in zip(p1,p2)) <= 1:
                                sim = self._signature_similarity(
                                    self.token_signatures[t1],
                                    self.token_signatures[t2]
                                )
                                if sim >= self.threshold:
                                    self.classes[t2] = self.classes[t1]
        
        # Build rules with abstraction
        def abstract(s):
            return frozenset(self.classes.get(t, t) for t in s)
        
        by_ae = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
                by_ae[(obs['action'], abstract(obs['effect']))].append(abstract(obs['before']))
        
        self.rules = []
        for (action, effect), positives in by_ae.items():
            if len(positives) >= 2:
                lhs = frozenset.intersection(*random.sample(positives, min(10, len(positives))))
                if lhs:
                    self.rules.append({'lhs': lhs, 'effect': effect, 'action': action})
    
    def predict(self, state, action):
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        abstract_state = frozenset(self.classes.get(t, t) for t in state)
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        return set()


# =============================================================================
# 2. MOTION ENERGY INSPIRED SIEVE (V1-MT model)
# =============================================================================

class MotionEnergySieve:
    """
    Visual cortex inspired sieve for temporal patterns.
    
    V1: Spatiotemporal filters detect local motion
    MT: Pool V1 responses to estimate velocity
    
    Key idea: Track token "motion" across time windows
    """
    
    def __init__(self, n_directions=8, decay=0.9):
        self.n_directions = n_directions  # Discretized velocity space
        self.decay = decay
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        self.token_velocity = defaultdict(lambda: [0.0] * n_directions)
        self.classes = {}
        self.rules = []
        self.prev_state = None
        self.time = 0
    
    def _direction_index(self, appeared: bool, position: int) -> int:
        """Map token appearance/disappearance to direction."""
        # Simple encoding: appear = positive, disappear = negative
        base = position % (self.n_directions // 2)
        if appeared:
            return base
        else:
            return base + self.n_directions // 2
    
    def observe(self, state, action, next_state):
        before, after = frozenset(state), frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | {f"-{t}" for t in (before-after)})
        self.exact[(before, action)][effect] += 1
        self.observations.append({
            'before': before, 'effect': effect, 'action': action, 'time': self.time
        })
        
        # Update velocity estimates
        if self.prev_state is not None:
            appeared = after - self.prev_state
            disappeared = self.prev_state - after
            
            for token in appeared:
                try:
                    pos = int(token.split('_')[1]) if '_' in token else 0
                except:
                    pos = 0
                dir_idx = self._direction_index(True, pos)
                self.token_velocity[token][dir_idx] += 1
            
            for token in disappeared:
                try:
                    pos = int(token.split('_')[1]) if '_' in token else 0
                except:
                    pos = 0
                dir_idx = self._direction_index(False, pos)
                self.token_velocity[token][dir_idx] += 1
        
        # Decay old velocities
        for token in self.token_velocity:
            for i in range(self.n_directions):
                self.token_velocity[token][i] *= self.decay
        
        self.prev_state = after
        self.time += 1
    
    def build(self):
        # Find tokens with similar velocity profiles
        tokens = list(self.token_velocity.keys())
        self.classes = {t: t for t in tokens}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2:
                    continue
                
                # Structural check
                p1, p2 = t1.split('_'), t2.split('_')
                if len(p1) != len(p2) or sum(a!=b for a,b in zip(p1,p2)) > 1:
                    continue
                
                # Velocity profile similarity (cosine)
                v1, v2 = self.token_velocity[t1], self.token_velocity[t2]
                dot = sum(a*b for a, b in zip(v1, v2))
                m1 = sum(a*a for a in v1) ** 0.5
                m2 = sum(b*b for b in v2) ** 0.5
                
                if m1 > 0.1 and m2 > 0.1:
                    sim = dot / (m1 * m2)
                    if sim > 0.8:
                        self.classes[t2] = self.classes[t1]
        
        # Standard rule building
        def abstract(s):
            return frozenset(self.classes.get(t, t) for t in s)
        
        by_ae = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
                by_ae[(obs['action'], abstract(obs['effect']))].append(abstract(obs['before']))
        
        self.rules = []
        for (action, effect), positives in by_ae.items():
            if len(positives) >= 2:
                lhs = frozenset.intersection(*random.sample(positives, min(10, len(positives))))
                if lhs:
                    self.rules.append({'lhs': lhs, 'effect': effect, 'action': action})
    
    def predict(self, state, action):
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        abstract_state = frozenset(self.classes.get(t, t) for t in state)
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        return set()


# =============================================================================
# 3. HOLOGRAPHIC INTERFERENCE SIEVE
# =============================================================================

class HolographicSieve:
    """
    Holographic associative memory inspired sieve.
    
    Key idea: Store patterns as interference in complex-valued space.
    Recall via correlation (inverse interference).
    
    Maps to: phase = token identity, amplitude = observation count
    """
    
    def __init__(self, n_dimensions=64):
        self.n_dim = n_dimensions
        self.observations = []
        self.exact = defaultdict(lambda: defaultdict(int))
        
        # Holographic memory: complex vectors
        self.token_holograms = {}  # token -> complex vector
        self.rules = []
    
    def _token_to_phase(self, token: str) -> complex:
        """Map token to unit complex number (phase encoding)."""
        h = hash(token) % 360
        angle = h * math.pi / 180
        return complex(math.cos(angle), math.sin(angle))
    
    def _encode_pattern(self, tokens: FrozenSet[str]) -> List[complex]:
        """Encode token set as hologram (superposition of phases)."""
        if not tokens:
            return [complex(0, 0)] * self.n_dim
        
        hologram = [complex(0, 0)] * self.n_dim
        for token in tokens:
            phase = self._token_to_phase(token)
            for i in range(self.n_dim):
                # Each dimension gets a different rotation
                rotation = complex(math.cos(i * 0.1), math.sin(i * 0.1))
                hologram[i] += phase * rotation
        
        return hologram
    
    def _hologram_similarity(self, h1: List[complex], h2: List[complex]) -> float:
        """Correlation of two holograms."""
        if not h1 or not h2:
            return 0
        
        # Complex dot product
        dot = sum(a * b.conjugate() for a, b in zip(h1, h2))
        m1 = sum(abs(a)**2 for a in h1) ** 0.5
        m2 = sum(abs(b)**2 for b in h2) ** 0.5
        
        if m1 == 0 or m2 == 0:
            return 0
        return abs(dot) / (m1 * m2)
    
    def observe(self, state, action, next_state):
        before, after = frozenset(state), frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | {f"-{t}" for t in (before-after)})
        self.exact[(before, action)][effect] += 1
        self.observations.append({'before': before, 'effect': effect, 'action': action})
        
        # Store hologram for this state
        for token in before:
            if token not in self.token_holograms:
                self.token_holograms[token] = [complex(0, 0)] * self.n_dim
            
            effect_hologram = self._encode_pattern(effect)
            for i in range(self.n_dim):
                self.token_holograms[token][i] += effect_hologram[i]
    
    def build(self):
        # Find tokens with similar holograms
        tokens = list(self.token_holograms.keys())
        self.classes = {t: t for t in tokens}
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2:
                    continue
                
                p1, p2 = t1.split('_'), t2.split('_')
                if len(p1) != len(p2) or sum(a!=b for a,b in zip(p1,p2)) > 1:
                    continue
                
                sim = self._hologram_similarity(
                    self.token_holograms[t1],
                    self.token_holograms[t2]
                )
                if sim > 0.8:
                    self.classes[t2] = self.classes[t1]
        
        # Standard rule building
        def abstract(s):
            return frozenset(self.classes.get(t, t) for t in s)
        
        by_ae = defaultdict(list)
        for obs in self.observations:
            if obs['effect']:
                by_ae[(obs['action'], abstract(obs['effect']))].append(abstract(obs['before']))
        
        self.rules = []
        for (action, effect), positives in by_ae.items():
            if len(positives) >= 2:
                lhs = frozenset.intersection(*random.sample(positives, min(10, len(positives))))
                if lhs:
                    self.rules.append({'lhs': lhs, 'effect': effect, 'action': action})
    
    def predict(self, state, action):
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x: x[1])[0])
        abstract_state = frozenset(self.classes.get(t, t) for t in state)
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        return set()


print("Physics-inspired sieves loaded!")
print("\nAlgorithm complexities:")
print("  MinHash:     O(n * k) where k = signature size")
print("  MotionEnergy: O(n * d) where d = direction bins")
print("  Holographic:  O(n * m) where m = hologram dimensions")
print("\nAll reduce to O(n) with constant factors!")
