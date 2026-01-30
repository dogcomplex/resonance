"""
PURE SIEVE: Zero Cheating
=========================

TRUE MINIMAL INPUT:
- Raw pixel array (no interpretation)
- Integer action (no label)
- Integer frame counter (no "time" concept)

ZERO HINTS:
- No semantic labels (no "bright", "wall", "linear", "x", "y", "t")
- No intensity thresholds we define
- No pre-named dimensions
- No categorization of any kind

The ONLY thing we do:
1. Hash pixel patterns -> anonymous token IDs
2. Track which tokens activate together (coupling)
3. Track which tokens activate in sequence (temporal coupling)
4. Let amplitudes drift (never delete)

Structure must EMERGE from:
- Co-activation patterns
- Temporal sequences
- Interference dynamics

Even "dimensions" are just: "when this token fires, what other
statistical properties tend to co-occur?" - emergent, not named.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import hashlib


@dataclass
class PureConfig:
    """Configuration - only numerical parameters, no semantic hints."""
    patch_size: int = 8
    n_patches: int = 10
    base_damping: float = 0.01
    min_amplitude: float = 1e-12
    coupling_strength: float = 0.1
    max_tokens: int = 2000
    frame_buffer_size: int = 5


class PureToken:
    """
    A token with NO semantic meaning.
    Just an ID and an amplitude.
    """

    def __init__(self, token_id: str):
        self.id = token_id
        self.amplitude: complex = complex(1e-8, 0)

        # Emergent properties - keyed by OTHER token IDs, not named dimensions
        # "When I fire, what else tends to fire?"
        self.co_occurrence: Dict[str, float] = defaultdict(float)

        # "When I fire, what fired just before?"
        self.preceded_by: Dict[str, float] = defaultdict(float)

        # "When I fire, what fires just after?"
        self.followed_by: Dict[str, float] = defaultdict(float)

        # Raw activation count
        self.activation_count: int = 0


class PureSieve:
    """
    A sieve with ZERO semantic hints.

    Input: pixel array, integer action, integer frame
    Output: emergent structure (couplings, amplitudes)

    NO labels. NO named dimensions. NO categorization.
    """

    def __init__(self, config: Optional[PureConfig] = None):
        self.config = config or PureConfig()

        self.tokens: Dict[str, PureToken] = {}
        self.couplings: Dict[Tuple[str, str], float] = defaultdict(float)

        self.frame_buffer: List[np.ndarray] = []
        self.activation_buffer: List[Set[str]] = []
        self.frame_count: int = 0

        self.entropy_history: List[float] = []
        self.damping = self.config.base_damping

    def _hash(self, data: np.ndarray) -> str:
        """Pure hash - no prefix, no interpretation."""
        quantized = (data.flatten() * 10).astype(np.int8)
        return hashlib.md5(quantized.tobytes()).hexdigest()[:16]

    def _get_token(self, token_id: str) -> PureToken:
        """Get or create token."""
        if token_id not in self.tokens:
            if len(self.tokens) >= self.config.max_tokens:
                # Merge into most similar existing token
                return self._find_similar(token_id)
            self.tokens[token_id] = PureToken(token_id)
        return self.tokens[token_id]

    def _find_similar(self, new_id: str) -> PureToken:
        """Find most activated token (crude similarity)."""
        return max(self.tokens.values(), key=lambda t: abs(t.amplitude))

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """
        Observe raw data. NO INTERPRETATION.

        frame: 2D numpy array of pixel values
        action: integer (we don't know what it means)
        frame_num: integer (we don't call it "time")
        """
        self.frame_count = frame_num

        # Normalize frame to [0,1] - pure numerical operation
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float) / 255.0

        # Buffer frames
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.config.frame_buffer_size:
            self.frame_buffer = self.frame_buffer[-self.config.frame_buffer_size:]

        # Current frame activations
        activated: Set[str] = set()

        # 1. Extract patches - NO CATEGORIZATION
        # Just hash whatever pattern is in each location
        patch_size = self.config.patch_size
        h, w = frame.shape
        stride_y = h // self.config.n_patches
        stride_x = w // self.config.n_patches

        for i in range(self.config.n_patches):
            for j in range(self.config.n_patches):
                y1 = i * stride_y
                y2 = min(h, y1 + patch_size)
                x1 = j * stride_x
                x2 = min(w, x1 + patch_size)

                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                # Hash the raw patch - no interpretation of what's "bright" or "wall"
                token_id = self._hash(patch)
                token = self._get_token(token_id)

                # Inject based on raw intensity (no threshold decisions)
                intensity = np.mean(patch)
                phase = intensity * np.pi
                injection = complex(intensity, 0) * np.exp(1j * phase)
                token.amplitude = token.amplitude * 0.9 + injection * 0.1
                token.activation_count += 1

                activated.add(token_id)

        # 2. Extract frame differences - NO "motion" label
        if len(self.frame_buffer) >= 2:
            diff = np.abs(self.frame_buffer[-1] - self.frame_buffer[-2])

            for i in range(self.config.n_patches):
                for j in range(self.config.n_patches):
                    y1 = i * stride_y
                    y2 = min(h, y1 + patch_size)
                    x1 = j * stride_x
                    x2 = min(w, x1 + patch_size)

                    diff_patch = diff[y1:y2, x1:x2]
                    if diff_patch.size == 0:
                        continue

                    diff_intensity = np.mean(diff_patch)
                    if diff_intensity > 0.01:  # Only hash if there's ANY difference
                        token_id = self._hash(diff_patch)
                        token = self._get_token(token_id)

                        phase = diff_intensity * np.pi
                        injection = complex(diff_intensity, 0) * np.exp(1j * phase)
                        token.amplitude = token.amplitude * 0.9 + injection * 0.1
                        token.activation_count += 1

                        activated.add(token_id)

        # 3. Extract second-order differences - NO "linear" or "acceleration" labels
        if len(self.frame_buffer) >= 3:
            d1 = self.frame_buffer[-2] - self.frame_buffer[-3]
            d2 = self.frame_buffer[-1] - self.frame_buffer[-2]
            dd = d2 - d1  # Second derivative

            # Hash the second derivative pattern
            token_id = self._hash(dd)
            token = self._get_token(token_id)

            # Inject based on how "consistent" the motion is
            # (low dd magnitude = consistent motion, but we don't call it "linear")
            consistency = 1.0 - np.clip(np.mean(np.abs(dd)), 0, 1)
            if consistency > 0.5:
                injection = complex(consistency, 0)
                token.amplitude = token.amplitude * 0.9 + injection * 0.1
                token.activation_count += 1
                activated.add(token_id)

        # 4. Action as anonymous token
        # We don't know what action "0" or "1" means - just hash it
        action_bytes = np.array([action], dtype=np.int8)
        action_id = self._hash(action_bytes)
        action_token = self._get_token(action_id)
        action_token.amplitude = action_token.amplitude * 0.9 + complex(0.3, 0) * 0.1
        action_token.activation_count += 1
        activated.add(action_id)

        # 5. Learn couplings from co-activation
        self._learn_structure(activated)

        # Buffer activations for temporal learning
        self.activation_buffer.append(activated)
        if len(self.activation_buffer) > self.config.frame_buffer_size:
            self.activation_buffer = self.activation_buffer[-self.config.frame_buffer_size:]

        # 6. Evolve
        self._evolve()

    def _learn_structure(self, activated: Set[str]):
        """
        Learn structure from activation patterns.
        NO named dimensions - just token-to-token relationships.
        """
        activated_list = list(activated)

        # Co-occurrence (same frame)
        for i, t1 in enumerate(activated_list):
            token1 = self.tokens.get(t1)
            if not token1:
                continue

            for t2 in activated_list[i+1:]:
                token2 = self.tokens.get(t2)
                if not token2:
                    continue

                # Symmetric coupling
                self.couplings[(t1, t2)] += 0.01
                self.couplings[(t2, t1)] += 0.01

                # Token-level co-occurrence tracking
                token1.co_occurrence[t2] += 0.01
                token2.co_occurrence[t1] += 0.01

        # Temporal structure (what preceded what)
        if len(self.activation_buffer) >= 2:
            prev_activated = self.activation_buffer[-2]

            for curr_id in activated:
                curr_token = self.tokens.get(curr_id)
                if not curr_token:
                    continue

                for prev_id in prev_activated:
                    prev_token = self.tokens.get(prev_id)
                    if not prev_token:
                        continue

                    # Directional coupling: prev -> curr
                    self.couplings[(prev_id, curr_id)] += 0.005

                    # Token-level temporal tracking
                    curr_token.preceded_by[prev_id] += 0.005
                    prev_token.followed_by[curr_id] += 0.005

        # Decay all couplings (but never delete)
        for key in self.couplings:
            self.couplings[key] *= 0.999

    def _evolve(self):
        """Evolve amplitudes through interference."""
        new_amplitudes: Dict[str, complex] = {}

        for token_id, token in self.tokens.items():
            # Damped self
            damped = token.amplitude * (1 - self.damping)
            if abs(damped) < self.config.min_amplitude:
                damped = complex(self.config.min_amplitude, 0)

            new_amp = damped

            # Coupling contributions
            for other_id, other in self.tokens.items():
                if other_id == token_id:
                    continue

                coupling = self.couplings.get((other_id, token_id), 0)
                if coupling > 0.0001:
                    transfer = coupling * other.amplitude * self.config.coupling_strength * 0.01
                    new_amp += transfer

            new_amplitudes[token_id] = new_amp

        # Normalize
        total = sum(abs(a) for a in new_amplitudes.values())
        if total > 100:
            scale = 100 / total
            new_amplitudes = {k: v * scale for k, v in new_amplitudes.items()}

        # Apply (floor at minimum)
        for token_id, amp in new_amplitudes.items():
            if abs(amp) < self.config.min_amplitude:
                amp = complex(self.config.min_amplitude, 0)
            self.tokens[token_id].amplitude = amp

        # Self-tune
        self._self_tune()

    def _self_tune(self):
        """Self-tune damping based on entropy."""
        amplitudes = [abs(t.amplitude) for t in self.tokens.values()
                     if abs(t.amplitude) > 0.001]

        if len(amplitudes) < 3:
            return

        total = sum(amplitudes)
        probs = [a / total for a in amplitudes]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(probs))
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        self.entropy_history.append(normalized)

        if len(self.entropy_history) < 10:
            return

        if normalized < 0.5:
            self.damping = max(0.001, self.damping * 0.95)
        elif normalized > 0.85:
            self.damping = min(0.1, self.damping * 1.05)

    def get_structure(self) -> Dict:
        """
        Query emergent structure.

        Returns anonymous IDs - interpretation is external.
        """
        # Significant tokens
        significant = [(tid, t) for tid, t in self.tokens.items()
                      if abs(t.amplitude) > 0.01]
        significant.sort(key=lambda x: -abs(x[1].amplitude))

        # Find "hub" tokens (highly connected)
        hub_scores = {}
        for tid, token in self.tokens.items():
            score = len(token.co_occurrence) + len(token.preceded_by) + len(token.followed_by)
            hub_scores[tid] = score

        hubs = sorted(hub_scores.items(), key=lambda x: -x[1])[:20]

        # Strong couplings
        strong_couplings = [(k, v) for k, v in self.couplings.items() if v > 0.01]
        strong_couplings.sort(key=lambda x: -x[1])

        # Temporal chains (A -> B -> C patterns)
        chains = []
        for tid, token in self.tokens.items():
            if abs(token.amplitude) < 0.01:
                continue
            # What does this token reliably precede?
            best_follow = max(token.followed_by.items(), key=lambda x: x[1], default=(None, 0))
            if best_follow[0] and best_follow[1] > 0.05:
                chains.append((tid, best_follow[0], best_follow[1]))

        return {
            'n_tokens': len(self.tokens),
            'n_significant': len(significant),
            'top_amplitudes': [(tid, abs(t.amplitude)) for tid, t in significant[:15]],
            'hub_tokens': hubs[:10],
            'strong_couplings': strong_couplings[:15],
            'temporal_chains': sorted(chains, key=lambda x: -x[2])[:10],
            'entropy': self.entropy_history[-1] if self.entropy_history else 0,
            'damping': self.damping,
        }

    def print_state(self):
        """Print state with anonymous IDs."""
        structure = self.get_structure()

        print(f"\n{'='*70}")
        print(f"PURE SIEVE STATE - Frame {self.frame_count}")
        print(f"{'='*70}")

        print(f"\nTokens: {structure['n_tokens']} total, {structure['n_significant']} significant")
        print(f"Entropy: {structure['entropy']:.3f}, Damping: {structure['damping']:.5f}")

        print(f"\nTop Amplitude Tokens (anonymous IDs):")
        for tid, amp in structure['top_amplitudes']:
            token = self.tokens[tid]
            n_co = len([v for v in token.co_occurrence.values() if v > 0.01])
            n_pre = len([v for v in token.preceded_by.values() if v > 0.01])
            n_fol = len([v for v in token.followed_by.values() if v > 0.01])
            print(f"  {tid}: amp={amp:.4f}, co={n_co}, pre={n_pre}, fol={n_fol}")

        print(f"\nHub Tokens (most connected):")
        for tid, score in structure['hub_tokens']:
            amp = abs(self.tokens[tid].amplitude)
            print(f"  {tid}: connections={score}, amp={amp:.4f}")

        print(f"\nTemporal Chains (A -> B):")
        for t1, t2, strength in structure['temporal_chains']:
            print(f"  {t1[:12]} -> {t2[:12]}: {strength:.4f}")

        print(f"\nStrongest Couplings:")
        for (t1, t2), strength in structure['strong_couplings'][:8]:
            print(f"  {t1[:12]} <-> {t2[:12]}: {strength:.4f}")


def test_pure_sieve():
    """Test pure sieve - NO INTERPRETATION in the test either."""
    print("=" * 70)
    print("PURE SIEVE: Zero Semantic Hints")
    print("=" * 70)

    sieve = PureSieve()

    # Simulation parameters - these are EXTERNAL to the sieve
    # The sieve doesn't know what "ball" or "paddle" means
    obj1_x, obj1_y = 42.0, 20.0
    obj1_dx, obj1_dy = 2.0, 1.5
    obj2_x = 42.0

    for frame_num in range(1500):
        # Create pixel array - just numbers, no meaning to the sieve
        pixels = np.zeros((84, 84), dtype=np.uint8)

        # Some bright pixels somewhere
        o1x, o1y = int(obj1_x), int(obj1_y)
        pixels[max(0,o1y-2):min(84,o1y+2), max(0,o1x-2):min(84,o1x+2)] = 255

        # More bright pixels at bottom
        o2x = int(obj2_x)
        pixels[78:82, max(0,o2x-8):min(84,o2x+8)] = 200

        # Dim pixels at edges
        pixels[0:2, :] = 100
        pixels[:, 0:2] = 100
        pixels[:, 82:84] = 100

        # Action - just an integer
        if obj1_x < obj2_x - 5:
            action = 0
        elif obj1_x > obj2_x + 5:
            action = 2
        else:
            action = 1

        if np.random.random() < 0.2:
            action = np.random.randint(0, 3)

        # Feed to sieve - ONLY raw data
        sieve.observe(pixels, action, frame_num)

        # Physics (completely hidden from sieve)
        obj1_x += obj1_dx
        obj1_y += obj1_dy

        if obj1_x <= 4 or obj1_x >= 80:
            obj1_dx *= -1
            obj1_x = np.clip(obj1_x, 4, 80)

        if obj1_y <= 4:
            obj1_dy = abs(obj1_dy)

        if obj1_y >= 78:
            obj1_y = 20
            obj1_x = np.random.uniform(20, 64)
            obj1_dx = np.random.choice([-2, -1.5, 1.5, 2])
            obj1_dy = abs(obj1_dy)

        if action == 0:
            obj2_x = max(10, obj2_x - 4)
        elif action == 2:
            obj2_x = min(74, obj2_x + 4)

        if (frame_num + 1) % 300 == 0:
            sieve.print_state()

    # Final analysis - EXTERNAL interpretation
    print("\n" + "=" * 70)
    print("STRUCTURE DISCOVERED (anonymous)")
    print("=" * 70)

    structure = sieve.get_structure()

    print(f"\nTotal tokens created: {structure['n_tokens']}")
    print(f"Significant (amp > 0.01): {structure['n_significant']}")

    print("\nThe sieve discovered patterns but has NO IDEA what they mean.")
    print("It just knows:")
    print("  - Some patterns have high amplitude (they persist/recur)")
    print("  - Some patterns co-occur (they're coupled)")
    print("  - Some patterns follow other patterns (temporal structure)")

    # Count dormant tokens
    dormant = sum(1 for t in sieve.tokens.values()
                  if 1e-8 < abs(t.amplitude) < 0.01)
    print(f"\nDormant tokens (preserving memory): {dormant}")

    # Analyze what the sieve found WITHOUT giving it names
    print("\n" + "=" * 70)
    print("EXTERNAL INTERPRETATION (we do this, not the sieve)")
    print("=" * 70)

    # The sieve just has anonymous IDs
    # WE (externally) can look at what patterns those correspond to
    # But the sieve itself has no idea

    # Check if there are tokens that activate consistently
    consistent_tokens = [(tid, t.activation_count)
                        for tid, t in sieve.tokens.items()
                        if t.activation_count > 100]
    consistent_tokens.sort(key=lambda x: -x[1])

    print(f"\nTokens that activated >100 times: {len(consistent_tokens)}")
    print("(These might correspond to stable features, but the sieve doesn't know)")

    # Check temporal structure
    temporal_patterns = structure['temporal_chains']
    print(f"\nTemporal chains discovered: {len(temporal_patterns)}")
    print("(These might encode causality, but the sieve doesn't label it)")

    return sieve


if __name__ == "__main__":
    sieve = test_pure_sieve()
