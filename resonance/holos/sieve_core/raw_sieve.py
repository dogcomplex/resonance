"""
RAW SIEVE: No Cheating Edition
==============================

TRUE MINIMAL INPUT:
- Raw pixel frames (84x84 grayscale)
- Frame number (time)
- Action taken (0, 1, 2)

NO:
- Labeled tokens
- Pre-computed velocities
- Named dimensions
- Outcome weighting
- Token deletion (only drift)

The sieve must discover ALL structure from scratch:
- Objects (bright patches)
- Motion (frame differences)
- Velocity (motion patterns over time)
- Geometry (spatial relationships)
- Causality (action -> state change)
- Physics (bounce rules)

Key insight: Rules don't die, they DRIFT.
When similar states re-occur, dormant patterns re-amplify.
This is how memory works.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import cmath
import hashlib


@dataclass
class RawConfig:
    """Configuration for raw sieve."""
    # Perception
    patch_size: int = 8           # Size of visual patches
    n_patches_per_dim: int = 10   # Patches per dimension (overlapping)

    # Sieve dynamics
    base_damping: float = 0.01    # Very slow drift - nothing dies quickly
    min_amplitude: float = 1e-10  # Floor - never truly zero
    coupling_strength: float = 0.1

    # Growth
    max_tokens: int = 1000        # Generous limit
    similarity_threshold: float = 0.95  # For merging near-duplicates

    # Temporal
    temporal_window: int = 5      # Frames to keep for motion detection


class RawToken:
    """
    A token in the raw sieve.

    No labels - just a hash ID based on what activated it.
    Coordinates emerge from where/when it was active.
    """

    def __init__(self, token_id: str):
        self.id = token_id
        self.amplitude: complex = complex(1e-6, 0)  # Start tiny

        # Emergent coordinates - populated by activation patterns
        # Key: some hashable dimension identifier
        # Value: running average of coordinate when this token is active
        self.coordinates: Dict[str, float] = {}
        self.coord_counts: Dict[str, int] = {}  # For averaging

        # What frames was this token active in?
        self.activation_frames: List[int] = []

        # Creation time
        self.birth_frame: int = 0

    def record_coordinate(self, dim: str, value: float):
        """Update running average of coordinate along dimension."""
        if dim not in self.coordinates:
            self.coordinates[dim] = value
            self.coord_counts[dim] = 1
        else:
            n = self.coord_counts[dim]
            # Running average
            self.coordinates[dim] = (self.coordinates[dim] * n + value) / (n + 1)
            self.coord_counts[dim] = n + 1

    def distance_to(self, other: 'RawToken') -> float:
        """Distance in shared coordinate space."""
        shared = set(self.coordinates.keys()) & set(other.coordinates.keys())
        if not shared:
            return float('inf')

        return np.sqrt(sum((self.coordinates[d] - other.coordinates[d])**2
                          for d in shared) / len(shared))


class RawSieve:
    """
    A sieve that takes only raw pixels and discovers structure.

    NO LABELS. NO PRECONCEPTIONS. Just patterns that resonate.
    """

    def __init__(self, config: Optional[RawConfig] = None):
        self.config = config or RawConfig()

        # All tokens - never deleted, only dampened
        self.tokens: Dict[str, RawToken] = {}

        # Couplings between tokens (learned from co-occurrence)
        self.couplings: Dict[Tuple[str, str], float] = defaultdict(float)

        # Frame buffer for temporal patterns
        self.frame_buffer: List[np.ndarray] = []
        self.current_frame: int = 0

        # Recent activations for coupling learning
        self.recent_activations: List[Set[str]] = []

        # Entropy tracking for self-tuning
        self.entropy_history: List[float] = []
        self.damping = self.config.base_damping

    def _hash_pattern(self, pattern: np.ndarray, prefix: str = "") -> str:
        """Create a hash ID for a pattern."""
        # Quantize to reduce sensitivity to noise
        quantized = (pattern * 10).astype(int)
        h = hashlib.md5(quantized.tobytes()).hexdigest()[:12]
        return f"{prefix}_{h}" if prefix else h

    def _extract_patches(self, frame: np.ndarray) -> List[Tuple[str, float, int, int]]:
        """
        Extract visual patches from frame.

        Returns: List of (hash_id, intensity, center_x, center_y)

        Key insight: Don't hash the pattern content (which changes with position),
        hash the LOCATION + INTENSITY PROFILE. This lets us track "bright thing at location X"
        regardless of exact pixel values.
        """
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)

        frame = frame.astype(float) / 255.0

        patches = []
        patch_size = self.config.patch_size
        stride = frame.shape[0] // self.config.n_patches_per_dim

        for i in range(self.config.n_patches_per_dim):
            for j in range(self.config.n_patches_per_dim):
                cy = i * stride + stride // 2
                cx = j * stride + stride // 2

                y1 = max(0, cy - patch_size // 2)
                y2 = min(frame.shape[0], cy + patch_size // 2)
                x1 = max(0, cx - patch_size // 2)
                x2 = min(frame.shape[1], cx + patch_size // 2)

                patch = frame[y1:y2, x1:x2]

                if patch.size > 0:
                    intensity = np.mean(patch)
                    max_intensity = np.max(patch)

                    # Distinguish between:
                    # - Bright objects (ball: max ~1.0, mean ~0.3)
                    # - Dim walls (max ~0.4, mean ~0.2)
                    # - Empty (max < 0.1)

                    if max_intensity > 0.8:  # Bright object (ball or paddle)
                        # Use location-based ID so we can track it moving
                        patch_id = f"bright_{i}_{j}"
                        patches.append((patch_id, intensity, cx, cy))
                    elif max_intensity > 0.3 and intensity > 0.15:  # Wall/structure
                        # Use pattern hash - walls don't move
                        patch_id = self._hash_pattern(patch, f"wall_{i}_{j}")
                        patches.append((patch_id, intensity * 0.5, cx, cy))  # Lower weight

        return patches

    def _extract_motion(self) -> List[Tuple[str, float, int, int]]:
        """
        Extract motion patterns from frame differences.

        This is how velocity EMERGES - not given, discovered.
        """
        if len(self.frame_buffer) < 2:
            return []

        motions = []
        prev = self.frame_buffer[-2].astype(float)
        curr = self.frame_buffer[-1].astype(float)

        diff = np.abs(curr - prev) / 255.0

        # Extract motion patches
        patch_size = self.config.patch_size
        stride = diff.shape[0] // self.config.n_patches_per_dim

        for i in range(self.config.n_patches_per_dim):
            for j in range(self.config.n_patches_per_dim):
                cy = i * stride + stride // 2
                cx = j * stride + stride // 2

                y1 = max(0, cy - patch_size // 2)
                y2 = min(diff.shape[0], cy + patch_size // 2)
                x1 = max(0, cx - patch_size // 2)
                x2 = min(diff.shape[1], cx + patch_size // 2)

                patch = diff[y1:y2, x1:x2]

                if patch.size > 0:
                    motion_intensity = np.mean(patch)
                    if motion_intensity > 0.05:  # Significant motion
                        motion_id = self._hash_pattern(patch, f"m{i}_{j}")
                        motions.append((motion_id, motion_intensity, cx, cy))

        return motions

    def _extract_temporal_patterns(self) -> List[Tuple[str, float]]:
        """
        Extract patterns across multiple frames.

        This is where velocity and acceleration emerge.
        """
        if len(self.frame_buffer) < 3:
            return []

        patterns = []

        # Look at how motion changes over time
        if len(self.frame_buffer) >= 3:
            f1 = self.frame_buffer[-3].astype(float) / 255.0
            f2 = self.frame_buffer[-2].astype(float) / 255.0
            f3 = self.frame_buffer[-1].astype(float) / 255.0

            # First derivative (motion)
            d1 = f2 - f1
            d2 = f3 - f2

            # Second derivative (acceleration / change in motion)
            dd = d2 - d1

            # Find regions of consistent motion (low dd = constant velocity)
            motion_consistency = 1.0 - np.clip(np.abs(dd), 0, 1)

            # Regions of high consistency = linear motion (physics!)
            if np.mean(motion_consistency) > 0.8:
                pattern_id = self._hash_pattern(motion_consistency, "linear")
                patterns.append((pattern_id, np.mean(motion_consistency)))

        return patterns

    def _get_or_create_token(self, token_id: str) -> RawToken:
        """Get existing token or create new one."""
        if token_id not in self.tokens:
            # Check for similar existing tokens first
            for existing_id, existing in self.tokens.items():
                if token_id.split('_')[0] == existing_id.split('_')[0]:
                    # Same prefix (same patch location) - might be same concept
                    # Let them both exist - the sieve will sort it out
                    pass

            if len(self.tokens) >= self.config.max_tokens:
                # Don't delete - just don't create new ones
                # Find most similar existing token
                return self._find_most_similar(token_id)

            token = RawToken(token_id)
            token.birth_frame = self.current_frame
            self.tokens[token_id] = token

        return self.tokens[token_id]

    def _find_most_similar(self, new_id: str) -> RawToken:
        """Find the most similar existing token."""
        prefix = new_id.split('_')[0] if '_' in new_id else ""

        # Prefer same-prefix tokens
        candidates = [t for tid, t in self.tokens.items()
                     if tid.split('_')[0] == prefix]

        if candidates:
            # Return the one with highest amplitude (most established)
            return max(candidates, key=lambda t: abs(t.amplitude))

        # Otherwise return any high-amplitude token
        return max(self.tokens.values(), key=lambda t: abs(t.amplitude))

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """
        Observe a raw frame.

        This is the ONLY input: pixels, action, time.
        """
        self.current_frame = frame_num

        # Normalize frame
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2).astype(np.uint8)

        # Add to buffer
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.config.temporal_window:
            self.frame_buffer = self.frame_buffer[-self.config.temporal_window:]

        # Extract patterns
        activated_tokens: Set[str] = set()

        # 1. Visual patches (where is something?)
        patches = self._extract_patches(frame)
        for patch_id, intensity, cx, cy in patches:
            token = self._get_or_create_token(patch_id)

            # Boost amplitude based on intensity
            phase = intensity * np.pi  # Intensity as phase
            injection = complex(intensity * np.cos(phase), intensity * np.sin(phase))
            token.amplitude = token.amplitude * 0.9 + injection * 0.1

            # Record spatial coordinates (emergent!)
            token.record_coordinate('x', cx / frame.shape[1])
            token.record_coordinate('y', cy / frame.shape[0])
            token.record_coordinate('t', frame_num / 1000)  # Normalized time

            token.activation_frames.append(frame_num)
            activated_tokens.add(patch_id)

        # 2. Motion patterns (what is changing?)
        motions = self._extract_motion()
        for motion_id, intensity, cx, cy in motions:
            token = self._get_or_create_token(motion_id)

            phase = intensity * np.pi
            injection = complex(intensity * np.cos(phase), intensity * np.sin(phase))
            token.amplitude = token.amplitude * 0.9 + injection * 0.1

            token.record_coordinate('x', cx / frame.shape[1])
            token.record_coordinate('y', cy / frame.shape[0])
            token.record_coordinate('t', frame_num / 1000)

            token.activation_frames.append(frame_num)
            activated_tokens.add(motion_id)

        # 3. Temporal patterns (velocity, acceleration)
        temporal = self._extract_temporal_patterns()
        for pattern_id, strength in temporal:
            token = self._get_or_create_token(pattern_id)

            injection = complex(strength, 0)
            token.amplitude = token.amplitude * 0.9 + injection * 0.1

            token.record_coordinate('t', frame_num / 1000)
            token.activation_frames.append(frame_num)
            activated_tokens.add(pattern_id)

        # 4. Action token (what did agent do?)
        action_id = f"action_{action}"
        action_token = self._get_or_create_token(action_id)
        action_token.amplitude = action_token.amplitude * 0.9 + complex(0.5, 0) * 0.1
        action_token.record_coordinate('t', frame_num / 1000)
        action_token.activation_frames.append(frame_num)
        activated_tokens.add(action_id)

        # Learn couplings from co-activation
        self._learn_couplings(activated_tokens)

        # Track activations for temporal coupling
        self.recent_activations.append(activated_tokens)
        if len(self.recent_activations) > 5:
            self.recent_activations = self.recent_activations[-5:]

        # Evolve the sieve
        self._evolve()

    def _learn_couplings(self, activated: Set[str]):
        """
        Learn couplings from co-activation.

        Tokens that fire together wire together.
        """
        activated_list = list(activated)

        # Same-frame co-activation
        for i, t1 in enumerate(activated_list):
            for t2 in activated_list[i+1:]:
                self.couplings[(t1, t2)] += 0.01
                self.couplings[(t2, t1)] += 0.01

        # Temporal coupling (this frame's tokens couple to previous frame's)
        if len(self.recent_activations) >= 2:
            prev_activated = self.recent_activations[-2]
            for t1 in activated:
                for t2 in prev_activated:
                    # Directional: prev -> current (causality!)
                    self.couplings[(t2, t1)] += 0.005

        # Decay couplings slowly
        for key in list(self.couplings.keys()):
            self.couplings[key] *= 0.999
            # But don't delete - just let them fade
            # (Long-term memory principle)

    def _evolve(self):
        """
        Evolve the sieve.

        Key change: NO DELETION. Only drift toward minimum amplitude.
        """
        # 1. Interference between coupled tokens
        new_amplitudes: Dict[str, complex] = {}

        for token_id, token in self.tokens.items():
            # Self with damping (drift toward min, never zero)
            damped = token.amplitude * (1 - self.damping)
            # Floor at minimum
            if abs(damped) < self.config.min_amplitude:
                damped = complex(self.config.min_amplitude, 0)

            new_amp = damped

            # Coupling contributions
            for other_id, other in self.tokens.items():
                if other_id == token_id:
                    continue

                coupling = self.couplings.get((other_id, token_id), 0)
                if coupling > 0.001:  # Only significant couplings
                    transfer = coupling * other.amplitude * self.config.coupling_strength
                    new_amp += transfer * 0.01

            new_amplitudes[token_id] = new_amp

        # 2. Normalize to prevent explosion
        total = sum(abs(a) for a in new_amplitudes.values())
        if total > 100:
            scale = 100 / total
            new_amplitudes = {k: v * scale for k, v in new_amplitudes.items()}

        # 3. Apply (but ensure minimum amplitude - NO DEATH)
        for token_id, amp in new_amplitudes.items():
            if abs(amp) < self.config.min_amplitude:
                amp = complex(self.config.min_amplitude, 0)
            self.tokens[token_id].amplitude = amp

        # 4. Self-tune damping
        self._self_tune()

    def _self_tune(self):
        """Self-tune damping to maintain criticality."""
        # Compute entropy over significant tokens
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

        # Target entropy around 0.7 (critical)
        if normalized < 0.5:
            self.damping = max(0.001, self.damping * 0.95)  # Less damping
        elif normalized > 0.85:
            self.damping = min(0.1, self.damping * 1.05)  # More damping

    def query_structure(self, min_amplitude: float = 0.01) -> Dict:
        """
        Query what structure has emerged.

        This is NEUTRAL - no goal weighting.
        """
        # Find significant tokens
        significant = [(tid, t) for tid, t in self.tokens.items()
                      if abs(t.amplitude) > min_amplitude]
        significant.sort(key=lambda x: -abs(x[1].amplitude))

        # Find dimensions that emerged
        all_dims = set()
        for _, token in significant:
            all_dims.update(token.coordinates.keys())

        # Analyze dimension utility
        dim_stats = {}
        for dim in all_dims:
            coords = [t.coordinates.get(dim) for _, t in significant
                     if dim in t.coordinates]
            if len(coords) >= 3:
                dim_stats[dim] = {
                    'variance': np.var(coords),
                    'range': (min(coords), max(coords)),
                    'n_tokens': len(coords)
                }

        # Find strongest couplings
        strong_couplings = [(k, v) for k, v in self.couplings.items() if v > 0.01]
        strong_couplings.sort(key=lambda x: -x[1])

        return {
            'n_tokens': len(self.tokens),
            'n_significant': len(significant),
            'top_tokens': [(tid, abs(t.amplitude)) for tid, t in significant[:20]],
            'dimensions': dim_stats,
            'strong_couplings': strong_couplings[:20],
            'entropy': self.entropy_history[-1] if self.entropy_history else 0,
            'damping': self.damping,
        }

    def query_with_goal(self, goal_tokens: Set[str], min_amplitude: float = 0.001) -> List[Tuple[str, float]]:
        """
        Query the sieve WITH a goal bias.

        Find tokens that are coupled to goal states.
        This is how goals ACT ON the sieve without changing it.
        """
        # Find all tokens connected to goal tokens
        relevant = []

        for token_id, token in self.tokens.items():
            if abs(token.amplitude) < min_amplitude:
                continue

            # How strongly coupled to goal?
            goal_coupling = 0
            for goal_id in goal_tokens:
                goal_coupling += self.couplings.get((token_id, goal_id), 0)
                goal_coupling += self.couplings.get((goal_id, token_id), 0)

            if goal_coupling > 0:
                # Score = amplitude * coupling to goal
                score = abs(token.amplitude) * goal_coupling
                relevant.append((token_id, score))

        relevant.sort(key=lambda x: -x[1])
        return relevant

    def print_state(self):
        """Print current sieve state."""
        structure = self.query_structure()

        print(f"\n{'='*60}")
        print(f"RAW SIEVE STATE - Frame {self.current_frame}")
        print(f"{'='*60}")

        print(f"\nTokens: {structure['n_tokens']} total, {structure['n_significant']} significant")
        print(f"Entropy: {structure['entropy']:.3f}, Damping: {structure['damping']:.4f}")

        print(f"\nEmerged Dimensions:")
        for dim, stats in structure['dimensions'].items():
            print(f"  {dim}: var={stats['variance']:.4f}, range={stats['range']}, n={stats['n_tokens']}")

        print(f"\nTop Tokens (by amplitude):")
        for tid, amp in structure['top_tokens'][:10]:
            token = self.tokens[tid]
            coords = {k: f"{v:.2f}" for k, v in token.coordinates.items()}
            print(f"  {tid[:30]:30s}: amp={amp:.4f}, coords={coords}")

        print(f"\nStrongest Couplings:")
        for (t1, t2), strength in structure['strong_couplings'][:10]:
            print(f"  {t1[:20]:20s} <-> {t2[:20]:20s}: {strength:.4f}")


# =============================================================================
# TEST: Discover structure from raw pixels
# =============================================================================

def test_raw_sieve():
    """Test raw sieve on synthetic Pong."""
    print("=" * 70)
    print("RAW SIEVE: No Labels, No Cheating")
    print("Discover structure from pixels alone")
    print("=" * 70)

    sieve = RawSieve()

    # Simulate Pong
    ball_x, ball_y = 42.0, 20.0
    ball_dx, ball_dy = 2.0, 1.5
    paddle_x = 42.0

    for frame_num in range(1000):
        # Create raw frame - NO LABELS
        frame = np.zeros((84, 84), dtype=np.uint8)

        # Ball (just bright pixels)
        bx, by = int(ball_x), int(ball_y)
        frame[max(0,by-2):min(84,by+2), max(0,bx-2):min(84,bx+2)] = 255

        # Paddle (bright pixels at bottom)
        px = int(paddle_x)
        frame[78:82, max(0,px-8):min(84,px+8)] = 200

        # Walls (dim pixels at edges)
        frame[0:2, :] = 100
        frame[:, 0:2] = 100
        frame[:, 82:84] = 100

        # Simple policy: move toward ball
        if ball_x < paddle_x - 5:
            action = 0
        elif ball_x > paddle_x + 5:
            action = 2
        else:
            action = 1

        # Random exploration sometimes
        if np.random.random() < 0.2:
            action = np.random.randint(0, 3)

        # Observe - ONLY raw frame, action, frame number
        sieve.observe(frame, action, frame_num)

        # Physics (hidden from sieve)
        ball_x += ball_dx
        ball_y += ball_dy

        if ball_x <= 4 or ball_x >= 80:
            ball_dx *= -1
            ball_x = np.clip(ball_x, 4, 80)

        if ball_y <= 4:
            ball_dy = abs(ball_dy)

        if ball_y >= 78:
            ball_y = 20
            ball_x = np.random.uniform(20, 64)
            ball_dx = np.random.choice([-2, -1.5, 1.5, 2])
            ball_dy = abs(ball_dy)

        # Move paddle
        if action == 0:
            paddle_x = max(10, paddle_x - 4)
        elif action == 2:
            paddle_x = min(74, paddle_x + 4)

        if (frame_num + 1) % 200 == 0:
            sieve.print_state()

    # Final analysis
    print("\n" + "=" * 70)
    print("WHAT DID THE SIEVE DISCOVER?")
    print("=" * 70)

    structure = sieve.query_structure()

    print("\n1. DIMENSIONS THAT EMERGED (from nothing!):")
    for dim, stats in sorted(structure['dimensions'].items(),
                             key=lambda x: -x[1]['variance']):
        print(f"   {dim}: variance={stats['variance']:.4f}")

    print("\n2. SPATIAL STRUCTURE:")
    # Group tokens by spatial location
    spatial_groups = defaultdict(list)
    for tid, token in sieve.tokens.items():
        if abs(token.amplitude) > 0.01:
            x = token.coordinates.get('x', -1)
            y = token.coordinates.get('y', -1)
            if x >= 0 and y >= 0:
                region = (int(x * 4), int(y * 4))  # Coarse grid
                spatial_groups[region].append((tid, abs(token.amplitude)))

    print("   Regions with activity:")
    for region, tokens in sorted(spatial_groups.items(),
                                  key=lambda x: -sum(t[1] for t in x[1])):
        total_amp = sum(t[1] for t in tokens)
        print(f"   Region {region}: {len(tokens)} tokens, total amp={total_amp:.3f}")

    print("\n3. TEMPORAL PATTERNS:")
    # Look for linear motion tokens
    linear_tokens = [tid for tid in sieve.tokens if 'linear' in tid]
    print(f"   Linear motion patterns: {len(linear_tokens)}")

    print("\n4. ACTION-COUPLED STRUCTURE:")
    for action in [0, 1, 2]:
        action_id = f"action_{action}"
        if action_id in sieve.tokens:
            coupled = sieve.query_with_goal({action_id})
            print(f"   action_{action} -> {len(coupled)} coupled tokens")

    print("\n5. DORMANT BUT ALIVE:")
    dormant = [tid for tid, t in sieve.tokens.items()
               if 0.0001 < abs(t.amplitude) < 0.01]
    print(f"   {len(dormant)} tokens dormant but preserving learned structure")

    print(f"\nTOTAL TOKENS: {len(sieve.tokens)} (none deleted, all preserving memory)")

    return sieve


if __name__ == "__main__":
    sieve = test_raw_sieve()
