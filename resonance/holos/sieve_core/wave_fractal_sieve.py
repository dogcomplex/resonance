"""
WAVE FRACTAL SIEVE
==================

The correct implementation: everything is waves in an infinite-dimensional space.

CORE PRINCIPLES:
1. NO DELETION - patterns never disappear, they just dampen
2. AMPLITUDE = RELEVANCE - used patterns amplify, unused ones fade
3. INTERFERENCE - patterns reinforce or cancel through phase
4. CONTEXT ACTIVATION - current state "illuminates" relevant patterns
5. RECURSIVE - patterns of patterns, sieves within sieves

This gives us:
- Context-dependent memory: only relevant patterns activate
- Infinite history: old patterns return when context matches
- Emergent abstraction: co-occurring waves combine into higher modes
- Anthropic selection: longer-lasting patterns dominate the spectrum

The wave function never collapses - it just has regions of high and low amplitude.
A pattern learned 10,000 frames ago stays dormant until context brings it back.
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
import hashlib
from collections import defaultdict
import cmath


class WaveToken:
    """
    A token represented as a complex wave.

    - amplitude: magnitude of the wave (relevance/activation)
    - phase: encodes value/relationship information
    - The wave never dies, just dampens toward zero
    """
    __slots__ = [
        'id',
        'wave',              # Complex number: amplitude * e^(i*phase)
        'flows_to',          # Coupling waves to other tokens
        'flows_from',        # Coupling waves from other tokens
        'sub_field',         # Internal wave field (recursive sieve)
        'components',        # For composite tokens: what they're made of
    ]

    def __init__(self, token_id: str):
        self.id = token_id
        self.wave: complex = complex(0, 0)  # Starts at zero amplitude
        self.flows_to: Dict[str, complex] = {}  # Coupling as complex waves
        self.flows_from: Dict[str, complex] = {}
        self.sub_field: Dict[str, complex] = {}  # Internal patterns (recursive)
        self.components: Set[str] = set()

    @property
    def amplitude(self) -> float:
        return abs(self.wave)

    @property
    def phase(self) -> float:
        return cmath.phase(self.wave)


class WaveFractalSieve:
    """
    Fractal sieve using wave mechanics.

    Every pattern exists as a wave. Observation adds energy.
    Unused patterns dampen but never disappear.
    Context determines which patterns resonate.
    """

    def __init__(self):
        # The infinite wave field - patterns are indexed by hash
        # but conceptually this is a continuous field
        self.field: Dict[str, WaveToken] = {}

        # Composite patterns (level 1+) - also waves
        self.composites: Dict[str, WaveToken] = {}

        # Frame state
        self.frame_num: int = 0
        self.prev_active: Set[str] = set()
        self.curr_active: Set[str] = set()
        self.current_action_id: Optional[str] = None
        self.prev_action_id: Optional[str] = None

        # Temporal trace for backward amplitude flow
        self._action_trace: List[Tuple[str, complex]] = []  # (action_id, wave_state)

        # Wave mechanics parameters
        self._damping: float = 0.001  # Very slow damping - waves persist
        self._coupling_strength: float = 0.1
        self._interference_rate: float = 0.05

        # Manifold
        self._manifold_built = False
        self._frame_shape: Optional[Tuple[int, int]] = None

    def _hash(self, data: str) -> str:
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _get_token(self, token_id: str) -> WaveToken:
        """Get or create token - tokens are never deleted."""
        if token_id not in self.field:
            self.field[token_id] = WaveToken(token_id)
        return self.field[token_id]

    def _get_composite(self, comp_id: str) -> WaveToken:
        """Get or create composite pattern."""
        if comp_id not in self.composites:
            self.composites[comp_id] = WaveToken(comp_id)
        return self.composites[comp_id]

    def _build_manifold(self, h: int, w: int):
        """Build spatial adjacency - the intrinsic structure of pixel space."""
        if self._manifold_built:
            return
        self._frame_shape = (h, w)

        # Create all pixel tokens (they exist even at zero amplitude)
        for y in range(h):
            for x in range(w):
                pixel_id = self._hash(f"p_{y}_{x}")
                token = self._get_token(pixel_id)

                # Spatial neighbors get weak coupling (structure of space)
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor_id = self._hash(f"p_{ny}_{nx}")
                        # Spatial coupling as small positive real wave
                        token.flows_to[neighbor_id] = complex(0.01, 0)
                        token.flows_from[neighbor_id] = complex(0.01, 0)

        # Create action tokens
        for a in range(10):
            action_id = self._hash(f"action_{a}")
            self._get_token(action_id)

        self._manifold_built = True

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """
        Observe frame and action - inject energy into the wave field.
        """
        self.frame_num = frame_num

        # Grayscale + normalize
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float)
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val

        h, w = frame.shape
        self._build_manifold(h, w)

        # Store previous state
        self.prev_active = self.curr_active.copy()
        self.prev_action_id = self.current_action_id
        self.curr_active = set()

        # === INJECT OBSERVATIONS INTO WAVE FIELD ===
        # Each observation adds energy to the corresponding wave
        for y in range(h):
            for x in range(w):
                intensity = frame[y, x]
                pixel_id = self._hash(f"p_{y}_{x}")
                token = self._get_token(pixel_id)

                # Add energy proportional to intensity
                # Phase encodes the intensity value
                if intensity > 0:
                    phase = intensity * np.pi  # [0, pi] based on intensity
                    energy = complex(np.cos(phase), np.sin(phase)) * intensity
                    token.wave += energy * 0.1  # Accumulate, don't replace
                    self.curr_active.add(pixel_id)

        # Action injection
        action_id = self._hash(f"action_{action}")
        action_token = self._get_token(action_id)
        action_token.wave += complex(1.0, 0)  # Add positive real energy
        self.current_action_id = action_id
        self.curr_active.add(action_id)

        # === BUILD TEMPORAL COUPLINGS THROUGH ACTION BOTTLENECK ===
        self._couple_through_action()

        # === WAVE INTERFERENCE ===
        # Active patterns interfere with each other
        self._interfere()

        # === BACKWARD AMPLITUDE PROPAGATION ===
        self._propagate_backward()

        # === COMPOSITE PATTERN DISCOVERY ===
        # Co-active waves create resonance at higher modes
        self._discover_composites()

        # === ACTIVATE COMPOSITES ===
        self._activate_composites()

        # === UNIVERSAL DAMPING ===
        # All waves slowly decay (but never to exactly zero)
        self._damp_all()

    def _couple_through_action(self):
        """
        Build coupling waves through the action bottleneck.

        Coupling strength is proportional to action wave amplitude.
        This is the anthropic principle: successful actions build stronger connections.
        """
        if not self.prev_action_id:
            return

        prev_action = self.field.get(self.prev_action_id)
        if not prev_action:
            return

        # Coupling strength scales with action amplitude
        base_cs = self._coupling_strength
        amp_factor = 1.0 + prev_action.amplitude * 0.1

        # prev_pixels -> prev_action (coupling wave)
        for pixel_id in self.prev_active:
            if pixel_id != self.prev_action_id:
                pixel = self.field.get(pixel_id)
                if pixel:
                    # Coupling wave: magnitude = strength, phase = relationship
                    coupling = complex(base_cs * amp_factor, 0)

                    # Accumulate coupling (interference)
                    pixel.flows_to[self.prev_action_id] = (
                        pixel.flows_to.get(self.prev_action_id, complex(0,0)) + coupling
                    )
                    prev_action.flows_from[pixel_id] = (
                        prev_action.flows_from.get(pixel_id, complex(0,0)) + coupling
                    )

        # prev_action -> curr_pixels
        for pixel_id in self.curr_active:
            if pixel_id != self.current_action_id:
                pixel = self.field.get(pixel_id)
                if pixel:
                    coupling = complex(base_cs * amp_factor, 0)

                    prev_action.flows_to[pixel_id] = (
                        prev_action.flows_to.get(pixel_id, complex(0,0)) + coupling
                    )
                    pixel.flows_from[self.prev_action_id] = (
                        pixel.flows_from.get(self.prev_action_id, complex(0,0)) + coupling
                    )

    def _interfere(self):
        """
        Wave interference between active patterns.

        Similar phases reinforce, opposite phases cancel.
        This is how context-dependent patterns emerge.
        """
        active_list = list(self.curr_active)
        ir = self._interference_rate

        for i, t1_id in enumerate(active_list):
            t1 = self.field.get(t1_id)
            if not t1 or t1.amplitude < 0.001:
                continue

            for t2_id in active_list[i+1:]:
                t2 = self.field.get(t2_id)
                if not t2 or t2.amplitude < 0.001:
                    continue

                # Phase difference determines interference type
                phase_diff = t1.phase - t2.phase

                # Constructive (similar phase) or destructive (opposite)
                interference = np.cos(phase_diff) * ir

                # Mutual influence through coupling
                if t2_id in t1.flows_to:
                    t1.flows_to[t2_id] += complex(interference, 0)
                if t1_id in t2.flows_to:
                    t2.flows_to[t1_id] += complex(interference, 0)

    def _propagate_backward(self):
        """
        Backward amplitude propagation - the anthropic principle.

        Each frame the universe continues to exist, energy flows backward
        to all actions that led to this moment.
        """
        # Add current action to trace
        if self.current_action_id:
            action = self.field.get(self.current_action_id)
            if action:
                self._action_trace.append((self.current_action_id, action.wave))

        # Credit assignment with recency weighting
        # Recent actions get more credit for current survival
        if self._action_trace:
            lambda_discount = 0.95  # Slower discount for persistent credit
            trace_len = len(self._action_trace)

            for i, (action_id, _) in enumerate(self._action_trace):
                action = self.field.get(action_id)
                if action:
                    age = trace_len - 1 - i
                    discount = lambda_discount ** age
                    # Add survival energy (positive real)
                    action.wave += complex(0.05 * discount, 0)

    def _discover_composites(self):
        """
        Discover composite patterns from co-active waves.

        When waves consistently co-activate, they create a resonance
        at a higher mode - a composite pattern.

        NO DELETION - composites persist forever, just at varying amplitude.
        """
        # Only check periodically
        if self.frame_num % 20 != 0:
            return

        active_list = list(self.curr_active)

        # Look for strongly co-active pairs
        for i, t1_id in enumerate(active_list):
            t1 = self.field.get(t1_id)
            if not t1 or t1.amplitude < 0.1:
                continue

            for t2_id in active_list[i+1:]:
                t2 = self.field.get(t2_id)
                if not t2 or t2.amplitude < 0.1:
                    continue

                # Composite ID from components
                comp_id = self._hash(f"comp_{min(t1_id, t2_id)}_{max(t1_id, t2_id)}")
                comp = self._get_composite(comp_id)

                # Set components if new
                if not comp.components:
                    comp.components = {t1_id, t2_id}

                # Add energy to composite proportional to component amplitudes
                # Phase is sum of component phases (wave combination)
                combined_phase = t1.phase + t2.phase
                combined_amp = (t1.amplitude * t2.amplitude) ** 0.5  # Geometric mean
                energy = cmath.rect(combined_amp * 0.01, combined_phase)
                comp.wave += energy

    def _activate_composites(self):
        """
        Activate composite patterns when their components are active.

        Composites add their energy to the field, enabling hierarchical
        pattern recognition.
        """
        for comp_id, comp in self.composites.items():
            if comp.amplitude < 0.001:
                continue

            # Check if components are active
            components_active = all(
                c in self.curr_active for c in comp.components
            )

            if components_active and comp.components:
                # Composite resonates - add survival energy
                comp.wave += complex(0.02, 0)

                # Composite influences action selection
                # (This happens through the sub_field mechanism)
                for c_id in comp.components:
                    comp.sub_field[c_id] = comp.sub_field.get(c_id, complex(0,0)) + comp.wave * 0.01

    def _damp_all(self):
        """
        Universal damping - all waves slowly decay but never reach zero.

        This is NOT deletion. A wave at 1e-10 amplitude is still there,
        waiting for context to amplify it again.
        """
        damping = 1.0 - self._damping

        # Damp main field
        for token in self.field.values():
            token.wave *= damping

            # Damp couplings
            for key in token.flows_to:
                token.flows_to[key] *= damping
            for key in token.flows_from:
                token.flows_from[key] *= damping

        # Damp composites
        for comp in self.composites.values():
            comp.wave *= damping
            for key in comp.sub_field:
                comp.sub_field[key] *= damping

    def signal_game_end(self):
        """
        Game boundary - clear trace but DON'T delete anything.

        The patterns learned during this game persist in the wave field.
        """
        self._action_trace = []

    def choose_action(self, num_actions: int = 3) -> int:
        """
        Choose action based on wave resonance with current context.

        The current active pixels create a query wave.
        Actions with strong coupling waves to this query resonate more.
        Composites also contribute through their internal fields.
        """
        action_scores = {}

        for a in range(num_actions):
            action_id = self._hash(f"action_{a}")
            action_token = self.field.get(action_id)

            if not action_token:
                action_scores[a] = 1.0
                continue

            # Score from direct couplings (pixel -> action)
            coupling_score = 0.0
            for pixel_id in self.curr_active:
                pixel = self.field.get(pixel_id)
                if pixel and action_id in pixel.flows_to:
                    # Magnitude of coupling wave
                    coupling_score += abs(pixel.flows_to[action_id])

            # Score from action's own amplitude (survival history)
            amplitude_score = action_token.amplitude

            # Score from composites that involve this action
            composite_score = 0.0
            for comp in self.composites.values():
                if action_id in comp.components and comp.amplitude > 0.01:
                    # Check if other components are active
                    other_active = any(
                        c in self.curr_active
                        for c in comp.components
                        if c != action_id
                    )
                    if other_active:
                        composite_score += comp.amplitude

            action_scores[a] = (
                coupling_score +
                0.1 * amplitude_score +
                0.5 * composite_score +
                0.01  # Small baseline
            )

        # Temperature-based selection
        scores_list = list(action_scores.values())
        if len(scores_list) > 1 and max(scores_list) > min(scores_list):
            spread = (max(scores_list) - min(scores_list)) / (max(scores_list) + 0.001)
            temperature = 0.3 + 2.0 * spread

            for a in action_scores:
                if action_scores[a] > 0:
                    action_scores[a] = action_scores[a] ** (1.0 / temperature)

        total = sum(action_scores.values())
        if total < 0.001:
            return np.random.randint(0, num_actions)

        probs = {a: s / total for a, s in action_scores.items()}
        r = np.random.random()
        cumulative = 0.0
        for a, p in probs.items():
            cumulative += p
            if r < cumulative:
                return a

        return list(action_scores.keys())[-1]

    def get_action_amplitudes(self) -> Dict[int, float]:
        result = {}
        for a in range(10):
            action_id = self._hash(f"action_{a}")
            if action_id in self.field:
                result[a] = self.field[action_id].amplitude
        return result

    def get_stats(self) -> Dict:
        # Count tokens by amplitude range
        high_amp = sum(1 for t in self.field.values() if t.amplitude > 1.0)
        mid_amp = sum(1 for t in self.field.values() if 0.01 < t.amplitude <= 1.0)
        low_amp = sum(1 for t in self.field.values() if t.amplitude <= 0.01)

        total_field_energy = sum(t.amplitude for t in self.field.values())
        total_composite_energy = sum(c.amplitude for c in self.composites.values())

        active_composites = sum(1 for c in self.composites.values() if c.amplitude > 0.01)

        return {
            'frame': self.frame_num,
            'field_tokens': len(self.field),
            'high_amp_tokens': high_amp,
            'mid_amp_tokens': mid_amp,
            'low_amp_tokens': low_amp,
            'total_field_energy': total_field_energy,
            'composites': len(self.composites),
            'active_composites': active_composites,
            'total_composite_energy': total_composite_energy,
            'trace_length': len(self._action_trace),
        }

    def print_state(self):
        stats = self.get_stats()
        amps = self.get_action_amplitudes()

        print(f"\n{'='*70}")
        print(f"WAVE FRACTAL SIEVE - Frame {self.frame_num}")
        print(f"{'='*70}")
        print(f"\nWave Field:")
        print(f"  Total tokens: {stats['field_tokens']}")
        print(f"  High amplitude (>1.0): {stats['high_amp_tokens']}")
        print(f"  Mid amplitude (0.01-1.0): {stats['mid_amp_tokens']}")
        print(f"  Low amplitude (<0.01): {stats['low_amp_tokens']}")
        print(f"  Total energy: {stats['total_field_energy']:.2f}")

        print(f"\nComposite Patterns:")
        print(f"  Total: {stats['composites']}")
        print(f"  Active (>0.01): {stats['active_composites']}")
        print(f"  Total energy: {stats['total_composite_energy']:.4f}")

        print(f"\nAction amplitudes:")
        for a, amp in sorted(amps.items()):
            print(f"  action_{a}: {amp:.4f}")

        # Show top composites
        if self.composites:
            top_comps = sorted(
                self.composites.values(),
                key=lambda c: c.amplitude,
                reverse=True
            )[:5]
            print(f"\nTop composite patterns:")
            for comp in top_comps:
                print(f"  {comp.id[:8]}...: amp={comp.amplitude:.4f}, "
                      f"phase={comp.phase:.2f}, components={len(comp.components)}")


# =============================================================================
# TEST
# =============================================================================

def test_wave_pong():
    """Test wave fractal sieve on Pong."""
    print("=" * 70)
    print("WAVE FRACTAL SIEVE - PONG")
    print("No deletion, infinite wave field, context-dependent activation")
    print("=" * 70)

    FRAME_SIZE = 21
    sieve = WaveFractalSieve()

    game_lengths = []
    hits = 0
    misses = 0

    ball_x, ball_y = 10.5, 5.0
    ball_dx, ball_dy = 0.5, 0.5
    paddle_x = 10.5
    current_game_length = 0

    for frame_num in range(10000):
        # Create frame
        pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

        # Ball
        bx, by = int(ball_x), int(ball_y)
        pixels[max(0,by-1):min(FRAME_SIZE,by+1), max(0,bx-1):min(FRAME_SIZE,bx+1)] = 255

        # Paddle
        px = int(paddle_x)
        pixels[FRAME_SIZE-2:FRAME_SIZE, max(0,px-2):min(FRAME_SIZE,px+2)] = 200

        # Walls
        pixels[0:1, :] = 50
        pixels[:, 0:1] = 50
        pixels[:, FRAME_SIZE-1:FRAME_SIZE] = 50

        # Choose action
        if frame_num < 100:
            action = np.random.randint(0, 3)
        else:
            action = sieve.choose_action(num_actions=3)

        # Observe
        sieve.observe(pixels, action, frame_num)

        # Physics
        ball_x += ball_dx
        ball_y += ball_dy

        if ball_x <= 1 or ball_x >= FRAME_SIZE - 1:
            ball_dx *= -1
            ball_x = np.clip(ball_x, 1, FRAME_SIZE - 1)

        if ball_y <= 1:
            ball_dy = abs(ball_dy)

        # Apply action
        if action == 0:
            paddle_x = max(3, paddle_x - 1)
        elif action == 2:
            paddle_x = min(FRAME_SIZE - 3, paddle_x + 1)

        current_game_length += 1

        # Check hit/miss
        if ball_y >= FRAME_SIZE - 2:
            hit = abs(ball_x - paddle_x) < 3
            if hit:
                hits += 1
                ball_dy = -abs(ball_dy)
                ball_y = FRAME_SIZE - 3
            else:
                misses += 1
                game_lengths.append(current_game_length)
                current_game_length = 0
                sieve.signal_game_end()

                ball_y = 5
                ball_x = np.random.uniform(5, FRAME_SIZE - 5)
                ball_dx = np.random.choice([-0.5, 0.5])
                ball_dy = 0.5

        # Progress report
        if (frame_num + 1) % 1000 == 0:
            total_games = hits + misses
            hit_rate = hits / total_games if total_games > 0 else 0
            recent_lengths = game_lengths[-10:] if len(game_lengths) >= 10 else game_lengths
            avg_length = np.mean(recent_lengths) if recent_lengths else 0

            print(f"\n--- Frame {frame_num + 1} ---")
            print(f"Games: {total_games} (hits={hits}, misses={misses})")
            print(f"Hit rate: {hit_rate:.1%}")
            print(f"Recent avg game length: {avg_length:.1f} frames")
            sieve.print_state()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_games = hits + misses
    print(f"Total games: {total_games}")
    print(f"Hits: {hits} ({100*hits/total_games:.1f}%)")
    print(f"Misses: {misses}")

    if len(game_lengths) >= 20:
        first_10 = np.mean(game_lengths[:10])
        last_10 = np.mean(game_lengths[-10:])
        print(f"\nGame length progression:")
        print(f"  First 10 games avg: {first_10:.1f} frames")
        print(f"  Last 10 games avg: {last_10:.1f} frames")
        if last_10 > first_10:
            print(f"  Improvement: {(last_10/first_10 - 1)*100:.1f}%")

    print(f"\nFinal action amplitudes:")
    amps = sieve.get_action_amplitudes()
    for a, amp in sorted(amps.items()):
        print(f"  action_{a}: {amp:.4f}")

    stats = sieve.get_stats()
    print(f"\nWave field state:")
    print(f"  Total tokens: {stats['field_tokens']}")
    print(f"  Total composites: {stats['composites']}")
    print(f"  Active composites: {stats['active_composites']}")

    return sieve, game_lengths


if __name__ == "__main__":
    test_wave_pong()
