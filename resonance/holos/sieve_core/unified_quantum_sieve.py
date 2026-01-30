"""
UNIFIED QUANTUM SIEVE
=====================

Consolidates the key insights from natural exploration testing:

1. THERMAL FLOOR + SUPERPOSITION are the SAME physics:
   - Thermal floor = zero-point energy = minimum fluctuation per mode
   - Superposition = wave function maintains all possibilities until collapse

   The unified view: Zero-point energy IS the superposition of vacuum fluctuations.
   Every mode has E = hbar*omega/2 even at T=0.

2. Core mechanism:
   - Actions exist as wave function amplitudes (complex numbers)
   - Probability = |amplitude|^2 (Born rule)
   - Zero-point energy: Every mode has minimum amplitude > 0
   - Collapse: Strong evidence shifts amplitude distribution
   - Thermal bath: Mismatch energy redistributed, not lost

3. This is ONE mechanism, not two layered:
   - The zero-point floor EMERGES from maintaining superposition
   - The superposition IS the thermal floor - they're dual descriptions

Test results that led here:
- Forced exploration: 95.6% accuracy
- Superposition: 98.5% simple / 180.9 steps dodge (7.6x random)
- Thermal floor: 99.1% simple / 43.0 steps dodge (1.8x random)
"""

import numpy as np
from typing import Dict, Set, Tuple, Optional, List
from collections import defaultdict
import hashlib
import cmath


class QuantumSieve:
    """
    Unified sieve based on quantum mechanics.

    Key insight: Superposition + thermal floor are the same physics.

    - Wave function: Psi(state, action) = complex amplitude
    - Zero-point energy: |Psi_min|^2 = hbar/2 (normalized)
    - Born rule: P(action|state) = |Psi(state,action)|^2 / sum|Psi|^2
    - Collapse: Evidence modifies amplitudes through interference
    - Decoherence: Heat bath couples to environment

    CRITICAL: Uses SPATIAL FEATURES (pixel positions) not just state hashes.
    This allows generalization across similar states.
    """

    # The fundamental constant: minimum amplitude (sqrt of zero-point energy)
    # In our units: E_0 = |psi_0|^2 = 0.01 per mode
    ZERO_POINT_AMPLITUDE = 0.1

    def __init__(self):
        # === SPATIAL COUPLING (the key to generalization) ===
        # pixel_action_coupling[pixel_id][action] = complex amplitude
        # This is how the sieve learns "ball here -> move this way"
        self.pixel_action_coupling: Dict[str, Dict[int, complex]] = defaultdict(dict)

        # Wave function: Psi[state_id][action] = complex amplitude
        # Starts in superposition (all equal)
        self.wave_function: Dict[str, Dict[int, complex]] = defaultdict(dict)

        # Global action amplitudes (marginalized over states)
        self.action_amplitudes: Dict[int, complex] = {}

        # Currently active pixels (for spatial coupling)
        self._active_pixels: Set[str] = set()

        # State history for credit assignment
        self._state_trace: List[str] = []
        self._action_trace: List[int] = []
        self._pixel_trace: List[Set[str]] = []  # Track which pixels were active

        # Heat bath (absorbed energy from collapses/mismatches)
        self.heat_bath: float = 1.0

        # Statistics
        self.frame_num: int = 0
        self._game_lengths: List[int] = []
        self._current_game_length: int = 0

        # Coupling graph: state-action -> subsequent states
        # This tracks what happens AFTER taking actions (for learning)
        self.transitions: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _hash(self, data) -> str:
        """Deterministic hash for state identity."""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, bytes):
            data = str(data).encode()
        return hashlib.md5(data).hexdigest()[:12]

    def _ensure_superposition(self, state_id: str, num_actions: int):
        """
        Ensure state has proper superposition initialized.

        Key physics: All actions start with EQUAL amplitude (symmetry).
        The zero-point amplitude ensures no action ever has zero probability.
        """
        if state_id not in self.wave_function:
            # Initialize in symmetric superposition
            # Equal amplitudes = equal probabilities = no bias
            for a in range(num_actions):
                # Start at sqrt(1/N) for normalization, but above zero-point
                amp = max(self.ZERO_POINT_AMPLITUDE, 1.0 / np.sqrt(num_actions))
                # Random phase to break symmetry naturally
                phase = np.random.uniform(0, 2 * np.pi)
                self.wave_function[state_id][a] = amp * cmath.exp(1j * phase)

    def _get_probabilities(self, state_id: str, num_actions: int) -> np.ndarray:
        """
        Compute Born rule probabilities.

        P(a|s) = |Psi(s,a)|^2 / sum_a' |Psi(s,a')|^2
        """
        self._ensure_superposition(state_id, num_actions)

        # Compute |amplitude|^2 for each action
        probs = np.zeros(num_actions)
        for a in range(num_actions):
            amp = self.wave_function[state_id].get(a, self.ZERO_POINT_AMPLITUDE)
            probs[a] = abs(amp) ** 2

        # Ensure zero-point energy floor
        probs = np.maximum(probs, self.ZERO_POINT_AMPLITUDE ** 2)

        # Normalize (Born rule)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(num_actions) / num_actions

        return probs

    def _get_pixel_id(self, y: int, x: int) -> str:
        """Get unique ID for pixel position."""
        return f"px_{y}_{x}"

    def _ensure_pixel_superposition(self, pixel_id: str, num_actions: int):
        """Ensure pixel has superposition initialized."""
        if pixel_id not in self.pixel_action_coupling:
            for a in range(num_actions):
                amp = self.ZERO_POINT_AMPLITUDE
                phase = np.random.uniform(0, 2 * np.pi)
                self.pixel_action_coupling[pixel_id][a] = amp * cmath.exp(1j * phase)

    def observe(self, frame: np.ndarray, action: int, frame_num: int, num_actions: int = 3):
        """
        Observe state-action pair. This is where learning happens.

        The wave function evolves based on:
        1. Constructive interference: action taken gains amplitude
        2. Coherence: if this action led to survival before, boost
        3. Decoherence: mismatch energy goes to heat bath

        CRITICAL: Builds PIXEL-ACTION couplings for spatial generalization.
        """
        self.frame_num = frame_num
        self._current_game_length += 1

        # Normalize frame
        if len(frame.shape) > 1:
            frame_norm = frame.astype(float)
            if frame_norm.max() > 0:
                frame_norm = frame_norm / frame_norm.max()
        else:
            frame_norm = frame.astype(float)

        # === EXTRACT ACTIVE PIXELS ===
        self._active_pixels = set()
        if len(frame_norm.shape) >= 2:
            h, w = frame_norm.shape[:2]
            for y in range(h):
                for x in range(w):
                    val = frame_norm[y, x] if len(frame_norm.shape) == 2 else frame_norm[y, x].mean()
                    if val > 0.1:  # Active pixel
                        pixel_id = self._get_pixel_id(y, x)
                        self._active_pixels.add(pixel_id)
                        self._ensure_pixel_superposition(pixel_id, num_actions)

        # Hash the state (for state-level tracking)
        state_id = self._hash(frame_norm)
        self._ensure_superposition(state_id, num_actions)

        # Record observation
        if self._state_trace:
            prev_state = self._state_trace[-1]
            prev_action = self._action_trace[-1]
            self.transitions[(prev_state, prev_action)][state_id] += 1.0

        self._state_trace.append(state_id)
        self._action_trace.append(action)
        self._pixel_trace.append(self._active_pixels.copy())

        # Limit trace length
        max_trace = 50
        if len(self._state_trace) > max_trace:
            self._state_trace = self._state_trace[-max_trace:]
            self._action_trace = self._action_trace[-max_trace:]
            self._pixel_trace = self._pixel_trace[-max_trace:]

        # === PIXEL-ACTION COUPLING UPDATE ===
        # This is the key: build spatial associations
        survival_factor = 1.0 + 0.02 * np.log1p(self._current_game_length)
        boost = 0.01 * survival_factor

        for pixel_id in self._active_pixels:
            # Boost coupling between this pixel and the chosen action
            old_amp = self.pixel_action_coupling[pixel_id].get(action, self.ZERO_POINT_AMPLITUDE)
            phase_drift = np.random.uniform(-0.05, 0.05)
            new_amp = old_amp * (1 + boost) * cmath.exp(1j * phase_drift)
            # Cap amplitude to prevent explosion
            if abs(new_amp) > 100:
                new_amp = 100 * cmath.exp(1j * cmath.phase(new_amp))
            self.pixel_action_coupling[pixel_id][action] = new_amp

        # === STATE-LEVEL UPDATE ===
        current_amp = self.wave_function[state_id].get(action, self.ZERO_POINT_AMPLITUDE)
        phase_drift = np.random.uniform(-0.1, 0.1)
        new_amp = current_amp * (1 + boost) * cmath.exp(1j * phase_drift)
        self.wave_function[state_id][action] = new_amp

        # Update global action amplitude
        self.action_amplitudes[action] = self.action_amplitudes.get(action, 1.0) + boost * 0.5

        # === DECOHERENCE / DAMPING ===
        damping = 0.998
        # Damping on pixel couplings
        for pixel_id in self._active_pixels:
            for a in list(self.pixel_action_coupling[pixel_id].keys()):
                old_amp = self.pixel_action_coupling[pixel_id][a]
                new_amp = old_amp * damping
                if abs(new_amp) < self.ZERO_POINT_AMPLITUDE:
                    phase = cmath.phase(new_amp) if abs(new_amp) > 0 else 0
                    new_amp = self.ZERO_POINT_AMPLITUDE * cmath.exp(1j * phase)
                self.pixel_action_coupling[pixel_id][a] = new_amp

    def choose_action(self, num_actions: int = 3) -> int:
        """
        Choose action based on wave function collapse.

        Uses Born rule: P(a) = |Psi(a)|^2

        KEY: Uses PIXEL-ACTION couplings for spatial reasoning.
        Sums up evidence from all currently active pixels.
        """
        # === PIXEL-BASED DECISION ===
        # Sum amplitude contributions from active pixels
        pixel_scores = np.zeros(num_actions, dtype=complex)

        if self._active_pixels:
            for pixel_id in self._active_pixels:
                if pixel_id in self.pixel_action_coupling:
                    for a in range(num_actions):
                        amp = self.pixel_action_coupling[pixel_id].get(a, self.ZERO_POINT_AMPLITUDE)
                        pixel_scores[a] += amp

        # Add global action bias
        for a in range(num_actions):
            global_amp = self.action_amplitudes.get(a, 1.0)
            pixel_scores[a] += global_amp * 0.1

        # Born rule: P(a) = |sum of amplitudes|^2
        probs = np.array([abs(amp) ** 2 for amp in pixel_scores])

        # Ensure zero-point floor
        probs = np.maximum(probs, self.ZERO_POINT_AMPLITUDE ** 2)

        # Normalize
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(num_actions) / num_actions

        # Ensure valid distribution
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum()

        return np.random.choice(num_actions, p=probs)

    def signal_game_end(self, game_length: int = 0, death: bool = True):
        """
        Signal end of episode.

        If death: negative interference on recent actions
        If not death: positive reinforcement

        The key physics: death = decoherence event
        The wave function partially collapses, and "bad" paths lose amplitude.

        CRITICAL: Affects PIXEL-ACTION couplings, not just state-level.
        """
        if game_length == 0:
            game_length = self._current_game_length

        self._game_lengths.append(game_length)

        if death and self._state_trace and self._action_trace:
            # === DEATH = DECOHERENCE EVENT ===
            # Recent pixel-action pairs get amplitude REDUCTION
            # This is the anthropic selection: paths that lead to death lose amplitude

            trace_len = len(self._state_trace)

            # Exponential decay of penalty with time before death
            for i in range(min(15, trace_len)):
                idx = trace_len - 1 - i
                if idx < 0:
                    break

                action = self._action_trace[idx]

                # Penalty decays exponentially from death
                penalty = 0.3 * np.exp(-i / 4.0)  # tau = 4 steps

                # === PIXEL-LEVEL PENALTY (key for spatial learning) ===
                if idx < len(self._pixel_trace):
                    pixels = self._pixel_trace[idx]
                    for pixel_id in pixels:
                        if pixel_id in self.pixel_action_coupling:
                            old_amp = self.pixel_action_coupling[pixel_id].get(action, self.ZERO_POINT_AMPLITUDE)
                            new_mag = max(self.ZERO_POINT_AMPLITUDE, abs(old_amp) * (1 - penalty))
                            phase = cmath.phase(old_amp) if abs(old_amp) > 0 else 0
                            self.pixel_action_coupling[pixel_id][action] = new_mag * cmath.exp(1j * phase)

                # State-level penalty
                state_id = self._state_trace[idx]
                if state_id in self.wave_function and action in self.wave_function[state_id]:
                    old_amp = self.wave_function[state_id][action]
                    new_mag = max(self.ZERO_POINT_AMPLITUDE, abs(old_amp) * (1 - penalty))
                    phase = cmath.phase(old_amp) if abs(old_amp) > 0 else 0
                    self.wave_function[state_id][action] = new_mag * cmath.exp(1j * phase)

                # Global action penalty (mild)
                old_global = self.action_amplitudes.get(action, 1.0)
                self.action_amplitudes[action] = max(0.1, old_global * (1 - penalty * 0.2))

            # Energy from collapse goes to heat bath
            self.heat_bath += 0.1 * trace_len

        else:
            # Survival / non-death: mild positive reinforcement on pixels
            for i in range(len(self._action_trace)):
                action = self._action_trace[i]
                boost = 0.01 * (i + 1) / len(self._action_trace)

                if i < len(self._pixel_trace):
                    pixels = self._pixel_trace[i]
                    for pixel_id in pixels:
                        if pixel_id in self.pixel_action_coupling:
                            old_amp = self.pixel_action_coupling[pixel_id].get(action, self.ZERO_POINT_AMPLITUDE)
                            new_mag = abs(old_amp) * (1 + boost)
                            phase = cmath.phase(old_amp) if abs(old_amp) > 0 else 0
                            self.pixel_action_coupling[pixel_id][action] = new_mag * cmath.exp(1j * phase)

        # Clear trace for next episode
        self._state_trace = []
        self._action_trace = []
        self._pixel_trace = []
        self._current_game_length = 0

        # Heat bath slowly cools (energy radiates)
        self.heat_bath *= 0.99

    def get_stats(self) -> Dict:
        """Get statistics about the sieve state."""
        total_states = len(self.wave_function)
        total_pixels = len(self.pixel_action_coupling)

        # Compute average pixel-action coupling per action
        pixel_action_amps = {0: [], 1: [], 2: []}
        for pixel_id, actions in self.pixel_action_coupling.items():
            for a, amp in actions.items():
                if a in pixel_action_amps:
                    pixel_action_amps[a].append(abs(amp))

        avg_pixel_amps = {a: np.mean(amps) if amps else 0 for a, amps in pixel_action_amps.items()}

        return {
            'total_states': total_states,
            'total_pixels': total_pixels,
            'heat_bath': self.heat_bath,
            'pixel_action_avg_amps': avg_pixel_amps,
            'global_action_amps': dict(self.action_amplitudes),
            'transitions_tracked': len(self.transitions),
            'avg_game_length': np.mean(self._game_lengths[-100:]) if self._game_lengths else 0,
        }


# =============================================================================
# PONG TEST
# =============================================================================

def test_quantum_sieve_pong():
    """
    Test unified quantum sieve on Pong.

    Same test setup as pure_wave_sieve for comparison.
    """
    print("=" * 70)
    print("UNIFIED QUANTUM SIEVE - Pong Test")
    print("Superposition + Thermal Floor = One Mechanism")
    print("=" * 70)

    FRAME_SIZE = 21
    sieve = QuantumSieve()

    # Game tracking
    game_lengths = []
    hits = 0
    misses = 0

    # Pong state
    ball_x, ball_y = 10.5, 5.0
    ball_dx, ball_dy = 0.5, 0.5
    paddle_x = 10.5
    current_game_length = 0

    # Progress milestones
    milestones = [1000, 5000, 10000, 20000, 30000, 40000, 50000]

    TOTAL_FRAMES = 50000

    for frame_num in range(TOTAL_FRAMES):
        # Create frame
        pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

        # Ball (2x2 block)
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
        if frame_num < 200:
            # Initial exploration phase
            action = np.random.randint(0, 3)
        else:
            action = sieve.choose_action(num_actions=3)

        # Observe
        sieve.observe(pixels, action, frame_num, num_actions=3)

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
                sieve.signal_game_end(current_game_length, death=True)
                current_game_length = 0

                ball_y = 5
                ball_x = np.random.uniform(5, FRAME_SIZE - 5)
                ball_dx = np.random.choice([-0.5, 0.5])
                ball_dy = 0.5

        # Progress report
        if frame_num + 1 in milestones:
            total_games = hits + misses
            hit_rate = hits / total_games if total_games > 0 else 0
            recent = game_lengths[-20:] if len(game_lengths) >= 20 else game_lengths
            avg_len = np.mean(recent) if recent else 0

            stats = sieve.get_stats()

            print(f"\nFrame {frame_num + 1}:")
            print(f"  Games: {total_games}, Hit rate: {hit_rate:.1%}")
            print(f"  Recent avg game length: {avg_len:.1f}")
            print(f"  Pixels tracked: {stats['total_pixels']}, States: {stats['total_states']}")
            print(f"  Heat bath: {stats['heat_bath']:.2f}")
            print(f"  Pixel-action amps: {stats['pixel_action_avg_amps']}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_games = hits + misses
    print(f"\nTotal frames: {TOTAL_FRAMES}")
    print(f"Total games: {total_games}")
    print(f"Hits: {hits} ({100*hits/total_games:.1f}%)")
    print(f"Misses: {misses}")

    if len(game_lengths) >= 20:
        first_20 = np.mean(game_lengths[:20])
        last_20 = np.mean(game_lengths[-20:])
        print(f"\nGame length evolution:")
        print(f"  First 20 games: {first_20:.1f} frames")
        print(f"  Last 20 games: {last_20:.1f} frames")
        print(f"  Improvement: {(last_20/first_20 - 1)*100:+.1f}%")

    # Comparison baselines
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("Random baseline: ~33% hit rate")
    print("Pure wave sieve: 45.2% hit rate, +42.9% game length")
    print(f"Quantum sieve:   {100*hits/total_games:.1f}% hit rate, ", end="")
    if len(game_lengths) >= 20:
        improvement = (np.mean(game_lengths[-20:]) / np.mean(game_lengths[:20]) - 1) * 100
        print(f"{improvement:+.1f}% game length")

    return sieve, game_lengths, hits, misses


if __name__ == "__main__":
    test_quantum_sieve_pong()
