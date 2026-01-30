"""
WAVE SIEVE v5 - Unified Resonant Cavity
========================================

A resonant cavity that learns by accumulating standing waves.
One free parameter: PSI_0 (zero-point amplitude).
Everything else derived from PSI_0 and N (number of modes).

THERMODYNAMICS:
===============
Open resonant cavity coupled to environment:

1. ENERGY PUMP: Each observation injects energy (broadband photons).
2. GLOBAL DECOHERENCE: All modes decay toward zero-point each frame.
3. SYMMETRIC SIGNALS: Success = phase 0, death = phase pi.
4. ENERGY = total_energy. Heat = total - mode_energy (derived).
5. RADIATIVE COOLING: Total energy decays toward equilibrium.

COUPLING (emergent cavities):
=============================
Modes couple in TWO ways, both in the same wave medium:
- TEMPORAL: mode at t-1 couples to mode at t (sequence memory).
- SPATIAL: modes observed simultaneously couple (co-occurrence).
Both are complex amplitudes in the same coupling dictionary.
Coupling IS the cavity — when a cluster of modes develops strong
mutual coupling with correlated phases, that IS a resonant
sub-structure. No separate objects needed. The cavity is the wave.

Like atoms forming molecules: spatial proximity creates coupling
between modes, and the correlated wavefunction IS the bond.

RESONANCE QUALITY (optimal stopping):
======================================
Q = contrast * selectivity * log(1 + signal_strength).
Snapshot/restore at peak Q. No oracle needed.

CONSTANTS (all from PSI_0 and N):
- Energy quantum per observation: n_observed * PSI_0^2
- Global decoherence: PSI_0^2 / N per mode per frame
- Pump interference amp: PSI_0
- Temporal coupling: PSI_0 / N
- Spatial coupling: PSI_0 / N (same strength — same physics)
- Locality: N (query), 1 (neighbor), 1/N (distant)
- Credit: 1/(1+r)^2
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import cmath


class WaveSieve:
    """Physics reservoir. One free parameter: PSI_0."""

    PSI_0 = 0.1

    def __init__(self):
        self.amplitude: Dict[Any, Dict[int, complex]] = defaultdict(dict)
        # Unified coupling: both temporal AND spatial, same medium.
        # coupling[mode_a][mode_b] = complex amplitude of their correlation.
        self.coupling: Dict[Any, Dict[Any, complex]] = defaultdict(
            lambda: defaultdict(complex)
        )
        self._prev_data: Optional[Dict[Any, Any]] = None
        self.total_energy: float = self.PSI_0 ** 2
        self._trace: List[Tuple[Dict[Any, Any], int]] = []
        self.frame: int = 0

        # Resonance quality tracking (optimal stopping)
        self._q_history: List[float] = []
        self._best_q: float = 0.0
        self._best_q_frame: int = 0
        self._best_snapshot: Optional[Dict] = None

    # === DERIVED QUANTITIES ===

    def _n_modes(self) -> int:
        return max(1, sum(len(acts) for acts in self.amplitude.values()))

    def _mode_energy(self) -> float:
        return sum(
            abs(a) ** 2
            for acts in self.amplitude.values()
            for a in acts.values()
        )

    def _heat(self) -> float:
        return max(0, self.total_energy - self._mode_energy())

    # === RESONANCE QUALITY (optimal stopping) ===

    def _signal_strength(self) -> float:
        """Total signal energy above vacuum = sum(|a|^2 - PSI_0^2)."""
        signal = 0.0
        floor = self.PSI_0 ** 2
        for actions in self.amplitude.values():
            for a in actions.values():
                excess = abs(a) ** 2 - floor
                if excess > 0:
                    signal += excess
        return signal

    def _mode_contrast(self) -> float:
        """Fringe visibility: (max - min) / (max + min) intensity per mode.
        Only counts modes with signal above the floor."""
        if not self.amplitude:
            return 0.0
        floor = self.PSI_0 ** 2
        contrasts = []
        for actions in self.amplitude.values():
            if not actions:
                continue
            intensities = [abs(a) ** 2 for a in actions.values()]
            max_i = max(intensities)
            if max_i <= floor * 1.5:
                continue
            min_i = min(intensities)
            if max_i + min_i > 0:
                contrasts.append((max_i - min_i) / (max_i + min_i))
        if not contrasts:
            return 0.0
        return sum(contrasts) / len(contrasts)

    def _mode_selectivity(self) -> float:
        """Fraction of modes carrying signal above the floor."""
        if not self.amplitude:
            return 0.0
        floor = self.PSI_0 ** 2
        total = 0
        differentiated = 0
        for actions in self.amplitude.values():
            if not actions:
                continue
            total += 1
            intensities = [abs(a) ** 2 for a in actions.values()]
            max_i = max(intensities)
            min_i = min(intensities)
            if max_i > floor * 1.5 and (max_i - min_i) > floor:
                differentiated += 1
        if total == 0:
            return 0.0
        return differentiated / total

    def resonance_quality(self) -> float:
        """Q = contrast * selectivity * log(1 + signal_strength)."""
        contrast = self._mode_contrast()
        selectivity = self._mode_selectivity()
        signal = np.log1p(self._signal_strength() / self.PSI_0 ** 2)
        return contrast * selectivity * signal

    def _update_quality(self):
        """Track Q and snapshot at peak. Sampled every N frames."""
        N = self._n_modes()
        if self.frame % max(1, N) != 0:
            return
        q = self.resonance_quality()
        self._q_history.append(q)
        if q > self._best_q * 1.01 + self.PSI_0 ** 2:
            self._best_q = q
            self._best_q_frame = self.frame
            self._best_snapshot = self._snapshot()

    def _snapshot(self) -> Dict:
        """Capture cavity state for restoration."""
        return {
            'amplitude': {k: dict(v) for k, v in self.amplitude.items()},
            'coupling': {k: dict(v) for k, v in self.coupling.items()},
            'total_energy': self.total_energy,
            'frame': self.frame,
        }

    def restore_best(self) -> int:
        """Restore to peak resonance quality state."""
        if self._best_snapshot is None:
            return self.frame
        snap = self._best_snapshot
        self.amplitude = defaultdict(dict)
        for k, v in snap['amplitude'].items():
            self.amplitude[k] = dict(v)
        self.coupling = defaultdict(lambda: defaultdict(complex))
        for k, v in snap['coupling'].items():
            self.coupling[k] = defaultdict(complex)
            self.coupling[k].update(v)
        self.total_energy = snap['total_energy']
        return snap['frame']

    def q_trend(self, window: int = 100) -> float:
        """Recent trend in Q. Positive = improving, negative = degrading."""
        if len(self._q_history) < window + 1:
            return 0.0
        recent = self._q_history[-window:]
        old = self._q_history[-(window + window // 2):-window // 2]
        if not old:
            return 0.0
        return (sum(recent) / len(recent)) - (sum(old) / len(old))

    # === INPUT ===

    def _normalize_input(self, data) -> Dict[Any, Any]:
        if isinstance(data, dict):
            return data
        elif isinstance(data, np.ndarray):
            flat = data.flatten()
            return {i: flat[i].item() for i in range(len(flat))}
        elif isinstance(data, (list, tuple)):
            return {i: v for i, v in enumerate(data)}
        elif isinstance(data, (bytes, str)):
            return {i: v for i, v in enumerate(data)}
        else:
            return {0: data}

    # === MODES ===

    def _mode_key(self, position: Any, value: Any) -> tuple:
        if isinstance(position, tuple):
            return (*position, value)
        return (position, value)

    def _ensure_mode(self, key: Any, num_actions: int):
        if key not in self.amplitude:
            for a in range(num_actions):
                phase = np.random.uniform(0, 2 * np.pi)
                self.amplitude[key][a] = self.PSI_0 * cmath.exp(1j * phase)
            self.total_energy += num_actions * self.PSI_0 ** 2

    # === OBSERVE ===

    def observe(self, data, action: int, num_actions: int):
        self.frame += 1
        config = self._normalize_input(data)

        for pos, val in config.items():
            self._ensure_mode(self._mode_key(pos, val), num_actions)

        N = self._n_modes()

        # Energy pump: broadband illumination
        n_obs = len(config)
        pump_energy = n_obs * self.PSI_0 ** 2
        self.total_energy += pump_energy

        # Global decoherence: all modes decay toward zero-point
        decay = self.PSI_0 ** 2 / N
        for mode_key in list(self.amplitude.keys()):
            self._decohere_mode(mode_key, decay)

        # Constructive interference: pump photon excites observed modes
        amp = self.PSI_0
        for pos, val in config.items():
            mode = self._mode_key(pos, val)
            old = self.amplitude[mode].get(action, self.PSI_0)
            phase = cmath.phase(old) if abs(old) > 0 else 0
            self.amplitude[mode][action] = old + amp * cmath.exp(1j * phase)

        # Radiative cooling
        radiation_rate = self.PSI_0 ** 2 / N
        self.total_energy *= (1 - radiation_rate)
        min_energy = N * self.PSI_0 ** 2
        self.total_energy = max(min_energy, self.total_energy)

        # === COUPLING: temporal + spatial in the same medium ===
        observed_modes = []
        for pos, val in config.items():
            observed_modes.append(self._mode_key(pos, val))

        tc = self.PSI_0 / N

        # Temporal coupling: mode at t-1 <-> mode at t (same position)
        if self._prev_data is not None:
            for pos, val in config.items():
                if pos in self._prev_data:
                    prev_val = self._prev_data[pos]
                    now_key = self._mode_key(pos, val)
                    prev_key = self._mode_key(pos, prev_val)
                    self._couple(now_key, prev_key, tc)

        # Spatial coupling: modes observed simultaneously couple.
        # Stochastic sampling — sqrt(n_obs) pairs per frame.
        # This IS cavity formation: correlated modes = bound state.
        # Same strength as temporal — same physics, same medium.
        if len(observed_modes) >= 2:
            n_samples = max(1, int(np.sqrt(len(observed_modes))))
            for _ in range(n_samples):
                i, j = np.random.choice(len(observed_modes), size=2,
                                        replace=False)
                self._couple(observed_modes[i], observed_modes[j], tc)

        self._trace.append((config, action))
        self._prev_data = dict(config)

        # Track resonance quality
        self._update_quality()

    def _couple(self, key_a: Any, key_b: Any, strength: float):
        """Symmetric coupling between two modes. Same operation for
        temporal and spatial — the wave medium doesn't distinguish."""
        old = self.coupling[key_a][key_b]
        p = cmath.phase(old) if abs(old) > 0 else np.random.uniform(
            0, 2 * np.pi)
        self.coupling[key_a][key_b] = old + strength * cmath.exp(1j * p)
        old_b = self.coupling[key_b][key_a]
        p_b = cmath.phase(old_b) if abs(old_b) > 0 else p
        self.coupling[key_b][key_a] = old_b + strength * cmath.exp(1j * p_b)

    def _decohere_mode(self, key: Any, rate: float):
        for a in list(self.amplitude[key].keys()):
            old = self.amplitude[key][a]
            new = old * (1 - rate)
            if abs(new) < self.PSI_0:
                p = cmath.phase(old) if abs(old) > 0 else 0
                new = self.PSI_0 * cmath.exp(1j * p)
            self.amplitude[key][a] = new

    # === CHOOSE ACTION ===

    def choose_action(self, data, num_actions: int,
                      neighbor_fn=None, query_pos=None) -> int:
        config = self._normalize_input(data)
        total_amp = np.zeros(num_actions, dtype=complex)
        N = max(1, len(config))

        neighbor_set = set()
        if query_pos is not None and neighbor_fn is not None:
            neighbor_set = set(neighbor_fn(query_pos))

        # Direct mode contributions (Green's function locality)
        for pos, val in config.items():
            mode = self._mode_key(pos, val)
            self._ensure_mode(mode, num_actions)

            if query_pos is not None and pos == query_pos:
                weight = float(N)
            elif pos in neighbor_set:
                weight = 1.0
            elif query_pos is not None:
                weight = 1.0 / N
            else:
                weight = 1.0

            for a in range(num_actions):
                total_amp[a] += weight * self.amplitude[mode].get(
                    a, self.PSI_0)

        # Coupling contributions: both temporal AND spatial.
        # Each observed mode pulls in amplitudes from all coupled modes.
        # This is how "cavities" contribute — through the coupling
        # network, not through separate objects.
        tc = self.PSI_0 / self._n_modes()
        for pos, val in config.items():
            mode = self._mode_key(pos, val)
            if mode in self.coupling:
                for coupled_mode, coupling_amp in self.coupling[mode].items():
                    if abs(coupling_amp) > 0 and coupled_mode in self.amplitude:
                        for a in range(num_actions):
                            partner_amp = self.amplitude[coupled_mode].get(
                                a, self.PSI_0)
                            total_amp[a] += coupling_amp * partner_amp * tc

        probs = np.array([abs(a) ** 2 for a in total_amp])
        probs = np.maximum(probs, self.PSI_0 ** 2)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_actions) / num_actions

        return np.random.choice(num_actions, p=probs)

    # === SIGNALS (SYMMETRIC) ===

    def _signal(self, phase_shift: float):
        """Success = phase 0, death = phase pi. Same operation, different phase.
        Credit = 1/(1+r)^2. Signal carries its own energy."""
        if not self._trace:
            return

        signal_energy = len(self._trace) * self.PSI_0 ** 2
        self.total_energy += signal_energy
        amp = self.PSI_0
        max_a = np.sqrt(max(self.PSI_0 ** 2, self.total_energy))

        for i, (config, action) in enumerate(reversed(self._trace)):
            credit = 1.0 / (1 + i) ** 2

            for pos, val in config.items():
                mode = self._mode_key(pos, val)
                if mode not in self.amplitude:
                    continue

                old = self.amplitude[mode].get(action, self.PSI_0)
                old_phase = cmath.phase(old) if abs(old) > 0 else 0

                signal = credit * amp * cmath.exp(
                    1j * (old_phase + phase_shift))
                new = old + signal

                if abs(new) < self.PSI_0:
                    new = self.PSI_0 * cmath.exp(1j * cmath.phase(new))
                if abs(new) > max_a:
                    new = max_a * cmath.exp(1j * cmath.phase(new))

                self.amplitude[mode][action] = new

        self._trace = []

    def signal_death(self):
        """Destructive interference = phase pi. THE NOT WAVE."""
        self._signal(np.pi)

    def signal_success(self):
        """Constructive interference = phase 0."""
        self._signal(0.0)

    # === ANNEALING ===

    def anneal(self, temperature: float = 1.0):
        self.total_energy += len(self.amplitude) * self.PSI_0 ** 2 * temperature

    def cool(self, rate: float = None):
        if rate is None:
            rate = self.PSI_0 ** 2 / self._n_modes()
        heat = self._heat()
        self.total_energy -= heat * rate

    # === EPISODE / STATS ===

    def reset_episode(self):
        self._prev_data = None
        self._trace = []

    def get_stats(self) -> Dict:
        me = self._mode_energy()
        ce = sum(
            abs(c) ** 2
            for targets in self.coupling.values()
            for c in targets.values()
        )
        q = self.resonance_quality()
        return {
            'total_energy': self.total_energy,
            'mode_energy': me,
            'heat_bath': self._heat(),
            'coupling_energy': ce,
            'n_modes': len(self.amplitude),
            'n_mode_actions': self._n_modes(),
            'n_couplings': sum(len(t) for t in self.coupling.values()),
            'resonance_q': q,
            'best_q': self._best_q,
            'best_q_frame': self._best_q_frame,
            'q_trend': self.q_trend(),
        }
