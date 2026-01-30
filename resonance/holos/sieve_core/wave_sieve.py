"""
WAVE SIEVE v3 - Physics Reservoir
==================================

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
5. RADIATIVE COOLING: Heat slowly leaves the system.

CONSTANTS (all from PSI_0 and N):
- Energy quantum per observation: n_observed * PSI_0^2
- Global decoherence: PSI_0^2 / N per mode per frame
- Interference amp: sqrt(heat / N)
- Temporal coupling: PSI_0 / N
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
        self.temporal: Dict[Any, Dict[Any, complex]] = defaultdict(
            lambda: defaultdict(complex)
        )
        self._prev_data: Optional[Dict[Any, Any]] = None
        self.total_energy: float = self.PSI_0 ** 2
        self._trace: List[Tuple[Dict[Any, Any], int]] = []
        self.frame: int = 0

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

        # Energy pump: broadband illumination.
        # The pump energy goes directly into the observed modes -
        # this IS the photon being absorbed by the detector.
        n_obs = len(config)
        pump_energy = n_obs * self.PSI_0 ** 2
        self.total_energy += pump_energy

        # Global decoherence: all modes decay toward zero-point.
        # This frees energy from modes, increasing heat.
        decay = self.PSI_0 ** 2 / N
        for mode_key in list(self.amplitude.keys()):
            self._decohere_mode(mode_key, decay)

        # Constructive interference: pump photon absorbed by observed modes.
        # Each observed mode gets pump_energy / n_obs = PSI_0^2 worth.
        # Amplitude boost = PSI_0 (sqrt of one photon's energy).
        # This is the incoming wave directly exciting the resonant modes.
        amp = self.PSI_0  # One photon per observed mode

        for pos, val in config.items():
            mode = self._mode_key(pos, val)
            old = self.amplitude[mode].get(action, self.PSI_0)
            phase = cmath.phase(old) if abs(old) > 0 else 0
            self.amplitude[mode][action] = old + amp * cmath.exp(1j * phase)

        # Radiative cooling: total energy decays toward equilibrium.
        # Rate = PSI_0^2 / N. This acts on TOTAL energy, not just heat.
        # The system radiates from its walls, losing structured energy too.
        # Equilibrium: pump (n_obs * PSI_0^2/frame) = radiation (E * PSI_0^2/N)
        # => E_eq = n_obs * N (steady state total energy)
        radiation_rate = self.PSI_0 ** 2 / N
        self.total_energy *= (1 - radiation_rate)
        # Floor at vacuum energy
        min_energy = N * self.PSI_0 ** 2
        self.total_energy = max(min_energy, self.total_energy)

        # Temporal coupling
        if self._prev_data is not None:
            tc = self.PSI_0 / N
            for pos, val in config.items():
                if pos in self._prev_data:
                    prev_val = self._prev_data[pos]
                    now_key = self._mode_key(pos, val)
                    prev_key = self._mode_key(pos, prev_val)
                    old = self.temporal[now_key][prev_key]
                    p = cmath.phase(old) if abs(old) > 0 else np.random.uniform(0, 2 * np.pi)
                    self.temporal[now_key][prev_key] = old + tc * cmath.exp(1j * p)
                    old_b = self.temporal[prev_key][now_key]
                    p_b = cmath.phase(old_b) if abs(old_b) > 0 else p
                    self.temporal[prev_key][now_key] = old_b + tc * cmath.exp(1j * p_b)

        self._trace.append((config, action))
        self._prev_data = dict(config)

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
                total_amp[a] += weight * self.amplitude[mode].get(a, self.PSI_0)

        if self._prev_data is not None:
            tc = self.PSI_0 / self._n_modes()
            for pos, val in config.items():
                mode = self._mode_key(pos, val)
                if mode in self.temporal:
                    for prev_mode, coupling in self.temporal[mode].items():
                        if abs(coupling) > 0 and prev_mode in self.amplitude:
                            for a in range(num_actions):
                                prev_amp = self.amplitude[prev_mode].get(a, self.PSI_0)
                                total_amp[a] += coupling * prev_amp * tc

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
        """
        Success = phase 0 (constructive), death = phase pi (destructive).
        Same operation, different phase. Credit = 1/(1+r)^2.

        The signal IS an observation from the environment - it injects
        energy just like observe() does. Death/success are boundary
        reflections that carry energy back into the cavity.
        """
        if not self._trace:
            return

        N = self._n_modes()

        # The signal is an energy-carrying event from the boundary.
        # It injects energy proportional to the trace length
        # (the boundary reflected the whole path).
        signal_energy = len(self._trace) * self.PSI_0 ** 2
        self.total_energy += signal_energy
        amp = self.PSI_0  # One photon per mode
        max_a = np.sqrt(max(self.PSI_0 ** 2, self.total_energy))

        for i, (config, action) in enumerate(reversed(self._trace)):
            credit = 1.0 / (1 + i) ** 2

            for pos, val in config.items():
                mode = self._mode_key(pos, val)
                if mode not in self.amplitude:
                    continue

                old = self.amplitude[mode].get(action, self.PSI_0)
                old_phase = cmath.phase(old) if abs(old) > 0 else 0

                signal = credit * amp * cmath.exp(1j * (old_phase + phase_shift))
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
        te = sum(
            abs(c) ** 2
            for targets in self.temporal.values()
            for c in targets.values()
        )
        return {
            'total_energy': self.total_energy,
            'mode_energy': me,
            'heat_bath': self._heat(),
            'temporal_energy': te,
            'n_modes': len(self.amplitude),
            'n_mode_actions': self._n_modes(),
            'n_temporal': sum(len(t) for t in self.temporal.values()),
        }
