"""
WAVE SIEVE v4 - Fractal Resonant Cavity
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
5. RADIATIVE COOLING: Heat slowly leaves the system.

RESONANCE QUALITY (optimal stopping):
======================================
The cavity measures its own quality factor Q from internal observables:
- Mode contrast: max|a|^2 / mean|a|^2 per mode (action differentiation)
- Phase coherence: alignment of winning-action phases across modes
- Energy partition: mode_energy / total_energy (structure vs thermal)
Q = contrast * coherence * partition. No oracle needed.
Snapshot/restore at peak Q to capture optimal learned state.

FRACTAL CAVITIES:
=================
Positions that co-occur form natural sub-cavities. When correlation
energy between a position cluster exceeds PSI_0^2, a child cavity
spawns. Parent energy flows to children through shared boundary modes.
Each child is itself a WaveSieve — fractal resonance all the way down.

CONSTANTS (all from PSI_0 and N):
- Energy quantum per observation: n_observed * PSI_0^2
- Global decoherence: PSI_0^2 / N per mode per frame
- Pump interference amp: PSI_0
- Temporal coupling: PSI_0 / N
- Locality: N (query), 1 (neighbor), 1/N (distant)
- Credit: 1/(1+r)^2
- Cavity spawn threshold: PSI_0^2 (correlation energy)
- Inter-cavity coupling: PSI_0 / depth
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import cmath


class WaveSieve:
    """Physics reservoir. One free parameter: PSI_0."""

    PSI_0 = 0.1

    def __init__(self, depth: int = 0):
        self.depth = depth
        self.amplitude: Dict[Any, Dict[int, complex]] = defaultdict(dict)
        self.temporal: Dict[Any, Dict[Any, complex]] = defaultdict(
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

        # Fractal cavities: position-cluster -> child WaveSieve
        self._cavities: Dict[Any, 'WaveSieve'] = {}
        self._cooccurrence: Dict[Tuple, float] = defaultdict(float)

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
        """How much amplitude has accumulated above the zero-point floor.
        Physics: total signal energy above vacuum = sum(|a|^2 - PSI_0^2)
        for all mode-actions where |a| > PSI_0. This measures how much
        the environment has imprinted on the cavity."""
        signal = 0.0
        floor = self.PSI_0 ** 2
        for actions in self.amplitude.values():
            for a in actions.values():
                excess = abs(a) ** 2 - floor
                if excess > 0:
                    signal += excess
        return signal

    def _mode_contrast(self) -> float:
        """How differentiated are action amplitudes within each mode?
        Physics: for each mode, ratio of (max - min) to (max + min) intensity.
        This is the fringe visibility. 0 = uniform, 1 = fully differentiated.
        Only counts modes that have accumulated signal above the floor."""
        if not self.amplitude:
            return 0.0
        floor = self.PSI_0 ** 2
        contrasts = []
        for mode_key, actions in self.amplitude.items():
            if not actions:
                continue
            intensities = [abs(a) ** 2 for a in actions.values()]
            max_i = max(intensities)
            # Only count modes that have learned something
            if max_i <= floor * 1.5:
                continue
            min_i = min(intensities)
            if max_i + min_i > 0:
                contrasts.append((max_i - min_i) / (max_i + min_i))
        if not contrasts:
            return 0.0
        return sum(contrasts) / len(contrasts)

    def _mode_selectivity(self) -> float:
        """Fraction of modes that have differentiated above the floor.
        Physics: what fraction of the cavity's modes carry signal.
        0 = nothing learned, 1 = all modes carry information."""
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
        """Combined quality factor Q. All from internal observables.
        Q = contrast * selectivity * log(1 + signal_strength).
        - contrast: how strongly modes differentiate between actions
        - selectivity: what fraction of modes carry signal
        - signal_strength: total energy above vacuum (log-scaled)
        Peaks when the cavity has optimal learned representation."""
        contrast = self._mode_contrast()
        selectivity = self._mode_selectivity()
        signal = np.log1p(self._signal_strength() / self.PSI_0 ** 2)
        return contrast * selectivity * signal

    def _update_quality(self):
        """Track Q and snapshot at peak. Sampled every N frames to avoid
        overhead from computing Q and snapshotting every frame."""
        N = self._n_modes()
        # Sample rate: every N frames (thermodynamic timescale)
        if self.frame % max(1, N) != 0:
            return
        q = self.resonance_quality()
        self._q_history.append(q)
        # Only snapshot if Q improved by at least 1% relative
        if q > self._best_q * 1.01 + self.PSI_0 ** 2:
            self._best_q = q
            self._best_q_frame = self.frame
            self._best_snapshot = self._snapshot()

    def _snapshot(self) -> Dict:
        """Capture cavity state for restoration."""
        return {
            'amplitude': {k: dict(v) for k, v in self.amplitude.items()},
            'temporal': {k: dict(v) for k, v in self.temporal.items()},
            'total_energy': self.total_energy,
            'frame': self.frame,
        }

    def restore_best(self) -> int:
        """Restore the cavity to its peak resonance quality state.
        Returns the frame number of the restored state."""
        if self._best_snapshot is None:
            return self.frame
        snap = self._best_snapshot
        self.amplitude = defaultdict(dict)
        for k, v in snap['amplitude'].items():
            self.amplitude[k] = dict(v)
        self.temporal = defaultdict(lambda: defaultdict(complex))
        for k, v in snap['temporal'].items():
            self.temporal[k] = defaultdict(complex)
            self.temporal[k].update(v)
        self.total_energy = snap['total_energy']
        return snap['frame']

    def q_trend(self, window: int = 100) -> float:
        """Recent trend in Q. Positive = improving, negative = degrading.
        Physics: dQ/dt averaged over window."""
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

        # Fractal cavity coupling: feed sub-cavities
        # Updated every sqrt(N) frames to balance learning with overhead
        if self.frame % max(1, int(np.sqrt(N))) == 0:
            self._update_cavities(config, action, num_actions)

        self._trace.append((config, action))
        self._prev_data = dict(config)

        # Track resonance quality
        self._update_quality()

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

        # Fractal cavity contributions
        total_amp += self._cavity_contribution(config, num_actions)

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

        # Propagate signal to child cavities (boundary reflection)
        for child in self._cavities.values():
            if child._trace:
                child._signal(phase_shift)

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

    # === FRACTAL CAVITIES ===

    def _update_cavities(self, config: Dict, action: int, num_actions: int):
        """Dynamically create and feed sub-cavities based on input geometry.

        Physics: when positions co-occur repeatedly, their correlation
        energy builds. When it exceeds the spawn threshold, a resonant
        sub-cavity forms at that cluster. The parent feeds the child
        with a coupling strength of PSI_0 / (depth + 2).

        Child cavities see a LOCAL view: only the positions in their
        cluster. This is how the universe creates fractal resonant
        structures from the geometry of its input.

        Efficiency: co-occurrence is sampled stochastically (Monte Carlo)
        rather than computed for all O(N^2) pairs. This is physically
        motivated — not all pair interactions happen every frame.
        """
        if self.depth >= 3:
            return

        positions = list(config.keys())
        if len(positions) < 2:
            return

        N = max(1, len(positions))

        # Stochastic co-occurrence sampling: pick sqrt(N) random pairs.
        # Physics: thermal fluctuations sample the pair space.
        n_samples = max(1, int(np.sqrt(N)))
        deposit = self.PSI_0 ** 2 / N  # Normalized per sample

        for _ in range(n_samples):
            i, j = np.random.choice(len(positions), size=2, replace=False)
            ki, kj = positions[i], positions[j]
            pair = (ki, kj) if hash(ki) <= hash(kj) else (kj, ki)
            self._cooccurrence[pair] += deposit

            # Spawn cavity when correlation energy is significant.
            # Threshold: sqrt(N) * PSI_0^2 — geometric mean between
            # one photon (too easy) and N photons (too hard).
            threshold = np.sqrt(N) * self.PSI_0 ** 2
            if pair not in self._cavities and self._cooccurrence[pair] >= threshold:
                # Cap total cavities at N (energy budget)
                if len(self._cavities) < N:
                    self._cavities[pair] = WaveSieve(depth=self.depth + 1)

        # Feed active sub-cavities with their local view
        for pair, child in self._cavities.items():
            p1, p2 = pair
            if p1 in config and p2 in config:
                local_config = {p1: config[p1], p2: config[p2]}
                child.observe(local_config, action, num_actions)

    def _cavity_contribution(self, config: Dict, num_actions: int) -> np.ndarray:
        """Collect action amplitudes from child cavities.
        Each child contributes its local view weighted by coupling."""
        contrib = np.zeros(num_actions, dtype=complex)
        if not self._cavities:
            return contrib

        coupling = self.PSI_0 / (self.depth + 2)
        for pair, child in self._cavities.items():
            p1, p2 = pair
            if p1 in config and p2 in config:
                local_config = {p1: config[p1], p2: config[p2]}
                child_config = child._normalize_input(local_config)
                child_amp = np.zeros(num_actions, dtype=complex)
                for pos, val in child_config.items():
                    mode = child._mode_key(pos, val)
                    if mode in child.amplitude:
                        for a in range(num_actions):
                            child_amp[a] += child.amplitude[mode].get(a, self.PSI_0)
                contrib += coupling * child_amp
        return contrib

    # === EPISODE / STATS ===

    def reset_episode(self):
        self._prev_data = None
        self._trace = []
        for child in self._cavities.values():
            child.reset_episode()

    def get_stats(self) -> Dict:
        me = self._mode_energy()
        te = sum(
            abs(c) ** 2
            for targets in self.temporal.values()
            for c in targets.values()
        )
        q = self.resonance_quality()
        return {
            'total_energy': self.total_energy,
            'mode_energy': me,
            'heat_bath': self._heat(),
            'temporal_energy': te,
            'n_modes': len(self.amplitude),
            'n_mode_actions': self._n_modes(),
            'n_temporal': sum(len(t) for t in self.temporal.values()),
            'resonance_q': q,
            'best_q': self._best_q,
            'best_q_frame': self._best_q_frame,
            'n_cavities': len(self._cavities),
            'cavity_depth': self.depth,
            'q_trend': self.q_trend(),
        }
