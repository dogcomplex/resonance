"""
WAVE SIEVE v7 — Holographic Wavefunction
==========================================

The medium is ONE wavefunction expanded in a discrete basis.
Every entry in the basis is a mode. Modes are labeled for bookkeeping
but they are all part of the SAME holographic interference pattern.

token = wave = vertex = rule = mode

There is exactly ONE operation on the medium: INTERFERE.
    interfere(mode, action, delta)
    where delta is a complex amplitude to add.

Everything else is a special case of interfere:
    observe    = interfere(mode, action, +PSI_0)          # inject rule
    curry      = interfere(A⊗B, action, +PSI_0/N)        # bind two rules
    decohere   = interfere(mode, action, -rate * current) # env anti-rule
    signal(+)  = interfere(mode, action, +credit * PSI_0) # reinforce
    signal(-)  = interfere(mode, action, -credit * PSI_0) # anti-reinforce (matter/antimatter)

Phase 0 = matter = curry = bind = constructive.
Phase π = antimatter = uncurry = unbind = destructive.
Same operation, opposite phase. That's all there is.

A "curried rule" is a mode whose key is the tensor product of
two simpler keys: (key_A, key_B). It lives in the SAME wavefunction.
It IS a higher harmonic of the same wave. When you observe A and B
together, their product creates the A⊗B harmonic — this is just
how waves work. Fourier products create sum/difference frequencies.

The dictionary is a basis expansion: Ψ = Σ_k Σ_a ψ(k,a) |k,a⟩
where k labels modes (of any order) and a labels actions.
The basis is adaptive — modes appear when needed and vanish when
they carry no information (all amplitudes ≈ PSI_0).

One free parameter: PSI_0 (zero-point amplitude = vacuum fluctuation).
One operation: interfere.
One medium: the wavefunction.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import cmath


class WaveSieve:
    """One wavefunction. One operation. One parameter."""

    PSI_0 = 0.1

    def __init__(self):
        # The wavefunction, expanded in an adaptive basis.
        # |Ψ⟩ = Σ_k Σ_a ψ[k][a] |k,a⟩
        # k: mode label (tuple), a: action index
        self.psi: Dict[Any, Dict[int, complex]] = defaultdict(dict)

        self._prev_modes: Optional[List[tuple]] = None
        self._prev_data: Optional[Dict[Any, Any]] = None
        self.total_energy: float = self.PSI_0 ** 2
        self._trace: List[Tuple[List[tuple], int]] = []  # [(mode_keys, action), ...]
        self.frame: int = 0

        # Resonance quality tracking (optimal stopping)
        self._q_history: List[float] = []
        self._best_q: float = 0.0
        self._best_q_frame: int = 0
        self._best_snapshot: Optional[Dict] = None

    # =================================================================
    # THE ONE OPERATION
    # =================================================================

    def _interfere(self, key: Any, action: int, delta: complex,
                   num_actions: int = None):
        """Add a complex amplitude to a mode-action.
        This is the ONLY way the wavefunction changes.
        Everything else calls this."""
        if key not in self.psi:
            if num_actions is None:
                return  # Can't create mode without knowing action count
            self._spawn_mode(key, num_actions)

        old = self.psi[key].get(action, self.PSI_0 + 0j)
        new = old + delta

        # Vacuum floor: amplitude never goes below PSI_0
        if abs(new) < self.PSI_0:
            new = self.PSI_0 * cmath.exp(1j * cmath.phase(new)
                                          if abs(new) > 1e-15 else 0j)

        # Energy ceiling: prevent runaway
        max_a = np.sqrt(max(self.PSI_0 ** 2, self.total_energy))
        if abs(new) > max_a:
            new = max_a * cmath.exp(1j * cmath.phase(new))

        self.psi[key][action] = new

    def _spawn_mode(self, key: Any, num_actions: int):
        """A new mode appears in the wavefunction with vacuum energy."""
        for a in range(num_actions):
            phase = np.random.uniform(0, 2 * np.pi)
            self.psi[key][a] = self.PSI_0 * cmath.exp(1j * phase)
        self.total_energy += num_actions * self.PSI_0 ** 2

    # =================================================================
    # DERIVED QUANTITIES
    # =================================================================

    def _n_modes(self) -> int:
        return max(1, sum(len(acts) for acts in self.psi.values()))

    def _mode_energy(self) -> float:
        return sum(abs(a) ** 2 for acts in self.psi.values()
                   for a in acts.values())

    def _heat(self) -> float:
        return max(0, self.total_energy - self._mode_energy())

    # =================================================================
    # MODE KEYS
    # =================================================================

    def _mode_key(self, position: Any, value: Any) -> tuple:
        """First-order mode label."""
        if isinstance(position, tuple):
            return (*position, value)
        return (position, value)

    def _curry_key(self, key_a: tuple, key_b: tuple) -> tuple:
        """Tensor product label. Canonical order by hash.
        A⊗B = B⊗A (symmetric curry)."""
        if hash(key_a) <= hash(key_b):
            return (key_a, key_b)
        return (key_b, key_a)

    def _is_curried(self, key) -> bool:
        """Is this a higher-order (composite) mode?"""
        return (isinstance(key, tuple) and len(key) == 2
                and isinstance(key[0], tuple)
                and isinstance(key[1], tuple))

    # =================================================================
    # INPUT NORMALIZATION
    # =================================================================

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

    # =================================================================
    # OBSERVE — inject rules + curry
    # =================================================================

    def observe(self, data, action: int, num_actions: int):
        """Observation = interfering rules into the wavefunction.
        All operations use _interfere. Nothing else."""
        self.frame += 1
        config = self._normalize_input(data)

        # Ensure all first-order modes exist
        observed_modes = []
        for pos, val in config.items():
            mode = self._mode_key(pos, val)
            if mode not in self.psi:
                self._spawn_mode(mode, num_actions)
            observed_modes.append(mode)

        N = self._n_modes()
        n_obs = len(config)

        # Energy pump
        self.total_energy += n_obs * self.PSI_0 ** 2

        # (1) DECOHERENCE — universal weak destructive interference
        # The environment IS a higher-order sieve applying phase π
        # to everything. Same operation: interfere with negative delta.
        decay = self.PSI_0 ** 2 / N
        for key in list(self.psi.keys()):
            for a in list(self.psi[key].keys()):
                old = self.psi[key][a]
                # Decay toward vacuum = subtract proportional to excess
                self._interfere(key, a, -decay * old)

        # (2) FIRST-ORDER RULES — constructive interference (phase 0)
        # "This configuration with this action was observed"
        for mode in observed_modes:
            old = self.psi[mode].get(action, self.PSI_0 + 0j)
            phase = cmath.phase(old) if abs(old) > 0 else 0
            self._interfere(mode, action,
                            self.PSI_0 * cmath.exp(1j * phase),
                            num_actions)

        # (3) TEMPORAL CURRYING — product of t-1 and t waves
        # When mode_prev and mode_now co-fire across time,
        # their product creates the curried harmonic
        tc = self.PSI_0 / N
        if self._prev_modes is not None and self._prev_data is not None:
            for mode_now, (pos, val) in zip(observed_modes, config.items()):
                if pos in self._prev_data:
                    prev_val = self._prev_data[pos]
                    mode_prev = self._mode_key(pos, prev_val)
                    ck = self._curry_key(mode_prev, mode_now)
                    if ck not in self.psi:
                        self._spawn_mode(ck, num_actions)
                    old = self.psi[ck].get(action, self.PSI_0 + 0j)
                    p = cmath.phase(old) if abs(old) > 0 else 0
                    self._interfere(ck, action,
                                    tc * cmath.exp(1j * p),
                                    num_actions)

        # (4) SPATIAL CURRYING — product of co-observed waves
        # Stochastic sampling: √N_obs pairs per frame
        if len(observed_modes) >= 2:
            n_samples = max(1, int(np.sqrt(len(observed_modes))))
            for _ in range(n_samples):
                i, j = np.random.choice(len(observed_modes), size=2,
                                        replace=False)
                ck = self._curry_key(observed_modes[i], observed_modes[j])
                if ck not in self.psi:
                    self._spawn_mode(ck, num_actions)
                old = self.psi[ck].get(action, self.PSI_0 + 0j)
                p = cmath.phase(old) if abs(old) > 0 else 0
                self._interfere(ck, action,
                                tc * cmath.exp(1j * p),
                                num_actions)

        # (5) RADIATIVE COOLING — energy dissipation
        radiation_rate = self.PSI_0 ** 2 / N
        self.total_energy *= (1 - radiation_rate)
        min_energy = N * self.PSI_0 ** 2
        self.total_energy = max(min_energy, self.total_energy)

        # Record trace for signal, update state
        self._trace.append((observed_modes, action))
        self._prev_modes = list(observed_modes)
        self._prev_data = dict(config)
        self._update_quality()

    # =================================================================
    # CHOOSE ACTION — measure the wavefunction
    # =================================================================

    def choose_action(self, data, num_actions: int,
                      neighbor_fn=None, query_pos=None) -> int:
        """Born rule measurement. Sum amplitudes from all matching modes.
        Probability = |Ψ(action)|². Same physics, no special cases."""
        config = self._normalize_input(data)
        total_amp = np.zeros(num_actions, dtype=complex)
        N = max(1, len(config))

        neighbor_set = set()
        if query_pos is not None and neighbor_fn is not None:
            neighbor_set = set(neighbor_fn(query_pos))

        # First-order mode contributions
        observed_modes = set()
        for pos, val in config.items():
            mode = self._mode_key(pos, val)
            if mode not in self.psi:
                self._spawn_mode(mode, num_actions)
            observed_modes.add(mode)

            # Green's function locality weighting
            if query_pos is not None and pos == query_pos:
                weight = float(N)
            elif pos in neighbor_set:
                weight = 1.0
            elif query_pos is not None:
                weight = 1.0 / N
            else:
                weight = 1.0

            for a in range(num_actions):
                total_amp[a] += weight * self.psi[mode].get(
                    a, self.PSI_0 + 0j)

        # Previous-frame modes for temporal curry matching
        prev_modes = set()
        if self._prev_modes is not None:
            prev_modes = set(self._prev_modes)

        # Curried mode contributions: when both parts are active,
        # the higher harmonic contributes to the total wavefunction
        all_active = observed_modes | prev_modes
        n_total = self._n_modes()
        tc = self.PSI_0 / n_total

        for key, actions in self.psi.items():
            if not self._is_curried(key):
                continue
            part_a, part_b = key
            if part_a in all_active and part_b in all_active:
                for a in range(num_actions):
                    total_amp[a] += tc * actions.get(a, self.PSI_0 + 0j)

        # Born rule: P(a) = |Ψ(a)|²
        probs = np.array([abs(a) ** 2 for a in total_amp])
        probs = np.maximum(probs, self.PSI_0 ** 2)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_actions) / num_actions

        return np.random.choice(num_actions, p=probs)

    # =================================================================
    # SIGNAL — meta-rule about the trace
    # =================================================================

    def _signal(self, phase_shift: float):
        """Signal = interfere with the trace modes.
        phase_shift = 0: constructive (matter, curry, bind, success)
        phase_shift = π: destructive (antimatter, uncurry, unbind, death)
        Credit = 1/(1+r)². Signal carries its own energy.
        This IS the same operation applied as a meta-rule."""
        if not self._trace:
            return

        signal_energy = len(self._trace) * self.PSI_0 ** 2
        self.total_energy += signal_energy

        for i, (trace_modes, action) in enumerate(reversed(self._trace)):
            credit = 1.0 / (1 + i) ** 2

            # Interfere with first-order modes on the trace
            for mode in trace_modes:
                if mode not in self.psi:
                    continue
                old = self.psi[mode].get(action, self.PSI_0 + 0j)
                old_phase = cmath.phase(old) if abs(old) > 0 else 0
                delta = credit * self.PSI_0 * cmath.exp(
                    1j * (old_phase + phase_shift))
                self._interfere(mode, action, delta)

            # Interfere with curried modes whose parts were active
            trace_set = set(trace_modes)
            for key in list(self.psi.keys()):
                if not self._is_curried(key):
                    continue
                part_a, part_b = key
                if part_a in trace_set and part_b in trace_set:
                    old = self.psi[key].get(action, self.PSI_0 + 0j)
                    old_phase = cmath.phase(old) if abs(old) > 0 else 0
                    # Curried signal is PSI_0 weaker (natural scaling)
                    delta = credit * self.PSI_0 ** 2 * cmath.exp(
                        1j * (old_phase + phase_shift))
                    self._interfere(key, action, delta)

        self._trace = []

    def signal_death(self):
        """Phase π interference. THE NOT WAVE. Antimatter."""
        self._signal(np.pi)

    def signal_success(self):
        """Phase 0 interference. Matter. Binding."""
        self._signal(0.0)

    # =================================================================
    # RESONANCE QUALITY (optimal stopping)
    # =================================================================

    def _signal_strength(self) -> float:
        floor = self.PSI_0 ** 2
        return sum(max(0, abs(a) ** 2 - floor)
                   for acts in self.psi.values()
                   for a in acts.values())

    def _mode_contrast(self) -> float:
        if not self.psi:
            return 0.0
        floor = self.PSI_0 ** 2
        contrasts = []
        for actions in self.psi.values():
            if not actions:
                continue
            intensities = [abs(a) ** 2 for a in actions.values()]
            max_i = max(intensities)
            if max_i <= floor * 1.5:
                continue
            min_i = min(intensities)
            if max_i + min_i > 0:
                contrasts.append((max_i - min_i) / (max_i + min_i))
        return sum(contrasts) / len(contrasts) if contrasts else 0.0

    def _mode_selectivity(self) -> float:
        if not self.psi:
            return 0.0
        floor = self.PSI_0 ** 2
        total = differentiated = 0
        for actions in self.psi.values():
            if not actions:
                continue
            total += 1
            intensities = [abs(a) ** 2 for a in actions.values()]
            max_i, min_i = max(intensities), min(intensities)
            if max_i > floor * 1.5 and (max_i - min_i) > floor:
                differentiated += 1
        return differentiated / total if total else 0.0

    def resonance_quality(self) -> float:
        contrast = self._mode_contrast()
        selectivity = self._mode_selectivity()
        signal = np.log1p(self._signal_strength() / self.PSI_0 ** 2)
        return contrast * selectivity * signal

    def _update_quality(self):
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
        return {
            'psi': {k: dict(v) for k, v in self.psi.items()},
            'total_energy': self.total_energy,
            'frame': self.frame,
        }

    def restore_best(self) -> int:
        if self._best_snapshot is None:
            return self.frame
        snap = self._best_snapshot
        self.psi = defaultdict(dict)
        for k, v in snap['psi'].items():
            self.psi[k] = dict(v)
        self.total_energy = snap['total_energy']
        return snap['frame']

    def q_trend(self, window: int = 100) -> float:
        if len(self._q_history) < window + 1:
            return 0.0
        recent = self._q_history[-window:]
        old = self._q_history[-(window + window // 2):-window // 2]
        return ((sum(recent) / len(recent)) - (sum(old) / len(old))
                if old else 0.0)

    # =================================================================
    # ANNEALING
    # =================================================================

    def anneal(self, temperature: float = 1.0):
        self.total_energy += len(self.psi) * self.PSI_0 ** 2 * temperature

    def cool(self, rate: float = None):
        if rate is None:
            rate = self.PSI_0 ** 2 / self._n_modes()
        self.total_energy -= self._heat() * rate

    # =================================================================
    # EPISODE / STATS
    # =================================================================

    def reset_episode(self):
        self._prev_modes = None
        self._prev_data = None
        self._trace = []

    def get_stats(self) -> Dict:
        first_order = sum(1 for k in self.psi if not self._is_curried(k))
        higher_order = sum(1 for k in self.psi if self._is_curried(k))
        q = self.resonance_quality()
        return {
            'total_energy': self.total_energy,
            'mode_energy': self._mode_energy(),
            'heat_bath': self._heat(),
            'n_modes': len(self.psi),
            'n_first_order': first_order,
            'n_curried': higher_order,
            'n_mode_actions': self._n_modes(),
            'n_couplings': higher_order,  # compat
            'resonance_q': q,
            'best_q': self._best_q,
            'best_q_frame': self._best_q_frame,
            'q_trend': self.q_trend(),
        }
