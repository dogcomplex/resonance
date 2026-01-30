"""
WAVE SIEVE v6 - Curried Resonance
===================================

Everything is a rule. A rule is a wave. A wave is a mode.

token = wave = vertex = rule

A mode is a tuple key of arbitrary depth mapped to action amplitudes.
First-order:  (pos, val) -> actions         "if pos=val, prefer action"
Second-order: ((p1,v1),(p2,v2)) -> actions  "if p1=v1 AND p2=v2, prefer action"
Nth-order:    curry of N first-order rules

CURRYING:
=========
When rules co-fire (are observed together), they CURRY into a composite
rule in the same amplitude dictionary. No separate coupling dict.
Currying = binding = constructive interference = phase 0 = matter.
Uncurrying = splitting = the composite contributes back to its parts.

Every operation is the same: inject a rule with a phase.
- observe(): inject first-order rules (phase 0)
- signal_success(): reinforce rules on trace (phase 0)
- signal_death(): anti-reinforce rules on trace (phase pi)
- decoherence: environment injects universal weak anti-rules (phase pi)
- curry: co-firing rules combine into higher-order rule (phase 0)

One free parameter: PSI_0 (zero-point amplitude).
One operation: interfere(mode, action, amplitude, phase).
One medium: the amplitude dictionary.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import cmath


class WaveSieve:
    """Everything is a rule. One parameter: PSI_0."""

    PSI_0 = 0.1

    def __init__(self):
        # THE medium. Every rule lives here, regardless of order.
        # key -> {action: complex_amplitude}
        # key can be (pos, val) or ((pos1,val1), (pos2,val2)) or deeper.
        self.amplitude: Dict[Any, Dict[int, complex]] = defaultdict(dict)
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

    def _order(self, key) -> int:
        """Depth of a curried rule. First-order = 1."""
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], tuple) and isinstance(key[1], tuple):
                # Composite: ((a,b), (c,d)) or deeper
                return self._order(key[0]) + self._order(key[1])
            # Simple: (pos, val) — first-order
            return 1
        return 1

    # === RESONANCE QUALITY (optimal stopping) ===

    def _signal_strength(self) -> float:
        signal = 0.0
        floor = self.PSI_0 ** 2
        for actions in self.amplitude.values():
            for a in actions.values():
                excess = abs(a) ** 2 - floor
                if excess > 0:
                    signal += excess
        return signal

    def _mode_contrast(self) -> float:
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
            'amplitude': {k: dict(v) for k, v in self.amplitude.items()},
            'total_energy': self.total_energy,
            'frame': self.frame,
        }

    def restore_best(self) -> int:
        if self._best_snapshot is None:
            return self.frame
        snap = self._best_snapshot
        self.amplitude = defaultdict(dict)
        for k, v in snap['amplitude'].items():
            self.amplitude[k] = dict(v)
        self.total_energy = snap['total_energy']
        return snap['frame']

    def q_trend(self, window: int = 100) -> float:
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

    def _curry_key(self, key_a: tuple, key_b: tuple) -> tuple:
        """Curry two rules into a composite. Canonical order by hash."""
        if hash(key_a) <= hash(key_b):
            return (key_a, key_b)
        return (key_b, key_a)

    def _ensure_mode(self, key: Any, num_actions: int):
        if key not in self.amplitude:
            for a in range(num_actions):
                phase = np.random.uniform(0, 2 * np.pi)
                self.amplitude[key][a] = self.PSI_0 * cmath.exp(1j * phase)
            self.total_energy += num_actions * self.PSI_0 ** 2

    # === OBSERVE ===

    def observe(self, data, action: int, num_actions: int):
        """Observation = injecting first-order rules into the medium.
        Co-observed rules curry into higher-order rules.
        Previous-frame rules curry temporally."""
        self.frame += 1
        config = self._normalize_input(data)

        for pos, val in config.items():
            self._ensure_mode(self._mode_key(pos, val), num_actions)

        N = self._n_modes()

        # Energy pump: each observation injects energy
        n_obs = len(config)
        self.total_energy += n_obs * self.PSI_0 ** 2

        # Global decoherence: the environment applies a universal
        # weak anti-rule to ALL modes. This IS the environment
        # being a higher-order sieve that decays everything.
        decay = self.PSI_0 ** 2 / N
        for mode_key in list(self.amplitude.keys()):
            self._decohere_mode(mode_key, decay)

        # Inject first-order rules: "this config + action was observed"
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

        # === CURRYING: co-firing rules combine ===
        observed_modes = []
        for pos, val in config.items():
            observed_modes.append(self._mode_key(pos, val))

        tc = self.PSI_0 / N

        # Temporal currying: rules from t-1 curry with rules at t
        if self._prev_data is not None:
            for pos, val in config.items():
                if pos in self._prev_data:
                    prev_val = self._prev_data[pos]
                    now_key = self._mode_key(pos, val)
                    prev_key = self._mode_key(pos, prev_val)
                    curry_key = self._curry_key(prev_key, now_key)
                    self._ensure_mode(curry_key, num_actions)
                    # The curried rule gets the action that was taken
                    old = self.amplitude[curry_key].get(action, self.PSI_0)
                    p = cmath.phase(old) if abs(old) > 0 else 0
                    self.amplitude[curry_key][action] = (
                        old + tc * cmath.exp(1j * p))

        # Spatial currying: co-observed rules curry
        # Stochastic — sqrt(N_obs) pairs sampled per frame
        if len(observed_modes) >= 2:
            n_samples = max(1, int(np.sqrt(len(observed_modes))))
            for _ in range(n_samples):
                i, j = np.random.choice(len(observed_modes), size=2,
                                        replace=False)
                curry_key = self._curry_key(observed_modes[i],
                                            observed_modes[j])
                self._ensure_mode(curry_key, num_actions)
                old = self.amplitude[curry_key].get(action, self.PSI_0)
                p = cmath.phase(old) if abs(old) > 0 else 0
                self.amplitude[curry_key][action] = (
                    old + tc * cmath.exp(1j * p))

        self._trace.append((config, action))
        self._prev_data = dict(config)
        self._update_quality()

    def _decohere_mode(self, key: Any, rate: float):
        """Decoherence IS the environment applying anti-rules.
        All modes decay toward zero-point, regardless of order."""
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
        """Sum amplitudes from ALL matching rules — first-order AND curried.
        A curried rule matches when ALL its constituent parts are present."""
        config = self._normalize_input(data)
        total_amp = np.zeros(num_actions, dtype=complex)
        N = max(1, len(config))

        neighbor_set = set()
        if query_pos is not None and neighbor_fn is not None:
            neighbor_set = set(neighbor_fn(query_pos))

        # Collect first-order modes from current observation
        observed_modes = set()
        for pos, val in config.items():
            mode = self._mode_key(pos, val)
            self._ensure_mode(mode, num_actions)
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
                total_amp[a] += weight * self.amplitude[mode].get(
                    a, self.PSI_0)

        # Add previous-frame modes for temporal curried rule matching
        prev_modes = set()
        if self._prev_data is not None:
            for pos, val in self._prev_data.items():
                prev_modes.add(self._mode_key(pos, val))

        # Curried rule contributions: scan all modes, find composites
        # whose parts are present in observed_modes or prev_modes.
        # Weight by PSI_0 / N (coupling strength, same as temporal).
        tc = self.PSI_0 / self._n_modes()
        all_active = observed_modes | prev_modes

        for key, actions in self.amplitude.items():
            # Skip first-order (already counted above)
            if not (isinstance(key, tuple) and len(key) == 2
                    and isinstance(key[0], tuple)
                    and isinstance(key[1], tuple)):
                continue

            # Check if both parts of this curried rule are active
            part_a, part_b = key
            if part_a in all_active and part_b in all_active:
                # Weight: higher-order rules contribute less per-rule
                # but there are more of them. PSI_0/N scales naturally.
                for a in range(num_actions):
                    total_amp[a] += tc * actions.get(a, self.PSI_0)

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
        """The signal is a rule about the trace. Phase 0 = curry/bind.
        Phase pi = uncurry/unbind. Same operation, different phase.
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

        # Signal also reinforces/anti-reinforces curried rules
        # that were active during the trace. The signal IS a
        # higher-order rule being applied to all active rules.
        for i, (config, action) in enumerate(reversed(self._trace)):
            credit = 1.0 / (1 + i) ** 2
            trace_modes = []
            for pos, val in config.items():
                trace_modes.append(self._mode_key(pos, val))

            # Reinforce curried rules whose parts were in this frame
            trace_set = set(trace_modes)
            for key in list(self.amplitude.keys()):
                if not (isinstance(key, tuple) and len(key) == 2
                        and isinstance(key[0], tuple)
                        and isinstance(key[1], tuple)):
                    continue
                part_a, part_b = key
                if part_a in trace_set and part_b in trace_set:
                    old = self.amplitude[key].get(action, self.PSI_0)
                    old_phase = cmath.phase(old) if abs(old) > 0 else 0
                    signal = credit * amp * self.PSI_0 * cmath.exp(
                        1j * (old_phase + phase_shift))
                    new = old + signal
                    if abs(new) < self.PSI_0:
                        new = self.PSI_0 * cmath.exp(
                            1j * cmath.phase(new))
                    if abs(new) > max_a:
                        new = max_a * cmath.exp(
                            1j * cmath.phase(new))
                    self.amplitude[key][action] = new

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
        # Count by order
        first_order = 0
        higher_order = 0
        for key in self.amplitude:
            if (isinstance(key, tuple) and len(key) == 2
                    and isinstance(key[0], tuple)
                    and isinstance(key[1], tuple)):
                higher_order += 1
            else:
                first_order += 1
        q = self.resonance_quality()
        return {
            'total_energy': self.total_energy,
            'mode_energy': me,
            'heat_bath': self._heat(),
            'n_modes': len(self.amplitude),
            'n_first_order': first_order,
            'n_curried': higher_order,
            'n_mode_actions': self._n_modes(),
            'n_couplings': higher_order,  # compat with old tests
            'resonance_q': q,
            'best_q': self._best_q,
            'best_q_frame': self._best_q_frame,
            'q_trend': self.q_trend(),
        }
