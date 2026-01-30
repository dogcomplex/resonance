"""
WAVE SIEVE v8 — Recursive Currying
====================================

Everything is a token. A curried rule IS a token. Tokens curry with
other tokens, including curried tokens. This creates arbitrary-depth
rules naturally: A, A⊗B, (A⊗B)⊗C, ((A⊗B)⊗C)⊗D, ...

The wavefunction is a single medium. Every mode — regardless of its
order (depth) — lives in the same dictionary, undergoes the same
decoherence, receives the same signals, and contributes to actions
through the same Born rule.

There is ONE operation: interfere(mode, action, delta).
There is ONE parameter: PSI_0.
There is ONE medium: psi.

Higher-order currying emerges naturally because _curry_key works on
ANY two keys. When a second-order mode A⊗B is "active" (its parts
match the current observation), it can curry with another active mode
C to create (A⊗B)⊗C. The system recursively builds whatever depth
of rules are needed by the task.

Pruning: modes whose amplitudes are ALL at vacuum (PSI_0) carry no
information — they are pure vacuum fluctuation. Periodically removing
them is like the vacuum "forgetting" patterns that never got reinforced.
Information isn't lost — it was never there. These modes were noise
that never crystallized into signal.

token = wave = vertex = rule = mode (of any order)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import cmath


class WaveSieve:
    """One wavefunction. One operation. One parameter. Recursive currying."""

    PSI_0 = 0.1

    def __init__(self):
        # |Ψ⟩ = Σ_k Σ_a ψ[k][a] |k,a⟩
        self.psi: Dict[Any, Dict[int, complex]] = defaultdict(dict)

        self._prev_modes: Optional[List[tuple]] = None
        self._prev_data: Optional[Dict[Any, Any]] = None
        self.total_energy: float = self.PSI_0 ** 2
        self._trace: List[Tuple[List[tuple], int]] = []
        self.frame: int = 0

        # Resonance quality tracking
        self._q_history: List[float] = []
        self._best_q: float = 0.0
        self._best_q_frame: int = 0
        self._best_snapshot: Optional[Dict] = None

    # =================================================================
    # THE ONE OPERATION
    # =================================================================

    def _interfere(self, key: Any, action: int, delta: complex,
                   num_actions: int = None):
        """Add complex amplitude to mode-action. The ONLY mutation."""
        if key not in self.psi:
            if num_actions is None:
                return
            self._spawn_mode(key, num_actions)

        old = self.psi[key].get(action, self.PSI_0 + 0j)
        new = old + delta

        # Vacuum floor
        if abs(new) < self.PSI_0:
            new = self.PSI_0 * cmath.exp(1j * cmath.phase(new)
                                          if abs(new) > 1e-15 else 0j)

        # Energy ceiling
        max_a = np.sqrt(max(self.PSI_0 ** 2, self.total_energy))
        if abs(new) > max_a:
            new = max_a * cmath.exp(1j * cmath.phase(new))

        self.psi[key][action] = new

    def _spawn_mode(self, key: Any, num_actions: int):
        """New mode at vacuum energy."""
        for a in range(num_actions):
            phase = np.random.uniform(0, 2 * np.pi)
            self.psi[key][a] = self.PSI_0 * cmath.exp(1j * phase)
        self.total_energy += num_actions * self.PSI_0 ** 2

    # =================================================================
    # MODE STRUCTURE
    # =================================================================

    def _n_modes(self) -> int:
        return max(1, sum(len(acts) for acts in self.psi.values()))

    def _mode_energy(self) -> float:
        return sum(abs(a) ** 2 for acts in self.psi.values()
                   for a in acts.values())

    def _heat(self) -> float:
        return max(0, self.total_energy - self._mode_energy())

    def _mode_key(self, position: Any, value: Any) -> tuple:
        """First-order mode label."""
        if isinstance(position, tuple):
            return (*position, value)
        return (position, value)

    def _curry_key(self, key_a, key_b) -> tuple:
        """Tensor product. Works on ANY two keys (any order).
        Canonical by hash. A⊗B = B⊗A."""
        if hash(key_a) <= hash(key_b):
            return (key_a, key_b)
        return (key_b, key_a)

    def _order(self, key) -> int:
        """Depth of a mode. First-order = 1, A⊗B = 2, (A⊗B)⊗C = 3, ..."""
        if not isinstance(key, tuple) or len(key) != 2:
            return 1
        if not isinstance(key[0], tuple) or not isinstance(key[1], tuple):
            return 1  # (pos, val) — first-order
        return self._order(key[0]) + self._order(key[1])

    def _is_composite(self, key) -> bool:
        """Is this mode a curry of two sub-modes?"""
        return (isinstance(key, tuple) and len(key) == 2
                and isinstance(key[0], tuple)
                and isinstance(key[1], tuple))

    def _leaf_modes(self, key) -> Set[tuple]:
        """Extract all first-order leaf modes from a (possibly deep) key."""
        if not self._is_composite(key):
            return {key}
        return self._leaf_modes(key[0]) | self._leaf_modes(key[1])

    def _parts(self, key) -> Tuple:
        """Get the two direct sub-keys of a composite mode."""
        if self._is_composite(key):
            return key[0], key[1]
        return None, None

    # =================================================================
    # INPUT
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
    # OBSERVE
    # =================================================================

    def observe(self, data, action: int, num_actions: int):
        """Observation = interference. All through _interfere."""
        self.frame += 1
        config = self._normalize_input(data)

        # Ensure first-order modes
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

        # (1) DECOHERENCE — environment anti-rule on all modes
        decay = self.PSI_0 ** 2 / N
        for key in list(self.psi.keys()):
            for a in list(self.psi[key].keys()):
                old = self.psi[key][a]
                self._interfere(key, a, -decay * old)

        # (2) FIRST-ORDER RULES
        for mode in observed_modes:
            old = self.psi[mode].get(action, self.PSI_0 + 0j)
            phase = cmath.phase(old) if abs(old) > 0 else 0
            self._interfere(mode, action,
                            self.PSI_0 * cmath.exp(1j * phase),
                            num_actions)

        # Build active set: current first-order modes
        active_first = set(observed_modes)

        # Previous-frame modes
        prev_first = set()
        if self._prev_modes is not None:
            prev_first = set(self._prev_modes)

        all_first = active_first | prev_first

        # (3) CURRYING — any active mode can curry with any other
        # This includes curried modes whose leaves are all active.
        # We find all "active" modes (any order) then curry pairs.
        tc = self.PSI_0 / N

        # Find all active modes: a composite is active if both its
        # direct sub-parts are active. Build bottom-up.
        active_modes = set(all_first)
        newly_active = set(all_first)

        # Propagate: check which composites become active
        # (their parts are both in active_modes)
        # Cap iterations to prevent infinite loops
        for _ in range(3):  # max depth to check per frame
            next_active = set()
            for key in self.psi:
                if key in active_modes:
                    continue
                if not self._is_composite(key):
                    continue
                pa, pb = key
                if pa in active_modes and pb in active_modes:
                    next_active.add(key)
            if not next_active:
                break
            active_modes |= next_active
            newly_active = next_active

        # Temporal currying: same-position prev→now
        if self._prev_data is not None:
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

        # Spatial currying: co-observed modes (first-order only)
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

        # HIGHER-ORDER CURRYING: active composites curry with active
        # first-order modes, but ONLY when the composite carries real
        # signal (above vacuum). No speculative mode creation.
        # This is physical: higher harmonics only appear when
        # fundamentals have enough energy.
        floor = self.PSI_0 ** 2
        signal_threshold = floor * 2.0  # must be 2x vacuum to curry up
        active_composites = []
        for k in active_modes:
            if not self._is_composite(k):
                continue
            if k not in self.psi:
                continue
            # Only curry if this composite has differentiated signal
            amps = self.psi[k]
            max_amp = max(abs(a) ** 2 for a in amps.values())
            if max_amp > signal_threshold:
                active_composites.append(k)

        if active_composites and observed_modes:
            max_order = 4
            # Very sparse: 1 sample per frame, only if composites have signal
            comp = active_composites[
                np.random.randint(len(active_composites))]
            if self._order(comp) < max_order:
                first = observed_modes[
                    np.random.randint(len(observed_modes))]
                ck = self._curry_key(comp, first)
                # Only strengthen existing modes OR create if
                # the composite is strongly above vacuum
                if ck in self.psi or max_amp > signal_threshold * 2:
                    ho_tc = tc * self.PSI_0
                    if ck not in self.psi:
                        self._spawn_mode(ck, num_actions)
                    old = self.psi[ck].get(action, self.PSI_0 + 0j)
                    p = cmath.phase(old) if abs(old) > 0 else 0
                    self._interfere(ck, action,
                                    ho_tc * cmath.exp(1j * p),
                                    num_actions)

        # (5) RADIATIVE COOLING
        radiation_rate = self.PSI_0 ** 2 / N
        self.total_energy *= (1 - radiation_rate)
        min_energy = N * self.PSI_0 ** 2
        self.total_energy = max(min_energy, self.total_energy)

        # (6) PRUNING — remove vacuum-only modes periodically
        # Modes at vacuum carry no information. This is physical:
        # vacuum fluctuations that never got reinforced.
        if self.frame % max(50, N // 4) == 0:
            self._prune()

        # Record trace, update state
        self._trace.append((list(active_first), action))
        self._prev_modes = list(observed_modes)
        self._prev_data = dict(config)
        self._update_quality()

    def _prune(self):
        """Remove modes at vacuum. No information lost — there was none.
        Higher-order modes have a higher pruning threshold — they must
        justify their existence with more signal. This is physical:
        higher harmonics are more fragile."""
        floor = self.PSI_0 ** 2
        to_remove = []
        for key, actions in self.psi.items():
            if not self._is_composite(key):
                continue  # never prune first-order modes
            order = self._order(key)
            # Higher order = stricter threshold to survive
            threshold = floor * (1.0 + 0.5 * order)
            max_amp = max(abs(a) ** 2 for a in actions.values())
            min_amp = min(abs(a) ** 2 for a in actions.values())
            # Prune if undifferentiated (no signal) or barely above vacuum
            if max_amp <= threshold or (max_amp - min_amp) < floor * 0.5:
                to_remove.append(key)
        for key in to_remove:
            energy = sum(abs(a) ** 2 for a in self.psi[key].values())
            self.total_energy -= energy
            del self.psi[key]

    # =================================================================
    # CHOOSE ACTION
    # =================================================================

    def choose_action(self, data, num_actions: int,
                      neighbor_fn=None, query_pos=None) -> int:
        """Born rule. All matching modes contribute."""
        config = self._normalize_input(data)
        total_amp = np.zeros(num_actions, dtype=complex)
        N = max(1, len(config))

        neighbor_set = set()
        if query_pos is not None and neighbor_fn is not None:
            neighbor_set = set(neighbor_fn(query_pos))

        # First-order contributions
        observed_modes = set()
        for pos, val in config.items():
            mode = self._mode_key(pos, val)
            if mode not in self.psi:
                self._spawn_mode(mode, num_actions)
            observed_modes.add(mode)

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

        # Build active set (same as observe)
        prev_modes = set()
        if self._prev_modes is not None:
            prev_modes = set(self._prev_modes)
        all_first = observed_modes | prev_modes

        # Find ALL active composite modes (bottom-up)
        active_modes = set(all_first)
        for _ in range(3):
            next_active = set()
            for key in self.psi:
                if key in active_modes:
                    continue
                if not self._is_composite(key):
                    continue
                pa, pb = key
                if pa in active_modes and pb in active_modes:
                    next_active.add(key)
            if not next_active:
                break
            active_modes |= next_active

        # Composite mode contributions
        n_total = self._n_modes()
        for key in active_modes:
            if not self._is_composite(key):
                continue
            if key not in self.psi:
                continue
            actions = self.psi[key]
            order = self._order(key)
            # Weight decreases with order: PSI_0^(order-1) / N
            weight = (self.PSI_0 ** (order - 1)) / n_total
            for a in range(num_actions):
                total_amp[a] += weight * actions.get(a, self.PSI_0 + 0j)

        # Born rule
        probs = np.array([abs(a) ** 2 for a in total_amp])
        probs = np.maximum(probs, self.PSI_0 ** 2)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_actions) / num_actions

        return np.random.choice(num_actions, p=probs)

    # =================================================================
    # SIGNAL
    # =================================================================

    def _signal(self, phase_shift: float):
        """Signal = meta-rule interference on trace."""
        if not self._trace:
            return

        signal_energy = len(self._trace) * self.PSI_0 ** 2
        self.total_energy += signal_energy

        for i, (trace_modes, action) in enumerate(reversed(self._trace)):
            credit = 1.0 / (1 + i) ** 2
            trace_set = set(trace_modes)

            # First-order modes
            for mode in trace_modes:
                if mode not in self.psi:
                    continue
                old = self.psi[mode].get(action, self.PSI_0 + 0j)
                old_phase = cmath.phase(old) if abs(old) > 0 else 0
                delta = credit * self.PSI_0 * cmath.exp(
                    1j * (old_phase + phase_shift))
                self._interfere(mode, action, delta)

            # ALL composite modes whose leaves intersect the trace
            for key in list(self.psi.keys()):
                if not self._is_composite(key):
                    continue
                leaves = self._leaf_modes(key)
                if leaves.issubset(trace_set):
                    old = self.psi[key].get(action, self.PSI_0 + 0j)
                    old_phase = cmath.phase(old) if abs(old) > 0 else 0
                    order = self._order(key)
                    # Signal strength scales with PSI_0^order
                    delta = (credit * self.PSI_0 ** order
                             * cmath.exp(1j * (old_phase + phase_shift)))
                    self._interfere(key, action, delta)

        self._trace = []

    def signal_death(self):
        """Phase π. Antimatter. NOT wave."""
        self._signal(np.pi)

    def signal_success(self):
        """Phase 0. Matter. Binding."""
        self._signal(0.0)

    # =================================================================
    # RESONANCE QUALITY
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
        first_order = 0
        by_order = defaultdict(int)
        for k in self.psi:
            o = self._order(k)
            by_order[o] += 1
            if o == 1:
                first_order += 1
        higher_order = sum(v for k, v in by_order.items() if k > 1)
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
            'order_distribution': dict(by_order),
            'resonance_q': q,
            'best_q': self._best_q,
            'best_q_frame': self._best_q_frame,
            'q_trend': self.q_trend(),
        }
