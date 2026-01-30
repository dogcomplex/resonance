"""
WAVE SIEVE v10 — Ternary Wavefunction (Scaled)
================================================

Ternary amplitudes {-1, 0, +1} with concentration = temperature.
Each mode-action stores [plus_count, minus_count].

v10 fixes:
- Decoherence preserves signal RATIO (multiplicative decay on both
  sides equally, like radioactive decay). Net magnetization preserved.
- Pruning uses mode CAP relative to first-order count.
- Softmax temperature derived from signal RANGE not absolute magnitude.
- Curried mode creation rate-limited to prevent explosion.

One operation: add trits. One medium: integer counts.
One parameter: SEED_COUNT = 1.

token = wave = vertex = rule = mode (of any order)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict


class WaveSieve:
    """Ternary wavefunction. Integer arithmetic. One parameter."""

    SEED_COUNT = 1
    PSI_0 = 0.1  # interface compat

    def __init__(self):
        self.psi: Dict[Any, Dict[int, List[int]]] = defaultdict(dict)
        self._prev_modes: Optional[List[tuple]] = None
        self._prev_data: Optional[Dict[Any, Any]] = None
        self.total_trits: int = 0
        self._trace: List[Tuple[List[tuple], int]] = []
        self.frame: int = 0
        self._n_first_order: int = 0  # track for mode cap

        # Resonance quality tracking
        self._q_history: List[float] = []
        self._best_q: float = 0.0
        self._best_q_frame: int = 0
        self._best_snapshot: Optional[Dict] = None

    # =================================================================
    # THE ONE OPERATION
    # =================================================================

    def _interfere(self, key: Any, action: int, sign: int, count: int = 1,
                   num_actions: int = None):
        """Add trits to a mode-action. The ONLY mutation."""
        if count <= 0:
            return
        if key not in self.psi:
            if num_actions is None:
                return
            self._spawn_mode(key, num_actions)

        if action not in self.psi[key]:
            self.psi[key][action] = [0, 0]

        pm = self.psi[key][action]
        if sign > 0:
            pm[0] += count
        else:
            pm[1] += count
        self.total_trits += count

        # Annihilate
        annihilated = min(pm[0], pm[1])
        if annihilated > 0:
            pm[0] -= annihilated
            pm[1] -= annihilated
            self.total_trits -= 2 * annihilated

    def _spawn_mode(self, key: Any, num_actions: int):
        """New mode at vacuum."""
        for a in range(num_actions):
            sign = np.random.choice([-1, 1])
            self.psi[key][a] = [1 if sign > 0 else 0,
                                1 if sign < 0 else 0]
            self.total_trits += 1
        if not self._is_composite(key):
            self._n_first_order += 1

    # =================================================================
    # DERIVED QUANTITIES
    # =================================================================

    def _net(self, key, action) -> int:
        if key not in self.psi or action not in self.psi[key]:
            return 0
        pm = self.psi[key][action]
        return pm[0] - pm[1]

    def _energy(self, key, action) -> int:
        if key not in self.psi or action not in self.psi[key]:
            return 0
        pm = self.psi[key][action]
        return pm[0] + pm[1]

    def _mode_total_trits(self, key) -> int:
        return sum(pm[0] + pm[1] for pm in self.psi[key].values())

    def _n_modes(self) -> int:
        return max(1, sum(len(acts) for acts in self.psi.values()))

    def _heat(self) -> float:
        return 0.0

    # =================================================================
    # MODE STRUCTURE
    # =================================================================

    def _mode_key(self, position: Any, value: Any) -> tuple:
        if isinstance(position, tuple):
            return (*position, value)
        return (position, value)

    def _curry_key(self, key_a, key_b) -> tuple:
        if hash(key_a) <= hash(key_b):
            return (key_a, key_b)
        return (key_b, key_a)

    def _order(self, key) -> int:
        if not isinstance(key, tuple) or len(key) != 2:
            return 1
        if not isinstance(key[0], tuple) or not isinstance(key[1], tuple):
            return 1
        return self._order(key[0]) + self._order(key[1])

    def _is_composite(self, key) -> bool:
        return (isinstance(key, tuple) and len(key) == 2
                and isinstance(key[0], tuple)
                and isinstance(key[1], tuple))

    def _leaf_modes(self, key) -> Set[tuple]:
        if not self._is_composite(key):
            return {key}
        return self._leaf_modes(key[0]) | self._leaf_modes(key[1])

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
        self.frame += 1
        config = self._normalize_input(data)

        observed_modes = []
        for pos, val in config.items():
            mode = self._mode_key(pos, val)
            if mode not in self.psi:
                self._spawn_mode(mode, num_actions)
            observed_modes.append(mode)

        N = self._n_modes()

        # (1) DECOHERENCE — proportional decay that PRESERVES net ratio
        # Remove fraction of trits from BOTH sides equally.
        # This shrinks total concentration but keeps net/total constant.
        # Like evaporation: both matter and antimatter evaporate.
        decay_rate = 1.0 / N
        for key in list(self.psi.keys()):
            for a in list(self.psi[key].keys()):
                pm = self.psi[key][a]
                # Decay each side independently
                for side in (0, 1):
                    if pm[side] <= 0:
                        continue
                    n_remove = pm[side] * decay_rate
                    n_rem_int = int(n_remove)
                    if np.random.random() < (n_remove - n_rem_int):
                        n_rem_int += 1
                    if n_rem_int > 0 and pm[0] + pm[1] > 1:
                        actual = min(n_rem_int, pm[side])
                        pm[side] -= actual
                        self.total_trits -= actual

        # (2) FIRST-ORDER RULES
        for mode in observed_modes:
            self._interfere(mode, action, +1, self.SEED_COUNT, num_actions)

        # Build active sets
        active_first = set(observed_modes)
        prev_first = set()
        if self._prev_modes is not None:
            prev_first = set(self._prev_modes)
        all_first = active_first | prev_first

        # Mode cap: max composites = N_first * N_first (quadratic)
        # Beyond this, skip currying to prevent explosion
        n_composites = len(self.psi) - self._n_first_order
        mode_cap = max(100, self._n_first_order ** 2)
        allow_new_modes = n_composites < mode_cap

        # (3) TEMPORAL CURRYING
        if self._prev_data is not None:
            for mode_now, (pos, val) in zip(observed_modes, config.items()):
                if pos in self._prev_data:
                    prev_val = self._prev_data[pos]
                    mode_prev = self._mode_key(pos, prev_val)
                    ck = self._curry_key(mode_prev, mode_now)
                    if ck in self.psi:
                        # Strengthen existing curried mode
                        self._interfere(ck, action, +1, self.SEED_COUNT,
                                        num_actions)
                    elif allow_new_modes:
                        self._spawn_mode(ck, num_actions)
                        self._interfere(ck, action, +1, self.SEED_COUNT,
                                        num_actions)

        # (4) SPATIAL CURRYING — only create new if under cap
        if len(observed_modes) >= 2:
            n_samples = max(1, int(np.sqrt(len(observed_modes))))
            for _ in range(n_samples):
                i, j = np.random.choice(len(observed_modes), size=2,
                                        replace=False)
                ck = self._curry_key(observed_modes[i], observed_modes[j])
                if ck in self.psi:
                    self._interfere(ck, action, +1, self.SEED_COUNT,
                                    num_actions)
                elif allow_new_modes:
                    self._spawn_mode(ck, num_actions)
                    self._interfere(ck, action, +1, self.SEED_COUNT,
                                    num_actions)

        # (5) HIGHER-ORDER CURRYING (gated by signal AND mode cap)
        active_modes = set(all_first)
        for _ in range(3):
            next_active = set()
            for key in self.psi:
                if key in active_modes or not self._is_composite(key):
                    continue
                pa, pb = key
                if pa in active_modes and pb in active_modes:
                    next_active.add(key)
            if not next_active:
                break
            active_modes |= next_active

        if allow_new_modes:
            active_composites = []
            for k in active_modes:
                if not self._is_composite(k) or k not in self.psi:
                    continue
                mt = self._mode_total_trits(k)
                if mt > num_actions * 2:
                    active_composites.append(k)

            if active_composites and observed_modes:
                comp = active_composites[
                    np.random.randint(len(active_composites))]
                if self._order(comp) < 4:
                    first = observed_modes[
                        np.random.randint(len(observed_modes))]
                    ck = self._curry_key(comp, first)
                    if ck in self.psi:
                        self._interfere(ck, action, +1, self.SEED_COUNT,
                                        num_actions)
                    elif self._mode_total_trits(comp) > num_actions * 4:
                        self._spawn_mode(ck, num_actions)
                        self._interfere(ck, action, +1, self.SEED_COUNT,
                                        num_actions)

        # (6) PRUNING
        if self.frame % max(50, len(self.psi) // 4) == 0:
            self._prune(num_actions)

        self._trace.append((list(active_first), action))
        self._prev_modes = list(observed_modes)
        self._prev_data = dict(config)
        self._update_quality()

    def _prune(self, num_actions: int = 4):
        """Remove undifferentiated composite modes."""
        to_remove = []
        for key, actions in self.psi.items():
            if not self._is_composite(key):
                continue
            nets = [self._net(key, a) for a in actions]
            max_net = max(nets) if nets else 0
            min_net = min(nets) if nets else 0
            total = self._mode_total_trits(key)
            order = self._order(key)
            threshold = num_actions * (1 + order)
            if total <= threshold or (max_net - min_net) <= 1:
                to_remove.append(key)
        for key in to_remove:
            trits = self._mode_total_trits(key)
            self.total_trits -= trits
            del self.psi[key]

    # =================================================================
    # CHOOSE ACTION
    # =================================================================

    def choose_action(self, data, num_actions: int,
                      neighbor_fn=None, query_pos=None) -> int:
        """Boltzmann selection on net magnetization."""
        config = self._normalize_input(data)
        total_net = np.zeros(num_actions, dtype=np.float64)
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
                total_net[a] += weight * self._net(mode, a)

        # Active composite modes
        prev_modes = set()
        if self._prev_modes is not None:
            prev_modes = set(self._prev_modes)
        all_first = observed_modes | prev_modes

        active_modes = set(all_first)
        for _ in range(3):
            next_active = set()
            for key in self.psi:
                if key in active_modes or not self._is_composite(key):
                    continue
                pa, pb = key
                if pa in active_modes and pb in active_modes:
                    next_active.add(key)
            if not next_active:
                break
            active_modes |= next_active

        for key in active_modes:
            if not self._is_composite(key) or key not in self.psi:
                continue
            order = self._order(key)
            weight = 1.0 / order
            for a in range(num_actions):
                total_net[a] += weight * self._net(key, a)

        # Boltzmann selection.
        # Temperature = based on RANGE of nets, not absolute values.
        # This way, a spread of [-450, -400, -350, -500] with range=150
        # gets T=15, making the differences meaningful.
        net_range = np.max(total_net) - np.min(total_net)
        T = max(1.0, net_range / 10.0)
        shifted = total_net / T
        shifted -= np.max(shifted)  # numerical stability
        probs = np.exp(shifted)
        probs = np.maximum(probs, 1e-10)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_actions) / num_actions

        return np.random.choice(num_actions, p=probs)

    # =================================================================
    # SIGNAL
    # =================================================================

    def _signal(self, sign: int):
        """Add trits with given sign to trace modes."""
        if not self._trace:
            return

        for i, (trace_modes, action) in enumerate(reversed(self._trace)):
            credit = max(1, self.SEED_COUNT * 4 // (1 + i) ** 2)
            trace_set = set(trace_modes)

            for mode in trace_modes:
                if mode not in self.psi:
                    continue
                self._interfere(mode, action, sign, credit)

            for key in list(self.psi.keys()):
                if not self._is_composite(key):
                    continue
                leaves = self._leaf_modes(key)
                if leaves.issubset(trace_set):
                    order = self._order(key)
                    ho_credit = max(1, credit // order)
                    self._interfere(key, action, sign, ho_credit)

        self._trace = []

    def signal_death(self):
        self._signal(-1)

    def signal_success(self):
        self._signal(+1)

    # =================================================================
    # RESONANCE QUALITY
    # =================================================================

    def _signal_strength(self) -> float:
        total = 0.0
        for actions in self.psi.values():
            for pm in actions.values():
                net = abs(pm[0] - pm[1])
                if net > 1:
                    total += net - 1
        return total

    def _mode_contrast(self) -> float:
        if not self.psi:
            return 0.0
        contrasts = []
        for actions in self.psi.values():
            if not actions:
                continue
            nets = [abs(pm[0] - pm[1]) for pm in actions.values()]
            max_n = max(nets)
            if max_n <= 1:
                continue
            min_n = min(nets)
            if max_n + min_n > 0:
                contrasts.append((max_n - min_n) / (max_n + min_n))
        return sum(contrasts) / len(contrasts) if contrasts else 0.0

    def _mode_selectivity(self) -> float:
        if not self.psi:
            return 0.0
        total = differentiated = 0
        for actions in self.psi.values():
            if not actions:
                continue
            total += 1
            nets = [abs(pm[0] - pm[1]) for pm in actions.values()]
            max_n, min_n = max(nets), min(nets)
            if max_n > 2 and (max_n - min_n) > 1:
                differentiated += 1
        return differentiated / total if total else 0.0

    def resonance_quality(self) -> float:
        contrast = self._mode_contrast()
        selectivity = self._mode_selectivity()
        signal = np.log1p(self._signal_strength())
        return contrast * selectivity * signal

    def _update_quality(self):
        N = self._n_modes()
        if self.frame % max(1, N) != 0:
            return
        q = self.resonance_quality()
        self._q_history.append(q)
        if q > self._best_q * 1.01 + 0.01:
            self._best_q = q
            self._best_q_frame = self.frame
            self._best_snapshot = self._snapshot()

    def _snapshot(self) -> Dict:
        return {
            'psi': {k: {a: list(pm) for a, pm in v.items()}
                    for k, v in self.psi.items()},
            'total_trits': self.total_trits,
            'frame': self.frame,
        }

    def restore_best(self) -> int:
        if self._best_snapshot is None:
            return self.frame
        snap = self._best_snapshot
        self.psi = defaultdict(dict)
        for k, v in snap['psi'].items():
            self.psi[k] = {a: list(pm) for a, pm in v.items()}
        self.total_trits = snap['total_trits']
        return snap['frame']

    def q_trend(self, window: int = 100) -> float:
        if len(self._q_history) < window + 1:
            return 0.0
        recent = self._q_history[-window:]
        old = self._q_history[-(window + window // 2):-window // 2]
        return ((sum(recent) / len(recent)) - (sum(old) / len(old))
                if old else 0.0)

    # =================================================================
    # COMPAT PROPERTIES
    # =================================================================

    @property
    def total_energy(self):
        return float(self.total_trits)

    @total_energy.setter
    def total_energy(self, val):
        self.total_trits = int(val)

    # =================================================================
    # ANNEALING
    # =================================================================

    def anneal(self, temperature: float = 1.0):
        for key in self.psi:
            for a in self.psi[key]:
                sign = np.random.choice([-1, 1])
                count = max(1, int(temperature))
                self._interfere(key, a, sign, count)

    def cool(self, rate: float = None):
        if rate is None:
            rate = 1.0 / max(1, len(self.psi))
        n_remove = max(1, int(self.total_trits * rate))
        all_keys = list(self.psi.keys())
        if not all_keys:
            return
        for _ in range(n_remove):
            k = all_keys[np.random.randint(len(all_keys))]
            actions = list(self.psi[k].keys())
            if not actions:
                continue
            a = actions[np.random.randint(len(actions))]
            pm = self.psi[k][a]
            if pm[0] + pm[1] > 1:
                if pm[0] >= pm[1]:
                    pm[0] -= 1
                else:
                    pm[1] -= 1
                self.total_trits -= 1

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
            'total_energy': float(self.total_trits),
            'total_trits': self.total_trits,
            'mode_energy': float(self.total_trits),
            'heat_bath': 0.0,
            'n_modes': len(self.psi),
            'n_first_order': first_order,
            'n_curried': higher_order,
            'n_mode_actions': self._n_modes(),
            'n_couplings': higher_order,
            'order_distribution': dict(by_order),
            'resonance_q': q,
            'best_q': self._best_q,
            'best_q_frame': self._best_q_frame,
            'q_trend': self.q_trend(),
        }
