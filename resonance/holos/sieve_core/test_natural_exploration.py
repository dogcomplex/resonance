"""
NATURAL EXPLORATION MECHANISMS
==============================

Testing 4 physics-aligned approaches to exploration:

1. TRUE ERGODICITY - Visit unvisited states with probability ∝ 1/visits
   (Emerges from dynamics, not forced)

2. THERMAL FLOOR - Minimum temperature prevents complete freezing
   (Quantum vacuum fluctuations, T > 0 always)

3. SYMMETRY PRESERVATION - All actions equal until evidence breaks symmetry
   (Spontaneous symmetry breaking)

4. SUPERPOSITION - Actions exist simultaneously, collapse on outcome
   (Quantum measurement model)

Goal: Find which mechanism produces natural exploration that is:
a) Most physically realistic
b) Most performant
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import hashlib
from collections import defaultdict


# =============================================================================
# BASE SIEVE (shared infrastructure)
# =============================================================================

class BaseSieve:
    """Common infrastructure for all sieve variants."""

    def __init__(self):
        self.couplings: Dict[Tuple[str, str], complex] = defaultdict(complex)
        self.action_amplitudes: Dict[int, complex] = defaultdict(complex)
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.state_action_visits: Dict[Tuple[str, int], int] = defaultdict(int)

        self.frame_num = 0
        self.current_state: Optional[str] = None

        self._state_trace: List[str] = []
        self._action_trace: List[int] = []

    def _hash(self, state) -> str:
        if isinstance(state, np.ndarray):
            return hashlib.md5(state.tobytes()).hexdigest()[:12]
        return hashlib.md5(str(state).encode()).hexdigest()[:12]

    def _damping(self):
        """Universal damping."""
        d = 0.995
        for k in self.couplings:
            self.couplings[k] *= d
        for k in self.action_amplitudes:
            self.action_amplitudes[k] *= d

    def signal_outcome(self, success: bool, strength: float = 1.0):
        """Signal success/failure."""
        if not self._state_trace or not self._action_trace:
            return

        last_state = self._state_trace[-1]
        last_action = self._action_trace[-1]
        key = (last_state, last_action)

        if success:
            self.couplings[key] += complex(strength, 0)
            self.action_amplitudes[last_action] += complex(strength * 0.1, 0)
        else:
            self.couplings[key] -= complex(strength, 0)

        self._state_trace = []
        self._action_trace = []


# =============================================================================
# 1. ERGODIC SIEVE - Visit unvisited states naturally
# =============================================================================

class ErgodicSieve(BaseSieve):
    """
    Ergodicity emerges from visit-counting.

    Physics principle: Given infinite time, an ergodic system visits
    all accessible states. We approximate this by boosting exploration
    to rarely-visited state-action pairs.

    The boost is 1/sqrt(visits) - diminishing but never zero.
    """

    def observe(self, state, action: int, num_actions: int = 3):
        self.frame_num += 1
        state_id = self._hash(state)
        self.current_state = state_id

        # Track visits
        self.state_visits[state_id] += 1
        self.state_action_visits[(state_id, action)] += 1

        # Traces
        self._state_trace.append(state_id)
        self._action_trace.append(action)
        if len(self._state_trace) > 50:
            self._state_trace = self._state_trace[-50:]
            self._action_trace = self._action_trace[-50:]

        # Energy injection
        phase = (self.frame_num * 0.1) % (2 * np.pi)
        self.couplings[(state_id, action)] += complex(np.cos(phase), np.sin(phase))
        self.action_amplitudes[action] += complex(0.1, 0)

        self._damping()

    def choose_action(self, state, num_actions: int = 3) -> int:
        state_id = self._hash(state)

        scores = []
        for a in range(num_actions):
            key = (state_id, a)

            # Base score from couplings
            coupling = abs(self.couplings.get(key, 0))
            amplitude = abs(self.action_amplitudes.get(a, 0))
            base_score = coupling + amplitude * 0.1

            # ERGODIC BOOST: 1/sqrt(visits) - explores unvisited
            visits = self.state_action_visits.get(key, 0) + 1
            ergodic_boost = 1.0 / np.sqrt(visits)

            scores.append(base_score + ergodic_boost)

        # Softmax
        scores = np.array(scores)
        scores = scores - scores.max()
        exp_scores = np.exp(scores / 0.5)
        probs = exp_scores / exp_scores.sum()

        return np.random.choice(num_actions, p=probs)


# =============================================================================
# 2. THERMAL FLOOR SIEVE - Minimum temperature
# =============================================================================

class ThermalFloorSieve(BaseSieve):
    """
    Maintains minimum temperature (thermal floor).

    Physics principle: Real systems never reach T=0 (third law of
    thermodynamics). Quantum vacuum fluctuations ensure minimum energy.

    We implement a temperature that can't drop below a floor.
    """

    def __init__(self):
        super().__init__()
        self.temperature = 2.0  # Current temperature
        self.temp_floor = 0.3   # Minimum temperature (vacuum fluctuations)
        self.heat_bath = 1.0    # Thermal reservoir

    def observe(self, state, action: int, num_actions: int = 3):
        self.frame_num += 1
        state_id = self._hash(state)
        self.current_state = state_id

        self.state_visits[state_id] += 1

        self._state_trace.append(state_id)
        self._action_trace.append(action)
        if len(self._state_trace) > 50:
            self._state_trace = self._state_trace[-50:]
            self._action_trace = self._action_trace[-50:]

        # Energy injection
        self.couplings[(state_id, action)] += complex(1.0, 0)
        self.action_amplitudes[action] += complex(0.1, 0)

        # Thermal fluctuations - random kicks to all actions
        for a in range(num_actions):
            kick = np.random.randn() * np.sqrt(self.temperature) * 0.1
            self.action_amplitudes[a] += complex(kick, 0)

        # Temperature dynamics (cooling but with floor)
        self.temperature = max(self.temp_floor,
                              self.temperature * 0.9995)

        self._damping()

    def signal_outcome(self, success: bool, strength: float = 1.0):
        super().signal_outcome(success, strength)

        # Failure heats up the system (mismatch → thermal energy)
        if not success:
            self.temperature = min(2.0, self.temperature + 0.1)
            self.heat_bath += strength * 0.5

    def choose_action(self, state, num_actions: int = 3) -> int:
        state_id = self._hash(state)

        scores = []
        for a in range(num_actions):
            key = (state_id, a)
            coupling = abs(self.couplings.get(key, 0))
            amplitude = abs(self.action_amplitudes.get(a, 0))
            scores.append(coupling + amplitude * 0.1 + 0.01)

        scores = np.array(scores)
        scores = scores - scores.max()
        # Temperature-scaled softmax
        exp_scores = np.exp(scores / self.temperature)
        probs = exp_scores / (exp_scores.sum() + 1e-10)

        return np.random.choice(num_actions, p=probs)


# =============================================================================
# 3. SYMMETRY SIEVE - Preserve symmetry until broken
# =============================================================================

class SymmetrySieve(BaseSieve):
    """
    Symmetry preserved until broken by evidence.

    Physics principle: The universe starts maximally symmetric.
    Structure emerges through spontaneous symmetry breaking when
    fluctuations get amplified by dynamics.

    We maintain a "symmetry score" per state. High symmetry = uniform
    action distribution. Evidence breaks symmetry gradually.
    """

    def __init__(self):
        super().__init__()
        self.symmetry: Dict[str, float] = defaultdict(lambda: 1.0)  # Start symmetric
        self.evidence: Dict[str, float] = defaultdict(float)  # Accumulated evidence

    def observe(self, state, action: int, num_actions: int = 3):
        self.frame_num += 1
        state_id = self._hash(state)
        self.current_state = state_id

        self.state_visits[state_id] += 1

        self._state_trace.append(state_id)
        self._action_trace.append(action)
        if len(self._state_trace) > 50:
            self._state_trace = self._state_trace[-50:]
            self._action_trace = self._action_trace[-50:]

        # Energy injection
        self.couplings[(state_id, action)] += complex(1.0, 0)
        self.action_amplitudes[action] += complex(0.1, 0)

        self._damping()

    def signal_outcome(self, success: bool, strength: float = 1.0):
        """Outcome provides evidence that breaks symmetry."""
        if not self._state_trace or not self._action_trace:
            return

        last_state = self._state_trace[-1]
        last_action = self._action_trace[-1]
        key = (last_state, last_action)

        if success:
            self.couplings[key] += complex(strength, 0)
            # Success breaks symmetry - this action is special
            self.evidence[last_state] += strength
        else:
            self.couplings[key] -= complex(strength, 0)
            # Failure also breaks symmetry
            self.evidence[last_state] += strength * 0.5

        # Symmetry decreases as evidence accumulates
        # symmetry = 1 / (1 + evidence) → approaches 0 with more evidence
        self.symmetry[last_state] = 1.0 / (1.0 + self.evidence[last_state])

        self._state_trace = []
        self._action_trace = []

    def choose_action(self, state, num_actions: int = 3) -> int:
        state_id = self._hash(state)
        sym = self.symmetry[state_id]

        # High symmetry → uniform distribution
        # Low symmetry → follow couplings

        scores = []
        for a in range(num_actions):
            key = (state_id, a)
            coupling = abs(self.couplings.get(key, 0))
            amplitude = abs(self.action_amplitudes.get(a, 0))

            # Blend: (1-sym)*learned + sym*uniform
            learned_score = coupling + amplitude * 0.1
            uniform_score = 1.0 / num_actions

            blended = (1 - sym) * learned_score + sym * uniform_score
            scores.append(blended + 0.01)

        scores = np.array(scores)
        probs = scores / scores.sum()

        return np.random.choice(num_actions, p=probs)


# =============================================================================
# 4. SUPERPOSITION SIEVE - Quantum-style parallel exploration
# =============================================================================

class SuperpositionSieve(BaseSieve):
    """
    Actions exist in superposition until outcome collapses them.

    Physics principle: In quantum mechanics, a system exists in all
    possible states simultaneously until measured. The outcome
    "collapses" the wave function.

    Implementation: We track a "wave function" over actions. Each
    observation adds amplitude. Outcomes cause collapse (amplify
    chosen, dampen others).
    """

    def __init__(self):
        super().__init__()
        # Wave function: state -> action -> complex amplitude
        self.wave_function: Dict[str, Dict[int, complex]] = defaultdict(
            lambda: defaultdict(complex)
        )
        self.collapsed: Dict[str, bool] = defaultdict(bool)

    def observe(self, state, action: int, num_actions: int = 3):
        self.frame_num += 1
        state_id = self._hash(state)
        self.current_state = state_id

        self.state_visits[state_id] += 1

        self._state_trace.append(state_id)
        self._action_trace.append(action)
        if len(self._state_trace) > 50:
            self._state_trace = self._state_trace[-50:]
            self._action_trace = self._action_trace[-50:]

        # Add amplitude to chosen action
        phase = (self.frame_num * 0.1) % (2 * np.pi)
        self.wave_function[state_id][action] += complex(np.cos(phase), np.sin(phase))

        # Also add small amplitude to ALL actions (superposition spreading)
        for a in range(num_actions):
            self.wave_function[state_id][a] += complex(0.05, 0)

        # Standard coupling update
        self.couplings[(state_id, action)] += complex(1.0, 0)

        self._damping()

        # Wave function also decoheres (damping)
        for s in self.wave_function:
            for a in self.wave_function[s]:
                self.wave_function[s][a] *= 0.99

    def signal_outcome(self, success: bool, strength: float = 1.0):
        """Outcome causes wave function collapse."""
        if not self._state_trace or not self._action_trace:
            return

        last_state = self._state_trace[-1]
        last_action = self._action_trace[-1]

        # COLLAPSE: The chosen action's amplitude dominates
        if success:
            # Constructive collapse - chosen action amplified
            self.wave_function[last_state][last_action] *= 2.0

            # Others get reduced (destructive interference with chosen path)
            for a in self.wave_function[last_state]:
                if a != last_action:
                    self.wave_function[last_state][a] *= 0.5
        else:
            # Failure: chosen action suppressed, others boosted
            self.wave_function[last_state][last_action] *= 0.3

            # "Path not taken" gets chance
            for a in self.wave_function[last_state]:
                if a != last_action:
                    self.wave_function[last_state][a] *= 1.2

        # Also update couplings
        key = (last_state, last_action)
        if success:
            self.couplings[key] += complex(strength, 0)
        else:
            self.couplings[key] -= complex(strength, 0)

        self._state_trace = []
        self._action_trace = []

    def choose_action(self, state, num_actions: int = 3) -> int:
        state_id = self._hash(state)

        # Probability from wave function: P(a) = |ψ(a)|²
        amplitudes = []
        for a in range(num_actions):
            amp = self.wave_function[state_id].get(a, complex(0.1, 0))
            amplitudes.append(abs(amp))

        # Born rule: probability ∝ |amplitude|²
        probs = np.array([a**2 for a in amplitudes])
        total = probs.sum()
        if total < 1e-10:
            probs = np.ones(num_actions) / num_actions  # uniform if collapsed
        else:
            probs = probs / total

        # Ensure valid probability distribution
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum()

        return np.random.choice(num_actions, p=probs)


# =============================================================================
# TEST ENVIRONMENT
# =============================================================================

class SimpleTask:
    """
    Simple state-action mapping task.
    5 states, 3 actions, optimal mapping known.
    """
    def __init__(self):
        self.num_states = 5
        self.num_actions = 3

    def optimal(self, state: int) -> int:
        return state % self.num_actions

    def get_state(self, state_idx: int) -> np.ndarray:
        return np.array([state_idx])


class DodgeBallTask:
    """Dodgeball with encoded states."""
    def __init__(self, width=5):
        self.width = width
        self.height = 6
        self.reset()

    def reset(self):
        self.agent_x = self.width // 2
        self.balls = []
        return self.get_state()

    def get_state(self) -> np.ndarray:
        danger_row = self.height - 2
        threats = [bx for bx, by in self.balls if by == danger_row]

        if threats:
            closest = min(threats, key=lambda bx: abs(bx - self.agent_x))
            rel = closest - self.agent_x
        else:
            rel = 99

        edge = 0 if self.agent_x == 0 else (2 if self.agent_x == self.width - 1 else 1)
        return np.array([rel, edge, self.agent_x])

    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        if action == 0:
            self.agent_x = max(0, self.agent_x - 1)
        elif action == 2:
            self.agent_x = min(self.width - 1, self.agent_x + 1)

        self.balls = [(bx, by + 1) for bx, by in self.balls]

        for bx, by in self.balls:
            if by == self.height - 1 and bx == self.agent_x:
                return self.get_state(), True

        self.balls = [(bx, by) for bx, by in self.balls if by < self.height]

        if np.random.random() < 0.3:
            self.balls.append((np.random.randint(self.width), 0))

        return self.get_state(), False


# =============================================================================
# TEST RUNNER
# =============================================================================

def test_sieve_on_simple(SieveClass, name: str, num_trials: int = 2000):
    """Test a sieve on simple mapping task."""
    print(f"\n--- {name} ---")

    task = SimpleTask()
    sieve = SieveClass()

    correct = 0
    total = 0
    recent = []

    for trial in range(num_trials):
        state_idx = np.random.randint(task.num_states)
        state = task.get_state(state_idx)
        optimal = task.optimal(state_idx)

        action = sieve.choose_action(state, task.num_actions)
        sieve.observe(state, action, task.num_actions)

        is_correct = (action == optimal)
        sieve.signal_outcome(is_correct)

        if is_correct:
            correct += 1
        total += 1

        recent.append(1 if is_correct else 0)
        if len(recent) > 100:
            recent.pop(0)

        if (trial + 1) % 500 == 0:
            print(f"  Trial {trial+1}: Overall={100*correct/total:.1f}%, "
                  f"Recent100={100*sum(recent)/len(recent):.1f}%")

    return correct / total


def test_sieve_on_dodgeball(SieveClass, name: str, num_games: int = 1000):
    """Test a sieve on dodgeball survival."""
    print(f"\n--- {name} ---")

    sieve = SieveClass()
    game_lengths = []

    for game in range(num_games):
        env = DodgeBallTask()
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            action = sieve.choose_action(state, num_actions=3)
            sieve.observe(state, action, num_actions=3)

            prev_state = state
            state, done = env.step(action)
            steps += 1

            # Survival credit
            if not done and prev_state[0] in [-1, 0, 1]:
                sieve.signal_outcome(True, strength=0.3)

        if done:
            sieve.signal_outcome(False, strength=1.0)

        game_lengths.append(steps)

        if (game + 1) % 250 == 0:
            recent = np.mean(game_lengths[-100:])
            print(f"  Game {game+1}: Recent avg = {recent:.1f}")

    return np.mean(game_lengths[-200:])


def run_all_tests():
    """Compare all four exploration mechanisms."""
    print()
    print("#" * 70)
    print("# NATURAL EXPLORATION MECHANISMS COMPARISON")
    print("#" * 70)

    sieves = [
        (ErgodicSieve, "1. ERGODIC (visit-counting)"),
        (ThermalFloorSieve, "2. THERMAL FLOOR (T > 0 always)"),
        (SymmetrySieve, "3. SYMMETRY (preserve until broken)"),
        (SuperpositionSieve, "4. SUPERPOSITION (quantum collapse)"),
    ]

    # Test 1: Simple mapping task
    print()
    print("=" * 70)
    print("TEST A: SIMPLE MAPPING (5 states -> 3 actions)")
    print("=" * 70)

    simple_results = {}
    for SieveClass, name in sieves:
        acc = test_sieve_on_simple(SieveClass, name)
        simple_results[name] = acc

    # Test 2: Dodgeball survival
    print()
    print("=" * 70)
    print("TEST B: DODGEBALL SURVIVAL")
    print("=" * 70)

    dodge_results = {}
    for SieveClass, name in sieves:
        avg_len = test_sieve_on_dodgeball(SieveClass, name)
        dodge_results[name] = avg_len

    # Random baselines
    print()
    print("Computing baselines...")

    # Simple random
    correct = sum(np.random.randint(3) == (s % 3) for s in
                 [np.random.randint(5) for _ in range(1000)])
    random_simple = correct / 1000

    # Dodgeball random
    random_lengths = []
    for _ in range(200):
        env = DodgeBallTask()
        env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            _, done = env.step(np.random.randint(3))
            steps += 1
        random_lengths.append(steps)
    random_dodge = np.mean(random_lengths)

    # Summary
    print()
    print("#" * 70)
    print("# RESULTS SUMMARY")
    print("#" * 70)
    print()
    print(f"{'Mechanism':<40} {'Simple Acc':<15} {'Dodge Survival':<15}")
    print("-" * 70)

    for name in simple_results:
        simple = simple_results[name]
        dodge = dodge_results[name]
        print(f"{name:<40} {100*simple:>6.1f}%        {dodge:>6.1f} steps")

    print("-" * 70)
    print(f"{'RANDOM BASELINE':<40} {100*random_simple:>6.1f}%        {random_dodge:>6.1f} steps")
    print()

    # Winner determination
    best_simple = max(simple_results, key=simple_results.get)
    best_dodge = max(dodge_results, key=dodge_results.get)

    print(f"Best on Simple: {best_simple} ({100*simple_results[best_simple]:.1f}%)")
    print(f"Best on Dodge:  {best_dodge} ({dodge_results[best_dodge]:.1f} steps)")

    return simple_results, dodge_results


if __name__ == "__main__":
    run_all_tests()
