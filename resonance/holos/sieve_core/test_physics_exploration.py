"""
PHYSICS-ALIGNED EXPLORATION TESTS
=================================

Testing the hypothesis: if reality IS the sieve, exploration should be NATURAL.

Physical principles guiding this:
1. ERGODICITY - systems naturally visit all accessible states over time
2. THERMAL EQUILIBRIUM - energy distributes equally among degrees of freedom
3. QUANTUM SUPERPOSITION - all paths explored simultaneously until measurement
4. SYMMETRY BREAKING - structure emerges from random fluctuations, not direction

The sieve shouldn't need "intelligent" exploration - it should emerge from:
- Energy equipartition (all actions get some energy)
- Thermal fluctuations (randomness from heat bath)
- Interference patterns (constructive/destructive based on outcomes)

Three test types:
1. Forced uniform exploration (baseline)
2. External encoders (pretrained embeddings)
3. Pure survival tasks (anthropic principle domain)
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
import hashlib
from collections import defaultdict


class PhysicsSieve:
    """
    Sieve with physics-aligned exploration.

    Key principle: EXPLORATION IS DEFAULT, CONCENTRATION REQUIRES SELECTION.

    Instead of:
        - Choose best action (exploitation)
        - Sometimes choose random (exploration)

    We do:
        - All actions exist in superposition
        - Each observation adds energy to current action
        - Interference patterns emerge from outcomes
        - "Best" action has highest amplitude after interference

    Exploration emerges from:
        - Energy equipartition (thermal equilibrium)
        - Zero-point energy (quantum minimum)
        - Phase randomness (thermal noise)
    """

    def __init__(self):
        # State-action couplings (the learned associations)
        self.couplings: Dict[Tuple[str, str], complex] = defaultdict(complex)

        # Action amplitudes (survival history)
        self.action_amplitudes: Dict[str, complex] = defaultdict(complex)

        # State visitation (for ergodicity tracking)
        self.state_visits: Dict[str, int] = defaultdict(int)

        # Heat bath - thermal energy reservoir
        self.heat_bath: float = 1.0  # Start with some thermal energy

        # Temperature - controls exploration/exploitation
        # High temp = more exploration (thermal fluctuations dominate)
        # Low temp = more exploitation (energy differences matter)
        self.temperature: float = 1.0

        # Zero-point energy - minimum energy per mode (quantum limit)
        self.zero_point: float = 0.1

        # Frame tracking
        self.frame_num: int = 0
        self.current_state: Optional[str] = None
        self.current_action: Optional[int] = None

        # Traces for credit assignment
        self._state_trace: List[str] = []
        self._action_trace: List[int] = []

    def _hash_state(self, state) -> str:
        """Hash state to string ID."""
        if isinstance(state, np.ndarray):
            return hashlib.md5(state.tobytes()).hexdigest()[:12]
        return hashlib.md5(str(state).encode()).hexdigest()[:12]

    def observe(self, state, action: int, num_actions: int = 3):
        """
        Observe state-action pair.

        Physics: Energy flows INTO the observed action.
        All other actions maintain zero-point energy (never truly zero).
        """
        self.frame_num += 1

        state_id = self._hash_state(state)
        self.current_state = state_id
        self.current_action = action

        # Track visits (for ergodicity analysis)
        self.state_visits[state_id] += 1

        # Add to traces
        self._state_trace.append(state_id)
        self._action_trace.append(action)
        if len(self._state_trace) > 100:
            self._state_trace = self._state_trace[-100:]
            self._action_trace = self._action_trace[-100:]

        # === ENERGY INJECTION ===
        # Observed action gets energy (phase encodes timing)
        phase = (self.frame_num * 0.1) % (2 * np.pi)
        energy = complex(1.0 * np.cos(phase), 1.0 * np.sin(phase))

        key = (state_id, action)
        self.couplings[key] += energy
        self.action_amplitudes[action] = self.action_amplitudes.get(action, complex(0)) + energy

        # === ZERO-POINT MAINTENANCE ===
        # All actions maintain minimum energy (quantum vacuum fluctuations)
        for a in range(num_actions):
            amp = abs(self.action_amplitudes.get(a, 0))
            if amp < self.zero_point:
                # Inject zero-point energy with random phase
                zp_phase = np.random.uniform(0, 2 * np.pi)
                self.action_amplitudes[a] = complex(
                    self.zero_point * np.cos(zp_phase),
                    self.zero_point * np.sin(zp_phase)
                )

        # === THERMAL FLUCTUATIONS ===
        # Heat bath causes random energy redistribution
        if self.heat_bath > 0.01:
            thermal_kick = self.heat_bath * 0.01
            for a in range(num_actions):
                # Random phase kick (incoherent thermal noise)
                kick_phase = np.random.uniform(0, 2 * np.pi)
                kick = complex(thermal_kick * np.cos(kick_phase),
                             thermal_kick * np.sin(kick_phase))
                self.action_amplitudes[a] = self.action_amplitudes.get(a, complex(0)) + kick

        # === DAMPING (energy dissipation) ===
        damping = 0.99
        for k in self.couplings:
            self.couplings[k] *= damping
        for a in self.action_amplitudes:
            self.action_amplitudes[a] *= damping

        # Heat bath also dissipates (radiative cooling)
        self.heat_bath *= 0.999

    def signal_outcome(self, success: bool, strength: float = 1.0):
        """
        Signal outcome - this is where selection happens.

        Physics interpretation:
        - Success = constructive interference (amplify)
        - Failure = destructive interference (dampen) + heat generation

        The key insight: failure energy doesn't disappear, it goes to heat bath.
        """
        if not self._state_trace or not self._action_trace:
            return

        last_state = self._state_trace[-1]
        last_action = self._action_trace[-1]
        key = (last_state, last_action)

        if success:
            # CONSTRUCTIVE INTERFERENCE
            # Amplify the state-action coupling
            current = self.couplings.get(key, complex(0))
            boost = complex(strength, 0)  # In-phase addition
            self.couplings[key] = current + boost

            # Also boost action amplitude
            self.action_amplitudes[last_action] = (
                self.action_amplitudes.get(last_action, complex(0)) + boost
            )
        else:
            # DESTRUCTIVE INTERFERENCE
            # Reduce coupling, but energy goes to heat bath
            current = self.couplings.get(key, complex(0))
            current_amp = abs(current)

            # Anti-phase addition (destructive)
            reduction = complex(-strength, 0)
            self.couplings[key] = current + reduction

            # Energy conservation: lost energy goes to heat
            self.heat_bath += strength * 0.5

            # Thermal re-excitation of other actions
            # (failed action's energy redistributes to alternatives)

        # Clear traces on outcome
        self._state_trace = []
        self._action_trace = []

    def choose_action(self, state, num_actions: int = 3) -> int:
        """
        Choose action based on wave interference.

        Physics: This is like a MEASUREMENT - we sample from
        the probability distribution defined by |amplitude|².

        At high temperature: nearly uniform (thermal equilibrium)
        At low temperature: concentrated on highest amplitude
        """
        state_id = self._hash_state(state)

        # Compute "wave function" for each action
        amplitudes = []
        for a in range(num_actions):
            key = (state_id, a)

            # State-action coupling
            coupling = self.couplings.get(key, complex(0))

            # Action's intrinsic amplitude
            action_amp = self.action_amplitudes.get(a, complex(self.zero_point, 0))

            # Total amplitude = coupling + base amplitude
            total = coupling + action_amp * 0.1
            amplitudes.append(total)

        # Probability = |amplitude|² (Born rule)
        probs = np.array([abs(a)**2 for a in amplitudes])

        # Temperature scaling (Boltzmann-like)
        # High temp = flatten distribution
        # Low temp = sharpen peaks
        if self.temperature > 0.01:
            # Convert to "energies" and apply Boltzmann
            energies = -np.log(probs + 1e-10)  # Lower prob = higher energy
            boltzmann = np.exp(-energies / self.temperature)
            probs = boltzmann / (boltzmann.sum() + 1e-10)
        else:
            # Zero temperature = deterministic (highest amplitude wins)
            probs = probs / (probs.sum() + 1e-10)

        # Ensure valid probabilities
        probs = np.clip(probs, 0, 1)
        probs = probs / (probs.sum() + 1e-10)

        return np.random.choice(num_actions, p=probs)

    def get_stats(self) -> Dict:
        """Get statistics."""
        action_amps = {a: abs(self.action_amplitudes.get(a, 0))
                      for a in range(10) if a in self.action_amplitudes}

        return {
            'frame': self.frame_num,
            'num_couplings': len(self.couplings),
            'num_states': len(self.state_visits),
            'heat_bath': self.heat_bath,
            'temperature': self.temperature,
            'action_amplitudes': action_amps,
        }


# =============================================================================
# TEST 1: FORCED UNIFORM EXPLORATION (BASELINE)
# =============================================================================

def test_forced_exploration():
    """
    Test with forced uniform exploration phase.

    This establishes a baseline: if we FORCE the sieve to see
    all state-action pairs equally, can it learn?
    """
    print("=" * 70)
    print("TEST 1: FORCED UNIFORM EXPLORATION")
    print("Force sieve to explore all state-action pairs equally first")
    print("=" * 70)
    print()

    # Simple environment: 5 states, 3 actions
    # Optimal: state 0 -> action 0, state 1 -> action 1, etc (mod 3)
    NUM_STATES = 5
    NUM_ACTIONS = 3

    def optimal_action(state: int) -> int:
        return state % NUM_ACTIONS

    sieve = PhysicsSieve()

    # Phase 1: Forced uniform exploration
    print("Phase 1: Forced exploration (visit each state-action equally)")
    EXPLORATION_ROUNDS = 50

    for round_num in range(EXPLORATION_ROUNDS):
        for state in range(NUM_STATES):
            for action in range(NUM_ACTIONS):
                # Force this state-action
                state_arr = np.array([state])
                sieve.observe(state_arr, action, NUM_ACTIONS)

                # Signal success/failure
                is_optimal = (action == optimal_action(state))
                sieve.signal_outcome(is_optimal, strength=1.0 if is_optimal else 0.5)

    print(f"  Completed {EXPLORATION_ROUNDS} rounds of forced exploration")
    print(f"  Each state-action pair seen {EXPLORATION_ROUNDS} times")

    # Phase 2: Free exploitation
    print()
    print("Phase 2: Free exploitation (sieve chooses)")

    correct = 0
    total = 0

    for trial in range(1000):
        state = np.random.randint(NUM_STATES)
        state_arr = np.array([state])

        action = sieve.choose_action(state_arr, NUM_ACTIONS)
        sieve.observe(state_arr, action, NUM_ACTIONS)

        is_correct = (action == optimal_action(state))
        if is_correct:
            correct += 1
        total += 1

        sieve.signal_outcome(is_correct)

        if (trial + 1) % 200 == 0:
            print(f"  Trial {trial+1}: Accuracy = {100*correct/total:.1f}%")

    print()
    print(f"FINAL: {100*correct/total:.1f}% accuracy")
    print(f"Random baseline: {100/NUM_ACTIONS:.1f}%")

    if correct/total > 0.7:
        print("STATUS: STRONG LEARNING")
    elif correct/total > 0.5:
        print("STATUS: MODERATE LEARNING")
    elif correct/total > 1/NUM_ACTIONS + 0.05:
        print("STATUS: WEAK LEARNING")
    else:
        print("STATUS: NOT LEARNING")

    return correct/total


# =============================================================================
# TEST 2: EXTERNAL ENCODER (Pretrained Embeddings)
# =============================================================================

def test_external_encoder():
    """
    Test with external encoder providing state representations.

    Instead of raw pixels, use a simple pretrained-style embedding.
    This tests if the sieve can learn when given good representations.
    """
    print()
    print("=" * 70)
    print("TEST 2: EXTERNAL ENCODER")
    print("Use structured embeddings instead of raw pixels")
    print("=" * 70)
    print()

    # Simulate a "pretrained encoder" that produces meaningful embeddings
    # For dodgeball: encode (agent_position, threat_relative_position)

    class EncodedDodgeBall:
        def __init__(self, width=5):
            self.width = width
            self.height = 6
            self.reset()

        def reset(self):
            self.agent_x = self.width // 2
            self.balls = []
            self.step_count = 0
            return self.get_encoded_state()

        def get_encoded_state(self) -> np.ndarray:
            """Return encoded state (not pixels)."""
            # Find nearest threat
            danger_row = self.height - 2
            threats = [bx for bx, by in self.balls if by == danger_row]

            if threats:
                closest = min(threats, key=lambda bx: abs(bx - self.agent_x))
                rel_threat = closest - self.agent_x  # -4 to +4
            else:
                rel_threat = 99  # No threat

            # Encoding: [agent_position, relative_threat, edge_flags]
            at_left = 1 if self.agent_x == 0 else 0
            at_right = 1 if self.agent_x == self.width - 1 else 0

            return np.array([self.agent_x, rel_threat, at_left, at_right])

        def step(self, action: int) -> Tuple[np.ndarray, bool]:
            self.step_count += 1

            if action == 0:
                self.agent_x = max(0, self.agent_x - 1)
            elif action == 2:
                self.agent_x = min(self.width - 1, self.agent_x + 1)

            self.balls = [(bx, by + 1) for bx, by in self.balls]

            for bx, by in self.balls:
                if by == self.height - 1 and bx == self.agent_x:
                    return self.get_encoded_state(), True

            self.balls = [(bx, by) for bx, by in self.balls if by < self.height]

            if np.random.random() < 0.3:
                self.balls.append((np.random.randint(0, self.width), 0))

            return self.get_encoded_state(), False

    env = EncodedDodgeBall(width=5)
    sieve = PhysicsSieve()
    sieve.temperature = 2.0  # Start warm for exploration

    NUM_GAMES = 1000
    game_lengths = []

    for game in range(NUM_GAMES):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            action = sieve.choose_action(state, num_actions=3)
            sieve.observe(state, action, num_actions=3)

            prev_state = state
            state, done = env.step(action)
            steps += 1

            # Survival credit (survived this step)
            if not done:
                # Check if was in danger
                rel_threat = prev_state[1]
                if rel_threat in [-1, 0, 1]:
                    # Survived danger! Positive signal
                    sieve.signal_outcome(True, strength=0.3)

        if done:
            # Death signal
            sieve.signal_outcome(False, strength=1.0)

        game_lengths.append(steps)

        # Anneal temperature over time
        sieve.temperature = max(0.5, 2.0 * np.exp(-game / 500))

        if (game + 1) % 400 == 0:
            recent = np.mean(game_lengths[-100:])
            stats = sieve.get_stats()
            print(f"Game {game+1}: Avg={recent:.1f}, Temp={sieve.temperature:.2f}, "
                  f"Heat={stats['heat_bath']:.2f}, States={stats['num_states']}")

    # Results
    print()
    first = np.mean(game_lengths[:200])
    last = np.mean(game_lengths[-200:])
    print(f"First 200: {first:.1f} steps")
    print(f"Last 200: {last:.1f} steps")
    print(f"Improvement: {(last/first - 1)*100:+.1f}%")

    # Random baseline
    random_lengths = []
    for _ in range(200):
        env2 = EncodedDodgeBall()
        state = env2.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            state, done = env2.step(np.random.randint(3))
            steps += 1
        random_lengths.append(steps)

    print(f"Random: {np.mean(random_lengths):.1f} steps")

    if last > np.mean(random_lengths) * 1.3:
        print("STATUS: STRONG LEARNING")
    elif last > np.mean(random_lengths) * 1.1:
        print("STATUS: MODERATE LEARNING")
    else:
        print("STATUS: WEAK/NO LEARNING")

    return last / np.mean(random_lengths)


# =============================================================================
# TEST 3: PURE SURVIVAL (Anthropic Principle Domain)
# =============================================================================

def test_pure_survival():
    """
    Test on truly continuous survival task.

    No discrete "correct" action - just survival.
    The anthropic principle should shine here.
    """
    print()
    print("=" * 70)
    print("TEST 3: PURE SURVIVAL TASK")
    print("No correct answer, just survival = success")
    print("=" * 70)
    print()

    class BalancingTask:
        """
        Balance a pole: stay in the middle region.

        State: position (-10 to +10)
        Actions: 0=left, 1=stay, 2=right
        Death: position reaches edge (-10 or +10)

        Optimal: move toward center
        """
        def __init__(self):
            self.reset()

        def reset(self):
            self.position = np.random.uniform(-3, 3)  # Start near center
            self.velocity = 0.0
            return self.get_state()

        def get_state(self) -> np.ndarray:
            # Encode position and velocity
            return np.array([self.position, self.velocity])

        def step(self, action: int) -> Tuple[np.ndarray, bool]:
            # Action affects velocity
            if action == 0:
                self.velocity -= 0.5
            elif action == 2:
                self.velocity += 0.5

            # Physics: velocity affects position, with friction
            self.velocity *= 0.9
            self.position += self.velocity

            # Random disturbance (makes it challenging)
            self.position += np.random.uniform(-0.3, 0.3)

            # Death at edges
            done = abs(self.position) >= 10

            return self.get_state(), done

    env = BalancingTask()
    sieve = PhysicsSieve()
    sieve.temperature = 3.0  # Start very warm

    NUM_EPISODES = 1500
    episode_lengths = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 500:
            action = sieve.choose_action(state, num_actions=3)
            sieve.observe(state, action, num_actions=3)

            prev_pos = env.position
            state, done = env.step(action)
            steps += 1

            # Survival credit every step
            if not done:
                # Bonus if moved toward center
                moved_toward_center = abs(env.position) < abs(prev_pos)
                sieve.signal_outcome(True, strength=0.1 + (0.2 if moved_toward_center else 0))

        if done:
            sieve.signal_outcome(False, strength=2.0)

        episode_lengths.append(steps)

        # Anneal temperature
        sieve.temperature = max(0.3, 3.0 * np.exp(-episode / 800))

        if (episode + 1) % 500 == 0:
            recent = np.mean(episode_lengths[-200:])
            print(f"Episode {episode+1}: Avg={recent:.1f}, Temp={sieve.temperature:.2f}")

    # Results
    print()
    first = np.mean(episode_lengths[:300])
    last = np.mean(episode_lengths[-300:])
    print(f"First 300: {first:.1f} steps")
    print(f"Last 300: {last:.1f} steps")
    print(f"Improvement: {(last/first - 1)*100:+.1f}%")

    # Random baseline
    random_lengths = []
    for _ in range(300):
        env2 = BalancingTask()
        state = env2.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            state, done = env2.step(np.random.randint(3))
            steps += 1
        random_lengths.append(steps)

    print(f"Random: {np.mean(random_lengths):.1f} steps")

    if last > np.mean(random_lengths) * 1.5:
        print("STATUS: STRONG LEARNING")
    elif last > np.mean(random_lengths) * 1.2:
        print("STATUS: MODERATE LEARNING")
    elif last > np.mean(random_lengths) * 1.05:
        print("STATUS: WEAK LEARNING")
    else:
        print("STATUS: NOT LEARNING")

    return last / np.mean(random_lengths)


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_physics_tests():
    """Run all physics-aligned exploration tests."""
    print()
    print("#" * 70)
    print("# PHYSICS-ALIGNED EXPLORATION TESTS")
    print("# Testing if exploration can emerge naturally from physics")
    print("#" * 70)

    results = {}

    results['forced'] = test_forced_exploration()
    results['encoder'] = test_external_encoder()
    results['survival'] = test_pure_survival()

    print()
    print("#" * 70)
    print("# SUMMARY")
    print("#" * 70)
    print()
    print(f"Forced Exploration: {100*results['forced']:.1f}% accuracy")
    print(f"External Encoder:   {results['encoder']:.2f}x random")
    print(f"Pure Survival:      {results['survival']:.2f}x random")
    print()

    return results


if __name__ == "__main__":
    run_all_physics_tests()
