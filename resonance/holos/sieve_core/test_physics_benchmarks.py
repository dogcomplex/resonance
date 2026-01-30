"""
PHYSICS BENCHMARKS - Universal sieve validation
================================================
Real-world physics and math tasks, not games.
Each has a known optimal/baseline for comparison.

1. Boolean functions (parity, majority) — combinatorial logic
2. Binary sequence prediction (Markov, noisy periodic) — information theory
3. Logistic map prediction — chaos / temporal patterns
4. Pendulum phase prediction — smooth physics dynamics
5. Non-stationary bandit — exploration / exploitation
"""

import numpy as np
from wave_sieve import WaveSieve


# ============================================================
# 1. BOOLEAN FUNCTIONS
# ============================================================

def test_boolean(fn_name, fn, n_bits=5, n_trials=5000):
    """Learn a boolean function from examples.
    Observation: binary vector. Action: 0 or 1. Feedback: correct/wrong."""
    print(f"\n  {fn_name} (n={n_bits}):")
    sieve = WaveSieve()
    correct = 0
    window = []

    for trial in range(n_trials):
        # Random binary input
        bits = tuple(np.random.randint(0, 2, size=n_bits))
        target = fn(bits)

        config = {i: bits[i] for i in range(n_bits)}

        if trial < 200:
            action = np.random.randint(0, 2)
        else:
            action = sieve.choose_action(config, num_actions=2)

        sieve.observe(config, action, num_actions=2)

        if action == target:
            correct += 1
            sieve.signal_success()
        else:
            sieve.signal_death()

        window.append(1 if action == target else 0)
        if len(window) > 500:
            window.pop(0)

        if (trial + 1) % 1000 == 0:
            recent = 100 * sum(window) / len(window)
            print(f"    Trial {trial+1}: Recent500={recent:.1f}%")

    final = 100 * sum(window) / len(window)
    print(f"    FINAL: {final:.1f}% (baseline: 50%)")
    return final


def parity(bits):
    return sum(bits) % 2


def majority(bits):
    return 1 if sum(bits) > len(bits) / 2 else 0


print("=" * 60)
print("BENCHMARK 1: BOOLEAN FUNCTIONS")
print("=" * 60)

maj_score = test_boolean("Majority-5", majority, n_bits=5)
par_score = test_boolean("Parity-5", parity, n_bits=5)


# ============================================================
# 2. BINARY SEQUENCE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("BENCHMARK 2: BINARY SEQUENCE PREDICTION")
print("=" * 60)


def test_markov_prediction(order=2, n_steps=10000):
    """Predict next bit from a Markov chain of given order."""
    print(f"\n  Markov order-{order}:")

    # Random transition probabilities
    n_contexts = 2 ** order
    probs = np.random.uniform(0.1, 0.9, size=n_contexts)
    # Compute entropy rate (optimal accuracy)
    entropy = 0
    for p in probs:
        entropy -= (p * np.log2(p + 1e-10) + (1 - p) * np.log2(1 - p + 1e-10))
    entropy /= n_contexts
    optimal_acc = 100 * np.mean(np.maximum(probs, 1 - probs))
    print(f"    Optimal accuracy: {optimal_acc:.1f}%")

    sieve = WaveSieve()
    history = [np.random.randint(0, 2) for _ in range(order)]
    window = []

    for step in range(n_steps):
        # Current context
        context = tuple(history[-order:])
        context_idx = sum(b * (2 ** i) for i, b in enumerate(context))

        # Generate next bit
        next_bit = 1 if np.random.random() < probs[context_idx] else 0

        # Sieve predicts
        config = {i: context[i] for i in range(order)}
        if step < 200:
            action = np.random.randint(0, 2)
        else:
            action = sieve.choose_action(config, num_actions=2)

        sieve.observe(config, action, num_actions=2)

        if action == next_bit:
            sieve.signal_success()
            window.append(1)
        else:
            sieve.signal_death()
            window.append(0)

        history.append(next_bit)
        if len(window) > 500:
            window.pop(0)

        if (step + 1) % 2000 == 0:
            recent = 100 * sum(window) / len(window)
            print(f"    Step {step+1}: Recent500={recent:.1f}%")

    final = 100 * sum(window) / len(window)
    print(f"    FINAL: {final:.1f}% (optimal: {optimal_acc:.1f}%, baseline: 50%)")
    return final, optimal_acc


def test_noisy_periodic(period=5, noise=0.05, n_steps=10000):
    """Predict next bit from a repeating pattern with noise."""
    print(f"\n  Noisy periodic (period={period}, noise={noise}):")
    pattern = [np.random.randint(0, 2) for _ in range(period)]
    optimal_acc = 100 * (1 - noise)
    print(f"    Pattern: {pattern}, Optimal: {optimal_acc:.1f}%")

    sieve = WaveSieve()
    window = []

    for step in range(n_steps):
        # Generate bit (pattern + noise)
        true_bit = pattern[step % period]
        if np.random.random() < noise:
            true_bit = 1 - true_bit

        # Context: position in sequence (modulo window)
        context_window = 3
        context = {}
        for k in range(context_window):
            if step - k - 1 >= 0:
                # Use previous bits as context
                prev_idx = (step - k - 1) % period
                context[k] = pattern[prev_idx] if np.random.random() > noise else 1 - pattern[prev_idx]

        if not context:
            context = {0: 0}

        if step < 200:
            action = np.random.randint(0, 2)
        else:
            action = sieve.choose_action(context, num_actions=2)

        sieve.observe(context, action, num_actions=2)

        if action == true_bit:
            sieve.signal_success()
            window.append(1)
        else:
            sieve.signal_death()
            window.append(0)

        if len(window) > 500:
            window.pop(0)

        if (step + 1) % 2000 == 0:
            recent = 100 * sum(window) / len(window)
            print(f"    Step {step+1}: Recent500={recent:.1f}%")

    final = 100 * sum(window) / len(window)
    print(f"    FINAL: {final:.1f}% (optimal: {optimal_acc:.1f}%, baseline: 50%)")
    return final, optimal_acc


markov_score, markov_optimal = test_markov_prediction(order=2)
periodic_score, periodic_optimal = test_noisy_periodic(period=5, noise=0.05)


# ============================================================
# 3. LOGISTIC MAP PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("BENCHMARK 3: LOGISTIC MAP (edge of chaos)")
print("=" * 60)


def test_logistic_map(r=3.9, n_bins=8, window_size=3, n_steps=10000):
    """Predict which bin the next value falls in."""
    print(f"\n  r={r}, bins={n_bins}, window={window_size}:")

    sieve = WaveSieve()
    x = np.random.random()
    history = []
    window = []

    for step in range(n_steps):
        x_next = r * x * (1 - x)
        current_bin = min(int(x * n_bins), n_bins - 1)
        next_bin = min(int(x_next * n_bins), n_bins - 1)

        history.append(current_bin)
        if len(history) > window_size:
            history.pop(0)

        config = {i: history[i] for i in range(len(history))}

        if step < 200:
            action = np.random.randint(0, n_bins)
        else:
            action = sieve.choose_action(config, num_actions=n_bins)

        sieve.observe(config, action, num_actions=n_bins)

        if action == next_bin:
            sieve.signal_success()
            window.append(1)
        else:
            sieve.signal_death()
            window.append(0)

        x = x_next
        if len(window) > 500:
            window.pop(0)

        if (step + 1) % 2000 == 0:
            recent = 100 * sum(window) / len(window)
            print(f"    Step {step+1}: Recent500={recent:.1f}%")

    final = 100 * sum(window) / len(window)
    baseline = 100.0 / n_bins
    print(f"    FINAL: {final:.1f}% (baseline: {baseline:.1f}%)")
    return final


logistic_score = test_logistic_map()


# ============================================================
# 4. PENDULUM PHASE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("BENCHMARK 4: PENDULUM PHASE SPACE")
print("=" * 60)


def test_pendulum(n_grid=8, n_steps=10000):
    """Predict next phase-space cell of a simple pendulum."""
    print(f"\n  Grid: {n_grid}x{n_grid} = {n_grid**2} cells:")
    g_L = 9.81  # g/L
    dt = 0.05
    n_cells = n_grid * n_grid

    def state_to_cell(theta, omega):
        # Map theta to [0, 2*pi), omega to [-6, 6]
        t_norm = (theta % (2 * np.pi)) / (2 * np.pi)
        o_norm = (omega + 6) / 12.0
        o_norm = np.clip(o_norm, 0, 0.999)
        t_idx = min(int(t_norm * n_grid), n_grid - 1)
        o_idx = min(int(o_norm * n_grid), n_grid - 1)
        return t_idx * n_grid + o_idx

    sieve = WaveSieve()
    window = []
    # Start with random initial condition
    theta = np.random.uniform(0, 2 * np.pi)
    omega = np.random.uniform(-4, 4)

    for step in range(n_steps):
        current_cell = state_to_cell(theta, omega)

        # RK4 integration
        def deriv(th, om):
            return om, -g_L * np.sin(th)

        k1t, k1o = deriv(theta, omega)
        k2t, k2o = deriv(theta + dt / 2 * k1t, omega + dt / 2 * k1o)
        k3t, k3o = deriv(theta + dt / 2 * k2t, omega + dt / 2 * k2o)
        k4t, k4o = deriv(theta + dt * k3t, omega + dt * k3o)

        theta_new = theta + dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        omega_new = omega + dt / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)

        next_cell = state_to_cell(theta_new, omega_new)

        config = {0: current_cell}

        if step < 200:
            action = np.random.randint(0, n_cells)
        else:
            action = sieve.choose_action(config, num_actions=n_cells)

        sieve.observe(config, action, num_actions=n_cells)

        if action == next_cell:
            sieve.signal_success()
            window.append(1)
        else:
            sieve.signal_death()
            window.append(0)

        theta, omega = theta_new, omega_new
        if len(window) > 500:
            window.pop(0)

        # Occasionally reset trajectory (new initial conditions)
        if (step + 1) % 500 == 0:
            theta = np.random.uniform(0, 2 * np.pi)
            omega = np.random.uniform(-4, 4)
            sieve.reset_episode()

        if (step + 1) % 2000 == 0:
            recent = 100 * sum(window) / len(window)
            print(f"    Step {step+1}: Recent500={recent:.1f}%")

    final = 100 * sum(window) / len(window)
    baseline = 100.0 / n_cells
    print(f"    FINAL: {final:.1f}% (baseline: {baseline:.1f}%)")
    return final


pendulum_score = test_pendulum()


# ============================================================
# 5. NON-STATIONARY BANDIT
# ============================================================

print("\n" + "=" * 60)
print("BENCHMARK 5: NON-STATIONARY BANDIT")
print("=" * 60)


def test_bandit(n_arms=10, n_steps=10000, drift=0.01):
    """Pull arms with drifting reward probabilities."""
    print(f"\n  Arms={n_arms}, drift={drift}:")

    # Initialize arm probabilities
    probs = np.random.uniform(0.2, 0.8, size=n_arms)

    sieve = WaveSieve()
    window_reward = []
    oracle_reward = []

    for step in range(n_steps):
        # Drift probabilities
        probs += np.random.normal(0, drift, size=n_arms)
        probs = np.clip(probs, 0.05, 0.95)

        # Oracle pulls best arm
        oracle_reward.append(probs[np.argmax(probs)])

        # Context: last arm pulled and result (or nothing for first step)
        config = {0: step % n_arms}  # Simple: just the step modulo
        # Better: provide recent history
        if step > 0:
            config = {0: last_arm, 1: last_reward}

        if step < 200:
            action = np.random.randint(0, n_arms)
        else:
            action = sieve.choose_action(config, num_actions=n_arms)

        # Get reward
        reward = 1 if np.random.random() < probs[action] else 0

        sieve.observe(config, action, num_actions=n_arms)
        if reward:
            sieve.signal_success()
        else:
            sieve.signal_death()

        last_arm = action
        last_reward = reward
        window_reward.append(reward)
        if len(window_reward) > 500:
            window_reward.pop(0)

        if (step + 1) % 2000 == 0:
            recent = 100 * sum(window_reward) / len(window_reward)
            oracle_recent = 100 * np.mean(oracle_reward[-500:])
            print(f"    Step {step+1}: Reward={recent:.1f}% "
                  f"Oracle={oracle_recent:.1f}%")

    final = 100 * sum(window_reward) / len(window_reward)
    oracle_final = 100 * np.mean(oracle_reward[-500:])
    random_baseline = 100 * np.mean(probs)
    print(f"    FINAL: {final:.1f}% (oracle: {oracle_final:.1f}%, "
          f"random: {random_baseline:.1f}%)")
    return final


bandit_score = test_bandit()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PHYSICS BENCHMARK SUMMARY")
print("=" * 60)
print(f"  Majority-5:        {maj_score:.1f}%  (baseline: 50%)")
print(f"  Parity-5:          {par_score:.1f}%  (baseline: 50%)")
print(f"  Markov order-2:    {markov_score:.1f}%  (optimal: {markov_optimal:.1f}%)")
print(f"  Noisy periodic:    {periodic_score:.1f}%  (optimal: {periodic_optimal:.1f}%)")
print(f"  Logistic map:      {logistic_score:.1f}%  (baseline: {100.0/8:.1f}%)")
print(f"  Pendulum phase:    {pendulum_score:.1f}%  (baseline: {100.0/64:.1f}%)")
print(f"  Bandit:            {bandit_score:.1f}%")
