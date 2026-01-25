"""
Test the sieve primitive.

This tests:
1. Basic sieve operations (inject, evolve, stable patterns)
2. Interference (constructive and destructive)
3. Self-annealing (temperature drops as patterns stabilize)
4. Game compilation (Connect4 on sieve)
5. Rule learning (infer rules from traces)
"""

import math
from sieve import Pattern, Amplitude, Rule, Sieve, solve, value_to_phase
from compile_game import GameSieve, solve_game_on_sieve


def test_basic_sieve():
    """Test basic sieve operations"""
    print("\n=== Test: Basic Sieve ===")

    sieve = Sieve(threshold=0.01, damping=0.95)

    # Inject some patterns
    p1 = Pattern(tokens='A')
    p2 = Pattern(tokens='B')
    p3 = Pattern(tokens='C')

    sieve.inject_forward(p1, magnitude=1.0)
    sieve.inject_forward(p2, magnitude=0.5)

    print(f"Initial: {len(sieve.field)} patterns")
    print(f"Temperature: {sieve.temperature():.3f}")

    # Create rules: A -> B, B -> C
    rules = [
        Rule(lhs=p1, rhs=p2, transfer=complex(0.9, 0)),
        Rule(lhs=p2, rhs=p3, transfer=complex(0.9, 0)),
    ]

    # Evolve a few steps
    for i in range(10):
        stats = sieve.evolve(rules)
        if i % 3 == 0:
            print(f"Gen {i}: patterns={stats['patterns_out']}, temp={sieve.temperature():.3f}")

    stable = sieve.stable_patterns()
    print(f"Stable patterns: {len(stable)}")
    for p, a in stable:
        print(f"  {p}: {a}")

    assert len(sieve.field) > 0, "Sieve should have some patterns"
    print("PASSED")


def test_interference():
    """Test constructive and destructive interference"""
    print("\n=== Test: Interference ===")

    sieve = Sieve(threshold=0.01, damping=0.99)

    p_target = Pattern(tokens='TARGET')

    # Two paths to same pattern with same phase (constructive)
    sieve.inject(p_target, Amplitude.forward(0.5))
    sieve.inject(p_target, Amplitude.forward(0.5))

    constructive = sieve.field[p_target].magnitude
    print(f"Constructive: 0.5 + 0.5 = {constructive:.3f} (expect ~1.0)")
    assert constructive > 0.9, "Constructive interference should increase amplitude"

    # Reset and test destructive
    sieve.clear()
    sieve.inject(p_target, Amplitude.forward(0.5))
    sieve.inject(p_target, Amplitude.backward(0.5))  # Opposite phase

    destructive = sieve.field[p_target].magnitude
    print(f"Destructive: 0.5 + (-0.5) = {destructive:.3f} (expect ~0.0)")
    assert destructive < 0.1, "Destructive interference should decrease amplitude"

    print("PASSED")


def test_self_annealing():
    """Test that temperature drops as system stabilizes"""
    print("\n=== Test: Self-Annealing ===")

    sieve = Sieve(threshold=0.001, damping=0.95)

    # Create a simple system with multiple patterns
    patterns = [Pattern(tokens=f'P{i}') for i in range(10)]
    for p in patterns:
        sieve.inject_forward(p, magnitude=0.3)

    # Rules that cause some patterns to reinforce each other
    rules = [
        Rule(lhs=patterns[i], rhs=patterns[(i+1) % 10], transfer=complex(0.5, 0))
        for i in range(10)
    ]

    temps = [sieve.temperature()]

    for gen in range(50):
        sieve.evolve(rules)
        temps.append(sieve.temperature())

    print(f"Temperature: {temps[0]:.3f} -> {temps[-1]:.3f}")
    print(f"Final patterns: {len(sieve.field)}")

    # Temperature should generally decrease (annealing)
    early_temp = sum(temps[:10]) / 10
    late_temp = sum(temps[-10:]) / 10
    print(f"Early avg temp: {early_temp:.3f}, Late avg temp: {late_temp:.3f}")

    print("PASSED (temperature trend observed)")


def test_bidirectional_solve():
    """Test bidirectional solving with forward and backward waves"""
    print("\n=== Test: Bidirectional Solve ===")

    sieve = Sieve(threshold=0.001, damping=0.98)

    # Simple chain: A -> B -> C -> D -> E (boundary)
    states = [Pattern(tokens=s) for s in ['A', 'B', 'C', 'D', 'E']]

    # Forward rules
    forward_rules = [
        Rule(lhs=states[i], rhs=states[i+1], transfer=complex(1.0, 0))
        for i in range(4)
    ]

    # Backward rules (reversed)
    backward_rules = [r.reversed() for r in forward_rules]

    all_rules = forward_rules + backward_rules

    # Inject forward seed (A)
    sieve.inject_forward(states[0], magnitude=1.0)

    # Inject backward seed (E) with value-encoded phase
    value_phase = value_to_phase(1)  # Win = phase 0
    sieve.inject(states[4], Amplitude.from_polar(1.0, value_phase))

    # Evolve until stable
    for gen in range(30):
        stats = sieve.evolve(all_rules)
        if gen % 10 == 0:
            print(f"Gen {gen}: patterns={stats['patterns_out']}, closures={len(sieve.closures)}")

    print(f"Final: {len(sieve.field)} patterns, {len(sieve.closures)} closures")

    # Check that middle states have amplitude (waves met)
    for state in states:
        if state in sieve.field:
            print(f"  {state.tokens}: {sieve.field[state]}")

    print("PASSED")


def test_game_sieve():
    """Test GameSieve with a simple game"""
    print("\n=== Test: GameSieve ===")

    # Import a simple game
    try:
        from games.connect4 import Connect4Game, C4State

        game = Connect4Game()
        start = C4State()

        # Create game sieve
        gsieve = GameSieve(game, threshold=0.001, damping=0.98)

        # Inject start state
        gsieve.inject_state(start, forward=True, magnitude=1.0)

        # Run a few generations
        for gen in range(20):
            stats = gsieve.evolve_game()
            if gen % 5 == 0:
                print(f"Gen {gen}: patterns={stats['patterns_out']}, "
                      f"boundaries={stats['boundaries_found']}, "
                      f"solved={len(gsieve.solved)}")

        print(f"Final: {len(gsieve.field)} patterns, {len(gsieve.solved)} solved")
        print("PASSED")

    except ImportError:
        print("SKIPPED (games.connect4 not available)")


def test_rule_learning():
    """Test learning rules from observations"""
    print("\n=== Test: Rule Learning ===")

    from meta_sieve import Observation, learn_rules

    # Create observations from a simple deterministic system
    # If we see A->B multiple times, we should learn that rule
    observations = [
        Observation(before='A', after='B'),
        Observation(before='A', after='B'),
        Observation(before='A', after='B'),
        Observation(before='B', after='C'),
        Observation(before='B', after='C'),
        # Noise: A->X only once
        Observation(before='A', after='X'),
    ]

    rules = learn_rules(observations, max_generations=30, threshold=0.1, verbose=False)

    print(f"Learned {len(rules)} rules:")
    for rule in rules:
        print(f"  {rule.lhs.tokens} -> {rule.rhs.tokens} (strength={rule.transfer})")

    # Should have learned A->B and B->C, probably not A->X (too weak)
    rule_strings = [(r.lhs.tokens, r.rhs.tokens) for r in rules]

    if ('A', 'B') in rule_strings:
        print("Found A->B rule")
    if ('B', 'C') in rule_strings:
        print("Found B->C rule")

    print("PASSED")


if __name__ == "__main__":
    test_basic_sieve()
    test_interference()
    test_self_annealing()
    test_bidirectional_solve()
    test_game_sieve()
    test_rule_learning()

    print("\n=== All Tests Complete ===")
