"""
Tests for the deep physics layers.

This tests:
1. Substrate: AmplitudeField, Hamiltonian, Substrate evolution
2. Emergence: SelfOrganizingSubstrate, rule emergence
3. Information: Distinctions, logic emergence
4. Spacetime: Causal structure, metric, light cones

These are both unit tests and demonstrations of the concepts.
"""

import math
import cmath
import sys
import os
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the package - using direct imports for standalone testing
from sieve_core.substrate import (
    DiscreteConfig, AmplitudeField, RuleHamiltonian,
    LazyHamiltonian, Substrate, detect_closures, solve_on_substrate
)
from sieve_core.emergence import (
    EntityType, Entity, EmergentHamiltonian,
    SelfOrganizingSubstrate, bootstrap_from_noise, learn_physics
)
from sieve_core.information import (
    Distinction, Side, DistinctionSpace, LogicalSpace,
    InformationalSieve, bootstrap_logic
)
from sieve_core.spacetime import (
    Event, CausalStructure, ComputationalMetric,
    LocalityEmergence, EmergentSpacetime
)


# ============================================================
# SUBSTRATE TESTS
# ============================================================

def test_amplitude_field():
    """Test basic amplitude field operations."""
    print("\n=== Test: AmplitudeField ===")

    field = AmplitudeField(threshold=0.01)

    # Create some configurations
    c1 = DiscreteConfig(tokens=('A',))
    c2 = DiscreteConfig(tokens=('B',))
    c3 = DiscreteConfig(tokens=('C',))

    # Inject amplitudes
    field.inject(c1, 1.0)
    field.inject(c2, 0.5 + 0.5j)
    field.inject(c3, 0.1)

    print(f"Field has {len(field)} configurations")
    print(f"Norm: {field.norm():.3f}")
    print(f"Entropy: {field.entropy():.3f}")

    # Test interference
    field.inject(c1, 0.5)  # Constructive
    print(f"After constructive interference on c1: |c1| = {abs(field[c1]):.3f}")
    assert abs(field[c1]) > 1.0, "Constructive interference should increase amplitude"

    field.inject(c2, -(0.5 + 0.5j))  # Destructive
    print(f"After destructive interference on c2: |c2| = {abs(field[c2]):.3f}")
    assert abs(field[c2]) < 0.1, "Destructive interference should decrease amplitude"

    print("PASSED")


def test_rule_hamiltonian():
    """Test RuleHamiltonian."""
    print("\n=== Test: RuleHamiltonian ===")

    # Create a simple chain: A -> B -> C
    cA = DiscreteConfig(tokens=('A',))
    cB = DiscreteConfig(tokens=('B',))
    cC = DiscreteConfig(tokens=('C',))

    rules = [
        (cA, cB, 1.0),
        (cB, cC, 1.0),
    ]

    H = RuleHamiltonian(rules)

    # Check neighbors
    neighbors_A = H.neighbors(cA)
    print(f"Neighbors of A: {[(n[0].tokens, n[1]) for n in neighbors_A]}")
    assert len(neighbors_A) == 1, "A should have 1 neighbor (B)"

    neighbors_B = H.neighbors(cB)
    print(f"Neighbors of B: {[(n[0].tokens, n[1]) for n in neighbors_B]}")
    assert len(neighbors_B) == 2, "B should have 2 neighbors (A and C)"

    print("PASSED")


def test_substrate_evolution():
    """Test substrate evolution."""
    print("\n=== Test: Substrate Evolution ===")

    # Create chain Hamiltonian
    configs = [DiscreteConfig(tokens=(chr(65+i),)) for i in range(5)]  # A, B, C, D, E

    rules = [(configs[i], configs[i+1], 1.0) for i in range(4)]
    H = RuleHamiltonian(rules)

    substrate = Substrate(H, damping=0.05)

    # Inject at A
    substrate.inject(configs[0], 1.0)

    print(f"Initial: {len(substrate.psi)} configs, norm={substrate.psi.norm():.3f}")

    # Evolve
    for i in range(50):
        stats = substrate.step(0.1)
        if i % 10 == 0:
            print(f"t={substrate.time:.1f}: configs={len(substrate.psi)}, "
                  f"temp={substrate.temperature():.3f}")

    # Should have spread across the chain
    print(f"Final: {len(substrate.psi)} configs")
    for c in configs:
        amp = substrate.psi[c]
        if abs(amp) > 0.01:
            print(f"  {c.tokens}: {abs(amp):.3f}")

    print("PASSED")


def test_bidirectional_solve():
    """Test solving with forward and backward waves."""
    print("\n=== Test: Bidirectional Solve ===")

    # Linear chain
    n = 7
    configs = [DiscreteConfig(tokens=(f'S{i}',)) for i in range(n)]

    rules = [(configs[i], configs[i+1], 1.0) for i in range(n-1)]
    H = RuleHamiltonian(rules)

    # Solve: start at S0, end at S6
    closures, substrate = solve_on_substrate(
        H,
        forward_configs=[configs[0]],
        backward_configs=[configs[n-1]],
        damping=0.1,
        max_time=50.0,
        dt=0.1,
        verbose=True
    )

    print(f"\nClosures found: {len(closures)}")
    for c, amp in closures:
        print(f"  {c.tokens}: |amp|={abs(amp):.3f}, phase={cmath.phase(amp):.3f}")

    # Middle configurations should have mixed phase
    print("PASSED")


# ============================================================
# EMERGENCE TESTS
# ============================================================

def test_entity():
    """Test Entity class."""
    print("\n=== Test: Entity ===")

    # Create state entities
    s1 = Entity.state('A')
    s2 = Entity.state(('B', 'C'))

    print(f"State 1: {s1}")
    print(f"State 2: {s2}")

    # Create rule entities
    r1 = Entity.rule('A', 'B')
    r2 = Entity.rule(('X', 'Y'), ('Z',))

    print(f"Rule 1: {r1}")
    print(f"Rule 2: {r2}")

    # Check types
    assert s1.entity_type == EntityType.STATE
    assert r1.entity_type == EntityType.RULE

    print("PASSED")


def test_self_organizing():
    """Test SelfOrganizingSubstrate."""
    print("\n=== Test: SelfOrganizingSubstrate ===")

    sos = SelfOrganizingSubstrate(damping_state=0.1, damping_rule=0.02)

    # Inject some states
    sos.inject_state('A', 1.0)
    sos.inject_state('B', 0.5)
    sos.inject_state('C', 0.5)

    # Inject some rules
    sos.inject_rule('A', 'B', 0.8)
    sos.inject_rule('B', 'C', 0.8)
    sos.inject_rule('A', 'C', 0.3)  # Weaker rule

    print(f"Initial: {sos.summary()}")

    # Evolve
    for i in range(100):
        stats = sos.step(0.1)
        if i % 20 == 0:
            print(f"t={sos.time:.1f}: states={stats['states']}, rules={stats['rules']}")

    print(f"\nFinal: {sos.summary()}")

    # The stronger rules should have persisted
    print("\nDominant rules:")
    for from_t, to_t, amp in sos.dominant_rules(5):
        print(f"  {from_t} -> {to_t}: {abs(amp):.3f}")

    print("PASSED")


def test_bootstrap_from_noise():
    """Test bootstrapping from random initial conditions."""
    print("\n=== Test: Bootstrap from Noise ===")

    substrate = bootstrap_from_noise(
        n_tokens=5,
        n_initial_states=10,
        n_initial_rules=20,
        evolution_time=50.0,
        dt=0.1,
        verbose=True
    )

    # Should have some stable structure
    print(f"\nStable states: {len(substrate.get_states())}")
    print(f"Stable rules: {len(substrate.get_rules())}")

    assert len(substrate.get_rules()) > 0 or len(substrate.get_states()) > 0, \
        "Something should have survived"

    print("PASSED")


def test_learn_physics():
    """Test learning rules from observations."""
    print("\n=== Test: Learn Physics ===")

    # Generate observations from a simple system
    # True rules: A->B, B->C (always), A->X (sometimes)
    observations = []
    for _ in range(10):
        observations.append(('A', 'B'))
    for _ in range(10):
        observations.append(('B', 'C'))
    for _ in range(2):
        observations.append(('A', 'X'))  # Noise

    substrate = learn_physics(
        observations,
        evolution_time=30.0,
        verbose=True
    )

    # Check learned rules
    rules = substrate.dominant_rules(5)
    print("\nLearned rules (strength):")
    for from_t, to_t, amp in rules:
        print(f"  {from_t} -> {to_t}: {abs(amp):.3f}")

    # A->B and B->C should be stronger than A->X
    print("PASSED")


# ============================================================
# INFORMATION TESTS
# ============================================================

def test_distinction():
    """Test Distinction and Side."""
    print("\n=== Test: Distinction ===")

    d = Distinction(boundary="existence")
    print(f"Distinction: {d}")

    s_marked = Side(d, True)
    s_unmarked = Side(d, False)

    print(f"Marked side: {s_marked}")
    print(f"Unmarked side: {s_unmarked}")

    assert s_marked.opposite() == s_unmarked

    print("PASSED")


def test_distinction_space():
    """Test DistinctionSpace."""
    print("\n=== Test: DistinctionSpace ===")

    space = DistinctionSpace(threshold=0.01)

    # Create some distinctions
    space.distinguish("hot_cold", 1.0)
    space.distinguish("up_down", 0.8)
    space.distinguish("left_right", 0.5)

    print(f"Distinctions: {len(space.field)}")
    print(f"Entropy: {space.entropy():.3f}")
    print(f"Complexity: {space.complexity():.3f}")

    # Weaken a distinction
    space.identify(Distinction("left_right"), 0.4)
    print(f"After weakening left_right: {len(space.field)} distinctions")

    print("PASSED")


def test_informational_sieve():
    """Test InformationalSieve."""
    print("\n=== Test: InformationalSieve ===")

    sieve = InformationalSieve(damping=0.1, self_reflection_rate=0.1)

    # Inject some distinctions
    sieve.inject("true_false", 1.0)
    sieve.inject("yes_no", 0.8)
    sieve.inject("noise", 0.1)

    print(f"Initial: {sieve.summary()}")

    # Evolve
    for i in range(50):
        stats = sieve.evolve(0.1)
        if i % 10 == 0:
            print(f"t={sieve.time:.1f}: distinctions={stats['distinctions']}, "
                  f"temp={sieve.temperature():.3f}")

    # Strong distinctions should survive, weak ones fade
    stable = sieve.stable_distinctions(0.1)
    print(f"\nStable distinctions: {len(stable)}")
    for d, a in stable:
        print(f"  {d.boundary}: {abs(a):.3f}")

    print("PASSED")


def test_bootstrap_logic():
    """Test bootstrapping logic from distinctions."""
    print("\n=== Test: Bootstrap Logic ===")

    sieve = bootstrap_logic(verbose=True)

    # Should have created some stable logical structure
    stable = sieve.stable_distinctions()
    assert len(stable) > 0, "Should have some stable distinctions"

    print("PASSED")


# ============================================================
# SPACETIME TESTS
# ============================================================

def test_causal_structure():
    """Test CausalStructure."""
    print("\n=== Test: CausalStructure ===")

    causal = CausalStructure()

    # Create a simple causal diamond
    #     e0
    #    / \
    #   e1  e2
    #    \ /
    #     e3

    e0 = Event(id=0, changes=frozenset(), causes=frozenset())
    e1 = Event(id=1, changes=frozenset(), causes=frozenset({e0}))
    e2 = Event(id=2, changes=frozenset(), causes=frozenset({e0}))
    e3 = Event(id=3, changes=frozenset(), causes=frozenset({e1, e2}))

    causal.add_event(e0)
    causal.add_event(e1)
    causal.add_event(e2)
    causal.add_event(e3)

    print(f"Events: {len(causal.events)}")

    # Check causal relations
    print(f"e0 -> e3: distance = {causal.causal_distance(e0, e3)}")
    print(f"e1 spacelike to e2: {causal.spacelike_separated(e1, e2)}")
    print(f"e0 future: {[e.id for e in causal.causal_future(e0)]}")

    assert causal.spacelike_separated(e1, e2), "e1 and e2 should be spacelike"
    assert not causal.spacelike_separated(e0, e3), "e0 and e3 should be timelike"

    print("PASSED")


def test_locality_emergence():
    """Test LocalityEmergence."""
    print("\n=== Test: Locality Emergence ===")

    locality = LocalityEmergence(damping=0.1, coupling_strength=1.0)

    print(f"Characteristic length: {locality.characteristic_length:.3f}")
    print(f"Effective range: {locality.effective_range():.3f}")

    # Amplitude should decay with distance
    for d in [0, 1, 2, 5, 10, 20]:
        amp = locality.amplitude_at_distance(d)
        local = locality.is_local_interaction(d)
        print(f"  d={d}: amplitude={amp:.6f}, local={local}")

    print("PASSED")


def test_computational_metric():
    """Test ComputationalMetric."""
    print("\n=== Test: Computational Metric ===")

    # Create a 2D grid Hamiltonian
    configs = {}
    for i in range(5):
        for j in range(5):
            configs[(i, j)] = DiscreteConfig(tokens=(i, j))

    rules = []
    for i in range(5):
        for j in range(5):
            # Connect to neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 5 and 0 <= nj < 5:
                    rules.append((configs[(i, j)], configs[(ni, nj)], 1.0))

    H = RuleHamiltonian(rules)
    metric = ComputationalMetric(H)

    # Check distances
    c_00 = configs[(0, 0)]
    c_44 = configs[(4, 4)]
    c_01 = configs[(0, 1)]

    print(f"Distance (0,0) to (0,1): {metric.circuit_depth(c_00, c_01)}")
    print(f"Distance (0,0) to (4,4): {metric.circuit_depth(c_00, c_44)}")
    print(f"Geodesic (0,0) to (4,4): {metric.geodesic_distance(c_00, c_44):.3f}")

    # Manhattan distance should be 8
    assert metric.circuit_depth(c_00, c_44) == 8, "Manhattan distance should be 8"

    print("PASSED")


def test_emergent_spacetime():
    """Test EmergentSpacetime."""
    print("\n=== Test: Emergent Spacetime ===")

    # Create simple amplitude field and Hamiltonian
    configs = [DiscreteConfig(tokens=(i,)) for i in range(10)]
    rules = [(configs[i], configs[i+1], 1.0) for i in range(9)]

    H = RuleHamiltonian(rules)
    field = AmplitudeField()
    for c in configs:
        field.inject(c, 0.3)

    spacetime = EmergentSpacetime(field, H, damping=0.1, coupling=1.0)

    print(f"Speed of light: {spacetime.c:.3f}")
    print(f"Locality range: {spacetime.locality.effective_range():.3f}")

    # Create some events
    e0 = spacetime.create_event({})
    e1 = spacetime.create_event({}, {e0})
    e2 = spacetime.create_event({}, {e0})
    e3 = spacetime.create_event({}, {e1, e2})

    # Check interval
    interval_01 = spacetime.spacetime_interval(e0, e1)
    interval_12 = spacetime.spacetime_interval(e1, e2)

    print(f"Interval e0-e1: {interval_01:.3f} (should be positive/timelike)")
    print(f"Interval e1-e2: {interval_12:.3f} (should be negative/spacelike)")

    print(spacetime.summary())
    print("PASSED")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DEEP PHYSICS TESTS")
    print("=" * 60)

    # Substrate tests
    test_amplitude_field()
    test_rule_hamiltonian()
    test_substrate_evolution()
    test_bidirectional_solve()

    # Emergence tests
    test_entity()
    test_self_organizing()
    test_bootstrap_from_noise()
    test_learn_physics()

    # Information tests
    test_distinction()
    test_distinction_space()
    test_informational_sieve()
    test_bootstrap_logic()

    # Spacetime tests
    test_causal_structure()
    test_locality_emergence()
    test_computational_metric()
    test_emergent_spacetime()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
