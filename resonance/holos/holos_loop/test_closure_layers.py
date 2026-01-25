"""
Test script for closure-aware layer system.

Tests:
1. Closure detection basics
2. Layer 1 path search
3. Layer 2 cover search
4. Layer 3 policy search
5. Full wave system integration
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from holos.games.connect4 import Connect4Game, C4State
from holos.closure import (
    ClosureDetector, ClosureEvent, ClosureType,
    PhaseAlignment, ModeEmergence, SimpleIrreducibilityChecker,
    compute_phase_closure, estimate_irreducibility_from_path
)
from holos.layer1_paths import (
    PathLayerSolver, PathGame, PartialPath, PathValue,
    create_path_solver, solve_with_paths
)
from holos.layer2_covers import (
    CoverLayerSolver, CoverGame, PathCover, CoverValue,
    create_cover_solver, find_optimal_cover
)
from holos.layer3_policy import (
    PolicyLayerSolver, PolicyGame, CoverPolicy, PolicyValue,
    create_policy_solver
)
from holos.wave_system import (
    WaveSystem, WaveState, LayerMedium,
    create_wave_system, run_wave_search
)


def test_closure_detection():
    """Test basic closure detection"""
    print("\n" + "="*60)
    print("TEST: Closure Detection")
    print("="*60)

    detector = ClosureDetector(phase_threshold=0.2)

    # Test phase alignment
    alignment1 = PhaseAlignment(100.0, 95.0, 100.0)
    print(f"Alignment 1: phase_diff={alignment1.phase_diff:.3f}, aligned={alignment1.is_aligned}")
    assert alignment1.is_aligned, "Should be aligned (within 20%)"

    alignment2 = PhaseAlignment(100.0, 50.0, 100.0)
    print(f"Alignment 2: phase_diff={alignment2.phase_diff:.3f}, aligned={alignment2.is_aligned}")
    assert not alignment2.is_aligned, "Should not be aligned (50% diff)"

    # Test closure detection
    event = detector.check_closure(
        state_hash=12345,
        forward_value=100.0,
        backward_value=98.0,
        layer=0,
        iteration=1
    )
    assert event is not None, "Should detect closure"
    print(f"Closure detected: {event}")
    assert event.closure_type in [ClosureType.IRREDUCIBLE, ClosureType.REDUCIBLE]

    # Test no closure when values differ
    event2 = detector.check_closure(
        state_hash=12346,
        forward_value=100.0,
        backward_value=50.0,
        layer=0,
        iteration=2
    )
    assert event2 is None, "Should not detect closure (values differ too much)"

    print(f"Detector stats: {detector.stats}")
    print("PASSED")


def test_phase_closure():
    """Test phase closure computation"""
    print("\n" + "="*60)
    print("TEST: Phase Closure Computation")
    print("="*60)

    # Closed sequence
    values1 = [0.0, 0.5, 1.0, 0.5, 0.0]
    is_closed, residual = compute_phase_closure(values1)
    print(f"Sequence 1: closed={is_closed}, residual={residual:.3f}")
    assert is_closed, "Should be closed (returns to start)"

    # Open sequence
    values2 = [0.0, 0.5, 1.0, 1.5, 2.0]
    is_closed, residual = compute_phase_closure(values2)
    print(f"Sequence 2: closed={is_closed}, residual={residual:.3f}")
    assert not is_closed, "Should not be closed (doesn't return)"

    print("PASSED")


def test_irreducibility_estimation():
    """Test irreducibility estimation"""
    print("\n" + "="*60)
    print("TEST: Irreducibility Estimation")
    print("="*60)

    # Long path, few alternatives = more irreducible
    irr1 = estimate_irreducibility_from_path(20, 2.0, 1)
    print(f"Long path, few alts: {irr1:.3f}")

    # Short path, many alternatives = more reducible
    irr2 = estimate_irreducibility_from_path(3, 10.0, 10)
    print(f"Short path, many alts: {irr2:.3f}")

    assert irr1 > irr2, "Long/constrained path should be more irreducible"
    print("PASSED")


def test_layer1_path_search():
    """Test Layer 1 path search with Connect4"""
    print("\n" + "="*60)
    print("TEST: Layer 1 Path Search")
    print("="*60)

    game = Connect4Game()
    start = C4State()

    # Create path solver
    solver = create_path_solver(game, max_path_length=20)

    # Build some starting positions
    positions = [start]
    for child, col in game.get_successors(start):
        if col in [2, 3, 4]:
            positions.append(child)

    # Run path search
    spines = solver.solve(
        forward_seeds=positions,
        backward_seeds=None,
        max_iterations=30,
        mode="balanced",
        verbose=True
    )

    print(f"\nFound {len(spines)} spines")
    for spine in spines[:3]:
        print(f"  {spine}")

    print(f"\nPath game stats: {solver.path_game.stats}")
    print(f"Closure stats: {solver.closure_detector.stats}")
    print("PASSED")


def test_layer2_cover_search():
    """Test Layer 2 cover search"""
    print("\n" + "="*60)
    print("TEST: Layer 2 Cover Search")
    print("="*60)

    # Create mock paths with coverage
    paths = [
        (1, "path1"),
        (2, "path2"),
        (3, "path3"),
        (4, "path4"),
        (5, "path5"),
    ]

    solver = create_cover_solver(paths, target=100)

    # Set path coverages
    solver.set_path_coverage(1, set(range(0, 30)))    # Covers 0-29
    solver.set_path_coverage(2, set(range(20, 50)))   # Covers 20-49
    solver.set_path_coverage(3, set(range(40, 70)))   # Covers 40-69
    solver.set_path_coverage(4, set(range(60, 90)))   # Covers 60-89
    solver.set_path_coverage(5, set(range(80, 110)))  # Covers 80-109

    # Run cover search
    covers = solver.solve(
        max_iterations=50,
        mode="balanced",
        verbose=True
    )

    print(f"\nFound {len(covers)} complete covers")
    for cover in covers[:3]:
        value = solver.cover_game.evaluate_cover(cover)
        print(f"  {cover.signature()}: {value}")

    print("PASSED")


def test_layer3_policy_search():
    """Test Layer 3 policy search"""
    print("\n" + "="*60)
    print("TEST: Layer 3 Policy Search")
    print("="*60)

    # Create mock covers
    covers = [
        (1, PathCover(paths=(1, 2))),
        (2, PathCover(paths=(2, 3))),
        (3, PathCover(paths=(3, 4))),
    ]

    solver = create_policy_solver(
        covers,
        total_problems=3,
        compute_budget=100.0,
        storage_budget=100.0
    )

    # Set cover costs and problems
    solver.set_cover_costs(1, compute=10.0, storage=5.0)
    solver.set_cover_costs(2, compute=15.0, storage=8.0)
    solver.set_cover_costs(3, compute=12.0, storage=6.0)

    solver.set_cover_problems(1, {0, 1})      # Solves problems 0, 1
    solver.set_cover_problems(2, {1, 2})      # Solves problems 1, 2
    solver.set_cover_problems(3, {0, 2})      # Solves problems 0, 2

    # Run policy search
    policies = solver.solve(
        max_iterations=50,
        mode="balanced",
        verbose=True
    )

    print(f"\nFound {len(policies)} complete policies")
    for policy in policies[:3]:
        value = solver.policy_game.evaluate_policy(policy)
        print(f"  {policy.signature()}: {value}")

    print("PASSED")


def test_mode_emergence():
    """Test mode emergence from closure state"""
    print("\n" + "="*60)
    print("TEST: Mode Emergence")
    print("="*60)

    detector = ClosureDetector()
    emergence = ModeEmergence(detector)

    # No closures, small frontier = lightning
    mode1 = emergence.get_emergent_mode(
        forward_frontier_size=50,
        backward_frontier_size=50,
        recent_closures=0,
        branching_factor=2.5
    )
    print(f"Small frontier, no closures: {mode1}")
    assert mode1 == "lightning", "Should emerge as lightning"

    # Many closures = crystal
    mode2 = emergence.get_emergent_mode(
        forward_frontier_size=1000,
        backward_frontier_size=1000,
        recent_closures=10,
        branching_factor=5.0
    )
    print(f"Many closures: {mode2}")
    assert mode2 == "crystal", "Should emerge as crystal"

    # Asymmetric frontiers = osmosis
    mode3 = emergence.get_emergent_mode(
        forward_frontier_size=100,
        backward_frontier_size=1000,
        recent_closures=2,
        branching_factor=5.0
    )
    print(f"Asymmetric frontiers: {mode3}")
    assert mode3 == "osmosis", "Should emerge as osmosis"

    # Default = wave
    mode4 = emergence.get_emergent_mode(
        forward_frontier_size=500,
        backward_frontier_size=500,
        recent_closures=2,
        branching_factor=5.0
    )
    print(f"Default case: {mode4}")
    assert mode4 == "wave", "Should emerge as wave"

    print("PASSED")


def test_wave_system_basic():
    """Test basic wave system setup"""
    print("\n" + "="*60)
    print("TEST: Wave System Basic")
    print("="*60)

    game = Connect4Game()
    start = C4State()

    # Create wave system
    system = create_wave_system(game, total_energy=100.0)

    # Build positions
    positions = [start]
    for child, col in game.get_successors(start):
        if col in [2, 3, 4]:
            positions.append(child)

    # Setup
    system.setup(positions)

    print(f"System initialized with {len(system.layers)} layers")
    for i, layer in enumerate(system.layers):
        print(f"  Layer {i} ({layer.name}): impedance={layer.impedance}, damping={layer.damping}")

    # Run a few steps
    for i in range(5):
        result = system.step()
        print(f"Step {i}: closures={sum(result['layer_closures'])}, modes={system.current_modes}")

    print("PASSED")


def test_wave_system_full():
    """Test full wave system run with Connect4"""
    print("\n" + "="*60)
    print("TEST: Wave System Full Run")
    print("="*60)

    game = Connect4Game()
    start = C4State()

    # Build position pool
    positions = [start]
    frontier = [start]
    seen = {game.hash_state(start)}

    for depth in range(2):
        next_frontier = []
        for pos in frontier:
            for child, col in game.get_successors(pos):
                h = game.hash_state(child)
                if h not in seen:
                    seen.add(h)
                    positions.append(child)
                    next_frontier.append(child)
        frontier = next_frontier[:20]

    print(f"Built {len(positions)} positions")

    # Run wave search
    result = run_wave_search(
        game,
        start_states=positions[:10],
        boundary_states=None,
        max_iterations=30,
        energy=200.0
    )

    print(f"\nResults: {result}")
    print("PASSED")


def run_all_tests():
    """Run all tests"""
    tests = [
        ("Closure Detection", test_closure_detection),
        ("Phase Closure", test_phase_closure),
        ("Irreducibility", test_irreducibility_estimation),
        ("Mode Emergence", test_mode_emergence),
        ("Layer 1 Paths", test_layer1_path_search),
        ("Layer 2 Covers", test_layer2_cover_search),
        ("Layer 3 Policies", test_layer3_policy_search),
        ("Wave System Basic", test_wave_system_basic),
        ("Wave System Full", test_wave_system_full),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\nFAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test closure-aware layer system")
    parser.add_argument("--test", choices=[
        "closure", "phase", "irr", "mode",
        "layer1", "layer2", "layer3",
        "wave_basic", "wave_full", "all"
    ], default="all", help="Which test to run")

    args = parser.parse_args()

    if args.test == "closure":
        test_closure_detection()
    elif args.test == "phase":
        test_phase_closure()
    elif args.test == "irr":
        test_irreducibility_estimation()
    elif args.test == "mode":
        test_mode_emergence()
    elif args.test == "layer1":
        test_layer1_path_search()
    elif args.test == "layer2":
        test_layer2_cover_search()
    elif args.test == "layer3":
        test_layer3_policy_search()
    elif args.test == "wave_basic":
        test_wave_system_basic()
    elif args.test == "wave_full":
        test_wave_system_full()
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
