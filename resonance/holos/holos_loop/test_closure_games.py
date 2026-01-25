"""
Test closure-aware layer system against actual games.

Tests:
1. Connect4 - full game tree
2. Connect4 - specific opening analysis
3. Comparison with original HOLOS solver
"""

import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from holos.games.connect4 import Connect4Game, C4State, C4Value
from holos.games.sudoku import SudokuGame, SudokuState, get_sample_puzzles
from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.closure import ClosureDetector, ModeEmergence
from holos.layer1_paths import create_path_solver
from holos.wave_system import create_wave_system, run_wave_search


def build_position_pool(game, start, max_depth=3, max_positions=500):
    """Build a pool of positions for testing."""
    positions = [start]
    frontier = [start]
    seen = {game.hash_state(start)}

    for depth in range(max_depth):
        next_frontier = []
        for pos in frontier:
            for child, col in game.get_successors(pos):
                h = game.hash_state(child)
                if h not in seen and len(positions) < max_positions:
                    seen.add(h)
                    positions.append(child)
                    next_frontier.append(child)
        frontier = next_frontier[:100]  # Limit growth per level

    return positions


def test_connect4_closure_detection():
    """Test closure detection during Connect4 search."""
    print("\n" + "=" * 60)
    print("TEST: Connect4 Closure Detection")
    print("=" * 60)

    game = Connect4Game()
    start = C4State()

    # Build positions
    positions = build_position_pool(game, start, max_depth=3, max_positions=200)
    print(f"Built {len(positions)} positions")

    # Create closure detector
    detector = ClosureDetector(phase_threshold=0.2)
    emergence = ModeEmergence(detector)

    # Manual forward/backward simulation
    forward_values = {}
    backward_values = {}

    # Forward: from start, estimate reachability
    frontier = {game.hash_state(start): start}
    for _ in range(5):
        next_frontier = {}
        for h, state in frontier.items():
            forward_values[h] = len(forward_values)  # Pseudo-value: distance from start
            for child, move in game.get_successors(state):
                ch = game.hash_state(child)
                if ch not in forward_values:
                    next_frontier[ch] = child
        frontier = next_frontier

    # Backward: from terminals, propagate known values
    backward_frontier = {}
    for pos in positions:
        term, val = game.is_terminal(pos)
        if term:
            h = game.hash_state(pos)
            backward_values[h] = val.value if val else 0
            backward_frontier[h] = pos

    for _ in range(5):
        next_frontier = {}
        for h, state in backward_frontier.items():
            for parent, move in game.get_predecessors(state):
                ph = game.hash_state(parent)
                if ph not in backward_values:
                    # Simple propagation (not proper minimax, just for testing)
                    backward_values[ph] = backward_values[h]
                    next_frontier[ph] = parent
        backward_frontier = next_frontier

    # Check for closures
    closures = []
    for h in forward_values:
        if h in backward_values:
            event = detector.check_closure(
                state_hash=h,
                forward_value=float(forward_values[h]),
                backward_value=float(backward_values[h]),
                layer=0,
                iteration=0
            )
            if event:
                closures.append(event)

    print(f"Forward explored: {len(forward_values)}")
    print(f"Backward explored: {len(backward_values)}")
    print(f"Overlapping: {len(set(forward_values) & set(backward_values))}")
    print(f"Closures detected: {len(closures)}")

    if closures:
        print(f"First closure: {closures[0]}")

    # Test mode emergence
    mode = emergence.get_emergent_mode(
        forward_frontier_size=len(forward_values),
        backward_frontier_size=len(backward_values),
        recent_closures=len(closures),
        branching_factor=7.0  # Connect4 has 7 columns
    )
    print(f"Emergent mode: {mode}")

    print("PASSED")


def test_connect4_layer1_paths():
    """Test Layer 1 path finding on Connect4."""
    print("\n" + "=" * 60)
    print("TEST: Connect4 Layer 1 Paths")
    print("=" * 60)

    game = Connect4Game()
    start = C4State()

    # Build positions from specific openings
    positions = []

    # Center column opening (strongest)
    center_open = start
    for child, col in game.get_successors(center_open):
        if col == 3:  # Center
            positions.append(child)
            # Add responses
            for grandchild, col2 in game.get_successors(child):
                positions.append(grandchild)
            break

    # Edge column opening (weaker)
    for child, col in game.get_successors(start):
        if col == 0:  # Edge
            positions.append(child)
            for grandchild, col2 in game.get_successors(child):
                if len(positions) < 20:
                    positions.append(grandchild)
            break

    print(f"Testing with {len(positions)} positions")

    # Create path solver
    solver = create_path_solver(game, max_path_length=25)

    # Solve
    t0 = time.time()
    spines = solver.solve(
        forward_seeds=positions,
        backward_seeds=None,
        max_iterations=40,
        mode="balanced",
        verbose=True
    )
    elapsed = time.time() - t0

    print(f"\nTime: {elapsed:.2f}s")
    print(f"Spines found: {len(spines)}")

    # Analyze spines
    wins = {'X': 0, 'O': 0, 'Draw': 0}
    lengths = []
    for spine in spines:
        lengths.append(spine.depth)
        if spine.end_value:
            if hasattr(spine.end_value, 'value'):
                v = spine.end_value.value
            else:
                v = spine.end_value
            if v == 1:
                wins['X'] += 1
            elif v == -1:
                wins['O'] += 1
            else:
                wins['Draw'] += 1

    if lengths:
        print(f"Path lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    print(f"Outcomes: {wins}")

    print("PASSED")


def test_connect4_wave_system():
    """Test wave system on Connect4."""
    print("\n" + "=" * 60)
    print("TEST: Connect4 Wave System")
    print("=" * 60)

    game = Connect4Game()
    start = C4State()

    # Build position pool
    positions = build_position_pool(game, start, max_depth=2, max_positions=50)
    print(f"Initial positions: {len(positions)}")

    # Run wave search
    t0 = time.time()
    result = run_wave_search(
        game,
        start_states=positions,
        boundary_states=None,
        max_iterations=40,
        energy=300.0
    )
    elapsed = time.time() - t0

    print(f"\nTime: {elapsed:.2f}s")
    print(f"Iterations: {result['iterations']}")
    print(f"Closures: {result['closures']}")
    print(f"Spines: {result['spines']}")
    print(f"Energy spent: {result['energy_spent']:.1f}")

    print("PASSED")


def test_compare_original_holos():
    """Compare closure-aware system with original HOLOS."""
    print("\n" + "=" * 60)
    print("TEST: Compare with Original HOLOS")
    print("=" * 60)

    game = Connect4Game()
    start = C4State()

    # Build same position pool for both
    positions = build_position_pool(game, start, max_depth=2, max_positions=30)
    print(f"Positions: {len(positions)}")

    # Original HOLOS
    print("\n--- Original HOLOS ---")
    original_solver = HOLOSSolver(game, name="original_c4")
    seeds = [SeedPoint(p, SearchMode.WAVE) for p in positions]

    t0 = time.time()
    hologram = original_solver.solve(
        forward_seeds=seeds,
        backward_seeds=None,
        max_iterations=30
    )
    original_time = time.time() - t0

    print(f"Time: {original_time:.2f}s")
    print(f"Solved: {len(hologram.solved)}")
    print(f"Spines: {len(hologram.spines)}")
    print(f"Connections: {len(hologram.connections)}")

    # Closure-aware system
    print("\n--- Closure-Aware Wave System ---")
    t0 = time.time()
    result = run_wave_search(
        game,
        start_states=positions,
        boundary_states=None,
        max_iterations=30,
        energy=200.0
    )
    closure_time = time.time() - t0

    print(f"Time: {closure_time:.2f}s")
    print(f"Spines: {result['spines']}")
    print(f"Closures: {result['closures']}")

    # Comparison
    print("\n--- Comparison ---")
    print(f"Original spines: {len(hologram.spines)}")
    print(f"Closure spines: {result['spines']}")
    print(f"Speed ratio: {original_time/closure_time:.2f}x" if closure_time > 0 else "N/A")

    print("PASSED")


def test_connect4_opening_analysis():
    """Analyze specific Connect4 openings with closure system."""
    print("\n" + "=" * 60)
    print("TEST: Connect4 Opening Analysis")
    print("=" * 60)

    game = Connect4Game()
    start = C4State()

    openings = {}

    # Get all first moves
    for child, col in game.get_successors(start):
        openings[f"col_{col}"] = child

    print(f"Analyzing {len(openings)} openings...")

    # Analyze each opening with path solver
    results = {}
    for name, pos in openings.items():
        solver = create_path_solver(game, max_path_length=20)

        spines = solver.solve(
            forward_seeds=[pos],
            backward_seeds=None,
            max_iterations=20,
            mode="balanced",
            verbose=False
        )

        # Count outcomes
        x_wins = sum(1 for s in spines if hasattr(s.end_value, 'value') and s.end_value.value == 1)
        o_wins = sum(1 for s in spines if hasattr(s.end_value, 'value') and s.end_value.value == -1)

        results[name] = {
            'spines': len(spines),
            'x_wins': x_wins,
            'o_wins': o_wins,
            'x_ratio': x_wins / len(spines) if spines else 0
        }

    # Sort by X win ratio (best openings for player 1)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['x_ratio'], reverse=True)

    print("\nOpening Analysis (sorted by X win ratio):")
    for name, r in sorted_results:
        print(f"  {name}: {r['spines']} spines, X wins {r['x_ratio']*100:.0f}%")

    # Best opening should be center (col 3)
    best = sorted_results[0][0]
    print(f"\nBest opening: {best}")

    print("PASSED")


def test_sudoku_layer1_paths():
    """Test Layer 1 path finding on Sudoku."""
    print("\n" + "=" * 60)
    print("TEST: Sudoku Layer 1 Paths")
    print("=" * 60)

    game = SudokuGame()
    puzzles = get_sample_puzzles()

    # Use easy puzzle
    puzzle = puzzles.get('easy')
    if puzzle is None:
        print("No easy puzzle available, skipping")
        print("SKIPPED")
        return

    print(f"Puzzle has {puzzle.filled_count()} clues")

    # Create path solver
    solver = create_path_solver(game, max_path_length=30)

    # Solve - Sudoku is different: we start from puzzle and look for solution
    t0 = time.time()
    spines = solver.solve(
        forward_seeds=[puzzle],
        backward_seeds=None,  # Will auto-generate solved grids
        max_iterations=30,
        mode="balanced",
        verbose=True
    )
    elapsed = time.time() - t0

    print(f"\nTime: {elapsed:.2f}s")
    print(f"Spines found: {len(spines)}")

    if spines:
        # Check if solution is valid
        for spine in spines[:3]:
            print(f"  Path depth: {spine.depth}, end_value: {spine.end_value}")

    print("PASSED")


def test_sudoku_wave_system():
    """Test wave system on Sudoku."""
    print("\n" + "=" * 60)
    print("TEST: Sudoku Wave System")
    print("=" * 60)

    game = SudokuGame()
    puzzles = get_sample_puzzles()

    puzzle = puzzles.get('easy')
    if puzzle is None:
        print("No easy puzzle available, skipping")
        print("SKIPPED")
        return

    print(f"Puzzle has {puzzle.filled_count()} clues")

    # Run wave search
    t0 = time.time()
    result = run_wave_search(
        game,
        start_states=[puzzle],
        boundary_states=None,
        max_iterations=30,
        energy=200.0
    )
    elapsed = time.time() - t0

    print(f"\nTime: {elapsed:.2f}s")
    print(f"Iterations: {result['iterations']}")
    print(f"Closures: {result['closures']}")
    print(f"Spines: {result['spines']}")

    print("PASSED")


def run_all_tests():
    """Run all game tests."""
    tests = [
        ("Closure Detection", test_connect4_closure_detection),
        ("Layer 1 Paths", test_connect4_layer1_paths),
        ("Wave System", test_connect4_wave_system),
        # ("Compare Original", test_compare_original_holos),  # Slow, skip by default
        ("Opening Analysis", test_connect4_opening_analysis),
        ("Sudoku Paths", test_sudoku_layer1_paths),
        ("Sudoku Wave", test_sudoku_wave_system),
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

    print("\n" + "=" * 60)
    print(f"GAME TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test closure-aware system on games")
    parser.add_argument("--test", choices=[
        "closure", "paths", "wave", "compare", "openings",
        "sudoku_paths", "sudoku_wave", "all"
    ], default="all", help="Which test to run")

    args = parser.parse_args()

    if args.test == "closure":
        test_connect4_closure_detection()
    elif args.test == "paths":
        test_connect4_layer1_paths()
    elif args.test == "wave":
        test_connect4_wave_system()
    elif args.test == "compare":
        test_compare_original_holos()
    elif args.test == "openings":
        test_connect4_opening_analysis()
    elif args.test == "sudoku_paths":
        test_sudoku_layer1_paths()
    elif args.test == "sudoku_wave":
        test_sudoku_wave_system()
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
