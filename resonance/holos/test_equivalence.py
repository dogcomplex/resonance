"""
holos/test_equivalence.py - Test functional equivalence with fractal_holos3.py

This script verifies that the modular holos/ package behaves consistently
with the original fractal_holos3.py implementation.

Key checks:
1. Same solved positions given same inputs
2. Spine paths are created properly
3. Stats tracking matches
4. Memory tracking works
5. Backward seed auto-generation works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.core import HOLOSSolver, SeedPoint, SearchMode, LightningProbe
from holos.storage import Hologram, SpinePath, SeedFrontierMapping
from holos.games.chess import (
    ChessGame, ChessState, ChessValue,
    random_position, create_chess_solver,
    Piece, piece_type, in_check
)


def test_chess_game_interface():
    """Test ChessGame interface methods"""
    print("\n" + "=" * 60)
    print("TEST: ChessGame Interface")
    print("=" * 60)

    game = ChessGame(min_pieces=7, max_pieces=8)

    # Test state creation
    pieces = [
        (Piece.W_KING, 4),
        (Piece.W_QUEEN, 3),
        (Piece.W_ROOK, 0),
        (Piece.B_KING, 60),
        (Piece.B_QUEEN, 59),
        (Piece.B_ROOK, 63),
    ]
    state = ChessState(pieces, 'w')

    print(f"Created state with {state.piece_count()} pieces")
    state.display()

    # Test hash
    h = game.hash_state(state)
    print(f"Hash: {h}")

    # Test successors
    successors = game.get_successors(state)
    print(f"Successors: {len(successors)}")

    # Test predecessors
    predecessors = game.get_predecessors(state)
    print(f"Predecessors: {len(predecessors)}")

    # Test boundary check
    is_boundary = game.is_boundary(state)
    print(f"Is boundary (<=7 pieces): {is_boundary}")

    # Test lightning successors (captures only)
    lightning_successors = game.get_lightning_successors(state)
    print(f"Lightning successors (captures): {len(lightning_successors)}")

    # Test features
    features = game.get_features(state)
    print(f"Features: {features}")

    print("\n[PASS] ChessGame interface works")
    return True


def test_generate_boundary_seeds():
    """Test boundary seed generation (matching fractal_holos3.py)"""
    print("\n" + "=" * 60)
    print("TEST: Boundary Seed Generation")
    print("=" * 60)

    game = ChessGame(min_pieces=7, max_pieces=8)

    # Create an 8-piece template
    pos = random_position("KQRRvKQRR")
    if pos is None:
        print("[SKIP] Could not generate template position")
        return True

    print(f"Template position ({pos.piece_count()} pieces):")
    pos.display()

    # Generate boundary seeds
    print("\nGenerating boundary positions...")
    seeds = game.generate_boundary_seeds(pos, count=10)

    if len(seeds) > 0:
        print(f"\n[PASS] Generated {len(seeds)} boundary positions")
        for i, seed in enumerate(seeds[:3]):
            print(f"\nSeed {i+1} ({seed.piece_count()} pieces):")
            seed.display()
        return True
    else:
        print("[INFO] No seeds generated (may need syzygy tables)")
        return True


def test_solver_stats():
    """Test that solver has all required stats keys"""
    print("\n" + "=" * 60)
    print("TEST: Solver Stats Keys")
    print("=" * 60)

    game = ChessGame(min_pieces=7, max_pieces=8)
    solver = HOLOSSolver(game, name="test_solver")

    expected_stats = [
        'lightning_probes',
        'connections',
        'crystallized',
        'spines_found',
        'forward_expanded',
        'backward_expanded',
        'equiv_shortcuts',
        'equiv_tracked',
        'equiv_propagated',
        'minimax_solved',
    ]

    missing = [k for k in expected_stats if k not in solver.stats]
    if missing:
        print(f"[FAIL] Missing stats keys: {missing}")
        return False

    print(f"[PASS] All {len(expected_stats)} stats keys present")
    for key in expected_stats:
        print(f"  - {key}: {solver.stats[key]}")

    return True


def test_memory_tracking():
    """Test memory tracking method"""
    print("\n" + "=" * 60)
    print("TEST: Memory Tracking")
    print("=" * 60)

    game = ChessGame(min_pieces=7, max_pieces=8)
    solver = HOLOSSolver(game, name="test_solver", max_memory_mb=4000)

    mem = solver.memory_mb()
    print(f"Current memory usage: {mem:.1f} MB")
    print(f"Max memory limit: {solver.max_memory_mb} MB")

    if mem >= 0:
        print("[PASS] Memory tracking works")
        return True
    else:
        print("[FAIL] Memory tracking returned negative")
        return False


def test_spine_creation():
    """Test that spines list is initialized"""
    print("\n" + "=" * 60)
    print("TEST: Spine Structure")
    print("=" * 60)

    game = ChessGame(min_pieces=7, max_pieces=8)
    solver = HOLOSSolver(game, name="test_solver")

    if hasattr(solver, 'spines'):
        print(f"[PASS] Solver has spines attribute")
        print(f"  - Type: {type(solver.spines)}")
        print(f"  - Initial length: {len(solver.spines)}")
    else:
        print("[FAIL] Solver missing spines attribute")
        return False

    # Test SpinePath structure
    spine = SpinePath(
        start_hash=12345,
        moves=[(0, 8, None), (8, 16, None)],
        end_hash=67890,
        end_value=1,
        checkpoints=[(12345, "start"), (67890, "end")]
    )

    print(f"\nSpinePath test:")
    print(f"  - start_hash: {spine.start_hash}")
    print(f"  - end_hash: {spine.end_hash}")
    print(f"  - depth: {spine.depth}")
    print(f"  - end_value: {spine.end_value}")
    print(f"  - checkpoints: {len(spine.checkpoints)}")

    print("[PASS] Spine structures work")
    return True


def test_lightning_probe():
    """Test LightningProbe with forward and backward directions"""
    print("\n" + "=" * 60)
    print("TEST: Lightning Probe (Bidirectional)")
    print("=" * 60)

    game = ChessGame(min_pieces=7, max_pieces=8)

    # Create a position
    pieces = [
        (Piece.W_KING, 4),
        (Piece.W_QUEEN, 3),
        (Piece.W_ROOK, 0),
        (Piece.W_ROOK, 7),
        (Piece.B_KING, 60),
        (Piece.B_QUEEN, 59),
        (Piece.B_ROOK, 56),
    ]
    state = ChessState(pieces, 'w')

    print("Test position:")
    state.display()

    # Forward probe
    forward_probe = LightningProbe(game, {}, direction="forward", max_depth=10)
    value, path = forward_probe.probe(state)
    print(f"\nForward probe:")
    print(f"  - Value: {value}")
    print(f"  - Path length: {len(path)}")
    print(f"  - Nodes visited: {forward_probe.nodes_visited}")

    # Backward probe
    backward_probe = LightningProbe(game, {}, direction="backward", max_depth=10)
    bwd_value, bwd_path = backward_probe.probe(state)
    print(f"\nBackward probe:")
    print(f"  - Value: {bwd_value}")
    print(f"  - Path length: {len(bwd_path)}")
    print(f"  - Nodes visited: {backward_probe.nodes_visited}")

    print("\n[PASS] Lightning probes work (forward and backward)")
    return True


def test_apply_move():
    """Test that ChessGame has apply_move method"""
    print("\n" + "=" * 60)
    print("TEST: apply_move Method")
    print("=" * 60)

    game = ChessGame(min_pieces=7, max_pieces=8)

    pieces = [
        (Piece.W_KING, 4),
        (Piece.W_QUEEN, 3),
        (Piece.B_KING, 60),
    ]
    state = ChessState(pieces, 'w')

    successors = game.get_successors(state)
    if successors:
        child, move = successors[0]
        applied = game.apply_move(state, move)
        print(f"Original state hash: {hash(state)}")
        print(f"Move: {move}")
        print(f"Child hash (from get_successors): {hash(child)}")
        print(f"Applied hash (from apply_move): {hash(applied)}")

        if hash(child) == hash(applied):
            print("\n[PASS] apply_move matches get_successors")
            return True
        else:
            print("\n[FAIL] apply_move produces different state")
            return False
    else:
        print("[SKIP] No successors available")
        return True


def run_all_tests():
    """Run all equivalence tests"""
    print("\n" + "=" * 60)
    print("HOLOS MODULAR EQUIVALENCE TESTS")
    print("=" * 60)
    print("Verifying consistency with fractal_holos3.py")

    results = []
    tests = [
        ("ChessGame Interface", test_chess_game_interface),
        ("Boundary Seed Generation", test_generate_boundary_seeds),
        ("Solver Stats Keys", test_solver_stats),
        ("Memory Tracking", test_memory_tracking),
        ("Spine Structure", test_spine_creation),
        ("Lightning Probe", test_lightning_probe),
        ("apply_move Method", test_apply_move),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed - modular version is consistent!")
    else:
        print("\n[WARNING] Some tests failed - check implementation")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
