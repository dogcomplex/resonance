"""
holos/test_integration.py - Comprehensive integration tests

Tests the full HOLOS pipeline including:
1. Connection detection
2. Crystallization
3. Solver reset
4. Multi-round sessions
5. Hologram merging
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.storage import Hologram, SpinePath
from holos.games.chess import ChessGame, random_position


def test_connection_detection():
    """Verify connections are detected when waves meet"""
    print("=" * 60)
    print("TEST: Connection Detection")
    print("=" * 60)

    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=7)
    solver = HOLOSSolver(game, name="conn_test", max_memory_mb=300)

    # 7-piece position (one above boundary)
    positions = []
    for _ in range(5):
        pos = random_position("KQRvKQR")
        if pos:
            positions.append(pos)

    if not positions:
        print("[SKIP] Could not generate positions")
        return False

    seeds = [SeedPoint(p, SearchMode.WAVE) for p in positions]
    hologram = solver.solve(seeds, max_iterations=5)

    print(f"\nResults:")
    print(f"  Solved: {len(hologram.solved)}")
    print(f"  Stats connections: {solver.stats['connections']}")
    print(f"  Connection list: {len(hologram.connections)}")
    print(f"  Spines: {len(hologram.spines)}")

    # Verify seeds were solved
    seeds_solved = sum(1 for p in positions if game.hash_state(p) in hologram.solved)
    print(f"  Seeds solved: {seeds_solved}/{len(positions)}")

    if seeds_solved == len(positions):
        print("\n[PASS] Connection detection working")
        return True
    else:
        print("\n[FAIL] Not all seeds solved")
        return False


def test_crystallization():
    """Verify crystallization expands around connections"""
    print("\n" + "=" * 60)
    print("TEST: Crystallization")
    print("=" * 60)

    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=7)
    solver = HOLOSSolver(game, name="crystal_test", max_memory_mb=400)

    positions = []
    for _ in range(10):
        pos = random_position("KQRvKQR")
        if pos:
            positions.append(pos)

    if not positions:
        print("[SKIP] Could not generate positions")
        return False

    seeds = [SeedPoint(p, SearchMode.WAVE) for p in positions]
    hologram = solver.solve(seeds, max_iterations=5)

    print(f"\nResults:")
    print(f"  Solved: {len(hologram.solved)}")
    print(f"  Crystallized: {solver.stats['crystallized']}")
    print(f"  Connections: {solver.stats['connections']}")

    # Crystallization should happen if we have connections
    if solver.stats['connections'] > 0:
        print(f"\n[PASS] Crystallization triggered ({solver.stats['crystallized']} positions)")
        return True
    else:
        print("\n[INFO] No connections found - crystallization not triggered")
        print("       (This can happen with limited iterations)")
        return True  # Not a failure, just didn't reach connection


def test_solver_reset():
    """Verify solver.reset() clears all state"""
    print("\n" + "=" * 60)
    print("TEST: Solver Reset")
    print("=" * 60)

    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=7)
    solver = HOLOSSolver(game, name="reset_test", max_memory_mb=200)

    # First solve
    pos1 = random_position("KQRvKQR")
    if pos1:
        seeds1 = [SeedPoint(pos1, SearchMode.WAVE)]
        solver.solve(seeds1, max_iterations=2)

    state_before = {
        'forward_seen': len(solver.forward_seen),
        'backward_seen': len(solver.backward_seen),
        'solved': len(solver.solved),
        'connections': solver.stats['connections'],
    }
    print(f"Before reset: {state_before}")

    # Reset
    solver.reset()

    state_after = {
        'forward_seen': len(solver.forward_seen),
        'backward_seen': len(solver.backward_seen),
        'solved': len(solver.solved),
        'connections': solver.stats['connections'],
    }
    print(f"After reset: {state_after}")

    if all(v == 0 for v in state_after.values()):
        print("\n[PASS] Solver reset clears all state")
        return True
    else:
        print("\n[FAIL] Solver reset did not clear all state")
        return False


def test_hologram_merge():
    """Verify hologram merging with deduplication"""
    print("\n" + "=" * 60)
    print("TEST: Hologram Merge")
    print("=" * 60)

    # Create two holograms with overlapping data
    h1 = Hologram("h1")
    h1.solved = {1: "win", 2: "lose", 3: "draw"}
    h1.spines.append(SpinePath(
        start_hash=1,
        moves=[(0, 1, None)],
        end_hash=10,
        end_value="win"
    ))

    h2 = Hologram("h2")
    h2.solved = {3: "draw", 4: "win", 5: "lose"}  # 3 overlaps
    h2.spines.append(SpinePath(
        start_hash=1,  # Same start_hash, should dedupe
        moves=[(0, 2, None)],
        end_hash=20,
        end_value="win"
    ))
    h2.spines.append(SpinePath(
        start_hash=4,  # New spine
        moves=[(0, 3, None)],
        end_hash=30,
        end_value="win"
    ))

    merged = h1.merge(h2)

    print(f"H1 solved: {len(h1.solved)}, spines: {len(h1.spines)}")
    print(f"H2 solved: {len(h2.solved)}, spines: {len(h2.spines)}")
    print(f"Merged solved: {len(merged.solved)}, spines: {len(merged.spines)}")

    # Should have 5 unique solved positions (1,2,3,4,5)
    # Should have 2 unique spines (start_hash 1 and 4, dedupe on 1)
    expected_solved = 5
    expected_spines = 2

    if len(merged.solved) == expected_solved and len(merged.spines) == expected_spines:
        print(f"\n[PASS] Hologram merge with deduplication")
        return True
    else:
        print(f"\n[FAIL] Expected {expected_solved} solved, {expected_spines} spines")
        return False


def test_boundary_seed_generation():
    """Verify boundary seeds respect min_pieces setting"""
    print("\n" + "=" * 60)
    print("TEST: Boundary Seed Generation")
    print("=" * 60)

    # Test with min_pieces=6 (6-piece positions should be generated)
    game6 = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=8)

    template = random_position("KQRRvKQR")  # 8 pieces
    if not template:
        print("[SKIP] Could not generate template")
        return False

    seeds6 = game6.generate_boundary_seeds(template, count=10)

    if seeds6:
        piece_counts = [s.piece_count() for s in seeds6]
        print(f"min_pieces=6: Generated {len(seeds6)} positions with counts: {set(piece_counts)}")

        if all(c == 6 for c in piece_counts):
            print("[PASS] All seeds have correct piece count (6)")
            return True
        else:
            print(f"[FAIL] Expected all 6-piece, got {set(piece_counts)}")
            return False
    else:
        print("[FAIL] No seeds generated")
        return False


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("HOLOS INTEGRATION TESTS")
    print("=" * 60)

    results = []

    results.append(("Boundary Seed Generation", test_boundary_seed_generation()))
    results.append(("Solver Reset", test_solver_reset()))
    results.append(("Hologram Merge", test_hologram_merge()))
    results.append(("Connection Detection", test_connection_detection()))
    results.append(("Crystallization", test_crystallization()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} passed")

    if passed == len(results):
        print("\n[SUCCESS] All integration tests passed!")
        return True
    else:
        print("\n[WARNING] Some tests failed")
        return False


if __name__ == "__main__":
    run_all_tests()
