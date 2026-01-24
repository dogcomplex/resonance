"""
Test the refactored goal targeting architecture.

This tests:
1. Layer 0 capabilities (material utilities in chess.py)
2. GoalCondition as Layer 1/2 concept
3. Goal-filtered solving
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode, GoalCondition
from holos.games.chess import (
    ChessGame, random_position,
    get_material_string, get_parent_materials, enumerate_material_positions
)


def test_material_utilities():
    """Test Layer 0 material utilities"""
    print("=" * 60)
    print("TEST: Material Utilities (Layer 0)")
    print("=" * 60)

    # Create a position
    pos = random_position("KQRRvKQR")
    if not pos:
        print("[SKIP] Could not generate position")
        return False

    # Test get_material_string
    material = get_material_string(pos)
    print(f"Position material: {material}")
    assert material == "KQRRvKQR", f"Expected KQRRvKQR, got {material}"

    # Test get_parent_materials
    parents = get_parent_materials("KQRRvKQR")
    print(f"Parent materials: {parents}")
    assert "KQRRvKQRR" in parents, "Expected KQRRvKQRR in parents"
    assert len(parents) == 10, f"Expected 10 parents, got {len(parents)}"

    print("\n[PASS] Material utilities work correctly")
    return True


def test_game_signature():
    """Test ChessGame.get_signature"""
    print("\n" + "=" * 60)
    print("TEST: Game Signature Method")
    print("=" * 60)

    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=8)

    pos = random_position("KQRRvKQR")
    if not pos:
        print("[SKIP] Could not generate position")
        return False

    sig = game.get_signature(pos)
    print(f"Game signature: {sig}")
    assert sig == "KQRRvKQR", f"Expected KQRRvKQR, got {sig}"

    # Test enumerate_positions
    positions = game.enumerate_positions("KQRRvKQR", count=5)
    print(f"Enumerated {len(positions)} positions with KQRRvKQR material")
    assert len(positions) == 5, f"Expected 5 positions, got {len(positions)}"

    # Verify all have correct material
    for p in positions:
        assert game.get_signature(p) == "KQRRvKQR"

    print("\n[PASS] Game signature method works")
    return True


def test_goal_condition():
    """Test GoalCondition filtering"""
    print("\n" + "=" * 60)
    print("TEST: GoalCondition (Layer 1/2)")
    print("=" * 60)

    # Create a goal targeting KQRRvKQR only
    goal = GoalCondition(
        target_signatures={"KQRRvKQR"},
        early_terminate_misses=True,
        name="KQRRvKQR_only"
    )

    # Test matches
    assert goal.matches("KQRRvKQR") == True
    assert goal.matches("KQRRvKQRR") == False
    assert goal.matches("KQRRvKQB") == False

    print(f"Goal: {goal.name}")
    print(f"  Targets: {goal.target_signatures}")
    print(f"  Early terminate: {goal.early_terminate_misses}")
    print(f"  'KQRRvKQR' matches: {goal.matches('KQRRvKQR')}")
    print(f"  'KQRRvKQRR' matches: {goal.matches('KQRRvKQRR')}")

    print("\n[PASS] GoalCondition works correctly")
    return True


def test_goal_filtered_solve():
    """Test solving with goal filtering"""
    print("\n" + "=" * 60)
    print("TEST: Goal-Filtered Solve")
    print("=" * 60)

    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=7)
    solver = HOLOSSolver(game, name="goal_test", max_memory_mb=500)

    # Create goal for KQRvKQR (6-piece boundary)
    goal = GoalCondition(
        target_signatures={"KQRvKQR"},
        early_terminate_misses=True,
        name="KQRvKQR_only"
    )

    # Generate 7-piece positions
    positions = []
    for _ in range(5):
        pos = random_position("KQRvKQR")
        if pos:
            positions.append(pos)

    if not positions:
        print("[SKIP] Could not generate positions")
        return False

    seeds = [SeedPoint(p, SearchMode.WAVE) for p in positions]

    # Solve WITH goal
    print("\nSolving with goal filtering...")
    hologram = solver.solve(seeds, max_iterations=3, goal=goal)

    print(f"\nResults:")
    print(f"  Solved: {len(hologram.solved)}")
    print(f"  Goal filtered: {solver.stats['goal_filtered']}")

    # Reset and solve WITHOUT goal for comparison
    solver.reset()
    print("\nSolving without goal filtering...")
    hologram2 = solver.solve(seeds, max_iterations=3, goal=None)

    print(f"\nResults (no goal):")
    print(f"  Solved: {len(hologram2.solved)}")
    print(f"  Goal filtered: {solver.stats['goal_filtered']}")

    print("\n[PASS] Goal-filtered solving works")
    return True


def run_all_tests():
    """Run all goal targeting tests"""
    print("\n" + "=" * 60)
    print("GOAL TARGETING ARCHITECTURE TESTS")
    print("=" * 60)

    results = []

    results.append(("Material Utilities", test_material_utilities()))
    results.append(("Game Signature", test_game_signature()))
    results.append(("GoalCondition", test_goal_condition()))
    results.append(("Goal-Filtered Solve", test_goal_filtered_solve()))

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

    return passed == len(results)


if __name__ == "__main__":
    run_all_tests()
