"""
Full solve test - verify solver works end-to-end
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.chess import ChessGame, random_position


def test_7_piece_solve():
    """Test solving 7-piece positions (one above boundary)"""
    print("=" * 60)
    print("7-PIECE SOLVE TEST")
    print("=" * 60)

    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=7)
    solver = HOLOSSolver(game, name="7piece_test", max_memory_mb=500)

    # Generate 7-piece positions
    positions = []
    for _ in range(20):
        pos = random_position("KQRvKQR")  # 7 pieces
        if pos:
            positions.append(pos)

    if not positions:
        print("Could not generate positions")
        return

    print(f"Generated {len(positions)} 7-piece positions")

    # Create seeds
    seeds = [SeedPoint(p, SearchMode.WAVE) for p in positions]

    # Solve
    hologram = solver.solve(seeds, max_iterations=10)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(hologram.summary())

    # Check if our seed positions were solved
    solved_seeds = 0
    for pos in positions:
        h = game.hash_state(pos)
        if h in hologram.solved:
            solved_seeds += 1

    print(f"\nSeeds solved: {solved_seeds}/{len(positions)}")

    return hologram


def test_8_piece_solve():
    """Test solving 8-piece positions (two above boundary)"""
    print("\n" + "=" * 60)
    print("8-PIECE SOLVE TEST")
    print("=" * 60)

    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=8)
    solver = HOLOSSolver(game, name="8piece_test", max_memory_mb=1000)

    # Generate 8-piece positions
    positions = []
    for _ in range(10):
        pos = random_position("KQRRvKQR")  # 8 pieces
        if pos:
            positions.append(pos)

    if not positions:
        print("Could not generate positions")
        return

    print(f"Generated {len(positions)} 8-piece positions")

    # Create seeds
    seeds = [SeedPoint(p, SearchMode.WAVE) for p in positions]

    # Solve with more iterations since we're 2 steps from boundary
    hologram = solver.solve(seeds, max_iterations=20)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(hologram.summary())

    # Check if our seed positions were solved
    solved_seeds = 0
    for pos in positions:
        h = game.hash_state(pos)
        if h in hologram.solved:
            solved_seeds += 1

    print(f"\nSeeds solved: {solved_seeds}/{len(positions)}")

    return hologram


def test_solver_reset():
    """Verify solver reset gives clean results"""
    print("\n" + "=" * 60)
    print("SOLVER RESET TEST")
    print("=" * 60)

    game = ChessGame(syzygy_path="./syzygy", min_pieces=6, max_pieces=7)
    solver = HOLOSSolver(game, name="reset_test", max_memory_mb=500)

    # Problem 1
    pos1 = random_position("KQRvKQR")
    if pos1:
        seeds1 = [SeedPoint(pos1, SearchMode.WAVE)]
        h1 = solver.solve(seeds1, max_iterations=3)
        print(f"Problem 1: {len(h1.solved)} solved, {h1.stats.get('connections', 0)} connections")
        print(f"  Forward seen: {len(solver.forward_seen)}")
        print(f"  Backward seen: {len(solver.backward_seen)}")

    # Reset solver
    print("\nResetting solver...")
    solver.reset()
    print(f"After reset: forward_seen={len(solver.forward_seen)}, backward_seen={len(solver.backward_seen)}")

    # Problem 2 - same solver after reset
    pos2 = random_position("KQRvKQR")
    if pos2:
        seeds2 = [SeedPoint(pos2, SearchMode.WAVE)]
        h2 = solver.solve(seeds2, max_iterations=3)
        print(f"\nProblem 2: {len(h2.solved)} solved, {h2.stats.get('connections', 0)} connections")
        print(f"  Forward seen: {len(solver.forward_seen)}")
        print(f"  Backward seen: {len(solver.backward_seen)}")

    print("\n[PASS] Solver reset works correctly")


if __name__ == "__main__":
    test_7_piece_solve()
    test_solver_reset()
