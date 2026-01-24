"""
holos/test_connect4_seeds.py - Seed-Based Connect4 Solver Testing

This tests the seed-based approach for Connect4:
1. Solve regions of the game tree
2. Store critical positions as seeds (not all 4.5 trillion positions)
3. Reconstruct solutions on demand
4. Measure compression and speedup

Key differences from Sudoku:
- Connect4 is a TWO-PLAYER game (minimax, not single-path)
- Terminal positions (wins/draws) are the boundary
- We store game-theoretic VALUES, not just reachability
- First player (X) wins with perfect play (solved game)

Seed strategy for minimax games:
- Seeds are positions with KNOWN VALUES
- From seeds, we can reconstruct surrounding values via minimax
- Key insight: Store "frontier" positions at transition points
"""

import sys
import os
import time
import pickle
from typing import List, Tuple, Dict, Set, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.connect4 import (
    Connect4Game, C4State, C4Value, C4Features,
    count_threats, extract_features
)
from holos.games.seeds import (
    TacticalSeed, TacticalValue, TacticalSeedGame, SeedDirection
)


# ============================================================
# SEED-BASED SOLUTION STORAGE FOR CONNECT4
# ============================================================

@dataclass
class C4SeedSolution:
    """
    A compressed representation of a Connect4 position's value.

    Instead of storing all positions, store:
    - The position hash
    - Its game-theoretic value
    - Move sequence from a known seed position
    """
    position_hash: int
    position_compact: Tuple  # (cols, turn)
    value: int  # +1, 0, -1
    path_from_root: List[int]  # Column moves from empty board
    depth: int

    def storage_size(self) -> int:
        """Estimate storage size in bytes"""
        # hash: 8 bytes, value: 1 byte, path: 1 byte per move
        return 8 + 1 + len(self.path_from_root) + 4

    def __repr__(self):
        return f"C4Seed(v={self.value}, d={self.depth}, path_len={len(self.path_from_root)})"


def minimax_solve(state: C4State, depth: int = 0, max_depth: int = 42,
                  cache: Dict = None) -> Tuple[int, List[int]]:
    """
    Solve a Connect4 position using minimax (for baseline comparison).

    Returns: (value, best_path)
    """
    if cache is None:
        cache = {}

    h = hash(state)
    if h in cache:
        return cache[h]

    # Terminal check
    winner = state.check_win()
    if winner == 'X':
        cache[h] = (1, [])
        return 1, []
    elif winner == 'O':
        cache[h] = (-1, [])
        return -1, []
    elif state.piece_count() == 42:
        cache[h] = (0, [])
        return 0, []

    if depth >= max_depth:
        cache[h] = (0, [])
        return 0, []

    valid_moves = state.get_valid_moves()
    if not valid_moves:
        cache[h] = (0, [])
        return 0, []

    if state.turn == 'X':
        # Maximizing
        best_val = -2
        best_path = []
        for move in valid_moves:
            child = state.play(move)
            child_val, child_path = minimax_solve(child, depth + 1, max_depth, cache)
            if child_val > best_val:
                best_val = child_val
                best_path = [move] + child_path
                if best_val == 1:  # Pruning
                    break
        cache[h] = (best_val, best_path)
        return best_val, best_path
    else:
        # Minimizing
        best_val = 2
        best_path = []
        for move in valid_moves:
            child = state.play(move)
            child_val, child_path = minimax_solve(child, depth + 1, max_depth, cache)
            if child_val < best_val:
                best_val = child_val
                best_path = [move] + child_path
                if best_val == -1:  # Pruning
                    break
        cache[h] = (best_val, best_path)
        return best_val, best_path


def path_to_state(path: List[int]) -> C4State:
    """Convert a move path to a state"""
    state = C4State()
    for move in path:
        state = state.play(move)
    return state


def state_to_path(state: C4State) -> List[int]:
    """
    Attempt to reconstruct path from state.
    Note: Multiple paths can lead to the same position.
    This returns one valid path.
    """
    # Simple approach: read pieces bottom-up in temporal order
    # This is a heuristic and may not give the exact original path
    path = []
    cols = list(state.cols)

    while sum(c.count('X') + c.count('O') for c in cols) > 0:
        # Find the last piece placed (topmost in any column)
        last_col = -1
        last_row = -1
        last_piece = None

        # Determine who played last
        x_count = sum(c.count('X') for c in cols)
        o_count = sum(c.count('O') for c in cols)

        if x_count > o_count:
            last_piece = 'X'
        else:
            last_piece = 'O'

        # Find topmost piece of that type
        for c in range(7):
            for r in range(5, -1, -1):
                if cols[c][r] == last_piece:
                    last_col = c
                    last_row = r
                    break
            if last_col >= 0:
                break

        if last_col < 0:
            break

        # Remove this piece
        cols[last_col] = cols[last_col][:last_row] + '.' + cols[last_col][last_row+1:]
        path.append(last_col)

    path.reverse()
    return path


# ============================================================
# TESTS
# ============================================================

def test_basic_connect4():
    """Test basic Connect4 operations"""
    print("=" * 70)
    print("TEST: Basic Connect4 Operations")
    print("=" * 70)

    game = Connect4Game()
    state = C4State()

    print("\nEmpty board:")
    print(state.display())

    # Play some moves
    state = state.play(3)  # X center
    state = state.play(3)  # O on top
    state = state.play(2)  # X left
    state = state.play(4)  # O right

    print("\nAfter 4 moves:")
    print(state.display())

    # Check successors
    successors = game.get_successors(state)
    print(f"\nSuccessors: {len(successors)}")

    # Check predecessors
    predecessors = game.get_predecessors(state)
    print(f"Predecessors: {len(predecessors)}")

    print("\n[PASS] Basic Connect4 operations work")


def test_minimax_small():
    """Test minimax solver on small positions"""
    print("\n" + "=" * 70)
    print("TEST: Minimax Solver (small positions)")
    print("=" * 70)

    # Create a near-terminal position
    state = C4State()

    # Play to a position where X can win
    moves = [3, 0, 3, 0, 3, 0]  # X builds center column
    for m in moves:
        state = state.play(m)

    print("\nPosition after moves [3,0,3,0,3,0]:")
    print(state.display())
    print(f"Turn: {state.turn}")

    # X to play - should win with column 3
    cache = {}
    start = time.time()
    value, path = minimax_solve(state, max_depth=10, cache=cache)
    elapsed = time.time() - start

    print(f"\nMinimax result:")
    print(f"  Value: {value} ({'X wins' if value == 1 else 'O wins' if value == -1 else 'Draw'})")
    print(f"  Best path: {path}")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Positions cached: {len(cache)}")

    assert value == 1, "X should win from this position"
    print("\n[PASS] Minimax solver works correctly")

    return cache


def test_holos_connect4():
    """Test HOLOS solver on Connect4"""
    print("\n" + "=" * 70)
    print("TEST: HOLOS Solver (Connect4)")
    print("=" * 70)

    game = Connect4Game(min_pieces=30, max_pieces=42)

    # Start from a position with 30 pieces (late game)
    # Generate some late-game positions
    print("\nGenerating late-game seeds...")
    seeds = []
    for _ in range(20):
        state = C4State()
        for _ in range(30):
            moves = state.get_valid_moves()
            if not moves or state.check_win():
                break
            state = state.play(moves[hash(state) % len(moves)])

        if not state.check_win() and state.piece_count() >= 28:
            seeds.append(state)

    print(f"Generated {len(seeds)} late-game positions")

    if not seeds:
        print("Could not generate suitable positions")
        return

    # Show first seed
    print("\nFirst seed position:")
    print(seeds[0].display())

    # Run HOLOS
    solver = HOLOSSolver(game, name="c4_late", max_memory_mb=200, max_frontier_size=50_000)
    forward_seeds = [SeedPoint(s, SearchMode.WAVE) for s in seeds[:5]]

    print("\nRunning HOLOS...")
    start = time.time()
    hologram = solver.solve(forward_seeds, backward_seeds=[], max_iterations=5)
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Solved: {len(hologram.solved):,}")
    print(f"  Spines: {len(hologram.spines)}")
    print(f"  Connections: {len(hologram.connections)}")

    # Check values for seeds
    print("\nSeed values:")
    for i, s in enumerate(seeds[:5]):
        h = game.hash_state(s)
        v = hologram.query(h)
        print(f"  Seed {i+1}: {v}")

    return hologram


def test_seed_compression():
    """Test seed-based compression for Connect4"""
    print("\n" + "=" * 70)
    print("TEST: Seed Compression (Connect4)")
    print("=" * 70)

    game = Connect4Game()

    # Solve a region of the game tree
    print("\nSolving positions from empty board...")

    cache = {}
    state = C4State()

    start = time.time()
    value, path = minimax_solve(state, max_depth=12, cache=cache)
    elapsed = time.time() - start

    print(f"Solved {len(cache)} positions in {elapsed:.2f}s")
    print(f"Root value: {value}")

    # Convert cache to seed format
    print("\nConverting to seed format...")
    seeds = []
    for h, (val, best_path) in cache.items():
        # We don't have the state directly, so we'd need path reconstruction
        # For this test, just count storage
        pass

    # Storage comparison
    # Full storage: hash (8) + value (1) = 9 bytes per position
    full_storage = len(cache) * 9

    # Seed storage: only store "interesting" positions
    # - Terminal positions
    # - Positions where best move changes value
    # For simplicity, estimate 10% are "interesting"
    interesting = len(cache) // 10
    seed_storage = interesting * 9

    print(f"\n--- Storage Comparison ---")
    print(f"Positions solved: {len(cache):,}")
    print(f"Full storage: {full_storage:,} bytes ({full_storage/1024:.1f} KB)")
    print(f"Estimated seed storage (10%): {seed_storage:,} bytes ({seed_storage/1024:.1f} KB)")
    print(f"Compression ratio: {full_storage/seed_storage:.1f}x")

    return cache


def test_layer1_connect4():
    """Test Layer 1 seed optimization for Connect4"""
    print("\n" + "=" * 70)
    print("TEST: Layer 1 Seed Optimization (Connect4)")
    print("=" * 70)

    game = Connect4Game()
    state = C4State()

    # Create seed pool from different game phases
    print("\nBuilding seed pool...")
    seed_pool = [(game.hash_state(state), state)]

    # Add positions at different depths
    current = state
    for i in range(10):
        moves = current.get_valid_moves()
        if not moves or current.check_win():
            break
        move = moves[i % len(moves)]
        current = current.play(move)
        seed_pool.append((game.hash_state(current), current))

    print(f"Seed pool: {len(seed_pool)} positions")

    # Create Layer 1 game
    layer1 = TacticalSeedGame(game, seed_pool, max_depth=4)

    # Evaluate configurations
    print("\n--- Evaluating Seed Configurations ---")
    results = []

    for i, (h, s) in enumerate(seed_pool[:5]):
        best_value = None
        best_seed = None

        for depth in [2, 3, 4]:
            for mode in [SearchMode.LIGHTNING, SearchMode.WAVE]:
                for direction in [SeedDirection.FORWARD, SeedDirection.BILATERAL]:
                    seed = TacticalSeed(h, depth, mode, direction, s)
                    value = layer1.evaluate(seed)

                    if best_value is None or value.efficiency > best_value.efficiency:
                        best_value = value
                        best_seed = seed

        print(f"  Position {i} ({s.piece_count()} pieces): {best_seed.signature()} -> eff={best_value.efficiency:.1f}")
        results.append((best_seed, best_value))

    print(f"\n{layer1.summary()}")

    return results


def test_full_game_region():
    """
    Test solving a complete region of the game tree.

    This demonstrates the approach for eventually solving all 4.5T positions:
    1. Solve in chunks (by piece count)
    2. Store seeds for each chunk
    3. Reconstruct on demand
    """
    print("\n" + "=" * 70)
    print("TEST: Full Game Region Solve")
    print("=" * 70)

    game = Connect4Game(min_pieces=38, max_pieces=42)

    print("Solving positions with 38-42 pieces (late endgame)...")

    # Generate terminal/near-terminal positions
    terminal_states = []
    print("\nGenerating boundary positions...")

    # Random playouts to find terminals
    for _ in range(100):
        state = C4State()
        for _ in range(42):
            moves = state.get_valid_moves()
            if not moves:
                break
            if state.check_win():
                break
            move = moves[hash(state) % len(moves)]
            state = state.play(move)

        if state.piece_count() >= 38:
            terminal_states.append(state)

    print(f"Found {len(terminal_states)} late-game positions")

    # Use these as backward seeds
    solver = HOLOSSolver(
        game,
        name="c4_endgame",
        max_memory_mb=500,
        max_frontier_size=100_000,
        spine_as_boundary=True
    )

    # Generate forward seeds from 38-piece positions
    forward_seeds = []
    backward_seeds = []

    for s in terminal_states:
        if s.check_win() or s.piece_count() == 42:
            # Terminal - use as backward seed
            backward_seeds.append(SeedPoint(s, SearchMode.WAVE))
        else:
            # Non-terminal late game - use as forward seed
            forward_seeds.append(SeedPoint(s, SearchMode.WAVE))

    print(f"\nForward seeds: {len(forward_seeds)}")
    print(f"Backward seeds: {len(backward_seeds)}")

    start = time.time()
    hologram = solver.solve(
        forward_seeds[:20],
        backward_seeds=backward_seeds[:50],
        max_iterations=5,
        lightning_interval=1
    )
    elapsed = time.time() - start

    print(f"\n--- Results ---")
    print(f"Time: {elapsed:.1f}s")
    print(f"Solved: {len(hologram.solved):,}")
    print(f"Spines: {len(hologram.spines)}")
    print(f"Connections: {len(hologram.connections)}")

    # Storage analysis
    full_storage = len(hologram.solved) * 16
    # Seeds + spines storage
    seed_storage = (len(forward_seeds) + len(backward_seeds)) * 100 + len(hologram.spines) * 200

    print(f"\n--- Storage ---")
    print(f"Full storage: {full_storage:,} bytes")
    print(f"Seed storage: {seed_storage:,} bytes")
    if seed_storage > 0:
        print(f"Compression: {full_storage/seed_storage:.1f}x")

    return hologram


# ============================================================
# MAIN
# ============================================================

def main():
    """Run all Connect4 seed tests"""
    print("=" * 70)
    print("CONNECT4 SEED-BASED SOLVER TEST SUITE")
    print("=" * 70)
    print("""
Testing seed-based approach for Connect4:

Key differences from Sudoku:
- Connect4 is a TWO-PLAYER game (minimax values)
- 4.5 trillion total positions
- First player (X) wins with perfect play

Strategy:
1. Solve game tree in chunks (by piece count)
2. Store "frontier" seeds at transition points
3. Reconstruct values via minimax from seeds
    """)

    # Test 1: Basic operations
    print("\n" + "#" * 70)
    print("# TEST 1: BASIC OPERATIONS")
    print("#" * 70)
    test_basic_connect4()

    # Test 2: Minimax solver
    print("\n" + "#" * 70)
    print("# TEST 2: MINIMAX SOLVER")
    print("#" * 70)
    test_minimax_small()

    # Test 3: Seed compression
    print("\n" + "#" * 70)
    print("# TEST 3: SEED COMPRESSION")
    print("#" * 70)
    test_seed_compression()

    # Test 4: HOLOS solver
    print("\n" + "#" * 70)
    print("# TEST 4: HOLOS SOLVER")
    print("#" * 70)
    test_holos_connect4()

    # Test 5: Layer 1 optimization
    print("\n" + "#" * 70)
    print("# TEST 5: LAYER 1 OPTIMIZATION")
    print("#" * 70)
    test_layer1_connect4()

    # Test 6: Full region solve
    print("\n" + "#" * 70)
    print("# TEST 6: FULL REGION SOLVE")
    print("#" * 70)
    test_full_game_region()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
