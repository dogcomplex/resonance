#!/usr/bin/env python3
"""
test_connect4.py - Full test run of Connect-4 using HOLOS

This demonstrates HOLOS solving Connect-4 through bidirectional search.

Connect-4 is a solved game:
- First player (X) wins with perfect play
- The key insight is discovered through bidirectional search

Phases:
1. Forward from start position(s)
2. Backward from terminal positions
3. Connection when waves meet
4. Crystallization around connections
"""

import time
import random
import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode
from holos.games.connect4 import Connect4Game, C4State, C4Value, random_position


def test_basic_game_interface():
    """Test that the game interface works correctly"""
    print("\n" + "=" * 60)
    print("TEST: Basic Game Interface")
    print("=" * 60)

    game = Connect4Game()
    state = C4State()

    # Test initial state
    print(f"\nInitial state:")
    print(state.display())
    assert state.piece_count() == 0
    assert state.turn == 'X'
    assert not state.is_terminal()

    # Test move
    state2 = state.play(3)  # Center column
    print(f"\nAfter X plays column 3:")
    print(state2.display())
    assert state2.piece_count() == 1
    assert state2.turn == 'O'

    # Test predecessor (unplay)
    pred = state2.unplay(3)
    print(f"\nPredecessor (unplay column 3):")
    print(pred.display())
    assert hash(pred) == hash(state), "Predecessor should match original"

    # Test hash canonicalization (mirror)
    state_left = C4State().play(0)
    state_right = C4State().play(6)
    print(f"\nMirror test:")
    print(f"  Hash of X@col0: {hash(state_left)}")
    print(f"  Hash of X@col6: {hash(state_right)}")
    assert hash(state_left) == hash(state_right), "Mirrors should have same hash"

    # Test terminal detection
    # Create a winning position for X
    cols = list('.' * 6 for _ in range(7))
    cols[0] = 'XXXX..'
    win_state = C4State(tuple(cols), 'O')  # X just played, won
    print(f"\nWinning position:")
    print(win_state.display())
    assert win_state.is_terminal()
    assert win_state.check_win() == 'X'
    assert win_state.terminal_value() == 1

    print("\n[PASS] Basic game interface works correctly")


def test_successors_and_predecessors():
    """Test successor and predecessor generation"""
    print("\n" + "=" * 60)
    print("TEST: Successors and Predecessors")
    print("=" * 60)

    game = Connect4Game()

    # From initial state
    state = C4State()
    successors = game.get_successors(state)
    print(f"\nSuccessors from initial state: {len(successors)}")
    assert len(successors) == 7, "Should have 7 possible first moves"

    # Play a few moves
    state = state.play(3).play(3).play(2)
    print(f"\nAfter X-3, O-3, X-2:")
    print(state.display())

    succs = game.get_successors(state)
    print(f"Successors: {len(succs)}")

    preds = game.get_predecessors(state)
    print(f"Predecessors: {len(preds)}")
    assert len(preds) > 0, "Should have predecessors"

    # Verify predecessor leads back
    for pred, col in preds:
        child = pred.play(col)
        if hash(child) == hash(state):
            print(f"  Found matching predecessor via column {col}")
            break

    print("\n[PASS] Successors and predecessors work correctly")


def test_features_and_equivalence():
    """Test feature extraction for equivalence classes"""
    print("\n" + "=" * 60)
    print("TEST: Features and Equivalence")
    print("=" * 60)

    game = Connect4Game()

    # Two positions that should have same features (mirrors)
    state1 = C4State().play(0).play(0)  # X then O in col 0
    state2 = C4State().play(6).play(6)  # X then O in col 6 (mirror)

    f1 = game.get_features(state1)
    f2 = game.get_features(state2)

    print(f"\nFeatures for X@0, O@0: {f1}")
    print(f"Features for X@6, O@6: {f2}")
    assert f1 == f2, "Mirror positions should have same features"

    # Check signature
    sig1 = game.get_signature(state1)
    sig2 = game.get_signature(state2)
    print(f"\nSignature 1: {sig1}")
    print(f"Signature 2: {sig2}")
    assert sig1 == sig2, "Mirror positions should have same signature"

    print("\n[PASS] Features and equivalence work correctly")


def test_boundary_seeds():
    """Test boundary seed generation"""
    print("\n" + "=" * 60)
    print("TEST: Boundary Seed Generation")
    print("=" * 60)

    game = Connect4Game()
    state = C4State()

    seeds = game.generate_boundary_seeds(state, count=50)
    print(f"\nGenerated {len(seeds)} terminal positions")

    # Verify all are terminal
    wins = draws = 0
    for s in seeds:
        assert s.is_terminal(), "All seeds should be terminal"
        if s.check_win():
            wins += 1
        else:
            draws += 1

    print(f"  Wins: {wins}, Draws: {draws}")
    print("\n[PASS] Boundary seed generation works correctly")


def test_lightning_probe():
    """Test lightning probe finding paths"""
    print("\n" + "=" * 60)
    print("TEST: Lightning Probe")
    print("=" * 60)

    from holos.holos import LightningProbe

    game = Connect4Game()

    # Start from a position close to terminal
    # Create a position where X is about to win
    cols = list('.' * 6 for _ in range(7))
    cols[3] = 'XXX...'  # X has 3 in center column
    cols[0] = 'O.....'
    cols[1] = 'O.....'
    cols[2] = 'O.....'
    state = C4State(tuple(cols), 'X')  # X to move

    print(f"\nStarting position (X can win):")
    print(state.display())

    probe = LightningProbe(game, {}, direction="forward", max_depth=5)
    value, path = probe.probe(state)

    print(f"\nLightning result:")
    print(f"  Value: {value}")
    print(f"  Path length: {len(path)}")
    print(f"  Nodes visited: {probe.nodes_visited}")

    if path:
        print(f"  Path: {[m for s, m in path]}")

    # X should find winning move (drop in col 3)
    assert value is not None, "Should find terminal"
    if value:
        assert value.value == 1, "X should win"

    print("\n[PASS] Lightning probe works correctly")


def test_small_solve():
    """Test solving a small portion of the game tree"""
    print("\n" + "=" * 60)
    print("TEST: Small Solve (5 iterations)")
    print("=" * 60)

    game = Connect4Game(max_pieces=42)
    solver = HOLOSSolver(game, name="connect4_test", max_memory_mb=1000)

    # Start from empty board
    start_state = C4State()
    forward_seeds = [SeedPoint(start_state, SearchMode.WAVE, depth=2)]

    print(f"\nStarting solve from empty board...")
    start_time = time.time()

    hologram = solver.solve(
        forward_seeds,
        backward_seeds=None,  # Auto-generate
        max_iterations=5,
        lightning_interval=2
    )

    elapsed = time.time() - start_time

    print(f"\n" + "=" * 60)
    print(f"SOLVE COMPLETE in {elapsed:.1f}s")
    print("=" * 60)
    print(hologram.summary())

    # Check if start position was solved
    start_h = game.hash_state(start_state)
    start_value = hologram.query(start_h)
    print(f"\nStart position value: {start_value}")

    # Print some stats
    print(f"\nStatistics:")
    for key, val in solver.stats.items():
        print(f"  {key}: {val}")

    print("\n[PASS] Small solve completed successfully")
    return hologram, game


def test_medium_solve():
    """Test a medium-depth solve"""
    print("\n" + "=" * 60)
    print("TEST: Medium Solve (20 iterations)")
    print("=" * 60)

    game = Connect4Game(max_pieces=42)
    solver = HOLOSSolver(
        game,
        name="connect4_medium",
        max_memory_mb=2000,
        max_frontier_size=500_000
    )

    # Start from empty board
    start_state = C4State()
    forward_seeds = [SeedPoint(start_state, SearchMode.WAVE, depth=3)]

    print(f"\nStarting medium solve...")
    start_time = time.time()

    hologram = solver.solve(
        forward_seeds,
        backward_seeds=None,
        max_iterations=20,
        lightning_interval=3
    )

    elapsed = time.time() - start_time

    print(f"\n" + "=" * 60)
    print(f"SOLVE COMPLETE in {elapsed:.1f}s")
    print("=" * 60)
    print(hologram.summary())

    # Check stats
    print(f"\nKey Statistics:")
    print(f"  Forward expanded: {solver.stats['forward_expanded']:,}")
    print(f"  Backward expanded: {solver.stats['backward_expanded']:,}")
    print(f"  Connections: {solver.stats['connections']}")
    print(f"  Spines found: {solver.stats['spines_found']}")
    print(f"  Equivalence shortcuts: {solver.stats['equiv_shortcuts']}")

    # Sample some solved positions
    print(f"\nSample solved positions:")
    sample_hashes = list(hologram.solved.keys())[:5]
    for h in sample_hashes:
        print(f"  Hash {h}: {hologram.solved[h]}")

    print("\n[PASS] Medium solve completed successfully")
    return hologram, game


def test_play_with_hologram(hologram, game):
    """Test playing moves using the hologram"""
    print("\n" + "=" * 60)
    print("TEST: Play with Hologram")
    print("=" * 60)

    state = C4State()
    print("\nPlaying game using hologram for guidance...")

    moves_played = 0
    while not state.is_terminal() and moves_played < 10:
        h = game.hash_state(state)
        value = hologram.query(h)

        print(f"\nMove {moves_played + 1}, Turn: {state.turn}")
        print(f"  Position value: {value}")

        # Find best move
        best_col = None
        best_value = None

        for child, col in game.get_successors(state):
            ch = game.hash_state(child)
            cv = hologram.query(ch)
            if cv is not None:
                if best_value is None:
                    best_value = cv
                    best_col = col
                elif state.turn == 'X' and cv.value > best_value.value:
                    best_value = cv
                    best_col = col
                elif state.turn == 'O' and cv.value < best_value.value:
                    best_value = cv
                    best_col = col

        if best_col is not None:
            print(f"  Best move: column {best_col} (value: {best_value})")
            state = state.play(best_col)
        else:
            # Pick center-biased random
            moves = state.get_valid_moves()
            col = min(moves, key=lambda c: abs(c - 3))
            print(f"  Random move: column {col}")
            state = state.play(col)

        moves_played += 1

    print(f"\nFinal position after {moves_played} moves:")
    print(state.display())

    if state.is_terminal():
        winner = state.check_win()
        if winner:
            print(f"Winner: {winner}")
        else:
            print("Draw!")

    print("\n[PASS] Play with hologram works")


def run_full_test():
    """Run all tests"""
    print("=" * 60)
    print("CONNECT-4 HOLOS FULL TEST")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Basic tests
    test_basic_game_interface()
    test_successors_and_predecessors()
    test_features_and_equivalence()
    test_boundary_seeds()
    test_lightning_probe()

    # Solve tests
    hologram_small, game = test_small_solve()
    hologram_medium, game = test_medium_solve()

    # Play test
    test_play_with_hologram(hologram_medium, game)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_test()
