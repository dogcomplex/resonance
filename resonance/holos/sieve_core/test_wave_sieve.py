"""
TEST WAVE SIEVE v2
==================

Tests the single unified WaveSieve on all discrete games.
One sieve class, N dimensions it discovers on its own.
No separate anti-modes. Death signal IS the NOT wave.

Tests:
1. Pattern Matching (4 patterns -> 4 actions)
2. Sequence Memory (0 -> 1 -> 2 -> 0)
3. TicTacToe (vs random opponent)
4. Mini Sudoku (4x4) - with locality (Green's function)

Each test uses the SAME WaveSieve class with the SAME interface.
The only difference is what raw data is fed in and whether
neighbor functions / query_pos are used.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from wave_sieve import WaveSieve


# =============================================================================
# TEST ENVIRONMENTS
# =============================================================================

class PatternMatch:
    """4 patterns -> 4 actions (one-to-one mapping)."""
    def __init__(self):
        self.patterns = [
            np.array([[1, 1], [0, 0]]),
            np.array([[0, 0], [1, 1]]),
            np.array([[1, 0], [1, 0]]),
            np.array([[0, 1], [0, 1]]),
        ]
        self.current_pattern = 0

    def reset(self):
        self.current_pattern = np.random.randint(0, 4)
        return self.patterns[self.current_pattern].copy()

    def check(self, action: int) -> bool:
        return action == self.current_pattern


class SequenceMemory:
    """Sequence memory: 0 -> 1 -> 2 -> 0 -> ..."""
    def __init__(self):
        self.current = 0

    def reset(self):
        self.current = np.random.randint(0, 3)
        return self.current

    def step(self, action: int) -> Tuple[bool, int]:
        expected = (self.current + 1) % 3
        correct = (action == expected)
        if correct:
            self.current = expected
        else:
            self.current = np.random.randint(0, 3)
        return correct, self.current


class TicTacToe:
    """TicTacToe vs random opponent."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        return self.get_config()

    def get_config(self) -> Dict[tuple, int]:
        """Return board as {(r,c): value} dict - raw data for sieve."""
        config = {}
        for r in range(3):
            for c in range(3):
                config[(r, c)] = int(self.board[r, c])
        return config

    def get_valid_actions(self) -> List[int]:
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]

    def step(self, action: int) -> Tuple[Dict, bool, int]:
        if action not in self.get_valid_actions():
            return self.get_config(), True, -1

        self.board[action // 3, action % 3] = self.current_player

        if self._check_win(self.current_player):
            return self.get_config(), True, 1 if self.current_player == 1 else -1

        if len(self.get_valid_actions()) == 0:
            return self.get_config(), True, 0

        self.current_player *= -1

        if self.current_player == -1:
            valid = self.get_valid_actions()
            if valid:
                opp_action = np.random.choice(valid)
                self.board[opp_action // 3, opp_action % 3] = -1
                if self._check_win(-1):
                    return self.get_config(), True, -1
                if len(self.get_valid_actions()) == 0:
                    return self.get_config(), True, 0
            self.current_player = 1

        return self.get_config(), False, 0

    def _check_win(self, player: int) -> bool:
        for i in range(3):
            if all(self.board[i, :] == player): return True
        for i in range(3):
            if all(self.board[:, i] == player): return True
        if all(self.board[i, i] == player for i in range(3)): return True
        if all(self.board[i, 2-i] == player for i in range(3)): return True
        return False


class MiniSudoku:
    """Mini 4x4 Sudoku with fixed solution, randomized blanks."""
    def __init__(self):
        self.size = 4
        self.reset()

    def reset(self):
        self.solution = np.array([
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]
        ], dtype=np.int8)
        self.board = self.solution.copy()
        self.empty_cells = []
        positions = [(i, j) for i in range(4) for j in range(4)]
        np.random.shuffle(positions)
        for pos in positions[:6]:
            self.board[pos] = 0
            self.empty_cells.append(pos)
        self.current_cell_idx = 0

    def get_current_cell(self) -> Optional[Tuple[int, int]]:
        if self.current_cell_idx < len(self.empty_cells):
            return self.empty_cells[self.current_cell_idx]
        return None

    def get_config(self, target_cell: Tuple[int, int]) -> Dict[tuple, int]:
        """Return board + target as raw field configuration."""
        config = {}
        for r in range(4):
            for c in range(4):
                config[(r, c)] = int(self.board[r, c])
        # Target cell: mark which cell we're filling
        # Use ('target', r, c) so it has spatial position the neighbor_fn can use
        config[('target', target_cell[0], target_cell[1])] = -1
        return config

    def step(self, digit: int) -> Tuple[bool, bool]:
        """Returns (done, correct)."""
        cell = self.get_current_cell()
        if cell is None:
            return True, True
        correct_digit = self.solution[cell[0], cell[1]]
        if digit == correct_digit:
            self.board[cell[0], cell[1]] = digit
            self.current_cell_idx += 1
            done = self.current_cell_idx >= len(self.empty_cells)
            return done, True
        else:
            return True, False


# =============================================================================
# NEIGHBOR FUNCTIONS (grid topology - geometry, not domain knowledge)
# =============================================================================

def ttt_neighbors(pos):
    """TicTacToe: same row, same column, same diagonal."""
    if not isinstance(pos, tuple) or len(pos) != 2:
        return []
    r, c = pos
    nbrs = []
    for r2 in range(3):
        for c2 in range(3):
            if (r2, c2) != (r, c):
                # Same row, same col, or same diagonal
                if r2 == r or c2 == c:
                    nbrs.append((r2, c2))
                elif abs(r2 - r) == abs(c2 - c):  # diagonal
                    nbrs.append((r2, c2))
    return nbrs


def sudoku_neighbors(pos):
    """Sudoku 4x4: same row, same column, same 2x2 box."""
    # Handle target marker: ('target', r, c) -> neighbors are the grid cells
    if isinstance(pos, tuple) and len(pos) == 3 and pos[0] == 'target':
        r, c = pos[1], pos[2]
    elif isinstance(pos, tuple) and len(pos) == 2:
        r, c = pos
    else:
        return []

    nbrs = set()
    for c2 in range(4):
        if c2 != c: nbrs.add((r, c2))
    for r2 in range(4):
        if r2 != r: nbrs.add((r2, c))
    br, bc = 2 * (r // 2), 2 * (c // 2)
    for r2 in range(br, br + 2):
        for c2 in range(bc, bc + 2):
            if (r2, c2) != (r, c):
                nbrs.add((r2, c2))
    return list(nbrs)


# =============================================================================
# TESTS
# =============================================================================

def test_pattern():
    """Test pattern matching with unified WaveSieve."""
    print("=" * 60)
    print("TEST: PATTERN MATCHING (4 patterns -> 4 actions)")
    print("=" * 60)

    env = PatternMatch()
    sieve = WaveSieve()

    NUM_TRIALS = 2000
    correct = 0
    total = 0
    window = []

    for trial in range(NUM_TRIALS):
        state = env.reset()
        target = env.current_pattern

        # Raw data: just pass the numpy array directly
        # WaveSieve._normalize_input handles it
        config = state

        if trial < 200:
            action = np.random.randint(0, 4)
        else:
            action = sieve.choose_action(config, num_actions=4)

        sieve.observe(config, action, num_actions=4)

        is_correct = (action == target)
        total += 1
        if is_correct:
            correct += 1
            sieve.signal_success()
        else:
            sieve.signal_death()

        sieve.reset_episode()  # Each trial is independent

        window.append(1 if is_correct else 0)
        if len(window) > 100: window.pop(0)

        if (trial + 1) % 500 == 0:
            recent = 100 * sum(window) / len(window)
            stats = sieve.get_stats()
            print(f"  Trial {trial+1}: Recent100={recent:.1f}%  "
                  f"[modes={stats['n_modes']} heat={stats['heat_bath']:.3f}]")

    print(f"\nFINAL: {100*correct/total:.1f}% (baseline: 25%)")
    return correct / total


def test_sequence():
    """Test sequence prediction."""
    print("\n" + "=" * 60)
    print("TEST: SEQUENCE (0 -> 1 -> 2 -> 0 -> ...)")
    print("=" * 60)

    env = SequenceMemory()
    sieve = WaveSieve()

    NUM_STEPS = 3000
    correct = 0
    total = 0
    window = []

    current = env.reset()

    for step in range(NUM_STEPS):
        # Raw data: just pass the scalar directly
        # WaveSieve._normalize_input turns it into {0: value}

        if step < 200:
            action = np.random.randint(0, 3)
        else:
            action = sieve.choose_action(current, num_actions=3)

        sieve.observe(current, action, num_actions=3)

        is_correct, current = env.step(action)
        total += 1
        if is_correct:
            correct += 1
            sieve.signal_success()
        else:
            sieve.signal_death()

        # Don't reset_episode - sequence is continuous

        window.append(1 if is_correct else 0)
        if len(window) > 100: window.pop(0)

        if (step + 1) % 1000 == 0:
            recent = 100 * sum(window) / len(window)
            stats = sieve.get_stats()
            print(f"  Step {step+1}: Recent100={recent:.1f}%  "
                  f"[modes={stats['n_modes']} temporal={stats['n_temporal']} "
                  f"heat={stats['heat_bath']:.3f}]")

    print(f"\nFINAL: {100*correct/total:.1f}% (baseline: 33%)")
    return correct / total


def test_tictactoe():
    """Test TicTacToe with WaveSieve."""
    print("\n" + "=" * 60)
    print("TEST: TICTACTOE (vs random opponent)")
    print("=" * 60)

    env = TicTacToe()
    sieve = WaveSieve()

    NUM_GAMES = 2000
    wins = 0
    losses = 0
    draws = 0

    for game in range(NUM_GAMES):
        config = env.reset()
        done = False
        sieve.reset_episode()

        while not done:
            valid_actions = env.get_valid_actions()

            if game < 200:
                action = np.random.choice(valid_actions)
            else:
                action = sieve.choose_action(
                    config, num_actions=9,
                    neighbor_fn=ttt_neighbors
                )
                if action not in valid_actions:
                    action = np.random.choice(valid_actions)

            sieve.observe(config, action, num_actions=9)
            config, done, reward = env.step(action)

        if reward == 1:
            wins += 1
            sieve.signal_success()
        elif reward == -1:
            losses += 1
            sieve.signal_death()
        else:
            draws += 1
            # Draw: mild success (survived)
            sieve.signal_success()

        if (game + 1) % 400 == 0:
            total = wins + losses + draws
            stats = sieve.get_stats()
            print(f"  Game {game+1}: W={100*wins/total:.1f}% "
                  f"L={100*losses/total:.1f}% D={100*draws/total:.1f}%  "
                  f"[modes={stats['n_modes']} heat={stats['heat_bath']:.3f}]")

    total = wins + losses + draws
    print(f"\nFINAL: W={100*wins/total:.1f}%, L={100*losses/total:.1f}%, "
          f"D={100*draws/total:.1f}%")
    print("(Random baseline: ~30% W, ~30% L, ~40% D)")
    return wins / total


def test_sudoku():
    """Test Sudoku with locality (Green's function weighting)."""
    print("\n" + "=" * 60)
    print("TEST: MINI SUDOKU (locality + death-as-NOT-wave)")
    print("=" * 60)

    sieve = WaveSieve()
    solved = 0
    total_moves = 0
    correct_moves = 0

    NUM_PUZZLES = 5000

    for puzzle in range(NUM_PUZZLES):
        env = MiniSudoku()
        env.reset()
        sieve.reset_episode()

        while True:
            cell = env.get_current_cell()
            if cell is None:
                solved += 1
                break

            config = env.get_config(cell)

            # Target cell position for locality weighting
            target_pos = ('target', cell[0], cell[1])

            if puzzle < 300:
                action = np.random.randint(0, 4)
            else:
                action = sieve.choose_action(
                    config, num_actions=4,
                    neighbor_fn=sudoku_neighbors,
                    query_pos=target_pos
                )

            sieve.observe(config, action, num_actions=4)
            total_moves += 1

            digit = action + 1
            done, correct = env.step(digit)

            if correct:
                correct_moves += 1
                if done:
                    solved += 1
                    sieve.signal_success()
                    break
            else:
                sieve.signal_death()
                break

        if (puzzle + 1) % 500 == 0:
            solve_rate = 100 * solved / (puzzle + 1)
            move_acc = 100 * correct_moves / max(1, total_moves)
            stats = sieve.get_stats()
            print(f"  Puzzle {puzzle+1}: Solved={solve_rate:.1f}%, "
                  f"Moves={move_acc:.1f}%  "
                  f"[modes={stats['n_modes']} heat={stats['heat_bath']:.3f}]")

    move_acc = 100 * correct_moves / max(1, total_moves)
    print(f"\nFINAL: Solved={100*solved/NUM_PUZZLES:.1f}%, "
          f"Moves={move_acc:.1f}% (baseline: 25%)")
    return solved / NUM_PUZZLES, correct_moves, total_moves


# =============================================================================
# FORMAT-AGNOSTIC TEST: prove the sieve works with raw bytes/strings
# =============================================================================

def test_pattern_raw_formats():
    """Test that _normalize_input works with various formats."""
    print("\n" + "=" * 60)
    print("TEST: FORMAT-AGNOSTIC INPUT (same patterns, different encodings)")
    print("=" * 60)

    env = PatternMatch()
    sieve = WaveSieve()

    NUM_TRIALS = 2000
    correct = 0
    total = 0
    window = []

    for trial in range(NUM_TRIALS):
        state = env.reset()
        target = env.current_pattern

        # Randomly choose input format each trial to prove format-agnosticism
        fmt = trial % 4
        if fmt == 0:
            # numpy array (raw)
            data = state
        elif fmt == 1:
            # flat list
            data = state.flatten().tolist()
        elif fmt == 2:
            # bytes
            data = bytes(state.flatten().tolist())
        else:
            # dict (old style)
            data = {(r, c): int(state[r, c]) for r in range(2) for c in range(2)}

        if trial < 200:
            action = np.random.randint(0, 4)
        else:
            action = sieve.choose_action(data, num_actions=4)

        sieve.observe(data, action, num_actions=4)

        is_correct = (action == target)
        total += 1
        if is_correct:
            correct += 1
            sieve.signal_success()
        else:
            sieve.signal_death()

        sieve.reset_episode()

        window.append(1 if is_correct else 0)
        if len(window) > 100: window.pop(0)

        if (trial + 1) % 500 == 0:
            recent = 100 * sum(window) / len(window)
            stats = sieve.get_stats()
            print(f"  Trial {trial+1}: Recent100={recent:.1f}%  "
                  f"[modes={stats['n_modes']} heat={stats['heat_bath']:.3f}]")

    print(f"\nFINAL: {100*correct/total:.1f}% (baseline: 25%)")
    print("NOTE: Different formats create different mode keys, so this tests")
    print("      that learning happens regardless of input representation.")
    return correct / total


# =============================================================================
# MAIN
# =============================================================================

def run_all():
    print("#" * 60)
    print("# WAVE SIEVE v2 - UNIFIED PHYSICS TEST SUITE")
    print("# One sieve. Any input format. Death IS the NOT wave.")
    print("# No anti-modes. No hashing. True to physics.")
    print("#" * 60)

    results = {}

    results['pattern'] = test_pattern()
    results['sequence'] = test_sequence()
    results['tictactoe'] = test_tictactoe()

    solved, correct_s, total_s = test_sudoku()
    results['sudoku_moves'] = 100 * correct_s / max(1, total_s)
    results['sudoku_solved'] = 100 * solved

    results['format_agnostic'] = test_pattern_raw_formats()

    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)
    print(f"\nPattern Match:           {100*results['pattern']:.1f}% (baseline: 25%)")
    print(f"Sequence Memory:         {100*results['sequence']:.1f}% (baseline: 33%)")
    print(f"TicTacToe Wins:          {100*results['tictactoe']:.1f}% (baseline: ~30%)")
    print(f"Sudoku Move Accuracy:    {results['sudoku_moves']:.1f}% (baseline: 25%)")
    print(f"Sudoku Full Solves:      {results['sudoku_solved']:.1f}%")
    print(f"Format-Agnostic Pattern: {100*results['format_agnostic']:.1f}% (baseline: 25%)")

    # Pass/fail
    tests = [
        ("Pattern", results['pattern'], 0.30),
        ("Sequence", results['sequence'], 0.38),
        ("TicTacToe", results['tictactoe'], 0.35),
        ("Sudoku Moves", results['sudoku_moves'], 28),
        ("Format-Agnostic", results['format_agnostic'], 0.28),
    ]

    passed = 0
    for name, val, threshold in tests:
        if val > threshold:
            passed += 1
            print(f"\n[PASS] {name}")
        else:
            print(f"\n[FAIL] {name}")

    print(f"\nOverall: {passed}/{len(tests)} passed")

    return results


if __name__ == "__main__":
    run_all()
