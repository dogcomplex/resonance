"""
SIMPLE SIEVE VALIDATION TESTS
=============================

Test the sieve on problems simpler than Pong to validate core mechanics:
1. TicTacToe - discrete state, clear winning strategies
2. Pattern Matching - learns to match input to output
3. Sequence Prediction - learns simple sequences

These avoid Pong's issues:
- No pixel encoding complexity
- No continuous physics
- Smaller state spaces
- Clear optimal strategies to compare against
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# Import the sieves we're testing
from spiral_sieve import SpiralSieve
from pure_wave_sieve import PureWaveSieve


# =============================================================================
# TEST 1: TICTACTOE
# =============================================================================

class TicTacToe:
    """Simple TicTacToe environment."""

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.int8)  # 0=empty, 1=X, 2=O
        self.current_player = 1  # X starts

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return board as 9x9 image for sieve (3x upscale)."""
        # Upscale for sieve - each cell becomes 3x3 block
        img = np.zeros((9, 9), dtype=np.uint8)
        for r in range(3):
            for c in range(3):
                cell = self.board[r, c]
                br, bc = r * 3, c * 3
                if cell == 1:  # X
                    img[br:br+3, bc:bc+3] = 255
                    img[br+1, bc+1] = 0  # Hollow center for X
                elif cell == 2:  # O
                    img[br:br+3, bc:bc+3] = 128
        return img

    def get_valid_actions(self) -> List[int]:
        """Return list of valid moves (0-8 for positions)."""
        valid = []
        for i in range(9):
            r, c = i // 3, i % 3
            if self.board[r, c] == 0:
                valid.append(i)
        return valid

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Make a move. Returns (state, reward, done)."""
        r, c = action // 3, action % 3

        if self.board[r, c] != 0:
            # Invalid move - loses immediately
            return self.get_state(), -1.0, True

        self.board[r, c] = self.current_player

        # Check win
        if self._check_win(self.current_player):
            return self.get_state(), 1.0, True

        # Check draw
        if len(self.get_valid_actions()) == 0:
            return self.get_state(), 0.0, True

        # Switch player
        self.current_player = 3 - self.current_player
        return self.get_state(), 0.0, False

    def _check_win(self, player: int) -> bool:
        # Rows
        for r in range(3):
            if all(self.board[r, c] == player for c in range(3)):
                return True
        # Cols
        for c in range(3):
            if all(self.board[r, c] == player for r in range(3)):
                return True
        # Diagonals
        if all(self.board[i, i] == player for i in range(3)):
            return True
        if all(self.board[i, 2-i] == player for i in range(3)):
            return True
        return False


def random_opponent_move(game: TicTacToe) -> Optional[int]:
    """Random opponent plays as O."""
    valid = game.get_valid_actions()
    if valid and game.current_player == 2:
        return np.random.choice(valid)
    return None


def test_tictactoe():
    """Test sieve on TicTacToe."""
    print("=" * 70)
    print("TEST 1: TICTACTOE")
    print("=" * 70)
    print("Sieve plays X, random opponent plays O")
    print("9 actions (positions 0-8)")
    print()

    sieve = SpiralSieve(embedding_dim=8)
    game = TicTacToe()

    wins = 0
    losses = 0
    draws = 0
    invalid_moves = 0

    NUM_GAMES = 500

    for game_num in range(NUM_GAMES):
        state = game.reset()
        done = False
        moves = 0

        while not done:
            # Sieve's turn (X)
            if game.current_player == 1:
                # Observe state
                frame_num = game_num * 20 + moves

                if game_num < 50:
                    # Explore randomly first
                    valid = game.get_valid_actions()
                    action = np.random.choice(valid) if valid else 0
                else:
                    # Use sieve
                    action = sieve.choose_action(num_actions=9)

                sieve.observe(state, action, frame_num)

                state, reward, done = game.step(action)
                moves += 1

                if reward == -1:  # Invalid move
                    invalid_moves += 1
                    losses += 1
                    done = True
                elif done and reward == 1:
                    wins += 1
                elif done and reward == 0:
                    draws += 1

            # Opponent's turn (O)
            else:
                opp_action = random_opponent_move(game)
                if opp_action is not None:
                    state, reward, done = game.step(opp_action)
                    moves += 1

                    if done and reward == 1:  # Opponent won
                        losses += 1
                    elif done and reward == 0:
                        draws += 1

        # Signal game end with length
        sieve.signal_game_end(moves)

        # Progress
        if (game_num + 1) % 100 == 0:
            total = wins + losses + draws
            print(f"Games {game_num+1}: W={wins} L={losses} D={draws} "
                  f"(W%={100*wins/total:.1f}%, Invalid={invalid_moves})")

    # Final stats
    print()
    print("FINAL RESULTS:")
    total = wins + losses + draws
    print(f"  Wins: {wins} ({100*wins/total:.1f}%)")
    print(f"  Losses: {losses} ({100*losses/total:.1f}%)")
    print(f"  Draws: {draws} ({100*draws/total:.1f}%)")
    print(f"  Invalid moves: {invalid_moves}")

    # Compare to random baseline (should be ~30% win against random)
    print()
    print("Expected vs random opponent: ~30% wins, ~20% draws, ~50% losses")

    return wins / total if total > 0 else 0


# =============================================================================
# TEST 2: PATTERN MATCHING
# =============================================================================

def test_pattern_matching():
    """
    Simple pattern matching: learn to output action that matches input pattern.

    Input: one of 4 patterns (encoded as 4x4 images)
    Output: action 0-3 matching the pattern

    This tests whether the sieve can learn simple associations.
    """
    print()
    print("=" * 70)
    print("TEST 2: PATTERN MATCHING")
    print("=" * 70)
    print("Learn to match 4 input patterns to 4 output actions")
    print()

    # Create 4 distinct patterns
    patterns = [
        np.array([[1,1,0,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]]) * 255,  # Top-left
        np.array([[0,0,1,1], [0,0,1,1], [0,0,0,0], [0,0,0,0]]) * 255,  # Top-right
        np.array([[0,0,0,0], [0,0,0,0], [1,1,0,0], [1,1,0,0]]) * 255,  # Bottom-left
        np.array([[0,0,0,0], [0,0,0,0], [0,0,1,1], [0,0,1,1]]) * 255,  # Bottom-right
    ]

    sieve = SpiralSieve(embedding_dim=8)

    correct = 0
    total = 0

    # Track performance over time
    window_correct = []

    NUM_TRIALS = 2000

    for trial in range(NUM_TRIALS):
        # Random pattern
        target = np.random.randint(0, 4)
        pattern = patterns[target].astype(np.uint8)

        # Sieve chooses action
        if trial < 100:
            action = np.random.randint(0, 4)
        else:
            action = sieve.choose_action(num_actions=4)

        # Observe pattern and action
        sieve.observe(pattern, action, trial)

        # Check if correct
        is_correct = (action == target)
        total += 1
        if is_correct:
            correct += 1

        window_correct.append(1 if is_correct else 0)
        if len(window_correct) > 100:
            window_correct.pop(0)

        # Reward signal: longer "game" if correct
        game_length = 10 if is_correct else 1
        sieve.signal_game_end(game_length)

        # Progress
        if (trial + 1) % 400 == 0:
            recent_acc = sum(window_correct) / len(window_correct) * 100
            print(f"Trial {trial+1}: Overall={100*correct/total:.1f}%, "
                  f"Recent100={recent_acc:.1f}%")

    print()
    print("FINAL RESULTS:")
    print(f"  Accuracy: {100*correct/total:.1f}%")
    print(f"  Random baseline: 25%")

    return correct / total if total > 0 else 0


# =============================================================================
# TEST 3: SEQUENCE PREDICTION
# =============================================================================

def test_sequence_prediction():
    """
    Learn to predict next in simple sequences.

    Sequence: 0 -> 1 -> 2 -> 0 -> 1 -> 2 -> ...

    The sieve sees current state as pattern, must output next action.
    """
    print()
    print("=" * 70)
    print("TEST 3: SEQUENCE PREDICTION")
    print("=" * 70)
    print("Learn sequence: 0 -> 1 -> 2 -> 0 -> ...")
    print()

    # Create visual patterns for each state
    state_patterns = [
        np.array([[1,0,0], [0,0,0], [0,0,0]]) * 255,  # State 0: top-left
        np.array([[0,1,0], [0,0,0], [0,0,0]]) * 255,  # State 1: top-middle
        np.array([[0,0,1], [0,0,0], [0,0,0]]) * 255,  # State 2: top-right
    ]

    sieve = SpiralSieve(embedding_dim=8)

    correct = 0
    total = 0
    window_correct = []

    current_state = 0

    NUM_STEPS = 3000

    for step in range(NUM_STEPS):
        # Current pattern
        pattern = state_patterns[current_state].astype(np.uint8)

        # Expected next state
        next_state = (current_state + 1) % 3

        # Sieve predicts
        if step < 100:
            action = np.random.randint(0, 3)
        else:
            action = sieve.choose_action(num_actions=3)

        # Observe
        sieve.observe(pattern, action, step)

        # Check prediction
        is_correct = (action == next_state)
        total += 1
        if is_correct:
            correct += 1

        window_correct.append(1 if is_correct else 0)
        if len(window_correct) > 100:
            window_correct.pop(0)

        # Reward: staying in sequence is "survival"
        if is_correct:
            current_state = next_state
        else:
            # Reset sequence on wrong prediction
            sieve.signal_game_end(1)
            current_state = np.random.randint(0, 3)

        # Progress
        if (step + 1) % 500 == 0:
            recent_acc = sum(window_correct) / len(window_correct) * 100
            print(f"Step {step+1}: Overall={100*correct/total:.1f}%, "
                  f"Recent100={recent_acc:.1f}%")

    print()
    print("FINAL RESULTS:")
    print(f"  Accuracy: {100*correct/total:.1f}%")
    print(f"  Random baseline: 33%")

    return correct / total if total > 0 else 0


# =============================================================================
# TEST 4: XOR LEARNING
# =============================================================================

def test_xor():
    """
    Learn XOR: (A, B) -> A XOR B

    Input patterns:
    - (0,0) -> 0
    - (0,1) -> 1
    - (1,0) -> 1
    - (1,1) -> 0

    This is a classic test - requires non-linear combination.
    """
    print()
    print("=" * 70)
    print("TEST 4: XOR LEARNING")
    print("=" * 70)
    print("Learn XOR: (A,B) -> A XOR B")
    print()

    # Visual patterns for inputs
    input_patterns = {
        (0, 0): np.array([[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]) * 255,
        (0, 1): np.array([[0,0,0,0], [0,0,0,0], [0,0,1,1], [0,0,1,1]]) * 255,
        (1, 0): np.array([[1,1,0,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]]) * 255,
        (1, 1): np.array([[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]]) * 255,
    }

    xor_targets = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0,
    }

    sieve = SpiralSieve(embedding_dim=8)

    correct = 0
    total = 0
    window_correct = []

    inputs = list(input_patterns.keys())

    NUM_TRIALS = 2000

    for trial in range(NUM_TRIALS):
        # Random input
        inp = inputs[trial % 4]  # Cycle through to ensure balance
        pattern = input_patterns[inp].astype(np.uint8)
        target = xor_targets[inp]

        # Sieve predicts
        if trial < 100:
            action = np.random.randint(0, 2)
        else:
            action = sieve.choose_action(num_actions=2)

        # Observe
        sieve.observe(pattern, action, trial)

        # Check
        is_correct = (action == target)
        total += 1
        if is_correct:
            correct += 1

        window_correct.append(1 if is_correct else 0)
        if len(window_correct) > 100:
            window_correct.pop(0)

        # Reward signal
        game_length = 10 if is_correct else 1
        sieve.signal_game_end(game_length)

        # Progress
        if (trial + 1) % 400 == 0:
            recent_acc = sum(window_correct) / len(window_correct) * 100
            print(f"Trial {trial+1}: Overall={100*correct/total:.1f}%, "
                  f"Recent100={recent_acc:.1f}%")

    print()
    print("FINAL RESULTS:")
    print(f"  Accuracy: {100*correct/total:.1f}%")
    print(f"  Random baseline: 50%")

    return correct / total if total > 0 else 0


# =============================================================================
# SIEVE WRAPPER FOR UNIFIED INTERFACE
# =============================================================================

class SieveWrapper:
    """Unified interface for testing different sieve implementations."""

    def __init__(self, sieve_type: str = "spiral"):
        self.sieve_type = sieve_type
        if sieve_type == "spiral":
            self.sieve = SpiralSieve(embedding_dim=8)
        else:
            self.sieve = PureWaveSieve()
        self.frame_num = 0

    def observe(self, state: np.ndarray, action: int):
        self.sieve.observe(state, action, self.frame_num)
        self.frame_num += 1

    def choose_action(self, num_actions: int) -> int:
        return self.sieve.choose_action(num_actions)

    def signal_game_end(self, length: int = 0):
        self.sieve.signal_game_end(length)


# =============================================================================
# COMPARATIVE TESTS
# =============================================================================

def test_pattern_matching_comparative():
    """
    Compare both sieves on pattern matching task.
    """
    print()
    print("=" * 70)
    print("COMPARATIVE TEST: PATTERN MATCHING")
    print("=" * 70)
    print()

    # Create 4 distinct patterns
    patterns = [
        np.array([[1,1,0,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]]) * 255,
        np.array([[0,0,1,1], [0,0,1,1], [0,0,0,0], [0,0,0,0]]) * 255,
        np.array([[0,0,0,0], [0,0,0,0], [1,1,0,0], [1,1,0,0]]) * 255,
        np.array([[0,0,0,0], [0,0,0,0], [0,0,1,1], [0,0,1,1]]) * 255,
    ]

    results = {}

    for sieve_type in ["spiral", "wave"]:
        print(f"\n--- Testing {sieve_type.upper()} SIEVE ---")

        sieve = SieveWrapper(sieve_type)

        correct = 0
        total = 0
        window_correct = []

        NUM_TRIALS = 2000

        for trial in range(NUM_TRIALS):
            target = np.random.randint(0, 4)
            pattern = patterns[target].astype(np.uint8)

            if trial < 100:
                action = np.random.randint(0, 4)
            else:
                action = sieve.choose_action(num_actions=4)

            sieve.observe(pattern, action)

            is_correct = (action == target)
            total += 1
            if is_correct:
                correct += 1

            window_correct.append(1 if is_correct else 0)
            if len(window_correct) > 100:
                window_correct.pop(0)

            game_length = 10 if is_correct else 1
            sieve.signal_game_end(game_length)

            if (trial + 1) % 500 == 0:
                recent_acc = sum(window_correct) / len(window_correct) * 100
                print(f"Trial {trial+1}: Overall={100*correct/total:.1f}%, Recent100={recent_acc:.1f}%")

        results[sieve_type] = correct / total
        print(f"FINAL: {100*correct/total:.1f}%")

    print()
    print("=" * 70)
    print(f"COMPARISON: Spiral={100*results['spiral']:.1f}% vs Wave={100*results['wave']:.1f}%")
    print(f"Random baseline: 25%")
    print("=" * 70)

    return results


def test_sequence_comparative():
    """Compare both sieves on sequence prediction."""
    print()
    print("=" * 70)
    print("COMPARATIVE TEST: SEQUENCE PREDICTION (0->1->2->0...)")
    print("=" * 70)

    state_patterns = [
        np.array([[1,0,0], [0,0,0], [0,0,0]]) * 255,
        np.array([[0,1,0], [0,0,0], [0,0,0]]) * 255,
        np.array([[0,0,1], [0,0,0], [0,0,0]]) * 255,
    ]

    results = {}

    for sieve_type in ["spiral", "wave"]:
        print(f"\n--- Testing {sieve_type.upper()} SIEVE ---")

        sieve = SieveWrapper(sieve_type)

        correct = 0
        total = 0
        window_correct = []
        current_state = 0

        NUM_STEPS = 3000

        for step in range(NUM_STEPS):
            pattern = state_patterns[current_state].astype(np.uint8)
            next_state = (current_state + 1) % 3

            if step < 100:
                action = np.random.randint(0, 3)
            else:
                action = sieve.choose_action(num_actions=3)

            sieve.observe(pattern, action)

            is_correct = (action == next_state)
            total += 1
            if is_correct:
                correct += 1

            window_correct.append(1 if is_correct else 0)
            if len(window_correct) > 100:
                window_correct.pop(0)

            if is_correct:
                current_state = next_state
            else:
                sieve.signal_game_end(1)
                current_state = np.random.randint(0, 3)

            if (step + 1) % 1000 == 0:
                recent_acc = sum(window_correct) / len(window_correct) * 100
                print(f"Step {step+1}: Overall={100*correct/total:.1f}%, Recent100={recent_acc:.1f}%")

        results[sieve_type] = correct / total
        print(f"FINAL: {100*correct/total:.1f}%")

    print()
    print("=" * 70)
    print(f"COMPARISON: Spiral={100*results['spiral']:.1f}% vs Wave={100*results['wave']:.1f}%")
    print(f"Random baseline: 33%")
    print("=" * 70)

    return results


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all validation tests."""
    print()
    print("#" * 70)
    print("# SIEVE VALIDATION TEST SUITE")
    print("#" * 70)
    print()

    results = {}

    # Run each test
    results['tictactoe'] = test_tictactoe()
    results['pattern_match'] = test_pattern_matching()
    results['sequence'] = test_sequence_prediction()
    results['xor'] = test_xor()

    # Summary
    print()
    print("#" * 70)
    print("# SUMMARY (SPIRAL SIEVE)")
    print("#" * 70)
    print()
    print(f"TicTacToe Win Rate:    {100*results['tictactoe']:.1f}% (random baseline: ~30%)")
    print(f"Pattern Matching:      {100*results['pattern_match']:.1f}% (random baseline: 25%)")
    print(f"Sequence Prediction:   {100*results['sequence']:.1f}% (random baseline: 33%)")
    print(f"XOR Learning:          {100*results['xor']:.1f}% (random baseline: 50%)")
    print()

    # Overall assessment
    above_random = 0
    if results['tictactoe'] > 0.30:
        above_random += 1
    if results['pattern_match'] > 0.25:
        above_random += 1
    if results['sequence'] > 0.33:
        above_random += 1
    if results['xor'] > 0.50:
        above_random += 1

    print(f"Tests above random: {above_random}/4")

    if above_random >= 3:
        print("STATUS: SIEVE SHOWS LEARNING")
    else:
        print("STATUS: SIEVE NOT LEARNING (needs investigation)")

    return results


def run_comparative_tests():
    """Run comparative tests between spiral and wave sieve."""
    print()
    print("#" * 70)
    print("# COMPARATIVE TESTS: SPIRAL vs WAVE SIEVE")
    print("#" * 70)

    pattern_results = test_pattern_matching_comparative()
    sequence_results = test_sequence_comparative()

    print()
    print("#" * 70)
    print("# FINAL COMPARISON")
    print("#" * 70)
    print()
    print(f"{'Task':<25} {'Spiral':<12} {'Wave':<12} {'Winner':<12}")
    print("-" * 60)

    for task, spiral_r, wave_r in [
        ("Pattern Matching", pattern_results['spiral'], pattern_results['wave']),
        ("Sequence Prediction", sequence_results['spiral'], sequence_results['wave']),
    ]:
        winner = "SPIRAL" if spiral_r > wave_r else "WAVE" if wave_r > spiral_r else "TIE"
        print(f"{task:<25} {100*spiral_r:<12.1f}% {100*wave_r:<12.1f}% {winner:<12}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        run_comparative_tests()
    else:
        run_all_tests()
