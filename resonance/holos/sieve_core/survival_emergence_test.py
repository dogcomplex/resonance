"""
SURVIVAL EMERGENCE TEST
=======================

HYPOTHESIS: Survival doesn't need explicit reward.
Patterns that survive longer naturally get amplified because:
1. More frames = more couplings
2. More couplings = more backward-propagating waves
3. Those waves constructively interfere with patterns that led to them
4. So surviving patterns dominate simply by EXISTING LONGER

This is the anthropic principle as wave mechanics.

TEST: Run a game with NO explicit survival signal.
Just observe, couple, propagate.
See if "good" actions (ones that lead to longer games) naturally emerge
with higher amplitude than "bad" actions.
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
import hashlib


class PureWaveToken:
    """Token with NO survival tracking. Just amplitude and couplings."""
    __slots__ = ['id', 'amplitude', 'spatial_neighbors', 'temporal_prev', 'temporal_next']

    def __init__(self, token_id: str):
        self.id = token_id
        self.amplitude: float = 0.0
        self.spatial_neighbors: Set[str] = set()
        self.temporal_prev: Dict[str, float] = {}  # What preceded this
        self.temporal_next: Dict[str, float] = {}  # What followed this


class PureWaveSieve:
    """
    Sieve with NO explicit survival/goal mechanism.

    ONLY:
    - Observe pixels and actions
    - Build couplings from co-occurrence and temporal sequence
    - Let amplitudes propagate through couplings

    HYPOTHESIS: Actions that lead to longer games will naturally
    have higher amplitude because they create more coupling chains.
    """

    def __init__(self):
        self.tokens: Dict[str, PureWaveToken] = {}
        self.frame_num: int = 0
        self.prev_active: Set[str] = set()
        self.current_active: Set[str] = set()

        # Parameters (will self-tune toward criticality)
        self._coupling_strength: float = 0.1
        self._decay_rate: float = 0.001

        self._manifold_built = False

    def _hash(self, data: str) -> str:
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _get_token(self, token_id: str) -> PureWaveToken:
        if token_id not in self.tokens:
            self.tokens[token_id] = PureWaveToken(token_id)
        return self.tokens[token_id]

    def _build_manifold(self, h: int, w: int):
        if self._manifold_built:
            return
        for y in range(h):
            for x in range(w):
                pixel_id = self._hash(f"p_{y}_{x}")
                token = self._get_token(pixel_id)
                if y > 0:
                    token.spatial_neighbors.add(self._hash(f"p_{y-1}_{x}"))
                if y < h - 1:
                    token.spatial_neighbors.add(self._hash(f"p_{y+1}_{x}"))
                if x > 0:
                    token.spatial_neighbors.add(self._hash(f"p_{y}_{x-1}"))
                if x < w - 1:
                    token.spatial_neighbors.add(self._hash(f"p_{y}_{x+1}"))
        self._manifold_built = True

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """Observe with NO survival signal. Just coupling."""
        self.frame_num = frame_num

        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float)
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val

        h, w = frame.shape
        self._build_manifold(h, w)

        self.prev_active = self.current_active.copy()
        self.current_active = set()

        # Observe all pixels
        for y in range(h):
            for x in range(w):
                intensity = frame[y, x]
                pixel_id = self._hash(f"p_{y}_{x}")
                token = self._get_token(pixel_id)
                token.amplitude = intensity
                if intensity > 0:
                    self.current_active.add(pixel_id)

        # Action token
        action_id = self._hash(f"action_{action}")
        action_token = self._get_token(action_id)
        action_token.amplitude = 1.0
        self.current_active.add(action_id)

        # Learn couplings
        self._learn_couplings()

        # Propagate amplitudes through couplings
        self._propagate()

    def _learn_couplings(self):
        """Build couplings from observation."""
        cs = self._coupling_strength
        dr = self._decay_rate

        for curr_id in self.current_active:
            curr_token = self.tokens.get(curr_id)
            if not curr_token:
                continue

            # Temporal: what was active before this?
            for prev_id in self.prev_active:
                # Self-link (same position across time)
                if prev_id == curr_id:
                    curr_token.temporal_prev[prev_id] = (
                        curr_token.temporal_prev.get(prev_id, 0) + cs
                    )
                # Neighbor-link (adjacent position was active)
                elif prev_id in curr_token.spatial_neighbors:
                    curr_token.temporal_prev[prev_id] = (
                        curr_token.temporal_prev.get(prev_id, 0) + cs
                    )
                    prev_token = self.tokens.get(prev_id)
                    if prev_token:
                        prev_token.temporal_next[curr_id] = (
                            prev_token.temporal_next.get(curr_id, 0) + cs
                        )

        # Decay all couplings
        for token in self.tokens.values():
            for key in list(token.temporal_prev.keys()):
                token.temporal_prev[key] *= (1 - dr)
                if token.temporal_prev[key] < 1e-12:
                    del token.temporal_prev[key]
            for key in list(token.temporal_next.keys()):
                token.temporal_next[key] *= (1 - dr)
                if token.temporal_next[key] < 1e-12:
                    del token.temporal_next[key]

    def _propagate(self):
        """
        Propagate amplitudes through the TEMPORAL CHAIN.

        KEY INSIGHT: The backward echo must flow through CAUSAL connections,
        not just co-occurrence. If A->B->C is a temporal chain, then
        high amplitude at C should flow back to B, then to A.

        This is survival emergence:
        - Longer games have longer chains
        - Amplitude propagates backward through chains
        - Early tokens in long chains get more amplitude

        The mechanism: each token's amplitude contributes to tokens that LED to it.
        """
        # Collect amplitude contributions
        contributions = {tid: 0.0 for tid in self.tokens}

        # Forward pass: currently active tokens boost their amplitude
        for token_id in self.current_active:
            contributions[token_id] += 0.1

        # BACKWARD pass: amplitude flows from current to PREDECESSORS
        # This is the key mechanism for survival emergence
        for token in self.tokens.values():
            if token.amplitude > 0:
                # For each thing that LED to this token (temporal_prev)
                for prev_id, coupling in token.temporal_prev.items():
                    if prev_id in self.tokens:
                        # Amplitude flows backward proportional to coupling
                        contributions[prev_id] += coupling * token.amplitude * 0.001

        # Apply contributions
        for tid, contrib in contributions.items():
            self.tokens[tid].amplitude += contrib

        # Decay ALL tokens
        for token in self.tokens.values():
            token.amplitude *= 0.99

    def get_action_amplitudes(self) -> Dict[int, float]:
        """Get amplitude of each action token."""
        result = {}
        for a in range(3):
            action_id = self._hash(f"action_{a}")
            if action_id in self.tokens:
                result[a] = self.tokens[action_id].amplitude
            else:
                result[a] = 0.0
        return result

    def choose_action(self) -> int:
        """
        Choose action based on CONTEXTUAL amplitude.

        The key insight: It's not "which action has highest amplitude overall"
        but "which action is most strongly coupled to WHAT'S CURRENTLY ACTIVE"

        This is where survival emergence should appear:
        - Action+context combinations that led to longer games
        - Have stronger couplings (more co-occurrences over more frames)
        - So they get chosen more often in similar contexts
        """
        action_scores = {}

        for a in range(3):
            action_id = self._hash(f"action_{a}")
            action_token = self.tokens.get(action_id)

            if not action_token:
                action_scores[a] = 0.0
                continue

            # Score = how strongly is this action coupled to current context?
            score = action_token.amplitude  # Base amplitude

            # Add coupling strength to currently active tokens
            # This is the "context" part
            for active_id in self.current_active:
                if active_id == action_id:
                    continue
                active_token = self.tokens.get(active_id)
                if active_token:
                    # How much did this active token and action co-occur?
                    coupling = action_token.temporal_prev.get(active_id, 0)
                    coupling += action_token.temporal_next.get(active_id, 0)
                    score += coupling * active_token.amplitude

            action_scores[a] = score

        # Choose based on scores
        total = sum(action_scores.values())
        if total < 0.001:
            return np.random.randint(0, 3)

        probs = {a: s / total for a, s in action_scores.items()}
        r = np.random.random()
        cumulative = 0.0
        for a, p in probs.items():
            cumulative += p
            if r < cumulative:
                return a
        return 2


def test_pattern_persistence():
    """
    Alternative test: Do PATTERNS from longer games dominate the sieve?

    Instead of asking "which action is best", ask:
    "Which patterns have highest amplitude?"
    "Do patterns from long games dominate patterns from short games?"
    """
    print("=" * 70)
    print("PATTERN PERSISTENCE TEST")
    print("Do patterns from longer games naturally dominate?")
    print("=" * 70)

    FRAME_SIZE = 21

    sieve = PureWaveSieve()

    # Play several games of different lengths and track total amplitude
    total_amplitude_per_game = []

    for game in range(30):
        ball_x, ball_y = 10.5, 5.0
        ball_dx = np.random.choice([-0.5, 0.5])
        ball_dy = 0.5
        paddle_x = 10.5

        # For this test, manipulate how long games last
        # Odd games: always miss (short)
        # Even games: always hit (long, until 200 frames)
        forced_outcome = (game % 2 == 0)

        frames = 0
        game_amplitude = 0.0

        for _ in range(200):
            pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
            bx, by = int(ball_x), int(ball_y)
            pixels[max(0,by-1):min(FRAME_SIZE,by+1), max(0,bx-1):min(FRAME_SIZE,bx+1)] = 255
            px = int(paddle_x)
            pixels[FRAME_SIZE-2:FRAME_SIZE, max(0,px-2):min(FRAME_SIZE,px+2)] = 200

            action = np.random.randint(0, 3)
            sieve.observe(pixels, action, sieve.frame_num + 1)

            # Track amplitude accumulated this frame
            game_amplitude += sum(t.amplitude for t in sieve.tokens.values())

            ball_x += ball_dx
            ball_y += ball_dy
            if ball_x <= 1 or ball_x >= FRAME_SIZE - 1:
                ball_dx *= -1
            if ball_y <= 1:
                ball_dy = abs(ball_dy)
            if action == 0:
                paddle_x = max(3, paddle_x - 1)
            elif action == 2:
                paddle_x = min(FRAME_SIZE - 3, paddle_x + 1)

            frames += 1

            if ball_y >= FRAME_SIZE - 2:
                if forced_outcome:
                    # Force hit
                    ball_dy = -abs(ball_dy)
                    ball_y = FRAME_SIZE - 3
                else:
                    # Force miss
                    break

        total_amplitude_per_game.append((frames, game_amplitude))
        print(f"Game {game}: {frames} frames, total_amp={game_amplitude:.1f}, type={'LONG' if forced_outcome else 'SHORT'}")

    # Compare amplitude from long vs short games
    long_games = [amp for (f, amp) in total_amplitude_per_game if f >= 100]
    short_games = [amp for (f, amp) in total_amplitude_per_game if f < 100]

    print(f"\nLong games avg amplitude: {np.mean(long_games):.1f}")
    print(f"Short games avg amplitude: {np.mean(short_games):.1f}")
    print(f"Ratio: {np.mean(long_games) / np.mean(short_games):.2f}x")

    return sieve


def test_survival_emergence():
    """
    Test whether survival-promoting actions naturally get higher amplitude
    WITHOUT any explicit survival signal.
    """
    print("=" * 70)
    print("SURVIVAL EMERGENCE TEST")
    print("NO explicit survival signal - just wave propagation")
    print("Hypothesis: Longer games naturally amplify the actions that led to them")
    print("=" * 70)

    FRAME_SIZE = 21  # Small for speed: 21x21 = 441 pixels
    NUM_GAMES = 100

    sieve = PureWaveSieve()

    game_lengths = []
    action_choices = {0: 0, 1: 0, 2: 0}

    for game_num in range(NUM_GAMES):
        # Reset game state
        ball_x, ball_y = 10.5, 5.0
        ball_dx = np.random.choice([-0.5, 0.5])
        ball_dy = 0.5
        paddle_x = 10.5

        frames_survived = 0

        while True:
            # Create frame
            pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

            # Ball
            bx, by = int(ball_x), int(ball_y)
            pixels[max(0,by-1):min(FRAME_SIZE,by+1), max(0,bx-1):min(FRAME_SIZE,bx+1)] = 255

            # Paddle
            px = int(paddle_x)
            pixels[FRAME_SIZE-2:FRAME_SIZE, max(0,px-2):min(FRAME_SIZE,px+2)] = 200

            # Choose action based purely on amplitude
            if game_num < 10:
                # First 10 games: random to bootstrap
                action = np.random.randint(0, 3)
            else:
                action = sieve.choose_action()

            action_choices[action] += 1

            # Observe - NO survival signal, just the frame
            sieve.observe(pixels, action, sieve.frame_num + 1)

            # Physics
            ball_x += ball_dx
            ball_y += ball_dy

            if ball_x <= 1 or ball_x >= FRAME_SIZE - 1:
                ball_dx *= -1
                ball_x = np.clip(ball_x, 1, FRAME_SIZE - 1)

            if ball_y <= 1:
                ball_dy = abs(ball_dy)

            # Apply action
            if action == 0:
                paddle_x = max(3, paddle_x - 1)
            elif action == 2:
                paddle_x = min(FRAME_SIZE - 3, paddle_x + 1)

            frames_survived += 1

            # Check for game over
            if ball_y >= FRAME_SIZE - 2:
                hit = abs(ball_x - paddle_x) < 3
                if hit:
                    # Ball bounces back - game continues
                    ball_dy = -abs(ball_dy)
                    ball_y = FRAME_SIZE - 3
                else:
                    # Miss - game ends
                    break

            # Safety limit
            if frames_survived > 500:
                break

        game_lengths.append(frames_survived)

        if (game_num + 1) % 10 == 0:
            avg_length = np.mean(game_lengths[-10:])
            amps = sieve.get_action_amplitudes()
            print(f"\nGames {game_num-8}-{game_num+1}:")
            print(f"  Avg game length: {avg_length:.1f} frames")
            print(f"  Action amplitudes: {amps}")
            print(f"  Action choices: {action_choices}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Compare first 10 games (random) vs last 10 games (amplitude-guided)
    first_10_avg = np.mean(game_lengths[:10])
    last_10_avg = np.mean(game_lengths[-10:])

    print(f"First 10 games (random): avg {first_10_avg:.1f} frames")
    print(f"Last 10 games (amplitude-guided): avg {last_10_avg:.1f} frames")
    print(f"Improvement: {(last_10_avg/first_10_avg - 1)*100:.1f}%")

    print(f"\nFinal action amplitudes:")
    amps = sieve.get_action_amplitudes()
    for a, amp in sorted(amps.items()):
        print(f"  action_{a}: {amp:.4f}")

    # The key test: do longer games correlate with action amplitude?
    # We need to track which actions led to longer games
    print(f"\nTotal action choices: {action_choices}")

    return sieve, game_lengths


def test_controlled_comparison():
    """
    More controlled test: Run two sieves on identical game sequences.
    One chooses randomly, one uses amplitudes.

    If survival emergence is real, the amplitude-guided sieve
    should naturally favor actions that led to longer games,
    even though it never received explicit reward.
    """
    print("\n" + "=" * 70)
    print("CONTROLLED COMPARISON TEST")
    print("=" * 70)

    FRAME_SIZE = 21

    # First: Run random games to collect "training" data
    print("\nPhase 1: Collecting experience with random play...")

    sieve = PureWaveSieve()

    for game in range(30):
        ball_x, ball_y = 10.5, 5.0
        ball_dx = np.random.choice([-0.5, 0.5])
        ball_dy = 0.5
        paddle_x = 10.5

        for _ in range(200):  # Max frames per game
            pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
            bx, by = int(ball_x), int(ball_y)
            pixels[max(0,by-1):min(FRAME_SIZE,by+1), max(0,bx-1):min(FRAME_SIZE,bx+1)] = 255
            px = int(paddle_x)
            pixels[FRAME_SIZE-2:FRAME_SIZE, max(0,px-2):min(FRAME_SIZE,px+2)] = 200

            action = np.random.randint(0, 3)
            sieve.observe(pixels, action, sieve.frame_num + 1)

            ball_x += ball_dx
            ball_y += ball_dy

            if ball_x <= 1 or ball_x >= FRAME_SIZE - 1:
                ball_dx *= -1
            if ball_y <= 1:
                ball_dy = abs(ball_dy)

            if action == 0:
                paddle_x = max(3, paddle_x - 1)
            elif action == 2:
                paddle_x = min(FRAME_SIZE - 3, paddle_x + 1)

            if ball_y >= FRAME_SIZE - 2:
                if abs(ball_x - paddle_x) < 3:
                    ball_dy = -abs(ball_dy)
                    ball_y = FRAME_SIZE - 3
                else:
                    break

    print(f"After training, action amplitudes:")
    amps = sieve.get_action_amplitudes()
    for a, amp in sorted(amps.items()):
        print(f"  action_{a}: {amp:.6f}")

    # Phase 2: Test amplitude-guided vs random
    print("\nPhase 2: Testing amplitude-guided play vs random...")

    random_lengths = []
    guided_lengths = []

    for test in range(20):
        # Same initial conditions
        np.random.seed(test * 42)
        ball_x_init = 10.5
        ball_dx_init = np.random.choice([-0.5, 0.5])

        # Random play
        ball_x, ball_y = ball_x_init, 5.0
        ball_dx, ball_dy = ball_dx_init, 0.5
        paddle_x = 10.5
        frames = 0
        for _ in range(300):
            action = np.random.randint(0, 3)
            ball_x += ball_dx
            ball_y += ball_dy
            if ball_x <= 1 or ball_x >= FRAME_SIZE - 1:
                ball_dx *= -1
            if ball_y <= 1:
                ball_dy = abs(ball_dy)
            if action == 0:
                paddle_x = max(3, paddle_x - 1)
            elif action == 2:
                paddle_x = min(FRAME_SIZE - 3, paddle_x + 1)
            frames += 1
            if ball_y >= FRAME_SIZE - 2:
                if abs(ball_x - paddle_x) < 3:
                    ball_dy = -abs(ball_dy)
                    ball_y = FRAME_SIZE - 3
                else:
                    break
        random_lengths.append(frames)

        # Amplitude-guided play (same starting conditions)
        # IMPORTANT: Must observe to update context for choose_action!
        ball_x, ball_y = ball_x_init, 5.0
        ball_dx, ball_dy = ball_dx_init, 0.5
        paddle_x = 10.5
        frames = 0
        for _ in range(300):
            # Create frame and observe it (so sieve has context)
            pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
            bx, by = int(ball_x), int(ball_y)
            pixels[max(0,by-1):min(FRAME_SIZE,by+1), max(0,bx-1):min(FRAME_SIZE,bx+1)] = 255
            px = int(paddle_x)
            pixels[FRAME_SIZE-2:FRAME_SIZE, max(0,px-2):min(FRAME_SIZE,px+2)] = 200

            # Get action BEFORE observing (so we choose based on current state)
            action = sieve.choose_action()

            # Now observe with the chosen action
            sieve.observe(pixels, action, sieve.frame_num + 1)

            ball_x += ball_dx
            ball_y += ball_dy
            if ball_x <= 1 or ball_x >= FRAME_SIZE - 1:
                ball_dx *= -1
            if ball_y <= 1:
                ball_dy = abs(ball_dy)
            if action == 0:
                paddle_x = max(3, paddle_x - 1)
            elif action == 2:
                paddle_x = min(FRAME_SIZE - 3, paddle_x + 1)
            frames += 1
            if ball_y >= FRAME_SIZE - 2:
                if abs(ball_x - paddle_x) < 3:
                    ball_dy = -abs(ball_dy)
                    ball_y = FRAME_SIZE - 3
                else:
                    break
        guided_lengths.append(frames)

    print(f"\nRandom play: avg {np.mean(random_lengths):.1f} frames (std {np.std(random_lengths):.1f})")
    print(f"Amplitude-guided: avg {np.mean(guided_lengths):.1f} frames (std {np.std(guided_lengths):.1f})")

    if np.mean(guided_lengths) > np.mean(random_lengths) * 1.1:
        print("\n*** SURVIVAL EMERGENCE CONFIRMED ***")
        print("Amplitude-guided play survives longer WITHOUT explicit reward!")
    else:
        print("\nNo significant improvement detected.")
        print("May need more training or different propagation mechanism.")


if __name__ == "__main__":
    test_pattern_persistence()
    print("\n\n")
    # test_survival_emergence()
    # print("\n\n")
    # test_controlled_comparison()
