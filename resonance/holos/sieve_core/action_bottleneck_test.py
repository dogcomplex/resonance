"""
ACTION AS TEMPORAL BOTTLENECK
=============================

Key insight: Actions are not "co-occurring tokens" - they are GATES.

The structure of causality is:
    state_t → action → state_t+1

NOT:
    state_t + action → state_t+1

Every pixel-to-pixel temporal transition flows THROUGH the action:
    pixel_0_t → action_X → pixel_0_t+1
    pixel_1_t → action_X → pixel_1_t+1
    ...

And ALL pixels in the same frame share the SAME action token.
So action becomes a MASSIVE BOTTLENECK where ALL temporal flow concentrates.

This means:
1. Actions accumulate amplitude from EVERY pixel transition
2. Different actions lead to different future pixel distributions
3. The action that leads to longer-lasting futures gets more accumulated
   amplitude flowing back through it

The laws of physics (action tokens) become the dominant features of reality
simply by being the bottleneck to long futures.
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
import hashlib


class BottleneckToken:
    """Token with temporal connections that flow through action bottlenecks."""
    __slots__ = ['id', 'amplitude', 'spatial_neighbors', 'flows_to', 'flows_from']

    def __init__(self, token_id: str):
        self.id = token_id
        self.amplitude: float = 0.0
        self.spatial_neighbors: Set[str] = set()
        # Temporal flow: these track directed connections
        self.flows_to: Dict[str, float] = {}    # This token -> other (forward in time)
        self.flows_from: Dict[str, float] = {}  # Other -> this token (backward in time)


class ActionBottleneckSieve:
    """
    Sieve where ALL temporal transitions flow through action tokens.

    Structure:
        pixels_t → action → pixels_t+1

    The action is the GATE between temporal slices.
    All amplitude flowing from future to past MUST pass through action tokens.
    """

    def __init__(self):
        self.tokens: Dict[str, BottleneckToken] = {}
        self.frame_num: int = 0
        self.prev_active_pixels: Set[str] = set()
        self.current_active_pixels: Set[str] = set()
        self.current_action_id: Optional[str] = None
        self.prev_action_id: Optional[str] = None

        # Parameters
        self._coupling_strength: float = 0.1
        self._decay_rate: float = 0.001
        self._propagation_strength: float = 0.01

        self._manifold_built = False

    def _hash(self, data: str) -> str:
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _get_token(self, token_id: str) -> BottleneckToken:
        if token_id not in self.tokens:
            self.tokens[token_id] = BottleneckToken(token_id)
        return self.tokens[token_id]

    def _build_manifold(self, h: int, w: int):
        """Build spatial adjacency for pixels."""
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
        """
        Observe frame and action.

        KEY: Temporal connections flow THROUGH the action, not around it.
        """
        self.frame_num = frame_num

        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float)
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val

        h, w = frame.shape
        self._build_manifold(h, w)

        # Store previous state
        self.prev_active_pixels = self.current_active_pixels.copy()
        self.prev_action_id = self.current_action_id
        self.current_active_pixels = set()

        # Observe all pixels
        for y in range(h):
            for x in range(w):
                intensity = frame[y, x]
                pixel_id = self._hash(f"p_{y}_{x}")
                token = self._get_token(pixel_id)
                token.amplitude = intensity
                if intensity > 0:
                    self.current_active_pixels.add(pixel_id)

        # Action token
        action_id = self._hash(f"action_{action}")
        action_token = self._get_token(action_id)
        action_token.amplitude = 1.0
        self.current_action_id = action_id

        # Learn temporal couplings THROUGH the action bottleneck
        self._learn_bottleneck_couplings()

        # Propagate amplitudes
        self._propagate()

    def _learn_bottleneck_couplings(self):
        """
        Build temporal couplings that flow THROUGH the action.

        Structure:
            prev_pixel → prev_action → current_pixel

        NOT:
            prev_pixel → current_pixel (direct)

        The action MEDIATES all temporal transitions.
        """
        cs = self._coupling_strength
        dr = self._decay_rate

        if not self.prev_action_id:
            return

        prev_action = self.tokens.get(self.prev_action_id)
        if not prev_action:
            return

        # FIRST HALF: prev_pixels → prev_action
        # Every active pixel in the previous frame connects TO the action that was taken
        for prev_pixel_id in self.prev_active_pixels:
            prev_pixel = self.tokens.get(prev_pixel_id)
            if prev_pixel:
                # prev_pixel flows TO prev_action
                prev_pixel.flows_to[self.prev_action_id] = (
                    prev_pixel.flows_to.get(self.prev_action_id, 0) + cs
                )
                # prev_action receives flow FROM prev_pixel
                prev_action.flows_from[prev_pixel_id] = (
                    prev_action.flows_from.get(prev_pixel_id, 0) + cs
                )

        # SECOND HALF: prev_action → current_pixels
        # The action that was taken connects TO every active pixel in the current frame
        for curr_pixel_id in self.current_active_pixels:
            curr_pixel = self.tokens.get(curr_pixel_id)
            if curr_pixel:
                # prev_action flows TO curr_pixel
                prev_action.flows_to[curr_pixel_id] = (
                    prev_action.flows_to.get(curr_pixel_id, 0) + cs
                )
                # curr_pixel receives flow FROM prev_action
                curr_pixel.flows_from[self.prev_action_id] = (
                    curr_pixel.flows_from.get(self.prev_action_id, 0) + cs
                )

        # Decay all couplings
        for token in self.tokens.values():
            for key in list(token.flows_to.keys()):
                token.flows_to[key] *= (1 - dr)
                if token.flows_to[key] < 1e-12:
                    del token.flows_to[key]
            for key in list(token.flows_from.keys()):
                token.flows_from[key] *= (1 - dr)
                if token.flows_from[key] < 1e-12:
                    del token.flows_from[key]

    def _propagate(self):
        """
        Propagate amplitudes through the temporal graph.

        Forward: amplitude flows from past to future (flows_to)
        Backward: amplitude flows from future to past (flows_from)

        The BACKWARD flow is what creates survival emergence:
        - Future states send amplitude back through their flows_from
        - That amplitude concentrates at the action bottlenecks
        - Actions that led to longer futures accumulate more amplitude
        """
        contributions = {tid: 0.0 for tid in self.tokens}

        # Active tokens get amplitude boost
        for pixel_id in self.current_active_pixels:
            contributions[pixel_id] += 0.1
        if self.current_action_id:
            contributions[self.current_action_id] += 0.1

        # BACKWARD propagation: amplitude flows from current to predecessors
        # This is the key mechanism for survival emergence
        for token in self.tokens.values():
            if token.amplitude > 0:
                # Send amplitude backward through flows_from connections
                for source_id, coupling in token.flows_from.items():
                    if source_id in self.tokens:
                        # Amplitude flows backward proportional to coupling
                        contributions[source_id] += coupling * token.amplitude * self._propagation_strength

        # Apply contributions
        for tid, contrib in contributions.items():
            self.tokens[tid].amplitude += contrib

        # Decay all tokens
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

    def get_action_flow_stats(self) -> Dict[int, Dict]:
        """Get detailed flow statistics for each action."""
        result = {}
        for a in range(3):
            action_id = self._hash(f"action_{a}")
            action_token = self.tokens.get(action_id)
            if action_token:
                result[a] = {
                    'amplitude': action_token.amplitude,
                    'flows_from_count': len(action_token.flows_from),
                    'flows_to_count': len(action_token.flows_to),
                    'total_inflow': sum(action_token.flows_from.values()),
                    'total_outflow': sum(action_token.flows_to.values()),
                }
            else:
                result[a] = {'amplitude': 0.0}
        return result

    def choose_action(self) -> int:
        """
        Choose action based on amplitude.

        With the bottleneck structure, action amplitude directly reflects
        how much "future" flows through that action.
        """
        amps = self.get_action_amplitudes()

        total = sum(amps.values())
        if total < 0.001:
            return np.random.randint(0, 3)

        # Softmax selection
        probs = {a: amp / total for a, amp in amps.items()}
        r = np.random.random()
        cumulative = 0.0
        for a, p in probs.items():
            cumulative += p
            if r < cumulative:
                return a
        return 2


def test_bottleneck_amplification():
    """
    Test whether actions accumulate more amplitude as bottlenecks.

    Compare:
    1. Total pixel amplitude
    2. Total action amplitude

    If actions are true bottlenecks, they should accumulate MORE amplitude
    per token than pixels, because ALL flow passes through them.
    """
    print("=" * 70)
    print("BOTTLENECK AMPLIFICATION TEST")
    print("Do actions accumulate more amplitude per token than pixels?")
    print("=" * 70)

    FRAME_SIZE = 21
    sieve = ActionBottleneckSieve()

    # Run some games
    for game in range(20):
        ball_x, ball_y = 10.5, 5.0
        ball_dx = np.random.choice([-0.5, 0.5])
        ball_dy = 0.5
        paddle_x = 10.5

        for frame in range(100):
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

    # Analyze amplitude distribution
    pixel_amps = []
    action_amps = []

    for tid, token in sieve.tokens.items():
        if tid.startswith(sieve._hash("action")[:4]) or any(
            tid == sieve._hash(f"action_{a}") for a in range(3)
        ):
            action_amps.append(token.amplitude)
        else:
            pixel_amps.append(token.amplitude)

    # Get action tokens specifically
    action_stats = sieve.get_action_flow_stats()

    print(f"\nPixel tokens: {len(pixel_amps)}")
    print(f"  Total amplitude: {sum(pixel_amps):.2f}")
    print(f"  Avg amplitude: {np.mean(pixel_amps):.4f}")
    print(f"  Max amplitude: {max(pixel_amps):.4f}")

    print(f"\nAction tokens: {len(action_amps)}")
    print(f"  Total amplitude: {sum(action_amps):.2f}")
    print(f"  Avg amplitude: {np.mean(action_amps):.4f}")

    print(f"\nAction flow statistics:")
    for a, stats in action_stats.items():
        print(f"  action_{a}:")
        print(f"    amplitude: {stats['amplitude']:.4f}")
        print(f"    flows_from (inputs): {stats.get('flows_from_count', 0)}")
        print(f"    flows_to (outputs): {stats.get('flows_to_count', 0)}")
        print(f"    total_inflow: {stats.get('total_inflow', 0):.2f}")
        print(f"    total_outflow: {stats.get('total_outflow', 0):.2f}")

    # The key metric: amplitude per token
    pixel_amp_per_token = np.mean(pixel_amps) if pixel_amps else 0
    action_amp_per_token = np.mean(action_amps) if action_amps else 0

    print(f"\nAmplitude concentration ratio:")
    if pixel_amp_per_token > 0:
        ratio = action_amp_per_token / pixel_amp_per_token
        print(f"  Action/Pixel amplitude ratio: {ratio:.2f}x")
        if ratio > 1:
            print("  *** BOTTLENECK EFFECT CONFIRMED ***")
            print("  Actions concentrate more amplitude per token than pixels!")

    return sieve


def test_survival_emergence_with_bottleneck():
    """
    Test whether the bottleneck structure leads to survival emergence.

    With actions as true bottlenecks:
    - Actions that lead to longer games have more future flowing through them
    - That future amplitude flows backward through the bottleneck
    - So actions that lead to survival accumulate more amplitude
    """
    print("\n" + "=" * 70)
    print("SURVIVAL EMERGENCE WITH BOTTLENECK")
    print("Do actions that lead to longer games accumulate more amplitude?")
    print("=" * 70)

    FRAME_SIZE = 21
    sieve = ActionBottleneckSieve()

    # Track which actions led to longer games
    action_game_lengths = {0: [], 1: [], 2: []}

    for game in range(50):
        ball_x, ball_y = 10.5, 5.0
        ball_dx = np.random.choice([-0.5, 0.5])
        ball_dy = 0.5
        paddle_x = 10.5

        frames = 0
        actions_taken = []

        for _ in range(300):
            pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
            bx, by = int(ball_x), int(ball_y)
            pixels[max(0,by-1):min(FRAME_SIZE,by+1), max(0,bx-1):min(FRAME_SIZE,bx+1)] = 255
            px = int(paddle_x)
            pixels[FRAME_SIZE-2:FRAME_SIZE, max(0,px-2):min(FRAME_SIZE,px+2)] = 200

            # Use sieve to choose action after warmup
            if game < 10:
                action = np.random.randint(0, 3)
            else:
                action = sieve.choose_action()

            actions_taken.append(action)
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

        # Record game length for each action taken
        for a in actions_taken:
            action_game_lengths[a].append(frames)

        if (game + 1) % 10 == 0:
            amps = sieve.get_action_amplitudes()
            print(f"\nAfter game {game + 1}:")
            print(f"  Game lasted: {frames} frames")
            print(f"  Action amplitudes: {amps}")

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    print("\nAction amplitudes vs avg game length when used:")
    amps = sieve.get_action_amplitudes()
    for a in range(3):
        avg_length = np.mean(action_game_lengths[a]) if action_game_lengths[a] else 0
        print(f"  action_{a}: amplitude={amps[a]:.4f}, avg_game_length={avg_length:.1f}")

    # Check correlation between amplitude and game length
    print("\nCorrelation check:")
    amp_list = [amps[a] for a in range(3)]
    length_list = [np.mean(action_game_lengths[a]) for a in range(3)]

    # Simple correlation
    amp_rank = np.argsort(amp_list)
    length_rank = np.argsort(length_list)

    print(f"  Amplitude ranking: {list(amp_rank)}")
    print(f"  Game length ranking: {list(length_rank)}")

    if list(amp_rank) == list(length_rank):
        print("  *** PERFECT CORRELATION ***")
        print("  Actions that lead to longer games have higher amplitude!")
    elif amp_rank[-1] == length_rank[-1]:
        print("  *** PARTIAL CORRELATION ***")
        print("  The highest-amplitude action also has the longest games!")

    return sieve


def test_long_vs_short_games():
    """
    Controlled test: Force long vs short games and see if actions differentiate.
    """
    print("\n" + "=" * 70)
    print("CONTROLLED LONG VS SHORT GAMES")
    print("Force long/short games to see action amplitude differentiation")
    print("=" * 70)

    FRAME_SIZE = 21
    sieve = ActionBottleneckSieve()

    # Phase 1: Games where action_0 leads to long games
    print("\nPhase 1: action_0 = long games, action_2 = short games")
    for game in range(20):
        ball_x, ball_y = 10.5, 5.0
        ball_dx = np.random.choice([-0.5, 0.5])
        ball_dy = 0.5
        paddle_x = 10.5

        # Alternate: action_0 = long (200 frames), action_2 = short (30 frames)
        forced_action = game % 2  # 0 or 1
        max_frames = 200 if forced_action == 0 else 30

        for frame in range(max_frames):
            pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
            bx, by = int(ball_x), int(ball_y)
            pixels[max(0,by-1):min(FRAME_SIZE,by+1), max(0,bx-1):min(FRAME_SIZE,bx+1)] = 255
            px = int(paddle_x)
            pixels[FRAME_SIZE-2:FRAME_SIZE, max(0,px-2):min(FRAME_SIZE,px+2)] = 200

            # Use the forced action
            action = forced_action
            sieve.observe(pixels, action, sieve.frame_num + 1)

            ball_x += ball_dx
            ball_y += ball_dy
            if ball_x <= 1 or ball_x >= FRAME_SIZE - 1:
                ball_dx *= -1
            if ball_y <= 1:
                ball_dy = abs(ball_dy)

    print(f"Action amplitudes after controlled games:")
    amps = sieve.get_action_amplitudes()
    for a, amp in sorted(amps.items()):
        print(f"  action_{a}: {amp:.4f}")

    if amps[0] > amps[1] * 1.5:
        print("\n*** SUCCESS ***")
        print("action_0 (long games) has significantly higher amplitude than action_1 (short games)!")
        print(f"Ratio: {amps[0]/amps[1]:.2f}x")
    else:
        print("\nAction amplitudes did not differentiate significantly.")

    return sieve


if __name__ == "__main__":
    test_bottleneck_amplification()
    print("\n\n")
    test_long_vs_short_games()
    print("\n\n")
    test_survival_emergence_with_bottleneck()
