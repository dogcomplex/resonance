"""
SURVIVAL TASK TEST
==================

Test the sieve on tasks where survival IS the objective.
The anthropic principle should shine here without artificial emphasis.

Task: GridWorld Survival
- Agent starts in center of grid
- Hazards appear randomly
- Agent must avoid hazards to survive
- Longer survival = success

This is pure anthropic selection: patterns that lead to survival persist.
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
import hashlib
import cmath
from collections import defaultdict


class MultiScaleSieve:
    """
    Sieve with multiple scales of pattern identity:
    1. Pixel-level: individual positions
    2. State-level: entire frame hash
    3. Gradient-level: frame-to-frame difference

    Also includes:
    - Bootstrap emphasis (seed energy in promising directions)
    - Death signal (negative interference on termination)
    """

    def __init__(self):
        # Three levels of pattern identity
        self.pixels: Dict[str, complex] = {}      # Individual pixels
        self.states: Dict[str, complex] = {}      # Whole-state hashes
        self.gradients: Dict[str, complex] = {}   # Frame differences

        # Couplings between all levels
        self.couplings: Dict[Tuple[str, str], complex] = defaultdict(complex)

        # Action states
        self.actions: Dict[str, complex] = {}
        self.discovered_actions: Set[str] = set()

        # Frame tracking
        self.frame_num = 0
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_state_id: Optional[str] = None
        self.current_action_id: Optional[str] = None
        self.prev_action_id: Optional[str] = None

        # Active sets
        self.curr_active_pixels: Set[str] = set()
        self.curr_active_states: Set[str] = set()
        self.curr_active_gradients: Set[str] = set()

        # Temporal trace for credit assignment
        self._action_trace: List[str] = []
        self._state_trace: List[str] = []

        # Bootstrap state
        self._bootstrap_active = True
        self._bootstrap_energy = 1.0
        self._bootstrap_targets: Set[str] = set()  # States to emphasize

        # Statistics
        self._game_lengths: List[int] = []
        self._current_game_length = 0

    def _hash(self, data) -> str:
        """Deterministic hash."""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, bytes):
            data = str(data).encode()
        return hashlib.md5(data).hexdigest()[:16]

    def _get_pixel_id(self, y: int, x: int) -> str:
        return self._hash(f"px_{y}_{x}")

    def _get_state_id(self, frame: np.ndarray) -> str:
        return self._hash(frame)

    def _get_gradient_id(self, diff: np.ndarray) -> str:
        # Quantize gradient to reduce noise
        quantized = (diff > 0.1).astype(np.int8) - (diff < -0.1).astype(np.int8)
        return self._hash(quantized)

    def _get_action_id(self, action: int) -> str:
        return self._hash(f"action_{action}")

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """
        Observe frame and action at all three scales.
        """
        self.frame_num = frame_num
        self._current_game_length += 1

        # Normalize frame
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float)
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val

        h, w = frame.shape

        # Store previous state
        prev_active_pixels = self.curr_active_pixels.copy()
        prev_active_states = self.curr_active_states.copy()
        self.prev_action_id = self.current_action_id

        # Reset current actives
        self.curr_active_pixels = set()
        self.curr_active_states = set()
        self.curr_active_gradients = set()

        # === PIXEL LEVEL ===
        for y in range(h):
            for x in range(w):
                if frame[y, x] > 0.01:
                    pid = self._get_pixel_id(y, x)
                    self.curr_active_pixels.add(pid)
                    # Inject energy (normalized by frame size)
                    energy = frame[y, x] / (h * w)
                    self.pixels[pid] = self.pixels.get(pid, 0) + energy

        # === STATE LEVEL ===
        state_id = self._get_state_id(frame)
        self.curr_active_states.add(state_id)
        self.states[state_id] = self.states.get(state_id, 0) + 1.0
        self._state_trace.append(state_id)

        # === GRADIENT LEVEL ===
        if self.prev_frame is not None:
            diff = frame - self.prev_frame
            grad_id = self._get_gradient_id(diff)
            self.curr_active_gradients.add(grad_id)
            self.gradients[grad_id] = self.gradients.get(grad_id, 0) + 1.0

        # === ACTION ===
        action_id = self._get_action_id(action)
        self.discovered_actions.add(action_id)
        self.actions[action_id] = self.actions.get(action_id, 0) + 1.0
        self.current_action_id = action_id
        self._action_trace.append(action_id)

        # === BUILD COUPLINGS ===
        self._build_couplings(state_id, action_id)

        # === BACKWARD PROPAGATION (anthropic credit) ===
        self._propagate_backward()

        # === BOOTSTRAP EMPHASIS ===
        if self._bootstrap_active and self._bootstrap_energy > 0:
            self._apply_bootstrap(state_id, action_id)

        # === DAMPING ===
        self._damp_all()

        # Store for next frame
        self.prev_frame = frame.copy()
        self.prev_state_id = state_id

    def _build_couplings(self, state_id: str, action_id: str):
        """Build couplings between state and action."""
        # State -> Action coupling
        key = (state_id, action_id)
        self.couplings[key] += 0.1

        # Previous action -> Current state coupling (temporal)
        if self.prev_action_id:
            key = (self.prev_action_id, state_id)
            self.couplings[key] += 0.1

        # Gradient -> Action coupling (motion predicts action)
        for grad_id in self.curr_active_gradients:
            key = (grad_id, action_id)
            self.couplings[key] += 0.05

    def _propagate_backward(self):
        """
        Backward amplitude flow - the anthropic principle.
        Each frame of survival sends energy back to actions that led here.
        """
        if not self._action_trace:
            return

        trace_len = len(self._action_trace)
        tau = max(1, trace_len / 3)

        # Normalize to total energy = 1.0
        total_discount = sum(np.exp(-(trace_len - 1 - i) / tau) for i in range(trace_len))

        for i, action_id in enumerate(self._action_trace):
            age = trace_len - 1 - i
            discount = np.exp(-age / tau)
            normalized_energy = discount / (total_discount + 1e-10)
            self.actions[action_id] = self.actions.get(action_id, 0) + normalized_energy

        # Also propagate to states
        for i, state_id in enumerate(self._state_trace[-100:]):
            age = len(self._state_trace) - 1 - i
            discount = np.exp(-age / tau)
            self.states[state_id] = self.states.get(state_id, 0) + discount * 0.1

    def _apply_bootstrap(self, state_id: str, action_id: str):
        """
        Bootstrap emphasis - seed energy in promising directions.
        This helps overcome the cold-start problem.
        """
        # If this state/action pair hasn't been seen much, give it a boost
        coupling_key = (state_id, action_id)
        current_coupling = abs(self.couplings.get(coupling_key, 0))

        if current_coupling < 1.0:  # Not well-established yet
            # Inject bootstrap energy
            boost = self._bootstrap_energy * 0.1
            self.couplings[coupling_key] += boost
            self.actions[action_id] = self.actions.get(action_id, 0) + boost

        # Decay bootstrap over time as real learning takes over
        self._bootstrap_energy *= 0.999

    def _damp_all(self):
        """Apply damping to all amplitudes."""
        damping = 0.99

        for k in self.pixels:
            self.pixels[k] *= damping
        for k in self.states:
            self.states[k] *= damping
        for k in self.gradients:
            self.gradients[k] *= damping
        for k in self.actions:
            self.actions[k] *= damping
        for k in self.couplings:
            self.couplings[k] *= damping

    def signal_game_end(self, game_length: int = 0, death: bool = True):
        """
        Game boundary signal.

        If death=True, apply negative interference to recent actions/states.
        This is the contrastive signal.
        """
        if game_length == 0:
            game_length = self._current_game_length

        self._game_lengths.append(game_length)

        if death:
            # STRONG NEGATIVE SIGNAL on death
            # Key insight: penalize the LAST state-action that led to death
            # This is the most direct cause

            if self._state_trace and self._action_trace:
                # Last state-action pair is most responsible
                last_state = self._state_trace[-1]
                last_action = self._action_trace[-1]

                # Strong penalty on immediate cause
                key = (last_state, last_action)
                self.couplings[key] = self.couplings.get(key, 0) - 1.0

                # Weaker penalty on second-to-last (contributing cause)
                if len(self._state_trace) >= 2 and len(self._action_trace) >= 2:
                    prev_state = self._state_trace[-2]
                    prev_action = self._action_trace[-2]
                    key2 = (prev_state, prev_action)
                    self.couplings[key2] = self.couplings.get(key2, 0) - 0.3

            # Also penalize the action itself
            if self._action_trace:
                last_action = self._action_trace[-1]
                self.actions[last_action] = self.actions.get(last_action, 0) - 0.5

        # Clear traces
        self._action_trace = []
        self._state_trace = []
        self._current_game_length = 0

    def choose_action(self, num_actions: int = 4) -> int:
        """
        Choose action based on resonance with current state.
        """
        if not self.curr_active_states:
            return np.random.randint(0, num_actions)

        state_id = list(self.curr_active_states)[0]
        action_scores = {}

        for a in range(num_actions):
            action_id = self._get_action_id(a)

            # State -> Action coupling
            coupling_key = (state_id, action_id)
            coupling_score = abs(self.couplings.get(coupling_key, 0))

            # Action amplitude (survival history)
            amplitude_score = abs(self.actions.get(action_id, 0))

            # Gradient -> Action coupling (if available)
            gradient_score = 0.0
            for grad_id in self.curr_active_gradients:
                grad_key = (grad_id, action_id)
                gradient_score += abs(self.couplings.get(grad_key, 0))

            # Combine scores
            total = coupling_score + amplitude_score + gradient_score + 0.01
            action_scores[a] = total

        # Softmax selection
        temperature = 1.0
        scores = np.array(list(action_scores.values()))
        scores = scores - scores.max()
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / exp_scores.sum()

        return np.random.choice(list(action_scores.keys()), p=probs)

    def get_stats(self) -> Dict:
        return {
            'num_pixels': len(self.pixels),
            'num_states': len(self.states),
            'num_gradients': len(self.gradients),
            'num_couplings': len(self.couplings),
            'bootstrap_energy': self._bootstrap_energy,
            'avg_game_length': np.mean(self._game_lengths[-20:]) if self._game_lengths else 0,
        }


# =============================================================================
# GRIDWORLD SURVIVAL ENVIRONMENT
# =============================================================================

class GridWorldSurvival:
    """
    Simple survival task:
    - Agent (white) must avoid hazards (gray)
    - Hazards spawn randomly and move
    - Survival time = success
    """

    def __init__(self, size: int = 7):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [self.size // 2, self.size // 2]
        self.hazards = []
        self.step_count = 0
        self.spawn_hazard()
        return self.get_state()

    def spawn_hazard(self):
        """Spawn a hazard at random edge position."""
        edge = np.random.randint(4)
        if edge == 0:  # Top
            pos = [0, np.random.randint(self.size)]
        elif edge == 1:  # Bottom
            pos = [self.size - 1, np.random.randint(self.size)]
        elif edge == 2:  # Left
            pos = [np.random.randint(self.size), 0]
        else:  # Right
            pos = [np.random.randint(self.size), self.size - 1]
        self.hazards.append(pos)

    def get_state(self) -> np.ndarray:
        """Return grid as image."""
        frame = np.zeros((self.size, self.size), dtype=np.uint8)

        # Agent (bright)
        frame[self.agent_pos[0], self.agent_pos[1]] = 255

        # Hazards (dim)
        for h in self.hazards:
            if 0 <= h[0] < self.size and 0 <= h[1] < self.size:
                frame[h[0], h[1]] = 128

        return frame

    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        """
        Take action (0=up, 1=down, 2=left, 3=right).
        Returns (state, done).
        """
        self.step_count += 1

        # Move agent
        dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_y = max(0, min(self.size - 1, self.agent_pos[0] + dy))
        new_x = max(0, min(self.size - 1, self.agent_pos[1] + dx))
        self.agent_pos = [new_y, new_x]

        # Move hazards toward agent
        for h in self.hazards:
            if np.random.random() < 0.5:  # 50% chance to move
                dy = np.sign(self.agent_pos[0] - h[0])
                dx = np.sign(self.agent_pos[1] - h[1])
                h[0] += dy
                h[1] += dx

        # Spawn new hazard occasionally
        if self.step_count % 20 == 0:
            self.spawn_hazard()

        # Check collision
        for h in self.hazards:
            if h[0] == self.agent_pos[0] and h[1] == self.agent_pos[1]:
                return self.get_state(), True  # Death

        # Remove hazards that went off-grid
        self.hazards = [h for h in self.hazards
                       if 0 <= h[0] < self.size and 0 <= h[1] < self.size]

        return self.get_state(), False


class DodgeBall:
    """
    Simpler survival: dodge balls coming from one direction.

    - Balls fall from top
    - Agent at bottom row, can move left/right/stay
    - Clear spatial-action relationship: ball above = move away
    """

    def __init__(self, width: int = 5):
        self.width = width
        self.height = 6
        self.reset()

    def reset(self):
        self.agent_x = self.width // 2
        self.balls = []  # List of (x, y) positions
        self.step_count = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return grid as image."""
        frame = np.zeros((self.height, self.width), dtype=np.uint8)

        # Agent at bottom (bright)
        frame[self.height - 1, self.agent_x] = 255

        # Balls (dim)
        for bx, by in self.balls:
            if 0 <= by < self.height:
                frame[by, bx] = 128

        return frame

    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        """
        Actions: 0=left, 1=stay, 2=right
        """
        self.step_count += 1

        # Move agent
        if action == 0:
            self.agent_x = max(0, self.agent_x - 1)
        elif action == 2:
            self.agent_x = min(self.width - 1, self.agent_x + 1)
        # action == 1: stay

        # Move balls down
        self.balls = [(bx, by + 1) for bx, by in self.balls]

        # Check collision (ball at agent position)
        for bx, by in self.balls:
            if by == self.height - 1 and bx == self.agent_x:
                return self.get_state(), True  # Hit!

        # Remove balls that fell off
        self.balls = [(bx, by) for bx, by in self.balls if by < self.height]

        # Spawn new ball at top (random x)
        if np.random.random() < 0.3:  # 30% chance each step
            self.balls.append((np.random.randint(0, self.width), 0))

        return self.get_state(), False


# =============================================================================
# TEST
# =============================================================================

def test_survival():
    """Test multi-scale sieve on survival task."""
    print("=" * 70)
    print("SURVIVAL TASK TEST")
    print("Multi-scale sieve with bootstrap and death signal")
    print("=" * 70)
    print()

    env = GridWorldSurvival(size=7)
    sieve = MultiScaleSieve()

    NUM_GAMES = 500
    game_lengths = []

    for game in range(NUM_GAMES):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 500:
            # Choose action
            if game < 20:
                action = np.random.randint(0, 4)
            else:
                action = sieve.choose_action(num_actions=4)

            # Observe
            sieve.observe(state, action, game * 500 + steps)

            # Step
            state, done = env.step(action)
            steps += 1

        # Signal game end (with death signal if died)
        sieve.signal_game_end(steps, death=done)
        game_lengths.append(steps)

        # Progress
        if (game + 1) % 50 == 0:
            recent = np.mean(game_lengths[-20:]) if len(game_lengths) >= 20 else np.mean(game_lengths)
            stats = sieve.get_stats()
            print(f"Game {game+1}: Recent avg length = {recent:.1f}, "
                  f"States={stats['num_states']}, Bootstrap={stats['bootstrap_energy']:.3f}")

    # Final analysis
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    first_50 = np.mean(game_lengths[:50])
    last_50 = np.mean(game_lengths[-50:])

    print(f"First 50 games avg: {first_50:.1f} steps")
    print(f"Last 50 games avg: {last_50:.1f} steps")
    print(f"Improvement: {(last_50/first_50 - 1)*100:+.1f}%")

    # Random baseline
    print()
    print("Running random baseline...")
    random_lengths = []
    for _ in range(100):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            action = np.random.randint(0, 4)
            state, done = env.step(action)
            steps += 1
        random_lengths.append(steps)

    print(f"Random baseline: {np.mean(random_lengths):.1f} steps")
    print()

    if last_50 > np.mean(random_lengths) * 1.2:
        print("STATUS: SIEVE SHOWS LEARNING (>20% above random)")
    elif last_50 > np.mean(random_lengths):
        print("STATUS: SIEVE SHOWS WEAK LEARNING")
    else:
        print("STATUS: SIEVE NOT LEARNING")

    return game_lengths


def test_dodgeball():
    """
    Test on DodgeBall - clearer spatial relationship.

    Ball falling toward agent = should move away.
    This is a cleaner test of whether the sieve can learn
    spatial-action associations.
    """
    print()
    print("=" * 70)
    print("DODGEBALL SURVIVAL TEST")
    print("Clearer spatial-action relationship")
    print("=" * 70)
    print()

    env = DodgeBall(width=5)
    sieve = MultiScaleSieve()

    NUM_GAMES = 1000
    game_lengths = []

    for game in range(NUM_GAMES):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            # Choose action
            if game < 50:
                action = np.random.randint(0, 3)
            else:
                action = sieve.choose_action(num_actions=3)

            # Observe BEFORE action (state that led to decision)
            sieve.observe(state, action, game * 200 + steps)

            # Step
            state, done = env.step(action)
            steps += 1

        # Signal game end
        sieve.signal_game_end(steps, death=done)
        game_lengths.append(steps)

        # Progress
        if (game + 1) % 100 == 0:
            recent = np.mean(game_lengths[-50:]) if len(game_lengths) >= 50 else np.mean(game_lengths)
            stats = sieve.get_stats()
            print(f"Game {game+1}: Recent avg = {recent:.1f} steps, "
                  f"States={stats['num_states']}, Couplings={stats['num_couplings']}")

    # Analysis
    print()
    print("=" * 70)
    print("DODGEBALL RESULTS")
    print("=" * 70)

    first_100 = np.mean(game_lengths[:100])
    last_100 = np.mean(game_lengths[-100:])

    print(f"First 100 games: {first_100:.1f} steps")
    print(f"Last 100 games: {last_100:.1f} steps")
    print(f"Improvement: {(last_100/first_100 - 1)*100:+.1f}%")

    # Random baseline
    print()
    print("Random baseline...")
    random_lengths = []
    for _ in range(200):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = np.random.randint(0, 3)
            state, done = env.step(action)
            steps += 1
        random_lengths.append(steps)

    print(f"Random: {np.mean(random_lengths):.1f} steps")

    # Oracle baseline (perfect avoidance)
    print()
    print("Oracle baseline (perfect play)...")
    oracle_lengths = []
    for _ in range(200):
        env2 = DodgeBall(width=5)
        state = env2.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            # Oracle: look at incoming balls, move away
            frame = state
            agent_x = env2.agent_x

            # Find balls in row above agent (about to hit)
            danger_row = env2.height - 2
            danger_cols = [bx for bx, by in env2.balls if by == danger_row]

            if agent_x in danger_cols:
                # Ball coming! Move away
                if agent_x > 0 and (agent_x - 1) not in danger_cols:
                    action = 0  # Left
                elif agent_x < env2.width - 1 and (agent_x + 1) not in danger_cols:
                    action = 2  # Right
                else:
                    action = 1  # Can't escape, stay
            else:
                action = 1  # Safe, stay

            state, done = env2.step(action)
            steps += 1
        oracle_lengths.append(steps)

    print(f"Oracle: {np.mean(oracle_lengths):.1f} steps")
    print()

    print(f"Learning gap: Random={np.mean(random_lengths):.1f} -> "
          f"Sieve={last_100:.1f} -> Oracle={np.mean(oracle_lengths):.1f}")

    if last_100 > np.mean(random_lengths) * 1.2:
        print("STATUS: SIEVE SHOWS LEARNING (>20% above random)")
    elif last_100 > np.mean(random_lengths) * 1.05:
        print("STATUS: SIEVE SHOWS WEAK LEARNING")
    else:
        print("STATUS: SIEVE NOT LEARNING")

    return game_lengths


def test_minimal_state():
    """
    Test with MINIMAL state encoding.

    Instead of full pixel grid, encode just:
    - Relative position of nearest threat (-2, -1, 0, +1, +2)

    This reduces state space dramatically and tests if the
    sieve mechanism itself works when given good representations.
    """
    print()
    print("=" * 70)
    print("MINIMAL STATE ENCODING TEST")
    print("State = relative position of nearest threat")
    print("=" * 70)
    print()

    class MinimalDodgeSieve:
        """Sieve with minimal state encoding."""

        def __init__(self):
            self.states: Dict[str, complex] = {}
            self.actions: Dict[str, complex] = {}
            self.couplings: Dict[Tuple[str, str], complex] = defaultdict(complex)

            self.frame_num = 0
            self.current_action_id: Optional[str] = None
            self.current_state_id: Optional[str] = None
            self.prev_state_id: Optional[str] = None
            self.prev_action_id: Optional[str] = None

            self._action_trace: List[str] = []
            self._state_trace: List[str] = []

        def _get_state_id(self, relative_pos: int, agent_x: int = None, width: int = 5) -> str:
            # State includes relative threat AND edge awareness
            # This is still minimal but captures the critical edge case
            if agent_x is not None:
                at_left = agent_x == 0
                at_right = agent_x == width - 1
                edge = "L" if at_left else ("R" if at_right else "M")  # Left/Right/Middle
                return f"rel_{relative_pos}_{edge}"
            return f"rel_{relative_pos}"

        def _get_action_id(self, action: int) -> str:
            return f"act_{action}"

        def observe(self, relative_threat: int, action: int, agent_x: int = None, width: int = 5):
            """
            Observe minimal state (relative threat position + edge).
            relative_threat: -2, -1, 0, +1, +2, or 99 (no threat)
            """
            self.frame_num += 1

            self.prev_state_id = self.current_state_id
            self.prev_action_id = self.current_action_id

            state_id = self._get_state_id(relative_threat, agent_x, width)
            action_id = self._get_action_id(action)

            self.current_state_id = state_id
            self.current_action_id = action_id

            # Add to traces
            self._state_trace.append(state_id)
            self._action_trace.append(action_id)
            if len(self._state_trace) > 50:
                self._state_trace = self._state_trace[-50:]
                self._action_trace = self._action_trace[-50:]

            # Inject energy
            self.states[state_id] = self.states.get(state_id, 0) + 1.0
            self.actions[action_id] = self.actions.get(action_id, 0) + 1.0

            # Build coupling: state -> action
            key = (state_id, action_id)
            self.couplings[key] += 0.1

            # Backward propagation
            trace_len = len(self._action_trace)
            if trace_len > 0:
                tau = max(1, trace_len / 3)
                for i, aid in enumerate(self._action_trace):
                    age = trace_len - 1 - i
                    discount = np.exp(-age / tau)
                    self.actions[aid] = self.actions.get(aid, 0) + discount * 0.1

            # Damping
            for k in self.states:
                self.states[k] *= 0.99
            for k in self.actions:
                self.actions[k] *= 0.99
            for k in self.couplings:
                self.couplings[k] *= 0.99

        def signal_death(self):
            """Strong negative signal on death."""
            if self._state_trace and self._action_trace:
                # Penalize last state-action
                last_state = self._state_trace[-1]
                last_action = self._action_trace[-1]
                key = (last_state, last_action)
                self.couplings[key] -= 1.0
                self.actions[last_action] = self.actions.get(last_action, 0) - 0.5

            self._state_trace = []
            self._action_trace = []

        def signal_death_with_cause(self, state_tuple, action: int, width: int = 5):
            """
            Strong negative signal on death, with explicit cause.

            state_tuple: (relative_threat, agent_x) when death-causing action was taken
            action: the action that led to death
            """
            relative, agent_x = state_tuple
            state_id = self._get_state_id(relative, agent_x, width)
            action_id = self._get_action_id(action)

            # Strong penalty on the causal state-action pair
            key = (state_id, action_id)
            self.couplings[key] -= 2.0  # Strong penalty

            # Also penalize the action amplitude
            self.actions[action_id] = self.actions.get(action_id, 0) - 1.0

            self._state_trace = []
            self._action_trace = []

        def signal_survival_credit(self, relative: int, action: int, agent_x: int, width: int):
            """
            Explicit positive credit for surviving danger.

            This is the bootstrap/directed learning you mentioned -
            explicitly rewarding escape from dangerous situations.
            """
            state_id = self._get_state_id(relative, agent_x, width)
            action_id = self._get_action_id(action)

            # Positive coupling for this state-action pair
            key = (state_id, action_id)
            self.couplings[key] += 0.5  # Explicit positive credit

        def choose_action(self, state: int, agent_x: int = None, width: int = 5) -> int:
            """Choose based on state-action coupling with exploration."""
            state_id = self._get_state_id(state, agent_x, width)

            # EPSILON-GREEDY EXPLORATION
            # This is the bootstrap - force exploration of alternatives
            epsilon = max(0.1, 0.5 * np.exp(-self.frame_num / 10000))  # Decay from 50% to 10%
            if np.random.random() < epsilon:
                return np.random.randint(0, 3)

            action_scores = {}
            for a in range(3):
                action_id = self._get_action_id(a)
                key = (state_id, action_id)

                coupling = self.couplings.get(key, 0)
                amplitude = self.actions.get(action_id, 0)

                # Use real part (can be negative from death signal!)
                score = coupling.real + amplitude.real * 0.1 + 0.01
                action_scores[a] = score

            # Softmax with higher temperature for exploration
            scores = np.array(list(action_scores.values()))
            temperature = max(0.5, 2.0 * np.exp(-self.frame_num / 20000))  # Anneal from 2.0 to 0.5
            scores = scores - scores.max()  # Numerical stability
            exp_scores = np.exp(scores / temperature)
            probs = exp_scores / (exp_scores.sum() + 1e-10)

            return np.random.choice([0, 1, 2], p=probs)

    # Run test
    env = DodgeBall(width=5)
    sieve = MinimalDodgeSieve()

    NUM_GAMES = 5000
    game_lengths = []
    action_counts = {0: 0, 1: 0, 2: 0}  # Track action distribution

    for game in range(NUM_GAMES):
        state = env.reset()
        done = False
        steps = 0
        last_state_before_death = None
        last_action_before_death = None

        while not done and steps < 200:
            # Compute relative threat position
            agent_x = env.agent_x
            danger_row = env.height - 2  # Row above agent

            # Find closest threat in danger row
            threats = [bx for bx, by in env.balls if by == danger_row]
            if threats:
                # Relative position of closest threat
                closest = min(threats, key=lambda bx: abs(bx - agent_x))
                relative = closest - agent_x  # -2 to +2
                relative = max(-2, min(2, relative))  # Clamp
            else:
                relative = 99  # No threat

            # Choose action - now with edge awareness
            if game < 100:
                action = np.random.randint(0, 3)
            else:
                action = sieve.choose_action(relative, agent_x, env.width)

            # Store state-action BEFORE step (this is what caused death)
            last_state_before_death = (relative, agent_x)
            last_action_before_death = action

            # Track action distribution
            if game >= 100:
                action_counts[action] += 1

            # Observe with edge awareness
            sieve.observe(relative, action, agent_x, env.width)

            # Step
            prev_relative = relative
            state, done = env.step(action)
            steps += 1

            # EXPLICIT POSITIVE CREDIT: survived a dangerous situation!
            if not done and prev_relative in [-1, 0, 1]:  # Was in danger
                # We survived! This action was good.
                sieve.signal_survival_credit(prev_relative, action, agent_x, env.width)

        if done:
            # Death signal with the actual causing state-action
            if game < 110 and game >= 100:
                # Debug first 10 learned games
                rel, ax = last_state_before_death
                action_name = ['L', 'S', 'R'][last_action_before_death]
                edge = "L" if ax == 0 else ("R" if ax == env.width - 1 else "M")
                print(f"  DEATH: state=rel_{rel}_{edge}, action={action_name}, "
                      f"agent_x={env.agent_x}, balls={env.balls}")
            sieve.signal_death_with_cause(last_state_before_death, last_action_before_death, env.width)
        game_lengths.append(steps)

        if (game + 1) % 500 == 0:
            recent = np.mean(game_lengths[-200:])
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                pcts = [100 * action_counts[a] / total_actions for a in range(3)]
                print(f"Game {game+1}: Avg={recent:.1f}, Actions: L={pcts[0]:.0f}% S={pcts[1]:.0f}% R={pcts[2]:.0f}%")

    # Results
    print()
    print("=" * 70)
    print("MINIMAL STATE RESULTS")
    print("=" * 70)

    first_500 = np.mean(game_lengths[:500])
    last_500 = np.mean(game_lengths[-500:])

    print(f"First 500 games: {first_500:.1f} steps")
    print(f"Last 500 games: {last_500:.1f} steps")
    print(f"Improvement: {(last_500/first_500 - 1)*100:+.1f}%")

    # Action distribution
    total_actions = sum(action_counts.values())
    print(f"\nAction distribution:")
    for a, name in [(0, 'LEFT'), (1, 'STAY'), (2, 'RIGHT')]:
        print(f"  {name}: {100 * action_counts[a] / total_actions:.1f}%")

    # Random baseline
    print()
    random_lengths = []
    for _ in range(500):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = np.random.randint(0, 3)
            state, done = env.step(action)
            steps += 1
        random_lengths.append(steps)

    print(f"Random: {np.mean(random_lengths):.1f} steps")
    print(f"Oracle: 200 steps (max)")
    print()

    # Show learned couplings
    print("Learned state-action couplings:")
    print("  (L=left, S=stay, R=right)")
    print("  Negative = led to death, Positive = led to survival")
    print()
    for state_id in sorted(set(s for s, a in sieve.couplings.keys())):
        row = []
        for a in range(3):
            action_id = sieve._get_action_id(a)
            key = (state_id, action_id)
            coupling = sieve.couplings.get(key, 0)
            row.append(f"{coupling.real:+.2f}")
        print(f"  {state_id:>10}: L={row[0]}, S={row[1]}, R={row[2]}")

    # What SHOULD be learned:
    print()
    print("Expected optimal couplings:")
    print("  rel_-2: R should be positive (move away from threat on left)")
    print("  rel_-1: R should be positive (move away)")
    print("  rel_0:  L or R positive, S negative (escape!)")
    print("  rel_1:  L should be positive (move away)")
    print("  rel_2:  L should be positive (move away from threat on right)")
    print("  rel_99: Any is fine (no threat)")

    if last_500 > np.mean(random_lengths) * 1.5:
        print("\nSTATUS: STRONG LEARNING")
    elif last_500 > np.mean(random_lengths) * 1.2:
        print("\nSTATUS: MODERATE LEARNING")
    elif last_500 > np.mean(random_lengths) * 1.05:
        print("\nSTATUS: WEAK LEARNING")
    else:
        print("\nSTATUS: NOT LEARNING")

    return game_lengths


if __name__ == "__main__":
    test_minimal_state()
