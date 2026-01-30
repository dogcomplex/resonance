"""
GOAL-DIRECTED FRACTAL SIEVE
===========================

The key insight: a sieve that PLAYS must have:
1. Prediction - what will happen if I do X?
2. Valuation - how good is outcome Y?
3. Selection - pick action that leads to best predicted outcome

This is still sieves all the way down - but with goal gradient.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import cmath


@dataclass
class GoalConfig:
    """Configuration for goal-directed sieve."""
    damping: float = 0.1
    coupling_strength: float = 0.2
    survival_percentile: float = 70.0
    reward_amplification: float = 2.0  # How much to boost reward-correlated patterns
    prediction_horizon: int = 5  # How far ahead to predict
    exploration_noise: float = 0.1  # Randomness in action selection


class PredictiveSieve:
    """
    A sieve that predicts outcomes of actions.

    Key insight: instead of just finding "A correlates with B",
    find "if action X, then state S with probability P".

    This IS still a sieve - but the tokens are (state, action, outcome) triples.
    """

    def __init__(self, n_actions: int, config: Optional[GoalConfig] = None):
        self.n_actions = n_actions
        self.config = config or GoalConfig()

        # Transition sieve: encodes (state_feature, action) -> outcome_feature
        # The amplitude encodes how reliably this transition occurs
        self.transitions: Dict[Tuple[str, int, str], complex] = defaultdict(complex)

        # Reward sieve: encodes state_feature -> expected reward
        self.rewards: Dict[str, complex] = defaultdict(complex)

        # Action-value sieve: encodes (state_summary, action) -> value
        self.action_values: Dict[Tuple[str, int], complex] = defaultdict(complex)

    def observe_transition(self, before: Dict[str, float], action: int,
                          after: Dict[str, float], reward: float):
        """
        Observe one transition and update the predictive sieve.
        """
        # For each feature that changed significantly
        for feature in before:
            if feature in after:
                delta = after[feature] - before[feature]

                if abs(delta) > 0.01:
                    # Encode: (feature, action) -> delta direction
                    direction = "up" if delta > 0 else "down"
                    key = (feature, action, direction)

                    # Amplitude encodes strength, phase encodes consistency
                    # Positive phase = reliable transition
                    phase = np.pi * 0.25  # Start optimistic
                    self.transitions[key] += complex(np.cos(phase), np.sin(phase)) * abs(delta)

        # Update reward associations
        # Which features correlate with reward?
        for feature, value in after.items():
            reward_phase = reward * np.pi  # reward in [-1,1] maps to phase
            self.rewards[feature] += complex(np.cos(reward_phase), np.sin(reward_phase)) * value

    def predict_outcome(self, state: Dict[str, float], action: int) -> Dict[str, float]:
        """
        Predict what state features will look like after action.
        """
        predicted = state.copy()

        for feature, current_value in state.items():
            # Check transitions for this feature and action
            up_key = (feature, action, "up")
            down_key = (feature, action, "down")

            up_amp = abs(self.transitions.get(up_key, 0))
            down_amp = abs(self.transitions.get(down_key, 0))

            # Predict change based on relative amplitudes
            if up_amp + down_amp > 0.1:
                expected_delta = (up_amp - down_amp) / (up_amp + down_amp) * 0.1
                predicted[feature] = max(0, min(1, current_value + expected_delta))

        return predicted

    def evaluate_state(self, state: Dict[str, float]) -> float:
        """
        Evaluate how good a state is based on reward correlations.
        """
        value = 0.0
        total_weight = 0.0

        for feature, feat_value in state.items():
            if feature in self.rewards:
                reward_amp = self.rewards[feature]
                # Phase indicates reward direction, amplitude indicates reliability
                phase = cmath.phase(reward_amp)
                reliability = min(1.0, abs(reward_amp))

                # Positive phase = positive reward correlation
                contribution = np.sin(phase) * feat_value * reliability
                value += contribution
                total_weight += reliability

        return value / max(0.1, total_weight)

    def select_action(self, state: Dict[str, float]) -> int:
        """
        Select action that leads to best predicted outcome.

        This is where the sieve becomes an AGENT.
        """
        action_values = []

        for action in range(self.n_actions):
            # Predict outcome
            predicted = self.predict_outcome(state, action)
            # Evaluate predicted outcome
            value = self.evaluate_state(predicted)
            action_values.append(value)

        # Softmax selection with exploration
        values = np.array(action_values)

        if self.config.exploration_noise > 0:
            # Add exploration noise
            values += np.random.normal(0, self.config.exploration_noise, len(values))

        # Return best action
        return int(np.argmax(values))

    def evolve(self):
        """
        Evolve the sieve - decay weak patterns, normalize.
        """
        # Decay transitions
        for key in list(self.transitions.keys()):
            self.transitions[key] *= (1 - self.config.damping)
            if abs(self.transitions[key]) < 0.01:
                del self.transitions[key]

        # Decay rewards
        for key in list(self.rewards.keys()):
            self.rewards[key] *= (1 - self.config.damping * 0.5)  # Slower decay for rewards
            if abs(self.rewards[key]) < 0.01:
                del self.rewards[key]

        # Normalize if needed
        total = sum(abs(v) for v in self.transitions.values())
        if total > 100:
            scale = 100 / total
            self.transitions = {k: v * scale for k, v in self.transitions.items()}


class GoalDirectedAgent:
    """
    A complete agent using goal-directed fractal sieve.

    Structure:
    - Perceptual sieve: raw observations -> features
    - Predictive sieve: (features, action) -> predicted features
    - Value sieve: features -> reward expectation
    - Action sieve: state -> action selection

    All sieves interact through shared amplitude fields.
    """

    def __init__(self, frame_shape: Tuple[int, int], n_actions: int,
                 patch_size: int = 8):
        self.frame_shape = frame_shape
        self.n_actions = n_actions
        self.patch_size = patch_size

        # The predictive core
        self.predictor = PredictiveSieve(n_actions)

        # Feature extraction state
        self.prev_features: Optional[Dict[str, float]] = None
        self.prev_action: Optional[int] = None

        # Performance tracking
        self.total_reward = 0.0
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0

        # Previous frame for motion detection
        self._prev_frame: Optional[np.ndarray] = None

    def extract_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract simple features from frame."""
        features = {}

        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float) / 255.0

        h, w = frame.shape
        ps = self.patch_size

        # Patch-based features
        for y in range(0, h - ps + 1, ps):
            for x in range(0, w - ps + 1, ps):
                patch = frame[y:y+ps, x:x+ps]
                features[f"p_{y}_{x}"] = np.mean(patch)

        # Global features
        features["brightness"] = np.mean(frame)

        # Motion features (if we have previous)
        if self._prev_frame is not None:
            diff = np.abs(frame - self._prev_frame)
            features["motion"] = np.mean(diff)
            features["motion_y"] = np.mean(diff[:h//2, :]) - np.mean(diff[h//2:, :])
            features["motion_x"] = np.mean(diff[:, :w//2]) - np.mean(diff[:, w//2:])

        self._prev_frame = frame.copy()

        return features

    def observe_and_act(self, frame: np.ndarray, reward: float = 0.0) -> int:
        """
        Observe current state, learn from transition, select action.
        """
        # Extract features
        features = self.extract_features(frame)

        # Learn from previous transition
        if self.prev_features is not None and self.prev_action is not None:
            self.predictor.observe_transition(
                self.prev_features,
                self.prev_action,
                features,
                reward
            )

        # Track reward
        self.current_episode_reward += reward
        self.total_reward += reward

        # Evolve predictive sieve
        self.predictor.evolve()

        # Select action
        action = self.predictor.select_action(features)

        # Store for next step
        self.prev_features = features
        self.prev_action = action

        return action

    def episode_end(self):
        """Called at end of episode."""
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        n_transitions = len(self.predictor.transitions)
        n_rewards = len(self.predictor.rewards)

        return {
            'n_transitions': n_transitions,
            'n_reward_associations': n_rewards,
            'total_reward': self.total_reward,
            'episodes': len(self.episode_rewards),
            'recent_avg_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
        }


# =============================================================================
# TEST: Can this learn to play Pong?
# =============================================================================

def test_goal_directed_pong():
    """
    Test goal-directed agent on synthetic Pong.
    """
    print("=" * 60)
    print("GOAL-DIRECTED FRACTAL SIEVE: Learning Pong")
    print("=" * 60)

    agent = GoalDirectedAgent(
        frame_shape=(84, 84),
        n_actions=3,  # left, stay, right
        patch_size=8
    )

    # Run multiple episodes
    n_episodes = 50
    episode_scores = []

    for episode in range(n_episodes):
        # Reset game state
        ball_x, ball_y = 42, 20
        ball_dx, ball_dy = 2, 1.5
        paddle_x = 42
        score = 0

        for step in range(200):
            # Create frame
            frame = np.zeros((84, 84), dtype=np.uint8)

            # Draw ball
            bx, by = int(ball_x), int(ball_y)
            frame[max(0,by-2):min(84,by+2), max(0,bx-2):min(84,bx+2)] = 255

            # Draw paddle
            px = int(paddle_x)
            frame[78:82, max(0,px-8):min(84,px+8)] = 200

            # Draw walls
            frame[0:2, :] = 100
            frame[:, 0:2] = 100
            frame[:, 82:84] = 100

            # Compute reward
            reward = 0.0
            if ball_y >= 76:  # Ball near paddle level
                if abs(ball_x - paddle_x) < 12:
                    reward = 1.0  # Hit!
                    score += 1
                else:
                    reward = -1.0  # Miss

            # Agent observes and acts
            action = agent.observe_and_act(frame, reward)

            # Execute action
            if action == 0:
                paddle_x = max(10, paddle_x - 4)
            elif action == 2:
                paddle_x = min(74, paddle_x + 4)

            # Update ball physics
            ball_x += ball_dx
            ball_y += ball_dy

            if ball_x <= 4 or ball_x >= 80:
                ball_dx *= -1
            if ball_y <= 4:
                ball_dy *= -1
            if ball_y >= 78:
                # Reset ball
                ball_y = 20
                ball_x = np.random.uniform(20, 64)
                ball_dx = np.random.choice([-2, 2])
                ball_dy = abs(ball_dy)

        agent.episode_end()
        episode_scores.append(score)

        if (episode + 1) % 10 == 0:
            stats = agent.get_statistics()
            recent_scores = episode_scores[-10:]
            print(f"\nEpisode {episode + 1}:")
            print(f"  Recent avg score: {np.mean(recent_scores):.1f}")
            print(f"  Recent avg reward: {stats['recent_avg_reward']:.2f}")
            print(f"  Learned transitions: {stats['n_transitions']}")
            print(f"  Reward associations: {stats['n_reward_associations']}")

    # Final analysis
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    stats = agent.get_statistics()
    print(f"Total episodes: {stats['episodes']}")
    print(f"Total reward: {stats['total_reward']:.1f}")

    # Compare first half vs second half
    first_half = np.mean(episode_scores[:25])
    second_half = np.mean(episode_scores[25:])
    print(f"\nLearning curve:")
    print(f"  First 25 episodes avg score: {first_half:.1f}")
    print(f"  Last 25 episodes avg score: {second_half:.1f}")
    print(f"  Improvement: {second_half - first_half:+.1f}")

    # Show what the agent learned
    print(f"\nLearned {stats['n_transitions']} transition patterns")
    print(f"Learned {stats['n_reward_associations']} reward associations")

    # Show top reward associations
    print("\nTop reward associations:")
    sorted_rewards = sorted(
        agent.predictor.rewards.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    for feature, amp in sorted_rewards:
        phase = cmath.phase(amp)
        phase_meaning = "GOOD" if phase > 0 else "BAD"
        print(f"  {feature}: strength={abs(amp):.3f}, correlation={phase_meaning}")

    return agent


# =============================================================================
# DEEPER ANALYSIS: What would make this work perfectly?
# =============================================================================

def analyze_what_perfect_play_needs():
    """
    Analyze what the sieve would need to learn for perfect Pong.
    """
    print("\n" + "=" * 60)
    print("WHAT PERFECT PONG PLAY REQUIRES")
    print("=" * 60)

    print("""
LEVEL 1: PERCEPTION
-------------------
The agent must learn to see:
  - Ball position (x, y)
  - Ball velocity (dx, dy)
  - Paddle position
  - Distance ball-to-paddle

Current implementation: Patch features capture this approximately.
The motion features help with velocity.

Missing: Explicit object detection. The sieve sees "brightness at location"
not "there is a ball at location". This works but is inefficient.


LEVEL 2: PREDICTION
-------------------
The agent must predict:
  - Where will ball be in N frames?
  - Will it hit paddle if I stay?
  - Will it hit if I move left/right?

Current implementation: PredictiveSieve learns (feature, action) -> delta.
This is one-step prediction.

Missing: Multi-step rollout. Perfect play needs to predict ball trajectory
across many frames, not just one. The sieve architecture could do this
by having "prediction sieves" that chain predictions.


LEVEL 3: PLANNING
-----------------
The agent must plan:
  - What sequence of actions intercepts the ball?
  - What's the minimum moves needed?

Current implementation: One-step lookahead (predict + evaluate).

Missing: Search/planning. The sieve finds patterns but doesn't
explore action sequences. Could be added with "planning sieves"
that evaluate multi-step action sequences.


LEVEL 4: EXECUTION
------------------
The agent must act smoothly:
  - Move toward predicted intercept point
  - Don't overshoot

Current implementation: Greedy action selection.

This is probably fine for Pong - the action space is simple.


THE KEY MISSING PIECE: TEMPORAL ABSTRACTION
-------------------------------------------
The ball doesn't need tracking every frame - it follows a predictable
trajectory. Perfect play would learn:

  "Ball moving right+down at (40, 30) -> will hit paddle zone at x=60"

This is a RULE that predicts far ahead. The current sieve finds
frame-to-frame correlations but not this higher-level pattern.

To fix: The fractal structure should go deeper - sub-sieves that
learn "trajectory patterns" not just "frame deltas".
""")


if __name__ == "__main__":
    agent = test_goal_directed_pong()
    analyze_what_perfect_play_needs()
