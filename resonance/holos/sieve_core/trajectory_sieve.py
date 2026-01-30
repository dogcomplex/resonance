"""
TRAJECTORY SIEVE: Temporal Abstraction for Game Playing
========================================================

Key insight: Instead of learning frame-to-frame transitions,
learn TRAJECTORY PATTERNS that predict outcomes.

A trajectory is: "object at position P moving with velocity V
will end up at position P' after T timesteps"

This is the missing layer that makes perfect play possible.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import defaultdict
from dataclasses import dataclass, field
import cmath


class ObjectState(NamedTuple):
    """Tracked object state."""
    x: float
    y: float
    vx: float
    vy: float


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory sieve."""
    damping: float = 0.001  # Very slow decay - trajectories are VERY stable
    position_bins: int = 12  # Finer position discretization
    velocity_bins: int = 5  # More velocity bins
    prediction_steps: int = 30  # How far to predict
    exploration: float = 0.02  # Less exploration as we learn


class ObjectTracker:
    """
    Simple object tracker for Pong.

    In a real system, this would emerge from the perceptual sieve.
    Here we cheat slightly by providing structure.
    """

    def __init__(self, frame_shape: Tuple[int, int]):
        self.frame_shape = frame_shape
        self.prev_frame: Optional[np.ndarray] = None
        self.ball_history: List[Tuple[float, float]] = []
        self.paddle_x: float = frame_shape[1] / 2

    def update(self, frame: np.ndarray) -> Tuple[Optional[ObjectState], float]:
        """
        Track objects in frame.

        Returns: (ball_state, paddle_x)
        """
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float)

        h, w = frame.shape

        # Find ball (brightest small region in upper 90% of screen)
        upper_region = frame[:int(h*0.9), :]
        ball_mask = upper_region > 200

        if np.any(ball_mask):
            ys, xs = np.where(ball_mask)
            ball_x = float(np.mean(xs))
            ball_y = float(np.mean(ys))
            self.ball_history.append((ball_x, ball_y))

            # Compute velocity from history
            if len(self.ball_history) >= 2:
                prev_x, prev_y = self.ball_history[-2]
                vx = ball_x - prev_x
                vy = ball_y - prev_y
            else:
                vx, vy = 0, 0

            ball_state = ObjectState(ball_x, ball_y, vx, vy)
        else:
            ball_state = None

        # Keep limited history
        if len(self.ball_history) > 10:
            self.ball_history = self.ball_history[-10:]

        # Find paddle (bright region in bottom of screen)
        paddle_region = frame[int(h*0.9):, :]
        paddle_mask = paddle_region > 150

        if np.any(paddle_mask):
            _, xs = np.where(paddle_mask)
            self.paddle_x = float(np.mean(xs))

        return ball_state, self.paddle_x


class TrajectorySieve:
    """
    A sieve that learns trajectory patterns.

    Key insight: discretize the state space and learn which
    discretized states lead to which outcomes.

    Trajectories are encoded as:
        (position_bin, velocity_bin) -> outcome_bin

    Where outcome_bin is "where will ball be at paddle level"
    """

    def __init__(self, config: Optional[TrajectoryConfig] = None):
        self.config = config or TrajectoryConfig()

        # Trajectory sieve: (pos_bin, vel_bin) -> predicted_intercept_bin
        # Amplitude encodes confidence in prediction
        self.trajectories: Dict[Tuple[int, int, int, int], Dict[int, complex]] = defaultdict(
            lambda: defaultdict(complex)
        )

        # Outcome history for learning
        self.pending_predictions: List[Tuple[Tuple[int, int, int, int], int]] = []

    def _discretize_position(self, x: float, y: float, width: float = 84, height: float = 84) -> Tuple[int, int]:
        """Discretize position to bin."""
        x_bin = int(np.clip(x / width * self.config.position_bins, 0, self.config.position_bins - 1))
        y_bin = int(np.clip(y / height * self.config.position_bins, 0, self.config.position_bins - 1))
        return (x_bin, y_bin)

    def _discretize_velocity(self, vx: float, vy: float) -> Tuple[int, int]:
        """Discretize velocity to bin."""
        # Velocity bins: -2, -1, 0, 1, 2 mapped to 0-4
        vx_bin = int(np.clip(np.sign(vx) * min(2, abs(vx)) + 2, 0, self.config.velocity_bins))
        vy_bin = int(np.clip(np.sign(vy) * min(2, abs(vy)) + 2, 0, self.config.velocity_bins))
        return (vx_bin, vy_bin)

    def _predict_intercept(self, ball: ObjectState, paddle_y: float = 78) -> float:
        """
        Predict where ball will intercept paddle level.

        This is simple physics - but the sieve should LEARN this.
        """
        if ball.vy <= 0:
            # Ball moving up, will bounce
            # For simplicity, just return current x
            return ball.x

        # Time to reach paddle
        if ball.vy > 0:
            time_to_paddle = (paddle_y - ball.y) / ball.vy
            # Where will ball be?
            x_at_paddle = ball.x + ball.vx * time_to_paddle

            # Handle wall bounces (simple approximation)
            while x_at_paddle < 0 or x_at_paddle > 84:
                if x_at_paddle < 0:
                    x_at_paddle = -x_at_paddle
                if x_at_paddle > 84:
                    x_at_paddle = 168 - x_at_paddle

            return x_at_paddle
        return ball.x

    def observe(self, ball: ObjectState, actual_intercept_x: Optional[float] = None):
        """
        Observe a ball state and optionally its actual intercept.
        """
        # Discretize current state
        pos_bin = self._discretize_position(ball.x, ball.y)
        vel_bin = self._discretize_velocity(ball.vx, ball.vy)
        state_key = (*pos_bin, *vel_bin)

        if actual_intercept_x is not None:
            # We know where it actually went - learn!
            intercept_bin = int(np.clip(
                actual_intercept_x / 84 * self.config.position_bins,
                0, self.config.position_bins - 1
            ))

            # Strengthen this trajectory prediction
            self.trajectories[state_key][intercept_bin] += complex(1.0, 0)

            # Weaken other predictions slightly (competition)
            for other_bin in self.trajectories[state_key]:
                if other_bin != intercept_bin:
                    self.trajectories[state_key][other_bin] *= 0.95

    def predict_intercept_bin(self, ball: ObjectState) -> int:
        """
        Predict where ball will intercept based on learned trajectories.

        NO PHYSICS FALLBACK - must learn from experience only!
        """
        pos_bin = self._discretize_position(ball.x, ball.y)
        vel_bin = self._discretize_velocity(ball.vx, ball.vy)
        state_key = (*pos_bin, *vel_bin)

        predictions = self.trajectories.get(state_key, {})

        if predictions:
            # Return most confident prediction
            best_bin = max(predictions.keys(), key=lambda b: abs(predictions[b]))
            return best_bin
        else:
            # No learned prediction - guess center (exploration)
            return self.config.position_bins // 2

    def evolve(self):
        """Decay weak trajectory patterns."""
        for state_key in list(self.trajectories.keys()):
            for intercept_bin in list(self.trajectories[state_key].keys()):
                self.trajectories[state_key][intercept_bin] *= (1 - self.config.damping)
                if abs(self.trajectories[state_key][intercept_bin]) < 0.01:
                    del self.trajectories[state_key][intercept_bin]
            if not self.trajectories[state_key]:
                del self.trajectories[state_key]


class TrajectoryAgent:
    """
    Agent that uses trajectory prediction for action selection.

    The key insight: instead of learning (state, action) -> value,
    learn trajectory -> intercept, then choose action to reach intercept.
    """

    def __init__(self, frame_shape: Tuple[int, int] = (84, 84), n_actions: int = 3):
        self.frame_shape = frame_shape
        self.n_actions = n_actions

        # Trajectory learning
        self.trajectory_sieve = TrajectorySieve()

        # Ball tracking (simple, direct)
        self.ball_history: List[Tuple[float, float]] = []
        self.trajectory_start_state: Optional[ObjectState] = None
        self.paddle_x: float = 42.0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.episode_hits = 0
        self.episode_misses = 0
        self.trajectories_learned = 0

    def _find_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find ball position in frame."""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)

        # Ball is brightest object in upper region
        upper = frame[:75, :]
        if np.max(upper) > 200:
            ys, xs = np.where(upper > 200)
            return (float(np.mean(xs)), float(np.mean(ys)))
        return None

    def _find_paddle(self, frame: np.ndarray) -> float:
        """Find paddle position."""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)

        lower = frame[75:, :]
        if np.max(lower) > 150:
            _, xs = np.where(lower > 150)
            return float(np.mean(xs))
        return self.paddle_x

    def observe_and_act(self, frame: np.ndarray, reward: float = 0.0) -> int:
        """
        Observe frame, learn from trajectory, select action.
        """
        # Track ball
        ball_pos = self._find_ball(frame)
        self.paddle_x = self._find_paddle(frame)

        if ball_pos is not None:
            self.ball_history.append(ball_pos)

            # Compute velocity from history
            if len(self.ball_history) >= 2:
                prev = self.ball_history[-2]
                vx = ball_pos[0] - prev[0]
                vy = ball_pos[1] - prev[1]

                ball_state = ObjectState(ball_pos[0], ball_pos[1], vx, vy)

                # Ball moving down?
                if vy > 0.5:
                    # Start tracking trajectory if not already
                    if self.trajectory_start_state is None:
                        self.trajectory_start_state = ball_state

                    # Ball near bottom? Learn the trajectory outcome
                    if ball_pos[1] >= 70:
                        if self.trajectory_start_state is not None:
                            self.trajectory_sieve.observe(self.trajectory_start_state, ball_pos[0])
                            self.trajectories_learned += 1
                            self.trajectory_start_state = None
                else:
                    # Ball moving up - reset
                    self.trajectory_start_state = None

        # Keep limited history
        if len(self.ball_history) > 10:
            self.ball_history = self.ball_history[-10:]

        # Track hits/misses
        if reward > 0:
            self.hits += 1
            self.episode_hits += 1
        elif reward < 0:
            self.misses += 1
            self.episode_misses += 1

        # Evolve sieve
        self.trajectory_sieve.evolve()

        # Select action
        ball_state = None
        if len(self.ball_history) >= 2:
            prev = self.ball_history[-2]
            curr = self.ball_history[-1]
            ball_state = ObjectState(curr[0], curr[1], curr[0]-prev[0], curr[1]-prev[1])

        action = self._select_action(ball_state, self.paddle_x)

        return action

    def _select_action(self, ball: Optional[ObjectState], paddle_x: float) -> int:
        """
        Select action to intercept predicted ball position.
        """
        if ball is None or ball.vy <= 0:
            # Ball moving away or not visible - stay
            return 1

        # Predict where ball will intercept
        intercept_bin = self.trajectory_sieve.predict_intercept_bin(ball)

        # Convert bin to position
        target_x = (intercept_bin + 0.5) * 84 / self.trajectory_sieve.config.position_bins

        # Move toward target
        diff = target_x - paddle_x

        # Add small exploration noise
        if np.random.random() < self.trajectory_sieve.config.exploration:
            return np.random.randint(0, 3)

        if diff < -4:
            return 0  # Move left
        elif diff > 4:
            return 2  # Move right
        else:
            return 1  # Stay

    def episode_end(self) -> Tuple[int, int]:
        """End episode, return (hits, misses)."""
        result = (self.episode_hits, self.episode_misses)
        self.episode_hits = 0
        self.episode_misses = 0
        return result

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        n_trajectories = sum(len(v) for v in self.trajectory_sieve.trajectories.values())
        return {
            'n_trajectory_patterns': len(self.trajectory_sieve.trajectories),
            'n_predictions': n_trajectories,
            'trajectories_learned': self.trajectories_learned,
            'total_hits': self.hits,
            'total_misses': self.misses,
            'hit_rate': self.hits / max(1, self.hits + self.misses),
        }


# =============================================================================
# TEST: Learning trajectories for Pong
# =============================================================================

def test_trajectory_agent():
    """
    Test trajectory-based agent on synthetic Pong.
    """
    print("=" * 60)
    print("TRAJECTORY SIEVE: Learning to Play Pong")
    print("=" * 60)

    agent = TrajectoryAgent()

    n_episodes = 200
    episode_stats = []

    for episode in range(n_episodes):
        # Reset game
        ball_x, ball_y = 42, 20
        ball_dx, ball_dy = np.random.choice([-2, 2]), 1.5
        paddle_x = 42.0

        for step in range(300):
            # Create frame
            frame = np.zeros((84, 84), dtype=np.uint8)

            # Draw ball
            bx, by = int(ball_x), int(ball_y)
            frame[max(0,by-2):min(84,by+2), max(0,bx-2):min(84,bx+2)] = 255

            # Draw paddle
            px = int(paddle_x)
            frame[78:82, max(0,px-8):min(84,px+8)] = 200

            # Walls
            frame[0:2, :] = 100
            frame[:, 0:2] = 100
            frame[:, 82:84] = 100

            # Reward
            reward = 0.0
            if ball_y >= 76 and ball_y < 80:
                if abs(ball_x - paddle_x) < 12:
                    reward = 1.0
                else:
                    reward = -1.0

            # Agent acts
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
                ball_x = np.clip(ball_x, 4, 80)
            if ball_y <= 4:
                ball_dy = abs(ball_dy)
            if ball_y >= 78:
                ball_y = 20
                ball_x = np.random.uniform(20, 64)
                ball_dx = np.random.choice([-2, 2])
                ball_dy = abs(ball_dy)

        hits, misses = agent.episode_end()
        episode_stats.append({'hits': hits, 'misses': misses})

        if (episode + 1) % 20 == 0:
            stats = agent.get_statistics()
            recent = episode_stats[-20:]
            recent_hits = sum(e['hits'] for e in recent)
            recent_misses = sum(e['misses'] for e in recent)
            hit_rate = recent_hits / max(1, recent_hits + recent_misses)

            print(f"\nEpisode {episode + 1}:")
            print(f"  Recent hit rate: {hit_rate:.1%}")
            print(f"  Trajectories observed: {stats['trajectories_learned']}")
            print(f"  Unique trajectory patterns: {stats['n_trajectory_patterns']}")
            print(f"  Total predictions: {stats['n_predictions']}")

    # Final analysis
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    stats = agent.get_statistics()
    print(f"Overall hit rate: {stats['hit_rate']:.1%}")
    print(f"Total hits: {stats['total_hits']}")
    print(f"Total misses: {stats['total_misses']}")

    # Learning curve
    first_quarter = episode_stats[:25]
    last_quarter = episode_stats[-25:]

    first_rate = sum(e['hits'] for e in first_quarter) / max(1, sum(e['hits'] + e['misses'] for e in first_quarter))
    last_rate = sum(e['hits'] for e in last_quarter) / max(1, sum(e['hits'] + e['misses'] for e in last_quarter))

    print(f"\nLearning curve:")
    print(f"  First 25 episodes hit rate: {first_rate:.1%}")
    print(f"  Last 25 episodes hit rate: {last_rate:.1%}")
    print(f"  Improvement: {(last_rate - first_rate)*100:+.1f}%")

    # Show learned trajectories
    print(f"\nLearned {stats['n_trajectory_patterns']} trajectory state patterns")
    print(f"Each predicts where ball will intercept paddle")

    # Sample some learned trajectories
    print("\nSample learned trajectories:")
    for state_key, predictions in list(agent.trajectory_sieve.trajectories.items())[:5]:
        px, py, vx, vy = state_key
        best_intercept = max(predictions.keys(), key=lambda k: abs(predictions[k]))
        confidence = abs(predictions[best_intercept])
        print(f"  Position bin ({px},{py}), velocity bin ({vx},{vy})")
        print(f"    -> Predicts intercept at bin {best_intercept} (confidence: {confidence:.2f})")

    return agent


if __name__ == "__main__":
    agent = test_trajectory_agent()
