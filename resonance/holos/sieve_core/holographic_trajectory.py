"""
HOLOGRAPHIC TRAJECTORY SIEVE
============================

Instead of discretizing trajectories, encode them holographically.
The phase encodes continuous values, and interference naturally
handles similarity/generalization.

Key insight: A trajectory is a point in a high-dimensional space
(x, y, vx, vy). We can encode this holographically and let
interference find similar trajectories.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import defaultdict
from dataclasses import dataclass
import cmath


class Trajectory(NamedTuple):
    """A trajectory state."""
    x: float
    y: float
    vx: float
    vy: float


@dataclass
class HoloConfig:
    """Configuration for holographic trajectory sieve."""
    memory_size: int = 256  # Size of holographic memory
    learning_rate: float = 0.1
    decay_rate: float = 0.001
    prediction_noise: float = 0.01


class HolographicMemory:
    """
    Holographic associative memory for trajectories.

    SIMPLER APPROACH: Instead of trying to decode phases,
    store associations directly and retrieve by similarity.

    Key insight: similar trajectories should retrieve similar intercepts.
    We achieve this by encoding trajectories as keys and storing
    intercept predictions as values, with similarity-based retrieval.
    """

    def __init__(self, config: Optional[HoloConfig] = None):
        self.config = config or HoloConfig()

        # Store (trajectory_encoding, intercept) pairs
        # The encoding is a fixed random projection
        np.random.seed(42)
        self.projection = np.random.randn(4, self.config.memory_size)

        # Memory stores weighted intercepts for each memory slot
        self.memory_intercepts = np.zeros(self.config.memory_size)
        self.memory_weights = np.zeros(self.config.memory_size)

    def _encode_trajectory(self, traj: Trajectory) -> np.ndarray:
        """
        Encode trajectory as a similarity-preserving pattern.

        Uses locality-sensitive hashing idea: similar inputs activate
        similar patterns.
        """
        # Normalize
        features = np.array([
            traj.x / 84.0,
            traj.y / 84.0,
            (traj.vx + 5) / 10.0,
            (traj.vy + 5) / 10.0
        ])

        # Project to memory space
        projected = np.dot(features, self.projection)

        # Soft activation (similar to RBF kernel)
        # Higher values for features that align with projection
        activation = np.exp(-np.abs(projected))

        return activation / (np.sum(activation) + 1e-6)  # Normalize

    def store(self, trajectory: Trajectory, intercept_x: float):
        """Store trajectory -> intercept association."""
        pattern = self._encode_trajectory(trajectory)

        # Update memory with weighted average
        self.memory_intercepts += pattern * intercept_x * self.config.learning_rate
        self.memory_weights += pattern * self.config.learning_rate

        # Decay
        self.memory_intercepts *= (1 - self.config.decay_rate)
        self.memory_weights *= (1 - self.config.decay_rate)

    def retrieve(self, trajectory: Trajectory) -> float:
        """Retrieve predicted intercept."""
        pattern = self._encode_trajectory(trajectory)

        # Weighted retrieval
        weighted_sum = np.sum(pattern * self.memory_intercepts)
        weight_sum = np.sum(pattern * self.memory_weights)

        if weight_sum < 0.01:
            return 42.0  # Default to center

        return weighted_sum / weight_sum


class HolographicTrajectoryAgent:
    """
    Agent using holographic trajectory memory for Pong.
    """

    def __init__(self):
        self.memory = HolographicMemory()

        # Ball tracking
        self.ball_history: List[Tuple[float, float]] = []
        self.trajectory_start: Optional[Trajectory] = None
        self.paddle_x: float = 42.0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.trajectories_learned = 0

    def _find_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find ball in frame."""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        upper = frame[:75, :]
        if np.max(upper) > 200:
            ys, xs = np.where(upper > 200)
            return (float(np.mean(xs)), float(np.mean(ys)))
        return None

    def _find_paddle(self, frame: np.ndarray) -> float:
        """Find paddle in frame."""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        lower = frame[75:, :]
        if np.max(lower) > 150:
            _, xs = np.where(lower > 150)
            return float(np.mean(xs))
        return self.paddle_x

    def observe_and_act(self, frame: np.ndarray, reward: float = 0.0) -> int:
        """Observe and act."""
        # Track ball
        ball_pos = self._find_ball(frame)
        self.paddle_x = self._find_paddle(frame)

        current_trajectory: Optional[Trajectory] = None

        if ball_pos is not None:
            self.ball_history.append(ball_pos)

            if len(self.ball_history) >= 2:
                prev = self.ball_history[-2]
                vx = ball_pos[0] - prev[0]
                vy = ball_pos[1] - prev[1]
                current_trajectory = Trajectory(ball_pos[0], ball_pos[1], vx, vy)

                # Ball moving down?
                if vy > 0.5:
                    if self.trajectory_start is None:
                        self.trajectory_start = current_trajectory

                    # Ball near bottom? Learn!
                    if ball_pos[1] >= 70:
                        if self.trajectory_start is not None:
                            self.memory.store(self.trajectory_start, ball_pos[0])
                            self.trajectories_learned += 1
                            self.trajectory_start = None
                else:
                    self.trajectory_start = None

        # Keep limited history
        if len(self.ball_history) > 10:
            self.ball_history = self.ball_history[-10:]

        # Track rewards
        if reward > 0:
            self.hits += 1
        elif reward < 0:
            self.misses += 1

        # Select action
        return self._select_action(current_trajectory)

    def _select_action(self, traj: Optional[Trajectory]) -> int:
        """Select action based on predicted intercept."""
        if traj is None or traj.vy <= 0:
            return 1  # Stay

        # Predict intercept
        predicted_x = self.memory.retrieve(traj)

        # Move toward predicted intercept
        diff = predicted_x - self.paddle_x

        if diff < -4:
            return 0  # Left
        elif diff > 4:
            return 2  # Right
        return 1  # Stay

    def get_statistics(self) -> Dict:
        """Get stats."""
        return {
            'trajectories_learned': self.trajectories_learned,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(1, self.hits + self.misses),
            'memory_weight_sum': np.sum(self.memory.memory_weights),
        }


def test_holographic_agent():
    """Test holographic trajectory agent."""
    print("=" * 60)
    print("HOLOGRAPHIC TRAJECTORY SIEVE: Learning Pong")
    print("=" * 60)

    agent = HolographicTrajectoryAgent()

    n_episodes = 200
    episode_stats = []

    for episode in range(n_episodes):
        # Reset game
        ball_x, ball_y = 42, 20
        ball_dx = np.random.choice([-2, -1.5, 1.5, 2])
        ball_dy = 1.5
        paddle_x = 42.0

        episode_hits = 0
        episode_misses = 0

        for step in range(300):
            # Create frame
            frame = np.zeros((84, 84), dtype=np.uint8)

            bx, by = int(ball_x), int(ball_y)
            frame[max(0,by-2):min(84,by+2), max(0,bx-2):min(84,bx+2)] = 255

            px = int(paddle_x)
            frame[78:82, max(0,px-8):min(84,px+8)] = 200

            frame[0:2, :] = 100
            frame[:, 0:2] = 100
            frame[:, 82:84] = 100

            # Reward
            reward = 0.0
            if ball_y >= 76 and ball_y < 80:
                if abs(ball_x - paddle_x) < 12:
                    reward = 1.0
                    episode_hits += 1
                else:
                    reward = -1.0
                    episode_misses += 1

            # Agent acts
            action = agent.observe_and_act(frame, reward)

            if action == 0:
                paddle_x = max(10, paddle_x - 4)
            elif action == 2:
                paddle_x = min(74, paddle_x + 4)

            # Physics
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
                ball_dx = np.random.choice([-2, -1.5, 1.5, 2])
                ball_dy = abs(ball_dy)

        episode_stats.append({'hits': episode_hits, 'misses': episode_misses})

        if (episode + 1) % 40 == 0:
            stats = agent.get_statistics()
            recent = episode_stats[-40:]
            recent_hits = sum(e['hits'] for e in recent)
            recent_misses = sum(e['misses'] for e in recent)
            hit_rate = recent_hits / max(1, recent_hits + recent_misses)

            print(f"\nEpisode {episode + 1}:")
            print(f"  Recent hit rate: {hit_rate:.1%}")
            print(f"  Trajectories learned: {stats['trajectories_learned']}")
            print(f"  Memory weights: {stats['memory_weight_sum']:.4f}")

    # Final
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    stats = agent.get_statistics()
    print(f"Overall hit rate: {stats['hit_rate']:.1%}")

    first_quarter = episode_stats[:50]
    last_quarter = episode_stats[-50:]
    first_rate = sum(e['hits'] for e in first_quarter) / max(1, sum(e['hits']+e['misses'] for e in first_quarter))
    last_rate = sum(e['hits'] for e in last_quarter) / max(1, sum(e['hits']+e['misses'] for e in last_quarter))

    print(f"\nLearning curve:")
    print(f"  First 50 episodes: {first_rate:.1%}")
    print(f"  Last 50 episodes: {last_rate:.1%}")
    print(f"  Improvement: {(last_rate - first_rate)*100:+.1f}%")

    return agent


if __name__ == "__main__":
    agent = test_holographic_agent()
