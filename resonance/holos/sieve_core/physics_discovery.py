"""
PHYSICS DISCOVERY SIEVE
=======================

The key insight: instead of learning (trajectory) -> (intercept) directly,
discover the STRUCTURE of trajectories:
  1. Ball moves linearly between bounces
  2. Velocity reverses at walls
  3. Intercept can be computed from initial state

This is what makes perfect play possible - discovering the physics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import defaultdict
from dataclasses import dataclass


class Observation(NamedTuple):
    """Single observation."""
    x: float
    y: float
    t: int


@dataclass
class PhysicsRule:
    """A discovered physics rule."""
    name: str
    confidence: float
    description: str


class PhysicsDiscoverySieve:
    """
    A sieve that discovers physics rules from observations.

    Instead of memorizing trajectories, discovers:
    - Motion patterns (linear, accelerating, etc.)
    - Boundary conditions (bounces, wraps, stops)
    - Conservation laws (momentum, energy)
    """

    def __init__(self):
        # Observation history
        self.observations: List[Observation] = []

        # Discovered rules
        self.rules: Dict[str, PhysicsRule] = {}

        # Pattern counters for rule discovery
        self.linear_motion_count = 0
        self.total_motion_count = 0
        self.bounce_left_count = 0
        self.bounce_right_count = 0
        self.bounce_top_count = 0

    def observe(self, x: float, y: float, t: int):
        """Add observation."""
        self.observations.append(Observation(x, y, t))

        # Keep limited history
        if len(self.observations) > 100:
            self.observations = self.observations[-100:]

        # Try to discover rules from observations
        self._discover_rules()

    def _discover_rules(self):
        """Discover physics rules from observation history."""
        if len(self.observations) < 10:
            return

        # Check for linear motion
        self._check_linear_motion()

        # Check for boundary bounces
        self._check_bounces()

        # Update rule confidences
        self._update_rules()

    def _check_linear_motion(self):
        """Check if recent motion is linear."""
        recent = self.observations[-10:]

        # Compute velocities between consecutive observations
        vxs = []
        vys = []
        for i in range(1, len(recent)):
            dt = recent[i].t - recent[i-1].t
            if dt > 0:
                vx = (recent[i].x - recent[i-1].x) / dt
                vy = (recent[i].y - recent[i-1].y) / dt
                vxs.append(vx)
                vys.append(vy)

        if len(vxs) < 3:
            return

        # Check if velocities are constant (linear motion)
        vx_std = np.std(vxs)
        vy_std = np.std(vys)

        self.total_motion_count += 1

        if vx_std < 0.5 and vy_std < 0.5:
            # Linear motion!
            self.linear_motion_count += 1

    def _check_bounces(self):
        """Check for boundary bounces."""
        recent = self.observations[-20:]

        for i in range(2, len(recent)):
            # Check for velocity reversal at boundaries
            x0, x1, x2 = recent[i-2].x, recent[i-1].x, recent[i].x
            y0, y1, y2 = recent[i-2].y, recent[i-1].y, recent[i].y

            vx_before = x1 - x0
            vx_after = x2 - x1
            vy_before = y1 - y0
            vy_after = y2 - y1

            # Check for X velocity reversal
            if vx_before * vx_after < -0.5:  # Sign changed
                if x1 <= 10:
                    self.bounce_left_count += 1
                elif x1 >= 74:
                    self.bounce_right_count += 1

            # Check for Y velocity reversal at top
            if vy_before * vy_after < -0.5:
                if y1 <= 10:
                    self.bounce_top_count += 1

    def _update_rules(self):
        """Update discovered rules based on counts."""
        # Linear motion rule
        if self.total_motion_count > 10:
            linearity = self.linear_motion_count / self.total_motion_count
            if linearity > 0.8:
                self.rules['linear_motion'] = PhysicsRule(
                    name='linear_motion',
                    confidence=linearity,
                    description='Ball moves in straight lines (constant velocity)'
                )

        # Bounce rules
        total_bounces = self.bounce_left_count + self.bounce_right_count + self.bounce_top_count
        if total_bounces > 5:
            self.rules['wall_bounce'] = PhysicsRule(
                name='wall_bounce',
                confidence=min(1.0, total_bounces / 10),
                description='Ball bounces off walls (velocity reverses)'
            )

    def get_discovered_rules(self) -> List[PhysicsRule]:
        """Get all discovered rules."""
        return list(self.rules.values())

    def predict_intercept(self, x: float, y: float, vx: float, vy: float,
                          paddle_y: float = 76) -> Optional[float]:
        """
        Predict intercept using discovered rules.

        Returns None if we haven't discovered enough physics yet.
        """
        # Need linear motion rule to predict
        if 'linear_motion' not in self.rules:
            return None

        if vy <= 0:
            return None  # Ball moving up

        # Predict using linear extrapolation
        time_to_paddle = (paddle_y - y) / vy
        predicted_x = x + vx * time_to_paddle

        # Apply bounce rule if discovered
        if 'wall_bounce' in self.rules:
            # Handle bounces
            while predicted_x < 4 or predicted_x > 80:
                if predicted_x < 4:
                    predicted_x = -predicted_x + 8  # Bounce off left
                if predicted_x > 80:
                    predicted_x = 160 - predicted_x  # Bounce off right

        return predicted_x


class PhysicsDiscoveryAgent:
    """
    Agent that discovers physics to play Pong.

    The key: first discover the physics, then use physics for prediction.
    This is how intelligent systems should work - understanding before acting.
    """

    def __init__(self):
        self.physics = PhysicsDiscoverySieve()

        # Ball tracking
        self.ball_history: List[Tuple[float, float]] = []
        self.paddle_x: float = 42.0
        self.frame_count: int = 0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.predictions_using_physics = 0
        self.predictions_using_fallback = 0

    def _find_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find ball in frame."""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        upper = frame[:75, :]
        if np.max(upper) > 200:
            ys, xs = np.where(upper > 200)
            return (float(np.mean(xs)), float(np.mean(ys)))
        return None

    def observe_and_act(self, frame: np.ndarray, reward: float = 0.0) -> int:
        """Observe and act."""
        self.frame_count += 1

        # Find ball
        ball_pos = self._find_ball(frame)

        if ball_pos is not None:
            self.ball_history.append(ball_pos)

            # Provide observation to physics discoverer
            self.physics.observe(ball_pos[0], ball_pos[1], self.frame_count)

        # Keep limited history
        if len(self.ball_history) > 10:
            self.ball_history = self.ball_history[-10:]

        # Track rewards
        if reward > 0:
            self.hits += 1
        elif reward < 0:
            self.misses += 1

        # Find paddle
        if len(frame.shape) == 3:
            frame_2d = np.mean(frame, axis=2)
        else:
            frame_2d = frame
        lower = frame_2d[75:, :]
        if np.max(lower) > 150:
            _, xs = np.where(lower > 150)
            self.paddle_x = float(np.mean(xs))

        # Select action
        return self._select_action()

    def _select_action(self) -> int:
        """Select action using discovered physics."""
        if len(self.ball_history) < 2:
            return 1  # Stay

        # Compute current state
        curr = self.ball_history[-1]
        prev = self.ball_history[-2]
        vx = curr[0] - prev[0]
        vy = curr[1] - prev[1]

        if vy <= 0:
            return 1  # Ball moving up - stay

        # Try to predict using physics
        predicted_x = self.physics.predict_intercept(curr[0], curr[1], vx, vy)

        if predicted_x is not None:
            self.predictions_using_physics += 1
            target = predicted_x
        else:
            # Fallback: move toward ball's current x
            self.predictions_using_fallback += 1
            target = curr[0]

        # Move toward target
        diff = target - self.paddle_x

        if diff < -4:
            return 0  # Left
        elif diff > 4:
            return 2  # Right
        return 1  # Stay

    def get_statistics(self) -> Dict:
        """Get stats."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(1, self.hits + self.misses),
            'physics_predictions': self.predictions_using_physics,
            'fallback_predictions': self.predictions_using_fallback,
            'rules_discovered': len(self.physics.rules),
        }


def test_physics_discovery():
    """Test physics discovery agent."""
    print("=" * 60)
    print("PHYSICS DISCOVERY SIEVE: Learning Pong Physics")
    print("=" * 60)

    agent = PhysicsDiscoveryAgent()

    n_episodes = 100
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

        if (episode + 1) % 20 == 0:
            stats = agent.get_statistics()
            recent = episode_stats[-20:]
            recent_hits = sum(e['hits'] for e in recent)
            recent_misses = sum(e['misses'] for e in recent)
            hit_rate = recent_hits / max(1, recent_hits + recent_misses)

            print(f"\nEpisode {episode + 1}:")
            print(f"  Recent hit rate: {hit_rate:.1%}")
            print(f"  Rules discovered: {stats['rules_discovered']}")
            print(f"  Physics predictions: {stats['physics_predictions']}")
            print(f"  Fallback predictions: {stats['fallback_predictions']}")

            # Show discovered rules
            rules = agent.physics.get_discovered_rules()
            if rules:
                print(f"  Discovered physics:")
                for rule in rules:
                    print(f"    - {rule.name}: {rule.description} (conf={rule.confidence:.2f})")

    # Final
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    stats = agent.get_statistics()
    print(f"Overall hit rate: {stats['hit_rate']:.1%}")
    print(f"Rules discovered: {stats['rules_discovered']}")

    # Learning curve
    first_quarter = episode_stats[:25]
    last_quarter = episode_stats[-25:]
    first_rate = sum(e['hits'] for e in first_quarter) / max(1, sum(e['hits']+e['misses'] for e in first_quarter))
    last_rate = sum(e['hits'] for e in last_quarter) / max(1, sum(e['hits']+e['misses'] for e in last_quarter))

    print(f"\nLearning curve:")
    print(f"  First 25 episodes: {first_rate:.1%}")
    print(f"  Last 25 episodes: {last_rate:.1%}")
    print(f"  Improvement: {(last_rate - first_rate)*100:+.1f}%")

    # Final discovered physics
    print("\nFinal Discovered Physics:")
    for rule in agent.physics.get_discovered_rules():
        print(f"  {rule.name}: {rule.description}")
        print(f"    Confidence: {rule.confidence:.2f}")

    return agent


if __name__ == "__main__":
    agent = test_physics_discovery()
