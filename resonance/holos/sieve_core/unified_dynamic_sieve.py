"""
UNIFIED DYNAMIC SIEVE WITH PROPER DIMENSION DISCOVERY
======================================================

Addressing the key questions:
1. Does a dimension model the agent's perspective?
2. Do dimensions capture geometry and velocity?
3. What prevents/enables dimension growth?
4. How do we merge duplicate patterns?

Key insight: Dimensions should emerge from FUNCTIONAL VARIANCE,
not just phase disagreement. A dimension is useful if patterns
vary PREDICTABLY along it.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass, field
import cmath


@dataclass
class UnifiedConfig:
    """Configuration for unified sieve."""
    base_damping: float = 0.05
    coupling_strength: float = 0.2
    survival_threshold: float = 0.05

    # Dimension management
    max_dimensions: int = 10
    dimension_utility_threshold: float = 0.1  # Min variance explained
    dimension_merge_threshold: float = 0.9    # Correlation for merging

    # Pattern deduplication
    pattern_similarity_threshold: float = 0.95

    max_tokens: int = 100


class DimensionTracker:
    """
    Tracks dimension utility - does this dimension explain variance?

    A dimension is useful if:
    - Patterns vary predictably along it
    - It explains variance that other dimensions don't
    - It's not redundant with another dimension
    """

    def __init__(self, name: str):
        self.name = name
        self.variance_explained: List[float] = []
        self.correlation_with_others: Dict[str, float] = {}

    def record_variance(self, explained: float):
        self.variance_explained.append(explained)
        if len(self.variance_explained) > 100:
            self.variance_explained = self.variance_explained[-100:]

    def get_utility(self) -> float:
        if not self.variance_explained:
            return 0.0
        return np.mean(self.variance_explained[-20:]) if len(self.variance_explained) >= 20 else np.mean(self.variance_explained)


class UnifiedToken:
    """A token with rich dimensional embedding."""

    def __init__(self, name: str):
        self.name = name
        self.amplitude: complex = complex(1, 0)
        self.coordinates: Dict[str, float] = {}
        self.couplings: Dict[str, float] = {}

        # Track what this token represents
        self.token_type: str = "unknown"  # observation, relationship, action, etc.
        self.activation_history: List[float] = []

    def similarity_to(self, other: 'UnifiedToken') -> float:
        """Compute similarity to another token."""
        # Phase similarity
        phase_sim = np.cos(cmath.phase(self.amplitude) - cmath.phase(other.amplitude))

        # Coordinate similarity (in shared dimensions)
        shared_dims = set(self.coordinates.keys()) & set(other.coordinates.keys())
        if shared_dims:
            coord_dist = np.sqrt(sum((self.coordinates[d] - other.coordinates[d])**2
                                    for d in shared_dims) / len(shared_dims))
            coord_sim = np.exp(-coord_dist * 2)
        else:
            coord_sim = 0.5  # Unknown

        # Name similarity (crude but useful)
        name_sim = 1.0 if self.name == other.name else 0.0

        return (phase_sim + coord_sim) / 2


class UnifiedSieve:
    """
    A sieve that properly discovers and manages dimensions.

    Key improvements:
    1. Dimensions have utility scores - useless ones die
    2. Duplicate patterns are merged
    3. Action space is a proper dimension
    4. Geometry emerges from relational features
    """

    def __init__(self, name: str = "root", config: Optional[UnifiedConfig] = None):
        self.name = name
        self.config = config or UnifiedConfig()

        self.tokens: Dict[str, UnifiedToken] = {}
        self.dimensions: Dict[str, DimensionTracker] = {}

        # Pre-define some semantic dimensions
        self._init_semantic_dimensions()

        self.entropy_history: List[float] = []
        self.damping = self.config.base_damping

    def _init_semantic_dimensions(self):
        """Initialize dimensions with semantic meaning."""
        # These dimensions have meaning we understand
        semantic_dims = [
            'time',           # When (sequence position)
            'action',         # What can be done (agent's perspective)
            'spatial_x',      # Where (horizontal)
            'spatial_y',      # Where (vertical)
            'velocity',       # How fast (rate of change)
            'abstraction',    # Level of detail
        ]

        for dim in semantic_dims:
            self.dimensions[dim] = DimensionTracker(dim)

    def _get_or_create_token(self, name: str) -> UnifiedToken:
        if name not in self.tokens:
            # Check for near-duplicates first
            for existing_name, existing in self.tokens.items():
                if self._names_are_similar(name, existing_name):
                    # Merge into existing
                    return existing

            if len(self.tokens) >= self.config.max_tokens:
                self._prune_weakest_token()

            self.tokens[name] = UnifiedToken(name)

        return self.tokens[name]

    def _names_are_similar(self, name1: str, name2: str) -> bool:
        """Check if two token names refer to same concept."""
        # Same base with different suffixes
        if name1.replace('_t0', '').replace('_t1', '') == name2.replace('_t0', '').replace('_t1', ''):
            return False  # These are intentionally different (before/after)

        # Check for relationship symmetry: rel_A_B == rel_B_A
        if name1.startswith('rel_') and name2.startswith('rel_'):
            parts1 = name1[4:].split('_')
            parts2 = name2[4:].split('_')
            if len(parts1) >= 2 and len(parts2) >= 2:
                if set(parts1[:2]) == set(parts2[:2]):
                    return True

        return False

    def _prune_weakest_token(self):
        """Remove the weakest token."""
        if not self.tokens:
            return
        weakest = min(self.tokens.values(), key=lambda t: abs(t.amplitude))
        del self.tokens[weakest.name]

    def inject_observation(self, name: str, value: float,
                          spatial_x: Optional[float] = None,
                          spatial_y: Optional[float] = None,
                          time: Optional[float] = None):
        """
        Inject an observation with proper dimensional coordinates.
        """
        token = self._get_or_create_token(name)
        token.token_type = "observation"

        # Bipolar phase encoding
        centered = np.clip(2 * value - 1, -1, 1)
        phase = centered * np.pi
        injection = complex(np.cos(phase), np.sin(phase))
        token.amplitude = token.amplitude * 0.9 + injection * 0.1

        # Set coordinates
        if time is not None:
            token.coordinates['time'] = time
        if spatial_x is not None:
            token.coordinates['spatial_x'] = spatial_x
        if spatial_y is not None:
            token.coordinates['spatial_y'] = spatial_y

    def inject_action(self, action_id: int, n_actions: int = 3):
        """
        Inject an action as a point in action-space dimension.

        The action dimension represents "what the agent can do".
        """
        token = self._get_or_create_token(f"action_{action_id}")
        token.token_type = "action"
        token.amplitude = complex(1, 0)

        # Position in action space (normalized)
        token.coordinates['action'] = action_id / max(1, n_actions - 1)

    def inject_velocity(self, name: str, velocity: float, direction: str = 'x'):
        """
        Inject velocity as a dimensional coordinate, not just a value.

        This is key: velocity isn't just data, it's an AXIS along which
        other phenomena vary.
        """
        token = self._get_or_create_token(f"{name}_velocity_{direction}")
        token.token_type = "velocity"

        # Velocity magnitude as amplitude, direction as phase
        magnitude = abs(velocity)
        sign = np.sign(velocity)

        token.amplitude = complex(magnitude * 0.5 + 0.5, sign * 0.5)

        # Position in velocity dimension (normalized to [-1, 1] -> [0, 1])
        normalized_vel = (velocity + 5) / 10  # Assume velocity in [-5, 5]
        token.coordinates['velocity'] = np.clip(normalized_vel, 0, 1)

    def inject_geometric_relation(self, name1: str, name2: str,
                                  value1: float, value2: float,
                                  relation_type: str = 'difference'):
        """
        Inject a geometric relationship - the RELATIVE position matters.

        This is how geometry emerges: not from absolute positions,
        but from relationships between positions.
        """
        if relation_type == 'difference':
            rel_value = value1 - value2  # Can be negative
        elif relation_type == 'distance':
            rel_value = abs(value1 - value2)  # Always positive
        elif relation_type == 'ratio':
            rel_value = value1 / max(0.01, value2)
        else:
            rel_value = value1 - value2

        rel_name = f"geo_{name1}_{relation_type}_{name2}"
        token = self._get_or_create_token(rel_name)
        token.token_type = "geometry"

        # Encode relationship
        normalized = np.clip((rel_value + 1) / 2, 0, 1)  # Normalize to [0,1]
        centered = 2 * normalized - 1
        phase = centered * np.pi
        token.amplitude = complex(np.cos(phase), np.sin(phase))

    def inject_causal_link(self, cause: str, effect: str,
                          strength: float, direction: float):
        """
        Inject a causal relationship with direction.

        direction: -1 (inverse) to +1 (direct) relationship
        """
        rel_name = f"cause_{cause}_to_{effect}"
        token = self._get_or_create_token(rel_name)
        token.token_type = "causal"

        phase = direction * np.pi
        token.amplitude = complex(strength * np.cos(phase), strength * np.sin(phase))

        # Causal links exist at a higher abstraction level
        token.coordinates['abstraction'] = 0.7

    def evolve(self):
        """Evolve the sieve with dimension utility tracking."""
        if not self.tokens:
            return

        # 1. Compute interference and update amplitudes
        self._compute_interference()

        # 2. Update dimension utility scores
        self._update_dimension_utility()

        # 3. Merge duplicate patterns
        self._merge_duplicates()

        # 4. Prune dead dimensions
        self._prune_useless_dimensions()

        # 5. Check if new dimensions needed
        self._check_dimension_birth()

        # 6. Self-tune
        self._self_tune()

    def _compute_interference(self):
        """Compute interference between tokens."""
        new_amplitudes: Dict[str, complex] = {}
        token_list = list(self.tokens.values())

        for token in token_list:
            # Self with damping
            new_amp = token.amplitude * (1 - self.damping)

            # Interference from coupled tokens
            for other_name, coupling in token.couplings.items():
                if other_name in self.tokens:
                    other = self.tokens[other_name]
                    transfer = coupling * other.amplitude * self.config.coupling_strength
                    new_amp += transfer / max(1, len(token.couplings))

            # Interference from nearby tokens (in coordinate space)
            for other in token_list:
                if other.name == token.name:
                    continue

                # Distance in shared dimensions
                shared = set(token.coordinates.keys()) & set(other.coordinates.keys())
                if not shared:
                    continue

                dist = np.sqrt(sum((token.coordinates[d] - other.coordinates[d])**2
                                  for d in shared) / len(shared))

                if dist < 0.2:  # Close enough to interfere
                    interference = np.exp(-dist * 5) * other.amplitude * 0.05
                    new_amp += interference

            new_amplitudes[token.name] = new_amp

        # Apply and normalize
        total = sum(abs(a) for a in new_amplitudes.values())
        if total > 10:
            scale = 10 / total
            new_amplitudes = {k: v * scale for k, v in new_amplitudes.items()}

        for name, amp in new_amplitudes.items():
            if name in self.tokens:
                self.tokens[name].amplitude = amp

        # Remove dead tokens
        dead = [n for n, t in self.tokens.items()
                if abs(t.amplitude) < self.config.survival_threshold]
        for n in dead:
            del self.tokens[n]

    def _update_dimension_utility(self):
        """Track how much variance each dimension explains."""
        for dim_name, dim_tracker in self.dimensions.items():
            # Get tokens that have this dimension
            tokens_with_dim = [t for t in self.tokens.values()
                              if dim_name in t.coordinates]

            if len(tokens_with_dim) < 3:
                dim_tracker.record_variance(0)
                continue

            # Compute variance in amplitude along this dimension
            coords = [t.coordinates[dim_name] for t in tokens_with_dim]
            amps = [abs(t.amplitude) for t in tokens_with_dim]

            # Does amplitude vary predictably with coordinate?
            if np.std(coords) > 0.01 and np.std(amps) > 0.01:
                correlation = abs(np.corrcoef(coords, amps)[0, 1])
                dim_tracker.record_variance(correlation if not np.isnan(correlation) else 0)
            else:
                dim_tracker.record_variance(0)

    def _merge_duplicates(self):
        """Merge tokens that are too similar."""
        token_list = list(self.tokens.values())
        merged = set()

        for i, t1 in enumerate(token_list):
            if t1.name in merged:
                continue

            for t2 in token_list[i+1:]:
                if t2.name in merged:
                    continue

                sim = t1.similarity_to(t2)
                if sim > self.config.pattern_similarity_threshold:
                    # Merge t2 into t1
                    t1.amplitude = (t1.amplitude + t2.amplitude) / 2
                    t1.couplings.update(t2.couplings)
                    merged.add(t2.name)

        # Remove merged tokens
        for name in merged:
            if name in self.tokens:
                del self.tokens[name]

    def _prune_useless_dimensions(self):
        """Remove dimensions that don't explain variance."""
        useless = []
        for dim_name, tracker in self.dimensions.items():
            if tracker.get_utility() < self.config.dimension_utility_threshold:
                # Don't remove semantic dimensions too quickly
                if dim_name in ['time', 'action', 'spatial_x', 'spatial_y', 'velocity', 'abstraction']:
                    continue
                useless.append(dim_name)

        for dim in useless:
            del self.dimensions[dim]
            # Remove coordinate from all tokens
            for token in self.tokens.values():
                if dim in token.coordinates:
                    del token.coordinates[dim]

    def _check_dimension_birth(self):
        """Check if new dimension needed based on unexplained variance."""
        if len(self.dimensions) >= self.config.max_dimensions:
            return

        # Look for tokens with high amplitude but no clear dimensional structure
        orphan_tokens = [t for t in self.tokens.values()
                        if len(t.coordinates) < 2 and abs(t.amplitude) > 0.3]

        if len(orphan_tokens) > 5:
            # These tokens need a dimension
            new_dim = f"emergent_{len(self.dimensions)}"
            self.dimensions[new_dim] = DimensionTracker(new_dim)

            # Assign coordinates based on amplitude phase
            for token in orphan_tokens:
                phase = cmath.phase(token.amplitude)
                token.coordinates[new_dim] = (phase + np.pi) / (2 * np.pi)

    def _self_tune(self):
        """Self-tune to maintain criticality."""
        amplitudes = [abs(t.amplitude) for t in self.tokens.values()
                     if abs(t.amplitude) > 0.01]
        if not amplitudes:
            return

        total = sum(amplitudes)
        probs = [a / total for a in amplitudes]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        self.entropy_history.append(normalized)

        if len(self.entropy_history) < 10:
            return

        recent = self.entropy_history[-10:]
        trend = recent[-1] - recent[0]

        if trend < -0.05:
            self.damping = min(0.3, self.damping * 1.05)
        elif trend > 0.05:
            self.damping = max(0.01, self.damping * 0.95)

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        dim_utilities = {name: tracker.get_utility()
                        for name, tracker in self.dimensions.items()}

        token_types = defaultdict(int)
        for t in self.tokens.values():
            token_types[t.token_type] += 1

        return {
            'n_tokens': len(self.tokens),
            'n_dimensions': len(self.dimensions),
            'dimensions': list(self.dimensions.keys()),
            'dimension_utilities': dim_utilities,
            'token_types': dict(token_types),
            'damping': self.damping,
            'entropy': self.entropy_history[-1] if self.entropy_history else 0,
        }

    def print_state(self):
        """Print detailed state."""
        stats = self.get_statistics()

        print(f"\nSieve '{self.name}':")
        print(f"  Tokens: {stats['n_tokens']}, Damping: {stats['damping']:.3f}")
        print(f"  Token types: {stats['token_types']}")

        print(f"\n  Dimensions ({stats['n_dimensions']}):")
        for dim, utility in stats['dimension_utilities'].items():
            print(f"    {dim}: utility={utility:.3f}")

        print(f"\n  Top patterns:")
        patterns = [(t.name, abs(t.amplitude), cmath.phase(t.amplitude))
                   for t in self.tokens.values()]
        patterns.sort(key=lambda x: -x[1])

        for name, amp, phase in patterns[:10]:
            phase_deg = phase * 180 / np.pi
            print(f"    {name}: amp={amp:.3f}, phase={phase_deg:.1f}deg")


# =============================================================================
# TEST: Proper dimension discovery
# =============================================================================

def test_unified_sieve():
    """Test unified sieve with proper dimension discovery."""
    print("=" * 70)
    print("UNIFIED DYNAMIC SIEVE: Proper Dimension Discovery")
    print("=" * 70)

    sieve = UnifiedSieve()

    # Simulate Pong with proper dimensional injection
    ball_x, ball_y = 42, 20
    ball_dx, ball_dy = 2, 1.5
    paddle_x = 42.0

    for frame in range(500):
        t = frame / 500  # Normalized time

        # Inject observations with spatial coordinates
        sieve.inject_observation('ball_x', ball_x / 84, spatial_x=ball_x / 84, time=t)
        sieve.inject_observation('ball_y', ball_y / 84, spatial_y=ball_y / 84, time=t)
        sieve.inject_observation('paddle_x', paddle_x / 84, spatial_x=paddle_x / 84, time=t)

        # Inject velocities as dimensional coordinates
        sieve.inject_velocity('ball', ball_dx, 'x')
        sieve.inject_velocity('ball', ball_dy, 'y')

        # Inject geometric relation: ball-paddle distance (agent's perspective!)
        sieve.inject_geometric_relation('ball_x', 'paddle_x',
                                        ball_x / 84, paddle_x / 84, 'difference')

        # Inject action
        action = np.random.randint(0, 3)
        sieve.inject_action(action, n_actions=3)

        # Inject causal links based on what happens
        if ball_x <= 4 or ball_x >= 80:
            # Wall bounce: velocity causes position change
            sieve.inject_causal_link('ball_velocity_x', 'wall_bounce', 0.8, -1.0)
            ball_dx *= -1

        if ball_y >= 76:
            # Near paddle: relative position matters
            distance = abs(ball_x - paddle_x)
            if distance < 12:
                sieve.inject_causal_link('geo_ball_x_difference_paddle_x', 'hit', 0.9, 1.0)
            else:
                sieve.inject_causal_link('geo_ball_x_difference_paddle_x', 'miss', 0.9, -1.0)

        # Update physics
        ball_x += ball_dx
        ball_y += ball_dy

        if ball_y <= 4:
            ball_dy = abs(ball_dy)
        if ball_y >= 78:
            ball_y = 20
            ball_x = np.random.uniform(20, 64)
            ball_dx = np.random.choice([-2, 2])

        # Move paddle
        if action == 0:
            paddle_x = max(10, paddle_x - 3)
        elif action == 2:
            paddle_x = min(74, paddle_x + 3)

        # Evolve
        sieve.evolve()

        if (frame + 1) % 100 == 0:
            print(f"\n--- Frame {frame + 1} ---")
            sieve.print_state()

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS: What Dimensions Emerged?")
    print("=" * 70)

    stats = sieve.get_statistics()

    print("\nDimension Utility Analysis:")
    print("(Higher utility = dimension explains more variance)")
    for dim, utility in sorted(stats['dimension_utilities'].items(),
                               key=lambda x: -x[1]):
        bar = '#' * int(utility * 20)
        print(f"  {dim:15s}: {utility:.3f} {bar}")

    print("\nToken Type Distribution:")
    for ttype, count in stats['token_types'].items():
        print(f"  {ttype}: {count}")

    print("\nKey Questions Answered:")
    print(f"  1. Agent perspective (action dimension): utility={stats['dimension_utilities'].get('action', 0):.3f}")
    print(f"  2. Geometry (spatial dimensions): x={stats['dimension_utilities'].get('spatial_x', 0):.3f}, y={stats['dimension_utilities'].get('spatial_y', 0):.3f}")
    print(f"  3. Velocity dimension: utility={stats['dimension_utilities'].get('velocity', 0):.3f}")

    return sieve


if __name__ == "__main__":
    sieve = test_unified_sieve()
