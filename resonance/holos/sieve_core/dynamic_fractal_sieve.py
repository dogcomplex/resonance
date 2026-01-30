"""
DYNAMIC N-DIMENSIONAL FRACTAL SIEVE
====================================

The key insight: don't predefine dimensions. Let them EMERGE.

A dimension is born when:
- Two patterns that SHOULD be distinguished can't be
- There's unexplained variance in predictions
- A new type of relationship is discovered

A dimension dies when:
- It carries no distinguishing information
- All patterns along it are equivalent

The sieve grows the geometry it needs to capture the structure it finds.

Core principle: TOKENS ARE SIEVES. The whole thing is recursive.
At any "level" you can:
- Zoom in: see the token as a sieve with its own tokens
- Zoom out: see the sieve as a token in a larger sieve
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from collections import defaultdict
from dataclasses import dataclass, field
import cmath
import hashlib


@dataclass
class DynamicConfig:
    """Configuration for dynamic sieve."""
    base_damping: float = 0.05
    coupling_strength: float = 0.2
    survival_threshold: float = 0.1
    dimension_birth_threshold: float = 0.3  # Unexplained variance triggers new dim
    dimension_death_threshold: float = 0.01  # Useless dims die
    max_dimensions: int = 20
    max_tokens_per_sieve: int = 100


class DynamicToken:
    """
    A token that is also a sieve.

    Key properties:
    - Has an amplitude (how "present" it is in parent sieve)
    - Has a phase (encodes its "value" or "type")
    - Contains sub-tokens (it's a sieve itself)
    - Has connections to other tokens (couplings)

    The token's "meaning" emerges from:
    - Its sub-structure (what it contains)
    - Its context (where it appears)
    - Its dynamics (how it evolves)
    """

    def __init__(self, name: str, parent: Optional['DynamicSieve'] = None):
        self.name = name
        self.parent = parent

        # State in parent's field
        self.amplitude: complex = complex(1, 0)

        # This token IS a sieve
        self.sub_sieve: Optional['DynamicSieve'] = None

        # Connections to other tokens (in same sieve)
        self.couplings: Dict[str, float] = {}

        # History for learning
        self.activation_history: List[float] = []

        # Dimensional coordinates (position in N-dim space)
        # Grows dynamically as dimensions are added
        self.coordinates: Dict[str, float] = {}

    def get_magnitude(self) -> float:
        return abs(self.amplitude)

    def get_phase(self) -> float:
        return cmath.phase(self.amplitude)

    def expand_to_sieve(self, config: DynamicConfig) -> 'DynamicSieve':
        """Expand this token into a full sieve."""
        if self.sub_sieve is None:
            self.sub_sieve = DynamicSieve(
                name=f"{self.name}_sub",
                parent_token=self,
                config=config
            )
        return self.sub_sieve

    def __repr__(self):
        return f"Token({self.name}, amp={self.get_magnitude():.3f}, phase={self.get_phase():.2f})"


class DynamicSieve:
    """
    An N-dimensional sieve that grows its geometry dynamically.

    Dimensions represent different "axes of variation":
    - Time (sequence position)
    - Space (location in observation)
    - Abstraction (level of detail)
    - Modality (type of information)
    - Hypothesis (competing interpretations)
    - ... any new axis the data requires

    New dimensions are born when existing ones can't explain the variance.
    """

    def __init__(self, name: str = "root",
                 parent_token: Optional[DynamicToken] = None,
                 config: Optional[DynamicConfig] = None):
        self.name = name
        self.parent_token = parent_token
        self.config = config or DynamicConfig()

        # Tokens in this sieve
        self.tokens: Dict[str, DynamicToken] = {}

        # Dimensions of this sieve's space
        # Each dimension has a name and range
        self.dimensions: Dict[str, Tuple[float, float]] = {}

        # The holographic field: maps coordinate tuples to amplitudes
        # This is sparse - only non-zero regions stored
        self.field: Dict[Tuple, complex] = defaultdict(complex)

        # Prediction errors (for dimension discovery)
        self.prediction_errors: List[float] = []

        # Statistics for self-tuning
        self.entropy_history: List[float] = []
        self.damping = self.config.base_damping

    def _get_or_create_token(self, name: str) -> DynamicToken:
        """Get existing token or create new one."""
        if name not in self.tokens:
            if len(self.tokens) >= self.config.max_tokens_per_sieve:
                # Prune weakest token
                weakest = min(self.tokens.values(), key=lambda t: t.get_magnitude())
                del self.tokens[weakest.name]

            self.tokens[name] = DynamicToken(name, parent=self)
        return self.tokens[name]

    def _ensure_dimension(self, dim_name: str, value: float):
        """Ensure dimension exists, expand if needed."""
        if dim_name not in self.dimensions:
            if len(self.dimensions) >= self.config.max_dimensions:
                return  # Can't add more
            self.dimensions[dim_name] = (value, value)  # Initial range
        else:
            low, high = self.dimensions[dim_name]
            self.dimensions[dim_name] = (min(low, value), max(high, value))

    def inject(self, name: str, value: float, coordinates: Optional[Dict[str, float]] = None):
        """
        Inject a value into the sieve.

        name: token name
        value: value to encode (will be mapped to phase)
        coordinates: position in N-dim space (optional, auto-assigned if missing)
        """
        token = self._get_or_create_token(name)

        # Bipolar phase encoding
        centered = 2 * value - 1 if 0 <= value <= 1 else value  # Handle both cases
        centered = np.clip(centered, -1, 1)
        phase = centered * np.pi
        injection = complex(np.cos(phase), np.sin(phase))

        # Update token amplitude
        token.amplitude = token.amplitude * 0.9 + injection * 0.1  # Smooth update
        token.activation_history.append(abs(injection))

        # Update coordinates (only numeric values)
        if coordinates:
            for dim, coord in coordinates.items():
                if isinstance(coord, (int, float)):
                    self._ensure_dimension(dim, coord)
                    token.coordinates[dim] = coord

    def inject_relationship(self, token1: str, token2: str, strength: float,
                            direction: Optional[float] = None):
        """
        Inject a relationship (coupling) between two tokens.

        Relationships are first-class: they become tokens themselves!
        This is how we discover structure.

        strength: magnitude of relationship
        direction: sign/direction of relationship (-1 to 1)
                   None = symmetric relationship
        """
        t1 = self._get_or_create_token(token1)
        t2 = self._get_or_create_token(token2)

        # Create coupling
        t1.couplings[token2] = strength
        t2.couplings[token1] = strength

        # The relationship itself becomes a token
        rel_name = f"rel_{token1}_{token2}"
        rel_token = self._get_or_create_token(rel_name)

        # Encode direction in phase if provided
        if direction is not None:
            phase = direction * np.pi  # -1 -> -pi, +1 -> +pi
            rel_token.amplitude = complex(strength * np.cos(phase), strength * np.sin(phase))
        else:
            rel_token.amplitude = complex(strength, 0)

    def inject_sequence(self, tokens: List[Tuple[str, float]], dim_name: str = "time"):
        """
        Inject a sequence of observations.

        Automatically assigns positions along the given dimension.
        """
        for i, (name, value) in enumerate(tokens):
            self.inject(name, value, {dim_name: i / max(1, len(tokens) - 1)})

    def compute_interference(self):
        """
        Compute interference between tokens based on their coordinates.

        Tokens that are "close" in the N-dim space interfere more strongly.
        This is how patterns emerge - coherent structures reinforce each other.
        """
        token_list = list(self.tokens.values())

        for i, t1 in enumerate(token_list):
            for t2 in token_list[i+1:]:
                # Compute distance in shared dimensions
                shared_dims = set(t1.coordinates.keys()) & set(t2.coordinates.keys())

                if shared_dims:
                    # Euclidean distance in shared space
                    dist_sq = sum((t1.coordinates[d] - t2.coordinates[d])**2
                                  for d in shared_dims)
                    distance = np.sqrt(dist_sq)

                    # Closer tokens interfere more strongly
                    interference_strength = np.exp(-distance * 2)

                    # Phase similarity determines constructive/destructive
                    phase_diff = t1.get_phase() - t2.get_phase()
                    interference = interference_strength * np.cos(phase_diff)

                    # Update couplings based on interference
                    old_coupling = t1.couplings.get(t2.name, 0)
                    t1.couplings[t2.name] = old_coupling * 0.9 + interference * 0.1
                    t2.couplings[t1.name] = t1.couplings[t2.name]

    def evolve(self):
        """
        One evolution step.

        1. Compute interference between tokens
        2. Update amplitudes based on couplings
        3. Decay weak tokens
        4. Check for new dimensions needed
        5. Self-tune parameters
        """
        if not self.tokens:
            return

        # 1. Interference
        self.compute_interference()

        # 2. Update amplitudes
        new_amplitudes: Dict[str, complex] = {}

        for name, token in self.tokens.items():
            # Self-contribution with damping
            new_amp = token.amplitude * (1 - self.damping)

            # Coupling contributions
            for other_name, coupling in token.couplings.items():
                if other_name in self.tokens:
                    other = self.tokens[other_name]
                    # Transfer proportional to coupling strength
                    transfer = coupling * other.amplitude * self.config.coupling_strength
                    new_amp += transfer / max(1, len(token.couplings))

            new_amplitudes[name] = new_amp

        # Apply new amplitudes
        for name, amp in new_amplitudes.items():
            self.tokens[name].amplitude = amp

        # 3. Normalize and decay
        total = sum(abs(t.amplitude) for t in self.tokens.values())
        if total > 10:
            scale = 10 / total
            for t in self.tokens.values():
                t.amplitude *= scale

        # Remove dead tokens
        dead = [n for n, t in self.tokens.items()
                if abs(t.amplitude) < self.config.survival_threshold * 0.1]
        for n in dead:
            del self.tokens[n]

        # 4. Check for new dimensions
        self._check_dimension_birth()

        # 5. Self-tune
        self._self_tune()

        # 6. Recursively evolve sub-sieves
        for token in self.tokens.values():
            if token.sub_sieve is not None:
                token.sub_sieve.evolve()

    def _check_dimension_birth(self):
        """
        Check if we need a new dimension.

        A new dimension is needed when:
        - Many token pairs have similar coordinates but different phases
        - This indicates there's a hidden variable we're not tracking

        Key insight: be CONSERVATIVE about spawning dimensions.
        Only spawn when there's SYSTEMATIC unexplained variance.
        """
        if len(self.dimensions) >= self.config.max_dimensions:
            return

        # Count how many token pairs need separation
        conflicting_pairs = []
        token_list = list(self.tokens.values())

        for i, t1 in enumerate(token_list):
            for t2 in token_list[i+1:]:
                shared = set(t1.coordinates.keys()) & set(t2.coordinates.keys())
                if len(shared) < 2:  # Need at least 2 shared dimensions
                    continue

                coord_dist = np.sqrt(sum((t1.coordinates.get(d, 0) - t2.coordinates.get(d, 0))**2
                                        for d in shared) / len(shared))

                phase_dist = abs(t1.get_phase() - t2.get_phase())

                # Very close coordinates but very different phases
                if coord_dist < 0.05 and phase_dist > 2.0:
                    conflicting_pairs.append((t1, t2))

        # Only spawn dimension if MANY pairs conflict (systematic issue)
        threshold = max(3, len(token_list) * 0.1)  # At least 3 or 10% of tokens

        if len(conflicting_pairs) >= threshold:
            new_dim = f"discovered_{len(self.dimensions)}"
            self._ensure_dimension(new_dim, 0)

            # Separate conflicting pairs
            for t1, t2 in conflicting_pairs[:5]:  # Only first few
                t1.coordinates[new_dim] = 0.0
                t2.coordinates[new_dim] = 1.0

    def _self_tune(self):
        """Self-tune damping to maintain criticality."""
        # Compute entropy
        amplitudes = [abs(t.amplitude) for t in self.tokens.values() if abs(t.amplitude) > 0.01]
        if not amplitudes:
            return

        total = sum(amplitudes)
        probs = [a / total for a in amplitudes]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        self.entropy_history.append(normalized_entropy)

        if len(self.entropy_history) < 10:
            return

        # Adjust damping based on entropy trend
        recent = self.entropy_history[-10:]
        trend = recent[-1] - recent[0]

        if trend < -0.05:  # Getting too ordered
            self.damping = min(0.3, self.damping * 1.05)
        elif trend > 0.05:  # Getting too chaotic
            self.damping = max(0.01, self.damping * 0.95)

    def predict(self, partial_state: Dict[str, float]) -> Dict[str, float]:
        """
        Predict missing values given partial state.

        This is where the sieve becomes useful - it fills in gaps
        based on learned patterns.
        """
        predictions = {}

        # Find tokens that match partial state
        matching_tokens = []
        for token in self.tokens.values():
            match_score = 0
            for name, value in partial_state.items():
                if name in token.name:  # Rough matching
                    match_score += 1
            if match_score > 0:
                matching_tokens.append((token, match_score))

        if not matching_tokens:
            return predictions

        # Use coupled tokens to predict others
        for token, _ in matching_tokens:
            for coupled_name, coupling in token.couplings.items():
                if coupled_name not in partial_state and coupling > 0.1:
                    coupled_token = self.tokens.get(coupled_name)
                    if coupled_token:
                        # Predict based on coupling and phase
                        predicted_phase = coupled_token.get_phase()
                        predicted_value = (predicted_phase / np.pi + 1) / 2
                        predictions[coupled_name] = predicted_value * abs(coupling)

        return predictions

    def get_stable_patterns(self, top_n: int = 10) -> List[Tuple[str, float, float]]:
        """Get most stable patterns (highest amplitude tokens)."""
        patterns = [(t.name, t.get_magnitude(), t.get_phase())
                    for t in self.tokens.values()]
        patterns.sort(key=lambda x: -x[1])
        return patterns[:top_n]

    def get_dimensions(self) -> List[str]:
        """Get current dimensions."""
        return list(self.dimensions.keys())

    def get_statistics(self) -> Dict:
        """Get sieve statistics."""
        return {
            'n_tokens': len(self.tokens),
            'n_dimensions': len(self.dimensions),
            'dimensions': list(self.dimensions.keys()),
            'damping': self.damping,
            'entropy': self.entropy_history[-1] if self.entropy_history else 0,
            'total_amplitude': sum(abs(t.amplitude) for t in self.tokens.values()),
        }

    def print_state(self, indent: int = 0):
        """Print current state."""
        prefix = "  " * indent
        stats = self.get_statistics()

        print(f"{prefix}Sieve '{self.name}':")
        print(f"{prefix}  Dimensions: {stats['dimensions']}")
        print(f"{prefix}  Tokens: {stats['n_tokens']}, Damping: {stats['damping']:.3f}")

        patterns = self.get_stable_patterns(5)
        for name, amp, phase in patterns:
            phase_deg = phase * 180 / np.pi
            print(f"{prefix}    {name}: amp={amp:.3f}, phase={phase_deg:.1f}deg")

        # Recursively print sub-sieves
        for token in self.tokens.values():
            if token.sub_sieve is not None:
                print(f"{prefix}  Sub-sieve of '{token.name}':")
                token.sub_sieve.print_state(indent + 2)


class FractalUniverse:
    """
    A complete fractal sieve universe.

    This is the top-level container that:
    - Receives observations
    - Routes them to appropriate depth
    - Discovers structure at all scales
    - Generates predictions/actions
    """

    def __init__(self, config: Optional[DynamicConfig] = None):
        self.config = config or DynamicConfig()
        self.root = DynamicSieve("universe", config=self.config)

        # Track observations for learning
        self.observation_count = 0
        self.prediction_accuracy: List[float] = []

    def observe(self, observation: Dict[str, float],
                context: Optional[Dict[str, float]] = None):
        """
        Observe a state (e.g., one game frame).

        observation: dict of feature_name -> value
        context: additional context (e.g., time step, action taken)
        """
        self.observation_count += 1

        # Inject all observations into root sieve
        for name, value in observation.items():
            coords = {'time': self.observation_count / 1000}  # Normalized time
            if context:
                coords.update(context)
            self.root.inject(name, value, coords)

        # Inject relationships between observations
        items = list(observation.items())
        for i, (n1, v1) in enumerate(items):
            for n2, v2 in items[i+1:]:
                # Relationship strength based on co-occurrence
                correlation = 1 - abs(v1 - v2)  # Similar values = stronger
                self.root.inject_relationship(n1, n2, correlation)

    def observe_transition(self, before: Dict[str, float], after: Dict[str, float],
                           action: Optional[int] = None):
        """
        Observe a state transition.

        This is key for learning dynamics - what changes and why.
        """
        # Inject before state
        for name, value in before.items():
            self.root.inject(f"{name}_t0", value, {'time_offset': 0})

        # Inject after state
        for name, value in after.items():
            self.root.inject(f"{name}_t1", value, {'time_offset': 1})

        # Inject deltas (changes)
        for name in before:
            if name in after:
                delta = after[name] - before[name]
                self.root.inject(f"{name}_delta", (delta + 1) / 2)  # Normalize to [0,1]

        # Inject action if provided
        if action is not None:
            self.root.inject(f"action_{action}", 1.0)

        # Create DIRECTIONAL relationships: before -> delta -> after
        for name in before:
            if name in after:
                delta = after[name] - before[name]
                direction = np.sign(delta)  # -1, 0, or 1

                # before -> delta (direction shows if increasing/decreasing)
                self.root.inject_relationship(f"{name}_t0", f"{name}_delta", 0.8, direction)

                # delta -> after (positive: delta leads to after)
                self.root.inject_relationship(f"{name}_delta", f"{name}_t1", 0.8, direction)

                # Action relates to delta with direction
                if action is not None:
                    self.root.inject_relationship(f"action_{action}", f"{name}_delta", 0.5, direction)

    def evolve(self, steps: int = 1):
        """Evolve the universe for given steps."""
        for _ in range(steps):
            self.root.evolve()

    def predict(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """Predict next state given current."""
        # Inject current state
        for name, value in current_state.items():
            self.root.inject(f"{name}_now", value)

        # Use sieve to predict
        predictions = self.root.predict({f"{name}_now": v for name, v in current_state.items()})

        # Extract next state predictions
        next_state = {}
        for name, value in predictions.items():
            if '_t1' in name:
                base_name = name.replace('_t1', '')
                next_state[base_name] = value

        return next_state

    def get_discovered_structure(self) -> Dict:
        """Get the structure the sieve has discovered."""
        return {
            'dimensions': self.root.get_dimensions(),
            'n_tokens': len(self.root.tokens),
            'top_patterns': self.root.get_stable_patterns(20),
            'statistics': self.root.get_statistics(),
        }


# =============================================================================
# TEST: Fractal Sieve Learning Pong
# =============================================================================

def test_fractal_pong():
    """Test fractal sieve on Pong."""
    print("=" * 60)
    print("DYNAMIC FRACTAL SIEVE: Learning Pong Structure")
    print("=" * 60)

    universe = FractalUniverse()

    # Simulate Pong
    ball_x, ball_y = 42, 20
    ball_dx, ball_dy = 2, 1.5
    paddle_x = 42.0

    for frame in range(500):
        # Current state
        state = {
            'ball_x': ball_x / 84,
            'ball_y': ball_y / 84,
            'ball_dx': (ball_dx + 3) / 6,
            'ball_dy': (ball_dy + 3) / 6,
            'paddle_x': paddle_x / 84,
        }

        # Observe state
        universe.observe(state, {'frame': frame / 500})

        # Update physics
        old_state = state.copy()

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

        # Random paddle movement
        action = np.random.randint(0, 3)
        if action == 0:
            paddle_x = max(10, paddle_x - 3)
        elif action == 2:
            paddle_x = min(74, paddle_x + 3)

        # New state
        new_state = {
            'ball_x': ball_x / 84,
            'ball_y': ball_y / 84,
            'ball_dx': (ball_dx + 3) / 6,
            'ball_dy': (ball_dy + 3) / 6,
            'paddle_x': paddle_x / 84,
        }

        # Observe transition
        universe.observe_transition(old_state, new_state, action)

        # Evolve
        universe.evolve(1)

        # Print periodically
        if (frame + 1) % 100 == 0:
            print(f"\n--- Frame {frame + 1} ---")
            universe.root.print_state()

    # Final analysis
    print("\n" + "=" * 60)
    print("DISCOVERED STRUCTURE")
    print("=" * 60)

    structure = universe.get_discovered_structure()
    print(f"\nDimensions discovered: {structure['dimensions']}")
    print(f"Tokens learned: {structure['n_tokens']}")

    print("\nTop stable patterns:")
    for name, amp, phase in structure['top_patterns'][:15]:
        phase_deg = phase * 180 / np.pi
        meaning = "positive" if phase_deg > 0 else "negative"
        print(f"  {name}: amp={amp:.3f}, phase={phase_deg:.1f}deg ({meaning})")

    # Look for discovered relationships
    print("\nStrongest relationships discovered:")
    rels = [(n, t.get_magnitude()) for n, t in universe.root.tokens.items()
            if n.startswith('rel_')]
    rels.sort(key=lambda x: -x[1])
    for name, strength in rels[:10]:
        print(f"  {name}: {strength:.3f}")

    return universe


if __name__ == "__main__":
    universe = test_fractal_pong()
