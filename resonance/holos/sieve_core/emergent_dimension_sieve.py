"""
EMERGENT DIMENSION SIEVE
========================

Fixing the dimension utility problem: dimensions need MULTIPLE tokens
spread across their range to show utility.

Key insight: Instead of updating the same token's coordinate repeatedly,
we need to create TEMPORAL tokens that capture state at different times,
allowing the dimension to show variance.

The agent's perspective emerges when:
1. Action tokens spread across the action dimension
2. Each action links to different outcomes
3. The dimension shows "if I'm HERE in action space, THIS happens"

Geometry emerges when:
1. Relative position tokens spread across spatial dimensions
2. Position predicts collision/miss outcomes
3. The dimension shows "if ball is HERE relative to paddle, THAT happens"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import cmath


@dataclass
class EmergentConfig:
    """Configuration."""
    base_damping: float = 0.03
    coupling_strength: float = 0.15
    survival_threshold: float = 0.02
    max_dimensions: int = 12
    max_tokens: int = 200
    temporal_window: int = 20  # Keep this many temporal snapshots


class EmergentToken:
    """A token with temporal identity and dimensional embedding."""

    def __init__(self, name: str, timestamp: int = 0):
        self.name = name
        self.timestamp = timestamp
        self.amplitude: complex = complex(1, 0)
        self.coordinates: Dict[str, float] = {}
        self.token_type: str = "unknown"
        self.outcome: Optional[str] = None  # What happened after this token?

    def full_id(self) -> str:
        """Unique identifier including timestamp."""
        return f"{self.name}@{self.timestamp}"


class DimensionStats:
    """Track dimension statistics for utility calculation."""

    def __init__(self, name: str):
        self.name = name
        self.coord_values: List[float] = []
        self.amp_values: List[float] = []
        self.outcome_mapping: Dict[float, str] = {}  # coord -> outcome

    def record(self, coord: float, amplitude: float, outcome: Optional[str] = None):
        self.coord_values.append(coord)
        self.amp_values.append(amplitude)
        if outcome:
            # Bin the coordinate
            binned = round(coord * 10) / 10
            self.outcome_mapping[binned] = outcome

        # Keep bounded
        if len(self.coord_values) > 100:
            self.coord_values = self.coord_values[-100:]
            self.amp_values = self.amp_values[-100:]

    def get_utility(self) -> float:
        """How much does this dimension explain variance?"""
        if len(self.coord_values) < 5:
            return 0.0

        # Variance in coordinates
        coord_var = np.var(self.coord_values)
        if coord_var < 0.001:
            return 0.0  # No variance along dimension

        # Correlation with amplitude
        if np.std(self.amp_values) > 0.001:
            corr = abs(np.corrcoef(self.coord_values, self.amp_values)[0, 1])
            if np.isnan(corr):
                corr = 0
        else:
            corr = 0

        # Outcome predictability (if we have outcomes)
        outcome_score = len(set(self.outcome_mapping.values())) / max(1, len(self.outcome_mapping))

        return (coord_var + corr + outcome_score) / 3

    def describes_agent_perspective(self) -> bool:
        """Does this dimension capture agent's action-outcome relationship?"""
        return 'hit' in self.outcome_mapping.values() or 'miss' in self.outcome_mapping.values()


class EmergentDimensionSieve:
    """
    A sieve where dimensions emerge from functional necessity.

    Key fix: Create temporal token snapshots so dimensions have
    multiple data points to show variance.
    """

    def __init__(self, name: str = "root", config: Optional[EmergentConfig] = None):
        self.name = name
        self.config = config or EmergentConfig()

        # Tokens indexed by full_id
        self.tokens: Dict[str, EmergentToken] = {}
        self.current_frame: int = 0

        # Dimensions with statistics
        self.dimensions: Dict[str, DimensionStats] = {
            'time': DimensionStats('time'),
            'action': DimensionStats('action'),
            'spatial_x': DimensionStats('spatial_x'),
            'spatial_y': DimensionStats('spatial_y'),
            'velocity_x': DimensionStats('velocity_x'),
            'velocity_y': DimensionStats('velocity_y'),
            'relative_x': DimensionStats('relative_x'),  # ball - paddle
        }

        # Couplings between tokens
        self.couplings: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Track outcomes for learning
        self.pending_outcome_tokens: List[str] = []

        self.entropy_history: List[float] = []
        self.damping = self.config.base_damping

    def _create_temporal_token(self, base_name: str, token_type: str) -> EmergentToken:
        """Create a token with temporal identity."""
        token = EmergentToken(base_name, self.current_frame)
        token.token_type = token_type
        token.coordinates['time'] = self.current_frame / 1000  # Normalized

        full_id = token.full_id()
        self.tokens[full_id] = token
        return token

    def observe_ball(self, x: float, y: float, vx: float, vy: float):
        """
        Observe ball state - creates temporal snapshot.

        The key: each observation becomes a SEPARATE token at a specific time,
        so dimensions can accumulate variance across multiple tokens.
        """
        # Normalize
        nx = x / 84.0
        ny = y / 84.0
        nvx = (vx + 5) / 10.0  # Assume velocity in [-5, 5]
        nvy = (vy + 5) / 10.0

        # Create temporal token for this observation
        token = self._create_temporal_token('ball_state', 'observation')

        # Set dimensional coordinates
        token.coordinates['spatial_x'] = nx
        token.coordinates['spatial_y'] = ny
        token.coordinates['velocity_x'] = nvx
        token.coordinates['velocity_y'] = nvy

        # Amplitude encodes "interestingness" - higher when moving toward paddle
        interest = max(0, vy) / 5.0  # Moving down is more relevant
        phase = nx * np.pi  # Phase encodes x position
        token.amplitude = complex(0.5 + interest * 0.5, 0) * cmath.exp(1j * phase)

        # Record in dimension stats
        self.dimensions['spatial_x'].record(nx, abs(token.amplitude))
        self.dimensions['spatial_y'].record(ny, abs(token.amplitude))
        self.dimensions['velocity_x'].record(nvx, abs(token.amplitude))
        self.dimensions['velocity_y'].record(nvy, abs(token.amplitude))

        # This token might get an outcome
        self.pending_outcome_tokens.append(token.full_id())

        return token

    def observe_paddle(self, x: float):
        """Observe paddle state."""
        nx = x / 84.0

        token = self._create_temporal_token('paddle_state', 'observation')
        token.coordinates['spatial_x'] = nx
        token.amplitude = complex(0.3, 0)  # Paddle is less dynamic

        self.dimensions['spatial_x'].record(nx, abs(token.amplitude))

        return token

    def observe_relative_position(self, ball_x: float, paddle_x: float):
        """
        Observe relative position - this is CRUCIAL for agent perspective.

        The agent cares about: "Where is the ball relative to ME?"
        """
        diff = (ball_x - paddle_x) / 84.0  # Normalized difference
        normalized = (diff + 1) / 2  # Map [-1, 1] to [0, 1]

        token = self._create_temporal_token('relative_position', 'geometry')
        token.coordinates['relative_x'] = normalized
        token.coordinates['spatial_x'] = ball_x / 84.0  # Also track absolute

        # Phase encodes direction: negative phase = ball left of paddle
        phase = diff * np.pi
        token.amplitude = complex(0.8 * np.cos(phase), 0.8 * np.sin(phase))

        # Record - this is where agent perspective emerges!
        self.dimensions['relative_x'].record(normalized, abs(token.amplitude))

        # Mark for outcome tracking
        self.pending_outcome_tokens.append(token.full_id())

        return token

    def observe_action(self, action: int, n_actions: int = 3):
        """
        Observe action taken - places token in action space.

        This IS the agent's perspective: "what I can do"
        """
        normalized = action / max(1, n_actions - 1)

        token = self._create_temporal_token(f'action_{action}', 'action')
        token.coordinates['action'] = normalized
        token.amplitude = complex(0.6, 0)

        self.dimensions['action'].record(normalized, abs(token.amplitude))
        self.pending_outcome_tokens.append(token.full_id())

        return token

    def observe_outcome(self, outcome: str):
        """
        Record outcome (hit/miss) for pending tokens.

        This is how the sieve learns: tokens that were active before
        a hit/miss get that outcome recorded, building the
        coord -> outcome mapping that makes dimensions useful.
        """
        # Assign outcome to recent tokens
        for full_id in self.pending_outcome_tokens[-10:]:
            if full_id in self.tokens:
                token = self.tokens[full_id]
                token.outcome = outcome

                # Record outcome in relevant dimensions
                for dim_name, coord in token.coordinates.items():
                    if dim_name in self.dimensions:
                        self.dimensions[dim_name].record(coord, abs(token.amplitude), outcome)

                # Boost amplitude of predictive tokens
                if outcome == 'hit':
                    token.amplitude *= 1.2
                else:
                    token.amplitude *= 0.8

        self.pending_outcome_tokens = []

    def create_causal_link(self, cause_id: str, effect_id: str, strength: float = 0.5):
        """Create coupling between tokens."""
        self.couplings[cause_id][effect_id] = strength

    def evolve(self):
        """Evolve the sieve."""
        self.current_frame += 1

        # 1. Interference between coupled tokens
        self._compute_interference()

        # 2. Prune old temporal tokens (keep recent window)
        self._prune_old_tokens()

        # 3. Check for emergent dimensions
        self._check_dimension_emergence()

        # 4. Self-tune
        self._self_tune()

    def _compute_interference(self):
        """Compute interference between tokens."""
        new_amplitudes: Dict[str, complex] = {}

        for full_id, token in self.tokens.items():
            # Self with damping
            new_amp = token.amplitude * (1 - self.damping)

            # Coupling-based interference
            if full_id in self.couplings:
                for other_id, coupling in self.couplings[full_id].items():
                    if other_id in self.tokens:
                        other = self.tokens[other_id]
                        transfer = coupling * other.amplitude * self.config.coupling_strength
                        new_amp += transfer

            # Proximity-based interference (tokens close in dimension space)
            for other_id, other in self.tokens.items():
                if other_id == full_id:
                    continue

                # Distance in shared dimensions
                shared = set(token.coordinates.keys()) & set(other.coordinates.keys())
                if len(shared) < 1:
                    continue

                dist_sq = sum((token.coordinates[d] - other.coordinates[d])**2
                             for d in shared) / len(shared)

                if dist_sq < 0.04:  # Close enough (within 0.2 in each dim)
                    interference = np.exp(-dist_sq * 10) * other.amplitude * 0.02
                    new_amp += interference

            new_amplitudes[full_id] = new_amp

        # Normalize
        total = sum(abs(a) for a in new_amplitudes.values())
        if total > 20:
            scale = 20 / total
            new_amplitudes = {k: v * scale for k, v in new_amplitudes.items()}

        # Apply
        for full_id, amp in new_amplitudes.items():
            if full_id in self.tokens:
                self.tokens[full_id].amplitude = amp

        # Remove dead tokens
        dead = [fid for fid, t in self.tokens.items()
                if abs(t.amplitude) < self.config.survival_threshold]
        for fid in dead:
            del self.tokens[fid]
            if fid in self.couplings:
                del self.couplings[fid]

    def _prune_old_tokens(self):
        """Prune tokens outside temporal window, keeping strongest."""
        min_frame = self.current_frame - self.config.temporal_window

        old_tokens = [(fid, t) for fid, t in self.tokens.items()
                      if t.timestamp < min_frame]

        # Keep the strongest old tokens (they represent stable patterns)
        old_tokens.sort(key=lambda x: -abs(x[1].amplitude))

        # Remove weak old tokens
        for fid, token in old_tokens[10:]:  # Keep top 10
            if fid in self.tokens:
                del self.tokens[fid]

    def _check_dimension_emergence(self):
        """Check if new dimensions should emerge."""
        if len(self.dimensions) >= self.config.max_dimensions:
            return

        # Look for unexplained variance patterns
        tokens_without_structure = [t for t in self.tokens.values()
                                   if len(t.coordinates) < 3 and abs(t.amplitude) > 0.2]

        if len(tokens_without_structure) > 10:
            # Group by outcome
            by_outcome: Dict[str, List[EmergentToken]] = defaultdict(list)
            for t in tokens_without_structure:
                outcome = t.outcome or 'unknown'
                by_outcome[outcome].append(t)

            # If outcomes are different but coordinates overlap -> need new dimension
            if len(by_outcome) > 1:
                new_dim = f"emergent_{len(self.dimensions)}"
                self.dimensions[new_dim] = DimensionStats(new_dim)

                # Assign coordinates based on outcome
                outcome_to_coord = {outcome: i / len(by_outcome)
                                   for i, outcome in enumerate(by_outcome.keys())}

                for outcome, tokens in by_outcome.items():
                    coord = outcome_to_coord[outcome]
                    for t in tokens:
                        t.coordinates[new_dim] = coord
                        self.dimensions[new_dim].record(coord, abs(t.amplitude), outcome)

    def _self_tune(self):
        """Self-tune damping based on entropy."""
        amplitudes = [abs(t.amplitude) for t in self.tokens.values()
                     if abs(t.amplitude) > 0.01]

        if len(amplitudes) < 3:
            return

        total = sum(amplitudes)
        probs = [a / total for a in amplitudes]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(probs))
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        self.entropy_history.append(normalized)

        if len(self.entropy_history) < 10:
            return

        recent = self.entropy_history[-10:]
        trend = recent[-1] - recent[0]

        # Target entropy around 0.7 (critical)
        if normalized < 0.6:
            self.damping = max(0.01, self.damping * 0.95)  # Less damping
        elif normalized > 0.8:
            self.damping = min(0.2, self.damping * 1.05)  # More damping

    def get_dimension_report(self) -> Dict:
        """Get detailed dimension analysis."""
        report = {}

        for dim_name, stats in self.dimensions.items():
            utility = stats.get_utility()
            n_points = len(stats.coord_values)
            variance = np.var(stats.coord_values) if stats.coord_values else 0
            agent_relevant = stats.describes_agent_perspective()

            report[dim_name] = {
                'utility': utility,
                'n_points': n_points,
                'variance': variance,
                'captures_outcomes': agent_relevant,
                'outcome_mapping': dict(stats.outcome_mapping),
            }

        return report

    def print_state(self):
        """Print sieve state."""
        print(f"\n{'='*60}")
        print(f"EMERGENT DIMENSION SIEVE - Frame {self.current_frame}")
        print(f"{'='*60}")

        print(f"\nTokens: {len(self.tokens)}, Damping: {self.damping:.4f}")

        # Token types
        types = defaultdict(int)
        for t in self.tokens.values():
            types[t.token_type] += 1
        print(f"Token types: {dict(types)}")

        # Dimension report
        print(f"\nDimension Analysis:")
        report = self.get_dimension_report()

        for dim_name, info in sorted(report.items(), key=lambda x: -x[1]['utility']):
            bar = '#' * int(info['utility'] * 20)
            agent_marker = " [AGENT]" if info['captures_outcomes'] else ""
            print(f"  {dim_name:15s}: utility={info['utility']:.3f} var={info['variance']:.3f} n={info['n_points']:3d} {bar}{agent_marker}")

            if info['outcome_mapping']:
                # Show coord -> outcome mapping
                outcomes = list(set(info['outcome_mapping'].values()))
                if len(outcomes) > 1:
                    print(f"                   Outcomes: {outcomes}")

        # Top patterns
        print(f"\nTop amplitude patterns:")
        patterns = [(t.name, abs(t.amplitude), t.outcome, t.timestamp)
                   for t in self.tokens.values()]
        patterns.sort(key=lambda x: -x[1])

        for name, amp, outcome, ts in patterns[:8]:
            outcome_str = f" -> {outcome}" if outcome else ""
            print(f"    {name}@{ts}: amp={amp:.3f}{outcome_str}")


# =============================================================================
# TEST: Does the agent perspective emerge?
# =============================================================================

def test_emergent_dimensions():
    """Test if agent perspective and geometry emerge."""
    print("=" * 70)
    print("EMERGENT DIMENSION SIEVE")
    print("Does the agent's perspective emerge from observations?")
    print("=" * 70)

    sieve = EmergentDimensionSieve()

    # Simulate Pong
    ball_x, ball_y = 42, 20
    ball_dx, ball_dy = 2, 1.5
    paddle_x = 42.0

    hits = 0
    misses = 0

    for frame in range(1000):
        # Observations
        sieve.observe_ball(ball_x, ball_y, ball_dx, ball_dy)
        sieve.observe_paddle(paddle_x)
        sieve.observe_relative_position(ball_x, paddle_x)

        # Action (MIXED policy: sometimes good, sometimes random to get misses)
        if np.random.random() < 0.3:  # 30% random
            action = np.random.randint(0, 3)
        elif ball_x < paddle_x - 5:
            action = 0  # Left
        elif ball_x > paddle_x + 5:
            action = 2  # Right
        else:
            action = 1  # Stay

        sieve.observe_action(action)

        # Physics
        ball_x += ball_dx
        ball_y += ball_dy

        # Wall bounces
        if ball_x <= 4 or ball_x >= 80:
            ball_dx *= -1
            ball_x = np.clip(ball_x, 4, 80)

        if ball_y <= 4:
            ball_dy = abs(ball_dy)

        # Paddle interaction
        if ball_y >= 76:
            distance = abs(ball_x - paddle_x)
            if distance < 12:
                sieve.observe_outcome('hit')
                hits += 1
            else:
                sieve.observe_outcome('miss')
                misses += 1

            # Reset ball
            ball_y = 20
            ball_x = np.random.uniform(20, 64)
            ball_dx = np.random.choice([-2, -1.5, 1.5, 2])
            ball_dy = abs(ball_dy)

        # Move paddle
        if action == 0:
            paddle_x = max(10, paddle_x - 4)
        elif action == 2:
            paddle_x = min(74, paddle_x + 4)

        # Evolve sieve
        sieve.evolve()

        if (frame + 1) % 200 == 0:
            sieve.print_state()
            print(f"\nGame stats: {hits} hits, {misses} misses, {hits/(hits+misses+1e-6):.1%} hit rate")

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS: What Did We Discover?")
    print("=" * 70)

    report = sieve.get_dimension_report()

    print("\n1. AGENT PERSPECTIVE (Action Dimension):")
    action_info = report.get('action', {})
    print(f"   Utility: {action_info.get('utility', 0):.3f}")
    print(f"   Captures hit/miss: {action_info.get('captures_outcomes', False)}")
    if action_info.get('outcome_mapping'):
        print(f"   Action -> Outcome mapping: {action_info['outcome_mapping']}")

    print("\n2. GEOMETRY (Relative Position):")
    rel_info = report.get('relative_x', {})
    print(f"   Utility: {rel_info.get('utility', 0):.3f}")
    print(f"   Captures hit/miss: {rel_info.get('captures_outcomes', False)}")
    if rel_info.get('outcome_mapping'):
        print(f"   Position -> Outcome mapping shows:")
        for coord, outcome in sorted(rel_info['outcome_mapping'].items()):
            print(f"      relative_x={coord:.1f} -> {outcome}")

    print("\n3. VELOCITY:")
    vx_info = report.get('velocity_x', {})
    vy_info = report.get('velocity_y', {})
    print(f"   X Utility: {vx_info.get('utility', 0):.3f}")
    print(f"   Y Utility: {vy_info.get('utility', 0):.3f}")

    print("\n4. SPATIAL:")
    sx_info = report.get('spatial_x', {})
    sy_info = report.get('spatial_y', {})
    print(f"   X Utility: {sx_info.get('utility', 0):.3f}, Variance: {sx_info.get('variance', 0):.3f}")
    print(f"   Y Utility: {sy_info.get('utility', 0):.3f}, Variance: {sy_info.get('variance', 0):.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("""
The agent's perspective SHOULD emerge in 'relative_x' dimension because:
- It measures ball position RELATIVE to paddle (what the agent cares about)
- It directly predicts outcomes (hit when close, miss when far)
- This is the dimension that matters for deciding actions

If 'relative_x' captures outcomes but 'spatial_x' doesn't, that proves
geometry is emerging from RELATIONSHIPS, not absolute positions.
""")

    return sieve


if __name__ == "__main__":
    sieve = test_emergent_dimensions()
