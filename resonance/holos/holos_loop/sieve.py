"""
holos_loop/sieve.py - The Universal Primitive

Everything else is parameterization of the sieve:
- Games are rule sets
- Layers are frequency bands
- Modes are resonance patterns
- Solutions are stable interference patterns

The sieve operates by:
1. Patterns exist with complex amplitudes
2. Rules transform patterns, transferring amplitude
3. Multiple paths to same pattern INTERFERE (sum amplitudes)
4. Threshold sieves out low-amplitude patterns
5. System self-anneals: interference eliminates inconsistency,
   survivors reinforce, temperature drops, stable modes emerge

This is what closures DO when you let them run continuously.
"""

import cmath
import math
from typing import Dict, List, Tuple, Any, Optional, Set, Callable, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum


# ============================================================
# CORE PRIMITIVES
# ============================================================

@dataclass(frozen=True)
class Pattern:
    """
    A configuration of tokens.

    Patterns are the "states" of the sieve, but more general:
    - Could be game positions
    - Could be rewrite rules themselves
    - Could be abstract tokens
    - Could be continuous coordinates (for analog sieve)

    Frozen for hashability - patterns are identity.
    """
    tokens: Any  # Must be hashable

    def __hash__(self):
        return hash(self.tokens)

    def __eq__(self, other):
        if not isinstance(other, Pattern):
            return False
        return self.tokens == other.tokens

    def __repr__(self):
        return f"Pattern({self.tokens})"


@dataclass
class Amplitude:
    """
    Complex amplitude - magnitude AND phase.

    Phase encodes "which direction" the wave came from:
    - Forward wave: phase ≈ 0
    - Backward wave: phase ≈ π
    - Closure: phases align (constructive interference)
    - Contradiction: phases oppose (destructive interference)

    The complex number representation naturally handles interference:
    - Addition = superposition
    - |z|² = probability/intensity
    - arg(z) = phase
    """
    value: complex = complex(1.0, 0.0)

    @property
    def magnitude(self) -> float:
        """How "real" is this pattern?"""
        return abs(self.value)

    @property
    def phase(self) -> float:
        """Which direction did we come from? (radians)"""
        return cmath.phase(self.value)

    @property
    def intensity(self) -> float:
        """Born rule: probability ~ |amplitude|²"""
        return self.magnitude ** 2

    def interfere(self, other: 'Amplitude') -> 'Amplitude':
        """
        Superposition - THE sieve operation.

        This is where the magic happens:
        - Same phase → constructive → amplitude grows
        - Opposite phase → destructive → amplitude shrinks
        - Orthogonal phase → rotation → amplitude changes direction
        """
        return Amplitude(self.value + other.value)

    def scale(self, factor: complex) -> 'Amplitude':
        """Scale amplitude (for damping, rule transfer, etc.)"""
        return Amplitude(self.value * factor)

    def is_coherent_with(self, other: 'Amplitude', threshold: float = 0.3) -> bool:
        """
        Are these amplitudes in phase? (potential closure)

        Threshold is in radians - how close phases must be.
        """
        phase_diff = abs(self.phase - other.phase)
        # Handle wrap-around
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
        return phase_diff < threshold

    @staticmethod
    def from_polar(magnitude: float, phase: float) -> 'Amplitude':
        """Create amplitude from magnitude and phase"""
        return Amplitude(complex(magnitude * math.cos(phase), magnitude * math.sin(phase)))

    @staticmethod
    def forward(magnitude: float = 1.0) -> 'Amplitude':
        """Forward-propagating amplitude (phase 0)"""
        return Amplitude(complex(magnitude, 0))

    @staticmethod
    def backward(magnitude: float = 1.0) -> 'Amplitude':
        """Backward-propagating amplitude (phase π)"""
        return Amplitude(complex(-magnitude, 0))

    @staticmethod
    def zero() -> 'Amplitude':
        """Zero amplitude (doesn't exist)"""
        return Amplitude(complex(0, 0))

    def __repr__(self):
        return f"Amp({self.magnitude:.3f}∠{math.degrees(self.phase):.1f}°)"


@dataclass
class Rule:
    """
    LHS => RHS rewrite rule with amplitude transfer.

    Rules are the "moves" of the sieve, but more general:
    - Could be game moves (position => position)
    - Could be logical inference (premise => conclusion)
    - Could be causal transitions (cause => effect)

    The transfer coefficient allows rules to:
    - Preserve amplitude (transfer = 1)
    - Dampen (|transfer| < 1)
    - Amplify (|transfer| > 1)
    - Rotate phase (transfer has imaginary component)
    """
    lhs: Pattern          # What to match
    rhs: Pattern          # What to produce
    transfer: complex = complex(1.0, 0.0)  # Amplitude multiplier
    name: str = ""        # Optional identifier

    def matches(self, pattern: Pattern) -> bool:
        """Does this rule apply to this pattern?"""
        return pattern == self.lhs

    def apply(self, amplitude: Amplitude) -> Tuple[Pattern, Amplitude]:
        """
        Apply rule: transform pattern, transfer amplitude.

        Returns (new_pattern, new_amplitude).
        """
        new_amplitude = amplitude.scale(self.transfer)
        return self.rhs, new_amplitude

    def reversed(self) -> 'Rule':
        """
        Create reverse rule (RHS => LHS).

        For bidirectional search: backward rules are forward rules
        with inverted transfer (conjugate to flip phase).
        """
        return Rule(
            lhs=self.rhs,
            rhs=self.lhs,
            transfer=1.0 / self.transfer if self.transfer != 0 else 0,
            name=f"{self.name}_rev" if self.name else ""
        )

    def __repr__(self):
        return f"Rule({self.lhs} => {self.rhs}, transfer={self.transfer})"


class MatchMode(Enum):
    """How rules match patterns"""
    EXACT = "exact"       # Pattern must equal LHS exactly
    CONTAINS = "contains" # Pattern must contain LHS as subpattern
    REGEX = "regex"       # Pattern matches LHS as regex (for token sequences)


@dataclass
class FlexibleRule(Rule):
    """
    Rule with flexible matching.

    For non-exact matching (contains, regex), the LHS acts as a template
    and the RHS describes the transformation.
    """
    match_mode: MatchMode = MatchMode.EXACT
    match_fn: Optional[Callable[[Pattern], bool]] = None
    transform_fn: Optional[Callable[[Pattern], Pattern]] = None

    def matches(self, pattern: Pattern) -> bool:
        if self.match_fn:
            return self.match_fn(pattern)
        if self.match_mode == MatchMode.EXACT:
            return pattern == self.lhs
        elif self.match_mode == MatchMode.CONTAINS:
            # Check if lhs tokens are subset of pattern tokens
            if hasattr(self.lhs.tokens, '__iter__') and hasattr(pattern.tokens, '__iter__'):
                return set(self.lhs.tokens).issubset(set(pattern.tokens))
        return False

    def apply(self, amplitude: Amplitude, source_pattern: Pattern = None) -> Tuple[Pattern, Amplitude]:
        new_amplitude = amplitude.scale(self.transfer)
        if self.transform_fn and source_pattern:
            new_pattern = self.transform_fn(source_pattern)
        else:
            new_pattern = self.rhs
        return new_pattern, new_amplitude


# ============================================================
# THE SIEVE
# ============================================================

class Sieve:
    """
    The universal substrate.

    A sieve is a field of amplitudes over patterns, evolving under rules.

    The sieve operation:
    1. For each pattern with amplitude above threshold
    2. Apply all matching rules
    3. Each rule produces new pattern with transferred amplitude
    4. Multiple paths to same pattern INTERFERE (amplitudes add)
    5. Patterns below threshold are sieved out (cease to exist)

    Self-annealing:
    - Early: many patterns survive → high temperature → exploration
    - Late: interference eliminates inconsistency → low temperature → convergence
    - Stable: only coherent patterns remain → solution found
    """

    def __init__(self, threshold: float = 0.001, damping: float = 0.99):
        """
        Args:
            threshold: Minimum amplitude to survive (the sieve mesh size)
            damping: Global damping per evolution step (energy loss)
        """
        self.field: Dict[Pattern, Amplitude] = defaultdict(Amplitude.zero)
        self.threshold = threshold
        self.damping = damping

        # History tracking
        self.generation = 0
        self.history: List[Dict[str, Any]] = []

        # Closure detection
        self.closures: List[Tuple[Pattern, Amplitude, int]] = []  # (pattern, amplitude, generation)

    def inject(self, pattern: Pattern, amplitude: Amplitude):
        """
        Add amplitude at a pattern (source/seed).

        Multiple injections at same pattern interfere.
        """
        self.field[pattern] = self.field[pattern].interfere(amplitude)

    def inject_forward(self, pattern: Pattern, magnitude: float = 1.0):
        """Inject forward-propagating amplitude (phase 0)"""
        self.inject(pattern, Amplitude.forward(magnitude))

    def inject_backward(self, pattern: Pattern, magnitude: float = 1.0):
        """Inject backward-propagating amplitude (phase π)"""
        self.inject(pattern, Amplitude.backward(magnitude))

    def evolve(self, rules: List[Rule], detect_closures: bool = True) -> Dict[str, Any]:
        """
        One evolution step.

        Returns statistics about this step.
        """
        new_field: Dict[Pattern, Amplitude] = defaultdict(Amplitude.zero)

        stats = {
            'generation': self.generation,
            'patterns_in': 0,
            'patterns_out': 0,
            'rules_applied': 0,
            'interferences': 0,
            'closures_detected': 0,
            'sieved_out': 0,
        }

        # Track which patterns receive amplitude from multiple sources
        amplitude_sources: Dict[Pattern, List[Tuple[Pattern, Rule]]] = defaultdict(list)

        for pattern, amplitude in list(self.field.items()):
            if amplitude.magnitude < self.threshold:
                stats['sieved_out'] += 1
                continue

            stats['patterns_in'] += 1

            # Apply damping
            damped = amplitude.scale(self.damping)

            # Apply all matching rules
            applied_any = False
            for rule in rules:
                if rule.matches(pattern):
                    new_pattern, new_amplitude = rule.apply(damped)

                    # Track source for interference detection
                    if new_pattern in new_field and new_field[new_pattern].magnitude > 0:
                        stats['interferences'] += 1

                    amplitude_sources[new_pattern].append((pattern, rule))
                    new_field[new_pattern] = new_field[new_pattern].interfere(new_amplitude)
                    stats['rules_applied'] += 1
                    applied_any = True

            # If no rules applied, pattern persists (boundary/stable)
            if not applied_any:
                new_field[pattern] = new_field[pattern].interfere(damped)

        # Detect closures: patterns with high amplitude from both phases
        if detect_closures:
            for pattern, sources in amplitude_sources.items():
                if len(sources) >= 2:
                    # Check if sources have different phases (forward/backward meeting)
                    phases = [self.field[src].phase for src, _ in sources if src in self.field]
                    if phases:
                        phase_spread = max(phases) - min(phases)
                        # Significant phase spread but still constructive = closure
                        if phase_spread > 0.5 and new_field[pattern].magnitude > self.threshold * 10:
                            self.closures.append((pattern, new_field[pattern], self.generation))
                            stats['closures_detected'] += 1

        # Apply threshold (the sieve)
        self.field = {p: a for p, a in new_field.items() if a.magnitude >= self.threshold}
        stats['patterns_out'] = len(self.field)

        self.generation += 1
        self.history.append(stats)

        return stats

    def evolve_bidirectional(self, forward_rules: List[Rule], backward_rules: List[Rule] = None) -> Dict[str, Any]:
        """
        Evolve with explicit forward and backward rule sets.

        If backward_rules not provided, generates them by reversing forward_rules.
        """
        if backward_rules is None:
            backward_rules = [r.reversed() for r in forward_rules]

        return self.evolve(forward_rules + backward_rules)

    def temperature(self) -> float:
        """
        Effective temperature = how "disordered" is the field?

        High temperature: many patterns, similar amplitudes (exploration)
        Low temperature: few patterns, one dominant (convergence)

        Computed as normalized entropy of amplitude distribution.
        """
        magnitudes = [a.magnitude for a in self.field.values() if a.magnitude >= self.threshold]
        if not magnitudes:
            return 0.0  # Frozen/dead

        total = sum(magnitudes)
        if total == 0:
            return 0.0

        probs = [m / total for m in magnitudes]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(len(magnitudes) + 1)

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def coherence(self) -> float:
        """
        How phase-aligned is the field?

        High coherence: all amplitudes point same direction (solved)
        Low coherence: amplitudes point all directions (unsolved)
        """
        if not self.field:
            return 0.0

        # Vector sum of unit phasors
        total = sum((a.value / abs(a.value) if abs(a.value) > 0 else 0
                     for a in self.field.values()), complex(0, 0))

        return abs(total) / len(self.field)

    def total_amplitude(self) -> float:
        """Total magnitude in the field (energy)"""
        return sum(a.magnitude for a in self.field.values())

    def total_intensity(self) -> float:
        """Total intensity (squared magnitude)"""
        return sum(a.intensity for a in self.field.values())

    def stable_patterns(self, min_amplitude: float = None) -> List[Tuple[Pattern, Amplitude]]:
        """
        Patterns that have survived with significant amplitude.

        These are the "solutions" - stable interference patterns.
        """
        thresh = min_amplitude or self.threshold * 10
        return [(p, a) for p, a in self.field.items() if a.magnitude >= thresh]

    def dominant_pattern(self) -> Optional[Tuple[Pattern, Amplitude]]:
        """The pattern with highest amplitude (if any)"""
        if not self.field:
            return None
        return max(self.field.items(), key=lambda x: x[1].magnitude)

    def is_stable(self, window: int = 5, tolerance: float = 0.01) -> bool:
        """
        Has the sieve reached equilibrium?

        Checks if temperature has stabilized over recent history.
        """
        if len(self.history) < window:
            return False

        recent_temps = [self.temperature()]  # Current
        # Reconstruct recent temperatures from pattern counts (approximation)
        for stats in self.history[-window:]:
            if stats['patterns_out'] > 0:
                recent_temps.append(stats['patterns_out'] / max(1, stats['patterns_in']))

        if len(recent_temps) < 2:
            return False

        variance = sum((t - sum(recent_temps)/len(recent_temps))**2 for t in recent_temps) / len(recent_temps)
        return variance < tolerance

    def clear(self):
        """Reset the sieve"""
        self.field.clear()
        self.generation = 0
        self.history.clear()
        self.closures.clear()

    def summary(self) -> str:
        """Human-readable state summary"""
        lines = [
            f"Sieve (gen {self.generation}):",
            f"  Patterns: {len(self.field)}",
            f"  Total amplitude: {self.total_amplitude():.3f}",
            f"  Temperature: {self.temperature():.3f}",
            f"  Coherence: {self.coherence():.3f}",
            f"  Closures: {len(self.closures)}",
        ]

        dominant = self.dominant_pattern()
        if dominant:
            p, a = dominant
            lines.append(f"  Dominant: {p} @ {a}")

        return "\n".join(lines)


# ============================================================
# CONVENIENCE: SOLVE LOOP
# ============================================================

def solve(
    sieve: Sieve,
    rules: List[Rule],
    forward_seeds: List[Pattern],
    backward_seeds: List[Pattern] = None,
    max_generations: int = 1000,
    stability_window: int = 10,
    verbose: bool = True
) -> Tuple[List[Tuple[Pattern, Amplitude]], Dict[str, Any]]:
    """
    Run the sieve until stable.

    The sieve self-anneals:
    1. Inject seeds (forward at phase 0, backward at phase π)
    2. Evolve until temperature stabilizes
    3. Return stable patterns (solutions)

    Args:
        sieve: The sieve to run
        rules: Rewrite rules (forward direction)
        forward_seeds: Starting patterns (injected at phase 0)
        backward_seeds: Target patterns (injected at phase π)
        max_generations: Safety limit
        stability_window: How many generations to check for stability
        verbose: Print progress

    Returns:
        (stable_patterns, statistics)
    """
    sieve.clear()

    # Inject seeds
    n_forward = len(forward_seeds)
    n_backward = len(backward_seeds) if backward_seeds else 0

    # Normalize amplitudes
    fwd_amp = 1.0 / math.sqrt(n_forward) if n_forward > 0 else 0
    bwd_amp = 1.0 / math.sqrt(n_backward) if n_backward > 0 else 0

    for seed in forward_seeds:
        sieve.inject_forward(seed, fwd_amp)

    if backward_seeds:
        for seed in backward_seeds:
            sieve.inject_backward(seed, bwd_amp)

    # Generate backward rules
    backward_rules = [r.reversed() for r in rules]
    all_rules = rules + backward_rules

    # Evolution loop
    stats_history = []
    prev_temp = 1.0

    for gen in range(max_generations):
        stats = sieve.evolve(all_rules)
        stats_history.append(stats)

        temp = sieve.temperature()

        if verbose and gen % 10 == 0:
            print(f"Gen {gen}: patterns={stats['patterns_out']}, "
                  f"temp={temp:.3f}, closures={len(sieve.closures)}")

        # Check stability
        if sieve.is_stable(stability_window):
            if verbose:
                print(f"Stable at generation {gen}")
            break

        # Check if frozen (no patterns left)
        if stats['patterns_out'] == 0:
            if verbose:
                print(f"Frozen at generation {gen}")
            break

        prev_temp = temp

    # Aggregate stats
    total_stats = {
        'generations': sieve.generation,
        'total_rules_applied': sum(s['rules_applied'] for s in stats_history),
        'total_interferences': sum(s['interferences'] for s in stats_history),
        'total_closures': len(sieve.closures),
        'final_patterns': len(sieve.field),
        'final_temperature': sieve.temperature(),
        'final_coherence': sieve.coherence(),
    }

    return sieve.stable_patterns(), total_stats


# ============================================================
# VALUE ENCODING
# ============================================================

def value_to_phase(value: Any) -> float:
    """
    Encode a game-theoretic value as a phase.

    Standard encoding:
    - Win (+1): phase = 0 (aligned with forward)
    - Loss (-1): phase = π (opposite to forward)
    - Draw (0): phase = π/2 (orthogonal)

    For continuous values in [-1, 1], linearly interpolate.
    """
    if isinstance(value, (int, float)):
        # Map [-1, 1] to [π, 0]
        # -1 -> π, 0 -> π/2, +1 -> 0
        normalized = max(-1, min(1, value))
        return math.pi * (1 - normalized) / 2

    # Symbolic values
    if value in (1, True, 'win', 'W', 'X'):
        return 0
    elif value in (-1, False, 'loss', 'L', 'O'):
        return math.pi
    elif value in (0, None, 'draw', 'D'):
        return math.pi / 2

    return math.pi / 2  # Default: unknown = orthogonal


def phase_to_value(phase: float) -> float:
    """
    Decode a phase back to a game-theoretic value.

    Returns value in [-1, 1].
    """
    # Map [0, π] to [1, -1]
    normalized_phase = abs(phase) % math.pi
    return 1 - 2 * normalized_phase / math.pi
