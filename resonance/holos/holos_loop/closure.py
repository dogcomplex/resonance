"""
holos/closure.py - Closure Detection and Wave Dynamics

Core insight from neuroscience: What we observe (modes, patterns, frequencies)
are READOUTS of whether a closure condition has been satisfied, not causes.

In HOLOS terms:
- Closure = forward wave meets backward wave with consistent values
- Phase closure condition: Σ Δφ = 2πk (integer phase alignment)
- Translated: predicted value ≈ actual value at meeting point

Three layers of closure:
1. SCALING (where closure CAN occur) - layer structure defines resonant scales
2. CLOSURE (which loops PERSIST) - irreducible closures survive coupling/noise
3. READOUT (what we OBSERVE) - modes emerge from closure state

Irreducibility:
- Reducible closure: can be decomposed into smaller sub-solutions
- Irreducible closure: atomic, cannot be simplified further
- Irreducible closures are more stable, more informative, more worth crystallizing

This module provides:
- ClosureEvent: Record of a closure occurrence
- ClosureDetector: Detects and classifies closures
- PhaseAlignment: Measures alignment between predictions and evaluations
- IrreducibilityChecker: Determines if a closure is atomic or decomposable
"""

from typing import List, Tuple, Optional, Any, Dict, Set, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import math

S = TypeVar('S')  # State type
V = TypeVar('V')  # Value type


# ============================================================
# CLOSURE TYPES
# ============================================================

class ClosureType(Enum):
    """Classification of closure events"""
    NONE = "none"              # No closure (waves haven't met)
    REDUCIBLE = "reducible"    # Closure exists but can be decomposed
    IRREDUCIBLE = "irreducible"  # Atomic closure, cannot be simplified
    RESONANT = "resonant"      # Irreducible AND reinforced by multiple paths


class WaveOrigin(Enum):
    """Which wave a state came from"""
    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"  # Connection point


# ============================================================
# CLOSURE EVENT
# ============================================================

@dataclass
class ClosureEvent:
    """
    Record of a closure occurrence.

    A closure happens when:
    1. Forward and backward waves meet (position closure)
    2. Predicted value matches actual value (value closure)

    We track both to understand search dynamics.
    """
    # Location
    state_hash: int
    layer: int  # Which layer (0=positions, 1=paths, 2=covers, 3=policy)

    # Closure quality
    closure_type: ClosureType
    phase_diff: float  # |predicted - actual| / scale (0 = perfect alignment)

    # Values involved
    forward_value: Any  # Value from forward wave
    backward_value: Any  # Value from backward wave

    # Timing
    iteration: int
    timestamp: float = 0.0

    # Context
    forward_path_length: int = 0
    backward_path_length: int = 0
    num_contributing_paths: int = 1  # For resonant closures

    def __repr__(self):
        return (f"ClosureEvent(layer={self.layer}, type={self.closure_type.value}, "
                f"phase_diff={self.phase_diff:.3f}, paths={self.num_contributing_paths})")


# ============================================================
# PHASE ALIGNMENT
# ============================================================

@dataclass
class PhaseAlignment:
    """
    Measures alignment between two values.

    In the closure frame, "phase alignment" means:
    - Forward wave prediction matches backward wave evaluation
    - The two waves agree on the value at a meeting point

    Perfect alignment (phase_diff = 0) means Σ Δφ = 2πk exactly.
    """
    predicted: float
    actual: float
    scale: float = 1.0  # Normalization factor

    @property
    def phase_diff(self) -> float:
        """Phase difference, normalized by scale"""
        if self.scale == 0:
            return float('inf')
        return abs(self.predicted - self.actual) / self.scale

    @property
    def is_aligned(self) -> bool:
        """Is phase aligned within threshold?"""
        return self.phase_diff < 0.2  # 20% tolerance

    @property
    def alignment_strength(self) -> float:
        """How strong is the alignment (1.0 = perfect, 0.0 = none)"""
        return max(0.0, 1.0 - self.phase_diff)

    def to_closure_type(self, is_irreducible: bool) -> ClosureType:
        """Convert alignment to closure type"""
        if not self.is_aligned:
            return ClosureType.NONE
        return ClosureType.IRREDUCIBLE if is_irreducible else ClosureType.REDUCIBLE


# ============================================================
# CLOSURE DETECTOR
# ============================================================

class ClosureDetector(Generic[S, V]):
    """
    Detects and classifies closure events.

    A closure occurs when:
    1. Forward and backward waves both reach a state
    2. The values they carry are consistent (phase aligned)

    The detector tracks:
    - All closure events
    - Statistics on closure types
    - Resonance patterns (multiple paths to same closure)
    """

    def __init__(self, phase_threshold: float = 0.2):
        """
        Args:
            phase_threshold: Maximum phase_diff for alignment (default 20%)
        """
        self.phase_threshold = phase_threshold

        # Closure history
        self.closures: List[ClosureEvent] = []
        self.closure_by_hash: Dict[int, List[ClosureEvent]] = {}

        # Statistics
        self.stats = {
            'total_closures': 0,
            'irreducible': 0,
            'reducible': 0,
            'resonant': 0,
            'avg_phase_diff': 0.0,
            'best_phase_diff': float('inf'),
        }

    def check_closure(self,
                      state_hash: int,
                      forward_value: Any,
                      backward_value: Any,
                      layer: int = 0,
                      iteration: int = 0,
                      irreducibility_fn: Optional[callable] = None) -> Optional[ClosureEvent]:
        """
        Check if a closure has occurred at this state.

        Args:
            state_hash: Hash of the state where waves meet
            forward_value: Value carried by forward wave
            backward_value: Value carried by backward wave
            layer: Which layer this closure is on
            iteration: Current iteration number
            irreducibility_fn: Optional function to check if closure is irreducible

        Returns:
            ClosureEvent if closure detected, None otherwise
        """
        # Extract numeric values for comparison
        fwd_num = self._value_to_number(forward_value)
        bwd_num = self._value_to_number(backward_value)

        # Calculate phase alignment
        scale = max(abs(fwd_num), abs(bwd_num), 1.0)
        alignment = PhaseAlignment(fwd_num, bwd_num, scale)

        if not alignment.is_aligned:
            return None

        # Determine irreducibility
        is_irreducible = True
        if irreducibility_fn is not None:
            is_irreducible = irreducibility_fn(state_hash, forward_value, backward_value)

        # Check for resonance (multiple closures at same location)
        existing = self.closure_by_hash.get(state_hash, [])
        num_paths = len(existing) + 1

        closure_type = alignment.to_closure_type(is_irreducible)
        if num_paths > 1 and closure_type == ClosureType.IRREDUCIBLE:
            closure_type = ClosureType.RESONANT

        event = ClosureEvent(
            state_hash=state_hash,
            layer=layer,
            closure_type=closure_type,
            phase_diff=alignment.phase_diff,
            forward_value=forward_value,
            backward_value=backward_value,
            iteration=iteration,
            num_contributing_paths=num_paths
        )

        # Record
        self.closures.append(event)
        if state_hash not in self.closure_by_hash:
            self.closure_by_hash[state_hash] = []
        self.closure_by_hash[state_hash].append(event)

        # Update stats
        self._update_stats(event)

        return event

    def _value_to_number(self, value: Any) -> float:
        """Extract numeric value for phase comparison"""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, 'efficiency'):
            return value.efficiency
        if hasattr(value, 'value'):
            return float(value.value)
        # Try to convert
        try:
            return float(value)
        except:
            return 0.0

    def _update_stats(self, event: ClosureEvent):
        """Update running statistics"""
        self.stats['total_closures'] += 1

        if event.closure_type == ClosureType.IRREDUCIBLE:
            self.stats['irreducible'] += 1
        elif event.closure_type == ClosureType.REDUCIBLE:
            self.stats['reducible'] += 1
        elif event.closure_type == ClosureType.RESONANT:
            self.stats['resonant'] += 1

        # Running average of phase diff
        n = self.stats['total_closures']
        old_avg = self.stats['avg_phase_diff']
        self.stats['avg_phase_diff'] = old_avg + (event.phase_diff - old_avg) / n

        if event.phase_diff < self.stats['best_phase_diff']:
            self.stats['best_phase_diff'] = event.phase_diff

    def get_resonant_closures(self) -> List[int]:
        """Get state hashes with resonant (multi-path) closures"""
        return [h for h, events in self.closure_by_hash.items()
                if len(events) > 1]

    def get_irreducible_closures(self) -> List[ClosureEvent]:
        """Get all irreducible closure events"""
        return [e for e in self.closures
                if e.closure_type in [ClosureType.IRREDUCIBLE, ClosureType.RESONANT]]


# ============================================================
# IRREDUCIBILITY CHECKER
# ============================================================

class IrreducibilityChecker(ABC, Generic[S, V]):
    """
    Abstract base for checking if a closure is irreducible.

    A closure is IRREDUCIBLE if it cannot be decomposed into
    smaller sub-closures that together produce the same result.

    From the neuroscience analogy:
    - Reducible closures can split into smaller repeating subloops
    - Irreducible closures can't be tiled or factorized
    - Irreducible closures are dramatically more stable under noise/coupling

    This is where "prime structure" enters - not as prime frequencies,
    but as a constraint on decomposability.
    """

    @abstractmethod
    def is_irreducible(self, state_hash: int, forward_value: V, backward_value: V) -> bool:
        """
        Check if a closure at this state is irreducible.

        Returns True if the closure cannot be decomposed into simpler closures.
        """
        pass


class SimpleIrreducibilityChecker(IrreducibilityChecker[S, V]):
    """
    Simple irreducibility checker based on uniqueness.

    A closure is considered irreducible if:
    1. It's not a duplicate of an existing closure
    2. Its value can't be derived from simpler known values
    """

    def __init__(self):
        self.known_closures: Set[int] = set()
        self.known_values: Dict[int, Any] = {}

    def is_irreducible(self, state_hash: int, forward_value: Any, backward_value: Any) -> bool:
        # New closure at new location = potentially irreducible
        if state_hash not in self.known_closures:
            self.known_closures.add(state_hash)
            self.known_values[state_hash] = forward_value
            return True

        # Closure at known location - check if value changed
        known_value = self.known_values.get(state_hash)
        if known_value != forward_value:
            # Different value at same location = new information = irreducible
            return True

        # Same value at same location = reducible (already known)
        return False


# ============================================================
# MODE EMERGENCE
# ============================================================

class ModeEmergence:
    """
    Determines which search mode should emerge from the current closure state.

    Key insight: Modes are not choices, they're READOUTS of the closure state.

    - Lightning emerges when there's a clear direct path (soliton)
    - Wave emerges when exploring uniformly (plane wave)
    - Crystal emerges when amplifying at closure points (standing wave)
    - Osmosis emerges when following a gradient toward closure (diffusion)
    """

    def __init__(self, closure_detector: ClosureDetector):
        self.detector = closure_detector

    def get_emergent_mode(self,
                          forward_frontier_size: int,
                          backward_frontier_size: int,
                          recent_closures: int,
                          branching_factor: float) -> str:
        """
        Determine which mode emerges from current state.

        Args:
            forward_frontier_size: Size of forward wave front
            backward_frontier_size: Size of backward wave front
            recent_closures: Closures in last N iterations
            branching_factor: Average branching factor

        Returns:
            SearchMode value ("lightning", "wave", "crystal", "osmosis")
        """
        # Check for soliton condition (lightning)
        # Low branching + small frontier = direct path available
        if branching_factor < 3 and forward_frontier_size < 100:
            return "lightning"

        # Check for standing wave condition (crystal)
        # Recent closures = resonance points to amplify
        if recent_closures > 5:
            return "crystal"

        # Check for gradient condition (osmosis)
        # Asymmetric frontiers = concentration gradient
        ratio = max(forward_frontier_size, 1) / max(backward_frontier_size, 1)
        if ratio > 5 or ratio < 0.2:
            return "osmosis"

        # Default: plane wave (uniform exploration)
        return "wave"

    def score_state_for_expansion(self,
                                   state_hash: int,
                                   wave_origin: WaveOrigin,
                                   neighbors_solved: int,
                                   distance_to_boundary: int) -> float:
        """
        Score a state for expansion priority (osmosis-style).

        Higher score = higher confidence = expand first.
        """
        score = 0.0

        # More solved neighbors = higher confidence
        score += neighbors_solved * 10.0

        # Closer to boundary = higher confidence
        if distance_to_boundary > 0:
            score += 50.0 / distance_to_boundary

        # Check if this location has had closures
        if state_hash in self.detector.closure_by_hash:
            closures = self.detector.closure_by_hash[state_hash]
            # More closures = resonance point = very high priority
            score += len(closures) * 100.0

            # Irreducible closures are even better
            irreducible_count = sum(1 for c in closures
                                    if c.closure_type in [ClosureType.IRREDUCIBLE, ClosureType.RESONANT])
            score += irreducible_count * 50.0

        return score


# ============================================================
# LAYER COUPLING
# ============================================================

@dataclass
class LayerCoupling:
    """
    Describes coupling between adjacent layers.

    When a wave hits the boundary between layers, some energy:
    - Reflects (stays in current layer)
    - Transmits (moves to adjacent layer)

    The coupling coefficients determine how much.
    """
    from_layer: int
    to_layer: int

    # Coupling strength (0.0 to 1.0)
    transmission: float = 0.3  # How much energy transmits to adjacent layer
    reflection: float = 0.7    # How much energy reflects back

    # Frequency-dependent coupling (optional)
    # Different closure types may couple differently
    type_weights: Dict[ClosureType, float] = field(default_factory=lambda: {
        ClosureType.IRREDUCIBLE: 1.0,  # Irreducible closures couple strongly
        ClosureType.REDUCIBLE: 0.5,    # Reducible couple weakly
        ClosureType.RESONANT: 1.5,     # Resonant couple very strongly
    })

    def get_transmission(self, closure_type: ClosureType) -> float:
        """Get effective transmission for this closure type"""
        weight = self.type_weights.get(closure_type, 1.0)
        return self.transmission * weight


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def compute_phase_closure(values: List[float]) -> Tuple[bool, float]:
    """
    Check if a sequence of values satisfies phase closure.

    Phase closure: Σ Δφ = 2πk (integer multiple of 2π)

    In our context, this means the cumulative value changes
    return to a consistent state.

    Returns:
        (is_closed, residual) where residual is the "leftover" phase
    """
    if len(values) < 2:
        return True, 0.0

    # Compute differences
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    total_diff = sum(diffs)

    # Check if total difference is near zero (or integer multiple of some period)
    # For game values, "closure" means we return to a consistent outcome
    residual = abs(total_diff)
    is_closed = residual < 0.1  # Threshold for "close enough"

    return is_closed, residual


def estimate_irreducibility_from_path(path_length: int,
                                       branching_factor: float,
                                       num_alternatives: int) -> float:
    """
    Estimate how irreducible a solution path is.

    Longer paths with fewer alternatives are more irreducible.

    Returns value 0.0 (highly reducible) to 1.0 (highly irreducible)
    """
    if path_length == 0:
        return 0.0

    # Longer paths are more specific (harder to factor)
    length_factor = 1.0 - math.exp(-path_length / 10.0)

    # Fewer alternatives = more irreducible
    if num_alternatives == 0:
        alt_factor = 1.0
    else:
        alt_factor = 1.0 / (1.0 + math.log(1 + num_alternatives))

    # Low branching = more forced = more irreducible
    branch_factor = 1.0 / (1.0 + math.log(1 + branching_factor))

    return length_factor * alt_factor * branch_factor
