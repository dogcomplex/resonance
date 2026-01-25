"""
sieve_core/information.py - Information as Fundamental Substrate

Going deeper than emergence.py.

The deepest question: What is information, before there are things to have information ABOUT?

Proposal: Information IS relationship. Not bits, not states - pure relationship.
A "bit" is already too structured - it presupposes two distinct states.
At the deepest level, there's just: things that could be the same or different.

This file explores:
1. Pre-information: Distinguishability itself
2. Information geometry: The space of possible distinctions
3. The emergence of logic from distinguishability
4. How the sieve operates on distinctions directly

Key insight: The sieve threshold isn't about "amplitude" - it's about
DISTINGUISHABILITY. Below threshold = indistinguishable = same thing.
"""

import cmath
import math
from typing import Dict, List, Tuple, Any, Optional, Set, FrozenSet, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np


# ============================================================
# DISTINGUISHABILITY: THE PRE-INFORMATIONAL LEVEL
# ============================================================

@dataclass(frozen=True)
class Distinction:
    """
    A distinction - the most primitive informational entity.

    A distinction is just: "this and that are different."
    It doesn't say WHAT they are, only that they're distinguishable.

    Formally: Distinction(A, B) means A ≠ B.

    This is more primitive than a bit:
    - A bit says "0 or 1" (presupposes two things)
    - A distinction says "different" (creates two things by distinguishing)

    The universe begins with the first distinction.
    (This is Spencer-Brown's "Laws of Form")
    """
    # Distinctions are identified by their boundary
    # The boundary is what separates the two sides
    boundary: Any  # Hashable identifier for this distinction

    def __repr__(self):
        return f"D[{self.boundary}]"


@dataclass(frozen=True)
class Side:
    """
    One side of a distinction.

    A side has meaning only relative to its distinction.
    "True" means nothing without "False".
    "Here" means nothing without "Not-here".

    Sides are the first things that can "exist" -
    existence means being on one side of a distinction.
    """
    distinction: Distinction
    which: bool  # True = marked/inside, False = unmarked/outside

    def opposite(self) -> 'Side':
        return Side(self.distinction, not self.which)

    def __repr__(self):
        mark = "T" if self.which else "F"
        return f"{mark}({self.distinction.boundary})"


# ============================================================
# DISTINCTION SPACE: WHERE DISTINCTIONS LIVE
# ============================================================

class DistinctionSpace:
    """
    A space of distinctions with interference.

    Key insight: Distinctions themselves can be distinguished!
    This creates a hierarchy of meta-distinctions.

    The amplitude field is over distinctions, not over states.
    High amplitude = clear distinction.
    Low amplitude = fuzzy distinction.
    Zero amplitude = no distinction (identity).

    This is pre-logical: logic emerges from stable distinction patterns.
    """

    def __init__(self, threshold: float = 1e-10):
        self.threshold = threshold
        # Amplitude over distinctions
        self.field: Dict[Distinction, complex] = {}
        # Meta-distinctions: distinctions between distinctions
        self.meta_field: Dict[Tuple[Distinction, Distinction], complex] = {}

    def distinguish(self, boundary: Any, amplitude: complex = 1.0):
        """Create or reinforce a distinction."""
        d = Distinction(boundary)
        self.field[d] = self.field.get(d, 0j) + amplitude
        self._cleanup()

    def identify(self, d: Distinction, amplitude: complex = 1.0):
        """Weaken a distinction (make things more same)."""
        if d in self.field:
            self.field[d] -= amplitude
            self._cleanup()

    def meta_distinguish(self, d1: Distinction, d2: Distinction, amplitude: complex = 1.0):
        """Create a distinction between distinctions."""
        key = (d1, d2) if hash(d1) < hash(d2) else (d2, d1)
        self.meta_field[key] = self.meta_field.get(key, 0j) + amplitude
        self._cleanup()

    def _cleanup(self):
        """Remove sub-threshold distinctions (they become identities)."""
        self.field = {d: a for d, a in self.field.items() if abs(a) >= self.threshold}
        self.meta_field = {k: a for k, a in self.meta_field.items() if abs(a) >= self.threshold}

    def interfere(self, other: 'DistinctionSpace') -> 'DistinctionSpace':
        """Combine two distinction spaces by interference."""
        result = DistinctionSpace(threshold=self.threshold)

        # All distinctions from both spaces
        all_d = set(self.field.keys()) | set(other.field.keys())
        for d in all_d:
            result.field[d] = self.field.get(d, 0j) + other.field.get(d, 0j)

        # All meta-distinctions from both
        all_meta = set(self.meta_field.keys()) | set(other.meta_field.keys())
        for key in all_meta:
            result.meta_field[key] = self.meta_field.get(key, 0j) + other.meta_field.get(key, 0j)

        result._cleanup()
        return result

    def entropy(self) -> float:
        """Entropy of the distinction field."""
        if not self.field:
            return 0.0
        mags = [abs(a)**2 for a in self.field.values()]
        total = sum(mags)
        if total == 0:
            return 0.0
        probs = [m/total for m in mags]
        return -sum(p * math.log(p + 1e-15) for p in probs if p > 0)

    def complexity(self) -> float:
        """
        Complexity = number of independent distinctions.

        Distinctions that can be composed from others don't count.
        This is like the rank of the distinction space.
        """
        # Simple measure: number of above-threshold distinctions
        return len(self.field) + len(self.meta_field) * 0.5


# ============================================================
# LOGICAL EMERGENCE: LOGIC FROM DISTINCTION
# ============================================================

class LogicalSpace:
    """
    Logic emerges from stable distinction patterns.

    Spencer-Brown showed:
    - One operation: distinction (mark/unmark)
    - Two laws: calling (double mark = no mark), crossing (inside/outside swap)

    From these, all of Boolean algebra emerges.

    In our framework:
    - Distinctions that persist = valid logical entities
    - Distinctions that destructively interfere = contradictions
    - The laws of logic are just: what doesn't self-destruct

    This is computational coherentism applied to logic itself.
    """

    def __init__(self, base: DistinctionSpace):
        self.base = base
        # Propositions are stable distinction patterns
        self.propositions: Dict[str, FrozenSet[Distinction]] = {}

    def define(self, name: str, distinctions: Set[Distinction]):
        """Define a proposition as a set of distinctions."""
        self.propositions[name] = frozenset(distinctions)

    def _to_amplitude(self, distinctions: FrozenSet[Distinction]) -> complex:
        """Get combined amplitude of a distinction pattern."""
        if not distinctions:
            return 1.0  # Empty = always true (no distinction = same)
        return sum(self.base.field.get(d, 0j) for d in distinctions) / len(distinctions)

    def truth_value(self, name: str) -> complex:
        """
        Truth is not binary - it's amplitude.

        |truth| = 1: definitely true or definitely false
        |truth| = 0: meaningless (no stable distinction)
        Complex phase: direction of truth (true vs false)
        """
        if name not in self.propositions:
            return 0j
        return self._to_amplitude(self.propositions[name])

    def logical_and(self, a: str, b: str) -> complex:
        """AND = intersection of distinction patterns."""
        if a not in self.propositions or b not in self.propositions:
            return 0j
        combined = self.propositions[a] | self.propositions[b]
        return self._to_amplitude(frozenset(combined))

    def logical_or(self, a: str, b: str) -> complex:
        """OR = union of distinction patterns."""
        if a not in self.propositions or b not in self.propositions:
            return 0j
        combined = self.propositions[a] & self.propositions[b]
        return self._to_amplitude(frozenset(combined))

    def logical_not(self, a: str) -> complex:
        """NOT = phase flip (opposite side of distinction)."""
        return -self.truth_value(a)

    def contradiction(self, a: str) -> bool:
        """A contradiction has zero amplitude (self-destructive interference)."""
        return abs(self.truth_value(a)) < self.base.threshold


# ============================================================
# INFORMATION GEOMETRY: THE SHAPE OF DISTINCTION
# ============================================================

class InformationManifold:
    """
    The space of possible distinction patterns has geometry.

    Key insight: Some distinctions are "closer" than others.
    Distance = how much information separates them.

    This is Fisher information geometry:
    - Points = probability distributions (here: amplitude patterns)
    - Distance = statistical distinguishability
    - Geodesics = natural transformations between patterns

    The sieve operates BY geometry:
    - Rules move you through the manifold
    - Damping shrinks distances (things become same)
    - Interference = superposition in the tangent space
    """

    def __init__(self, dim: int):
        self.dim = dim  # Dimension of the distinction space

    def fisher_distance(
        self,
        field1: Dict[Any, complex],
        field2: Dict[Any, complex]
    ) -> float:
        """
        Fisher information distance between two amplitude patterns.

        This is the "natural" distance in probability/amplitude space.
        It measures: how distinguishable are these two patterns?
        """
        all_keys = set(field1.keys()) | set(field2.keys())
        if not all_keys:
            return 0.0

        # Convert to probabilities
        total1 = sum(abs(a)**2 for a in field1.values()) or 1.0
        total2 = sum(abs(a)**2 for a in field2.values()) or 1.0

        # Bhattacharyya coefficient (related to Fisher distance)
        bc = 0.0
        for k in all_keys:
            p1 = abs(field1.get(k, 0j))**2 / total1
            p2 = abs(field2.get(k, 0j))**2 / total2
            bc += math.sqrt(p1 * p2)

        # Distance = arccos(BC) for Bhattacharyya
        bc = min(bc, 1.0)  # Numerical stability
        return math.acos(bc) if bc < 1.0 else 0.0

    def geodesic(
        self,
        start: Dict[Any, complex],
        end: Dict[Any, complex],
        t: float
    ) -> Dict[Any, complex]:
        """
        Interpolate along geodesic between two patterns.

        t=0: start pattern
        t=1: end pattern
        t=0.5: "halfway" in information space

        This is how patterns naturally transform into each other.
        """
        all_keys = set(start.keys()) | set(end.keys())
        result = {}

        for k in all_keys:
            a = start.get(k, 0j)
            b = end.get(k, 0j)
            # Spherical interpolation (respect amplitude geometry)
            result[k] = a * (1 - t) + b * t  # Linear approximation

        return result


# ============================================================
# THE WHEELER-IT-FROM-BIT SUBSTRATE
# ============================================================

class ItFromBit:
    """
    John Wheeler's radical idea: Physics comes from information.

    "It from bit. Every it — every particle, every field of force,
    even the spacetime continuum itself — derives its function,
    its meaning, its very existence entirely from binary choices,
    bits, yes-no questions."

    This class explores: what if the SUBSTRATE is pure information?

    Structure emerges from questions:
    - Each yes/no question = a distinction
    - Asking a question = creating amplitude in that distinction
    - Answering a question = the distinction stabilizing
    - Objects = stable patterns of answered questions

    The universe as a self-answering set of questions.
    """

    def __init__(self, threshold: float = 1e-10):
        self.threshold = threshold

        # The "it" - configurations that have emerged
        self.entities: Dict[str, Dict[Distinction, complex]] = {}

        # The "bit" - unanswered questions (pending distinctions)
        self.questions: Dict[Distinction, complex] = {}

        # The answers - stabilized distinctions
        self.answers: Dict[Distinction, Side] = {}

        # Time (number of question-answering cycles)
        self.time = 0

    def ask(self, boundary: Any, amplitude: complex = 1.0):
        """Ask a yes/no question (create a distinction)."""
        d = Distinction(boundary)
        self.questions[d] = self.questions.get(d, 0j) + amplitude

    def answer(self, d: Distinction, which: bool):
        """Answer a question (stabilize a distinction)."""
        if d in self.questions:
            del self.questions[d]
        self.answers[d] = Side(d, which)

    def evolve(self) -> Dict[str, Any]:
        """
        One step of it-from-bit evolution.

        Questions with high enough amplitude become answered.
        Answers that lose support become questions again.

        This is the fundamental dynamics of reality.
        """
        stats = {
            'time': self.time,
            'questions': len(self.questions),
            'answers': len(self.answers),
            'entities': len(self.entities)
        }

        new_answers = dict(self.answers)
        new_questions = dict(self.questions)

        # Questions with high amplitude become answers
        for d, amp in list(self.questions.items()):
            if abs(amp) > 0.5:  # Threshold for "answering"
                # Answer based on phase
                which = cmath.phase(amp) > 0
                new_answers[d] = Side(d, which)
                del new_questions[d]

        # Natural decay of unanswered questions
        for d in list(new_questions.keys()):
            new_questions[d] *= 0.99
            if abs(new_questions[d]) < self.threshold:
                del new_questions[d]

        # Check for consistency - contradictory answers destabilize
        # (This is where the sieve-like interference happens)
        # For now: just track

        self.questions = new_questions
        self.answers = new_answers
        self.time += 1

        return stats

    def define_entity(self, name: str, question_pattern: Dict[Distinction, complex]):
        """
        Define an entity as a pattern of distinctions.

        Entities don't exist independently - they're patterns
        in the distinction field. A particle is just a stable
        pattern of answered questions about location, momentum, etc.
        """
        self.entities[name] = question_pattern

    def entity_exists(self, name: str) -> float:
        """
        To what degree does this entity exist?

        Existence isn't binary - it's the amplitude with which
        the entity's question pattern is answered consistently.
        """
        if name not in self.entities:
            return 0.0

        pattern = self.entities[name]
        total = 0.0
        count = 0

        for d, expected_amp in pattern.items():
            if d in self.answers:
                # Check if answer matches expected
                answer = self.answers[d]
                expected_side = cmath.phase(expected_amp) > 0
                if answer.which == expected_side:
                    total += abs(expected_amp)
                else:
                    total -= abs(expected_amp)
                count += 1
            elif d in self.questions:
                # Partially answered
                total += abs(self.questions[d]) * 0.5
                count += 1

        return total / max(count, 1)


# ============================================================
# CATEGORICAL STRUCTURE: MORPHISMS BETWEEN DISTINCTIONS
# ============================================================

class Arrow:
    """
    A morphism (arrow) in the category of distinctions.

    Category theory is about structure-preserving maps.
    Here: maps between distinction patterns that preserve
    the "distinguishing" relationship.

    An arrow from pattern A to pattern B means:
    "If you can distinguish things in A, you can distinguish
    the corresponding things in B."
    """

    def __init__(
        self,
        source: FrozenSet[Distinction],
        target: FrozenSet[Distinction],
        map_fn: Callable[[Distinction], Optional[Distinction]]
    ):
        self.source = source
        self.target = target
        self.map = map_fn

    def apply(self, d: Distinction) -> Optional[Distinction]:
        """Apply the arrow to a distinction."""
        if d in self.source:
            return self.map(d)
        return None

    def compose(self, other: 'Arrow') -> Optional['Arrow']:
        """Compose with another arrow (if compatible)."""
        if self.target != other.source:
            return None

        def composed(d):
            mid = self.apply(d)
            if mid is not None:
                return other.apply(mid)
            return None

        return Arrow(self.source, other.target, composed)


class DistinctionCategory:
    """
    The category of distinctions.

    Objects: Distinction patterns (sets of distinctions)
    Morphisms: Structure-preserving maps between patterns

    This category has special properties:
    - It's self-dual (distinctions about distinctions)
    - It has products (combining distinctions)
    - It has coproducts (choosing between distinctions)
    - It's cartesian closed (function spaces exist)

    This is the foundation for typed computation:
    Types = distinction patterns
    Functions = arrows
    Type checking = arrow existence
    """

    def __init__(self):
        self.objects: Set[FrozenSet[Distinction]] = set()
        self.arrows: Dict[Tuple[FrozenSet[Distinction], FrozenSet[Distinction]], List[Arrow]] = {}

    def add_object(self, pattern: Set[Distinction]):
        """Add a distinction pattern as an object."""
        self.objects.add(frozenset(pattern))

    def add_arrow(self, arrow: Arrow):
        """Add a morphism."""
        key = (arrow.source, arrow.target)
        if key not in self.arrows:
            self.arrows[key] = []
        self.arrows[key].append(arrow)

    def identity(self, pattern: FrozenSet[Distinction]) -> Arrow:
        """Identity arrow (every object has one)."""
        return Arrow(pattern, pattern, lambda d: d)

    def hom_set(
        self,
        source: FrozenSet[Distinction],
        target: FrozenSet[Distinction]
    ) -> List[Arrow]:
        """Get all arrows from source to target."""
        return self.arrows.get((source, target), [])

    def product(
        self,
        a: FrozenSet[Distinction],
        b: FrozenSet[Distinction]
    ) -> FrozenSet[Distinction]:
        """Product of two objects (conjunction of distinctions)."""
        return a | b

    def coproduct(
        self,
        a: FrozenSet[Distinction],
        b: FrozenSet[Distinction]
    ) -> FrozenSet[Distinction]:
        """Coproduct of two objects (disjunction of distinctions)."""
        return a & b


# ============================================================
# THE DEEPEST: SELF-REFERENTIAL INFORMATION
# ============================================================

class SelfReference:
    """
    The deepest structure: information that is about itself.

    Gödel showed: any sufficiently rich formal system can express
    statements about itself.

    Here: the distinction field can contain distinctions ABOUT
    the distinction field itself.

    This is where things get strange:
    - Fixed points: distinctions that are their own value
    - Paradoxes: self-referential distinctions that oscillate
    - Consciousness?: the distinction field "distinguishing itself"

    Self-reference is not a bug - it's the source of depth.
    """

    def __init__(self, base: DistinctionSpace):
        self.base = base
        # Meta-level: distinctions about the base field
        self.meta = DistinctionSpace(threshold=base.threshold)
        # Meta-meta-level: distinctions about the meta field
        self.meta_meta = DistinctionSpace(threshold=base.threshold)
        # Can continue indefinitely, but three levels is enough for now

    def reflect(self):
        """
        Create distinctions about the current state.

        This is the fundamental self-referential operation.
        """
        # For each distinction in base, create a meta-distinction
        # "Is this distinction stable?"
        for d, amp in self.base.field.items():
            stability_d = Distinction(f"stable_{d.boundary}")
            stable_amp = complex(abs(amp), 0)  # Magnitude = stability
            self.meta.distinguish(stability_d, stable_amp)

    def fixed_point(self) -> Optional[Distinction]:
        """
        Find a fixed point: distinction that equals its reflection.

        This is a Gödelian fixed point - a self-describing description.
        """
        for d, amp in self.base.field.items():
            reflected = Distinction(f"stable_{d.boundary}")
            if reflected in self.meta.field:
                reflected_amp = self.meta.field[reflected]
                # Check if they match (within tolerance)
                if abs(amp - reflected_amp) < self.base.threshold:
                    return d
        return None

    def detect_paradox(self) -> List[Distinction]:
        """
        Find paradoxes: distinctions that negate themselves.

        Like the liar paradox: "This statement is false."
        In distinction terms: a distinction whose truth flips its value.
        """
        paradoxes = []
        for d, amp in self.base.field.items():
            # A paradox oscillates: high -> low -> high -> ...
            # We detect this by checking if reflection has opposite phase
            reflected = Distinction(f"stable_{d.boundary}")
            if reflected in self.meta.field:
                reflected_amp = self.meta.field[reflected]
                phase_diff = abs(cmath.phase(amp) - cmath.phase(reflected_amp))
                if abs(phase_diff - math.pi) < 0.3:  # Opposite phase
                    paradoxes.append(d)
        return paradoxes


# ============================================================
# SYNTHESIS: THE INFORMATIONAL SIEVE
# ============================================================

class InformationalSieve:
    """
    The sieve operating at the level of pure information.

    This is below the pattern level - it's the sieve operating
    on distinctions themselves.

    Instead of:
    - Patterns with amplitude (holos_loop/sieve.py)
    - Configurations with amplitude (substrate.py)
    - Entities with amplitude (emergence.py)

    We have:
    - Distinctions with amplitude

    Everything else emerges from stable distinction patterns.
    """

    def __init__(
        self,
        threshold: float = 1e-10,
        damping: float = 0.1,
        self_reflection_rate: float = 0.01
    ):
        self.threshold = threshold
        self.damping = damping
        self.self_reflection_rate = self_reflection_rate

        # The distinction field
        self.distinctions = DistinctionSpace(threshold=threshold)

        # Self-referential layer
        self.self_ref = SelfReference(self.distinctions)

        # Logical layer (emerges from stable distinctions)
        self.logic = LogicalSpace(self.distinctions)

        # Information geometry
        self.geometry = InformationManifold(dim=100)  # Approximate

        # Time
        self.time = 0.0
        self.dt = 0.1

        # History
        self.history: List[Dict[str, Any]] = []

    def inject(self, boundary: Any, amplitude: complex = 1.0):
        """Inject a distinction."""
        self.distinctions.distinguish(boundary, amplitude)

    def evolve(self, dt: float = None) -> Dict[str, Any]:
        """
        One evolution step of the informational sieve.

        1. Apply damping (distinctions fade)
        2. Interference (distinctions combine)
        3. Self-reflection (create meta-distinctions)
        4. Threshold (sub-threshold distinctions vanish)
        """
        dt = dt or self.dt
        stats = {
            'time': self.time,
            'distinctions': len(self.distinctions.field),
            'meta_distinctions': len(self.distinctions.meta_field),
            'entropy': self.distinctions.entropy(),
            'complexity': self.distinctions.complexity(),
        }

        # 1. Damping
        for d in list(self.distinctions.field.keys()):
            self.distinctions.field[d] *= (1 - self.damping * dt)

        # 2. Interference already happens through injection
        # (amplitudes add when same distinction is injected)

        # 3. Self-reflection
        if self.self_reflection_rate > 0:
            # Occasionally reflect on the field
            if self.time % (1.0 / self.self_reflection_rate) < dt:
                self.self_ref.reflect()

        # 4. Threshold
        self.distinctions._cleanup()

        # Check for special structures
        fixed_point = self.self_ref.fixed_point()
        paradoxes = self.self_ref.detect_paradox()

        stats['fixed_point'] = fixed_point is not None
        stats['paradoxes'] = len(paradoxes)

        self.time += dt
        self.history.append(stats)

        return stats

    def stable_distinctions(self, min_amplitude: float = 0.1) -> List[Tuple[Distinction, complex]]:
        """Get distinctions that have stabilized."""
        return [
            (d, a) for d, a in self.distinctions.field.items()
            if abs(a) >= min_amplitude
        ]

    def temperature(self) -> float:
        """Normalized entropy of the distinction field."""
        if len(self.distinctions.field) <= 1:
            return 0.0
        max_entropy = math.log(len(self.distinctions.field))
        return self.distinctions.entropy() / max_entropy if max_entropy > 0 else 0.0

    def summary(self) -> str:
        n_d = len(self.distinctions.field)
        n_m = len(self.distinctions.meta_field)
        return (f"InformationalSieve(t={self.time:.2f}):\n"
                f"  Distinctions: {n_d}\n"
                f"  Meta-distinctions: {n_m}\n"
                f"  Temperature: {self.temperature():.3f}\n"
                f"  Complexity: {self.distinctions.complexity():.3f}")


# ============================================================
# DEMONSTRATION: LOGIC FROM NOTHING
# ============================================================

def bootstrap_logic(verbose: bool = True) -> InformationalSieve:
    """
    Demonstrate how logic emerges from pure distinction.

    Start with nothing.
    Inject the first distinction: something vs nothing.
    Let the sieve evolve.
    See logic emerge.
    """
    sieve = InformationalSieve(damping=0.05, self_reflection_rate=0.1)

    if verbose:
        print("Bootstrapping logic from nothing...")
        print()

    # The first distinction: existence itself
    # "There is something" vs "There is nothing"
    sieve.inject("existence", 1.0)

    if verbose:
        print("Step 1: First distinction (existence)")
        print(f"  {sieve.summary()}")
        print()

    # Evolve a bit
    for _ in range(10):
        sieve.evolve()

    # The first distinction creates the possibility of more
    # Once you can distinguish, you can distinguish distinctions
    sieve.inject("true_false", 0.8)
    sieve.inject("same_different", 0.8)

    if verbose:
        print("Step 2: Core logical distinctions")
        print(f"  {sieve.summary()}")
        print()

    # Evolve more
    for _ in range(50):
        sieve.evolve()

    # Define logical propositions from stable distinctions
    stable = sieve.stable_distinctions(min_amplitude=0.1)

    if verbose:
        print("Step 3: After evolution")
        print(f"  Stable distinctions: {len(stable)}")
        for d, a in stable:
            print(f"    {d.boundary}: {abs(a):.3f}")
        print()

    # Logic has emerged!
    # The stable distinctions are the "atoms" of logic
    # Their interference patterns are the logical operations

    if verbose:
        print("Logic has bootstrapped from pure distinction!")
        print()

    return sieve
