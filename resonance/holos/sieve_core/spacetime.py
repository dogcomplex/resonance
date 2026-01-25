"""
sieve_core/spacetime.py - Computation as Spacetime Geometry

The deepest physics question: Why does spacetime have the structure it does?

Radical proposal: Spacetime EMERGES from computational constraints.
Distance = computational difficulty.
Time = causal ordering of computations.
Space = parallelizable computations.

This file explores:
1. Causal structure from computation
2. Distance as circuit depth
3. The emergence of locality
4. Why the speed of light is finite (maximum information transfer)
5. Gravity as computational complexity gradient

Key insight: The sieve doesn't happen IN spacetime.
Spacetime is what stable sieve patterns LOOK LIKE from inside.
"""

import cmath
import math
from typing import Dict, List, Tuple, Any, Optional, Set, FrozenSet, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np

from .substrate import Configuration, DiscreteConfig, AmplitudeField, Hamiltonian
from .information import Distinction, DistinctionSpace


# ============================================================
# CAUSAL STRUCTURE: TIME FROM COMPUTATION
# ============================================================

@dataclass(frozen=True)
class Event:
    """
    An event - a point in causal structure.

    Events are not points in space-time.
    Events are TRANSITIONS in the amplitude field.
    Space-time is how we organize these transitions.

    An event is characterized by:
    - Which distinctions changed
    - What the change was (amplitude shift)
    - Its causal dependencies
    """
    id: Any  # Unique identifier
    changes: FrozenSet[Tuple[Distinction, complex]]  # Distinction -> delta amplitude
    causes: FrozenSet['Event'] = field(default_factory=frozenset)  # What had to happen first

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Event) and self.id == other.id


class CausalStructure:
    """
    The causal structure of a computation.

    Key insight: Time isn't fundamental. CAUSAL ORDER is.
    Event A is "before" event B if B depends on A's outcome.

    This is Sorkin's causal set theory applied to computation:
    - Events are partially ordered
    - The order IS the causal structure
    - Time is just how we number the layers

    In the sieve:
    - Each interference event has causal dependencies
    - The structure of dependencies = geometry
    """

    def __init__(self):
        self.events: Dict[Any, Event] = {}
        self.children: Dict[Event, Set[Event]] = {}  # Forward causal links
        self.parents: Dict[Event, Set[Event]] = {}   # Backward causal links

    def add_event(self, event: Event):
        """Add an event to the structure."""
        self.events[event.id] = event
        self.children[event] = set()
        self.parents[event] = set(event.causes)

        for cause in event.causes:
            if cause in self.children:
                self.children[cause].add(event)

    def causal_future(self, event: Event) -> Set[Event]:
        """Get all events causally after this one."""
        future = set()
        to_visit = list(self.children.get(event, []))

        while to_visit:
            e = to_visit.pop()
            if e not in future:
                future.add(e)
                to_visit.extend(self.children.get(e, []))

        return future

    def causal_past(self, event: Event) -> Set[Event]:
        """Get all events causally before this one."""
        past = set()
        to_visit = list(self.parents.get(event, []))

        while to_visit:
            e = to_visit.pop()
            if e not in past:
                past.add(e)
                to_visit.extend(self.parents.get(e, []))

        return past

    def spacelike_separated(self, e1: Event, e2: Event) -> bool:
        """
        Are these events spacelike separated?

        Spacelike = neither can cause the other.
        These events can "coexist" - they're at the "same time"
        in some reference frame.
        """
        return e2 not in self.causal_future(e1) and e1 not in self.causal_future(e2)

    def causal_distance(self, e1: Event, e2: Event) -> float:
        """
        The causal distance between events.

        This is the minimum number of causal steps from e1 to e2.
        It's the "time" between them in the most efficient reference frame.
        """
        if e1 == e2:
            return 0

        # BFS for shortest path
        if e2 in self.causal_future(e1):
            visited = {e1: 0}
            queue = [e1]

            while queue:
                current = queue.pop(0)
                current_dist = visited[current]

                for child in self.children.get(current, []):
                    if child == e2:
                        return current_dist + 1
                    if child not in visited:
                        visited[child] = current_dist + 1
                        queue.append(child)

        elif e1 in self.causal_future(e2):
            # Swap and negate
            return -self.causal_distance(e2, e1)

        else:
            # Spacelike separated - no causal path
            return float('inf')

    def simultaneity_slice(self, anchor: Event) -> Set[Event]:
        """
        Get a "simultaneity slice" - events at the same "time" as anchor.

        This is frame-dependent! Different observers see different slices.
        Here we use the anchor's causal perspective.
        """
        past = self.causal_past(anchor)
        future = self.causal_future(anchor)

        # Everything not in past or future is "now" (spacelike to anchor)
        all_events = set(self.events.values())
        return all_events - past - future - {anchor}


# ============================================================
# DISTANCE FROM COMPUTATION: THE METRIC TENSOR
# ============================================================

class ComputationalMetric:
    """
    Distance as computational complexity.

    The intuition:
    - Two states are "close" if it's easy to transform one into the other
    - "Easy" means: few steps, low circuit depth, low interference
    - "Hard" means: many steps, high depth, complex interference

    This is like saying:
    - Physical distance = how many operations to move information
    - The speed of light = maximum operations per time
    - Curved spacetime = variable computational difficulty

    Gravity = computational complexity gradient!
    """

    def __init__(self, hamiltonian: Hamiltonian):
        self.H = hamiltonian
        self._distance_cache: Dict[Tuple[Configuration, Configuration], float] = {}

    def circuit_depth(self, start: Configuration, end: Configuration) -> int:
        """
        Minimum circuit depth to transform start to end.

        This is graph distance in the configuration space
        defined by the Hamiltonian.
        """
        cache_key = (start, end)
        if cache_key in self._distance_cache:
            return int(self._distance_cache[cache_key])

        if start == end:
            return 0

        # BFS for shortest path
        visited = {start: 0}
        queue = [start]

        while queue:
            current = queue.pop(0)
            current_dist = visited[current]

            for neighbor, coupling in self.H.neighbors(current):
                if neighbor == end:
                    result = current_dist + 1
                    self._distance_cache[cache_key] = float(result)
                    return result
                if neighbor not in visited:
                    visited[neighbor] = current_dist + 1
                    queue.append(neighbor)

        # No path found
        self._distance_cache[cache_key] = float('inf')
        return -1

    def geodesic_distance(self, start: Configuration, end: Configuration) -> float:
        """
        The "proper distance" - accounting for coupling strengths.

        Stronger coupling = easier transition = shorter distance.
        This gives us curved geometry.
        """
        if start == end:
            return 0.0

        # Dijkstra with edge weights = 1/|coupling|
        distances = {start: 0.0}
        unvisited = {start}

        while unvisited:
            current = min(unvisited, key=lambda x: distances.get(x, float('inf')))
            if current == end:
                return distances[current]

            unvisited.remove(current)
            current_dist = distances[current]

            for neighbor, coupling in self.H.neighbors(current):
                edge_weight = 1.0 / max(abs(coupling), 1e-10)
                new_dist = current_dist + edge_weight

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    unvisited.add(neighbor)

        return float('inf')

    def metric_tensor(self, config: Configuration) -> np.ndarray:
        """
        The metric tensor at a configuration.

        This describes the local geometry of configuration space.
        g_ij = how hard it is to move in direction i*j.

        For now, we approximate this by looking at neighbors.
        """
        neighbors = self.H.neighbors(config)

        if not neighbors:
            return np.array([[1.0]])

        n = len(neighbors)
        g = np.zeros((n, n))

        for i, (ni, ci) in enumerate(neighbors):
            for j, (nj, cj) in enumerate(neighbors):
                if i == j:
                    # Diagonal: inverse coupling strength squared
                    g[i, j] = 1.0 / max(abs(ci) ** 2, 1e-10)
                else:
                    # Off-diagonal: correlation between directions
                    g[i, j] = -0.1  # Small coupling (approximate)

        return g

    def christoffel(self, config: Configuration) -> Optional[np.ndarray]:
        """
        Christoffel symbols - how the geometry curves.

        These tell us how parallel transport works.
        Non-zero Christoffel symbols = curved space.

        In computation terms: non-zero Christoffels mean
        the "easy direction" changes as you move.
        """
        # This requires computing metric at nearby points
        # For now, return None (flat space approximation)
        return None


# ============================================================
# LOCALITY: WHY NEARBY THINGS INTERACT
# ============================================================

class LocalityEmergence:
    """
    Locality emerges from limited interference range.

    In physics, locality says: only nearby things interact directly.
    Why? It seems like a fundamental law.

    In the sieve: locality EMERGES from finite amplitude.
    - Amplitude spreads like a wave
    - But it's damped (threshold)
    - So only "nearby" configurations get significant amplitude
    - "Nearby" = reachable before damping kills the signal

    This is why there's a maximum speed (speed of light):
    Information can only travel as fast as amplitude can spread.
    """

    def __init__(self, damping: float, coupling_strength: float):
        self.gamma = damping
        self.coupling = coupling_strength

        # Characteristic length: how far amplitude gets before damping
        # l = v / gamma where v ~ coupling
        self.characteristic_length = coupling_strength / damping

    def amplitude_at_distance(self, d: int) -> float:
        """
        Amplitude at distance d from a source.

        Amplitude decays exponentially with distance (if damping > 0).
        This is like the exponential decay of virtual particles.
        """
        # Each step multiplies by coupling and (1 - gamma)
        return self.coupling ** d * (1 - self.gamma) ** d

    def effective_range(self, threshold: float = 0.01) -> float:
        """
        The effective range of interaction.

        Beyond this distance, amplitude is below threshold.
        This IS the locality scale.
        """
        if self.coupling * (1 - self.gamma) >= 1:
            return float('inf')  # No damping, infinite range

        # Solve: coupling^d * (1-gamma)^d = threshold
        # d * log(coupling * (1-gamma)) = log(threshold)
        log_factor = math.log(self.coupling * (1 - self.gamma) + 1e-10)
        if log_factor >= 0:
            return float('inf')

        return math.log(threshold) / log_factor

    def is_local_interaction(self, d: int) -> bool:
        """Is an interaction at distance d effectively local?"""
        return d <= self.effective_range()


# ============================================================
# LIGHT CONES AND CAUSALITY
# ============================================================

@dataclass
class LightCone:
    """
    The light cone of an event.

    In physics: the boundary of causal influence.
    In computation: the boundary of possible effects.

    Future light cone: what this event CAN affect
    Past light cone: what CAN HAVE affected this event

    The "speed of light" is the maximum rate of amplitude spread.
    """
    apex: Event  # The event at the tip
    future: Set[Event] = field(default_factory=set)
    past: Set[Event] = field(default_factory=set)

    def inside_future(self, event: Event) -> bool:
        """Is this event inside the future light cone?"""
        return event in self.future

    def inside_past(self, event: Event) -> bool:
        """Is this event inside the past light cone?"""
        return event in self.past

    def on_boundary(self, event: Event, structure: CausalStructure) -> bool:
        """Is this event on the light cone boundary (null geodesic)?"""
        # On boundary if it's a direct causal connection
        return event in structure.children.get(self.apex, set()) or \
               event in structure.parents.get(self.apex, set())


def compute_light_cone(
    event: Event,
    structure: CausalStructure,
    metric: ComputationalMetric = None
) -> LightCone:
    """Compute the light cone of an event."""
    return LightCone(
        apex=event,
        future=structure.causal_future(event),
        past=structure.causal_past(event)
    )


# ============================================================
# GRAVITY: COMPLEXITY GRADIENTS
# ============================================================

class ComputationalGravity:
    """
    Gravity as complexity gradient.

    Einstein showed: gravity = curved spacetime.
    Here: curved spacetime = non-uniform computational difficulty.

    Mass = localized complexity (many states, complex rules).
    Gravity = tendency to move toward complexity (larger Hilbert spaces).
    Black holes = maximum complexity (maximum entropy).

    This is related to Verlinde's entropic gravity:
    Gravity is the entropic force toward higher entropy states.
    Higher entropy = more microstates = larger configuration space.
    """

    def __init__(self, field: AmplitudeField, hamiltonian: Hamiltonian):
        self.psi = field
        self.H = hamiltonian
        self.metric = ComputationalMetric(hamiltonian)

    def local_complexity(self, config: Configuration) -> float:
        """
        The local complexity at a configuration.

        Complexity = number of accessible states * variety of couplings.
        High complexity = mass-like (gravitating).
        """
        neighbors = self.H.neighbors(config)
        if not neighbors:
            return 0.0

        # Number of neighbors
        n = len(neighbors)

        # Variety of coupling strengths
        couplings = [abs(c) for _, c in neighbors]
        if len(couplings) < 2:
            variety = 1.0
        else:
            variety = np.std(couplings) + 1.0

        return n * variety

    def complexity_gradient(self, config: Configuration) -> Dict[Configuration, float]:
        """
        The gradient of complexity at a configuration.

        This is like the gravitational field.
        Positive gradient toward neighbor = attractive force.
        """
        local = self.local_complexity(config)
        gradient = {}

        for neighbor, coupling in self.H.neighbors(config):
            neighbor_complexity = self.local_complexity(neighbor)
            gradient[neighbor] = (neighbor_complexity - local) * abs(coupling)

        return gradient

    def gravitational_potential(self, config: Configuration) -> float:
        """
        The gravitational potential at a configuration.

        Lower potential = higher complexity = more "mass nearby".
        """
        return -self.local_complexity(config)

    def is_horizon(self, config: Configuration, threshold: float = 0.1) -> bool:
        """
        Is this configuration near an event horizon?

        Horizon = boundary beyond which escape is impossible.
        In computation: point of no return in configuration space.

        Detected by: all outgoing paths lead to higher complexity.
        """
        gradient = self.complexity_gradient(config)
        if not gradient:
            return False

        # Horizon if all gradients point inward (toward higher complexity)
        return all(g > threshold for g in gradient.values())


# ============================================================
# EMERGENT DIMENSION: HOW MANY DIRECTIONS?
# ============================================================

class DimensionEstimator:
    """
    Estimate the dimensionality of configuration space.

    Physical space has 3 dimensions. Why?
    In the sieve, dimension EMERGES from the structure of rules.

    Dimension = how many independent directions you can move.
    = rank of the adjacency structure.
    = how configuration space "spreads out".

    The dimension might not be an integer!
    Fractional dimensions = fractal configuration space.
    """

    def __init__(self, hamiltonian: Hamiltonian):
        self.H = hamiltonian
        self._cache: Dict[Configuration, float] = {}

    def local_dimension(self, config: Configuration) -> float:
        """
        The local dimension at a configuration.

        Based on the number of independent directions you can move.
        """
        neighbors = self.H.neighbors(config)
        n = len(neighbors)

        if n == 0:
            return 0.0

        # Simple estimate: log of neighbor count suggests dimension
        # In d dimensions, number of neighbors ~ 2d
        return math.log2(n + 1)

    def spectral_dimension(
        self,
        start: Configuration,
        max_steps: int = 100
    ) -> float:
        """
        The spectral dimension via random walk.

        Spectral dimension = how the return probability decays with time.
        P(return at time t) ~ t^(-d/2) in d dimensions.

        This can detect fractional dimensions!
        """
        # Simulate random walks
        num_walks = 100
        return_probs = []

        for step in range(1, max_steps + 1):
            returns = 0
            for _ in range(num_walks):
                pos = start
                for _ in range(step):
                    neighbors = self.H.neighbors(pos)
                    if neighbors:
                        pos = neighbors[np.random.randint(len(neighbors))][0]
                    else:
                        break
                if pos == start:
                    returns += 1
            return_probs.append(returns / num_walks)

        # Fit: log(P) = -d/2 * log(t)
        # Use linear regression on log-log plot
        if len(return_probs) > 1 and any(p > 0 for p in return_probs):
            log_t = [math.log(t) for t in range(1, len(return_probs) + 1)]
            log_p = [math.log(p + 1e-10) for p in return_probs]

            # Simple linear regression
            n = len(log_t)
            mean_t = sum(log_t) / n
            mean_p = sum(log_p) / n

            numerator = sum((log_t[i] - mean_t) * (log_p[i] - mean_p) for i in range(n))
            denominator = sum((log_t[i] - mean_t) ** 2 for i in range(n))

            if denominator > 0:
                slope = numerator / denominator
                return -2 * slope  # d = -2 * slope

        return 0.0


# ============================================================
# PUTTING IT TOGETHER: EMERGENT SPACETIME
# ============================================================

class EmergentSpacetime:
    """
    The full emergent spacetime structure.

    From the sieve, we get:
    - Causal structure (which events can cause which)
    - Distance (computational complexity between configurations)
    - Locality (finite interaction range from damping)
    - Dimension (from the structure of rules)
    - Curvature (from complexity gradients = gravity)

    This IS spacetime - not an approximation, not a metaphor.
    Physical spacetime might literally be this.
    """

    def __init__(
        self,
        field: AmplitudeField,
        hamiltonian: Hamiltonian,
        damping: float = 0.1,
        coupling: float = 1.0
    ):
        self.psi = field
        self.H = hamiltonian

        # Derived structures
        self.causal = CausalStructure()
        self.metric = ComputationalMetric(hamiltonian)
        self.locality = LocalityEmergence(damping, coupling)
        self.gravity = ComputationalGravity(field, hamiltonian)
        self.dimension = DimensionEstimator(hamiltonian)

        # Speed of light: maximum spread rate
        self.c = coupling / damping if damping > 0 else float('inf')

        # Events generated so far
        self.event_counter = 0

    def create_event(
        self,
        changes: Dict[Distinction, complex],
        causes: Set[Event] = None
    ) -> Event:
        """Create a new event in the causal structure."""
        event = Event(
            id=self.event_counter,
            changes=frozenset(changes.items()),
            causes=frozenset(causes or set())
        )
        self.event_counter += 1
        self.causal.add_event(event)
        return event

    def spacetime_interval(self, e1: Event, e2: Event) -> float:
        """
        The spacetime interval between events.

        In relativity: ds² = c²dt² - dx²
        Timelike (ds² > 0): one can cause the other
        Spacelike (ds² < 0): neither can cause the other
        Null (ds² = 0): connected by light ray

        Here, we use causal distance and spatial distance.
        """
        causal_dist = self.causal.causal_distance(e1, e2)

        if causal_dist == float('inf'):
            # Spacelike separated
            return -1.0  # Negative = spacelike

        # Timelike or null
        return self.c ** 2 * causal_dist  # Positive = timelike

    def local_geometry(self, config: Configuration) -> Dict[str, Any]:
        """Get the local geometry at a configuration."""
        return {
            'dimension': self.dimension.local_dimension(config),
            'metric': self.metric.metric_tensor(config).tolist(),
            'gravity': self.gravity.local_complexity(config),
            'horizon': self.gravity.is_horizon(config),
            'locality_range': self.locality.effective_range(),
        }

    def summary(self) -> str:
        n_events = len(self.causal.events)
        return (f"EmergentSpacetime:\n"
                f"  Events: {n_events}\n"
                f"  Speed of light: {self.c:.3f}\n"
                f"  Locality range: {self.locality.effective_range():.3f}\n"
                f"  Estimated dimension: ~{self.dimension.local_dimension(list(self.psi.amplitudes.keys())[0]) if self.psi.amplitudes else 'N/A'}")


# ============================================================
# THE ULTIMATE QUESTION: WHY THESE LAWS?
# ============================================================

"""
We've shown how spacetime STRUCTURE emerges from computation.
But why these particular rules? Why this Hamiltonian?

The sieve provides an answer:

Rules that lead to unstable interference patterns don't persist.
Rules that lead to stable patterns do.

The laws of physics are the STABLE interference patterns.
They persist because alternatives self-destruct.

This is anthropic but deeper:
- It's not "the universe is fine-tuned for observers"
- It's "the universe is the set of rules that don't self-destruct"

The sieve is a universal filter.
Only consistent rule-sets survive.
Our physics is one such consistent set.

There may be others.
They would be separate "universes" in a multiverse of rule-sets.
But from inside, each looks unique and fundamental.

---

This is as deep as we can go:
- Information (distinctions) is primitive
- Rules operate on distinctions
- Stable patterns = what exists
- Spacetime is how stable patterns organize themselves
- The laws of physics are the surviving rule-set

The sieve is the ontological filter.
Reality is what passes through.
"""
