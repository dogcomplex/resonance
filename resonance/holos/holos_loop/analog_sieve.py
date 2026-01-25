"""
holos_loop/analog_sieve.py - Continuous Field Sieve

The discrete sieve is a sampled version of this continuous field.

In the analog sieve:
- Patterns are points in a continuous manifold
- Amplitude is a complex field ψ(x) over the manifold
- Rules become differential operators
- Evolution follows a wave equation

The physics:
    ∂ψ/∂t = -i·H·ψ - γ·ψ + D·∇²ψ + sources

Where:
- H = Hamiltonian (encodes the rules as potential energy landscape)
- γ = damping (the sieve threshold, energy loss)
- D = diffusion (smoothing, allows exploration)
- sources = seed injections

Closures occur where:
- Forward wave (phase 0) meets backward wave (phase π)
- They have comparable magnitudes
- The combined amplitude is above threshold

This is the continuous limit of what the discrete sieve does.
"""

import math
import cmath
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from abc import ABC, abstractmethod


# ============================================================
# MANIFOLD ABSTRACTION
# ============================================================

class Manifold(ABC):
    """
    Abstract manifold - the space where patterns live.

    Different games have different natural manifolds:
    - Connect4: 7^6 discrete points (column heights)
    - Chess: High-dimensional discrete space
    - Continuous control: Actual continuous manifold

    The manifold provides:
    - Distance metric (how far apart are states?)
    - Neighbors (what's "nearby" in state space?)
    - Coordinates (how to parameterize states)
    """

    @abstractmethod
    def distance(self, x: Any, y: Any) -> float:
        """Distance between two points"""
        pass

    @abstractmethod
    def neighbors(self, x: Any, radius: float) -> List[Any]:
        """Points within radius of x"""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the manifold"""
        pass


class GridManifold(Manifold):
    """
    Simple grid manifold for discrete games treated as continuous.

    Points are integer tuples, but we can interpolate between them.
    """

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        self.ndim = len(shape)

    def distance(self, x: Tuple[int, ...], y: Tuple[int, ...]) -> float:
        """Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

    def neighbors(self, x: Tuple[int, ...], radius: float = 1.5) -> List[Tuple[int, ...]]:
        """Grid points within radius (including diagonals for radius > 1)"""
        neighbors = []
        r_int = int(math.ceil(radius))

        # Generate all points in hypercube, filter by distance
        ranges = [range(max(0, x[i] - r_int), min(self.shape[i], x[i] + r_int + 1))
                  for i in range(self.ndim)]

        from itertools import product
        for point in product(*ranges):
            if point != x and self.distance(x, point) <= radius:
                neighbors.append(point)

        return neighbors

    def dimension(self) -> int:
        return self.ndim

    def all_points(self) -> List[Tuple[int, ...]]:
        """Enumerate all grid points"""
        from itertools import product
        return list(product(*[range(s) for s in self.shape]))


class EmbeddedManifold(Manifold):
    """
    Manifold where points are embedded in a vector space.

    Each state maps to a vector, and we use vector space distance.
    """

    def __init__(self, embed_fn: Callable[[Any], np.ndarray], dim: int):
        self.embed_fn = embed_fn
        self._dim = dim

    def distance(self, x: Any, y: Any) -> float:
        vx = self.embed_fn(x)
        vy = self.embed_fn(y)
        return np.linalg.norm(vx - vy)

    def neighbors(self, x: Any, radius: float) -> List[Any]:
        # Requires knowing the state space - not implemented for general case
        raise NotImplementedError("EmbeddedManifold requires explicit state enumeration")

    def dimension(self) -> int:
        return self._dim


# ============================================================
# CONTINUOUS FIELD
# ============================================================

@dataclass
class FieldPoint:
    """A point in the field with its amplitude"""
    coords: Any  # Coordinates on the manifold
    amplitude: complex
    velocity: complex = 0j  # For wave equation (∂ψ/∂t)

    @property
    def magnitude(self) -> float:
        return abs(self.amplitude)

    @property
    def phase(self) -> float:
        return cmath.phase(self.amplitude)


class AnalogSieve:
    """
    Continuous amplitude field over a manifold.

    The field evolves according to a wave equation with damping:
    ∂²ψ/∂t² = c²∇²ψ - γ∂ψ/∂t + V(x)ψ + sources

    Or in first-order form:
    ∂ψ/∂t = v
    ∂v/∂t = c²∇²ψ - γv + V(x)ψ + sources

    Where:
    - c = wave speed
    - γ = damping coefficient
    - V(x) = potential (encodes rules as energy landscape)
    - sources = seed injections

    The sieve operates by:
    - Damping removes energy globally
    - Potential shapes where amplitude concentrates
    - Interference creates nodes (zero amplitude) and antinodes (high amplitude)
    - Stable patterns = standing waves
    """

    def __init__(
        self,
        manifold: Manifold,
        wave_speed: float = 1.0,
        damping: float = 0.1,
        diffusion: float = 0.1,
        threshold: float = 0.001
    ):
        self.manifold = manifold
        self.c = wave_speed
        self.gamma = damping
        self.D = diffusion
        self.threshold = threshold

        # Field storage (sparse - only store non-zero points)
        self.field: Dict[Any, FieldPoint] = {}

        # Potential function (defines the "rules" as energy landscape)
        self.potential: Callable[[Any], complex] = lambda x: 0j

        # Source function (ongoing injections)
        self.sources: Dict[Any, complex] = {}

        # Statistics
        self.time = 0.0
        self.dt = 0.1  # Time step

    def set_potential(self, V: Callable[[Any], complex]):
        """
        Set the potential function.

        V(x) > 0: Point x is "uphill" (unstable, amplitude flows away)
        V(x) < 0: Point x is "downhill" (stable, amplitude flows toward)
        V(x) = 0: Neutral

        For games: V(x) = 0 for legal states, V(x) = ∞ for illegal states
        """
        self.potential = V

    def inject(self, coords: Any, amplitude: complex):
        """Inject amplitude at a point"""
        if coords not in self.field:
            self.field[coords] = FieldPoint(coords, 0j, 0j)
        self.field[coords].amplitude += amplitude

    def inject_source(self, coords: Any, rate: complex):
        """Add a continuous source at a point"""
        self.sources[coords] = rate

    def remove_source(self, coords: Any):
        """Remove a continuous source"""
        if coords in self.sources:
            del self.sources[coords]

    def evolve(self, dt: float = None) -> Dict[str, Any]:
        """
        One time step of field evolution.

        Uses a simple finite-difference scheme.
        """
        dt = dt or self.dt
        stats = {
            'time': self.time,
            'points': len(self.field),
            'total_amplitude': 0.0,
            'max_amplitude': 0.0,
        }

        new_field: Dict[Any, FieldPoint] = {}

        # Get all points to update (existing + neighbors for diffusion)
        points_to_update = set(self.field.keys())
        for coords in list(self.field.keys()):
            if self.field[coords].magnitude > self.threshold:
                try:
                    neighbors = self.manifold.neighbors(coords, 1.5)
                    points_to_update.update(neighbors)
                except NotImplementedError:
                    pass

        for coords in points_to_update:
            # Get current state
            if coords in self.field:
                current = self.field[coords]
                psi = current.amplitude
                v = current.velocity
            else:
                psi = 0j
                v = 0j

            # Compute Laplacian (∇²ψ) from neighbors
            laplacian = 0j
            try:
                neighbors = self.manifold.neighbors(coords, 1.5)
                if neighbors:
                    neighbor_sum = sum(
                        self.field[n].amplitude if n in self.field else 0j
                        for n in neighbors
                    )
                    laplacian = (neighbor_sum - len(neighbors) * psi) / len(neighbors)
            except NotImplementedError:
                pass

            # Potential at this point
            V = self.potential(coords)

            # Source at this point
            source = self.sources.get(coords, 0j)

            # Wave equation: ∂v/∂t = c²∇²ψ - γv + Vψ + source
            dv_dt = self.c**2 * laplacian - self.gamma * v + V * psi + source

            # Update velocity and amplitude
            new_v = v + dv_dt * dt
            new_psi = psi + new_v * dt

            # Apply diffusion separately (smoothing)
            if self.D > 0 and abs(laplacian) > 0:
                new_psi += self.D * laplacian * dt

            # Store if above threshold
            if abs(new_psi) >= self.threshold:
                new_field[coords] = FieldPoint(coords, new_psi, new_v)
                stats['total_amplitude'] += abs(new_psi)
                stats['max_amplitude'] = max(stats['max_amplitude'], abs(new_psi))

        self.field = new_field
        self.time += dt
        stats['points'] = len(self.field)

        return stats

    def detect_closures(self, phase_tolerance: float = 0.5) -> List[Tuple[Any, complex]]:
        """
        Detect closure points.

        Closures are where:
        - Amplitude is significant (above threshold * 10)
        - Phase is near 0 or π (forward-backward meeting)
        - Neighbors have different phases (interference)
        """
        closures = []

        for coords, point in self.field.items():
            if point.magnitude < self.threshold * 10:
                continue

            # Check if phase indicates closure (0 or π = forward/backward)
            phase = abs(point.phase)
            if phase < phase_tolerance or abs(phase - math.pi) < phase_tolerance:
                # Check neighbors for phase diversity
                try:
                    neighbors = self.manifold.neighbors(coords, 1.5)
                    neighbor_phases = [
                        self.field[n].phase for n in neighbors
                        if n in self.field and self.field[n].magnitude > self.threshold
                    ]

                    if neighbor_phases:
                        phase_spread = max(neighbor_phases) - min(neighbor_phases)
                        if phase_spread > math.pi / 2:
                            # Significant phase spread = interference point
                            closures.append((coords, point.amplitude))
                except NotImplementedError:
                    # Can't check neighbors - just use amplitude
                    closures.append((coords, point.amplitude))

        return closures

    def temperature(self) -> float:
        """
        Effective temperature of the field.

        Based on amplitude entropy.
        """
        magnitudes = [p.magnitude for p in self.field.values()]
        if not magnitudes:
            return 0.0

        total = sum(magnitudes)
        if total == 0:
            return 0.0

        probs = [m / total for m in magnitudes]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        max_entropy = math.log(len(magnitudes) + 1)

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def total_energy(self) -> float:
        """
        Total energy in the field.

        E = ∫ (|∂ψ/∂t|² + c²|∇ψ|² + V|ψ|²) dx

        Approximated as sum over points.
        """
        energy = 0.0
        for coords, point in self.field.items():
            kinetic = abs(point.velocity) ** 2
            potential = abs(self.potential(coords)) * point.magnitude ** 2
            energy += kinetic + potential
        return energy

    def sample(self, coords_list: List[Any]) -> Dict[Any, complex]:
        """Sample the field at specific coordinates"""
        return {
            coords: self.field[coords].amplitude if coords in self.field else 0j
            for coords in coords_list
        }

    def summary(self) -> str:
        """Human-readable summary"""
        return (f"AnalogSieve (t={self.time:.2f}):\n"
                f"  Points: {len(self.field)}\n"
                f"  Total amplitude: {sum(p.magnitude for p in self.field.values()):.3f}\n"
                f"  Temperature: {self.temperature():.3f}\n"
                f"  Energy: {self.total_energy():.3f}")


# ============================================================
# DISCRETIZATION: ANALOG -> DISCRETE
# ============================================================

def discretize_field(
    analog: AnalogSieve,
    grid_points: List[Any],
    threshold: float = 0.01
) -> Dict[Any, complex]:
    """
    Sample the analog field at discrete points.

    This converts the continuous field back to discrete patterns
    for use with the discrete sieve.
    """
    samples = analog.sample(grid_points)
    return {coords: amp for coords, amp in samples.items() if abs(amp) >= threshold}


def continuize_patterns(
    patterns: Dict[Any, complex],
    manifold: Manifold,
    smoothing: float = 0.5
) -> AnalogSieve:
    """
    Create an analog field from discrete patterns.

    The discrete patterns become point sources, then we smooth.
    """
    analog = AnalogSieve(manifold, diffusion=smoothing)

    for coords, amplitude in patterns.items():
        analog.inject(coords, amplitude)

    # Run a few smoothing steps
    for _ in range(5):
        analog.evolve()

    return analog


# ============================================================
# GAME POTENTIAL: RULES AS ENERGY LANDSCAPE
# ============================================================

def create_game_potential(
    legal_states: set,
    terminal_states: Dict[Any, float],  # state -> value
    illegal_penalty: float = 1e6
) -> Callable[[Any], complex]:
    """
    Create a potential function from game rules.

    - Legal states: V = 0 (free propagation)
    - Illegal states: V = large positive (barrier)
    - Terminal states: V = -value (attracts toward wins, repels from losses)
    """
    def V(coords):
        if coords not in legal_states:
            return complex(illegal_penalty, 0)

        if coords in terminal_states:
            value = terminal_states[coords]
            # Wins (value > 0) are attractive (V < 0)
            # Losses (value < 0) are repulsive (V > 0)
            return complex(-value * 10, 0)

        return 0j

    return V
