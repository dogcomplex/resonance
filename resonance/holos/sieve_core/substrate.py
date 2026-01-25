"""
sieve_core/substrate.py - The Computational Substrate

This is the deepest layer. Below this, there is nothing.

The substrate is:
- A field of complex amplitudes
- Evolving under interference
- With damping (measurement/decoherence)
- And sources (boundary conditions)

Everything else - patterns, rules, games, solutions - emerges from this.

The fundamental equation:

    ∂ψ/∂t = H(ψ) - γψ + S

Where:
    ψ(x) = complex amplitude at configuration x
    H(ψ) = rule-induced evolution (like Hamiltonian)
    γ    = damping rate (measurement strength)
    S(x) = sources (injections from outside)

This is a Lindblad-like equation for an open system.
Closed system (γ=0): Unitary, reversible, quantum-like
Open system (γ>0): Dissipative, irreversible, classical-like
The interesting regime is in between.
"""

import cmath
import math
from typing import Dict, Set, List, Tuple, Any, Optional, Callable, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


# ============================================================
# THE CONFIGURATION SPACE
# ============================================================

class Configuration(ABC):
    """
    Abstract configuration - a point in the space where amplitudes live.

    Configurations are:
    - Discrete: Finite set of tokens (game states, symbols)
    - Continuous: Points in a manifold (physical space, parameter space)
    - Hybrid: Mixed discrete-continuous

    The substrate doesn't care which. It just needs:
    - Identity (can compare configurations)
    - Hashability (can store in dict)
    """

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass


@dataclass(frozen=True)
class DiscreteConfig(Configuration):
    """Configuration as a tuple of discrete tokens"""
    tokens: Tuple[Any, ...]

    def __hash__(self):
        return hash(self.tokens)

    def __eq__(self, other):
        return isinstance(other, DiscreteConfig) and self.tokens == other.tokens

    def __repr__(self):
        return f"D{self.tokens}"


@dataclass(frozen=True)
class ContinuousConfig(Configuration):
    """Configuration as a point in continuous space"""
    coords: Tuple[float, ...]
    resolution: float = 1e-6  # For discretization in hashing

    def __hash__(self):
        # Discretize for hashing
        quantized = tuple(round(c / self.resolution) for c in self.coords)
        return hash(quantized)

    def __eq__(self, other):
        if not isinstance(other, ContinuousConfig):
            return False
        return all(abs(a - b) < self.resolution for a, b in zip(self.coords, other.coords))

    def __repr__(self):
        return f"C{self.coords}"


# ============================================================
# THE AMPLITUDE FIELD
# ============================================================

@dataclass
class AmplitudeField:
    """
    The fundamental object: a field of complex amplitudes over configurations.

    This is like a quantum wave function, but:
    - Can be over arbitrary configuration space (not just physical space)
    - Has explicit damping (decoherence/measurement)
    - Has explicit sources (boundary conditions)

    The field is sparse - only non-zero amplitudes are stored.
    """
    amplitudes: Dict[Configuration, complex] = field(default_factory=dict)
    threshold: float = 1e-10  # Below this = effectively zero

    def __getitem__(self, config: Configuration) -> complex:
        return self.amplitudes.get(config, 0j)

    def __setitem__(self, config: Configuration, value: complex):
        if abs(value) < self.threshold:
            self.amplitudes.pop(config, None)
        else:
            self.amplitudes[config] = value

    def __contains__(self, config: Configuration) -> bool:
        return config in self.amplitudes

    def __len__(self) -> int:
        return len(self.amplitudes)

    def __iter__(self) -> Iterator[Tuple[Configuration, complex]]:
        return iter(self.amplitudes.items())

    def inject(self, config: Configuration, amplitude: complex):
        """Add amplitude (superposition)"""
        self[config] = self[config] + amplitude

    def scale(self, factor: complex):
        """Scale all amplitudes"""
        for config in list(self.amplitudes.keys()):
            self.amplitudes[config] *= factor
            if abs(self.amplitudes[config]) < self.threshold:
                del self.amplitudes[config]

    def norm(self) -> float:
        """Total squared amplitude (like probability)"""
        return sum(abs(a) ** 2 for a in self.amplitudes.values())

    def total_amplitude(self) -> float:
        """Total amplitude (not squared)"""
        return sum(abs(a) for a in self.amplitudes.values())

    def normalize(self):
        """Normalize to unit norm"""
        n = self.norm()
        if n > 0:
            self.scale(1.0 / math.sqrt(n))

    def entropy(self) -> float:
        """Von Neumann-like entropy of amplitude distribution"""
        probs = [abs(a) ** 2 for a in self.amplitudes.values()]
        total = sum(probs)
        if total == 0:
            return 0.0
        probs = [p / total for p in probs]
        return -sum(p * math.log(p + 1e-15) for p in probs if p > 0)

    def copy(self) -> 'AmplitudeField':
        """Deep copy"""
        new_field = AmplitudeField(threshold=self.threshold)
        new_field.amplitudes = dict(self.amplitudes)
        return new_field


# ============================================================
# THE HAMILTONIAN (RULE-INDUCED EVOLUTION)
# ============================================================

class Hamiltonian(ABC):
    """
    Abstract Hamiltonian - defines how configurations evolve.

    In physics, H generates time evolution: ψ(t) = e^{-iHt} ψ(0)
    Here, H encodes the rules: which configurations can transition to which.

    The Hamiltonian is:
    - Hermitian (for unitary/reversible evolution)
    - Sparse (only neighboring configs couple)
    - Local (rules are local rewrites)
    """

    @abstractmethod
    def apply(self, field: AmplitudeField) -> AmplitudeField:
        """Apply H to a field, returning H|ψ⟩"""
        pass

    @abstractmethod
    def neighbors(self, config: Configuration) -> List[Tuple[Configuration, complex]]:
        """
        Get configurations that couple to this one, with coupling strength.

        Returns [(config', H_{config,config'}), ...]
        """
        pass


class RuleHamiltonian(Hamiltonian):
    """
    Hamiltonian defined by rewrite rules.

    Each rule A → B contributes:
        H_{A,B} = coupling (amplitude transfer from A to B)
        H_{B,A} = coupling* (Hermitian: reverse transfer)

    The evolution is:
        ψ'(B) += coupling * ψ(A) for each rule A → B
    """

    def __init__(self, rules: List[Tuple[Configuration, Configuration, complex]]):
        """
        Args:
            rules: List of (from_config, to_config, coupling) tuples
        """
        # Store as adjacency: config -> [(neighbor, coupling), ...]
        self.forward: Dict[Configuration, List[Tuple[Configuration, complex]]] = {}
        self.backward: Dict[Configuration, List[Tuple[Configuration, complex]]] = {}

        for from_c, to_c, coupling in rules:
            if from_c not in self.forward:
                self.forward[from_c] = []
            self.forward[from_c].append((to_c, coupling))

            # Hermitian conjugate for reverse
            if to_c not in self.backward:
                self.backward[to_c] = []
            self.backward[to_c].append((from_c, coupling.conjugate()))

    def neighbors(self, config: Configuration) -> List[Tuple[Configuration, complex]]:
        result = []
        if config in self.forward:
            result.extend(self.forward[config])
        if config in self.backward:
            result.extend(self.backward[config])
        return result

    def apply(self, field: AmplitudeField) -> AmplitudeField:
        """Apply Hamiltonian: H|ψ⟩"""
        result = AmplitudeField(threshold=field.threshold)

        for config, amplitude in field:
            # Transfer to neighbors
            for neighbor, coupling in self.neighbors(config):
                result.inject(neighbor, coupling * amplitude)

        return result


class LazyHamiltonian(Hamiltonian):
    """
    Hamiltonian with lazy neighbor generation.

    For large/infinite configuration spaces, we can't enumerate all rules.
    Instead, provide a function that generates neighbors on demand.
    """

    def __init__(self, neighbor_fn: Callable[[Configuration], List[Tuple[Configuration, complex]]]):
        """
        Args:
            neighbor_fn: Function mapping config -> [(neighbor, coupling), ...]
        """
        self.neighbor_fn = neighbor_fn
        self._cache: Dict[Configuration, List[Tuple[Configuration, complex]]] = {}

    def neighbors(self, config: Configuration) -> List[Tuple[Configuration, complex]]:
        if config not in self._cache:
            self._cache[config] = self.neighbor_fn(config)
        return self._cache[config]

    def apply(self, field: AmplitudeField) -> AmplitudeField:
        result = AmplitudeField(threshold=field.threshold)

        for config, amplitude in field:
            for neighbor, coupling in self.neighbors(config):
                result.inject(neighbor, coupling * amplitude)

        return result

    def clear_cache(self):
        self._cache.clear()


# ============================================================
# THE SUBSTRATE
# ============================================================

class Substrate:
    """
    The computational substrate - where everything happens.

    Evolution equation:
        ∂ψ/∂t = -i·H(ψ) - γ·ψ + S

    Discretized (one timestep):
        ψ' = (1 - γ·dt)·ψ + dt·(-i·H(ψ) + S)

    The substrate has three components:
    1. Hamiltonian H: Rules (unitary evolution)
    2. Damping γ: Measurement/decoherence (non-unitary)
    3. Sources S: Boundary conditions (injection from outside)

    The ratio γ/||H|| determines the regime:
    - γ → 0: Quantum-like (coherent, reversible)
    - γ → ∞: Classical-like (immediate collapse)
    - γ ~ ||H||: Interesting (partial coherence)
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        damping: float = 0.1,
        threshold: float = 1e-10
    ):
        self.H = hamiltonian
        self.gamma = damping
        self.threshold = threshold

        # The field
        self.psi = AmplitudeField(threshold=threshold)

        # Sources (continuous injection)
        self.sources: Dict[Configuration, complex] = {}

        # Time tracking
        self.time = 0.0
        self.dt = 0.1

        # History
        self.history: List[Dict[str, Any]] = []

    def inject(self, config: Configuration, amplitude: complex):
        """Inject amplitude at a configuration"""
        self.psi.inject(config, amplitude)

    def inject_source(self, config: Configuration, rate: complex):
        """Add continuous source"""
        self.sources[config] = rate

    def remove_source(self, config: Configuration):
        """Remove continuous source"""
        self.sources.pop(config, None)

    def step(self, dt: float = None) -> Dict[str, Any]:
        """
        One evolution step.

        ψ' = (1 - γ·dt)·ψ + dt·(-i·H(ψ) + S)
        """
        dt = dt or self.dt
        stats = {
            'time': self.time,
            'configs': len(self.psi),
            'norm': self.psi.norm(),
            'entropy': self.psi.entropy(),
        }

        # Apply Hamiltonian: H|ψ⟩
        H_psi = self.H.apply(self.psi)

        # New field
        new_psi = AmplitudeField(threshold=self.threshold)

        # Damped original: (1 - γ·dt)·ψ
        damping_factor = complex(1.0 - self.gamma * dt, 0)
        for config, amplitude in self.psi:
            new_psi.inject(config, damping_factor * amplitude)

        # Hamiltonian contribution: -i·dt·H(ψ)
        # The -i gives oscillation rather than exponential growth/decay
        H_factor = complex(0, -dt)
        for config, amplitude in H_psi:
            new_psi.inject(config, H_factor * amplitude)

        # Sources: dt·S
        for config, rate in self.sources.items():
            new_psi.inject(config, dt * rate)

        self.psi = new_psi
        self.time += dt

        stats['new_configs'] = len(self.psi)
        stats['new_norm'] = self.psi.norm()
        self.history.append(stats)

        return stats

    def evolve(self, duration: float, dt: float = None) -> List[Dict[str, Any]]:
        """Evolve for a duration"""
        dt = dt or self.dt
        steps = int(duration / dt)
        stats = []
        for _ in range(steps):
            stats.append(self.step(dt))
        return stats

    def measure(self, config: Configuration) -> Tuple[float, float]:
        """
        'Measure' a configuration.

        Returns (probability, phase) where:
        - probability = |amplitude|^2
        - phase = arg(amplitude)

        Note: This doesn't collapse the wavefunction (that's what damping does).
        This is just reading the current amplitude.
        """
        amp = self.psi[config]
        return abs(amp) ** 2, cmath.phase(amp)

    def dominant_configs(self, n: int = 10) -> List[Tuple[Configuration, complex]]:
        """Get the n highest-amplitude configurations"""
        items = list(self.psi)
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        return items[:n]

    def temperature(self) -> float:
        """
        Effective temperature (normalized entropy).

        High temperature = many configs with similar amplitude (disorder)
        Low temperature = few configs dominate (order)
        """
        if len(self.psi) == 0:
            return 0.0
        max_entropy = math.log(len(self.psi))
        if max_entropy == 0:
            return 0.0
        return self.psi.entropy() / max_entropy

    def coherence(self) -> float:
        """
        Phase coherence.

        High coherence = amplitudes point same direction
        Low coherence = amplitudes point randomly
        """
        if len(self.psi) == 0:
            return 0.0

        # Sum of unit phasors
        total = sum(a / abs(a) if abs(a) > 0 else 0j for _, a in self.psi)
        return abs(total) / len(self.psi)

    def is_equilibrium(self, window: int = 10, tol: float = 0.01) -> bool:
        """Check if system has reached equilibrium"""
        if len(self.history) < window:
            return False

        # Check if norm is stable
        norms = [h['norm'] for h in self.history[-window:]]
        norm_var = np.var(norms) if norms else 0
        return norm_var < tol

    def summary(self) -> str:
        return (f"Substrate(t={self.time:.3f}):\n"
                f"  Configs: {len(self.psi)}\n"
                f"  Norm: {self.psi.norm():.6f}\n"
                f"  Temperature: {self.temperature():.3f}\n"
                f"  Coherence: {self.coherence():.3f}")


# ============================================================
# CLOSURE DETECTION (EMERGENT FROM SUBSTRATE)
# ============================================================

def detect_closures(
    substrate: Substrate,
    forward_phase: float = 0.0,
    backward_phase: float = math.pi,
    phase_tolerance: float = 0.3,
    amplitude_threshold: float = 0.01
) -> List[Tuple[Configuration, complex]]:
    """
    Detect closure points in the substrate.

    A closure is where:
    1. Amplitude is significant (above threshold)
    2. Multiple paths contributed (interference)
    3. Phase indicates forward-backward meeting

    In the substrate, we detect this by:
    - Finding high-amplitude configs
    - Checking if phase is between forward and backward
    - These are the "solutions"
    """
    closures = []

    for config, amplitude in substrate.psi:
        if abs(amplitude) < amplitude_threshold:
            continue

        phase = cmath.phase(amplitude)

        # Closure: phase is "in between" forward and backward
        # This means both waves contributed constructively
        dist_forward = abs(phase - forward_phase)
        dist_backward = abs(phase - backward_phase)
        dist_backward = min(dist_backward, 2 * math.pi - dist_backward)

        # If not purely forward and not purely backward, it's mixed
        if dist_forward > phase_tolerance and dist_backward > phase_tolerance:
            closures.append((config, amplitude))

    return closures


# ============================================================
# CONVENIENCE: SOLVE AS SUBSTRATE EVOLUTION
# ============================================================

def solve_on_substrate(
    hamiltonian: Hamiltonian,
    forward_configs: List[Configuration],
    backward_configs: List[Configuration],
    forward_amplitude: complex = 1.0,
    backward_amplitude: complex = -1.0,  # Opposite phase
    damping: float = 0.1,
    max_time: float = 100.0,
    dt: float = 0.1,
    verbose: bool = True
) -> Tuple[List[Tuple[Configuration, complex]], Substrate]:
    """
    Solve by evolving the substrate until equilibrium.

    Args:
        hamiltonian: The rules
        forward_configs: Starting points (phase 0)
        backward_configs: Targets/boundaries (phase π)
        forward_amplitude: Initial amplitude for forward
        backward_amplitude: Initial amplitude for backward
        damping: Decoherence rate
        max_time: Maximum evolution time
        dt: Time step
        verbose: Print progress

    Returns:
        (closures, substrate)
    """
    substrate = Substrate(hamiltonian, damping=damping)

    # Inject forward seeds
    n_fwd = len(forward_configs)
    fwd_amp = forward_amplitude / math.sqrt(n_fwd) if n_fwd > 0 else 0
    for config in forward_configs:
        substrate.inject(config, fwd_amp)

    # Inject backward seeds
    n_bwd = len(backward_configs)
    bwd_amp = backward_amplitude / math.sqrt(n_bwd) if n_bwd > 0 else 0
    for config in backward_configs:
        substrate.inject(config, bwd_amp)

    # Evolve
    steps = int(max_time / dt)
    for i in range(steps):
        stats = substrate.step(dt)

        if verbose and i % 50 == 0:
            print(f"t={substrate.time:.1f}: configs={len(substrate.psi)}, "
                  f"temp={substrate.temperature():.3f}, "
                  f"coherence={substrate.coherence():.3f}")

        if substrate.is_equilibrium():
            if verbose:
                print(f"Equilibrium at t={substrate.time:.1f}")
            break

    closures = detect_closures(substrate)

    if verbose:
        print(f"Found {len(closures)} closures")

    return closures, substrate
