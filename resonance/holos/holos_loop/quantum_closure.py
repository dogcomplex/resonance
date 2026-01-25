"""
holos/quantum_closure.py - Quantum-Inspired Closure System

Integrates three key features:
1. Value propagation through closure points (minimax decryption)
2. Equivalence-based closure detection (feature compression)
3. Quantum-like amplitude tracking (superposition/interference)

The quantum analogy:
- States exist in superposition until measured (closure)
- Amplitudes track probability of reaching each state
- Interference: paths can reinforce or cancel
- Measurement collapses to definite values

From poor_mans_quantum.txt:
- Real/imaginary components via time-multiplexing (even/odd steps)
- Polynomial sensors for n "qubits"
- 2^n energy cost is unavoidable but can be managed
- Maxwell daemon (closure detector) boosts readable terms
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import math
import time
import cmath

from holos.holos import GameInterface
from holos.closure import ClosureType, ClosureEvent, WaveOrigin
from holos.storage import SpinePath


# ============================================================
# QUANTUM STATE REPRESENTATION
# ============================================================

@dataclass
class QuantumState:
    """
    A state with quantum-like properties.

    Amplitude is complex: real part from forward wave, imaginary from backward.
    Phase tracks the "direction" of the wave at this point.
    """
    hash: int
    amplitude: complex = complex(1.0, 0.0)
    phase: float = 0.0
    origin: WaveOrigin = WaveOrigin.FORWARD
    depth: int = 0

    @property
    def probability(self) -> float:
        """Born rule: probability = |amplitude|^2"""
        return abs(self.amplitude) ** 2

    @property
    def is_coherent(self) -> bool:
        """State is coherent if both real and imaginary parts are significant."""
        return abs(self.amplitude.real) > 0.01 and abs(self.amplitude.imag) > 0.01


@dataclass
class EquivalenceClass:
    """A class of states with equivalent features."""
    features: Any  # Hashable feature tuple
    members: Set[int] = field(default_factory=set)
    known_value: Any = None
    confidence: float = 0.0  # How confident are we in the value?

    def add_member(self, state_hash: int, value: Any = None):
        self.members.add(state_hash)
        if value is not None and self.known_value is None:
            self.known_value = value
            self.confidence = 1.0
        elif value is not None and value == self.known_value:
            self.confidence = min(1.0, self.confidence + 0.1)


# ============================================================
# VALUE PROPAGATION (MINIMAX DECRYPTION)
# ============================================================

class ValuePropagator:
    """
    Propagates game-theoretic values through the search graph.

    When closures form, values flow from boundaries inward.
    This "decrypts" the interior - computing minimax values.
    """

    def __init__(self, game: GameInterface):
        self.game = game
        self.values: Dict[int, Any] = {}
        self.parents: Dict[int, Set[int]] = defaultdict(set)  # child -> parents
        self.children: Dict[int, Set[int]] = defaultdict(set)  # parent -> children
        self.pending: Set[int] = set()  # States waiting for propagation

    def register_edge(self, parent_hash: int, child_hash: int):
        """Record parent-child relationship for propagation."""
        self.parents[child_hash].add(parent_hash)
        self.children[parent_hash].add(child_hash)

    def set_value(self, state_hash: int, value: Any, state: Any = None):
        """Set a known value and trigger propagation."""
        if state_hash in self.values:
            return  # Already known

        self.values[state_hash] = value

        # Parents may now be solvable
        for parent_hash in self.parents[state_hash]:
            if parent_hash not in self.values:
                self.pending.add(parent_hash)

    def propagate(self, states: Dict[int, Any], max_iterations: int = 1000) -> int:
        """
        Propagate values through pending states.
        Returns number of newly solved states.
        """
        solved = 0
        iterations = 0

        while self.pending and iterations < max_iterations:
            iterations += 1
            state_hash = self.pending.pop()

            if state_hash in self.values:
                continue

            state = states.get(state_hash)
            if state is None:
                continue

            # Get all children's values
            child_hashes = self.children[state_hash]
            child_values = []
            all_known = True

            for ch in child_hashes:
                if ch in self.values:
                    child_values.append(self.values[ch])
                else:
                    all_known = False

            # Can we propagate?
            if child_values:
                # Use game's propagation rule (minimax for games)
                propagated = self.game.propagate_value(state, child_values)

                if propagated is not None:
                    self.values[state_hash] = propagated
                    solved += 1

                    # Add parents to pending
                    for parent_hash in self.parents[state_hash]:
                        if parent_hash not in self.values:
                            self.pending.add(parent_hash)

        return solved


# ============================================================
# EQUIVALENCE-BASED CLOSURE
# ============================================================

class EquivalenceClosureDetector:
    """
    Detects closures based on feature equivalence, not just position identity.

    Key insight: If two positions have the same features and one is solved,
    the other likely has the same value (with some confidence).
    """

    def __init__(self, game: GameInterface, feature_extractor: Callable = None):
        self.game = game
        self.feature_extractor = feature_extractor or self._default_features

        # Equivalence tracking
        self.classes: Dict[Any, EquivalenceClass] = {}
        self.state_to_class: Dict[int, Any] = {}

        # Statistics
        self.stats = {
            'equiv_closures': 0,
            'equiv_propagated': 0,
            'classes_created': 0,
        }

    def _default_features(self, state: Any) -> Any:
        """Default feature extraction - use game's get_features if available."""
        if hasattr(self.game, 'get_features'):
            return self.game.get_features(state)
        return None  # No features available

    def register_state(self, state_hash: int, state: Any, value: Any = None):
        """Register a state and its features."""
        features = self.feature_extractor(state)
        if features is None:
            return None

        # Get or create equivalence class
        if features not in self.classes:
            self.classes[features] = EquivalenceClass(features=features)
            self.stats['classes_created'] += 1

        eq_class = self.classes[features]
        eq_class.add_member(state_hash, value)
        self.state_to_class[state_hash] = features

        return eq_class

    def check_equivalence_closure(self, state_hash: int, state: Any) -> Optional[ClosureEvent]:
        """
        Check if state can be solved via equivalence.

        If another state with same features has a known value,
        this state likely has the same value.
        """
        features = self.feature_extractor(state)
        if features is None:
            return None

        if features not in self.classes:
            return None

        eq_class = self.classes[features]

        if eq_class.known_value is not None and eq_class.confidence > 0.5:
            self.stats['equiv_closures'] += 1
            return ClosureEvent(
                state_hash=state_hash,
                layer=0,
                closure_type=ClosureType.REDUCIBLE,  # Equivalence is a form of reduction
                phase_diff=1.0 - eq_class.confidence,
                forward_value=eq_class.known_value,
                backward_value=eq_class.known_value,
                iteration=0,
                num_contributing_paths=len(eq_class.members)
            )

        return None

    def propagate_to_class(self, features: Any, value: Any) -> int:
        """Propagate a value to all members of an equivalence class."""
        if features not in self.classes:
            return 0

        eq_class = self.classes[features]
        eq_class.known_value = value
        eq_class.confidence = 1.0

        self.stats['equiv_propagated'] += len(eq_class.members)
        return len(eq_class.members)


# ============================================================
# QUANTUM-LIKE WAVE SYSTEM
# ============================================================

class QuantumWaveSystem:
    """
    Quantum-inspired wave propagation with amplitude tracking.

    Key quantum concepts:
    - Superposition: Multiple states explored simultaneously
    - Amplitude: Complex number tracking probability of each path
    - Interference: Amplitudes can add (constructive) or cancel (destructive)
    - Measurement: Closure events collapse superposition to definite values

    From poor_mans_quantum.txt:
    - Time-multiplex real/imaginary: even steps = real, odd = imaginary
    - Polynomial sensors (seeds) for log(n) "qubits"
    - 2^n energy cost unavoidable
    - Daemon (closure detector) reads fast medium to guide slow
    """

    def __init__(self, game: GameInterface, feature_extractor: Callable = None):
        self.game = game

        # State storage with quantum properties
        self.states: Dict[int, Any] = {}
        self.amplitudes: Dict[int, complex] = {}
        self.phases: Dict[int, float] = {}

        # Wave tracking
        self.forward_reached: Set[int] = set()
        self.backward_reached: Set[int] = set()

        # Value propagation
        self.propagator = ValuePropagator(game)

        # Equivalence detection
        self.equiv_detector = EquivalenceClosureDetector(game, feature_extractor)

        # Closure tracking
        self.closures: Set[int] = set()
        self.closure_events: List[ClosureEvent] = []

        # Quantum statistics
        self.stats = {
            'iterations': 0,
            'total_amplitude': 0.0,
            'coherent_states': 0,
            'interference_events': 0,
            'constructive': 0,
            'destructive': 0,
            'measurements': 0,
            'values_propagated': 0,
        }

    def initialize(self, forward_seeds: List[Any], backward_seeds: List[Any] = None):
        """
        Initialize quantum superposition over seeds.

        Forward seeds get real amplitude, backward get imaginary.
        """
        # Forward seeds: real amplitude
        n_fwd = len(forward_seeds)
        amp_fwd = 1.0 / math.sqrt(n_fwd) if n_fwd > 0 else 0

        for state in forward_seeds:
            h = self.game.hash_state(state)
            self.states[h] = state
            self.amplitudes[h] = complex(amp_fwd, 0)
            self.phases[h] = 0.0
            self.forward_reached.add(h)
            self.equiv_detector.register_state(h, state)

        # Backward seeds: imaginary amplitude
        if backward_seeds:
            n_bwd = len(backward_seeds)
            amp_bwd = 1.0 / math.sqrt(n_bwd) if n_bwd > 0 else 0

            for state in backward_seeds:
                h = self.game.hash_state(state)
                self.states[h] = state
                self.backward_reached.add(h)

                # Add imaginary component
                existing = self.amplitudes.get(h, complex(0, 0))
                self.amplitudes[h] = complex(existing.real, amp_bwd)
                self.phases[h] = math.pi  # Backward wave has opposite phase

                # Set value if boundary
                if self.game.is_boundary(state):
                    value = self.game.get_boundary_value(state)
                    self.propagator.set_value(h, value, state)
                    self.equiv_detector.register_state(h, state, value)
        else:
            self._generate_backward_seeds(forward_seeds)

    def _generate_backward_seeds(self, templates: List[Any], count: int = 100):
        """Generate backward seeds from boundary."""
        if not templates:
            return

        seeds = self.game.generate_boundary_seeds(templates[0], count)
        n_bwd = len(seeds)
        amp_bwd = 1.0 / math.sqrt(n_bwd) if n_bwd > 0 else 0

        for state in seeds[:count]:
            h = self.game.hash_state(state)
            self.states[h] = state
            self.backward_reached.add(h)

            existing = self.amplitudes.get(h, complex(0, 0))
            self.amplitudes[h] = complex(existing.real, amp_bwd)
            self.phases[h] = math.pi

            if self.game.is_boundary(state):
                value = self.game.get_boundary_value(state)
                self.propagator.set_value(h, value, state)
                self.equiv_detector.register_state(h, state, value)

    def step(self, t: int) -> Dict[str, Any]:
        """
        One quantum evolution step.

        Even steps: expand forward (real component)
        Odd steps: expand backward (imaginary component)

        This time-multiplexing is key to the poor_mans_quantum approach.
        """
        self.stats['iterations'] += 1
        new_closures = []
        expanded = 0

        if t % 2 == 0:
            # Even step: forward expansion (real component)
            expanded, new_closures = self._expand_forward()
        else:
            # Odd step: backward expansion (imaginary component)
            expanded, new_closures = self._expand_backward()

        # Process closures (measurement events)
        for event in new_closures:
            self.closures.add(event.state_hash)
            self.closure_events.append(event)
            self.stats['measurements'] += 1

        # Propagate values through graph
        propagated = self.propagator.propagate(self.states, max_iterations=100)
        self.stats['values_propagated'] += propagated

        # Update quantum statistics
        self._update_stats()

        return {
            'expanded': expanded,
            'new_closures': len(new_closures),
            'propagated': propagated,
            'total_amplitude': self.stats['total_amplitude'],
            'coherent_states': self.stats['coherent_states'],
        }

    def _expand_forward(self) -> Tuple[int, List[ClosureEvent]]:
        """Expand forward wave, adding to real component."""
        expanded = 0
        new_closures = []

        # Sort by amplitude (highest probability first)
        frontier = [(h, abs(self.amplitudes.get(h, 0)))
                   for h in self.forward_reached
                   if h in self.states]
        frontier.sort(key=lambda x: -x[1])

        for h, amp in frontier[:20]:  # Expand top 20
            if amp < 0.001:
                continue

            state = self.states[h]
            parent_amp = self.amplitudes.get(h, complex(1, 0))

            successors = list(self.game.get_successors(state))
            if not successors:
                continue

            # Distribute amplitude to children (unitary-ish)
            child_amp_factor = 1.0 / math.sqrt(len(successors))

            for child, move in successors:
                ch = self.game.hash_state(child)

                # Register edge for value propagation
                self.propagator.register_edge(h, ch)

                if ch not in self.states:
                    self.states[ch] = child
                    expanded += 1

                self.forward_reached.add(ch)

                # Add amplitude (forward = real component)
                old_amp = self.amplitudes.get(ch, complex(0, 0))
                new_real = old_amp.real + parent_amp.real * child_amp_factor
                self.amplitudes[ch] = complex(new_real, old_amp.imag)

                # Track interference
                if old_amp != 0:
                    self._check_interference(ch, old_amp, self.amplitudes[ch])

                # Register with equivalence detector
                self.equiv_detector.register_state(ch, child)

                # Check for closure (both waves reached)
                if ch in self.backward_reached:
                    event = self._check_closure(ch)
                    if event:
                        new_closures.append(event)

                # Check for equivalence closure
                equiv_event = self.equiv_detector.check_equivalence_closure(ch, child)
                if equiv_event and ch not in self.closures:
                    new_closures.append(equiv_event)

                # Check boundary
                if self.game.is_boundary(child):
                    value = self.game.get_boundary_value(child)
                    self.propagator.set_value(ch, value, child)

        return expanded, new_closures

    def _expand_backward(self) -> Tuple[int, List[ClosureEvent]]:
        """Expand backward wave, adding to imaginary component."""
        expanded = 0
        new_closures = []

        frontier = [(h, abs(self.amplitudes.get(h, 0)))
                   for h in self.backward_reached
                   if h in self.states]
        frontier.sort(key=lambda x: -x[1])

        for h, amp in frontier[:20]:
            if amp < 0.001:
                continue

            state = self.states[h]
            parent_amp = self.amplitudes.get(h, complex(0, 1))

            predecessors = list(self.game.get_predecessors(state))
            if not predecessors:
                continue

            child_amp_factor = 1.0 / math.sqrt(len(predecessors))

            for parent, move in predecessors:
                ph = self.game.hash_state(parent)

                # Register edge (reversed for backward)
                self.propagator.register_edge(ph, h)

                if ph not in self.states:
                    self.states[ph] = parent
                    expanded += 1

                self.backward_reached.add(ph)

                # Add amplitude (backward = imaginary component)
                old_amp = self.amplitudes.get(ph, complex(0, 0))
                new_imag = old_amp.imag + parent_amp.imag * child_amp_factor
                self.amplitudes[ph] = complex(old_amp.real, new_imag)

                if old_amp != 0:
                    self._check_interference(ph, old_amp, self.amplitudes[ph])

                self.equiv_detector.register_state(ph, parent)

                if ph in self.forward_reached:
                    event = self._check_closure(ph)
                    if event:
                        new_closures.append(event)

                equiv_event = self.equiv_detector.check_equivalence_closure(ph, parent)
                if equiv_event and ph not in self.closures:
                    new_closures.append(equiv_event)

        return expanded, new_closures

    def _check_closure(self, h: int) -> Optional[ClosureEvent]:
        """Check if state is a closure point (both waves reached)."""
        if h not in self.forward_reached or h not in self.backward_reached:
            return None

        amp = self.amplitudes.get(h, complex(0, 0))

        # Closure when both components are significant (coherent)
        if abs(amp.real) > 0.001 and abs(amp.imag) > 0.001:
            # Get value if known
            value = self.propagator.values.get(h)
            closure_type = ClosureType.IRREDUCIBLE if value else ClosureType.REDUCIBLE

            return ClosureEvent(
                state_hash=h,
                layer=0,
                closure_type=closure_type,
                phase_diff=abs(cmath.phase(amp)),  # Phase difference
                forward_value=value,
                backward_value=value,
                iteration=self.stats['iterations'],
                num_contributing_paths=1
            )

        return None

    def _check_interference(self, h: int, old_amp: complex, new_amp: complex):
        """Track interference events."""
        self.stats['interference_events'] += 1

        if abs(new_amp) > abs(old_amp):
            self.stats['constructive'] += 1
        elif abs(new_amp) < abs(old_amp):
            self.stats['destructive'] += 1

    def _update_stats(self):
        """Update quantum statistics."""
        total_amp = sum(abs(a) for a in self.amplitudes.values())
        coherent = sum(1 for a in self.amplitudes.values()
                      if abs(a.real) > 0.01 and abs(a.imag) > 0.01)

        self.stats['total_amplitude'] = total_amp
        self.stats['coherent_states'] = coherent

    def measure(self, top_k: int = 10) -> List[Tuple[int, float, Any]]:
        """
        Collapse superposition, return highest probability states.

        This is the "readout" - what we can observe from the computation.
        """
        # Probability = |amplitude|^2
        probs = [(h, abs(amp)**2, self.propagator.values.get(h))
                for h, amp in self.amplitudes.items()]

        # Normalize
        total = sum(p for _, p, _ in probs)
        if total > 0:
            probs = [(h, p/total, v) for h, p, v in probs]

        # Sort by probability, return top k
        probs.sort(key=lambda x: -x[1])
        return probs[:top_k]

    def run(self, max_iterations: int = 100, target_closures: int = None) -> Dict[str, Any]:
        """Run quantum wave evolution."""
        t0 = time.time()

        for t in range(max_iterations):
            result = self.step(t)

            if t % 20 == 0:
                print(f"  Iter {t}: closures={len(self.closures)}, "
                      f"coherent={result['coherent_states']}, "
                      f"values={len(self.propagator.values)}")

            if target_closures and len(self.closures) >= target_closures:
                break

        elapsed = time.time() - t0

        # Final measurement
        observations = self.measure(top_k=20)

        return {
            'iterations': self.stats['iterations'],
            'elapsed': elapsed,
            'states_explored': len(self.states),
            'closures': len(self.closures),
            'values_computed': len(self.propagator.values),
            'equiv_classes': len(self.equiv_detector.classes),
            'interference_events': self.stats['interference_events'],
            'constructive_ratio': self.stats['constructive'] / max(1, self.stats['interference_events']),
            'observations': observations,
            'stats': self.stats,
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def run_quantum_search(
    game: GameInterface,
    start_states: List[Any],
    backward_states: List[Any] = None,
    max_iterations: int = 100,
    feature_extractor: Callable = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run quantum-inspired closure search.

    Combines:
    - Amplitude tracking (superposition)
    - Value propagation (minimax decryption)
    - Equivalence detection (feature compression)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Quantum Closure Search")
        print("=" * 60)

    system = QuantumWaveSystem(game, feature_extractor)
    system.initialize(start_states, backward_states)

    if verbose:
        print(f"  Forward seeds: {len(start_states)}")
        print(f"  Backward seeds: {len(system.backward_reached)}")
        print()

    result = system.run(max_iterations=max_iterations)

    if verbose:
        print(f"\nResults:")
        print(f"  States explored: {result['states_explored']}")
        print(f"  Closures: {result['closures']}")
        print(f"  Values computed: {result['values_computed']}")
        print(f"  Equivalence classes: {result['equiv_classes']}")
        print(f"  Interference events: {result['interference_events']}")
        print(f"  Constructive ratio: {result['constructive_ratio']:.1%}")

        print(f"\nTop observations (by probability):")
        for h, prob, value in result['observations'][:5]:
            print(f"  {h}: prob={prob:.3f}, value={value}")

    return result
