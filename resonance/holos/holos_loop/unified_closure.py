"""
holos/unified_closure.py - Unified Closure-Aware HOLOS System

This module integrates all components:

FROM ORIGINAL LOCUS.md:
- SpinePath with checkpoints
- SeedFrontierMapping for compression
- Hologram storage structure
- Session management (phase transitions)
- Crystallization around connections

FROM CLOSURE PHYSICS:
- Pressure dynamics (continuous expansion)
- Interior detection and decryption
- Equivalence-based closure
- Quantum-like amplitude tracking

FROM POOR_MANS_QUANTUM:
- Time-multiplexed real/imaginary components
- Polynomial sensors for log(n) qubits
- Daemon that reads fast medium to guide slow
- Interference (constructive/destructive)

The unified system treats search as a RESERVOIR COMPUTER:
- Input: Seeds (polynomial scaling)
- Reservoir: Game graph (exponential state space)
- Dynamics: Bidirectional wave propagation
- Readout: Closure events + value propagation
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import math
import time
import cmath

from holos.holos import GameInterface
from holos.storage import SpinePath, SeedFrontierMapping, Hologram
from holos.closure import ClosureType, ClosureEvent, WaveOrigin, ModeEmergence, ClosureDetector


# ============================================================
# UNIFIED CONFIGURATION
# ============================================================

@dataclass
class UnifiedConfig:
    """Configuration for unified closure system."""
    # Pressure dynamics
    initial_pressure: float = 100.0
    closure_damping: float = 0.3
    saturation_threshold: float = 0.01

    # Quantum settings
    enable_quantum: bool = True
    amplitude_threshold: float = 0.001

    # Equivalence settings
    enable_equivalence: bool = True
    equiv_confidence_threshold: float = 0.5

    # Crystallization
    crystal_radius: int = 3
    crystal_on_closure: bool = True

    # Interior detection
    interior_radius: int = 5
    decrypt_on_interior: bool = True

    # Session phases
    lightning_rounds: int = 3
    wave_rounds: int = 10
    crystal_rounds: int = 5


# ============================================================
# UNIFIED STATE TRACKING
# ============================================================

@dataclass
class UnifiedState:
    """
    Complete state for a position in the unified system.

    Combines:
    - Game state
    - Quantum amplitude
    - Pressure
    - Equivalence class
    - Value (if known)
    """
    hash: int
    state: Any

    # Quantum properties
    amplitude: complex = complex(1.0, 0.0)
    phase: float = 0.0

    # Wave tracking
    forward_reached: bool = False
    backward_reached: bool = False
    forward_depth: int = 0
    backward_depth: int = 0

    # Pressure
    pressure: float = 100.0

    # Value and equivalence
    value: Any = None
    features: Any = None
    equiv_class_id: Any = None

    # Closure status
    is_closure: bool = False
    closure_type: ClosureType = ClosureType.NONE

    @property
    def is_coherent(self) -> bool:
        """Both waves have reached this state."""
        return self.forward_reached and self.backward_reached

    @property
    def probability(self) -> float:
        """Born rule probability."""
        return abs(self.amplitude) ** 2


class Phase(Enum):
    """Session phases - matches original LOCUS.md."""
    INIT = "init"
    LIGHTNING = "lightning"
    WAVE = "wave"
    CRYSTAL = "crystal"
    COMPLETE = "complete"


# ============================================================
# UNIFIED CLOSURE SYSTEM
# ============================================================

class UnifiedClosureSystem:
    """
    The complete closure-aware HOLOS system.

    Integrates all features from LOCUS.md and closure physics.
    """

    def __init__(
        self,
        game: GameInterface,
        config: UnifiedConfig = None,
        feature_extractor: Callable = None
    ):
        self.game = game
        self.config = config or UnifiedConfig()
        self.feature_extractor = feature_extractor or self._default_features

        # Unified state storage
        self.states: Dict[int, UnifiedState] = {}

        # Frontier tracking (for expansion)
        self.forward_frontier: Set[int] = set()
        self.backward_frontier: Set[int] = set()

        # Value propagation graph
        self.parents: Dict[int, Set[int]] = defaultdict(set)
        self.children: Dict[int, Set[int]] = defaultdict(set)
        self.propagation_pending: Set[int] = set()

        # Equivalence classes
        self.equiv_classes: Dict[Any, Set[int]] = defaultdict(set)
        self.equiv_values: Dict[Any, Any] = {}

        # Closure tracking
        self.closure_detector = ClosureDetector()
        self.mode_emergence = ModeEmergence(self.closure_detector)
        self.closures: Set[int] = set()
        self.closure_events: List[ClosureEvent] = []

        # Interior tracking
        self.interiors: List[Dict] = []

        # Output structures (from original HOLOS)
        self.spines: List[SpinePath] = []
        self.connections: List[Tuple[int, int, Any]] = []
        self.seed_mappings: List[SeedFrontierMapping] = []

        # Session state
        self.phase = Phase.INIT
        self.iteration = 0

        # Statistics
        self.stats = {
            'iterations': 0,
            'forward_expanded': 0,
            'backward_expanded': 0,
            'closures': 0,
            'irreducible_closures': 0,
            'equiv_closures': 0,
            'values_propagated': 0,
            'interiors_found': 0,
            'crystallized': 0,
            'interference_events': 0,
            'constructive': 0,
            'destructive': 0,
            'spines_found': 0,
        }

    def _default_features(self, state: Any) -> Any:
        """Default feature extraction."""
        if hasattr(self.game, 'get_features'):
            return self.game.get_features(state)
        return None

    # ========================================
    # INITIALIZATION
    # ========================================

    def initialize(
        self,
        forward_seeds: List[Any],
        backward_seeds: List[Any] = None
    ):
        """Initialize the system with seeds."""
        # Initialize forward seeds
        n_fwd = len(forward_seeds)
        amp_fwd = 1.0 / math.sqrt(n_fwd) if n_fwd > 0 else 0

        for state in forward_seeds:
            h = self.game.hash_state(state)
            features = self.feature_extractor(state)

            self.states[h] = UnifiedState(
                hash=h,
                state=state,
                amplitude=complex(amp_fwd, 0),
                phase=0.0,
                forward_reached=True,
                forward_depth=0,
                pressure=self.config.initial_pressure,
                features=features
            )

            self.forward_frontier.add(h)
            self._register_equivalence(h, features)

        # Initialize backward seeds
        if backward_seeds:
            n_bwd = len(backward_seeds)
            amp_bwd = 1.0 / math.sqrt(n_bwd) if n_bwd > 0 else 0

            for state in backward_seeds:
                h = self.game.hash_state(state)
                features = self.feature_extractor(state)

                if h in self.states:
                    us = self.states[h]
                    us.amplitude = complex(us.amplitude.real, amp_bwd)
                    us.backward_reached = True
                    us.backward_depth = 0
                else:
                    self.states[h] = UnifiedState(
                        hash=h,
                        state=state,
                        amplitude=complex(0, amp_bwd),
                        phase=math.pi,
                        backward_reached=True,
                        backward_depth=0,
                        pressure=self.config.initial_pressure,
                        features=features
                    )

                self.backward_frontier.add(h)
                self._register_equivalence(h, features)

                # Set value if boundary
                if self.game.is_boundary(state):
                    value = self.game.get_boundary_value(state)
                    self._set_value(h, value)
        else:
            self._generate_backward_seeds(forward_seeds)

        self.phase = Phase.LIGHTNING

    def _generate_backward_seeds(self, templates: List[Any], count: int = 100):
        """Generate backward seeds from boundary."""
        if not templates:
            return

        seeds = self.game.generate_boundary_seeds(templates[0], count)
        n_bwd = len(seeds)
        amp_bwd = 1.0 / math.sqrt(n_bwd) if n_bwd > 0 else 0

        for state in seeds[:count]:
            h = self.game.hash_state(state)
            features = self.feature_extractor(state)

            if h in self.states:
                us = self.states[h]
                us.amplitude = complex(us.amplitude.real, amp_bwd)
                us.backward_reached = True
            else:
                self.states[h] = UnifiedState(
                    hash=h,
                    state=state,
                    amplitude=complex(0, amp_bwd),
                    phase=math.pi,
                    backward_reached=True,
                    backward_depth=0,
                    pressure=self.config.initial_pressure,
                    features=features
                )

            self.backward_frontier.add(h)
            self._register_equivalence(h, features)

            if self.game.is_boundary(state):
                value = self.game.get_boundary_value(state)
                self._set_value(h, value)

    # ========================================
    # EQUIVALENCE TRACKING
    # ========================================

    def _register_equivalence(self, h: int, features: Any, value: Any = None):
        """Register a state in its equivalence class."""
        if features is None:
            return

        us = self.states.get(h)
        if us:
            us.equiv_class_id = features

        self.equiv_classes[features].add(h)

        if value is not None:
            if features not in self.equiv_values:
                self.equiv_values[features] = value
            self._propagate_equiv_value(features, value)

    def _propagate_equiv_value(self, features: Any, value: Any):
        """Propagate value to all members of equivalence class."""
        for h in self.equiv_classes[features]:
            us = self.states.get(h)
            if us and us.value is None:
                self._set_value(h, value, from_equiv=True)
                self.stats['equiv_closures'] += 1

    def _check_equiv_closure(self, h: int) -> Optional[ClosureEvent]:
        """Check if state can be solved via equivalence."""
        us = self.states.get(h)
        if not us or us.features is None:
            return None

        if us.features in self.equiv_values and us.value is None:
            value = self.equiv_values[us.features]
            return ClosureEvent(
                state_hash=h,
                layer=0,
                closure_type=ClosureType.REDUCIBLE,
                phase_diff=0.0,
                forward_value=value,
                backward_value=value,
                iteration=self.iteration,
                num_contributing_paths=len(self.equiv_classes[us.features])
            )
        return None

    # ========================================
    # VALUE PROPAGATION
    # ========================================

    def _set_value(self, h: int, value: Any, from_equiv: bool = False):
        """Set a value and trigger propagation."""
        us = self.states.get(h)
        if not us or us.value is not None:
            return

        us.value = value

        # Register in equivalence
        if us.features and us.features not in self.equiv_values:
            self.equiv_values[us.features] = value

        # Add parents to propagation queue
        for parent_h in self.parents[h]:
            self.propagation_pending.add(parent_h)

        self.stats['values_propagated'] += 1

    def _propagate_values(self, max_steps: int = 100) -> int:
        """Propagate values through the graph."""
        propagated = 0

        for _ in range(max_steps):
            if not self.propagation_pending:
                break

            h = self.propagation_pending.pop()
            us = self.states.get(h)
            if not us or us.value is not None:
                continue

            # Get children's values
            child_values = []
            all_known = True
            for child_h in self.children[h]:
                child_us = self.states.get(child_h)
                if child_us and child_us.value is not None:
                    child_values.append(child_us.value)
                else:
                    all_known = False

            if child_values:
                # Use game's propagation (minimax)
                prop_value = self.game.propagate_value(us.state, child_values)
                if prop_value is not None:
                    self._set_value(h, prop_value)
                    propagated += 1

        return propagated

    # ========================================
    # EXPANSION
    # ========================================

    def _expand_forward(self) -> Tuple[int, List[ClosureEvent]]:
        """Expand forward frontier."""
        expanded = 0
        new_closures = []

        # Sort by amplitude/pressure
        frontier_list = list(self.forward_frontier)
        frontier_list.sort(key=lambda h: -(self.states[h].probability + self.states[h].pressure / 1000))

        for h in frontier_list[:20]:
            us = self.states[h]
            if us.probability < self.config.amplitude_threshold:
                continue

            successors = list(self.game.get_successors(us.state))
            if not successors:
                continue

            amp_factor = 1.0 / math.sqrt(len(successors)) if successors else 0

            for child, move in successors:
                ch = self.game.hash_state(child)

                # Register edge
                self.parents[ch].add(h)
                self.children[h].add(ch)

                if ch not in self.states:
                    features = self.feature_extractor(child)
                    self.states[ch] = UnifiedState(
                        hash=ch,
                        state=child,
                        amplitude=complex(0, 0),
                        forward_reached=True,
                        forward_depth=us.forward_depth + 1,
                        pressure=us.pressure * 0.9,
                        features=features
                    )
                    expanded += 1
                    self._register_equivalence(ch, features)

                child_us = self.states[ch]
                child_us.forward_reached = True

                # Update amplitude (forward = real component)
                old_amp = child_us.amplitude
                new_real = old_amp.real + us.amplitude.real * amp_factor
                child_us.amplitude = complex(new_real, old_amp.imag)

                # Track interference
                if abs(old_amp) > 0:
                    self._track_interference(old_amp, child_us.amplitude)

                # Check for closure
                if child_us.backward_reached:
                    event = self._check_closure(ch)
                    if event:
                        new_closures.append(event)

                # Check equiv closure
                if self.config.enable_equivalence:
                    equiv_event = self._check_equiv_closure(ch)
                    if equiv_event and ch not in self.closures:
                        new_closures.append(equiv_event)

                # Check boundary
                if self.game.is_boundary(child):
                    value = self.game.get_boundary_value(child)
                    self._set_value(ch, value)

                self.forward_frontier.add(ch)

            self.forward_frontier.discard(h)

        self.stats['forward_expanded'] += expanded
        return expanded, new_closures

    def _expand_backward(self) -> Tuple[int, List[ClosureEvent]]:
        """Expand backward frontier."""
        expanded = 0
        new_closures = []

        frontier_list = list(self.backward_frontier)
        frontier_list.sort(key=lambda h: -(self.states[h].probability + self.states[h].pressure / 1000))

        for h in frontier_list[:20]:
            us = self.states[h]
            if us.probability < self.config.amplitude_threshold:
                continue

            predecessors = list(self.game.get_predecessors(us.state))
            if not predecessors:
                continue

            amp_factor = 1.0 / math.sqrt(len(predecessors)) if predecessors else 0

            for parent, move in predecessors:
                ph = self.game.hash_state(parent)

                # Register edge (reversed)
                self.parents[h].add(ph)
                self.children[ph].add(h)

                if ph not in self.states:
                    features = self.feature_extractor(parent)
                    self.states[ph] = UnifiedState(
                        hash=ph,
                        state=parent,
                        amplitude=complex(0, 0),
                        backward_reached=True,
                        backward_depth=us.backward_depth + 1,
                        pressure=us.pressure * 0.9,
                        features=features
                    )
                    expanded += 1
                    self._register_equivalence(ph, features)

                parent_us = self.states[ph]
                parent_us.backward_reached = True

                # Update amplitude (backward = imaginary component)
                old_amp = parent_us.amplitude
                new_imag = old_amp.imag + us.amplitude.imag * amp_factor
                parent_us.amplitude = complex(old_amp.real, new_imag)

                if abs(old_amp) > 0:
                    self._track_interference(old_amp, parent_us.amplitude)

                if parent_us.forward_reached:
                    event = self._check_closure(ph)
                    if event:
                        new_closures.append(event)

                if self.config.enable_equivalence:
                    equiv_event = self._check_equiv_closure(ph)
                    if equiv_event and ph not in self.closures:
                        new_closures.append(equiv_event)

                self.backward_frontier.add(ph)

            self.backward_frontier.discard(h)

        self.stats['backward_expanded'] += expanded
        return expanded, new_closures

    # ========================================
    # CLOSURE DETECTION
    # ========================================

    def _check_closure(self, h: int) -> Optional[ClosureEvent]:
        """Check if state is a closure point."""
        us = self.states.get(h)
        if not us:
            return None

        if not (us.forward_reached and us.backward_reached):
            return None

        # Both waves reached - this is a closure
        amp = us.amplitude
        is_coherent = abs(amp.real) > 0.001 and abs(amp.imag) > 0.001

        if us.value is not None:
            closure_type = ClosureType.IRREDUCIBLE
            self.stats['irreducible_closures'] += 1
        elif is_coherent:
            closure_type = ClosureType.REDUCIBLE
        else:
            return None

        return ClosureEvent(
            state_hash=h,
            layer=0,
            closure_type=closure_type,
            phase_diff=abs(cmath.phase(amp)) if amp != 0 else 0,
            forward_value=us.value,
            backward_value=us.value,
            iteration=self.iteration,
            num_contributing_paths=1
        )

    def _process_closure(self, event: ClosureEvent):
        """Process a closure event."""
        h = event.state_hash
        us = self.states.get(h)

        if us:
            us.is_closure = True
            us.closure_type = event.closure_type
            us.pressure *= self.config.closure_damping

        self.closures.add(h)
        self.closure_events.append(event)
        self.stats['closures'] += 1

        # Record connection
        self.connections.append((h, h, event.forward_value))

        # Crystallize around closure
        if self.config.crystal_on_closure:
            self._crystallize(h)

        # Try to form interior
        if self.config.decrypt_on_interior:
            self._try_form_interior(h)

    # ========================================
    # CRYSTALLIZATION
    # ========================================

    def _crystallize(self, center_h: int):
        """Crystallize (local BFS) around a closure point."""
        us = self.states.get(center_h)
        if not us:
            return

        # Local BFS
        local = {center_h: us.state}
        local_seen = {center_h}

        for _ in range(self.config.crystal_radius):
            next_local = {}
            for h, state in local.items():
                for child, move in self.game.get_successors(state):
                    ch = self.game.hash_state(child)
                    if ch not in local_seen:
                        local_seen.add(ch)
                        next_local[ch] = child

                        if self.game.is_boundary(child):
                            value = self.game.get_boundary_value(child)
                            if ch not in self.states:
                                features = self.feature_extractor(child)
                                self.states[ch] = UnifiedState(
                                    hash=ch,
                                    state=child,
                                    features=features
                                )
                            self._set_value(ch, value)
                            self.stats['crystallized'] += 1

            local = next_local

    # ========================================
    # INTERIOR DETECTION
    # ========================================

    def _try_form_interior(self, center_h: int):
        """Try to detect and decrypt an interior around a closure."""
        boundary = set()
        interior = set()
        queue = [(center_h, 0)]
        visited = set()

        while queue:
            h, depth = queue.pop(0)
            if h in visited:
                continue
            visited.add(h)

            us = self.states.get(h)
            if not us:
                continue

            # Is this boundary?
            if h in self.closures and us.value is not None:
                boundary.add(h)
                continue

            if depth >= self.config.interior_radius:
                boundary.add(h)
                continue

            interior.add(h)

            # Explore children
            for child_h in self.children[h]:
                if child_h not in visited:
                    queue.append((child_h, depth + 1))

        # Check if boundary is fully valued
        boundary_solved = all(
            self.states.get(h) and self.states[h].value is not None
            for h in boundary
        )

        if boundary_solved and interior:
            self.interiors.append({
                'center': center_h,
                'boundary': boundary,
                'interior': interior,
                'time': time.time()
            })
            self.stats['interiors_found'] += 1

            # Decrypt interior
            self._decrypt_interior(boundary, interior)

    def _decrypt_interior(self, boundary: Set[int], interior: Set[int]):
        """Propagate values from boundary into interior."""
        for h in boundary:
            self.propagation_pending.add(h)

        propagated = self._propagate_values(max_steps=len(interior) * 2)

    # ========================================
    # INTERFERENCE TRACKING
    # ========================================

    def _track_interference(self, old_amp: complex, new_amp: complex):
        """Track interference events."""
        self.stats['interference_events'] += 1

        if abs(new_amp) > abs(old_amp):
            self.stats['constructive'] += 1
        elif abs(new_amp) < abs(old_amp):
            self.stats['destructive'] += 1

    # ========================================
    # MODE EMERGENCE
    # ========================================

    def get_emergent_mode(self) -> str:
        """Get the emergent search mode based on closure state."""
        return self.mode_emergence.get_emergent_mode(
            forward_frontier_size=len(self.forward_frontier),
            backward_frontier_size=len(self.backward_frontier),
            recent_closures=len([e for e in self.closure_events[-10:]]),
            branching_factor=7.0  # Default, should be game-specific
        )

    # ========================================
    # MAIN LOOP
    # ========================================

    def step(self) -> Dict[str, Any]:
        """One step of the unified system."""
        self.iteration += 1
        self.stats['iterations'] = self.iteration

        new_closures = []
        expanded = 0

        # Time-multiplexed expansion (quantum-like)
        if self.iteration % 2 == 0:
            exp, closures = self._expand_forward()
        else:
            exp, closures = self._expand_backward()

        expanded += exp
        new_closures.extend(closures)

        # Process closures
        for event in new_closures:
            if event.state_hash not in self.closures:
                self._process_closure(event)

        # Propagate values
        propagated = self._propagate_values()

        # Get emergent mode
        mode = self.get_emergent_mode()

        return {
            'iteration': self.iteration,
            'expanded': expanded,
            'new_closures': len(new_closures),
            'total_closures': len(self.closures),
            'values_known': sum(1 for us in self.states.values() if us.value is not None),
            'propagated': propagated,
            'mode': mode,
            'interiors': len(self.interiors),
        }

    def run(
        self,
        max_iterations: int = 100,
        target_closures: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run the unified system."""
        t0 = time.time()

        for i in range(max_iterations):
            result = self.step()

            if verbose and i % 10 == 0:
                print(f"  Iter {i}: closures={result['total_closures']}, "
                      f"values={result['values_known']}, "
                      f"mode={result['mode']}, "
                      f"interiors={result['interiors']}")

            if target_closures and result['total_closures'] >= target_closures:
                break

        elapsed = time.time() - t0

        return {
            'iterations': self.iteration,
            'elapsed': elapsed,
            'states': len(self.states),
            'closures': len(self.closures),
            'irreducible': self.stats['irreducible_closures'],
            'values': sum(1 for us in self.states.values() if us.value is not None),
            'equiv_classes': len(self.equiv_classes),
            'interiors': len(self.interiors),
            'spines': len(self.spines),
            'stats': self.stats,
        }

    # ========================================
    # OUTPUT GENERATION
    # ========================================

    def to_hologram(self, name: str = "unified") -> Hologram:
        """Convert results to standard Hologram format."""
        hologram = Hologram(name=name)

        # Copy solved values
        for h, us in self.states.items():
            if us.value is not None:
                hologram.solved[h] = us.value
                if us.features:
                    hologram.add_with_features(h, us.value, us.features)

        # Copy spines
        hologram.spines = self.spines.copy()

        # Copy connections
        hologram.connections = self.connections.copy()

        # Copy equivalence
        hologram.equiv_classes = dict(self.equiv_classes)
        hologram.equiv_outcomes = dict(self.equiv_values)

        # Copy stats
        hologram.stats = dict(self.stats)

        return hologram


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def run_unified_search(
    game: GameInterface,
    start_states: List[Any],
    backward_states: List[Any] = None,
    max_iterations: int = 100,
    config: UnifiedConfig = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], Hologram]:
    """
    Run unified closure-aware search.

    Returns (results_dict, hologram).
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Unified Closure Search")
        print("=" * 60)

    system = UnifiedClosureSystem(game, config)
    system.initialize(start_states, backward_states)

    if verbose:
        print(f"  Forward seeds: {len(start_states)}")
        print(f"  Backward seeds: {len(system.backward_frontier)}")
        print()

    result = system.run(max_iterations=max_iterations, verbose=verbose)

    hologram = system.to_hologram()

    if verbose:
        print(f"\nResults:")
        print(f"  States: {result['states']}")
        print(f"  Closures: {result['closures']} ({result['irreducible']} irreducible)")
        print(f"  Values: {result['values']}")
        print(f"  Equivalence classes: {result['equiv_classes']}")
        print(f"  Interiors: {result['interiors']}")

    return result, hologram
