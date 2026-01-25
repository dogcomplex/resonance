"""
holos/wave_system.py - Unified Multi-Layer Wave Orchestration

This module orchestrates wave propagation across all layers simultaneously,
implementing the "one big wave function" paradigm.

Core insight from neuroscience:
- Brain rhythms are not clocks, they're closure events
- What we observe (modes/patterns) are READOUTS of closure conditions
- Structure (geometry) determines where closures CAN form
- Dynamics (waves) explore for closures
- Irreducibility determines which closures PERSIST

The multi-layer system:
- Each layer is a "medium" with its own properties
- Waves propagate within each layer (HOLOS search)
- Waves couple BETWEEN layers (closure events trigger cross-layer effects)
- Energy (compute budget) flows through the system
- Modes EMERGE from closure conditions, not selected

Layer coupling:
- When Layer 0 finds a spine, Layer 1 gains a new path
- When Layer 1 finds a complete path, Layer 2 gains a new cover option
- When Layer 2 finds a complete cover, Layer 3 gains a new policy option
- Information flows BOTH directions (results up, priorities down)

This implements the "wave function jumping up a layer, propagating there,
and jumping back down in a new spot" pattern.
"""

from typing import List, Tuple, Optional, Any, Dict, Set, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time

from holos.holos import GameInterface, SearchMode, HOLOSSolver, SeedPoint
from holos.storage import SpinePath
from holos.closure import (
    ClosureDetector, ClosureEvent, ClosureType,
    ModeEmergence, LayerCoupling, WaveOrigin
)
from holos.layer1_paths import PathLayerSolver, PartialPath, PathGame
from holos.layer2_covers import CoverLayerSolver, PathCover, CoverGame
from holos.layer3_policy import PolicyLayerSolver, CoverPolicy, PolicyGame


# ============================================================
# WAVE STATE
# ============================================================

@dataclass
class WaveState:
    """
    State of wave propagation at one layer.

    Tracks:
    - Amplitude: "energy" at each state (priority/confidence)
    - Phase: progress toward closure
    - Frontier: active states being expanded
    """
    # Amplitude at each state (hash -> amplitude)
    amplitude: Dict[int, float] = field(default_factory=dict)

    # Forward and backward frontiers
    forward_frontier: Set[int] = field(default_factory=set)
    backward_frontier: Set[int] = field(default_factory=set)

    # Solved states
    solved: Set[int] = field(default_factory=set)

    # Total energy in this layer
    total_energy: float = 0.0

    @property
    def is_active(self) -> bool:
        """Is this wave still propagating?"""
        return len(self.forward_frontier) > 0 or len(self.backward_frontier) > 0


# ============================================================
# LAYER MEDIUM
# ============================================================

@dataclass
class LayerMedium:
    """
    A layer as a wave propagation medium.

    Properties:
    - Impedance: How easily waves propagate (lower = faster)
    - Damping: How much energy dissipates per step
    - Coupling: How energy transfers to adjacent layers
    """
    layer_index: int
    name: str

    # Medium properties
    impedance: float = 1.0    # Lower = faster propagation
    damping: float = 0.05     # Fraction of energy lost per step

    # Coupling to adjacent layers
    coupling_up: float = 0.3   # Energy transmitted upward on closure
    coupling_down: float = 0.2  # Energy transmitted downward on closure

    # Solver for this layer
    solver: Any = None

    # Closure detector
    closure_detector: ClosureDetector = None

    def __post_init__(self):
        if self.closure_detector is None:
            self.closure_detector = ClosureDetector()


# ============================================================
# MULTI-LAYER WAVE SYSTEM
# ============================================================

class WaveSystem:
    """
    Unified wave propagation across all layers.

    Implements tandem execution where:
    1. All layers propagate waves simultaneously
    2. Closures at one layer emit energy to adjacent layers
    3. Modes emerge from closure conditions
    4. Energy (compute) is conserved across the system

    The key insight: this is all ONE wave function propagating
    through a multi-scale medium. The layers are just different
    resonant scales where standing waves can form.
    """

    def __init__(self,
                 layer0_game: GameInterface,
                 total_energy: float = 1000.0,
                 verbose: bool = True):
        """
        Args:
            layer0_game: The underlying Layer 0 game
            total_energy: Total compute budget (energy)
            verbose: Print progress
        """
        self.layer0_game = layer0_game
        self.total_energy = total_energy
        self.verbose = verbose

        # Shared closure detector (observes all layers)
        self.closure_detector = ClosureDetector()

        # Mode emergence (determines search mode from closure state)
        self.mode_emergence = ModeEmergence(self.closure_detector)

        # Layer media (will be initialized in setup)
        self.layers: List[LayerMedium] = []

        # Wave states per layer
        self.wave_states: List[WaveState] = []

        # Energy allocation
        self.energy_spent: float = 0.0
        self.energy_per_layer: Dict[int, float] = defaultdict(float)

        # Results from each layer
        self.layer0_spines: List[SpinePath] = []
        self.layer1_paths: List[PartialPath] = []
        self.layer2_covers: List[PathCover] = []
        self.layer3_policies: List[CoverPolicy] = []

        # Statistics
        self.stats = {
            'total_iterations': 0,
            'layer_iterations': [0, 0, 0, 0],
            'closures_per_layer': [0, 0, 0, 0],
            'energy_transfers': 0,
            'mode_changes': 0,
        }

        # Current emergent mode per layer
        self.current_modes: List[str] = ["wave", "wave", "wave", "wave"]

    def setup(self,
              forward_seeds: List[Any],
              backward_seeds: List[Any] = None):
        """
        Initialize the wave system with seeds.

        Args:
            forward_seeds: Starting states for Layer 0
            backward_seeds: Boundary states for Layer 0 (optional)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Wave System Initialization")
            print(f"  Total energy: {self.total_energy}")
            print(f"  Forward seeds: {len(forward_seeds)}")
            print(f"{'='*60}")

        # Create Layer 0 medium and solver
        layer0_solver = HOLOSSolver(self.layer0_game, name="wave_layer0")
        self.layers.append(LayerMedium(
            layer_index=0,
            name="positions",
            impedance=1.0,
            damping=0.02,
            coupling_up=0.3,
            solver=layer0_solver,
            closure_detector=self.closure_detector
        ))

        # Create Layer 1 medium and solver
        layer1_solver = PathLayerSolver(
            self.layer0_game,
            self.closure_detector
        )
        self.layers.append(LayerMedium(
            layer_index=1,
            name="paths",
            impedance=2.0,  # Higher impedance = slower propagation
            damping=0.05,
            coupling_up=0.25,
            coupling_down=0.3,
            solver=layer1_solver,
            closure_detector=self.closure_detector
        ))

        # Create Layer 2 medium (will be populated as Layer 1 produces paths)
        self.layers.append(LayerMedium(
            layer_index=2,
            name="covers",
            impedance=4.0,
            damping=0.1,
            coupling_up=0.2,
            coupling_down=0.25,
            solver=None,  # Created dynamically when paths are available
            closure_detector=self.closure_detector
        ))

        # Create Layer 3 medium (will be populated as Layer 2 produces covers)
        self.layers.append(LayerMedium(
            layer_index=3,
            name="policies",
            impedance=8.0,
            damping=0.15,
            coupling_up=0.0,  # No layer above
            coupling_down=0.2,
            solver=None,  # Created dynamically
            closure_detector=self.closure_detector
        ))

        # Initialize wave states
        for _ in range(4):
            self.wave_states.append(WaveState())

        # Seed Layer 0
        for state in forward_seeds:
            layer1_solver.add_forward_seed(state)
            h = self.layer0_game.hash_state(state)
            self.wave_states[0].forward_frontier.add(h)
            self.wave_states[0].amplitude[h] = 1.0

        # Seed Layer 0 backward
        if backward_seeds is None:
            backward_seeds = self.layer0_game.generate_boundary_seeds(
                forward_seeds[0], count=100
            )

        for state in backward_seeds:
            layer1_solver.add_backward_seed(state)
            h = self.layer0_game.hash_state(state)
            self.wave_states[0].backward_frontier.add(h)
            self.wave_states[0].amplitude[h] = 1.0

        # Initialize Layer 1 with seeds
        for state in forward_seeds:
            self.layers[1].solver.add_forward_seed(state)

        for state in backward_seeds:
            self.layers[1].solver.add_backward_seed(state)

        if self.verbose:
            print(f"  Layer 0 frontiers: fwd={len(self.wave_states[0].forward_frontier)}, "
                  f"bwd={len(self.wave_states[0].backward_frontier)}")

    def step(self) -> Dict:
        """
        Perform one step of wave propagation across all layers.

        This is the core "one big wave function" step:
        1. Propagate within each layer (based on emergent mode)
        2. Detect closures
        3. Transfer energy between layers on closure
        4. Update modes based on new closure state

        Returns:
            Dict with step statistics
        """
        self.stats['total_iterations'] += 1
        result = {
            'layer_closures': [0, 0, 0, 0],
            'energy_transfers': 0,
            'mode_changes': 0,
        }

        # Check energy budget
        if self.energy_spent >= self.total_energy:
            return result

        # Step each layer (in parallel conceptually, sequential here)
        for layer_idx, layer in enumerate(self.layers):
            if layer.solver is None:
                continue

            if not self.wave_states[layer_idx].is_active:
                continue

            # Allocate energy to this step
            step_energy = min(
                1.0,  # Base step cost
                self.total_energy - self.energy_spent
            )

            # Get emergent mode for this layer
            mode = self._get_emergent_mode(layer_idx)
            if mode != self.current_modes[layer_idx]:
                self.current_modes[layer_idx] = mode
                result['mode_changes'] += 1

            # Propagate wave
            layer_result = self._step_layer(layer_idx, mode, step_energy)

            # Record closures
            result['layer_closures'][layer_idx] = layer_result.get('closures', 0)
            self.stats['closures_per_layer'][layer_idx] += layer_result.get('closures', 0)
            self.stats['layer_iterations'][layer_idx] += 1

            # Update energy
            self.energy_spent += step_energy
            self.energy_per_layer[layer_idx] += step_energy

            # Handle cross-layer energy transfer on closure
            if layer_result.get('closures', 0) > 0:
                transfers = self._transfer_energy(layer_idx, layer_result)
                result['energy_transfers'] += transfers

        self.stats['energy_transfers'] += result['energy_transfers']
        self.stats['mode_changes'] += result['mode_changes']

        return result

    def _step_layer(self, layer_idx: int, mode: str, energy: float) -> Dict:
        """Step a single layer"""
        layer = self.layers[layer_idx]
        result = {'closures': 0, 'new_states': 0}

        if layer_idx == 0:
            # Layer 0: Run HOLOS on positions (via Layer 1's underlying solver)
            layer1_solver = self.layers[1].solver
            step_result = layer1_solver.step(mode=mode)
            result['closures'] = step_result.get('connections', 0)
            result['new_states'] = step_result.get('extended', 0)

            # Collect spines
            for spine in layer1_solver.completed_spines:
                if spine not in self.layer0_spines:
                    self.layer0_spines.append(spine)

        elif layer_idx == 1:
            # Layer 1: Run path search
            step_result = layer.solver.step(mode=mode)
            result['closures'] = step_result.get('spines', 0)
            result['new_states'] = step_result.get('extended', 0)

            # Collect paths
            for spine in layer.solver.completed_spines:
                # Convert spine to PartialPath for Layer 2
                pass  # Paths are stored in solver

        elif layer_idx == 2 and layer.solver is not None:
            # Layer 2: Run cover search
            step_result = layer.solver.step(mode="balanced")
            result['closures'] = step_result.get('complete', 0)

        elif layer_idx == 3 and layer.solver is not None:
            # Layer 3: Run policy search
            step_result = layer.solver.step(mode="balanced")
            result['closures'] = step_result.get('complete', 0)

        return result

    def _get_emergent_mode(self, layer_idx: int) -> str:
        """Get emergent mode for a layer based on closure state"""
        wave_state = self.wave_states[layer_idx]

        # Get recent closure count for this layer
        recent_closures = sum(
            1 for c in self.closure_detector.closures[-20:]
            if c.layer == layer_idx
        )

        # Estimate branching factor
        branching = 10.0 / (1 + layer_idx)  # Higher layers have lower branching

        return self.mode_emergence.get_emergent_mode(
            forward_frontier_size=len(wave_state.forward_frontier),
            backward_frontier_size=len(wave_state.backward_frontier),
            recent_closures=recent_closures,
            branching_factor=branching
        )

    def _transfer_energy(self, source_layer: int, layer_result: Dict) -> int:
        """Transfer energy to adjacent layers on closure"""
        transfers = 0
        layer = self.layers[source_layer]

        # Transfer up (closure results inform higher layer)
        if source_layer < 3 and layer.coupling_up > 0:
            energy_up = layer_result.get('closures', 0) * layer.coupling_up
            self.wave_states[source_layer + 1].total_energy += energy_up

            # When Layer 0 produces spines, Layer 1 gets new paths
            if source_layer == 0 and layer_result.get('closures', 0) > 0:
                self._propagate_up_from_layer0()
                transfers += 1

            # When Layer 1 produces paths, Layer 2 gets new cover options
            if source_layer == 1 and layer_result.get('closures', 0) > 0:
                self._propagate_up_from_layer1()
                transfers += 1

        # Transfer down (higher layer priorities inform lower layer)
        if source_layer > 0 and layer.coupling_down > 0:
            energy_down = layer_result.get('closures', 0) * layer.coupling_down
            self.wave_states[source_layer - 1].total_energy += energy_down
            transfers += 1

        return transfers

    def _propagate_up_from_layer0(self):
        """Propagate Layer 0 spines to Layer 1"""
        # Spines found at Layer 0 become completed paths at Layer 1
        # This is already handled by the shared solver
        pass

    def _propagate_up_from_layer1(self):
        """Propagate Layer 1 paths to Layer 2"""
        layer1_solver = self.layers[1].solver

        # If we have enough paths, initialize Layer 2
        if len(layer1_solver.completed_spines) >= 5 and self.layers[2].solver is None:
            paths = [(hash(s), s) for s in layer1_solver.completed_spines]
            self.layers[2].solver = CoverLayerSolver(
                paths,
                target_coverage=100,
                closure_detector=self.closure_detector
            )
            if self.verbose:
                print(f"  Layer 2 initialized with {len(paths)} paths")

    def run(self,
            max_iterations: int = 100,
            target_closures: int = None) -> Dict:
        """
        Run the wave system until completion or budget exhausted.

        Args:
            max_iterations: Maximum total iterations
            target_closures: Stop when this many closures found (optional)

        Returns:
            Dict with run statistics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Wave System Run")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Energy budget: {self.total_energy}")
            print(f"{'='*60}")

        start_time = time.time()

        for i in range(max_iterations):
            result = self.step()

            # Check stopping conditions
            if self.energy_spent >= self.total_energy:
                if self.verbose:
                    print(f"\n  Energy budget exhausted at iteration {i}")
                break

            if target_closures and self.closure_detector.stats['total_closures'] >= target_closures:
                if self.verbose:
                    print(f"\n  Target closures reached at iteration {i}")
                break

            # Progress report
            if self.verbose and i % 10 == 0:
                self._print_progress(i)

            # Check if all layers are inactive
            if not any(ws.is_active for ws in self.wave_states):
                if self.verbose:
                    print(f"\n  All layers inactive at iteration {i}")
                break

        elapsed = time.time() - start_time

        if self.verbose:
            self._print_summary(elapsed)

        return {
            'iterations': self.stats['total_iterations'],
            'elapsed': elapsed,
            'closures': self.closure_detector.stats['total_closures'],
            'spines': len(self.layer0_spines),
            'energy_spent': self.energy_spent,
        }

    def _print_progress(self, iteration: int):
        """Print progress report"""
        print(f"\n  Iteration {iteration}:")
        print(f"    Energy: {self.energy_spent:.1f}/{self.total_energy:.1f}")
        print(f"    Closures: {self.closure_detector.stats['total_closures']} "
              f"(irr={self.closure_detector.stats['irreducible']})")
        print(f"    Modes: {self.current_modes}")

        for i, layer in enumerate(self.layers):
            ws = self.wave_states[i]
            print(f"    L{i} ({layer.name}): fwd={len(ws.forward_frontier)}, "
                  f"bwd={len(ws.backward_frontier)}, solved={len(ws.solved)}")

    def _print_summary(self, elapsed: float):
        """Print final summary"""
        print(f"\n{'='*60}")
        print(f"Wave System Summary")
        print(f"{'='*60}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Iterations: {self.stats['total_iterations']}")
        print(f"  Energy spent: {self.energy_spent:.1f}")

        print(f"\n  Closures:")
        print(f"    Total: {self.closure_detector.stats['total_closures']}")
        print(f"    Irreducible: {self.closure_detector.stats['irreducible']}")
        print(f"    Resonant: {self.closure_detector.stats['resonant']}")
        print(f"    By layer: {self.stats['closures_per_layer']}")

        print(f"\n  Results:")
        print(f"    Layer 0 spines: {len(self.layer0_spines)}")

        if self.layers[1].solver:
            print(f"    Layer 1 paths: {len(self.layers[1].solver.completed_spines)}")

        print(f"\n  Energy distribution:")
        for i, energy in self.energy_per_layer.items():
            pct = energy / max(1, self.energy_spent) * 100
            print(f"    Layer {i}: {energy:.1f} ({pct:.1f}%)")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_wave_system(layer0_game: GameInterface,
                       total_energy: float = 1000.0) -> WaveSystem:
    """Create a wave system for the given game"""
    return WaveSystem(layer0_game, total_energy)


def run_wave_search(layer0_game: GameInterface,
                    start_states: List[Any],
                    boundary_states: List[Any] = None,
                    max_iterations: int = 100,
                    energy: float = 1000.0) -> Dict:
    """
    Run wave search on a game.

    Args:
        layer0_game: The game to solve
        start_states: Starting positions
        boundary_states: Boundary positions
        max_iterations: Maximum iterations
        energy: Total energy budget

    Returns:
        Dict with results
    """
    system = create_wave_system(layer0_game, energy)
    system.setup(start_states, boundary_states)
    return system.run(max_iterations)
