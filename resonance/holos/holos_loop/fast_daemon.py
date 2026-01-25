"""
holos/fast_daemon.py - Fast Daemon for Quantum-Like Speedup

The quantum speedup analogy from poor_mans_quantum.txt:
- Wave medium: slow, exponential expansion
- Daemon: fast, can read and boost signals

If the daemon operates FASTER than the wave, it can:
1. Read multiple positions simultaneously
2. Detect interference patterns before they dissipate
3. Boost low-amplitude but high-value paths
4. Collapse superpositions strategically

In hardware terms:
- CPU: Sequential daemon, ~10^6 ops/sec
- GPU: Parallel daemon, ~10^9 ops/sec (1000x speedup)
- Quantum: True superposition, exponential speedup

This module simulates what a "fast daemon" could do, even without actual
GPU implementation. The key insight: batch operations and parallel checks.

Effects we could utilize with speedup:
1. GLOBAL INTERFERENCE: Check all pairs of paths for constructive/destructive
2. AMPLITUDE FOCUSING: Concentrate probability on promising regions
3. GROVER-LIKE SEARCH: Amplitude amplification on marked states
4. PARALLEL COLLAPSE: Measure multiple regions simultaneously
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import math
import time

from holos.holos import GameInterface
from holos.closure import ClosureType, ClosureEvent


# ============================================================
# FAST DAEMON OPERATIONS
# ============================================================

@dataclass
class DaemonMetrics:
    """Track what the daemon observes across the wave."""
    total_amplitude: float = 0.0
    max_amplitude: float = 0.0
    amplitude_concentration: float = 0.0  # Gini coefficient
    interference_potential: float = 0.0   # How much paths could interfere
    closure_density: float = 0.0          # Fraction of frontier that's closed
    value_coverage: float = 0.0           # Fraction with known values


class FastDaemon:
    """
    A daemon that operates faster than the wave expansion.

    Key operations (all should be parallelizable):
    1. scan_all(): Read entire frontier in one "tick"
    2. detect_interference(): Find path pairs that could interfere
    3. amplify(): Boost amplitudes of promising states
    4. collapse_region(): Force measurement in a subregion
    """

    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold
        self.scan_history: List[DaemonMetrics] = []

    def scan_all(
        self,
        amplitudes: Dict[int, complex],
        values: Dict[int, Any],
        closures: Set[int]
    ) -> DaemonMetrics:
        """
        Scan entire state space in one operation.

        On GPU: This would be a single parallel reduction.
        """
        if not amplitudes:
            return DaemonMetrics()

        # Compute all metrics in "parallel"
        amps = [abs(a) for a in amplitudes.values()]
        total = sum(amps)
        max_amp = max(amps) if amps else 0

        # Concentration (simplified Gini)
        if total > 0:
            sorted_amps = sorted(amps)
            n = len(sorted_amps)
            cumsum = 0
            gini_sum = 0
            for i, a in enumerate(sorted_amps):
                cumsum += a
                gini_sum += cumsum
            concentration = 1 - 2 * gini_sum / (n * total) if n > 0 else 0
        else:
            concentration = 0

        # Closure and value coverage
        n_states = len(amplitudes)
        closure_density = len(closures) / n_states if n_states > 0 else 0
        value_coverage = len(values) / n_states if n_states > 0 else 0

        metrics = DaemonMetrics(
            total_amplitude=total,
            max_amplitude=max_amp,
            amplitude_concentration=concentration,
            closure_density=closure_density,
            value_coverage=value_coverage
        )

        self.scan_history.append(metrics)
        return metrics

    def detect_interference_potential(
        self,
        amplitudes: Dict[int, complex],
        neighbors: Dict[int, Set[int]]
    ) -> Dict[int, float]:
        """
        Find states where interference could occur.

        Interference happens when multiple paths reach the same state.
        On GPU: Parallel neighbor lookup and amplitude comparison.
        """
        potential = {}

        for h, amp in amplitudes.items():
            if h not in neighbors:
                continue

            # Sum amplitudes of neighbors
            neighbor_amp_sum = sum(
                abs(amplitudes.get(nh, 0))
                for nh in neighbors[h]
            )

            # High potential = our amplitude similar to neighbor sum
            # (means paths are converging here)
            if neighbor_amp_sum > 0:
                potential[h] = abs(amp) * neighbor_amp_sum

        return potential

    def find_amplification_targets(
        self,
        amplitudes: Dict[int, complex],
        values: Dict[int, Any],
        value_heuristic: Callable[[Any], float] = None
    ) -> List[Tuple[int, float]]:
        """
        Find states that should be amplified (Grover-like).

        Targets are states with:
        1. Low current amplitude (being missed)
        2. High value or potential value

        On GPU: Parallel score computation.
        """
        if value_heuristic is None:
            value_heuristic = lambda v: 1.0 if v is not None else 0.0

        targets = []

        for h, amp in amplitudes.items():
            current_prob = abs(amp) ** 2
            value = values.get(h)
            value_score = value_heuristic(value)

            # Low prob + high value = amplify
            if current_prob < 0.1 and value_score > 0.5:
                boost_needed = value_score / max(0.01, current_prob)
                targets.append((h, boost_needed))

        # Sort by boost needed
        targets.sort(key=lambda x: -x[1])
        return targets[:100]  # Top 100 targets

    def apply_amplification(
        self,
        amplitudes: Dict[int, complex],
        targets: List[Tuple[int, float]],
        boost_factor: float = 2.0
    ) -> Dict[int, complex]:
        """
        Amplify target states (Grover oracle + diffusion).

        This is the "signal boosting" from poor_mans_quantum.txt.
        On GPU: Parallel amplitude updates.
        """
        new_amplitudes = dict(amplitudes)

        # Apply boost to targets
        for h, _ in targets:
            if h in new_amplitudes:
                old = new_amplitudes[h]
                new_amplitudes[h] = old * boost_factor

        # Renormalize (preserve total probability)
        total = sum(abs(a)**2 for a in new_amplitudes.values())
        if total > 0:
            norm = math.sqrt(total)
            new_amplitudes = {h: a/norm for h, a in new_amplitudes.items()}

        return new_amplitudes

    def suggest_collapse_regions(
        self,
        amplitudes: Dict[int, complex],
        closures: Set[int],
        min_density: float = 0.3
    ) -> List[Set[int]]:
        """
        Find regions ripe for collapse (measurement).

        A region should be collapsed when:
        1. High closure density (mostly solved)
        2. High amplitude concentration (clear winner)

        On GPU: Parallel clustering.
        """
        # Simple clustering by amplitude threshold
        high_amp_states = {
            h for h, a in amplitudes.items()
            if abs(a) > self.threshold * 10
        }

        # Find connected components of high-amplitude states
        # (simplified: just return states near closures)
        collapse_candidates = high_amp_states & closures

        if len(collapse_candidates) > min_density * len(high_amp_states):
            return [collapse_candidates]

        return []


# ============================================================
# FAST DAEMON WAVE SYSTEM
# ============================================================

class FastDaemonWaveSystem:
    """
    Wave system with fast daemon for quantum-like effects.

    The daemon operates "between" wave steps, performing:
    1. Global state scan
    2. Interference detection
    3. Amplitude amplification
    4. Strategic collapse
    """

    def __init__(
        self,
        game: GameInterface,
        daemon_frequency: int = 5,  # Daemon acts every N steps
        enable_amplification: bool = True,
        enable_collapse: bool = True
    ):
        self.game = game
        self.daemon = FastDaemon()
        self.daemon_frequency = daemon_frequency
        self.enable_amplification = enable_amplification
        self.enable_collapse = enable_collapse

        # State
        self.states: Dict[int, Any] = {}
        self.amplitudes: Dict[int, complex] = {}
        self.values: Dict[int, Any] = {}
        self.closures: Set[int] = set()

        # Graph structure (for interference detection)
        self.neighbors: Dict[int, Set[int]] = defaultdict(set)

        # Tracking
        self.forward_reached: Set[int] = set()
        self.backward_reached: Set[int] = set()

        self.iteration = 0
        self.daemon_actions: List[Dict] = []

        self.stats = {
            'amplifications': 0,
            'collapses': 0,
            'daemon_scans': 0,
        }

    def initialize(self, forward_seeds: List[Any], backward_seeds: List[Any] = None):
        """Initialize with seeds."""
        # Forward
        n_fwd = len(forward_seeds)
        amp = 1.0 / math.sqrt(n_fwd) if n_fwd > 0 else 0

        for state in forward_seeds:
            h = self.game.hash_state(state)
            self.states[h] = state
            self.amplitudes[h] = complex(amp, 0)
            self.forward_reached.add(h)

        # Backward
        if backward_seeds:
            n_bwd = len(backward_seeds)
            amp_bwd = 1.0 / math.sqrt(n_bwd) if n_bwd > 0 else 0

            for state in backward_seeds:
                h = self.game.hash_state(state)
                self.states[h] = state
                self.backward_reached.add(h)

                old = self.amplitudes.get(h, complex(0, 0))
                self.amplitudes[h] = complex(old.real, amp_bwd)

                if self.game.is_boundary(state):
                    self.values[h] = self.game.get_boundary_value(state)
        else:
            self._generate_backward_seeds(forward_seeds)

    def _generate_backward_seeds(self, templates: List[Any], count: int = 100):
        """Generate backward seeds."""
        if not templates:
            return

        seeds = self.game.generate_boundary_seeds(templates[0], count)
        n_bwd = len(seeds)
        amp = 1.0 / math.sqrt(n_bwd) if n_bwd > 0 else 0

        for state in seeds[:count]:
            h = self.game.hash_state(state)
            self.states[h] = state
            self.backward_reached.add(h)

            old = self.amplitudes.get(h, complex(0, 0))
            self.amplitudes[h] = complex(old.real, amp)

            if self.game.is_boundary(state):
                self.values[h] = self.game.get_boundary_value(state)

    def step(self) -> Dict[str, Any]:
        """One wave step with daemon intervention."""
        self.iteration += 1
        expanded = 0
        new_closures = []

        # Expand (alternating forward/backward)
        if self.iteration % 2 == 0:
            expanded, new_closures = self._expand_forward()
        else:
            expanded, new_closures = self._expand_backward()

        # Process closures
        for event in new_closures:
            self.closures.add(event.state_hash)

        # Daemon intervention
        daemon_result = None
        if self.iteration % self.daemon_frequency == 0:
            daemon_result = self._daemon_intervention()

        return {
            'iteration': self.iteration,
            'expanded': expanded,
            'new_closures': len(new_closures),
            'total_closures': len(self.closures),
            'values': len(self.values),
            'daemon_acted': daemon_result is not None,
        }

    def _expand_forward(self) -> Tuple[int, List[ClosureEvent]]:
        """Expand forward wave."""
        expanded = 0
        closures = []

        frontier = [(h, abs(self.amplitudes.get(h, 0)))
                   for h in self.forward_reached
                   if h in self.states]
        frontier.sort(key=lambda x: -x[1])

        for h, amp in frontier[:20]:
            if amp < 0.001:
                continue

            state = self.states[h]
            successors = list(self.game.get_successors(state))
            if not successors:
                continue

            factor = 1.0 / math.sqrt(len(successors))
            parent_amp = self.amplitudes.get(h, complex(1, 0))

            for child, move in successors:
                ch = self.game.hash_state(child)

                # Track neighbor relationship
                self.neighbors[h].add(ch)
                self.neighbors[ch].add(h)

                if ch not in self.states:
                    self.states[ch] = child
                    expanded += 1

                self.forward_reached.add(ch)

                old = self.amplitudes.get(ch, complex(0, 0))
                new_real = old.real + parent_amp.real * factor
                self.amplitudes[ch] = complex(new_real, old.imag)

                # Closure check
                if ch in self.backward_reached:
                    if abs(self.amplitudes[ch].real) > 0.001 and abs(self.amplitudes[ch].imag) > 0.001:
                        closures.append(ClosureEvent(
                            state_hash=ch,
                            layer=0,
                            closure_type=ClosureType.REDUCIBLE,
                            phase_diff=0.0,
                            forward_value=self.values.get(ch),
                            backward_value=self.values.get(ch),
                            iteration=self.iteration,
                            num_contributing_paths=1
                        ))

                # Boundary check
                if self.game.is_boundary(child):
                    self.values[ch] = self.game.get_boundary_value(child)

        return expanded, closures

    def _expand_backward(self) -> Tuple[int, List[ClosureEvent]]:
        """Expand backward wave."""
        expanded = 0
        closures = []

        frontier = [(h, abs(self.amplitudes.get(h, 0)))
                   for h in self.backward_reached
                   if h in self.states]
        frontier.sort(key=lambda x: -x[1])

        for h, amp in frontier[:20]:
            if amp < 0.001:
                continue

            state = self.states[h]
            predecessors = list(self.game.get_predecessors(state))
            if not predecessors:
                continue

            factor = 1.0 / math.sqrt(len(predecessors))
            parent_amp = self.amplitudes.get(h, complex(0, 1))

            for parent, move in predecessors:
                ph = self.game.hash_state(parent)

                self.neighbors[h].add(ph)
                self.neighbors[ph].add(h)

                if ph not in self.states:
                    self.states[ph] = parent
                    expanded += 1

                self.backward_reached.add(ph)

                old = self.amplitudes.get(ph, complex(0, 0))
                new_imag = old.imag + parent_amp.imag * factor
                self.amplitudes[ph] = complex(old.real, new_imag)

                if ph in self.forward_reached:
                    if abs(self.amplitudes[ph].real) > 0.001 and abs(self.amplitudes[ph].imag) > 0.001:
                        closures.append(ClosureEvent(
                            state_hash=ph,
                            layer=0,
                            closure_type=ClosureType.REDUCIBLE,
                            phase_diff=0.0,
                            forward_value=self.values.get(ph),
                            backward_value=self.values.get(ph),
                            iteration=self.iteration,
                            num_contributing_paths=1
                        ))

        return expanded, closures

    def _daemon_intervention(self) -> Dict[str, Any]:
        """Daemon scans and potentially intervenes."""
        self.stats['daemon_scans'] += 1

        # 1. Global scan
        metrics = self.daemon.scan_all(self.amplitudes, self.values, self.closures)

        action = {
            'iteration': self.iteration,
            'metrics': metrics,
            'amplified': False,
            'collapsed': False,
        }

        # 2. Amplification (Grover-like)
        if self.enable_amplification and metrics.value_coverage < 0.5:
            # Define value heuristic based on game
            def value_heuristic(v):
                if v is None:
                    return 0.0
                # Prioritize winning values
                if hasattr(v, 'value'):
                    return 1.0 if v.value != 0 else 0.5
                return 1.0

            targets = self.daemon.find_amplification_targets(
                self.amplitudes, self.values, value_heuristic
            )

            if targets:
                self.amplitudes = self.daemon.apply_amplification(
                    self.amplitudes, targets, boost_factor=1.5
                )
                self.stats['amplifications'] += 1
                action['amplified'] = True
                action['targets_boosted'] = len(targets)

        # 3. Strategic collapse
        if self.enable_collapse and metrics.closure_density > 0.3:
            regions = self.daemon.suggest_collapse_regions(
                self.amplitudes, self.closures
            )

            if regions:
                # "Collapse" = propagate values aggressively
                for region in regions:
                    self._collapse_region(region)
                self.stats['collapses'] += 1
                action['collapsed'] = True
                action['regions_collapsed'] = len(regions)

        self.daemon_actions.append(action)
        return action

    def _collapse_region(self, region: Set[int]):
        """Force value propagation in a region."""
        # Find valued states in region
        valued = {h for h in region if h in self.values}

        # Propagate to unvalued neighbors
        for h in valued:
            value = self.values[h]
            state = self.states.get(h)
            if not state:
                continue

            # Propagate to children
            for child, _ in self.game.get_successors(state):
                ch = self.game.hash_state(child)
                if ch in region and ch not in self.values:
                    # Simple propagation (should use game.propagate_value)
                    if ch in self.states:
                        child_state = self.states[ch]
                        prop = self.game.propagate_value(child_state, [value])
                        if prop is not None:
                            self.values[ch] = prop

    def run(self, max_iterations: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """Run with daemon intervention."""
        t0 = time.time()

        for i in range(max_iterations):
            result = self.step()

            if verbose and i % 10 == 0:
                print(f"  Iter {i}: closures={result['total_closures']}, "
                      f"values={result['values']}, "
                      f"daemon={'acted' if result['daemon_acted'] else '-'}")

        elapsed = time.time() - t0

        # Final metrics
        final_metrics = self.daemon.scan_all(self.amplitudes, self.values, self.closures)

        return {
            'iterations': self.iteration,
            'elapsed': elapsed,
            'states': len(self.states),
            'closures': len(self.closures),
            'values': len(self.values),
            'value_coverage': final_metrics.value_coverage,
            'amplitude_concentration': final_metrics.amplitude_concentration,
            'daemon_stats': self.stats,
            'daemon_actions': len(self.daemon_actions),
        }


# ============================================================
# GPU-LIKE PARALLEL OPERATIONS (SIMULATED)
# ============================================================

def parallel_closure_check(
    forward_reached: Set[int],
    backward_reached: Set[int],
    amplitudes: Dict[int, complex],
    threshold: float = 0.001
) -> Set[int]:
    """
    Check ALL positions for closure in parallel.

    On GPU: This would be a single kernel launch checking all positions.
    Returns set of closure points.
    """
    # In real GPU: this is O(1) parallel time, O(n) work
    overlap = forward_reached & backward_reached

    closures = set()
    for h in overlap:
        amp = amplitudes.get(h, complex(0, 0))
        if abs(amp.real) > threshold and abs(amp.imag) > threshold:
            closures.add(h)

    return closures


def parallel_amplitude_update(
    amplitudes: Dict[int, complex],
    updates: Dict[int, complex]
) -> Dict[int, complex]:
    """
    Apply all amplitude updates in parallel.

    On GPU: Single kernel for element-wise addition.
    """
    result = dict(amplitudes)
    for h, delta in updates.items():
        old = result.get(h, complex(0, 0))
        result[h] = old + delta
    return result


def parallel_value_propagation(
    values: Dict[int, Any],
    edges: Dict[int, Set[int]],  # parent -> children
    propagate_fn: Callable[[Any, List[Any]], Any],
    max_waves: int = 10
) -> Dict[int, Any]:
    """
    Propagate values through graph in parallel waves.

    Each wave propagates one level. Total time O(depth) not O(nodes).
    On GPU: Each wave is a parallel kernel.
    """
    result = dict(values)

    for wave in range(max_waves):
        updates = {}

        # Find all nodes that can be updated this wave
        for parent, children in edges.items():
            if parent in result:
                continue

            child_values = [result[c] for c in children if c in result]
            if child_values:
                prop = propagate_fn(None, child_values)  # Simplified
                if prop is not None:
                    updates[parent] = prop

        if not updates:
            break

        result.update(updates)

    return result


# ============================================================
# CONVENIENCE
# ============================================================

def run_fast_daemon_search(
    game: GameInterface,
    start_states: List[Any],
    backward_states: List[Any] = None,
    max_iterations: int = 100,
    daemon_frequency: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run search with fast daemon."""
    if verbose:
        print("\n" + "=" * 60)
        print("Fast Daemon Wave Search")
        print("=" * 60)

    system = FastDaemonWaveSystem(
        game,
        daemon_frequency=daemon_frequency,
        enable_amplification=True,
        enable_collapse=True
    )
    system.initialize(start_states, backward_states)

    if verbose:
        print(f"  Forward seeds: {len(start_states)}")
        print(f"  Backward seeds: {len(system.backward_reached)}")
        print(f"  Daemon frequency: every {daemon_frequency} steps")
        print()

    result = system.run(max_iterations=max_iterations, verbose=verbose)

    if verbose:
        print(f"\nResults:")
        print(f"  States: {result['states']}")
        print(f"  Closures: {result['closures']}")
        print(f"  Values: {result['values']} ({result['value_coverage']*100:.1f}% coverage)")
        print(f"  Daemon actions: {result['daemon_actions']}")
        print(f"  Amplifications: {result['daemon_stats']['amplifications']}")
        print(f"  Collapses: {result['daemon_stats']['collapses']}")

    return result
