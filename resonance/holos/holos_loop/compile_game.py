"""
holos_loop/compile_game.py - Compile Games to Sieve Rules

Games are special cases of sieves where:
- Patterns are game states
- Rules are legal moves (deterministic, known)
- Boundaries are terminal states with known values

This module provides adapters to run existing GameInterface
implementations on the sieve substrate.

The compiler doesn't enumerate all rules upfront (exponential).
Instead, it creates a RuleGenerator that produces rules lazily
as the sieve requests them for specific patterns.
"""

from typing import List, Tuple, Any, Optional, Callable, Iterator, Set, Dict
from dataclasses import dataclass
import math

from .sieve import Pattern, Amplitude, Rule, Sieve, solve, value_to_phase
from .holos import GameInterface


# ============================================================
# PATTERN ADAPTERS
# ============================================================

def state_to_pattern(state: Any, game: GameInterface) -> Pattern:
    """Convert a game state to a sieve pattern"""
    # Use game's hash as the pattern token
    h = game.hash_state(state)
    return Pattern(tokens=(h, state))  # Store both hash and state


def pattern_to_state(pattern: Pattern) -> Tuple[int, Any]:
    """Extract hash and state from a pattern"""
    h, state = pattern.tokens
    return h, state


def pattern_hash(pattern: Pattern) -> int:
    """Get just the hash from a pattern"""
    return pattern.tokens[0]


# ============================================================
# RULE GENERATION
# ============================================================

class LazyRuleGenerator:
    """
    Generates rules lazily from a GameInterface.

    Instead of enumerating all possible rules (exponential),
    this generates rules on-demand for patterns in the sieve.

    The sieve calls `rules_for(pattern)` to get applicable rules.
    """

    def __init__(self, game: GameInterface, include_backward: bool = True):
        self.game = game
        self.include_backward = include_backward
        self._rule_cache: Dict[int, List[Rule]] = {}

    def rules_for(self, pattern: Pattern) -> List[Rule]:
        """Get all rules applicable to this pattern"""
        h, state = pattern_to_state(pattern)

        if h in self._rule_cache:
            return self._rule_cache[h]

        rules = []

        # Forward rules (successors)
        try:
            for child_state, move in self.game.get_successors(state):
                child_pattern = state_to_pattern(child_state, self.game)
                rules.append(Rule(
                    lhs=pattern,
                    rhs=child_pattern,
                    transfer=complex(1.0, 0),  # Forward: no phase rotation
                    name=f"fwd_{h}_{move}"
                ))
        except Exception:
            pass  # State might not support successors

        # Backward rules (predecessors)
        if self.include_backward:
            try:
                for parent_state, move in self.game.get_predecessors(state):
                    parent_pattern = state_to_pattern(parent_state, self.game)
                    rules.append(Rule(
                        lhs=pattern,
                        rhs=parent_pattern,
                        transfer=complex(-1.0, 0),  # Backward: phase flip
                        name=f"bwd_{h}_{move}"
                    ))
            except Exception:
                pass

        self._rule_cache[h] = rules
        return rules

    def clear_cache(self):
        """Clear the rule cache (for memory management)"""
        self._rule_cache.clear()


# ============================================================
# SIEVE ADAPTER
# ============================================================

class GameSieve(Sieve):
    """
    A sieve specialized for games.

    Extends the base Sieve with:
    - Lazy rule generation from GameInterface
    - Boundary detection and value injection
    - Game-specific statistics
    """

    def __init__(
        self,
        game: GameInterface,
        threshold: float = 0.001,
        damping: float = 0.99
    ):
        super().__init__(threshold=threshold, damping=damping)
        self.game = game
        self.rule_generator = LazyRuleGenerator(game)

        # Track game-specific info
        self.solved: Dict[int, Any] = {}  # hash -> value
        self.boundaries_hit: Set[int] = set()

    def inject_state(self, state: Any, forward: bool = True, magnitude: float = 1.0):
        """Inject a game state into the sieve"""
        pattern = state_to_pattern(state, self.game)
        if forward:
            self.inject_forward(pattern, magnitude)
        else:
            self.inject_backward(pattern, magnitude)

    def inject_boundary(self, state: Any, value: Any, magnitude: float = 1.0):
        """
        Inject a boundary state with its known value.

        The value is encoded as a phase.
        """
        pattern = state_to_pattern(state, self.game)
        phase = value_to_phase(value)
        self.inject(pattern, Amplitude.from_polar(magnitude, phase))

        # Record the boundary
        h = self.game.hash_state(state)
        self.solved[h] = value
        self.boundaries_hit.add(h)

    def evolve_game(self) -> Dict[str, Any]:
        """
        One evolution step using game rules.

        Differs from base evolve:
        - Rules generated lazily per pattern
        - Boundaries detected and recorded
        - Values propagated
        """
        new_field: Dict[Pattern, Amplitude] = {}
        stats = {
            'generation': self.generation,
            'patterns_in': 0,
            'patterns_out': 0,
            'rules_applied': 0,
            'boundaries_found': 0,
            'closures_detected': 0,
        }

        # Track amplitude contributions for closure detection
        contributions: Dict[Pattern, List[Amplitude]] = {}

        for pattern, amplitude in list(self.field.items()):
            if amplitude.magnitude < self.threshold:
                continue

            stats['patterns_in'] += 1
            h, state = pattern_to_state(pattern)

            # Check if this is a boundary
            if self.game.is_boundary(state):
                if h not in self.solved:
                    value = self.game.get_boundary_value(state)
                    self.solved[h] = value
                    self.boundaries_hit.add(h)
                    stats['boundaries_found'] += 1

                    # Re-inject with value-encoded phase
                    phase = value_to_phase(value)
                    boundary_amp = Amplitude.from_polar(amplitude.magnitude, phase)
                    if pattern not in new_field:
                        new_field[pattern] = Amplitude.zero()
                    new_field[pattern] = new_field[pattern].interfere(boundary_amp)
                    continue

            # Apply damping
            damped = amplitude.scale(self.damping)

            # Get rules for this pattern
            rules = self.rule_generator.rules_for(pattern)

            if not rules:
                # No rules = persist (stable pattern)
                if pattern not in new_field:
                    new_field[pattern] = Amplitude.zero()
                new_field[pattern] = new_field[pattern].interfere(damped)
            else:
                # Apply rules
                for rule in rules:
                    new_pattern, new_amplitude = rule.apply(damped)

                    if new_pattern not in new_field:
                        new_field[new_pattern] = Amplitude.zero()
                        contributions[new_pattern] = []

                    new_field[new_pattern] = new_field[new_pattern].interfere(new_amplitude)
                    contributions[new_pattern].append(new_amplitude)
                    stats['rules_applied'] += 1

        # Detect closures: patterns with contributions from different phases
        for pattern, amps in contributions.items():
            if len(amps) >= 2:
                phases = [a.phase for a in amps]
                phase_spread = max(phases) - min(phases)
                # Forward (0) and backward (π) meeting
                if phase_spread > math.pi / 2 and new_field[pattern].magnitude > self.threshold * 5:
                    self.closures.append((pattern, new_field[pattern], self.generation))
                    stats['closures_detected'] += 1

                    # Record solution
                    h, state = pattern_to_state(pattern)
                    if h not in self.solved:
                        # Value from phase of combined amplitude
                        from .sieve import phase_to_value
                        self.solved[h] = phase_to_value(new_field[pattern].phase)

        # Apply threshold
        self.field = {p: a for p, a in new_field.items() if a.magnitude >= self.threshold}
        stats['patterns_out'] = len(self.field)

        self.generation += 1
        self.history.append(stats)

        return stats

    def solve_game(
        self,
        forward_seeds: List[Any],
        backward_seeds: List[Any] = None,
        max_generations: int = 1000,
        verbose: bool = True
    ) -> Dict[int, Any]:
        """
        Solve the game using the sieve.

        Args:
            forward_seeds: Starting game states
            backward_seeds: Target/boundary states (if None, auto-detect)
            max_generations: Safety limit
            verbose: Print progress

        Returns:
            Dictionary of hash -> value for solved states
        """
        self.clear()
        self.solved.clear()
        self.boundaries_hit.clear()

        # Inject forward seeds
        n_fwd = len(forward_seeds)
        fwd_amp = 1.0 / math.sqrt(n_fwd) if n_fwd > 0 else 1.0
        for state in forward_seeds:
            self.inject_state(state, forward=True, magnitude=fwd_amp)

        # Inject backward seeds (boundaries)
        if backward_seeds:
            n_bwd = len(backward_seeds)
            bwd_amp = 1.0 / math.sqrt(n_bwd) if n_bwd > 0 else 1.0
            for state in backward_seeds:
                if self.game.is_boundary(state):
                    value = self.game.get_boundary_value(state)
                    self.inject_boundary(state, value, bwd_amp)
                else:
                    self.inject_state(state, forward=False, magnitude=bwd_amp)

        # Evolution loop
        for gen in range(max_generations):
            stats = self.evolve_game()

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: patterns={stats['patterns_out']}, "
                      f"solved={len(self.solved)}, "
                      f"closures={len(self.closures)}, "
                      f"temp={self.temperature():.3f}")

            # Check stability
            if self.is_stable():
                if verbose:
                    print(f"Stable at generation {gen}")
                break

            # Check if frozen
            if stats['patterns_out'] == 0:
                if verbose:
                    print(f"Frozen at generation {gen}")
                break

        return self.solved

    def query(self, state: Any) -> Optional[Any]:
        """Query solved value for a state"""
        h = self.game.hash_state(state)
        return self.solved.get(h)


# ============================================================
# COMPILATION FUNCTIONS
# ============================================================

def compile_game_to_rules(
    game: GameInterface,
    seed_states: List[Any],
    max_depth: int = 5
) -> List[Rule]:
    """
    Compile a game to explicit rules by BFS from seeds.

    This is for small games where explicit enumeration is feasible.
    For large games, use GameSieve with lazy rule generation.

    Args:
        game: The game interface
        seed_states: Starting states
        max_depth: How deep to enumerate

    Returns:
        List of explicit rules
    """
    rules = []
    seen: Set[int] = set()
    frontier = list(seed_states)

    for depth in range(max_depth):
        next_frontier = []

        for state in frontier:
            h = game.hash_state(state)
            if h in seen:
                continue
            seen.add(h)

            pattern = state_to_pattern(state, game)

            # Forward rules
            for child_state, move in game.get_successors(state):
                child_pattern = state_to_pattern(child_state, game)
                rules.append(Rule(
                    lhs=pattern,
                    rhs=child_pattern,
                    transfer=complex(1.0, 0),
                    name=f"fwd_{h}_{move}"
                ))

                child_h = game.hash_state(child_state)
                if child_h not in seen:
                    next_frontier.append(child_state)

            # Backward rules (optional, for explicit enumeration)
            for parent_state, move in game.get_predecessors(state):
                parent_pattern = state_to_pattern(parent_state, game)
                rules.append(Rule(
                    lhs=pattern,
                    rhs=parent_pattern,
                    transfer=complex(-1.0, 0),
                    name=f"bwd_{h}_{move}"
                ))

        frontier = next_frontier
        if not frontier:
            break

    return rules


# ============================================================
# CONVENIENCE: RUN GAME ON SIEVE
# ============================================================

def solve_game_on_sieve(
    game: GameInterface,
    start_states: List[Any],
    boundary_states: List[Any] = None,
    max_generations: int = 500,
    threshold: float = 0.001,
    damping: float = 0.99,
    verbose: bool = True
) -> Tuple[Dict[int, Any], GameSieve]:
    """
    Convenience function to solve a game using the sieve.

    Returns:
        (solved_dict, sieve) - The solutions and the sieve object for inspection
    """
    sieve = GameSieve(game, threshold=threshold, damping=damping)
    solved = sieve.solve_game(
        forward_seeds=start_states,
        backward_seeds=boundary_states,
        max_generations=max_generations,
        verbose=verbose
    )
    return solved, sieve
