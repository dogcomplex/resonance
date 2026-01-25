"""
holos_loop/meta_sieve.py - Stacked Sieves and Rule Learning

What if one sieve's stable patterns become another sieve's rules?

This is how abstraction emerges without being designed:
- Base sieve operates on game states, outputs stable patterns
- Meta sieve treats stable patterns as candidate rules, filters them
- The rules that survive the meta-sieve are "real" rules

This is also how rule learning works:
- Observe state transitions (patterns)
- Patterns that persist = likely real transitions
- Patterns that interfere destructively = noise/coincidence

The stacking can continue:
- Meta-meta-sieve: rules about rules (abstractions of abstractions)
- Each level has slower dynamics (longer time constant)
- Higher levels see coarser structure
"""

from typing import Dict, List, Tuple, Any, Optional, Set, Callable
from dataclasses import dataclass, field
import math

from .sieve import Pattern, Amplitude, Rule, Sieve, solve


# ============================================================
# OBSERVATION -> PATTERN CONVERSION
# ============================================================

@dataclass(frozen=True)
class Observation:
    """
    An observed transition: before -> after

    Observations are the raw data from which rules are inferred.
    """
    before: Any  # State before transition
    after: Any   # State after transition
    context: Any = None  # Optional context/metadata

    def as_pattern(self) -> Pattern:
        """Convert observation to a pattern for the sieve"""
        return Pattern(tokens=(self.before, self.after, self.context))

    def as_rule(self, transfer: complex = 1.0) -> Rule:
        """Convert observation to a candidate rule"""
        return Rule(
            lhs=Pattern(tokens=self.before),
            rhs=Pattern(tokens=self.after),
            transfer=transfer,
            name=f"obs_{hash(self.before)}_{hash(self.after)}"
        )


def observations_to_patterns(observations: List[Observation]) -> List[Pattern]:
    """Convert observations to patterns for sieving"""
    return [obs.as_pattern() for obs in observations]


def patterns_to_rules(patterns: List[Tuple[Pattern, Amplitude]], threshold: float = 0.1) -> List[Rule]:
    """
    Convert surviving patterns back to rules.

    Only patterns with amplitude above threshold become rules.
    Amplitude magnitude -> rule transfer coefficient
    """
    rules = []
    for pattern, amplitude in patterns:
        if amplitude.magnitude < threshold:
            continue

        tokens = pattern.tokens
        if len(tokens) >= 2:
            before, after = tokens[0], tokens[1]
            rules.append(Rule(
                lhs=Pattern(tokens=before),
                rhs=Pattern(tokens=after),
                transfer=complex(amplitude.magnitude, 0),  # Use magnitude as transfer
                name=f"learned_{hash(before)}_{hash(after)}"
            ))

    return rules


# ============================================================
# META-SIEVE: SIEVE OF SIEVES
# ============================================================

class MetaSieve:
    """
    A sieve whose rules come from another sieve's outputs.

    Structure:
        Base sieve: Operates on states, produces stable state-patterns
        Meta sieve: Treats state-patterns as transition-rules, filters them

    The meta sieve answers: "Which transitions are real rules?"
    by letting candidate rules interfere. Rules that persist
    across many observations are probably real.
    """

    def __init__(
        self,
        base_threshold: float = 0.001,
        meta_threshold: float = 0.01,
        coupling: float = 0.5
    ):
        """
        Args:
            base_threshold: Threshold for base sieve (state patterns)
            meta_threshold: Threshold for meta sieve (rule patterns)
            coupling: How much base stable patterns affect meta rules
        """
        self.base = Sieve(threshold=base_threshold)
        self.meta = Sieve(threshold=meta_threshold)
        self.coupling = coupling

        # Learned rules (stable patterns from meta sieve)
        self.rules: List[Rule] = []

        # Statistics
        self.generation = 0
        self.rule_history: List[int] = []  # Number of rules per generation

    def inject_observation(self, obs: Observation, forward: bool = True):
        """
        Inject an observation into the base sieve.

        The observation is treated as a pattern to be tested.
        """
        pattern = obs.as_pattern()
        if forward:
            self.base.inject_forward(pattern)
        else:
            self.base.inject_backward(pattern)

    def inject_observations(self, observations: List[Observation]):
        """Inject multiple observations"""
        n = len(observations)
        amp = 1.0 / math.sqrt(n) if n > 0 else 1.0
        for obs in observations:
            self.base.inject(obs.as_pattern(), Amplitude.forward(amp))

    def evolve(self) -> Dict[str, Any]:
        """
        One evolution step of the coupled system.

        1. Evolve base sieve (filters state patterns)
        2. Promote stable patterns to meta sieve as rule candidates
        3. Evolve meta sieve (filters rule patterns)
        4. Extract surviving rules
        """
        stats = {
            'generation': self.generation,
            'base_patterns': 0,
            'meta_patterns': 0,
            'rules_extracted': 0,
        }

        # Step 1: Evolve base sieve
        # (Need rules for base - use current learned rules or identity)
        base_rules = self.rules if self.rules else [
            # Default: patterns persist with damping
        ]
        self.base.evolve(base_rules)
        stats['base_patterns'] = len(self.base.field)

        # Step 2: Promote stable base patterns to meta sieve
        stable_base = self.base.stable_patterns(min_amplitude=self.base.threshold * 5)
        for pattern, amplitude in stable_base:
            # Convert state-pattern to rule-pattern for meta sieve
            tokens = pattern.tokens
            if len(tokens) >= 2:
                # This is a transition pattern (before, after)
                rule_pattern = Pattern(tokens=('rule', tokens[0], tokens[1]))
                # Inject into meta with amplitude scaled by coupling
                meta_amp = Amplitude(amplitude.value * self.coupling)
                self.meta.inject(rule_pattern, meta_amp)

        # Step 3: Evolve meta sieve
        # Meta rules: how rules interact
        meta_rules = self._meta_rules()
        self.meta.evolve(meta_rules)
        stats['meta_patterns'] = len(self.meta.field)

        # Step 4: Extract rules from stable meta patterns
        stable_meta = self.meta.stable_patterns(min_amplitude=self.meta.threshold * 5)
        self.rules = self._extract_rules(stable_meta)
        stats['rules_extracted'] = len(self.rules)

        self.generation += 1
        self.rule_history.append(len(self.rules))

        return stats

    def _meta_rules(self) -> List[Rule]:
        """
        Rules for the meta sieve - how rule-patterns interact.

        This defines the "physics" of rule learning:
        - Consistent rules reinforce each other
        - Contradictory rules interfere destructively
        - Unused rules fade
        """
        # For now: rules just persist with damping
        # More sophisticated: rules that share LHS compete
        return []

    def _extract_rules(self, stable: List[Tuple[Pattern, Amplitude]]) -> List[Rule]:
        """Convert stable meta patterns to actual rules"""
        rules = []
        for pattern, amplitude in stable:
            tokens = pattern.tokens
            if len(tokens) >= 3 and tokens[0] == 'rule':
                before, after = tokens[1], tokens[2]
                rules.append(Rule(
                    lhs=Pattern(tokens=before),
                    rhs=Pattern(tokens=after),
                    transfer=complex(amplitude.magnitude, 0),
                    name=f"meta_{hash(before)}_{hash(after)}"
                ))
        return rules

    def temperature(self) -> Tuple[float, float]:
        """Temperatures of base and meta sieves"""
        return self.base.temperature(), self.meta.temperature()

    def is_stable(self) -> bool:
        """Is the coupled system stable?"""
        # Stable if both sieves are stable and rule count is stable
        if len(self.rule_history) < 5:
            return False

        recent = self.rule_history[-5:]
        variance = sum((r - sum(recent)/5)**2 for r in recent) / 5
        return variance < 1 and self.base.is_stable() and self.meta.is_stable()

    def summary(self) -> str:
        """Human-readable summary"""
        base_temp, meta_temp = self.temperature()
        return (f"MetaSieve (gen {self.generation}):\n"
                f"  Base: {len(self.base.field)} patterns, temp={base_temp:.3f}\n"
                f"  Meta: {len(self.meta.field)} patterns, temp={meta_temp:.3f}\n"
                f"  Rules: {len(self.rules)}")


# ============================================================
# HIERARCHICAL SIEVE: ARBITRARY DEPTH
# ============================================================

class HierarchicalSieve:
    """
    A stack of sieves with arbitrary depth.

    Each level:
    - Operates on patterns from the level below
    - Has slower dynamics (longer time constant)
    - Sees coarser structure

    This is the full generalization of meta-sieve:
    - Level 0: State patterns
    - Level 1: Transition patterns (rules)
    - Level 2: Rule patterns (meta-rules)
    - Level 3: Meta-rule patterns (meta-meta-rules)
    - ...
    """

    def __init__(
        self,
        depth: int = 3,
        base_threshold: float = 0.001,
        threshold_growth: float = 10.0,
        base_damping: float = 0.99,
        damping_decay: float = 0.9
    ):
        """
        Args:
            depth: Number of sieve levels
            base_threshold: Threshold for level 0
            threshold_growth: Each level's threshold is this factor higher
            base_damping: Damping for level 0
            damping_decay: Each level's damping is this factor smaller (slower)
        """
        self.depth = depth
        self.levels: List[Sieve] = []

        for i in range(depth):
            thresh = base_threshold * (threshold_growth ** i)
            damp = base_damping * (damping_decay ** i)
            self.levels.append(Sieve(threshold=thresh, damping=damp))

        # Coupling between levels (how much lower level feeds upper)
        self.couplings = [0.5] * (depth - 1)

        # Rules at each level
        self.rules_by_level: List[List[Rule]] = [[] for _ in range(depth)]

        self.generation = 0

    def inject(self, pattern: Pattern, level: int = 0):
        """Inject a pattern at a specific level"""
        if 0 <= level < self.depth:
            self.levels[level].inject_forward(pattern)

    def evolve(self) -> Dict[str, Any]:
        """
        Evolve all levels, with upward coupling.

        Lower levels evolve first, then stable patterns promote upward.
        """
        stats = {'generation': self.generation, 'levels': []}

        for i in range(self.depth):
            # Evolve this level with its rules
            rules = self.rules_by_level[i]
            level_stats = self.levels[i].evolve(rules if rules else [])
            stats['levels'].append(level_stats)

            # Promote stable patterns to next level (if not top)
            if i < self.depth - 1:
                stable = self.levels[i].stable_patterns()
                coupling = self.couplings[i]

                for pattern, amplitude in stable:
                    # Transform pattern for next level
                    upper_pattern = self._promote_pattern(pattern, i)
                    upper_amp = Amplitude(amplitude.value * coupling)
                    self.levels[i + 1].inject(upper_pattern, upper_amp)

        # Extract rules from each level
        for i in range(self.depth):
            self.rules_by_level[i] = self._extract_rules_at_level(i)

        self.generation += 1
        return stats

    def _promote_pattern(self, pattern: Pattern, from_level: int) -> Pattern:
        """Transform a pattern for the next level up"""
        # Wrap pattern with level tag
        return Pattern(tokens=(f'L{from_level + 1}', pattern.tokens))

    def _extract_rules_at_level(self, level: int) -> List[Rule]:
        """Extract rules from stable patterns at a level"""
        if level == 0:
            return []  # Level 0 rules come from game/environment

        stable = self.levels[level].stable_patterns()
        rules = []

        for pattern, amplitude in stable:
            tokens = pattern.tokens
            # Patterns are (level_tag, inner_tokens)
            if isinstance(tokens, tuple) and len(tokens) >= 2:
                tag, inner = tokens[0], tokens[1:]
                # inner contains the rule structure
                if len(inner) >= 2:
                    rules.append(Rule(
                        lhs=Pattern(tokens=inner[0]),
                        rhs=Pattern(tokens=inner[1]) if len(inner) > 1 else Pattern(tokens=inner[0]),
                        transfer=complex(amplitude.magnitude, 0),
                        name=f"L{level}_{hash(inner)}"
                    ))

        return rules

    def temperature_profile(self) -> List[float]:
        """Temperature at each level"""
        return [level.temperature() for level in self.levels]

    def is_stable(self) -> bool:
        """Is the entire hierarchy stable?"""
        return all(level.is_stable() for level in self.levels)

    def summary(self) -> str:
        """Human-readable summary"""
        lines = [f"HierarchicalSieve ({self.depth} levels, gen {self.generation}):"]
        temps = self.temperature_profile()
        for i, level in enumerate(self.levels):
            rules = len(self.rules_by_level[i])
            lines.append(f"  L{i}: {len(level.field)} patterns, temp={temps[i]:.3f}, rules={rules}")
        return "\n".join(lines)


# ============================================================
# RULE LEARNING FROM OBSERVATIONS
# ============================================================

def learn_rules(
    observations: List[Observation],
    max_generations: int = 100,
    threshold: float = 0.01,
    verbose: bool = True
) -> List[Rule]:
    """
    Learn rules from observations using the meta-sieve.

    This is unsupervised rule discovery:
    - Inject all observations
    - Let them interfere
    - Surviving patterns are likely rules

    Args:
        observations: List of before->after observations
        max_generations: How long to run
        threshold: Minimum amplitude for a rule to survive
        verbose: Print progress

    Returns:
        List of learned rules
    """
    meta = MetaSieve(meta_threshold=threshold)

    # Inject all observations
    meta.inject_observations(observations)

    # Run until stable
    for gen in range(max_generations):
        stats = meta.evolve()

        if verbose and gen % 10 == 0:
            print(f"Gen {gen}: {meta.summary()}")

        if meta.is_stable():
            if verbose:
                print(f"Stable at generation {gen}")
            break

    return meta.rules


def infer_rules_from_traces(
    traces: List[List[Any]],
    max_generations: int = 100,
    threshold: float = 0.01,
    verbose: bool = True
) -> List[Rule]:
    """
    Infer rules from state traces (sequences of states).

    Each trace is a sequence: [s0, s1, s2, ...] implying transitions
    s0->s1, s1->s2, etc.

    Args:
        traces: List of state sequences
        max_generations: How long to run
        threshold: Minimum amplitude for a rule to survive
        verbose: Print progress

    Returns:
        List of inferred rules
    """
    observations = []
    for trace in traces:
        for i in range(len(trace) - 1):
            observations.append(Observation(before=trace[i], after=trace[i + 1]))

    return learn_rules(observations, max_generations, threshold, verbose)
