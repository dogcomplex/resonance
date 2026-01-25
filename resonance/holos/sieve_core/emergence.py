"""
sieve_core/emergence.py - Emergence from Pure Interference

The deepest layer dissolves even the distinction between configurations and rules.

Insight: A "rule" A → B is just another pattern in a higher-order space.
If we put rules and configurations in the same space, rules emerge from
stable interference patterns, just like solutions do.

This is the realization of computational coherentism:
- No fundamental rules
- No fundamental configurations
- Just interference all the way down
- Stable patterns = "what exists"

The universe as self-organizing interference.
"""

import cmath
import math
from typing import Dict, List, Tuple, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

from .substrate import (
    Configuration, DiscreteConfig, AmplitudeField,
    Hamiltonian, Substrate
)


# ============================================================
# UNIFIED SPACE: CONFIGURATIONS AND RULES TOGETHER
# ============================================================

class EntityType(Enum):
    """What kind of entity is this?"""
    STATE = auto()      # A configuration/state
    RULE = auto()       # A transition rule
    META_RULE = auto()  # A rule about rules
    # ... can extend indefinitely


@dataclass(frozen=True)
class Entity(Configuration):
    """
    Unified entity that can be either a state or a rule.

    States: (STATE, tokens)
    Rules: (RULE, from_tokens, to_tokens)
    Meta-rules: (META_RULE, rule1_tokens, rule2_tokens)

    By putting everything in the same space, we can have:
    - Amplitude over states (quantum-like superposition)
    - Amplitude over rules (uncertain which rules apply)
    - Amplitude over meta-rules (uncertain which physics)
    """
    entity_type: EntityType
    content: Tuple[Any, ...]

    def __hash__(self):
        return hash((self.entity_type, self.content))

    def __eq__(self, other):
        return (isinstance(other, Entity) and
                self.entity_type == other.entity_type and
                self.content == other.content)

    def __repr__(self):
        if self.entity_type == EntityType.STATE:
            return f"S{self.content}"
        elif self.entity_type == EntityType.RULE:
            return f"R({self.content[0]}->{self.content[1]})"
        else:
            return f"M{self.content}"

    @staticmethod
    def state(tokens: Any) -> 'Entity':
        if not isinstance(tokens, tuple):
            tokens = (tokens,)
        return Entity(EntityType.STATE, tokens)

    @staticmethod
    def rule(from_tokens: Any, to_tokens: Any) -> 'Entity':
        if not isinstance(from_tokens, tuple):
            from_tokens = (from_tokens,)
        if not isinstance(to_tokens, tuple):
            to_tokens = (to_tokens,)
        return Entity(EntityType.RULE, (from_tokens, to_tokens))


# ============================================================
# SELF-MODIFYING HAMILTONIAN
# ============================================================

class EmergentHamiltonian(Hamiltonian):
    """
    A Hamiltonian that modifies itself based on rule-entity amplitudes.

    The coupling between states A and B is determined by the amplitude
    of the rule-entity (A, B). This means:
    - Rules with high amplitude strongly couple states
    - Rules with low amplitude weakly couple states
    - Rules that destructively interfere effectively don't exist

    The system becomes self-consistent: rules affect states,
    and state evolution affects which rules persist.
    """

    def __init__(self, field: AmplitudeField, base_coupling: complex = 1.0):
        """
        Args:
            field: The unified amplitude field (contains both states and rules)
            base_coupling: Default coupling when rule amplitude is 1
        """
        self.field = field
        self.base_coupling = base_coupling

    def neighbors(self, config: Configuration) -> List[Tuple[Configuration, complex]]:
        """Get neighbors based on rule amplitudes"""
        if not isinstance(config, Entity) or config.entity_type != EntityType.STATE:
            return []

        neighbors = []
        state_tokens = config.content

        # Find all rules that have this state as source
        for entity, amplitude in self.field:
            if not isinstance(entity, Entity):
                continue
            if entity.entity_type != EntityType.RULE:
                continue

            from_tokens, to_tokens = entity.content

            if from_tokens == state_tokens:
                # This rule applies
                target = Entity.state(to_tokens)
                # Coupling strength = rule amplitude * base coupling
                coupling = amplitude * self.base_coupling
                neighbors.append((target, coupling))

        return neighbors

    def apply(self, field: AmplitudeField) -> AmplitudeField:
        """Apply Hamiltonian using current rule amplitudes"""
        result = AmplitudeField(threshold=field.threshold)

        # Only evolve state entities
        for entity, amplitude in field:
            if isinstance(entity, Entity) and entity.entity_type == EntityType.STATE:
                for neighbor, coupling in self.neighbors(entity):
                    result.inject(neighbor, coupling * amplitude)

        return result


# ============================================================
# SELF-ORGANIZING SUBSTRATE
# ============================================================

class SelfOrganizingSubstrate:
    """
    A substrate where rules and states co-evolve.

    The key insight: put rules in the same amplitude field as states.
    Let them interfere with each other.
    Stable configurations = what exists (states)
    Stable rule-configurations = what's possible (physics)

    This is emergence: the laws of physics aren't given, they emerge
    from self-consistent interference patterns.
    """

    def __init__(
        self,
        damping_state: float = 0.1,
        damping_rule: float = 0.01,  # Rules are more stable than states
        threshold: float = 1e-10
    ):
        self.damping_state = damping_state
        self.damping_rule = damping_rule
        self.threshold = threshold

        # Unified field
        self.field = AmplitudeField(threshold=threshold)

        # Self-referential Hamiltonian
        self.H = EmergentHamiltonian(self.field)

        # Time
        self.time = 0.0
        self.dt = 0.1

        # History
        self.history: List[Dict[str, Any]] = []

    def inject_state(self, tokens: Any, amplitude: complex = 1.0):
        """Inject a state"""
        entity = Entity.state(tokens)
        self.field.inject(entity, amplitude)

    def inject_rule(self, from_tokens: Any, to_tokens: Any, amplitude: complex = 1.0):
        """Inject a rule"""
        entity = Entity.rule(from_tokens, to_tokens)
        self.field.inject(entity, amplitude)

    def inject_bidirectional_rule(self, a_tokens: Any, b_tokens: Any, amplitude: complex = 1.0):
        """Inject a rule in both directions (reversible)"""
        self.inject_rule(a_tokens, b_tokens, amplitude)
        self.inject_rule(b_tokens, a_tokens, amplitude.conjugate())

    def step(self, dt: float = None) -> Dict[str, Any]:
        """
        One evolution step.

        States and rules evolve together but with different damping.
        """
        dt = dt or self.dt

        # Count states and rules
        n_states = sum(1 for e, _ in self.field
                       if isinstance(e, Entity) and e.entity_type == EntityType.STATE)
        n_rules = sum(1 for e, _ in self.field
                      if isinstance(e, Entity) and e.entity_type == EntityType.RULE)

        stats = {
            'time': self.time,
            'states': n_states,
            'rules': n_rules,
            'total': len(self.field),
            'norm': self.field.norm(),
        }

        # Apply Hamiltonian to states
        H_field = self.H.apply(self.field)

        # New field
        new_field = AmplitudeField(threshold=self.threshold)

        # Evolve each entity with type-specific damping
        for entity, amplitude in self.field:
            if isinstance(entity, Entity):
                if entity.entity_type == EntityType.STATE:
                    gamma = self.damping_state
                else:
                    gamma = self.damping_rule
            else:
                gamma = self.damping_state

            # Damped original
            damped = amplitude * complex(1.0 - gamma * dt, 0)
            new_field.inject(entity, damped)

        # Add Hamiltonian contribution (only for states)
        H_factor = complex(0, -dt)
        for entity, amplitude in H_field:
            new_field.inject(entity, H_factor * amplitude)

        # Rule reinforcement: rules that successfully mediate transitions get stronger
        # (This makes the system self-organizing)
        self._reinforce_rules(new_field, dt)

        # Normalize to prevent overflow (cap maximum amplitude)
        max_amp = 100.0
        for entity in list(new_field.amplitudes.keys()):
            amp = new_field[entity]
            if abs(amp) > max_amp:
                new_field[entity] = amp * (max_amp / abs(amp))

        self.field = new_field
        self.H.field = new_field  # Update Hamiltonian's reference
        self.time += dt

        self.history.append(stats)
        return stats

    def _reinforce_rules(self, field: AmplitudeField, dt: float):
        """
        Reinforce rules that mediate actual transitions.

        If state A has amplitude and rule A→B exists, and B gets amplitude,
        then the rule A→B is "doing work" and should be reinforced.

        This creates positive feedback for useful rules.
        """
        reinforcement = 0.01 * dt

        for entity, amplitude in list(field):
            if not isinstance(entity, Entity) or entity.entity_type != EntityType.RULE:
                continue

            from_tokens, to_tokens = entity.content
            from_entity = Entity.state(from_tokens)
            to_entity = Entity.state(to_tokens)

            # Check if both endpoints have amplitude
            from_amp = field[from_entity]
            to_amp = field[to_entity]

            if abs(from_amp) > self.threshold and abs(to_amp) > self.threshold:
                # Rule is mediating a transition - reinforce it
                # Reinforcement proportional to geometric mean of endpoint amplitudes
                boost = reinforcement * math.sqrt(abs(from_amp) * abs(to_amp))
                field.inject(entity, boost * (amplitude / abs(amplitude) if abs(amplitude) > 0 else 1))

    def get_states(self) -> List[Tuple[Any, complex]]:
        """Get all states and their amplitudes"""
        return [(e.content, a) for e, a in self.field
                if isinstance(e, Entity) and e.entity_type == EntityType.STATE]

    def get_rules(self) -> List[Tuple[Any, Any, complex]]:
        """Get all rules and their amplitudes"""
        return [(e.content[0], e.content[1], a) for e, a in self.field
                if isinstance(e, Entity) and e.entity_type == EntityType.RULE]

    def dominant_states(self, n: int = 10) -> List[Tuple[Any, complex]]:
        """Get highest-amplitude states"""
        states = self.get_states()
        states.sort(key=lambda x: abs(x[1]), reverse=True)
        return states[:n]

    def dominant_rules(self, n: int = 10) -> List[Tuple[Any, Any, complex]]:
        """Get highest-amplitude rules"""
        rules = self.get_rules()
        rules.sort(key=lambda x: abs(x[2]), reverse=True)
        return rules[:n]

    def temperature(self) -> float:
        """Effective temperature (state entropy)"""
        states = [abs(a) ** 2 for _, a in self.get_states()]
        if not states:
            return 0.0
        total = sum(states)
        if total == 0:
            return 0.0
        probs = [s / total for s in states]
        entropy = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
        max_entropy = math.log(len(states)) if len(states) > 1 else 1
        return entropy / max_entropy

    def rule_entropy(self) -> float:
        """Entropy over rules (how uncertain is the physics?)"""
        rules = [abs(a) ** 2 for _, _, a in self.get_rules()]
        if not rules:
            return 0.0
        total = sum(rules)
        if total == 0:
            return 0.0
        probs = [r / total for r in rules]
        entropy = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
        max_entropy = math.log(len(rules)) if len(rules) > 1 else 1
        return entropy / max_entropy

    def summary(self) -> str:
        n_states = len(self.get_states())
        n_rules = len(self.get_rules())
        return (f"SelfOrganizingSubstrate(t={self.time:.2f}):\n"
                f"  States: {n_states}, Rules: {n_rules}\n"
                f"  State temperature: {self.temperature():.3f}\n"
                f"  Rule entropy: {self.rule_entropy():.3f}\n"
                f"  Top states: {self.dominant_states(3)}\n"
                f"  Top rules: {self.dominant_rules(3)}")


# ============================================================
# EMERGENCE FROM NOISE
# ============================================================

def bootstrap_from_noise(
    n_tokens: int = 10,
    n_initial_states: int = 20,
    n_initial_rules: int = 50,
    evolution_time: float = 100.0,
    dt: float = 0.1,
    verbose: bool = True
) -> SelfOrganizingSubstrate:
    """
    Bootstrap a self-organizing system from random initial conditions.

    This demonstrates emergence: start with random states and rules,
    let them evolve, see what structure emerges.

    The stable patterns that emerge are the "physics" of this universe.
    """
    import random

    substrate = SelfOrganizingSubstrate()

    # Random tokens
    tokens = list(range(n_tokens))

    # Inject random states
    for _ in range(n_initial_states):
        token = random.choice(tokens)
        phase = random.uniform(0, 2 * math.pi)
        mag = random.uniform(0.1, 1.0)
        substrate.inject_state(token, mag * cmath.exp(1j * phase))

    # Inject random rules
    for _ in range(n_initial_rules):
        from_token = random.choice(tokens)
        to_token = random.choice(tokens)
        if from_token != to_token:
            phase = random.uniform(0, 2 * math.pi)
            mag = random.uniform(0.1, 1.0)
            substrate.inject_rule(from_token, to_token, mag * cmath.exp(1j * phase))

    if verbose:
        print(f"Initial: {substrate.summary()}")
        print()

    # Evolve
    steps = int(evolution_time / dt)
    for i in range(steps):
        stats = substrate.step(dt)

        if verbose and i % 100 == 0:
            print(f"t={substrate.time:.1f}: states={stats['states']}, rules={stats['rules']}, "
                  f"temp={substrate.temperature():.3f}, rule_ent={substrate.rule_entropy():.3f}")

    if verbose:
        print()
        print(f"Final: {substrate.summary()}")

    return substrate


# ============================================================
# DISCOVERING RULES FROM OBSERVATIONS
# ============================================================

def learn_physics(
    observations: List[Tuple[Any, Any]],  # (before, after) pairs
    prior_rules: List[Tuple[Any, Any]] = None,
    evolution_time: float = 50.0,
    verbose: bool = True
) -> SelfOrganizingSubstrate:
    """
    Learn the rules (physics) that explain observations.

    Observations are (before, after) state pairs.
    We inject:
    - States for all observed tokens
    - Candidate rules for all observed transitions
    - Let them interfere

    Rules that consistently explain observations get reinforced.
    Rules that contradict observations get damped.

    The surviving rules are the "laws" that best explain the data.
    """
    substrate = SelfOrganizingSubstrate(
        damping_state=0.2,   # States come and go
        damping_rule=0.05    # Rules are more persistent
    )

    # Inject observed transitions as candidate rules
    seen_transitions = set()
    for before, after in observations:
        key = (before, after)
        if key not in seen_transitions:
            seen_transitions.add(key)
            # Inject rule with amplitude proportional to observation count
            count = sum(1 for b, a in observations if b == before and a == after)
            substrate.inject_rule(before, after, math.sqrt(count))

    # Inject prior rules if any
    if prior_rules:
        for from_t, to_t in prior_rules:
            substrate.inject_bidirectional_rule(from_t, to_t, 0.5)

    # Inject states from observations
    all_tokens = set()
    for before, after in observations:
        all_tokens.add(before)
        all_tokens.add(after)

    for token in all_tokens:
        substrate.inject_state(token, 0.3)

    if verbose:
        print(f"Learning from {len(observations)} observations...")
        print(f"Candidate rules: {len(seen_transitions)}")

    # Evolve
    steps = int(evolution_time / substrate.dt)
    for i in range(steps):
        substrate.step()

        # Re-inject observations periodically (they're evidence)
        if i % 20 == 0:
            for before, after in observations:
                substrate.inject_state(before, 0.1)
                substrate.inject_state(after, 0.1)

        if verbose and i % 100 == 0:
            print(f"t={substrate.time:.1f}: temp={substrate.temperature():.3f}, "
                  f"rules={len(substrate.get_rules())}")

    if verbose:
        print()
        print("Learned rules (by strength):")
        for from_t, to_t, amp in substrate.dominant_rules(10):
            print(f"  {from_t} -> {to_t}: {abs(amp):.3f}")

    return substrate
