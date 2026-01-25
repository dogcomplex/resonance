"""
sieve_core - The Deep Physics of Computation

This package explores what computation IS at the deepest level.

Layers (from practical to fundamental):

1. substrate.py - The computational substrate
   - Configuration, AmplitudeField, Hamiltonian
   - The basic "wave function over configurations"

2. emergence.py - Self-organizing rules
   - Entity (unified states and rules)
   - SelfOrganizingSubstrate
   - Rules emerge from stable patterns

3. information.py - Information as fundamental
   - Distinction (the primitive)
   - DistinctionSpace
   - Logic emerges from distinctions

4. spacetime.py - Emergent spacetime
   - CausalStructure
   - ComputationalMetric
   - EmergentSpacetime

See UNIFIED_THEORY.md for the full theoretical framework.
"""

__version__ = "0.1.0"

# Level 4: Substrate
from .substrate import (
    Configuration,
    DiscreteConfig,
    ContinuousConfig,
    AmplitudeField,
    Hamiltonian,
    RuleHamiltonian,
    LazyHamiltonian,
    Substrate,
    detect_closures,
    solve_on_substrate,
)

# Level 3: Emergence
from .emergence import (
    EntityType,
    Entity,
    EmergentHamiltonian,
    SelfOrganizingSubstrate,
    bootstrap_from_noise,
    learn_physics,
)

# Level 2: Information
from .information import (
    Distinction,
    Side,
    DistinctionSpace,
    LogicalSpace,
    InformationManifold,
    ItFromBit,
    Arrow,
    DistinctionCategory,
    SelfReference,
    InformationalSieve,
    bootstrap_logic,
)

# Level 1: Spacetime
from .spacetime import (
    Event,
    CausalStructure,
    ComputationalMetric,
    LocalityEmergence,
    LightCone,
    compute_light_cone,
    ComputationalGravity,
    DimensionEstimator,
    EmergentSpacetime,
)

__all__ = [
    # Substrate
    "Configuration",
    "DiscreteConfig",
    "ContinuousConfig",
    "AmplitudeField",
    "Hamiltonian",
    "RuleHamiltonian",
    "LazyHamiltonian",
    "Substrate",
    "detect_closures",
    "solve_on_substrate",
    # Emergence
    "EntityType",
    "Entity",
    "EmergentHamiltonian",
    "SelfOrganizingSubstrate",
    "bootstrap_from_noise",
    "learn_physics",
    # Information
    "Distinction",
    "Side",
    "DistinctionSpace",
    "LogicalSpace",
    "InformationManifold",
    "ItFromBit",
    "Arrow",
    "DistinctionCategory",
    "SelfReference",
    "InformationalSieve",
    "bootstrap_logic",
    # Spacetime
    "Event",
    "CausalStructure",
    "ComputationalMetric",
    "LocalityEmergence",
    "LightCone",
    "compute_light_cone",
    "ComputationalGravity",
    "DimensionEstimator",
    "EmergentSpacetime",
]
