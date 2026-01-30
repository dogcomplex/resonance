"""
cosmology.py - Cosmological Implications of the Sieve Framework

This file explores:
1. Stable universe configurations and their categorical structure
2. Rules mapped to category theory functors
3. Implications for the origin of the universe
4. Mapping to fundamental forces (strong, weak, EM, gravity)
5. Pseudorandom vs true random initialization
6. Universe-spawning rules and recursive cosmology

Key question: If the sieve is fundamental, what does it tell us about
why OUR universe has the physics it does?
"""

import math
import cmath
import random
import hashlib
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, FrozenSet, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sieve_core.emergence import SelfOrganizingSubstrate
from sieve_core.substrate import DiscreteConfig, RuleHamiltonian, Substrate, AmplitudeField


# ============================================================
# PART 1: STABLE UNIVERSE CONFIGURATIONS
# ============================================================

@dataclass
class UniverseConfiguration:
    """A stable universe configuration with full analysis."""
    rules: FrozenSet[Tuple]
    trial_ids: List[int]

    # Structural properties
    n_rules: int = 0
    reversible_pairs: int = 0
    self_loops: int = 0
    flow_rules: int = 0

    # Categorical properties
    is_groupoid: bool = False  # All morphisms invertible
    is_monoid: bool = False    # Has identity and composition
    has_terminal: bool = False # Has terminal object
    has_initial: bool = False  # Has initial object

    # Physical interpretation
    conserved_quantities: List[int] = field(default_factory=list)
    symmetry_group: str = ""
    force_structure: str = ""


def analyze_stable_configurations(n_tokens: int = 4, n_trials: int = 500):
    """
    Collect and deeply analyze all stable universe configurations.
    """
    print("=" * 70)
    print("PART 1: STABLE UNIVERSE CONFIGURATIONS")
    print("=" * 70)

    universes = defaultdict(list)

    print(f"Running {n_trials} trials with {n_tokens} tokens...")

    for trial in range(n_trials):
        random.seed(trial * 104729)  # Large prime
        np.random.seed(trial * 104729)

        substrate = SelfOrganizingSubstrate(
            damping_state=0.1,
            damping_rule=0.02,
            threshold=0.01
        )

        for _ in range(15):
            i, j = random.randint(0, n_tokens-1), random.randint(0, n_tokens-1)
            amp = complex(random.gauss(0, 1), random.gauss(0, 1))
            substrate.inject_rule((i,), (j,), amp)

        for _ in range(100):
            substrate.step(0.1)

        final_rules = frozenset(
            (f, t) for f, t, a in substrate.get_rules() if abs(a) > 0.5
        )

        universes[final_rules].append(trial)

    print(f"\nFound {len(universes)} distinct stable configurations")

    # Analyze each configuration
    configs = []
    for rules, trials in universes.items():
        config = UniverseConfiguration(rules=rules, trial_ids=trials)
        config.n_rules = len(rules)

        # Count structure types
        for r in rules:
            if r[0] == r[1]:
                config.self_loops += 1
            elif (r[1], r[0]) in rules:
                config.reversible_pairs += 0.5  # Count each pair once
            else:
                config.flow_rules += 1

        config.reversible_pairs = int(config.reversible_pairs)

        # Categorical analysis
        analyze_categorical_structure(config, n_tokens)

        # Physical interpretation
        analyze_physical_structure(config, n_tokens)

        configs.append(config)

    # Sort by frequency
    configs.sort(key=lambda c: -len(c.trial_ids))

    return configs, universes


def analyze_categorical_structure(config: UniverseConfiguration, n_tokens: int):
    """
    Analyze the categorical structure of a universe configuration.

    Category theory interpretation:
    - Objects = tokens (0, 1, 2, ...)
    - Morphisms = rules (arrows between tokens)
    - Composition = rule chaining
    - Identity = self-loops
    """
    rules = config.rules

    # Check for identities (self-loops for each token)
    tokens_with_identity = {r[0][0] for r in rules if r[0] == r[1]}
    all_tokens = {t for r in rules for t in r[0] + r[1]}

    config.is_monoid = len(tokens_with_identity) == len(all_tokens) and len(all_tokens) > 0

    # Check for invertibility (groupoid)
    reversible_count = sum(1 for r in rules if (r[1], r[0]) in rules)
    non_self_loops = sum(1 for r in rules if r[0] != r[1])

    config.is_groupoid = (reversible_count == non_self_loops) and non_self_loops > 0

    # Check for terminal object (all arrows point to it)
    in_degree = Counter()
    out_degree = Counter()
    for r in rules:
        if r[0] != r[1]:  # Exclude self-loops
            for t in r[0]:
                out_degree[t] += 1
            for t in r[1]:
                in_degree[t] += 1

    if in_degree:
        max_in = max(in_degree.values())
        terminals = [t for t, d in in_degree.items() if d == max_in and out_degree.get(t, 0) == 0]
        config.has_terminal = len(terminals) > 0

    if out_degree:
        max_out = max(out_degree.values())
        initials = [t for t, d in out_degree.items() if d == max_out and in_degree.get(t, 0) == 0]
        config.has_initial = len(initials) > 0


def analyze_physical_structure(config: UniverseConfiguration, n_tokens: int):
    """
    Analyze the physical interpretation of a universe configuration.
    """
    rules = config.rules

    # Find conserved quantities (tokens with equal in/out flow)
    in_flow = Counter()
    out_flow = Counter()

    for r in rules:
        if r[0] != r[1]:
            for t in r[0]:
                out_flow[t] += 1
            for t in r[1]:
                in_flow[t] += 1

    config.conserved_quantities = [
        t for t in set(in_flow.keys()) | set(out_flow.keys())
        if in_flow.get(t, 0) == out_flow.get(t, 0) and in_flow.get(t, 0) > 0
    ]

    # Determine symmetry structure
    if config.is_groupoid:
        config.symmetry_group = "Full symmetry (groupoid)"
    elif config.reversible_pairs > config.flow_rules:
        config.symmetry_group = "Partial symmetry (mostly reversible)"
    elif config.self_loops > config.n_rules // 2:
        config.symmetry_group = "Identity-dominated (static)"
    else:
        config.symmetry_group = "Asymmetric (directional)"

    # Force structure interpretation
    if config.n_rules <= 2:
        config.force_structure = "Minimal (one force)"
    elif config.has_terminal and config.has_initial:
        config.force_structure = "Hierarchical (gravity-like)"
    elif config.is_groupoid:
        config.force_structure = "Gauge (EM-like)"
    elif len(config.conserved_quantities) > 0:
        config.force_structure = "Conserved (strong-like)"
    else:
        config.force_structure = "Dissipative (weak-like)"


# ============================================================
# PART 2: CATEGORY THEORY FUNCTORS
# ============================================================

class Functor(Enum):
    """Category theory functors that rules can represent."""
    IDENTITY = auto()      # Id: C -> C, does nothing
    FORGETFUL = auto()     # U: C -> Set, loses structure
    FREE = auto()          # F: Set -> C, adds structure
    HOM = auto()           # Hom(A, -): C -> Set, morphisms from A
    PRODUCT = auto()       # A x -: C -> C, pairs with A
    COPRODUCT = auto()     # A + -: C -> C, disjoint union with A
    EXPONENTIAL = auto()   # (-)^A: C -> C, function space
    MONAD = auto()         # T: C -> C with unit and multiplication
    COMONAD = auto()       # W: C -> C, dual of monad


@dataclass
class RuleFunctor:
    """A rule interpreted as a categorical functor."""
    rule: Tuple
    functor_type: Functor
    description: str
    physical_analog: str


def classify_rule_as_functor(rule: Tuple, all_rules: FrozenSet) -> RuleFunctor:
    """
    Classify a single rule as a category theory functor.
    """
    from_t, to_t = rule

    # Self-loop = identity functor
    if from_t == to_t:
        return RuleFunctor(
            rule=rule,
            functor_type=Functor.IDENTITY,
            description=f"Identity on {from_t}",
            physical_analog="Rest mass / vacuum stability"
        )

    # Check if reverse exists
    has_reverse = (to_t, from_t) in all_rules

    if has_reverse:
        # Reversible pair = groupoid morphism
        # This is like a gauge transformation
        return RuleFunctor(
            rule=rule,
            functor_type=Functor.HOM,
            description=f"Invertible morphism {from_t} <-> {to_t}",
            physical_analog="Gauge boson / force carrier"
        )

    # Check connectivity patterns
    # Does this rule's target have many inputs? (Terminal-like)
    inputs_to_target = sum(1 for r in all_rules if r[1] == to_t and r[0] != to_t)

    if inputs_to_target > 2:
        return RuleFunctor(
            rule=rule,
            functor_type=Functor.FORGETFUL,
            description=f"Many-to-one: {from_t} -> {to_t} (attractor)",
            physical_analog="Gravitational attraction / entropy increase"
        )

    # Does this rule's source have many outputs? (Initial-like)
    outputs_from_source = sum(1 for r in all_rules if r[0] == from_t and r[1] != from_t)

    if outputs_from_source > 2:
        return RuleFunctor(
            rule=rule,
            functor_type=Functor.FREE,
            description=f"One-to-many: {from_t} -> multiple targets",
            physical_analog="Particle decay / spontaneous emission"
        )

    # Check for cycle participation
    # A -> B -> ... -> A forms a monad-like structure
    visited = {from_t}
    current = to_t
    cycle_length = 1

    while current not in visited and cycle_length < 10:
        visited.add(current)
        next_steps = [r[1] for r in all_rules if r[0] == current and r[0] != r[1]]
        if not next_steps:
            break
        current = next_steps[0]  # Follow first available
        cycle_length += 1

    if current == from_t and cycle_length > 1:
        return RuleFunctor(
            rule=rule,
            functor_type=Functor.MONAD,
            description=f"Part of cycle length {cycle_length}",
            physical_analog="Oscillation / bound state"
        )

    # Default: simple flow
    return RuleFunctor(
        rule=rule,
        functor_type=Functor.COPRODUCT,
        description=f"Simple transition {from_t} -> {to_t}",
        physical_analog="State transition / quantum jump"
    )


def analyze_functors(configs: List[UniverseConfiguration]):
    """
    Analyze all rules across configurations as functors.
    """
    print("\n" + "=" * 70)
    print("PART 2: RULES AS CATEGORY THEORY FUNCTORS")
    print("=" * 70)

    functor_counts = Counter()
    functor_examples = defaultdict(list)

    for config in configs[:50]:  # Top 50 most common
        for rule in config.rules:
            rf = classify_rule_as_functor(rule, config.rules)
            functor_counts[rf.functor_type] += len(config.trial_ids)
            if len(functor_examples[rf.functor_type]) < 3:
                functor_examples[rf.functor_type].append(rf)

    print("\nFunctor distribution across universes:")
    total = sum(functor_counts.values())

    for functor, count in functor_counts.most_common():
        pct = 100 * count / total
        print(f"\n  {functor.name}: {pct:.1f}%")
        for ex in functor_examples[functor]:
            print(f"    Example: {ex.rule}")
            print(f"      {ex.description}")
            print(f"      Physical: {ex.physical_analog}")

    return functor_counts, functor_examples


# ============================================================
# PART 3: IMPLICATIONS FOR FUNDAMENTAL FORCES
# ============================================================

def map_to_fundamental_forces(configs: List[UniverseConfiguration]):
    """
    Map universe configurations to fundamental force structures.
    """
    print("\n" + "=" * 70)
    print("PART 3: MAPPING TO FUNDAMENTAL FORCES")
    print("=" * 70)

    print("""
THE FOUR FORCES IN TERMS OF SIEVE STRUCTURE:

1. GRAVITY (Spacetime Curvature)
   - Sieve analog: Complexity gradient / many-to-one funneling
   - Functor: Forgetful functor (loses structure, increases entropy)
   - Rule signature: Multiple sources -> single sink
   - Emerges from: Asymmetric rule distributions

2. ELECTROMAGNETISM (Gauge Symmetry U(1))
   - Sieve analog: Reversible exchange / groupoid structure
   - Functor: Hom functor (invertible morphisms)
   - Rule signature: A <-> B with full reversibility
   - Emerges from: Conservation + symmetry

3. WEAK FORCE (Broken Gauge Symmetry)
   - Sieve analog: Partially reversible / symmetry breaking
   - Functor: Not quite Hom (some inverses missing)
   - Rule signature: A -> B exists, B -> A weak or absent
   - Emerges from: Asymmetric damping

4. STRONG FORCE (SU(3) Confinement)
   - Sieve analog: Cyclic closure / monad structure
   - Functor: Monad (cycles that close)
   - Rule signature: A -> B -> C -> A (confined loop)
   - Emerges from: Self-reinforcing cycles
    """)

    # Analyze which universes have which force structures
    force_universes = defaultdict(list)

    for config in configs:
        rules = config.rules

        # Detect gravity-like (many-to-one)
        in_degrees = Counter()
        for r in rules:
            if r[0] != r[1]:
                for t in r[1]:
                    in_degrees[t] += 1

        if in_degrees and max(in_degrees.values()) >= 3:
            force_universes['gravity'].append(config)

        # Detect EM-like (full groupoid)
        if config.is_groupoid:
            force_universes['electromagnetism'].append(config)

        # Detect weak-like (partial symmetry)
        elif config.reversible_pairs > 0 and config.flow_rules > config.reversible_pairs:
            force_universes['weak'].append(config)

        # Detect strong-like (cycles)
        # Check for cycles
        has_cycle = False
        for r in rules:
            if r[0] != r[1]:
                visited = {r[0]}
                current = r[1]
                for _ in range(len(rules)):
                    next_steps = [r2[1] for r2 in rules if r2[0] == current and r2[0] != r2[1]]
                    if r[0] in [n for n in next_steps]:
                        has_cycle = True
                        break
                    if not next_steps:
                        break
                    current = next_steps[0]
                if has_cycle:
                    break

        if has_cycle:
            force_universes['strong'].append(config)

    print("\nForce structure prevalence:")
    for force, configs_list in force_universes.items():
        total_trials = sum(len(c.trial_ids) for c in configs_list)
        print(f"  {force.upper()}: {len(configs_list)} universes, {total_trials} trials")

    # Find universes with multiple forces (like ours!)
    print("\nUniverses with multiple force types:")
    multi_force = []
    for config in configs:
        forces = []
        if config in force_universes.get('gravity', []):
            forces.append('G')
        if config in force_universes.get('electromagnetism', []):
            forces.append('EM')
        if config in force_universes.get('weak', []):
            forces.append('W')
        if config in force_universes.get('strong', []):
            forces.append('S')

        if len(forces) >= 2:
            multi_force.append((config, forces))

    for config, forces in multi_force[:5]:
        print(f"  {'+'.join(forces)}: {len(config.trial_ids)} trials, {config.n_rules} rules")

    return force_universes


# ============================================================
# PART 4: PSEUDORANDOM VS TRUE RANDOM
# ============================================================

def analyze_pseudorandom_effects(n_tokens: int = 4, n_trials: int = 200):
    """
    Compare true random vs pseudorandom initialization.

    Key question: Does the PRNG algorithm affect which universes emerge?
    """
    print("\n" + "=" * 70)
    print("PART 4: PSEUDORANDOM VS TRUE RANDOM EFFECTS")
    print("=" * 70)

    # Test different PRNG methods
    prng_methods = {
        'python_random': lambda seed: random.seed(seed),
        'numpy_random': lambda seed: np.random.seed(seed),
        'sha256_hash': lambda seed: None,  # Will use hash-based
    }

    results = {}

    for method_name, seed_fn in prng_methods.items():
        universes = defaultdict(list)

        for trial in range(n_trials):
            if method_name == 'sha256_hash':
                # Use cryptographic hash for "more random" sequence
                hash_input = f"trial_{trial}_salt_cosmology".encode()
                hash_bytes = hashlib.sha256(hash_input).digest()
                # Convert to sequence of floats
                random_floats = [b / 255.0 for b in hash_bytes]
                random_idx = [0]

                def get_random():
                    idx = random_idx[0] % len(random_floats)
                    random_idx[0] += 1
                    return random_floats[idx]

                def get_randint(a, b):
                    return int(get_random() * (b - a + 1)) + a

                def get_gauss(mu, sigma):
                    # Box-Muller approximation with safety bounds
                    u1 = max(0.001, min(0.999, get_random()))
                    u2 = get_random()
                    return mu + sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            else:
                seed_fn(trial * 31337)
                get_randint = lambda a, b: random.randint(a, b)
                get_gauss = lambda mu, sigma: random.gauss(mu, sigma)

            substrate = SelfOrganizingSubstrate(
                damping_state=0.1,
                damping_rule=0.02,
                threshold=0.01
            )

            for _ in range(15):
                i, j = get_randint(0, n_tokens-1), get_randint(0, n_tokens-1)
                amp = complex(get_gauss(0, 1), get_gauss(0, 1))
                substrate.inject_rule((i,), (j,), amp)

            for _ in range(100):
                substrate.step(0.1)

            final_rules = frozenset(
                (f, t) for f, t, a in substrate.get_rules() if abs(a) > 0.5
            )

            universes[final_rules].append(trial)

        results[method_name] = universes

        n_unique = len(universes)
        top_5_coverage = sum(len(t) for t in sorted(universes.values(), key=len, reverse=True)[:5])

        print(f"\n{method_name}:")
        print(f"  Unique universes: {n_unique}")
        print(f"  Top 5 coverage: {top_5_coverage}/{n_trials} ({100*top_5_coverage/n_trials:.1f}%)")

    # Compare: Do different PRNGs produce different universe distributions?
    print("\n--- PRNG Comparison ---")

    all_universes = set()
    for method_universes in results.values():
        all_universes.update(method_universes.keys())

    print(f"Total unique universes across all methods: {len(all_universes)}")

    # Find universes that appear in all methods
    common = all_universes.copy()
    for method_universes in results.values():
        common &= set(method_universes.keys())

    print(f"Universes appearing in ALL methods: {len(common)}")

    # Find method-specific universes
    for method, method_universes in results.items():
        unique_to_method = set(method_universes.keys())
        for other_method, other_universes in results.items():
            if other_method != method:
                unique_to_method -= set(other_universes.keys())
        print(f"  Unique to {method}: {len(unique_to_method)}")

    return results


# ============================================================
# PART 5: UNIVERSE-SPAWNING RULES
# ============================================================

def find_universe_spawning_rules(n_tokens: int = 4, n_trials: int = 100):
    """
    Find rules capable of generating pseudorandomness,
    which could spawn child universes.

    Key insight: If a rule-set can produce unpredictable outputs
    from deterministic inputs, it can seed new universes.
    """
    print("\n" + "=" * 70)
    print("PART 5: UNIVERSE-SPAWNING RULES")
    print("=" * 70)

    print("""
UNIVERSE SPAWNING MECHANISM:

A rule-set can spawn a child universe if it can:
1. AMPLIFY small differences (chaos/sensitivity)
2. PRODUCE varied outputs (entropy generation)
3. MAINTAIN coherence (not just noise)

These are the same properties as a good PRNG:
- Deterministic but unpredictable
- Wide output distribution
- Long period before repetition
    """)

    # Collect universe data
    universes = defaultdict(list)
    trajectories = []

    for trial in range(n_trials):
        random.seed(trial * 7919)
        np.random.seed(trial * 7919)

        substrate = SelfOrganizingSubstrate(
            damping_state=0.1,
            damping_rule=0.02,
            threshold=0.01
        )

        initial = set()
        for _ in range(15):
            i, j = random.randint(0, n_tokens-1), random.randint(0, n_tokens-1)
            amp = complex(random.gauss(0, 1), random.gauss(0, 1))
            substrate.inject_rule((i,), (j,), amp)
            initial.add(((i,), (j,)))

        # Track state at each step
        state_sequence = []
        for step in range(100):
            substrate.step(0.1)
            if step % 5 == 0:
                state = frozenset(
                    (f, t, round(abs(a), 2))
                    for f, t, a in substrate.get_rules()
                    if abs(a) > 0.1
                )
                state_sequence.append(state)

        final_rules = frozenset(
            (f, t) for f, t, a in substrate.get_rules() if abs(a) > 0.5
        )

        universes[final_rules].append(trial)
        trajectories.append({
            'trial': trial,
            'initial': frozenset(initial),
            'sequence': state_sequence,
            'final': final_rules
        })

    # Analyze which rule-sets produce the most "entropy" (varied behavior)
    print("\n--- Entropy Generation Analysis ---")

    rule_entropy = {}

    for config in set(universes.keys()):
        if len(universes[config]) < 2:
            continue

        # Get all trajectories that led to this configuration
        config_trajs = [t for t in trajectories if t['final'] == config]

        # Measure trajectory diversity
        all_intermediate_states = []
        for traj in config_trajs:
            all_intermediate_states.extend(traj['sequence'])

        unique_states = len(set(all_intermediate_states))
        total_states = len(all_intermediate_states)

        if total_states > 0:
            entropy = unique_states / total_states
            rule_entropy[config] = entropy

    # Find high-entropy configurations
    sorted_by_entropy = sorted(rule_entropy.items(), key=lambda x: -x[1])

    print("\nHighest entropy configurations (potential universe spawners):")
    for config, entropy in sorted_by_entropy[:5]:
        print(f"\n  Entropy: {entropy:.3f}")
        print(f"  Rules: {list(config)[:5]}...")
        print(f"  Appearances: {len(universes[config])}")

        # Analyze structure
        n_reversible = sum(1 for r in config if (r[1], r[0]) in config) // 2
        n_cycles = 0

        # Detect cycles
        for r in config:
            if r[0] != r[1]:
                visited = {r[0]}
                current = r[1]
                for _ in range(len(config)):
                    nexts = [r2[1] for r2 in config if r2[0] == current and r2[0] != r2[1]]
                    if r[0] in nexts:
                        n_cycles += 1
                        break
                    if not nexts:
                        break
                    current = nexts[0]
                    if current in visited:
                        break
                    visited.add(current)

        print(f"  Reversible pairs: {n_reversible}")
        print(f"  Cycles detected: {n_cycles}")

    # The key insight: Which structural features enable entropy generation?
    print("\n--- Structural Features of Entropy Generators ---")

    high_entropy_configs = [c for c, e in sorted_by_entropy[:10]]
    low_entropy_configs = [c for c, e in sorted_by_entropy[-10:] if e > 0]

    def analyze_features(configs, label):
        if not configs:
            return {}

        features = {
            'avg_rules': np.mean([len(c) for c in configs]),
            'avg_reversible': np.mean([sum(1 for r in c if (r[1], r[0]) in c) / 2 for c in configs]),
            'avg_self_loops': np.mean([sum(1 for r in c if r[0] == r[1]) for c in configs]),
        }

        print(f"\n{label}:")
        for feat, val in features.items():
            print(f"  {feat}: {val:.2f}")

        return features

    high_feat = analyze_features(high_entropy_configs, "High entropy (spawners)")
    low_feat = analyze_features(low_entropy_configs, "Low entropy (stable)")

    return universes, trajectories, rule_entropy


# ============================================================
# PART 6: RECURSIVE COSMOLOGY
# ============================================================

def simulate_recursive_cosmology(n_generations: int = 3, n_tokens: int = 4):
    """
    Simulate universes spawning child universes.

    A "parent" universe's stable rules are used to initialize
    the PRNG for child universes.
    """
    print("\n" + "=" * 70)
    print("PART 6: RECURSIVE COSMOLOGY")
    print("=" * 70)

    print("""
RECURSIVE UNIVERSE SPAWNING:

Generation 0: True random initialization
Generation 1: Use Gen-0's stable rules as PRNG seed
Generation 2: Use Gen-1's stable rules as PRNG seed
...

Question: Do child universes inherit structure from parents?
    """)

    generations = []

    # Generation 0: True random
    gen0_universes = []

    print("\n--- Generation 0 (True Random) ---")
    for trial in range(20):
        random.seed(trial * 999961)
        np.random.seed(trial * 999961)

        substrate = SelfOrganizingSubstrate(
            damping_state=0.1,
            damping_rule=0.02
        )

        for _ in range(15):
            i, j = random.randint(0, n_tokens-1), random.randint(0, n_tokens-1)
            substrate.inject_rule((i,), (j,), complex(random.gauss(0, 1), random.gauss(0, 1)))

        for _ in range(100):
            substrate.step(0.1)

        final = frozenset((f, t) for f, t, a in substrate.get_rules() if abs(a) > 0.5)
        gen0_universes.append(final)

    generations.append(gen0_universes)
    print(f"  Created {len(gen0_universes)} universes")
    print(f"  Unique: {len(set(gen0_universes))}")

    # Subsequent generations: Use parent rules as seed
    for gen in range(1, n_generations):
        print(f"\n--- Generation {gen} (Seeded from Gen-{gen-1}) ---")

        gen_universes = []
        parent_universes = generations[gen - 1]

        for parent_idx, parent in enumerate(parent_universes):
            # Use parent's rules as seed
            seed_string = str(sorted(parent))
            seed_hash = int(hashlib.sha256(seed_string.encode()).hexdigest()[:8], 16)

            random.seed(seed_hash)
            np.random.seed(seed_hash % (2**32))

            substrate = SelfOrganizingSubstrate(
                damping_state=0.1,
                damping_rule=0.02
            )

            for _ in range(15):
                i, j = random.randint(0, n_tokens-1), random.randint(0, n_tokens-1)
                substrate.inject_rule((i,), (j,), complex(random.gauss(0, 1), random.gauss(0, 1)))

            for _ in range(100):
                substrate.step(0.1)

            final = frozenset((f, t) for f, t, a in substrate.get_rules() if abs(a) > 0.5)
            gen_universes.append(final)

        generations.append(gen_universes)
        print(f"  Created {len(gen_universes)} universes")
        print(f"  Unique: {len(set(gen_universes))}")

        # Check similarity to parent generation
        same_as_parent = sum(1 for i, u in enumerate(gen_universes)
                            if u == parent_universes[i])
        print(f"  Identical to parent: {same_as_parent}")

        # Check rule overlap
        overlaps = []
        for i, child in enumerate(gen_universes):
            parent = parent_universes[i]
            if len(parent) > 0 and len(child) > 0:
                overlap = len(parent & child) / len(parent | child)
                overlaps.append(overlap)

        if overlaps:
            print(f"  Average rule overlap with parent: {np.mean(overlaps):.3f}")

    # Analyze lineages
    print("\n--- Lineage Analysis ---")

    # Track which rules persist across generations
    rule_persistence = Counter()

    for gen_idx, gen in enumerate(generations):
        for universe in gen:
            for rule in universe:
                rule_persistence[rule] += 1

    print("\nMost persistent rules (across all generations):")
    for rule, count in rule_persistence.most_common(5):
        print(f"  {rule}: {count} appearances")

    # Find "immortal" lineages (same rules persist through generations)
    print("\nImmortality analysis:")
    for trial_idx in range(min(5, len(generations[0]))):
        lineage = [gen[trial_idx] for gen in generations if trial_idx < len(gen)]
        persistent = lineage[0]
        for gen_rules in lineage[1:]:
            persistent = persistent & gen_rules

        if persistent:
            print(f"  Lineage {trial_idx}: {len(persistent)} immortal rules")
            print(f"    {list(persistent)[:3]}...")

    return generations


# ============================================================
# PART 7: THE ORIGIN OF THE UNIVERSE
# ============================================================

def analyze_cosmological_implications(configs: List[UniverseConfiguration]):
    """
    What does all this imply about the origin of our universe?
    """
    print("\n" + "=" * 70)
    print("PART 7: COSMOLOGICAL IMPLICATIONS")
    print("=" * 70)

    print("""
IMPLICATIONS FOR THE ORIGIN OF THE UNIVERSE:

1. THE INITIAL STATE
   - Our universe emerged from a random/pseudorandom initial configuration
   - The specific laws we observe are ONE stable solution among many
   - Different initial conditions would yield different physics

2. WHY THESE LAWS?
   - Not anthropic selection, not fine-tuning
   - These are the rules that SURVIVED interference
   - Other rule-sets self-destructed through inconsistency

3. THE FOUR FORCES
   - GRAVITY: Emerges from complexity gradients (many-to-one funneling)
   - EM: Emerges from perfect symmetry (groupoid structure)
   - WEAK: Emerges from broken symmetry (partial reversibility)
   - STRONG: Emerges from cyclic closure (confined loops)

4. CONSERVATION LAWS
   - Energy/momentum conservation = balanced in/out flow
   - Charge conservation = tokens that appear equally as source and target
   - These aren't imposed - they EMERGE from stable configurations

5. THE BIG BANG
   - May have been the moment rules "crystallized" from quantum foam
   - Not an explosion of matter, but a phase transition of RULES
   - Time begins when rules stabilize (causal structure emerges)

6. MULTIVERSE STRUCTURE
   - Infinitely many possible universes (combinatorial)
   - Finite vocabulary of fundamental rules
   - Our universe is typical, not special
   - Child universes inherit structure from parents

7. THE ANTHROPIC PRINCIPLE (REINTERPRETED)
   - We observe THIS universe because it supports observers
   - But ANY stable universe would have "observers" in some form
   - Consciousness = self-referential rule patterns
   - We're not special; we're inevitable
    """)

    # Find the most "realistic" configuration (closest to our physics)
    print("\n--- Finding Our Universe ---")

    our_features = {
        'has_gravity': True,      # Many-to-one structure
        'has_em': True,           # Groupoid symmetry
        'has_weak': True,         # Partial symmetry
        'has_strong': True,       # Cyclic structure
        'has_conservation': True, # Conserved quantities
    }

    candidates = []

    for config in configs:
        score = 0

        # Check each feature
        in_degrees = Counter()
        for r in config.rules:
            if r[0] != r[1]:
                for t in r[1]:
                    in_degrees[t] += 1

        has_gravity = in_degrees and max(in_degrees.values()) >= 2
        has_em = config.is_groupoid
        has_weak = config.reversible_pairs > 0 and config.flow_rules > 0
        has_conservation = len(config.conserved_quantities) > 0

        # Check for cycles (strong force)
        has_strong = False
        for r in config.rules:
            if r[0] != r[1]:
                visited = {r[0]}
                current = r[1]
                for _ in range(10):
                    nexts = [r2[1] for r2 in config.rules if r2[0] == current and r2[0] != r2[1]]
                    if r[0] in nexts:
                        has_strong = True
                        break
                    if not nexts:
                        break
                    current = nexts[0]
                if has_strong:
                    break

        score = sum([has_gravity, has_em, has_weak, has_strong, has_conservation])
        if score >= 3:
            candidates.append((config, score, {
                'gravity': has_gravity,
                'em': has_em,
                'weak': has_weak,
                'strong': has_strong,
                'conservation': has_conservation
            }))

    candidates.sort(key=lambda x: (-x[1], -len(x[0].trial_ids)))

    print(f"\nFound {len(candidates)} candidate 'realistic' universes")

    for config, score, features in candidates[:3]:
        print(f"\nScore: {score}/5")
        print(f"  Trials: {len(config.trial_ids)}")
        print(f"  Rules: {config.n_rules}")
        print(f"  Features: {features}")
        print(f"  Rules: {list(config.rules)[:5]}...")


# ============================================================
# MAIN
# ============================================================

def main():
    n_tokens = 4  # Small for tractability, but enough for rich structure

    print("=" * 70)
    print("COSMOLOGICAL ANALYSIS OF THE SIEVE FRAMEWORK")
    print("=" * 70)

    # Part 1: Stable configurations
    configs, universes = analyze_stable_configurations(n_tokens=n_tokens, n_trials=500)

    # Part 2: Category theory
    functor_counts, functor_examples = analyze_functors(configs)

    # Part 3: Fundamental forces
    force_universes = map_to_fundamental_forces(configs)

    # Part 4: Pseudorandom effects
    prng_results = analyze_pseudorandom_effects(n_tokens=n_tokens, n_trials=150)

    # Part 5: Universe-spawning rules
    spawn_universes, spawn_trajectories, rule_entropy = find_universe_spawning_rules(
        n_tokens=n_tokens, n_trials=100
    )

    # Part 6: Recursive cosmology
    generations = simulate_recursive_cosmology(n_generations=4, n_tokens=n_tokens)

    # Part 7: Implications
    analyze_cosmological_implications(configs)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
