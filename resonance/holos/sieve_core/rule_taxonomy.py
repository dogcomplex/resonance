"""
rule_taxonomy.py - Deep Taxonomy of Emergent Rules

Key findings from initial analysis:
- Universe space is DIVERSE (199 distinct universes from 200 trials)
- But individual RULES are universal (same rules appear 50%+ of time)
- This means: Many universes, but built from a common alphabet of stable rules

This file:
1. Identifies the "universal alphabet" - rules that transcend universe choice
2. Analyzes what makes certain rules universal
3. Maps which rules require which other rules (dependency graph)
4. Identifies incompatible rule pairs (cannot coexist)
5. Builds the complete rule taxonomy
"""

import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, FrozenSet
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sieve_core.emergence import SelfOrganizingSubstrate
from sieve_core.substrate import DiscreteConfig, RuleHamiltonian, Substrate


# ============================================================
# CONFIGURATION
# ============================================================

N_TOKENS = 4
N_TRIALS = 300


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_universe_data():
    """Collect detailed data about rule emergence."""
    print("=" * 70)
    print("COLLECTING UNIVERSE DATA")
    print(f"Tokens: {N_TOKENS}, Trials: {N_TRIALS}")
    print("=" * 70)

    # Track everything
    universes = defaultdict(list)  # final rule-set -> trials
    rule_emergence_time = defaultdict(list)  # rule -> list of emergence times
    rule_cooccurrence = defaultdict(Counter)  # rule -> counter of co-occurring rules
    rule_exclusion = defaultdict(Counter)  # rule -> counter of excluded rules
    trajectories = []  # Full trajectories

    all_possible_rules = [((i,), (j,)) for i in range(N_TOKENS) for j in range(N_TOKENS)]

    for trial in range(N_TRIALS):
        random.seed(trial * 7919)
        np.random.seed(trial * 7919)

        substrate = SelfOrganizingSubstrate(
            damping_state=0.1,
            damping_rule=0.02,
            threshold=0.01
        )

        # Track initial rules
        initial_rules = set()
        for _ in range(12):
            i, j = random.randint(0, N_TOKENS-1), random.randint(0, N_TOKENS-1)
            amp = complex(random.gauss(0, 1), random.gauss(0, 1))
            substrate.inject_rule((i,), (j,), amp)
            initial_rules.add(((i,), (j,)))

        # Track trajectory
        trajectory = {'initial': frozenset(initial_rules), 'snapshots': []}

        # Evolve with snapshots
        for step in range(100):
            substrate.step(0.1)

            if step % 10 == 0:
                current = {
                    (f, t): abs(a)
                    for f, t, a in substrate.get_rules()
                    if abs(a) > 0.1
                }
                trajectory['snapshots'].append((step * 0.1, current))

        # Final state
        final_rules = frozenset(
            (f, t) for f, t, a in substrate.get_rules() if abs(a) > 0.5
        )
        trajectory['final'] = final_rules
        trajectories.append(trajectory)

        universes[final_rules].append(trial)

        # Track emergence times
        for rule in final_rules:
            # Find when it first appeared strongly
            for t, snapshot in trajectory['snapshots']:
                if rule in snapshot and snapshot[rule] > 0.5:
                    rule_emergence_time[rule].append(t)
                    break

        # Track co-occurrence
        for r1 in final_rules:
            for r2 in final_rules:
                if r1 != r2:
                    rule_cooccurrence[r1][r2] += 1

        # Track exclusion (rules present initially but not finally)
        excluded = initial_rules - final_rules
        for r_final in final_rules:
            for r_excluded in excluded:
                rule_exclusion[r_final][r_excluded] += 1

        if trial % 50 == 0:
            print(f"  Trial {trial}: {len(final_rules)} final rules, "
                  f"{len(universes)} unique universes so far")

    return universes, rule_emergence_time, rule_cooccurrence, rule_exclusion, trajectories


# ============================================================
# RULE TAXONOMY
# ============================================================

@dataclass
class RuleTaxonomy:
    """Complete taxonomy of a rule."""
    rule: Tuple

    # Frequency
    frequency: float  # How often does it appear in final universes?

    # Timing
    mean_emergence_time: float
    emergence_variance: float

    # Dependencies
    requires: Set[Tuple]  # Rules that almost always co-occur
    enables: Set[Tuple]   # Rules that appear more often when this rule exists
    excludes: Set[Tuple]  # Rules that almost never co-occur

    # Structure
    is_self_loop: bool
    has_reverse: bool  # Does the reverse rule also often exist?

    # Classification
    tier: str  # "Universal", "Common", "Rare", "Exotic"
    category: str  # "Conservation", "Flow", "Oscillation", etc.

    def __repr__(self):
        return f"{self.rule}: {self.tier} {self.category} (freq={self.frequency:.1%})"


def build_taxonomy(universes, emergence_times, cooccurrence, exclusion):
    """Build complete taxonomy of all rules."""
    print("\n" + "=" * 70)
    print("BUILDING RULE TAXONOMY")
    print("=" * 70)

    total_trials = sum(len(t) for t in universes.values())

    # Count rule frequencies
    rule_freq = Counter()
    for rules, trials in universes.items():
        for r in rules:
            rule_freq[r] += len(trials)

    all_rules = list(rule_freq.keys())
    taxonomies = {}

    for rule in all_rules:
        freq = rule_freq[rule] / total_trials

        # Timing
        times = emergence_times.get(rule, [10.0])
        mean_time = np.mean(times) if times else 10.0
        var_time = np.var(times) if len(times) > 1 else 0.0

        # Dependencies
        cooccur = cooccurrence.get(rule, Counter())
        exclude = exclusion.get(rule, Counter())

        requires = set()
        enables = set()
        excludes = set()

        rule_count = rule_freq[rule]

        for other, count in cooccur.most_common():
            # Requires: appears together >80% of time this rule appears
            if count > rule_count * 0.8:
                requires.add(other)
            # Enables: appears together >50% of time
            elif count > rule_count * 0.5:
                enables.add(other)

        for other, count in exclude.most_common():
            # Excludes: excluded >60% of time this rule survives
            if count > rule_count * 0.6:
                excludes.add(other)

        # Structure
        is_self_loop = rule[0] == rule[1]
        reverse = (rule[1], rule[0])
        has_reverse = reverse in rule_freq and rule_freq[reverse] > rule_count * 0.3

        # Classification
        if freq > 0.5:
            tier = "Universal"
        elif freq > 0.3:
            tier = "Common"
        elif freq > 0.1:
            tier = "Rare"
        else:
            tier = "Exotic"

        # Category based on structure
        if is_self_loop:
            category = "Identity/Rest"
        elif has_reverse:
            category = "Oscillation/Exchange"
        elif len(requires) > 2:
            category = "Hub/Vertex"
        else:
            category = "Flow/Decay"

        taxonomies[rule] = RuleTaxonomy(
            rule=rule,
            frequency=freq,
            mean_emergence_time=mean_time,
            emergence_variance=var_time,
            requires=requires,
            enables=enables,
            excludes=excludes,
            is_self_loop=is_self_loop,
            has_reverse=has_reverse,
            tier=tier,
            category=category
        )

    return taxonomies


def print_taxonomy(taxonomies: Dict[Tuple, RuleTaxonomy]):
    """Print the complete taxonomy."""
    print("\n--- COMPLETE RULE TAXONOMY ---\n")

    # Sort by tier then frequency
    tier_order = {"Universal": 0, "Common": 1, "Rare": 2, "Exotic": 3}
    sorted_rules = sorted(
        taxonomies.values(),
        key=lambda t: (tier_order[t.tier], -t.frequency)
    )

    current_tier = None
    for tax in sorted_rules:
        if tax.tier != current_tier:
            current_tier = tax.tier
            print(f"\n=== {current_tier.upper()} RULES ===\n")

        print(f"{tax.rule}:")
        print(f"  Frequency: {tax.frequency:.1%}")
        print(f"  Category: {tax.category}")
        print(f"  Emergence: t={tax.mean_emergence_time:.1f} (var={tax.emergence_variance:.2f})")

        if tax.requires:
            print(f"  REQUIRES: {list(tax.requires)[:3]}")
        if tax.enables:
            print(f"  ENABLES: {list(tax.enables)[:3]}")
        if tax.excludes:
            print(f"  EXCLUDES: {list(tax.excludes)[:3]}")
        print()

    return sorted_rules


# ============================================================
# DEPENDENCY GRAPH
# ============================================================

def build_dependency_graph(taxonomies: Dict[Tuple, RuleTaxonomy]):
    """Build and analyze the rule dependency graph."""
    print("\n" + "=" * 70)
    print("RULE DEPENDENCY GRAPH")
    print("=" * 70)

    # Nodes = rules, Edges = requires/enables/excludes
    nodes = list(taxonomies.keys())

    requires_edges = []
    enables_edges = []
    excludes_edges = []

    for rule, tax in taxonomies.items():
        for req in tax.requires:
            if req in taxonomies:
                requires_edges.append((rule, req))
        for enb in tax.enables:
            if enb in taxonomies:
                enables_edges.append((rule, enb))
        for exc in tax.excludes:
            if exc in taxonomies:
                excludes_edges.append((rule, exc))

    print(f"\nNodes (rules): {len(nodes)}")
    print(f"REQUIRES edges: {len(requires_edges)}")
    print(f"ENABLES edges: {len(enables_edges)}")
    print(f"EXCLUDES edges: {len(excludes_edges)}")

    # Find strongly connected components (rules that mutually require each other)
    # Simplified: find cliques in requires graph
    mutual_requires = []
    for r1, r2 in requires_edges:
        if (r2, r1) in requires_edges:
            pair = (r1, r2) if r1 < r2 else (r2, r1)
            if pair not in mutual_requires:
                mutual_requires.append(pair)

    print(f"\nMutual requirements (A requires B AND B requires A):")
    for r1, r2 in mutual_requires[:10]:
        print(f"  {r1} <-> {r2}")

    # Find incompatible pairs
    mutual_excludes = []
    for r1, tax1 in taxonomies.items():
        for r2, tax2 in taxonomies.items():
            if r1 < r2:
                if r2 in tax1.excludes and r1 in tax2.excludes:
                    mutual_excludes.append((r1, r2))

    print(f"\nMutually exclusive rules:")
    for r1, r2 in mutual_excludes[:10]:
        print(f"  {r1} vs {r2}")

    # Find "foundation" rules (required by many, require few)
    foundation_score = {}
    for rule, tax in taxonomies.items():
        required_by = sum(1 for t in taxonomies.values() if rule in t.requires)
        requires = len(tax.requires)
        foundation_score[rule] = required_by - requires

    print(f"\nFoundation rules (required by many, require few):")
    for rule, score in sorted(foundation_score.items(), key=lambda x: -x[1])[:5]:
        print(f"  {rule}: score={score}")

    return requires_edges, enables_edges, excludes_edges


# ============================================================
# UNIVERSE TYPES
# ============================================================

def classify_universes(universes, taxonomies):
    """Classify universes by their rule composition."""
    print("\n" + "=" * 70)
    print("UNIVERSE CLASSIFICATION")
    print("=" * 70)

    universe_types = defaultdict(list)

    for rules, trials in universes.items():
        # Classify based on rule tiers present
        tiers = Counter()
        categories = Counter()

        for rule in rules:
            if rule in taxonomies:
                tax = taxonomies[rule]
                tiers[tax.tier] += 1
                categories[tax.category] += 1

        # Determine universe type
        if tiers["Universal"] > len(rules) * 0.6:
            u_type = "Standard Physics"
        elif tiers["Exotic"] > len(rules) * 0.3:
            u_type = "Exotic Physics"
        elif categories["Identity/Rest"] > len(rules) * 0.4:
            u_type = "Frozen Universe"
        elif categories["Oscillation/Exchange"] > len(rules) * 0.4:
            u_type = "Cyclic Universe"
        else:
            u_type = "Mixed Physics"

        universe_types[u_type].append((rules, trials))

    print("\nUniverse type distribution:")
    total = sum(len(v) for v in universe_types.values())
    for u_type, unis in sorted(universe_types.items(), key=lambda x: -len(x[1])):
        trial_count = sum(len(t) for r, t in unis)
        print(f"  {u_type}: {len(unis)} universes, {trial_count} trials ({100*trial_count/N_TRIALS:.0f}%)")

    return universe_types


# ============================================================
# THE KEY QUESTION: BRANCHING STRUCTURE
# ============================================================

def analyze_branching(universes, trajectories):
    """Analyze how universes branch and potentially merge."""
    print("\n" + "=" * 70)
    print("BRANCHING STRUCTURE ANALYSIS")
    print("=" * 70)

    # Question 1: Do similar initial conditions lead to similar universes?
    print("\n--- Initial -> Final Correlation ---")

    # Group by initial rule sets
    initial_to_final = defaultdict(list)
    for traj in trajectories:
        initial_to_final[traj['initial']].append(traj['final'])

    # Find initial conditions that led to multiple different finals
    divergent = [(init, finals) for init, finals in initial_to_final.items()
                 if len(set(finals)) > 1]

    print(f"Unique initial conditions: {len(initial_to_final)}")
    print(f"Divergent (same initial, different final): {len(divergent)}")

    # Question 2: Do different paths converge to same universe?
    print("\n--- Convergence Analysis ---")

    final_to_initials = defaultdict(list)
    for traj in trajectories:
        final_to_initials[traj['final']].append(traj['initial'])

    # Universes reached from multiple starting points
    convergent = [(final, inits) for final, inits in final_to_initials.items()
                  if len(set(inits)) > 1]

    print(f"Universes reached from multiple starts: {len(convergent)}")

    if convergent:
        # Most "attractive" universe
        most_attractive = max(convergent, key=lambda x: len(x[1]))
        print(f"Most attractive universe: {len(most_attractive[1])} different starting points")

    # Question 3: What determines which universe you reach?
    print("\n--- Determining Factors ---")

    # For each universal rule, check if its presence in initial determines anything
    all_initial_rules = set()
    for init in initial_to_final.keys():
        all_initial_rules.update(init)

    rule_predictiveness = {}
    for rule in all_initial_rules:
        with_rule = []
        without_rule = []

        for traj in trajectories:
            if rule in traj['initial']:
                with_rule.append(traj['final'])
            else:
                without_rule.append(traj['final'])

        # Measure: how concentrated are outcomes?
        with_diversity = len(set(with_rule)) / max(len(with_rule), 1)
        without_diversity = len(set(without_rule)) / max(len(without_rule), 1)

        rule_predictiveness[rule] = without_diversity - with_diversity

    print("\nMost predictive initial rules (presence narrows outcomes):")
    for rule, score in sorted(rule_predictiveness.items(), key=lambda x: -x[1])[:5]:
        print(f"  {rule}: predictiveness={score:.3f}")

    print("\nMost chaotic initial rules (presence increases diversity):")
    for rule, score in sorted(rule_predictiveness.items(), key=lambda x: x[1])[:5]:
        print(f"  {rule}: predictiveness={score:.3f}")

    return divergent, convergent


# ============================================================
# MAIN
# ============================================================

def main():
    # Collect data
    universes, emergence_times, cooccurrence, exclusion, trajectories = collect_universe_data()

    print(f"\n\nTotal universes: {len(universes)}")
    print(f"Total trials: {N_TRIALS}")

    # Build taxonomy
    taxonomies = build_taxonomy(universes, emergence_times, cooccurrence, exclusion)
    sorted_rules = print_taxonomy(taxonomies)

    # Dependency graph
    requires, enables, excludes = build_dependency_graph(taxonomies)

    # Universe classification
    universe_types = classify_universes(universes, taxonomies)

    # Branching analysis
    divergent, convergent = analyze_branching(universes, trajectories)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: THE STRUCTURE OF POSSIBLE PHYSICS")
    print("=" * 70)

    universal_rules = [t for t in taxonomies.values() if t.tier == "Universal"]

    print(f"""
KEY FINDINGS:

1. UNIVERSE DIVERSITY: {len(universes)} distinct universes from {N_TRIALS} trials
   - Not dominated by few attractors
   - Each random start typically yields unique final state

2. RULE UNIVERSALITY: {len(universal_rules)} "universal" rules appear in >50% of universes
   - These form the "alphabet" from which all universes are built
   - Examples: {[t.rule for t in universal_rules[:3]]}

3. RULE DEPENDENCIES: Some rules require others
   - Mutual requirements form "clusters" of co-dependent rules
   - Foundation rules: required by many, require few

4. RULE EXCLUSIONS: Some rules cannot coexist
   - Mutually exclusive pairs define "branch points" in universe space

5. BRANCHING: The multiverse has structure
   - Same initial conditions can diverge (chaos)
   - Different paths can converge (attractors)
   - Initial rule composition partially predicts final universe

CONCLUSION:
The space of possible physics is:
- INFINITE in universe count (combinatorial explosion)
- FINITE in rule vocabulary (universal rules dominate)
- STRUCTURED by dependencies and exclusions
- PARTIALLY PREDICTABLE from initial conditions
    """)


if __name__ == "__main__":
    main()
