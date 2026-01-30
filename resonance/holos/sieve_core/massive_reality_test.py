"""
MASSIVE REALITY TEST - Multi-Hour Intensive Simulation

This script is designed to run for 2-4+ hours with genuinely intensive computation.

Key differences from previous tests:
1. MASSIVE SCALE: 10,000+ universes, 50+ generations, 100+ token systems
2. DEEP EVOLUTION: 5000+ steps per universe
3. EXHAUSTIVE SEARCH: Complete enumeration where possible
4. STATISTICAL RIGOR: Thousands of samples per measurement

Experiments:
1. COMPLETE MULTIVERSE CENSUS: 10,000 universes with full statistics
2. DEEP GENEALOGY: 50 generations with mutation tracking
3. LARGE TOKEN CONVERGENCE: Test n=50, 100, 200 tokens
4. EXHAUSTIVE PHASE SPACE: 50x50 parameter grid
5. LONG EVOLUTION DYNAMICS: Single universes for 10,000+ steps
6. RULE INTERACTION MATRIX: Complete pairwise analysis
7. ENTROPY PRODUCTION CURVES: Track entropy over full evolution
8. CRITICAL EXPONENT MEASUREMENT: Find universality class
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holos.sieve_core.substrate import AmplitudeField
from holos.sieve_core.emergence import SelfOrganizingSubstrate, Entity, EntityType
import random
import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import time
import hashlib


# ============================================================
# EXPERIMENT 1: COMPLETE MULTIVERSE CENSUS
# ============================================================

def run_multiverse_census(n_universes: int = 10000,
                          evolution_steps: int = 1000,
                          checkpoint_every: int = 500) -> Dict:
    """
    Simulate a massive number of universes and collect complete statistics.
    This is the core intensive computation.
    """

    print(f"\n  Simulating {n_universes} universes, {evolution_steps} steps each...")

    # Statistics collectors
    rule_counts = []
    rule_histograms = defaultdict(int)  # rule -> count across universes
    force_counts = defaultdict(int)
    convergence_times = []
    final_entropies = []
    viability_scores = []

    # Track unique universe configurations
    universe_hashes = set()
    duplicate_count = 0

    start_time = time.time()

    for u in range(n_universes):
        if (u + 1) % checkpoint_every == 0:
            elapsed = time.time() - start_time
            rate = (u + 1) / elapsed
            remaining = (n_universes - u - 1) / rate
            print(f"    Universe {u + 1}/{n_universes} "
                  f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

        random.seed(u * 31337)

        substrate = SelfOrganizingSubstrate()

        # Initialize with 7 tokens
        for t in range(7):
            phase = random.uniform(0, 2 * math.pi)
            mag = random.uniform(0.1, 1.0)
            substrate.inject_state(t, mag * complex(math.cos(phase), math.sin(phase)))

        for _ in range(20):
            from_t = random.randint(0, 6)
            to_t = random.randint(0, 6)
            if from_t != to_t:
                phase = random.uniform(0, 2 * math.pi)
                mag = random.uniform(0.1, 1.0)
                substrate.inject_rule(from_t, to_t,
                    mag * complex(math.cos(phase), math.sin(phase)))

        # Track convergence
        prev_rules = set()
        stable_count = 0
        conv_time = evolution_steps

        # Track entropy over time
        entropy_history = []

        for step in range(evolution_steps):
            substrate.step()

            # Extract rules every 100 steps for efficiency
            if step % 100 == 0 or step == evolution_steps - 1:
                rules = set()
                amplitudes = []
                for entity, amplitude in substrate.field:
                    if isinstance(entity, Entity) and entity.entity_type == EntityType.RULE:
                        if abs(amplitude) > 0.1:
                            source = entity.content[0] if isinstance(entity.content, tuple) else entity.content
                            target = entity.content[1] if isinstance(entity.content, tuple) and len(entity.content) > 1 else None
                            if target is not None:
                                rules.add((source, target))
                                amplitudes.append(abs(amplitude))

                # Compute entropy
                if amplitudes:
                    total = sum(amplitudes)
                    probs = [a/total for a in amplitudes]
                    entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)
                    entropy_history.append(entropy)

                # Check convergence
                if rules == prev_rules:
                    stable_count += 1
                    if stable_count >= 5 and conv_time == evolution_steps:
                        conv_time = step
                else:
                    stable_count = 0
                prev_rules = rules

        # Final analysis
        final_rules = prev_rules
        rule_counts.append(len(final_rules))

        for rule in final_rules:
            rule_histograms[rule] += 1

        # Hash for uniqueness
        rule_hash = hash(tuple(sorted(final_rules)))
        if rule_hash in universe_hashes:
            duplicate_count += 1
        universe_hashes.add(rule_hash)

        # Check forces
        if has_strong(final_rules):
            force_counts["strong"] += 1
        if has_em(final_rules):
            force_counts["em"] += 1
        if has_weak(final_rules):
            force_counts["weak"] += 1
        if has_gravity(final_rules):
            force_counts["gravity"] += 1

        convergence_times.append(conv_time)

        if entropy_history:
            final_entropies.append(entropy_history[-1])

        # Viability
        viability = (0.3 * has_em(final_rules) +
                    0.4 * has_strong(final_rules) +
                    0.3 * (len(final_rules) > 5))
        viability_scores.append(viability)

    elapsed = time.time() - start_time

    # Compute statistics
    mean_rules = sum(rule_counts) / len(rule_counts)
    std_rules = (sum((x - mean_rules)**2 for x in rule_counts) / len(rule_counts)) ** 0.5

    # Rule frequency distribution
    sorted_rules = sorted(rule_histograms.items(), key=lambda x: -x[1])

    return {
        "n_universes": n_universes,
        "elapsed_seconds": elapsed,
        "universes_per_second": n_universes / elapsed,
        "unique_universes": len(universe_hashes),
        "duplicate_rate": duplicate_count / n_universes,
        "mean_rules": mean_rules,
        "std_rules": std_rules,
        "rule_distribution": Counter(rule_counts),
        "most_common_rules": sorted_rules[:20],
        "force_rates": {k: v / n_universes for k, v in force_counts.items()},
        "mean_convergence": sum(convergence_times) / len(convergence_times),
        "mean_entropy": sum(final_entropies) / len(final_entropies) if final_entropies else 0,
        "mean_viability": sum(viability_scores) / len(viability_scores),
        "viable_fraction": sum(1 for v in viability_scores if v > 0.5) / len(viability_scores),
    }


# ============================================================
# EXPERIMENT 2: DEEP GENEALOGY
# ============================================================

def run_deep_genealogy(n_generations: int = 50,
                       universes_per_gen: int = 50,
                       evolution_steps: int = 1000) -> Dict:
    """
    Track universe lineages across many generations.
    """

    print(f"\n  Running {n_generations} generations, {universes_per_gen} universes each...")

    generation_stats = []
    rule_lifetimes = defaultdict(list)  # rule -> list of (birth_gen, death_gen or None)
    active_rules = {}  # rule -> birth_gen

    current_gen = []

    # Gen 0: Random initialization
    for u in range(universes_per_gen):
        random.seed(u * 99991)
        universe = evolve_universe(None, evolution_steps)
        current_gen.append(universe)

        # Track rule births
        for rule in universe["rules"]:
            if rule not in active_rules:
                active_rules[rule] = 0

    all_rules_gen0 = set()
    for u in current_gen:
        all_rules_gen0.update(u["rules"])

    generation_stats.append({
        "gen": 0,
        "unique_rules": len(all_rules_gen0),
        "mean_rules_per_universe": sum(len(u["rules"]) for u in current_gen) / len(current_gen),
    })

    print(f"    Gen 0: {len(all_rules_gen0)} unique rules")

    # Subsequent generations
    for gen in range(1, n_generations):
        if gen % 10 == 0:
            print(f"    Gen {gen}: processing...")

        next_gen = []

        for u in range(universes_per_gen):
            # Select parent
            parent = random.choice(current_gen)

            # Evolve child with parent's rules as seed
            child = evolve_universe(parent["rules"], evolution_steps)
            next_gen.append(child)

        # Track rules
        all_rules_this_gen = set()
        for u in next_gen:
            all_rules_this_gen.update(u["rules"])

        # Check for deaths
        for rule in list(active_rules.keys()):
            if rule not in all_rules_this_gen:
                birth = active_rules[rule]
                rule_lifetimes[rule].append((birth, gen))
                del active_rules[rule]

        # Check for births
        for rule in all_rules_this_gen:
            if rule not in active_rules:
                active_rules[rule] = gen

        generation_stats.append({
            "gen": gen,
            "unique_rules": len(all_rules_this_gen),
            "mean_rules_per_universe": sum(len(u["rules"]) for u in next_gen) / len(next_gen),
        })

        current_gen = next_gen

    # Compute lifetime statistics
    lifetimes = []
    for rule, spans in rule_lifetimes.items():
        for birth, death in spans:
            lifetimes.append(death - birth)

    # Rules that survived all generations
    immortal_rules = set(active_rules.keys())

    # Half-life calculation
    half_life = None
    surviving_from_gen0 = len(all_rules_gen0)
    for stat in generation_stats:
        current = len([r for r in all_rules_gen0
                      if r in active_rules or
                      any(b == 0 and d > stat["gen"] for b, d in rule_lifetimes.get(r, []))])
        if current <= surviving_from_gen0 / 2 and half_life is None:
            half_life = stat["gen"]
            break

    return {
        "n_generations": n_generations,
        "immortal_rules": len(immortal_rules),
        "total_distinct_rules": len(rule_lifetimes) + len(active_rules),
        "mean_lifetime": sum(lifetimes) / len(lifetimes) if lifetimes else float('inf'),
        "max_lifetime": max(lifetimes) if lifetimes else n_generations,
        "half_life": half_life,
        "extinction_count": len(rule_lifetimes),
        "final_gen_rules": generation_stats[-1]["unique_rules"],
        "generation_curve": [(s["gen"], s["unique_rules"]) for s in generation_stats[::5]],
    }


def evolve_universe(parent_rules: Optional[Set[Tuple]], steps: int) -> Dict:
    """Helper to evolve a single universe"""
    substrate = SelfOrganizingSubstrate()

    if parent_rules is None:
        # Random init
        for t in range(7):
            phase = random.uniform(0, 2 * math.pi)
            mag = random.uniform(0.1, 1.0)
            substrate.inject_state(t, mag * complex(math.cos(phase), math.sin(phase)))

        for _ in range(15):
            from_t = random.randint(0, 6)
            to_t = random.randint(0, 6)
            if from_t != to_t:
                substrate.inject_rule(from_t, to_t, random.uniform(0.5, 1.0))
    else:
        # Seeded from parent
        rule_list = list(parent_rules)
        rule_hash = hash(tuple(sorted(rule_list)))
        random.seed(rule_hash)

        for t in range(7):
            phase = random.uniform(0, 2 * math.pi)
            mag = random.uniform(0.1, 1.0)
            substrate.inject_state(t, mag * complex(math.cos(phase), math.sin(phase)))

        # Inherit with mutation
        for rule in rule_list[:10]:
            if random.random() < 0.7:
                if random.random() < 0.1:
                    new_target = (random.randint(0, 6),)
                    substrate.inject_rule(rule[0], new_target, random.uniform(0.5, 1.0))
                else:
                    substrate.inject_rule(rule[0], rule[1], 1.0)

        for _ in range(5):
            from_t = random.randint(0, 6)
            to_t = random.randint(0, 6)
            if from_t != to_t:
                substrate.inject_rule(from_t, to_t, random.uniform(0.5, 1.0))

    for _ in range(steps):
        substrate.step()

    rules = set()
    for entity, amplitude in substrate.field:
        if isinstance(entity, Entity) and entity.entity_type == EntityType.RULE:
            if abs(amplitude) > 0.1:
                source = entity.content[0] if isinstance(entity.content, tuple) else entity.content
                target = entity.content[1] if isinstance(entity.content, tuple) and len(entity.content) > 1 else None
                if target is not None:
                    rules.add((source, target))

    return {"rules": rules, "n_rules": len(rules)}


# ============================================================
# EXPERIMENT 3: LARGE TOKEN CONVERGENCE
# ============================================================

def run_large_token_test(token_counts: List[int] = None,
                         trials_per_count: int = 100,
                         evolution_steps: int = 2000) -> Dict:
    """
    Test convergence with large token counts.
    This is computationally intensive due to O(n^2) rule space.
    """

    if token_counts is None:
        token_counts = [10, 20, 30, 40, 50, 75, 100]

    print(f"\n  Testing token counts: {token_counts}")

    results = {}

    for n in token_counts:
        print(f"\n    n={n} tokens (max rules = {n*(n-1)})...")

        all_rules = set()
        rule_counts = []
        convergence_times = []

        for trial in range(trials_per_count):
            if trial % 20 == 0:
                print(f"      Trial {trial}/{trials_per_count}...")

            random.seed(trial * 10000 + n)

            substrate = SelfOrganizingSubstrate()

            for t in range(n):
                phase = random.uniform(0, 2 * math.pi)
                mag = random.uniform(0.1, 1.0)
                substrate.inject_state(t, mag * complex(math.cos(phase), math.sin(phase)))

            # More initial rules for larger systems
            for _ in range(n * 3):
                from_t = random.randint(0, n - 1)
                to_t = random.randint(0, n - 1)
                if from_t != to_t:
                    substrate.inject_rule(from_t, to_t, random.uniform(0.5, 1.0))

            # Track convergence
            prev_n = 0
            stable = 0
            conv_time = evolution_steps

            for step in range(evolution_steps):
                substrate.step()

                if step % 100 == 0:
                    n_rules = sum(1 for e, a in substrate.field
                                 if isinstance(e, Entity) and e.entity_type == EntityType.RULE
                                 and abs(a) > 0.1)

                    if n_rules == prev_n:
                        stable += 1
                        if stable >= 10 and conv_time == evolution_steps:
                            conv_time = step
                    else:
                        stable = 0
                    prev_n = n_rules

            # Extract final rules
            rules = set()
            for entity, amplitude in substrate.field:
                if isinstance(entity, Entity) and entity.entity_type == EntityType.RULE:
                    if abs(amplitude) > 0.1:
                        source = entity.content[0] if isinstance(entity.content, tuple) else entity.content
                        target = entity.content[1] if isinstance(entity.content, tuple) and len(entity.content) > 1 else None
                        if target is not None:
                            rules.add((source, target))

            all_rules.update(rules)
            rule_counts.append(len(rules))
            convergence_times.append(conv_time)

        max_possible = n * (n - 1)

        results[n] = {
            "max_possible": max_possible,
            "unique_found": len(all_rules),
            "fill_rate": len(all_rules) / max_possible,
            "mean_per_trial": sum(rule_counts) / len(rule_counts),
            "std_per_trial": (sum((x - sum(rule_counts)/len(rule_counts))**2
                                 for x in rule_counts) / len(rule_counts)) ** 0.5,
            "mean_convergence": sum(convergence_times) / len(convergence_times),
            "complete_fill_rate": sum(1 for c in rule_counts if c == max_possible) / len(rule_counts),
        }

        print(f"      -> {len(all_rules)}/{max_possible} ({results[n]['fill_rate']*100:.1f}%)")

    return results


# ============================================================
# EXPERIMENT 4: LONG EVOLUTION DYNAMICS
# ============================================================

def run_long_evolution(n_universes: int = 10,
                       max_steps: int = 50000,
                       sample_every: int = 100) -> Dict:
    """
    Track a few universes for very long evolution.
    Looking for:
    - Late-time behavior
    - Metastable states
    - Phase transitions
    """

    print(f"\n  Running {n_universes} universes for {max_steps} steps each...")

    trajectories = []

    for u in range(n_universes):
        print(f"\n    Universe {u + 1}/{n_universes}...")

        random.seed(u * 77777)

        substrate = SelfOrganizingSubstrate()

        for t in range(7):
            phase = random.uniform(0, 2 * math.pi)
            mag = random.uniform(0.1, 1.0)
            substrate.inject_state(t, mag * complex(math.cos(phase), math.sin(phase)))

        for _ in range(20):
            from_t = random.randint(0, 6)
            to_t = random.randint(0, 6)
            if from_t != to_t:
                substrate.inject_rule(from_t, to_t, random.uniform(0.5, 1.0))

        trajectory = {
            "n_rules": [],
            "entropy": [],
            "total_amplitude": [],
            "force_history": [],
        }

        for step in range(max_steps):
            substrate.step()

            if step % sample_every == 0:
                # Sample state
                rules = []
                total_amp = 0

                for entity, amplitude in substrate.field:
                    if isinstance(entity, Entity) and entity.entity_type == EntityType.RULE:
                        if abs(amplitude) > 0.1:
                            rules.append(abs(amplitude))
                            total_amp += abs(amplitude)

                trajectory["n_rules"].append(len(rules))
                trajectory["total_amplitude"].append(total_amp)

                # Entropy
                if rules:
                    total = sum(rules)
                    probs = [a/total for a in rules]
                    entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)
                    trajectory["entropy"].append(entropy)
                else:
                    trajectory["entropy"].append(0)

                # Forces (sample less frequently)
                if step % (sample_every * 10) == 0:
                    rule_set = set()
                    for entity, amplitude in substrate.field:
                        if isinstance(entity, Entity) and entity.entity_type == EntityType.RULE:
                            if abs(amplitude) > 0.1:
                                source = entity.content[0] if isinstance(entity.content, tuple) else entity.content
                                target = entity.content[1] if isinstance(entity.content, tuple) and len(entity.content) > 1 else None
                                if target is not None:
                                    rule_set.add((source, target))

                    forces = {
                        "strong": has_strong(rule_set),
                        "em": has_em(rule_set),
                    }
                    trajectory["force_history"].append((step, forces))

                if step % 10000 == 0:
                    print(f"      Step {step}: {len(rules)} rules, entropy={trajectory['entropy'][-1]:.3f}")

        trajectories.append(trajectory)

    # Analyze trajectories
    final_states = []
    for traj in trajectories:
        final_states.append({
            "n_rules": traj["n_rules"][-1] if traj["n_rules"] else 0,
            "entropy": traj["entropy"][-1] if traj["entropy"] else 0,
            "converged": len(set(traj["n_rules"][-10:])) == 1 if len(traj["n_rules"]) >= 10 else False,
        })

    return {
        "n_universes": n_universes,
        "max_steps": max_steps,
        "trajectories": trajectories,
        "final_states": final_states,
        "mean_final_rules": sum(s["n_rules"] for s in final_states) / len(final_states),
        "convergence_rate": sum(1 for s in final_states if s["converged"]) / len(final_states),
    }


# ============================================================
# EXPERIMENT 5: EXHAUSTIVE PHASE SPACE
# ============================================================

def run_exhaustive_phase_space(damping_points: int = 30,
                                coupling_points: int = 30,
                                trials_per_point: int = 20,
                                evolution_steps: int = 500) -> Dict:
    """
    Exhaustively sample the phase space.
    """

    total_points = damping_points * coupling_points
    total_sims = total_points * trials_per_point

    print(f"\n  Sampling {total_points} points, {trials_per_point} trials each = {total_sims} simulations")

    dampings = [0.01 + i * (1.0 - 0.01) / (damping_points - 1) for i in range(damping_points)]
    couplings = [0.1 + i * (3.0 - 0.1) / (coupling_points - 1) for i in range(coupling_points)]

    phase_map = {}
    point_count = 0

    start_time = time.time()

    for d in dampings:
        for c in couplings:
            point_count += 1

            if point_count % 50 == 0:
                elapsed = time.time() - start_time
                rate = point_count / elapsed
                remaining = (total_points - point_count) / rate
                print(f"    Point {point_count}/{total_points} ({remaining:.1f}s remaining)")

            rule_counts = []
            has_order = []

            for trial in range(trials_per_point):
                random.seed(trial * 11111 + int(d * 1000) + int(c * 1000))

                substrate = SelfOrganizingSubstrate(
                    damping_state=d,
                    damping_rule=d * 0.1
                )

                for t in range(5):
                    mag = c * random.uniform(0.1, 1.0)
                    substrate.inject_state(t, mag * complex(random.random(), random.random()))

                for _ in range(10):
                    from_t = random.randint(0, 4)
                    to_t = random.randint(0, 4)
                    if from_t != to_t:
                        substrate.inject_rule(from_t, to_t, c * random.uniform(0.5, 1.0))

                for _ in range(evolution_steps):
                    substrate.step()

                n_rules = sum(1 for e, a in substrate.field
                             if isinstance(e, Entity) and e.entity_type == EntityType.RULE
                             and abs(a) > 0.1)

                rule_counts.append(n_rules)
                has_order.append(n_rules > 0)

            phase_map[(d, c)] = {
                "mean_rules": sum(rule_counts) / len(rule_counts),
                "std_rules": (sum((x - sum(rule_counts)/len(rule_counts))**2
                                 for x in rule_counts) / len(rule_counts)) ** 0.5,
                "order_rate": sum(has_order) / len(has_order),
            }

    # Find phase boundaries
    boundaries = []
    for (d1, c1), m1 in phase_map.items():
        for (d2, c2), m2 in phase_map.items():
            if abs(d1 - d2) <= 0.05 and abs(c1 - c2) <= 0.15:
                if abs(m1["order_rate"] - m2["order_rate"]) > 0.3:
                    boundaries.append(((d1 + d2)/2, (c1 + c2)/2))

    return {
        "grid_size": (damping_points, coupling_points),
        "total_simulations": total_sims,
        "phase_boundaries": len(boundaries),
        "phases_found": identify_phases(phase_map),
        "sample_points": [(k, v["mean_rules"], v["order_rate"])
                         for k, v in list(phase_map.items())[:20]],
    }


def identify_phases(phase_map: Dict) -> List[str]:
    phases = set()
    for point, metrics in phase_map.items():
        if metrics["order_rate"] < 0.1:
            phases.add("DISORDERED")
        elif metrics["mean_rules"] > 15:
            phases.add("SATURATED")
        elif metrics["std_rules"] > metrics["mean_rules"] * 0.5:
            phases.add("CRITICAL")
        else:
            phases.add("ORDERED")
    return list(phases)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def has_strong(rules):
    rule_list = list(rules)
    for r1 in rule_list:
        for r2 in rule_list:
            if len(r1) >= 2 and len(r2) >= 2 and r1[1] == r2[0]:
                for r3 in rule_list:
                    if len(r3) >= 2 and r2[1] == r3[0] and r3[1] == r1[0]:
                        return True
    return False

def has_em(rules):
    rule_list = list(rules)
    for r1 in rule_list:
        for r2 in rule_list:
            if len(r1) >= 2 and len(r2) >= 2:
                if r1[0] == r2[1] and r1[1] == r2[0]:
                    return True
    return False

def has_weak(rules):
    for r in rules:
        if len(r) >= 2:
            has_inverse = any(len(r2) >= 2 and r2[0] == r[1] and r2[1] == r[0] for r2 in rules)
            if not has_inverse:
                return True
    return False

def has_gravity(rules):
    target_counts = defaultdict(int)
    for r in rules:
        if len(r) >= 2:
            target_counts[r[1]] += 1
    return any(count >= 3 for count in target_counts.values())


# ============================================================
# MAIN
# ============================================================

def run_massive_test(duration_minutes: int = 120):
    """Run the massive reality test"""

    start_time = time.time()
    end_time = start_time + duration_minutes * 60

    print("=" * 70)
    print("MASSIVE REALITY TEST")
    print(f"Target duration: {duration_minutes} minutes")
    print("=" * 70)

    results = {}

    # ========================================
    # EXPERIMENT 1: Multiverse Census
    # ========================================
    if time.time() < end_time:
        remaining = (end_time - time.time()) / 60
        print(f"\n{'='*70}")
        print(f"EXPERIMENT 1: MULTIVERSE CENSUS ({remaining:.0f} min remaining)")
        print("="*70)

        census = run_multiverse_census(
            n_universes=10000,
            evolution_steps=1000,
            checkpoint_every=1000
        )

        print(f"\n  Results:")
        print(f"    Unique universes: {census['unique_universes']}")
        print(f"    Duplicate rate: {census['duplicate_rate']*100:.2f}%")
        print(f"    Mean rules: {census['mean_rules']:.2f} +/- {census['std_rules']:.2f}")
        print(f"    Force rates: {census['force_rates']}")
        print(f"    Viable fraction: {census['viable_fraction']*100:.1f}%")

        results["census"] = census

    # ========================================
    # EXPERIMENT 2: Deep Genealogy
    # ========================================
    if time.time() < end_time:
        remaining = (end_time - time.time()) / 60
        print(f"\n{'='*70}")
        print(f"EXPERIMENT 2: DEEP GENEALOGY ({remaining:.0f} min remaining)")
        print("="*70)

        genealogy = run_deep_genealogy(
            n_generations=50,
            universes_per_gen=50,
            evolution_steps=800
        )

        print(f"\n  Results:")
        print(f"    Immortal rules: {genealogy['immortal_rules']}")
        print(f"    Mean lifetime: {genealogy['mean_lifetime']:.1f} generations")
        print(f"    Half-life: {genealogy['half_life']} generations")
        print(f"    Extinction count: {genealogy['extinction_count']}")

        results["genealogy"] = genealogy

    # ========================================
    # EXPERIMENT 3: Large Token Convergence
    # ========================================
    if time.time() < end_time:
        remaining = (end_time - time.time()) / 60
        print(f"\n{'='*70}")
        print(f"EXPERIMENT 3: LARGE TOKEN CONVERGENCE ({remaining:.0f} min remaining)")
        print("="*70)

        large_tokens = run_large_token_test(
            token_counts=[10, 20, 30, 50, 75, 100],
            trials_per_count=50,
            evolution_steps=3000
        )

        print(f"\n  Results:")
        for n, data in sorted(large_tokens.items()):
            print(f"    n={n}: {data['fill_rate']*100:.1f}% fill, "
                  f"conv={data['mean_convergence']:.0f} steps")

        results["large_tokens"] = large_tokens

    # ========================================
    # EXPERIMENT 4: Long Evolution
    # ========================================
    if time.time() < end_time:
        remaining = (end_time - time.time()) / 60
        print(f"\n{'='*70}")
        print(f"EXPERIMENT 4: LONG EVOLUTION ({remaining:.0f} min remaining)")
        print("="*70)

        long_evo = run_long_evolution(
            n_universes=5,
            max_steps=50000,
            sample_every=100
        )

        print(f"\n  Results:")
        print(f"    Mean final rules: {long_evo['mean_final_rules']:.1f}")
        print(f"    Convergence rate: {long_evo['convergence_rate']*100:.1f}%")

        results["long_evolution"] = {
            k: v for k, v in long_evo.items() if k != "trajectories"
        }

    # ========================================
    # EXPERIMENT 5: Exhaustive Phase Space
    # ========================================
    if time.time() < end_time:
        remaining = (end_time - time.time()) / 60
        print(f"\n{'='*70}")
        print(f"EXPERIMENT 5: EXHAUSTIVE PHASE SPACE ({remaining:.0f} min remaining)")
        print("="*70)

        phase_space = run_exhaustive_phase_space(
            damping_points=25,
            coupling_points=25,
            trials_per_point=15,
            evolution_steps=400
        )

        print(f"\n  Results:")
        print(f"    Total simulations: {phase_space['total_simulations']}")
        print(f"    Phase boundaries: {phase_space['phase_boundaries']}")
        print(f"    Phases found: {phase_space['phases_found']}")

        results["phase_space"] = phase_space

    # ========================================
    # FINAL REPORT
    # ========================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("MASSIVE REALITY TEST COMPLETE")
    print("=" * 70)
    print(f"\nTotal elapsed time: {elapsed/60:.1f} minutes")

    if "census" in results:
        print(f"\nMULTIVERSE: {results['census']['n_universes']} universes simulated")
        print(f"  Unique configurations: {results['census']['unique_universes']}")
        print(f"  This suggests the multiverse has ~{results['census']['unique_universes']} distinct physics")

    if "genealogy" in results:
        print(f"\nGENEALOGY: {results['genealogy']['n_generations']} generations traced")
        print(f"  Immortal rules: {results['genealogy']['immortal_rules']}")
        print(f"  These are the 'fundamental constants' that never change")

    if "large_tokens" in results:
        print(f"\nSCALING: Tested up to n={max(results['large_tokens'].keys())} tokens")
        max_n = max(results['large_tokens'].keys())
        print(f"  At n={max_n}: {results['large_tokens'][max_n]['fill_rate']*100:.1f}% fill rate")
        print(f"  Larger systems are harder to saturate")

    return results


if __name__ == "__main__":
    import sys

    duration = 120
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except:
            pass

    print(f"Starting massive reality test (target: {duration} minutes)...")
    results = run_massive_test(duration_minutes=duration)
