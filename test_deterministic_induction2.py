"""
TEST: Deterministic induction algorithm on concrete examples (fixed)
"""

from typing import Set, FrozenSet, List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)  # Make hashable
class Observation:
    before: FrozenSet[str]
    after: FrozenSet[str]
    
    @property
    def effect(self) -> FrozenSet[str]:
        added = self.after - self.before
        removed = self.before - self.after
        return frozenset({f"+{t}" for t in added} | {f"-{t}" for t in removed})


@dataclass(frozen=True)  # Make hashable
class Rule:
    lhs: FrozenSet[str]
    rhs: FrozenSet[str]
    
    def __repr__(self):
        lhs_str = " ∧ ".join(sorted(self.lhs)) if self.lhs else "∅"
        rhs_str = ", ".join(sorted(self.rhs))
        return f"{lhs_str} → {rhs_str}"


def deterministic_induction(observations: List[Observation]) -> List[Rule]:
    """
    Find minimal rules that explain all observations.
    """
    
    # Group by effect
    by_effect: Dict[FrozenSet[str], List[FrozenSet[str]]] = defaultdict(list)
    for obs in observations:
        if obs.effect:
            by_effect[obs.effect].append(obs.before)
    
    rules = []
    
    for effect, positive_states in by_effect.items():
        # Step 1: Intersection
        lhs = frozenset.intersection(*positive_states)
        
        # Step 2: Discrimination
        for obs in observations:
            if obs.effect != effect and lhs <= obs.before:
                for pos_state in positive_states:
                    diff = pos_state - obs.before
                    if diff:
                        discriminator = next(iter(diff))
                        lhs = lhs | {discriminator}
                        break
        
        rules.append(Rule(lhs=lhs, rhs=effect))
    
    return rules


def validate_rules(rules: List[Rule], observations: List[Observation]):
    """Validate each rule against all observations."""
    print("\nValidation:")
    for rule in rules:
        hits = 0
        misses = 0
        
        for obs in observations:
            if rule.lhs <= obs.before:
                if rule.rhs == obs.effect:
                    hits += 1
                else:
                    misses += 1
        
        status = "✓" if misses == 0 else "✗"
        print(f"  {status} {rule}: {hits} hits, {misses} misses")


# ============ TEST CASES ============

print("="*70)
print("TEST 1: Simple crafting - wood + bench → plank")
print("="*70)

observations1 = [
    Observation(frozenset({"wood", "bench", "axe"}), frozenset({"plank", "bench", "axe"})),
    Observation(frozenset({"wood", "bench", "hammer"}), frozenset({"plank", "bench", "hammer"})),
    Observation(frozenset({"wood", "axe"}), frozenset({"wood", "axe"})),  # No bench
    Observation(frozenset({"bench", "hammer"}), frozenset({"bench", "hammer"})),  # No wood
]

print("\nObservations:")
for obs in observations1:
    eff = set(obs.effect) if obs.effect else "∅"
    print(f"  {set(obs.before)} → {eff}")

rules1 = deterministic_induction(observations1)
print("\nDiscovered rules:")
for rule in rules1:
    print(f"  {rule}")

validate_rules(rules1, observations1)

print("\n" + "="*70)
print("TEST 2: Farm planting - seeds + water + field → crop")  
print("="*70)

observations2 = [
    Observation(frozenset({"seeds", "water", "field", "sunny"}), 
                frozenset({"crop", "field", "sunny"})),
    Observation(frozenset({"seeds", "water", "field", "cloudy"}), 
                frozenset({"crop", "field", "cloudy"})),
    Observation(frozenset({"seeds", "field"}), frozenset({"seeds", "field"})),  # No water
    Observation(frozenset({"water", "field"}), frozenset({"water", "field"})),  # No seeds
    Observation(frozenset({"seeds", "water"}), frozenset({"seeds", "water"})),  # No field
]

print("\nObservations:")
for obs in observations2:
    eff = set(obs.effect) if obs.effect else "∅"
    print(f"  {set(obs.before)} → {eff}")

rules2 = deterministic_induction(observations2)
print("\nDiscovered rules:")
for rule in rules2:
    print(f"  {rule}")

validate_rules(rules2, observations2)

print("\n" + "="*70)
print("TEST 3: Complex - requires 3 tokens, has distractors")  
print("="*70)

# True rule: A + B + C → +X, -A
# Distractors: D, E, F appear randomly
observations3 = [
    Observation(frozenset({"A", "B", "C"}), frozenset({"B", "C", "X"})),
    Observation(frozenset({"A", "B", "C", "D"}), frozenset({"B", "C", "D", "X"})),
    Observation(frozenset({"A", "B", "C", "E"}), frozenset({"B", "C", "E", "X"})),
    Observation(frozenset({"A", "B", "C", "D", "E"}), frozenset({"B", "C", "D", "E", "X"})),
    # Negatives
    Observation(frozenset({"A", "B"}), frozenset({"A", "B"})),  # No C
    Observation(frozenset({"A", "C"}), frozenset({"A", "C"})),  # No B
    Observation(frozenset({"B", "C"}), frozenset({"B", "C"})),  # No A
    Observation(frozenset({"A", "B", "D"}), frozenset({"A", "B", "D"})),  # No C
    Observation(frozenset({"D", "E", "F"}), frozenset({"D", "E", "F"})),  # None of A,B,C
]

print("\nObservations:")
for obs in observations3:
    eff = set(obs.effect) if obs.effect else "∅"
    print(f"  {set(obs.before)} → {eff}")

rules3 = deterministic_induction(observations3)
print("\nDiscovered rules:")
for rule in rules3:
    print(f"  {rule}")

validate_rules(rules3, observations3)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
The algorithm:
1. INTERSECTION: Find tokens in ALL positive examples
2. DISCRIMINATION: Add tokens to distinguish from negatives

Complexity: O(E × N × T) - linear in observations!

This is NOT exploring 2^N hypotheses.
It's directly computing the minimal LHS using set operations.

The magic: Negative examples (where nothing happened) let us
eliminate unnecessary conditions via discrimination.
""")
