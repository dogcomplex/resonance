"""
PROBABILISTIC SIEVE V3: Correct probability estimation

The insight: If rule fires with probability P, then:
- P fraction of observations are positives
- (1-P) fraction are negatives WITH THE SAME LHS

When we sample:
- Sample K positives → intersection gives candidate LHS
- Sample M negatives → if any has LHS ⊆ neg, it's a "contradiction"

The probability of hitting a contradiction in our negative sample is:
- Let q = (1-P) = fraction of all relevant obs that are negative
- With sample ratio r, we sample r*N negatives
- P(hit contradiction) ≈ q if we sample enough negatives

So survival rate ≈ P (the true probability)!

BUT: We need to count ONLY negatives that have our LHS.
Currently we're counting ALL negatives.
"""

import random
from typing import Set, FrozenSet, List, Dict
from collections import defaultdict
from dataclasses import dataclass

random.seed(42)


@dataclass(frozen=True)
class Observation:
    before: FrozenSet[str]
    after: FrozenSet[str]
    
    @property
    def effect(self) -> FrozenSet[str]:
        added = self.after - self.before
        removed = self.before - self.after
        return frozenset({f"+{t}" for t in added} | {f"-{t}" for t in removed})


def generate_observations(lhs, effect, probability, n, extras):
    observations = []
    for _ in range(n):
        before = set(lhs)
        num_extras = min(random.randint(0, 2), len(extras))
        if num_extras > 0:
            before.update(random.sample(extras, k=num_extras))
        
        if random.random() < probability:
            after = set(before)
            for e in effect:
                if e.startswith('+'):
                    after.add(e[1:])
                elif e.startswith('-'):
                    after.discard(e[1:])
            observations.append(Observation(frozenset(before), frozenset(after)))
        else:
            observations.append(Observation(frozenset(before), frozenset(before)))
    
    return observations


def estimate_rule_probability(
    observations: List[Observation],
    candidate_lhs: FrozenSet[str],
    target_effect: FrozenSet[str]
) -> float:
    """
    Estimate probability of rule: candidate_lhs → target_effect
    
    P(rule) = count(lhs present AND effect occurred) / count(lhs present)
    """
    lhs_present = 0
    effect_occurred = 0
    
    for obs in observations:
        if candidate_lhs <= obs.before:  # LHS is present
            lhs_present += 1
            if obs.effect == target_effect:
                effect_occurred += 1
    
    if lhs_present == 0:
        return 0.0
    
    return effect_occurred / lhs_present


def probabilistic_sieve_v3(
    observations: List[Observation],
    num_samples: int = 100,
    sample_ratio: float = 0.7
) -> List[tuple]:
    """
    Find rules with probability estimates.
    
    1. Sample positives, compute intersection → candidate LHS
    2. For each candidate, compute P(effect | LHS present)
    
    This is just counting - no search needed!
    """
    
    # Group by effect
    by_effect = defaultdict(list)
    for obs in observations:
        if obs.effect:
            by_effect[obs.effect].append(obs.before)
    
    results = []
    
    for effect, positives in by_effect.items():
        if len(positives) < 2:
            continue
        
        # Generate candidate LHS via sampling
        candidate_lhs_set = set()
        
        for _ in range(num_samples):
            sample_size = max(2, int(len(positives) * sample_ratio))
            sample = random.sample(positives, min(sample_size, len(positives)))
            lhs = frozenset.intersection(*sample)
            if lhs:
                candidate_lhs_set.add(lhs)
        
        # For each candidate, estimate probability
        for lhs in candidate_lhs_set:
            prob = estimate_rule_probability(observations, lhs, effect)
            results.append((lhs, effect, prob))
    
    return sorted(results, key=lambda x: -x[2])


# ============ TEST ============

print("="*70)
print("PROBABILISTIC SIEVE V3: Direct Probability Estimation")
print("="*70)

print("""
Key insight: 
  P(rule) = count(LHS present AND effect) / count(LHS present)
  
This is just COUNTING, not sieving!
The sieve just generates candidate LHS via intersection sampling.
""")


for true_prob in [1.0, 0.8, 0.5, 0.3]:
    print(f"\n--- True probability: {true_prob:.0%} ---")
    
    obs = generate_observations(
        lhs={"A", "B"}, 
        effect={"+X"}, 
        probability=true_prob,
        n=200, 
        extras=["C", "D", "E", "F"]
    )
    
    actual_positives = sum(1 for o in obs if o.effect == frozenset({"+X"}))
    actual_with_lhs = sum(1 for o in obs if {"A", "B"} <= o.before)
    actual_rate = actual_positives / actual_with_lhs if actual_with_lhs > 0 else 0
    
    print(f"  Actual: {actual_positives}/{actual_with_lhs} = {actual_rate:.1%}")
    
    results = probabilistic_sieve_v3(obs, num_samples=100)
    
    for lhs, eff, prob in results[:3]:
        match = "✓" if {"A", "B"} <= set(lhs) else ""
        print(f"  Found: {set(lhs)} → {set(eff)} @ {prob:.1%} {match}")


print("\n" + "="*70)
print("THE ALGORITHM SIMPLIFIED")
print("="*70)
print("""
STEP 1: Sample positives, intersect → candidate LHS
        (This generates plausible rules without 2^N enumeration)

STEP 2: For each candidate, count:
        P(effect | LHS) = #(LHS ∧ effect) / #(LHS)
        
STEP 3: Bucket by probability:
        100%: Deterministic
        70-99%: Strong probabilistic
        50-69%: Medium
        <50%: Weak or noise

COMPLEXITY: O(num_samples × N) for candidate generation
            O(num_candidates × N) for probability counting
            Total: O(N) with reasonable sample counts

STILL JUST COUNTING AND SET OPERATIONS!
""")
