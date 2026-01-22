"""
THE HYPOTHESIS GENERATOR

This is the ONLY new component needed.
Everything else delegates to your existing engines.
"""

from typing import Set, List, FrozenSet
from dataclasses import dataclass


@dataclass
class CandidateRule:
    """A hypothesized rule to be tested"""
    lhs: FrozenSet[str]      # Preconditions
    rhs: FrozenSet[str]      # Effects (+token, -token)
    specificity: int         # |LHS| - simpler = more general
    prior: float             # Initial confidence before testing
    source: str              # Which heuristic generated this


def generate_hypotheses(
    before: Set[str], 
    after: Set[str], 
    action: int = None
) -> List[CandidateRule]:
    """
    Generate candidate rules from a single observation.
    
    This is the INDUCTIVE step - going from specific observation
    to general rule candidates.
    
    Heuristics used:
    1. CONSUMED: Tokens that disappeared were probably required
    2. PRODUCED: Tokens that appeared are the effect
    3. CATALYSTS: Tokens present in both might be conditions
    4. ACTION: If action space is small, action is likely relevant
    5. OCCAM: Simpler rules (fewer LHS tokens) preferred
    """
    
    # Compute what changed
    added = after - before
    removed = before - after
    unchanged = before & after
    
    # Effects are the ground truth
    effects = frozenset({f"+{t}" for t in added} | {f"-{t}" for t in removed})
    
    if not effects:
        return []  # No change observed
    
    candidates = []
    action_token = f"__A:{action}" if action is not None else None
    
    # === HEURISTIC 1: Minimal - just action (if available) ===
    if action_token:
        candidates.append(CandidateRule(
            lhs=frozenset({action_token}),
            rhs=effects,
            specificity=1,
            prior=0.1,  # Low prior - probably too simple
            source="action_only"
        ))
    
    # === HEURISTIC 2: Consumed tokens are preconditions ===
    # If B disappeared, B was probably required
    for t in removed:
        lhs = {t}
        if action_token:
            lhs.add(action_token)
        candidates.append(CandidateRule(
            lhs=frozenset(lhs),
            rhs=effects,
            specificity=len(lhs),
            prior=0.6,  # High prior - consumption is strong signal
            source="consumed"
        ))
    
    # === HEURISTIC 3: Consumed + Produced suggests transformation ===
    # B → C pattern (B consumed, C produced)
    if removed and added:
        for r in removed:
            for a in added:
                # Rule: r + action → -r, +a
                lhs = {r}
                if action_token:
                    lhs.add(action_token)
                specific_effects = frozenset({f"-{r}", f"+{a}"})
                candidates.append(CandidateRule(
                    lhs=frozenset(lhs),
                    rhs=specific_effects,
                    specificity=len(lhs),
                    prior=0.5,
                    source="transformation"
                ))
    
    # === HEURISTIC 4: Full observation (exact match) ===
    # Memorize the exact state - will overfit but always correct
    full_lhs = set(before)
    if action_token:
        full_lhs.add(action_token)
    candidates.append(CandidateRule(
        lhs=frozenset(full_lhs),
        rhs=effects,
        specificity=len(full_lhs),
        prior=0.3,  # Medium prior - correct but overfits
        source="exact_match"
    ))
    
    # === HEURISTIC 5: Unchanged tokens as catalysts ===
    # Things that were present but didn't change might be conditions
    for t in unchanged:
        if t == action_token:
            continue
        lhs = {t}
        if action_token:
            lhs.add(action_token)
        candidates.append(CandidateRule(
            lhs=frozenset(lhs),
            rhs=effects,
            specificity=len(lhs),
            prior=0.2,  # Lower prior - weaker signal
            source="catalyst"
        ))
    
    # === HEURISTIC 6: Pairs of consumed tokens ===
    # Maybe two tokens are both required
    removed_list = list(removed)
    for i in range(len(removed_list)):
        for j in range(i+1, len(removed_list)):
            lhs = {removed_list[i], removed_list[j]}
            if action_token:
                lhs.add(action_token)
            candidates.append(CandidateRule(
                lhs=frozenset(lhs),
                rhs=effects,
                specificity=len(lhs),
                prior=0.4,
                source="consumed_pair"
            ))
    
    # === HEURISTIC 7: No action variant ===
    # Maybe the action doesn't matter
    if action_token:
        for t in removed:
            candidates.append(CandidateRule(
                lhs=frozenset({t}),
                rhs=effects,
                specificity=1,
                prior=0.3,
                source="no_action"
            ))
    
    # Deduplicate by (lhs, rhs)
    seen = set()
    unique = []
    for c in candidates:
        key = (c.lhs, c.rhs)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    # Sort by prior (descending) then specificity (ascending = simpler first)
    unique.sort(key=lambda c: (-c.prior, c.specificity))
    
    return unique


def print_candidates(candidates: List[CandidateRule]):
    """Pretty print candidate rules"""
    print(f"\n{'LHS':<40} {'RHS':<30} {'Prior':>6} {'Source':<15}")
    print("-" * 95)
    for c in candidates:
        lhs_str = ", ".join(sorted(c.lhs))[:38]
        rhs_str = ", ".join(sorted(c.rhs))[:28]
        print(f"{lhs_str:<40} {rhs_str:<30} {c.prior:>6.2f} {c.source:<15}")


# === TEST ===
if __name__ == "__main__":
    print("="*70)
    print("HYPOTHESIS GENERATOR TEST")
    print("="*70)
    
    # Example: Crafting wood into plank
    before = {"has_wood", "has_axe", "at_workbench"}
    after = {"has_plank", "has_axe", "at_workbench"}
    action = 5  # CRAFT action
    
    print(f"\nObservation:")
    print(f"  Before: {before}")
    print(f"  Action: {action}")
    print(f"  After:  {after}")
    print(f"  Δ: +has_plank, -has_wood")
    
    candidates = generate_hypotheses(before, after, action)
    
    print(f"\nGenerated {len(candidates)} candidate rules:")
    print_candidates(candidates)
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
Top candidates (by prior):
1. 'has_wood' consumed → High prior because it disappeared
2. 'has_wood' + action_5 → Consumption + action
3. Transformation: has_wood → has_plank

The TESTING phase (your engine) will determine:
- Does 'has_wood + action_5 → +plank, -wood' hold universally?
- Or do we also need 'at_workbench'?
- Or is 'has_axe' required?

The generator proposes. Your engine disposes.
""")
