"""
Bidirectional Wave Integration with Curried Chemistry

This connects our physics insights to the actual rule learning system.

Key ideas:
1. Forward wave (deduction): Given observations, what rules could apply?
2. Backward wave (abduction): Given effects, what rules produced them?
3. Standing wave (induction): Where waves meet = discovered rules
4. Multiple crunches = multiple valid endpoints to learn toward
5. Adversarial = filtering waves by which crunch they reach
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class Token:
    """A token in the system - can be observation, intermediate, or effect"""
    name: str
    properties: Dict[str, any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name

@dataclass 
class Rule:
    """A causal rule: conditions → effects"""
    conditions: Tuple[str, ...]
    effects: Tuple[str, ...]
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.conditions, self.effects))
    
    def __eq__(self, other):
        return self.conditions == other.conditions and self.effects == other.effects
    
    def __str__(self):
        conds = ' + '.join(self.conditions)
        effs = ' + '.join(self.effects)
        return f"{conds} → {effs} ({self.confidence:.2f})"

@dataclass
class CurriedIntermediate:
    """A partially-bound rule waiting for more conditions"""
    bound_conditions: Tuple[str, ...]
    remaining_conditions: Tuple[str, ...]
    effects: Tuple[str, ...]
    parent_rule: Rule
    
    def __str__(self):
        bound = ' + '.join(self.bound_conditions)
        remaining = ' + '.join(self.remaining_conditions)
        effs = ' + '.join(self.effects)
        return f"[{bound}] + ({remaining}) → {effs}"


class BidirectionalRuleLearner:
    """
    Learn rules using bidirectional wave interference.
    
    Forward wave: From observations, expand reachable states
    Backward wave: From goals/endpoints, trace back what could cause them
    Standing wave: Rules that appear in both directions
    """
    
    def __init__(self):
        self.known_rules: Set[Rule] = set()
        self.candidate_rules: Dict[Rule, float] = {}  # rule -> resonance
        self.observations: List[Tuple[Set[str], Set[str]]] = []  # (before, after) pairs
        self.endpoints: Set[str] = set()  # Known "crunch" tokens
        
        # Wave caches
        self.forward_reach: Dict[str, Set[str]] = {}  # token -> reachable tokens
        self.backward_reach: Dict[str, Set[str]] = {}  # token -> tokens that reach it
        
    def add_endpoint(self, token: str):
        """Add a known endpoint (crunch) to learn toward"""
        self.endpoints.add(token)
    
    def observe(self, before: Set[str], after: Set[str]):
        """Record an observation of state transition"""
        self.observations.append((frozenset(before), frozenset(after)))
    
    def forward_expand(self, tokens: Set[str], depth: int = 5) -> Dict[str, List[Rule]]:
        """
        Deduction: Given tokens, what rules COULD apply and what would result?
        Returns: token -> [rules that produce it]
        """
        reachable = {t: [] for t in tokens}
        frontier = tokens.copy()
        
        for _ in range(depth):
            new_frontier = set()
            for rule in self.known_rules:
                # Check if all conditions are met
                if all(c in frontier for c in rule.conditions):
                    # Rule can fire - add effects to reachable
                    for effect in rule.effects:
                        if effect not in reachable:
                            reachable[effect] = []
                        reachable[effect].append(rule)
                        new_frontier.add(effect)
            frontier = frontier | new_frontier
        
        return reachable
    
    def backward_expand(self, goals: Set[str], depth: int = 5) -> Dict[str, List[Rule]]:
        """
        Abduction: Given goals, what rules COULD have produced them?
        Returns: token -> [rules that require it as condition]
        """
        needed = {g: [] for g in goals}
        frontier = goals.copy()
        
        for _ in range(depth):
            new_frontier = set()
            for rule in self.known_rules:
                # Check if any effect is in our goals
                if any(e in frontier for e in rule.effects):
                    # This rule could produce our goals - we need its conditions
                    for cond in rule.conditions:
                        if cond not in needed:
                            needed[cond] = []
                        needed[cond].append(rule)
                        new_frontier.add(cond)
            frontier = frontier | new_frontier
        
        return needed
    
    def find_standing_waves(self, before: Set[str], after: Set[str]) -> List[Rule]:
        """
        Induction: Find rules that connect before → after
        These are the "standing waves" - rules that appear in both directions
        """
        forward = self.forward_expand(before)
        backward = self.backward_expand(after)
        
        # Intersection: tokens reachable from both directions
        intersection = set(forward.keys()) & set(backward.keys())
        
        # Standing wave rules: rules that contribute to reaching the intersection
        standing_rules = set()
        
        for token in intersection:
            # Rules that produce this token (forward)
            for rule in forward.get(token, []):
                standing_rules.add(rule)
            # Rules that need this token (backward)
            for rule in backward.get(token, []):
                standing_rules.add(rule)
        
        return list(standing_rules)
    
    def propose_rules(self, before: Set[str], after: Set[str]) -> List[Rule]:
        """
        Given a (before, after) observation, propose candidate rules.
        This is the core induction step.
        """
        # What tokens appeared?
        new_tokens = after - before
        
        # What tokens disappeared?
        consumed = before - after
        
        # What tokens persisted?
        persisted = before & after
        
        candidates = []
        
        # Simple rule: all of before → all new tokens
        if new_tokens:
            candidates.append(Rule(
                conditions=tuple(sorted(before)),
                effects=tuple(sorted(new_tokens)),
                confidence=0.5
            ))
        
        # Simpler rules: subsets of before → new tokens
        for size in range(1, min(4, len(before) + 1)):
            for combo in self._combinations(list(before), size):
                candidates.append(Rule(
                    conditions=tuple(sorted(combo)),
                    effects=tuple(sorted(new_tokens)),
                    confidence=0.3
                ))
        
        # Curried rules: condition → intermediate → effect
        # (We'll detect these by seeing if any token appears temporarily)
        
        return candidates
    
    def _combinations(self, items, size):
        """Generate combinations of given size"""
        if size == 0:
            yield []
            return
        if not items:
            return
        first, rest = items[0], items[1:]
        # Include first
        for combo in self._combinations(rest, size - 1):
            yield [first] + combo
        # Exclude first
        for combo in self._combinations(rest, size):
            yield combo
    
    def learn_from_observations(self):
        """
        Main learning loop: process observations and induce rules.
        Uses bidirectional waves to filter candidate rules.
        """
        for before, after in self.observations:
            # Propose candidate rules
            candidates = self.propose_rules(before, after)
            
            # Score each candidate by how well it explains the observation
            for rule in candidates:
                # Does this rule's conditions match before?
                if not all(c in before for c in rule.conditions):
                    continue
                
                # Does this rule's effects appear in after?
                if not all(e in after for e in rule.effects):
                    continue
                
                # Resonance: increment if we've seen this rule work before
                if rule not in self.candidate_rules:
                    self.candidate_rules[rule] = 0
                self.candidate_rules[rule] += 1
        
        # Promote high-resonance candidates to known rules
        threshold = 2  # Must appear in at least 2 observations
        for rule, count in self.candidate_rules.items():
            if count >= threshold:
                self.known_rules.add(rule)
    
    def learn_with_endpoints(self):
        """
        Learn rules with awareness of multiple endpoints (crunches).
        Different endpoints may require different rules.
        """
        if not self.endpoints:
            return self.learn_from_observations()
        
        # Group observations by which endpoint they reach
        obs_by_endpoint = defaultdict(list)
        
        for before, after in self.observations:
            # Check which endpoints are in 'after'
            reached = after & self.endpoints
            for endpoint in reached:
                obs_by_endpoint[endpoint].append((before, after))
            if not reached:
                obs_by_endpoint['unknown'].append((before, after))
        
        # Learn rules separately for each endpoint
        rules_by_endpoint = {}
        
        for endpoint, obs_list in obs_by_endpoint.items():
            endpoint_learner = BidirectionalRuleLearner()
            for before, after in obs_list:
                endpoint_learner.observe(before, after)
            endpoint_learner.learn_from_observations()
            rules_by_endpoint[endpoint] = endpoint_learner.known_rules
        
        # Rules that appear for MULTIPLE endpoints are more fundamental
        rule_endpoint_count = defaultdict(set)
        for endpoint, rules in rules_by_endpoint.items():
            for rule in rules:
                rule_endpoint_count[rule].add(endpoint)
        
        # Rank rules by how many endpoints they contribute to
        print("\nRules by endpoint coverage:")
        for rule, endpoints in sorted(rule_endpoint_count.items(), 
                                      key=lambda x: -len(x[1])):
            print(f"  {rule}: {endpoints}")
        
        return rules_by_endpoint


def test_simple_learning():
    """Test basic bidirectional rule learning"""
    print("=" * 70)
    print("TEST: Simple Bidirectional Rule Learning")
    print("=" * 70)
    
    learner = BidirectionalRuleLearner()
    
    # Add some observations (TTT-like)
    # Before: current board tokens, After: next board + terminal
    
    # X plays in center, O plays corner
    learner.observe(
        {'empty_board'},
        {'X@4', 'O@0', 'turn_X'}
    )
    
    # More observations
    learner.observe(
        {'X@4', 'turn_O'},
        {'X@4', 'O@0', 'turn_X'}
    )
    
    learner.observe(
        {'X@4', 'O@0', 'turn_X'},
        {'X@4', 'O@0', 'X@8', 'turn_O'}
    )
    
    # X wins
    learner.observe(
        {'X@0', 'X@4', 'X@8', 'turn_O'},
        {'X@0', 'X@4', 'X@8', 'WIN_X'}
    )
    
    # O wins
    learner.observe(
        {'O@0', 'O@1', 'O@2', 'turn_X'},
        {'O@0', 'O@1', 'O@2', 'WIN_O'}
    )
    
    # Learn rules
    learner.learn_from_observations()
    
    print(f"\nLearned {len(learner.known_rules)} rules:")
    for rule in learner.known_rules:
        print(f"  {rule}")
    
    print(f"\nCandidate rules ({len(learner.candidate_rules)}):")
    for rule, count in sorted(learner.candidate_rules.items(), key=lambda x: -x[1])[:10]:
        print(f"  ({count}) {rule}")


def test_endpoint_aware_learning():
    """Test learning with multiple endpoints"""
    print("\n" + "=" * 70)
    print("TEST: Endpoint-Aware Learning (Multiple Crunches)")
    print("=" * 70)
    
    learner = BidirectionalRuleLearner()
    
    # Define endpoints
    learner.add_endpoint('WIN_X')
    learner.add_endpoint('WIN_O')
    learner.add_endpoint('TIE')
    
    # Observations leading to X win
    learner.observe({'A', 'B'}, {'A', 'B', 'C'})
    learner.observe({'A', 'B', 'C'}, {'A', 'B', 'C', 'WIN_X'})
    learner.observe({'X', 'Y'}, {'X', 'Y', 'Z'})
    learner.observe({'X', 'Y', 'Z'}, {'X', 'Y', 'Z', 'WIN_X'})
    
    # Observations leading to O win
    learner.observe({'P', 'Q'}, {'P', 'Q', 'R'})
    learner.observe({'P', 'Q', 'R'}, {'P', 'Q', 'R', 'WIN_O'})
    
    # Observations leading to tie
    learner.observe({'A', 'B'}, {'A', 'B', 'D'})
    learner.observe({'A', 'B', 'D'}, {'A', 'B', 'D', 'TIE'})
    
    # Learn with endpoint awareness
    rules_by_endpoint = learner.learn_with_endpoints()
    
    for endpoint, rules in rules_by_endpoint.items():
        print(f"\nRules for {endpoint}:")
        for rule in rules:
            print(f"  {rule}")


def test_curried_wave_learning():
    """Test learning curried (multi-step) rules via waves"""
    print("\n" + "=" * 70)
    print("TEST: Curried Rule Learning via Waves")
    print("=" * 70)
    
    learner = BidirectionalRuleLearner()
    
    # Simulate curried reaction: A + B → [A•B] → C
    # We observe: {A, B} → {A, B, [A•B]} → {C}
    
    learner.observe({'A', 'B'}, {'A', 'B', 'AB_intermediate'})
    learner.observe({'AB_intermediate'}, {'C'})
    
    # Same pattern with different tokens
    learner.observe({'X', 'Y'}, {'X', 'Y', 'XY_intermediate'})
    learner.observe({'XY_intermediate'}, {'Z'})
    
    # Direct observation (skipping intermediate)
    learner.observe({'A', 'B'}, {'C'})
    learner.observe({'X', 'Y'}, {'Z'})
    
    learner.learn_from_observations()
    
    print(f"\nLearned rules:")
    for rule in learner.known_rules:
        print(f"  {rule}")
    
    # Now test forward and backward expansion
    print("\nForward expansion from {A, B}:")
    forward = learner.forward_expand({'A', 'B'})
    for token, rules in forward.items():
        if rules:
            print(f"  {token}: via {[str(r) for r in rules]}")
    
    print("\nBackward expansion from {C}:")
    backward = learner.backward_expand({'C'})
    for token, rules in backward.items():
        if rules:
            print(f"  {token}: needed by {[str(r) for r in rules]}")
    
    print("\nStanding waves connecting {A, B} → {C}:")
    standing = learner.find_standing_waves({'A', 'B'}, {'C'})
    for rule in standing:
        print(f"  {rule}")


def demonstrate_perspective_filtering():
    """Show how different perspectives filter the same rules"""
    print("\n" + "=" * 70)
    print("TEST: Perspective Filtering on Rules")
    print("=" * 70)
    
    # Imagine we have rules for a game
    all_rules = [
        Rule(('X_move', 'empty_4'), ('X@4',)),
        Rule(('O_move', 'empty_4'), ('O@4',)),
        Rule(('X@0', 'X@4', 'X@8'), ('WIN_X',)),
        Rule(('O@0', 'O@1', 'O@2'), ('WIN_O',)),
        Rule(('full_board', 'no_winner'), ('TIE',)),
    ]
    
    # Define perspectives
    perspectives = {
        'X_player': {
            'control': lambda r: 'X_move' in r.conditions or 'X@' in str(r.effects),
            'goal': 'WIN_X'
        },
        'O_player': {
            'control': lambda r: 'O_move' in r.conditions or 'O@' in str(r.effects),
            'goal': 'WIN_O'
        },
        'Tie_seeker': {
            'control': lambda r: True,  # Controls all
            'goal': 'TIE'
        }
    }
    
    for name, perspective in perspectives.items():
        print(f"\n{name}'s perspective:")
        
        # Filter rules by control
        my_rules = [r for r in all_rules if perspective['control'](r)]
        print(f"  My rules: {len(my_rules)}")
        for r in my_rules:
            print(f"    {r}")
        
        # Filter rules by goal
        goal_rules = [r for r in all_rules if perspective['goal'] in r.effects]
        print(f"  Goal rules: {goal_rules}")


if __name__ == "__main__":
    test_simple_learning()
    test_endpoint_aware_learning()
    test_curried_wave_learning()
    demonstrate_perspective_filtering()
    
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)
    print("""
How Bidirectional Waves Enhance Curried Chemistry:

1. FORWARD WAVE (Deduction):
   - Given current tokens, what rules can fire?
   - Expand reachable states through known rules
   - This is what we already do in rule application

2. BACKWARD WAVE (Abduction):
   - Given desired endpoint, what rules are needed?
   - Trace back through effects → conditions
   - NEW: Helps focus rule search on relevant patterns

3. STANDING WAVE (Induction):
   - Rules that appear in both forward and backward passes
   - These are the rules that ACTUALLY connect observation to goal
   - Higher resonance = more likely to be real rules

4. MULTI-ENDPOINT (Multiple Crunches):
   - Different endpoints (W/L/T/E) are different attractors
   - Rules can be endpoint-specific or universal
   - Universal rules are more fundamental

5. PERSPECTIVE FILTERING:
   - Same rules, different views
   - Agent filters for rules they control
   - Goal filters for rules that lead there
   - No extra computation - just filtering!

6. CURRIED RULES VIA WAVES:
   - Intermediates appear in forward expansion
   - Get "explained" by backward expansion
   - Standing wave confirms the curry chain

This unifies:
- Rule learning (induction via standing waves)
- Rule application (deduction via forward waves)
- Goal-directed reasoning (abduction via backward waves)
- Multi-agent dynamics (perspective filtering)
""")
