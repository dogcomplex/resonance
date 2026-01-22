"""
RULE CHAINING SYSTEM

A complete system for:
1. Learning atomic rules from anonymized observations
2. Chaining rules via A* search to reach goals
3. Discovering state abstractions from behavioral equivalence
4. Generating compound "macro" rules

Based on the insight from space.txt:
- Spatial understanding emerges from simple causal rules
- Geometry naturally limits combinatorial explosion
- Paths compress into reusable compound rules
"""

import random
from collections import defaultdict
import heapq
from typing import Dict, List, Tuple, Set, Callable, Optional


class RuleLearner:
    """
    Learns rules from (before, action, after) observations.
    """
    
    def __init__(self, min_support: int = 3, min_confidence: float = 0.9):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.observations: List[Tuple] = []
        self.rule_counts: Dict[Tuple, Dict[Tuple, int]] = defaultdict(lambda: defaultdict(int))
    
    def observe(self, before: tuple, action: str, after: tuple):
        """Record an observation."""
        self.observations.append((before, action, after))
        key = (before, action)
        self.rule_counts[key][after] += 1
    
    def get_rules(self) -> List[Dict]:
        """Extract high-confidence rules."""
        rules = []
        for (before, action), outcomes in self.rule_counts.items():
            total = sum(outcomes.values())
            if total < self.min_support:
                continue
            
            for after, count in outcomes.items():
                conf = count / total
                if conf >= self.min_confidence:
                    rules.append({
                        "before": before,
                        "action": action,
                        "after": after,
                        "confidence": conf,
                        "support": count,
                    })
        
        return rules


class RuleChainer:
    """
    Chains rules via search to find paths between states.
    """
    
    def __init__(self, rules: List[Dict]):
        # Index rules by (before, action)
        self.transitions: Dict[tuple, List[Tuple]] = defaultdict(list)
        self.all_actions: Set[str] = set()
        
        for rule in rules:
            key = (rule["before"], rule["action"])
            self.transitions[key].append((rule["after"], rule["confidence"]))
            self.all_actions.add(rule["action"])
    
    def get_transitions(self, state: tuple) -> List[Tuple[str, tuple, float]]:
        """Get all possible (action, next_state, confidence) from state."""
        trans = []
        for action in self.all_actions:
            key = (state, action)
            for next_state, conf in self.transitions.get(key, []):
                trans.append((action, next_state, conf))
        return trans
    
    def search(self, 
               start: tuple, 
               goal: Callable[[tuple], bool],
               max_depth: int = 10) -> Optional[Tuple[List, float]]:
        """
        A* search from start to goal.
        Returns (path, cost) or None.
        """
        # Priority queue: (cost, state, path)
        frontier = [(0, start, [])]
        visited = set()
        
        while frontier:
            cost, state, path = heapq.heappop(frontier)
            
            if state in visited:
                continue
            visited.add(state)
            
            if goal(state):
                return path, cost
            
            if len(path) >= max_depth:
                continue
            
            for action, next_state, conf in self.get_transitions(state):
                if next_state not in visited:
                    # Cost = steps + uncertainty
                    new_cost = cost + 1 + (1 - conf)
                    new_path = path + [(action, next_state)]
                    heapq.heappush(frontier, (new_cost, next_state, new_path))
        
        return None, float('inf')
    
    def find_all_reachable(self, start: tuple, max_depth: int = 5) -> Dict[tuple, List]:
        """Find all states reachable from start with their paths."""
        reachable = {start: []}
        frontier = [(start, [])]
        
        while frontier:
            state, path = frontier.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            for action, next_state, conf in self.get_transitions(state):
                new_path = path + [(action, next_state)]
                
                if next_state not in reachable:
                    reachable[next_state] = new_path
                    frontier.append((next_state, new_path))
        
        return reachable


class AbstractionDiscoverer:
    """
    Discovers state abstractions based on behavioral equivalence.
    """
    
    def __init__(self, chainer: RuleChainer):
        self.chainer = chainer
    
    def find_equivalences(self, states: Set[tuple]) -> Dict[str, Set[tuple]]:
        """
        Group states by transition signature.
        States with identical transitions are equivalent.
        """
        signatures: Dict[tuple, List[tuple]] = defaultdict(list)
        
        for state in states:
            trans = self.chainer.get_transitions(state)
            # Signature = sorted (action, result) pairs
            sig = tuple(sorted((a, s) for a, s, c in trans))
            signatures[sig].append(state)
        
        # Filter to groups with >1 state
        abstractions = {}
        for i, (sig, equiv_states) in enumerate(signatures.items()):
            if len(equiv_states) > 1:
                # Find common properties for naming
                name = self._name_abstraction(equiv_states)
                abstractions[name] = set(equiv_states)
        
        return abstractions
    
    def _name_abstraction(self, states: List[tuple]) -> str:
        """Generate descriptive name for an abstraction."""
        if not states:
            return "Empty"
        
        # Find dimensions that vary vs are constant
        n_dims = len(states[0])
        const_dims = []
        vary_dims = []
        
        for i in range(n_dims):
            vals = set(s[i] for s in states)
            if len(vals) == 1:
                const_dims.append(f"{i}={list(vals)[0]}")
            else:
                vary_dims.append(f"{i}=*")
        
        return f"Abstract({','.join(const_dims + vary_dims)})"


class CompoundRuleGenerator:
    """
    Generates compound (macro) rules from paths.
    """
    
    def __init__(self, chainer: RuleChainer):
        self.chainer = chainer
        self.compound_rules: List[Dict] = []
    
    def generate_for_goal(self, 
                          start_states: Set[tuple],
                          goal: Callable[[tuple], bool],
                          max_depth: int = 6) -> List[Dict]:
        """Generate compound rules from each start state to goal."""
        for start in start_states:
            result = self.chainer.search(start, goal, max_depth)
            if result[0] is not None:
                path, cost = result
                actions = tuple(a for a, s in path)
                final = path[-1][1] if path else start
                
                self.compound_rules.append({
                    "from": start,
                    "actions": actions,
                    "to": final,
                    "length": len(actions),
                    "cost": cost,
                })
        
        return self.compound_rules
    
    def generate_all_paths(self, max_depth: int = 4) -> List[Dict]:
        """Generate all reachable paths (for exhaustive exploration)."""
        # Get all known states
        all_states = set()
        for (before, action), outcomes in self.chainer.transitions.items():
            all_states.add(before)
            for after, conf in outcomes:
                all_states.add(after)
        
        # Find paths from each state
        for start in all_states:
            reachable = self.chainer.find_all_reachable(start, max_depth)
            
            for end, path in reachable.items():
                if path:  # Skip trivial self-paths
                    actions = tuple(a for a, s in path)
                    self.compound_rules.append({
                        "from": start,
                        "actions": actions,
                        "to": end,
                        "length": len(actions),
                    })
        
        return self.compound_rules


# =============================================================================
# INTEGRATED SYSTEM
# =============================================================================

class HonestRuleSystem:
    """
    Complete system integrating all components.
    """
    
    def __init__(self, min_support=3, min_confidence=0.9):
        self.learner = RuleLearner(min_support, min_confidence)
        self.chainer = None
        self.abstraction_discoverer = None
        self.compound_generator = None
    
    def observe(self, before: tuple, action: str, after: tuple):
        """Add observation."""
        self.learner.observe(before, action, after)
    
    def build(self):
        """Build all components from learned rules."""
        rules = self.learner.get_rules()
        self.chainer = RuleChainer(rules)
        self.abstraction_discoverer = AbstractionDiscoverer(self.chainer)
        self.compound_generator = CompoundRuleGenerator(self.chainer)
        return rules
    
    def find_path_to_goal(self, start: tuple, goal: Callable[[tuple], bool], max_depth=10):
        """Find path from start to goal."""
        if not self.chainer:
            self.build()
        return self.chainer.search(start, goal, max_depth)
    
    def discover_abstractions(self) -> Dict[str, Set[tuple]]:
        """Find behavioral equivalences."""
        if not self.abstraction_discoverer:
            self.build()
        
        # Get all states
        all_states = set()
        for obs in self.learner.observations:
            all_states.add(obs[0])
            all_states.add(obs[2])
        
        return self.abstraction_discoverer.find_equivalences(all_states)
    
    def generate_compound_rules(self, goal: Callable[[tuple], bool], max_depth=6):
        """Generate macro rules for reaching goal."""
        if not self.compound_generator:
            self.build()
        
        # Get all states
        all_states = set()
        for obs in self.learner.observations:
            all_states.add(obs[0])
            all_states.add(obs[2])
        
        return self.compound_generator.generate_for_goal(all_states, goal, max_depth)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RULE CHAINING SYSTEM DEMO")
    print("=" * 60)
    
    # Create system
    system = HonestRuleSystem(min_support=2, min_confidence=0.9)
    
    # Add some example rules (simulating learned observations)
    # States: (direction, front_tile)
    directions = ["U", "R", "D", "L"]
    tiles = ["floor", "wall", "goal"]
    
    # Turn rules
    for i, d in enumerate(directions):
        for t in tiles:
            # Turn left: direction rotates counterclockwise
            system.observe((d, t), "left", (directions[(i-1)%4], t))
            system.observe((d, t), "right", (directions[(i+1)%4], t))
    
    # Forward rules
    for d in directions:
        # Wall blocks
        system.observe((d, "wall"), "fwd", (d, "wall"))
        # Floor allows movement (leads to various tiles)
        system.observe((d, "floor"), "fwd", (d, "floor"))
        system.observe((d, "floor"), "fwd", (d, "wall"))
        system.observe((d, "floor"), "fwd", (d, "goal"))
        # Goal
        system.observe((d, "goal"), "fwd", (d, "goal"))
    
    # Build
    rules = system.build()
    print(f"\nLearned {len(rules)} atomic rules")
    
    # Find path to goal
    print("\n--- Finding path to GOAL ---")
    start = ("U", "floor")
    goal = lambda s: s[1] == "goal"
    
    path, cost = system.find_path_to_goal(start, goal)
    if path:
        print(f"Path from {start}:")
        for action, state in path:
            print(f"  {action} -> {state}")
        print(f"Total cost: {cost:.2f}")
    
    # Discover abstractions
    print("\n--- Discovering abstractions ---")
    abstractions = system.discover_abstractions()
    print(f"Found {len(abstractions)} equivalence classes")
    for name, states in abstractions.items():
        print(f"  {name}: {len(states)} states")
    
    # Generate compound rules
    print("\n--- Generating compound rules ---")
    compound = system.generate_compound_rules(goal, max_depth=4)
    print(f"Generated {len(compound)} compound rules")
    
    # Show shortest
    if compound:
        shortest = min(compound, key=lambda r: r["length"])
        print(f"Shortest to goal: {shortest['from']} --[{shortest['length']}]--> {shortest['to']}")
        print(f"  Actions: {' -> '.join(shortest['actions'])}")
