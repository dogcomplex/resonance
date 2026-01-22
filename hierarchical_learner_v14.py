"""
Hierarchical Learner V14 - Clean Meta-Rule Foundation

This version:
1. Keeps V13's working core (exact_memory + rules)
2. Adds meta-rule infrastructure WITHOUT breaking existing functionality
3. Uses __ prefix for meta-tokens to ensure no domain collision

The meta-rule engine is SEPARATE and can be used for:
- Batch processing observations
- Parallel confidence updates
- GPU-acceleratable operations
"""

from typing import Set, Dict, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


@dataclass
class QueryResult:
    """Query result with metadata"""
    effects: Set[str]
    confidence: float
    source: str
    support: int = 0
    alternatives: List[Tuple[Set[str], float]] = field(default_factory=list)


class HierarchicalLearnerV14:
    """
    V14: V13 core + meta-rule foundation for future GPU acceleration.
    
    Meta-tokens use __ prefix and NEVER mix with domain tokens.
    """
    
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        
        # === CORE STORAGE (V13 proven design) ===
        self.exact_memory: Dict[Tuple[FrozenSet[str], int], Dict[FrozenSet[str], int]] = \
            defaultdict(lambda: defaultdict(int))
        self.rules: Dict[Tuple[FrozenSet[str], int], Dict] = {}
        self._effect_producers: Dict[str, List[Tuple[FrozenSet[str], int]]] = defaultdict(list)
        
        # === META-RULE INFRASTRUCTURE ===
        # These could become GPU tensors:
        # - rule_lhs[i] = set of required tokens for rule i
        # - rule_rhs[i] = set of effects for rule i  
        # - rule_action[i] = action for rule i
        # - rule_count[i] = observation count for rule i
        self._meta_rules: List[Dict] = []  # Future: convert to tensors
        
        self.total_observations = 0
    
    # === CORE OPERATIONS (V13 proven) ===
    
    def observe(self, before: Set[str], action: int, after: Set[str]):
        """Record transition - unchanged from V13"""
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        added = after_fs - before_fs
        removed = before_fs - after_fs
        effects = frozenset({f"+{t}" for t in added} | {f"-{t}" for t in removed})
        
        # Exact memory (store even empty effects for counting)
        self.exact_memory[(before_fs, action)][effects] += 1
        
        # Hierarchical rules
        self._update_rules(before_fs, action, effects)
        
        # Effect index
        for effect in effects:
            self._effect_producers[effect].append((before_fs, action))
        
        self.total_observations += 1
    
    def _update_rules(self, state: FrozenSet[str], action: int, effects: FrozenSet[str]):
        """Update rules at multiple specificity levels"""
        self._add_rule(state, action, effects)
        for token in state:
            self._add_rule(frozenset([token]), action, effects)
    
    def _add_rule(self, pattern: FrozenSet[str], action: int, effects: FrozenSet[str]):
        """Add or update rule"""
        key = (pattern, action)
        if key not in self.rules:
            self.rules[key] = {
                'effect_counts': defaultdict(int),
                'observations': 0,
                'specificity': len(pattern)
            }
        self.rules[key]['effect_counts'][effects] += 1
        self.rules[key]['observations'] += 1
    
    def reset_episode(self):
        pass
    
    def predict(self, state: Set[str], action: int) -> Set[str]:
        """Predict effects"""
        return self.predict_with_confidence(state, action).effects
    
    def predict_with_confidence(self, state: Set[str], action: int) -> QueryResult:
        """Predict with confidence - V13 logic exactly"""
        state_fs = frozenset(state)
        key = (state_fs, action)
        
        # Exact match first
        if key in self.exact_memory:
            outcomes = self.exact_memory[key]
            total = sum(outcomes.values())
            sorted_out = sorted(outcomes.items(), key=lambda x: -x[1])
            best_eff, best_cnt = sorted_out[0]
            
            return QueryResult(
                effects=set(best_eff),
                confidence=best_cnt / total,
                source="exact",
                support=total,
                alternatives=[(set(e), c/total) for e, c in sorted_out[1:4]]
            )
        
        # Rule match
        best_key = self._find_best_rule(state_fs, action)
        if best_key:
            data = self.rules[best_key]
            outcomes = data['effect_counts']
            total = sum(outcomes.values())
            sorted_out = sorted(outcomes.items(), key=lambda x: -x[1])
            best_eff, best_cnt = sorted_out[0]
            
            base_conf = best_cnt / total
            spec_factor = min(1.0, data['specificity'] / 10)
            obs_factor = min(1.0, data['observations'] / 20)
            
            return QueryResult(
                effects=set(best_eff),
                confidence=base_conf * spec_factor * obs_factor,
                source="rule",
                support=data['observations'],
                alternatives=[(set(e), c/total) for e, c in sorted_out[1:4]]
            )
        
        return QueryResult(effects=set(), confidence=0.1, source="unknown")
    
    def _find_best_rule(self, state: FrozenSet[str], action: int):
        """Find most specific matching rule"""
        candidates = []
        for (pattern, act), data in self.rules.items():
            if act != action or not pattern.issubset(state):
                continue
            candidates.append(((pattern, act), data['specificity'], data['observations']))
        
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        return candidates[0][0]
    
    # === ABDUCTION ===
    
    def abduce(self, target_effect: str) -> Dict:
        """Backward reasoning"""
        producers = self._effect_producers.get(target_effect, [])
        if not producers:
            return {'enabling_actions': [], 'required_tokens': set(), 'confidence': 0.0}
        
        action_counts = defaultdict(int)
        all_tokens = set()
        for state, action in producers:
            action_counts[action] += 1
            all_tokens |= state
        
        total = sum(action_counts.values())
        best = max(action_counts.items(), key=lambda x: x[1])
        
        return {
            'enabling_actions': sorted(action_counts.keys(), key=lambda a: -action_counts[a])[:5],
            'required_tokens': all_tokens,
            'confidence': best[1] / total
        }
    
    def what_produces(self, token: str) -> List[Tuple[int, float]]:
        return self._action_stats(f"+{token}")
    
    def what_consumes(self, token: str) -> List[Tuple[int, float]]:
        return self._action_stats(f"-{token}")
    
    def _action_stats(self, effect: str) -> List[Tuple[int, float]]:
        producers = self._effect_producers.get(effect, [])
        counts = defaultdict(int)
        for _, action in producers:
            counts[action] += 1
        total = sum(counts.values())
        if total == 0:
            return []
        return sorted([(a, c/total) for a, c in counts.items()], key=lambda x: -x[1])
    
    # === INDUCTION ===
    
    def induce_path(self, before: Set[str], after: Set[str], 
                    max_steps: int = 10) -> List[Tuple[List, float]]:
        """A* path search"""
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        if before_fs == after_fs:
            return [([], 1.0)]
        
        # Build rule set
        rules = {}
        for key, outcomes in self.exact_memory.items():
            state, _ = key
            for effects, count in outcomes.items():
                total = sum(outcomes.values())
                conf = count / total
                rk = (state, effects)
                if rk not in rules or conf > rules[rk]:
                    rules[rk] = conf
        
        # A* search
        frontier = [(0, 0, (before_fs, (), 1.0))]
        visited = {before_fs}
        solutions = []
        counter = 1
        
        while frontier and len(solutions) < 5 and counter < 5000:
            _, _, (current, path, conf) = heapq.heappop(frontier)
            
            if current == after_fs:
                solutions.append((list(path), conf))
                continue
            
            if len(path) >= max_steps:
                continue
            
            for (lhs, rhs), rule_conf in rules.items():
                if not lhs.issubset(current):
                    continue
                
                new_tokens = set(current)
                for e in rhs:
                    if e.startswith('+'):
                        new_tokens.add(e[1:])
                    elif e.startswith('-'):
                        new_tokens.discard(e[1:])
                new_fs = frozenset(new_tokens)
                
                if new_fs in visited:
                    continue
                
                visited.add(new_fs)
                new_conf = conf * rule_conf
                new_path = path + ((lhs, rhs),)
                dist = len(new_fs - after_fs) + len(after_fs - new_fs)
                heapq.heappush(frontier, (dist - new_conf, counter, (new_fs, new_path, new_conf)))
                counter += 1
        
        solutions.sort(key=lambda x: -x[1])
        return solutions
    
    def explain_transition(self, before: Set[str], after: Set[str]) -> str:
        paths = self.induce_path(before, after, max_steps=5)
        if not paths:
            return f"No path. Diff: -{before-after}, +{after-before}"
        rules, conf = paths[0]
        lines = [f"Path ({conf:.0%}):"]
        for i, (_, rhs) in enumerate(rules):
            c = {e[1:] for e in rhs if e.startswith('-')}
            p = {e[1:] for e in rhs if e.startswith('+')}
            lines.append(f"  {i+1}. -{c} → +{p}")
        return "\n".join(lines)
    
    # === EXPLORATION ===
    
    def novelty_score(self, state: Set[str], action: int) -> float:
        key = (frozenset(state), action)
        if key in self.exact_memory:
            count = sum(self.exact_memory[key].values())
            return 1.0 / (1.0 + count * 0.5)
        return 1.0
    
    def exploration_value(self, state: Set[str], action: int) -> float:
        novelty = self.novelty_score(state, action)
        result = self.predict_with_confidence(state, action)
        return 0.5 * novelty + 0.5 * (1.0 - result.confidence)
    
    # === META-RULE BATCH OPERATIONS ===
    # These are designed for future GPU acceleration
    
    def batch_observe(self, transitions: List[Tuple[Set[str], int, Set[str]]]):
        """
        Batch observation - can be parallelized.
        
        Each transition is independent, so this maps well to GPU.
        """
        for before, action, after in transitions:
            self.observe(before, action, after)
    
    def batch_predict(self, queries: List[Tuple[Set[str], int]]) -> List[QueryResult]:
        """
        Batch prediction - can be parallelized.
        
        On GPU this would be:
        1. Encode all query states as sparse vectors
        2. Matrix multiply against rule LHS
        3. Find matches in parallel
        4. Look up RHS for matches
        """
        return [self.predict_with_confidence(state, action) for state, action in queries]
    
    def stats(self) -> Dict:
        return {
            'total_observations': self.total_observations,
            'exact_pairs': len(self.exact_memory),
            'rules': len(self.rules),
            'effects_indexed': len(self._effect_producers),
        }


HierarchicalLearner = HierarchicalLearnerV14
