"""
Hierarchical Learner V13 v2 - Matching V12 behavior exactly

The key insight: V12's success comes from:
1. Exact memory as first priority (100% confidence for seen pairs)
2. Hierarchical rules with specificity-based selection
3. Single-token rules for generalization

Let's keep the same structure but make it cleaner.
"""

from typing import Set, Dict, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


@dataclass
class QueryResult:
    """Unified query result"""
    effects: Set[str]
    confidence: float
    source: str
    support: int = 0
    alternatives: List[Tuple[Set[str], float]] = field(default_factory=list)


class HierarchicalLearnerV13:
    """
    Simplified V13 that maintains V12 accuracy.
    
    Core principle: Exact matches trump everything, then most-specific rule wins.
    """
    
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        
        # LAYER 1: Exact memory (highest priority)
        # (state, action) → {effects → count}
        self.exact_memory: Dict[Tuple[FrozenSet[str], int], Dict[FrozenSet[str], int]] = defaultdict(lambda: defaultdict(int))
        
        # LAYER 2: Hierarchical rules 
        # (pattern, action) → {effects → count, observations, specificity}
        self.rules: Dict[Tuple[FrozenSet[str], int], Dict] = {}
        
        # Reverse index for abduction
        self._effect_producers: Dict[str, List[Tuple[FrozenSet[str], int]]] = defaultdict(list)
        
        # Temporal
        self._prev_action: Optional[int] = None
        self._action_pairs: Dict[Tuple[int, int], int] = defaultdict(int)
        
        self.total_observations = 0
    
    def observe(self, before: Set[str], action: int, after: Set[str]):
        """Record transition"""
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        # Compute effects
        added = after_fs - before_fs
        removed = before_fs - after_fs
        effects = frozenset({f"+{t}" for t in added} | {f"-{t}" for t in removed})
        
        # 1. Exact memory
        self.exact_memory[(before_fs, action)][effects] += 1
        
        # 2. Hierarchical rules
        self._update_rules(before_fs, action, effects)
        
        # 3. Reverse index
        for effect in effects:
            self._effect_producers[effect].append((before_fs, action))
        
        # Temporal
        if self._prev_action is not None:
            self._action_pairs[(self._prev_action, action)] += 1
        self._prev_action = action
        
        self.total_observations += 1
    
    def _update_rules(self, state: FrozenSet[str], action: int, effects: FrozenSet[str]):
        """Update hierarchical rules at multiple specificity levels"""
        # Full state rule
        self._add_rule(state, action, effects)
        
        # Single-token rules (for generalization)
        for token in state:
            self._add_rule(frozenset([token]), action, effects)
    
    def _add_rule(self, pattern: FrozenSet[str], action: int, effects: FrozenSet[str]):
        """Add or update a rule"""
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
        self._prev_action = None
    
    def predict(self, state: Set[str], action: int) -> Set[str]:
        """Basic prediction - return most likely effects"""
        result = self.predict_with_confidence(state, action)
        return result.effects
    
    def predict_with_confidence(self, state: Set[str], action: int) -> QueryResult:
        """Predict with confidence"""
        state_fs = frozenset(state)
        key = (state_fs, action)
        
        # 1. Exact match - highest confidence
        if key in self.exact_memory:
            outcomes = self.exact_memory[key]
            total = sum(outcomes.values())
            
            sorted_outcomes = sorted(outcomes.items(), key=lambda x: -x[1])
            best_effects, best_count = sorted_outcomes[0]
            
            alternatives = [(set(eff), cnt/total) for eff, cnt in sorted_outcomes[1:4]]
            
            return QueryResult(
                effects=set(best_effects),
                confidence=best_count / total,
                source="exact",
                support=total,
                alternatives=alternatives
            )
        
        # 2. Find best matching rule
        best_rule_key = self._find_best_rule(state_fs, action)
        
        if best_rule_key:
            rule_data = self.rules[best_rule_key]
            outcomes = rule_data['effect_counts']
            total = sum(outcomes.values())
            
            sorted_outcomes = sorted(outcomes.items(), key=lambda x: -x[1])
            best_effects, best_count = sorted_outcomes[0]
            
            # Confidence scales with specificity and observations (V12 formula)
            base_confidence = best_count / total
            specificity_factor = min(1.0, rule_data['specificity'] / 10)
            observation_factor = min(1.0, rule_data['observations'] / 20)
            confidence = base_confidence * specificity_factor * observation_factor
            
            return QueryResult(
                effects=set(best_effects),
                confidence=confidence,
                source="rule",
                support=rule_data['observations'],
                alternatives=[(set(eff), cnt/total) for eff, cnt in sorted_outcomes[1:4]]
            )
        
        # 3. Unknown
        return QueryResult(effects=set(), confidence=0.1, source="unknown")
    
    def _find_best_rule(self, state: FrozenSet[str], action: int) -> Optional[Tuple[FrozenSet[str], int]]:
        """Find most specific matching rule"""
        candidates = []
        
        for (pattern, act), rule_data in self.rules.items():
            if act != action:
                continue
            if not pattern.issubset(state):
                continue
            
            candidates.append((
                (pattern, act),
                rule_data['specificity'],
                rule_data['observations']
            ))
        
        if not candidates:
            return None
        
        # Sort by specificity (desc), then observations (desc)
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        return candidates[0][0]
    
    # === ABDUCTION ===
    
    def abduce(self, target_effect: str) -> Dict:
        """Backward reasoning: what produces target effect?"""
        producers = self._effect_producers.get(target_effect, [])
        
        if not producers:
            return {'enabling_actions': [], 'required_tokens': set(), 'confidence': 0.0}
        
        # Deduplicate and count
        action_counts = defaultdict(int)
        all_tokens = set()
        
        for state, action in producers:
            action_counts[action] += 1
            all_tokens |= state
        
        total = sum(action_counts.values())
        best_action = max(action_counts.items(), key=lambda x: x[1])
        
        return {
            'enabling_actions': sorted(action_counts.keys(), key=lambda a: -action_counts[a])[:5],
            'required_tokens': all_tokens,
            'confidence': best_action[1] / total
        }
    
    def what_produces(self, token: str) -> List[Tuple[int, float]]:
        return self._get_action_stats(f"+{token}")
    
    def what_consumes(self, token: str) -> List[Tuple[int, float]]:
        return self._get_action_stats(f"-{token}")
    
    def _get_action_stats(self, effect: str) -> List[Tuple[int, float]]:
        producers = self._effect_producers.get(effect, [])
        action_counts = defaultdict(int)
        for _, action in producers:
            action_counts[action] += 1
        total = sum(action_counts.values())
        if total == 0:
            return []
        return sorted([(a, c/total) for a, c in action_counts.items()], key=lambda x: -x[1])
    
    # === INDUCTION ===
    
    def induce_path(self, before: Set[str], after: Set[str], 
                    max_steps: int = 10) -> List[Tuple[List[Tuple[FrozenSet, FrozenSet]], float]]:
        """Find rule chain connecting states"""
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        if before_fs == after_fs:
            return [([], 1.0)]
        
        # Build unique rules
        unique_rules = {}
        for key, outcomes in self.exact_memory.items():
            state, action = key
            for effects, count in outcomes.items():
                total = sum(outcomes.values())
                conf = count / total
                rule_key = (state, effects)
                if rule_key not in unique_rules or conf > unique_rules[rule_key]:
                    unique_rules[rule_key] = conf
        
        # A* search
        initial = (before_fs, (), 1.0)
        frontier = [(0, 0, initial)]
        visited = {before_fs}
        solutions = []
        counter = 1
        
        while frontier and len(solutions) < 5 and counter < 10000:
            _, _, (current, path, conf) = heapq.heappop(frontier)
            
            if current == after_fs:
                solutions.append((list(path), conf))
                continue
            
            if len(path) >= max_steps:
                continue
            
            for (lhs, rhs), rule_conf in unique_rules.items():
                if not lhs.issubset(current):
                    continue
                
                new_tokens = set(current)
                for effect in rhs:
                    if effect.startswith('+'):
                        new_tokens.add(effect[1:])
                    elif effect.startswith('-'):
                        new_tokens.discard(effect[1:])
                new_fs = frozenset(new_tokens)
                
                if new_fs in visited:
                    continue
                
                visited.add(new_fs)
                new_conf = conf * rule_conf
                new_path = path + ((lhs, rhs),)
                
                dist = len(new_fs - after_fs) + len(after_fs - new_fs)
                priority = dist + len(new_path) * 0.5 - new_conf
                
                heapq.heappush(frontier, (priority, counter, (new_fs, new_path, new_conf)))
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
        uncertainty = 1.0 - result.confidence
        return 0.5 * novelty + 0.5 * uncertainty
    
    # === UTILITIES ===
    
    def stats(self) -> Dict:
        return {
            'total_observations': self.total_observations,
            'exact_pairs': len(self.exact_memory),
            'rules': len(self.rules),
            'effects_indexed': len(self._effect_producers),
        }


HierarchicalLearner = HierarchicalLearnerV13
