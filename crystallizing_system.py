"""
CRYSTALLIZING RULE SYSTEM

A complete implementation for achieving 100% rule coverage in deterministic games.
"""

from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass
import random


@dataclass
class CrystallizedRule:
    """A crystallized (minimal, high-confidence) rule."""
    pattern: FrozenSet[str]
    action: int
    effect: str
    precision: float
    observations: int
    contexts_tested: int
    
    def __str__(self):
        pattern_str = ', '.join(sorted(self.pattern)) if self.pattern else '(any)'
        return f"{{{pattern_str}}} + a{self.action} → {self.effect} ({self.precision:.0%}, n={self.observations})"


class CrystallizingRuleSystem:
    """System for achieving 100% rule coverage through crystallization."""
    
    def __init__(self, n_actions: int = 6):
        self.n_actions = n_actions
        self.observations: List[Tuple[FrozenSet, int, FrozenSet]] = []
        self.vocabulary: Set[str] = set()
        self.crystallized_rules: List[CrystallizedRule] = []
        
        # Coverage tracking
        self.state_action_counts: Dict[Tuple[FrozenSet, int], int] = defaultdict(int)
        self.token_action_counts: Dict[Tuple[str, int], int] = defaultdict(int)
        
        # Effect tracking for crystallization
        self.effect_positive_contexts: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
    def observe(self, before: Set[str], action: int, after: Set[str]):
        """Record an observation."""
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        self.observations.append((before_fs, action, after_fs))
        self.vocabulary.update(before)
        self.vocabulary.update(after)
        
        # Update coverage
        self.state_action_counts[(before_fs, action)] += 1
        for token in before:
            self.token_action_counts[(token, action)] += 1
        
        # Track effects
        for token in after - before:
            self.effect_positive_contexts[(f"+{token}", action)].append(before_fs)
        for token in before - after:
            self.effect_positive_contexts[(f"-{token}", action)].append(before_fs)
    
    def crystallize(self, min_observations: int = 5, min_confidence: float = 0.95) -> List[CrystallizedRule]:
        """Extract minimal rules via ablation."""
        rules = []
        
        for (effect, action), contexts in self.effect_positive_contexts.items():
            if len(contexts) < min_observations:
                continue
            
            # Find common tokens across all positive examples
            common = set(contexts[0])
            for ctx in contexts[1:]:
                common &= ctx
            
            # Ablate: find necessary tokens
            necessary = set()
            for token in common:
                other = common - {token}
                
                # Check precision without this token
                matches = effect_count = 0
                for before, a, after in self.observations:
                    if a != action or not other.issubset(before):
                        continue
                    matches += 1
                    target = effect[1:]
                    if effect[0] == '+' and target in after and target not in before:
                        effect_count += 1
                    elif effect[0] == '-' and target in before and target not in after:
                        effect_count += 1
                
                # Token is necessary if removing it drops precision
                if matches > 0 and effect_count / matches < min_confidence:
                    necessary.add(token)
            
            if not necessary:
                necessary = common
            
            # Compute final precision
            matches = effect_count = 0
            for before, a, after in self.observations:
                if a != action or not necessary.issubset(before):
                    continue
                matches += 1
                target = effect[1:]
                if effect[0] == '+' and target in after and target not in before:
                    effect_count += 1
                elif effect[0] == '-' and target in before and target not in after:
                    effect_count += 1
            
            if matches > 0 and effect_count >= min_observations:
                precision = effect_count / matches
                if precision >= min_confidence:
                    rules.append(CrystallizedRule(
                        pattern=frozenset(necessary),
                        action=action,
                        effect=effect,
                        precision=precision,
                        observations=effect_count,
                        contexts_tested=len(contexts)
                    ))
        
        self.crystallized_rules = rules
        return rules
    
    def get_coverage_gaps(self, min_coverage: int = 5) -> List[Dict]:
        """Identify areas needing more exploration."""
        gaps = []
        
        # Find low-coverage token-action pairs
        for token in self.vocabulary:
            for action in range(self.n_actions):
                count = self.token_action_counts[(token, action)]
                if count < min_coverage:
                    gaps.append({
                        'type': 'low_coverage',
                        'token': token,
                        'action': action,
                        'count': count,
                        'priority': min_coverage - count
                    })
        
        # Find uncrystallized effects
        crystallized = {(r.effect, r.action) for r in self.crystallized_rules}
        for (effect, action) in self.effect_positive_contexts:
            if (effect, action) not in crystallized:
                gaps.append({
                    'type': 'uncrystallized',
                    'effect': effect,
                    'action': action,
                    'priority': 10
                })
        
        gaps.sort(key=lambda g: -g['priority'])
        return gaps
    
    def suggest_action(self, current_tokens: Set[str]) -> int:
        """Suggest action to maximize coverage."""
        current_fs = frozenset(current_tokens)
        
        # Prefer untested state-action pairs
        counts = [self.state_action_counts[(current_fs, a)] for a in range(self.n_actions)]
        if min(counts) == 0:
            return counts.index(0)
        
        # Otherwise prefer least-tested
        min_count = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_count]
        return random.choice(candidates)
    
    def is_complete(self, min_coverage: int = 5) -> Tuple[bool, str]:
        """Check if coverage is complete."""
        gaps = self.get_coverage_gaps(min_coverage)
        critical = [g for g in gaps if g['priority'] >= 5]
        
        if critical:
            return False, f"{len(critical)} critical gaps remain"
        
        return True, "Coverage complete"
    
    def get_minimal_ruleset(self) -> List[CrystallizedRule]:
        """Return crystallized rules sorted by action and effect."""
        return sorted(self.crystallized_rules, key=lambda r: (r.action, r.effect))
    
    def stats(self) -> Dict:
        """Return statistics."""
        gaps = self.get_coverage_gaps()
        complete, status = self.is_complete()
        
        return {
            'observations': len(self.observations),
            'vocabulary': len(self.vocabulary),
            'unique_states': len(set(b for b, _, _ in self.observations)),
            'rules_crystallized': len(self.crystallized_rules),
            'coverage_gaps': len(gaps),
            'complete': complete,
            'status': status
        }


if __name__ == "__main__":
    print("CrystallizingRuleSystem loaded successfully")
