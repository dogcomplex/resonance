"""
Unified Best Learner

Combines insights from all learners:
1. Pure rules (100% confidence) for strong predictions
2. Soft rules (high confidence) for coverage
3. Prior-weighted voting when no rules match
4. Discriminative scoring for rare labels
5. Phase-based confidence tracking

Target: Maximize accuracy at ALL coverage levels
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field


@dataclass
class Pattern:
    """A pattern with label statistics."""
    positions: FrozenSet[Tuple[int, str]]
    label_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_matches: int = 0
    
    def matches(self, board: str) -> bool:
        for pos, val in self.positions:
            if pos >= len(board) or board[pos] != val:
                return False
        return True
    
    def observe(self, label: int):
        self.label_counts[label] += 1
        self.total_matches += 1
    
    def precision_for(self, label: int) -> float:
        if self.total_matches == 0:
            return 0.0
        return self.label_counts[label] / self.total_matches
    
    def best_label(self) -> Optional[int]:
        if not self.label_counts:
            return None
        return max(self.label_counts, key=self.label_counts.get)
    
    def best_precision(self) -> float:
        best = self.best_label()
        return self.precision_for(best) if best is not None else 0.0
    
    def is_pure(self, min_obs: int = 2) -> bool:
        if self.total_matches < min_obs:
            return False
        return len([c for c in self.label_counts.values() if c > 0]) == 1
    
    def specificity(self) -> int:
        return len(self.positions)


class UnifiedLearner:
    """
    Best-of-breed few-shot learner.
    
    Key innovations:
    1. Tracks ALL patterns (not just rules with single output)
    2. Uses patterns with >90% precision as "soft rules"
    3. Uses patterns with 100% precision as "hard rules"
    4. Weighted prediction combining rules and prior
    """
    
    def __init__(self, num_outputs: int = 5, board_size: int = None,
                 label_names: List[str] = None,
                 pattern_sizes: List[int] = None,
                 hard_threshold: float = 1.0,
                 soft_threshold: float = 0.85,
                 min_support: int = 2,
                 **kwargs):
        self.num_outputs = num_outputs
        self.board_size = board_size
        self.label_names = label_names
        self.pattern_sizes = pattern_sizes or [3]
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.min_support = min_support
        
        # Pattern storage
        self.patterns: Dict[FrozenSet[Tuple[int, str]], Pattern] = {}
        
        # Classified rules
        self.hard_rules: Dict[int, List[Pattern]] = defaultdict(list)  # 100% precision
        self.soft_rules: Dict[int, List[Pattern]] = defaultdict(list)  # >85% precision
        
        # Observations
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        self.label_counts: Dict[int, int] = defaultdict(int)
        self.observed_values: Set[str] = set()
        
        self.stats = {
            'patterns': 0,
            'hard_rules': 0,
            'soft_rules': 0,
            'predictions': 0,
        }
    
    def _extract_patterns(self, board: str) -> List[FrozenSet[Tuple[int, str]]]:
        """Extract all patterns from board."""
        patterns = []
        for size in self.pattern_sizes:
            if size > len(board):
                continue
            for positions in combinations(range(len(board)), size):
                pattern = frozenset((pos, board[pos]) for pos in positions)
                patterns.append(pattern)
        return patterns
    
    def _update_patterns(self, board: str, label: int):
        """Update pattern statistics."""
        for pattern_key in self._extract_patterns(board):
            if pattern_key not in self.patterns:
                self.patterns[pattern_key] = Pattern(positions=pattern_key)
            self.patterns[pattern_key].observe(label)
        self.stats['patterns'] = len(self.patterns)
    
    def _classify_rules(self):
        """Classify patterns into hard and soft rules."""
        self.hard_rules.clear()
        self.soft_rules.clear()
        
        for pattern in self.patterns.values():
            if pattern.total_matches < self.min_support:
                continue
            
            best_label = pattern.best_label()
            if best_label is None:
                continue
            
            precision = pattern.best_precision()
            
            if precision >= self.hard_threshold:
                self.hard_rules[best_label].append(pattern)
            elif precision >= self.soft_threshold:
                self.soft_rules[best_label].append(pattern)
        
        # Sort by specificity then support
        for label in list(self.hard_rules.keys()) + list(self.soft_rules.keys()):
            if label in self.hard_rules:
                self.hard_rules[label].sort(
                    key=lambda p: (-p.specificity(), -p.total_matches)
                )
            if label in self.soft_rules:
                self.soft_rules[label].sort(
                    key=lambda p: (-p.specificity(), -p.best_precision(), -p.total_matches)
                )
        
        self.stats['hard_rules'] = sum(len(v) for v in self.hard_rules.values())
        self.stats['soft_rules'] = sum(len(v) for v in self.soft_rules.values())
    
    def predict(self, observation: str) -> int:
        """Predict using tiered rule matching."""
        self.stats['predictions'] += 1
        
        # Infer structure
        if self.board_size is None:
            self.board_size = len(observation)
        for char in observation:
            self.observed_values.add(char)
        
        # Calculate label rarity
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts.get(l, 0) / total 
                     for l in range(self.num_outputs)}
        
        # TIER 1: Hard rules (100% precision), rarest labels first
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for pattern in self.hard_rules.get(label, []):
                if pattern.matches(observation) and pattern.total_matches >= 3:
                    return label
        
        # TIER 2: Soft rules with weighted voting
        votes: Dict[int, float] = defaultdict(float)
        
        for label in range(self.num_outputs):
            for pattern in self.soft_rules.get(label, []):
                if pattern.matches(observation):
                    rarity_boost = 1.0 / (label_freq.get(label, 0.1) + 0.01)
                    weight = (
                        pattern.best_precision() * 
                        (pattern.specificity() ** 2) * 
                        pattern.total_matches * 
                        rarity_boost
                    )
                    votes[label] += weight
        
        # Also add hard rules with less support
        for label in range(self.num_outputs):
            for pattern in self.hard_rules.get(label, []):
                if pattern.matches(observation):
                    rarity_boost = 1.0 / (label_freq.get(label, 0.1) + 0.01)
                    weight = (
                        1.5 *  # Bonus for being pure
                        (pattern.specificity() ** 2) * 
                        pattern.total_matches * 
                        rarity_boost
                    )
                    votes[label] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        # TIER 3: Prior-based
        return self._prior_predict()
    
    def _prior_predict(self) -> int:
        """Prior-based prediction."""
        if not self.label_counts:
            return random.randint(0, self.num_outputs - 1)
        
        total = sum(self.label_counts.values())
        r = random.random() * total
        cumsum = 0
        for label, count in sorted(self.label_counts.items(), key=lambda x: -x[1]):
            cumsum += count
            if r <= cumsum:
                return label
        return max(self.label_counts, key=self.label_counts.get)
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        """Learn from observation."""
        self.history.append((observation, guess, correct_label))
        self.observations.append((observation, correct_label))
        self.label_counts[correct_label] += 1
        
        # Infer structure
        if self.board_size is None:
            self.board_size = len(observation)
        for char in observation:
            self.observed_values.add(char)
        
        # Update pattern statistics
        self._update_patterns(observation, correct_label)
        
        # Reclassify rules periodically
        if len(self.observations) % 10 == 0:
            self._classify_rules()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'observations': len(self.observations),
            'rules': self.stats['hard_rules'] + self.stats['soft_rules'],
            'pure_rules': self.stats['hard_rules'],
        }
    
    def describe_knowledge(self) -> str:
        """Describe learned knowledge."""
        lines = ["=== Unified Learner Knowledge ===\n"]
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Patterns tracked: {len(self.patterns)}")
        lines.append(f"Hard rules (100%): {self.stats['hard_rules']}")
        lines.append(f"Soft rules (>85%): {self.stats['soft_rules']}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for idx in range(self.num_outputs):
            count = self.label_counts.get(idx, 0)
            pct = count / total * 100 if total > 0 else 0
            label = self.label_names[idx] if self.label_names else str(idx)
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        
        lines.append("\n--- Hard Rules by Label ---")
        for label_idx in range(self.num_outputs):
            label_name = self.label_names[label_idx] if self.label_names else str(label_idx)
            rules = self.hard_rules.get(label_idx, [])
            lines.append(f"\n  {label_name}: {len(rules)} hard rules")
            for pattern in rules[:5]:
                pat_str = ', '.join(f"p{p}={v}" for p, v in sorted(pattern.positions))
                lines.append(f"    {pat_str} (n={pattern.total_matches})")
        
        return '\n'.join(lines)


# Alias
pure_rules = property(lambda self: self.hard_rules)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE
    
    print("="*70)
    print("Testing Unified Learner")
    print("="*70)
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    learner = UnifiedLearner(num_outputs=5, label_names=LABEL_SPACE)
    
    correct = 0
    checkpoints = [10, 25, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 6000]
    
    for i in range(7000):
        obs = gen.next()
        if obs is None:
            print(f"All {i} states observed!")
            break
        
        board, true_idx = obs
        pred_idx = learner.predict(board)
        
        if pred_idx == true_idx:
            correct += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if (i + 1) in checkpoints:
            acc = correct / (i + 1)
            stats = learner.get_stats()
            print(f"  @{i+1:5d}: acc={acc:.1%} hard={stats['hard_rules']} soft={stats['soft_rules']}")
    
    print(f"\nFinal Accuracy: {correct/(i+1):.1%}")
    print(f"Coverage: {gen.coverage():.1%}")
    print("\n" + learner.describe_knowledge())
