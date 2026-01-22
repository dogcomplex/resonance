"""
Hybrid Few-Shot Learner

Combines multiple strategies for different phases of learning:

Phase 1 (0-50 observations): Prior-based + early pattern discovery
Phase 2 (50-200 observations): Hypothesis voting with rare label boost  
Phase 3 (200+ observations): Confidence-weighted patterns

Key innovations:
1. Separate hypothesis pools for each label
2. Discriminative scoring (patterns that distinguish labels)
3. Phase-appropriate strategies

NO CHEATING - purely observational learning.
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field


@dataclass
class Pattern:
    """A pattern with statistics per label."""
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
    
    def is_pure_for(self, label: int, min_obs: int = 2) -> bool:
        """Returns True if pattern ONLY predicts this label."""
        if self.total_matches < min_obs:
            return False
        return self.label_counts[label] == self.total_matches
    
    def discriminative_score(self, label: int, label_prior: float) -> float:
        """
        Score how well this pattern discriminates for this label.
        Higher = more predictive beyond prior.
        """
        if self.total_matches == 0:
            return 0.0
        precision = self.precision_for(label)
        # How much better than random?
        lift = precision / label_prior if label_prior > 0 else precision
        return lift * self.total_matches
    
    def specificity(self) -> int:
        return len(self.positions)


class HybridLearner:
    """
    Hybrid few-shot learner with phase-appropriate strategies.
    """
    
    def __init__(self, num_outputs: int = 5, board_size: int = None,
                 label_names: List[str] = None,
                 pattern_sizes: List[int] = None,
                 **kwargs):
        self.num_outputs = num_outputs
        self.board_size = board_size
        self.label_names = label_names
        self.pattern_sizes = pattern_sizes or [3]  # Focus on 3-patterns
        
        # Pattern storage - keyed by frozenset of (pos, val) pairs
        self.patterns: Dict[FrozenSet[Tuple[int, str]], Pattern] = {}
        
        # Pure rules (patterns that perfectly predict a label)
        self.pure_rules: Dict[int, List[Pattern]] = defaultdict(list)
        
        # Observations
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        self.label_counts: Dict[int, int] = defaultdict(int)
        
        # Track observed values
        self.observed_values: Set[str] = set()
        
        self.stats = {
            'patterns': 0,
            'pure_rules': 0,
            'predictions': 0,
        }
    
    def _extract_patterns(self, board: str) -> List[FrozenSet[Tuple[int, str]]]:
        """Extract all patterns of configured sizes from board."""
        patterns = []
        for size in self.pattern_sizes:
            if size > len(board):
                continue
            for positions in combinations(range(len(board)), size):
                pattern = frozenset((pos, board[pos]) for pos in positions)
                patterns.append(pattern)
        return patterns
    
    def _update_patterns(self, board: str, label: int):
        """Update pattern statistics with this observation."""
        for pattern_key in self._extract_patterns(board):
            if pattern_key not in self.patterns:
                self.patterns[pattern_key] = Pattern(positions=pattern_key)
            self.patterns[pattern_key].observe(label)
        
        self.stats['patterns'] = len(self.patterns)
    
    def _discover_pure_rules(self):
        """Find patterns that perfectly predict labels."""
        self.pure_rules.clear()
        
        for pattern in self.patterns.values():
            for label in range(self.num_outputs):
                if pattern.is_pure_for(label, min_obs=2):
                    self.pure_rules[label].append(pattern)
        
        # Sort by specificity (higher = better) then by support
        for label in self.pure_rules:
            self.pure_rules[label].sort(
                key=lambda p: (-p.specificity(), -p.total_matches)
            )
        
        self.stats['pure_rules'] = sum(len(v) for v in self.pure_rules.values())
    
    def predict(self, observation: str) -> int:
        """Predict using phase-appropriate strategy."""
        self.stats['predictions'] += 1
        
        # Infer structure
        if self.board_size is None:
            self.board_size = len(observation)
        for char in observation:
            self.observed_values.add(char)
        
        n_obs = len(self.observations)
        
        # Strategy depends on phase
        if n_obs < 20:
            return self._predict_early(observation)
        elif n_obs < 100:
            return self._predict_mid(observation)
        else:
            return self._predict_late(observation)
    
    def _predict_early(self, observation: str) -> int:
        """Early phase: mostly prior-based with any pure rules."""
        # Check pure rules first (prioritize rare labels)
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts[l] / total for l in range(self.num_outputs)}
        
        # Check rarest labels first
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for pattern in self.pure_rules.get(label, []):
                if pattern.matches(observation):
                    return label
        
        # Fall back to prior
        return self._prior_predict()
    
    def _predict_mid(self, observation: str) -> int:
        """Mid phase: weighted voting with rare label boost."""
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts[l] / total for l in range(self.num_outputs)}
        
        # Check pure rules first (prioritize rare labels)
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for pattern in self.pure_rules.get(label, []):
                if pattern.matches(observation):
                    return label
        
        # Weighted voting among all matching patterns
        votes: Dict[int, float] = defaultdict(float)
        
        for pattern in self.patterns.values():
            if not pattern.matches(observation):
                continue
            
            for label in range(self.num_outputs):
                if pattern.label_counts[label] == 0:
                    continue
                
                precision = pattern.precision_for(label)
                rarity_boost = 1.0 / (label_freq.get(label, 0.1) + 0.01)  # Boost rare labels
                weight = precision * pattern.specificity() * rarity_boost
                votes[label] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        return self._prior_predict()
    
    def _predict_late(self, observation: str) -> int:
        """Late phase: confidence-weighted with strong pure rule priority."""
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts[l] / total for l in range(self.num_outputs)}
        
        # Pure rules are VERY reliable by now - check them first
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for pattern in self.pure_rules.get(label, []):
                if pattern.matches(observation) and pattern.total_matches >= 3:
                    return label
        
        # Discriminative scoring
        scores: Dict[int, float] = defaultdict(float)
        
        for pattern in self.patterns.values():
            if not pattern.matches(observation):
                continue
            
            for label in range(self.num_outputs):
                score = pattern.discriminative_score(label, label_freq.get(label, 0.1))
                scores[label] += score
        
        if scores:
            return max(scores, key=scores.get)
        
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
        
        # Periodically discover pure rules
        if len(self.observations) % 5 == 0:
            self._discover_pure_rules()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'observations': len(self.observations),
            'rules': self.stats['pure_rules'],
        }
    
    def describe_knowledge(self) -> str:
        """Describe learned patterns."""
        lines = ["=== Hybrid Learner Knowledge ===\n"]
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Patterns tracked: {len(self.patterns)}")
        lines.append(f"Pure rules: {self.stats['pure_rules']}")
        lines.append(f"Observed values: {sorted(self.observed_values)}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for idx in range(self.num_outputs):
            count = self.label_counts.get(idx, 0)
            pct = count / total * 100 if total > 0 else 0
            label = self.label_names[idx] if self.label_names else str(idx)
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        
        lines.append("\n--- Pure Rules by Label ---")
        for label_idx in range(self.num_outputs):
            label_name = self.label_names[label_idx] if self.label_names else str(label_idx)
            rules = self.pure_rules.get(label_idx, [])
            lines.append(f"\n  {label_name}: {len(rules)} pure rules")
            
            for pattern in rules[:8]:
                pattern_str = ', '.join(f"p{p}={v}" for p, v in sorted(pattern.positions))
                lines.append(f"    {pattern_str} (n={pattern.total_matches})")
        
        return '\n'.join(lines)


class MultiSizeHybridLearner(HybridLearner):
    """
    Hybrid learner that tracks multiple pattern sizes and
    uses the most specific matching rule.
    """
    
    def __init__(self, **kwargs):
        # Override to use multiple sizes
        kwargs['pattern_sizes'] = [2, 3, 4]
        super().__init__(**kwargs)
    
    def _predict_late(self, observation: str) -> int:
        """Use most specific matching rule."""
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts[l] / total for l in range(self.num_outputs)}
        
        # Group pure rules by specificity
        rules_by_specificity: Dict[int, List[Tuple[int, Pattern]]] = defaultdict(list)
        for label, patterns in self.pure_rules.items():
            for pattern in patterns:
                if pattern.matches(observation) and pattern.total_matches >= 2:
                    rules_by_specificity[pattern.specificity()].append((label, pattern))
        
        # Check most specific rules first
        for spec in sorted(rules_by_specificity.keys(), reverse=True):
            rules = rules_by_specificity[spec]
            if rules:
                # Vote among rules at this specificity, boosting rare labels
                votes: Dict[int, float] = defaultdict(float)
                for label, pattern in rules:
                    rarity = 1.0 / (label_freq.get(label, 0.1) + 0.01)
                    votes[label] += pattern.total_matches * rarity
                
                if votes:
                    return max(votes, key=votes.get)
        
        # Fall back to discriminative scoring
        return super()._predict_late(observation)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    from test_harness import test_learner, compare_learners, print_comparison
    
    print("="*70)
    print("Testing Hybrid Learner")
    print("="*70)
    
    result = test_learner(
        HybridLearner, tictactoe, random_board, label_space,
        rounds=500, verbose=True
    )
    
    print(f"\nFinal Accuracy: {result['final_accuracy']:.1%}")
    
    print("\nPer-Label Accuracy:")
    for label, acc in result['per_label_accuracy'].items():
        count = result['per_label_counts'][label]
        print(f"  {label:8s}: {acc:.1%} ({count} samples)")
    
    print("\n" + result['learner'].describe_knowledge())
    
    print("\n" + "="*70)
    print("Testing Multi-Size Hybrid Learner")
    print("="*70)
    
    result2 = test_learner(
        MultiSizeHybridLearner, tictactoe, random_board, label_space,
        rounds=500, verbose=True
    )
    
    print(f"\nFinal Accuracy: {result2['final_accuracy']:.1%}")
    
    print("\nPer-Label Accuracy:")
    for label, acc in result2['per_label_accuracy'].items():
        count = result2['per_label_counts'][label]
        print(f"  {label:8s}: {acc:.1%} ({count} samples)")
