"""
Pattern Discovery Learner

A principled approach to few-shot learning that:
1. Systematically enumerates ALL possible k-position patterns
2. Tracks which patterns correlate with which labels
3. Discovers high-confidence rules through observation
4. NO embedded game knowledge - purely observational

Key insight: Instead of randomly generating rules, we systematically
track statistics for ALL possible patterns and discover which ones
are predictive.
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet
from collections import defaultdict
from itertools import combinations, product
from dataclasses import dataclass, field


@dataclass
class PatternStats:
    """Statistics for a specific pattern."""
    label_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_seen: int = 0
    
    def observe(self, label: int):
        self.label_counts[label] += 1
        self.total_seen += 1
    
    def confidence(self, label: int) -> float:
        if self.total_seen == 0:
            return 0.0
        return self.label_counts[label] / self.total_seen
    
    def best_label(self) -> Optional[int]:
        if not self.label_counts:
            return None
        return max(self.label_counts, key=self.label_counts.get)
    
    def is_deterministic(self, min_observations: int = 2) -> bool:
        """Returns True if pattern always predicts same label."""
        if self.total_seen < min_observations:
            return False
        return len([c for c in self.label_counts.values() if c > 0]) == 1


class PatternDiscoveryLearner:
    """
    Learner that discovers predictive patterns through systematic enumeration.
    
    Strategy:
    1. Track statistics for all observed k-position patterns
    2. Identify patterns that are strongly predictive (high confidence)
    3. Use discovered patterns for prediction
    4. Fall back to label prior when no patterns match
    
    NO CHEATING:
    - Doesn't know what positions mean
    - Doesn't know what values mean  
    - Doesn't know about rows/cols/diags
    - Only learns from (input, label) pairs
    """
    
    def __init__(self, num_outputs: int = 5, board_size: int = None,
                 label_names: List[str] = None, 
                 max_pattern_size: int = 4,
                 min_confidence: float = 0.95,
                 min_observations: int = 3,
                 **kwargs):
        self.num_outputs = num_outputs
        self.board_size = board_size
        self.label_names = label_names
        self.max_pattern_size = max_pattern_size
        self.min_confidence = min_confidence
        self.min_observations = min_observations
        
        # Pattern -> Stats mapping
        # Pattern is tuple of (position, value) pairs
        self.pattern_stats: Dict[FrozenSet[Tuple[int, str]], PatternStats] = defaultdict(PatternStats)
        
        # Discovered high-confidence rules
        # Maps pattern -> (label, confidence, support)
        self.discovered_rules: Dict[FrozenSet[Tuple[int, str]], Tuple[int, float, int]] = {}
        
        # Observations
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        self.label_counts: Dict[int, int] = defaultdict(int)
        
        # Track observed values
        self.observed_values: Set[str] = set()
        
        self.stats = {
            'patterns_tracked': 0,
            'rules_discovered': 0,
            'predictions': 0,
        }
    
    def _extract_patterns(self, board: str, max_size: int = None) -> List[FrozenSet[Tuple[int, str]]]:
        """Extract all k-position patterns from a board (k=1 to max_size)."""
        if max_size is None:
            max_size = self.max_pattern_size
        
        patterns = []
        positions = list(range(len(board)))
        
        for k in range(1, min(max_size + 1, len(board) + 1)):
            for pos_combo in combinations(positions, k):
                pattern = frozenset((pos, board[pos]) for pos in pos_combo)
                patterns.append(pattern)
        
        return patterns
    
    def _pattern_matches(self, pattern: FrozenSet[Tuple[int, str]], board: str) -> bool:
        """Check if pattern matches board."""
        for pos, val in pattern:
            if pos >= len(board) or board[pos] != val:
                return False
        return True
    
    def predict(self, observation: str) -> int:
        """Predict using discovered rules."""
        self.stats['predictions'] += 1
        
        # Infer structure
        if self.board_size is None:
            self.board_size = len(observation)
        for char in observation:
            self.observed_values.add(char)
        
        # Find matching discovered rules, prioritize by specificity and confidence
        matching_rules = []
        for pattern, (label, conf, support) in self.discovered_rules.items():
            if self._pattern_matches(pattern, observation):
                matching_rules.append((len(pattern), conf, support, label, pattern))
        
        if matching_rules:
            # Sort by specificity (desc), then confidence (desc), then support (desc)
            matching_rules.sort(key=lambda x: (-x[0], -x[1], -x[2]))
            
            # Weighted voting among top matches
            votes: Dict[int, float] = defaultdict(float)
            for spec, conf, sup, label, pattern in matching_rules[:20]:
                weight = spec * conf * (sup + 1)
                votes[label] += weight
            
            if votes:
                return max(votes, key=votes.get)
        
        # Fall back to prior
        return self._prior_predict()
    
    def _prior_predict(self) -> int:
        """Predict based on label distribution."""
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
        patterns = self._extract_patterns(observation)
        for pattern in patterns:
            self.pattern_stats[pattern].observe(correct_label)
        
        self.stats['patterns_tracked'] = len(self.pattern_stats)
        
        # Discover new rules periodically
        if len(self.observations) % 10 == 0:
            self._discover_rules()
    
    def _discover_rules(self):
        """Identify high-confidence patterns as rules."""
        new_discoveries = 0
        
        for pattern, stats in self.pattern_stats.items():
            if pattern in self.discovered_rules:
                continue
            
            if stats.total_seen < self.min_observations:
                continue
            
            best_label = stats.best_label()
            if best_label is None:
                continue
            
            conf = stats.confidence(best_label)
            if conf >= self.min_confidence:
                self.discovered_rules[pattern] = (best_label, conf, stats.total_seen)
                new_discoveries += 1
        
        self.stats['rules_discovered'] = len(self.discovered_rules)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'observations': len(self.observations),
            'rules': len(self.discovered_rules),
        }
    
    def describe_knowledge(self) -> str:
        """Describe discovered patterns."""
        lines = ["=== Pattern Discovery Learner ===\n"]
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Patterns tracked: {len(self.pattern_stats)}")
        lines.append(f"Rules discovered: {len(self.discovered_rules)}")
        lines.append(f"Observed values: {sorted(self.observed_values)}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for idx in range(self.num_outputs):
            count = self.label_counts.get(idx, 0)
            pct = count / total * 100 if total > 0 else 0
            label = self.label_names[idx] if self.label_names else str(idx)
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        
        # Group discovered rules by label
        lines.append("\n--- Discovered Rules by Label ---")
        rules_by_label: Dict[int, List] = defaultdict(list)
        for pattern, (label, conf, support) in self.discovered_rules.items():
            rules_by_label[label].append((pattern, conf, support))
        
        for label_idx in range(self.num_outputs):
            label_name = self.label_names[label_idx] if self.label_names else str(label_idx)
            rules = rules_by_label.get(label_idx, [])
            lines.append(f"\n  {label_name}: {len(rules)} rules")
            
            # Show top rules by specificity
            sorted_rules = sorted(rules, key=lambda x: (-len(x[0]), -x[1], -x[2]))
            for pattern, conf, support in sorted_rules[:5]:
                pattern_str = ', '.join(f"p{p}={v}" for p, v in sorted(pattern))
                lines.append(f"    [{conf:.0%}] {pattern_str} (n={support})")
        
        return '\n'.join(lines)
    
    def get_rules_for_label(self, label_idx: int) -> List[Tuple[FrozenSet, float, int]]:
        """Get all discovered rules for a specific label."""
        rules = []
        for pattern, (label, conf, support) in self.discovered_rules.items():
            if label == label_idx:
                rules.append((pattern, conf, support))
        return sorted(rules, key=lambda x: (-len(x[0]), -x[1], -x[2]))


class AdaptivePatternLearner(PatternDiscoveryLearner):
    """
    Enhanced pattern learner that adapts to label rarity.
    
    Key improvements:
    - Lower confidence threshold for rare labels
    - More aggressive pattern tracking for rare labels
    - Stratified sampling awareness
    """
    
    def __init__(self, rare_threshold: float = 0.15, 
                 rare_min_confidence: float = 0.85,
                 **kwargs):
        super().__init__(**kwargs)
        self.rare_threshold = rare_threshold
        self.rare_min_confidence = rare_min_confidence
    
    def _discover_rules(self):
        """Discover rules with adaptive thresholds for rare labels."""
        total_obs = sum(self.label_counts.values())
        if total_obs == 0:
            return
        
        # Calculate label frequencies
        label_freq = {label: count / total_obs 
                     for label, count in self.label_counts.items()}
        
        new_discoveries = 0
        
        for pattern, stats in self.pattern_stats.items():
            if pattern in self.discovered_rules:
                continue
            
            best_label = stats.best_label()
            if best_label is None:
                continue
            
            # Adaptive thresholds based on label rarity
            is_rare = label_freq.get(best_label, 0) < self.rare_threshold
            
            min_obs = 2 if is_rare else self.min_observations
            min_conf = self.rare_min_confidence if is_rare else self.min_confidence
            
            if stats.total_seen < min_obs:
                continue
            
            conf = stats.confidence(best_label)
            if conf >= min_conf:
                self.discovered_rules[pattern] = (best_label, conf, stats.total_seen)
                new_discoveries += 1
        
        self.stats['rules_discovered'] = len(self.discovered_rules)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    from test_harness import test_learner
    
    print("="*70)
    print("Testing Pattern Discovery Learner")
    print("="*70)
    
    result = test_learner(
        PatternDiscoveryLearner, tictactoe, random_board, label_space,
        rounds=500, verbose=True
    )
    
    print(f"\nFinal Accuracy: {result['final_accuracy']:.1%}")
    
    print("\nPer-Label Accuracy:")
    for label, acc in result['per_label_accuracy'].items():
        count = result['per_label_counts'][label]
        print(f"  {label:8s}: {acc:.1%} ({count} samples)")
    
    print("\n" + result['learner'].describe_knowledge())
    
    print("\n" + "="*70)
    print("Testing Adaptive Pattern Learner")
    print("="*70)
    
    result2 = test_learner(
        AdaptivePatternLearner, tictactoe, random_board, label_space,
        rounds=500, verbose=True
    )
    
    print(f"\nFinal Accuracy: {result2['final_accuracy']:.1%}")
    
    print("\nPer-Label Accuracy:")
    for label, acc in result2['per_label_accuracy'].items():
        count = result2['per_label_counts'][label]
        print(f"  {label:8s}: {acc:.1%} ({count} samples)")
