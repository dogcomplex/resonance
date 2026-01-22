"""
Hypothesis Elimination Learner

A principled few-shot learner based on strict hypothesis elimination.

Key insight: We maintain hypotheses of the form "pattern P implies label L"
and ELIMINATE any hypothesis that is contradicted by an observation.

This is theoretically sound: once we see a counterexample, the hypothesis is dead.

For prediction, we vote among surviving hypotheses that match the input.

NO CHEATING:
- No knowledge of rows/cols/diagonals
- No knowledge of what symbols mean
- No knowledge of what labels represent
- Purely learns from (input, label) pairs
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass


@dataclass
class Hypothesis:
    """
    A hypothesis: "If board matches pattern P, then label is L"
    
    Once contradicted (pattern matches but label differs), hypothesis is eliminated.
    """
    pattern: FrozenSet[Tuple[int, str]]  # Set of (position, value) constraints
    label: int
    support: int = 0  # Times pattern matched AND label was correct
    alive: bool = True
    
    def matches(self, board: str) -> bool:
        """Check if board matches this pattern."""
        for pos, val in self.pattern:
            if pos >= len(board) or board[pos] != val:
                return False
        return True
    
    def specificity(self) -> int:
        return len(self.pattern)
    
    def __hash__(self):
        return hash((self.pattern, self.label))


class HypothesisEliminationLearner:
    """
    Few-shot learner using strict hypothesis elimination.
    
    Algorithm:
    1. Generate hypotheses from observations
    2. STRICTLY eliminate any hypothesis contradicted by ANY observation
    3. Predict via weighted voting among surviving hypotheses
    
    Key properties:
    - Hypotheses that survive NEVER produce wrong predictions on seen data
    - Generalization comes from pattern specificity
    - More observations = fewer surviving hypotheses = better predictions
    """
    
    def __init__(self, num_outputs: int = 5, board_size: int = None,
                 label_names: List[str] = None,
                 max_hypotheses: int = 50000,
                 pattern_sizes: List[int] = None,
                 **kwargs):
        self.num_outputs = num_outputs
        self.board_size = board_size
        self.label_names = label_names
        self.max_hypotheses = max_hypotheses
        self.pattern_sizes = pattern_sizes or [2, 3, 4]
        
        # Hypothesis storage
        self.hypotheses: Dict[Tuple[FrozenSet, int], Hypothesis] = {}
        
        # Observations
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        self.label_counts: Dict[int, int] = defaultdict(int)
        
        # Track observed values
        self.observed_values: Set[str] = set()
        
        self.stats = {
            'generated': 0,
            'eliminated': 0,
            'predictions': 0,
        }
    
    def _generate_hypotheses_from_observation(self, board: str, label: int):
        """Generate hypotheses consistent with this observation."""
        generated = 0
        
        for size in self.pattern_sizes:
            if size > len(board):
                continue
            
            # Generate ALL size-k patterns from this board
            for positions in combinations(range(len(board)), size):
                pattern = frozenset((pos, board[pos]) for pos in positions)
                key = (pattern, label)
                
                if key not in self.hypotheses:
                    hyp = Hypothesis(pattern=pattern, label=label, support=1)
                    self.hypotheses[key] = hyp
                    generated += 1
                else:
                    # Hypothesis exists - increment support
                    self.hypotheses[key].support += 1
        
        self.stats['generated'] += generated
    
    def _eliminate_contradicted(self, board: str, true_label: int):
        """Eliminate hypotheses contradicted by this observation."""
        eliminated = 0
        
        for key, hyp in list(self.hypotheses.items()):
            if not hyp.alive:
                continue
            
            if hyp.matches(board) and hyp.label != true_label:
                # CONTRADICTION: pattern matches but wrong label
                hyp.alive = False
                eliminated += 1
        
        self.stats['eliminated'] += eliminated
        
        # Periodically clean up dead hypotheses to save memory
        if eliminated > 1000:
            self.hypotheses = {k: v for k, v in self.hypotheses.items() if v.alive}
    
    def predict(self, observation: str) -> int:
        """Predict using voting among matching surviving hypotheses."""
        self.stats['predictions'] += 1
        
        # Infer structure
        if self.board_size is None:
            self.board_size = len(observation)
        for char in observation:
            self.observed_values.add(char)
        
        # Collect votes from alive hypotheses that match
        votes: Dict[int, float] = defaultdict(float)
        
        for hyp in self.hypotheses.values():
            if hyp.alive and hyp.matches(observation):
                # Weight by specificity and support
                weight = (hyp.specificity() ** 2) * (hyp.support + 1)
                votes[hyp.label] += weight
        
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
        
        # First eliminate, then generate
        # Order matters: we don't want to generate hypotheses just to eliminate them
        self._eliminate_contradicted(observation, correct_label)
        self._generate_hypotheses_from_observation(observation, correct_label)
    
    def get_stats(self) -> Dict[str, Any]:
        alive = sum(1 for h in self.hypotheses.values() if h.alive)
        return {
            **self.stats,
            'hypotheses_alive': alive,
            'hypotheses_total': len(self.hypotheses),
            'observations': len(self.observations),
            'rules': alive,  # Compatibility
        }
    
    def describe_knowledge(self) -> str:
        """Describe surviving hypotheses."""
        lines = ["=== Hypothesis Elimination Learner ===\n"]
        
        alive = [h for h in self.hypotheses.values() if h.alive]
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Alive hypotheses: {len(alive)}")
        lines.append(f"Eliminated: {self.stats['eliminated']}")
        lines.append(f"Observed values: {sorted(self.observed_values)}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for idx in range(self.num_outputs):
            count = self.label_counts.get(idx, 0)
            pct = count / total * 100 if total > 0 else 0
            label = self.label_names[idx] if self.label_names else str(idx)
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        
        # Group by label
        lines.append("\n--- Surviving Hypotheses by Label ---")
        by_label: Dict[int, List[Hypothesis]] = defaultdict(list)
        for hyp in alive:
            by_label[hyp.label].append(hyp)
        
        for label_idx in range(self.num_outputs):
            label_name = self.label_names[label_idx] if self.label_names else str(label_idx)
            hyps = by_label.get(label_idx, [])
            lines.append(f"\n  {label_name}: {len(hyps)} hypotheses")
            
            # Show top by support
            sorted_hyps = sorted(hyps, key=lambda h: (-h.specificity(), -h.support))
            for hyp in sorted_hyps[:5]:
                pattern_str = ', '.join(f"p{p}={v}" for p, v in sorted(hyp.pattern))
                lines.append(f"    [{hyp.specificity()}] {pattern_str} (sup={hyp.support})")
        
        return '\n'.join(lines)
    
    def get_surviving_patterns_for_label(self, label_idx: int) -> List[Hypothesis]:
        """Get all surviving hypotheses for a label."""
        return [h for h in self.hypotheses.values() 
                if h.alive and h.label == label_idx]


class FocusedHypothesisLearner(HypothesisEliminationLearner):
    """
    Enhanced hypothesis learner that focuses on rare labels.
    
    Key insight: For imbalanced data, we need more hypotheses for rare labels
    because they have fewer observations to learn from.
    """
    
    def __init__(self, 
                 rare_label_boost: int = 5,
                 focus_pattern_sizes: List[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.rare_label_boost = rare_label_boost
        self.focus_pattern_sizes = focus_pattern_sizes or [3]  # Focus on 3-patterns for wins
    
    def _generate_hypotheses_from_observation(self, board: str, label: int):
        """Generate hypotheses with extra focus on rare labels."""
        # Standard generation
        super()._generate_hypotheses_from_observation(board, label)
        
        # Extra generation for rare labels
        total = sum(self.label_counts.values())
        if total == 0:
            return
        
        freq = self.label_counts[label] / total
        if freq < 0.2:  # Rare label
            # Generate extra hypotheses with focused pattern sizes
            for size in self.focus_pattern_sizes:
                if size > len(board):
                    continue
                
                # Generate ALL patterns at this size
                for positions in combinations(range(len(board)), size):
                    pattern = frozenset((pos, board[pos]) for pos in positions)
                    key = (pattern, label)
                    
                    if key not in self.hypotheses:
                        hyp = Hypothesis(pattern=pattern, label=label, support=1)
                        self.hypotheses[key] = hyp


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    from test_harness import test_learner
    
    print("="*70)
    print("Testing Hypothesis Elimination Learner")
    print("="*70)
    
    result = test_learner(
        HypothesisEliminationLearner, tictactoe, random_board, label_space,
        rounds=500, verbose=True, pattern_sizes=[3]
    )
    
    print(f"\nFinal Accuracy: {result['final_accuracy']:.1%}")
    
    print("\nPer-Label Accuracy:")
    for label, acc in result['per_label_accuracy'].items():
        count = result['per_label_counts'][label]
        print(f"  {label:8s}: {acc:.1%} ({count} samples)")
    
    # Show surviving win patterns
    learner = result['learner']
    print("\n--- Win1 Patterns (3-position, high support) ---")
    win1_hyps = learner.get_surviving_patterns_for_label(1)
    win1_3 = [h for h in win1_hyps if h.specificity() == 3]
    for hyp in sorted(win1_3, key=lambda h: -h.support)[:10]:
        pattern_str = ', '.join(f"p{p}={v}" for p, v in sorted(hyp.pattern))
        print(f"  {pattern_str} (support={hyp.support})")
    
    print("\n" + "="*70)
    print("Testing Focused Hypothesis Learner")
    print("="*70)
    
    result2 = test_learner(
        FocusedHypothesisLearner, tictactoe, random_board, label_space,
        rounds=500, verbose=True, pattern_sizes=[2, 3, 4], focus_pattern_sizes=[3]
    )
    
    print(f"\nFinal Accuracy: {result2['final_accuracy']:.1%}")
    
    print("\nPer-Label Accuracy:")
    for label, acc in result2['per_label_accuracy'].items():
        count = result2['per_label_counts'][label]
        print(f"  {label:8s}: {acc:.1%} ({count} samples)")
