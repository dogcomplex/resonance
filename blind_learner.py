"""
Truly Blind Few-Shot Learner

This learner has NO embedded game knowledge:
- Doesn't know about rows/cols/diagonals
- Doesn't know what '0', '1', '2' mean
- Doesn't know what labels represent
- Only learns from observations through hypothesis elimination

The goal is honest few-shot learning without cheating.
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations


@dataclass
class Rule:
    """A rule mapping input pattern to output label."""
    pattern: Dict[int, str]  # position -> required value
    output: int  # output label index
    confidence: float = 0.5
    support: int = 0
    
    def matches(self, board: str) -> bool:
        """Check if observation matches this rule's pattern."""
        for pos, val in self.pattern.items():
            if pos >= len(board) or board[pos] != val:
                return False
        return True
    
    def specificity(self) -> int:
        """More conditions = more specific."""
        return len(self.pattern)
    
    def signature(self) -> tuple:
        return (tuple(sorted(self.pattern.items())), self.output)


class BlindLearner:
    """
    Truly blind few-shot learner.
    
    Only knows:
    - Input length (inferred from first observation)
    - Number of possible output labels
    
    Does NOT know:
    - What input characters mean
    - What labels represent
    - Any structural patterns (lines, etc.)
    - Any game-specific semantics
    """
    
    def __init__(self, num_outputs: int = 5, board_size: int = None, 
                 label_names: List[str] = None, max_rules: int = 5000,
                 rules_per_obs: int = 50, **kwargs):
        self.num_outputs = num_outputs
        self.board_size = board_size  # Will be inferred if None
        self.label_names = label_names  # Only for reporting
        self.max_rules = max_rules
        self.rules_per_obs = rules_per_obs
        
        self.rules: List[Rule] = []
        self.rule_signatures: Set[tuple] = set()
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        
        # Track observed values at each position
        self.observed_values: Set[str] = set()
        self.label_counts: Dict[int, int] = defaultdict(int)
        
        self.stats = {
            'generated': 0,
            'eliminated': 0,
            'predictions': 0,
        }
    
    def predict(self, observation: str) -> int:
        """Predict using only learned rules - no structural assumptions."""
        self.stats['predictions'] += 1
        
        # Infer board size from first observation
        if self.board_size is None:
            self.board_size = len(observation)
        
        # Track observed values
        for char in observation:
            self.observed_values.add(char)
        
        if not self.rules:
            return self._prior_predict()
        
        # Find matching rules
        matching = [r for r in self.rules if r.matches(observation)]
        
        if not matching:
            return self._prior_predict()
        
        # Weighted voting among matching rules
        votes: Dict[int, float] = defaultdict(float)
        for rule in matching:
            weight = (rule.specificity() ** 2) * rule.confidence * (rule.support + 1)
            votes[rule.output] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        return self._prior_predict()
    
    def _prior_predict(self) -> int:
        """Predict based on observed label distribution."""
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
        
        # Generate rules from this observation
        self._generate_rules(observation, correct_label)
        
        # Eliminate contradicted rules
        self._eliminate_contradicted(observation, correct_label)
        
        # Update confidences periodically
        if len(self.observations) % 25 == 0:
            self._update_confidences()
        
        # Prune if needed
        if len(self.rules) > self.max_rules:
            self._prune()
    
    def _generate_rules(self, board: str, label: int):
        """Generate rules from observation - no structural bias."""
        generated = 0
        
        # Generate rules of varying specificity
        for specificity in range(1, min(7, len(board) + 1)):
            attempts = 0
            while generated < self.rules_per_obs and attempts < self.rules_per_obs * 3:
                attempts += 1
                
                # Random positions
                positions = random.sample(range(len(board)), specificity)
                pattern = {pos: board[pos] for pos in positions}
                
                rule = Rule(pattern=pattern, output=label, support=1)
                sig = rule.signature()
                
                if sig not in self.rule_signatures:
                    self.rule_signatures.add(sig)
                    self.rules.append(rule)
                    generated += 1
                    self.stats['generated'] += 1
    
    def _eliminate_contradicted(self, board: str, label: int):
        """Remove rules contradicted by observation."""
        valid = []
        eliminated = 0
        
        for rule in self.rules:
            if rule.matches(board) and rule.output != label:
                eliminated += 1
                self.rule_signatures.discard(rule.signature())
            else:
                valid.append(rule)
        
        self.rules = valid
        self.stats['eliminated'] += eliminated
    
    def _update_confidences(self):
        """Update rule confidences based on recent observations."""
        recent = self.observations[-100:]
        
        for rule in self.rules:
            support = 0
            applicable = 0
            
            for board, label in recent:
                if rule.matches(board):
                    applicable += 1
                    if rule.output == label:
                        support += 1
            
            rule.support = support
            rule.confidence = support / applicable if applicable > 0 else 0.5
    
    def _prune(self):
        """Keep best rules."""
        for rule in self.rules:
            rule._score = (
                rule.specificity() * 5 +
                rule.confidence * 10 +
                min(rule.support, 10)
            )
        
        self.rules.sort(key=lambda r: -r._score)
        self.rules = self.rules[:self.max_rules]
        self.rule_signatures = {r.signature() for r in self.rules}
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'rules': len(self.rules),
            'observations': len(self.observations),
            'unique_values': len(self.observed_values),
        }
    
    def describe_knowledge(self) -> str:
        """Describe what the learner has discovered."""
        lines = ["=== Blind Learner Knowledge ===\n"]
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Rules: {len(self.rules)}")
        lines.append(f"Observed values: {sorted(self.observed_values)}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for idx in range(self.num_outputs):
            count = self.label_counts[idx]
            pct = count / total * 100 if total > 0 else 0
            label = self.label_names[idx] if self.label_names and idx < len(self.label_names) else str(idx)
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        
        # Show top rules by confidence
        lines.append("\n--- Top Rules (confidence >= 0.8, support >= 2) ---")
        sorted_rules = sorted(self.rules, key=lambda r: (-r.confidence, -r.support))
        shown = 0
        for rule in sorted_rules:
            if rule.confidence >= 0.8 and rule.support >= 2 and shown < 20:
                label = self.label_names[rule.output] if self.label_names else str(rule.output)
                pattern_str = ', '.join(f"p{k}={v}" for k, v in sorted(rule.pattern.items()))
                lines.append(f"  [{rule.confidence:.2f}] {pattern_str} => {label} (x{rule.support})")
                shown += 1
        
        return '\n'.join(lines)
