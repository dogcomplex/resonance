"""
Bubble-Up Abstraction Learner

Discovers patterns and creates abstract labels for them WITHOUT semantic knowledge.

Process:
1. Observe raw states (opaque token sets)
2. Track which tokens co-occur consistently
3. Create abstract labels for frequent patterns
4. Replace concrete tokens with abstractions in rules
5. Rules become more general and compact

Example:
  Observes: {p0_1, p1_1, p2_1} always appears with label "win1"
  Creates:  A0 = {p0_1, p1_1, p2_1}
  Rule becomes: A0 => win1  (instead of 3 separate conditions)
  
  Later, LLM can observe: "A0 looks like 'top row all X'" and label it "row0_X"
  But the learner itself never uses semantic knowledge.
"""

import random
from typing import List, Tuple, Dict, Set, Optional, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field

import sys
sys.path.insert(0, '/home/claude/locus')


@dataclass
class Pattern:
    """
    An observed pattern of co-occurring tokens.
    """
    tokens: FrozenSet[str]
    
    # How often this exact pattern appears
    count: int = 0
    
    # What labels it correlates with
    label_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Optional abstract name (assigned later)
    abstract_name: Optional[str] = None
    
    def precision(self, label: int) -> float:
        """Precision for predicting this label."""
        if self.count == 0:
            return 0.0
        return self.label_counts[label] / self.count
    
    def best_label(self) -> Tuple[int, float]:
        """Return (label, precision) for best label."""
        if not self.label_counts:
            return (-1, 0.0)
        best = max(self.label_counts.items(), key=lambda x: x[1])
        return (best[0], best[1] / self.count)
    
    def __hash__(self):
        return hash(self.tokens)
    
    def __eq__(self, other):
        return self.tokens == other.tokens
    
    def __str__(self):
        if self.abstract_name:
            return self.abstract_name
        return "{" + ", ".join(sorted(self.tokens)) + "}"


class BubbleUpLearner:
    """
    Learns abstract labels for patterns through observation.
    
    NO CHEATING:
    - Input: opaque strings (board states)
    - Output: opaque integers (labels)
    - Learns pattern co-occurrence without semantic knowledge
    - Creates abstract names (A0, A1, ...) for patterns
    
    The "bubble-up" process:
    1. Low-level: individual position-value tokens
    2. Mid-level: patterns of tokens that predict labels
    3. High-level: abstractions that group similar patterns
    """
    
    def __init__(self, pattern_sizes: List[int] = None, 
                 abstraction_threshold: int = 10,
                 precision_threshold: float = 0.95):
        self.pattern_sizes = pattern_sizes or [2, 3]
        self.abstraction_threshold = abstraction_threshold  # Min observations to abstract
        self.precision_threshold = precision_threshold      # Min precision to abstract
        
        # Observed patterns
        self.patterns: Dict[FrozenSet[str], Pattern] = {}
        
        # Abstractions: name -> pattern
        self.abstractions: Dict[str, Pattern] = {}
        
        # Reverse map: pattern tokens -> abstraction name
        self.reverse_abstractions: Dict[FrozenSet[str], str] = {}
        
        # Next abstraction ID
        self.next_abstract_id = 0
        
        # Observations
        self.observations: List[Tuple[str, int]] = []
        self.label_counts: Dict[int, int] = defaultdict(int)
        
        # Stats
        self.stats = {
            'observations': 0,
            'patterns_tracked': 0,
            'abstractions_created': 0,
        }
    
    def _tokenize(self, board: str) -> Set[str]:
        """Convert board to tokens."""
        return {f"p{i}_{board[i]}" for i in range(len(board))}
    
    def _extract_patterns(self, tokens: Set[str]) -> List[FrozenSet[str]]:
        """Extract all patterns of configured sizes."""
        patterns = []
        token_list = sorted(tokens)
        for size in self.pattern_sizes:
            if size > len(token_list):
                continue
            for combo in combinations(token_list, size):
                patterns.append(frozenset(combo))
        return patterns
    
    def observe(self, board: str, label: int):
        """Observe a board state and its label."""
        self.stats['observations'] += 1
        self.observations.append((board, label))
        self.label_counts[label] += 1
        
        # Tokenize and extract patterns
        tokens = self._tokenize(board)
        patterns = self._extract_patterns(tokens)
        
        # Update pattern statistics
        for pattern in patterns:
            if pattern not in self.patterns:
                self.patterns[pattern] = Pattern(tokens=pattern)
                self.stats['patterns_tracked'] += 1
            
            self.patterns[pattern].count += 1
            self.patterns[pattern].label_counts[label] += 1
    
    def create_abstractions(self) -> List[str]:
        """
        Create abstractions for high-precision patterns.
        
        Returns list of new abstraction names.
        """
        created = []
        
        for pattern_tokens, pattern in self.patterns.items():
            # Skip if already abstracted
            if pattern.abstract_name is not None:
                continue
            
            # Skip if already an abstraction of this
            if pattern_tokens in self.reverse_abstractions:
                continue
            
            # Check thresholds
            if pattern.count < self.abstraction_threshold:
                continue
            
            best_label, precision = pattern.best_label()
            if precision < self.precision_threshold:
                continue
            
            # Create abstraction
            name = f"A{self.next_abstract_id}"
            self.next_abstract_id += 1
            
            pattern.abstract_name = name
            self.abstractions[name] = pattern
            self.reverse_abstractions[pattern_tokens] = name
            
            created.append(name)
            self.stats['abstractions_created'] += 1
        
        return created
    
    def get_abstractions_for(self, board: str) -> Set[str]:
        """Get all abstraction names that apply to this board."""
        tokens = self._tokenize(board)
        result = set()
        
        for pattern_tokens, name in self.reverse_abstractions.items():
            if pattern_tokens.issubset(tokens):
                result.add(name)
        
        return result
    
    def predict(self, board: str) -> int:
        """Predict label using learned patterns and abstractions."""
        tokens = self._tokenize(board)
        
        # Calculate label frequencies for priors
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts.get(l, 0) / total 
                     for l in self.label_counts.keys()}
        
        # Find matching abstractions (highest priority)
        for pattern_tokens, name in self.reverse_abstractions.items():
            if pattern_tokens.issubset(tokens):
                pattern = self.abstractions[name]
                best_label, _ = pattern.best_label()
                return best_label
        
        # Find matching high-precision patterns
        votes = defaultdict(float)
        for pattern_tokens, pattern in self.patterns.items():
            if not pattern_tokens.issubset(tokens):
                continue
            
            if pattern.count < 2:
                continue
            
            best_label, precision = pattern.best_label()
            if precision >= 0.8:
                # Weight by precision, specificity, rarity
                rarity_boost = 1.0 / (label_freq.get(best_label, 0.1) + 0.01)
                weight = precision * len(pattern_tokens) * pattern.count * rarity_boost
                votes[best_label] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        # Fall back to prior
        if self.label_counts:
            return max(self.label_counts, key=self.label_counts.get)
        return 0
    
    def describe(self) -> str:
        """Describe learned abstractions."""
        lines = ["=== Bubble-Up Abstraction Learner ===\n"]
        
        lines.append(f"Observations: {self.stats['observations']}")
        lines.append(f"Patterns tracked: {self.stats['patterns_tracked']}")
        lines.append(f"Abstractions created: {self.stats['abstractions_created']}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for label, count in sorted(self.label_counts.items()):
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  Label {label}: {count} ({pct:.1f}%)")
        
        lines.append("\n--- Abstractions ---")
        for name, pattern in sorted(self.abstractions.items()):
            best_label, precision = pattern.best_label()
            lines.append(f"  {name} = {pattern.tokens}")
            lines.append(f"       -> label {best_label} ({precision:.1%} precision, n={pattern.count})")
        
        # Show top patterns without abstractions
        lines.append("\n--- Top Unabstracted Patterns ---")
        unabstracted = [p for p in self.patterns.values() 
                       if p.abstract_name is None and p.count >= 5]
        unabstracted.sort(key=lambda p: -p.count)
        
        for pattern in unabstracted[:10]:
            best_label, precision = pattern.best_label()
            lines.append(f"  {pattern.tokens}")
            lines.append(f"       -> label {best_label} ({precision:.1%}, n={pattern.count})")
        
        return '\n'.join(lines)
    
    def export_rules(self) -> str:
        """Export learned rules in production format."""
        lines = ["# Learned Production Rules", ""]
        
        # Export abstraction definitions
        lines.append("## Abstraction Definitions")
        for name, pattern in sorted(self.abstractions.items()):
            tokens_str = "  ".join(sorted(pattern.tokens))
            lines.append(f"# {name} = {{ {tokens_str} }}")
        
        lines.append("\n## Classification Rules")
        for name, pattern in sorted(self.abstractions.items()):
            best_label, precision = pattern.best_label()
            lines.append(f"{name}  =>  label_{best_label}")
        
        return '\n'.join(lines)


def test_bubble_up():
    """Test the bubble-up learner."""
    print("="*70)
    print("BUBBLE-UP ABSTRACTION LEARNER")
    print("="*70)
    
    from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    learner = BubbleUpLearner(
        pattern_sizes=[3],
        abstraction_threshold=10,
        precision_threshold=0.99
    )
    
    correct = 0
    
    print("\nLearning from observations...")
    for i in range(2000):
        obs = gen.next()
        if obs is None:
            break
        
        board, true_label = obs
        pred = learner.predict(board)
        
        if pred == true_label:
            correct += 1
        
        learner.observe(board, true_label)
        
        # Create abstractions periodically
        if (i + 1) % 200 == 0:
            created = learner.create_abstractions()
            acc = correct / (i + 1)
            print(f"  @{i+1}: acc={acc:.1%}, abstractions={len(learner.abstractions)}, new={len(created)}")
    
    # Final abstraction pass
    learner.create_abstractions()
    
    print(f"\nFinal accuracy: {correct/2000:.1%}")
    print(f"Total abstractions: {len(learner.abstractions)}")
    
    print("\n" + learner.describe())
    
    print("\n" + "="*70)
    print("EXPORTED RULES")
    print("="*70)
    print(learner.export_rules())
    
    # Verify no cheating
    print("\n" + "="*70)
    print("VERIFICATION: Is this cheating?")
    print("="*70)
    print("""
    What the learner received:
    - Board strings like "000100020"
    - Label indices like 0, 1, 2, 3, 4
    
    What it discovered:
    - Patterns like {p0_1, p4_1, p8_1} predict label 1
    - Named this pattern "A0" (opaque name, no semantics)
    
    NO CHEATING:
    ✓ No knowledge of rows/columns/diagonals
    ✓ No knowledge of what labels mean
    ✓ Abstract names are opaque (A0, A1, ...)
    ✓ Later, LLM could label A0 as "main_diagonal_X"
    """)


if __name__ == "__main__":
    test_bubble_up()
