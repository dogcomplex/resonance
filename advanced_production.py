"""
Advanced Production Rule Learner

Based on the LHS => RHS token framework:
- LHS: Required conditions (consumed or checked)
- RHS: Productions (output label)
- Tokens on both sides: Catalysts/conditionals

Key features:
1. Strict hypothesis elimination (contradicted rules are dead)
2. Confidence-based phase transitions (not just observation count)
3. Rule confidence tracking
4. Discriminative rule scoring

NO CHEATING - purely observational learning.
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet
from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field
from enum import Enum


class LearnerPhase(Enum):
    """Learning phases based on rule confidence."""
    EXPLORATION = "exploration"      # Building hypotheses
    REFINEMENT = "refinement"        # Eliminating bad rules
    EXPLOITATION = "exploitation"    # Using confident rules


@dataclass(frozen=True)
class Condition:
    """A condition: position has value."""
    position: int
    value: str
    
    def check(self, board: str) -> bool:
        return self.position < len(board) and board[self.position] == self.value
    
    def __str__(self):
        return f"p{self.position}={self.value}"


@dataclass
class ProductionRule:
    """
    Production rule: IF conditions THEN label
    
    LHS = frozenset of Conditions (all must be true)
    RHS = output label index
    """
    lhs: FrozenSet[Condition]
    rhs: int
    
    # Statistics
    times_matched: int = 0
    times_correct: int = 0
    times_wrong: int = 0
    
    # Status
    alive: bool = True
    
    def matches(self, board: str) -> bool:
        """Check if all conditions are satisfied."""
        return all(c.check(board) for c in self.lhs)
    
    def fire(self, correct_label: int) -> bool:
        """
        Fire the rule and record result.
        Returns True if rule was correct.
        """
        self.times_matched += 1
        if self.rhs == correct_label:
            self.times_correct += 1
            return True
        else:
            self.times_wrong += 1
            return False
    
    def confidence(self) -> float:
        """Confidence = correct / matched."""
        if self.times_matched == 0:
            return 0.5  # Prior
        return self.times_correct / self.times_matched
    
    def is_pure(self) -> bool:
        """Pure rule = never wrong."""
        return self.times_matched >= 2 and self.times_wrong == 0
    
    def specificity(self) -> int:
        return len(self.lhs)
    
    def __hash__(self):
        return hash((self.lhs, self.rhs))
    
    def signature(self) -> Tuple:
        return (self.lhs, self.rhs)


class AdvancedProductionLearner:
    """
    Advanced production rule learner with:
    1. Strict hypothesis elimination
    2. Confidence-based phase transitions
    3. Pure rule tracking
    4. Discriminative scoring
    """
    
    def __init__(self, num_outputs: int = 5, board_size: int = None,
                 label_names: List[str] = None,
                 pattern_sizes: List[int] = None,
                 # Phase transition thresholds
                 exploration_until_confidence: float = 0.7,
                 refinement_until_confidence: float = 0.9,
                 **kwargs):
        self.num_outputs = num_outputs
        self.board_size = board_size
        self.label_names = label_names
        self.pattern_sizes = pattern_sizes or [3]
        
        self.exploration_threshold = exploration_until_confidence
        self.refinement_threshold = refinement_until_confidence
        
        # Rule storage
        self.rules: Dict[Tuple, ProductionRule] = {}
        
        # Pure rules by label (high confidence)
        self.pure_rules: Dict[int, List[ProductionRule]] = defaultdict(list)
        
        # Observations
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        self.label_counts: Dict[int, int] = defaultdict(int)
        
        # Tracked values
        self.observed_values: Set[str] = set()
        
        # Phase tracking
        self.current_phase = LearnerPhase.EXPLORATION
        self.phase_history: List[Tuple[int, LearnerPhase]] = []
        
        # Stats
        self.stats = {
            'generated': 0,
            'eliminated': 0,
            'predictions': 0,
            'phase_changes': 0,
        }
    
    def _board_to_conditions(self, board: str) -> Set[Condition]:
        """Convert board to set of conditions."""
        return {Condition(pos, val) for pos, val in enumerate(board)}
    
    def _generate_rules(self, board: str, label: int):
        """Generate production rules from observation."""
        conditions = list(self._board_to_conditions(board))
        
        for size in self.pattern_sizes:
            if size > len(conditions):
                continue
            
            for combo in combinations(conditions, size):
                lhs = frozenset(combo)
                sig = (lhs, label)
                
                if sig not in self.rules:
                    rule = ProductionRule(lhs=lhs, rhs=label)
                    self.rules[sig] = rule
                    self.stats['generated'] += 1
    
    def _eliminate_contradicted(self, board: str, label: int):
        """Eliminate rules that match but predict wrong label."""
        eliminated = 0
        
        for rule in self.rules.values():
            if not rule.alive:
                continue
            
            if rule.matches(board):
                was_correct = rule.fire(label)
                if not was_correct:
                    # STRICT ELIMINATION: one wrong = dead
                    rule.alive = False
                    eliminated += 1
        
        self.stats['eliminated'] += eliminated
    
    def _update_pure_rules(self):
        """Update the set of pure (never-wrong) rules."""
        self.pure_rules.clear()
        
        for rule in self.rules.values():
            if rule.alive and rule.is_pure():
                self.pure_rules[rule.rhs].append(rule)
        
        # Sort by specificity (higher = better) then support
        for label in self.pure_rules:
            self.pure_rules[label].sort(
                key=lambda r: (-r.specificity(), -r.times_correct)
            )
    
    def _compute_overall_confidence(self) -> float:
        """Compute overall learner confidence."""
        alive_rules = [r for r in self.rules.values() if r.alive]
        if not alive_rules:
            return 0.0
        
        # Weight by times matched
        total_matched = sum(r.times_matched for r in alive_rules)
        if total_matched == 0:
            return 0.5
        
        weighted_conf = sum(
            r.confidence() * r.times_matched 
            for r in alive_rules
        ) / total_matched
        
        return weighted_conf
    
    def _update_phase(self):
        """Update learning phase based on confidence."""
        conf = self._compute_overall_confidence()
        
        old_phase = self.current_phase
        
        if conf < self.exploration_threshold:
            self.current_phase = LearnerPhase.EXPLORATION
        elif conf < self.refinement_threshold:
            self.current_phase = LearnerPhase.REFINEMENT
        else:
            self.current_phase = LearnerPhase.EXPLOITATION
        
        if self.current_phase != old_phase:
            self.phase_history.append((len(self.observations), self.current_phase))
            self.stats['phase_changes'] += 1
    
    def predict(self, observation: str) -> int:
        """Predict using phase-appropriate strategy."""
        self.stats['predictions'] += 1
        
        # Infer structure
        if self.board_size is None:
            self.board_size = len(observation)
        for char in observation:
            self.observed_values.add(char)
        
        # Compute label rarity for boosting
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts.get(l, 0) / total 
                     for l in range(self.num_outputs)}
        
        # Strategy based on phase
        if self.current_phase == LearnerPhase.EXPLOITATION:
            return self._predict_exploitation(observation, label_freq)
        elif self.current_phase == LearnerPhase.REFINEMENT:
            return self._predict_refinement(observation, label_freq)
        else:
            return self._predict_exploration(observation, label_freq)
    
    def _predict_exploration(self, board: str, label_freq: Dict) -> int:
        """Exploration: prior-based with any high-confidence rules."""
        # Check pure rules, rare labels first
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for rule in self.pure_rules.get(label, []):
                if rule.matches(board):
                    return label
        
        # Fall back to prior
        return self._prior_predict()
    
    def _predict_refinement(self, board: str, label_freq: Dict) -> int:
        """Refinement: weighted voting with rare label boost."""
        # Check pure rules first
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for rule in self.pure_rules.get(label, []):
                if rule.matches(board):
                    return label
        
        # Weighted voting among all alive matching rules
        votes: Dict[int, float] = defaultdict(float)
        
        for rule in self.rules.values():
            if not rule.alive or not rule.matches(board):
                continue
            
            rarity_boost = 1.0 / (label_freq.get(rule.rhs, 0.1) + 0.01)
            weight = rule.confidence() * rule.specificity() * rarity_boost
            votes[rule.rhs] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        return self._prior_predict()
    
    def _predict_exploitation(self, board: str, label_freq: Dict) -> int:
        """Exploitation: use most confident matching rules."""
        # Pure rules are king
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for rule in self.pure_rules.get(label, []):
                if rule.matches(board) and rule.times_correct >= 3:
                    return label
        
        # Find best matching rule by confidence
        best_rule = None
        best_score = -1
        
        for rule in self.rules.values():
            if not rule.alive or not rule.matches(board):
                continue
            
            rarity_boost = 1.0 / (label_freq.get(rule.rhs, 0.1) + 0.01)
            score = rule.confidence() * (rule.specificity() ** 2) * rarity_boost
            
            if score > best_score:
                best_score = score
                best_rule = rule
        
        if best_rule:
            return best_rule.rhs
        
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
        
        # Eliminate first, then generate
        self._eliminate_contradicted(observation, correct_label)
        self._generate_rules(observation, correct_label)
        
        # Update pure rules and phase
        if len(self.observations) % 10 == 0:
            self._update_pure_rules()
            self._update_phase()
    
    def get_stats(self) -> Dict[str, Any]:
        alive = sum(1 for r in self.rules.values() if r.alive)
        pure_count = sum(len(rules) for rules in self.pure_rules.values())
        
        return {
            **self.stats,
            'rules': alive,
            'pure_rules': pure_count,
            'observations': len(self.observations),
            'phase': self.current_phase.value,
            'confidence': self._compute_overall_confidence(),
        }
    
    def describe_knowledge(self) -> str:
        """Describe learned rules."""
        lines = ["=== Advanced Production Learner ===\n"]
        
        alive = [r for r in self.rules.values() if r.alive]
        pure_count = sum(len(rules) for rules in self.pure_rules.values())
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Alive rules: {len(alive)}")
        lines.append(f"Pure rules: {pure_count}")
        lines.append(f"Phase: {self.current_phase.value}")
        lines.append(f"Overall confidence: {self._compute_overall_confidence():.1%}")
        
        lines.append("\n--- Phase History ---")
        for obs_num, phase in self.phase_history:
            lines.append(f"  @{obs_num}: {phase.value}")
        
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
            
            for rule in rules[:8]:
                lhs_str = ' AND '.join(str(c) for c in sorted(rule.lhs, key=str))
                lines.append(f"    IF {lhs_str} => {label_name} (n={rule.times_correct}, conf={rule.confidence():.0%})")
        
        return '\n'.join(lines)
    
    def get_discovered_rules(self) -> Dict[int, List[ProductionRule]]:
        """Get all pure rules organized by label."""
        return dict(self.pure_rules)


class MultiPatternProductionLearner(AdvancedProductionLearner):
    """Extended with multiple pattern sizes."""
    
    def __init__(self, **kwargs):
        kwargs['pattern_sizes'] = [2, 3, 4]
        super().__init__(**kwargs)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE
    
    print("="*70)
    print("Testing Advanced Production Learner")
    print("="*70)
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    learner = AdvancedProductionLearner(
        num_outputs=5, 
        label_names=LABEL_SPACE
    )
    
    correct = 0
    checkpoints = [10, 25, 50, 100, 200, 500, 1000]
    
    for i in range(1000):
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
            stats = learner.get_stats()
            acc = correct / (i + 1)
            print(f"  R{i+1:4d}: {acc:.1%} | phase={stats['phase'][:4]} conf={stats['confidence']:.0%} rules={stats['rules']} pure={stats['pure_rules']}")
    
    print(f"\nFinal Accuracy: {correct/min(1000, i+1):.1%}")
    print(f"Coverage: {gen.coverage():.1%}")
    print("\n" + learner.describe_knowledge())
