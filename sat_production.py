"""
Production Rule System with SAT-like Inference

Core concept: Rules are token transformations
  LHS => RHS
  
Where:
- LHS: Required tokens (conditions that must be present)
- RHS: Produced tokens (conclusions/outputs)
- Catalyst: Token on BOTH sides (checked but not consumed)

This enables:
1. Simple classification: IF pattern THEN label
2. Chained inference: IF A THEN B, IF B THEN C => IF A THEN C
3. Boolean combinations: Multiple rules can fire, votes combine
4. Negative inference: IF pattern THEN NOT label

SAT-like features:
- Each rule is a clause (disjunction of literals)
- Observations constrain which clauses can be true
- Consistency checking: no contradictions allowed
- Unit propagation: forced conclusions from observations

NO external SAT solver - pure Python implementation.
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet, Union
from collections import defaultdict
from itertools import combinations, product
from dataclasses import dataclass, field
from enum import Enum


class TokenType(Enum):
    """Types of tokens in the system."""
    POSITION = "position"      # Board position has value: p0=1
    LABEL = "label"            # Output label: label=win1
    META = "meta"              # Meta-info: board_full, has_empty
    DERIVED = "derived"        # Derived facts: three_in_row


@dataclass(frozen=True)
class Token:
    """
    A token represents a fact or condition.
    
    Examples:
    - Token("p0", "1") = position 0 has value 1
    - Token("label", 1) = output is label index 1
    - Token("board_full", True) = board has no empty spaces
    """
    name: str
    value: Any
    token_type: TokenType = TokenType.POSITION
    negated: bool = False
    
    def __str__(self):
        neg = "NOT " if self.negated else ""
        return f"{neg}{self.name}={self.value}"
    
    def negate(self) -> 'Token':
        """Return negated version of this token."""
        return Token(self.name, self.value, self.token_type, not self.negated)
    
    def matches(self, other: 'Token') -> bool:
        """Check if this token matches another (ignoring negation)."""
        return self.name == other.name and self.value == other.value


@dataclass
class ProductionRule:
    """
    A production rule: LHS => RHS
    
    Semantics:
    - If ALL tokens in LHS are satisfied, then RHS is concluded
    - Tokens can be negated (NOT p0=1 means position 0 is not 1)
    - Confidence tracks how often the rule has been correct
    """
    lhs: FrozenSet[Token]  # Conditions (all must be true)
    rhs: Token             # Conclusion
    
    # Statistics
    fires: int = 0         # Times LHS was satisfied
    correct: int = 0       # Times RHS matched observation
    
    # Status
    alive: bool = True     # Not contradicted
    
    def __post_init__(self):
        # Ensure lhs is frozenset
        if not isinstance(self.lhs, frozenset):
            object.__setattr__(self, 'lhs', frozenset(self.lhs))
    
    def lhs_satisfied(self, facts: Set[Token]) -> bool:
        """Check if all LHS conditions are met."""
        for token in self.lhs:
            if token.negated:
                # Negated token: the positive version must NOT be in facts
                positive = token.negate()
                if positive in facts:
                    return False
            else:
                # Positive token: must be in facts
                if token not in facts:
                    return False
        return True
    
    def fire(self, facts: Set[Token], true_label: Optional[int] = None) -> Optional[Token]:
        """
        Attempt to fire the rule.
        
        Returns the RHS token if LHS is satisfied, None otherwise.
        Updates statistics if true_label is provided.
        """
        if not self.lhs_satisfied(facts):
            return None
        
        self.fires += 1
        
        if true_label is not None:
            # Check if RHS matches truth
            if self.rhs.token_type == TokenType.LABEL:
                if self.rhs.value == true_label:
                    self.correct += 1
                else:
                    # Contradiction!
                    self.alive = False
        
        return self.rhs
    
    def confidence(self) -> float:
        """Return confidence (correct/fires)."""
        if self.fires == 0:
            return 0.5
        return self.correct / self.fires
    
    def precision(self) -> float:
        """Alias for confidence."""
        return self.confidence()
    
    def specificity(self) -> int:
        """Number of conditions in LHS."""
        return len(self.lhs)
    
    def __hash__(self):
        return hash((self.lhs, self.rhs))
    
    def __str__(self):
        lhs_str = " AND ".join(str(t) for t in sorted(self.lhs, key=str))
        return f"IF {lhs_str} THEN {self.rhs}"


class FactBase:
    """
    A collection of known facts (tokens).
    
    Supports:
    - Adding facts from observations
    - Querying facts
    - Deriving new facts via rules
    """
    
    def __init__(self):
        self.facts: Set[Token] = set()
        self.derived: Set[Token] = set()  # Facts derived via rules
    
    def add(self, token: Token):
        """Add a fact."""
        self.facts.add(token)
    
    def add_derived(self, token: Token):
        """Add a derived fact."""
        self.derived.add(token)
    
    def contains(self, token: Token) -> bool:
        """Check if a fact is known."""
        if token.negated:
            return token.negate() not in self.facts and token.negate() not in self.derived
        return token in self.facts or token in self.derived
    
    def all_facts(self) -> Set[Token]:
        """Return all facts (base + derived)."""
        return self.facts | self.derived
    
    def clear_derived(self):
        """Clear derived facts."""
        self.derived.clear()
    
    def clear(self):
        """Clear all facts."""
        self.facts.clear()
        self.derived.clear()
    
    @staticmethod
    def from_board(board: str) -> 'FactBase':
        """Create a FactBase from a board string."""
        fb = FactBase()
        for pos, val in enumerate(board):
            fb.add(Token(f"p{pos}", val, TokenType.POSITION))
        
        # Add meta-facts
        if '0' not in board:
            fb.add(Token("board_full", True, TokenType.META))
        else:
            fb.add(Token("has_empty", True, TokenType.META))
        
        return fb


class InferenceEngine:
    """
    Forward-chaining inference engine.
    
    Given a set of rules and facts, derives all possible conclusions.
    """
    
    def __init__(self, rules: List[ProductionRule]):
        self.rules = rules
    
    def infer(self, facts: FactBase, max_iterations: int = 10) -> Set[Token]:
        """
        Run forward chaining inference.
        
        Returns all derived tokens.
        """
        derived = set()
        
        for _ in range(max_iterations):
            new_derived = set()
            
            for rule in self.rules:
                if not rule.alive:
                    continue
                
                result = rule.fire(facts.all_facts())
                if result and result not in derived and result not in facts.facts:
                    new_derived.add(result)
                    facts.add_derived(result)
            
            if not new_derived:
                break  # Fixed point reached
            
            derived |= new_derived
        
        return derived


class SATLikeReasoner:
    """
    SAT-like reasoning without external solver.
    
    Key operations:
    1. Unit propagation: If a clause has one unfixed literal, it must be true
    2. Conflict detection: Check for contradictions
    3. Model counting: Count consistent assignments
    """
    
    def __init__(self):
        self.clauses: List[FrozenSet[Token]] = []  # CNF clauses
    
    def add_clause(self, literals: Set[Token]):
        """Add a clause (disjunction of literals)."""
        self.clauses.append(frozenset(literals))
    
    def add_rule_as_clause(self, rule: ProductionRule):
        """
        Convert rule to CNF clause.
        
        IF A AND B THEN C
        = NOT(A AND B) OR C
        = NOT A OR NOT B OR C
        """
        clause = set()
        for token in rule.lhs:
            clause.add(token.negate())
        clause.add(rule.rhs)
        self.add_clause(clause)
    
    def is_consistent(self, assignment: Dict[Tuple[str, Any], bool]) -> bool:
        """
        Check if assignment is consistent with all clauses.
        
        Assignment maps (name, value) -> True/False
        """
        for clause in self.clauses:
            clause_satisfied = False
            for literal in clause:
                key = (literal.name, literal.value)
                if key in assignment:
                    val = assignment[key]
                    if literal.negated:
                        val = not val
                    if val:
                        clause_satisfied = True
                        break
            
            if not clause_satisfied:
                # Check if clause has unassigned literals
                has_unassigned = any(
                    (lit.name, lit.value) not in assignment 
                    for lit in clause
                )
                if not has_unassigned:
                    return False  # Clause is false
        
        return True
    
    def unit_propagate(self, facts: Set[Token]) -> Set[Token]:
        """
        Find forced conclusions via unit propagation.
        
        If a clause has all but one literal false, the remaining must be true.
        """
        forced = set()
        assignment = {(t.name, t.value): not t.negated for t in facts}
        
        for clause in self.clauses:
            unsat_literals = []
            clause_sat = False
            
            for literal in clause:
                key = (literal.name, literal.value)
                if key in assignment:
                    val = assignment[key]
                    if literal.negated:
                        val = not val
                    if val:
                        clause_sat = True
                        break
                else:
                    unsat_literals.append(literal)
            
            if not clause_sat and len(unsat_literals) == 1:
                # Unit clause - this literal must be true
                forced.add(unsat_literals[0])
        
        return forced


class ProductionRuleEngine:
    """
    Complete production rule learning and inference system.
    
    Combines:
    1. Rule generation from observations
    2. Rule elimination on contradiction
    3. Forward chaining inference
    4. SAT-like consistency checking
    5. Weighted prediction
    """
    
    def __init__(self, num_outputs: int = 5, board_size: int = 9,
                 label_names: List[str] = None,
                 pattern_sizes: List[int] = None,
                 use_meta_facts: bool = True,
                 use_negative_rules: bool = True,
                 **kwargs):
        self.num_outputs = num_outputs
        self.board_size = board_size
        self.label_names = label_names or [f"label_{i}" for i in range(num_outputs)]
        self.pattern_sizes = pattern_sizes or [3]
        self.use_meta_facts = use_meta_facts
        self.use_negative_rules = use_negative_rules
        
        # Rule storage
        self.rules: Dict[int, ProductionRule] = {}  # id -> rule
        self.rule_id = 0
        
        # Index for fast lookup
        self.rules_by_rhs: Dict[int, List[int]] = defaultdict(list)  # label -> [rule_ids]
        
        # Observations
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        self.label_counts: Dict[int, int] = defaultdict(int)
        
        # SAT reasoner
        self.sat = SATLikeReasoner()
        
        self.stats = {
            'rules_generated': 0,
            'rules_eliminated': 0,
            'predictions': 0,
            'inferences': 0,
        }
    
    def _board_to_facts(self, board: str) -> FactBase:
        """Convert board to fact base."""
        return FactBase.from_board(board)
    
    def _generate_rules(self, board: str, label: int):
        """Generate production rules from observation."""
        facts = self._board_to_facts(board)
        position_tokens = [t for t in facts.facts if t.token_type == TokenType.POSITION]
        
        rhs = Token("label", label, TokenType.LABEL)
        
        for size in self.pattern_sizes:
            if size > len(position_tokens):
                continue
            
            for combo in combinations(position_tokens, size):
                lhs = frozenset(combo)
                
                # Check if rule already exists
                rule_key = (lhs, rhs)
                existing = None
                for rid, rule in self.rules.items():
                    if rule.lhs == lhs and rule.rhs.value == rhs.value:
                        existing = rid
                        break
                
                if existing is None:
                    rule = ProductionRule(lhs=lhs, rhs=rhs)
                    self.rules[self.rule_id] = rule
                    self.rules_by_rhs[label].append(self.rule_id)
                    self.rule_id += 1
                    self.stats['rules_generated'] += 1
        
        # Generate negative rules (IF pattern THEN NOT other_label)
        if self.use_negative_rules and len(self.observations) > 50:
            self._generate_negative_rules(board, label)
    
    def _generate_negative_rules(self, board: str, label: int):
        """Generate rules that exclude other labels."""
        # For now, skip this - adds complexity without clear benefit
        pass
    
    def _eliminate_contradicted(self, board: str, label: int):
        """Eliminate rules that fire but predict wrong label."""
        facts = self._board_to_facts(board)
        all_facts = facts.all_facts()
        eliminated = 0
        
        for rid, rule in list(self.rules.items()):
            if not rule.alive:
                continue
            
            if rule.lhs_satisfied(all_facts):
                rule.fires += 1
                if rule.rhs.token_type == TokenType.LABEL:
                    if rule.rhs.value == label:
                        rule.correct += 1
                    else:
                        rule.alive = False
                        eliminated += 1
        
        self.stats['rules_eliminated'] += eliminated
    
    def _get_alive_rules(self) -> List[ProductionRule]:
        """Get all alive rules."""
        return [r for r in self.rules.values() if r.alive]
    
    def _get_pure_rules(self, min_support: int = 2) -> Dict[int, List[ProductionRule]]:
        """Get rules with 100% precision grouped by label."""
        pure = defaultdict(list)
        for rule in self._get_alive_rules():
            if rule.fires >= min_support and rule.correct == rule.fires:
                if rule.rhs.token_type == TokenType.LABEL:
                    pure[rule.rhs.value].append(rule)
        
        # Sort by specificity
        for label in pure:
            pure[label].sort(key=lambda r: (-r.specificity(), -r.fires))
        
        return pure
    
    def _get_high_confidence_rules(self, min_confidence: float = 0.85,
                                    min_support: int = 2) -> Dict[int, List[ProductionRule]]:
        """Get rules with high confidence grouped by label."""
        confident = defaultdict(list)
        for rule in self._get_alive_rules():
            if rule.fires >= min_support and rule.confidence() >= min_confidence:
                if rule.rhs.token_type == TokenType.LABEL:
                    confident[rule.rhs.value].append(rule)
        
        for label in confident:
            confident[label].sort(key=lambda r: (-r.confidence(), -r.specificity(), -r.fires))
        
        return confident
    
    def predict(self, observation: str) -> int:
        """Predict label using production rules."""
        self.stats['predictions'] += 1
        
        facts = self._board_to_facts(observation)
        all_facts = facts.all_facts()
        
        # Calculate label priors
        total = sum(self.label_counts.values()) or 1
        label_freq = {l: self.label_counts.get(l, 0) / total 
                     for l in range(self.num_outputs)}
        
        # TIER 1: Pure rules (100% precision), rare labels first
        pure_rules = self._get_pure_rules(min_support=3)
        for label in sorted(range(self.num_outputs), key=lambda l: label_freq.get(l, 0)):
            for rule in pure_rules.get(label, []):
                if rule.lhs_satisfied(all_facts):
                    return label
        
        # TIER 2: High confidence rules with weighted voting
        confident_rules = self._get_high_confidence_rules(min_confidence=0.8, min_support=2)
        votes: Dict[int, float] = defaultdict(float)
        
        for label in range(self.num_outputs):
            for rule in confident_rules.get(label, []):
                if rule.lhs_satisfied(all_facts):
                    rarity_boost = 1.0 / (label_freq.get(label, 0.1) + 0.01)
                    weight = (
                        rule.confidence() * 
                        (rule.specificity() ** 2) * 
                        rule.fires * 
                        rarity_boost
                    )
                    votes[label] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        # TIER 3: All alive rules
        for rule in self._get_alive_rules():
            if rule.lhs_satisfied(all_facts) and rule.rhs.token_type == TokenType.LABEL:
                label = rule.rhs.value
                rarity_boost = 1.0 / (label_freq.get(label, 0.1) + 0.01)
                weight = rule.specificity() * (rule.fires + 1) * rarity_boost
                votes[label] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        # TIER 4: Prior
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
        
        # Eliminate first, then generate
        self._eliminate_contradicted(observation, correct_label)
        self._generate_rules(observation, correct_label)
    
    def get_stats(self) -> Dict[str, Any]:
        alive = len(self._get_alive_rules())
        pure = sum(len(v) for v in self._get_pure_rules().values())
        return {
            **self.stats,
            'rules': alive,
            'pure_rules': pure,
            'observations': len(self.observations),
        }
    
    def describe_knowledge(self) -> str:
        """Describe learned rules."""
        lines = ["=== Production Rule Engine ===\n"]
        
        alive_rules = self._get_alive_rules()
        pure_rules = self._get_pure_rules()
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Total rules: {len(self.rules)}")
        lines.append(f"Alive rules: {len(alive_rules)}")
        lines.append(f"Pure rules: {sum(len(v) for v in pure_rules.values())}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for idx in range(self.num_outputs):
            count = self.label_counts.get(idx, 0)
            pct = count / total * 100 if total > 0 else 0
            label = self.label_names[idx]
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        
        lines.append("\n--- Pure Rules (100% precision) ---")
        for label_idx in range(self.num_outputs):
            label_name = self.label_names[label_idx]
            rules = pure_rules.get(label_idx, [])
            lines.append(f"\n  {label_name}: {len(rules)} rules")
            for rule in rules[:8]:
                lines.append(f"    {rule} (n={rule.fires})")
        
        return '\n'.join(lines)
    
    def export_rules_cnf(self) -> str:
        """Export rules in CNF-like format."""
        lines = ["c Production Rules in CNF format", "c"]
        
        for rule in self._get_alive_rules():
            if rule.fires < 2:
                continue
            
            # Convert to clause: NOT(LHS) OR RHS
            clause_parts = []
            for token in rule.lhs:
                clause_parts.append(f"-{token.name}={token.value}")
            clause_parts.append(f"+label={rule.rhs.value}")
            
            lines.append(" ".join(clause_parts))
        
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE
    
    print("="*70)
    print("Testing Production Rule Engine")
    print("="*70)
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    engine = ProductionRuleEngine(
        num_outputs=5,
        label_names=LABEL_SPACE,
        pattern_sizes=[3]
    )
    
    correct = 0
    checkpoints = [10, 25, 50, 100, 200, 500, 1000, 2000]
    
    for i in range(3000):
        obs = gen.next()
        if obs is None:
            print(f"All {i} states observed!")
            break
        
        board, true_idx = obs
        pred_idx = engine.predict(board)
        
        if pred_idx == true_idx:
            correct += 1
        
        engine.update_history(board, pred_idx, true_idx)
        
        if (i + 1) in checkpoints:
            acc = correct / (i + 1)
            stats = engine.get_stats()
            print(f"  @{i+1:5d}: acc={acc:.1%} rules={stats['rules']} pure={stats['pure_rules']}")
    
    print(f"\nFinal Accuracy: {correct/(i+1):.1%}")
    print("\n" + engine.describe_knowledge())
    
    print("\n--- Sample CNF Export ---")
    print(engine.export_rules_cnf()[:500])
