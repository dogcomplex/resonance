"""
Unified Production Rule System

Clean representation:  LHS => RHS

Examples:
  turn_X  cell_empty  =>  turn_O  cell_X     # Move (turn_X is catalyst-ish, but consumed)
  cell_X  cell_X  cell_X  =>  win_X          # Win detection (cells are catalysts)
  gold3  =>  sword1                           # Crafting with quantities
  attack  =>  hit%70  miss%30                 # Probabilistic outcome

Key concepts:
  - CATALYST: Token on both sides (checked, not consumed)
  - CONSUMED: Token on LHS only (removed)
  - PRODUCED: Token on RHS only (added)
  - QUANTITY: Token with count (gold3 = 3 gold)
  - PROBABILITY: Token with % (hit%70 = 70% chance)

Abstraction:
  - Labels become placeholders for complex states
  - "state_A" might represent (cell0_X AND cell1_X AND cell2_X)
  - Enables bubble-up generalization without semantic knowledge
"""

import random
import re
from typing import List, Tuple, Dict, Set, Optional, FrozenSet, Union
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations


@dataclass(frozen=True)
class Token:
    """
    A token with optional quantity and probability.
    
    Formats:
      name       -> Token("name", qty=1, prob=None)
      name3      -> Token("name", qty=3, prob=None)  
      name%70    -> Token("name", qty=1, prob=0.70)
      name3%50   -> Token("name", qty=3, prob=0.50)
    """
    name: str
    qty: int = 1
    prob: Optional[float] = None
    
    def __str__(self):
        s = self.name
        if self.qty != 1:
            s += str(self.qty)
        if self.prob is not None:
            s += f"%{int(self.prob * 100)}"
        return s
    
    def __repr__(self):
        return str(self)
    
    def base(self) -> 'Token':
        """Return token without quantity or probability."""
        return Token(self.name, qty=1, prob=None)
    
    def with_qty(self, qty: int) -> 'Token':
        """Return token with different quantity."""
        return Token(self.name, qty=qty, prob=self.prob)
    
    def with_prob(self, prob: float) -> 'Token':
        """Return token with probability."""
        return Token(self.name, qty=self.qty, prob=prob)
    
    @staticmethod
    def parse(s: str) -> 'Token':
        """
        Parse token from string.
        
        Examples:
          "gold" -> Token("gold")
          "gold3" -> Token("gold", qty=3)
          "hit%70" -> Token("hit", prob=0.70)
          "gold3%50" -> Token("gold", qty=3, prob=0.50)
        """
        s = s.strip()
        
        # Try to parse probability suffix first
        prob = None
        if '%' in s:
            parts = s.rsplit('%', 1)
            s = parts[0]
            prob = int(parts[1]) / 100.0
        
        # Now parse name and optional quantity
        # Match letters/underscores, then optional digits at end
        match = re.match(r'^([a-zA-Z_][a-zA-Z_]*)(\d*)$', s)
        if not match:
            # Fallback: treat entire string as name
            return Token(s, 1, prob)
        
        name = match.group(1)
        qty_str = match.group(2)
        qty = int(qty_str) if qty_str else 1
        
        return Token(name, qty, prob)


@dataclass
class Rule:
    """
    A production rule: LHS => RHS
    
    Representation: space-separated tokens with => separator
    
    Examples:
      "turn_X cell_empty => turn_O cell_X"
      "gold3 => sword1"
      "attack => hit%70 miss%30"
      "cell_X cell_X cell_X => cell_X cell_X cell_X win_X"  # Catalysts
    """
    lhs: Tuple[Token, ...]
    rhs: Tuple[Token, ...]
    
    # Statistics
    fires: int = 0
    successes: int = 0
    
    def __post_init__(self):
        if not isinstance(self.lhs, tuple):
            object.__setattr__(self, 'lhs', tuple(self.lhs))
        if not isinstance(self.rhs, tuple):
            object.__setattr__(self, 'rhs', tuple(self.rhs))
    
    @property
    def catalysts(self) -> Set[str]:
        """Tokens present on both sides (by name)."""
        lhs_names = {t.name for t in self.lhs}
        rhs_names = {t.name for t in self.rhs}
        return lhs_names & rhs_names
    
    @property
    def consumed(self) -> Dict[str, int]:
        """Tokens consumed (LHS only, with quantities)."""
        result = defaultdict(int)
        rhs_names = {t.name for t in self.rhs}
        for t in self.lhs:
            if t.name not in rhs_names:
                result[t.name] += t.qty
        return dict(result)
    
    @property
    def produced(self) -> Dict[str, int]:
        """Tokens produced (RHS only, with quantities)."""
        result = defaultdict(int)
        lhs_names = {t.name for t in self.lhs}
        for t in self.rhs:
            if t.name not in lhs_names:
                result[t.name] += t.qty
        return dict(result)
    
    @property
    def is_probabilistic(self) -> bool:
        """True if RHS has probabilistic tokens."""
        return any(t.prob is not None for t in self.rhs)
    
    def can_fire(self, state: Dict[str, int]) -> bool:
        """Check if rule can fire given current state (quantities)."""
        # Count required quantities per token
        required = defaultdict(int)
        for t in self.lhs:
            required[t.name] += t.qty
        
        # Check all requirements met
        for name, qty in required.items():
            if state.get(name, 0) < qty:
                return False
        return True
    
    def fire(self, state: Dict[str, int]) -> Optional[Dict[str, int]]:
        """
        Fire the rule, returning new state.
        Returns None if rule cannot fire.
        
        For probabilistic rules, samples the outcome.
        """
        if not self.can_fire(state):
            return None
        
        self.fires += 1
        new_state = state.copy()
        
        # Compute net change per token
        lhs_counts = defaultdict(int)
        rhs_counts = defaultdict(int)
        
        for t in self.lhs:
            lhs_counts[t.name] += t.qty
        
        # Handle probabilistic RHS
        if self.is_probabilistic:
            prob_tokens = [t for t in self.rhs if t.prob is not None]
            non_prob_tokens = [t for t in self.rhs if t.prob is None]
            
            # Add non-probabilistic tokens
            for t in non_prob_tokens:
                rhs_counts[t.name] += t.qty
            
            # Sample one probabilistic token
            total_prob = sum(t.prob for t in prob_tokens)
            if total_prob > 0:
                r = random.random() * total_prob
                cumsum = 0
                for t in prob_tokens:
                    cumsum += t.prob
                    if r <= cumsum:
                        rhs_counts[t.name] += t.qty
                        break
        else:
            for t in self.rhs:
                rhs_counts[t.name] += t.qty
        
        # Apply changes
        all_tokens = set(lhs_counts.keys()) | set(rhs_counts.keys())
        for name in all_tokens:
            delta = rhs_counts[name] - lhs_counts[name]
            new_qty = new_state.get(name, 0) + delta
            if new_qty > 0:
                new_state[name] = new_qty
            elif name in new_state:
                del new_state[name]
        
        return new_state
    
    def __str__(self):
        lhs_str = "  ".join(str(t) for t in self.lhs)
        rhs_str = "  ".join(str(t) for t in self.rhs)
        return f"{lhs_str}  =>  {rhs_str}"
    
    def __repr__(self):
        return str(self)
    
    @staticmethod
    def parse(s: str) -> 'Rule':
        """
        Parse rule from string.
        
        Format: "lhs_tokens => rhs_tokens"
        """
        if "=>" not in s:
            raise ValueError(f"Rule must contain '=>': {s}")
        
        lhs_str, rhs_str = s.split("=>", 1)
        
        lhs_tokens = tuple(Token.parse(t) for t in lhs_str.split() if t)
        rhs_tokens = tuple(Token.parse(t) for t in rhs_str.split() if t)
        
        return Rule(lhs=lhs_tokens, rhs=rhs_tokens)


class State:
    """
    A state represented as token counts.
    
    Supports:
    - Adding/removing tokens with quantities
    - Checking token presence
    - Applying rules
    """
    
    def __init__(self, tokens: Dict[str, int] = None):
        self.tokens = tokens.copy() if tokens else {}
    
    def add(self, name: str, qty: int = 1):
        """Add tokens."""
        self.tokens[name] = self.tokens.get(name, 0) + qty
    
    def remove(self, name: str, qty: int = 1) -> bool:
        """Remove tokens. Returns False if insufficient."""
        current = self.tokens.get(name, 0)
        if current < qty:
            return False
        self.tokens[name] = current - qty
        if self.tokens[name] <= 0:
            del self.tokens[name]
        return True
    
    def has(self, name: str, qty: int = 1) -> bool:
        """Check if state has at least qty of token."""
        return self.tokens.get(name, 0) >= qty
    
    def count(self, name: str) -> int:
        """Get count of token."""
        return self.tokens.get(name, 0)
    
    def apply_rule(self, rule: Rule) -> bool:
        """Apply a rule. Returns True if fired."""
        new_tokens = rule.fire(self.tokens)
        if new_tokens is not None:
            self.tokens = new_tokens
            return True
        return False
    
    def copy(self) -> 'State':
        return State(self.tokens)
    
    def __str__(self):
        parts = []
        for name, qty in sorted(self.tokens.items()):
            if qty == 1:
                parts.append(name)
            else:
                parts.append(f"{name}{qty}")
        return "{ " + "  ".join(parts) + " }" if parts else "{ }"
    
    def __repr__(self):
        return str(self)


class AbstractionLayer:
    """
    Creates abstract labels for complex states.
    
    Enables bubble-up generalization:
    - Observe patterns that co-occur
    - Create abstract label for the pattern
    - Replace occurrences with the label
    
    Example:
      Sees (p0=1, p1=1, p2=1) always with win1
      Creates abstract: "row0_X" = {p0=1, p1=1, p2=1}
      Now rules can use "row0_X" instead of 3 tokens
    """
    
    def __init__(self):
        # Map: abstract_name -> concrete token set
        self.abstractions: Dict[str, FrozenSet[str]] = {}
        
        # Reverse map: concrete set -> abstract name
        self.reverse: Dict[FrozenSet[str], str] = {}
        
        # Pattern co-occurrence statistics
        self.cooccurrence: Dict[FrozenSet[str], int] = defaultdict(int)
        
        # Next abstract ID
        self.next_id = 0
    
    def observe(self, tokens: Set[str]):
        """Observe a set of tokens, tracking co-occurrences."""
        # Track pairs that co-occur
        token_list = sorted(tokens)
        for i in range(len(token_list)):
            for j in range(i + 1, len(token_list)):
                pair = frozenset([token_list[i], token_list[j]])
                self.cooccurrence[pair] += 1
    
    def create_abstraction(self, tokens: Set[str], name: str = None) -> str:
        """Create an abstract label for a token set."""
        frozen = frozenset(tokens)
        
        if frozen in self.reverse:
            return self.reverse[frozen]
        
        if name is None:
            name = f"A{self.next_id}"
            self.next_id += 1
        
        self.abstractions[name] = frozen
        self.reverse[frozen] = name
        
        return name
    
    def auto_abstract(self, min_cooccurrence: int = 10, min_size: int = 2) -> List[str]:
        """
        Automatically create abstractions for frequently co-occurring patterns.
        
        Returns list of created abstraction names.
        """
        created = []
        
        # Find highly correlated token sets
        candidates = [(pattern, count) for pattern, count in self.cooccurrence.items()
                     if count >= min_cooccurrence and len(pattern) >= min_size]
        candidates.sort(key=lambda x: -x[1])
        
        for pattern, count in candidates[:20]:  # Limit
            if pattern not in self.reverse:
                name = self.create_abstraction(pattern)
                created.append(name)
        
        return created
    
    def expand(self, abstract_name: str) -> Set[str]:
        """Expand an abstraction to its concrete tokens."""
        return set(self.abstractions.get(abstract_name, {abstract_name}))
    
    def compress(self, tokens: Set[str]) -> Set[str]:
        """Replace concrete tokens with abstractions where possible."""
        result = tokens.copy()
        
        for name, pattern in self.abstractions.items():
            if pattern.issubset(result):
                result -= pattern
                result.add(name)
        
        return result
    
    def describe(self) -> str:
        """Describe current abstractions."""
        lines = ["=== Abstractions ==="]
        for name, pattern in sorted(self.abstractions.items()):
            lines.append(f"  {name} = {{ {', '.join(sorted(pattern))} }}")
        return '\n'.join(lines)


class RuleEngine:
    """
    Engine for learning and applying production rules.
    
    Supports:
    - Rule parsing and representation
    - State transitions
    - Abstraction/generalization
    - Rule discovery from observations
    """
    
    def __init__(self):
        self.rules: List[Rule] = []
        self.abstraction = AbstractionLayer()
        
        # Statistics
        self.stats = {
            'observations': 0,
            'rules_learned': 0,
        }
    
    def add_rule(self, rule: Union[Rule, str]):
        """Add a rule (Rule object or string)."""
        if isinstance(rule, str):
            rule = Rule.parse(rule)
        self.rules.append(rule)
    
    def get_applicable_rules(self, state: State) -> List[Rule]:
        """Get all rules that can fire in current state."""
        return [r for r in self.rules if r.can_fire(state.tokens)]
    
    def step(self, state: State) -> List[Tuple[Rule, State]]:
        """
        Execute one step, returning all possible (rule, new_state) pairs.
        """
        results = []
        for rule in self.get_applicable_rules(state):
            new_state = state.copy()
            new_state.apply_rule(rule)
            results.append((rule, new_state))
        return results
    
    def observe_transition(self, before: Dict[str, int], after: Dict[str, int]):
        """
        Learn from an observed state transition.
        
        Infers a rule that explains the change.
        """
        self.stats['observations'] += 1
        
        # Track co-occurrences for abstraction
        self.abstraction.observe(set(before.keys()))
        self.abstraction.observe(set(after.keys()))
        
        # Compute change
        consumed = {}
        produced = {}
        catalysts = set()
        
        all_tokens = set(before.keys()) | set(after.keys())
        
        for token in all_tokens:
            before_qty = before.get(token, 0)
            after_qty = after.get(token, 0)
            
            if before_qty > 0 and after_qty > 0:
                # Present in both - might be catalyst
                if before_qty == after_qty:
                    catalysts.add(token)
                elif before_qty > after_qty:
                    consumed[token] = before_qty - after_qty
                else:
                    produced[token] = after_qty - before_qty
            elif before_qty > 0:
                consumed[token] = before_qty
            elif after_qty > 0:
                produced[token] = after_qty
        
        # Create LHS (consumed + catalysts) and RHS (produced + catalysts)
        lhs_tokens = []
        for token, qty in consumed.items():
            lhs_tokens.append(Token(token, qty=qty))
        for token in catalysts:
            lhs_tokens.append(Token(token, qty=before[token]))
        
        rhs_tokens = []
        for token, qty in produced.items():
            rhs_tokens.append(Token(token, qty=qty))
        for token in catalysts:
            rhs_tokens.append(Token(token, qty=after[token]))
        
        if lhs_tokens or rhs_tokens:
            rule = Rule(lhs=tuple(lhs_tokens), rhs=tuple(rhs_tokens))
            
            # Check if we already have this rule
            rule_str = str(rule)
            existing = [r for r in self.rules if str(r) == rule_str]
            if not existing:
                self.rules.append(rule)
                self.stats['rules_learned'] += 1
    
    def describe(self) -> str:
        """Describe engine state."""
        lines = ["=== Rule Engine ==="]
        lines.append(f"Rules: {len(self.rules)}")
        lines.append(f"Observations: {self.stats['observations']}")
        
        lines.append("\n--- Rules ---")
        for rule in self.rules[:20]:
            lines.append(f"  {rule}")
        if len(self.rules) > 20:
            lines.append(f"  ... and {len(self.rules) - 20} more")
        
        if self.abstraction.abstractions:
            lines.append("\n" + self.abstraction.describe())
        
        return '\n'.join(lines)


def demo():
    """Demonstrate the production rule system."""
    print("="*70)
    print("PRODUCTION RULE SYSTEM DEMO")
    print("="*70)
    
    # Parse and display rules
    print("\n--- Rule Parsing ---")
    
    examples = [
        "turn_X  cell_empty  =>  turn_O  cell_X",
        "gold3  =>  sword1",
        "attack  =>  hit%70  miss%30",
        "cell_X  cell_X  cell_X  =>  cell_X  cell_X  cell_X  win_X",
        "wood2  stone1  =>  pickaxe1",
    ]
    
    for s in examples:
        rule = Rule.parse(s)
        print(f"\n  Input:  {s}")
        print(f"  Parsed: {rule}")
        print(f"    Catalysts: {rule.catalysts}")
        print(f"    Consumed:  {rule.consumed}")
        print(f"    Produced:  {rule.produced}")
        print(f"    Probabilistic: {rule.is_probabilistic}")
    
    # State manipulation
    print("\n" + "-"*70)
    print("--- State Manipulation ---")
    
    state = State({"gold": 10, "wood": 5})
    print(f"Initial state: {state}")
    
    rule = Rule.parse("gold3 => sword1")
    print(f"Rule: {rule}")
    
    while rule.can_fire(state.tokens):
        state.apply_rule(rule)
        print(f"  After firing: {state}")
    
    # Probabilistic rules
    print("\n" + "-"*70)
    print("--- Probabilistic Rules ---")
    
    state = State({"attack": 10})
    rule = Rule.parse("attack => hit%70 miss%30")
    print(f"Rule: {rule}")
    print(f"Initial: {state}")
    
    outcomes = {"hit": 0, "miss": 0}
    for _ in range(10):
        if rule.can_fire(state.tokens):
            before = state.copy()
            state.apply_rule(rule)
            gained = set(state.tokens.keys()) - set(before.tokens.keys())
            for g in gained:
                outcomes[g] = outcomes.get(g, 0) + 1
    
    print(f"After 10 attacks: {state}")
    print(f"Outcomes: {outcomes}")
    
    # Rule learning
    print("\n" + "-"*70)
    print("--- Rule Learning from Observations ---")
    
    engine = RuleEngine()
    
    # Observe some transitions (simulating TicTacToe moves as token changes)
    transitions = [
        ({"turn_X": 1, "c0_empty": 1}, {"turn_O": 1, "c0_X": 1}),
        ({"turn_O": 1, "c4_empty": 1}, {"turn_X": 1, "c4_O": 1}),
        ({"turn_X": 1, "c8_empty": 1}, {"turn_O": 1, "c8_X": 1}),
        ({"gold": 5}, {"gold": 2, "sword": 1}),  # Crafting
    ]
    
    for before, after in transitions:
        engine.observe_transition(before, after)
    
    print(engine.describe())
    
    # TicTacToe rules in new format
    print("\n" + "="*70)
    print("TICTACTOE RULES IN PRODUCTION FORMAT")
    print("="*70)
    
    print("""
Move rules (catalyst: nothing; consumed: turn+empty; produced: turn+piece):
  turn_X  c0_empty  =>  turn_O  c0_X
  turn_X  c1_empty  =>  turn_O  c1_X
  ... (18 total: 9 positions × 2 players)

Win detection (catalyst: cells; produced: result):
  c0_X  c1_X  c2_X  =>  c0_X  c1_X  c2_X  win_X
  c0_X  c4_X  c8_X  =>  c0_X  c4_X  c8_X  win_X
  ... (16 total: 8 lines × 2 players)

Generalized (using abstraction):
  turn_X  cN_empty  =>  turn_O  cN_X    # N is any position
  row_X  =>  row_X  win_X               # row_X abstracts 3-in-a-row
""")


if __name__ == "__main__":
    demo()
