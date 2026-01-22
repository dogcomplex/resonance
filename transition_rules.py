"""
Full Production Rule System with State Transitions

This is the complete LHS => RHS framework:

1. STATE TRANSITIONS
   - LHS tokens are consumed (removed from state)
   - RHS tokens are produced (added to state)
   - Models game moves: "empty at p0" + "player X turn" => "X at p0" + "player O turn"

2. CATALYSTS (tokens on both sides)
   - Checked but not consumed
   - Acts as IF-THEN conditional
   - Example: "player X turn" on both sides = "if it's X's turn"

3. TEMPORAL CHAINING
   - Rules fire in sequence
   - State evolves over time
   - Can model full game trees

4. SAT ENCODING
   - All rules as CNF clauses
   - Consistency checking
   - Model enumeration

This enables learning FULL game rules, not just classification.
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet, Union
from collections import defaultdict
from itertools import combinations, permutations
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy


class TokenType(Enum):
    """Types of tokens."""
    CELL = "cell"           # Board cell state: cell_0_X, cell_0_O, cell_0_empty
    TURN = "turn"           # Whose turn: turn_X, turn_O
    RESULT = "result"       # Game result: result_win_X, result_win_O, result_draw
    COUNT = "count"         # Move count: move_1, move_2, ...
    META = "meta"           # Meta state: game_over, valid_move
    DERIVED = "derived"     # Derived facts: three_in_row_X


@dataclass(frozen=True)
class Token:
    """
    An atomic fact/token.
    
    Tokens represent discrete facts about game state:
    - cell_0_X = cell 0 contains X
    - turn_X = it's X's turn
    - result_win_X = X has won
    """
    name: str
    negated: bool = False
    
    def __str__(self):
        return f"{'!' if self.negated else ''}{self.name}"
    
    def negate(self) -> 'Token':
        return Token(self.name, not self.negated)
    
    @staticmethod
    def cell(pos: int, value: str) -> 'Token':
        """Create a cell token."""
        return Token(f"c{pos}_{value}")
    
    @staticmethod
    def turn(player: str) -> 'Token':
        """Create a turn token."""
        return Token(f"turn_{player}")
    
    @staticmethod
    def result(outcome: str) -> 'Token':
        """Create a result token."""
        return Token(f"result_{outcome}")


@dataclass
class TransitionRule:
    """
    A state transition rule: LHS => RHS
    
    Semantics:
    - CONSUMED: Tokens in LHS but not RHS are removed from state
    - PRODUCED: Tokens in RHS but not LHS are added to state
    - CATALYSTS: Tokens in BOTH LHS and RHS are checked but unchanged
    
    Example (X plays at position 0):
        LHS: {cell_0_empty, turn_X}
        RHS: {cell_0_X, turn_O}
        
        Consumed: cell_0_empty, turn_X
        Produced: cell_0_X, turn_O
        Catalysts: (none in this case)
    
    Example with catalyst (win check):
        LHS: {cell_0_X, cell_1_X, cell_2_X, turn_O}  # O's turn to detect X won
        RHS: {cell_0_X, cell_1_X, cell_2_X, result_win_X}
        
        Consumed: turn_O
        Produced: result_win_X
        Catalysts: cell_0_X, cell_1_X, cell_2_X (checked but not changed)
    """
    lhs: FrozenSet[Token]
    rhs: FrozenSet[Token]
    
    # Derived
    _consumed: FrozenSet[Token] = field(default=None, repr=False)
    _produced: FrozenSet[Token] = field(default=None, repr=False)
    _catalysts: FrozenSet[Token] = field(default=None, repr=False)
    
    # Statistics
    fires: int = 0
    
    def __post_init__(self):
        if not isinstance(self.lhs, frozenset):
            object.__setattr__(self, 'lhs', frozenset(self.lhs))
        if not isinstance(self.rhs, frozenset):
            object.__setattr__(self, 'rhs', frozenset(self.rhs))
        
        # Compute consumed, produced, catalysts
        object.__setattr__(self, '_consumed', self.lhs - self.rhs)
        object.__setattr__(self, '_produced', self.rhs - self.lhs)
        object.__setattr__(self, '_catalysts', self.lhs & self.rhs)
    
    @property
    def consumed(self) -> FrozenSet[Token]:
        return self._consumed
    
    @property
    def produced(self) -> FrozenSet[Token]:
        return self._produced
    
    @property
    def catalysts(self) -> FrozenSet[Token]:
        return self._catalysts
    
    def can_fire(self, state: Set[Token]) -> bool:
        """Check if rule can fire given current state."""
        # All LHS tokens must be present (positive) or absent (negated)
        for token in self.lhs:
            if token.negated:
                if token.negate() in state:
                    return False
            else:
                if token not in state:
                    return False
        return True
    
    def fire(self, state: Set[Token]) -> Optional[Set[Token]]:
        """
        Fire the rule, returning new state.
        Returns None if rule cannot fire.
        """
        if not self.can_fire(state):
            return None
        
        self.fires += 1
        
        # New state = (old state - consumed) + produced
        new_state = state.copy()
        new_state -= self.consumed
        new_state |= self.produced
        
        return new_state
    
    def __str__(self):
        lhs_str = ", ".join(str(t) for t in sorted(self.lhs, key=str))
        rhs_str = ", ".join(str(t) for t in sorted(self.rhs, key=str))
        
        parts = []
        if self.catalysts:
            parts.append(f"IF {', '.join(str(t) for t in self.catalysts)}")
        if self.consumed:
            parts.append(f"CONSUME {', '.join(str(t) for t in self.consumed)}")
        if self.produced:
            parts.append(f"PRODUCE {', '.join(str(t) for t in self.produced)}")
        
        return " | ".join(parts) if parts else f"{lhs_str} => {rhs_str}"


class GameState:
    """
    A game state represented as a set of tokens.
    """
    
    def __init__(self, tokens: Set[Token] = None):
        self.tokens = tokens or set()
        self.history: List[Set[Token]] = []
    
    def add(self, token: Token):
        self.tokens.add(token)
    
    def remove(self, token: Token):
        self.tokens.discard(token)
    
    def has(self, token: Token) -> bool:
        if token.negated:
            return token.negate() not in self.tokens
        return token in self.tokens
    
    def has_all(self, tokens: Set[Token]) -> bool:
        return all(self.has(t) for t in tokens)
    
    def apply_rule(self, rule: TransitionRule) -> bool:
        """Apply a rule, returning True if it fired."""
        new_state = rule.fire(self.tokens)
        if new_state is not None:
            self.history.append(self.tokens.copy())
            self.tokens = new_state
            return True
        return False
    
    def copy(self) -> 'GameState':
        gs = GameState(self.tokens.copy())
        gs.history = self.history.copy()
        return gs
    
    def __str__(self):
        return "{" + ", ".join(str(t) for t in sorted(self.tokens, key=str)) + "}"
    
    @staticmethod
    def from_board(board: str, current_turn: str = None) -> 'GameState':
        """
        Create game state from board string.
        
        Board uses: 0=empty, 1=X, 2=O
        """
        state = GameState()
        
        x_count = 0
        o_count = 0
        
        for pos, val in enumerate(board):
            if val == '0':
                state.add(Token.cell(pos, 'empty'))
            elif val == '1':
                state.add(Token.cell(pos, 'X'))
                x_count += 1
            elif val == '2':
                state.add(Token.cell(pos, 'O'))
                o_count += 1
        
        # Determine turn if not specified
        if current_turn is None:
            current_turn = 'X' if x_count == o_count else 'O'
        
        state.add(Token.turn(current_turn))
        
        return state
    
    def to_board(self) -> str:
        """Convert state back to board string."""
        board = ['0'] * 9
        for token in self.tokens:
            if token.name.startswith('c') and '_' in token.name:
                parts = token.name.split('_')
                pos = int(parts[0][1:])
                val = parts[1]
                if val == 'X':
                    board[pos] = '1'
                elif val == 'O':
                    board[pos] = '2'
        return ''.join(board)


class TicTacToeRuleGenerator:
    """
    Generates the complete rule set for TicTacToe.
    
    Rule types:
    1. MOVE RULES: Player places piece
    2. WIN RULES: Check for three-in-a-row
    3. DRAW RULES: Check for full board with no winner
    """
    
    WIN_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6],              # Diagonals
    ]
    
    @classmethod
    def generate_move_rules(cls) -> List[TransitionRule]:
        """Generate rules for placing pieces."""
        rules = []
        
        for player in ['X', 'O']:
            next_player = 'O' if player == 'X' else 'X'
            
            for pos in range(9):
                # IF cell is empty AND it's player's turn
                # THEN cell gets player's piece AND turn switches
                lhs = frozenset([
                    Token.cell(pos, 'empty'),
                    Token.turn(player)
                ])
                rhs = frozenset([
                    Token.cell(pos, player),
                    Token.turn(next_player)
                ])
                
                rules.append(TransitionRule(lhs=lhs, rhs=rhs))
        
        return rules
    
    @classmethod
    def generate_win_rules(cls) -> List[TransitionRule]:
        """Generate rules for detecting wins."""
        rules = []
        
        for player in ['X', 'O']:
            for line in cls.WIN_LINES:
                # IF three cells have player's piece (catalysts)
                # THEN result is win for player
                # The cell tokens are catalysts (on both sides)
                cell_tokens = frozenset([Token.cell(pos, player) for pos in line])
                
                lhs = cell_tokens
                rhs = cell_tokens | {Token.result(f'win_{player}')}
                
                rules.append(TransitionRule(lhs=lhs, rhs=rhs))
        
        return rules
    
    @classmethod
    def generate_draw_rules(cls) -> List[TransitionRule]:
        """Generate rule for detecting draw."""
        # This is complex - need NO empty cells AND no win
        # For simplicity, we'll handle this differently
        return []
    
    @classmethod
    def generate_all_rules(cls) -> List[TransitionRule]:
        """Generate complete rule set."""
        rules = []
        rules.extend(cls.generate_move_rules())
        rules.extend(cls.generate_win_rules())
        rules.extend(cls.generate_draw_rules())
        return rules


class TransitionEngine:
    """
    Engine for executing transition rules.
    
    Supports:
    - Single-step execution
    - Fixed-point iteration (apply rules until none fire)
    - Temporal simulation
    """
    
    def __init__(self, rules: List[TransitionRule]):
        self.rules = rules
        self.move_rules = [r for r in rules if 'empty' in str(r.consumed)]
        self.win_rules = [r for r in rules if 'result' in str(r.produced)]
    
    def step(self, state: GameState, rule_filter: List[TransitionRule] = None) -> List[Tuple[TransitionRule, GameState]]:
        """
        Execute one step, returning all possible (rule, new_state) pairs.
        """
        rules = rule_filter or self.rules
        results = []
        
        for rule in rules:
            if rule.can_fire(state.tokens):
                new_state = state.copy()
                new_state.apply_rule(rule)
                results.append((rule, new_state))
        
        return results
    
    def derive_to_fixpoint(self, state: GameState) -> GameState:
        """
        Apply derivation rules until no more fire.
        Used for checking win conditions after moves.
        """
        changed = True
        while changed:
            changed = False
            for rule in self.win_rules:
                if rule.can_fire(state.tokens):
                    # Check if this would add new tokens
                    if not rule.produced.issubset(state.tokens):
                        state.apply_rule(rule)
                        changed = True
        return state
    
    def get_legal_moves(self, state: GameState) -> List[Tuple[TransitionRule, GameState]]:
        """Get all legal moves from current state."""
        return self.step(state, self.move_rules)
    
    def simulate_game(self, max_moves: int = 9) -> List[GameState]:
        """Simulate a random game, returning state sequence."""
        state = GameState.from_board('0' * 9, 'X')
        states = [state.copy()]
        
        for _ in range(max_moves):
            # Check for win first
            state = self.derive_to_fixpoint(state)
            
            # Check if game over
            has_result = any(t.name.startswith('result_') for t in state.tokens)
            if has_result:
                break
            
            # Get legal moves
            moves = self.get_legal_moves(state)
            if not moves:
                break
            
            # Pick random move
            rule, new_state = random.choice(moves)
            state = new_state
            states.append(state.copy())
        
        return states


class TransitionLearner:
    """
    Learn transition rules from game traces.
    
    Given sequences of (before_state, after_state), infer rules.
    """
    
    def __init__(self):
        self.observed_transitions: List[Tuple[Set[Token], Set[Token]]] = []
        self.candidate_rules: Dict[Tuple, TransitionRule] = {}
        self.confirmed_rules: List[TransitionRule] = []
    
    def observe_transition(self, before: Set[Token], after: Set[Token]):
        """Record an observed state transition."""
        self.observed_transitions.append((before.copy(), after.copy()))
        
        # Infer potential rule
        consumed = before - after
        produced = after - before
        catalysts = before & after
        
        # Create candidate rule
        lhs = consumed | catalysts
        rhs = produced | catalysts
        
        if lhs and rhs:  # Non-trivial transition
            rule = TransitionRule(lhs=lhs, rhs=rhs)
            key = (rule.lhs, rule.rhs)
            
            if key not in self.candidate_rules:
                self.candidate_rules[key] = rule
            self.candidate_rules[key].fires += 1
    
    def observe_game_trace(self, states: List[Set[Token]]):
        """Observe a full game trace."""
        for i in range(len(states) - 1):
            self.observe_transition(states[i], states[i + 1])
    
    def get_confirmed_rules(self, min_observations: int = 2) -> List[TransitionRule]:
        """Get rules observed multiple times."""
        return [r for r in self.candidate_rules.values() if r.fires >= min_observations]
    
    def describe_learned_rules(self) -> str:
        """Describe all learned rules."""
        lines = ["=== Learned Transition Rules ===\n"]
        
        confirmed = self.get_confirmed_rules()
        lines.append(f"Total candidate rules: {len(self.candidate_rules)}")
        lines.append(f"Confirmed rules (2+ obs): {len(confirmed)}")
        
        # Group by type
        move_rules = []
        win_rules = []
        other_rules = []
        
        for rule in confirmed:
            if any('empty' in str(t) for t in rule.consumed):
                move_rules.append(rule)
            elif any('result' in str(t) for t in rule.produced):
                win_rules.append(rule)
            else:
                other_rules.append(rule)
        
        lines.append(f"\n--- Move Rules ({len(move_rules)}) ---")
        for rule in move_rules[:5]:
            lines.append(f"  {rule}")
        if len(move_rules) > 5:
            lines.append(f"  ... and {len(move_rules) - 5} more")
        
        lines.append(f"\n--- Win Rules ({len(win_rules)}) ---")
        for rule in win_rules[:10]:
            lines.append(f"  {rule}")
        if len(win_rules) > 10:
            lines.append(f"  ... and {len(win_rules) - 10} more")
        
        if other_rules:
            lines.append(f"\n--- Other Rules ({len(other_rules)}) ---")
            for rule in other_rules[:5]:
                lines.append(f"  {rule}")
        
        return '\n'.join(lines)


class CNFEncoder:
    """
    Encode rules and states as CNF for SAT solving.
    """
    
    def __init__(self):
        self.var_map: Dict[str, int] = {}
        self.next_var = 1
        self.clauses: List[List[int]] = []
    
    def get_var(self, token: Token) -> int:
        """Get variable number for token."""
        key = token.name
        if key not in self.var_map:
            self.var_map[key] = self.next_var
            self.next_var += 1
        return self.var_map[key]
    
    def add_rule_as_implication(self, rule: TransitionRule, time_step: int = 0):
        """
        Add rule as implication: LHS(t) => RHS(t+1)
        
        In CNF: NOT(LHS) OR RHS
        = (NOT l1 OR NOT l2 OR ... OR r1) AND (NOT l1 OR NOT l2 OR ... OR r2) AND ...
        """
        lhs_vars = [self.get_var(Token(f"{t.name}_t{time_step}")) for t in rule.lhs]
        rhs_vars = [self.get_var(Token(f"{t.name}_t{time_step + 1}")) for t in rule.rhs]
        
        # For each RHS token, add clause: NOT(all LHS) OR rhs
        for rhs_var in rhs_vars:
            clause = [-v for v in lhs_vars] + [rhs_var]
            self.clauses.append(clause)
    
    def add_state_constraint(self, state: Set[Token], time_step: int = 0):
        """Add constraint that state holds at time step."""
        for token in state:
            var = self.get_var(Token(f"{token.name}_t{time_step}"))
            if token.negated:
                self.clauses.append([-var])
            else:
                self.clauses.append([var])
    
    def to_dimacs(self) -> str:
        """Export in DIMACS CNF format."""
        lines = [f"p cnf {self.next_var - 1} {len(self.clauses)}"]
        for clause in self.clauses:
            lines.append(" ".join(str(v) for v in clause) + " 0")
        return "\n".join(lines)


if __name__ == "__main__":
    print("="*70)
    print("Full Production Rule System Demo")
    print("="*70)
    
    # Generate TicTacToe rules
    print("\n--- Generating TicTacToe Rules ---")
    rules = TicTacToeRuleGenerator.generate_all_rules()
    print(f"Generated {len(rules)} rules")
    
    move_rules = [r for r in rules if 'empty' in str(r.consumed)]
    win_rules = [r for r in rules if 'result' in str(r.produced)]
    
    print(f"  Move rules: {len(move_rules)}")
    print(f"  Win rules: {len(win_rules)}")
    
    print("\nSample move rule:")
    print(f"  {move_rules[0]}")
    
    print("\nSample win rule (with catalysts):")
    print(f"  {win_rules[0]}")
    print(f"    Catalysts: {win_rules[0].catalysts}")
    print(f"    Consumed: {win_rules[0].consumed}")
    print(f"    Produced: {win_rules[0].produced}")
    
    # Simulate games
    print("\n--- Simulating Games ---")
    engine = TransitionEngine(rules)
    
    for game_num in range(3):
        print(f"\nGame {game_num + 1}:")
        states = engine.simulate_game()
        
        for i, state in enumerate(states):
            board = state.to_board()
            turn = [t for t in state.tokens if t.name.startswith('turn_')]
            result = [t for t in state.tokens if t.name.startswith('result_')]
            
            print(f"  Move {i}: {board[:3]}|{board[3:6]}|{board[6:9]}", end="")
            if turn:
                print(f"  (turn: {turn[0].name.split('_')[1]})", end="")
            if result:
                print(f"  -> {result[0].name}", end="")
            print()
    
    # Learn from simulated games
    print("\n--- Learning from Game Traces ---")
    learner = TransitionLearner()
    
    for _ in range(100):
        states = engine.simulate_game()
        token_states = [s.tokens for s in states]
        learner.observe_game_trace(token_states)
    
    print(learner.describe_learned_rules())
    
    # CNF encoding
    print("\n--- CNF Encoding Sample ---")
    encoder = CNFEncoder()
    
    # Encode first move rule
    encoder.add_rule_as_implication(move_rules[0], time_step=0)
    
    # Encode initial state
    initial_state = GameState.from_board('000000000', 'X')
    encoder.add_state_constraint(initial_state.tokens, time_step=0)
    
    print(f"Variables: {encoder.next_var - 1}")
    print(f"Clauses: {len(encoder.clauses)}")
    print("\nDIMACS format (first 10 lines):")
    dimacs = encoder.to_dimacs()
    for line in dimacs.split('\n')[:10]:
        print(f"  {line}")
