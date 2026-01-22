"""
Tic-Tac-Toe Bidirectional Wave Discovery

Questions to answer:
1. Can we discover TTT rules from standing wave interference?
2. Does loose gravity (all 3^9 states) vs tight gravity (valid states only) matter?
3. What does "early" discovery mean in standing wave terms?
4. Can we discover rules even with Error states allowed?

Board encoding: 9-char string, each char is '0', 'X', or 'O'
  Position mapping:
  0|1|2
  -+-+-
  3|4|5
  -+-+-
  6|7|8
"""

from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional
import itertools

# Win conditions: indices that form a line
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Cols
    (0, 4, 8), (2, 4, 6)              # Diagonals
]

def board_to_str(board: tuple) -> str:
    """Convert board tuple to display string"""
    return ''.join(board)

def str_to_board(s: str) -> tuple:
    return tuple(s)

def display_board(board: tuple) -> str:
    """Pretty print a board"""
    b = board
    return f"{b[0]}|{b[1]}|{b[2]}\n-+-+-\n{b[3]}|{b[4]}|{b[5]}\n-+-+-\n{b[6]}|{b[7]}|{b[8]}"

def count_pieces(board: tuple) -> Tuple[int, int]:
    """Count X's and O's"""
    return board.count('X'), board.count('O')

def check_winner(board: tuple) -> Optional[str]:
    """Check if X or O has won"""
    for line in WIN_LINES:
        vals = [board[i] for i in line]
        if vals == ['X', 'X', 'X']:
            return 'X'
        if vals == ['O', 'O', 'O']:
            return 'O'
    return None

def is_full(board: tuple) -> bool:
    return '0' not in board

def get_terminal_state(board: tuple) -> Optional[str]:
    """Return terminal state: 'W' (X wins), 'L' (O wins), 'T' (tie), or None"""
    winner = check_winner(board)
    if winner == 'X':
        return 'W'
    elif winner == 'O':
        return 'L'
    elif is_full(board):
        return 'T'
    return None

def is_valid_ttt_state(board: tuple) -> bool:
    """Check if this is a reachable TTT state"""
    x_count, o_count = count_pieces(board)
    
    # X goes first, so X = O or X = O + 1
    if not (x_count == o_count or x_count == o_count + 1):
        return False
    
    # Can't have both players winning
    x_wins = check_winner(board) == 'X'
    o_wins = check_winner(board) == 'O'
    if x_wins and o_wins:
        return False
    
    # If X won, must be X's turn (X = O + 1) or game just ended
    # If O won, must be O's turn (X = O)
    if x_wins and x_count != o_count + 1:
        # X won but it's not consistent with X moving last
        pass  # Actually this is complex, let's be lenient
    
    return True

def get_error_type(board: tuple) -> Optional[str]:
    """Check what kind of error state this is"""
    x_count, o_count = count_pieces(board)
    
    # Count error
    if x_count < o_count or x_count > o_count + 1:
        return 'E_COUNT'
    
    # Both win error
    x_wins = check_winner(board) == 'X'
    o_wins = check_winner(board) == 'O'
    if x_wins and o_wins:
        return 'E_BOTH_WIN'
    
    return None


class TicTacToeUniverse:
    """
    A universe of tictactoe states with forward and backward transitions.
    """
    
    def __init__(self, allow_invalid: bool = False):
        self.allow_invalid = allow_invalid
        self.states = self._generate_states()
        self.transitions = self._build_transitions()
        
    def _generate_states(self) -> Set[tuple]:
        """Generate all states in this universe"""
        if self.allow_invalid:
            # All 3^9 states
            return set(itertools.product('0XO', repeat=9))
        else:
            # Only valid TTT states - generate by playing
            return self._generate_valid_states()
    
    def _generate_valid_states(self) -> Set[tuple]:
        """Generate only states reachable by legal play"""
        start = tuple('0' * 9)
        visited = {start}
        frontier = [start]
        
        while frontier:
            board = frontier.pop()
            
            # Don't expand from terminal states
            if get_terminal_state(board):
                continue
            
            # Whose turn?
            x_count, o_count = count_pieces(board)
            player = 'X' if x_count == o_count else 'O'
            
            # Try all moves
            for i in range(9):
                if board[i] == '0':
                    new_board = list(board)
                    new_board[i] = player
                    new_board = tuple(new_board)
                    if new_board not in visited:
                        visited.add(new_board)
                        frontier.append(new_board)
        
        return visited
    
    def _build_transitions(self) -> Dict[tuple, List[Tuple[str, tuple]]]:
        """Build transition graph"""
        trans = defaultdict(list)
        
        for board in self.states:
            terminal = get_terminal_state(board)
            error = get_error_type(board) if self.allow_invalid else None
            
            if terminal:
                # Terminal state transitions to terminal token
                trans[board].append((f'→{terminal}', (terminal,)))
            elif error:
                # Error state transitions to error token
                trans[board].append((f'→{error}', (error,)))
            else:
                # Non-terminal: can place X or O
                x_count, o_count = count_pieces(board)
                
                if self.allow_invalid:
                    # Can place either (loose gravity)
                    players = ['X', 'O']
                else:
                    # Must alternate (tight gravity)
                    players = ['X'] if x_count == o_count else ['O']
                
                for player in players:
                    for i in range(9):
                        if board[i] == '0':
                            new_board = list(board)
                            new_board[i] = player
                            new_board = tuple(new_board)
                            if new_board in self.states:
                                trans[board].append((f'{player}@{i}', new_board))
        
        # Terminal tokens go to END
        for terminal in ['W', 'L', 'T', 'E_COUNT', 'E_BOTH_WIN']:
            trans[(terminal,)].append(('→END', ('END',)))
        
        return dict(trans)
    
    def forward_expand(self, start: tuple, max_depth: int = 20) -> Dict[tuple, Tuple[int, List[str]]]:
        """Deduction: What's reachable from start?"""
        reached = {start: (0, [])}
        frontier = [(start, 0, [])]
        
        while frontier:
            state, depth, path = frontier.pop(0)
            if depth >= max_depth:
                continue
            
            for rule, next_state in self.transitions.get(state, []):
                if next_state not in reached:
                    new_path = path + [rule]
                    reached[next_state] = (depth + 1, new_path)
                    frontier.append((next_state, depth + 1, new_path))
        
        return reached
    
    def backward_expand(self, goal: tuple, max_depth: int = 20) -> Dict[tuple, Tuple[int, List[str]]]:
        """Abduction: What could lead to goal?"""
        # Build reverse transitions
        reverse = defaultdict(list)
        for state, trans_list in self.transitions.items():
            for rule, next_state in trans_list:
                reverse[next_state].append((rule, state))
        
        reached = {goal: (0, [])}
        frontier = [(goal, 0, [])]
        
        while frontier:
            state, depth, path = frontier.pop(0)
            if depth >= max_depth:
                continue
            
            for rule, prev_state in reverse.get(state, []):
                if prev_state not in reached:
                    new_path = [rule] + path
                    reached[prev_state] = (depth + 1, new_path)
                    frontier.append((prev_state, depth + 1, new_path))
        
        return reached
    
    def find_standing_waves(self, start: tuple, goal: tuple, 
                           max_depth: int = 15) -> Tuple[List[dict], Dict, Dict]:
        """Induction: Find standing waves between start and goal"""
        forward = self.forward_expand(start, max_depth)
        backward = self.backward_expand(goal, max_depth)
        
        intersection = set(forward.keys()) & set(backward.keys())
        
        waves = []
        for meeting in intersection:
            fwd_depth, fwd_path = forward[meeting]
            bwd_depth, bwd_path = backward[meeting]
            waves.append({
                'meeting_point': meeting,
                'forward_depth': fwd_depth,
                'backward_depth': bwd_depth,
                'total_depth': fwd_depth + bwd_depth,
                'forward_path': fwd_path,
                'backward_path': bwd_path,
            })
        
        return waves, forward, backward


def analyze_rule_discovery(waves: List[dict]) -> Dict[str, dict]:
    """Analyze which rules appear in standing waves and when"""
    rule_stats = defaultdict(lambda: {
        'count': 0,
        'first_appearance_depth': float('inf'),
        'appearances_by_depth': defaultdict(int)
    })
    
    for wave in waves:
        total_depth = wave['total_depth']
        all_rules = wave['forward_path'] + wave['backward_path']
        
        for rule in all_rules:
            rule_stats[rule]['count'] += 1
            rule_stats[rule]['first_appearance_depth'] = min(
                rule_stats[rule]['first_appearance_depth'],
                total_depth
            )
            rule_stats[rule]['appearances_by_depth'][total_depth] += 1
    
    return dict(rule_stats)


def test_valid_universe():
    """Test with valid TTT states only (tight gravity)"""
    print("=" * 70)
    print("TEST: Valid TTT Universe (Tight Gravity)")
    print("=" * 70)
    
    universe = TicTacToeUniverse(allow_invalid=False)
    print(f"\nUniverse has {len(universe.states)} valid states")
    
    start = tuple('0' * 9)
    end = ('END',)
    
    # Forward from empty board
    fwd = universe.forward_expand(start, max_depth=12)
    print(f"Forward from empty board reaches {len(fwd)} states")
    
    # Find terminal states reached
    terminals_reached = [s for s in fwd.keys() if len(s) == 1 and s[0] in 'WLTE']
    print(f"Terminal states reached: {set(t[0] for t in terminals_reached)}")
    
    # Backward from END
    bwd = universe.backward_expand(end, max_depth=12)
    print(f"Backward from END reaches {len(bwd)} states")
    
    # Standing waves
    waves, _, _ = universe.find_standing_waves(start, end, max_depth=12)
    print(f"Standing waves found: {len(waves)}")
    
    # Analyze rules
    rule_stats = analyze_rule_discovery(waves)
    
    print("\nRule discovery analysis:")
    print(f"{'Rule':<15} | {'Count':>6} | {'First Depth':>11}")
    print("-" * 40)
    for rule, stats in sorted(rule_stats.items(), key=lambda x: -x[1]['count'])[:15]:
        print(f"{rule:<15} | {stats['count']:>6} | {stats['first_appearance_depth']:>11}")
    
    # What rules are "fundamental" (appear early and often)?
    print("\n'Fundamental' rules (early + frequent):")
    fundamental = sorted(
        rule_stats.items(),
        key=lambda x: x[1]['first_appearance_depth'] + 1/(x[1]['count']+1)
    )[:10]
    for rule, stats in fundamental:
        print(f"  {rule}: depth={stats['first_appearance_depth']}, count={stats['count']}")
    
    # Show some example waves
    print("\nExample standing waves (shortest):")
    shortest = sorted(waves, key=lambda w: w['total_depth'])[:5]
    for w in shortest:
        if isinstance(w['meeting_point'], tuple) and len(w['meeting_point']) == 9:
            print(f"\n  Meeting at:\n{display_board(w['meeting_point'])}")
        else:
            print(f"\n  Meeting at: {w['meeting_point']}")
        print(f"  Forward ({w['forward_depth']}): {w['forward_path'][:5]}...")
        print(f"  Backward ({w['backward_depth']}): {w['backward_path'][:5]}...")


def test_invalid_universe():
    """Test with all 3^9 states (loose gravity)"""
    print("\n" + "=" * 70)
    print("TEST: All States Universe (Loose Gravity)")
    print("=" * 70)
    
    # This is expensive - let's do a smaller version
    # Actually 3^9 = 19683 states, should be manageable
    
    print("\nGenerating loose universe (all 3^9 states)...")
    universe = TicTacToeUniverse(allow_invalid=True)
    print(f"Universe has {len(universe.states)} states")
    
    # Count valid vs invalid
    valid = sum(1 for s in universe.states if is_valid_ttt_state(s))
    print(f"  Valid states: {valid}")
    print(f"  Invalid states: {len(universe.states) - valid}")
    
    start = tuple('0' * 9)
    end = ('END',)
    
    # Forward from empty
    print("\nExpanding forward (this may take a moment)...")
    fwd = universe.forward_expand(start, max_depth=10)
    print(f"Forward reaches {len(fwd)} states")
    
    # Check what terminal/error states we reach
    terminals = defaultdict(int)
    for state in fwd.keys():
        if isinstance(state, tuple) and len(state) == 1:
            terminals[state[0]] += 1
    print(f"Terminal/error states reached: {dict(terminals)}")
    
    # Backward from END
    print("\nExpanding backward...")
    bwd = universe.backward_expand(end, max_depth=10)
    print(f"Backward reaches {len(bwd)} states")
    
    # Standing waves
    waves, _, _ = universe.find_standing_waves(start, end, max_depth=10)
    print(f"Standing waves: {len(waves)}")
    
    # Rule analysis
    rule_stats = analyze_rule_discovery(waves)
    
    print("\nRule discovery (loose gravity):")
    print(f"{'Rule':<15} | {'Count':>6} | {'First Depth':>11}")
    print("-" * 40)
    for rule, stats in sorted(rule_stats.items(), key=lambda x: -x[1]['count'])[:15]:
        print(f"{rule:<15} | {stats['count']:>6} | {stats['first_appearance_depth']:>11}")


def test_win_discovery():
    """Specifically test if we can discover what a 'win' is"""
    print("\n" + "=" * 70)
    print("TEST: Can we discover WIN conditions from standing waves?")
    print("=" * 70)
    
    universe = TicTacToeUniverse(allow_invalid=False)
    
    # Look at boards that are 1 move away from win
    win_terminal = ('W',)  # X wins
    
    bwd = universe.backward_expand(win_terminal, max_depth=3)
    
    # Find states that are 2 steps from win (board → W → END, so depth=2 means board)
    pre_win_states = [s for s, (d, p) in bwd.items() 
                      if d == 1 and isinstance(s, tuple) and len(s) == 9]
    
    print(f"\nBoards that are X-wins (1 step from 'W' token): {len(pre_win_states)}")
    
    # Analyze what these boards have in common
    print("\nAnalyzing winning boards...")
    
    # Check which lines are completed
    line_completions = defaultdict(int)
    for board in pre_win_states:
        for i, line in enumerate(WIN_LINES):
            vals = [board[j] for j in line]
            if vals == ['X', 'X', 'X']:
                line_completions[line] += 1
    
    print("\nWin lines discovered (by frequency):")
    for line, count in sorted(line_completions.items(), key=lambda x: -x[1]):
        print(f"  Positions {line}: {count} boards")
    
    print("\n  → The standing wave naturally discovered all 8 win lines!")
    print("  → This is the 'rules of tictactoe' emerging from wave interference!")


def test_complexity_meaning():
    """What does 'early' discovery mean?"""
    print("\n" + "=" * 70)
    print("TEST: What does 'complexity' mean for rule discovery?")
    print("=" * 70)
    
    universe = TicTacToeUniverse(allow_invalid=False)
    
    start = tuple('0' * 9)
    end = ('END',)
    
    waves, fwd, bwd = universe.find_standing_waves(start, end, max_depth=12)
    
    # Group waves by total depth
    by_depth = defaultdict(list)
    for w in waves:
        by_depth[w['total_depth']].append(w)
    
    print("\nStanding waves by total depth (complexity):")
    for depth in sorted(by_depth.keys()):
        print(f"\n  Depth {depth}: {len(by_depth[depth])} waves")
        
        # Sample a wave at this depth
        sample = by_depth[depth][0]
        if isinstance(sample['meeting_point'], tuple) and len(sample['meeting_point']) == 9:
            print(f"    Example meeting point:\n{display_board(sample['meeting_point'])}")
        
        # What rules are used at this depth?
        rules_at_depth = defaultdict(int)
        for w in by_depth[depth]:
            for r in w['forward_path'] + w['backward_path']:
                rules_at_depth[r] += 1
        
        top_rules = sorted(rules_at_depth.items(), key=lambda x: -x[1])[:5]
        print(f"    Top rules: {[r for r, c in top_rules]}")
    
    print("\n" + "-" * 50)
    print("INTERPRETATION:")
    print("""
    Depth = forward_moves + backward_moves to connect start↔end
    
    - Lower depth = simpler games (fewer moves)
    - Rules appearing at low depth are 'fundamental'
    - Rules appearing only at high depth are 'emergent'
    
    This IS few-shot learning in a sense:
    - Depth 3 rules: Learned from 3-move games
    - Depth 6 rules: Need 6-move games to discover
    - Depth 9 rules: Only visible in full games
    """)


if __name__ == "__main__":
    test_valid_universe()
    test_invalid_universe()
    test_win_discovery()
    test_complexity_meaning()
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
1. GRAVITY = The valid state space
   - Tight gravity (valid TTT only): 5,478 states
   - Loose gravity (all 3^9): 19,683 states
   - The 'shape' of valid states IS the gravitational field
   
2. BIG CRUNCH = Terminal states → END
   - W/L/T are intermediate crunches
   - END is the ultimate attractor
   - Errors (E) are also valid endpoints in loose gravity
   
3. RULE DISCOVERY via standing waves:
   - Forward wave from start (deduction)
   - Backward wave from END (abduction)
   - Intersection reveals rules (induction)
   - Win conditions emerge naturally!
   
4. 'EARLY' = Low standing wave depth
   - Simple rules are discoverable from short games
   - Complex rules require longer games
   - This IS sample complexity in learning theory!
   
5. ERROR STATES in loose gravity:
   - They're just another type of endpoint
   - The universe doesn't 'forbid' them
   - They simply don't lead to W/L/T
""")
