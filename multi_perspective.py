"""
Multi-Perspective Agent Analysis

Key insight: A single bidirectional wave run contains ALL possible perspectives.
We just need to filter/weight the standing waves differently for each agent.

Perspectives to explore:
1. X-player (control X moves, want W)
2. O-player (control O moves, want L)
3. Position-agent (control only one square, want specific outcome)
4. Backwards agent (blank tokens trying to reach T)
5. Chaos agent (want E - error states in loose universe)
6. Observer (control nothing, predict outcomes)
"""

from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional, Callable
import itertools

# Win conditions
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]

def check_winner(board: tuple) -> Optional[str]:
    for line in WIN_LINES:
        vals = [board[i] for i in line]
        if vals == ['X', 'X', 'X']:
            return 'X'
        if vals == ['O', 'O', 'O']:
            return 'O'
    return None

def is_full(board: tuple) -> bool:
    return '0' not in board

def count_pieces(board: tuple) -> Tuple[int, int]:
    return board.count('X'), board.count('O')

def get_terminal(board: tuple) -> Optional[str]:
    winner = check_winner(board)
    if winner == 'X':
        return 'W'
    elif winner == 'O':
        return 'L'
    elif is_full(board):
        return 'T'
    return None

def display_board(board: tuple) -> str:
    b = board
    return f"{b[0]}|{b[1]}|{b[2]}\n-+-+-\n{b[3]}|{b[4]}|{b[5]}\n-+-+-\n{b[6]}|{b[7]}|{b[8]}"


class UniversalWaveAnalysis:
    """
    Compute the complete wave structure once, then extract any perspective.
    """
    
    def __init__(self, allow_invalid: bool = False):
        self.allow_invalid = allow_invalid
        self.states = self._generate_states()
        self.transitions = self._build_transitions()
        
        # Compute ONCE: all paths from start to all endpoints
        self.all_paths = self._compute_all_paths()
        
    def _generate_states(self) -> Set[tuple]:
        if self.allow_invalid:
            return set(itertools.product('0XO', repeat=9))
        else:
            return self._generate_valid_states()
    
    def _generate_valid_states(self) -> Set[tuple]:
        start = tuple('0' * 9)
        visited = {start}
        frontier = [start]
        
        while frontier:
            board = frontier.pop()
            if get_terminal(board):
                continue
            
            x_count, o_count = count_pieces(board)
            player = 'X' if x_count == o_count else 'O'
            
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
        trans = defaultdict(list)
        
        for board in self.states:
            terminal = get_terminal(board)
            if terminal:
                trans[board].append((f'→{terminal}', (terminal,)))
                continue
            
            x_count, o_count = count_pieces(board)
            
            if self.allow_invalid:
                players = ['X', 'O']
            else:
                players = ['X'] if x_count == o_count else ['O']
            
            for player in players:
                for i in range(9):
                    if board[i] == '0':
                        new_board = list(board)
                        new_board[i] = player
                        new_board = tuple(new_board)
                        if new_board in self.states:
                            trans[board].append((f'{player}@{i}', new_board))
        
        for terminal in ['W', 'L', 'T', 'E']:
            trans[(terminal,)].append(('→END', ('END',)))
        
        return dict(trans)
    
    def _compute_all_paths(self) -> Dict[str, List[List[str]]]:
        """
        Compute ALL paths from start to each endpoint.
        This is the universal wave function of the game.
        """
        start = tuple('0' * 9)
        endpoints = {'W': [], 'L': [], 'T': [], 'E': []}
        
        def dfs(state, path, visited):
            if state == ('END',):
                return
            
            if isinstance(state, tuple) and len(state) == 1 and state[0] in 'WLTE':
                endpoints[state[0]].append(path.copy())
                return
            
            for rule, next_state in self.transitions.get(state, []):
                if next_state not in visited:
                    visited.add(next_state)
                    path.append(rule)
                    dfs(next_state, path, visited)
                    path.pop()
                    visited.remove(next_state)
        
        # This is expensive but gives us everything
        # For TTT it's tractable
        visited = {start}
        dfs(start, [], visited)
        
        return endpoints
    
    def get_paths_to(self, endpoint: str) -> List[List[str]]:
        """Get all paths leading to a specific endpoint"""
        return self.all_paths.get(endpoint, [])
    
    def filter_paths_by_control(self, paths: List[List[str]], 
                                 controlled_moves: Set[str]) -> List[List[str]]:
        """
        Filter paths where the agent ONLY makes controlled moves.
        Non-controlled moves are "environmental" and can be anything.
        """
        # Actually, we want paths where our controlled moves are OPTIMAL
        # This is trickier - let's think about it differently
        return paths  # For now, return all
    
    def compute_move_value(self, move: str, target_endpoint: str) -> Dict:
        """
        Compute how much a move contributes to reaching target endpoint.
        Value = (paths through move to target) / (total paths through move)
        """
        paths_to_target = self.all_paths.get(target_endpoint, [])
        
        paths_with_move = [p for p in paths_to_target if move in p]
        
        # Also count total paths containing this move (to any endpoint)
        total_with_move = 0
        for ep, paths in self.all_paths.items():
            total_with_move += sum(1 for p in paths if move in p)
        
        return {
            'paths_to_target': len(paths_with_move),
            'total_paths': total_with_move,
            'value': len(paths_with_move) / max(1, total_with_move)
        }


class AgentPerspective:
    """
    An agent with a specific perspective on the game.
    """
    
    def __init__(self, name: str, 
                 controlled_moves: Callable[[str], bool],
                 target_endpoint: str,
                 universe: UniversalWaveAnalysis):
        self.name = name
        self.controlled_moves = controlled_moves  # Function: move -> bool
        self.target_endpoint = target_endpoint
        self.universe = universe
        
    def evaluate_move(self, move: str) -> float:
        """How good is this move for achieving my goal?"""
        if not self.controlled_moves(move):
            return 0.0  # Not my move
        
        stats = self.universe.compute_move_value(move, self.target_endpoint)
        return stats['value']
    
    def get_best_moves(self) -> List[Tuple[str, float]]:
        """Get all my moves ranked by value toward goal"""
        all_moves = set()
        for paths in self.universe.all_paths.values():
            for path in paths:
                all_moves.update(path)
        
        my_moves = [m for m in all_moves if self.controlled_moves(m)]
        
        ranked = [(m, self.evaluate_move(m)) for m in my_moves]
        return sorted(ranked, key=lambda x: -x[1])
    
    def summarize(self):
        """Summarize this agent's perspective"""
        paths_to_goal = len(self.universe.get_paths_to(self.target_endpoint))
        total_paths = sum(len(p) for p in self.universe.all_paths.values())
        
        print(f"\n{'='*50}")
        print(f"Agent: {self.name}")
        print(f"Goal: Reach {self.target_endpoint}")
        print(f"Paths to goal: {paths_to_goal} / {total_paths} ({100*paths_to_goal/max(1,total_paths):.1f}%)")
        
        best = self.get_best_moves()[:5]
        print(f"Best moves:")
        for move, value in best:
            print(f"  {move}: {value:.3f}")


def test_standard_agents():
    """Test standard X and O player perspectives"""
    print("=" * 70)
    print("TEST: Standard Player Perspectives")
    print("=" * 70)
    
    # Use smaller path counting to avoid explosion
    # Actually let's use the adversarial analysis we already have
    from ttt_adversarial import AdversarialTTTUniverse
    
    universe = AdversarialTTTUniverse()
    start = tuple('0' * 9)
    
    analysis = universe.analyze_adversarial_waves(start)
    
    print("\nX-Player Perspective (wants W):")
    print(f"  Favorable paths: {analysis['paths']['W']}")
    print(f"  Random win rate: {analysis['random_play']['W']:.1%}")
    print(f"  Perfect play outcome: {analysis['perfect_play']['outcome']}")
    
    print("\nO-Player Perspective (wants L):")
    print(f"  Favorable paths: {analysis['paths']['L']}")
    print(f"  Random win rate: {analysis['random_play']['L']:.1%}")
    print(f"  Perfect play outcome: {analysis['perfect_play']['outcome']}")


def test_position_agents():
    """Test single-position agent perspectives"""
    print("\n" + "=" * 70)
    print("TEST: Position Agent Perspectives")
    print("=" * 70)
    print("Each position 'wants' to be part of a winning line")
    
    from ttt_adversarial import AdversarialTTTUniverse
    universe = AdversarialTTTUniverse()
    
    # For each position, analyze: if I'm X, how often am I in a winning line?
    print("\nPosition value for X (being in the winning line when X wins):")
    
    position_values = {}
    
    for pos in range(9):
        # Count: in how many X-wins is this position X?
        wins_with_pos_x = 0
        total_x_wins = 0
        
        # Check all terminal states
        for board, terminal in universe.terminal_transitions.items():
            if terminal == 'W':  # X won
                total_x_wins += 1
                if board[pos] == 'X':
                    # Check if this position is in the winning line
                    for line in WIN_LINES:
                        if pos in line:
                            vals = [board[i] for i in line]
                            if vals == ['X', 'X', 'X']:
                                wins_with_pos_x += 1
                                break
        
        position_values[pos] = wins_with_pos_x / max(1, total_x_wins)
    
    # Display as grid
    print("\n  Position values (frequency in winning line):")
    print(f"  {position_values[0]:.2f} | {position_values[1]:.2f} | {position_values[2]:.2f}")
    print(f"  -----+------+-----")
    print(f"  {position_values[3]:.2f} | {position_values[4]:.2f} | {position_values[5]:.2f}")
    print(f"  -----+------+-----")
    print(f"  {position_values[6]:.2f} | {position_values[7]:.2f} | {position_values[8]:.2f}")
    
    print("\n  → Center (4) is most often in winning line!")
    print("  → Corners are next")
    print("  → Edges are least likely to be in winning line")


def test_blank_agent():
    """The blank (0) tokens trying to survive until Tie"""
    print("\n" + "=" * 70)
    print("TEST: Blank Agent (0's want to reach Tie)")
    print("=" * 70)
    print("The 'blank' perspective: remain unfilled as long as possible")
    
    from ttt_adversarial import AdversarialTTTUniverse
    universe = AdversarialTTTUniverse()
    
    # In tie games, which positions remain blank longest?
    # Actually in ties, NO positions remain blank!
    # So blank's "goal" is impossible - it always loses
    
    print("\nBlank tokens' fate:")
    print("  - In W (X wins): Blanks 'lose' (game ends with blanks remaining)")
    print("  - In L (O wins): Blanks 'lose' (game ends with blanks remaining)")
    print("  - In T (Tie): Blanks REALLY lose (all get filled!)")
    
    # Count average blanks remaining at each terminal
    blanks_at_terminal = {'W': [], 'L': [], 'T': []}
    
    for board, terminal in universe.terminal_transitions.items():
        blanks = board.count('0')
        blanks_at_terminal[terminal].append(blanks)
    
    for terminal, blanks_list in blanks_at_terminal.items():
        avg = sum(blanks_list) / len(blanks_list) if blanks_list else 0
        print(f"\n  At {terminal}: avg {avg:.2f} blanks remaining, {len(blanks_list)} states")
    
    print("\n  → Blanks 'prefer' quick X wins (more blanks survive)")
    print("  → Blanks 'hate' ties (all blanks die)")
    print("  → This is the 'backwards' perspective!")


def test_chaos_agent():
    """Agent that wants error states (loose gravity only)"""
    print("\n" + "=" * 70)
    print("TEST: Chaos Agent (wants Error states)")
    print("=" * 70)
    
    from ttt_bidirectional import TicTacToeUniverse
    
    # Need loose gravity for errors
    print("\nIn loose gravity universe:")
    loose = TicTacToeUniverse(allow_invalid=True)
    
    start = tuple('0' * 9)
    fwd = loose.forward_expand(start, max_depth=10)
    
    # Find error states reachable
    error_states = [s for s in fwd.keys() 
                    if isinstance(s, tuple) and len(s) == 1 and s[0].startswith('E')]
    
    print(f"  Error states reachable: {len(error_states)}")
    
    # What's the shortest path to an error?
    shortest_error = None
    shortest_depth = float('inf')
    for state in fwd.keys():
        if isinstance(state, tuple) and len(state) == 1 and state[0].startswith('E'):
            depth, path = fwd[state]
            if depth < shortest_depth:
                shortest_depth = depth
                shortest_error = (state, path)
    
    if shortest_error:
        print(f"\n  Shortest path to error: {shortest_error[1]}")
        print(f"  Depth: {shortest_depth}")
        print("\n  → Chaos agent's optimal play: make illegal moves ASAP!")


def count_perspectives():
    """Enumerate all possible meaningful perspectives"""
    print("\n" + "=" * 70)
    print("TEST: Counting All Perspectives")
    print("=" * 70)
    
    # Define perspective dimensions
    control_options = [
        ("X-player", lambda m: m.startswith('X@')),
        ("O-player", lambda m: m.startswith('O@')),
    ]
    
    # Add position-specific control
    for pos in range(9):
        control_options.append(
            (f"Position-{pos}", lambda m, p=pos: m.endswith(f'@{p}'))
        )
    
    # Add piece-at-position control
    for pos in range(9):
        for piece in ['X', 'O']:
            control_options.append(
                (f"{piece}@{pos}-only", lambda m, piece=piece, pos=pos: m == f'{piece}@{pos}')
            )
    
    goal_options = ['W', 'L', 'T', 'E']
    
    total_perspectives = len(control_options) * len(goal_options)
    
    print(f"\nControl options: {len(control_options)}")
    for name, _ in control_options[:5]:
        print(f"  - {name}")
    print(f"  ... and {len(control_options)-5} more")
    
    print(f"\nGoal options: {len(goal_options)}")
    for g in goal_options:
        print(f"  - Want {g}")
    
    print(f"\nTotal perspectives: {total_perspectives}")
    
    # Which are "meaningful"?
    print("\nMeaningful perspectives (agent has SOME control toward goal):")
    
    meaningful = [
        ("X-player → W", "Standard X play"),
        ("O-player → L", "Standard O play"),
        ("X-player → T", "X playing for tie (defensive)"),
        ("O-player → T", "O playing for tie (defensive)"),
        ("Position-4 → W", "Center square agent (wants to be in X's win)"),
        ("X@4-only → W", "Single-move agent (only plays X in center)"),
        ("Any → E", "Chaos agent (wants errors)"),
        ("None → *", "Observer (predicts but doesn't act)"),
    ]
    
    for name, desc in meaningful:
        print(f"  {name}: {desc}")
    
    print(f"\nMany perspectives are degenerate:")
    print("  - O-player → W: O can't CAUSE X to win, only fail to prevent")
    print("  - Position-7 → W: Edge position rarely in winning line")
    print("  - Single-move agents: Very limited influence")


def test_perfect_strategies():
    """Can we extract 'perfect' strategy for each perspective?"""
    print("\n" + "=" * 70)
    print("TEST: Perfect Strategies from Wave Analysis")
    print("=" * 70)
    
    from ttt_adversarial import AdversarialTTTUniverse
    universe = AdversarialTTTUniverse()
    
    # For each perspective, the "perfect strategy" is:
    # At each state, choose the move that maximizes paths to goal
    
    perspectives = [
        ("X → W", 'X', 'W', max),  # X wants to maximize W
        ("X → T", 'X', 'T', max),  # X plays defensively for tie
        ("O → L", 'O', 'L', max),  # O wants to maximize L
        ("O → T", 'O', 'T', max),  # O plays defensively for tie
    ]
    
    print("\nOptimal opening moves by perspective:")
    
    start = tuple('0' * 9)
    
    for name, player, goal, opt_fn in perspectives:
        # Get available moves
        if player == 'X':
            moves = universe.x_transitions.get(start, [])
        else:
            # O doesn't move first in standard TTT
            continue
        
        # Evaluate each move
        move_values = []
        for pos, next_board in moves:
            analysis = universe.analyze_adversarial_waves(next_board)
            goal_key = {'W': 'W', 'L': 'L', 'T': 'T'}[goal]
            paths_to_goal = analysis['paths'][goal_key]
            total_paths = analysis['total_paths']
            value = paths_to_goal / total_paths if total_paths > 0 else 0
            move_values.append((pos, value))
        
        # Find optimal
        if move_values:
            best = opt_fn(move_values, key=lambda x: x[1])
            print(f"\n  {name}:")
            print(f"    Best opening: X@{best[0]} (value: {best[1]:.3f})")
            
            # Show all moves
            for pos, val in sorted(move_values, key=lambda x: -x[1]):
                marker = "←" if pos == best[0] else ""
                print(f"      X@{pos}: {val:.3f} {marker}")


def analyze_derived_knowledge():
    """What can be derived from a single wave run?"""
    print("\n" + "=" * 70)
    print("ANALYSIS: What Can Be Derived from Single Wave Run?")
    print("=" * 70)
    
    print("""
From a single bidirectional wave computation (START ↔ END), we can derive:

1. EXISTENCE QUERIES:
   ✓ Can X win? (Are there paths to W?)
   ✓ Can O win? (Are there paths to L?)
   ✓ Is tie possible? (Are there paths to T?)
   ✓ Are errors reachable? (Loose gravity only)

2. PROBABILITY UNDER RANDOM PLAY:
   ✓ P(W), P(L), P(T) for any starting position
   ✓ Expected outcome distribution
   ✓ Variance in outcomes

3. MOVE VALUES:
   ✓ How much does each move favor each outcome?
   ✓ Differential = paths_to_W - paths_to_L
   ✓ This gives strategic value of moves

4. PERFECT PLAY OUTCOMES:
   ✓ Minimax value at each position
   ✓ Game-theoretic solution
   ✓ Whether game is "solved" (determined)

5. PERSPECTIVE-SPECIFIC STRATEGIES:
   ✓ Best moves for X wanting W
   ✓ Best moves for X wanting T (defensive)
   ✓ Best moves for O wanting L
   ✓ Best moves for O wanting T (defensive)
   ✓ Best moves for position-agents

6. STRUCTURAL INSIGHTS:
   ✓ Which positions are "important"? (Frequency in winning lines)
   ✓ Which moves are "forcing"? (Reduce opponent's options)
   ✓ What is the "complexity" of each position? (Branching factor)

7. LEARNING INSIGHTS:
   ✓ How many games needed to discover each rule?
   ✓ Which patterns emerge early vs late?
   ✓ What is the "sample complexity" of TTT?

What CANNOT be derived:
   ✗ Opponent's actual strategy (only their optimal one)
   ✗ Human psychological factors
   ✗ Specific game history
   ✗ Non-deterministic rules
""")


if __name__ == "__main__":
    test_standard_agents()
    test_position_agents()
    test_blank_agent()
    test_chaos_agent()
    count_perspectives()
    test_perfect_strategies()
    analyze_derived_knowledge()
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Universal Wave = Universal Knowledge")
    print("=" * 70)
    print("""
The single bidirectional wave run IS the "universal wave function" of the game.

Every possible perspective is just a FILTER on this wave:
- X-player filters for X-controlled moves
- O-player filters for O-controlled moves  
- Position-agent filters for position-specific moves
- Goal (W/L/T/E) filters for paths to that endpoint

The wave contains ALL strategies simultaneously:
- Cooperative play (both want same endpoint)
- Adversarial play (want opposite endpoints)
- Chaotic play (want error states)
- Defensive play (want ties)

This is like quantum mechanics:
- The wave function contains all possible outcomes
- "Measurement" (choosing a perspective) collapses to specific behavior
- Interference between perspectives creates game dynamics

NO SEPARATE RUNS NEEDED - just different filters on the same data!
""")
