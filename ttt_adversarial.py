"""
Adversarial Wave Analysis for Tic-Tac-Toe

Key questions:
1. Can we understand adversarial play from a single bidirectional run?
2. What does it mean to "be X" vs "be O"?
3. Are adversarial games two INTERFERING waves with different crunches?
4. Can we extract optimal play from wave analysis?
"""

from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional
import itertools

# Win conditions
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Cols
    (0, 4, 8), (2, 4, 6)              # Diagonals
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


class AdversarialTTTUniverse:
    """
    TTT Universe that tracks X-moves vs O-moves separately.
    This allows us to analyze adversarial dynamics.
    """
    
    def __init__(self):
        self.states = self._generate_valid_states()
        self.x_transitions = {}  # X's moves only
        self.o_transitions = {}  # O's moves only
        self.terminal_transitions = {}  # Board → terminal
        self._build_transitions()
        
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
    
    def _build_transitions(self):
        for board in self.states:
            terminal = get_terminal(board)
            if terminal:
                self.terminal_transitions[board] = terminal
                continue
            
            x_count, o_count = count_pieces(board)
            player = 'X' if x_count == o_count else 'O'
            
            moves = []
            for i in range(9):
                if board[i] == '0':
                    new_board = list(board)
                    new_board[i] = player
                    moves.append((i, tuple(new_board)))
            
            if player == 'X':
                self.x_transitions[board] = moves
            else:
                self.o_transitions[board] = moves
    
    def get_whose_turn(self, board: tuple) -> str:
        x_count, o_count = count_pieces(board)
        return 'X' if x_count == o_count else 'O'
    
    def count_paths_to_outcomes(self, board: tuple, 
                                 memo: Dict = None) -> Dict[str, int]:
        """
        Count number of paths from this board to each outcome (W/L/T).
        This is the key insight: each board has a "destiny distribution".
        """
        if memo is None:
            memo = {}
        
        if board in memo:
            return memo[board]
        
        terminal = get_terminal(board)
        if terminal:
            result = {'W': 0, 'L': 0, 'T': 0}
            result[terminal] = 1
            memo[board] = result
            return result
        
        # Get moves for current player
        player = self.get_whose_turn(board)
        moves = self.x_transitions.get(board, []) if player == 'X' else self.o_transitions.get(board, [])
        
        # Sum paths through all possible moves
        total = {'W': 0, 'L': 0, 'T': 0}
        for _, next_board in moves:
            child_paths = self.count_paths_to_outcomes(next_board, memo)
            for outcome, count in child_paths.items():
                total[outcome] += count
        
        memo[board] = total
        return total
    
    def compute_game_theoretic_value(self, board: tuple,
                                      memo: Dict = None) -> Tuple[str, float]:
        """
        Compute minimax value: what's the outcome with perfect play?
        Returns (outcome, confidence)
        """
        if memo is None:
            memo = {}
        
        if board in memo:
            return memo[board]
        
        terminal = get_terminal(board)
        if terminal:
            # W = +1, T = 0, L = -1 from X's perspective
            value = {'W': 1.0, 'T': 0.0, 'L': -1.0}[terminal]
            memo[board] = (terminal, value)
            return (terminal, value)
        
        player = self.get_whose_turn(board)
        moves = self.x_transitions.get(board, []) if player == 'X' else self.o_transitions.get(board, [])
        
        if not moves:
            memo[board] = ('T', 0.0)
            return ('T', 0.0)
        
        child_values = []
        for _, next_board in moves:
            _, value = self.compute_game_theoretic_value(next_board, memo)
            child_values.append(value)
        
        if player == 'X':
            # X maximizes
            best_value = max(child_values)
        else:
            # O minimizes
            best_value = min(child_values)
        
        if best_value > 0:
            outcome = 'W'
        elif best_value < 0:
            outcome = 'L'
        else:
            outcome = 'T'
        
        memo[board] = (outcome, best_value)
        return (outcome, best_value)
    
    def analyze_adversarial_waves(self, board: tuple):
        """
        Analyze the adversarial wave structure at a board position.
        
        X's wave: Pulls toward W
        O's wave: Pulls toward L
        
        The interference determines the outcome.
        """
        paths = self.count_paths_to_outcomes(board)
        total_paths = sum(paths.values())
        
        # Probabilities under random play
        p_w = paths['W'] / total_paths if total_paths > 0 else 0
        p_l = paths['L'] / total_paths if total_paths > 0 else 0
        p_t = paths['T'] / total_paths if total_paths > 0 else 0
        
        # Game-theoretic value under perfect play
        outcome, value = self.compute_game_theoretic_value(board)
        
        return {
            'paths': paths,
            'total_paths': total_paths,
            'random_play': {'W': p_w, 'L': p_l, 'T': p_t},
            'perfect_play': {'outcome': outcome, 'value': value},
        }


def test_adversarial_from_start():
    """Analyze adversarial dynamics from empty board"""
    print("=" * 70)
    print("TEST: Adversarial Wave Analysis from Start")
    print("=" * 70)
    
    universe = AdversarialTTTUniverse()
    print(f"\nUniverse has {len(universe.states)} states")
    print(f"  X-move states: {len(universe.x_transitions)}")
    print(f"  O-move states: {len(universe.o_transitions)}")
    print(f"  Terminal states: {len(universe.terminal_transitions)}")
    
    # Analyze from empty board
    start = tuple('0' * 9)
    analysis = universe.analyze_adversarial_waves(start)
    
    print(f"\nFrom empty board:")
    print(f"  Total possible games: {analysis['total_paths']}")
    print(f"  Paths to X-win (W): {analysis['paths']['W']}")
    print(f"  Paths to O-win (L): {analysis['paths']['L']}")
    print(f"  Paths to Tie (T): {analysis['paths']['T']}")
    
    print(f"\n  Under random play:")
    print(f"    P(X wins) = {analysis['random_play']['W']:.3f}")
    print(f"    P(O wins) = {analysis['random_play']['L']:.3f}")
    print(f"    P(Tie)    = {analysis['random_play']['T']:.3f}")
    
    print(f"\n  Under perfect play:")
    print(f"    Outcome: {analysis['perfect_play']['outcome']}")
    print(f"    Value: {analysis['perfect_play']['value']}")
    
    print("\n  → X has more winning paths, but perfect play leads to TIE")
    print("  → The adversarial interference cancels out X's advantage!")


def test_adversarial_at_positions():
    """Analyze how adversarial waves look at different board states"""
    print("\n" + "=" * 70)
    print("TEST: Adversarial Waves at Different Positions")
    print("=" * 70)
    
    universe = AdversarialTTTUniverse()
    
    # Test positions
    test_boards = [
        tuple('000000000'),  # Empty
        tuple('X00000000'),  # X in corner
        tuple('X000O0000'),  # X corner, O center
        tuple('XXX000OO0'),  # X about to win
        tuple('XO0X00O00'),  # Mid-game
    ]
    
    for board in test_boards:
        print(f"\n{'='*40}")
        print(display_board(board))
        print(f"{'='*40}")
        
        terminal = get_terminal(board)
        if terminal:
            print(f"  Terminal state: {terminal}")
            continue
        
        analysis = universe.analyze_adversarial_waves(board)
        
        print(f"  Turn: {universe.get_whose_turn(board)}")
        print(f"  Total paths: {analysis['total_paths']}")
        print(f"  Random play: W={analysis['random_play']['W']:.2f}, "
              f"L={analysis['random_play']['L']:.2f}, T={analysis['random_play']['T']:.2f}")
        print(f"  Perfect play: {analysis['perfect_play']['outcome']} "
              f"(value={analysis['perfect_play']['value']:.1f})")
        
        # What's the "wave differential"?
        w_pull = analysis['paths']['W'] / max(1, analysis['total_paths'])
        l_pull = analysis['paths']['L'] / max(1, analysis['total_paths'])
        wave_differential = w_pull - l_pull
        print(f"  Wave differential (X pull - O pull): {wave_differential:+.3f}")


def test_move_as_wave_steering():
    """Show how each move 'steers' the wave toward different outcomes"""
    print("\n" + "=" * 70)
    print("TEST: Moves as Wave Steering")
    print("=" * 70)
    
    universe = AdversarialTTTUniverse()
    
    # From empty board, analyze each X opening move
    start = tuple('0' * 9)
    
    print("\nX's opening moves - how do they steer the wave?")
    print(f"{'Move':<8} | {'Paths W':>8} | {'Paths L':>8} | {'Paths T':>8} | {'Differential':>12} | {'Perfect':>8}")
    print("-" * 70)
    
    x_moves = universe.x_transitions[start]
    move_analysis = []
    
    for pos, next_board in x_moves:
        analysis = universe.analyze_adversarial_waves(next_board)
        differential = (analysis['paths']['W'] - analysis['paths']['L']) / max(1, analysis['total_paths'])
        
        move_analysis.append({
            'pos': pos,
            'paths': analysis['paths'],
            'differential': differential,
            'perfect': analysis['perfect_play']['outcome']
        })
    
    # Sort by differential (best moves first)
    for m in sorted(move_analysis, key=lambda x: -x['differential']):
        print(f"X@{m['pos']:<6} | {m['paths']['W']:>8} | {m['paths']['L']:>8} | "
              f"{m['paths']['T']:>8} | {m['differential']:>+12.3f} | {m['perfect']:>8}")
    
    print("\n  → Center (4) has highest wave differential!")
    print("  → Corners (0,2,6,8) are next best")
    print("  → Edges (1,3,5,7) are worst")
    print("  → This matches known TTT strategy!")


def test_wave_interference_visualization():
    """Visualize how X and O waves interfere"""
    print("\n" + "=" * 70)
    print("TEST: Wave Interference Visualization")
    print("=" * 70)
    
    universe = AdversarialTTTUniverse()
    
    # Track wave strength at each depth
    def trace_wave_by_depth(board, player_perspective, depth=0, max_depth=9, memo=None):
        """Trace how 'favorable' states are distributed by depth"""
        if memo is None:
            memo = {}
        
        key = (board, depth)
        if key in memo:
            return memo[key]
        
        terminal = get_terminal(board)
        if terminal:
            # From X's perspective: W=+1, T=0, L=-1
            value = {'W': 1, 'T': 0, 'L': -1}[terminal]
            result = {depth: [value]}
            memo[key] = result
            return result
        
        if depth >= max_depth:
            return {depth: [0]}
        
        whose_turn = universe.get_whose_turn(board)
        moves = universe.x_transitions.get(board, []) if whose_turn == 'X' else universe.o_transitions.get(board, [])
        
        combined = defaultdict(list)
        for _, next_board in moves:
            child_result = trace_wave_by_depth(next_board, player_perspective, depth + 1, max_depth, memo)
            for d, values in child_result.items():
                combined[d].extend(values)
        
        memo[key] = dict(combined)
        return dict(combined)
    
    start = tuple('0' * 9)
    trace = trace_wave_by_depth(start, 'X', max_depth=10)
    
    print("\nWave strength by game depth (from X's perspective):")
    print(f"{'Depth':<6} | {'# Outcomes':>10} | {'Avg Value':>10} | {'X Favorable':>12} | {'O Favorable':>12}")
    print("-" * 65)
    
    for depth in sorted(trace.keys()):
        values = trace[depth]
        avg = sum(values) / len(values)
        x_fav = sum(1 for v in values if v > 0) / len(values)
        o_fav = sum(1 for v in values if v < 0) / len(values)
        
        print(f"{depth:<6} | {len(values):>10} | {avg:>+10.3f} | {x_fav:>11.1%} | {o_fav:>11.1%}")
    
    print("\n  → At depth 5 (earliest X win possible), X-favorable outcomes appear")
    print("  → At depth 6, O-favorable outcomes appear")
    print("  → Deeper depths show more ties as paths narrow")


def test_single_run_extrapolation():
    """Can we extract adversarial understanding from our original single run?"""
    print("\n" + "=" * 70)
    print("TEST: Extracting Adversarial Info from Single Bidirectional Run")
    print("=" * 70)
    
    # Import our original bidirectional analysis
    from ttt_bidirectional import TicTacToeUniverse, analyze_rule_discovery
    
    universe = TicTacToeUniverse(allow_invalid=False)
    start = tuple('0' * 9)
    end = ('END',)
    
    waves, fwd, bwd = universe.find_standing_waves(start, end, max_depth=12)
    
    print(f"\nFrom single bidirectional run:")
    print(f"  Total standing waves: {len(waves)}")
    
    # Separate waves by which terminal they pass through
    waves_through_W = [w for w in waves if '→W' in w['forward_path'] + w['backward_path']]
    waves_through_L = [w for w in waves if '→L' in w['forward_path'] + w['backward_path']]
    waves_through_T = [w for w in waves if '→T' in w['forward_path'] + w['backward_path']]
    
    print(f"  Waves through W (X wins): {len(waves_through_W)}")
    print(f"  Waves through L (O wins): {len(waves_through_L)}")
    print(f"  Waves through T (Tie):    {len(waves_through_T)}")
    
    # Analyze rules specific to each outcome
    print("\nRules associated with X winning:")
    w_rules = analyze_rule_discovery(waves_through_W)
    for rule, stats in sorted(w_rules.items(), key=lambda x: -x[1]['count'])[:8]:
        if not rule.startswith('→'):
            print(f"  {rule}: {stats['count']}")
    
    print("\nRules associated with O winning:")
    l_rules = analyze_rule_discovery(waves_through_L)
    for rule, stats in sorted(l_rules.items(), key=lambda x: -x[1]['count'])[:8]:
        if not rule.startswith('→'):
            print(f"  {rule}: {stats['count']}")
    
    # Compare: which moves favor which player?
    print("\n" + "-" * 50)
    print("Move differential (appearances in W-paths minus L-paths):")
    
    all_moves = set(w_rules.keys()) | set(l_rules.keys())
    move_diff = []
    for move in all_moves:
        if move.startswith('→'):
            continue
        w_count = w_rules.get(move, {'count': 0})['count']
        l_count = l_rules.get(move, {'count': 0})['count']
        diff = w_count - l_count
        move_diff.append((move, diff, w_count, l_count))
    
    print(f"{'Move':<10} | {'In W-paths':>10} | {'In L-paths':>10} | {'Difference':>10}")
    print("-" * 50)
    for move, diff, w, l in sorted(move_diff, key=lambda x: -x[1]):
        print(f"{move:<10} | {w:>10} | {l:>10} | {diff:>+10}")
    
    print("\n  → X moves appear more in W-paths (expected)")
    print("  → O moves appear more in L-paths (expected)")
    print("  → The RATIO tells us which moves are 'stronger' for each player!")
    print("\n  → YES! We can extract adversarial info from single run!")


if __name__ == "__main__":
    test_adversarial_from_start()
    test_adversarial_at_positions()
    test_move_as_wave_steering()
    test_wave_interference_visualization()
    test_single_run_extrapolation()
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS: Adversarial Waves")
    print("=" * 70)
    print("""
1. TWO WAVES, TWO CRUNCHES:
   - X's wave pulls toward W (X wins)
   - O's wave pulls toward L (O wins)
   - They INTERFERE in the state space
   - Tie (T) emerges where they cancel out!

2. FROM SINGLE RUN:
   - We CAN extract adversarial dynamics!
   - Count which paths go through W vs L
   - Move "strength" = differential in path counts
   - No need for separate X and O runs!

3. WAVE STEERING:
   - Each move "steers" the combined wave
   - X wants moves that increase W-paths, decrease L-paths
   - O wants the opposite
   - Center square (4) maximally steers toward balance (hence ties)

4. PERFECT PLAY = INTERFERENCE CANCELLATION:
   - Random play: X wins more often (more paths to W)
   - Perfect play: Tie (waves perfectly cancel)
   - The adversary's "counter-wave" neutralizes your wave!

5. NO NEED FOR PILOT WAVES PER PLAYER:
   - Single bidirectional run captures everything
   - Separate the waves by which crunch they reach
   - The "universe" already contains both perspectives!
""")
