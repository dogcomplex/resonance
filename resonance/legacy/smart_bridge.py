"""
Connect-4: Smart Bridge Solver
Uses equivalence classes to connect early game to solved late game.
"""
import pickle
import os
import time
import sys

state_dir = "./c4_bidir_state"

# Define C4Features class to match what was pickled
class C4Features:
    """Features for equivalence class - must match original definition"""
    __slots__ = ['x_count', 'o_count', 'x_threats', 'o_threats', 'height_profile', 'turn']
    
    def __init__(self, x_count, o_count, x_threats, o_threats, height_profile, turn):
        self.x_count = x_count
        self.o_count = o_count
        self.x_threats = x_threats
        self.o_threats = o_threats
        self.height_profile = height_profile
        self.turn = turn
    
    def __hash__(self):
        return hash((self.x_count, self.o_count, self.x_threats, self.o_threats, 
                     self.height_profile, self.turn))
    
    def __eq__(self, other):
        return (self.x_count == other.x_count and self.o_count == other.o_count and
                self.x_threats == other.x_threats and self.o_threats == other.o_threats and
                self.height_profile == other.height_profile and self.turn == other.turn)

# Inject into __main__ so pickle can find it
sys.modules['__main__'].C4Features = C4Features

print("Loading data...")
with open(f"{state_dir}/solved.pkl", 'rb') as f:
    solved = pickle.load(f)
print(f"  Solved states: {len(solved):,}")

try:
    with open(f"{state_dir}/equiv.pkl", 'rb') as f:
        equiv = pickle.load(f)
    print(f"  Equivalence classes: {len(equiv):,}")
except Exception as e:
    print(f"  Could not load equiv.pkl: {e}")
    equiv = {}

try:
    with open(f"{state_dir}/equiv_out.pkl", 'rb') as f:
        equiv_outcomes = pickle.load(f)
    print(f"  Equiv classes with outcomes: {len(equiv_outcomes):,}")
except Exception as e:
    print(f"  Could not load equiv_out.pkl: {e}")
    equiv_outcomes = {}

class C4State:
    """Compact Connect-4 state"""
    __slots__ = ['board', 'turn']
    
    def __init__(self, board=None, turn=1):
        self.board = board if board else tuple([0] * 42)
        self.turn = turn
    
    def __hash__(self):
        return hash((self.board, self.turn))
    
    def __eq__(self, other):
        return self.board == other.board and self.turn == other.turn
    
    def copy(self):
        return C4State(self.board, self.turn)
    
    def get_valid_moves(self):
        moves = []
        for col in range(7):
            if self.board[col * 6 + 5] == 0:
                moves.append(col)
        return moves
    
    def make_move(self, col):
        board = list(self.board)
        for row in range(6):
            idx = col * 6 + row
            if board[idx] == 0:
                board[idx] = self.turn
                return C4State(tuple(board), -self.turn)
        return None
    
    def check_win(self):
        board = self.board
        for row in range(6):
            for col in range(4):
                idx = col * 6 + row
                if board[idx] != 0:
                    if (board[idx] == board[idx + 6] == 
                        board[idx + 12] == board[idx + 18]):
                        return board[idx]
        for col in range(7):
            for row in range(3):
                idx = col * 6 + row
                if board[idx] != 0:
                    if (board[idx] == board[idx + 1] == 
                        board[idx + 2] == board[idx + 3]):
                        return board[idx]
        for col in range(4):
            for row in range(3):
                idx = col * 6 + row
                if board[idx] != 0:
                    if (board[idx] == board[idx + 7] == 
                        board[idx + 14] == board[idx + 21]):
                        return board[idx]
        for col in range(4):
            for row in range(3, 6):
                idx = col * 6 + row
                if board[idx] != 0:
                    if (board[idx] == board[idx + 5] == 
                        board[idx + 10] == board[idx + 15]):
                        return board[idx]
        return 0
    
    def is_full(self):
        return all(c != 0 for c in self.board)
    
    def piece_count(self):
        return sum(1 for c in self.board if c != 0)
    
    def get_features(self):
        """Extract equivalence features - returns C4Features object"""
        board = self.board
        x_count = sum(1 for c in board if c == 1)
        o_count = sum(1 for c in board if c == -1)
        
        heights = []
        for col in range(7):
            h = 0
            for row in range(6):
                if board[col * 6 + row] != 0:
                    h = row + 1
            heights.append(h)
        
        def count_threats(player):
            threats = 0
            for row in range(6):
                for col in range(4):
                    window = [board[c * 6 + row] for c in range(col, col + 4)]
                    if window.count(player) == 3 and window.count(0) == 1:
                        threats += 1
            for col in range(7):
                for row in range(3):
                    window = [board[col * 6 + r] for r in range(row, row + 4)]
                    if window.count(player) == 3 and window.count(0) == 1:
                        threats += 1
            for col in range(4):
                for row in range(3):
                    window = [board[(col + i) * 6 + row + i] for i in range(4)]
                    if window.count(player) == 3 and window.count(0) == 1:
                        threats += 1
            for col in range(4):
                for row in range(3, 6):
                    window = [board[(col + i) * 6 + row - i] for i in range(4)]
                    if window.count(player) == 3 and window.count(0) == 1:
                        threats += 1
            return threats
        
        x_threats = count_threats(1)
        o_threats = count_threats(-1)
        
        return C4Features(x_count, o_count, x_threats, o_threats, tuple(sorted(heights)), self.turn)
    
    def display(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\n 0 1 2 3 4 5 6")
        for row in range(5, -1, -1):
            print("|", end="")
            for col in range(7):
                print(symbols[self.board[col * 6 + row]], end="|")
            print()
        print("-" * 15)


def lookup_value(state):
    """Look up state value via hash or equivalence class"""
    h = hash(state)
    
    # Direct lookup
    if h in solved:
        return solved[h], "direct"
    
    # Equivalence class lookup
    features = state.get_features()
    if features in equiv_outcomes:
        return equiv_outcomes[features], "equiv"
    
    return None, None


def minimax_with_equiv(state, depth, alpha, beta, maximizing, cache, stats):
    """Minimax with equivalence class lookup"""
    h = hash(state)
    
    # Check solved (direct or equiv)
    val, method = lookup_value(state)
    if val is not None:
        if method == "direct":
            stats['direct_hits'] += 1
        else:
            stats['equiv_hits'] += 1
        return val, None
    
    # Cache lookup
    if h in cache:
        stats['cache_hits'] += 1
        return cache[h], None
    
    stats['nodes'] += 1
    
    # Terminal checks
    winner = state.check_win()
    if winner != 0:
        return winner, None
    if state.is_full():
        return 0, None
    
    # Depth limit
    if depth == 0:
        stats['depth_limits'] += 1
        return 0, None
    
    moves = state.get_valid_moves()
    moves.sort(key=lambda c: abs(c - 3))  # Center first
    
    best_move = moves[0]
    
    if maximizing:
        max_eval = -2
        for move in moves:
            child = state.make_move(move)
            eval_score, _ = minimax_with_equiv(child, depth - 1, alpha, beta, False, cache, stats)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                stats['pruned'] += 1
                break
        cache[h] = max_eval
        return max_eval, best_move
    else:
        min_eval = 2
        for move in moves:
            child = state.make_move(move)
            eval_score, _ = minimax_with_equiv(child, depth - 1, alpha, beta, True, cache, stats)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                stats['pruned'] += 1
                break
        cache[h] = min_eval
        return min_eval, best_move


def solve_start():
    """Solve from start position"""
    start = C4State()
    
    # First check if start is directly solvable
    val, method = lookup_value(start)
    if val is not None:
        print(f"\n🎉 START POSITION ALREADY SOLVED via {method}! 🎉")
        print(f"Value: {val}")
        return val, None
    
    print(f"\n{'='*60}")
    print("SOLVING FROM START (with equivalence lookup)")
    print(f"{'='*60}")
    
    cache = {}
    stats = {'nodes': 0, 'direct_hits': 0, 'equiv_hits': 0, 
             'cache_hits': 0, 'pruned': 0, 'depth_limits': 0}
    
    start_time = time.time()
    
    # Iterative deepening
    for depth in range(1, 43):
        cache.clear()
        stats = {k: 0 for k in stats}
        
        t0 = time.time()
        value, best_move = minimax_with_equiv(start, depth, -2, 2, True, cache, stats)
        dt = time.time() - t0
        
        total_hits = stats['direct_hits'] + stats['equiv_hits']
        print(f"Depth {depth:2d}: val={value:+d}, move=col{best_move}, "
              f"nodes={stats['nodes']:,}, hits={total_hits:,} "
              f"(d:{stats['direct_hits']}, e:{stats['equiv_hits']}), "
              f"time={dt:.2f}s")
        
        # Found definite result
        if value != 0 or stats['depth_limits'] == 0:
            break
        
        # Timeout protection
        if dt > 300:  # 5 minutes per depth
            print("Timeout - stopping iterative deepening")
            break
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    outcome = {1: "FIRST PLAYER (X) WINS", 0: "DRAW", -1: "SECOND PLAYER (O) WINS"}
    print(f"🎉 RESULT: {outcome[value]} 🎉")
    print(f"{'='*60}")
    print(f"Optimal first move: Column {best_move}")
    print(f"Total time: {total_time:.2f}s")
    
    return value, best_move


if __name__ == "__main__":
    # Check what piece counts we have in equiv_outcomes
    print("\nAnalyzing equivalence classes...")
    
    if equiv_outcomes:
        piece_counts = {}
        for features in equiv_outcomes.keys():
            total = features.x_count + features.o_count
            piece_counts[total] = piece_counts.get(total, 0) + 1
        
        print(f"\nEquiv classes by piece count:")
        for pc in sorted(piece_counts.keys()):
            print(f"  {pc:2d} pieces: {piece_counts[pc]:,} classes")
        
        min_pieces = min(piece_counts.keys()) if piece_counts else "N/A"
        print(f"\nMinimum pieces with solved equiv class: {min_pieces}")
    else:
        print("No equivalence outcomes loaded")
    
    # Now solve
    print("\n" + "="*60)
    value, move = solve_start()
