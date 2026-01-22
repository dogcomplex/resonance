"""
Connect-4: Bridge the Gap Solver
Traces from START position to connect with the solved region.
Uses minimax with the solved states as a lookup table.
"""
import pickle
import os
import time
from collections import deque

state_dir = "./c4_bidir_state"

print("Loading solved states...")
with open(f"{state_dir}/solved.pkl", 'rb') as f:
    solved = pickle.load(f)
print(f"Loaded {len(solved):,} solved states")

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
        """Return list of valid columns (0-6)"""
        moves = []
        board = list(self.board)
        for col in range(7):
            if board[col * 6 + 5] == 0:  # Top cell empty
                moves.append(col)
        return moves
    
    def make_move(self, col):
        """Return new state after making move in column"""
        board = list(self.board)
        # Find lowest empty cell in column
        for row in range(6):
            idx = col * 6 + row
            if board[idx] == 0:
                board[idx] = self.turn
                return C4State(tuple(board), -self.turn)
        return None
    
    def check_win(self):
        """Check if last player won. Returns winner (1/-1) or 0"""
        board = self.board
        # Check all possible 4-in-a-rows
        # Horizontal
        for row in range(6):
            for col in range(4):
                idx = col * 6 + row
                if board[idx] != 0:
                    if (board[idx] == board[idx + 6] == 
                        board[idx + 12] == board[idx + 18]):
                        return board[idx]
        # Vertical
        for col in range(7):
            for row in range(3):
                idx = col * 6 + row
                if board[idx] != 0:
                    if (board[idx] == board[idx + 1] == 
                        board[idx + 2] == board[idx + 3]):
                        return board[idx]
        # Diagonal up-right
        for col in range(4):
            for row in range(3):
                idx = col * 6 + row
                if board[idx] != 0:
                    if (board[idx] == board[idx + 7] == 
                        board[idx + 14] == board[idx + 21]):
                        return board[idx]
        # Diagonal down-right
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
    
    def display(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\n 0 1 2 3 4 5 6")
        for row in range(5, -1, -1):
            print("|", end="")
            for col in range(7):
                print(symbols[self.board[col * 6 + row]], end="|")
            print()
        print("-" * 15)


def minimax(state, depth, alpha, beta, maximizing, cache, stats):
    """
    Minimax with alpha-beta pruning and solved state lookup.
    Returns (value, best_move)
    """
    h = hash(state)
    
    # Check if already solved
    if h in solved:
        stats['solved_hits'] += 1
        return solved[h], None
    
    # Check cache
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
    
    # Depth limit - return heuristic (just 0 for now, we want exact solve)
    if depth == 0:
        stats['depth_limits'] += 1
        return 0, None  # Unknown, treat as draw
    
    moves = state.get_valid_moves()
    # Move ordering: prefer center columns
    moves.sort(key=lambda c: abs(c - 3))
    
    best_move = moves[0]
    
    if maximizing:  # X's turn (turn == 1)
        max_eval = -2
        for move in moves:
            child = state.make_move(move)
            eval_score, _ = minimax(child, depth - 1, alpha, beta, False, cache, stats)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                stats['pruned'] += 1
                break
        cache[h] = max_eval
        return max_eval, best_move
    else:  # O's turn (turn == -1)
        min_eval = 2
        for move in moves:
            child = state.make_move(move)
            eval_score, _ = minimax(child, depth - 1, alpha, beta, True, cache, stats)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                stats['pruned'] += 1
                break
        cache[h] = min_eval
        return min_eval, best_move


def solve_from_start(max_depth=42):
    """Solve Connect-4 from the start position"""
    start = C4State()
    
    print(f"\n{'='*60}")
    print("SOLVING CONNECT-4 FROM START POSITION")
    print(f"{'='*60}")
    print(f"Using {len(solved):,} pre-solved states as lookup table")
    print(f"Max search depth: {max_depth}")
    
    cache = {}
    stats = {
        'nodes': 0,
        'solved_hits': 0,
        'cache_hits': 0,
        'pruned': 0,
        'depth_limits': 0
    }
    
    start_time = time.time()
    
    # Iterative deepening
    for depth in range(1, max_depth + 1):
        cache.clear()
        stats = {k: 0 for k in stats}
        
        iter_start = time.time()
        value, best_move = minimax(start, depth, -2, 2, True, cache, stats)
        iter_time = time.time() - iter_start
        
        print(f"\nDepth {depth:2d}: value={value:+d}, best_move=col{best_move}, "
              f"nodes={stats['nodes']:,}, solved_hits={stats['solved_hits']:,}, "
              f"time={iter_time:.2f}s")
        
        # If we found a definite win/loss, we're done
        if value != 0 or stats['depth_limits'] == 0:
            break
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    outcome = {1: "FIRST PLAYER (X) WINS", 0: "DRAW", -1: "SECOND PLAYER (O) WINS"}
    print(f"🎉 RESULT: {outcome[value]} 🎉")
    print(f"{'='*60}")
    print(f"Optimal first move: Column {best_move}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Final stats: {stats}")
    
    # Add start position to solved
    start_hash = hash(start)
    solved[start_hash] = value
    
    # Save updated solved states
    print(f"\nSaving updated solved states...")
    with open(f"{state_dir}/solved.pkl", 'wb') as f:
        pickle.dump(solved, f)
    print(f"Saved {len(solved):,} solved states")
    
    return value, best_move


def analyze_first_moves():
    """Analyze all 7 possible first moves"""
    start = C4State()
    
    print(f"\n{'='*60}")
    print("ANALYZING ALL FIRST MOVES")
    print(f"{'='*60}")
    
    cache = {}
    stats = {
        'nodes': 0,
        'solved_hits': 0,
        'cache_hits': 0,
        'pruned': 0,
        'depth_limits': 0
    }
    
    results = {}
    for col in range(7):
        print(f"\nAnalyzing first move: Column {col}...")
        child = start.make_move(col)
        
        cache.clear()
        stats = {k: 0 for k in stats}
        
        start_time = time.time()
        # After X plays, it's O's turn (minimizing)
        value, _ = minimax(child, 42, -2, 2, False, cache, stats)
        elapsed = time.time() - start_time
        
        results[col] = value
        outcome = {1: "X wins", 0: "Draw", -1: "O wins"}
        print(f"  Column {col}: {outcome[value]} (value={value:+d}), "
              f"nodes={stats['nodes']:,}, time={elapsed:.2f}s")
    
    print(f"\n{'='*60}")
    print("FIRST MOVE SUMMARY")
    print(f"{'='*60}")
    for col, value in sorted(results.items()):
        outcome = {1: "X wins ✓", 0: "Draw ~", -1: "O wins ✗"}
        print(f"  Column {col}: {outcome[value]}")
    
    optimal = [c for c, v in results.items() if v == max(results.values())]
    print(f"\nOptimal first move(s): {optimal}")
    
    return results


if __name__ == "__main__":
    print("Connect-4 Gap Bridge Solver")
    print("=" * 60)
    
    # First, try to solve from start
    value, best_move = solve_from_start(max_depth=42)
    
    # Then analyze all first moves
    if value == 1:  # If X wins
        analyze_first_moves()
