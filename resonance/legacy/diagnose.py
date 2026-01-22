"""
Diagnostic: What did we actually solve?
"""
import pickle

state_dir = "./c4_bidir_state"

print("Loading solved states...")
with open(f"{state_dir}/solved.pkl", 'rb') as f:
    solved = pickle.load(f)
print(f"Total solved: {len(solved):,}")

# The solved dict maps hash -> value
# We can't directly get piece count from hash
# But we CAN check if specific early-game positions are solved

class C4State:
    def __init__(self, board=None, turn=1):
        self.board = board if board else tuple([0] * 42)
        self.turn = turn
    
    def __hash__(self):
        return hash((self.board, self.turn))
    
    def make_move(self, col):
        board = list(self.board)
        for row in range(6):
            idx = col * 6 + row
            if board[idx] == 0:
                board[idx] = self.turn
                return C4State(tuple(board), -self.turn)
        return None
    
    def piece_count(self):
        return sum(1 for c in self.board if c != 0)

# Check start position
start = C4State()
print(f"\nStart position (0 pieces): {hash(start) in solved}")

# Check all 1-piece positions (after X's first move)
print("\n1-piece positions (X's first move):")
for col in range(7):
    state = start.make_move(col)
    h = hash(state)
    if h in solved:
        print(f"  Col {col}: SOLVED, value={solved[h]}")
    else:
        print(f"  Col {col}: not solved")

# Check some 2-piece positions
print("\n2-piece positions (sample - X col3, then O moves):")
state1 = start.make_move(3)  # X plays center
for col in range(7):
    state2 = state1.make_move(col)
    h = hash(state2)
    if h in solved:
        print(f"  X:3, O:{col}: SOLVED, value={solved[h]}")
    else:
        print(f"  X:3, O:{col}: not solved")

# Try to find the MINIMUM piece count in solved
# We'll do this by checking positions at each depth
print("\nSearching for minimum solved piece count...")

from collections import deque

def bfs_find_solved(max_depth=20):
    """BFS from start to find first solved position"""
    visited = set()
    queue = deque([(start, 0)])  # (state, depth)
    visited.add(hash(start))
    
    nodes_by_depth = {}
    solved_by_depth = {}
    
    while queue:
        state, depth = queue.popleft()
        
        if depth > max_depth:
            break
        
        nodes_by_depth[depth] = nodes_by_depth.get(depth, 0) + 1
        
        h = hash(state)
        if h in solved:
            solved_by_depth[depth] = solved_by_depth.get(depth, 0) + 1
            if depth <= 10:  # Report early finds
                print(f"  Found solved at depth {depth}!")
        
        # Expand children
        if depth < max_depth:
            board = list(state.board)
            for col in range(7):
                if board[col * 6 + 5] == 0:  # Column not full
                    child = state.make_move(col)
                    ch = hash(child)
                    if ch not in visited:
                        visited.add(ch)
                        queue.append((child, depth + 1))
        
        # Progress
        if len(visited) % 100000 == 0:
            print(f"  Visited {len(visited):,} states...")
    
    print(f"\nBFS Summary (up to depth {max_depth}):")
    for d in sorted(nodes_by_depth.keys()):
        s = solved_by_depth.get(d, 0)
        n = nodes_by_depth[d]
        pct = 100 * s / n if n > 0 else 0
        print(f"  Depth {d:2d}: {n:>10,} nodes, {s:>8,} solved ({pct:.1f}%)")

# Run BFS
bfs_find_solved(max_depth=15)

# Also check: what values do we have?
print("\nValue distribution:")
wins = sum(1 for v in solved.values() if v == 1)
losses = sum(1 for v in solved.values() if v == -1)
draws = sum(1 for v in solved.values() if v == 0)
print(f"  X wins:  {wins:,}")
print(f"  O wins:  {losses:,}")
print(f"  Draws:   {draws:,}")
