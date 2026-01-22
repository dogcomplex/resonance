"""
TicTacToe Oracle - The ground truth game rules we're trying to infer.

This module provides:
- Oracle functions that return the correct label for any board state
- Board generation utilities
- Multiple game variants for testing generalization
"""

import random

square_state_space = ['0', '1', '2']  # 0=empty, 1=X, 2=O
label_space = ['ok', 'win1', 'win2', 'draw', 'error']


def tictactoe(board):
    """Standard TicTacToe oracle function."""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6],              # Diagonals
    ]
    
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == '1':
            return "win1"
    
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == '2':
            return "win2"
    
    if '0' not in board:
        return "draw"
    
    count_1 = board.count('1')
    count_2 = board.count('2')
    if count_1 == count_2:
        return "ok"
    if count_1 != count_2 and count_1 != count_2 + 1:
        return "error"
    
    return "ok"


def tictactoe_no_diags(board):
    """TicTacToe variant where diagonals don't count as wins."""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        # No diagonals!
    ]
    
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == '1':
            return "win1"
    
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == '2':
            return "win2"
    
    if '0' not in board:
        return "draw"
    
    count_1 = board.count('1')
    count_2 = board.count('2')
    if count_1 == count_2:
        return "ok"
    if count_1 != count_2 and count_1 != count_2 + 1:
        return "error"
    
    return "ok"


# Generate all valid game states
boards = {}

def _generate_all_boards():
    """Generate all reachable board states."""
    def explore(board='000000000'):
        if board in boards:
            return
        label = tictactoe(board)
        if label == 'error':
            return
        boards[board] = label
        if label == 'ok':
            count = board.count('0')
            turn = '1' if count % 2 == 1 else '2'
            for i in range(9):
                if board[i] == '0':
                    new_board = board[:i] + turn + board[i+1:]
                    explore(new_board)
    explore()

print("Initializing TicTacToe states...")
_generate_all_boards()
print(f"Generated {len(boards)} valid board states")


def random_board():
    """Return a random valid board state."""
    return random.choice(list(boards.keys()))


def get_label_distribution():
    """Get distribution of labels in all valid boards."""
    from collections import Counter
    return Counter(boards.values())
