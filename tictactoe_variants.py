"""
TicTacToe and variants for testing rule learning:
1. Standard TicTacToe
2. Misere (lose by getting 3 in a row)
3. Wild (can play either X or O)
4. Random rules (pure chaos)
"""
import random
from typing import Set, Tuple, Optional

class TicTacToeBase:
    """Base class for 3x3 board games"""
    
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.board = [0] * 9  # 0=empty, 1=X, 2=O
        self.current_player = 1  # X goes first
        self.done = False
        self.winner = None
    
    def reset(self, seed=None) -> Set[str]:
        if seed is not None:
            random.seed(seed)
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
        self.winner = None
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for i, val in enumerate(self.board):
            tokens.add(f"cell_{i}_{val}")  # cell_0_0, cell_0_1, cell_0_2
        tokens.add(f"player_{self.current_player}")
        tokens.add(f"done_{self.done}")
        if self.winner is not None:
            tokens.add(f"winner_{self.winner}")
        return tokens
    
    def _check_win(self, player) -> bool:
        """Check if player has won"""
        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # cols
            [0,4,8], [2,4,6]  # diags
        ]
        for line in lines:
            if all(self.board[i] == player for i in line):
                return True
        return False
    
    def _is_full(self) -> bool:
        return all(c != 0 for c in self.board)
    
    def get_valid_actions(self):
        if self.done:
            return []
        return [i for i in range(9) if self.board[i] == 0]
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        raise NotImplementedError


class StandardTicTacToe(TicTacToeBase):
    """Standard rules: 3 in a row wins"""
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done or action < 0 or action > 8 or self.board[action] != 0:
            return self._get_state(), 0, self.done, {}
        
        self.board[action] = self.current_player
        
        if self._check_win(self.current_player):
            self.winner = self.current_player
            self.done = True
            reward = 1 if self.current_player == 1 else -1
        elif self._is_full():
            self.done = True
            reward = 0
        else:
            self.current_player = 3 - self.current_player
            reward = 0
        
        return self._get_state(), reward, self.done, {}


class MisereTicTacToe(TicTacToeBase):
    """Misere: 3 in a row LOSES"""
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done or action < 0 or action > 8 or self.board[action] != 0:
            return self._get_state(), 0, self.done, {}
        
        self.board[action] = self.current_player
        
        if self._check_win(self.current_player):
            # Getting 3 in a row means you LOSE
            self.winner = 3 - self.current_player
            self.done = True
            reward = -1 if self.current_player == 1 else 1
        elif self._is_full():
            self.done = True
            reward = 0
        else:
            self.current_player = 3 - self.current_player
            reward = 0
        
        return self._get_state(), reward, self.done, {}


class WildTicTacToe(TicTacToeBase):
    """Wild: can play X (action 0-8) or O (action 9-17)"""
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, self.done, {}
        
        # Actions 0-8: place X, 9-17: place O
        if action < 9:
            cell, piece = action, 1
        else:
            cell, piece = action - 9, 2
        
        if cell < 0 or cell > 8 or self.board[cell] != 0:
            return self._get_state(), 0, self.done, {}
        
        self.board[cell] = piece
        
        # Check if either player won (by either piece)
        if self._check_win(1) or self._check_win(2):
            self.winner = self.current_player
            self.done = True
            reward = 1 if self.current_player == 1 else -1
        elif self._is_full():
            self.done = True
            reward = 0
        else:
            self.current_player = 3 - self.current_player
            reward = 0
        
        return self._get_state(), reward, self.done, {}
    
    def get_valid_actions(self):
        if self.done:
            return []
        empty = [i for i in range(9) if self.board[i] == 0]
        return empty + [i + 9 for i in empty]


class RandomRulesTicTacToe(TicTacToeBase):
    """Pure chaos: random state transitions, no consistent rules"""
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, self.done, {}
        
        # Randomly modify 1-3 cells
        num_changes = random.randint(1, 3)
        cells_to_change = random.sample(range(9), min(num_changes, 9))
        for cell in cells_to_change:
            self.board[cell] = random.randint(0, 2)
        
        # Randomly end game
        if random.random() < 0.15:
            self.done = True
            self.winner = random.choice([None, 1, 2])
        
        # Randomly switch player (or not)
        if random.random() < 0.7:
            self.current_player = 3 - self.current_player
        
        return self._get_state(), random.uniform(-1, 1), self.done, {}


class PartialRandomTicTacToe(TicTacToeBase):
    """Semi-chaos: normal placement but random win detection"""
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done or action < 0 or action > 8 or self.board[action] != 0:
            return self._get_state(), 0, self.done, {}
        
        # Normal placement
        self.board[action] = self.current_player
        
        # Random win check (30% chance of random outcome)
        if random.random() < 0.3:
            if random.random() < 0.5:
                self.winner = random.choice([1, 2])
                self.done = True
        elif self._check_win(self.current_player):
            self.winner = self.current_player
            self.done = True
        elif self._is_full():
            self.done = True
        
        if not self.done:
            self.current_player = 3 - self.current_player
        
        reward = 0
        if self.winner == 1:
            reward = 1
        elif self.winner == 2:
            reward = -1
        
        return self._get_state(), reward, self.done, {}


if __name__ == "__main__":
    # Quick test
    for cls, name in [(StandardTicTacToe, "Standard"), 
                       (MisereTicTacToe, "Misere"),
                       (WildTicTacToe, "Wild"),
                       (RandomRulesTicTacToe, "Random"),
                       (PartialRandomTicTacToe, "PartialRandom")]:
        env = cls(seed=42)
        state = env.reset()
        print(f"{name}: {len(state)} tokens, {len(env.get_valid_actions())} actions")
