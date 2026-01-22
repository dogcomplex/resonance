"""
New Game Environments for Testing Hierarchical Learner

These test capabilities not covered by existing environments:
1. Minesweeper - Logic deduction, partial information
2. Game2048 - Numeric combination rules
3. Snake - Physics-lite, growing state
4. LightSwitch - State machine, boolean dependencies
5. Mastermind - Deduction from feedback
6. TradingGame - Economic relationships
"""

import random
from typing import Set, List, Tuple, Dict, Optional
from collections import defaultdict


class Minesweeper:
    """
    Small Minesweeper (5x5, 3 mines).
    Tests: Logic deduction, partial observation.
    
    State: revealed cells with neighbor counts, flag positions
    Actions: reveal cell (0-24) or flag cell (25-49)
    """
    def __init__(self, seed=42, size=5, n_mines=3):
        self.size = size
        self.n_mines = n_mines
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None) -> Set[str]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        # Place mines
        all_cells = [(x, y) for x in range(self.size) for y in range(self.size)]
        self.mines = set(self.rng.sample(all_cells, self.n_mines))
        
        # Game state
        self.revealed = set()
        self.flagged = set()
        self.done = False
        self.won = False
        
        return self._get_state()
    
    def _count_neighbors(self, x, y) -> int:
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx, dy) != (0, 0):
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in self.mines:
                        count += 1
        return count
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) in self.revealed:
                    count = self._count_neighbors(x, y)
                    tokens.add(f"cell_{x}_{y}_revealed_{count}")
                elif (x, y) in self.flagged:
                    tokens.add(f"cell_{x}_{y}_flagged")
                else:
                    tokens.add(f"cell_{x}_{y}_hidden")
        tokens.add(f"done_{self.done}")
        tokens.add(f"won_{self.won}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        if self.done:
            return []
        actions = []
        for i in range(self.size * self.size):
            x, y = i % self.size, i // self.size
            if (x, y) not in self.revealed:
                actions.append(i)  # reveal
                actions.append(i + self.size * self.size)  # flag
        return actions
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, True, {}
        
        is_flag = action >= self.size * self.size
        idx = action % (self.size * self.size)
        x, y = idx % self.size, idx // self.size
        
        if is_flag:
            if (x, y) in self.flagged:
                self.flagged.remove((x, y))
            elif (x, y) not in self.revealed:
                self.flagged.add((x, y))
        else:
            if (x, y) in self.revealed or (x, y) in self.flagged:
                return self._get_state(), 0, False, {}
            
            if (x, y) in self.mines:
                self.done = True
                self.won = False
                return self._get_state(), -1, True, {}
            
            # Reveal cell (and flood fill if 0)
            self._reveal(x, y)
            
            # Check win
            if len(self.revealed) == self.size * self.size - self.n_mines:
                self.done = True
                self.won = True
                return self._get_state(), 1, True, {}
        
        return self._get_state(), 0, False, {}
    
    def _reveal(self, x, y):
        if (x, y) in self.revealed or (x, y) in self.mines:
            return
        self.revealed.add((x, y))
        if self._count_neighbors(x, y) == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        self._reveal(nx, ny)


class Game2048:
    """
    2048 on a 3x3 board (smaller for tractable state space).
    Tests: Numeric combination rules, deterministic merging.
    
    State: cell values (0, 2, 4, 8, 16, 32, 64, 128...)
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    def __init__(self, seed=42, size=3):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None) -> Set[str]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        self.board = [[0] * self.size for _ in range(self.size)]
        self._add_tile()
        self._add_tile()
        self.done = False
        self.score = 0
        
        return self._get_state()
    
    def _add_tile(self):
        empty = [(x, y) for x in range(self.size) for y in range(self.size) 
                 if self.board[y][x] == 0]
        if empty:
            x, y = self.rng.choice(empty)
            self.board[y][x] = 2 if self.rng.random() < 0.9 else 4
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for y in range(self.size):
            for x in range(self.size):
                val = self.board[y][x]
                tokens.add(f"cell_{x}_{y}_{val}")
        tokens.add(f"score_{self.score // 100}")  # Bucketed score
        tokens.add(f"done_{self.done}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        if self.done:
            return []
        return [0, 1, 2, 3]
    
    def _slide_row(self, row: List[int]) -> Tuple[List[int], int]:
        """Slide and merge a row to the left."""
        # Remove zeros
        non_zero = [x for x in row if x != 0]
        
        # Merge adjacent equal values
        merged = []
        score = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                score += non_zero[i] * 2
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        
        # Pad with zeros
        while len(merged) < len(row):
            merged.append(0)
        
        return merged, score
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, True, {}
        
        old_board = [row[:] for row in self.board]
        move_score = 0
        
        if action == 0:  # Up
            for x in range(self.size):
                col = [self.board[y][x] for y in range(self.size)]
                new_col, sc = self._slide_row(col)
                for y in range(self.size):
                    self.board[y][x] = new_col[y]
                move_score += sc
        elif action == 1:  # Down
            for x in range(self.size):
                col = [self.board[y][x] for y in range(self.size - 1, -1, -1)]
                new_col, sc = self._slide_row(col)
                for i, y in enumerate(range(self.size - 1, -1, -1)):
                    self.board[y][x] = new_col[i]
                move_score += sc
        elif action == 2:  # Left
            for y in range(self.size):
                self.board[y], sc = self._slide_row(self.board[y])
                move_score += sc
        elif action == 3:  # Right
            for y in range(self.size):
                row = self.board[y][::-1]
                new_row, sc = self._slide_row(row)
                self.board[y] = new_row[::-1]
                move_score += sc
        
        # Check if board changed
        if self.board != old_board:
            self.score += move_score
            self._add_tile()
        
        # Check game over
        if not self._can_move():
            self.done = True
        
        return self._get_state(), move_score / 100, self.done, {}
    
    def _can_move(self) -> bool:
        # Check for empty cells
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == 0:
                    return True
        # Check for possible merges
        for y in range(self.size):
            for x in range(self.size):
                val = self.board[y][x]
                if x + 1 < self.size and self.board[y][x + 1] == val:
                    return True
                if y + 1 < self.size and self.board[y + 1][x] == val:
                    return True
        return False


class Snake:
    """
    Snake game on a small grid.
    Tests: Growing state, collision physics, trajectory.
    
    State: head position, body segments, food position
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    def __init__(self, seed=42, size=6):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None) -> Set[str]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        center = self.size // 2
        self.snake = [(center, center)]  # Head first
        self.direction = (1, 0)  # Moving right
        self.food = self._place_food()
        self.done = False
        self.score = 0
        
        return self._get_state()
    
    def _place_food(self) -> Tuple[int, int]:
        empty = [(x, y) for x in range(self.size) for y in range(self.size)
                 if (x, y) not in self.snake]
        return self.rng.choice(empty) if empty else (-1, -1)
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        head = self.snake[0]
        tokens.add(f"head_{head[0]}_{head[1]}")
        
        # Direction
        tokens.add(f"dir_{self.direction[0]}_{self.direction[1]}")
        
        # Body segments (limit to avoid state explosion)
        for i, seg in enumerate(self.snake[1:min(4, len(self.snake))]):
            tokens.add(f"body_{i}_{seg[0]}_{seg[1]}")
        
        tokens.add(f"length_{min(len(self.snake), 6)}")
        tokens.add(f"food_{self.food[0]}_{self.food[1]}")
        
        # Relative food position
        fx, fy = self.food[0] - head[0], self.food[1] - head[1]
        tokens.add(f"food_rel_{1 if fx > 0 else -1 if fx < 0 else 0}_{1 if fy > 0 else -1 if fy < 0 else 0}")
        
        tokens.add(f"done_{self.done}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        if self.done:
            return []
        return [0, 1, 2, 3]
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, True, {}
        
        # Map action to direction (can't reverse)
        new_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_dir = new_dirs[action]
        
        # Can't reverse
        if (new_dir[0] + self.direction[0], new_dir[1] + self.direction[1]) != (0, 0):
            self.direction = new_dir
        
        # Move head
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Check wall collision
        if not (0 <= new_head[0] < self.size and 0 <= new_head[1] < self.size):
            self.done = True
            return self._get_state(), -1, True, {}
        
        # Check self collision
        if new_head in self.snake[:-1]:
            self.done = True
            return self._get_state(), -1, True, {}
        
        # Move
        self.snake.insert(0, new_head)
        
        # Check food
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 1
        else:
            self.snake.pop()
            reward = 0
        
        return self._get_state(), reward, False, {}


class LightSwitch:
    """
    Light switch puzzle with dependencies.
    Tests: Boolean logic, state machines.
    
    Some switches depend on others being in certain states.
    Goal: Turn all lights on.
    """
    def __init__(self, seed=42, n_switches=4):
        self.n_switches = n_switches
        self.rng = random.Random(seed)
        self._setup_dependencies()
        self.reset()
    
    def _setup_dependencies(self):
        """Each switch may depend on another switch's state."""
        self.dependencies = {}
        for i in range(self.n_switches):
            if self.rng.random() < 0.5 and i > 0:
                dep = self.rng.randint(0, i - 1)
                req_state = self.rng.choice([True, False])
                self.dependencies[i] = (dep, req_state)
    
    def reset(self, seed=None) -> Set[str]:
        self.switches = [False] * self.n_switches
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for i, on in enumerate(self.switches):
            tokens.add(f"switch_{i}_{'on' if on else 'off'}")
        
        # Encode dependencies (the rules)
        for sw, (dep, req) in self.dependencies.items():
            tokens.add(f"dep_{sw}_needs_{dep}_{'on' if req else 'off'}")
        
        tokens.add(f"all_on_{all(self.switches)}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        return list(range(self.n_switches))
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if action < 0 or action >= self.n_switches:
            return self._get_state(), 0, False, {}
        
        # Check if action is allowed
        if action in self.dependencies:
            dep, req_state = self.dependencies[action]
            if self.switches[dep] != req_state:
                # Dependency not met - switch doesn't toggle
                return self._get_state(), -0.1, False, {}
        
        # Toggle switch
        self.switches[action] = not self.switches[action]
        
        # Check win
        if all(self.switches):
            return self._get_state(), 1, True, {}
        
        return self._get_state(), 0, False, {}


class Mastermind:
    """
    Code-breaking game.
    Tests: Deduction from feedback.
    
    Guess a 4-digit code, get feedback on correct positions and colors.
    """
    def __init__(self, seed=42, code_length=3, n_colors=4):
        self.code_length = code_length
        self.n_colors = n_colors
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None) -> Set[str]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        self.secret = tuple(self.rng.randint(0, self.n_colors - 1) 
                           for _ in range(self.code_length))
        self.guesses = []
        self.feedbacks = []
        self.done = False
        self.won = False
        
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        tokens.add(f"n_guesses_{len(self.guesses)}")
        
        # Include last few guesses and feedbacks
        for i, (guess, (exact, color)) in enumerate(zip(self.guesses[-3:], self.feedbacks[-3:])):
            for j, g in enumerate(guess):
                tokens.add(f"guess_{i}_{j}_{g}")
            tokens.add(f"feedback_{i}_exact_{exact}")
            tokens.add(f"feedback_{i}_color_{color}")
        
        tokens.add(f"done_{self.done}")
        tokens.add(f"won_{self.won}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        if self.done:
            return []
        # Actions encode guesses: action = c0 + c1*n_colors + c2*n_colors^2 + ...
        return list(range(self.n_colors ** self.code_length))
    
    def _decode_action(self, action: int) -> Tuple[int, ...]:
        guess = []
        for _ in range(self.code_length):
            guess.append(action % self.n_colors)
            action //= self.n_colors
        return tuple(guess)
    
    def _get_feedback(self, guess: Tuple[int, ...]) -> Tuple[int, int]:
        exact = sum(g == s for g, s in zip(guess, self.secret))
        
        # Count color matches (not in right position)
        guess_counts = defaultdict(int)
        secret_counts = defaultdict(int)
        for g, s in zip(guess, self.secret):
            if g != s:
                guess_counts[g] += 1
                secret_counts[s] += 1
        
        color = sum(min(guess_counts[c], secret_counts[c]) for c in guess_counts)
        
        return exact, color
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, True, {}
        
        guess = self._decode_action(action)
        feedback = self._get_feedback(guess)
        
        self.guesses.append(guess)
        self.feedbacks.append(feedback)
        
        if guess == self.secret:
            self.done = True
            self.won = True
            return self._get_state(), 1, True, {}
        
        # Limit guesses
        if len(self.guesses) >= 10:
            self.done = True
            return self._get_state(), -1, True, {}
        
        return self._get_state(), 0, False, {}


class TradingGame:
    """
    Simple trading/economy game.
    Tests: Learning price relationships, buy low/sell high.
    
    Buy and sell goods as prices fluctuate.
    """
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.goods = ['wheat', 'iron', 'gold']
        self.reset()
    
    def reset(self, seed=None) -> Set[str]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        self.money = 100
        self.inventory = {g: 0 for g in self.goods}
        self.prices = {g: self.rng.randint(5, 20) for g in self.goods}
        self.turn = 0
        self.done = False
        
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        tokens.add(f"money_{self.money // 20}")  # Bucketed
        
        for g in self.goods:
            tokens.add(f"inv_{g}_{min(self.inventory[g], 5)}")
            tokens.add(f"price_{g}_{self.prices[g] // 5}")  # Bucketed price
        
        tokens.add(f"turn_{self.turn}")
        tokens.add(f"done_{self.done}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        if self.done:
            return []
        # 0-2: buy goods, 3-5: sell goods, 6: wait
        actions = [6]  # Can always wait
        for i, g in enumerate(self.goods):
            if self.money >= self.prices[g]:
                actions.append(i)  # Can buy
            if self.inventory[g] > 0:
                actions.append(i + 3)  # Can sell
        return actions
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, True, {}
        
        reward = 0
        
        if action < 3:  # Buy
            good = self.goods[action]
            if self.money >= self.prices[good]:
                self.money -= self.prices[good]
                self.inventory[good] += 1
        elif action < 6:  # Sell
            good = self.goods[action - 3]
            if self.inventory[good] > 0:
                self.inventory[good] -= 1
                self.money += self.prices[good]
        # action == 6: wait
        
        # Update prices (some randomness but with trends)
        for g in self.goods:
            change = self.rng.randint(-3, 3)
            self.prices[g] = max(1, min(30, self.prices[g] + change))
        
        self.turn += 1
        if self.turn >= 20:
            self.done = True
            # Final score based on money + inventory value
            total = self.money + sum(self.inventory[g] * self.prices[g] for g in self.goods)
            reward = (total - 100) / 100  # Normalized profit
        
        return self._get_state(), reward, self.done, {}


# ============================================================================
# Test all new environments
# ============================================================================

if __name__ == "__main__":
    print("Testing new game environments...")
    
    games = [
        ("Minesweeper", Minesweeper, 50),  # 25 reveal + 25 flag
        ("2048", Game2048, 4),
        ("Snake", Snake, 4),
        ("LightSwitch", LightSwitch, 4),
        ("Mastermind", Mastermind, 64),  # 4^3 guesses
        ("TradingGame", TradingGame, 7),
    ]
    
    for name, game_cls, n_actions in games:
        game = game_cls(seed=42)
        state = game.reset()
        print(f"\n{name}:")
        print(f"  Initial state: {len(state)} tokens")
        print(f"  Sample tokens: {list(state)[:5]}")
        print(f"  Valid actions: {game.get_valid_actions()[:5]}...")
        
        # Play a few random steps
        for _ in range(3):
            actions = game.get_valid_actions()
            if not actions:
                break
            action = random.choice(actions)
            state, reward, done, _ = game.step(action)
        
        print(f"  After 3 steps: {len(state)} tokens, done={done}")
    
    print("\nAll environments working!")
