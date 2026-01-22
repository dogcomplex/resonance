"""
Chaos variants for testing learnability:

1. Seeded Deterministic Chaos - random but stable rules, should converge to 100%
2. True Chaos - rules shift every step, unlearnable
3. Seeded Probabilistic Chaos - stable probabilistic rules, learnable distributions
4. True Probabilistic Chaos - shifting probabilistic rules, unlearnable
"""
import random
from typing import Set, Tuple, Dict, List

class SeededDeterministicChaos:
    """
    Type 1: Random rules chosen once at seed, then stable forever.
    Should eventually reach ~100% prediction accuracy.
    """
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
        
        # Generate stable random rules at init
        # Rule: (board_state_hash, action) -> resulting board state
        self._transition_rules: Dict[Tuple[tuple, int], List[int]] = {}
        self._generate_rules()
    
    def _generate_rules(self):
        """Pre-generate deterministic transitions for all possible states."""
        # For each possible board state and action, define what happens
        # We'll generate lazily on first encounter instead
        pass
    
    def _get_transition(self, board_tuple: tuple, action: int) -> List[int]:
        """Get or create deterministic transition for this state+action."""
        key = (board_tuple, action)
        if key not in self._transition_rules:
            # Generate a random but stable transition
            new_board = list(board_tuple)
            # Randomly modify 1-2 cells
            num_changes = self.rng.randint(1, 2)
            for _ in range(num_changes):
                cell = self.rng.randint(0, 8)
                new_board[cell] = self.rng.randint(0, 2)
            self._transition_rules[key] = new_board
        return self._transition_rules[key]
    
    def reset(self, seed=None) -> Set[str]:
        # Don't reseed rng - keep same rules
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for i, val in enumerate(self.board):
            tokens.add(f"cell_{i}_{val}")
        tokens.add(f"player_{self.current_player}")
        tokens.add(f"done_{self.done}")
        return tokens
    
    def get_valid_actions(self):
        return list(range(9)) if not self.done else []
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done or action < 0 or action > 8:
            return self._get_state(), 0, self.done, {}
        
        board_tuple = tuple(self.board)
        self.board = self._get_transition(board_tuple, action)
        
        # Deterministic end condition based on state
        end_key = (tuple(self.board), -1)  # -1 = end check
        if end_key not in self._transition_rules:
            self._transition_rules[end_key] = [1 if self.rng.random() < 0.1 else 0]
        
        if self._transition_rules[end_key][0]:
            self.done = True
        else:
            self.current_player = 3 - self.current_player
        
        return self._get_state(), 0, self.done, {}


class TrueChaos:
    """
    Type 2: Rules change every step. Completely unlearnable.
    """
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
    
    def reset(self, seed=None) -> Set[str]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for i, val in enumerate(self.board):
            tokens.add(f"cell_{i}_{val}")
        tokens.add(f"player_{self.current_player}")
        tokens.add(f"done_{self.done}")
        return tokens
    
    def get_valid_actions(self):
        return list(range(9)) if not self.done else []
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, self.done, {}
        
        # Completely random transition every time
        num_changes = self.rng.randint(1, 3)
        for _ in range(num_changes):
            cell = self.rng.randint(0, 8)
            self.board[cell] = self.rng.randint(0, 2)
        
        if self.rng.random() < 0.15:
            self.done = True
        else:
            self.current_player = 3 - self.current_player
        
        return self._get_state(), 0, self.done, {}


class SeededProbabilisticChaos:
    """
    Type 3: Stable probabilistic rules. Each state+action has fixed probability
    distribution over outcomes. Learnable - should match the probabilities.
    """
    def __init__(self, seed=42):
        self.rule_rng = random.Random(seed)  # For generating rules
        self.play_rng = random.Random()       # For sampling during play
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
        
        # Rule: (board_state_hash, action) -> list of (probability, new_board)
        self._transition_rules: Dict[Tuple[tuple, int], List[Tuple[float, List[int]]]] = {}
    
    def _get_transition_dist(self, board_tuple: tuple, action: int) -> List[Tuple[float, List[int]]]:
        """Get or create probabilistic transition distribution."""
        key = (board_tuple, action)
        if key not in self._transition_rules:
            # Generate 2-4 possible outcomes with random probabilities
            num_outcomes = self.rule_rng.randint(2, 4)
            outcomes = []
            remaining_prob = 1.0
            
            for i in range(num_outcomes):
                new_board = list(board_tuple)
                num_changes = self.rule_rng.randint(1, 2)
                for _ in range(num_changes):
                    cell = self.rule_rng.randint(0, 8)
                    new_board[cell] = self.rule_rng.randint(0, 2)
                
                if i == num_outcomes - 1:
                    prob = remaining_prob
                else:
                    prob = self.rule_rng.uniform(0.1, remaining_prob - 0.1 * (num_outcomes - i - 1))
                    remaining_prob -= prob
                
                outcomes.append((prob, new_board))
            
            self._transition_rules[key] = outcomes
        return self._transition_rules[key]
    
    def reset(self, seed=None) -> Set[str]:
        if seed is not None:
            self.play_rng = random.Random(seed)
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for i, val in enumerate(self.board):
            tokens.add(f"cell_{i}_{val}")
        tokens.add(f"player_{self.current_player}")
        tokens.add(f"done_{self.done}")
        return tokens
    
    def get_valid_actions(self):
        return list(range(9)) if not self.done else []
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done or action < 0 or action > 8:
            return self._get_state(), 0, self.done, {}
        
        board_tuple = tuple(self.board)
        outcomes = self._get_transition_dist(board_tuple, action)
        
        # Sample from distribution
        r = self.play_rng.random()
        cumulative = 0
        for prob, new_board in outcomes:
            cumulative += prob
            if r <= cumulative:
                self.board = list(new_board)
                break
        
        # Probabilistic end
        if self.play_rng.random() < 0.1:
            self.done = True
        else:
            self.current_player = 3 - self.current_player
        
        return self._get_state(), 0, self.done, {}


class TrueProbabilisticChaos:
    """
    Type 4: Shifting probabilistic rules. Completely unlearnable.
    """
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
    
    def reset(self, seed=None) -> Set[str]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for i, val in enumerate(self.board):
            tokens.add(f"cell_{i}_{val}")
        tokens.add(f"player_{self.current_player}")
        tokens.add(f"done_{self.done}")
        return tokens
    
    def get_valid_actions(self):
        return list(range(9)) if not self.done else []
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done:
            return self._get_state(), 0, self.done, {}
        
        # Generate fresh random distribution each time
        num_changes = self.rng.randint(1, 3)
        for _ in range(num_changes):
            if self.rng.random() < self.rng.random():  # Meta-random
                cell = self.rng.randint(0, 8)
                self.board[cell] = self.rng.randint(0, 2)
        
        if self.rng.random() < self.rng.uniform(0.05, 0.2):
            self.done = True
        else:
            self.current_player = 3 - self.current_player
        
        return self._get_state(), 0, self.done, {}


if __name__ == "__main__":
    print("Testing chaos variants...")
    
    for cls, name in [
        (SeededDeterministicChaos, "Seeded Deterministic"),
        (TrueChaos, "True Chaos"),
        (SeededProbabilisticChaos, "Seeded Probabilistic"),
        (TrueProbabilisticChaos, "True Probabilistic"),
    ]:
        env = cls(seed=42)
        state = env.reset()
        print(f"\n{name}: {len(state)} tokens, {len(env.get_valid_actions())} actions")
        
        # Take a few steps
        for i in range(3):
            action = random.choice(env.get_valid_actions()) if env.get_valid_actions() else 0
            next_state, _, done, _ = env.step(action)
            changes = len(next_state ^ state)
            print(f"  Step {i+1}: {changes} token changes")
            state = next_state
            if done:
                break
