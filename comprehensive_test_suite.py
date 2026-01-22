"""
COMPREHENSIVE TEST SUITE for Hierarchical Learner V11

Tests across multiple game types:
1. Board games (TicTacToe, Connect Four)
2. Chaos/Probabilistic systems
3. Grid worlds (MiniGrid variants)
4. Crafting/Inventory systems
5. Puzzle games (Sokoban)
6. Combat systems (probabilistic)
"""

import random
import sys
from typing import Set, List, Dict, Tuple
from collections import defaultdict

sys.path.insert(0, '/home/claude')
sys.path.insert(0, '/mnt/user-data/outputs')

from hierarchical_learner_v11 import HierarchicalLearner as V11
from hierarchical_learner_v9 import HierarchicalLearner as V9
from tictactoe_variants import StandardTicTacToe, RandomRulesTicTacToe
from chaos_variants import SeededDeterministicChaos, SeededProbabilisticChaos, TrueChaos
from minigrid_official import EmptyEnv, FourRoomsEnv, DoorKeyEnv, LavaGapEnv


# ============================================================================
# Additional Game Implementations
# ============================================================================

class ConnectFour:
    """Connect Four on 4x4 board"""
    def __init__(self, seed=42, width=4, height=4):
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None) -> Set[str]:
        if seed: self.rng = random.Random(seed)
        self.board = [[0]*self.height for _ in range(self.width)]
        self.current_player = 1
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for x in range(self.width):
            for y in range(self.height):
                tokens.add(f"cell_{x}_{y}_{self.board[x][y]}")
        tokens.add(f"player_{self.current_player}")
        tokens.add(f"done_{self.done}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        if self.done: return []
        return [x for x in range(self.width) if self.board[x][-1] == 0]
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done or action not in self.get_valid_actions():
            return self._get_state(), 0, self.done, {}
        
        for y in range(self.height):
            if self.board[action][y] == 0:
                self.board[action][y] = self.current_player
                break
        
        # Check horizontal wins
        for y in range(self.height):
            for x in range(self.width - 3):
                if self.board[x][y] == self.board[x+1][y] == self.board[x+2][y] == self.board[x+3][y] != 0:
                    self.done = True
        
        if not self.done and all(self.board[x][-1] != 0 for x in range(self.width)):
            self.done = True
        
        if not self.done:
            self.current_player = 3 - self.current_player
        
        return self._get_state(), 0, self.done, {}


class CraftingSystem:
    """Simple crafting with recipes"""
    RECIPES = {
        ('wood', 'wood'): 'plank',
        ('plank', 'plank'): 'stick',
        ('stick', 'stone'): 'pickaxe',
        ('iron', 'stick'): 'sword',
        ('plank', 'stick'): 'sign',
    }
    
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None) -> Set[str]:
        self.inventory = {'wood': 4, 'stone': 2, 'iron': 1}
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        for item, count in self.inventory.items():
            tokens.add(f"has_{item}_{min(count, 5)}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        return list(range(8))
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if action < 5:
            recipes = list(self.RECIPES.items())
            if action < len(recipes):
                (item1, item2), result = recipes[action]
                if self.inventory.get(item1, 0) >= 1 and self.inventory.get(item2, 0) >= 1:
                    self.inventory[item1] -= 1
                    self.inventory[item2] -= 1
                    self.inventory[result] = self.inventory.get(result, 0) + 1
        else:
            raw = ['wood', 'stone', 'iron'][action - 5]
            self.inventory[raw] = self.inventory.get(raw, 0) + 1
        return self._get_state(), 0, False, {}


class SimpleSokoban:
    """Simple Sokoban puzzle"""
    def __init__(self, seed=42, size=5):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None) -> Set[str]:
        if seed: self.rng = random.Random(seed)
        self.player = [1, 1]
        self.box = [2, 2]
        self.goal = [3, 3]
        self.walls = {(0, y) for y in range(self.size)} | \
                    {(self.size-1, y) for y in range(self.size)} | \
                    {(x, 0) for x in range(self.size)} | \
                    {(x, self.size-1) for x in range(self.size)}
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        tokens = set()
        tokens.add(f"player_{self.player[0]}_{self.player[1]}")
        tokens.add(f"box_{self.box[0]}_{self.box[1]}")
        tokens.add(f"goal_{self.goal[0]}_{self.goal[1]}")
        tokens.add(f"box_rel_{self.box[0]-self.player[0]}_{self.box[1]-self.player[1]}")
        tokens.add(f"done_{self.box == self.goal}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        return [0, 1, 2, 3]
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        new_player = [self.player[0]+dx, self.player[1]+dy]
        
        if tuple(new_player) in self.walls:
            return self._get_state(), 0, False, {}
        
        if new_player == self.box:
            new_box = [self.box[0]+dx, self.box[1]+dy]
            if tuple(new_box) in self.walls:
                return self._get_state(), 0, False, {}
            self.box = new_box
        
        self.player = new_player
        done = self.box == self.goal
        return self._get_state(), 1 if done else 0, done, {}


class SimpleCombat:
    """Turn-based combat with damage rolls"""
    def __init__(self, seed=42):
        self.rule_rng = random.Random(seed)
        self.attacks = {
            0: (self.rule_rng.randint(2, 4), self.rule_rng.randint(5, 8)),
            1: (self.rule_rng.randint(3, 5), self.rule_rng.randint(6, 9)),
            2: (self.rule_rng.randint(1, 2), self.rule_rng.randint(8, 12)),
        }
        self.play_rng = random.Random()
        self.reset()
    
    def reset(self, seed=None) -> Set[str]:
        if seed: self.play_rng = random.Random(seed)
        self.player_hp = 20
        self.enemy_hp = 15
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> Set[str]:
        return {
            f"player_hp_{self.player_hp // 5}",
            f"enemy_hp_{self.enemy_hp // 5}",
            f"done_{self.done}"
        }
    
    def get_valid_actions(self) -> List[int]:
        return [0, 1, 2] if not self.done else []
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if self.done: return self._get_state(), 0, True, {}
        
        min_dmg, max_dmg = self.attacks[action]
        damage = self.play_rng.randint(min_dmg, max_dmg)
        self.enemy_hp = max(0, self.enemy_hp - damage)
        
        if self.enemy_hp <= 0:
            self.done = True
            return self._get_state(), 1, True, {}
        
        enemy_dmg = self.play_rng.randint(2, 5)
        self.player_hp = max(0, self.player_hp - enemy_dmg)
        
        if self.player_hp <= 0:
            self.done = True
        
        return self._get_state(), 0, self.done, {}


# ============================================================================
# Test Helpers
# ============================================================================

def tokenize_mg(obs, env):
    tokens = set()
    image = obs.get('image', obs) if hasattr(obs, 'get') else obs
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x][0] != 0:
                tokens.add(f"obs_{y}_{x}_t{image[y][x][0]}")
    tokens.add(f"pos_{env.agent_pos[0]}_{env.agent_pos[1]}")
    tokens.add(f"dir_{env.agent_dir}")
    return tokens


def test_board_game(learner_cls, env_cls, n_actions=9, n_train=300, n_test=100, env_seed=42):
    random.seed(42)
    env = env_cls(seed=env_seed)
    learner = learner_cls(n_actions=n_actions)
    seen = set()
    
    for ep in range(n_train):
        state = env.reset(seed=ep*1000)
        if hasattr(learner, 'reset_episode'): learner.reset_episode()
        for _ in range(20):
            valid = env.get_valid_actions()
            if not valid: break
            action = random.choice(valid)
            seen.add((frozenset(state), action))
            next_state, _, done, _ = env.step(action)
            learner.observe(state, action, next_state)
            state = next_state
            if done: break
    
    tp_s, fp_s, fn_s = 0, 0, 0
    tp_u, fp_u, fn_u = 0, 0, 0
    
    for ep in range(n_test):
        state = env.reset(seed=80000+ep)
        if hasattr(learner, 'reset_episode'): learner.reset_episode()
        for _ in range(20):
            valid = env.get_valid_actions()
            if not valid: break
            action = random.choice(valid)
            is_seen = (frozenset(state), action) in seen
            next_state, _, done, _ = env.step(action)
            
            actual = {f"+{t}" for t in (next_state - state)} | {f"-{t}" for t in (state - next_state)}
            predicted = learner.predict(state, action)
            
            for e in predicted:
                if e in actual:
                    if is_seen: tp_s += 1
                    else: tp_u += 1
                else:
                    if is_seen: fp_s += 1
                    else: fp_u += 1
            for e in actual:
                if e not in predicted:
                    if is_seen: fn_s += 1
                    else: fn_u += 1
            
            state = next_state
            if done: break
    
    f1_s = 2*tp_s/(2*tp_s+fp_s+fn_s) if (2*tp_s+fp_s+fn_s) > 0 else 0
    f1_u = 2*tp_u/(2*tp_u+fp_u+fn_u) if (2*tp_u+fp_u+fn_u) > 0 else 0
    
    if hasattr(learner, 'close'): learner.close()
    return f1_s, f1_u


def test_minigrid(learner_cls, env_factory, n_train=150, n_test=50):
    random.seed(42)
    learner = learner_cls(n_actions=7)
    seen = set()
    
    for ep in range(n_train):
        env = env_factory(ep)
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        if hasattr(learner, 'reset_episode'): learner.reset_episode()
        state = tokenize_mg(obs, env)
        for _ in range(30):
            action = random.randint(0, 6)
            seen.add((frozenset(state), action))
            result = env.step(action)
            learner.observe(state, action, tokenize_mg(result[0], env))
            state = tokenize_mg(result[0], env)
            if result[2]: break
    
    tp_s, fp_s, fn_s = 0, 0, 0
    
    for ep in range(n_test):
        env = env_factory(50000+ep)
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        if hasattr(learner, 'reset_episode'): learner.reset_episode()
        state = tokenize_mg(obs, env)
        for _ in range(25):
            action = random.randint(0, 6)
            if (frozenset(state), action) not in seen:
                result = env.step(action)
                state = tokenize_mg(result[0], env)
                if result[2]: break
                continue
            result = env.step(action)
            next_state = tokenize_mg(result[0], env)
            actual = {f"+{t}" for t in (next_state - state)} | {f"-{t}" for t in (state - next_state)}
            predicted = learner.predict(state, action)
            
            for e in predicted:
                if e in actual: tp_s += 1
                else: fp_s += 1
            for e in actual:
                if e not in predicted: fn_s += 1
            
            state = next_state
            if result[2]: break
    
    f1_s = 2*tp_s/(2*tp_s+fp_s+fn_s) if (2*tp_s+fp_s+fn_s) > 0 else 0
    if hasattr(learner, 'close'): learner.close()
    return f1_s


# ============================================================================
# Main Test Suite
# ============================================================================

def run_test_suite():
    print("="*80)
    print("COMPREHENSIVE TEST SUITE: V9 vs V11")
    print("="*80)
    
    results = []
    
    # Board Games
    print("\n[BOARD GAMES]")
    print(f"{'Game':<30} {'V9 Seen':>10} {'V11 Seen':>10} {'V9 Unsn':>10} {'V11 Unsn':>10}")
    print("-"*80)
    
    board_tests = [
        ("TicTacToe", StandardTicTacToe, 9),
        ("Connect Four (4x4)", ConnectFour, 4),
    ]
    
    for name, env_cls, n_actions in board_tests:
        v9_s, v9_u = test_board_game(V9, env_cls, n_actions)
        v11_s, v11_u = test_board_game(V11, env_cls, n_actions)
        print(f"{name:<30} {v9_s:>10.1%} {v11_s:>10.1%} {v9_u:>10.1%} {v11_u:>10.1%}")
        results.append((name, v9_s, v11_s))
    
    # Chaos Systems
    print("\n[CHAOS SYSTEMS]")
    print(f"{'System':<30} {'V9 Seen':>10} {'V11 Seen':>10} {'Expected':>10}")
    print("-"*80)
    
    chaos_tests = [
        ("Seeded Deterministic", SeededDeterministicChaos, "100%"),
        ("Seeded Probabilistic", SeededProbabilisticChaos, "~70%"),
        ("True Chaos", TrueChaos, "~50%"),
        ("Random Rules TTT", RandomRulesTicTacToe, "~40%"),
    ]
    
    for name, env_cls, expected in chaos_tests:
        v9_s, _ = test_board_game(V9, env_cls, 9)
        v11_s, _ = test_board_game(V11, env_cls, 9)
        print(f"{name:<30} {v9_s:>10.1%} {v11_s:>10.1%} {expected:>10}")
        results.append((name, v9_s, v11_s))
    
    # MiniGrid Environments
    print("\n[MINIGRID ENVIRONMENTS]")
    print(f"{'Environment':<30} {'V9 Seen':>10} {'V11 Seen':>10}")
    print("-"*80)
    
    mg_tests = [
        ("Empty-8x8", lambda s: EmptyEnv(size=8, seed=s)),
        ("FourRooms", lambda s: FourRoomsEnv(seed=s)),
        ("DoorKey-6x6", lambda s: DoorKeyEnv(size=6, seed=s)),
        ("LavaGap-5x5", lambda s: LavaGapEnv(size=5, seed=s)),
    ]
    
    for name, env_factory in mg_tests:
        v9_s = test_minigrid(V9, env_factory)
        v11_s = test_minigrid(V11, env_factory)
        print(f"{name:<30} {v9_s:>10.1%} {v11_s:>10.1%}")
        results.append((name, v9_s, v11_s))
    
    # Other Domains
    print("\n[OTHER DOMAINS]")
    print(f"{'Domain':<30} {'V9 Seen':>10} {'V11 Seen':>10}")
    print("-"*80)
    
    other_tests = [
        ("Crafting System", CraftingSystem, 8),
        ("Simple Sokoban", SimpleSokoban, 4),
        ("Combat (Probabilistic)", SimpleCombat, 3),
    ]
    
    for name, env_cls, n_actions in other_tests:
        v9_s, _ = test_board_game(V9, env_cls, n_actions, n_train=200, n_test=50)
        v11_s, _ = test_board_game(V11, env_cls, n_actions, n_train=200, n_test=50)
        print(f"{name:<30} {v9_s:>10.1%} {v11_s:>10.1%}")
        results.append((name, v9_s, v11_s))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = 0
    failed_tests = []
    for name, v9_s, v11_s in results:
        diff = v11_s - v9_s
        if abs(diff) < 0.02:
            passed += 1
        else:
            failed_tests.append((name, v9_s, v11_s, diff))
    
    print(f"Tests passed (V11 ≈ V9): {passed}/{len(results)}")
    
    if failed_tests:
        print(f"\nTests with differences >2%:")
        for name, v9_s, v11_s, diff in failed_tests:
            status = "⚠️" if diff < 0 else "↑"
            print(f"  {status} {name}: V9={v9_s:.1%}, V11={v11_s:.1%}, diff={diff:+.1%}")
    else:
        print("\n✓ All tests passed! V11 matches V9 on all environments.")
    
    return results


if __name__ == "__main__":
    run_test_suite()
