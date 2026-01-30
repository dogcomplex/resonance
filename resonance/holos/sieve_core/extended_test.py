"""Extended learning curve test - are we still improving or plateaued?"""
import numpy as np
from wave_sieve import WaveSieve

# ============ EXTENDED SEQUENCE TEST ============
print("=" * 60)
print("EXTENDED SEQUENCE: 20,000 steps")
print("=" * 60)

class SequenceMemory:
    def __init__(self):
        self.current = 0
    def reset(self):
        self.current = np.random.randint(0, 3)
        return self.current
    def step(self, action):
        expected = (self.current + 1) % 3
        correct = (action == expected)
        if correct:
            self.current = expected
        else:
            self.current = np.random.randint(0, 3)
        return correct, self.current

env = SequenceMemory()
sieve = WaveSieve()
current = env.reset()
window = []
for step in range(20000):
    if step < 200:
        action = np.random.randint(0, 3)
    else:
        action = sieve.choose_action(current, num_actions=3)
    sieve.observe(current, action, num_actions=3)
    is_correct, current = env.step(action)
    if is_correct:
        sieve.signal_success()
    else:
        sieve.signal_death()
    window.append(1 if is_correct else 0)
    if len(window) > 500: window.pop(0)
    if (step + 1) % 2000 == 0:
        recent = 100 * sum(window) / len(window)
        stats = sieve.get_stats()
        print(f"  Step {step+1:6d}: Recent500={recent:.1f}%  "
              f"[modes={stats['n_modes']} heat={stats['heat_bath']:.4f} "
              f"total_E={stats['total_energy']:.3f}]")


# ============ EXTENDED SUDOKU TEST ============
print("\n" + "=" * 60)
print("EXTENDED SUDOKU: 20,000 puzzles")
print("=" * 60)

class MiniSudoku:
    def __init__(self):
        self.size = 4
        self.reset()
    def reset(self):
        self.solution = np.array([
            [1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]
        ], dtype=np.int8)
        self.board = self.solution.copy()
        self.empty_cells = []
        positions = [(i,j) for i in range(4) for j in range(4)]
        np.random.shuffle(positions)
        for pos in positions[:6]:
            self.board[pos] = 0
            self.empty_cells.append(pos)
        self.current_cell_idx = 0
    def get_current_cell(self):
        if self.current_cell_idx < len(self.empty_cells):
            return self.empty_cells[self.current_cell_idx]
        return None
    def get_config(self, target_cell):
        config = {}
        for r in range(4):
            for c in range(4):
                config[(r,c)] = int(self.board[r,c])
        config[('target', target_cell[0], target_cell[1])] = -1
        return config
    def step(self, digit):
        cell = self.get_current_cell()
        if cell is None: return True, True
        correct_digit = self.solution[cell[0], cell[1]]
        if digit == correct_digit:
            self.board[cell[0], cell[1]] = digit
            self.current_cell_idx += 1
            done = self.current_cell_idx >= len(self.empty_cells)
            return done, True
        else:
            return True, False

def sudoku_neighbors(pos):
    if isinstance(pos, tuple) and len(pos) == 3 and pos[0] == 'target':
        r, c = pos[1], pos[2]
    elif isinstance(pos, tuple) and len(pos) == 2:
        r, c = pos
    else:
        return []
    nbrs = set()
    for c2 in range(4):
        if c2 != c: nbrs.add((r, c2))
    for r2 in range(4):
        if r2 != r: nbrs.add((r2, c))
    br, bc = 2*(r//2), 2*(c//2)
    for r2 in range(br, br+2):
        for c2 in range(bc, bc+2):
            if (r2,c2) != (r,c): nbrs.add((r2,c2))
    return list(nbrs)

sieve2 = WaveSieve()
solved = 0
total_moves = 0
correct_moves = 0

window_size = 500
recent_solved = []
recent_moves_correct = []
recent_moves_total = []

for puzzle in range(20000):
    env2 = MiniSudoku()
    env2.reset()
    sieve2.reset_episode()

    puzzle_correct = 0
    puzzle_total = 0
    puzzle_solved = False

    while True:
        cell = env2.get_current_cell()
        if cell is None:
            puzzle_solved = True
            break
        config = env2.get_config(cell)
        target_pos = ('target', cell[0], cell[1])

        if puzzle < 300:
            action = np.random.randint(0, 4)
        else:
            action = sieve2.choose_action(config, num_actions=4,
                                          neighbor_fn=sudoku_neighbors,
                                          query_pos=target_pos)
        sieve2.observe(config, action, num_actions=4)
        puzzle_total += 1
        total_moves += 1

        digit = action + 1
        done, correct = env2.step(digit)

        if correct:
            puzzle_correct += 1
            correct_moves += 1
            if done:
                puzzle_solved = True
                sieve2.signal_success()
                break
        else:
            sieve2.signal_death()
            break

    if puzzle_solved:
        solved += 1

    recent_solved.append(1 if puzzle_solved else 0)
    recent_moves_correct.append(puzzle_correct)
    recent_moves_total.append(puzzle_total)
    if len(recent_solved) > window_size:
        recent_solved.pop(0)
        recent_moves_correct.pop(0)
        recent_moves_total.pop(0)

    if (puzzle + 1) % 2000 == 0:
        r_solve = 100 * sum(recent_solved) / len(recent_solved)
        r_moves = 100 * sum(recent_moves_correct) / max(1, sum(recent_moves_total))
        cumul_solve = 100 * solved / (puzzle + 1)
        cumul_moves = 100 * correct_moves / max(1, total_moves)
        stats = sieve2.get_stats()
        print(f"  Puzzle {puzzle+1:6d}: Recent500 Solve={r_solve:.1f}% "
              f"Moves={r_moves:.1f}%  |  Cumul Solve={cumul_solve:.1f}% "
              f"Moves={cumul_moves:.1f}%  [modes={stats['n_modes']} "
              f"heat={stats['heat_bath']:.4f} total_E={stats['total_energy']:.3f}]")


# ============ EXTENDED TICTACTOE TEST ============
print("\n" + "=" * 60)
print("EXTENDED TICTACTOE: 10,000 games")
print("=" * 60)

class TicTacToe:
    def __init__(self):
        self.reset()
    def reset(self):
        self.board = np.zeros((3,3), dtype=np.int8)
        self.current_player = 1
        return self.get_config()
    def get_config(self):
        config = {}
        for r in range(3):
            for c in range(3):
                config[(r,c)] = int(self.board[r,c])
        return config
    def get_valid_actions(self):
        return [i for i in range(9) if self.board[i//3, i%3] == 0]
    def step(self, action):
        if action not in self.get_valid_actions():
            return self.get_config(), True, -1
        self.board[action//3, action%3] = self.current_player
        if self._check_win(self.current_player):
            return self.get_config(), True, 1 if self.current_player == 1 else -1
        if len(self.get_valid_actions()) == 0:
            return self.get_config(), True, 0
        self.current_player *= -1
        if self.current_player == -1:
            valid = self.get_valid_actions()
            if valid:
                opp_action = np.random.choice(valid)
                self.board[opp_action//3, opp_action%3] = -1
                if self._check_win(-1):
                    return self.get_config(), True, -1
                if len(self.get_valid_actions()) == 0:
                    return self.get_config(), True, 0
            self.current_player = 1
        return self.get_config(), False, 0
    def _check_win(self, player):
        for i in range(3):
            if all(self.board[i,:] == player): return True
        for i in range(3):
            if all(self.board[:,i] == player): return True
        if all(self.board[i,i] == player for i in range(3)): return True
        if all(self.board[i,2-i] == player for i in range(3)): return True
        return False

def ttt_neighbors(pos):
    if not isinstance(pos, tuple) or len(pos) != 2: return []
    r, c = pos
    nbrs = []
    for r2 in range(3):
        for c2 in range(3):
            if (r2,c2) != (r,c):
                if r2 == r or c2 == c: nbrs.append((r2,c2))
                elif abs(r2-r) == abs(c2-c): nbrs.append((r2,c2))
    return nbrs

env3 = TicTacToe()
sieve3 = WaveSieve()
wins = 0; losses = 0; draws = 0
recent_results = []

for game in range(10000):
    config = env3.reset()
    done = False
    sieve3.reset_episode()
    while not done:
        valid_actions = env3.get_valid_actions()
        if game < 200:
            action = np.random.choice(valid_actions)
        else:
            action = sieve3.choose_action(config, num_actions=9,
                                          neighbor_fn=ttt_neighbors)
            if action not in valid_actions:
                action = np.random.choice(valid_actions)
        sieve3.observe(config, action, num_actions=9)
        config, done, reward = env3.step(action)
    if reward == 1:
        wins += 1; sieve3.signal_success(); recent_results.append('W')
    elif reward == -1:
        losses += 1; sieve3.signal_death(); recent_results.append('L')
    else:
        draws += 1; sieve3.signal_success(); recent_results.append('D')
    if len(recent_results) > 500: recent_results.pop(0)

    if (game + 1) % 2000 == 0:
        rw = 100 * recent_results.count('W') / len(recent_results)
        rl = 100 * recent_results.count('L') / len(recent_results)
        rd = 100 * recent_results.count('D') / len(recent_results)
        total = wins + losses + draws
        stats = sieve3.get_stats()
        print(f"  Game {game+1:6d}: Recent500 W={rw:.1f}% L={rl:.1f}% "
              f"D={rd:.1f}%  |  Cumul W={100*wins/total:.1f}% "
              f"L={100*losses/total:.1f}%  [modes={stats['n_modes']} "
              f"heat={stats['heat_bath']:.4f}]")

print("\n" + "=" * 60)
print("EXTENDED TEST SUMMARY")
print("=" * 60)
print("Look at the trends: still improving, or plateaued?")
