"""
BACKTEST - Test ResonanceCrystal across multiple environments

Environments:
1. PixelPong - Continuous physics, spatial abstraction
2. Minesweeper - Logic deduction, partial information
3. Game2048 - Numeric rules, deterministic merging
4. Snake - Growing state, self-collision
5. LightSwitch - State machines, boolean dependencies
"""

import random
import sys
sys.path.insert(0, '/home/claude')

from pixel_environments import PixelPong, tokenize_diff
from new_game_environments import Minesweeper, Game2048
from resonance_crystal_fixed import ResonanceCrystalFixed

def test_pong(seed=42, n_train=50, n_test=15):
    """Test on PixelPong with grid tokenization."""
    random.seed(seed)
    learner = ResonanceCrystalFixed(coherence=0.95, include_delta=False)
    seen = set()
    
    for ep in range(n_train):
        env = PixelPong(seed=ep)
        screen = env.reset()
        state = tokenize_diff(screen, screen.copy(), 7)
        learner.reset_episode()
        for _ in range(50):
            seen.add(frozenset(state))
            action = random.randint(0, 2)
            nxt, _, done, _ = env.step(action)
            nxt_state = tokenize_diff(nxt, screen, 7)
            learner.observe(state, action, nxt_state)
            screen, state = nxt, nxt_state
            if done: break
    
    learner.build()
    
    tp, fp, fn = 0, 0, 0
    for ep in range(n_test):
        env = PixelPong(seed=50000 + ep)
        screen = env.reset()
        state = tokenize_diff(screen, screen.copy(), 7)
        for _ in range(40):
            action = random.randint(0, 2)
            nxt, _, done, _ = env.step(action)
            nxt_state = tokenize_diff(nxt, screen, 7)
            if frozenset(state) not in seen:
                actual = {f"+{t}" for t in (nxt_state - state)} | {f"-{t}" for t in (state - nxt_state)}
                pred = learner.predict(state, action)
                for e in pred:
                    if e in actual: tp += 1
                    else: fp += 1
                for e in actual:
                    if e not in pred: fn += 1
            screen, state = nxt, nxt_state
            if done: break
    
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    return f1, len(learner.rules), len(set(learner.classes.values()))

def test_minesweeper(seed=42, n_train=100, n_test=30):
    """Test on Minesweeper - logic deduction."""
    random.seed(seed)
    learner = ResonanceCrystalFixed(coherence=0.90, include_delta=False)  # Lower coherence for discrete
    seen = set()
    
    for ep in range(n_train):
        env = Minesweeper(seed=ep)
        state = env.reset()
        learner.reset_episode()
        for _ in range(25):
            seen.add(frozenset(state))
            valid = env.get_valid_actions()
            if not valid: break
            action = random.choice(valid)
            nxt_state, _, done, _ = env.step(action)
            learner.observe(state, action, nxt_state)
            state = nxt_state
            if done: break
    
    learner.build()
    
    tp, fp, fn = 0, 0, 0
    for ep in range(n_test):
        env = Minesweeper(seed=50000 + ep)
        state = env.reset()
        for _ in range(25):
            valid = env.get_valid_actions()
            if not valid: break
            action = random.choice(valid)
            nxt_state, _, done, _ = env.step(action)
            if frozenset(state) not in seen:
                actual = (nxt_state - state) | {f"-{t}" for t in (state - nxt_state) if not t.startswith('-')}
                pred = learner.predict(state, action)
                # Simplified scoring for set environments
                pred_set = {e.lstrip('+-') for e in pred}
                actual_set = {e.lstrip('+-') for e in actual}
                tp += len(pred_set & actual_set)
                fp += len(pred_set - actual_set)
                fn += len(actual_set - pred_set)
            state = nxt_state
            if done: break
    
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    return f1, len(learner.rules), len(set(learner.classes.values()))

def test_2048(seed=42, n_train=100, n_test=30):
    """Test on 2048 - numeric combination rules."""
    random.seed(seed)
    learner = ResonanceCrystalFixed(coherence=0.90, include_delta=False)
    seen = set()
    
    for ep in range(n_train):
        env = Game2048(seed=ep)
        state = env.reset()
        learner.reset_episode()
        for _ in range(50):
            seen.add(frozenset(state))
            action = random.randint(0, 3)
            nxt_state, _, done, _ = env.step(action)
            learner.observe(state, action, nxt_state)
            state = nxt_state
            if done: break
    
    learner.build()
    
    tp, fp, fn = 0, 0, 0
    for ep in range(n_test):
        env = Game2048(seed=50000 + ep)
        state = env.reset()
        for _ in range(50):
            action = random.randint(0, 3)
            nxt_state, _, done, _ = env.step(action)
            if frozenset(state) not in seen:
                actual = (nxt_state - state) | {f"-{t}" for t in (state - nxt_state) if not t.startswith('-')}
                pred = learner.predict(state, action)
                pred_set = {e.lstrip('+-') for e in pred}
                actual_set = {e.lstrip('+-') for e in actual}
                tp += len(pred_set & actual_set)
                fp += len(pred_set - actual_set)
                fn += len(actual_set - pred_set)
            state = nxt_state
            if done: break
    
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    return f1, len(learner.rules), len(set(learner.classes.values()))

# Run tests
print("RESONANCE CRYSTAL BACKTEST")
print("="*60)

seeds = [42, 123, 456]

print("\n1. PONG (continuous physics, spatial abstraction)")
for s in seeds:
    f1, rules, classes = test_pong(seed=s)
    print(f"   Seed {s}: F1={f1:.1%}, {rules} rules, {classes} classes")
avg = sum(test_pong(seed=s)[0] for s in seeds)/3
print(f"   Average: {avg:.1%}")

print("\n2. MINESWEEPER (logic deduction, partial info)")
for s in seeds:
    f1, rules, classes = test_minesweeper(seed=s)
    print(f"   Seed {s}: F1={f1:.1%}, {rules} rules, {classes} classes")
avg = sum(test_minesweeper(seed=s)[0] for s in seeds)/3
print(f"   Average: {avg:.1%}")

print("\n3. 2048 (numeric combination rules)")
for s in seeds:
    f1, rules, classes = test_2048(seed=s)
    print(f"   Seed {s}: F1={f1:.1%}, {rules} rules, {classes} classes")
avg = sum(test_2048(seed=s)[0] for s in seeds)/3
print(f"   Average: {avg:.1%}")
