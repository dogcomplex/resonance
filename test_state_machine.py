"""
Test on a simple deterministic state machine.
This should get ~100% if the sieve is working correctly.
"""
import random, sys
sys.path.insert(0, '/home/claude')
from resonance_crystal_fixed import ResonanceCrystalFixed

class SimpleStateMachine:
    """
    A deterministic state machine with clear rules:
    
    States: A, B, C, D
    Actions: 0, 1
    
    Rules:
    - A + action=0 → B
    - A + action=1 → C
    - B + action=0 → C
    - B + action=1 → D
    - C + action=0 → D
    - C + action=1 → A
    - D + action=0 → A
    - D + action=1 → B
    """
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.state = 'A'
    
    def reset(self):
        self.state = self.rng.choice(['A', 'B', 'C', 'D'])
        return {f"state_{self.state}"}
    
    def step(self, action):
        transitions = {
            ('A', 0): 'B', ('A', 1): 'C',
            ('B', 0): 'C', ('B', 1): 'D',
            ('C', 0): 'D', ('C', 1): 'A',
            ('D', 0): 'A', ('D', 1): 'B',
        }
        self.state = transitions[(self.state, action)]
        return {f"state_{self.state}"}

def test_state_machine(seed=42, n_train=50, n_test=20):
    random.seed(seed)
    learner = ResonanceCrystalFixed(coherence=0.90, include_delta=False)
    seen = set()
    
    for ep in range(n_train):
        env = SimpleStateMachine(seed=ep)
        state = env.reset()
        learner.reset_episode()
        for _ in range(20):
            seen.add(frozenset(state))
            action = random.randint(0, 1)
            nxt_state = env.step(action)
            learner.observe(state, action, nxt_state)
            state = nxt_state
    
    learner.build()
    
    tp, fp, fn = 0, 0, 0
    for ep in range(n_test):
        env = SimpleStateMachine(seed=50000 + ep)
        state = env.reset()
        for _ in range(20):
            action = random.randint(0, 1)
            nxt_state = env.step(action)
            # Don't filter by seen - we want to test all
            actual = {f"+{t}" for t in (nxt_state - state)} | {f"-{t}" for t in (state - nxt_state)}
            pred = learner.predict(state, action)
            for e in pred:
                if e in actual: tp += 1
                else: fp += 1
            for e in actual:
                if e not in pred: fn += 1
            state = nxt_state
    
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
    return f1, len(learner.rules), len(set(learner.classes.values()))

print("SIMPLE STATE MACHINE TEST")
print("="*50)

seeds = [42, 123, 456]
for s in seeds:
    f1, rules, classes = test_state_machine(seed=s)
    print(f"Seed {s}: F1={f1:.1%}, {rules} rules, {classes} classes")

avg = sum(test_state_machine(seed=s)[0] for s in seeds)/3
print(f"\nAverage: {avg:.1%}")

# Check rules
random.seed(42)
learner = ResonanceCrystalFixed(coherence=0.90, include_delta=False)
for ep in range(50):
    env = SimpleStateMachine(seed=ep)
    state = env.reset()
    learner.reset_episode()
    for _ in range(20):
        action = random.randint(0, 1)
        nxt_state = env.step(action)
        learner.observe(state, action, nxt_state)
        state = nxt_state
learner.build()

print(f"\nLearned {len(learner.rules)} rules:")
for r in learner.rules:
    lhs = list(r['lhs'])[0] if len(r['lhs']) == 1 else r['lhs']
    eff = list(r['effect'])
    print(f"  {lhs} + action={r['action']} → {eff}")
