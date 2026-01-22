"""
Test on stochastic state machine.
This tests the probability dimension.
"""
import random, sys
sys.path.insert(0, '/home/claude')
from resonance_crystal_fixed import ResonanceCrystalFixed

class StochasticStateMachine:
    """
    Stochastic transitions - tests probability handling.
    
    Rules:
    - A + action=0 → B (80%) or C (20%)
    - A + action=1 → C (70%) or D (30%)
    - B + action=0 → C (90%) or A (10%)
    - etc.
    """
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.state = 'A'
    
    def reset(self):
        self.state = self.rng.choice(['A', 'B', 'C', 'D'])
        return {f"state_{self.state}"}
    
    def step(self, action):
        transitions = {
            ('A', 0): [('B', 0.8), ('C', 0.2)],
            ('A', 1): [('C', 0.7), ('D', 0.3)],
            ('B', 0): [('C', 0.9), ('A', 0.1)],
            ('B', 1): [('D', 0.6), ('B', 0.4)],
            ('C', 0): [('D', 0.85), ('C', 0.15)],
            ('C', 1): [('A', 0.75), ('B', 0.25)],
            ('D', 0): [('A', 0.95), ('D', 0.05)],
            ('D', 1): [('B', 0.5), ('C', 0.5)],
        }
        
        options = transitions[(self.state, action)]
        r = self.rng.random()
        cumsum = 0
        for next_state, prob in options:
            cumsum += prob
            if r < cumsum:
                self.state = next_state
                break
        
        return {f"state_{self.state}"}

def test_stochastic(seed=42, n_train=200, n_test=50):
    random.seed(seed)
    learner = ResonanceCrystalFixed(coherence=0.90, include_delta=False)
    seen = set()
    
    for ep in range(n_train):
        env = StochasticStateMachine(seed=ep)
        state = env.reset()
        learner.reset_episode()
        for _ in range(30):
            seen.add(frozenset(state))
            action = random.randint(0, 1)
            nxt_state = env.step(action)
            learner.observe(state, action, nxt_state)
            state = nxt_state
    
    learner.build()
    
    # Test: does it predict the MOST LIKELY outcome?
    correct, total = 0, 0
    for ep in range(n_test):
        env = StochasticStateMachine(seed=50000 + ep)
        state = env.reset()
        for _ in range(30):
            action = random.randint(0, 1)
            
            # Get prediction
            pred = learner.predict(state, action)
            
            # Get actual (but we're checking if pred matches most likely)
            nxt_state = env.step(action)
            actual = {f"+{t}" for t in (nxt_state - state)} | {f"-{t}" for t in (state - nxt_state)}
            
            # Score
            if pred:
                total += 1
                if pred == actual:
                    correct += 1
            
            state = nxt_state
    
    acc = correct / total if total else 0
    return acc, len(learner.rules), len(set(learner.classes.values()))

print("STOCHASTIC STATE MACHINE TEST")
print("="*50)

seeds = [42, 123, 456]
for s in seeds:
    acc, rules, classes = test_stochastic(seed=s)
    print(f"Seed {s}: Acc={acc:.1%}, {rules} rules, {classes} classes")

avg = sum(test_stochastic(seed=s)[0] for s in seeds)/3
print(f"\nAverage accuracy: {avg:.1%}")
print("(Expected upper bound ~75% since predicting most likely outcome)")

# Check what rules it learned
random.seed(42)
learner = ResonanceCrystalFixed(coherence=0.90, include_delta=False)
for ep in range(200):
    env = StochasticStateMachine(seed=ep)
    state = env.reset()
    learner.reset_episode()
    for _ in range(30):
        action = random.randint(0, 1)
        nxt_state = env.step(action)
        learner.observe(state, action, nxt_state)
        state = nxt_state
learner.build()

print(f"\nLearned {len(learner.rules)} rules (showing first 8):")
for r in learner.rules[:8]:
    lhs = list(r['lhs'])[0] if len(r['lhs']) == 1 else r['lhs']
    eff = list(r['effect'])
    print(f"  {lhs} + a={r['action']} → {eff} (temp={r['temperature']:.2f})")
