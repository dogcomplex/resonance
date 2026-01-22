"""Sweep cosine threshold to find optimal value."""

import random
import sys
from collections import defaultdict

sys.path.insert(0, '/home/claude')
from pixel_environments import PixelPong, tokenize_diff

class ThresholdSieve:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.obs = []
        self.exact = defaultdict(lambda: defaultdict(int))
        self.classes = {}
        self.rules = []
    
    def observe(self, state, action, next_state):
        before, after = frozenset(state), frozenset(next_state)
        effect = frozenset({f"+{t}" for t in (after-before)} | {f"-{t}" for t in (before-after)})
        self.exact[(before, action)][effect] += 1
        self.obs.append({'before': before, 'effect': effect, 'action': action})
    
    def build(self):
        counts = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        for o in self.obs:
            for t in o['before']:
                totals[t] += 1
                for e in o['effect']: counts[t][e] += 1
        
        tokens = list(totals.keys())
        self.classes = {t: t for t in tokens}
        merged = 0
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2: continue
                p1, p2 = t1.split('_'), t2.split('_')
                if len(p1) != len(p2) or sum(a!=b for a,b in zip(p1,p2)) > 1: continue
                if totals[t1] < 5 or totals[t2] < 5: continue
                all_e = set(counts[t1]) | set(counts[t2])
                if not all_e: continue
                dot = sum(counts[t1].get(e,0)/totals[t1] * counts[t2].get(e,0)/totals[t2] for e in all_e)
                m1 = sum((counts[t1].get(e,0)/totals[t1])**2 for e in all_e)**0.5
                m2 = sum((counts[t2].get(e,0)/totals[t2])**2 for e in all_e)**0.5
                if m1 > 0 and m2 > 0 and dot/(m1*m2) > self.threshold:
                    self.classes[t2] = self.classes[t1]
                    merged += 1
        
        def abstract(s): return frozenset(self.classes.get(t,t) for t in s)
        
        by_ae = defaultdict(list)
        for o in self.obs:
            if o['effect']:
                by_ae[(o['action'], abstract(o['effect']))].append(abstract(o['before']))
        
        self.rules = []
        for (action, effect), pos in by_ae.items():
            if len(pos) >= 2:
                lhs = frozenset.intersection(*random.sample(pos, min(10, len(pos))))
                if lhs: self.rules.append({'lhs': lhs, 'effect': effect, 'action': action})
        
        return merged
    
    def predict(self, state, action):
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x:x[1])[0])
        abstract_state = frozenset(self.classes.get(t,t) for t in state)
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        return set()

def test(threshold, n_train=50):
    random.seed(42)
    learner = ThresholdSieve(threshold)
    seen = set()
    
    for ep in range(n_train):
        env = PixelPong(seed=ep)
        screen = env.reset()
        prev = screen.copy()
        state = tokenize_diff(screen, prev, 7)
        for _ in range(50):
            seen.add(frozenset(state))
            action = random.randint(0,2)
            nxt, _, done, _ = env.step(action)
            nxt_state = tokenize_diff(nxt, screen, 7)
            learner.observe(state, action, nxt_state)
            prev, screen, state = screen, nxt, nxt_state
            if done: break
    
    merged = learner.build()
    
    tp_u, fp_u, fn_u = 0, 0, 0
    for ep in range(20):
        env = PixelPong(seed=50000+ep)
        screen = env.reset()
        prev = screen.copy()
        state = tokenize_diff(screen, prev, 7)
        for _ in range(50):
            action = random.randint(0,2)
            nxt, _, done, _ = env.step(action)
            nxt_state = tokenize_diff(nxt, screen, 7)
            
            if frozenset(state) not in seen:  # UNSEEN only
                actual = {f"+{t}" for t in (nxt_state-state)} | {f"-{t}" for t in (state-nxt_state)}
                pred = learner.predict(state, action)
                for e in pred:
                    if e in actual: tp_u += 1
                    else: fp_u += 1
                for e in actual:
                    if e not in pred: fn_u += 1
            
            prev, screen, state = screen, nxt, nxt_state
            if done: break
    
    f1_u = 2*tp_u/(2*tp_u+fp_u+fn_u) if (2*tp_u+fp_u+fn_u) else 0
    return f1_u, merged, len(learner.rules)

print("Cosine Threshold Sweep (N=50 episodes)")
print("="*55)
print(f"{'Threshold':<12}{'UNSEEN':>12}{'Merged':>12}{'Rules':>12}")
print("-"*55)

for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
    u, m, r = test(thresh)
    print(f"{thresh:<12.2f}{u:>12.1%}{m:>12}{r:>12}")
