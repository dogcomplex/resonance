"""Compare different similarity functions for token equivalence."""

import random
import sys
from collections import defaultdict

sys.path.insert(0, '/home/claude')
from pixel_environments import PixelPong, tokenize_diff

def jaccard(c1, c2, t1, t2):
    """Jaccard similarity of effect sets."""
    s1 = set(c1.keys())
    s2 = set(c2.keys())
    if not s1 and not s2: return 1.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0

def cosine(c1, c2, t1, t2):
    """Cosine of normalized effect vectors."""
    all_e = set(c1) | set(c2)
    if not all_e: return 1.0
    dot = sum(c1.get(e,0)/t1 * c2.get(e,0)/t2 for e in all_e)
    m1 = sum((c1.get(e,0)/t1)**2 for e in all_e)**0.5
    m2 = sum((c2.get(e,0)/t2)**2 for e in all_e)**0.5
    return dot/(m1*m2) if m1 and m2 else 0

def dice(c1, c2, t1, t2):
    """Dice coefficient (like F1)."""
    s1, s2 = set(c1.keys()), set(c2.keys())
    if not s1 and not s2: return 1.0
    inter = len(s1 & s2)
    return 2*inter / (len(s1)+len(s2)) if (len(s1)+len(s2)) else 0

class SimilaritySieve:
    def __init__(self, sim_fn, threshold):
        self.sim_fn = sim_fn
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
        
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if self.classes[t2] != t2: continue
                p1, p2 = t1.split('_'), t2.split('_')
                if len(p1) != len(p2) or sum(a!=b for a,b in zip(p1,p2)) > 1: continue
                if totals[t1] < 5 or totals[t2] < 5: continue
                
                sim = self.sim_fn(counts[t1], counts[t2], totals[t1], totals[t2])
                if sim > self.threshold:
                    self.classes[t2] = self.classes[t1]
        
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
    
    def predict(self, state, action):
        key = (frozenset(state), action)
        if key in self.exact and self.exact[key]:
            return set(max(self.exact[key].items(), key=lambda x:x[1])[0])
        abstract_state = frozenset(self.classes.get(t,t) for t in state)
        for r in self.rules:
            if r['action'] == action and r['lhs'] <= abstract_state:
                return set(r['effect'])
        return set()

def test(sim_fn, threshold, n_train=50):
    random.seed(42)
    learner = SimilaritySieve(sim_fn, threshold)
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
    
    learner.build()
    
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
            
            if frozenset(state) not in seen:
                actual = {f"+{t}" for t in (nxt_state-state)} | {f"-{t}" for t in (state-nxt_state)}
                pred = learner.predict(state, action)
                for e in pred:
                    if e in actual: tp_u += 1
                    else: fp_u += 1
                for e in actual:
                    if e not in pred: fn_u += 1
            
            prev, screen, state = screen, nxt, nxt_state
            if done: break
    
    return 2*tp_u/(2*tp_u+fp_u+fn_u) if (2*tp_u+fp_u+fn_u) else 0

print("Similarity Function Comparison (N=50, Pong)")
print("="*50)
print(f"{'Function':<12}{'Thresh':>10}{'UNSEEN':>12}")
print("-"*50)

for name, fn in [("Cosine", cosine), ("Jaccard", jaccard), ("Dice", dice)]:
    for thresh in [0.7, 0.9]:
        u = test(fn, thresh)
        print(f"{name:<12}{thresh:>10.1f}{u:>12.1%}")

print("-"*50)
print("No merging (baseline):")
print(f"{'None':<12}{'1.0':>10}{test(cosine, 1.0):>12.1%}")
