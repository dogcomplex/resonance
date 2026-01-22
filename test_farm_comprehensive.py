"""
Comprehensive Farm Game Test - Direct Rule Sampling

This test:
1. Generates synthetic states where each rule can fire
2. Observes the transition (with anonymized tokens)
3. Tests if the learner correctly predicts outcomes

This tests RULE LEARNING, not exploration.
The learner must discover patterns from observations alone.
"""

import sys
sys.path.insert(0, '/home/claude')
sys.path.insert(0, '/mnt/user-data/outputs')

from farm_game_v3 import parse_recipes, Token
from hierarchical_learner_v11 import HierarchicalLearner as V11
from hierarchical_learner_v9 import HierarchicalLearner as V9
import random
from collections import defaultdict


class RuleTestOracle:
    """Generate test cases for each rule"""
    
    def __init__(self, rules, anon_seed=999):
        self.rules = rules
        self.rng = random.Random(42)
        self.anon_rng = random.Random(anon_seed)
        
        # Build anonymization
        all_tokens = set()
        for rule in rules:
            for t in rule.lhs:
                all_tokens.add(t.base)
            for o in rule.rhs:
                for t in o.tokens:
                    all_tokens.add(t.base)
        
        tokens_list = sorted(all_tokens)
        self.anon_rng.shuffle(tokens_list)
        self.tok2anon = {t: f"T{i:03d}" for i, t in enumerate(tokens_list)}
        
        # Anonymize actions
        actions_list = [r.id for r in rules]
        self.anon_rng.shuffle(actions_list)
        self.act2anon = {a: i for i, a in enumerate(actions_list)}
        self.anon2act = {i: a for a, i in self.act2anon.items()}
        
        self.n_actions = len(actions_list)
    
    def _anon_state(self, raw):
        """Convert raw state to anonymized tokens"""
        tokens = set()
        for base, count in raw.items():
            if count > 0 and base in self.tok2anon:
                anon = self.tok2anon[base]
                if count > 100:
                    bucket = "100+"
                elif count > 10:
                    bucket = f"{(count//10)*10}"
                else:
                    bucket = str(count)
                tokens.add(f"{anon}_{bucket}")
        return tokens
    
    def generate_transition(self, rule, seed=None):
        """Generate one transition for this rule"""
        if seed is not None:
            self.rng = random.Random(seed)
        
        # Build minimal state where rule applies
        before = defaultdict(int)
        for token in rule.lhs:
            if token.consume_all:
                before[token.base] = self.rng.randint(0, 5)
            else:
                before[token.base] = token.quantity
        
        # Apply rule
        after = defaultdict(int, before)
        
        # Consume
        for token in rule.lhs:
            if token.consume_all:
                after[token.base] = 0
            else:
                after[token.base] -= token.quantity
        
        # Produce
        if rule.rhs:
            total_prob = sum(o.probability for o in rule.rhs)
            
            if len(rule.rhs) == 1 and rule.rhs[0].probability >= 100:
                outcome = rule.rhs[0]
            else:
                roll = self.rng.random() * 100
                outcome = None
                if roll < total_prob:
                    cumulative = 0
                    for o in rule.rhs:
                        cumulative += o.probability
                        if roll < cumulative:
                            outcome = o
                            break
            
            if outcome:
                for token in outcome.tokens:
                    if token.is_range:
                        qty = self.rng.randint(token.range_low, token.range_high)
                    else:
                        qty = token.quantity
                    after[token.base] += qty
        
        before_clean = {k: v for k, v in before.items() if v > 0}
        after_clean = {k: v for k, v in after.items() if v > 0}
        
        return (
            self._anon_state(before_clean),
            self.act2anon[rule.id],
            self._anon_state(after_clean)
        )


def test_learner(learner_cls, oracle, n_samples_per_rule=5, train_ratio=0.8):
    """
    Test learner on all rules.
    
    For each rule:
    1. Generate multiple samples (different RNG seeds)
    2. Use 80% for training, 20% for testing
    3. Measure prediction accuracy
    """
    learner = learner_cls(n_actions=oracle.n_actions)
    
    # Generate all samples
    all_samples = []
    for rule in oracle.rules:
        for i in range(n_samples_per_rule):
            seed = hash((rule.id, i)) % 1000000
            before, action, after = oracle.generate_transition(rule, seed=seed)
            all_samples.append((before, action, after, rule.id))
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Split train/test
    split = int(len(all_samples) * train_ratio)
    train = all_samples[:split]
    test = all_samples[split:]
    
    # Track which (state, action) pairs we've seen
    seen = set()
    
    # Train
    for before, action, after, rule_id in train:
        seen.add((frozenset(before), action))
        learner.observe(before, action, after)
    
    # Test
    results = {'det': {'tp': 0, 'fp': 0, 'fn': 0, 'n': 0},
               'prob': {'tp': 0, 'fp': 0, 'fn': 0, 'n': 0}}
    
    # Track outcomes per (state, action) to identify probabilistic
    outcomes_seen = defaultdict(set)
    for before, action, after, _ in train:
        key = (frozenset(before), action)
        effects = frozenset({f"+{t}" for t in (after - before)} | 
                           {f"-{t}" for t in (before - after)})
        outcomes_seen[key].add(effects)
    
    for before, action, after, rule_id in test:
        key = (frozenset(before), action)
        
        # Only test on seen state-action pairs
        if key not in seen:
            continue
        
        # Determine if probabilistic
        is_prob = len(outcomes_seen[key]) > 1
        cat = 'prob' if is_prob else 'det'
        results[cat]['n'] += 1
        
        predicted = learner.predict(before, action)
        actual = {f"+{t}" for t in (after - before)} | {f"-{t}" for t in (before - after)}
        
        for e in predicted:
            if e in actual:
                results[cat]['tp'] += 1
            else:
                results[cat]['fp'] += 1
        for e in actual:
            if e not in predicted:
                results[cat]['fn'] += 1
    
    if hasattr(learner, 'close'):
        learner.close()
    
    return results, len(train), len(seen)


def main():
    rules = parse_recipes('/mnt/user-data/uploads/recipes.csv')
    print(f"Loaded {len(rules)} rules from Farm Game")
    
    oracle = RuleTestOracle(rules, anon_seed=999)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE FARM GAME TEST - ALL 380 RULES")
    print("="*70)
    print("\nThis test generates synthetic states for EVERY rule,")
    print("trains the learner on 80%, and tests on 20%.\n")
    
    for learner_cls, name in [(V9, "V9"), (V11, "V11")]:
        print(f"\n--- Testing {name} ---")
        
        results, n_train, n_seen = test_learner(
            learner_cls, oracle, 
            n_samples_per_rule=10,  # 10 samples per rule = 3800 total
            train_ratio=0.8
        )
        
        print(f"Training samples: {n_train}")
        print(f"Unique state-action pairs: {n_seen}")
        
        for cat, label in [('det', 'Deterministic'), ('prob', 'Probabilistic')]:
            r = results[cat]
            if r['n'] == 0:
                continue
            
            prec = r['tp'] / (r['tp'] + r['fp']) if (r['tp'] + r['fp']) > 0 else 0
            rec = r['tp'] / (r['tp'] + r['fn']) if (r['tp'] + r['fn']) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            print(f"\n  {label}:")
            print(f"    Tested: {r['n']}")
            print(f"    Precision: {prec:.1%}")
            print(f"    Recall: {rec:.1%}")
            print(f"    F1: {f1:.1%}")
        
        # Overall
        total_tp = sum(r['tp'] for r in results.values())
        total_fp = sum(r['fp'] for r in results.values())
        total_fn = sum(r['fn'] for r in results.values())
        total_n = sum(r['n'] for r in results.values())
        
        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
        
        print(f"\n  OVERALL ({total_n} tests):")
        print(f"    Precision: {overall_prec:.1%}")
        print(f"    Recall: {overall_rec:.1%}")
        print(f"    F1: {overall_f1:.1%}")


if __name__ == "__main__":
    random.seed(42)
    main()
