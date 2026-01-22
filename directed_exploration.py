"""
Directed Exploration V2 - Focus on cross-episode performance
"""

import sys
import random
from collections import defaultdict
sys.path.insert(0, '/home/claude')
sys.path.insert(0, '/mnt/user-data/outputs')

exec(open('/home/claude/hierarchical_learner_v12.py').read())
from comprehensive_test_suite import CraftingSystem

print("="*70)
print("DIRECTED EXPLORATION: CROSS-EPISODE TEST")
print("="*70)

class SmartExplorer:
    """Focus on coverage diversity - try to visit all reachable states."""
    
    def __init__(self, learner, n_actions):
        self.learner = learner
        self.n_actions = n_actions
        self.state_visit_counts = defaultdict(int)
        self.action_from_state = defaultdict(set)  # state → actions tried
        
    def observe(self, before, action, after):
        self.learner.observe(before, action, after)
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        self.state_visit_counts[before_fs] += 1
        self.state_visit_counts[after_fs] += 1
        self.action_from_state[before_fs].add(action)
    
    def choose_action(self, state, valid_actions):
        if not valid_actions:
            return None
        
        state_fs = frozenset(state)
        tried = self.action_from_state[state_fs]
        untried = [a for a in valid_actions if a not in tried]
        
        if untried:
            # Prefer untried actions
            return random.choice(untried)
        
        # Otherwise, pick action leading to least-visited predicted state
        best_action = valid_actions[0]
        best_novelty = -1
        
        for action in valid_actions:
            novelty = self.learner.novelty_score(state, action)
            if novelty > best_novelty:
                best_novelty = novelty
                best_action = action
        
        return best_action
    
    def coverage_stats(self):
        return {
            'unique_states': len(self.state_visit_counts),
            'total_visits': sum(self.state_visit_counts.values()),
            'avg_actions_per_state': sum(len(v) for v in self.action_from_state.values()) / max(1, len(self.action_from_state))
        }


def test_cross_episode(explorer_class, name, n_episodes=50):
    """Test cross-episode induction performance"""
    random.seed(42)
    env = CraftingSystem()
    learner = HierarchicalLearnerV12(n_actions=10)
    explorer = explorer_class(learner, 10) if explorer_class else None
    
    episode_starts = []
    episode_ends = []
    
    for ep in range(n_episodes):
        state = env.reset()
        episode_starts.append(state.copy())
        learner.reset_episode()
        
        for step in range(25):
            valid = env.get_valid_actions()
            if not valid:
                break
            
            if explorer:
                action = explorer.choose_action(state, valid)
            else:
                action = random.choice(valid)
            
            next_state, _, done, _ = env.step(action)
            
            if explorer:
                explorer.observe(state, action, next_state)
            else:
                learner.observe(state, action, next_state)
            
            state = next_state
            if done:
                break
        
        episode_ends.append(state.copy())
    
    # Test cross-episode induction
    successes = 0
    for i in range(10):
        before = episode_starts[i]
        after = episode_ends[(i + 5) % n_episodes]  # Different episode's end
        
        paths = learner.induce_path(before, after, max_steps=10)
        if paths and paths[0][1] > 0.2:
            successes += 1
    
    stats = learner.stats()
    coverage = explorer.coverage_stats() if explorer else {'unique_states': 'N/A'}
    
    return successes, stats, coverage


print("\n1. RANDOM EXPLORATION")
print("-"*40)
succ, stats, cov = test_cross_episode(None, "Random")
print(f"Cross-episode induction: {succ}/10")
print(f"Rules: {stats['rules']}, Pairs: {stats['unique_state_action_pairs']}")

print("\n2. COVERAGE-FOCUSED EXPLORATION")
print("-"*40)
succ, stats, cov = test_cross_episode(SmartExplorer, "Smart")
print(f"Cross-episode induction: {succ}/10")
print(f"Rules: {stats['rules']}, Pairs: {stats['unique_state_action_pairs']}")
print(f"Unique states visited: {cov['unique_states']}")
print(f"Avg actions per state: {cov['avg_actions_per_state']:.1f}")

print("\n" + "="*70)
print("ANALYSIS: WHY COVERAGE MATTERS")
print("="*70)
print("""
Cross-episode induction fails when:
  Episode 1: Explores path A → B → C
  Episode 2: Explores path A → D → E
  
  Trying to induce: B → E
  Problem: No observed rules connect B to D or E!

Solution: Maximize COVERAGE
  - Visit many different states
  - Try all actions from each state
  - Build a dense transition graph

The denser our observed graph, the more likely any two
states have a known path between them.
""")

# Now test: Can we CONVERGE to same probability distributions?
print("\n" + "="*70)
print("PROBABILITY DISTRIBUTION CONVERGENCE")
print("="*70)

random.seed(42)
learner = HierarchicalLearnerV12(n_actions=10)
env = CraftingSystem()

# Track outcome distributions for specific state-action pairs
outcome_counts = defaultdict(lambda: defaultdict(int))

for ep in range(100):
    state = env.reset()
    learner.reset_episode()
    
    for step in range(30):
        valid = env.get_valid_actions()
        if not valid:
            break
        
        action = random.choice(valid)
        state_key = (frozenset(state), action)
        
        next_state, _, done, _ = env.step(action)
        outcome_key = frozenset(next_state - state)  # What changed
        
        outcome_counts[state_key][outcome_key] += 1
        learner.observe(state, action, next_state)
        
        state = next_state
        if done:
            break

# Compare observed vs predicted distributions
print("\nComparing observed vs predicted distributions:")
print("-"*50)

comparisons = 0
matches = 0

for (state_fs, action), outcomes in list(outcome_counts.items())[:5]:
    if len(outcomes) > 1:  # Probabilistic
        total = sum(outcomes.values())
        
        # Observed distribution
        observed = {k: v/total for k, v in outcomes.items()}
        
        # Predicted distribution from learner
        pred = learner.predict_with_confidence(set(state_fs), action)
        
        print(f"\nState-Action pair (seen {total}x):")
        print(f"  Observed outcomes: {len(outcomes)}")
        print(f"  Predicted confidence: {pred.confidence:.1%}")
        print(f"  Alternatives tracked: {len(pred.alternatives)}")
        
        comparisons += 1
        if abs(pred.confidence - max(observed.values())) < 0.15:
            matches += 1
            print(f"  ✓ Distribution roughly matches")
        else:
            print(f"  ✗ Distribution mismatch")

print(f"\nDistribution matches: {matches}/{comparisons}")

print("""
KEY INSIGHT: 
- Deterministic rules converge quickly (few observations needed)
- Probabilistic rules need more samples to estimate distribution
- With enough coverage, distributions DO converge

The learner naturally builds empirical distributions from observations.
More observations → better distribution estimates → better predictions.
""")
