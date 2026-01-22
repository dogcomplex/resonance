"""
CURIOSITY-DRIVEN AGENT

Uses the accurate world model with curiosity-based exploration:
1. Prefer actions leading to less-visited states
2. Try all actions on novel front tokens
3. BFS toward interesting tokens (not just "goal" - any rare token)
4. Use learned rules for planning

NO DOMAIN KNOWLEDGE - just graph exploration on learned state space.
"""
import random
from collections import defaultdict, deque
from typing import Set, Dict, List, Tuple, Optional, FrozenSet

class CuriosityAgent:
    """Agent with curiosity-driven exploration."""
    
    def __init__(self, n_actions=6):
        self.n_actions = n_actions
        
        # World model: (front_token, action) -> {effect: count}
        self.rules = defaultdict(lambda: defaultdict(int))
        
        # State graph: state_hash -> {action: [next_state_hash, ...]}
        self.state_graph = defaultdict(lambda: defaultdict(list))
        
        # Visit counts for curiosity
        self.state_visits = defaultdict(int)
        self.front_action_tries = defaultdict(int)
        
        # Token tracking
        self.all_tokens = set()
        self.token_counts = defaultdict(int)
        self.success_tokens = defaultdict(int)
        
        # Success patterns: (front, action) that led to success
        self.success_patterns = []
        
    def get_front(self, tokens: Set[str]) -> str:
        """Extract front=X token."""
        for t in tokens:
            if t.startswith('front='):
                return t
        return 'front=unknown'
    
    def state_hash(self, tokens: Set[str]) -> FrozenSet[str]:
        """Hash state for graph."""
        return frozenset(tokens)
    
    def observe(self, before: Set[str], action: int, after: Set[str], success: bool = False):
        """Learn from transition."""
        front = self.get_front(before)
        
        # Effect
        added = frozenset(after - before)
        removed = frozenset(before - after)
        effect = (added, removed)
        
        # Update world model
        self.rules[(front, action)][effect] += 1
        self.front_action_tries[(front, action)] += 1
        
        # Update state graph
        before_h = self.state_hash(before)
        after_h = self.state_hash(after)
        if after_h not in self.state_graph[before_h][action]:
            self.state_graph[before_h][action].append(after_h)
        
        # Update visits
        self.state_visits[after_h] += 1
        
        # Update token tracking
        for t in after:
            self.all_tokens.add(t)
            self.token_counts[t] += 1
        
        # Record success
        if success:
            for t in after:
                self.success_tokens[t] += 1
            self.success_patterns.append((front, action))
    
    def get_unexplored_actions(self, tokens: Set[str]) -> List[int]:
        """Get actions we haven't tried much for this front token."""
        front = self.get_front(tokens)
        unexplored = []
        for a in range(self.n_actions):
            if self.front_action_tries[(front, a)] < 3:
                unexplored.append(a)
        return unexplored
    
    def get_curiosity_scores(self, tokens: Set[str]) -> Dict[int, float]:
        """Score each action by curiosity value."""
        state_h = self.state_hash(tokens)
        scores = {}
        
        for a in range(self.n_actions):
            score = 0.0
            
            # Bonus for unexplored (front, action) pairs
            front = self.get_front(tokens)
            tries = self.front_action_tries[(front, a)]
            if tries == 0:
                score += 20.0  # Never tried this
            elif tries < 3:
                score += 10.0 / tries
            
            # Bonus for leading to less-visited states
            next_states = self.state_graph[state_h][a]
            if next_states:
                avg_visits = sum(self.state_visits[ns] for ns in next_states) / len(next_states)
                score += 5.0 / (avg_visits + 1)
            else:
                score += 3.0  # Unknown outcome
            
            # Bonus for leading to rare tokens
            for ns in next_states:
                for t in ns:
                    if self.token_counts.get(t, 0) < 5:
                        score += 2.0  # Rare token
            
            scores[a] = score
        
        return scores
    
    def get_success_action(self, tokens: Set[str]) -> Optional[int]:
        """Check if we have a known success action for this front."""
        front = self.get_front(tokens)
        for pat_front, pat_action in self.success_patterns:
            if pat_front == front:
                return pat_action
        return None
    
    def find_path_to_token(self, current: Set[str], target_token: str, max_depth: int = 15) -> Optional[List[int]]:
        """BFS to find path to state containing target token."""
        start_h = self.state_hash(current)
        
        queue = deque([(start_h, [])])
        visited = {start_h}
        
        while queue and len(queue[0][1]) < max_depth:
            state_h, path = queue.popleft()
            
            # Check if target in this state
            if target_token in state_h:
                return path
            
            # Explore neighbors
            for action in range(self.n_actions):
                for next_h in self.state_graph[state_h][action]:
                    if next_h not in visited:
                        visited.add(next_h)
                        queue.append((next_h, path + [action]))
        
        return None
    
    def choose_action(self, tokens: Set[str], explore_rate: float = 0.3) -> int:
        """Choose action balancing exploitation and curiosity."""
        
        # Check for known success action
        success_action = self.get_success_action(tokens)
        if success_action is not None and random.random() < 0.9:
            return success_action
        
        # Try to find path to success-correlated tokens
        if self.success_tokens and random.random() < 0.3:
            # Pick a success-correlated token to aim for
            target = max(self.success_tokens.keys(), key=lambda t: self.success_tokens[t])
            if target not in tokens:  # Not already there
                path = self.find_path_to_token(tokens, target)
                if path:
                    return path[0]
        
        # Curiosity-driven exploration
        if random.random() < explore_rate:
            # Prioritize unexplored actions
            unexplored = self.get_unexplored_actions(tokens)
            if unexplored:
                return random.choice(unexplored)
            return random.randint(0, self.n_actions - 1)
        
        # Use curiosity scores
        scores = self.get_curiosity_scores(tokens)
        best = max(scores.values())
        best_actions = [a for a, s in scores.items() if s >= best - 0.1]
        return random.choice(best_actions)
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'rules': len(self.rules),
            'states': len(self.state_visits),
            'tokens': len(self.all_tokens),
            'success_patterns': len(self.success_patterns),
            'success_tokens': len(self.success_tokens),
        }
