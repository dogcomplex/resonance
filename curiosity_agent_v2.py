"""
CURIOSITY AGENT V2 - With Backward Chaining

Improvements:
1. Backward chaining from success tokens to find prerequisites
2. Subgoal discovery - what tokens enable success actions?
3. Better state space exploration
"""
import random
from collections import defaultdict, deque
from typing import Set, Dict, List, Tuple, Optional, FrozenSet

class CuriosityAgentV2:
    """Agent with curiosity + backward chaining."""
    
    def __init__(self, n_actions=6):
        self.n_actions = n_actions
        
        # World model
        self.rules = defaultdict(lambda: defaultdict(int))  # (front, action) -> {effect: count}
        
        # State graph
        self.state_graph = defaultdict(lambda: defaultdict(list))
        self.reverse_graph = defaultdict(list)  # next_state -> [(prev_state, action), ...]
        
        # Visits
        self.state_visits = defaultdict(int)
        self.front_action_tries = defaultdict(int)
        
        # Token tracking
        self.all_tokens = set()
        self.token_counts = defaultdict(int)
        
        # Success tracking
        self.success_states = set()  # States that led to success
        self.success_fronts = defaultdict(int)  # front -> success count
        self.success_actions = defaultdict(int)  # (front, action) -> success count
        
        # Subgoals: tokens that enable success
        self.enabling_tokens = defaultdict(int)  # token -> how often it was present at success
        
        # Discovered subgoal chain
        self.subgoal_chain = []  # [(front, action), ...] leading to success
        
    def get_front(self, tokens: Set[str]) -> str:
        for t in tokens:
            if t.startswith('front='):
                return t
        return 'front=unknown'
    
    def state_hash(self, tokens: Set[str]) -> FrozenSet[str]:
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
        
        # Update graphs
        before_h = self.state_hash(before)
        after_h = self.state_hash(after)
        
        if after_h not in self.state_graph[before_h][action]:
            self.state_graph[before_h][action].append(after_h)
        
        self.reverse_graph[after_h].append((before_h, action))
        
        # Update visits
        self.state_visits[after_h] += 1
        
        # Token tracking
        for t in after:
            self.all_tokens.add(t)
            self.token_counts[t] += 1
        
        # Success tracking
        if success:
            self.success_states.add(after_h)
            self.success_fronts[front] += 1
            self.success_actions[(front, action)] += 1
            
            # Track which tokens enabled this success
            for t in before:
                self.enabling_tokens[t] += 1
            
            # Try to extend subgoal chain backward
            self._update_subgoal_chain(before_h, action)
    
    def _update_subgoal_chain(self, success_state: FrozenSet[str], success_action: int):
        """Backward chain from success to find prerequisites."""
        # Look at reverse graph to find what leads to success states
        visited = {success_state}
        queue = deque([(success_state, [(self.get_front(success_state), success_action)])])
        
        while queue and len(queue) < 100:
            state, chain = queue.popleft()
            
            # Check predecessors
            for prev_state, prev_action in self.reverse_graph[state]:
                if prev_state not in visited:
                    visited.add(prev_state)
                    prev_front = self.get_front(prev_state)
                    new_chain = [(prev_front, prev_action)] + chain
                    
                    # If this chain is longer than current, save it
                    if len(new_chain) > len(self.subgoal_chain):
                        self.subgoal_chain = new_chain
                    
                    queue.append((prev_state, new_chain))
    
    def get_best_action_for_subgoal(self, tokens: Set[str]) -> Optional[int]:
        """Check if current state matches a subgoal in the chain."""
        front = self.get_front(tokens)
        
        # Check if we have a direct success action
        for (f, a), count in self.success_actions.items():
            if f == front and count > 0:
                return a
        
        # Check subgoal chain
        for i, (chain_front, chain_action) in enumerate(self.subgoal_chain):
            if chain_front == front:
                return chain_action
        
        return None
    
    def get_enabling_token_target(self, tokens: Set[str]) -> Optional[str]:
        """Find a high-value enabling token we should try to reach."""
        # Sort enabling tokens by value
        for t, count in sorted(self.enabling_tokens.items(), key=lambda x: -x[1]):
            if t not in tokens and count > 2:
                return t
        return None
    
    def find_path_to_token(self, current: Set[str], target_token: str, max_depth: int = 15) -> Optional[List[int]]:
        """BFS to find path to state with target token."""
        start_h = self.state_hash(current)
        
        queue = deque([(start_h, [])])
        visited = {start_h}
        
        while queue and len(queue[0][1]) < max_depth:
            state_h, path = queue.popleft()
            
            if target_token in state_h:
                return path
            
            for action in range(self.n_actions):
                for next_h in self.state_graph[state_h][action]:
                    if next_h not in visited:
                        visited.add(next_h)
                        queue.append((next_h, path + [action]))
        
        return None
    
    def get_curiosity_scores(self, tokens: Set[str]) -> Dict[int, float]:
        """Score actions by curiosity value."""
        state_h = self.state_hash(tokens)
        front = self.get_front(tokens)
        scores = {}
        
        for a in range(self.n_actions):
            score = 0.0
            
            # Unexplored bonus
            tries = self.front_action_tries[(front, a)]
            if tries == 0:
                score += 15.0
            elif tries < 3:
                score += 8.0 / tries
            
            # Novelty bonus
            next_states = self.state_graph[state_h][a]
            if next_states:
                avg_visits = sum(self.state_visits[ns] for ns in next_states) / len(next_states)
                score += 5.0 / (avg_visits + 1)
            else:
                score += 3.0
            
            # Enabling token bonus
            for ns in next_states:
                for t in ns:
                    if t in self.enabling_tokens:
                        score += self.enabling_tokens[t] * 0.5
            
            scores[a] = score
        
        return scores
    
    def choose_action(self, tokens: Set[str], explore_rate: float = 0.3) -> int:
        """Choose action with subgoal awareness."""
        
        # Check for known good action
        subgoal_action = self.get_best_action_for_subgoal(tokens)
        if subgoal_action is not None and random.random() < 0.85:
            return subgoal_action
        
        # Try to reach enabling tokens
        if random.random() < 0.3:
            target = self.get_enabling_token_target(tokens)
            if target:
                path = self.find_path_to_token(tokens, target)
                if path:
                    return path[0]
        
        # Curiosity exploration
        if random.random() < explore_rate:
            front = self.get_front(tokens)
            unexplored = [a for a in range(self.n_actions) if self.front_action_tries[(front, a)] < 3]
            if unexplored:
                return random.choice(unexplored)
            return random.randint(0, self.n_actions - 1)
        
        # Use curiosity scores
        scores = self.get_curiosity_scores(tokens)
        best = max(scores.values())
        best_actions = [a for a, s in scores.items() if s >= best - 0.1]
        return random.choice(best_actions)
    
    def get_stats(self) -> Dict:
        return {
            'rules': len(self.rules),
            'states': len(self.state_visits),
            'success_states': len(self.success_states),
            'enabling_tokens': len(self.enabling_tokens),
            'subgoal_chain_length': len(self.subgoal_chain),
        }
