"""
Bidirectional Wave Sieve - Testing Deduction/Abduction/Induction

The hypothesis:
- Deduction (forward wave): Given start, what's reachable?
- Abduction (backward wave): Given goal, what could lead there?
- Induction (standing wave): Where they meet = discovered rules

Questions:
1. What defines the "Big Crunch" (end boundary) in arbitrary universes?
2. Can we see rule emergence from wave interference?
3. Do we get distinct "force-like" patterns?
"""

from collections import defaultdict
from typing import Dict, Set, List, Tuple, Callable, Any

class BidirectionalWaveSieve:
    def __init__(self, state_space: range, ops: Dict[str, Callable], 
                 inverse_ops: Dict[str, Callable]):
        self.state_space = state_space
        self.ops = ops
        self.inverse_ops = inverse_ops
        self.rule_resonance = defaultdict(int)  # How often each rule appears
        
    def is_valid(self, state) -> bool:
        return state in self.state_space
    
    def forward_expand(self, start: int, max_depth: int = 5) -> Dict[int, List[str]]:
        """Deduction: What's reachable from start?"""
        reached = {start: []}
        frontier = [(start, [])]
        
        for depth in range(max_depth):
            new_frontier = []
            for state, path in frontier:
                for op_name, op_fn in self.ops.items():
                    try:
                        new_state = op_fn(state)
                        if self.is_valid(new_state) and new_state not in reached:
                            new_path = path + [op_name]
                            reached[new_state] = new_path
                            new_frontier.append((new_state, new_path))
                    except:
                        pass
            frontier = new_frontier
        return reached
    
    def backward_expand(self, goal: int, max_depth: int = 5) -> Dict[int, List[str]]:
        """Abduction: What could lead to goal?"""
        sources = {goal: []}
        frontier = [(goal, [])]
        
        for depth in range(max_depth):
            new_frontier = []
            for state, path in frontier:
                for op_name, inv_fn in self.inverse_ops.items():
                    try:
                        prev_state = inv_fn(state)
                        if prev_state is not None and self.is_valid(prev_state) and prev_state not in sources:
                            new_path = [op_name] + path
                            sources[prev_state] = new_path
                            new_frontier.append((prev_state, new_path))
                    except:
                        pass
            frontier = new_frontier
        return sources
    
    def find_standing_waves(self, start: int, goal: int, 
                           max_depth: int = 5) -> Tuple[List[dict], Dict, Dict]:
        """Induction: Where do forward and backward waves meet?"""
        forward = self.forward_expand(start, max_depth)
        backward = self.backward_expand(goal, max_depth)
        
        # Interference: states reached by BOTH waves
        intersection = set(forward.keys()) & set(backward.keys())
        
        # Standing waves: complete paths through intersection points
        standing_waves = []
        for meeting_point in intersection:
            forward_path = forward[meeting_point]
            backward_path = backward[meeting_point]
            full_path = forward_path + backward_path
            standing_waves.append({
                'meeting_point': meeting_point,
                'forward_path': forward_path,
                'backward_path': backward_path,
                'full_path': full_path,
                'length': len(full_path)
            })
        
        return standing_waves, forward, backward
    
    def observe_transition(self, start: int, goal: int, max_depth: int = 5):
        """Observe a (start, goal) pair and accumulate rule resonance"""
        waves, fwd, bwd = self.find_standing_waves(start, goal, max_depth)
        
        # Count which operations appear in successful paths
        for wave in waves:
            for op in wave['full_path']:
                self.rule_resonance[op] += 1
                
        return waves
    
    def get_induced_rules(self, threshold: int = 1) -> Dict[str, int]:
        """Return rules with sufficient resonance (induction result)"""
        return {r: c for r, c in self.rule_resonance.items() if c >= threshold}


def test_number_chain():
    """Test 1: Simple number chain universe"""
    print("=" * 60)
    print("TEST 1: Number Chain Universe (1-20)")
    print("=" * 60)
    
    ops = {
        '+1': lambda x: x + 1,
        '+2': lambda x: x + 2,
        '*2': lambda x: x * 2,
        '-1': lambda x: x - 1,
    }
    
    inverse_ops = {
        '+1': lambda x: x - 1,
        '+2': lambda x: x - 2,
        '*2': lambda x: x // 2 if x % 2 == 0 else None,
        '-1': lambda x: x + 1,
    }
    
    sieve = BidirectionalWaveSieve(range(1, 21), ops, inverse_ops)
    
    # Single observation
    start, goal = 1, 10
    waves, fwd, bwd = sieve.find_standing_waves(start, goal, max_depth=5)
    
    print(f"\nSingle observation: {start} → {goal}")
    print(f"  Forward wave reached {len(fwd)} states")
    print(f"  Backward wave reached {len(bwd)} states")
    print(f"  Intersection (standing waves): {len(waves)} paths")
    
    print(f"\nShortest paths through interference:")
    for w in sorted(waves, key=lambda x: x['length'])[:5]:
        print(f"  {start} →{w['forward_path']}→ {w['meeting_point']} →{w['backward_path']}→ {goal}")
        print(f"    Full path: {w['full_path']}")
    
    # Multiple observations - what rules generalize?
    print("\n" + "-" * 40)
    print("Multiple observations - rule induction:")
    
    sieve2 = BidirectionalWaveSieve(range(1, 21), ops, inverse_ops)
    test_pairs = [
        (1, 10), (2, 15), (3, 12), (1, 8), (5, 20),
        (2, 8), (4, 16), (1, 16), (3, 9), (7, 14)
    ]
    
    for start, goal in test_pairs:
        waves = sieve2.observe_transition(start, goal, max_depth=4)
        print(f"  {start} → {goal}: {len(waves)} standing waves")
    
    print(f"\nInduced rules (by resonance):")
    for rule, count in sorted(sieve2.rule_resonance.items(), key=lambda x: -x[1]):
        print(f"  {rule}: {count} appearances")
    
    return sieve2


def test_with_end_state():
    """Test 2: Universe with explicit END state (Big Crunch)"""
    print("\n" + "=" * 60)
    print("TEST 2: Universe with END state (Big Crunch)")
    print("=" * 60)
    
    # State space: 1-19 are normal, 20 is END (absorbing)
    # Once you reach 20, nothing escapes
    
    ops = {
        '+1': lambda x: min(x + 1, 20),  # Caps at 20
        '+2': lambda x: min(x + 2, 20),
        '*2': lambda x: min(x * 2, 20),
        'END': lambda x: 20 if x >= 15 else None,  # Can jump to END from high states
    }
    
    inverse_ops = {
        '+1': lambda x: x - 1 if x > 1 else None,
        '+2': lambda x: x - 2 if x > 2 else None,
        '*2': lambda x: x // 2 if x % 2 == 0 and x > 1 else None,
        'END': lambda x: None,  # Can't reverse out of END
    }
    
    # Actually, let's allow backward from END to represent "what could have caused end"
    inverse_ops['END'] = lambda x: 15 if x == 20 else (16 if x == 20 else None)
    
    # Cleaner: END can be reached from 15-19
    def inverse_end(x):
        if x == 20:
            return 15  # One possible predecessor
        return None
    
    inverse_ops['END'] = inverse_end
    
    sieve = BidirectionalWaveSieve(range(1, 21), ops, inverse_ops)
    
    # Forward from 1
    fwd = sieve.forward_expand(1, max_depth=6)
    print(f"\nForward from 1 reaches: {sorted(fwd.keys())}")
    print(f"  Path to END (20): {fwd.get(20, 'not reached')}")
    
    # Backward from END (20)
    bwd = sieve.backward_expand(20, max_depth=6)
    print(f"\nBackward from END (20) traces to: {sorted(bwd.keys())}")
    
    # Standing waves from 1 to END
    waves, _, _ = sieve.find_standing_waves(1, 20, max_depth=6)
    print(f"\nStanding waves (1 → END): {len(waves)} paths")
    for w in sorted(waves, key=lambda x: x['length'])[:3]:
        print(f"  Path: {w['full_path']} (via {w['meeting_point']})")


def test_game_lattice():
    """Test 3: Game-like lattice with win/lose states"""
    print("\n" + "=" * 60)
    print("TEST 3: Game Lattice (Tic-Tac-Toe-like)")
    print("=" * 60)
    
    # Simplified: States are (my_score, opponent_score)
    # Win = my_score reaches 3, Lose = opponent reaches 3
    # This creates a lattice converging on terminal states
    
    # Encode states as single int: my*10 + opp
    def encode(my, opp): return my * 10 + opp
    def decode(s): return s // 10, s % 10
    
    valid_states = set()
    for my in range(4):
        for opp in range(4):
            if my < 3 and opp < 3:  # Neither has won yet
                valid_states.add(encode(my, opp))
    # Add terminal states
    valid_states.add(encode(3, 0))  # Win states
    valid_states.add(encode(3, 1))
    valid_states.add(encode(3, 2))
    valid_states.add(encode(0, 3))  # Lose states
    valid_states.add(encode(1, 3))
    valid_states.add(encode(2, 3))
    
    ops = {
        'I_score': lambda s: encode(decode(s)[0] + 1, decode(s)[1]) if decode(s)[0] < 3 else None,
        'Opp_scores': lambda s: encode(decode(s)[0], decode(s)[1] + 1) if decode(s)[1] < 3 else None,
    }
    
    inverse_ops = {
        'I_score': lambda s: encode(decode(s)[0] - 1, decode(s)[1]) if decode(s)[0] > 0 else None,
        'Opp_scores': lambda s: encode(decode(s)[0], decode(s)[1] - 1) if decode(s)[1] > 0 else None,
    }
    
    sieve = BidirectionalWaveSieve(valid_states, ops, inverse_ops)
    
    start = encode(0, 0)  # Game start
    win = encode(3, 0)    # Clean win
    
    print(f"\nForward from start (0,0):")
    fwd = sieve.forward_expand(start, max_depth=6)
    for s in sorted(fwd.keys()):
        my, opp = decode(s)
        if my == 3 or opp == 3:
            print(f"  Terminal ({my},{opp}): {fwd[s]}")
    
    print(f"\nBackward from win (3,0):")
    bwd = sieve.backward_expand(win, max_depth=6)
    for s in sorted(bwd.keys()):
        my, opp = decode(s)
        print(f"  ({my},{opp}): {bwd[s]}")
    
    print(f"\nStanding waves (start → win):")
    waves, _, _ = sieve.find_standing_waves(start, win, max_depth=6)
    for w in waves:
        print(f"  {w['full_path']} (meeting at {decode(w['meeting_point'])})")


def test_multiple_endpoints():
    """Test 4: What if there are multiple 'Big Crunch' endpoints?"""
    print("\n" + "=" * 60)
    print("TEST 4: Multiple Endpoints (Many Big Crunches)")
    print("=" * 60)
    
    # Universe where multiple absorbing states exist
    # Like chemistry: different stable products
    
    ops = {
        '+1': lambda x: x + 1,
        '*2': lambda x: x * 2,
        'stabilize_10': lambda x: 100 if x == 10 else None,  # 10 → stable_A
        'stabilize_12': lambda x: 120 if x == 12 else None,  # 12 → stable_B
        'stabilize_16': lambda x: 160 if x == 16 else None,  # 16 → stable_C
    }
    
    inverse_ops = {
        '+1': lambda x: x - 1 if x > 1 and x < 100 else None,
        '*2': lambda x: x // 2 if x % 2 == 0 and x > 1 and x < 100 else None,
        'stabilize_10': lambda x: 10 if x == 100 else None,
        'stabilize_12': lambda x: 12 if x == 120 else None,
        'stabilize_16': lambda x: 16 if x == 160 else None,
    }
    
    valid = set(range(1, 21)) | {100, 120, 160}  # Include stable endpoints
    sieve = BidirectionalWaveSieve(valid, ops, inverse_ops)
    
    start = 1
    endpoints = [100, 120, 160]
    
    print(f"\nFrom start={start}, finding paths to each endpoint:")
    
    for end in endpoints:
        waves, fwd, bwd = sieve.find_standing_waves(start, end, max_depth=8)
        print(f"\n  → Endpoint {end}:")
        print(f"    Forward reached: {len([k for k in fwd.keys() if k < 100])} normal states")
        print(f"    Standing waves: {len(waves)}")
        if waves:
            shortest = min(waves, key=lambda x: x['length'])
            print(f"    Shortest path: {shortest['full_path']}")
    
    # The "true" Big Crunch might be whichever endpoint is most reachable?
    # Or: all endpoints are equally valid "futures" - many worlds!


def test_interference_patterns():
    """Test 5: Visualize the interference pattern directly"""
    print("\n" + "=" * 60)
    print("TEST 5: Visualizing Wave Interference")
    print("=" * 60)
    
    ops = {
        '+1': lambda x: x + 1,
        '+2': lambda x: x + 2,
        '*2': lambda x: x * 2,
    }
    
    inverse_ops = {
        '+1': lambda x: x - 1 if x > 1 else None,
        '+2': lambda x: x - 2 if x > 2 else None,
        '*2': lambda x: x // 2 if x % 2 == 0 else None,
    }
    
    sieve = BidirectionalWaveSieve(range(1, 33), ops, inverse_ops)
    
    start, goal = 1, 32
    fwd = sieve.forward_expand(start, max_depth=6)
    bwd = sieve.backward_expand(goal, max_depth=6)
    
    print(f"\nState-by-state interference ({start} → {goal}):")
    print(f"{'State':>5} | {'Fwd Depth':>10} | {'Bwd Depth':>10} | {'Interference':>12}")
    print("-" * 50)
    
    for state in range(1, 33):
        fwd_depth = len(fwd.get(state, [])) if state in fwd else -1
        bwd_depth = len(bwd.get(state, [])) if state in bwd else -1
        
        if fwd_depth >= 0 and bwd_depth >= 0:
            interference = "★ STANDING"
        elif fwd_depth >= 0:
            interference = "→ forward"
        elif bwd_depth >= 0:
            interference = "← backward"
        else:
            interference = "  (none)"
        
        if fwd_depth >= 0 or bwd_depth >= 0:
            print(f"{state:>5} | {fwd_depth:>10} | {bwd_depth:>10} | {interference:>12}")
    
    # Show where waves meet
    waves, _, _ = sieve.find_standing_waves(start, goal, max_depth=6)
    meeting_points = set(w['meeting_point'] for w in waves)
    print(f"\nMeeting points (standing wave nodes): {sorted(meeting_points)}")
    
    # What rules dominate at meeting points?
    print(f"\nRules used in standing waves:")
    rule_at_meeting = defaultdict(int)
    for w in waves:
        for i, op in enumerate(w['full_path']):
            rule_at_meeting[op] += 1
    
    for rule, count in sorted(rule_at_meeting.items(), key=lambda x: -x[1]):
        print(f"  {rule}: {count}")


if __name__ == "__main__":
    test_number_chain()
    test_with_end_state()
    test_game_lattice()
    test_multiple_endpoints()
    test_interference_patterns()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: What defines the Big Crunch?")
    print("=" * 60)
    print("""
In our tests:
1. Number chain: No natural endpoint - universe expands forever
2. END state: Explicit absorbing state = Big Crunch
3. Game lattice: Multiple endpoints (win/lose) = branching futures
4. Multiple endpoints: Many valid "crunches" = many worlds?

The Big Crunch might be:
- An absorbing state (can reach but not leave)
- A convergence point (all paths lead here eventually)
- Simply "where backward wave originates" (goal-directed)

For arbitrary games: The END_GAME token IS the Big Crunch.
It's the boundary condition that makes abduction possible.
Without an endpoint, backward waves have nowhere to start.
""")
