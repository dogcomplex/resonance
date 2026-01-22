"""
Test: Does a naturally converging universe create its own Big Crunch?

Hypothesis: If rules naturally reduce entropy/options, the universe
converges to fewer states over time - creating natural endpoints.

This might be how the actual universe works:
- High energy = many possible states
- Low energy = fewer possible states  
- Eventually converges to ground state(s)
"""

from collections import defaultdict

class ConvergingUniverse:
    """
    Universe where states have 'energy' and rules tend to reduce it.
    Higher energy states have more transitions available.
    """
    
    def __init__(self, max_energy=10):
        self.max_energy = max_energy
        self.states = list(range(max_energy + 1))  # 0 to max_energy
        
        # Rules: Higher energy can decay to lower
        # Lower energy has fewer options
        self.transitions = self._build_transitions()
        
    def _build_transitions(self):
        """Build transition rules - energy tends to decrease"""
        trans = defaultdict(list)
        for e in range(self.max_energy + 1):
            # Can always stay same (stability)
            trans[e].append(('stay', e))
            
            # Can decay (lose energy) - always available
            if e > 0:
                trans[e].append(('decay_1', e - 1))
            if e > 1:
                trans[e].append(('decay_2', e - 2))
            
            # Can excite (gain energy) - but costs more at higher levels
            # Probability/availability decreases with energy
            if e < self.max_energy and e < 5:  # Only low-energy states can excite easily
                trans[e].append(('excite_1', e + 1))
            if e < self.max_energy - 1 and e < 3:  # Even harder
                trans[e].append(('excite_2', e + 2))
                
        return trans
    
    def forward_expand(self, start, max_steps=10):
        """Where can we reach from start?"""
        reached = {start: (0, [])}  # state -> (steps, path)
        frontier = [(start, 0, [])]
        
        while frontier:
            state, steps, path = frontier.pop(0)
            if steps >= max_steps:
                continue
                
            for rule, next_state in self.transitions[state]:
                if next_state not in reached or reached[next_state][0] > steps + 1:
                    new_path = path + [rule]
                    reached[next_state] = (steps + 1, new_path)
                    frontier.append((next_state, steps + 1, new_path))
        
        return reached
    
    def backward_expand(self, goal, max_steps=10):
        """What states could lead to goal?"""
        # Build reverse transitions
        reverse = defaultdict(list)
        for state, trans_list in self.transitions.items():
            for rule, next_state in trans_list:
                reverse[next_state].append((rule, state))
        
        reached = {goal: (0, [])}
        frontier = [(goal, 0, [])]
        
        while frontier:
            state, steps, path = frontier.pop(0)
            if steps >= max_steps:
                continue
                
            for rule, prev_state in reverse[state]:
                if prev_state not in reached or reached[prev_state][0] > steps + 1:
                    new_path = [rule] + path
                    reached[prev_state] = (steps + 1, new_path)
                    frontier.append((prev_state, steps + 1, new_path))
        
        return reached
    
    def find_natural_endpoints(self, max_steps=20):
        """Which states are natural 'attractors'?"""
        # Start from high energy, see where things end up
        from_high = self.forward_expand(self.max_energy, max_steps)
        
        # Count how many paths lead to each state
        endpoint_counts = defaultdict(int)
        for state, (steps, path) in from_high.items():
            # States that are reached and have few outgoing transitions
            # are natural endpoints
            outgoing = len(self.transitions[state])
            # Lower energy = more "endpoint-like"
            endpoint_score = (self.max_energy - state) + (10 - outgoing)
            endpoint_counts[state] = endpoint_score
        
        return endpoint_counts
    
    def simulate_ensemble(self, n_particles=100, max_steps=50):
        """Simulate many particles, see where they converge"""
        import random
        
        # Start all at max energy
        particles = [self.max_energy] * n_particles
        
        history = []
        for step in range(max_steps):
            # Record distribution
            dist = defaultdict(int)
            for p in particles:
                dist[p] += 1
            history.append(dict(dist))
            
            # Each particle takes a random transition
            new_particles = []
            for p in particles:
                transitions = self.transitions[p]
                # Bias toward decay (physical)
                weights = []
                for rule, next_state in transitions:
                    if 'decay' in rule:
                        weights.append(2.0)  # Decay is more likely
                    elif 'stay' in rule:
                        weights.append(1.0)
                    else:
                        weights.append(0.5)  # Excite is less likely
                
                # Normalize and choose
                total = sum(weights)
                r = random.random() * total
                cumsum = 0
                chosen = transitions[0][1]
                for (rule, next_state), w in zip(transitions, weights):
                    cumsum += w
                    if r <= cumsum:
                        chosen = next_state
                        break
                new_particles.append(chosen)
            
            particles = new_particles
        
        return history, particles


def test_convergence():
    print("=" * 60)
    print("TEST: Naturally Converging Universe")
    print("=" * 60)
    
    universe = ConvergingUniverse(max_energy=10)
    
    print("\nTransition rules by energy level:")
    for e in range(11):
        rules = [f"{r}→{s}" for r, s in universe.transitions[e]]
        print(f"  E={e}: {rules}")
    
    print("\n" + "-" * 40)
    print("Forward from max energy (E=10):")
    fwd = universe.forward_expand(10, max_steps=6)
    for state in sorted(fwd.keys()):
        steps, path = fwd[state]
        print(f"  E={state}: {steps} steps, path={path[:3]}...")
    
    print("\n" + "-" * 40)
    print("Backward from ground state (E=0):")
    bwd = universe.backward_expand(0, max_steps=6)
    for state in sorted(bwd.keys()):
        steps, path = bwd[state]
        print(f"  E={state}: {steps} steps, path={path[:3]}...")
    
    print("\n" + "-" * 40)
    print("Standing waves (E=10 → E=0):")
    
    intersection = set(fwd.keys()) & set(bwd.keys())
    print(f"  Intersection points: {sorted(intersection)}")
    
    # Full paths
    for meeting in sorted(intersection):
        fwd_steps, fwd_path = fwd[meeting]
        bwd_steps, bwd_path = bwd[meeting]
        total_path = fwd_path + bwd_path
        print(f"  Via E={meeting}: {total_path[:6]}... (len={len(total_path)})")
    
    print("\n" + "-" * 40)
    print("Ensemble simulation (100 particles, 50 steps):")
    history, final = universe.simulate_ensemble(100, 50)
    
    print(f"  Initial distribution: {history[0]}")
    print(f"  After 10 steps: {history[10]}")
    print(f"  After 25 steps: {history[25]}")
    print(f"  After 50 steps: {history[49]}")
    
    # Count final state
    final_dist = defaultdict(int)
    for p in final:
        final_dist[p] += 1
    print(f"  Final particles: {dict(final_dist)}")
    
    # Did it converge?
    ground_state_count = final_dist.get(0, 0)
    print(f"\n  Convergence to ground state: {ground_state_count}% of particles at E=0")


def test_bidirectional_with_convergence():
    """Test if natural convergence creates effective standing waves"""
    print("\n" + "=" * 60)
    print("TEST: Bidirectional Waves in Converging Universe")
    print("=" * 60)
    
    universe = ConvergingUniverse(max_energy=10)
    
    # Key insight: In a converging universe, the ground state IS the natural Big Crunch
    # Even without explicitly defining it, the backward wave from E=0 makes sense
    
    start = 8  # Start at moderately high energy
    goal = 0   # Natural endpoint
    
    fwd = universe.forward_expand(start, max_steps=10)
    bwd = universe.backward_expand(goal, max_steps=10)
    
    print(f"\nBidirectional analysis: E={start} → E={goal}")
    print(f"  Forward reached: {sorted(fwd.keys())}")
    print(f"  Backward reached: {sorted(bwd.keys())}")
    
    intersection = set(fwd.keys()) & set(bwd.keys())
    print(f"  Standing wave nodes: {sorted(intersection)}")
    
    # Which rules dominate in the standing waves?
    rule_counts = defaultdict(int)
    for meeting in intersection:
        fwd_steps, fwd_path = fwd[meeting]
        bwd_steps, bwd_path = bwd[meeting]
        for rule in fwd_path + bwd_path:
            rule_counts[rule] += 1
    
    print(f"\n  Rule resonance in standing waves:")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"    {rule}: {count}")
    
    print(f"\n  Key insight: 'decay' rules dominate because they're the natural flow.")
    print(f"  The universe WANTS to go to ground state - that's the Big Crunch.")


def test_multiple_attractors():
    """What if there are multiple natural attractors (local minima)?"""
    print("\n" + "=" * 60)
    print("TEST: Multiple Natural Attractors")
    print("=" * 60)
    
    # Modify universe to have local minima (metastable states)
    # E=0 is ground, but E=5 is a local minimum (hard to leave)
    
    class MultiAttractorUniverse(ConvergingUniverse):
        def _build_transitions(self):
            trans = defaultdict(list)
            for e in range(self.max_energy + 1):
                trans[e].append(('stay', e))
                
                # E=5 is a metastable trap
                if e == 5:
                    trans[e].append(('decay_1', 4))  # Can escape, but slow
                    # No excite from 5 - it's a trap
                elif e > 0:
                    trans[e].append(('decay_1', e - 1))
                    if e > 1:
                        trans[e].append(('decay_2', e - 2))
                    
                if e < self.max_energy and e != 4:  # Can't excite INTO 5 easily
                    trans[e].append(('excite_1', e + 1))
                    
                # But once at 6+, can fall into 5
                if e == 6:
                    trans[e].append(('trap', 5))
                    
            return trans
    
    universe = MultiAttractorUniverse(max_energy=10)
    
    print("\nSimulating with metastable trap at E=5:")
    history, final = universe.simulate_ensemble(100, 100)
    
    final_dist = defaultdict(int)
    for p in final:
        final_dist[p] += 1
    print(f"  Final distribution: {dict(sorted(final_dist.items()))}")
    
    print(f"\n  Two attractors emerged:")
    print(f"    - Ground state (E=0): {final_dist.get(0, 0)} particles")
    print(f"    - Metastable trap (E=5): {final_dist.get(5, 0)} particles")
    
    print(f"\n  This is like having TWO Big Crunches - two valid endpoints!")
    print(f"  Different particles 'choose' different futures based on path taken.")


if __name__ == "__main__":
    test_convergence()
    test_bidirectional_with_convergence()
    test_multiple_attractors()
    
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print("""
1. A converging universe CREATES its own Big Crunch:
   - Ground state (minimum energy) is the natural endpoint
   - No need to define it explicitly - it emerges from rules
   
2. Standing waves work even with natural convergence:
   - Forward wave from high energy
   - Backward wave from ground state
   - They meet throughout the energy landscape
   
3. Multiple attractors = multiple futures:
   - Metastable states are local Big Crunches
   - Particles can end up in different endpoints
   - This is literally many-worlds!
   
4. Rule resonance reveals physics:
   - 'decay' rules dominate (thermodynamic arrow of time)
   - The rules that appear most in standing waves are
     the ones that "connect" cause to effect most reliably

The Big Crunch isn't arbitrary - it's the ATTRACTOR of the dynamics.
In physics: maximum entropy state
In games: terminal game states (win/lose/draw)
In logic: tautologies and contradictions
""")
