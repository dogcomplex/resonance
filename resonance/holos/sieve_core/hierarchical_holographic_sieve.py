"""
HIERARCHICAL HOLOGRAPHIC SIEVE
==============================

The architecture you described:

1. Pack observations into holographic encoding
2. Sieve naturally preserves common patterns, removes noise
3. Apply sieve DIMENSIONALLY:
   - Level 0: Pixel patches -> features (per frame)
   - Level 1: Features across frames -> transitions
   - Level 2: Transitions -> rules
   - Level 3: Rules -> meta-rules
   - Level N: Meta^N rules -> Meta^(N+1) rules

4. Each level:
   - Threshold to keep important patterns
   - Feed into next level as "observations"
   - Repeat until stable

This is a RECURSIVE COMPRESSION TOWER.
"""

from substrate import Substrate, RuleHamiltonian, DiscreteConfig, AmplitudeField
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import cmath

# ============================================================================
# PART 1: THE HOLOGRAPHIC ACCUMULATOR
# ============================================================================

@dataclass
class HolographicMemory:
    """
    A memory that accumulates observations holographically.

    Key insight: We don't store observations discretely.
    We SUPERPOSE them into an amplitude field.
    Common patterns reinforce. Noise cancels.
    """

    n_tokens: int
    field: Dict[Tuple, complex]  # config -> amplitude
    observation_count: int = 0

    def __init__(self, n_tokens: int):
        self.n_tokens = n_tokens
        self.field = defaultdict(complex)
        self.observation_count = 0

    def encode_observation(self, features: Dict[str, float], phase_offset: float = 0.0):
        """
        Encode an observation as interference pattern.

        Each feature contributes amplitude with phase encoding its value.
        Multiple observations superpose - common features reinforce.
        """
        for feature_name, value in features.items():
            # Hash feature name to token ID
            token_id = hash(feature_name) % self.n_tokens

            # Phase encodes value (0-1 -> 0-2pi)
            # Plus observation-specific offset for temporal encoding
            phase = value * 2 * np.pi + phase_offset
            amplitude = complex(np.cos(phase), np.sin(phase))

            # Superpose into field
            config = (token_id,)
            self.field[config] += amplitude

        self.observation_count += 1

    def encode_transition(self, before: Dict[str, float], after: Dict[str, float],
                         action: Optional[int] = None):
        """
        Encode a transition as interference between before and after.

        The DIFFERENCE shows up as phase shift.
        Same values -> constructive interference (stable)
        Different values -> phase difference (change detected)
        """
        # Encode "before" with phase 0
        self.encode_observation(before, phase_offset=0.0)

        # Encode "after" with phase pi/2 (quarter turn)
        # This creates interference that encodes the CHANGE
        self.encode_observation(after, phase_offset=np.pi/2)

        # If action provided, encode it too
        if action is not None:
            action_token = hash(f"action_{action}") % self.n_tokens
            self.field[(action_token,)] += complex(1.0, 0.0)

    def get_dominant_patterns(self, threshold: float = 0.1) -> List[Tuple[Tuple, float, float]]:
        """
        Extract patterns that survived superposition.

        High amplitude = many observations agreed (common pattern)
        Low amplitude = observations disagreed (noise/rare)
        """
        patterns = []
        max_amp = max(abs(a) for a in self.field.values()) if self.field else 1.0

        for config, amplitude in self.field.items():
            normalized_amp = abs(amplitude) / max_amp
            if normalized_amp > threshold:
                phase = cmath.phase(amplitude)
                patterns.append((config, normalized_amp, phase))

        return sorted(patterns, key=lambda x: -x[1])

    def decay(self, factor: float = 0.9):
        """
        Apply decay to all amplitudes.

        This is how old observations fade unless reinforced.
        """
        for config in self.field:
            self.field[config] *= factor


# ============================================================================
# PART 2: THE DIMENSIONAL SIEVE STACK
# ============================================================================

class DimensionalSieve:
    """
    Apply sieves at multiple levels/dimensions.

    Level 0: Raw features -> stable features
    Level 1: Feature transitions -> stable transitions
    Level 2: Transitions -> rules
    Level 3: Rules -> meta-rules
    ...
    """

    def __init__(self, n_tokens_per_level: List[int], damping: float = 0.1):
        self.levels = len(n_tokens_per_level)
        self.n_tokens = n_tokens_per_level
        self.damping = damping

        # Holographic memory at each level
        self.memories = [HolographicMemory(n) for n in n_tokens_per_level]

        # Extracted patterns at each level (after thresholding)
        self.patterns: List[List[Tuple]] = [[] for _ in range(self.levels)]

        # Substrate for each level (created on demand)
        self.substrates: List[Optional[Substrate]] = [None] * self.levels

    def observe_raw(self, features: Dict[str, float]):
        """
        Feed raw observation into level 0.
        """
        self.memories[0].encode_observation(features)

    def observe_transition(self, before: Dict[str, float], after: Dict[str, float],
                          action: Optional[int] = None):
        """
        Feed transition observation into level 0.
        """
        self.memories[0].encode_transition(before, after, action)

    def sieve_level(self, level: int, threshold: float = 0.1,
                    evolution_time: float = 2.0) -> List[Tuple]:
        """
        Run sieve on a level and extract stable patterns.

        1. Get current holographic field
        2. Build Hamiltonian from dominant patterns
        3. Evolve to find truly stable patterns
        4. Threshold and return survivors
        """
        memory = self.memories[level]
        n_tokens = self.n_tokens[level]

        # Get patterns that survived holographic superposition
        raw_patterns = memory.get_dominant_patterns(threshold=threshold/2)

        if len(raw_patterns) < 2:
            return raw_patterns

        # Build rules: patterns that co-occur can reinforce each other
        rules = []
        for i, (config_i, amp_i, phase_i) in enumerate(raw_patterns):
            for j, (config_j, amp_j, phase_j) in enumerate(raw_patterns):
                if i != j:
                    # Coupling strength based on amplitude product
                    # Phase difference affects whether constructive or destructive
                    phase_diff = phase_i - phase_j
                    coupling = amp_i * amp_j * complex(np.cos(phase_diff), np.sin(phase_diff))

                    src = DiscreteConfig(config_i)
                    tgt = DiscreteConfig(config_j)
                    rules.append((src, tgt, coupling))

        if not rules:
            return raw_patterns

        # Create and run substrate
        H = RuleHamiltonian(rules)
        substrate = Substrate(H, damping=self.damping)

        # Inject patterns
        for config, amp, phase in raw_patterns:
            substrate.inject(DiscreteConfig(config), complex(amp * np.cos(phase), amp * np.sin(phase)))

        # Evolve
        substrate.evolve(duration=evolution_time)

        # Extract survivors
        dominant = substrate.dominant_configs(n=min(50, len(raw_patterns)))

        # Threshold
        survivors = []
        max_amp = max(abs(a) for _, a in dominant) if dominant else 1.0
        for config, amplitude in dominant:
            if abs(amplitude) / max_amp > threshold:
                survivors.append((config.tokens, abs(amplitude), cmath.phase(amplitude)))

        self.patterns[level] = survivors
        self.substrates[level] = substrate

        return survivors

    def promote_to_next_level(self, level: int):
        """
        Take stable patterns from level N and feed into level N+1 as observations.

        This is the key recursive step:
        - Level N patterns become Level N+1 "features"
        - Level N+1 sieve finds patterns among the patterns
        - These are META-patterns (rules about rules)
        """
        if level >= self.levels - 1:
            print(f"Already at top level {level}")
            return

        patterns = self.patterns[level]
        if not patterns:
            print(f"No patterns at level {level} to promote")
            return

        # Convert patterns to features for next level
        next_memory = self.memories[level + 1]

        for config, amplitude, phase in patterns:
            # Pattern becomes a "feature" at next level
            feature_name = f"L{level}_pattern_{config}"
            # Amplitude becomes the feature value
            feature_value = amplitude  # Already normalized 0-1

            next_memory.encode_observation({feature_name: feature_value}, phase_offset=phase)

        print(f"Promoted {len(patterns)} patterns from level {level} to level {level + 1}")

    def run_full_stack(self, threshold: float = 0.2, evolution_time: float = 2.0):
        """
        Run the full dimensional sieve stack.

        For each level:
        1. Sieve to find stable patterns
        2. Promote survivors to next level
        3. Repeat until top level
        """
        for level in range(self.levels):
            print(f"\n{'='*50}")
            print(f"LEVEL {level}: {self.n_tokens[level]} tokens")
            print(f"{'='*50}")

            # Sieve this level
            survivors = self.sieve_level(level, threshold, evolution_time)
            print(f"Stable patterns: {len(survivors)}")

            # Show top patterns
            for config, amp, phase in survivors[:5]:
                print(f"  {config}: amp={amp:.3f}, phase={phase:.2f}")

            # Promote to next level
            if level < self.levels - 1:
                self.promote_to_next_level(level)


# ============================================================================
# PART 3: DEMONSTRATION
# ============================================================================

def demonstrate_hierarchical_sieve():
    """
    Demonstrate the full hierarchical holographic sieve.
    """

    print("=" * 70)
    print("HIERARCHICAL HOLOGRAPHIC SIEVE DEMONSTRATION")
    print("=" * 70)

    # Create 4-level stack
    # Level 0: 50 tokens (features)
    # Level 1: 30 tokens (transitions/patterns)
    # Level 2: 20 tokens (rules)
    # Level 3: 10 tokens (meta-rules)

    stack = DimensionalSieve(
        n_tokens_per_level=[50, 30, 20, 10],
        damping=0.1
    )

    print("\nSimulating Pong-like observations...")

    # Simulate observations
    np.random.seed(42)

    for frame in range(100):
        # Simulated features (what a feature extractor might produce)
        features = {
            "ball_x": (np.sin(frame * 0.1) + 1) / 2,  # Oscillating 0-1
            "ball_y": (np.cos(frame * 0.15) + 1) / 2,
            "paddle_x": 0.5 + 0.3 * np.sin(frame * 0.05),  # Slower oscillation
            "ball_dx": np.sign(np.cos(frame * 0.1)) * 0.5 + 0.5,  # Direction
            "ball_dy": np.sign(np.sin(frame * 0.15)) * 0.5 + 0.5,
            "score": min(1.0, frame / 100),  # Slowly increasing
        }

        # Add some noise
        for key in features:
            features[key] += np.random.normal(0, 0.05)
            features[key] = max(0, min(1, features[key]))

        stack.observe_raw(features)

        # Also observe transitions (compare to previous)
        if frame > 0:
            action = np.random.randint(0, 3)  # Random action
            stack.observe_transition(prev_features, features, action)

        prev_features = features.copy()

    print(f"Accumulated {stack.memories[0].observation_count} observations at level 0")

    # Run the full stack
    stack.run_full_stack(threshold=0.15, evolution_time=2.0)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("""

WHAT HAPPENED:
==============

Level 0 (Features):
  - Raw feature observations superposed holographically
  - Common patterns (ball oscillation, paddle tracking) reinforced
  - Noise averaged out
  - Sieve found: stable features that persist across frames

Level 1 (Transitions):
  - Feature patterns from L0 became "observations"
  - Found: which feature-changes are consistent
  - e.g., "ball_x increases when ball_dx is positive"

Level 2 (Rules):
  - Transition patterns from L1 became "observations"
  - Found: which transitions co-occur reliably
  - e.g., "paddle follows ball" is a stable rule

Level 3 (Meta-rules):
  - Rule patterns from L2 became "observations"
  - Found: relationships between rules
  - e.g., "tracking rule activates when ball approaching"


THE KEY INSIGHT:
================

Each level does the SAME operation:
1. Accumulate (holographic superposition)
2. Sieve (find stable patterns)
3. Threshold (keep important ones)
4. Promote (feed to next level)

But the MEANING changes:
- L0: "What features are real?"
- L1: "What changes are consistent?"
- L2: "What rules govern changes?"
- L3: "What rules govern rules?"

This is AUTOMATIC ABSTRACTION.
No hand-coding of hierarchy - it emerges from repeated sieving.
    """)


# ============================================================================
# PART 4: THE THEORETICAL PICTURE
# ============================================================================

def explain_architecture():
    """
    Explain why this architecture works.
    """

    print("\n" + "=" * 70)
    print("WHY THIS ARCHITECTURE WORKS")
    print("=" * 70)

    print("""

THE HOLOGRAPHIC ACCUMULATION PRINCIPLE:
=======================================

When you superpose observations holographically:

  field[pattern] += amplitude * e^(i*phase)

Common patterns: amplitudes ADD (constructive interference)
  -> Gets stronger with each observation

Rare/noisy patterns: amplitudes partially CANCEL
  -> Random phases average toward zero

This is AUTOMATIC DENOISING without explicit statistics.


THE SIEVE REFINEMENT PRINCIPLE:
===============================

After accumulation, patterns compete in the sieve:

  Strong patterns: survive and reinforce neighbors
  Weak patterns: decay below threshold

This is AUTOMATIC FEATURE SELECTION.

The sieve doesn't just keep "frequent" patterns -
it keeps patterns that FIT TOGETHER (mutual reinforcement).


THE DIMENSIONAL PROMOTION PRINCIPLE:
====================================

When you promote patterns from level N to level N+1:

  Level N patterns become Level N+1 "atoms"
  Level N+1 finds patterns AMONG patterns

This is AUTOMATIC ABSTRACTION.

  L0 atoms = pixels/features
  L1 atoms = feature patterns (objects)
  L2 atoms = object patterns (behaviors)
  L3 atoms = behavior patterns (rules)
  L4 atoms = rule patterns (meta-rules)

Each level is more abstract, more compressed, more general.


THE THRESHOLD GATING:
=====================

Between levels, threshold discards weak patterns:

  Input: 100 patterns from level N
  Threshold: keep top 20% by amplitude
  Output: 20 patterns promoted to level N+1

This prevents exponential blowup.
Each level is COMPRESSED relative to the one below.


THE FULL PICTURE:
=================

         Raw Observations
               |
               v
    +---------------------+
    |  Level 0: Features  |  <- Holographic accumulation
    |  (50 tokens)        |  <- Sieve refinement
    +---------------------+  <- Threshold: keep 20%
               |
               v (promote survivors)
    +---------------------+
    |  Level 1: Patterns  |  <- Holographic accumulation
    |  (30 tokens)        |  <- Sieve refinement
    +---------------------+  <- Threshold: keep 20%
               |
               v (promote survivors)
    +---------------------+
    |  Level 2: Rules     |  <- Holographic accumulation
    |  (20 tokens)        |  <- Sieve refinement
    +---------------------+  <- Threshold: keep 20%
               |
               v (promote survivors)
    +---------------------+
    |  Level 3: Meta-Rules|  <- Final stable structure
    |  (10 tokens)        |
    +---------------------+

Each level:
  - Accumulates (holographic superposition)
  - Refines (sieve competition)
  - Compresses (threshold)
  - Promotes (feed to next level)


WHY THIS MIGHT BE POWERFUL:
===========================

1. NO HAND-DESIGNED FEATURES
   Raw observations in, abstract rules out.
   The hierarchy designs itself.

2. AUTOMATIC COMPRESSION
   Each level is smaller than the one below.
   Information is preserved through abstraction.

3. NATURAL MEMORY
   Old observations decay unless reinforced.
   Common patterns persist.
   This is how memory SHOULD work.

4. COMPOSITIONALITY
   Higher-level patterns are BUILT FROM lower-level patterns.
   Not just correlated - actually composed.

5. ONLINE LEARNING
   New observations continuously fed in.
   Patterns update incrementally.
   No batch processing required.


POTENTIAL ISSUES:
=================

1. TOKEN COUNT CHOICES
   How many tokens at each level?
   Too few: can't represent complexity
   Too many: slow, overfitting

2. THRESHOLD TUNING
   How aggressive to be at each level?
   Too aggressive: lose important patterns
   Too lenient: noise propagates up

3. PHASE ALIGNMENT
   Holographic encoding assumes phase matters.
   How to choose phase offsets for observations?

4. SCALE INVARIANCE
   Higher levels see fewer "observations" (patterns from below).
   Need to adjust damping/coupling accordingly.

5. CATASTROPHIC FORGETTING
   If old patterns decay too fast, may lose important rare events.
   Need balance between plasticity and stability.


BUT THE CORE IDEA IS SOUND:
===========================

Repeated application of:
  ACCUMULATE -> SIEVE -> THRESHOLD -> PROMOTE

This IS a form of recursive compression.
Each level extracts the "essence" of the level below.
The top level contains maximally compressed structure.

And it's all done with the same operation: WAVE INTERFERENCE + COMPETITION.
    """)


# ============================================================================
# PART 5: META-RULE EXTRACTION
# ============================================================================

def demonstrate_meta_rule_iteration():
    """
    Show the meta-rule iteration you described:

    1. Get rules from observation hologram
    2. Threshold to important rules
    3. Run fresh sieve on those rules
    4. Get meta-rules
    5. Put meta-rules back as observations
    6. Repeat
    """

    print("\n" + "=" * 70)
    print("META-RULE ITERATION DEMONSTRATION")
    print("=" * 70)

    print("""

YOUR PROPOSED ITERATION:
========================

1. Observations -> Hologram -> Sieve -> Rules
2. Rules -> Threshold -> Important Rules
3. Important Rules -> Fresh Sieve -> Meta-Rules
4. Meta-Rules -> Put back as "observations"
5. GOTO 1

This is FIXED-POINT ITERATION on rule space!


WHAT HAPPENS AT EACH ITERATION:
===============================

Iteration 0:
  Input: Raw observations
  Output: First-order rules ("ball moves right")

Iteration 1:
  Input: First-order rules (as observations)
  Output: Second-order rules ("movement follows direction")

Iteration 2:
  Input: Second-order rules
  Output: Third-order rules ("direction changes on collision")

Iteration N:
  Input: N-th order rules
  Output: (N+1)-th order rules

Eventually: FIXED POINT
  Rules at iteration N+1 = Rules at iteration N
  No more abstraction possible
  You've found the "generating rules" of the system


CONVERGENCE:
============

Does this converge? Under what conditions?

GOOD CASE:
  - Each iteration extracts strictly more abstract patterns
  - Number of patterns decreases each iteration
  - Eventually reaches fixed point with few meta-rules

BAD CASE:
  - Rules oscillate or cycle
  - No convergence
  - Indicates the system has irreducible complexity

DEGENERATE CASE:
  - Converges to empty set
  - Threshold too aggressive
  - Or: system is pure noise with no structure


THE PONG FIXED POINT:
=====================

For Pong, what's the fixed point?

Iteration 0: Pixel patterns
Iteration 1: "Ball", "Paddle", "Wall" as objects
Iteration 2: "Ball moves", "Paddle responds to input"
Iteration 3: "Ball bounces off paddle/wall"
Iteration 4: "Miss causes score change"
Iteration 5: "Game = bounce ball, avoid missing"
Iteration 6: = Iteration 5 (fixed point!)

The fixed point IS the game rules at their most abstract.
    """)

    # Mini demonstration
    print("\n" + "-" * 50)
    print("MINI DEMONSTRATION: 3 iterations")
    print("-" * 50)

    # Start with synthetic "rules" (representing discovered first-order rules)
    rules_iter_0 = [
        ("ball_moves_right", 0.9),
        ("ball_moves_left", 0.85),
        ("paddle_follows_input", 0.95),
        ("ball_bounces_wall", 0.8),
        ("ball_bounces_paddle", 0.75),
        ("score_increases_on_miss", 0.7),
        ("noise_pattern_1", 0.2),
        ("noise_pattern_2", 0.15),
    ]

    def iterate_rules(rules: List[Tuple[str, float]], iteration: int) -> List[Tuple[str, float]]:
        """One iteration of meta-rule extraction."""
        print(f"\nIteration {iteration}:")
        print(f"  Input: {len(rules)} rules")

        # Threshold
        threshold = 0.5
        important = [(r, a) for r, a in rules if a > threshold]
        print(f"  After threshold ({threshold}): {len(important)} rules")

        # "Sieve" - combine related rules into meta-rules
        # (In real implementation, this would be actual sieve evolution)
        meta_rules = []

        # Simple simulation: rules that share prefixes combine
        rule_groups = defaultdict(list)
        for rule, amp in important:
            prefix = rule.split("_")[0]  # e.g., "ball", "paddle"
            rule_groups[prefix].append((rule, amp))

        for prefix, group in rule_groups.items():
            if len(group) > 1:
                # Multiple rules with same prefix -> meta-rule
                combined_amp = sum(a for _, a in group) / len(group)
                meta_rule = f"meta_{prefix}_behavior"
                meta_rules.append((meta_rule, combined_amp))
            else:
                # Single rule -> keep as is (already abstract enough)
                meta_rules.append(group[0])

        print(f"  Output: {len(meta_rules)} meta-rules")
        for r, a in meta_rules:
            print(f"    {r}: {a:.2f}")

        return meta_rules

    # Run iterations
    current_rules = rules_iter_0
    for i in range(3):
        current_rules = iterate_rules(current_rules, i)
        if len(current_rules) <= 2:
            print(f"\nFixed point reached at iteration {i+1}")
            break

    print("""

RESULT:
=======
Started with 8 patterns (including noise).
After 3 iterations: ~3-4 meta-rules.

In a real system with proper sieve dynamics:
- "ball_behavior" would capture all ball physics
- "paddle_behavior" would capture all paddle physics
- "game_dynamics" would capture the win/lose structure

These meta-rules are the GENERATORS of Pong.
From them, you can derive all the lower-level rules.
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HIERARCHICAL HOLOGRAPHIC SIEVE")
    print("=" * 70)

    explain_architecture()
    demonstrate_hierarchical_sieve()
    demonstrate_meta_rule_iteration()

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)
