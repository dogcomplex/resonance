"""
INDUCTIVE DISCOVERY: Learning Rules from Observations
=====================================================

The inverse problem:
- Traditional sieve: Given rules, find stable patterns
- Inductive sieve: Given observations, discover rules

Applied to: Learning Pong (or any game) from raw experience
"""

from substrate import Substrate, RuleHamiltonian, LazyHamiltonian, DiscreteConfig
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import hashlib

# ============================================================================
# PART 1: THE FUNDAMENTAL CHALLENGE
# ============================================================================

def explain_inductive_challenge():
    """
    Why is learning rules from observations hard?
    """

    print("=" * 70)
    print("PART 1: THE INDUCTIVE DISCOVERY CHALLENGE")
    print("=" * 70)

    print("""

THE PROBLEM:
============
You're playing Pong for the first time.
- Screen: 210x160 pixels, 3 colors = 3^33600 possible states
- Actions: Left, Right, Stay = 3 choices
- You see: state_1 -> action -> state_2 -> action -> state_3 -> ...

Question: What are the "rules" of Pong?


WHY THIS IS HARD (TRADITIONAL VIEW):
====================================

1. STATE SPACE EXPLOSION
   - 3^33600 possible screens (effectively infinite)
   - Each screen is a unique "token"
   - N^2 rules for N tokens = impossible

2. GENERALIZATION PROBLEM
   - You see screen_A -> screen_B
   - Does this mean screen_A ALWAYS goes to screen_B?
   - Or only with this action?
   - Or only in this context?

3. ABSTRACTION PROBLEM
   - The "paddle" isn't labeled in the pixels
   - You have to DISCOVER that some pixels "move together"
   - And that moving-together-thing responds to actions


THE SIEVE APPROACH - KEY INSIGHT:
=================================

Don't tokenize STATES. Tokenize TRANSITIONS.

Traditional:
  Token = screen configuration
  Rule = screen_A -> screen_B
  Problem: infinite tokens

Inductive:
  Token = (local_pattern, action, local_change)
  Rule = "when you see X and do Y, Z happens"
  These are MUCH fewer and COMPOSE


THE PONG EXAMPLE:
=================

Instead of:
  "Screen with paddle at x=50, ball at (30,40)"

Tokenize as:
  "Paddle exists" (discovered by finding thing that moves with actions)
  "Ball exists" (discovered by finding thing that moves without actions)
  "Ball approaching paddle" (spatial relationship)
  "Action = Left" (part of observation)

Rules that emerge:
  (ball_approaching, action_left, paddle_moves_left)
  (ball_approaching, action_stay, paddle_stays)
  (paddle_intercepts, _, ball_bounces)
  (paddle_misses, _, score_changes)

These are the ACTUAL rules of Pong, not pixel configurations.


THE DISCOVERY PROCESS:
======================

1. OBSERVE: Collect (state, action, next_state) tuples
2. HASH: Convert states to compressed representations
3. DETECT: Find which parts of state change with/without action
4. CLUSTER: Group similar transitions
5. TOKENIZE: Create tokens for discovered patterns
6. SIEVE: Let patterns compete to find stable rules
7. ITERATE: Refine tokens based on what survives
    """)


# ============================================================================
# PART 2: HIERARCHICAL TOKENIZATION
# ============================================================================

def explain_hierarchical_tokenization():
    """
    How to avoid the pixel explosion problem.
    """

    print("\n" + "=" * 70)
    print("PART 2: HIERARCHICAL TOKENIZATION")
    print("=" * 70)

    print("""

THE PIXEL PROBLEM:
==================
Raw pixels are the WRONG level of abstraction.
Like trying to understand a novel by analyzing ink molecules.

SOLUTION: HIERARCHICAL TOKENS
=============================

Level 0: Raw pixels (too fine, don't use directly)
Level 1: Local patterns (3x3, 5x5 patches) - ~1000 types
Level 2: Objects (connected regions of similar patterns)
Level 3: Relationships (object A is left of object B)
Level 4: Dynamics (object A is moving toward object B)
Level 5: Rules (when A hits B, A bounces)


HOW TO BUILD THE HIERARCHY:
===========================

LEVEL 1: Local Patterns
-----------------------
Divide screen into patches (e.g., 8x8 pixels).
Hash each patch to a pattern ID.
Pong has only ~10-20 distinct patch types:
  - Empty (black)
  - Paddle (white rectangle)
  - Ball (white square)
  - Score digit patterns
  - Border

Now you have: 26x20 = 520 patches per frame, ~20 pattern types
Much better than 3^33600!


LEVEL 2: Objects (Emergent)
---------------------------
Find CONNECTED patches of same/similar type.
Track which patches move TOGETHER across frames.

Discovery process:
  Frame 1: patches A,B,C are white, adjacent
  Frame 2: patches A,B,C are still white, all shifted right
  Conclusion: A,B,C form an "object"

In Pong, this discovers:
  - The paddle (moves with actions)
  - The ball (moves without actions)
  - The score (changes rarely)
  - The border (never changes)


LEVEL 3: Relationships
----------------------
Once objects exist, compute relationships:
  - ball.x < paddle.x (ball left of paddle)
  - ball.y approaching paddle.y (ball moving toward paddle)
  - distance(ball, paddle) < threshold (ball near paddle)

These relationships are the TOKENS for the sieve.


LEVEL 4: Dynamics (Via the Sieve!)
----------------------------------
Now run the sieve on relationship tokens:
  - Inject observed (relationship, action, next_relationship) tuples
  - Let the sieve find which transitions are STABLE (rules)
  - Unstable transitions are noise or rare events


THE ANALOG QUESTION:
====================
"What if we used analog instead of discrete pixels?"

Interesting! Options:

1. CONTINUOUS PATCHES
   Instead of hashing to discrete pattern IDs,
   use continuous features (mean brightness, gradient, etc.)
   Map to DiscreteConfig by quantizing

2. FOURIER REPRESENTATION
   Transform patches to frequency domain
   Low frequencies = coarse structure
   High frequencies = details
   Truncate high frequencies for compression

3. LEARNED EMBEDDINGS
   Use a small neural net to compress patches
   But then we're mixing paradigms (NN + sieve)

For pure sieve approach: quantize analog to ~100-1000 levels
This preserves more information than binary
But keeps token space manageable
    """)


# ============================================================================
# PART 3: THE INDUCTIVE SIEVE ALGORITHM
# ============================================================================

class InductiveSieve:
    """
    A sieve that learns rules from observations.
    """

    def __init__(self, n_pattern_types: int = 100, relationship_types: int = 20):
        self.n_patterns = n_pattern_types
        self.n_relations = relationship_types

        # Observation memory
        self.transitions: List[Tuple[tuple, int, tuple]] = []  # (state_hash, action, next_state_hash)

        # Discovered patterns
        self.pattern_counts: Dict[tuple, int] = defaultdict(int)
        self.transition_counts: Dict[Tuple[tuple, int, tuple], int] = defaultdict(int)

        # Token registry
        self.tokens: Dict[str, int] = {}  # token_name -> token_id
        self.next_token_id = 0

        # The sieve (built dynamically)
        self.rules: List[Tuple[DiscreteConfig, DiscreteConfig, complex]] = []
        self.substrate = None

    def observe(self, state_hash: tuple, action: int, next_state_hash: tuple):
        """Record an observation."""
        self.transitions.append((state_hash, action, next_state_hash))
        self.pattern_counts[state_hash] += 1
        self.pattern_counts[next_state_hash] += 1
        self.transition_counts[(state_hash, action, next_state_hash)] += 1

    def get_or_create_token(self, name: str) -> int:
        """Get token ID, creating if needed."""
        if name not in self.tokens:
            self.tokens[name] = self.next_token_id
            self.next_token_id += 1
        return self.tokens[name]

    def build_sieve_from_observations(self, min_count: int = 2):
        """
        Build a sieve from observed transitions.

        Key insight: Frequent transitions become AMPLIFIED rules.
        Rare transitions become WEAK rules.
        The sieve then finds which rules are STABLE.
        """
        self.rules = []

        # Convert transitions to rules with coupling proportional to frequency
        max_count = max(self.transition_counts.values()) if self.transition_counts else 1

        for (state, action, next_state), count in self.transition_counts.items():
            if count < min_count:
                continue

            # Token for (state, action) combination
            src_name = f"s{hash(state) % 1000}_a{action}"
            src_id = self.get_or_create_token(src_name)

            # Token for next_state
            tgt_name = f"s{hash(next_state) % 1000}"
            tgt_id = self.get_or_create_token(tgt_name)

            # Coupling strength proportional to observed frequency
            coupling = complex(count / max_count, 0)

            self.rules.append((
                DiscreteConfig((src_id,)),
                DiscreteConfig((tgt_id,)),
                coupling
            ))

        if self.rules:
            H = RuleHamiltonian(self.rules)
            self.substrate = Substrate(H, damping=0.1)

            # Inject observed states
            for name, token_id in self.tokens.items():
                self.substrate.inject(DiscreteConfig((token_id,)), complex(1.0, 0))

    def discover_rules(self, evolution_time: float = 2.0) -> List[Tuple[str, str, float]]:
        """
        Evolve the sieve and extract stable rules.
        """
        if self.substrate is None:
            return []

        self.substrate.evolve(duration=evolution_time)

        # Find dominant configurations
        dominant = self.substrate.dominant_configs(n=20)

        # Map back to human-readable rules
        id_to_name = {v: k for k, v in self.tokens.items()}

        stable_rules = []
        for config, amplitude in dominant:
            if len(config.tokens) == 1:
                token_id = config.tokens[0]
                name = id_to_name.get(token_id, f"unknown_{token_id}")
                stable_rules.append((name, abs(amplitude)))

        return stable_rules


def demonstrate_inductive_learning():
    """
    Simulate learning Pong-like rules from observations.
    """

    print("\n" + "=" * 70)
    print("PART 3: DEMONSTRATION - LEARNING PONG RULES")
    print("=" * 70)

    # Simulate Pong observations (simplified)
    # State = (ball_x_region, ball_y_region, paddle_region, ball_direction)
    # Regions: 0=left, 1=center, 2=right for x; 0=top, 1=mid, 2=bottom for y

    sieve = InductiveSieve()

    print("\nSimulating 100 Pong observations...")

    # Generate synthetic Pong-like transitions
    np.random.seed(42)

    for _ in range(100):
        # Random state
        ball_x = np.random.randint(0, 3)
        ball_y = np.random.randint(0, 3)
        paddle = np.random.randint(0, 3)
        direction = np.random.randint(0, 4)  # 0=up-left, 1=up-right, 2=down-left, 3=down-right

        state = (ball_x, ball_y, paddle, direction)

        # Random action (0=left, 1=stay, 2=right)
        action = np.random.randint(0, 3)

        # Simulate next state (simplified physics)
        # Paddle moves with action
        new_paddle = max(0, min(2, paddle + (action - 1)))

        # Ball moves in direction (with some noise)
        dx = 1 if direction in [1, 3] else -1
        dy = 1 if direction in [2, 3] else -1
        new_ball_x = max(0, min(2, ball_x + dx))
        new_ball_y = max(0, min(2, ball_y + dy))

        # Ball bounces off edges
        new_direction = direction
        if new_ball_x == 0 or new_ball_x == 2:
            new_direction = direction ^ 1  # Flip horizontal
        if new_ball_y == 0 or new_ball_y == 2:
            new_direction = direction ^ 2  # Flip vertical

        next_state = (new_ball_x, new_ball_y, new_paddle, new_direction)

        sieve.observe(state, action, next_state)

    print(f"Collected {len(sieve.transitions)} transitions")
    print(f"Unique states: {len(sieve.pattern_counts)}")
    print(f"Unique transitions: {len(sieve.transition_counts)}")

    # Build and run sieve
    print("\nBuilding sieve from observations...")
    sieve.build_sieve_from_observations(min_count=1)

    print(f"Created {len(sieve.tokens)} tokens")
    print(f"Created {len(sieve.rules)} rules")

    # Discover stable rules
    print("\nEvolving to find stable patterns...")
    stable = sieve.discover_rules(evolution_time=2.0)

    print("\nMost stable patterns (discovered rules):")
    print("-" * 50)
    for name, amplitude in sorted(stable, key=lambda x: -x[1])[:15]:
        print(f"  {name}: amplitude={amplitude:.3f}")

    print("""

INTERPRETATION:
===============
The sieve discovered which (state, action) -> next_state transitions
are CONSISTENT (appear frequently and survive competition).

In a real Pong scenario with proper tokenization:
- "paddle_moves_with_action" would emerge as stable
- "ball_bounces_off_wall" would emerge as stable
- "ball_continues_in_direction" would emerge as stable
- Random noise transitions would decay away

The sieve is doing RULE INDUCTION:
- Frequent patterns get amplified
- Consistent patterns reinforce each other
- Contradictory/noisy patterns interfere and decay
    """)


# ============================================================================
# PART 4: MEMORY AND HOLOGRAMS
# ============================================================================

def explain_memory_and_holograms():
    """
    How does memory fit in? What about holographic encoding?
    """

    print("\n" + "=" * 70)
    print("PART 4: MEMORY AND HOLOGRAPHIC ENCODING")
    print("=" * 70)

    print("""

THE MEMORY QUESTION:
====================
The sieve naturally HAS memory - it's the amplitude field.
- Past observations inject amplitude
- Amplitude persists (with damping)
- The CURRENT field state IS the memory

But this is SHORT-TERM memory (decays with damping).


LONG-TERM MEMORY OPTIONS:
=========================

1. RULE CRYSTALLIZATION
   When a rule becomes very stable (high amplitude, long survival),
   "crystallize" it: permanently amplify its coupling.

   This is like moving from working memory to long-term memory:
   - Short-term: amplitude in the field
   - Long-term: coupling strength in the Hamiltonian

2. HIERARCHICAL MEMORY
   Lower levels remember concrete patterns.
   Higher levels remember abstract rules.

   Pong example:
   - Level 1: "I saw these specific pixels"
   - Level 2: "There was a ball at position X"
   - Level 3: "The ball was approaching the paddle"
   - Level 4: "The ball bounces when it hits the paddle"

   Higher levels are more compressed and last longer.


THE HOLOGRAPHIC INSIGHT:
========================

Holograms have a remarkable property:
- Information is distributed across the whole medium
- Any piece contains the whole (at lower resolution)
- Interference patterns encode relationships

This maps to the sieve beautifully:

1. DISTRIBUTED ENCODING
   Don't store "ball at (50, 30)" as a single token.
   Store it as interference between:
   - "ball exists" token
   - "x ~ 50" token (with phase encoding exact position)
   - "y ~ 30" token (with phase encoding exact position)

   The COMBINATION of phases encodes the state.
   Any subset gives partial information.

2. PHASE AS MEMORY
   We've been using amplitude for "how strong" a pattern is.
   But PHASE encodes "how patterns relate".

   Two tokens with same phase: constructively interfere (consistent)
   Two tokens with opposite phase: destructively interfere (contradictory)

   This is holographic memory: relationships stored as phases.

3. ASSOCIATIVE RECALL
   In a hologram, shining part of the reference beam recalls the full image.
   In the sieve: injecting partial state activates associated patterns.

   See ball position -> recall ball direction (from phase correlations)
   This is CONTENT-ADDRESSABLE memory.


IMPLEMENTING HOLOGRAPHIC MEMORY:
================================

class HolographicSieve:
    def encode_state(self, state_features):
        '''
        Encode a state as interference pattern.

        Each feature gets a token.
        The PHASE of injection encodes the feature VALUE.
        '''
        for feature_name, feature_value in state_features:
            token = self.get_token(feature_name)
            # Phase encodes value: value in [0,1] -> phase in [0, 2*pi]
            phase = feature_value * 2 * np.pi
            amplitude = complex(np.cos(phase), np.sin(phase))
            self.substrate.inject(token, amplitude)

    def recall(self, partial_state):
        '''
        Given partial state, recall associated features.

        Inject partial state, evolve briefly, read out phases.
        '''
        # Inject known features
        for feature_name, feature_value in partial_state:
            self.encode_feature(feature_name, feature_value)

        # Brief evolution lets correlations activate
        self.substrate.step()

        # Read out all features
        recalled = {}
        for feature_name, token in self.tokens.items():
            config = DiscreteConfig((token,))
            amplitude = self.substrate.psi[config]
            # Phase decodes value
            phase = np.angle(amplitude)
            value = phase / (2 * np.pi)
            recalled[feature_name] = value

        return recalled

This is speculative but follows from the sieve's wave mechanics.


WHY HOLOGRAMS FIT:
==================

The sieve IS a kind of hologram:
- Distributed storage (all rules interact)
- Phase relationships (interference)
- Associative recall (inject partial -> get whole)
- Graceful degradation (lose some, keep approximation)

The difference:
- Optical holograms are continuous
- Sieve is discrete (tokens)
- But the DYNAMICS are the same
    """)


# ============================================================================
# PART 5: EXPANDING THE TOKEN BASE
# ============================================================================

def explain_token_expansion():
    """
    Should we increase tokens as we observe more?
    """

    print("\n" + "=" * 70)
    print("PART 5: DYNAMIC TOKEN EXPANSION")
    print("=" * 70)

    print("""

THE EXPANSION QUESTION:
=======================
"Is the correct way to explore more observation space
 to increase token base each time?"

ANSWER: Yes, but CAREFULLY.


NAIVE APPROACH (Bad):
=====================
See new state? Add new token!

Problem: Token count explodes.
After 1000 observations: 1000 tokens, 10^6 rules.
After 10000 observations: unusable.


SMART APPROACH: HIERARCHICAL EXPANSION
======================================

Level 1 tokens: Fixed, small set (~100)
  These are your "alphabet" - primitive features
  E.g., patch types, actions, basic relationships

Level 2 tokens: Grow slowly
  These are "words" - compositions of primitives
  Only add when a NEW composition is seen REPEATEDLY

Level 3 tokens: Grow even slower
  These are "phrases" - compositions of compositions
  Only add when truly novel structure appears


THE COMPRESSION PRINCIPLE:
==========================

Add a new token ONLY when:
1. A pattern appears frequently (not noise)
2. It cannot be well-approximated by existing tokens
3. Adding it would REDUCE total description length

This is Minimum Description Length (MDL) principle:
  cost = code_length(model) + code_length(data|model)
  New token pays off if it compresses data more than it costs


PRACTICAL ALGORITHM:
====================

class AdaptiveSieve:
    def __init__(self, max_tokens=1000):
        self.max_tokens = max_tokens
        self.token_usage = defaultdict(int)
        self.candidate_patterns = defaultdict(int)

    def observe(self, pattern):
        # Check if pattern matches existing token
        match = self.find_closest_token(pattern)

        if match and self.distance(pattern, match) < threshold:
            # Use existing token
            self.token_usage[match] += 1
        else:
            # Candidate for new token
            self.candidate_patterns[pattern] += 1

    def maybe_add_tokens(self):
        # Periodically, promote frequent candidates to tokens
        for pattern, count in self.candidate_patterns.items():
            if count > MIN_COUNT and len(self.tokens) < self.max_tokens:
                # Check MDL criterion
                if self.mdl_improvement(pattern) > 0:
                    self.add_token(pattern)

    def prune_tokens(self):
        # Remove tokens that are rarely used
        for token, count in list(self.token_usage.items()):
            if count < MIN_USAGE:
                self.remove_token(token)


THE PONG TRAJECTORY:
====================

T=0: Start with primitive tokens
  - Patches: ~20 types
  - Actions: 3
  - Basic relationships: ~10

T=100: Discover objects
  - "Paddle" emerges (patches that move together with action)
  - "Ball" emerges (patches that move without action)
  - Add tokens for these objects

T=500: Discover dynamics
  - "Ball approaching paddle" (relationship + direction)
  - "Ball bouncing" (direction change pattern)
  - Add tokens for these dynamics

T=1000: Discover rules
  - "Paddle intercept causes bounce" (stable rule)
  - "Miss causes score change" (stable rule)
  - These are RULE tokens, not state tokens

Final token count: ~50-100, not 10000
Because abstraction compresses


THE KEY INSIGHT:
================

Don't expand tokens to COVER observation space.
Expand tokens to COMPRESS observation space.

Good tokens are ones that:
- Appear frequently
- Participate in stable rules
- Compose well with other tokens
- Reduce description length

Bad tokens are ones that:
- Appear rarely
- Don't participate in rules
- Are redundant with other tokens
- Add complexity without compression
    """)


# ============================================================================
# PART 6: SYNTHESIS - THE DISCOVERY METHODOLOGY
# ============================================================================

def synthesis_discovery_methodology():
    """
    Putting it all together: How to discover any game.
    """

    print("\n" + "=" * 70)
    print("PART 6: SYNTHESIS - DISCOVERING ANY GAME")
    print("=" * 70)

    print("""

THE UNIVERSAL DISCOVERY METHODOLOGY:
====================================

STEP 1: PRIMITIVE TOKENIZATION
------------------------------
Start with LOW-LEVEL tokens that are:
- Domain-agnostic (work for any game)
- Small in number (~100)
- Based on LOCAL features

For visual games:
  - Patch types (8x8 pixel patterns)
  - Edges, corners, regions
  - Color histograms

For abstract games:
  - Symbols in observation
  - Positions, counts
  - Basic predicates


STEP 2: OBSERVATION COLLECTION
------------------------------
Watch gameplay (or play randomly).
Record: (observation, action, next_observation)

Don't tokenize yet - just collect raw data.
Need enough to see patterns: ~100-1000 transitions.


STEP 3: FEATURE EXTRACTION
--------------------------
Convert observations to feature vectors.
Use LOCAL, COMPOSABLE features.

Apply hierarchical compression:
  Raw pixels -> patches -> objects -> relationships


STEP 4: TRANSITION ANALYSIS
---------------------------
For each (state, action, next_state):
  - What changed? (active features)
  - What stayed same? (stable features)
  - What correlates with action? (controllable)
  - What correlates with time? (dynamics)

This separates:
  - Player-controlled elements (paddle)
  - Environment dynamics (ball)
  - Static elements (score display, borders)


STEP 5: INDUCTIVE SIEVE
-----------------------
Build sieve from discovered transitions:
  - Tokens = discovered features/objects/relationships
  - Rules = observed transitions
  - Coupling = frequency of observation

Evolve and find stable patterns.
Stable patterns = reliable rules of the game.


STEP 6: TOKEN REFINEMENT
------------------------
Based on what survives:
  - KEEP tokens that participate in stable rules
  - MERGE tokens that always appear together
  - SPLIT tokens that behave differently in different contexts
  - REMOVE tokens that never participate in rules

Iterate steps 5-6 until stable vocabulary.


STEP 7: RULE EXTRACTION
-----------------------
The stable patterns ARE the rules.
Express in human terms:

Sieve output: "token_ball_right + action_left -> token_paddle_left"
Human rule: "When ball is on right and you press left, paddle moves left"


WHAT THIS DISCOVERS:
====================

For Pong:
  - Paddle responds to actions (controllability)
  - Ball moves independently (physics)
  - Ball bounces off walls and paddle (collision rules)
  - Missing ball changes score (lose condition)

For Chess (from watching):
  - Pieces move in specific patterns (movement rules)
  - Taking removes opponent piece (capture)
  - Some moves are forbidden (illegal move detection)
  - Game ends in certain configurations (win/lose)

For ANY game:
  - What you control
  - What moves on its own
  - How things interact
  - What ends the game


LIMITATIONS:
============

1. SAMPLE COMPLEXITY
   Rare rules need many observations to discover.
   Pong's "paddle intercept" might take 1000 games.
   Chess's "castling" might take 10000 games.

2. ABSTRACTION CEILING
   The sieve finds rules at the token level.
   High-level strategy requires high-level tokens.
   "Fork attack" needs "piece threatening squares" tokens.

3. NO PLANNING
   Discovering rules != knowing how to win.
   Still need search/planning on top of discovered rules.


THE SIEVE'S ROLE:
=================

The sieve is a RULE DISCOVERER, not a GAME SOLVER.

It answers: "What patterns are consistent in this environment?"
It doesn't answer: "What should I do to win?"

But knowing the rules is NECESSARY for good play.
And the sieve finds rules WITHOUT being told them.

This is the value: AUTOMATIC RULE INDUCTION.
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("INDUCTIVE DISCOVERY: Learning Rules from Observations")
    print("=" * 70)

    explain_inductive_challenge()
    explain_hierarchical_tokenization()
    demonstrate_inductive_learning()
    explain_memory_and_holograms()
    explain_token_expansion()
    synthesis_discovery_methodology()

    print("\n" + "=" * 70)
    print("INDUCTIVE DISCOVERY EXPLORATION COMPLETE")
    print("=" * 70)
