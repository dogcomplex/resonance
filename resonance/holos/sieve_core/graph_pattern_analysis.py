"""
GRAPH PATTERN ANALYSIS - Visual Mapping of Rules to Functors

This answers:
1. What do the 10 immortal graph patterns actually look like?
2. Why do we call each pattern by its functor name?
3. How rigorous is this mapping academically?
4. What do the 14 thermal extras contribute?
5. What are the ephemeral (mortal) rules?
6. Can we reach N tokens with the right randomness?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holos.sieve_core.substrate import AmplitudeField
from holos.sieve_core.emergence import SelfOrganizingSubstrate, Entity, EntityType
import random
import math
import cmath
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional, Any


# ============================================================
# PART 1: THE 10 IMMORTAL PATTERNS - VISUAL GRAPHS
# ============================================================

def show_immortal_patterns():
    """
    Show exactly what each of the 10 immortal pattern TYPES looks like
    as a graph, and explain the functor analogy.
    """

    print("=" * 80)
    print("THE 10 IMMORTAL PATTERN TYPES - GRAPH STRUCTURES")
    print("=" * 80)

    patterns = [
        {
            "name": "IDENTITY",
            "symbol": "Id",
            "functor": "Id (Identity Functor)",
            "graph": """
    +---+
    | a |--+
    +---+  |
      ^----+
    (self-loop)
            """,
            "algebraic": "f: A -> A where f = id",
            "why_functor": """
WHY "IDENTITY FUNCTOR"?

In category theory, the identity functor Id: C -> C maps:
  - Every object A to itself: Id(A) = A
  - Every morphism f to itself: Id(f) = f

A self-loop rule a -> a does exactly this:
  - Token a maps to token a
  - The transition is trivial (no change)

RIGOR: ***** (Perfect analogy)
This IS the identity morphism in the category of tokens.
            """,
            "programming": "x => x, lambda x: x, pass-through",
            "survival": "Always survives - does nothing, costs nothing"
        },

        {
            "name": "UNARY FLOW",
            "symbol": "Fl",
            "functor": "Hom(A,B) (Hom Functor)",
            "graph": """
    +---+     +---+
    | a | --> | b |
    +---+     +---+
    (directed edge)
            """,
            "algebraic": "f: A -> B (single morphism)",
            "why_functor": """
WHY "HOM FUNCTOR"?

In category theory, Hom(A,B) is the set of all morphisms from A to B.
A single rule a -> b is ONE ELEMENT of this Hom set.

More precisely: the rule IS a morphism in the category.
The "Hom functor" label means "this is a basic morphism."

RIGOR: ****o (Good analogy)
The rule is a morphism. Calling it "Hom functor" is slightly loose -
it's an element of Hom(a,b), not the functor Hom(-,-) itself.
But in the context of "what kind of thing is this?", it's accurate.
            """,
            "programming": "map(f), pipe, transform, state transition",
            "survival": "Survives if endpoints are active and it mediates real transitions"
        },

        {
            "name": "2-CYCLE (Symmetric Pair)",
            "symbol": "C2",
            "functor": "Aut_2 / Iso (Order-2 Automorphism / Isomorphism)",
            "graph": """
    +---+     +---+
    | a | <=> | b |
    +---+     +---+
    (bidirectional: a->b AND b->a)
            """,
            "algebraic": "f: A -> B, g: B -> A where gof = id_A, fog = id_B",
            "why_functor": """
WHY "ISOMORPHISM" or "Aut_2"?

Two rules a->b and b->a together form an ISOMORPHISM:
  - You can go from a to b
  - You can go back from b to a
  - Round trip = identity (in terms of reachability)

Aut_2 means "order-2 automorphism" - applying twice returns to start:
  f(f(x)) = x  (like NOT(NOT(x)) = x)

RIGOR: ***** (Excellent analogy)
This IS an isomorphism in the graph category.
The pair (f, f^-1) satisfies the isomorphism axioms exactly.
            """,
            "programming": "encode/decode, encrypt/decrypt, toggle, swap",
            "survival": "Very stable - mutual reinforcement between the two rules"
        },

        {
            "name": "3-CYCLE",
            "symbol": "C3",
            "functor": "Aut_3 (Order-3 Automorphism)",
            "graph": """
        +---+
        | a |
        +---+
       /     \
    +---+   +---+
    | c | -> | b |
    +---+   +---+
    (triangle)
            """,
            "algebraic": "f: A->B, g: B->C, h: C->A where hogof = id (mod 3)",
            "why_functor": """
WHY "Aut_3" (Order-3 Automorphism)?

A 3-cycle implements a Z/3Z group action:
  - Apply once: a -> b
  - Apply twice: a -> b -> c
  - Apply three times: a -> b -> c -> a (back to start!)

This is like the cube roots of unity: omega^3 = 1
Or RGB color rotation: R -> G -> B -> R

RIGOR: ****o (Good analogy)
The cycle does form a Z/3Z action on the three tokens.
"Automorphism" is slightly loose since it's a permutation of 3 objects,
not an automorphism of a single object. But the group structure is exact.
            """,
            "programming": "state = (state + 1) % 3, round-robin, traffic light",
            "survival": "Very stable - all three rules reinforce each other cyclically"
        },

        {
            "name": "BROADCAST (Fan-out)",
            "symbol": "Bc",
            "functor": "Delta (Delta / Diagonal Functor)",
            "graph": """
           +---+
           | b |
           +---+
            ^
    +---+   |   +---+
    | a | --+-> | c |
    +---+   |   +---+
            v
           +---+
           | d |
           +---+
    (one source, many targets)
            """,
            "algebraic": "Delta(x) = (x, x, x, ...) - diagonal/copy",
            "why_functor": """
WHY "DELTA / DIAGONAL FUNCTOR"?

The diagonal functor Delta: C -> CxCx... sends:
  - Object A to (A, A, A, ...)
  - Morphism f to (f, f, f, ...)

A broadcast source does exactly this:
  - One input token
  - Multiple copies sent to different targets
  - Same "information" replicated

RIGOR: ***oo (Decent analogy)
The pattern LOOKS like the diagonal functor's action.
But it's not literally a functor - it's a subgraph pattern.
The analogy is structural/visual, not formal.
            """,
            "programming": "fork(), multicast, pub/sub publish, tee",
            "survival": "Survives if source token is active and targets exist"
        },

        {
            "name": "COLLECTOR (Fan-in)",
            "symbol": "Cl",
            "functor": "Nabla (Nabla / Codiagonal Functor)",
            "graph": """
           +---+
           | b |
           +---+
            v
    +---+   |   +---+
    | a | --+-> | d |
    +---+   |   +---+
            ^
           +---+
           | c |
           +---+
    (many sources, one target)
            """,
            "algebraic": "Nabla(x, y, z) = x + y + z - codiagonal/merge",
            "why_functor": """
WHY "NABLA / CODIAGONAL FUNCTOR"?

The codiagonal Nabla: C+C+... -> C sends:
  - Multiple copies to a single object
  - It's the "folding" operation that merges branches

A collector target does exactly this:
  - Multiple input tokens
  - All flow into one target
  - Information merges/aggregates

RIGOR: ***oo (Decent analogy)
Same caveat as broadcast - it's a structural pattern that
resembles what the codiagonal does, not a literal functor.
            """,
            "programming": "reduce(), merge(), join(), aggregate",
            "survival": "Survives if target token is a stable attractor"
        },

        {
            "name": "BRIDGE (Natural Transformation)",
            "symbol": "Br",
            "functor": "eta: F => G (Natural Transformation)",
            "graph": """
    Region 1        Region 2
    +---+           +---+
    | a | --------> | c |
    +---+           +---+
      |               |
    +---+           +---+
    | b |           | d |
    +---+           +---+
    (connects separate clusters)
            """,
            "algebraic": "eta_X: F(X) -> G(X) - component at X",
            "why_functor": """
WHY "NATURAL TRANSFORMATION"?

A natural transformation eta: F => G provides a systematic way
to transform one functor's output to another's.

A bridge rule connects two otherwise separate regions:
  - Without it, regions are disconnected
  - It provides a "translation" between them
  - Like an adapter or gateway

RIGOR: **ooo (Loose analogy)
This is the weakest mapping. Natural transformations are very
specific mathematical objects. A "bridge" is just a graph pattern.
The analogy is: both "connect different structures systematically."
            """,
            "programming": "adapter pattern, API gateway, protocol translator",
            "survival": "Survives if both regions are active and need connection"
        },

        {
            "name": "SYMMETRIC PAIR",
            "symbol": "Sy",
            "functor": "Iso (Isomorphism Functor)",
            "graph": """
    +---+     +---+
    | a | <=> | b |
    +---+     +---+
    (same as 2-cycle, but emphasized as equivalence)
            """,
            "algebraic": "A ~= B (isomorphic objects)",
            "why_functor": """
WHY "ISOMORPHISM"?

Same as 2-cycle. The emphasis here is on the EQUIVALENCE:
  - a and b are "the same" up to relabeling
  - You can substitute one for the other
  - Information is preserved both ways

RIGOR: ***** (Exact)
This IS an isomorphism. No analogy needed.
            """,
            "programming": "JSON.stringify/parse, serialize/deserialize",
            "survival": "Very stable due to mutual reinforcement"
        },

        {
            "name": "SERIAL COMPOSE",
            "symbol": "Cs",
            "functor": "o (Composition)",
            "graph": """
    +---+     +---+     +---+
    | a | --> | b | --> | c |
    +---+     +---+     +---+
    (chain)
            """,
            "algebraic": "g o f: A -> C (composition of f: A->B and g: B->C)",
            "why_functor": """
WHY "COMPOSITION"?

Composition is THE fundamental operation in category theory:
  - Given f: A->B and g: B->C
  - You get gof: A->C

A chain of rules implements exactly this:
  - Rule 1: a->b
  - Rule 2: b->c
  - Combined effect: a can reach c

RIGOR: ***** (Exact)
This IS composition of morphisms. The sieve implements it directly.
            """,
            "programming": "pipe, chain, middleware, Unix pipes |",
            "survival": "Survives if all links in the chain are active"
        },

        {
            "name": "CONSERVATION (Balanced)",
            "symbol": "Cm",
            "functor": "Ker/Coker (Conservation Functor)",
            "graph": """
         in=3
    a ----->
    b -----> [X] -----> d    out=3
    c ----->     -----> e
                -----> f
    (sum of inputs = sum of outputs)
            """,
            "algebraic": "sum inputs = sum outputs (Kirchhoff-like)",
            "why_functor": """
WHY "CONSERVATION / KERNEL"?

In algebra, the kernel of a map is what gets "conserved" (mapped to 0).
Conservation laws (Noether currents) have div(J) = 0.

A balanced node where in-degree = out-degree:
  - What flows in equals what flows out
  - No accumulation, no deficit
  - Like charge conservation or mass balance

RIGOR: **ooo (Loose analogy)
The connection to Ker/Coker is metaphorical.
The graph pattern resembles conservation, but isn't literally a kernel.
Better to just say "balanced flow" than claim it's a functor.
            """,
            "programming": "producer-consumer balance, connection pool",
            "survival": "Survives if system reaches equilibrium"
        },
    ]

    for p in patterns:
        print("\n" + "=" * 80)
        print(f"{p['name']} ({p['symbol']})")
        print(f"Functor: {p['functor']}")
        print("=" * 80)
        print(f"\nGRAPH PATTERN:{p['graph']}")
        print(f"ALGEBRAIC: {p['algebraic']}")
        print(p['why_functor'])
        print(f"\nPROGRAMMING: {p['programming']}")
        print(f"SURVIVAL: {p['survival']}")


# ============================================================
# PART 2: RIGOR SUMMARY
# ============================================================

def rigor_summary():
    """
    Summarize how rigorous each mapping is.
    """

    print("\n" + "=" * 80)
    print("RIGOR ASSESSMENT: How Academic Are These Mappings?")
    print("=" * 80)

    print("""
TIER 1: EXACT CORRESPONDENCES (*****)
=====================================
These ARE the mathematical objects, not analogies:

- IDENTITY (Id): A self-loop IS the identity morphism. Period.
- ISOMORPHISM (Iso): A bidirectional pair IS an isomorphism.
- COMPOSITION (o): A chain IS morphism composition.

These would pass peer review as correct mathematical statements.


TIER 2: STRONG ANALOGIES (****o)
=================================
These capture the essential structure accurately:

- HOM (Hom): A rule IS an element of Hom(a,b). Calling it "the Hom functor"
  is slightly loose, but the intuition is correct.

- Aut_2, Aut_3: Cycles DO form group actions. The Z/nZ structure is exact.
  "Automorphism" is slightly loose (it's a permutation, not an automorphism
  of a single object), but the group theory is sound.


TIER 3: STRUCTURAL ANALOGIES (***oo)
=====================================
These look like the functor patterns but aren't literally functors:

- DELTA (Delta): A high-out-degree source LOOKS LIKE the diagonal functor.
  But it's a graph pattern, not a functor between categories.

- NABLA (Nabla): Same issue - it's the SHAPE of codiagonal, not the functor.

These are useful mnemonics but wouldn't survive formal scrutiny.


TIER 4: LOOSE METAPHORS (**ooo)
================================
These are suggestive but quite loose:

- BRIDGE/NAT: "Connects different regions" vaguely resembles what
  natural transformations do, but the formal connection is weak.

- CONSERVATION/KER: The balanced-flow pattern is like conservation laws,
  but calling it "kernel" is a stretch.

These are pedagogically useful but mathematically imprecise.


PRACTICAL RECOMMENDATION:
=========================
For communication: Use the names. They convey intuition.
For formal work: Stick to graph-theoretic terminology:
  - "Bidirectional edge" not "isomorphism"
  - "High out-degree node" not "delta functor"
  - "3-cycle" not "Aut_3"

The sieve produces GRAPHS. Functor labels are INTERPRETATIONS.
""")


# ============================================================
# PART 3: THE 8th TOKEN - WHAT DOES IT CHANGE?
# ============================================================

def analyze_8th_token():
    """
    When thermal noise creates token 7 (the 8th token, 0-indexed),
    what happens to the existing 42 rules?
    """

    print("\n" + "=" * 80)
    print("THE 8th TOKEN: What Do the Extra 14 Rules Mean?")
    print("=" * 80)

    print("""
WHAT ARE THE 14 EXTRA RULES?
============================
With uniform random (n=7 tokens): 42 rules = 7 x 6
With thermal noise (n=8 tokens): 56 rules = 8 x 7

The 14 extras are ALL rules involving token 7:
  - 7 rules FROM others TO token 7: (0,)->(7,), (1,)->(7,), ..., (6,)->(7,)
  - 7 rules FROM token 7 TO others: (7,)->(0,), (7,)->(1,), ..., (7,)->(6,)

Token 7 is FULLY CONNECTED to all other tokens.


WHAT FUNCTOR TYPES ARE THESE?
=============================
""")

    # Analyze the 14 rules
    n = 8
    rules_to_7 = [(i, 7) for i in range(7)]
    rules_from_7 = [(7, i) for i in range(7)]

    print("Rules TO token 7 (Collector pattern):")
    print("  " + "  ".join(f"({i},)->(7,)" for i in range(7)))
    print(f"  Token 7 has in-degree 7 -> This is a COLLECTOR (Nabla)")
    print(f"  Token 7 receives from EVERYONE")
    print()

    print("Rules FROM token 7 (Broadcast pattern):")
    print("  " + "  ".join(f"(7,)->({i},)" for i in range(7)))
    print(f"  Token 7 has out-degree 7 -> This is a BROADCASTER (Delta)")
    print(f"  Token 7 sends to EVERYONE")
    print()

    print("""
COMBINED: Token 7 is a HUB (*)
==============================
Token 7 is connected TO and FROM every other token.
This makes it a CENTRAL HUB - the highest-connectivity node.

    0 <--> 7 <--> 1
    ^    |    ^
    6 <--> 7 <--> 2
    v    |    v
    5 <--> 7 <--> 3
         |
         4

In category theory terms:
  - Token 7 looks like a PRODUCT/COPRODUCT object
  - Everything can go through it
  - It's like a "terminal + initial" object combined


DOES THIS CHANGE THE OTHER 42?
==============================
The original 42 rules (between tokens 0-6) are UNCHANGED in structure.
But their ROLE changes:

BEFORE (7 tokens):
  - Every token is "equal" - symmetric graph K_7
  - No special center
  - All rules have similar importance

AFTER (8 tokens with central hub):
  - Token 7 is special (highest connectivity)
  - Original rules are now "peripheral"
  - Token 7 mediates many paths: a->7->b is often shorter than a->...->b

The original isomorphisms (e.g., 0<->1) still exist but are now
SUPPLEMENTED by paths through 7 (0<->7<->1).


PHYSICAL ANALOGY:
================
Token 7 is like the HIGGS FIELD:
  - Connected to everything
  - Mediates interactions between all particles
  - A "universal coupler"

Or like a ROUTER in a network:
  - All traffic CAN go through it
  - It's the central hub
  - Direct peer-to-peer still works, but hub is always available
""")


# ============================================================
# PART 4: EPHEMERAL (MORTAL) RULES
# ============================================================

def analyze_ephemeral_rules():
    """
    What rules DON'T survive? What are the mortal rules?
    """

    print("\n" + "=" * 80)
    print("EPHEMERAL RULES: What Dies and Why?")
    print("=" * 80)

    print("""
In a 7-token fully-connected system, ALL 42 rules can be immortal.
But in LARGER systems or with DIFFERENT dynamics, some rules die.

Let's experimentally find which rules are most likely to die.
""")

    # Run experiments with larger token counts
    n_tokens_list = [10, 15, 20]

    for n in n_tokens_list:
        print(f"\n--- {n} tokens (max {n*(n-1)} rules) ---")

        rule_survival = defaultdict(int)
        n_trials = 30

        for trial in range(n_trials):
            random.seed(trial * 1000)

            substrate = SelfOrganizingSubstrate()

            for t in range(n):
                phase = random.uniform(0, 2 * math.pi)
                mag = random.uniform(0.1, 1.0)
                substrate.inject_state(t, mag * cmath.exp(1j * phase))

            for _ in range(n * 2):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a != b:
                    phase = random.uniform(0, 2 * math.pi)
                    mag = random.uniform(0.1, 1.0)
                    substrate.inject_rule(a, b, mag * cmath.exp(1j * phase))

            for _ in range(500):
                substrate.step()

            # Count surviving rules
            for entity, amplitude in substrate.field:
                if isinstance(entity, Entity) and entity.entity_type == EntityType.RULE:
                    if abs(amplitude) > 0.1:
                        src = entity.content[0][0] if isinstance(entity.content[0], tuple) else entity.content[0]
                        tgt = entity.content[1][0] if isinstance(entity.content[1], tuple) else entity.content[1]
                        rule_survival[(src, tgt)] += 1

        # Find always-surviving and often-dying rules
        always_survive = [(r, c) for r, c in rule_survival.items() if c == n_trials]
        often_die = [(r, c) for r, c in rule_survival.items() if c < n_trials // 2]
        never_seen = n*(n-1) - len(rule_survival)

        print(f"  Always survive: {len(always_survive)} rules")
        print(f"  Often die (< 50% survival): {len(often_die)} rules")
        print(f"  Never seen: {never_seen} rules")

        # What makes a rule survive?
        if always_survive:
            # Analyze the survivors
            survivor_rules = [r for r, c in always_survive]

            # Check for symmetry
            symmetric_survivors = sum(1 for (a, b) in survivor_rules
                                     if (b, a) in survivor_rules)
            print(f"  Symmetric pairs among survivors: {symmetric_survivors // 2}")

            # Check for low-index bias
            avg_src = sum(a for (a, b) in survivor_rules) / len(survivor_rules)
            avg_tgt = sum(b for (a, b) in survivor_rules) / len(survivor_rules)
            print(f"  Survivor avg source: {avg_src:.1f}, avg target: {avg_tgt:.1f}")

    print("""

WHY DO RULES DIE?
=================

1. ISOLATION: Rules whose endpoints don't connect to active regions
   - If token 15 is never visited, rules involving 15 decay

2. ASYMMETRY: One-way rules without reinforcement
   - a->b survives better if b->a also exists (mutual reinforcement)
   - Isolated one-way rules decay

3. COMPETITION: Rules compete for amplitude
   - Strongly-reinforced rules "starve out" weakly-reinforced ones
   - First-mover advantage: early-activated rules stay strong

4. PHASE CANCELLATION: Destructive interference
   - If multiple paths lead to same rule with opposite phases, it cancels
   - Rules in "phase-matched" cycles survive; others cancel

5. RANDOM FLUCTUATION: Sometimes just bad luck
   - Initial conditions matter
   - Same rule might survive in one trial, die in another


THE EPHEMERAL FUNCTOR TYPES:
===========================
Rules that tend to die:
  - ISOLATED FLOWS: a->b where a and b aren't connected to much else
  - LONG CHAINS: a->b->c->d... where middle links can break
  - WEAK BRIDGES: connections between sparse regions

Rules that tend to survive:
  - SYMMETRIC PAIRS: mutual reinforcement
  - SHORT CYCLES: 2-cycles and 3-cycles are very stable
  - HUB CONNECTIONS: anything connected to high-degree nodes
""")


# ============================================================
# PART 5: CAN WE REACH N TOKENS?
# ============================================================

def analyze_token_scaling():
    """
    Can we find randomness sources that stabilize more tokens?
    What's the fundamental limit?
    """

    print("\n" + "=" * 80)
    print("SCALING ANALYSIS: Can We Reach N Tokens?")
    print("=" * 80)

    print("""
QUESTION: Why did thermal noise give us 8 tokens instead of 7?

ANSWER: It's not about "unlocking" tokens, it's about STATISTICS.

The code does:
    token = int(random_sample * n)

For uniform random in [0,1):
    - Tokens 0-6 are equally likely
    - Token 7 would require sample >= 7/7 = 1.0, which never happens

For thermal (Boltzmann) distribution:
    - Has heavier tails
    - Occasionally produces values that, when scaled, map to higher indices
    - The bug/feature: int(sample * n) can exceed n-1 for some distributions

THIS IS A CODE ARTIFACT, not a fundamental physics result.
""")

    # Let's test what different distributions actually produce
    print("\nTesting token index distributions for n=7:")
    print("-" * 50)

    def sample_uniform(n):
        return int(random.random() * n)

    def sample_thermal(n, kT=0.3):
        u1 = max(0.001, min(0.999, random.random()))
        u2 = random.random()
        val = abs(math.sqrt(-2 * kT * math.log(u1)) * math.cos(2 * math.pi * u2))
        return int(min(1.0, val) * n)

    def sample_thermal_unbounded(n, kT=0.3):
        # Don't clamp - let it overflow
        u1 = max(0.001, random.random())
        u2 = random.random()
        val = abs(math.sqrt(-2 * kT * math.log(u1)) * math.cos(2 * math.pi * u2))
        return int(val * n)  # Can exceed n-1!

    n = 7
    n_samples = 10000

    for name, sampler in [("Uniform", sample_uniform),
                          ("Thermal (clamped)", sample_thermal),
                          ("Thermal (unbounded)", sample_thermal_unbounded)]:
        random.seed(42)
        counts = Counter(sampler(n) for _ in range(n_samples))
        max_token = max(counts.keys())
        tokens_used = len(counts)

        print(f"\n{name}:")
        print(f"  Max token index: {max_token}")
        print(f"  Distinct tokens: {tokens_used}")
        if max_token >= n:
            overflow = sum(c for t, c in counts.items() if t >= n)
            print(f"  Overflow (>= {n}): {overflow} ({overflow/n_samples*100:.2f}%)")

    print("""

FUNDAMENTAL LIMITS:
==================

Q: Is there a fundamental limit to how many tokens can be stable?

A: YES, but it's not about randomness. It's about:

1. CONNECTIVITY REQUIREMENTS
   For n tokens to all be immortal, they need enough rules connecting them.
   Random initialization with k initial rules leaves some tokens isolated
   if k < n (roughly).

2. AMPLITUDE DILUTION
   Total amplitude is conserved (modulo damping).
   With more tokens/rules, amplitude spreads thinner.
   Weak rules fall below threshold and die.

3. SYMMETRY BREAKING
   In large systems, random initialization breaks symmetry.
   Some tokens get strong early; others starve.
   This is spontaneous symmetry breaking.

4. COMPUTATIONAL LIMITS
   The sieve evolution takes time O(states x rules x steps).
   Large systems converge slower or not at all.


REACHING N TOKENS:
=================

To get N stable tokens, you need:

1. INITIALIZE ALL N: Explicitly inject states for all N tokens
2. SUFFICIENT RULES: At least ~N rules to keep everyone connected
3. ENOUGH TIME: Let the system equilibrate
4. FAVORABLE INITIALIZATION: Symmetric or balanced starting conditions

The randomness source doesn't fundamentally limit N.
It's the INITIALIZATION and DYNAMICS that matter.

Example: To guarantee 100 stable tokens:
  - Inject all 100 tokens with equal amplitude
  - Inject a spanning set of rules (at least 99, forming a connected graph)
  - Evolve until stable
  - Result: potentially all 100 can survive

The "42 immortals" result is specific to:
  - n=7 tokens (hard-coded in the test)
  - Sparse random initialization
  - Specific evolution parameters

It's not a universal constant; it's 7x6 = nx(n-1) for the chosen n.
""")

    # Demonstrate that we CAN get more tokens if we initialize properly
    print("\n" + "-" * 50)
    print("DEMONSTRATION: Explicit N-token initialization")
    print("-" * 50)

    for n in [10, 15, 20]:
        random.seed(42)
        substrate = SelfOrganizingSubstrate()

        # Initialize ALL n tokens explicitly
        for t in range(n):
            substrate.inject_state(t, 1.0)

        # Initialize a connected set of rules (cycle through all)
        for t in range(n):
            substrate.inject_rule(t, (t+1) % n, 0.5)  # Forward cycle
            substrate.inject_rule((t+1) % n, t, 0.5)  # Backward cycle

        # Add some random rules
        for _ in range(n):
            a, b = random.sample(range(n), 2)
            substrate.inject_rule(a, b, random.uniform(0.3, 0.7))

        # Evolve
        for _ in range(1000):
            substrate.step()

        # Count survivors
        surviving_rules = set()
        surviving_tokens = set()
        for entity, amplitude in substrate.field:
            if isinstance(entity, Entity):
                if entity.entity_type == EntityType.RULE and abs(amplitude) > 0.1:
                    src = entity.content[0][0] if isinstance(entity.content[0], tuple) else entity.content[0]
                    tgt = entity.content[1][0] if isinstance(entity.content[1], tuple) else entity.content[1]
                    surviving_rules.add((src, tgt))
                    surviving_tokens.add(src)
                    surviving_tokens.add(tgt)

        max_rules = n * (n - 1)
        print(f"\nn={n}: {len(surviving_rules)}/{max_rules} rules survive "
              f"({len(surviving_rules)/max_rules*100:.1f}%), "
              f"{len(surviving_tokens)}/{n} tokens active")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    show_immortal_patterns()
    rigor_summary()
    analyze_8th_token()
    analyze_ephemeral_rules()
    analyze_token_scaling()
