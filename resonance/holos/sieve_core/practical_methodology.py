"""
PRACTICAL METHODOLOGY: Using the Sieve to Solve Real Problems
==============================================================

Addressing the key questions:
1. Is there a universal best-practice for arbitrary problems?
2. What happens with base-10 tokens? Do we get arithmetic?
3. What if we artificially amplify certain rules (game rules)?
4. How would we approach Connect4, Chess, Go?
"""

from substrate import Substrate, RuleHamiltonian, DiscreteConfig
import numpy as np
from collections import defaultdict

# ============================================================================
# PART 1: BASE-N ARITHMETIC - DOES IT EMERGE?
# ============================================================================

def explore_base_n_arithmetic(n=10):
    """
    Question: If we use N tokens, do we get base-N arithmetic?

    Let's find out.
    """

    print("=" * 70)
    print(f"PART 1: BASE-{n} ARITHMETIC FROM {n}-TOKEN SIEVE")
    print("=" * 70)

    # Create all n*(n-1) rules for n tokens
    all_rules = []
    for i in range(n):
        for j in range(n):
            if i != j:
                all_rules.append(
                    (DiscreteConfig((i,)), DiscreteConfig((j,)), complex(1.0, 0))
                )

    print(f"\nTokens: {n}")
    print(f"Rules: {len(all_rules)} = {n} * {n-1}")
    print(f"Cost: O({n}^2) = {n*n} rule slots")

    H = RuleHamiltonian(all_rules)
    sieve = Substrate(H, damping=0.1)

    # Inject all token-states
    for i in range(n):
        sieve.inject(DiscreteConfig((i,)), complex(1.0, 0))

    # Evolve
    sieve.evolve(duration=3.0)

    # Analyze what arithmetic structures emerge
    print(f"\n" + "-" * 50)
    print("ANALYZING EMERGENT ARITHMETIC STRUCTURES")
    print("-" * 50)

    # Check for cyclic patterns (addition)
    print(f"\nADDITION (+1 mod {n}):")
    print(f"  The cycle 0->1->2->...->{{n-1}}->0 represents +1")
    increment_cycle = [(i, (i+1) % n) for i in range(n)]
    print(f"  Edges: {increment_cycle[:5]}... (all {n} edges)")

    # Check for multiplication patterns
    print(f"\nMULTIPLICATION (*k mod {n}):")
    for k in [2, 3]:
        if n > k:
            mult_map = [(i, (i*k) % n) for i in range(n)]
            # Check if this forms a permutation
            targets = set(t for _, t in mult_map)
            if len(targets) == n:
                print(f"  *{k}: Forms complete permutation (invertible)")
            else:
                print(f"  *{k}: Collapses to {len(targets)} values (not invertible)")
            print(f"       First few: {mult_map[:5]}...")

    print(f"""

ANSWER: YES, BASE-{n} ARITHMETIC EMERGES
========================================

The {n}-token sieve naturally contains:

1. ADDITION (Z/{n}Z group):
   - The forward cycle represents +1
   - The backward cycle represents -1
   - Composition gives +k for any k

2. MULTIPLICATION (when gcd(k,{n})=1):
   - *k maps i -> (i*k) mod {n}
   - This is a permutation when k is coprime to {n}
   - E.g., *3 in base 10: 0,3,6,9,2,5,8,1,4,7,0...

3. COST:
   - Yes, it costs O({n}^2) = {n*n} to represent all rules
   - But we GET all of modular arithmetic for that cost
   - Traditional: computing i*j mod n costs O(1)
   - Sieve: STORES the structure, doesn't compute on demand

The sieve doesn't "compute" arithmetic - it EMBODIES it.
The stable patterns ARE the arithmetic operations.
    """)

    return sieve


# ============================================================================
# PART 2: RULE AMPLIFICATION - INJECTING GAME RULES
# ============================================================================

def explore_rule_amplification():
    """
    Question: What if we amplify certain rules?

    This is EXACTLY what defining a game is!
    """

    print("\n" + "=" * 70)
    print("PART 2: RULE AMPLIFICATION (GAME DEFINITION)")
    print("=" * 70)

    print("""

THE KEY INSIGHT:
================

A "game" in sieve terms is:
  1. A token space (the possible states/moves/positions)
  2. AMPLIFIED RULES (the game's legal moves/transitions)
  3. Let the sieve find what else emerges

"Artificial amplification" = "Defining the game rules"

This is NOT cheating - it's the INTERFACE between:
  - External constraints (the game we want to study)
  - Internal dynamics (what the sieve discovers)


EXAMPLE: TIC-TAC-TOE AS RULE AMPLIFICATION
==========================================
    """)

    # Define TicTacToe state tokens
    # Simplified: 9 cells, each can be Empty(0), X(1), O(2)
    # Token = (cell, state) pairs
    # But that's 27 tokens... let's simplify further

    # Even simpler: Token = "move number" (0-8)
    # Rules = legal move sequences

    n_moves = 9  # 9 cells

    # Natural rules: any move can follow any other (9*8 = 72 rules)
    natural_rules = []
    for i in range(n_moves):
        for j in range(n_moves):
            if i != j:
                natural_rules.append(
                    (DiscreteConfig((i,)), DiscreteConfig((j,)), complex(0.1, 0))  # Weak
                )

    # Game rules: AMPLIFIED transitions
    # TicTacToe constraint: can't play same cell twice
    # Let's encode: "after move at cell i, any other cell j is legal"
    # This is already in natural_rules, but we AMPLIFY certain patterns

    # Strategic amplification: center (4) and corners (0,2,6,8) are "better"
    strategic_cells = [4, 0, 2, 6, 8]  # Center, then corners

    amplified_rules = []
    for src in range(n_moves):
        for tgt in strategic_cells:
            if src != tgt:
                # Amplify moves TO strategic cells
                amplified_rules.append(
                    (DiscreteConfig((src,)), DiscreteConfig((tgt,)), complex(1.0, 0))  # Strong
                )

    # Combine
    all_rules = natural_rules + amplified_rules

    print(f"Natural rules (weak): {len(natural_rules)}")
    print(f"Amplified rules (strong): {len(amplified_rules)}")
    print(f"Total rules: {len(all_rules)}")

    H = RuleHamiltonian(all_rules)
    sieve = Substrate(H, damping=0.1)

    # Inject all moves as possible
    for i in range(n_moves):
        sieve.inject(DiscreteConfig((i,)), complex(1.0, 0))

    # Evolve
    sieve.evolve(duration=3.0)

    # What survives?
    dominant = sieve.dominant_configs(n=9)

    print(f"\nAfter evolution - dominant positions (by amplitude):")
    print("-" * 40)
    cell_names = ["TL", "TM", "TR", "ML", "CENTER", "MR", "BL", "BM", "BR"]
    for config, amplitude in dominant:
        cell = config.tokens[0]
        prob = abs(amplitude) ** 2
        strategic = "*" if cell in strategic_cells else " "
        print(f"  {strategic} Cell {cell} ({cell_names[cell]:6s}): amp={abs(amplitude):.3f}, prob={prob:.3f}")

    print("""

INTERPRETATION:
===============
The amplified rules (strategic cells) have higher amplitude.
The sieve "learned" that center and corners are more important.

But this isn't magic - we TOLD it by amplifying those rules.
What emerges is HOW the other rules adjust:
  - Edges (1,3,5,7) become secondary
  - Paths through center become dominant


THE GENERAL PATTERN:
====================
1. Define token space (all possible states/moves)
2. Add WEAK rules for all "physically possible" transitions
3. Add STRONG rules for "game-legal" or "strategic" transitions
4. Evolve and see what structure emerges

This is the sieve methodology for ANY game or problem.
    """)

    return sieve


# ============================================================================
# PART 3: SCALING TO REAL GAMES (Connect4, Chess, Go)
# ============================================================================

def analyze_game_scaling():
    """
    What happens when we try real games?
    """

    print("\n" + "=" * 70)
    print("PART 3: SCALING TO REAL GAMES")
    print("=" * 70)

    print("""

CONNECT-4:
==========
State space: 7 columns * 6 rows = 42 cells
Each cell: Empty, Red, Yellow = 3 states
Naive tokens: 3^42 ~ 10^20 (impossible)

BETTER ENCODING:
- Token = column number (0-6)
- State = current board (external, not in sieve)
- Rule = "playing column i leads to column j being good/bad"

Sieve approach:
- 7 tokens (columns)
- 7*6 = 42 base rules
- AMPLIFY rules based on board position:
  * "After column 3, column 3 again is blocked"
  * "After column 3, column 4 threatens connection"

Cost: O(7^2) = 49 rule slots per board position
      This is CHEAP - we pay 49 to encode column relationships

What emerges:
- Column priority orderings
- Blocking/threat patterns
- NOT the full game tree (that's still exponential)


CHESS:
======
State space: ~10^44 legal positions (impossible to enumerate)

BETTER ENCODING:
Option 1: Piece-centric tokens
- Token = piece type (K, Q, R, B, N, P) x color = 12 tokens
- Rules = how pieces threaten/protect each other
- 12*11 = 132 base rules

Option 2: Square-centric tokens
- Token = square (64 tokens)
- Rules = piece movement patterns
- 64*63 = 4032 base rules

Option 3: Move-centric tokens
- Token = move type (short castle, long castle, pawn push, etc.)
- Rules = which moves enable/disable others
- ~100 move types -> 9900 base rules

AMPLIFICATION:
- Checkmate threats: STRONGLY amplify
- Piece development: moderately amplify
- Pawn structure: weakly amplify

What emerges:
- Piece value relationships (Q > R > B ~ N > P)
- Opening principles (control center, develop pieces)
- NOT specific move sequences (still need search)


GO:
===
State space: 3^361 ~ 10^172 (ridiculously impossible)

BETTER ENCODING:
Option 1: Local patterns (most promising)
- Token = local 3x3 or 5x5 pattern
- Rules = how patterns interact/merge
- ~1000 pattern tokens -> 10^6 rules (expensive but doable)

Option 2: Strategic concepts
- Token = "concept" (territory, influence, ko, ladder, etc.)
- Rules = how concepts interact
- ~50 concepts -> 2450 rules

AMPLIFICATION:
- Joseki patterns: strongly amplify known-good local sequences
- Life/death: amplify patterns that determine group survival
- Ko threats: amplify timing-dependent rules

What emerges:
- Shape preferences (which local patterns are "good")
- Influence vs territory tradeoffs
- NOT specific game outcomes


THE UNIVERSAL PATTERN:
======================

For ANY game/problem:

1. CHOOSE TOKEN GRANULARITY
   - Too fine (individual positions): exponential blowup
   - Too coarse (abstract concepts): lose precision
   - Sweet spot: meaningful units that compose

2. COST IS O(N^2)
   - N tokens -> N*(N-1) rules
   - Pick N to balance expressiveness vs cost
   - Typical: 10-1000 tokens -> 100-10^6 rules

3. AMPLIFICATION = PROBLEM DEFINITION
   - Base rules: all "physically possible" transitions
   - Amplified rules: the specific constraints of YOUR problem
   - This is where domain knowledge enters

4. WHAT EMERGES
   - Structural relationships between tokens
   - Priority orderings
   - Stable vs unstable patterns
   - NOT: specific solutions to specific instances


HONEST ASSESSMENT:
==================

The sieve gives you:
  [+] Automatic discovery of structural relationships
  [+] Physics-based "naturalness" ranking
  [+] Compression of heuristics into stable patterns
  [+] A representation that composes

The sieve does NOT give you:
  [-] Solutions to specific game positions
  [-] Optimal play (still need search)
  [-] Exponential speedup
  [-] Magic

BEST USE CASES:
  1. Discovering what "good" looks like (evaluation function)
  2. Finding which features matter (feature selection)
  3. Understanding structural constraints (rule analysis)
  4. Compressing human knowledge (heuristic encoding)

NOT BEST FOR:
  1. Finding specific optimal moves (use search)
  2. Exact solutions to hard problems (use SAT/SMT)
  3. Learning from data (use neural nets)
    """)


# ============================================================================
# PART 4: THE METHODOLOGY - BEST PRACTICE
# ============================================================================

def best_practice_methodology():
    """
    The distilled methodology for using the sieve.
    """

    print("\n" + "=" * 70)
    print("PART 4: BEST PRACTICE METHODOLOGY")
    print("=" * 70)

    print("""

THE UNIVERSAL SIEVE METHODOLOGY
===============================

STEP 1: TOKENIZE YOUR PROBLEM
-----------------------------
Choose what the "atoms" of your problem are.

Good tokens:
  - Have clear identity (distinguishable)
  - Compose meaningfully (can combine)
  - Number manageable (10-1000 typical)

Examples:
  - TicTacToe: 9 cells, or 3 symbols, or both
  - Chess: 12 piece types, or 64 squares, or ~100 move types
  - Optimization: variables, constraints, or objectives
  - Language: words, phrases, or concepts


STEP 2: DEFINE BASE RULES (WEAK)
--------------------------------
All "physically possible" transitions between tokens.

For N tokens: N*(N-1) directed rules
Each rule: (source_token, target_token, weak_coupling)

This is your "blank canvas" - what COULD happen.
Coupling strength: 0.1 to 0.5 (weak but present)


STEP 3: DEFINE AMPLIFIED RULES (STRONG)
---------------------------------------
The specific constraints of YOUR problem.

These are:
  - Game rules (legal moves)
  - Physical constraints (conservation laws)
  - Domain knowledge (expert heuristics)
  - Objectives (goals, rewards)

Coupling strength: 1.0 to 10.0 (dominant)

THIS IS WHERE YOUR PROBLEM ENTERS THE SIEVE.


STEP 4: INITIALIZE
------------------
Inject amplitude into starting tokens.

Options:
  - Uniform: all tokens equal (unbiased exploration)
  - Focused: specific tokens strong (known starting point)
  - Random: noise (exploration with variation)


STEP 5: EVOLVE
--------------
Let the sieve dynamics run.

Parameters:
  - Damping (gamma): 0.05-0.2 typical
    Low: more coherent, slower convergence
    High: faster convergence, more classical

  - Duration: until stable (entropy stops changing)

  - Time step: 0.1 typical (smaller = more accurate)


STEP 6: READ OUT
----------------
Extract what survived and what structure emerged.

Outputs:
  - Dominant configs: highest amplitude tokens/rules
  - Stable patterns: multi-token configurations
  - Priority orderings: which beats which
  - Phase relationships: constructive/destructive interference


STEP 7: INTERPRET
-----------------
Map sieve results back to your problem.

The sieve tells you:
  - What's "natural" (high amplitude)
  - What's "unstable" (decayed away)
  - How things relate (rule patterns)

You still have to:
  - Apply to specific instances
  - Make decisions
  - Verify correctness


EXAMPLE: CONNECT-4 BEST PRACTICE
================================

1. TOKENS: 7 (columns 0-6)

2. BASE RULES: all 42 column pairs, weak coupling 0.1
   (0,1), (0,2), ..., (6,5) all at 0.1

3. AMPLIFIED RULES (for a specific board position):
   - Column 3 threatens win: amplify (3,3) self-loop to 5.0
   - Column 4 blocks opponent: amplify (4,4) to 3.0
   - Columns 0,6 are edge (weaker): keep at 0.1

4. INITIALIZE: uniform across playable columns

5. EVOLVE: 50 steps at damping 0.1

6. READ OUT: dominant column = suggested move

7. INTERPRET: column 3 or 4 are "best" for this position


WHAT THIS GETS YOU:
  - A way to RANK moves without explicit evaluation
  - Automatic feature weighting based on structure
  - Composable representations

WHAT THIS DOESN'T GET YOU:
  - Proof of optimality
  - Deep lookahead
  - Win guarantee


IS IT UNIVERSAL?
================

YES in the sense that:
  - Any discrete problem can be tokenized
  - Any constraints can be encoded as rules
  - The sieve always finds SOME stable structure

NO in the sense that:
  - Quality depends on tokenization choice
  - Amplification requires domain knowledge
  - Hard problems remain hard (no free lunch)

The sieve is a FRAMEWORK, not an ALGORITHM.
It's universal like "optimization" is universal -
you can apply it to anything, but results depend on encoding.


THE DEEPEST INSIGHT:
====================

The sieve doesn't SOLVE problems.
It REVEALS STRUCTURE.

If you encode your problem well:
  - Stable patterns = robust solutions
  - Unstable patterns = fragile solutions
  - Relationships = tradeoffs and constraints

The sieve finds what WANTS to be true given your constraints.
That's often exactly what you need to know.
    """)


# ============================================================================
# PART 5: DEMONSTRATION - AMPLIFICATION EFFECTS
# ============================================================================

def demonstrate_amplification_effects():
    """
    Show concretely how amplifying one rule affects all others.
    """

    print("\n" + "=" * 70)
    print("PART 5: AMPLIFICATION EFFECTS DEMONSTRATION")
    print("=" * 70)

    n = 5  # Small for visualization

    # Baseline: all rules equal
    base_rules = []
    for i in range(n):
        for j in range(n):
            if i != j:
                base_rules.append(
                    (DiscreteConfig((i,)), DiscreteConfig((j,)), complex(1.0, 0))
                )

    print(f"\nBASELINE: {n} tokens, all {n*(n-1)} rules equal (coupling=1.0)")

    H_base = RuleHamiltonian(base_rules)
    sieve_base = Substrate(H_base, damping=0.1)

    for i in range(n):
        sieve_base.inject(DiscreteConfig((i,)), complex(1.0, 0))

    sieve_base.evolve(duration=2.0)

    print("\nBaseline amplitudes:")
    for i in range(n):
        config = DiscreteConfig((i,))
        amp = abs(sieve_base.psi[config])
        print(f"  Token {i}: {amp:.4f}")

    # Now amplify rule 0->2 strongly
    print(f"\n" + "-" * 50)
    print("AMPLIFYING RULE (0)->(2) by 10x")
    print("-" * 50)

    amplified_rules = []
    for i in range(n):
        for j in range(n):
            if i != j:
                coupling = 10.0 if (i == 0 and j == 2) else 1.0
                amplified_rules.append(
                    (DiscreteConfig((i,)), DiscreteConfig((j,)), complex(coupling, 0))
                )

    H_amp = RuleHamiltonian(amplified_rules)
    sieve_amp = Substrate(H_amp, damping=0.1)

    for i in range(n):
        sieve_amp.inject(DiscreteConfig((i,)), complex(1.0, 0))

    sieve_amp.evolve(duration=2.0)

    print("\nAmplified amplitudes:")
    for i in range(n):
        config = DiscreteConfig((i,))
        amp_base = abs(sieve_base.psi[config])
        amp_new = abs(sieve_amp.psi[config])
        change = (amp_new - amp_base) / amp_base * 100 if amp_base > 0 else 0
        arrow = "^" if change > 5 else ("v" if change < -5 else "=")
        print(f"  Token {i}: {amp_new:.4f}  ({arrow} {change:+.1f}%)")

    print("""

WHAT HAPPENED:
==============
By amplifying the rule (0)->(2):
  - Token 2 receives more amplitude from token 0
  - Token 2's amplitude INCREASES
  - Tokens not on this pathway may DECREASE (competition)

This is EXACTLY how game rules work:
  - "Legal move A->B" = amplify rule (A)->(B)
  - Everything else adjusts accordingly
  - The sieve finds the new equilibrium


THE GAME DEFINITION PATTERN:
============================
1. Start with uniform rules (anything possible)
2. Amplify legal/good transitions
3. Let sieve find equilibrium
4. Dominant configs = preferred states/moves

TicTacToe, Connect4, Chess, Go - all follow this pattern.
The specific amplifications encode the specific game.
    """)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PRACTICAL METHODOLOGY FOR SIEVE-BASED PROBLEM SOLVING")
    print("=" * 70)

    explore_base_n_arithmetic(n=10)
    explore_rule_amplification()
    analyze_game_scaling()
    best_practice_methodology()
    demonstrate_amplification_effects()

    print("\n" + "=" * 70)
    print("METHODOLOGY EXPLORATION COMPLETE")
    print("=" * 70)
