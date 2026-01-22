# COHERENCE REPORT - Locus Project
## Date: 2026-01-11
## Reviewer: Claude (current session)

---

## PROJECT OVERVIEW

This project has TWO DISTINCT WORKSTREAMS:

### Workstream A: TicTacToe Classification (Static Board → Label)
- **Problem**: Given a board state string, predict label (ok/win1/win2/draw/error)
- **Best**: HybridLearner (88% accuracy, discovers all 8 win conditions)
- **Files**: tictactoe.py, test_harness.py, game_oracle.py, few_shot_algs/*

### Workstream B: MiniGrid Navigation (Sequential Actions → Goal)
- **Problem**: Navigate agent through grid environments using only observations
- **Best**: FairLearnerV3 (94% Empty-8x8, 81% DoorKey-6x6)
- **Files**: fair_learner_v2.py, plus inline code from this session

**KEY INSIGHT**: These are DIFFERENT problems! Workstream A is classification, 
Workstream B is sequential decision-making with partial observability.

---

## CURRENT BEST APPROACHES

### For TicTacToe (Workstream A): HybridLearner
- Pattern extraction (k=3 position combos)
- Pure rule discovery (patterns that always predict same label)
- Phase-appropriate strategies
- **88% overall, discovers all 8 win conditions**

### For MiniGrid (Workstream B): FairLearnerV3
- Target-based exploration (cycle through all seen tokens)
- 2-cycle detection for rotation actions  
- Interaction discovery for pickup/toggle
- Goal learning from wins
- **94% Empty-8x8, 81% DoorKey-6x6**

---

## FILE REVIEW

### 1. README.md (4.5K)
**Status:** REVIEWED ✓
**Domain:** Workstream A (TicTacToe)
**Summary:** Documents HybridLearner as best (88%), shows discovered win patterns
**Fairness:** ✓ CLEAN - explicitly states "no cheating policy"
**Integration:** N/A - different problem domain from current work

---

### 2. fair_learner_v2.py (143 lines)
**Status:** REVIEWED ✓
**Domain:** Workstream B (MiniGrid)
**Summary:** Current session's work - target-based exploration
**Fairness:** ✓ CLEAN - no domain knowledge
**Integration:** 
- MISSING: FairLearnerV3 with interaction discovery not saved as file!
- ACTION NEEDED: Save FairLearnerV3 to file

---

### 3. comprehensive_test.py (237 lines)
**Status:** REVIEWED ✓
**Domain:** Workstream A (TicTacToe)
**Summary:** Test suite with unique observations, random rule variants
**Fairness:** ✓ CLEAN - tests generalization to novel rules
**Integration:** Good testing pattern - could adapt for MiniGrid

---

### 4. game_oracle.py (313 lines)
**Status:** REVIEWED ✓
**Domain:** Workstream A (TicTacToe)
**Summary:** Oracle framework with state enumeration, unique observation generator
**Fairness:** ✓ CLEAN - oracle is separate from learner
**Integration:** N/A

---

### 5. test_harness.py (171 lines)
**Status:** REVIEWED ✓
**Domain:** Workstream A (TicTacToe)
**Summary:** Evaluation framework with metrics
**Fairness:** ✓ CLEAN
**Integration:** N/A

---

### 6. tictactoe.py (108 lines)
**Status:** REVIEWED ✓
**Domain:** Workstream A (TicTacToe)
**Summary:** Oracle functions, board generation, variant rules
**Fairness:** ✓ CLEAN - this is the ground truth, not the learner
**Integration:** N/A

---

### 7. token_learner.py (320 lines)
**Status:** REVIEWED ✓
**Domain:** GENERAL - could apply to both
**Summary:** Token-based state transition learner framework
**Fairness:** ✓ CLEAN - generic token system
**Integration:** 
- ⚠️ VALUABLE: This is a general framework for learning transitions!
- Uses Observation (frozenset of tokens), Transition (before, action, after)
- Learns pattern → effect rules from transitions
- **COULD BE USEFUL** for MiniGrid work - provides clean abstraction

---

### 8. variant_tests.py (280 lines)
**Status:** REVIEWED ✓
**Domain:** Workstream A (TicTacToe)
**Summary:** Tests learners on novel game variants to detect cheating
**Fairness:** ✓ CLEAN - explicitly designed to catch cheating
**Integration:** Good methodology - apply to MiniGrid testing

---

## few_shot_algs/ Directory

### 9. __init__.py (27 lines)
**Status:** REVIEWED ✓
**Summary:** Base Algorithm interface with predict/update_history
**Fairness:** ✓ CLEAN

### 10. hybrid_learner.py (382 lines) ⭐ BEST FOR TICTACTOE
**Status:** REVIEWED ✓
**Domain:** Workstream A
**Summary:** Best performer (88%) - phase-appropriate strategies, pure rule discovery
**Fairness:** ✓ CLEAN - no hardcoded patterns
**Integration:** 
- ⚠️ VALUABLE CONCEPT: "Pure rules" (patterns that ALWAYS predict same label)
- Could apply to MiniGrid: cycles that ALWAYS return to start = reliable invariants

### 11. blind_learner.py (257 lines)
**Status:** REVIEWED ✓
**Domain:** Workstream A
**Summary:** Baseline blind learner (79%), hypothesis generation + elimination
**Fairness:** ✓ CLEAN - explicitly designed to be blind
**Integration:** Pattern generation approach could inform exploration

### 12. blind_transition.py (354 lines)
**Status:** REVIEWED ✓
**Domain:** GENERAL - transitions!
**Summary:** Learns state transitions from opaque observations
**Fairness:** ✓ CLEAN - explicitly "truly blind"
**Integration:**
- ⚠️ VALUABLE: Tracks change patterns between states
- Generalizes patterns (ignoring specific positions)
- **DIRECTLY APPLICABLE** to MiniGrid - this is what we're doing!
- Key insight: "same generalized pattern in MANY contexts = true rule"

### 13-34. (Other few_shot_algs files)
**Status:** TO REVIEW IN DETAIL - likely most are Workstream A focused

Key files to check:
- production.py (634 lines) - Production rules LHS => RHS
- sat_production.py (663 lines) - SAT-like inference
- transition_rules.py (630 lines) - Transition learning

---

## INTEGRATION NOTES

### MUST DO:
1. **Save FairLearnerV3** - current best MiniGrid agent not saved to file!

### SHOULD INTEGRATE:
1. **token_learner.py concepts** - Clean Observation/Transition abstraction
   - Current MiniGrid work uses ad-hoc tuples, could use this framework
   
2. **blind_transition.py concepts** - Change pattern generalization
   - "If same pattern in many contexts = true rule"
   - We're doing this with cycle detection, but less formally

3. **hybrid_learner.py "pure rules"** concept
   - Patterns that NEVER fail are reliable
   - Apply to: cycles that always return to start

### CONSIDER:
1. **production.py** - LHS => RHS rules with catalysts
   - Could model: (front=DOOR, has_key) + TOGGLE => front=OPEN
   - More expressive than current ad-hoc approach

---

## FAIRNESS FLAGS

### ✓ ALL CLEAN - No cheating detected

The project has a strong "no cheating" culture:
- README explicitly states policy
- variant_tests.py designed to catch cheating
- blind_* files explicitly avoid domain knowledge
- fair_learner_v2.py learns from experience only

### Potential Concerns (MINOR):
1. **fair_learner_v2.py line 77**: `if won and action == Action.A2:`
   - Hardcodes A2 as the "forward/move" action
   - Should learn this from 2-cycle detection instead
   - **FIX**: Use learned rotation actions to infer movement action

2. **Observations include "G" for goal state** (line 110)
   - This is environment feedback, not domain knowledge
   - ✓ ACCEPTABLE - equivalent to "reward signal"

---

## SUMMARY

The project is COHERENT but has TWO DISTINCT PROBLEM DOMAINS:
- **TicTacToe**: Well-developed, HybridLearner at 88%
- **MiniGrid**: Current session work, FairLearnerV3 at 81-94%

**ACTION ITEMS:**
1. Save FairLearnerV3 to file
2. Consider integrating token_learner.py abstractions
3. Apply blind_transition.py pattern generalization concepts
4. Review remaining few_shot_algs files for transferable ideas

---
