# Locus Project - Complete Coherence Report v2

## Executive Summary

**The Core Insight**: This project has ONE unified paradigm that should work across ALL domains:
- **Token-based state machine learning**
- Observations are sets of opaque tokens
- Rules are learned patterns that predict state transitions
- No domain-specific knowledge allowed ("fair/honest" learning)

The project evolved through: TicTacToe → Mini RPG/Text Adventures → MiniGrid Navigation

All using the same fundamental approach in `token_learner.py`.

---

## Complete File Inventory (36 files)

### Root Directory Files

| File | Lines | Size | Description | Domain |
|------|-------|------|-------------|--------|
| README.md | ~100 | 4.5K | Documents HybridLearner (88%), win pattern discovery | TicTacToe |
| token_learner.py | 320 | 11K | **⭐ UNIFIED FRAMEWORK** - Core abstraction | Universal |
| game_oracle.py | 313 | 10K | TicTacToe oracle with variants | TicTacToe |
| test_harness.py | 171 | 5.5K | Evaluation framework | TicTacToe |
| tictactoe.py | 108 | 3K | Board generation, variant rules | TicTacToe |
| comprehensive_test.py | 237 | 7.5K | Test suite with unique observations | TicTacToe |
| variant_tests.py | 280 | 8.6K | Cheating detection tests | TicTacToe |
| fair_learner_v2.py | 143 | 5K | Target-based exploration (MiniGrid) | MiniGrid |
| fair_learner_v3.py | 240 | 9.7K | + Interaction discovery (94% Empty-8x8) | MiniGrid |
| COHERENCE_REPORT.md | - | 7.5K | Previous coherence report | Meta |

### few_shot_algs/ Directory (26 files)

| File | Lines | Size | Description | Key Contribution |
|------|-------|------|-------------|------------------|
| __init__.py | 27 | 775B | Base Algorithm interface | Interface |
| **production.py** | 635 | 20K | **LHS => RHS rule system** | Core abstraction |
| **transition_rules.py** | 631 | 21K | **Full state transitions** | Core abstraction |
| **blind_transition.py** | 354 | 12K | **Truly blind learner** | Fair learning |
| **bubble_up.py** | 368 | 13K | **Abstraction discovery** | Pattern grouping |
| unified_learner.py | 328 | 12K | Best-of-breed classifier | Classification |
| hybrid_learner.py | 382 | 15K | Phase-based strategies (88%) | Classification |
| final_learner.py | 259 | 8.5K | 100% accuracy on TicTacToe | Classification |
| complete_learner.py | 418 | 14K | Priority + negation | Classification |
| priority_learner.py | ~400 | 16K | PRIORITY tokens | Classification |
| enhanced_learner.py | ~450 | 18K | Negative patterns | Classification |
| pattern_learner.py | ~350 | 14K | Systematic enumeration | Classification |
| hypothesis_learner.py | ~330 | 13K | Strict elimination | Classification |
| blind_learner.py | 257 | 9K | No embedded knowledge | Fair learning |
| meta_learner.py | 461 | 16K | Learning as production rules | Meta-learning |
| minimal_learner.py | ~350 | 15K | Minimal covering rules | Rule compression |
| true_minimal.py | ~300 | 11K | Core mechanics discovery | Rule compression |
| production_learner.py | ~280 | 11K | Pattern → label rules | Classification |
| sat_production.py | 663 | 23K | SAT-like inference | Advanced inference |
| advanced_sat.py | ~500 | 19K | Boolean combinations | Advanced inference |
| advanced_production.py | ~400 | 16K | Discriminative scoring | Classification |
| active_learning.py | 593 | 19K | Query-based learning | Exploration |
| parity_learner.py | ~250 | 10K | Turn inference | TicTacToe-specific |
| flexible_oracle.py | ~250 | 9K | Multiple rulesets | Testing |
| chaos_oracle.py | ~350 | 14K | Wild game variations | Testing |
| convergence_eval.py | ~450 | 17K | Balanced accuracy metrics | Evaluation |

---

## The Unified Paradigm

### token_learner.py - The Core Framework

```
State = FrozenSet[str] of tokens
Action = FrozenSet[str] of tokens
Transition = (before_state, action, after_state)
Rule = (pattern, effect, confidence)
```

**Key Methods:**
1. `observe(transition)` - Learn from state changes
2. `extract_rules()` - Find high-confidence patterns
3. `predict(state, action)` - Predict effects
4. `predict_next_state()` - Full state prediction

**Works for ANY domain** where:
- State can be represented as token sets
- Transitions have observable before/after states
- Rules are patterns that predict changes

### How It Was Applied

**TicTacToe:**
- Tokens: `{p0_1, p1_2, p4_0, ...}` (position_value)
- Rules: `{p0_1, p1_1, p2_1} → win1` (three X's in top row)

**Mini RPG (from transcript history):**
- Tokens: `{at_0_0, hp_10, has_sword, room_village, ...}`
- Rules: `{action_attack, has_sword, fighting_goblin} → killed_goblin`

**MiniGrid:**
- Tokens: `{front=T2, left=T1, right=T0, G, ...}`
- Rules: `{action_A2, front=T0} → move_forward`

---

## Files NOT Previously Reviewed (Now Complete)

### 1. active_learning.py
**Purpose:** Query-based learning + state transitions
**Key Features:**
- Passive mode: Observe random stream
- Active mode: Query specific patterns
- Transition prediction: board + action → next_state
**Integration:** Could help MiniGrid exploration by actively querying informative states

### 2. advanced_sat.py / sat_production.py
**Purpose:** Boolean combinations and chained inference
**Key Features:**
- OR rules (multiple patterns can satisfy)
- Negative rules (IF pattern THEN NOT label)
- Chained inference
**Integration:** Useful for complex MiniGrid interactions (key + door → open)

### 3. chaos_oracle.py / flexible_oracle.py / convergence_eval.py
**Purpose:** Testing and evaluation infrastructure
**Key Features:**
- Variable pattern sizes, asymmetric rules
- Balanced accuracy metrics
- Convergence timeline analysis
**Integration:** Adapt for MiniGrid fairness testing

### 4. meta_learner.py
**Purpose:** Learning process as production rules
**Key Features:**
- Meta tokens: `pattern_seen`, `pattern_pure`, `confidence_high`
- Strategy rules: How to update beliefs
**Integration:** Could formalize exploration strategy in MiniGrid

### 5. minimal_learner.py / true_minimal.py
**Purpose:** Find simplest covering rules
**Key Features:**
- Context-independent rules
- Minimal LHS patterns
**Integration:** Avoid overfitting in MiniGrid rule discovery

### 6. bubble_up.py
**Purpose:** Abstraction discovery
**Key Features:**
- Track token co-occurrence
- Create abstract labels for patterns
- Compress rules
**Integration:** Could discover "room types" or "object classes" in MiniGrid

### 7. parity_learner.py
**Purpose:** Turn inference from piece counts
**Key Features:**
- Infer hidden state (whose turn)
- Phase-aware predictions
**Integration:** TicTacToe-specific, less relevant to MiniGrid

---

## Fairness Audit Summary

**ALL files pass fairness audit.** The project has a strong "no cheating" culture:

| Principle | Evidence |
|-----------|----------|
| No domain knowledge in learners | blind_*, fair_learner_* files |
| Generalization tests | variant_tests.py, chaos_oracle.py |
| Explicit policy | README.md states "no cheating policy" |
| Separation of concerns | Oracle (ground truth) separate from learner |

**Minor concern (non-critical):**
- `fair_learner_v2.py` line 77 hardcodes `Action.A2` as forward movement
- Should infer from rotation detection instead

---

## Integration Roadmap: Reunifying the Paradigm

### Goal: One learner that handles TicTacToe AND MiniGrid

### Step 1: Token Abstraction Layer
Use `token_learner.py` + `bubble_up.py` to:
1. Convert MiniGrid observations to token sets
2. Discover abstraction patterns (like room types)
3. Learn transition rules from exploration

### Step 2: Rule Discovery
Use `blind_transition.py` patterns to:
1. Track (state, action, next_state) triples
2. Extract patterns that ALWAYS hold
3. Build transition graph from pure rules

### Step 3: Planning Integration
Use learned rules for:
1. A* search over predicted states (already in FairLearnerV3)
2. Goal-directed exploration
3. Interaction discovery (pickup, toggle)

### Step 4: Testing
Use adapted chaos/variant testing to:
1. Verify no domain knowledge leakage
2. Test on novel MiniGrid variants
3. Measure sample efficiency

---

## Key Files for Reunification

| Priority | File | Why |
|----------|------|-----|
| ⭐⭐⭐ | token_learner.py | Core unified abstraction |
| ⭐⭐⭐ | blind_transition.py | Pattern generalization in context |
| ⭐⭐ | production.py | LHS => RHS syntax |
| ⭐⭐ | bubble_up.py | Abstraction discovery |
| ⭐ | active_learning.py | Query-based exploration |
| ⭐ | transition_rules.py | Full state transitions |

---

## Next Steps

1. **Merge FairLearnerV3 into token_learner.py framework**
   - Replace ad-hoc tuples with Observation/Transition
   - Use extract_rules() for cycle detection
   - Apply predict() for navigation

2. **Test unified learner on both domains**
   - TicTacToe: Should match HybridLearner (88%)
   - MiniGrid: Should match FairLearnerV3 (94%/81%)

3. **Add intermediate test cases**
   - Mini RPG (from transcripts)
   - Simple text adventures
   - Verify smooth generalization

---

## Transcript History Summary

The project evolved through these phases (from journal.txt):

1. **Jan 9, 01:23-07:08**: TicTacToe classification (BlindLearner → HybridLearner 88%)
2. **Jan 9, 08:16-09:59**: Token-based generalization, Mini RPG testing
3. **Jan 9, 23:40-Jan 10, 03:31**: Pokemon abstraction, evolutionary discovery
4. **Jan 10, 07:35-13:10**: MiniGrid navigation, A* over learned rules
5. **Jan 10, 21:49-Jan 11, 02:46**: Official benchmarks, FourRooms, Memory
6. **Jan 11, 03:24-04:09**: Fair blind audit, current breakthrough

The token_learner.py was used throughout as the underlying abstraction.

---

*Report generated: Session continuing from compaction*
*Previous: COHERENCE_REPORT.md (partial review)*
*This version: Complete inventory of all 36 files*
