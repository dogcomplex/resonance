# LOCUS Proposals - Architectural Design Changes

This file documents proposed design changes for discussion before implementation.

---

## Proposal 1: Goal Conditions as Layer 1/2 Concern

**Date**: 2026-01-22
**Status**: In Progress

### Current State

`chess_targeted.py` implements material filtering as a **separate game class**:
- `TargetedChessGame` overrides `is_boundary()` and `get_successors()`
- Material filtering is baked into the game logic
- Works, but conflates Layer 0 (game rules) with Layer 1/2 (strategy)

### Proposed Architecture

Split into proper layers:

#### Layer 0 (chess.py) - Game Capabilities

Add **capabilities** without decision-making:

```python
class ChessGame:
    def get_material_signature(self, state) -> str:
        """Returns 'KQRRvKQR' format string"""

    def captures_to_material(self, state, move) -> Optional[str]:
        """If move is capture, what material results? None if not capture."""

    def enumerate_syzygy_positions(self, material: str, count: int) -> List[ChessState]:
        """Generate positions with given material from syzygy"""

    def get_valid_parent_materials(self, target_material: str) -> List[str]:
        """What materials can capture down to target? (8-piece variants)"""
```

These are **pure game logic** - no decisions about what to search.

#### Layer 1 (Seed Selection) - Goal Conditions

Add **goal condition** concept to seed selection:

```python
@dataclass
class GoalCondition:
    """Defines what counts as 'success' for a search"""
    target_materials: Set[str]      # Materials that count as boundary
    source_materials: Set[str]      # Materials to seed from
    early_terminate: bool = True    # Stop on wrong-material captures?

class SeedSelector:
    def select_seeds_for_goal(self, goal: GoalCondition, game: ChessGame):
        """Select seeds targeting a specific goal"""
        backward_seeds = game.enumerate_syzygy_positions(goal.target_materials)
        forward_seeds = [generate_source_positions(m) for m in goal.source_materials]
        return forward_seeds, backward_seeds
```

#### Layer 2 (Strategy) - Goal Selection

Decide **which goals** to pursue:

```python
class MetaStrategy:
    def select_next_goal(self, progress: Dict[str, float]) -> GoalCondition:
        """Choose which material transition to focus on next"""
        # Could prioritize:
        # - Materials with high uncertainty
        # - Materials that unlock other goals
        # - Materials with fastest convergence
```

### Benefits

1. **Composability**: Can combine goals (e.g., "find paths to KQRRvKQR OR KQRRvKQRR")
2. **Reusability**: Same goal conditions work for any game implementing the interface
3. **Clean Separation**: Layer 0 doesn't know about strategy, Layer 2 doesn't know about game rules
4. **Reproducibility**: Goals are data, can be saved/loaded/shared

### Implementation Plan

1. Add material utility functions to `chess.py`
2. Create `GoalCondition` dataclass in `core.py`
3. Add `filter_by_goal()` method to `HOLOSSolver`
4. Modify seed selection to accept goal conditions
5. Create meta-strategy hooks for Layer 2

### Questions for Discussion

1. Should `GoalCondition` be game-agnostic (in core.py) or game-specific?
   - Leaning: Game-agnostic with game-specific implementations

2. Should early termination be solver behavior or game behavior?
   - Leaning: Solver behavior (game just reports what would happen)

3. How do we handle composite goals (A OR B)?
   - Leaning: GoalCondition can have multiple target materials

---

## Proposal 2: Tactics (L1) vs Strategy (L2) Distinction

**Date**: 2026-01-22
**Status**: Discussion

### Current Understanding

- **Layer 1 (Tactics)**: "How do I execute this search efficiently?"
  - Seed selection within a known goal
  - Mode selection (lightning/wave/crystal)
  - Depth allocation per seed

- **Layer 2 (Strategy)**: "What should I be searching for?"
  - Goal selection (which material transitions to target)
  - Budget allocation across goals
  - Progress monitoring and goal adjustment

### Key Distinction

Layer 1 optimizes **efficiency** (coverage/cost) for a given goal.
Layer 2 optimizes **effectiveness** (progress toward ultimate objective).

### Implications

The material targeting we just built is **Layer 2** behavior:
- Deciding to focus on KQRRvKQR is a strategic choice
- Deciding HOW to search it (seeds, modes, depths) is tactical

This suggests:
- `chess_targeted.py` should become a Layer 2 strategy
- The underlying search mechanics stay in Layer 0/1
- Layer 2 passes goal conditions down to Layer 1

---
