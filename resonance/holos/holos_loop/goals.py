"""
holos_loop/goals.py - Goal Conditions and Targeting

Imported from original holos.py to enable targeted search.

GoalCondition allows Layer 1/2 strategies to define WHAT to search for,
while Layer 0 (game) provides the raw capabilities.

The physics interpretation:
- Goals act as FILTERS on the wave function
- States not matching the goal experience destructive interference
- This is analogous to quantum measurement projecting onto a subspace
"""

from typing import Set, Optional
from dataclasses import dataclass


@dataclass
class GoalCondition:
    """
    Defines what counts as 'success' for a targeted search.

    This is a Layer 1/2 concept - the STRATEGY of what to search for.
    Layer 0 (game) provides the capabilities, Layer 1/2 decides the goals.

    Attributes:
        target_signatures: Set of state signatures that count as goal states.
                          For chess, this might be material strings like {'KQRRvKQR'}.
                          For Connect4, this might be win patterns.
        early_terminate_misses: If True, stop expanding paths that can't reach goals.
        name: Human-readable name for this goal condition.

    Physics interpretation:
        Goals act as projection operators on the wave function.
        When early_terminate_misses=True, non-goal paths experience
        destructive interference (amplitude -> 0).

    The game must implement get_signature(state) for this to work.
    """
    target_signatures: Set[str]
    early_terminate_misses: bool = True
    name: str = "unnamed_goal"

    def matches(self, signature: str) -> bool:
        """Check if a state signature matches this goal"""
        return signature in self.target_signatures

    def __hash__(self):
        return hash((frozenset(self.target_signatures), self.early_terminate_misses, self.name))

    def __repr__(self):
        return f"GoalCondition({self.name}, targets={len(self.target_signatures)})"


@dataclass
class GoalAllocation:
    """
    Resource allocation for a specific goal.

    Used by Layer 2/3 to distribute compute/energy across multiple goals.

    Physics interpretation:
        This is like allocating amplitude to different measurement outcomes.
        Higher priority goals receive more of the wave function's energy.
    """
    goal: GoalCondition
    priority: float = 1.0  # Higher = more resources
    budget_fraction: float = 0.0  # Fraction of total budget (0-1)

    # Tracking
    states_explored: int = 0
    states_solved: int = 0

    @property
    def efficiency(self) -> float:
        """States solved per budget unit"""
        if self.budget_fraction <= 0:
            return 0.0
        return self.states_solved / self.budget_fraction

    def __repr__(self):
        return f"GoalAllocation({self.goal.name}, priority={self.priority:.2f}, solved={self.states_solved})"


def create_material_goal(materials: Set[str], early_terminate: bool = True) -> GoalCondition:
    """
    Convenience function to create a chess material-based goal.

    Args:
        materials: Set of material signatures like {"KQRRvKQR", "KQRBvKQR"}
        early_terminate: Whether to stop paths that can't reach goal materials

    Returns:
        GoalCondition configured for material targeting
    """
    name = f"materials_{len(materials)}" if len(materials) > 2 else "_".join(sorted(materials))
    return GoalCondition(
        target_signatures=materials,
        early_terminate_misses=early_terminate,
        name=name
    )


def create_pattern_goal(patterns: Set[str], name: str = "pattern_goal") -> GoalCondition:
    """
    Create a goal based on pattern signatures.

    Patterns can be any string that game.get_signature() returns.
    This is game-agnostic - works with any game that implements signatures.
    """
    return GoalCondition(
        target_signatures=patterns,
        early_terminate_misses=True,
        name=name
    )
