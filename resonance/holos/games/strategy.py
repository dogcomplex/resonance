"""
holos/games/strategy.py - Strategy Meta-Game (Layer 2)

This implements GameInterface for the META-META-GAME of goal/budget allocation.

Layer 2 decides:
- WHAT goals to pursue (which material signatures to target)
- HOW MUCH budget to allocate to each goal
- WHEN to switch between goals

Layer 1 (seeds.py) then executes the strategy by selecting optimal seeds.
Layer 0 (chess.py) provides the capabilities.

The key insight: Strategy is about RESOURCE ALLOCATION across goals.
"""

from typing import List, Tuple, Optional, Any, Set, FrozenSet, Dict
from dataclasses import dataclass, field
from collections import defaultdict

# GoalCondition is defined in holos.py (the engine) since HOLOSSolver uses it directly
# Re-export here for Layer 2 convenience
from holos.holos import GoalCondition


# ============================================================
# STRATEGY STATE - Budget allocation across goals
# ============================================================

@dataclass(frozen=True)
class GoalAllocation:
    """Allocation of resources to a single goal"""
    goal: GoalCondition
    budget_fraction: float  # 0.0 to 1.0
    priority: int  # Higher = process first

    def __hash__(self):
        return hash((self.goal, self.budget_fraction, self.priority))


@dataclass(frozen=True)
class StrategyState:
    """
    A complete strategy - allocation of budget across goals.

    This is the STATE in the Layer 2 meta-game.
    """
    allocations: FrozenSet[GoalAllocation]
    total_budget: float  # Total compute budget available
    material_universe: str  # What material space we're solving

    _hash: int = field(default=None, hash=False, compare=False)

    def __hash__(self):
        if self._hash is None:
            object.__setattr__(self, '_hash', hash((self.allocations, self.total_budget)))
        return self._hash

    def budget_for(self, goal_name: str) -> float:
        """Get budget allocated to a specific goal"""
        for alloc in self.allocations:
            if alloc.goal.name == goal_name:
                return self.total_budget * alloc.budget_fraction
        return 0.0


# ============================================================
# STRATEGY VALUE - Quality of a strategy
# ============================================================

@dataclass(frozen=True)
class StrategyValue:
    """Value of a strategy: overall solution quality"""
    total_solved: int       # Total positions solved across all goals
    goal_coverage: Dict[str, float]  # Coverage per goal (0-1)
    efficiency: float       # Solved / budget_used
    completeness: float     # Fraction of goals with >90% coverage

    def __repr__(self):
        return f"StrategyValue(solved={self.total_solved}, eff={self.efficiency:.1f}, complete={self.completeness:.1%})"

    def __lt__(self, other):
        # Prioritize completeness, then efficiency
        if self.completeness != other.completeness:
            return self.completeness < other.completeness
        return self.efficiency < other.efficiency

    def __eq__(self, other):
        if not isinstance(other, StrategyValue):
            return False
        return self.completeness == other.completeness and self.efficiency == other.efficiency


# ============================================================
# STRATEGY GAME INTERFACE
# ============================================================

# Import here to avoid circular imports
def _get_game_interface():
    from holos.holos import GameInterface
    return GameInterface


class StrategyGame:
    """
    The Layer 2 meta-game of goal/budget allocation.

    This is a LATTICE game over strategy space:
    - Moving "up" = adding goals or increasing budgets
    - Moving "down" = removing goals or decreasing budgets
    - Moving "sideways" = reallocating budget between goals

    The game searches for COMPLETE and EFFICIENT strategies.
    """

    def __init__(self, material_universe: str = "KQRRvKQR",
                 available_goals: List[GoalCondition] = None,
                 total_budget: float = 1000.0):
        self.material_universe = material_universe
        self.available_goals = available_goals or []
        self.total_budget = total_budget

        # Cache of evaluated strategies
        self.eval_cache: Dict[int, StrategyValue] = {}

    def hash_state(self, state: StrategyState) -> int:
        return hash(state)

    def get_successors(self, state: StrategyState) -> List[Tuple[StrategyState, Any]]:
        """
        Generate successor strategies (more goals or higher budgets).
        """
        successors = []
        allocations = set(state.allocations)
        current_goals = {a.goal.name for a in allocations}

        # ADD: Add a new goal
        for goal in self.available_goals:
            if goal.name not in current_goals:
                # Add with 10% budget, reduce others proportionally
                new_alloc = GoalAllocation(goal, 0.1, priority=len(allocations))
                # Scale down existing
                scaled = {GoalAllocation(a.goal, a.budget_fraction * 0.9, a.priority)
                          for a in allocations}
                new_state = StrategyState(
                    frozenset(scaled | {new_alloc}),
                    state.total_budget,
                    state.material_universe
                )
                successors.append((new_state, ('add_goal', goal.name)))

        # INCREASE: Increase budget for existing goal
        for alloc in allocations:
            if alloc.budget_fraction < 0.9:
                new_alloc = GoalAllocation(alloc.goal, min(1.0, alloc.budget_fraction + 0.1), alloc.priority)
                # Scale down others
                others = allocations - {alloc}
                scale = (1.0 - new_alloc.budget_fraction) / sum(a.budget_fraction for a in others) if others else 1.0
                scaled = {GoalAllocation(a.goal, a.budget_fraction * scale, a.priority) for a in others}
                new_state = StrategyState(
                    frozenset(scaled | {new_alloc}),
                    state.total_budget,
                    state.material_universe
                )
                successors.append((new_state, ('increase', alloc.goal.name)))

        return successors[:50]

    def get_predecessors(self, state: StrategyState) -> List[Tuple[StrategyState, Any]]:
        """
        Generate predecessor strategies (fewer goals or lower budgets).
        """
        predecessors = []
        allocations = set(state.allocations)

        # REMOVE: Remove a goal
        for alloc in allocations:
            if len(allocations) > 1:
                others = allocations - {alloc}
                # Redistribute budget
                scale = 1.0 / sum(a.budget_fraction for a in others)
                scaled = {GoalAllocation(a.goal, a.budget_fraction * scale, a.priority) for a in others}
                new_state = StrategyState(
                    frozenset(scaled),
                    state.total_budget,
                    state.material_universe
                )
                predecessors.append((new_state, ('remove_goal', alloc.goal.name)))

        # DECREASE: Decrease budget for existing goal
        for alloc in allocations:
            if alloc.budget_fraction > 0.1:
                new_alloc = GoalAllocation(alloc.goal, alloc.budget_fraction - 0.1, alloc.priority)
                # Scale up others
                others = allocations - {alloc}
                if others:
                    scale = (1.0 - new_alloc.budget_fraction) / sum(a.budget_fraction for a in others)
                    scaled = {GoalAllocation(a.goal, a.budget_fraction * scale, a.priority) for a in others}
                else:
                    scaled = set()
                new_state = StrategyState(
                    frozenset(scaled | {new_alloc}),
                    state.total_budget,
                    state.material_universe
                )
                predecessors.append((new_state, ('decrease', alloc.goal.name)))

        return predecessors

    def is_boundary(self, state: StrategyState) -> bool:
        """Boundary: empty strategy or previously evaluated"""
        if len(state.allocations) == 0:
            return True
        return hash(state) in self.eval_cache

    def get_boundary_value(self, state: StrategyState) -> Optional[StrategyValue]:
        """Get cached value or compute for boundary"""
        h = hash(state)

        if h in self.eval_cache:
            return self.eval_cache[h]

        if len(state.allocations) == 0:
            return StrategyValue(0, {}, 0.0, 0.0)

        return None

    def is_terminal(self, state: StrategyState) -> Tuple[bool, Optional[StrategyValue]]:
        """Terminal if all goals have 100% coverage"""
        return False, None

    def propagate_value(self, state: StrategyState,
                        child_values: List[StrategyValue]) -> Optional[StrategyValue]:
        """
        Value propagation: best completeness, then efficiency.
        """
        if not child_values:
            return None

        return max(child_values, key=lambda v: (v.completeness, v.efficiency))

    def get_features(self, state: StrategyState) -> Any:
        """Equivalence features"""
        num_goals = len(state.allocations)
        budget_dist = tuple(sorted(a.budget_fraction for a in state.allocations))
        return (num_goals, budget_dist)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_goal_from_material(material: str, early_terminate: bool = True) -> GoalCondition:
    """Create a GoalCondition targeting a specific material"""
    return GoalCondition(
        target_signatures={material},
        early_terminate_misses=early_terminate,
        name=f"goal_{material}"
    )


def create_multi_material_goal(materials: List[str], name: str = None,
                                early_terminate: bool = True) -> GoalCondition:
    """Create a GoalCondition targeting multiple materials"""
    return GoalCondition(
        target_signatures=set(materials),
        early_terminate_misses=early_terminate,
        name=name or f"goal_{'_'.join(materials[:3])}"
    )


def create_strategy_solver(material_universe: str, goals: List[GoalCondition],
                           total_budget: float = 1000.0):
    """Create a HOLOS solver for the strategy game"""
    from holos.holos import HOLOSSolver

    game = StrategyGame(material_universe, goals, total_budget)
    solver = HOLOSSolver(game, name=f"strategy_{material_universe}")
    return solver, game
