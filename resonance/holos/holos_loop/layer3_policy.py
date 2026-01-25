"""
holos/layer3_policy.py - Policy Scale (Layer 3)

Layer 3 operates at the POLICY scale:
- State: A policy (SEQUENCE of Layer 2 covers, ordered by priority)
- Value: Policy quality (total problem coverage, resource efficiency)
- Moves: Add/remove/reorder covers in the policy
- Boundary: Complete policies (those solving the full problem)

Key insight: A policy is a SEQUENCE of covers, not a static allocation.
The order determines which subproblems to solve first.

The two waves at Layer 3:
- Forward wave: Building policies by adding covers
- Backward wave: Refining policies by removing redundant covers
- Closure: When a policy efficiently solves the full problem

This layer manages compute/storage tradeoffs:
- Which covers to compute (compute cost)
- Which covers to store vs recompute (storage cost)
- How to sequence cover computation (ordering)

The HOLOS pattern at Layer 3:
- State space: All possible orderings of covers
- Bidirectional search finds optimal cover sequences
- Closures indicate efficient complete policies
"""

from typing import List, Tuple, Optional, Any, Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict

from holos.holos import GameInterface
from holos.closure import (
    ClosureDetector, ClosureEvent, ClosureType,
    WaveOrigin, LayerCoupling
)
from holos.layer2_covers import PathCover, CoverValue


# ============================================================
# LAYER 3 STATE: POLICY (Sequence of Covers)
# ============================================================

@dataclass
class CoverPolicy:
    """
    A policy: an ordered sequence of covers.

    This is the STATE at Layer 3 - we're searching over cover orderings.

    The ORDER matters:
    - Earlier covers have higher priority
    - Resources are allocated in sequence order
    - Each cover addresses a subproblem

    A policy is "complete" when all subproblems are solved.
    """
    # Ordered sequence of cover hashes
    covers: Tuple[int, ...]

    # Resource tracking
    total_compute: float = 0.0  # Total compute cost
    total_storage: float = 0.0  # Total storage cost

    # Coverage tracking
    problems_solved: int = 0
    total_problems: int = 1

    # Metadata
    origin: WaveOrigin = WaveOrigin.FORWARD

    # Cache
    _cover_objects: Dict[int, PathCover] = field(default_factory=dict, hash=False, compare=False)

    def __hash__(self):
        return hash(self.covers)

    def __len__(self):
        return len(self.covers)

    @property
    def num_covers(self) -> int:
        return len(self.covers)

    @property
    def completeness(self) -> float:
        return self.problems_solved / max(1, self.total_problems)

    def add_cover(self, cover_hash: int, cover_obj: PathCover = None) -> 'CoverPolicy':
        """Create new policy with cover added at end"""
        new_covers = self.covers + (cover_hash,)
        new_objs = dict(self._cover_objects)
        if cover_obj is not None:
            new_objs[cover_hash] = cover_obj
        return CoverPolicy(
            covers=new_covers,
            origin=self.origin,
            _cover_objects=new_objs
        )

    def prepend_cover(self, cover_hash: int, cover_obj: PathCover = None) -> 'CoverPolicy':
        """Create new policy with cover added at start (highest priority)"""
        new_covers = (cover_hash,) + self.covers
        new_objs = dict(self._cover_objects)
        if cover_obj is not None:
            new_objs[cover_hash] = cover_obj
        return CoverPolicy(
            covers=new_covers,
            origin=self.origin,
            _cover_objects=new_objs
        )

    def remove_cover(self, index: int) -> 'CoverPolicy':
        """Create new policy with cover at index removed"""
        if index < 0 or index >= len(self.covers):
            return self
        new_covers = self.covers[:index] + self.covers[index+1:]
        return CoverPolicy(
            covers=new_covers,
            origin=self.origin,
            _cover_objects=self._cover_objects
        )

    def swap(self, i: int, j: int) -> 'CoverPolicy':
        """Create new policy with covers i and j swapped"""
        if i < 0 or j < 0 or i >= len(self.covers) or j >= len(self.covers):
            return self
        cover_list = list(self.covers)
        cover_list[i], cover_list[j] = cover_list[j], cover_list[i]
        return CoverPolicy(
            covers=tuple(cover_list),
            origin=self.origin,
            _cover_objects=self._cover_objects
        )

    def signature(self) -> str:
        return f"Policy({self.origin.value}, {self.num_covers} covers, {self.completeness:.1%} complete)"


# ============================================================
# LAYER 3 VALUE: POLICY QUALITY
# ============================================================

@dataclass(frozen=True)
class PolicyValue:
    """
    Value of a policy at Layer 3.

    Measures policy quality:
    - completeness: Fraction of total problem solved
    - compute_efficiency: coverage / compute_cost
    - storage_efficiency: coverage / storage_cost
    - priority: Combined priority score
    """
    completeness: float      # Fraction of problem solved (0-1)
    compute_cost: float      # Total compute resources used
    storage_cost: float      # Total storage resources used
    compute_efficiency: float  # coverage / compute
    storage_efficiency: float  # coverage / storage
    priority: float          # Combined priority score

    # For complete policies
    is_complete: bool = False

    def __repr__(self):
        status = "COMPLETE" if self.is_complete else "partial"
        return f"PolicyValue({status}, {self.completeness:.1%}, comp_eff={self.compute_efficiency:.1f})"

    def __lt__(self, other):
        # Primary: completeness, Secondary: compute efficiency
        if self.completeness != other.completeness:
            return self.completeness < other.completeness
        return self.compute_efficiency < other.compute_efficiency


# ============================================================
# LAYER 3 GAME INTERFACE
# ============================================================

class PolicyGame(GameInterface[CoverPolicy, PolicyValue]):
    """
    Layer 3 game: Search over policies (cover orderings).

    State space: All orderings of available covers
    Successors: Add cover to policy
    Predecessors: Remove cover from policy
    Boundary: Complete policies (solve full problem)
    Value: Policy quality (completeness, efficiency)
    """

    def __init__(self,
                 available_covers: List[Tuple[int, PathCover]],
                 total_problems: int = 1,
                 compute_budget: float = 1000.0,
                 storage_budget: float = 1000.0,
                 closure_detector: ClosureDetector = None):
        """
        Args:
            available_covers: Pool of (hash, cover) pairs
            total_problems: Number of subproblems to solve
            compute_budget: Maximum compute resources
            storage_budget: Maximum storage resources
            closure_detector: Shared closure detector
        """
        self.available_covers = {h: cover for h, cover in available_covers}
        self.cover_hashes = list(self.available_covers.keys())
        self.total_problems = total_problems
        self.compute_budget = compute_budget
        self.storage_budget = storage_budget
        self.closure_detector = closure_detector or ClosureDetector()

        # Cover cost/value tracking
        self.cover_compute_cost: Dict[int, float] = {}
        self.cover_storage_cost: Dict[int, float] = {}
        self.cover_problems_solved: Dict[int, Set[int]] = {}  # cover_hash -> problem indices

        # Statistics
        self.stats = {
            'policies_created': 0,
            'policies_extended': 0,
            'complete_policies': 0,
        }

    def set_cover_costs(self, cover_hash: int, compute: float, storage: float):
        """Set compute and storage costs for a cover"""
        self.cover_compute_cost[cover_hash] = compute
        self.cover_storage_cost[cover_hash] = storage

    def set_cover_problems(self, cover_hash: int, problems: Set[int]):
        """Set which problems a cover solves"""
        self.cover_problems_solved[cover_hash] = problems

    def hash_state(self, policy: CoverPolicy) -> int:
        return hash(policy)

    def get_successors(self, policy: CoverPolicy) -> List[Tuple[CoverPolicy, Any]]:
        """Extend policy by adding covers"""
        successors = []
        current_covers = set(policy.covers)

        # Check resource constraints
        current_compute = sum(self.cover_compute_cost.get(h, 0) for h in policy.covers)
        current_storage = sum(self.cover_storage_cost.get(h, 0) for h in policy.covers)

        for cover_hash in self.cover_hashes:
            if cover_hash not in current_covers:
                # Check if adding this cover would exceed budgets
                new_compute = current_compute + self.cover_compute_cost.get(cover_hash, 0)
                new_storage = current_storage + self.cover_storage_cost.get(cover_hash, 0)

                if new_compute <= self.compute_budget and new_storage <= self.storage_budget:
                    cover_obj = self.available_covers.get(cover_hash)
                    new_policy = policy.add_cover(cover_hash, cover_obj)
                    successors.append((new_policy, ('add', cover_hash)))
                    self.stats['policies_extended'] += 1

        # Reordering moves
        for i in range(len(policy.covers) - 1):
            swapped = policy.swap(i, i + 1)
            successors.append((swapped, ('swap', i, i + 1)))

        return successors

    def get_predecessors(self, policy: CoverPolicy) -> List[Tuple[CoverPolicy, Any]]:
        """Simplify policy by removing covers"""
        predecessors = []

        for i in range(len(policy.covers)):
            reduced = policy.remove_cover(i)
            if len(reduced.covers) > 0:
                predecessors.append((reduced, ('remove', i)))

        return predecessors

    def is_boundary(self, policy: CoverPolicy) -> bool:
        """Policy is at boundary if it solves all problems"""
        solved = self._compute_problems_solved(policy)
        return len(solved) >= self.total_problems

    def get_boundary_value(self, policy: CoverPolicy) -> Optional[PolicyValue]:
        """Get value for complete policy"""
        if not self.is_boundary(policy):
            return None

        return self._evaluate_policy(policy, is_complete=True)

    def is_terminal(self, policy: CoverPolicy) -> Tuple[bool, Optional[PolicyValue]]:
        """Policy is terminal if complete with no redundancy"""
        if not self.is_boundary(policy):
            return False, None

        # Check for redundancy
        for i in range(len(policy.covers)):
            reduced = policy.remove_cover(i)
            if len(reduced.covers) > 0 and self.is_boundary(reduced):
                return False, None

        return True, self.get_boundary_value(policy)

    def propagate_value(self, policy: CoverPolicy, child_values: List[PolicyValue]) -> Optional[PolicyValue]:
        """Propagate value from child policies"""
        if not child_values:
            return None

        best = max(child_values, key=lambda v: (v.completeness, v.compute_efficiency))

        return PolicyValue(
            completeness=best.completeness * 0.9,
            compute_cost=best.compute_cost * 0.95,
            storage_cost=best.storage_cost * 0.95,
            compute_efficiency=best.compute_efficiency * 0.95,
            storage_efficiency=best.storage_efficiency * 0.95,
            priority=best.priority * 0.9
        )

    def get_features(self, policy: CoverPolicy) -> Any:
        """Feature extraction"""
        return (policy.num_covers, len(self._compute_problems_solved(policy)))

    def _compute_problems_solved(self, policy: CoverPolicy) -> Set[int]:
        """Compute total problems solved by policy"""
        solved = set()
        for cover_hash in policy.covers:
            if cover_hash in self.cover_problems_solved:
                solved.update(self.cover_problems_solved[cover_hash])
        return solved

    def _evaluate_policy(self, policy: CoverPolicy, is_complete: bool = False) -> PolicyValue:
        """Evaluate a policy's quality"""
        solved = self._compute_problems_solved(policy)
        completeness = len(solved) / max(1, self.total_problems)

        compute_cost = sum(self.cover_compute_cost.get(h, 0) for h in policy.covers)
        storage_cost = sum(self.cover_storage_cost.get(h, 0) for h in policy.covers)

        compute_eff = completeness / max(0.001, compute_cost) * 1000
        storage_eff = completeness / max(0.001, storage_cost) * 1000

        priority = completeness * compute_eff * 10

        return PolicyValue(
            completeness=completeness,
            compute_cost=compute_cost,
            storage_cost=storage_cost,
            compute_efficiency=compute_eff,
            storage_efficiency=storage_eff,
            priority=priority,
            is_complete=is_complete or completeness >= 1.0
        )

    # ==================== Policy-Specific Methods ====================

    def create_empty_policy(self) -> CoverPolicy:
        """Create empty policy"""
        self.stats['policies_created'] += 1
        return CoverPolicy(covers=(), origin=WaveOrigin.FORWARD)

    def create_full_policy(self) -> CoverPolicy:
        """Create policy with all covers"""
        self.stats['policies_created'] += 1
        return CoverPolicy(
            covers=tuple(self.cover_hashes),
            origin=WaveOrigin.BACKWARD,
            _cover_objects=self.available_covers
        )

    def evaluate_policy(self, policy: CoverPolicy) -> PolicyValue:
        """Public evaluation method"""
        return self._evaluate_policy(policy)


# ============================================================
# LAYER 3 SOLVER
# ============================================================

class PolicyLayerSolver:
    """
    Solver for Layer 3 (policy scale).

    Uses bidirectional search over policy space:
    - Forward: Build policies by adding covers
    - Backward: Refine policies by removing redundancy
    - Closure: Efficient complete policies
    """

    def __init__(self,
                 available_covers: List[Tuple[int, PathCover]],
                 total_problems: int = 1,
                 compute_budget: float = 1000.0,
                 storage_budget: float = 1000.0,
                 closure_detector: ClosureDetector = None):
        self.closure_detector = closure_detector or ClosureDetector()
        self.policy_game = PolicyGame(
            available_covers, total_problems,
            compute_budget, storage_budget,
            self.closure_detector
        )

        # Policy fronts
        self.forward_front: Dict[int, CoverPolicy] = {}
        self.backward_front: Dict[int, CoverPolicy] = {}

        # Priority queues
        self.forward_queue: List[Tuple[float, CoverPolicy]] = []
        self.backward_queue: List[Tuple[float, CoverPolicy]] = []

        # Results
        self.best_policies: List[CoverPolicy] = []

        # Statistics
        self.stats = {
            'iterations': 0,
            'forward_extensions': 0,
            'backward_extensions': 0,
            'complete_policies': 0,
        }

    def set_cover_costs(self, cover_hash: int, compute: float, storage: float):
        """Set costs for a cover"""
        self.policy_game.set_cover_costs(cover_hash, compute, storage)

    def set_cover_problems(self, cover_hash: int, problems: Set[int]):
        """Set which problems a cover solves"""
        self.policy_game.set_cover_problems(cover_hash, problems)

    def initialize(self):
        """Initialize fronts"""
        empty = self.policy_game.create_empty_policy()
        value = self.policy_game.evaluate_policy(empty)
        h = hash(empty)
        self.forward_front[h] = empty
        self.forward_queue.append((-value.priority, empty))

        full = self.policy_game.create_full_policy()
        value = self.policy_game.evaluate_policy(full)
        h = hash(full)
        self.backward_front[h] = full
        self.backward_queue.append((-value.priority, full))

    def step(self, mode: str = "balanced") -> Dict:
        """Perform one step"""
        self.stats['iterations'] += 1
        result = {'extended': 0, 'complete': 0}

        if mode == "forward":
            self._extend_forward()
            result['extended'] = 1
        elif mode == "backward":
            self._extend_backward()
            result['extended'] = 1
        else:
            self._extend_forward()
            self._extend_backward()
            result['extended'] = 2

        return result

    def _extend_forward(self):
        """Extend forward front"""
        if not self.forward_queue:
            return

        _, policy = self.forward_queue.pop(0)

        for new_policy, move in self.policy_game.get_successors(policy):
            value = self.policy_game.evaluate_policy(new_policy)
            h = hash(new_policy)
            if h not in self.forward_front:
                self.forward_front[h] = new_policy
                self.forward_queue.append((-value.priority, new_policy))
                self.stats['forward_extensions'] += 1

                if value.is_complete:
                    self.best_policies.append(new_policy)
                    self.stats['complete_policies'] += 1

        self.forward_queue.sort(key=lambda x: x[0])

    def _extend_backward(self):
        """Extend backward front"""
        if not self.backward_queue:
            return

        _, policy = self.backward_queue.pop(0)

        for new_policy, move in self.policy_game.get_predecessors(policy):
            value = self.policy_game.evaluate_policy(new_policy)
            h = hash(new_policy)
            if h not in self.backward_front:
                self.backward_front[h] = new_policy
                self.backward_queue.append((-value.priority, new_policy))
                self.stats['backward_extensions'] += 1

        self.backward_queue.sort(key=lambda x: x[0])

    def solve(self,
              max_iterations: int = 100,
              mode: str = "balanced",
              verbose: bool = True) -> List[CoverPolicy]:
        """Solve: find optimal policies"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Layer 3 Policy Solver")
            print(f"  Available covers: {len(self.policy_game.available_covers)}")
            print(f"  Total problems: {self.policy_game.total_problems}")
            print(f"  Compute budget: {self.policy_game.compute_budget}")
            print(f"  Storage budget: {self.policy_game.storage_budget}")
            print(f"{'='*60}")

        self.initialize()

        for i in range(max_iterations):
            result = self.step(mode=mode)

            if verbose and i % 10 == 0:
                print(f"  Iter {i}: fwd={len(self.forward_front)}, bwd={len(self.backward_front)}, "
                      f"complete={len(self.best_policies)}")

            if not self.forward_queue and not self.backward_queue:
                break

        if verbose:
            print(f"\n{'='*60}")
            print(f"Complete: {len(self.best_policies)} policies found")
            print(f"{'='*60}")

        return self.best_policies


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_policy_solver(covers: List[Tuple[int, PathCover]],
                         total_problems: int = 1,
                         compute_budget: float = 1000.0,
                         storage_budget: float = 1000.0) -> PolicyLayerSolver:
    """Create a Layer 3 policy solver"""
    return PolicyLayerSolver(covers, total_problems, compute_budget, storage_budget)
