"""
holos/session.py - Session Manager for Multi-Round Solving

This module handles:
1. Multi-round solving (incremental HOLOS runs)
2. Session state persistence
3. Budget allocation across rounds
4. Progress tracking and resumption

Key Question: Is SessionManager a Layer 1 or Layer 2 decision?

Answer: It depends on what we're optimizing:
- Layer 1 (Seed Selection): How to choose seeds for THIS round
- Layer 2 (Meta-Strategy): How to allocate compute across MULTIPLE rounds

SessionManager operates at BOTH levels:
- It maintains state across rounds (Layer 2 concern)
- It influences seed selection for each round (Layer 1 concern)

The session decides:
- When to stop a round (budget exhausted, progress stalled)
- What to prioritize next round (new seeds, deeper search, different mode)
- How to merge results across rounds
"""

from typing import Dict, Set, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import pickle
import os

from .storage import Hologram


class SessionPhase(Enum):
    """Current phase of the session"""
    INIT = "init"           # Just started
    LIGHTNING = "lightning"  # Fast probing phase
    WAVE = "wave"           # Broad expansion phase
    CRYSTAL = "crystal"     # Focused deepening
    COMPLETE = "complete"   # Session finished


@dataclass
class RoundStats:
    """Statistics for a single solving round"""
    round_id: int
    phase: SessionPhase
    start_time: float
    end_time: Optional[float] = None

    # Progress metrics
    states_explored: int = 0
    states_solved: int = 0
    connections_found: int = 0
    spines_found: int = 0

    # Resource usage
    memory_mb: float = 0.0
    iterations: int = 0

    # Efficiency
    solve_rate: float = 0.0  # states_solved / time
    explore_rate: float = 0.0  # states_explored / time

    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class SessionState:
    """
    Persistent state for a multi-round solving session.

    This is what gets saved between rounds (or program restarts).
    """
    session_id: str
    game_name: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Current phase
    phase: SessionPhase = SessionPhase.INIT
    current_round: int = 0

    # Accumulated results
    total_solved: int = 0
    total_explored: int = 0
    total_connections: int = 0

    # Round history
    rounds: List[RoundStats] = field(default_factory=list)

    # Seeds to explore (persisted between rounds)
    pending_seeds: List[int] = field(default_factory=list)  # State hashes
    explored_seeds: Set[int] = field(default_factory=set)

    # Budget tracking
    total_budget: float = 0.0  # Total compute budget (in arbitrary units)
    budget_used: float = 0.0

    # Meta-learning data
    feature_success: Dict[Any, float] = field(default_factory=dict)  # features -> success_rate
    mode_success: Dict[str, float] = field(default_factory=dict)  # mode -> success_rate

    def save(self, path: str):
        """Save session state"""
        self.updated_at = time.time()
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'SessionState':
        """Load session state"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def budget_remaining(self) -> float:
        return max(0, self.total_budget - self.budget_used)

    def summary(self) -> str:
        lines = [
            f"Session: {self.session_id}",
            f"  Phase: {self.phase.value}",
            f"  Rounds: {self.current_round}",
            f"  Solved: {self.total_solved:,}",
            f"  Explored: {self.total_explored:,}",
            f"  Connections: {self.total_connections:,}",
            f"  Budget: {self.budget_used:.1f} / {self.total_budget:.1f}",
            f"  Pending seeds: {len(self.pending_seeds)}",
        ]
        return "\n".join(lines)


class SessionManager:
    """
    Manages multi-round solving sessions.

    Responsibilities:
    1. Track progress across rounds
    2. Decide when to advance/change phases
    3. Select seeds for next round
    4. Merge results into hologram
    5. Save/restore session state

    The SessionManager is ABOVE the solver - it calls solver.solve()
    multiple times with different parameters based on what it learns.
    """

    def __init__(self, session_id: str, game_name: str,
                 save_dir: str = "./holos_sessions",
                 total_budget: float = 1000.0):
        self.session_id = session_id
        self.game_name = game_name
        self.save_dir = save_dir
        self.total_budget = total_budget

        os.makedirs(save_dir, exist_ok=True)

        # Try to load existing session
        self.state = self._load_or_create()

        # Accumulated hologram
        self.hologram: Optional[Hologram] = None
        self._load_hologram()

    def _state_path(self) -> str:
        return os.path.join(self.save_dir, f"{self.session_id}_state.pkl")

    def _hologram_path(self) -> str:
        return os.path.join(self.save_dir, f"{self.session_id}_hologram.pkl")

    def _load_or_create(self) -> SessionState:
        """Load existing session or create new one"""
        path = self._state_path()
        if os.path.exists(path):
            print(f"Resuming session: {self.session_id}")
            return SessionState.load(path)
        else:
            print(f"Creating new session: {self.session_id}")
            return SessionState(
                session_id=self.session_id,
                game_name=self.game_name,
                total_budget=self.total_budget,
            )

    def _load_hologram(self):
        """Load existing hologram if available"""
        path = self._hologram_path()
        if os.path.exists(path):
            self.hologram = Hologram.load(path)
            print(f"Loaded hologram: {len(self.hologram.solved):,} solved")
        else:
            self.hologram = Hologram(self.session_id)

    def save(self):
        """Save session state and hologram"""
        self.state.save(self._state_path())
        if self.hologram:
            self.hologram.save(self._hologram_path())

    def run_round(self, solver, seeds: List[Any],
                  max_iterations: int = 50,
                  budget: float = 100.0) -> RoundStats:
        """
        Run a single solving round.

        Args:
            solver: HOLOSSolver instance
            seeds: List of seed states (or SeedPoints)
            max_iterations: Max iterations for this round
            budget: Compute budget for this round

        Returns:
            RoundStats for this round
        """
        from .holos import SeedPoint, SearchMode

        # Create round stats
        round_stats = RoundStats(
            round_id=self.state.current_round,
            phase=self.state.phase,
            start_time=time.time(),
        )

        # Convert seeds to SeedPoints if needed
        seed_points = []
        for seed in seeds:
            if isinstance(seed, SeedPoint):
                seed_points.append(seed)
            else:
                seed_points.append(SeedPoint(seed, SearchMode.WAVE))

        # Run solver
        print(f"\n--- Round {round_stats.round_id} ({self.state.phase.value}) ---")
        result = solver.solve(seed_points, max_iterations=max_iterations)

        # Update round stats
        round_stats.end_time = time.time()
        round_stats.states_solved = len(result.solved)
        round_stats.states_explored = solver.stats.get('forward_expanded', 0) + \
                                       solver.stats.get('backward_expanded', 0)
        round_stats.connections_found = len(result.connections)
        round_stats.spines_found = len(result.spines)
        round_stats.iterations = max_iterations

        duration = round_stats.duration()
        if duration > 0:
            round_stats.solve_rate = round_stats.states_solved / duration
            round_stats.explore_rate = round_stats.states_explored / duration

        # Merge into hologram
        if self.hologram:
            self.hologram = self.hologram.merge(result)
        else:
            self.hologram = result

        # Update session state
        self.state.rounds.append(round_stats)
        self.state.current_round += 1
        self.state.total_solved = len(self.hologram.solved)
        self.state.total_explored += round_stats.states_explored
        self.state.total_connections += round_stats.connections_found
        self.state.budget_used += budget

        # Update explored seeds
        for sp in seed_points:
            h = solver.game.hash_state(sp.state)
            self.state.explored_seeds.add(h)

        # Auto-save
        self.save()

        print(f"Round complete: {round_stats.states_solved:,} solved, "
              f"{round_stats.connections_found} connections")

        return round_stats

    def select_next_seeds(self, game, num_seeds: int = 20) -> List[Any]:
        """
        Select seeds for the next round based on session history.

        Strategy depends on phase:
        - LIGHTNING: Sample from unsolved frontier
        - WAVE: Expand from highest-value connections
        - CRYSTAL: Focus on promising features

        This is a Layer 1 decision informed by Layer 2 data.
        """
        from .holos import SeedPoint, SearchMode

        seeds = []

        # Get pending seeds that haven't been explored
        pending = [h for h in self.state.pending_seeds
                   if h not in self.state.explored_seeds]

        if pending:
            # Use pending seeds first
            for h in pending[:num_seeds]:
                # We need to reconstruct state from hash
                # This requires game-specific logic, so we skip for now
                pass

        # If we have hologram with connections, expand from there
        if self.hologram and self.hologram.connections:
            # Get recent connections
            recent = self.hologram.connections[-20:]
            for fh, bh, value in recent:
                if fh not in self.state.explored_seeds:
                    self.state.pending_seeds.append(fh)

        return seeds

    def should_continue(self) -> bool:
        """Check if session should continue"""
        # Budget exhausted
        if self.state.budget_remaining() <= 0:
            return False

        # Check for progress stall
        if len(self.state.rounds) >= 3:
            recent = self.state.rounds[-3:]
            avg_solve = sum(r.states_solved for r in recent) / 3
            if avg_solve < 10:  # Progress stalled
                return False

        return True

    def advance_phase(self):
        """
        Advance to next phase based on progress.

        Phase transitions:
        - INIT -> LIGHTNING: Always start with fast probing
        - LIGHTNING -> WAVE: When lightning probes saturate
        - WAVE -> CRYSTAL: When wave expansion slows
        - CRYSTAL -> COMPLETE: When no more progress
        """
        current = self.state.phase

        if current == SessionPhase.INIT:
            self.state.phase = SessionPhase.LIGHTNING

        elif current == SessionPhase.LIGHTNING:
            # Check if lightning is still productive
            if len(self.state.rounds) >= 2:
                recent = self.state.rounds[-2:]
                avg_spines = sum(r.spines_found for r in recent) / 2
                if avg_spines < 2:  # Lightning becoming unproductive
                    self.state.phase = SessionPhase.WAVE

        elif current == SessionPhase.WAVE:
            # Check if wave is still expanding
            if len(self.state.rounds) >= 3:
                recent = self.state.rounds[-3:]
                growth_rate = (recent[-1].states_explored -
                               recent[0].states_explored) / len(recent)
                if growth_rate < 100:  # Growth slowing
                    self.state.phase = SessionPhase.CRYSTAL

        elif current == SessionPhase.CRYSTAL:
            # Check if crystal is still refining
            if len(self.state.rounds) >= 2:
                recent = self.state.rounds[-2:]
                avg_solve = sum(r.states_solved for r in recent) / 2
                if avg_solve < 5:  # Crystal done
                    self.state.phase = SessionPhase.COMPLETE

        if self.state.phase != current:
            print(f"Phase transition: {current.value} -> {self.state.phase.value}")

    def get_mode_for_phase(self) -> 'SearchMode':
        """Get search mode based on current phase"""
        from .holos import SearchMode

        phase_modes = {
            SessionPhase.INIT: SearchMode.LIGHTNING,
            SessionPhase.LIGHTNING: SearchMode.LIGHTNING,
            SessionPhase.WAVE: SearchMode.WAVE,
            SessionPhase.CRYSTAL: SearchMode.CRYSTAL,
            SessionPhase.COMPLETE: SearchMode.WAVE,
        }
        return phase_modes.get(self.state.phase, SearchMode.WAVE)

    def run_session(self, solver, initial_seeds: List[Any],
                    rounds_per_phase: int = 3,
                    iterations_per_round: int = 50):
        """
        Run complete solving session with automatic phase management.

        Args:
            solver: HOLOSSolver instance
            initial_seeds: Starting positions
            rounds_per_phase: Rounds before considering phase transition
            iterations_per_round: Iterations per round
        """
        print(f"\n{'='*60}")
        print(f"HOLOS SESSION: {self.session_id}")
        print(f"{'='*60}")
        print(self.state.summary())

        # Initialize with seeds
        seeds = initial_seeds

        while self.should_continue():
            # Advance phase if needed
            if self.state.current_round > 0 and \
               self.state.current_round % rounds_per_phase == 0:
                self.advance_phase()

            if self.state.phase == SessionPhase.COMPLETE:
                break

            # Get budget for this round
            budget = min(100.0, self.state.budget_remaining())

            # Run round
            stats = self.run_round(
                solver, seeds,
                max_iterations=iterations_per_round,
                budget=budget
            )

            # Select seeds for next round
            seeds = self.select_next_seeds(solver.game)
            if not seeds:
                # No more seeds, reuse initial with different mode
                seeds = initial_seeds[:10]

        # Final summary
        print(f"\n{'='*60}")
        print("SESSION COMPLETE")
        print(f"{'='*60}")
        print(self.state.summary())
        if self.hologram:
            print(self.hologram.summary())


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_session(session_id: str, game_name: str,
                   save_dir: str = "./holos_sessions",
                   budget: float = 1000.0) -> SessionManager:
    """Create or resume a session"""
    return SessionManager(session_id, game_name, save_dir, budget)


def quick_solve(solver, seeds: List[Any],
                max_iterations: int = 100) -> Hologram:
    """Quick solve without session management"""
    from .holos import SeedPoint, SearchMode

    seed_points = [SeedPoint(s, SearchMode.WAVE) for s in seeds]
    return solver.solve(seed_points, max_iterations=max_iterations)
