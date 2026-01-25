"""
holos_loop/session_state.py - Session State and Persistence

Imported and adapted from original session.py for the closure-aware system.

Key insight: Sessions are about PERSISTENCE across solving rounds.
The physics interpretation is that sessions track the HISTORY of
wave propagation, enabling the system to resume from checkpoints.

Unlike the original SessionManager (which actively controls solving),
this module focuses on STATE tracking. The physics-based wave system
handles actual control flow.
"""

from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import pickle
import os


class SessionPhase(Enum):
    """
    Current phase of the session.

    Physics interpretation:
        These phases represent different REGIMES of wave propagation:
        - INIT: Preparing the medium (setting initial conditions)
        - LIGHTNING: High-frequency, low-amplitude probing (fast signals)
        - WAVE: Broad wavefront expansion (coherent propagation)
        - CRYSTAL: Local solidification around closures (phase transitions)
        - COMPLETE: Equilibrium reached (no more flow)

    Note: In the closure system, phases EMERGE from closure conditions
    rather than being explicitly transitioned. This enum tracks what
    the system has DETECTED, not what it's been TOLD to do.
    """
    INIT = "init"
    LIGHTNING = "lightning"
    WAVE = "wave"
    CRYSTAL = "crystal"
    COMPLETE = "complete"


@dataclass
class RoundStats:
    """
    Statistics for a single solving round.

    A "round" is a bounded period of wave propagation.
    Physics interpretation: Each round is like a pulse of energy
    injected into the system, with measured outcomes.
    """
    round_id: int
    phase: SessionPhase
    start_time: float
    end_time: Optional[float] = None

    # Progress metrics
    states_explored: int = 0
    states_solved: int = 0
    connections_found: int = 0
    closures_found: int = 0
    spines_found: int = 0

    # Closure-specific metrics
    irreducible_closures: int = 0
    interiors_found: int = 0
    equiv_shortcuts: int = 0

    # Energy/pressure metrics (new for closure system)
    energy_start: float = 0.0
    energy_end: float = 0.0
    pressure_peak: float = 0.0

    # Resource usage
    memory_mb: float = 0.0
    iterations: int = 0

    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def energy_dissipated(self) -> float:
        """Energy lost to damping during this round"""
        return self.energy_start - self.energy_end

    @property
    def closure_rate(self) -> float:
        """Closures per unit energy dissipated"""
        if self.energy_dissipated <= 0:
            return 0.0
        return self.closures_found / self.energy_dissipated

    def summary(self) -> str:
        return (f"Round {self.round_id} ({self.phase.value}): "
                f"explored={self.states_explored}, solved={self.states_solved}, "
                f"closures={self.closures_found}, interiors={self.interiors_found}")


@dataclass
class SessionState:
    """
    Persistent state for a multi-round solving session.

    This is what gets saved between rounds (or program restarts).
    The session state is a CHECKPOINT of the wave system's history.

    Physics interpretation:
        This is like recording the state of a physical system at discrete times.
        The session allows "rewinding" to a previous state and resuming propagation.
    """
    session_id: str
    game_name: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Current phase (detected, not forced)
    phase: SessionPhase = SessionPhase.INIT
    current_round: int = 0

    # Accumulated results
    total_solved: int = 0
    total_explored: int = 0
    total_closures: int = 0
    total_connections: int = 0
    total_interiors: int = 0

    # Round history
    rounds: List[RoundStats] = field(default_factory=list)

    # Seeds to explore (persisted between rounds)
    pending_seeds: List[int] = field(default_factory=list)
    explored_seeds: Set[int] = field(default_factory=set)

    # Energy/pressure tracking (new)
    initial_energy: float = 0.0
    energy_remaining: float = 0.0
    peak_pressure: float = 0.0

    # Budget tracking (legacy compatibility)
    total_budget: float = 0.0
    budget_used: float = 0.0

    # Closure-specific tracking
    closure_history: List[Dict[str, Any]] = field(default_factory=list)
    mode_history: List[str] = field(default_factory=list)  # Emergent modes observed

    def save(self, path: str):
        """Save session state"""
        self.updated_at = time.time()

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'SessionState':
        """Load session state"""
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def exists(path: str) -> bool:
        """Check if session file exists"""
        return os.path.exists(path)

    def energy_fraction_remaining(self) -> float:
        """What fraction of initial energy remains?"""
        if self.initial_energy <= 0:
            return 0.0
        return self.energy_remaining / self.initial_energy

    def summary(self) -> str:
        lines = [
            f"Session: {self.session_id}",
            f"  Game: {self.game_name}",
            f"  Phase: {self.phase.value}",
            f"  Rounds: {self.current_round}",
            f"  Solved: {self.total_solved:,}",
            f"  Explored: {self.total_explored:,}",
            f"  Closures: {self.total_closures:,}",
            f"  Interiors: {self.total_interiors:,}",
            f"  Energy: {self.energy_remaining:.1f} / {self.initial_energy:.1f}",
            f"  Pending seeds: {len(self.pending_seeds)}",
        ]
        return "\n".join(lines)

    def record_round(self, stats: RoundStats):
        """Record a completed round"""
        stats.end_time = time.time()
        self.rounds.append(stats)
        self.current_round += 1

        # Update totals
        self.total_solved += stats.states_solved
        self.total_explored += stats.states_explored
        self.total_closures += stats.closures_found
        self.total_connections += stats.connections_found
        self.total_interiors += stats.interiors_found

        # Update energy tracking
        self.energy_remaining = stats.energy_end

    def detect_phase(self) -> SessionPhase:
        """
        Detect current phase from closure conditions.

        Physics interpretation:
            Phase is an EMERGENT property, not a control variable.
            We observe the system's behavior to determine what phase it's in.
        """
        if not self.rounds:
            return SessionPhase.INIT

        recent = self.rounds[-1] if self.rounds else None

        if recent is None:
            return SessionPhase.INIT

        # If many closures are forming, we're crystallizing
        if recent.closures_found > 10:
            return SessionPhase.CRYSTAL

        # If we found spines (fast paths), we're in lightning
        if recent.spines_found > 0 and recent.iterations < 5:
            return SessionPhase.LIGHTNING

        # If energy is depleted, we're complete
        if self.energy_fraction_remaining() < 0.01:
            return SessionPhase.COMPLETE

        # Default: wave expansion
        return SessionPhase.WAVE


def create_session(
    session_id: str,
    game_name: str,
    energy: float = 1000.0
) -> SessionState:
    """Create a new session with given initial energy"""
    return SessionState(
        session_id=session_id,
        game_name=game_name,
        initial_energy=energy,
        energy_remaining=energy,
        total_budget=energy,  # Legacy compatibility
    )


def load_or_create_session(
    path: str,
    session_id: str,
    game_name: str,
    energy: float = 1000.0
) -> SessionState:
    """Load existing session or create new one"""
    if SessionState.exists(path):
        session = SessionState.load(path)
        print(f"Loaded existing session: {session.session_id}")
        return session
    else:
        session = create_session(session_id, game_name, energy)
        print(f"Created new session: {session.session_id}")
        return session
