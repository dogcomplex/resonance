"""
holos/c4_holos_solve.py - Connect4 Solver Using HOLOS Framework

This demonstrates solving Connect4 using the generalized HOLOSSolver,
proving that HOLOS has all the features from c4_crystal.py.

Expected to replicate c4_crystal.py's ~57.5 second solve time.

Key HOLOS features used:
- spine_as_boundary=True (crystallization from solved paths)
- Bidirectional lightning probes
- Equivalence class propagation
- Early termination via minimax
- Auto-generated backward seeds from terminals

Run: python -m holos.c4_holos_solve
"""

import os
import sys
import time
import pickle
import gzip
from typing import Dict, Set, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from holos.holos import HOLOSSolver, SeedPoint, SearchMode, GoalCondition
from holos.games.connect4 import Connect4Game, C4State, C4Value, C4Features
from holos.storage import Hologram


# ============================================================
# CONFIGURATION
# ============================================================

SAVE_DIR = Path("./c4_holos_state")
MAX_MEMORY_MB = 4000
MAX_ITERATIONS = 100
LIGHTNING_INTERVAL = 3  # More frequent lightning for faster spine finding


# ============================================================
# ENHANCED CONNECT4 GAME WITH BETTER SEED GENERATION
# ============================================================

class EnhancedConnect4Game(Connect4Game):
    """
    Enhanced Connect4 with better backward seed generation.

    The key insight from c4_crystal.py: we need DIVERSE terminal
    positions to seed the backward wave effectively.
    """

    def __init__(self, max_pieces: int = 42):
        super().__init__(max_pieces=max_pieces)
        self._terminal_cache: List[C4State] = []

    def generate_boundary_seeds(self, template: C4State, count: int = 500) -> List[C4State]:
        """
        Generate diverse terminal positions for backward seeding.

        Strategy: Play random games to completion, collect unique terminals.
        More seeds = better backward coverage = faster solution.
        """
        import random

        positions = []
        seen = set()

        # Try different random seeds for diversity
        attempts = count * 50  # Many attempts to get diverse positions

        for attempt in range(attempts):
            if len(positions) >= count:
                break

            # Play random game to terminal
            state = C4State()
            moves_played = []

            while not state.is_terminal():
                moves = state.get_valid_moves()
                if not moves:
                    break

                # Prefer center columns (better game dynamics)
                if attempt % 3 == 0:
                    moves.sort(key=lambda c: abs(c - 3))
                    col = moves[0]
                else:
                    col = random.choice(moves)

                state = state.play(col)
                moves_played.append(col)

            if state.is_terminal():
                h = hash(state)
                if h not in seen:
                    seen.add(h)
                    positions.append(state)

        # Also add some early terminals (quick wins)
        # These are closer to start and help bridge the gap
        for _ in range(count // 5):
            state = C4State()
            for _ in range(7):  # Minimum moves for a win
                if state.is_terminal():
                    break
                moves = state.get_valid_moves()
                if not moves:
                    break
                # Bias toward creating/blocking threats
                state = state.play(random.choice(moves))

            if state.is_terminal():
                h = hash(state)
                if h not in seen:
                    seen.add(h)
                    positions.append(state)

        print(f"Generated {len(positions)} unique terminal positions for backward seeding")

        # Cache for potential reuse
        self._terminal_cache = positions[:count]

        return positions[:count]

    def get_lightning_successors(self, state: C4State) -> List[Tuple[C4State, int]]:
        """
        Lightning successors: prioritize winning moves and center.
        """
        if state.is_terminal():
            return []

        successors = []
        for col in [3, 2, 4, 1, 5, 0, 6]:  # Center first
            if state.can_play(col):
                child = state.play(col)
                successors.append((child, col))

        # Sort: winning moves first, then by threats
        def score(item):
            child, col = item
            if child.check_win() == state.turn:
                return 1000  # Winning move!
            # Count threats created
            return 10 - abs(col - 3)

        successors.sort(key=score, reverse=True)
        return successors[:5]  # Limit branching for lightning

    def score_for_lightning(self, state: C4State, move: int) -> float:
        """Score moves for lightning prioritization"""
        child = state.play(move)
        if child and child.check_win() == state.turn:
            return 100.0  # Winning move
        return 10.0 - abs(move - 3)  # Prefer center


# ============================================================
# SOLVER WITH PROGRESS TRACKING
# ============================================================

@dataclass
class SolveProgress:
    """Track solving progress for checkpointing"""
    iteration: int = 0
    solved_count: int = 0
    start_solved: bool = False
    start_value: Optional[int] = None
    lightning_time: float = 0.0
    total_time: float = 0.0
    spines_found: int = 0
    connections: int = 0


def check_start_solved(solver: HOLOSSolver) -> Tuple[bool, Optional[int]]:
    """Check if the start position is solved"""
    start = C4State()
    h = hash(start)
    if h in solver.solved:
        return True, solver.solved[h].value if hasattr(solver.solved[h], 'value') else solver.solved[h]
    return False, None


def save_progress(solver: HOLOSSolver, progress: SolveProgress):
    """Save solver state for checkpointing"""
    SAVE_DIR.mkdir(exist_ok=True)

    data = {
        'solved': {h: (v.value if hasattr(v, 'value') else v) for h, v in solver.solved.items()},
        'forward_seen': solver.forward_seen,
        'backward_seen': solver.backward_seen,
        'stats': solver.stats,
        'phase_timing': solver.phase_timing,
        'progress': progress,
    }

    with gzip.open(SAVE_DIR / "state.pkl.gz", 'wb') as f:
        pickle.dump(data, f)

    print(f"  [Saved: {progress.solved_count:,} solved, iter {progress.iteration}]")


def load_progress(solver: HOLOSSolver) -> Optional[SolveProgress]:
    """Load solver state from checkpoint"""
    state_file = SAVE_DIR / "state.pkl.gz"
    if not state_file.exists():
        return None

    try:
        with gzip.open(state_file, 'rb') as f:
            data = pickle.load(f)

        # Restore solved states (convert back to C4Value)
        for h, v in data['solved'].items():
            solver.solved[h] = C4Value(v) if isinstance(v, int) else v

        solver.forward_seen = data['forward_seen']
        solver.backward_seen = data['backward_seen']
        solver.stats = data['stats']
        solver.phase_timing = data['phase_timing']

        progress = data['progress']
        print(f"Loaded checkpoint: {progress.solved_count:,} solved, iteration {progress.iteration}")
        return progress

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None


# ============================================================
# MAIN SOLVER
# ============================================================

def solve_connect4(max_iterations: int = MAX_ITERATIONS,
                   resume: bool = True) -> Tuple[bool, Optional[int], Dict]:
    """
    Solve Connect4 using HOLOS framework.

    Returns:
        (solved, value, stats) where:
        - solved: True if start position was solved
        - value: Game-theoretic value (+1 X wins, -1 O wins, 0 draw)
        - stats: Solver statistics
    """
    print("="*60)
    print("[C4] CONNECT-4 HOLOS SOLVER [C4]")
    print("Using HOLOSSolver with spine_as_boundary=True")
    print("="*60)

    # Create enhanced game interface
    game = EnhancedConnect4Game(max_pieces=42)

    # Create solver with spine-as-boundary enabled
    solver = HOLOSSolver(
        game,
        name="connect4_holos",
        max_memory_mb=MAX_MEMORY_MB,
        max_frontier_size=2_000_000,
        spine_as_boundary=True,  # Key insight from c4_crystal.py!
    )

    # Try to resume from checkpoint
    progress = None
    if resume:
        progress = load_progress(solver)

    if progress is None:
        progress = SolveProgress()

    # Check if already solved
    solved, value = check_start_solved(solver)
    if solved:
        print(f"\n*** Already solved! Value: {value}")
        return True, value, solver.stats

    start_time = time.time()

    # Create forward seed (start position)
    start_state = C4State()
    forward_seeds = [SeedPoint(start_state, SearchMode.LIGHTNING, priority=10, depth=20)]

    # Auto-generate backward seeds
    backward_seeds = None  # Let solver auto-generate from game.generate_boundary_seeds()

    print(f"\nStarting solve from iteration {progress.iteration}...")
    print(f"  spine_as_boundary: ENABLED")
    print(f"  lightning_interval: {LIGHTNING_INTERVAL}")
    print(f"  max_iterations: {max_iterations}")

    # Run main solve loop
    try:
        hologram = solver.solve(
            forward_seeds=forward_seeds,
            backward_seeds=backward_seeds,
            max_iterations=max_iterations - progress.iteration,
            lightning_interval=LIGHTNING_INTERVAL,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        progress.iteration = max_iterations
        progress.solved_count = len(solver.solved)
        progress.total_time = time.time() - start_time
        save_progress(solver, progress)
        raise

    total_time = time.time() - start_time

    # Check final result
    solved, value = check_start_solved(solver)

    # Update progress
    progress.solved_count = len(solver.solved)
    progress.start_solved = solved
    progress.start_value = value
    progress.total_time = total_time
    progress.spines_found = solver.stats.get('spines_found', 0)
    progress.connections = len(solver.connections)
    progress.lightning_time = solver.phase_timing.get('lightning_time', 0)

    # Save final state
    save_progress(solver, progress)

    # Print results
    print("\n" + "="*60)
    print("SOLVE COMPLETE")
    print("="*60)

    print(f"\n[STATS] Statistics:")
    print(f"  Total solved: {len(solver.solved):,}")
    print(f"  Forward seen: {len(solver.forward_seen):,}")
    print(f"  Backward seen: {len(solver.backward_seen):,}")
    print(f"  Spines found: {solver.stats.get('spines_found', 0):,}")
    print(f"  Connections: {len(solver.connections):,}")
    print(f"  Equiv shortcuts: {solver.stats.get('equiv_shortcuts', 0):,}")
    print(f"  Equiv propagated: {solver.stats.get('equiv_propagated', 0):,}")
    print(f"  Spine seeds added: {solver.stats.get('spine_seeds_added', 0):,}")

    print(f"\n[TIME]  Timing:")
    print(f"  Lightning: {solver.phase_timing.get('lightning_time', 0):.1f}s")
    print(f"  Wave: {solver.phase_timing.get('wave_time', 0):.1f}s")
    print(f"  Crystal: {solver.phase_timing.get('crystal_time', 0):.1f}s")
    print(f"  Propagation: {solver.phase_timing.get('propagation_time', 0):.1f}s")
    print(f"  Total: {total_time:.1f}s")

    if solved:
        outcome = {1: "FIRST PLAYER (X) WINS", 0: "DRAW", -1: "SECOND PLAYER (O) WINS"}
        print(f"\n*** CONNECT-4 SOLVED: {outcome.get(value, f'Value={value}')} ***")

        # Try to show principal variation
        print(f"\n[PV] Attempting to extract principal variation...")
        pv = extract_principal_variation(solver, game)
        if pv:
            print(f"  Optimal moves: {pv}")
    else:
        print(f"\n[WARN]  Start position not yet solved.")
        print(f"  Run more iterations or check backward seed coverage.")

    return solved, value, solver.stats


def extract_principal_variation(solver: HOLOSSolver, game: Connect4Game,
                                 max_moves: int = 20) -> List[int]:
    """Extract the principal variation (optimal play sequence)"""
    pv = []
    state = C4State()

    for _ in range(max_moves):
        h = hash(state)
        if h not in solver.solved:
            break
        if state.is_terminal():
            break

        # Find best child
        best_move = None
        best_value = None

        for child, col in game.get_successors(state):
            ch = hash(child)
            if ch not in solver.solved:
                continue

            cv = solver.solved[ch]
            if hasattr(cv, 'value'):
                cv = cv.value

            if best_value is None:
                best_value = cv
                best_move = col
            elif state.turn == 'X' and cv > best_value:
                best_value = cv
                best_move = col
            elif state.turn == 'O' and cv < best_value:
                best_value = cv
                best_move = col

        if best_move is None:
            break

        pv.append(best_move)
        state = state.play(best_move)

    return pv


# ============================================================
# COMPRESSION STATISTICS
# ============================================================

def analyze_compression(solver: HOLOSSolver, game: Connect4Game):
    """Analyze compression potential of solved states"""
    from holos.compression import StateRepresentationComparer

    print("\n" + "="*60)
    print("COMPRESSION ANALYSIS")
    print("="*60)

    # Group by layer (piece count)
    by_layer: Dict[int, List[Tuple[int, Any]]] = {}

    for h, v in solver.solved.items():
        # We'd need the actual state to get piece count
        # For now, just count total
        pass

    # Basic stats
    total = len(solver.solved)
    print(f"\nTotal solved states: {total:,}")

    # Estimate storage
    # Each entry: hash (8 bytes) + value (1 byte) = 9 bytes
    raw_bytes = total * 9
    print(f"Raw storage: {raw_bytes:,} bytes ({raw_bytes/1024/1024:.1f} MB)")

    # With gzip (typical 8x compression)
    gzip_est = raw_bytes / 8
    print(f"Gzip estimate: {gzip_est:,.0f} bytes ({gzip_est/1024:.1f} KB)")

    # Equivalence class analysis
    equiv_classes = len(solver.equiv_classes)
    equiv_with_outcome = sum(1 for v in solver.equiv_outcomes.values() if v is not None)
    print(f"\nEquivalence classes: {equiv_classes:,}")
    print(f"  With known outcome: {equiv_with_outcome:,}")

    # Seed ratio (from stats)
    if 'equiv_shortcuts' in solver.stats and total > 0:
        seed_ratio = 1.0 - (solver.stats['equiv_shortcuts'] / total) if total > 0 else 1.0
        print(f"  Effective seed ratio: {seed_ratio:.2%}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve Connect4 with HOLOS")
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS,
                       help="Maximum iterations")
    parser.add_argument("--fresh", action="store_true",
                       help="Start fresh (ignore checkpoint)")
    parser.add_argument("--analyze", action="store_true",
                       help="Run compression analysis after solve")

    args = parser.parse_args()

    try:
        solved, value, stats = solve_connect4(
            max_iterations=args.iterations,
            resume=not args.fresh,
        )

        if args.analyze:
            # Would need to reload solver for this
            print("\n(Compression analysis requires solver state - skipped)")

        # Exit code
        sys.exit(0 if solved else 1)

    except KeyboardInterrupt:
        print("\nSolver interrupted. Progress saved.")
        sys.exit(130)
