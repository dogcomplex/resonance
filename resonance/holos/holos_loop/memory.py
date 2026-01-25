"""
holos_loop/memory.py - Memory Management Utilities

Provides memory tracking and limits for long-running wave propagation.

Physics interpretation:
    Memory is a FINITE RESOURCE, like energy in a physical system.
    When memory is exhausted, the system must either:
    1. Stop (equilibrium reached due to resource constraint)
    2. Compress/forget (lossy compression, entropy increase)
    3. Checkpoint (save state, restart fresh)

    This is analogous to thermodynamic limits - you can't do infinite
    computation with finite resources.
"""

import os
import sys
from typing import Optional, Callable
from dataclasses import dataclass

# Try to import psutil for accurate memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class MemoryConfig:
    """
    Configuration for memory management.

    Attributes:
        max_memory_mb: Maximum process memory before stopping
        warning_threshold: Fraction of max at which to warn (0.0-1.0)
        critical_threshold: Fraction of max at which to take action (0.0-1.0)
        max_frontier_size: Hard cap on frontier dictionary size
        max_solved_size: Hard cap on solved dictionary size
        gc_interval: How often to run garbage collection (iterations)
    """
    max_memory_mb: float = 4000.0
    warning_threshold: float = 0.7
    critical_threshold: float = 0.9
    max_frontier_size: int = 2_000_000
    max_solved_size: int = 10_000_000
    gc_interval: int = 100


class MemoryTracker:
    """
    Tracks memory usage and enforces limits.

    Physics interpretation:
        This is like a GOVERNOR on an engine - it prevents runaway
        resource consumption that would crash the system.

        The thresholds create a "pressure" that increases as memory fills:
        - Below warning: Free expansion
        - Warning to critical: Increased damping, slower expansion
        - Above critical: System halt, checkpoint required
    """

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self._process = None
        self._peak_mb = 0.0
        self._samples = []

        if HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())

    def current_mb(self) -> float:
        """Get current process memory in MB"""
        if self._process is not None:
            try:
                info = self._process.memory_info()
                mb = info.rss / (1024 * 1024)
                self._peak_mb = max(self._peak_mb, mb)
                return mb
            except Exception:
                pass

        # Fallback: estimate from sys.getsizeof (very rough)
        return 0.0

    def peak_mb(self) -> float:
        """Get peak memory usage"""
        self.current_mb()  # Update peak
        return self._peak_mb

    def fraction_used(self) -> float:
        """Get fraction of max memory used"""
        return self.current_mb() / self.config.max_memory_mb

    def is_warning(self) -> bool:
        """Are we above warning threshold?"""
        return self.fraction_used() >= self.config.warning_threshold

    def is_critical(self) -> bool:
        """Are we above critical threshold?"""
        return self.fraction_used() >= self.config.critical_threshold

    def should_stop(self) -> bool:
        """Should we stop due to memory pressure?"""
        return self.fraction_used() >= self.config.critical_threshold

    def check_frontier_size(self, size: int) -> bool:
        """Is frontier size within limits?"""
        return size <= self.config.max_frontier_size

    def check_solved_size(self, size: int) -> bool:
        """Is solved dict size within limits?"""
        return size <= self.config.max_solved_size

    def memory_pressure(self) -> float:
        """
        Get memory pressure as a 0-1 value.

        Can be used to scale expansion rate, damping, etc.

        Physics interpretation:
            Pressure increases as we approach the limit,
            like gas pressure in a finite container.
        """
        frac = self.fraction_used()
        if frac < self.config.warning_threshold:
            return 0.0
        elif frac < self.config.critical_threshold:
            # Linear ramp from warning to critical
            range_size = self.config.critical_threshold - self.config.warning_threshold
            return (frac - self.config.warning_threshold) / range_size
        else:
            return 1.0

    def status(self) -> str:
        """Get human-readable status"""
        mb = self.current_mb()
        frac = self.fraction_used()
        pressure = self.memory_pressure()

        if self.is_critical():
            level = "CRITICAL"
        elif self.is_warning():
            level = "WARNING"
        else:
            level = "OK"

        return f"Memory: {mb:.0f}/{self.config.max_memory_mb:.0f} MB ({frac*100:.1f}%) [{level}] pressure={pressure:.2f}"

    def suggest_action(self) -> str:
        """Suggest what to do based on memory state"""
        if self.is_critical():
            return "STOP: Save checkpoint and halt"
        elif self.is_warning():
            return "COMPRESS: Consider clearing caches, reducing frontier"
        else:
            return "CONTINUE: Memory is healthy"


# ============================================================
# MEMORY-AWARE EXPANSION CONTROL
# ============================================================

def create_memory_damping(tracker: MemoryTracker) -> Callable[[float], float]:
    """
    Create a damping function that increases with memory pressure.

    Returns a function that takes base_damping and returns adjusted damping.

    Physics interpretation:
        As memory pressure increases, we increase damping to slow expansion.
        This is like viscosity increasing as a fluid approaches a phase transition.
    """
    def damping_fn(base_damping: float) -> float:
        pressure = tracker.memory_pressure()
        # Damping increases linearly with pressure
        # At max pressure, damping is 3x base
        return base_damping * (1.0 + 2.0 * pressure)

    return damping_fn


def create_memory_gate(tracker: MemoryTracker) -> Callable[[], bool]:
    """
    Create a gate function that returns False when memory is critical.

    Use to guard expansion operations.
    """
    def gate_fn() -> bool:
        return not tracker.should_stop()

    return gate_fn


# ============================================================
# GLOBAL TRACKER
# ============================================================

_global_tracker: Optional[MemoryTracker] = None


def get_memory_tracker(config: MemoryConfig = None) -> MemoryTracker:
    """Get or create global memory tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MemoryTracker(config)
    return _global_tracker


def memory_mb() -> float:
    """Quick access to current memory in MB"""
    return get_memory_tracker().current_mb()


def memory_ok() -> bool:
    """Quick check if memory is below critical threshold"""
    return not get_memory_tracker().should_stop()


# ============================================================
# CONTEXT MANAGER
# ============================================================

class MemoryBudget:
    """
    Context manager for memory-bounded operations.

    Usage:
        with MemoryBudget(max_mb=2000) as budget:
            while budget.ok():
                # do expansion
                pass
            if budget.exceeded:
                # handle checkpoint
                pass

    Physics interpretation:
        This is like a "fuel tank" for computation.
        When the tank is empty, the engine stops.
    """

    def __init__(self, max_mb: float = 4000.0, warn_mb: float = None):
        self.max_mb = max_mb
        self.warn_mb = warn_mb or (max_mb * 0.7)
        self.tracker = MemoryTracker(MemoryConfig(
            max_memory_mb=max_mb,
            warning_threshold=self.warn_mb / max_mb,
        ))
        self.start_mb = 0.0
        self.exceeded = False
        self.warned = False

    def __enter__(self):
        self.start_mb = self.tracker.current_mb()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def ok(self) -> bool:
        """Check if we can continue"""
        if self.tracker.should_stop():
            self.exceeded = True
            return False
        if self.tracker.is_warning() and not self.warned:
            self.warned = True
            print(f"Memory warning: {self.tracker.status()}")
        return True

    def used_mb(self) -> float:
        """Memory used since entering context"""
        return self.tracker.current_mb() - self.start_mb

    def remaining_mb(self) -> float:
        """Estimated remaining before limit"""
        return self.max_mb - self.tracker.current_mb()
