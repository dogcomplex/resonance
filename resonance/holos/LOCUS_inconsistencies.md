# LOCUS Inconsistencies Report

**Generated**: 2026-01-23
**Updated**: 2026-01-23 (batched_solver removed, issues 4 and 9 resolved)
**Scope**: holos/ package analysis

---

## 1. GoalCondition Dual-Definition

**Severity**: Medium (API confusion)

- `holos.py` (lines 58-83): Defines `GoalCondition` directly in the engine
- `games/__init__.py` (line 43): Imports `GoalCondition` from `strategy.py`
- `__init__.py` (line 31): Exports from `holos.holos`

LOCUS.md describes GoalCondition as a "Layer 1/2 concept" but it lives in holos.py (THE ENGINE). Either it belongs in the engine or in the strategy layer, not both.

---

## 2. LOCUS.md Line Count Discrepancies

**Severity**: Low (documentation drift)

| File | Documented | Actual |
|------|-----------|--------|
| holos.py | ~1400 lines | ~1400 lines (accurate with osmosis) |
| __init__.py | ~65 lines | 64 lines |
| storage.py | 407 lines | 408 lines |

---

## ~~3. SearchMode.OSMOSIS Undocumented~~ RESOLVED

OSMOSIS mode is now documented in LOCUS.md (Update 11).

---

## ~~4. batched_solver.py Goal Parameter Unused~~ RESOLVED

batched_solver.py has been removed. Use the pattern in run_targeted_kqrr.py instead.

---

## 5. feature_success Placeholder

**Severity**: Low (stale code)

`session.py` line 106 declares `feature_success: Dict[Any, float]` but it's never populated. Documented in LOCUS.md as known issue but remains unfixed.

---

## 6. apply_move Return Type Inconsistency

**Severity**: Low (type safety)

`holos.py` lines 180-186:
```python
def apply_move(self, state: S, move: Any) -> S:
    ...
    return None  # Type says S, can return None
```

---

## 7. SpinePath.truncate Potential Off-by-One

**Severity**: Low (edge case)

`storage.py` lines 77-86: If `depth == len(self.checkpoints)` exactly, uses `self.end_hash`, but checkpoints might not include the end state.

---

## 8. estimate_compression Ignores num_seeds

**Severity**: Low (unused parameter)

`storage.py` lines 400-407:
```python
def estimate_compression(num_seeds: int, avg_depth: float) -> float:
    expansion_per_seed = 10 ** avg_depth
    return expansion_per_seed  # num_seeds never used
```

---

## ~~9. BatchedSolver Propagation Differs from HOLOSSolver~~ RESOLVED

batched_solver.py has been removed.

---

## 10. LOCUS.md Claims chess_targeted.py is DEPRECATED

**Severity**: Medium (documentation error)

LOCUS.md line 101-102 says:
```
└── chess_targeted.py           [DEPRECATED - ~350 lines]
    Note: Use GoalCondition from holos.py instead
```

But `chess_targeted.py` is **actively used** by:
- `full_search.py` - Uses `TargetedChessGame` for seed generation and material filtering
- `run_targeted_subprocess.py` - Uses `TargetedChessGame`
- `run_targeted_kqrr.py` - Uses `TargetedChessGame`

The file is NOT deprecated - it provides essential functionality for material-targeted searches that `GoalCondition` alone cannot provide (seed generation, material filtering in get_successors/get_predecessors).

---

## Summary

**Remaining Issues**:

**Medium Priority**:
1. GoalCondition dual-definition confusion
10. chess_targeted.py incorrectly marked as deprecated

**Low Priority**:
2. Line count discrepancies (minor)
5. feature_success placeholder
6. apply_move return type
7. SpinePath.truncate edge case
8. estimate_compression unused parameter
