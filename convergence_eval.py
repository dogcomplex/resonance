"""
Comprehensive Convergence Evaluation

Tests learner against:
1. True random oracles (no patterns - unlearnable)
2. Pattern-based random oracles (learnable patterns)
3. Structured oracles (TicTacToe variants)

Metrics:
- Raw accuracy (inflated by majority class)
- Balanced accuracy (mean of per-class)
- Weighted accuracy (rare classes count more)
- Macro F1

Convergence timeline at: 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, full
"""

import random
from collections import defaultdict
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional, Set
import math


LABELS = ['ok', 'winX', 'winO', 'draw', 'error']


# =============================================================================
# ORACLES
# =============================================================================

def enumerate_all_boards():
    return [''.join(c) for c in product('012', repeat=9)]


class TrueRandomOracle:
    """Truly random state->label mapping (unlearnable)."""
    
    def __init__(self, label_dist: Dict[str, float] = None, seed: int = None):
        if seed is not None:
            random.seed(seed)
        
        self.label_dist = label_dist or {
            'ok': 0.60, 'winX': 0.20, 'winO': 0.15, 'draw': 0.04, 'error': 0.01
        }
        
        self._all_states = {}
        self._states_by_label = {l: [] for l in LABELS}
        
        for board in enumerate_all_boards():
            r = random.random()
            cumsum = 0
            label = 'ok'
            for l, prob in self.label_dist.items():
                cumsum += prob
                if r <= cumsum:
                    label = l
                    break
            self._all_states[board] = label
            self._states_by_label[label].append(board)
        
        self._unseen = set(self._all_states.keys())
    
    def random_board(self, unique: bool = True) -> Optional[Tuple[str, str]]:
        if unique:
            if not self._unseen:
                return None
            board = random.choice(list(self._unseen))
            self._unseen.remove(board)
        else:
            board = random.choice(list(self._all_states.keys()))
        return (board, self._all_states[board])
    
    def reset(self):
        self._unseen = set(self._all_states.keys())
    
    @property
    def total_states(self):
        return len(self._all_states)


class PatternBasedRandomOracle:
    """Random patterns -> random labels (learnable)."""
    
    def __init__(self, n_patterns: int = 10, 
                 pattern_size_range: Tuple[int, int] = (2, 5),
                 seed: int = None):
        if seed is not None:
            random.seed(seed)
        
        self.patterns = []
        non_ok_labels = ['winX', 'winO', 'draw', 'error']
        
        for _ in range(n_patterns):
            size = random.randint(*pattern_size_range)
            positions = tuple(sorted(random.sample(range(9), size)))
            values = tuple(random.choice('012') for _ in range(size))
            label = random.choice(non_ok_labels)
            self.patterns.append((positions, values, label))
        
        self._all_states = {}
        self._states_by_label = {l: [] for l in LABELS}
        
        for board in enumerate_all_boards():
            label = 'ok'
            for pos, val, lbl in self.patterns:
                if all(board[p] == v for p, v in zip(pos, val)):
                    label = lbl
                    break
            self._all_states[board] = label
            self._states_by_label[label].append(board)
        
        self._unseen = set(self._all_states.keys())
    
    def random_board(self, unique: bool = True) -> Optional[Tuple[str, str]]:
        if unique:
            if not self._unseen:
                return None
            board = random.choice(list(self._unseen))
            self._unseen.remove(board)
        else:
            board = random.choice(list(self._all_states.keys()))
        return (board, self._all_states[board])
    
    def reset(self):
        self._unseen = set(self._all_states.keys())
    
    @property
    def total_states(self):
        return len(self._all_states)


class StructuredOracle:
    """Structured game rules (most learnable)."""
    
    STANDARD_LINES = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    
    def __init__(self, x_lines: List[Tuple] = None, 
                 o_lines: List[Tuple] = None,
                 mode: str = 'standard'):
        
        self.x_lines = [tuple(l) for l in (x_lines or self.STANDARD_LINES)]
        self.o_lines = [tuple(l) for l in (o_lines or self.x_lines)]
        self.mode = mode
        
        self._all_states = {}
        self._states_by_label = {l: [] for l in LABELS}
        
        for board in enumerate_all_boards():
            label = self._compute_label(board)
            if label == 'error' and mode == 'standard':
                continue
            self._all_states[board] = label
            self._states_by_label[label].append(board)
        
        self._unseen = set(self._all_states.keys())
    
    def _is_valid_parity(self, board: str) -> bool:
        x, o = board.count('1'), board.count('2')
        return x == o or x == o + 1
    
    def _check_win(self, board: str, player: str, lines: List[Tuple]) -> bool:
        val = '1' if player == 'X' else '2'
        return any(all(board[i] == val for i in line) for line in lines)
    
    def _compute_label(self, board: str) -> str:
        if not self._is_valid_parity(board):
            return 'error'
        
        x_wins = self._check_win(board, 'X', self.x_lines)
        o_wins = self._check_win(board, 'O', self.o_lines)
        
        if x_wins and o_wins:
            return 'error'
        if x_wins:
            return 'winX'
        if o_wins:
            return 'winO'
        if '0' not in board:
            return 'draw'
        return 'ok'
    
    def random_board(self, unique: bool = True) -> Optional[Tuple[str, str]]:
        if unique:
            if not self._unseen:
                return None
            board = random.choice(list(self._unseen))
            self._unseen.remove(board)
        else:
            board = random.choice(list(self._all_states.keys()))
        return (board, self._all_states[board])
    
    def reset(self):
        self._unseen = set(self._all_states.keys())
    
    @property
    def total_states(self):
        return len(self._all_states)


# =============================================================================
# LEARNER (General Pattern Learner)
# =============================================================================

class GeneralPatternLearner:
    """
    General pattern learner that can discover arbitrary patterns.
    
    Not limited to homogeneous patterns!
    """
    
    def __init__(self, 
                 pattern_sizes: List[int] = None,
                 precision_threshold: float = 0.95,
                 min_support: int = 3):
        
        self.pattern_sizes = pattern_sizes or [2, 3, 4]
        self.precision_threshold = precision_threshold
        self.min_support = min_support
        
        # Pattern stats: (positions, values) -> {label -> count}
        self.pattern_counts = defaultdict(int)
        self.pattern_labels = defaultdict(lambda: defaultdict(int))
        
        # Confirmed rules: (positions, values) -> label
        self.confirmed_rules = {}
        
        self.label_counts = defaultdict(int)
        self.observations = 0
    
    def _is_valid_parity(self, board: str) -> bool:
        x, o = board.count('1'), board.count('2')
        return x == o or x == o + 1
    
    def _get_patterns(self, board: str):
        """Generate all sub-patterns of the board."""
        for size in self.pattern_sizes:
            for positions in combinations(range(9), size):
                values = tuple(board[p] for p in positions)
                yield (positions, values)
    
    def observe(self, board: str, label: str):
        self.observations += 1
        self.label_counts[label] += 1
        
        for pattern in self._get_patterns(board):
            self.pattern_counts[pattern] += 1
            self.pattern_labels[pattern][label] += 1
            
            total = self.pattern_counts[pattern]
            
            if total >= self.min_support:
                # Check for high-precision pattern
                for lbl, count in self.pattern_labels[pattern].items():
                    precision = count / total
                    if precision >= self.precision_threshold and lbl != 'ok':
                        self.confirmed_rules[pattern] = lbl
                        break
                else:
                    # No longer high precision
                    if pattern in self.confirmed_rules:
                        del self.confirmed_rules[pattern]
    
    def predict(self, board: str) -> str:
        # Check confirmed rules (most specific first)
        best_match = None
        best_size = 0
        
        for pattern in self._get_patterns(board):
            if pattern in self.confirmed_rules:
                positions, values = pattern
                if len(positions) > best_size:
                    best_size = len(positions)
                    best_match = self.confirmed_rules[pattern]
        
        if best_match:
            return best_match
        
        # Check parity for error
        if not self._is_valid_parity(board):
            return 'error'
        
        # Default
        if '0' not in board:
            return 'draw'
        
        return 'ok'


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(predictions: List[str], 
                    true_labels: List[str]) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    per_class = {l: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0} for l in LABELS}
    
    correct = 0
    for pred, true in zip(predictions, true_labels):
        if pred == true:
            correct += 1
            per_class[true]['tp'] += 1
        else:
            per_class[pred]['fp'] += 1
            per_class[true]['fn'] += 1
        per_class[true]['total'] += 1
    
    # Raw accuracy
    raw_acc = correct / len(predictions) if predictions else 0
    
    # Per-class accuracy
    class_accs = {}
    for label in LABELS:
        total = per_class[label]['total']
        if total > 0:
            class_accs[label] = per_class[label]['tp'] / total
    
    # Balanced accuracy
    balanced_acc = sum(class_accs.values()) / len(class_accs) if class_accs else 0
    
    # Weighted accuracy (inverse frequency)
    total_samples = len(predictions)
    weighted_acc = 0
    weight_sum = 0
    for label, acc in class_accs.items():
        freq = per_class[label]['total'] / total_samples
        weight = 1.0 / (freq + 0.01)
        weighted_acc += acc * weight
        weight_sum += weight
    weighted_acc = weighted_acc / weight_sum if weight_sum > 0 else 0
    
    # Macro F1
    f1_scores = []
    for label in LABELS:
        tp = per_class[label]['tp']
        fp = per_class[label]['fp']
        fn = per_class[label]['fn']
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if prec + rec > 0:
            f1_scores.append(2 * prec * rec / (prec + rec))
    
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return {
        'raw': raw_acc,
        'balanced': balanced_acc,
        'weighted': weighted_acc,
        'f1': macro_f1,
        'per_class': class_accs,
    }


# =============================================================================
# CONVERGENCE EVALUATION
# =============================================================================

def evaluate_convergence(oracle, learner_class, learner_kwargs: dict = None,
                        checkpoints: List[int] = None,
                        n_trials: int = 3) -> Dict:
    """
    Evaluate learner convergence over time.
    
    Returns metrics at each checkpoint, averaged over trials.
    """
    learner_kwargs = learner_kwargs or {}
    checkpoints = checkpoints or [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    
    # Filter checkpoints to max states
    max_states = oracle.total_states
    checkpoints = [cp for cp in checkpoints if cp <= max_states] + [max_states]
    
    results = {cp: [] for cp in checkpoints}
    
    for trial in range(n_trials):
        oracle.reset()
        learner = learner_class(**learner_kwargs)
        
        # Collect all observations
        all_preds = []
        all_true = []
        
        obs_count = 0
        while True:
            result = oracle.random_board(unique=True)
            if result is None:
                break
            
            board, true_label = result
            pred = learner.predict(board)
            
            all_preds.append(pred)
            all_true.append(true_label)
            
            learner.observe(board, true_label)
            obs_count += 1
            
            if obs_count in checkpoints:
                metrics = compute_metrics(all_preds, all_true)
                results[obs_count].append(metrics)
    
    # Average over trials
    avg_results = {}
    for cp in checkpoints:
        if results[cp]:
            avg_results[cp] = {
                'raw': sum(r['raw'] for r in results[cp]) / len(results[cp]),
                'balanced': sum(r['balanced'] for r in results[cp]) / len(results[cp]),
                'weighted': sum(r['weighted'] for r in results[cp]) / len(results[cp]),
                'f1': sum(r['f1'] for r in results[cp]) / len(results[cp]),
            }
    
    return avg_results


def run_full_evaluation():
    """Run comprehensive evaluation across oracle types."""
    print("="*70)
    print("COMPREHENSIVE CONVERGENCE EVALUATION")
    print("="*70)
    
    # Oracle configurations
    oracles = [
        ("TrueRandom", TrueRandomOracle, {'seed': 42}),
        ("Pattern10", PatternBasedRandomOracle, {'n_patterns': 10, 'seed': 42}),
        ("Pattern50", PatternBasedRandomOracle, {'n_patterns': 50, 'seed': 42}),
        ("Structured", StructuredOracle, {}),
    ]
    
    learner_kwargs = {
        'pattern_sizes': [2, 3, 4],
        'precision_threshold': 0.95,
        'min_support': 3,
    }
    
    checkpoints = [50, 100, 200, 500, 1000, 2000, 5000]
    
    all_results = {}
    
    for name, OracleClass, oracle_kwargs in oracles:
        print(f"\n--- {name} ---")
        oracle = OracleClass(**oracle_kwargs)
        
        # Show distribution
        dist = {l: len(oracle._states_by_label[l]) for l in LABELS}
        total = sum(dist.values())
        print(f"Distribution: " + ", ".join(f"{l}:{dist[l]/total:.0%}" for l in LABELS if dist[l] > 0))
        
        results = evaluate_convergence(
            oracle, GeneralPatternLearner, learner_kwargs,
            checkpoints=checkpoints, n_trials=3
        )
        
        all_results[name] = results
    
    # Print comparison table
    print("\n" + "="*70)
    print("RAW ACCURACY (inflated by majority class)")
    print("="*70)
    print(f"{'Oracle':<15}", end="")
    for cp in checkpoints:
        print(f" @{cp:>4}", end="")
    print()
    print("-"*70)
    for name in all_results:
        print(f"{name:<15}", end="")
        for cp in checkpoints:
            if cp in all_results[name]:
                print(f" {all_results[name][cp]['raw']:>5.1%}", end="")
            else:
                print(f" {'N/A':>5}", end="")
        print()
    
    print("\n" + "="*70)
    print("BALANCED ACCURACY (mean of per-class)")
    print("="*70)
    print(f"{'Oracle':<15}", end="")
    for cp in checkpoints:
        print(f" @{cp:>4}", end="")
    print()
    print("-"*70)
    for name in all_results:
        print(f"{name:<15}", end="")
        for cp in checkpoints:
            if cp in all_results[name]:
                print(f" {all_results[name][cp]['balanced']:>5.1%}", end="")
            else:
                print(f" {'N/A':>5}", end="")
        print()
    
    print("\n" + "="*70)
    print("WEIGHTED ACCURACY (rare classes count more)")
    print("="*70)
    print(f"{'Oracle':<15}", end="")
    for cp in checkpoints:
        print(f" @{cp:>4}", end="")
    print()
    print("-"*70)
    for name in all_results:
        print(f"{name:<15}", end="")
        for cp in checkpoints:
            if cp in all_results[name]:
                print(f" {all_results[name][cp]['weighted']:>5.1%}", end="")
            else:
                print(f" {'N/A':>5}", end="")
        print()


if __name__ == "__main__":
    run_full_evaluation()
