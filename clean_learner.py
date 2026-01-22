"""
Clean Pattern Learner - NO implicit game knowledge

All game-specific details must be passed in via GameConfig.
Nothing is hardcoded about:
- Board size
- Cell symbols
- Pattern sizes
- Win conditions
- Turn order
"""

from collections import defaultdict
from itertools import combinations, product
from typing import Optional, Tuple, List, Set, Dict, Any, Callable
import random


class GameConfig:
    """
    Complete game configuration.
    
    User MUST provide all of these - no defaults that assume any specific game.
    """
    
    def __init__(self, 
                 board_size: int,
                 symbols: List[str],
                 empty_symbol: str,
                 pattern_sizes: List[int]):
        """
        Args:
            board_size: Number of cells (e.g., 9 for 3x3, 16 for 4x4)
            symbols: All valid cell values (e.g., ['0', '1', '2'])
            empty_symbol: Which symbol means empty (e.g., '0')
            pattern_sizes: Sizes to search (e.g., [3] for 3-in-a-row)
        """
        self.board_size = board_size
        self.symbols = symbols
        self.empty_symbol = empty_symbol
        self.pattern_sizes = pattern_sizes
        
        # Derived
        self.player_symbols = [s for s in symbols if s != empty_symbol]


class CleanOracle:
    """
    Oracle that generates observations from a label function.
    
    NO hardcoded game rules - the label_fn defines what the game is.
    """
    
    def __init__(self, 
                 config: GameConfig, 
                 label_fn: Callable[[str], Optional[Any]], 
                 seed: int = None):
        """
        Args:
            config: Game configuration
            label_fn: Function(board_str) -> label or None if invalid
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        self.config = config
        self.label_fn = label_fn
        
        # Build all valid states
        self._all: Dict[str, Any] = {}
        for board_tuple in product(config.symbols, repeat=config.board_size):
            board = ''.join(board_tuple)
            label = label_fn(board)
            if label is not None:
                self._all[board] = label
        
        self._order = list(self._all.keys())
        random.shuffle(self._order)
        self._idx = 0
    
    def next(self) -> Optional[Tuple[str, Any]]:
        """Get next random observation."""
        if self._idx >= len(self._order):
            return None
        board = self._order[self._idx]
        self._idx += 1
        return (board, self._all[board])
    
    def query_pattern(self, positions: Tuple[int, ...], 
                      symbol: str) -> List[Tuple[str, Any]]:
        """Query all boards where positions have symbol."""
        return [(b, l) for b, l in self._all.items()
                if all(b[p] == symbol for p in positions)]
    
    def reset(self, seed: int = None):
        """Reset for new evaluation run."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._order)
        self._idx = 0
    
    @property
    def total_states(self) -> int:
        return len(self._all)
    
    @property
    def label_distribution(self) -> Dict[Any, int]:
        """Get distribution of labels."""
        dist = defaultdict(int)
        for label in self._all.values():
            dist[label] += 1
        return dict(dist)


class CleanPatternLearner:
    """
    Pattern learner with NO implicit game knowledge.
    
    Learns: pattern -> label associations
    
    Does NOT assume anything about:
    - Board size (from config)
    - Symbol meanings (from config)
    - Pattern sizes (from config)
    - Turn order (not used)
    - Win conditions (discovered)
    """
    
    def __init__(self, 
                 config: GameConfig, 
                 precision_threshold: float = 0.95,
                 min_support: int = 3):
        """
        Args:
            config: Game configuration
            precision_threshold: Required precision to confirm a rule
            min_support: Minimum observations to confirm a rule
        """
        self.config = config
        self.precision = precision_threshold
        self.min_support = min_support
        
        # Statistics: (positions, symbol) -> {label: count}
        self.pattern_stats: Dict[Tuple, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.pattern_counts: Dict[Tuple, int] = defaultdict(int)
        
        # Confirmed rules
        self.rules: Dict[Tuple, Any] = {}
        
        self.observations = 0
    
    def _extract_patterns(self, board: str):
        """Extract all homogeneous patterns from board."""
        for size in self.config.pattern_sizes:
            for positions in combinations(range(self.config.board_size), size):
                values = [board[p] for p in positions]
                first = values[0]
                
                # Check for homogeneous non-empty pattern
                if first != self.config.empty_symbol and all(v == first for v in values):
                    yield (positions, first)
    
    def observe(self, board: str, label: Any):
        """Learn from an observation."""
        self.observations += 1
        
        for positions, symbol in self._extract_patterns(board):
            key = (positions, symbol)
            self.pattern_stats[key][label] += 1
            self.pattern_counts[key] += 1
            
            total = self.pattern_counts[key]
            if total >= self.min_support:
                best_label = max(self.pattern_stats[key], 
                                key=self.pattern_stats[key].get)
                best_count = self.pattern_stats[key][best_label]
                
                if best_count / total >= self.precision:
                    self.rules[key] = best_label
                else:
                    self.rules.pop(key, None)
    
    def predict(self, board: str, default_label: Any = None) -> Any:
        """Predict label for board."""
        for positions, symbol in self._extract_patterns(board):
            key = (positions, symbol)
            if key in self.rules:
                return self.rules[key]
        return default_label
    
    def get_stats(self) -> Dict:
        return {
            'observations': self.observations,
            'patterns_tracked': len(self.pattern_counts),
            'rules_confirmed': len(self.rules),
        }
    
    def get_rules_for_label(self, label: Any) -> List[Tuple[Tuple, str]]:
        """Get all rules that predict a specific label."""
        return [(pos, sym) for (pos, sym), lbl in self.rules.items() if lbl == label]


class CleanActiveLearner:
    """
    Active pattern learner with NO implicit game knowledge.
    
    Discovers patterns by querying - no assumed candidates.
    """
    
    def __init__(self, 
                 config: GameConfig,
                 precision_threshold: float = 0.95):
        self.config = config
        self.precision = precision_threshold
        
        # Discovered patterns: symbol -> set of position tuples
        self.patterns: Dict[str, Set[Tuple]] = defaultdict(set)
        self.queries = 0
    
    def discover(self, 
                 oracle: CleanOracle, 
                 target_labels: List[Any],
                 max_queries: int = 1000) -> int:
        """
        Discover patterns that produce target labels.
        
        Args:
            oracle: Query oracle
            target_labels: Labels to search for
            max_queries: Query budget
        
        Returns:
            Number of queries used
        """
        for size in self.config.pattern_sizes:
            for positions in combinations(range(self.config.board_size), size):
                if self.queries >= max_queries:
                    return self.queries
                
                for symbol in self.config.player_symbols:
                    results = oracle.query_pattern(positions, symbol)
                    self.queries += 1
                    
                    if not results:
                        continue
                    
                    labels = [l for _, l in results]
                    for target in target_labels:
                        if labels.count(target) / len(labels) >= self.precision:
                            self.patterns[symbol].add(positions)
                            break
        
        return self.queries
    
    def predict(self, board: str, 
                label_map: Dict[str, Any],
                default_label: Any = None) -> Any:
        """
        Predict label using discovered patterns.
        
        Args:
            board: Board state
            label_map: symbol -> label (e.g., {'1': 'winX'})
            default_label: Default if no match
        """
        for symbol, patterns in self.patterns.items():
            for positions in patterns:
                if all(board[p] == symbol for p in positions):
                    return label_map.get(symbol, default_label)
        return default_label


# ============================================================================
# Utility functions
# ============================================================================

def compute_accuracy(predictions: List[Any], 
                     true_labels: List[Any],
                     per_class: bool = True) -> Dict:
    """Compute accuracy metrics."""
    results = {'total': 0, 'correct': 0}
    
    if per_class:
        class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for pred, true in zip(predictions, true_labels):
            class_stats[true]['total'] += 1
            if pred == true:
                class_stats[true]['correct'] += 1
        results['per_class'] = {
            k: v['correct'] / v['total'] if v['total'] > 0 else 0
            for k, v in class_stats.items()
        }
    
    results['total'] = len(predictions)
    results['correct'] = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    return results
