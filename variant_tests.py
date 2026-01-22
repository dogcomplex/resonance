"""
Comprehensive Variant Testing

Test learners on multiple game variants to:
1. Verify no cheating (embedded game knowledge)
2. Measure generalization ability
3. Compare convergence rates

A learner that "knows" TicTacToe will fail on novel variants.
"""

import random
import sys
sys.path.insert(0, '/home/claude/locus')

from tictactoe import label_space
from collections import defaultdict


def make_variant_oracle(win_conditions):
    """Create an oracle with custom win conditions."""
    def oracle(board):
        for condition in win_conditions:
            if all(board[i] == '1' for i in condition):
                return 'win1'
        for condition in win_conditions:
            if all(board[i] == '2' for i in condition):
                return 'win2'
        if '0' not in board:
            return 'draw'
        count_1 = board.count('1')
        count_2 = board.count('2')
        if count_1 < count_2 or count_1 > count_2 + 1:
            return 'error'
        return 'ok'
    return oracle


# Game variants
VARIANTS = {
    'standard': [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6],              # Diagonals
    ],
    'no_diag': [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    ],
    'only_diag': [
        [0, 4, 8], [2, 4, 6],  # Only diagonals
    ],
    'corners': [
        [0, 2, 6], [0, 2, 8], [0, 6, 8], [2, 6, 8],  # Corner combinations
    ],
    'edges': [
        [0, 1, 2], [6, 7, 8],  # Top and bottom rows only
        [0, 3, 6], [2, 5, 8],  # Left and right columns only
    ],
    'l_shapes': [
        [0, 1, 3], [1, 2, 5], [3, 6, 7], [5, 7, 8],  # L shapes
    ],
}


def generate_boards(oracle, max_boards=5000):
    """Generate valid boards for a game variant."""
    boards = {}
    
    def explore(board='000000000'):
        if board in boards or len(boards) >= max_boards:
            return
        label = oracle(board)
        if label == 'error':
            return
        boards[board] = label
        if label == 'ok':
            count = board.count('0')
            turn = '1' if (9 - count) % 2 == 0 else '2'
            for i in range(9):
                if board[i] == '0':
                    new_board = board[:i] + turn + board[i+1:]
                    explore(new_board)
    
    explore()
    return boards


def test_on_variant(learner_class, variant_name, rounds=500, verbose=False, **kwargs):
    """Test a learner on a specific game variant."""
    oracle = make_variant_oracle(VARIANTS[variant_name])
    boards = generate_boards(oracle)
    board_list = list(boards.keys())
    
    learner = learner_class(
        num_outputs=5, 
        label_names=label_space,
        **kwargs
    )
    
    correct = 0
    per_label_correct = defaultdict(int)
    per_label_total = defaultdict(int)
    
    checkpoints = {}
    
    for i in range(rounds):
        board = random.choice(board_list)
        true_label = oracle(board)
        true_idx = label_space.index(true_label)
        
        pred_idx = learner.predict(board)
        
        if pred_idx == true_idx:
            correct += 1
            per_label_correct[true_label] += 1
        per_label_total[true_label] += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if (i + 1) in [10, 25, 50, 100, 200, 500]:
            checkpoints[i + 1] = correct / (i + 1)
            if verbose:
                print(f"  R{i+1:3d}: {correct/(i+1):.1%}")
    
    # Calculate per-label accuracy
    per_label_acc = {}
    for label in label_space:
        total = per_label_total[label]
        corr = per_label_correct[label]
        per_label_acc[label] = corr / total if total > 0 else 0.0
    
    return {
        'final_accuracy': correct / rounds,
        'checkpoints': checkpoints,
        'per_label': per_label_acc,
        'per_label_counts': dict(per_label_total),
    }


def comprehensive_test(learner_classes, variants=None, rounds=500, runs=1):
    """Test multiple learners on multiple variants."""
    if variants is None:
        variants = list(VARIANTS.keys())
    
    results = defaultdict(dict)
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Variant: {variant}")
        print(f"Win conditions: {VARIANTS[variant]}")
        print(f"{'='*60}")
        
        for learner_class in learner_classes:
            name = learner_class.__name__
            
            # Multiple runs for stability
            accs = []
            win_accs = []
            for run in range(runs):
                result = test_on_variant(learner_class, variant, rounds)
                accs.append(result['final_accuracy'])
                # Average win accuracy
                win_acc = (result['per_label'].get('win1', 0) + 
                          result['per_label'].get('win2', 0)) / 2
                win_accs.append(win_acc)
            
            mean_acc = sum(accs) / len(accs)
            mean_win = sum(win_accs) / len(win_accs)
            
            results[variant][name] = {
                'accuracy': mean_acc,
                'win_accuracy': mean_win,
                'checkpoints': result['checkpoints'],
            }
            
            print(f"  {name:30s}: {mean_acc:.1%} (wins: {mean_win:.1%})")
    
    return results


def print_comparison_table(results):
    """Print a comparison table."""
    variants = list(results.keys())
    learners = list(results[variants[0]].keys())
    
    print("\n" + "="*80)
    print("COMPARISON TABLE - Overall Accuracy")
    print("="*80)
    
    # Header
    print(f"\n{'Variant':<15}", end="")
    for learner in learners:
        print(f"{learner[:12]:>14}", end="")
    print()
    print("-" * (15 + 14 * len(learners)))
    
    for variant in variants:
        print(f"{variant:<15}", end="")
        for learner in learners:
            acc = results[variant][learner]['accuracy']
            print(f"{acc:>13.1%}", end="")
        print()
    
    print("\n" + "="*80)
    print("COMPARISON TABLE - Win Detection Accuracy")
    print("="*80)
    
    print(f"\n{'Variant':<15}", end="")
    for learner in learners:
        print(f"{learner[:12]:>14}", end="")
    print()
    print("-" * (15 + 14 * len(learners)))
    
    for variant in variants:
        print(f"{variant:<15}", end="")
        for learner in learners:
            win_acc = results[variant][learner]['win_accuracy']
            print(f"{win_acc:>13.1%}", end="")
        print()


def detect_cheating(results):
    """Analyze results for signs of cheating."""
    print("\n" + "="*80)
    print("CHEATING ANALYSIS")
    print("="*80)
    
    learners = list(results['standard'].keys())
    
    for learner in learners:
        std_acc = results['standard'][learner]['accuracy']
        std_win = results['standard'][learner]['win_accuracy']
        
        # Compare to novel variants
        novel_accs = []
        novel_wins = []
        for variant in ['corners', 'l_shapes']:
            if variant in results:
                novel_accs.append(results[variant][learner]['accuracy'])
                novel_wins.append(results[variant][learner]['win_accuracy'])
        
        if novel_accs:
            novel_avg = sum(novel_accs) / len(novel_accs)
            novel_win_avg = sum(novel_wins) / len(novel_wins)
            
            print(f"\n{learner}:")
            print(f"  Standard TicTacToe: {std_acc:.1%} (wins: {std_win:.1%})")
            print(f"  Novel variants avg: {novel_avg:.1%} (wins: {novel_win_avg:.1%})")
            
            drop = std_acc - novel_avg
            if drop > 0.15:
                print(f"  ⚠️  POSSIBLE CHEATING: {drop:.1%} drop on novel variants")
            elif drop > 0.05:
                print(f"  ⚡ Moderate drop: {drop:.1%} - may have some bias")
            else:
                print(f"  ✓ Generalizes well: only {drop:.1%} drop")


if __name__ == "__main__":
    from few_shot_algs.blind_learner import BlindLearner
    from few_shot_algs.hybrid_learner import HybridLearner, MultiSizeHybridLearner
    from few_shot_algs.hypothesis_learner import HypothesisEliminationLearner
    
    learners = [
        BlindLearner,
        HybridLearner,
        # MultiSizeHybridLearner,
        HypothesisEliminationLearner,
    ]
    
    results = comprehensive_test(
        learners,
        variants=['standard', 'no_diag', 'only_diag', 'corners', 'l_shapes'],
        rounds=500,
        runs=1
    )
    
    print_comparison_table(results)
    detect_cheating(results)
