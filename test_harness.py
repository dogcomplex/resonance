"""
Test Harness for Few-Shot Learners

Evaluates learners on various game variants with detailed metrics.
"""

import random
import time
from typing import List, Dict, Callable, Type, Any
from collections import defaultdict
import sys

sys.path.insert(0, '/home/claude/locus')


def test_learner(learner_class: Type, oracle: Callable, board_generator: Callable,
                 label_space: List[str], rounds: int = 500, 
                 checkpoints: List[int] = None, verbose: bool = True,
                 **learner_kwargs) -> Dict[str, Any]:
    """
    Test a learner on a game.
    
    Returns detailed metrics including accuracy over time.
    """
    if checkpoints is None:
        checkpoints = [10, 25, 50, 100, 200, 300, 500]
    
    learner = learner_class(
        num_outputs=len(label_space),
        label_names=label_space,
        **learner_kwargs
    )
    
    correct = 0
    correct_at_checkpoint = {}
    per_label_correct = defaultdict(int)
    per_label_total = defaultdict(int)
    
    start_time = time.time()
    
    for i in range(rounds):
        board = board_generator()
        true_label = oracle(board)
        true_idx = label_space.index(true_label)
        
        pred_idx = learner.predict(board)
        
        is_correct = pred_idx == true_idx
        if is_correct:
            correct += 1
            per_label_correct[true_label] += 1
        per_label_total[true_label] += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if (i + 1) in checkpoints:
            acc = correct / (i + 1)
            correct_at_checkpoint[i + 1] = acc
            if verbose:
                stats = learner.get_stats()
                print(f"  R{i+1:3d}: {acc:.1%} | rules={stats.get('rules', 0)}")
    
    runtime = time.time() - start_time
    
    # Per-label accuracy
    per_label_acc = {}
    for label in label_space:
        total = per_label_total[label]
        corr = per_label_correct[label]
        per_label_acc[label] = corr / total if total > 0 else 0.0
    
    return {
        'final_accuracy': correct / rounds,
        'accuracy_over_time': correct_at_checkpoint,
        'per_label_accuracy': per_label_acc,
        'per_label_counts': dict(per_label_total),
        'runtime': runtime,
        'final_stats': learner.get_stats(),
        'learner': learner,
    }


def compare_learners(learner_classes: List[Type], oracle: Callable, 
                     board_generator: Callable, label_space: List[str],
                     rounds: int = 500, runs: int = 3) -> Dict[str, Dict]:
    """Compare multiple learners with multiple runs for statistical significance."""
    
    results = defaultdict(list)
    
    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")
        for learner_class in learner_classes:
            name = learner_class.__name__
            print(f"\nTesting {name}...")
            result = test_learner(
                learner_class, oracle, board_generator, label_space,
                rounds=rounds, verbose=False
            )
            results[name].append(result)
            print(f"  Final: {result['final_accuracy']:.1%}")
    
    # Aggregate results
    summary = {}
    for name, run_results in results.items():
        accs = [r['final_accuracy'] for r in run_results]
        summary[name] = {
            'mean_accuracy': sum(accs) / len(accs),
            'min_accuracy': min(accs),
            'max_accuracy': max(accs),
            'convergence': {
                cp: sum(r['accuracy_over_time'].get(cp, 0) for r in run_results) / len(run_results)
                for cp in [10, 25, 50, 100, 200, 500] if cp <= rounds
            }
        }
    
    return summary


def print_comparison(summary: Dict[str, Dict]):
    """Print comparison table."""
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # Header
    print(f"\n{'Learner':<25} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    
    for name, stats in sorted(summary.items(), key=lambda x: -x[1]['mean_accuracy']):
        print(f"{name:<25} {stats['mean_accuracy']:>7.1%} {stats['min_accuracy']:>7.1%} {stats['max_accuracy']:>7.1%}")
    
    # Convergence
    print("\n--- Convergence (accuracy at N observations) ---")
    checkpoints = [10, 25, 50, 100, 200, 500]
    
    print(f"{'Learner':<25}", end="")
    for cp in checkpoints:
        print(f"  @{cp:<4}", end="")
    print()
    print("-" * 70)
    
    for name, stats in sorted(summary.items(), key=lambda x: -x[1]['mean_accuracy']):
        print(f"{name:<25}", end="")
        for cp in checkpoints:
            acc = stats['convergence'].get(cp, 0)
            print(f" {acc:>5.1%}", end="")
        print()


if __name__ == "__main__":
    from tictactoe import tictactoe, random_board, label_space
    from few_shot_algs.blind_learner import BlindLearner
    
    print("="*70)
    print("BASELINE TEST: BlindLearner on Standard TicTacToe")
    print("="*70)
    
    result = test_learner(
        BlindLearner, tictactoe, random_board, label_space,
        rounds=500, verbose=True
    )
    
    print(f"\nFinal Accuracy: {result['final_accuracy']:.1%}")
    print(f"Runtime: {result['runtime']:.2f}s")
    
    print("\nPer-Label Accuracy:")
    for label, acc in result['per_label_accuracy'].items():
        count = result['per_label_counts'][label]
        print(f"  {label:8s}: {acc:.1%} ({count} samples)")
    
    print("\n" + result['learner'].describe_knowledge())
