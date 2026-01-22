"""
Comprehensive Test Suite

Tests learners with:
1. Unique observations (no duplicates)
2. Standard TicTacToe
3. Random rule variants
4. Full coverage tracking
5. Per-label accuracy breakdown
"""

import sys
sys.path.insert(0, '/home/claude/locus')

import random
from collections import defaultdict
from typing import List, Dict, Any, Type

from game_oracle import (
    TicTacToeOracle, RandomRuleOracle, 
    UniqueObservationGenerator, LABEL_SPACE
)


def test_learner_unique(learner_class: Type, oracle, 
                        max_rounds: int = 1000,
                        checkpoints: List[int] = None,
                        verbose: bool = True,
                        **learner_kwargs) -> Dict[str, Any]:
    """
    Test a learner with unique observations.
    
    Returns when max_rounds reached OR all states observed.
    """
    if checkpoints is None:
        checkpoints = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    
    gen = UniqueObservationGenerator(oracle)
    learner = learner_class(
        num_outputs=len(LABEL_SPACE),
        label_names=LABEL_SPACE,
        **learner_kwargs
    )
    
    correct = 0
    per_label_correct = defaultdict(int)
    per_label_total = defaultdict(int)
    accuracy_history = {}
    
    rounds = 0
    for rounds in range(max_rounds):
        obs = gen.next()
        if obs is None:
            break  # All states seen!
        
        board, true_idx = obs
        true_label = LABEL_SPACE[true_idx]
        
        pred_idx = learner.predict(board)
        
        if pred_idx == true_idx:
            correct += 1
            per_label_correct[true_label] += 1
        per_label_total[true_label] += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if (rounds + 1) in checkpoints:
            acc = correct / (rounds + 1)
            accuracy_history[rounds + 1] = acc
            if verbose:
                stats = learner.get_stats()
                phase = stats.get('phase', 'N/A')[:4]
                rules = stats.get('rules', 0)
                pure = stats.get('pure_rules', stats.get('rules', 0))
                print(f"  R{rounds+1:4d}: {acc:.1%} | phase={phase} rules={rules} pure={pure}")
    
    rounds += 1  # Account for 0-indexing
    
    # Calculate per-label accuracy
    per_label_acc = {}
    for label in LABEL_SPACE:
        total = per_label_total.get(label, 0)
        corr = per_label_correct.get(label, 0)
        per_label_acc[label] = corr / total if total > 0 else 0.0
    
    return {
        'final_accuracy': correct / rounds if rounds > 0 else 0,
        'total_rounds': rounds,
        'coverage': gen.coverage(),
        'remaining': gen.remaining(),
        'accuracy_history': accuracy_history,
        'per_label_accuracy': per_label_acc,
        'per_label_counts': dict(per_label_total),
        'learner': learner,
    }


def test_on_random_rules(learner_class: Type, num_seeds: int = 5,
                         max_rounds: int = 1000, verbose: bool = True) -> Dict:
    """Test learner on multiple random rule sets."""
    results = []
    
    for seed in range(num_seeds):
        oracle = RandomRuleOracle(
            num_win_conditions=8,
            win_size=3,
            seed=seed * 100
        )
        
        if verbose:
            print(f"\n--- Random Rules Seed {seed * 100} ---")
            print(f"Win conditions: {oracle.win_conditions[:4]}...")
        
        result = test_learner_unique(
            learner_class, oracle, max_rounds, verbose=verbose
        )
        results.append(result)
        
        if verbose:
            print(f"Final: {result['final_accuracy']:.1%}")
    
    # Aggregate
    avg_acc = sum(r['final_accuracy'] for r in results) / len(results)
    avg_win = sum(
        (r['per_label_accuracy'].get('win1', 0) + r['per_label_accuracy'].get('win2', 0)) / 2
        for r in results
    ) / len(results)
    
    return {
        'mean_accuracy': avg_acc,
        'mean_win_accuracy': avg_win,
        'individual_results': results,
    }


def run_comprehensive_test():
    """Run full test suite."""
    from few_shot_algs.blind_learner import BlindLearner
    from few_shot_algs.hybrid_learner import HybridLearner
    from few_shot_algs.advanced_production import AdvancedProductionLearner
    
    learners = [
        ("BlindLearner", BlindLearner),
        ("HybridLearner", HybridLearner),
        ("AdvancedProduction", AdvancedProductionLearner),
    ]
    
    print("="*80)
    print("COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Test on standard TicTacToe with unique observations
    print("\n" + "="*80)
    print("1. STANDARD TICTACTOE (Unique Observations)")
    print("="*80)
    
    standard_results = {}
    oracle = TicTacToeOracle()
    print(f"Total states: {oracle.total_states()}")
    print(f"Distribution: {oracle.label_distribution()}")
    
    for name, learner_class in learners:
        print(f"\n--- {name} ---")
        oracle.reset_seen()
        result = test_learner_unique(learner_class, oracle, max_rounds=2000)
        standard_results[name] = result
        
        win_acc = (result['per_label_accuracy'].get('win1', 0) + 
                   result['per_label_accuracy'].get('win2', 0)) / 2
        print(f"Final: {result['final_accuracy']:.1%}, Wins: {win_acc:.1%}, Coverage: {result['coverage']:.1%}")
    
    # Test on random rules
    print("\n" + "="*80)
    print("2. RANDOM RULE VARIANTS")
    print("="*80)
    
    random_results = {}
    for name, learner_class in learners:
        print(f"\n=== {name} ===")
        result = test_on_random_rules(learner_class, num_seeds=3, max_rounds=1000, verbose=False)
        random_results[name] = result
        print(f"Mean accuracy: {result['mean_accuracy']:.1%}")
        print(f"Mean win accuracy: {result['mean_win_accuracy']:.1%}")
    
    # Summary tables
    print("\n" + "="*80)
    print("SUMMARY: Standard TicTacToe")
    print("="*80)
    
    print(f"\n{'Learner':<25} {'Overall':>10} {'ok':>8} {'win1':>8} {'win2':>8} {'Coverage':>10}")
    print("-" * 75)
    
    for name in standard_results:
        r = standard_results[name]
        print(f"{name:<25} {r['final_accuracy']:>9.1%} "
              f"{r['per_label_accuracy'].get('ok', 0):>7.1%} "
              f"{r['per_label_accuracy'].get('win1', 0):>7.1%} "
              f"{r['per_label_accuracy'].get('win2', 0):>7.1%} "
              f"{r['coverage']:>9.1%}")
    
    print("\n" + "="*80)
    print("SUMMARY: Random Rules (avg of 3 seeds)")
    print("="*80)
    
    print(f"\n{'Learner':<25} {'Mean Acc':>12} {'Mean Wins':>12}")
    print("-" * 50)
    
    for name in random_results:
        r = random_results[name]
        print(f"{name:<25} {r['mean_accuracy']:>11.1%} {r['mean_win_accuracy']:>11.1%}")
    
    # Convergence analysis
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    checkpoints = [10, 25, 50, 100, 200, 500, 1000]
    
    print(f"\n{'Learner':<25}", end="")
    for cp in checkpoints:
        print(f" @{cp:<5}", end="")
    print()
    print("-" * 75)
    
    for name in standard_results:
        print(f"{name:<25}", end="")
        for cp in checkpoints:
            acc = standard_results[name]['accuracy_history'].get(cp, 0)
            print(f" {acc:>5.1%}", end="")
        print()
    
    return standard_results, random_results


if __name__ == "__main__":
    run_comprehensive_test()
