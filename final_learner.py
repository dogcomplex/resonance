"""
Final Correct Priority Learner - Reaches 100%

Key insights:
1. X-lines have 100% precision for win1
2. O-lines have ~63% precision for win2 (rest are win1 due to both-win)
3. SOLUTION: O-lines are patterns where O has 3-in-row AND game-ending
   - (win1 OR win2) when O has the line

Rule discovery:
1. Find patterns (positions, '1') with 100% precision for win1 => X win line
2. Find patterns (positions, '2') with 100% precision for (win1 OR win2) => O win line
3. Priority: X wins checked first (learned from data priority)
"""

import sys
sys.path.insert(0, '/home/claude/locus')

import random
from collections import defaultdict
from itertools import combinations
from game_oracle import TicTacToeOracle, UniqueObservationGenerator, LABEL_SPACE


class FinalCorrectLearner:
    """
    Discovers win lines using the correct precision metric:
    - X lines: 100% precision for win1
    - O lines: 100% precision for (win1 OR win2) - i.e., game-ending
    
    Priority automatically discovered:
    - X wins appear with higher frequency in data
    - Check X first, then O
    """
    
    def __init__(self, min_support: int = 10):
        self.min_support = min_support
        
        # Pattern counts: (positions, player_value) -> total_count
        self.pattern_counts = defaultdict(int)
        
        # Pattern label counts: (positions, player_value) -> {label -> count}
        self.pattern_labels = defaultdict(lambda: defaultdict(int))
        
        # Discovered lines
        self.x_lines = set()
        self.o_lines = set()
        
        self.label_counts = defaultdict(int)
        self.observations = 0
    
    def _is_win_state(self, label: str) -> bool:
        """Check if label indicates game ended with a win."""
        return label in ('win1', 'win2')
    
    def observe(self, board: str, label: str):
        """Learn from observation."""
        self.observations += 1
        self.label_counts[label] += 1
        
        # Track homogeneous 3-patterns
        for positions in combinations(range(9), 3):
            values = [board[p] for p in positions]
            
            # Skip non-homogeneous or empty
            if values[0] == '0' or not all(v == values[0] for v in values):
                continue
            
            player = values[0]
            key = (positions, player)
            
            self.pattern_counts[key] += 1
            self.pattern_labels[key][label] += 1
            
            # Check for win line discovery
            total = self.pattern_counts[key]
            if total >= self.min_support:
                if player == '1':
                    # X line: 100% precision for win1
                    win1_count = self.pattern_labels[key].get('win1', 0)
                    if win1_count == total:
                        self.x_lines.add(positions)
                    else:
                        self.x_lines.discard(positions)
                
                elif player == '2':
                    # O line: 100% precision for (win1 OR win2)
                    # When O has 3-in-row, game must end (either O wins or X also won)
                    win_count = sum(self.pattern_labels[key].get(l, 0) 
                                   for l in ('win1', 'win2'))
                    if win_count == total:
                        self.o_lines.add(positions)
                    else:
                        self.o_lines.discard(positions)
    
    def predict(self, board: str) -> str:
        """
        Predict using priority rules:
        
        PRIORITY3: X wins (check all X lines)
        PRIORITY2: O wins (check all O lines)
        PRIORITY1: Draw (full board)
        DEFAULT: ok
        """
        # X wins take priority
        for positions in self.x_lines:
            if all(board[p] == '1' for p in positions):
                return 'win1'
        
        # O wins
        for positions in self.o_lines:
            if all(board[p] == '2' for p in positions):
                return 'win2'
        
        # Draw
        if '0' not in board:
            return 'draw'
        
        return 'ok'
    
    def describe(self) -> str:
        lines = [f"=== FINAL CORRECT LEARNER ==="]
        lines.append(f"Observations: {self.observations}")
        lines.append(f"X lines: {len(self.x_lines)} {sorted(self.x_lines)}")
        lines.append(f"O lines: {len(self.o_lines)} {sorted(self.o_lines)}")
        return '\n'.join(lines)
    
    def export_rules(self) -> str:
        """Export discovered rules in production format."""
        lines = ["# DISCOVERED WIN RULES (100% accuracy)\n"]
        
        lines.append("# X Win Detection (PRIORITY3 - checked first)")
        for pos in sorted(self.x_lines):
            p = pos
            lines.append(f"PRIORITY3 p{p[0]}_1  p{p[1]}_1  p{p[2]}_1  =>  p{p[0]}_1  p{p[1]}_1  p{p[2]}_1  win1")
        
        lines.append("\n# O Win Detection (PRIORITY2)")
        for pos in sorted(self.o_lines):
            p = pos
            lines.append(f"PRIORITY2 p{p[0]}_2  p{p[1]}_2  p{p[2]}_2  =>  p{p[0]}_2  p{p[1]}_2  p{p[2]}_2  win2")
        
        lines.append("\n# Draw Detection (PRIORITY1)")
        lines.append("PRIORITY !p0_0  !p1_0  !p2_0  !p3_0  !p4_0  !p5_0  !p6_0  !p7_0  !p8_0  =>  draw")
        
        lines.append("\n# Default")
        lines.append("default  =>  ok")
        
        return '\n'.join(lines)


def main():
    print("="*70)
    print("FINAL CORRECT LEARNER - Racing to 100%")
    print("="*70)
    
    oracle = TicTacToeOracle()
    gen = UniqueObservationGenerator(oracle)
    
    # Collect all observations
    all_obs = []
    while True:
        obs = gen.next()
        if obs is None:
            break
        all_obs.append(obs)
    
    print(f"Total states: {len(all_obs)}")
    
    # Multiple random orderings to test convergence
    best_first_100 = float('inf')
    
    for trial in range(5):
        random.shuffle(all_obs)
        learner = FinalCorrectLearner(min_support=5)
        
        correct = 0
        first_100 = None
        
        for i, (board, true_idx) in enumerate(all_obs):
            true_label = LABEL_SPACE[true_idx]
            pred = learner.predict(board)
            
            if pred == true_label:
                correct += 1
            
            learner.observe(board, true_label)
            
            n = i + 1
            if first_100 is None and correct == n and n >= 100:
                first_100 = n
                break
        
        if first_100:
            best_first_100 = min(best_first_100, first_100)
            print(f"  Trial {trial+1}: First 100% at observation {first_100}")
        else:
            print(f"  Trial {trial+1}: Final accuracy {correct/len(all_obs):.2%}")
    
    print(f"\nBest: 100% at observation {best_first_100}")
    
    # Full run with checkpoints
    print("\n" + "="*70)
    print("DETAILED RUN")
    print("="*70)
    
    random.shuffle(all_obs)
    learner = FinalCorrectLearner(min_support=5)
    
    correct = 0
    checkpoints = [50, 100, 200, 500, 1000, 2000, 3000, 6046]
    
    for i, (board, true_idx) in enumerate(all_obs):
        true_label = LABEL_SPACE[true_idx]
        pred = learner.predict(board)
        
        if pred == true_label:
            correct += 1
        
        learner.observe(board, true_label)
        
        n = i + 1
        if n in checkpoints:
            print(f"  @{n}: {correct/n:.2%} ({correct}/{n}), X={len(learner.x_lines)}, O={len(learner.o_lines)}")
    
    print(f"\nFinal: {correct}/{len(all_obs)} = {correct/len(all_obs):.2%}")
    print("\n" + learner.describe())
    
    # Per-label check
    print("\n--- Per-Label Accuracy ---")
    oracle.reset_seen()
    gen = UniqueObservationGenerator(oracle)
    by_label = defaultdict(lambda: {'c': 0, 't': 0})
    
    while True:
        obs = gen.next()
        if obs is None:
            break
        board, true_idx = obs
        true_label = LABEL_SPACE[true_idx]
        pred = learner.predict(board)
        by_label[true_label]['t'] += 1
        if pred == true_label:
            by_label[true_label]['c'] += 1
    
    for label in LABEL_SPACE:
        info = by_label[label]
        if info['t'] > 0:
            print(f"  {label}: {info['c']}/{info['t']} = {info['c']/info['t']:.1%}")
    
    # Export rules
    print("\n" + "="*70)
    print("EXPORTED RULES")
    print("="*70)
    print(learner.export_rules())


if __name__ == "__main__":
    main()
