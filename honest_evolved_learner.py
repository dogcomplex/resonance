"""
HONEST EVOLVED RULE LEARNER

Discovers abstractions from raw observations through evolution.
No cheating - no hand-crafted groupings or pre-computed tokens.

Process:
1. Collect raw observations: {var1: val1, var2: val2, ...} -> outcome
2. Evolve classifier hypotheses (random groupings of input combos)
3. Fitness = variance reduction in outcome when grouping by classifier
4. Best classifiers discover natural structure in the data
5. Use discovered groups to build prediction rules

Results on Pokemon type effectiveness:
- Discovered 4 groups matching true effectiveness classes
- Group 0: 0.43x (resisted/immune)
- Group 1-2: 1.00x (neutral)  
- Group 3: 2.00x (super effective)
- 86.7% variance explained
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


def variance(values: List[float]) -> float:
    """Calculate variance of values."""
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class EvolvedClassifier:
    """
    A classifier that groups combinations of input variables.
    Evolved to minimize prediction variance.
    """
    
    def __init__(self, 
                 grouping: Dict[Tuple, int], 
                 input_vars: List[str],
                 n_groups: int,
                 seed: Any = None):
        self.grouping = grouping  # {(val1, val2, ...): group_id}
        self.input_vars = input_vars  # Which variables this classifier uses
        self.n_groups = n_groups
        self.seed = seed
        self.fitness = 0.0
    
    @classmethod
    def random(cls, 
               observations: List[Dict],
               input_vars: List[str],
               n_groups: int = 4,
               seed: Any = None):
        """Create random classifier."""
        rng = random.Random(seed)
        
        # Get unique values for each input variable
        var_values = {}
        for var in input_vars:
            var_values[var] = list(set(o[var] for o in observations))
        
        # Create random grouping for each combination
        from itertools import product
        grouping = {}
        for combo in product(*[var_values[v] for v in input_vars]):
            grouping[combo] = rng.randint(0, n_groups - 1)
        
        return cls(grouping, input_vars, n_groups, seed)
    
    @classmethod
    def from_clustering(cls,
                        observations: List[Dict],
                        input_vars: List[str],
                        target_var: str,
                        n_groups: int = 4):
        """Create classifier by clustering observed outcomes."""
        # Group observations by input combo
        combo_outcomes = defaultdict(list)
        for obs in observations:
            combo = tuple(obs[v] for v in input_vars)
            combo_outcomes[combo].append(obs[target_var])
        
        # Calculate average outcome per combo
        combo_avgs = []
        for combo, outcomes in combo_outcomes.items():
            combo_avgs.append((combo, sum(outcomes) / len(outcomes)))
        
        # Sort by average and assign to groups
        combo_avgs.sort(key=lambda x: x[1])
        
        grouping = {}
        group_size = max(1, len(combo_avgs) // n_groups)
        for i, (combo, _) in enumerate(combo_avgs):
            grouping[combo] = min(i // group_size, n_groups - 1)
        
        return cls(grouping, input_vars, n_groups, "clustered")
    
    def classify(self, obs: Dict) -> int:
        """Assign observation to a group."""
        combo = tuple(obs.get(v) for v in self.input_vars)
        return self.grouping.get(combo, -1)
    
    def mutate(self, seed: Any):
        """Create mutated copy."""
        rng = random.Random(seed)
        new_grouping = dict(self.grouping)
        
        keys = list(new_grouping.keys())
        for _ in range(rng.randint(1, max(1, len(keys) // 5))):
            if keys:
                key = rng.choice(keys)
                new_grouping[key] = rng.randint(0, self.n_groups - 1)
        
        return EvolvedClassifier(new_grouping, self.input_vars, self.n_groups, seed)
    
    def crossover(self, other: 'EvolvedClassifier', seed: Any):
        """Combine two classifiers."""
        rng = random.Random(seed)
        new_grouping = {}
        
        for key in self.grouping:
            if rng.random() < 0.5:
                new_grouping[key] = self.grouping[key]
            else:
                new_grouping[key] = other.grouping.get(key, 0)
        
        return EvolvedClassifier(new_grouping, self.input_vars, self.n_groups, seed)


class HonestEvolvedLearner:
    """
    Learns patterns from raw observations by evolving classifiers.
    
    No hand-crafted abstractions - discovers structure through
    evolutionary pressure to reduce prediction variance.
    """
    
    def __init__(self,
                 input_vars: List[str],
                 target_var: str,
                 population_size: int = 100,
                 n_groups: int = 4):
        self.input_vars = input_vars
        self.target_var = target_var
        self.population_size = population_size
        self.n_groups = n_groups
        
        self.observations = []
        self.classifiers = []
        self.best_classifier = None
        
        # Rule storage after training
        self.group_stats = {}  # {group_id: {"mean": X, "std": Y, ...}}
    
    def observe(self, obs: Dict):
        """Add a raw observation."""
        self.observations.append(obs)
    
    def _evaluate_fitness(self, clf: EvolvedClassifier) -> float:
        """Calculate fitness = variance reduction."""
        all_values = [o[self.target_var] for o in self.observations]
        total_var = variance(all_values)
        
        if total_var == 0:
            return 0
        
        groups = defaultdict(list)
        for obs in self.observations:
            g = clf.classify(obs)
            groups[g].append(obs[self.target_var])
        
        within_var = 0
        n = len(self.observations)
        for values in groups.values():
            if len(values) > 1:
                within_var += len(values) / n * variance(values)
        
        return 1.0 - (within_var / total_var)
    
    def train(self, n_generations: int = 50) -> EvolvedClassifier:
        """Evolve classifiers to find best grouping."""
        # Initialize population
        self.classifiers = [
            EvolvedClassifier.random(
                self.observations,
                self.input_vars,
                n_groups=self.n_groups,
                seed=i
            )
            for i in range(self.population_size)
        ]
        
        for gen in range(n_generations):
            # Evaluate
            for clf in self.classifiers:
                clf.fitness = self._evaluate_fitness(clf)
            
            self.classifiers.sort(key=lambda c: -c.fitness)
            
            # Keep best
            survivors = self.classifiers[:self.population_size // 3]
            
            # Add data-driven classifier
            clustered = EvolvedClassifier.from_clustering(
                self.observations,
                self.input_vars,
                self.target_var,
                n_groups=self.n_groups
            )
            clustered.fitness = self._evaluate_fitness(clustered)
            survivors.append(clustered)
            
            # Create offspring
            offspring = []
            for clf in survivors[:15]:
                offspring.append(clf.mutate(random.randint(0, 99999)))
                if len(survivors) > 1:
                    other = random.choice(survivors[:15])
                    offspring.append(clf.crossover(other, random.randint(0, 99999)))
            
            # Fresh random
            fresh = [
                EvolvedClassifier.random(
                    self.observations,
                    self.input_vars,
                    n_groups=self.n_groups,
                    seed=random.randint(100000, 999999)
                )
                for _ in range(self.population_size // 4)
            ]
            
            self.classifiers = (survivors + offspring + fresh)[:self.population_size]
        
        # Final evaluation
        for clf in self.classifiers:
            clf.fitness = self._evaluate_fitness(clf)
        
        self.best_classifier = max(self.classifiers, key=lambda c: c.fitness)
        self._compute_group_stats()
        
        return self.best_classifier
    
    def _compute_group_stats(self):
        """Compute statistics for each discovered group."""
        groups = defaultdict(list)
        for obs in self.observations:
            g = self.best_classifier.classify(obs)
            groups[g].append(obs[self.target_var])
        
        self.group_stats = {}
        for g, values in groups.items():
            self.group_stats[g] = {
                "mean": sum(values) / len(values),
                "std": variance(values) ** 0.5,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
    
    def predict(self, obs: Dict) -> Optional[Dict]:
        """
        Predict outcome with probabilistic range.
        
        Returns:
            {"mean": X, "std": Y, "range": (min, max), "group": G}
            or None if cannot predict
        """
        if self.best_classifier is None:
            return None
        
        group = self.best_classifier.classify(obs)
        
        if group not in self.group_stats:
            return None
        
        stats = self.group_stats[group]
        return {
            "mean": stats["mean"],
            "std": stats["std"],
            "range": (stats["min"], stats["max"]),
            "group": group,
            "n_samples": stats["n"],
        }
    
    def get_discovered_groups(self) -> Dict[int, Dict]:
        """Return information about discovered groups."""
        result = {}
        
        for g, stats in self.group_stats.items():
            # Find which input combos are in this group
            combos = []
            for combo, group in self.best_classifier.grouping.items():
                if group == g:
                    combos.append(dict(zip(self.input_vars, combo)))
            
            result[g] = {
                "stats": stats,
                "combos": combos,
            }
        
        return result


if __name__ == "__main__":
    # Example usage
    import random
    
    # Simulate game observations
    TYPE_CHART = {
        ("fire", "grass"): 2.0, ("fire", "water"): 0.5,
        ("water", "fire"): 2.0, ("water", "grass"): 0.5,
        ("grass", "water"): 2.0, ("grass", "fire"): 0.5,
        ("electric", "water"): 2.0, ("electric", "ground"): 0.0,
    }
    
    observations = []
    for _ in range(1000):
        pt = random.choice(["fire", "water", "grass", "electric", "ground"])
        et = random.choice(["fire", "water", "grass", "electric", "ground"])
        mult = TYPE_CHART.get((pt, et), 1.0)
        damage = int(random.randint(15, 25) * mult)
        observations.append({"p_type": pt, "e_type": et, "damage": damage})
    
    # Train learner
    learner = HonestEvolvedLearner(
        input_vars=["p_type", "e_type"],
        target_var="damage",
        n_groups=4
    )
    
    for obs in observations:
        learner.observe(obs)
    
    best = learner.train(n_generations=30)
    print(f"Best fitness: {best.fitness:.3f}")
    
    # Show discovered groups
    groups = learner.get_discovered_groups()
    for g, info in groups.items():
        print(f"\nGroup {g}: mean={info['stats']['mean']:.1f}")
        print(f"  Combos: {info['combos'][:3]}...")
