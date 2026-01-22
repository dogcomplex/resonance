"""
import heapq
Hierarchical Learner V12 - Dream-Ready World Model

Building on V11, adds:
1. Confidence/Uncertainty estimates for predictions
2. Abductive reasoning (backward: effect → causes)
3. Inductive reasoning (pattern discovery across rules)
4. Temporal dependency tracking
5. Imagination rollout mode
6. Novelty scoring for exploration

Epistemological Framework:
- DEDUCTIVE:  state + action → effects (forward prediction)
- ABDUCTIVE:  desired_effect → possible (state, action) pairs (backward reasoning)  
- INDUCTIVE:  observations → general patterns/laws (rule compression)
"""

import sqlite3
import heapq
import re
import math
from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass, field
import hashlib


@dataclass
class Prediction:
    """A prediction with confidence and provenance"""
    effects: Set[str]
    confidence: float  # 0.0 - 1.0
    source: str  # "exact", "rule", "abduced", "induced", "unknown"
    support: int  # Number of observations supporting this
    alternatives: List[Tuple[Set[str], float]] = field(default_factory=list)  # Other possible outcomes


@dataclass 
class AbductiveResult:
    """Result of backward reasoning"""
    required_tokens: Set[str]  # Tokens that must be present
    enabling_actions: List[int]  # Actions that can produce the target
    confidence: float
    observed_count: int


@dataclass
class InducedPattern:
    """A discovered general pattern"""
    pattern_type: str  # "sell", "craft", "transform", etc.
    input_pattern: str  # Regex or template for inputs
    output_pattern: str  # Regex or template for outputs
    confidence: float
    coverage: int  # How many rules this explains
    examples: List[Tuple[int, Set[str], Set[str]]]  # (action, inputs, outputs)


@dataclass
class TemporalDependency:
    """Tracks what typically precedes an action"""
    action: int
    prerequisites: Dict[int, float]  # action → P(precedes)
    min_steps_from_start: float  # Average steps before this action appears
    typical_sequence: List[int]  # Most common path to this action


class HierarchicalLearnerV12:
    """
    Dream-ready world model with deductive, abductive, and inductive reasoning.
    """
    
    def __init__(self, n_actions: int, db_path: str = ":memory:"):
        self.n_actions = n_actions
        self.db_path = db_path
        
        # Core storage (from V11)
        self.exact_memory: Dict[Tuple[FrozenSet[str], int], Dict[FrozenSet[str], int]] = defaultdict(lambda: defaultdict(int))
        self.rules: Dict[Tuple[FrozenSet[str], int], Dict[str, any]] = {}
        
        # NEW: Reverse index for abductive reasoning
        self.effect_to_causes: Dict[str, List[Tuple[FrozenSet[str], int, int]]] = defaultdict(list)
        # effect → [(state_pattern, action, count), ...]
        
        # NEW: Token production/consumption tracking
        self.token_producers: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # token → {action → count}
        self.token_consumers: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # NEW: Temporal tracking
        self.action_sequences: List[List[int]] = []  # Episode action histories
        self.current_episode: List[int] = []
        self.action_step_counts: Dict[int, List[int]] = defaultdict(list)  # action → [step numbers]
        self.action_predecessors: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # action → {predecessor_action → count}
        
        # NEW: Pattern discovery for induction
        self.action_signatures: Dict[int, List[Tuple[Set[str], Set[str]]]] = defaultdict(list)
        # action → [(consumed_tokens, produced_tokens), ...]
        self.induced_patterns: List[InducedPattern] = []
        
        # NEW: Observation counts for confidence
        self.state_action_counts: Dict[Tuple[FrozenSet[str], int], int] = defaultdict(int)
        self.total_observations = 0
        
        # Trajectory tracking (from V11)
        self._prev_values: Dict[str, int] = {}
        
        # SQLite for persistence (optional)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite for large-scale storage"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                state_hash TEXT,
                action INTEGER,
                effects TEXT,
                count INTEGER,
                PRIMARY KEY (state_hash, action, effects)
            );
            
            CREATE TABLE IF NOT EXISTS effect_index (
                effect TEXT,
                state_hash TEXT,
                action INTEGER,
                count INTEGER
            );
            
            CREATE INDEX IF NOT EXISTS idx_effect ON effect_index(effect);
            
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_type TEXT,
                input_pattern TEXT,
                output_pattern TEXT,
                confidence REAL,
                coverage INTEGER
            );
        """)
        self.conn.commit()
    
    # =========================================================================
    # CORE OBSERVATION (Enhanced from V11)
    # =========================================================================
    
    def observe(self, before: Set[str], action: int, after: Set[str]):
        """
        Record a state transition with full indexing for all reasoning modes.
        """
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        # Compute effects
        added = after_fs - before_fs
        removed = before_fs - after_fs
        effects = frozenset({f"+{t}" for t in added} | {f"-{t}" for t in removed})
        
        # 1. EXACT MEMORY (deductive)
        self.exact_memory[(before_fs, action)][effects] += 1
        self.state_action_counts[(before_fs, action)] += 1
        self.total_observations += 1
        
        # 2. REVERSE INDEX (abductive)
        for effect in effects:
            self.effect_to_causes[effect].append((before_fs, action, 1))
        
        # 3. TOKEN TRACKING (abductive)
        for token in added:
            self.token_producers[token][action] += 1
        for token in removed:
            self.token_consumers[token][action] += 1
        
        # 4. TEMPORAL TRACKING
        self.current_episode.append(action)
        step = len(self.current_episode)
        self.action_step_counts[action].append(step)
        
        if step > 1:
            prev_action = self.current_episode[-2]
            self.action_predecessors[action][prev_action] += 1
        
        # 5. SIGNATURE TRACKING (inductive)
        consumed = {t for t in removed}
        produced = {t for t in added}
        self.action_signatures[action].append((consumed, produced))
        
        # 6. HIERARCHICAL RULES (from V11)
        self._update_rules(before_fs, action, effects)
    
    def _update_rules(self, state: FrozenSet[str], action: int, effects: FrozenSet[str]):
        """Update hierarchical rules (preserved from V11)"""
        # Single-token patterns
        for token in state:
            key = (frozenset([token]), action)
            if key not in self.rules:
                self.rules[key] = {
                    'observations': 0,
                    'effect_counts': defaultdict(int),
                    'specificity': 1
                }
            self.rules[key]['observations'] += 1
            self.rules[key]['effect_counts'][effects] += 1
        
        # Full state pattern
        key = (state, action)
        if key not in self.rules:
            self.rules[key] = {
                'observations': 0,
                'effect_counts': defaultdict(int),
                'specificity': len(state)
            }
        self.rules[key]['observations'] += 1
        self.rules[key]['effect_counts'][effects] += 1
    
    def reset_episode(self):
        """Mark episode boundary for temporal tracking"""
        if self.current_episode:
            self.action_sequences.append(self.current_episode)
        self.current_episode = []
        self._prev_values = {}
    
    # =========================================================================
    # DEDUCTIVE REASONING (Forward Prediction)
    # =========================================================================
    
    def predict(self, state: Set[str], action: int) -> Set[str]:
        """
        Basic forward prediction (V11 compatible).
        Returns most likely effects.
        """
        result = self.predict_with_confidence(state, action)
        return result.effects
    
    def predict_with_confidence(self, state: Set[str], action: int) -> Prediction:
        """
        Forward prediction with full confidence and alternatives.
        
        This is what the dreamer needs for imagination rollouts.
        """
        state_fs = frozenset(state)
        key = (state_fs, action)
        
        # 1. Check exact memory
        if key in self.exact_memory:
            outcomes = self.exact_memory[key]
            total = sum(outcomes.values())
            
            # Sort by frequency
            sorted_outcomes = sorted(outcomes.items(), key=lambda x: -x[1])
            best_effects, best_count = sorted_outcomes[0]
            
            confidence = best_count / total
            
            # Collect alternatives
            alternatives = [(set(eff), cnt/total) for eff, cnt in sorted_outcomes[1:4]]
            
            return Prediction(
                effects=set(best_effects),
                confidence=confidence,
                source="exact",
                support=total,
                alternatives=alternatives
            )
        
        # 2. Check hierarchical rules
        best_rule = self._find_best_rule(state_fs, action)
        if best_rule:
            rule_data = self.rules[best_rule]
            outcomes = rule_data['effect_counts']
            total = sum(outcomes.values())
            
            sorted_outcomes = sorted(outcomes.items(), key=lambda x: -x[1])
            best_effects, best_count = sorted_outcomes[0]
            
            # Confidence scales with specificity and observations
            base_confidence = best_count / total
            specificity_factor = min(1.0, rule_data['specificity'] / 10)
            observation_factor = min(1.0, rule_data['observations'] / 20)
            confidence = base_confidence * specificity_factor * observation_factor
            
            return Prediction(
                effects=set(best_effects),
                confidence=confidence,
                source="rule",
                support=rule_data['observations'],
                alternatives=[(set(eff), cnt/total) for eff, cnt in sorted_outcomes[1:4]]
            )
        
        # 3. Unknown - return empty with low confidence
        return Prediction(
            effects=set(),
            confidence=0.1,
            source="unknown",
            support=0,
            alternatives=[]
        )
    
    def _find_best_rule(self, state: FrozenSet[str], action: int) -> Optional[Tuple[FrozenSet[str], int]]:
        """Find most specific matching rule"""
        candidates = []
        for (pattern, act), rule_data in self.rules.items():
            if act == action and pattern.issubset(state):
                candidates.append(((pattern, act), rule_data['specificity'], rule_data['observations']))
        
        if not candidates:
            return None
        
        # Sort by specificity, then observations
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        return candidates[0][0]
    
    # =========================================================================
    # ABDUCTIVE REASONING (Backward Prediction)
    # =========================================================================
    
    def abduce(self, target_effect: str) -> AbductiveResult:
        """
        Given a desired effect, find what states/actions could produce it.
        
        "I want +💰_100 - what do I need to do?"
        """
        # Check if this is a production (+) or consumption (-) effect
        if target_effect.startswith('+'):
            token = target_effect[1:]
            # What actions produce this token?
            producers = self.token_producers.get(token, {})
            
            if not producers:
                return AbductiveResult(
                    required_tokens=set(),
                    enabling_actions=[],
                    confidence=0.0,
                    observed_count=0
                )
            
            # Find the best actions
            total = sum(producers.values())
            sorted_actions = sorted(producers.items(), key=lambda x: -x[1])
            
            # What states enabled these actions?
            required = set()
            for action, count in sorted_actions[:5]:
                # Look at states where this action produced this token
                for (state, act), outcomes in self.exact_memory.items():
                    if act == action:
                        for effects, cnt in outcomes.items():
                            if target_effect in effects:
                                required.update(state)
                                break
            
            return AbductiveResult(
                required_tokens=required,
                enabling_actions=[a for a, _ in sorted_actions[:5]],
                confidence=sorted_actions[0][1] / total if sorted_actions else 0,
                observed_count=total
            )
        
        elif target_effect.startswith('-'):
            token = target_effect[1:]
            # What actions consume this token?
            consumers = self.token_consumers.get(token, {})
            
            if not consumers:
                return AbductiveResult(
                    required_tokens={token},  # Need the token to consume it
                    enabling_actions=[],
                    confidence=0.0,
                    observed_count=0
                )
            
            total = sum(consumers.values())
            sorted_actions = sorted(consumers.items(), key=lambda x: -x[1])
            
            return AbductiveResult(
                required_tokens={token},
                enabling_actions=[a for a, _ in sorted_actions[:5]],
                confidence=sorted_actions[0][1] / total if sorted_actions else 0,
                observed_count=total
            )
        
        return AbductiveResult(set(), [], 0.0, 0)
    
    def what_produces(self, token: str) -> List[Tuple[int, float]]:
        """
        Which actions can produce this token?
        Returns [(action, probability), ...]
        """
        producers = self.token_producers.get(token, {})
        if not producers:
            return []
        
        total = sum(producers.values())
        return [(a, c/total) for a, c in sorted(producers.items(), key=lambda x: -x[1])]
    
    def what_consumes(self, token: str) -> List[Tuple[int, float]]:
        """
        Which actions consume this token?
        """
        consumers = self.token_consumers.get(token, {})
        if not consumers:
            return []
        
        total = sum(consumers.values())
        return [(a, c/total) for a, c in sorted(consumers.items(), key=lambda x: -x[1])]
    
    def what_requires(self, action: int) -> Set[str]:
        """
        What tokens are typically present when this action succeeds?
        """
        required = defaultdict(int)
        total = 0
        
        for (state, act), outcomes in self.exact_memory.items():
            if act == action:
                count = sum(outcomes.values())
                total += count
                for token in state:
                    required[token] += count
        
        if total == 0:
            return set()
        
        # Return tokens present in >50% of successful applications
        return {t for t, c in required.items() if c / total > 0.5}
    
    # =========================================================================
    # INDUCTIVE REASONING (Pattern Discovery)
    # =========================================================================
    
    def induce_patterns(self) -> List[InducedPattern]:
        """
        Discover general patterns across actions.
        
        "All sell actions convert an item to money"
        "All harvest actions consume growth tokens"
        """
        patterns = []
        
        # Group actions by their effect signatures
        signature_groups: Dict[str, List[int]] = defaultdict(list)
        
        for action, signatures in self.action_signatures.items():
            if not signatures:
                continue
            
            # Aggregate signature for this action
            all_consumed = set()
            all_produced = set()
            for consumed, produced in signatures:
                all_consumed.update(consumed)
                all_produced.update(produced)
            
            # Create a normalized signature
            # Extract token "types" by removing numeric suffixes
            def normalize(tokens):
                normalized = set()
                for t in tokens:
                    # Remove quantity buckets like _100+ or _3
                    base = re.sub(r'_\d+\+?$', '', t)
                    normalized.add(base)
                return frozenset(normalized)
            
            sig_key = (normalize(all_consumed), normalize(all_produced))
            signature_groups[str(sig_key)].append(action)
        
        # Find patterns with multiple actions
        for sig_str, actions in signature_groups.items():
            if len(actions) >= 2:
                # This is a pattern!
                # Analyze what's common
                
                # Get example signatures
                examples = []
                for action in actions[:5]:
                    sigs = self.action_signatures[action]
                    if sigs:
                        consumed, produced = sigs[0]
                        examples.append((action, consumed, produced))
                
                # Try to infer pattern type
                all_produced = set()
                for _, _, produced in examples:
                    all_produced.update(produced)
                
                if any('💰' in str(p) or 'money' in str(p).lower() for p in all_produced):
                    pattern_type = "sell"
                elif any('🎯' in str(p) for p in all_produced):
                    pattern_type = "tool_use"
                else:
                    pattern_type = "transform"
                
                pattern = InducedPattern(
                    pattern_type=pattern_type,
                    input_pattern=sig_str.split(',')[0] if ',' in sig_str else sig_str,
                    output_pattern=sig_str.split(',')[1] if ',' in sig_str else "",
                    confidence=len(actions) / len(self.action_signatures),
                    coverage=len(actions),
                    examples=examples
                )
                patterns.append(pattern)
        
        self.induced_patterns = sorted(patterns, key=lambda p: -p.coverage)
        return self.induced_patterns
    
    def predict_by_induction(self, state: Set[str], action: int) -> Optional[Prediction]:
        """
        Predict using induced patterns when exact/rule match fails.
        
        "I've never seen this exact sell action, but all sell actions
        produce money, so this probably does too."
        """
        if not self.induced_patterns:
            self.induce_patterns()
        
        # Get this action's signature
        signatures = self.action_signatures.get(action, [])
        if not signatures:
            return None
        
        consumed, produced = signatures[0] if signatures else (set(), set())
        
        # Find matching pattern
        for pattern in self.induced_patterns:
            # Check if this action matches the pattern
            if action in [ex[0] for ex in pattern.examples]:
                continue  # Already covered by exact knowledge
            
            # Check signature similarity
            pattern_examples = pattern.examples
            if not pattern_examples:
                continue
            
            # If signature is similar to pattern examples...
            example_consumed = pattern_examples[0][1]
            example_produced = pattern_examples[0][2]
            
            # Very rough similarity check
            if len(consumed & example_consumed) > 0 or len(produced & example_produced) > 0:
                # Predict based on pattern
                predicted_effects = set()
                for ex_action, ex_consumed, ex_produced in pattern_examples:
                    for p in ex_produced:
                        # Generalize: if pattern produces X, this might too
                        predicted_effects.add(f"+{p}")
                    for c in ex_consumed:
                        predicted_effects.add(f"-{c}")
                
                return Prediction(
                    effects=predicted_effects,
                    confidence=pattern.confidence * 0.5,  # Lower confidence for induction
                    source="induced",
                    support=pattern.coverage,
                    alternatives=[]
                )
        
        return None
    
    # =========================================================================
    # TEMPORAL REASONING
    # =========================================================================
    
    def get_temporal_dependencies(self, action: int) -> TemporalDependency:
        """
        What typically precedes this action?
        """
        predecessors = self.action_predecessors.get(action, {})
        total = sum(predecessors.values())
        
        if total == 0:
            return TemporalDependency(
                action=action,
                prerequisites={},
                min_steps_from_start=0,
                typical_sequence=[]
            )
        
        # Normalize predecessor probabilities
        prereq_probs = {a: c/total for a, c in predecessors.items()}
        
        # Average step number
        steps = self.action_step_counts.get(action, [])
        avg_step = sum(steps) / len(steps) if steps else 0
        
        # Find typical sequence leading to this action
        typical_seq = self._find_typical_sequence(action)
        
        return TemporalDependency(
            action=action,
            prerequisites=prereq_probs,
            min_steps_from_start=avg_step,
            typical_sequence=typical_seq
        )
    
    def _find_typical_sequence(self, target_action: int, max_len: int = 5) -> List[int]:
        """Find most common action sequence leading to target"""
        # Work backwards from target
        sequence = [target_action]
        current = target_action
        
        for _ in range(max_len - 1):
            predecessors = self.action_predecessors.get(current, {})
            if not predecessors:
                break
            
            # Most common predecessor
            best_pred = max(predecessors.items(), key=lambda x: x[1])[0]
            sequence.insert(0, best_pred)
            current = best_pred
        
        return sequence
    
    # =========================================================================
    # IMAGINATION / DREAMING
    # =========================================================================
    
    def imagine_trajectory(self, 
                          initial_state: Set[str], 
                          action_sequence: List[int],
                          confidence_threshold: float = 0.3) -> Tuple[List[Set[str]], float, int]:
        """
        Imagine a trajectory through state space.
        
        Returns:
        - states: List of predicted states
        - cumulative_confidence: Overall confidence in the trajectory
        - steps_before_uncertain: How many steps before confidence dropped below threshold
        """
        states = [initial_state.copy()]
        cumulative_confidence = 1.0
        steps_certain = 0
        
        current_state = initial_state.copy()
        
        for i, action in enumerate(action_sequence):
            prediction = self.predict_with_confidence(current_state, action)
            
            # Apply predicted effects
            new_state = current_state.copy()
            for effect in prediction.effects:
                if effect.startswith('+'):
                    new_state.add(effect[1:])
                elif effect.startswith('-'):
                    token = effect[1:]
                    if token in new_state:
                        new_state.remove(token)
            
            states.append(new_state)
            cumulative_confidence *= prediction.confidence
            
            if cumulative_confidence >= confidence_threshold:
                steps_certain = i + 1
            
            current_state = new_state
        
        return states, cumulative_confidence, steps_certain
    
    def dream_rollout(self, 
                     initial_state: Set[str],
                     policy_fn,  # Callable[[Set[str]], int]
                     max_steps: int = 50,
                     min_confidence: float = 0.2) -> Tuple[List[Tuple[Set[str], int, float]], float]:
        """
        Perform a full dream rollout using a policy function.
        
        Returns trajectory of (state, action, step_confidence) tuples
        and overall trajectory confidence.
        """
        trajectory = []
        state = initial_state.copy()
        cumulative_conf = 1.0
        
        for step in range(max_steps):
            if cumulative_conf < min_confidence:
                break
            
            action = policy_fn(state)
            prediction = self.predict_with_confidence(state, action)
            
            trajectory.append((state.copy(), action, prediction.confidence))
            
            # Apply effects
            for effect in prediction.effects:
                if effect.startswith('+'):
                    state.add(effect[1:])
                elif effect.startswith('-'):
                    token = effect[1:]
                    state.discard(token)
            
            cumulative_conf *= prediction.confidence
        
        return trajectory, cumulative_conf
    
    # =========================================================================
    # NOVELTY / EXPLORATION BONUS
    # =========================================================================
    
    def novelty_score(self, state: Set[str], action: int) -> float:
        """
        How novel is this state-action pair?
        
        1.0 = completely novel (never seen)
        0.0 = very familiar (seen many times)
        
        Useful as intrinsic reward for exploration.
        """
        state_fs = frozenset(state)
        key = (state_fs, action)
        
        # Exact match count
        exact_count = self.state_action_counts.get(key, 0)
        if exact_count > 0:
            # Seen before - low novelty
            # Use log scale: 1 obs = 0.5, 10 obs = 0.2, 100 obs = 0.1
            return 1.0 / (1.0 + math.log1p(exact_count))
        
        # Check rule coverage
        rule = self._find_best_rule(state_fs, action)
        if rule:
            rule_obs = self.rules[rule]['observations']
            # Partially covered by rules - medium novelty
            return 0.5 / (1.0 + math.log1p(rule_obs) * 0.5)
        
        # Completely novel
        return 1.0
    
    def exploration_value(self, state: Set[str], action: int) -> float:
        """
        Combined exploration value considering:
        - Novelty (how new is this?)
        - Uncertainty (how unsure are we?)
        - Information gain (will this teach us something?)
        """
        novelty = self.novelty_score(state, action)
        
        prediction = self.predict_with_confidence(state, action)
        uncertainty = 1.0 - prediction.confidence
        
        # Information gain: how many alternatives? More = more to learn
        n_alternatives = len(prediction.alternatives)
        info_gain = min(1.0, n_alternatives / 5)
        
        # Combine (weights can be tuned)
        return 0.4 * novelty + 0.4 * uncertainty + 0.2 * info_gain
    
    # =========================================================================
    # INDUCTIVE REASONING (Path Inference) 
    # =========================================================================
    
    def induce_path(self, before: Set[str], after: Set[str], 
                    max_steps: int = 10) -> List[Tuple[List[Tuple[FrozenSet, FrozenSet]], float]]:
        """
        Given two states, infer the most likely sequence of rules that
        transformed 'before' into 'after'.
        
        This is TRUE INDUCTION - finding the "missing middle" between
        observed states by factorizing the diff into known primitive rules.
        
        Uses the currying principle: prefer explanations with fewer steps
        that use well-established rules.
        
        Returns: List of (rule_sequence, confidence) where each rule is
                 represented as (consumed_tokens, produced_tokens)
        """
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        if before_fs == after_fs:
            return [([], 1.0)]  # No transformation needed
        
        # Build rule index from our observations
        rules_consuming: Dict[str, List] = defaultdict(list)
        rules_producing: Dict[str, List] = defaultdict(list)
        
        for (state, action), outcomes in self.exact_memory.items():
            for effects, count in outcomes.items():
                consumed = frozenset(e[1:] for e in effects if e.startswith('-'))
                produced = frozenset(e[1:] for e in effects if e.startswith('+'))
                
                if consumed or produced:
                    total = sum(outcomes.values())
                    confidence = count / total
                    rule = (consumed, produced, confidence, count)
                    
                    for token in consumed:
                        rules_consuming[token].append(rule)
                    for token in produced:
                        rules_producing[token].append(rule)
        
        # A* search for best factorization
        def score(current, target, rules_applied, conf):
            distance = len(current - target) + len(target - current)
            return conf - len(rules_applied) * 0.05 - distance * 0.1
        
        initial = (before_fs, (), 1.0)
        frontier = [(-score(before_fs, after_fs, (), 1.0), 0, initial)]
        visited = {before_fs}
        solutions = []
        counter = 1
        
        while frontier and len(solutions) < 10:
            _, _, (current, rules_applied, conf) = heapq.heappop(frontier)
            
            if current == after_fs:
                solutions.append((list(rules_applied), conf))
                continue
            
            if len(rules_applied) >= max_steps:
                continue
            
            # Find applicable rules that help
            to_remove = current - after_fs
            to_add = after_fs - current
            
            candidate_rules = set()
            for token in to_remove:
                for r in rules_consuming.get(token, []):
                    candidate_rules.add(r)
            for token in to_add:
                for r in rules_producing.get(token, []):
                    candidate_rules.add(r)
            
            for consumed, produced, rule_conf, count in candidate_rules:
                if not consumed.issubset(current):
                    continue
                
                new_tokens = (current - consumed) | produced
                if new_tokens in visited:
                    continue
                
                visited.add(new_tokens)
                new_conf = conf * rule_conf
                new_rules = rules_applied + ((consumed, produced),)
                
                s = score(new_tokens, after_fs, new_rules, new_conf)
                heapq.heappush(frontier, (-s, counter, (new_tokens, new_rules, new_conf)))
                counter += 1
        
        solutions.sort(key=lambda x: -x[1])
        return solutions
    
    def explain_transition(self, before: Set[str], after: Set[str]) -> str:
        """
        Human-readable explanation of what rules likely fired between states.
        
        This is the "show your work" function - given X and Y, explain
        the most likely sequence of transformations.
        """
        paths = self.induce_path(before, after, max_steps=5)
        
        if not paths:
            consumed = before - after
            produced = after - before
            return f"No path found. Raw diff: -{consumed}, +{produced}"
        
        rules, confidence = paths[0]
        
        lines = [f"Inferred path ({confidence:.1%} confidence):"]
        
        current = before.copy()
        for i, (consumed, produced) in enumerate(rules):
            lines.append(f"  {i+1}. -{set(consumed)} → +{set(produced)}")
            current = (current - set(consumed)) | set(produced)
        
        if len(paths) > 1:
            lines.append(f"\n  ({len(paths)-1} alternative explanations found)")
        
        return "\n".join(lines)
    
    def minimum_steps_estimate(self, before: Set[str], after: Set[str]) -> int:
        """
        Estimate minimum steps via currying principle.
        
        If the diff is N token changes and our largest rule changes M tokens,
        we need at least ceil(N/M) steps.
        """
        consumed = before - after
        produced = after - before
        total_diff = len(consumed) + len(produced)
        
        if total_diff == 0:
            return 0
        
        # Find max rule size from observations
        max_rule_size = 1
        for outcomes in self.exact_memory.values():
            for effects in outcomes.keys():
                size = len(effects)
                if size > max_rule_size:
                    max_rule_size = size
        
        return max(1, (total_diff + max_rule_size - 1) // max_rule_size)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def stats(self) -> Dict:
        """Return statistics about learned knowledge"""
        return {
            'total_observations': self.total_observations,
            'unique_state_action_pairs': len(self.exact_memory),
            'rules': len(self.rules),
            'tokens_tracked': len(self.token_producers) + len(self.token_consumers),
            'induced_patterns': len(self.induced_patterns),
            'episodes_seen': len(self.action_sequences),
            'actions_with_temporal_data': len(self.action_predecessors)
        }
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'conn'):
            self.conn.close()


# Alias for compatibility
HierarchicalLearner = HierarchicalLearnerV12
