"""
UNIFIED FAIR LEARNER

A single learner that handles:
1. Classification tasks (TicTacToe: state → label)
2. Navigation tasks (MiniGrid: state + action → next_state)
3. Sequential decision making (Mini RPG, text adventures)

Key principles:
- NO DOMAIN KNOWLEDGE - all patterns learned from observation
- Token-based representation - state = frozenset of opaque tokens
- Pure rule discovery - patterns with 100% confidence
- Cycle detection - for action semantics (rotation vs movement)
- Interaction learning - for state-changing actions

The core abstraction:
    Observation = frozenset of tokens
    Transition = (before, action, after)
    Rule = (pattern → effect) with confidence

This unifies the TicTacToe HybridLearner (88%) and MiniGrid FairLearnerV3 (94%).
"""

from dataclasses import dataclass, field
from typing import Set, FrozenSet, Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
from itertools import combinations
from enum import Enum
import random


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Observation:
    """
    An observation = immutable set of opaque tokens.
    
    Tokens are just strings with no semantic meaning to the learner.
    Examples:
        TicTacToe: {'p0=1', 'p1=2', 'p4=0', ...}
        MiniGrid:  {'front=T2', 'left=T1', 'right=T0', 'G'}
        Mini RPG:  {'at_0_0', 'hp=10', 'has_sword', 'room=village'}
    """
    tokens: FrozenSet[str]
    
    def __init__(self, tokens):
        if isinstance(tokens, str):
            tokens = {tokens}
        if isinstance(tokens, (list, tuple)):
            tokens = set(tokens)
        object.__setattr__(self, 'tokens', frozenset(tokens))
    
    def __contains__(self, item):
        return item in self.tokens
    
    def __iter__(self):
        return iter(self.tokens)
    
    def __len__(self):
        return len(self.tokens)
    
    def __or__(self, other):
        if isinstance(other, Observation):
            return Observation(self.tokens | other.tokens)
        return Observation(self.tokens | set(other))
    
    def __and__(self, other):
        if isinstance(other, Observation):
            return Observation(self.tokens & other.tokens)
        return Observation(self.tokens & set(other))
    
    def __sub__(self, other):
        if isinstance(other, Observation):
            return Observation(self.tokens - other.tokens)
        return Observation(self.tokens - set(other))
    
    def matches(self, pattern: FrozenSet[str]) -> bool:
        """Does this observation contain all tokens in pattern?"""
        return pattern <= self.tokens
    
    def __repr__(self):
        return f"Obs({set(self.tokens)})"


@dataclass
class Transition:
    """A state transition with optional action."""
    before: Observation
    action: Optional[Observation]
    after: Observation
    outcome: Optional[str] = None  # For classification: the label
    
    @property
    def input_tokens(self) -> FrozenSet[str]:
        """Combined input tokens (state + action)."""
        if self.action:
            return self.before.tokens | self.action.tokens
        return self.before.tokens
    
    @property
    def added(self) -> FrozenSet[str]:
        """Tokens added in this transition."""
        return self.after.tokens - self.before.tokens
    
    @property
    def removed(self) -> FrozenSet[str]:
        """Tokens removed in this transition."""
        return self.before.tokens - self.after.tokens
    
    @property
    def changed(self) -> bool:
        """Did the state change?"""
        return self.before.tokens != self.after.tokens


@dataclass
class Rule:
    """A learned rule: pattern → effect."""
    pattern: FrozenSet[str]
    effect: str  # "+token" for add, "-token" for remove, or label name
    confidence: float
    support: int
    
    @property
    def is_addition(self) -> bool:
        return self.effect.startswith('+')
    
    @property
    def is_removal(self) -> bool:
        return self.effect.startswith('-')
    
    @property
    def is_classification(self) -> bool:
        return not (self.is_addition or self.is_removal)
    
    @property
    def target_token(self) -> str:
        if self.is_addition or self.is_removal:
            return self.effect[1:]
        return self.effect
    
    def matches(self, obs: Observation) -> bool:
        return self.pattern <= obs.tokens
    
    def __repr__(self):
        return f"Rule({set(self.pattern)} → {self.effect}, conf={self.confidence:.2f}, n={self.support})"


# =============================================================================
# UNIFIED FAIR LEARNER
# =============================================================================

class UnifiedFairLearner:
    """
    A unified learner for classification and navigation tasks.
    
    NO DOMAIN KNOWLEDGE - learns everything from observation:
    - Pattern → label rules (classification)
    - Pattern → state change rules (navigation)
    - Action semantics (rotation vs movement)
    - Interaction effects (pickup, toggle)
    - Goal discovery (what leads to success)
    
    Works for: TicTacToe, MiniGrid, Mini RPG, text adventures, etc.
    """
    
    def __init__(self,
                 max_pattern_size: int = 3,
                 min_support: int = 3,
                 min_confidence: float = 0.8,
                 pure_threshold: float = 1.0):
        """
        Args:
            max_pattern_size: Maximum number of tokens in a pattern
            min_support: Minimum observations to consider a rule
            min_confidence: Minimum confidence to extract a rule
            pure_threshold: Confidence threshold for "pure" rules (no exceptions)
        """
        self.max_pattern_size = max_pattern_size
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.pure_threshold = pure_threshold
        
        # ===== TRANSITION LEARNING =====
        # Pattern → effect statistics
        self.effect_counts: Dict[FrozenSet[str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pattern_counts: Dict[FrozenSet[str], int] = defaultdict(int)
        
        # ===== CLASSIFICATION LEARNING =====
        # Pattern → label statistics (for static classification like TicTacToe)
        self.label_counts: Dict[FrozenSet[str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.label_totals: Dict[str, int] = defaultdict(int)
        
        # ===== ACTION SEMANTICS =====
        # Cycle detection for rotation vs movement
        self.two_cycles: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"returns": 0, "total": 0})
        self.action_types: Dict[str, str] = {}  # action → 'rotation' | 'movement' | 'interaction' | 'unknown'
        
        # ===== INTERACTION LEARNING =====
        # (front_token, action, has_special) → success/fail counts
        self.interaction_stats: Dict[Tuple, Dict[str, int]] = defaultdict(lambda: {"success": 0, "fail": 0})
        
        # ===== GOAL LEARNING =====
        # Token → count of times it preceded success
        self.pre_success_tokens: Dict[str, int] = defaultdict(int)
        self.goal_token: Optional[str] = None
        
        # ===== EXTRACTED RULES =====
        self.rules: List[Rule] = []
        self.pure_rules: Dict[str, List[Rule]] = defaultdict(list)  # label → pure rules
        
        # ===== VOCABULARY =====
        self.vocabulary: Set[str] = set()
        self.action_vocabulary: Set[str] = set()
        
        # ===== TRACKING =====
        self.last_obs: Optional[Observation] = None
        self.last_action: Optional[str] = None
        self.transitions_observed: int = 0
        self.classifications_observed: int = 0
    
    # =========================================================================
    # OBSERVATION METHODS
    # =========================================================================
    
    def observe_transition(self, transition: Transition):
        """
        Learn from a state transition.
        
        Used for navigation/sequential tasks.
        """
        self.transitions_observed += 1
        
        # Update vocabulary
        self.vocabulary.update(transition.before.tokens)
        self.vocabulary.update(transition.after.tokens)
        if transition.action:
            self.action_vocabulary.update(transition.action.tokens)
        
        input_tokens = transition.input_tokens
        
        # Track pattern → effect associations
        for size in range(1, min(self.max_pattern_size + 1, len(input_tokens) + 1)):
            for pattern in combinations(sorted(input_tokens), size):
                pattern_set = frozenset(pattern)
                self.pattern_counts[pattern_set] += 1
                
                for token in transition.added:
                    self.effect_counts[pattern_set][f"+{token}"] += 1
                for token in transition.removed:
                    self.effect_counts[pattern_set][f"-{token}"] += 1
        
        # Track 2-cycles for rotation detection
        # A 2-cycle is: state S1 --(action A)--> S2 --(action B)--> S1
        # We need: did doing action B return us to state BEFORE action A?
        
        if transition.action and len(transition.action.tokens) == 1:
            action = next(iter(transition.action.tokens))
            
            # Check 2-cycle: if we return to self._pre_last_obs
            if hasattr(self, '_pre_last_obs') and self._pre_last_obs and self.last_action:
                key = (self.last_action, action)
                self.two_cycles[key]["total"] += 1
                if self._pre_last_obs.tokens == transition.after.tokens:
                    self.two_cycles[key]["returns"] += 1
            
            # Update the chain for next check
            self._pre_last_obs = transition.before
            self.last_action = action
            self.last_obs = transition.after
        else:
            self._pre_last_obs = None
            self.last_action = None
            self.last_obs = transition.after
    
    def observe_classification(self, obs: Observation, label: str):
        """
        Learn from a classification example.
        
        Used for static classification tasks like TicTacToe.
        """
        self.classifications_observed += 1
        self.vocabulary.update(obs.tokens)
        self.label_totals[label] += 1
        
        # Track pattern → label associations
        for size in range(1, min(self.max_pattern_size + 1, len(obs.tokens) + 1)):
            for pattern in combinations(sorted(obs.tokens), size):
                pattern_set = frozenset(pattern)
                self.pattern_counts[pattern_set] += 1
                self.label_counts[pattern_set][label] += 1
    
    def observe_interaction(self, front_token: str, action: str, 
                           has_special: bool, success: bool):
        """
        Learn from an interaction attempt.
        
        Used for pickup/toggle/use actions.
        """
        key = (front_token, action, has_special)
        if success:
            self.interaction_stats[key]["success"] += 1
        else:
            self.interaction_stats[key]["fail"] += 1
    
    def observe_success(self, final_obs: Observation, last_action: Optional[str] = None):
        """
        Learn from a successful episode ending.
        
        Used for goal discovery.
        """
        # Track tokens that were present before success
        for token in final_obs.tokens:
            self.pre_success_tokens[token] += 1
        
        # Update goal token estimate
        if self.pre_success_tokens:
            self.goal_token = max(self.pre_success_tokens, key=self.pre_success_tokens.get)
    
    def reset_episode(self):
        """Reset per-episode tracking."""
        self.last_obs = None
        self.last_action = None
    
    # =========================================================================
    # RULE EXTRACTION
    # =========================================================================
    
    def extract_rules(self) -> List[Rule]:
        """Extract high-confidence rules from observations."""
        self.rules = []
        self.pure_rules.clear()
        
        # Extract transition rules (pattern → effect)
        for pattern, effects in self.effect_counts.items():
            support = self.pattern_counts[pattern]
            if support < self.min_support:
                continue
            
            for effect, count in effects.items():
                confidence = count / support
                if confidence >= self.min_confidence:
                    rule = Rule(pattern, effect, confidence, support)
                    self.rules.append(rule)
                    
                    if confidence >= self.pure_threshold:
                        self.pure_rules[effect].append(rule)
        
        # Extract classification rules (pattern → label)
        for pattern, labels in self.label_counts.items():
            support = self.pattern_counts[pattern]
            if support < self.min_support:
                continue
            
            for label, count in labels.items():
                confidence = count / support
                if confidence >= self.min_confidence:
                    rule = Rule(pattern, label, confidence, support)
                    self.rules.append(rule)
                    
                    if confidence >= self.pure_threshold:
                        self.pure_rules[label].append(rule)
        
        # Sort by confidence, then support, then simplicity
        self.rules.sort(key=lambda r: (-r.confidence, -r.support, len(r.pattern)))
        
        for label in self.pure_rules:
            self.pure_rules[label].sort(key=lambda r: (-r.confidence, -r.support, len(r.pattern)))
        
        return self.rules
    
    def discover_action_types(self):
        """
        Discover action semantics from 2-cycle analysis.
        
        Rotations: action pairs that return to same state (A0↔A1)
        Movement: actions that don't form 2-cycles with rotations
        """
        self.action_types.clear()
        
        # Find rotation pairs (high 2-cycle return rate in both directions)
        rotation_pairs = []
        for (a1, a2), stats in self.two_cycles.items():
            if stats["total"] < 10:
                continue
            fwd_rate = stats["returns"] / stats["total"]
            
            # Check reverse direction
            rev_stats = self.two_cycles.get((a2, a1), {"returns": 0, "total": 0})
            if rev_stats["total"] < 10:
                continue
            rev_rate = rev_stats["returns"] / rev_stats["total"]
            
            if fwd_rate > 0.9 and rev_rate > 0.9:
                rotation_pairs.append((a1, a2, (fwd_rate + rev_rate) / 2))
        
        # Mark rotation actions
        for a1, a2, _ in rotation_pairs:
            self.action_types[a1] = 'rotation'
            self.action_types[a2] = 'rotation'
        
        # Non-rotation actions with state effects are movement
        for action in self.action_vocabulary:
            if action not in self.action_types:
                # Check if this action causes state changes
                has_effects = any(
                    action in pattern 
                    for pattern in self.effect_counts 
                    if self.effect_counts[pattern]
                )
                if has_effects:
                    self.action_types[action] = 'movement'
                else:
                    self.action_types[action] = 'unknown'
        
        return self.action_types
    
    def get_rotation_pair(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the discovered rotation action pair (CCW, CW)."""
        rotations = [a for a, t in self.action_types.items() if t == 'rotation']
        if len(rotations) >= 2:
            return rotations[0], rotations[1]
        return None, None
    
    def get_movement_action(self) -> Optional[str]:
        """Get the discovered forward movement action."""
        movements = [a for a, t in self.action_types.items() if t == 'movement']
        return movements[0] if movements else None
    
    # =========================================================================
    # PREDICTION METHODS
    # =========================================================================
    
    def predict_effects(self, obs: Observation, 
                       action: Optional[Observation] = None) -> Dict[str, float]:
        """
        Predict effects of transitioning from state (with optional action).
        
        Returns: {effect: confidence} for all predicted effects
        """
        input_tokens = obs.tokens
        if action:
            input_tokens = input_tokens | action.tokens
        
        predictions = {}
        for rule in self.rules:
            if rule.pattern <= input_tokens:
                if rule.effect not in predictions or predictions[rule.effect] < rule.confidence:
                    predictions[rule.effect] = rule.confidence
        
        return predictions
    
    def predict_next_state(self, obs: Observation,
                          action: Optional[Observation] = None,
                          threshold: float = 0.5) -> Observation:
        """
        Predict the next state after a transition.
        """
        predictions = self.predict_effects(obs, action)
        
        next_tokens = set(obs.tokens)
        
        for effect, confidence in predictions.items():
            if confidence >= threshold:
                if effect.startswith('+'):
                    next_tokens.add(effect[1:])
                elif effect.startswith('-'):
                    next_tokens.discard(effect[1:])
        
        return Observation(next_tokens)
    
    def predict_label(self, obs: Observation) -> Optional[str]:
        """
        Predict classification label for observation.
        
        Uses phase-appropriate strategy like HybridLearner.
        """
        n_obs = self.classifications_observed
        
        # Calculate label frequencies for rare label boosting
        total = sum(self.label_totals.values()) or 1
        label_freq = {l: c / total for l, c in self.label_totals.items()}
        
        # Strategy 1: Check pure rules first (prioritize rare labels)
        for label in sorted(self.pure_rules.keys(), key=lambda l: label_freq.get(l, 0)):
            for rule in self.pure_rules.get(label, []):
                if rule.matches(obs) and rule.support >= 2:
                    return label
        
        # Strategy 2: Weighted voting among all matching rules
        if n_obs >= 20:
            votes: Dict[str, float] = defaultdict(float)
            
            for rule in self.rules:
                if rule.is_classification and rule.matches(obs):
                    label = rule.effect
                    rarity_boost = 1.0 / (label_freq.get(label, 0.1) + 0.01)
                    weight = rule.confidence * len(rule.pattern) * rarity_boost
                    votes[label] += weight
            
            if votes:
                return max(votes, key=votes.get)
        
        # Strategy 3: Fall back to prior
        if self.label_totals:
            return max(self.label_totals, key=self.label_totals.get)
        
        return None
    
    def interaction_works(self, front_token: str, action: str, has_special: bool) -> Optional[bool]:
        """Check if an interaction is known to work."""
        key = (front_token, action, has_special)
        stats = self.interaction_stats.get(key)
        
        if not stats:
            return None  # Unknown
        
        total = stats["success"] + stats["fail"]
        if total < 3:
            return None  # Not enough data
        
        return stats["success"] / total > 0.5
    
    def should_explore_interaction(self, front_token: str, action: str, has_special: bool) -> bool:
        """Should we explore this interaction?"""
        result = self.interaction_works(front_token, action, has_special)
        if result is None:
            return True  # Unknown, worth trying
        return result  # Try if it works
    
    # =========================================================================
    # NAVIGATION SUPPORT
    # =========================================================================
    
    def get_action_for_target(self, obs: Observation, target_token: str,
                             rng: random.Random) -> Optional[str]:
        """
        Get action to navigate toward target token.
        
        Uses learned rotation/movement actions.
        """
        tokens = list(obs.tokens)
        
        # Find positional tokens (front, left, right or similar)
        front = next((t for t in tokens if t.startswith('front=')), None)
        left = next((t for t in tokens if t.startswith('left=')), None)
        right = next((t for t in tokens if t.startswith('right=')), None)
        
        # Get action types
        rot_ccw, rot_cw = self.get_rotation_pair()
        move_fwd = self.get_movement_action()
        
        if front and left and right and rot_ccw and rot_cw and move_fwd:
            front_val = front.split('=')[1] if '=' in front else front
            left_val = left.split('=')[1] if '=' in left else left
            right_val = right.split('=')[1] if '=' in right else right
            
            if front_val == target_token:
                return move_fwd
            elif left_val == target_token:
                return rot_ccw
            elif right_val == target_token:
                return rot_cw
        
        # Fallback: random movement
        if move_fwd:
            if rng.random() < 0.6:
                return move_fwd
            elif rot_ccw and rot_cw:
                return rng.choice([rot_ccw, rot_cw])
        
        return None
    
    # =========================================================================
    # STATISTICS AND DEBUGGING
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            'transitions_observed': self.transitions_observed,
            'classifications_observed': self.classifications_observed,
            'vocabulary_size': len(self.vocabulary),
            'action_vocabulary': list(self.action_vocabulary),
            'patterns_tracked': len(self.pattern_counts),
            'rules_extracted': len(self.rules),
            'pure_rules': {k: len(v) for k, v in self.pure_rules.items()},
            'action_types': dict(self.action_types),
            'goal_token': self.goal_token,
        }
    
    def describe_knowledge(self) -> str:
        """Describe what the learner has discovered."""
        lines = ["=" * 60]
        lines.append("UNIFIED FAIR LEARNER - Knowledge Summary")
        lines.append("=" * 60)
        
        lines.append(f"\nObservations:")
        lines.append(f"  Transitions: {self.transitions_observed}")
        lines.append(f"  Classifications: {self.classifications_observed}")
        lines.append(f"  Vocabulary: {len(self.vocabulary)} tokens")
        lines.append(f"  Actions: {list(self.action_vocabulary)}")
        
        lines.append(f"\nAction Semantics:")
        for action, atype in sorted(self.action_types.items()):
            lines.append(f"  {action}: {atype}")
        
        rot_ccw, rot_cw = self.get_rotation_pair()
        move_fwd = self.get_movement_action()
        lines.append(f"  Rotation pair: {rot_ccw} ↔ {rot_cw}")
        lines.append(f"  Forward: {move_fwd}")
        
        lines.append(f"\nGoal Discovery:")
        lines.append(f"  Goal token: {self.goal_token}")
        lines.append(f"  Pre-success votes: {dict(self.pre_success_tokens)}")
        
        lines.append(f"\nRules Extracted: {len(self.rules)}")
        lines.append(f"  Pure rules by effect:")
        for effect, rules in sorted(self.pure_rules.items()):
            lines.append(f"    {effect}: {len(rules)} rules")
        
        lines.append(f"\nTop Rules:")
        for rule in self.rules[:10]:
            lines.append(f"  {rule}")
        
        lines.append(f"\nInteraction Knowledge:")
        for (front, action, has_special), stats in sorted(self.interaction_stats.items()):
            total = stats["success"] + stats["fail"]
            if total >= 3:
                rate = stats["success"] / total
                lines.append(f"  {front} + {action} (special={has_special}): {rate:.0%}")
        
        return '\n'.join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def obs_from_tictactoe(board: str) -> Observation:
    """Convert TicTacToe board string to Observation."""
    tokens = {f"p{i}={board[i]}" for i in range(len(board))}
    return Observation(tokens)

def obs_from_minigrid(front, left, right, *extras) -> Observation:
    """Convert MiniGrid view to Observation."""
    tokens = {f"front={front}", f"left={left}", f"right={right}"}
    tokens.update(extras)
    return Observation(tokens)

def obs_from_dict(state: Dict[str, Any]) -> Observation:
    """Convert state dictionary to Observation."""
    tokens = set()
    for key, value in state.items():
        if isinstance(value, bool):
            if value:
                tokens.add(key)
        else:
            tokens.add(f"{key}={value}")
    return Observation(tokens)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED FAIR LEARNER - Basic Tests")
    print("=" * 60)
    
    learner = UnifiedFairLearner()
    
    # Test 1: Classification (TicTacToe-style)
    print("\n--- Test 1: Classification ---")
    
    # Observe some TicTacToe patterns
    learner.observe_classification(obs_from_tictactoe("111000000"), "winX")
    learner.observe_classification(obs_from_tictactoe("111220000"), "winX")
    learner.observe_classification(obs_from_tictactoe("111022000"), "winX")
    learner.observe_classification(obs_from_tictactoe("000222000"), "winO")
    learner.observe_classification(obs_from_tictactoe("100222000"), "winO")
    learner.observe_classification(obs_from_tictactoe("010222000"), "winO")
    learner.observe_classification(obs_from_tictactoe("100020000"), "ok")
    learner.observe_classification(obs_from_tictactoe("010020000"), "ok")
    learner.observe_classification(obs_from_tictactoe("001020000"), "ok")
    
    rules = learner.extract_rules()
    print(f"Rules extracted: {len(rules)}")
    
    # Test prediction
    test_board = obs_from_tictactoe("111002000")
    pred = learner.predict_label(test_board)
    print(f"Prediction for '111002000': {pred} (expected: winX)")
    
    # Test 2: Navigation (MiniGrid-style)
    print("\n--- Test 2: Navigation ---")
    
    learner2 = UnifiedFairLearner()
    
    # Simulate rotation: A0, A1 are rotations, A2 is forward
    for _ in range(30):
        # A0 then A1 returns to same state
        obs1 = obs_from_minigrid("T1", "T2", "T0")
        learner2.observe_transition(Transition(
            before=obs1,
            action=Observation("A0"),
            after=obs_from_minigrid("T0", "T1", "T2")
        ))
        learner2.observe_transition(Transition(
            before=obs_from_minigrid("T0", "T1", "T2"),
            action=Observation("A1"),
            after=obs1
        ))
        
        # A2 changes position
        learner2.observe_transition(Transition(
            before=obs_from_minigrid("T0", "T1", "T2"),
            action=Observation("A2"),
            after=obs_from_minigrid("T3", "T0", "T1")
        ))
    
    learner2.discover_action_types()
    print(f"Action types: {learner2.action_types}")
    
    rot_ccw, rot_cw = learner2.get_rotation_pair()
    move_fwd = learner2.get_movement_action()
    print(f"Rotation pair: {rot_ccw}, {rot_cw}")
    print(f"Forward: {move_fwd}")
    
    print("\n--- Summary ---")
    print(learner.describe_knowledge())
