"""
Token-Based State Machine Learner

A general framework for learning state transition rules from observations.

Key concepts:
- State = FrozenSet[str] of tokens (unordered)
- Action = FrozenSet[str] of tokens (optional)
- Rule = (pattern, effect, confidence)

Works for:
- Board games (TicTacToe, Chess positions as tokens)
- Adventure games (location, inventory, HP as tokens)
- Video games (Pokemon - sprites, UI elements as tokens)
"""

from dataclasses import dataclass, field
from typing import Set, FrozenSet, Dict, List, Tuple, Optional, Any
from collections import defaultdict
from itertools import combinations
import random


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Observation:
    """An observation = immutable set of tokens."""
    tokens: FrozenSet[str]
    
    def __init__(self, tokens):
        if isinstance(tokens, str):
            tokens = {tokens}
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
        return Observation(self.tokens | other)
    
    def __and__(self, other):
        if isinstance(other, Observation):
            return Observation(self.tokens & other.tokens)
        return Observation(self.tokens & other)
    
    def __sub__(self, other):
        if isinstance(other, Observation):
            return Observation(self.tokens - other.tokens)
        return Observation(self.tokens - other)
    
    def matches(self, pattern: 'Observation') -> bool:
        """Does this observation contain all tokens in pattern?"""
        return pattern.tokens <= self.tokens
    
    def masked(self, visible: Set[str]) -> 'Observation':
        """Return observation with only specified tokens visible."""
        return Observation(self.tokens & visible)


@dataclass
class Transition:
    """A state transition."""
    before: Observation
    action: Optional[Observation]
    after: Observation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def input_tokens(self) -> FrozenSet[str]:
        """Combined input tokens."""
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


@dataclass
class Rule:
    """A learned rule: pattern -> effect."""
    pattern: FrozenSet[str]
    effect: str  # "+token" for added, "-token" for removed
    confidence: float
    support: int
    
    @property
    def is_addition(self) -> bool:
        return self.effect.startswith('+')
    
    @property
    def is_removal(self) -> bool:
        return self.effect.startswith('-')
    
    @property
    def target_token(self) -> str:
        return self.effect[1:]


# ============================================================================
# LEARNER
# ============================================================================

class TokenTransitionLearner:
    """
    Learns state transition rules from token-based observations.
    
    No hardcoded game knowledge - discovers patterns from data.
    """
    
    def __init__(self, 
                 max_pattern_size: int = 3,
                 min_support: int = 5,
                 min_confidence: float = 0.8):
        self.max_pattern_size = max_pattern_size
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        # Statistics
        self.effects: Dict[FrozenSet[str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.pattern_counts: Dict[FrozenSet[str], int] = defaultdict(int)
        
        # Extracted rules
        self.rules: List[Rule] = []
        
        # Tracking
        self.transitions_observed = 0
        self.vocabulary: Set[str] = set()
    
    def observe(self, transition: Transition):
        """Learn from a transition."""
        self.transitions_observed += 1
        
        input_tokens = transition.input_tokens
        self.vocabulary.update(input_tokens)
        self.vocabulary.update(transition.after.tokens)
        
        # Track pattern -> effect associations
        for size in range(1, min(self.max_pattern_size + 1, len(input_tokens) + 1)):
            for pattern in combinations(sorted(input_tokens), size):
                pattern_set = frozenset(pattern)
                self.pattern_counts[pattern_set] += 1
                
                for token in transition.added:
                    self.effects[pattern_set][f"+{token}"] += 1
                for token in transition.removed:
                    self.effects[pattern_set][f"-{token}"] += 1
    
    def observe_batch(self, transitions: List[Transition]):
        """Learn from multiple transitions."""
        for t in transitions:
            self.observe(t)
    
    def extract_rules(self) -> List[Rule]:
        """Extract high-confidence rules from observations."""
        self.rules = []
        
        for pattern, effects in self.effects.items():
            support = self.pattern_counts[pattern]
            if support < self.min_support:
                continue
            
            for effect, count in effects.items():
                confidence = count / support
                if confidence >= self.min_confidence:
                    self.rules.append(Rule(pattern, effect, confidence, support))
        
        # Sort by confidence, then support, then simplicity
        self.rules.sort(key=lambda r: (-r.confidence, -r.support, len(r.pattern)))
        return self.rules
    
    def predict(self, 
                state: Observation, 
                action: Optional[Observation] = None) -> Dict[str, float]:
        """
        Predict effects of transitioning from state (with optional action).
        
        Returns: {effect: confidence} for all predicted effects
        """
        input_tokens = state.tokens
        if action:
            input_tokens = input_tokens | action.tokens
        
        predictions = {}
        for rule in self.rules:
            if rule.pattern <= input_tokens:
                # Pattern matches - record prediction with highest confidence
                if rule.effect not in predictions or predictions[rule.effect] < rule.confidence:
                    predictions[rule.effect] = rule.confidence
        
        return predictions
    
    def predict_next_state(self,
                           state: Observation,
                           action: Optional[Observation] = None,
                           threshold: float = 0.5) -> Observation:
        """
        Predict the next state after a transition.
        
        Args:
            state: Current state
            action: Action taken (optional)
            threshold: Minimum confidence to apply a rule
        
        Returns:
            Predicted next state
        """
        predictions = self.predict(state, action)
        
        next_tokens = set(state.tokens)
        
        for effect, confidence in predictions.items():
            if confidence >= threshold:
                if effect.startswith('+'):
                    next_tokens.add(effect[1:])
                elif effect.startswith('-'):
                    next_tokens.discard(effect[1:])
        
        return Observation(next_tokens)
    
    def get_rules_for_effect(self, effect: str) -> List[Rule]:
        """Get all rules that predict a specific effect."""
        return [r for r in self.rules if r.effect == effect]
    
    def get_minimal_rules(self, effect: str) -> List[Rule]:
        """Get minimal (non-redundant) rules for an effect."""
        relevant = self.get_rules_for_effect(effect)
        
        minimal = []
        for r1 in relevant:
            is_minimal = True
            for r2 in relevant:
                # r2 is a proper subset with similar or better confidence
                if (r2.pattern < r1.pattern and 
                    r2.confidence >= r1.confidence * 0.95):
                    is_minimal = False
                    break
            if is_minimal:
                minimal.append(r1)
        
        return minimal
    
    def predict_with_partial_obs(self,
                                  visible: Observation,
                                  action: Optional[Observation] = None,
                                  hidden_fraction: float = 0.0) -> Dict[str, float]:
        """
        Predict with partial observability.
        
        Args:
            visible: Tokens that are visible
            action: Action taken
            hidden_fraction: Estimated fraction of state that's hidden
        
        Returns:
            {effect: probability} with adjusted confidences
        """
        input_tokens = visible.tokens
        if action:
            input_tokens = input_tokens | action.tokens
        
        predictions = defaultdict(float)
        
        for rule in self.rules:
            if rule.pattern <= input_tokens:
                # Full match
                predictions[rule.effect] = max(predictions[rule.effect], rule.confidence)
            elif hidden_fraction > 0:
                # Partial match - some pattern tokens might be hidden
                visible_match = rule.pattern & input_tokens
                hidden_match = rule.pattern - input_tokens
                
                if len(hidden_match) <= len(rule.pattern) * hidden_fraction:
                    # Pattern might match if hidden tokens are present
                    verified = len(visible_match) / len(rule.pattern)
                    adjusted = rule.confidence * verified * 0.5
                    predictions[rule.effect] = max(predictions[rule.effect], adjusted)
        
        return dict(predictions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            'transitions_observed': self.transitions_observed,
            'vocabulary_size': len(self.vocabulary),
            'patterns_tracked': len(self.pattern_counts),
            'rules_extracted': len(self.rules),
        }


# ============================================================================
# UTILITIES
# ============================================================================

def create_transition(before_tokens, action_tokens, after_tokens, **metadata):
    """Helper to create transitions from token iterables."""
    return Transition(
        before=Observation(before_tokens),
        action=Observation(action_tokens) if action_tokens else None,
        after=Observation(after_tokens),
        metadata=metadata
    )
