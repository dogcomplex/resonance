"""
Base interface for few-shot learning algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class Algorithm(ABC):
    """Base class for few-shot learners."""
    
    def __init__(self):
        self.history: List[Tuple[str, int, int]] = []  # (observation, guess, correct)
    
    @abstractmethod
    def predict(self, observation: str) -> int:
        """Predict label index for observation."""
        pass
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        """Learn from observation."""
        self.history.append((observation, guess, correct_label))
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return current learner statistics."""
        pass
