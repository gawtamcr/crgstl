from typing import Dict, Callable, Any

class RewardRegistry:
    """
    Registry to map predicate names to reward functions.
    """
    def __init__(self):
        self.rewards: Dict[str, Callable[[Any], float]] = {}
        self.penalties: Dict[str, Callable[[Any], float]] = {}

    def register_objective(self, name: str, func: Callable[[Any], float]):
        """Register a dense reward function for a progress predicate."""
        self.rewards[name] = func

    def register_safety(self, name: str, func: Callable[[Any], float]):
        """Register a soft penalty function for a safety predicate."""
        self.penalties[name] = func

    def get_reward(self, name: str, obs: Any) -> float:
        if name in self.rewards:
            return self.rewards[name](obs)
        return 0.0

    def get_penalty(self, name: str, obs: Any) -> float:
        if name in self.penalties:
            return self.penalties[name](obs)
        return 0.0
