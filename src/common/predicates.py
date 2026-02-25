import numpy as np
from typing import Dict, Callable

class STLPredicate:
    def __init__(self, robustness_fn: Callable[[Dict], float], reward_scale: float = 1.0):
        self.robustness_fn = robustness_fn
        self.reward_scale = reward_scale

    def __call__(self, obs: Dict) -> float:
        return self.robustness_fn(obs)

    def compute_reward(self, obs: Dict, action: np.ndarray) -> float:
        return self.robustness_fn(obs) * self.reward_scale

def define_predicates() -> Dict[str, STLPredicate]:
    
    def dist_ee_obj(o):
        return np.linalg.norm(o['observation'][:3] - o['achieved_goal'][:3])

    def dist_obj_goal(o):
        return np.linalg.norm(o['achieved_goal'][:3] - o['desired_goal'][:3])

    def obj_height(o):
        return o['achieved_goal'][2]

    return {
        "approach": STLPredicate(
            robustness_fn=lambda o: 0.02 - dist_ee_obj(o),
            reward_scale=5.0
        ),
        "grasp": STLPredicate(
            robustness_fn=lambda o: obj_height(o) - 0.05,
            reward_scale=20.0
        ),
        "move": STLPredicate(
            robustness_fn=lambda o: 0.05 - dist_obj_goal(o),
            reward_scale=10.0
        )
    }