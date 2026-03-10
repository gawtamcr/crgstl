import numpy as np
from typing import Dict, Callable, Optional

class STLPredicate:
    def __init__(   self, robustness_fn: Callable[[Dict], float], 
                    reward_scale: float = 1.0, 
                    target_fn: Optional[Callable[[Dict], np.ndarray]] = None):
        self.robustness_fn = robustness_fn
        self.reward_scale = reward_scale
        self.target_fn = target_fn

    def __call__(self, obs: Dict) -> float:
        return self.robustness_fn(obs)

    def compute_reward(self, obs: Dict, action: np.ndarray) -> float:
        return self.robustness_fn(obs) * self.reward_scale

    def get_target(self, obs: Dict) -> np.ndarray:
        if self.target_fn:
            return self.target_fn(obs)
        return np.zeros(3, dtype=np.float32)

def define_predicates() -> Dict[str, STLPredicate]:
    
    def dist_ee_obj(o):
        return np.linalg.norm(o['observation'][:3] - o['achieved_goal'][:3])

    def dist_obj_goal(o):
        return np.linalg.norm(o['achieved_goal'][:3] - o['desired_goal'][:3])

    def obj_height(o):
        return o['achieved_goal'][2]

    def get_obj_pos(o):
        return o['achieved_goal'][:3]

    def get_goal_pos(o):
        return o['desired_goal'][:3]

    return {
        "approach": STLPredicate(
            robustness_fn=lambda o: 0.02 - dist_ee_obj(o),
            reward_scale=5.0,
            target_fn=get_obj_pos
        ),
        "grasp": STLPredicate(
            robustness_fn=lambda o: obj_height(o) - 0.05,
            reward_scale=20.0,
            target_fn=get_obj_pos
        ),
        "move": STLPredicate(
            robustness_fn=lambda o: 0.05 - dist_obj_goal(o),
            reward_scale=10.0,
            target_fn=get_goal_pos
        )
    }

def define_stacking_predicates() -> Dict[str, STLPredicate]:
    # Offsets based on PandaGym observation structure:
    # [0:7] Robot (EE pos at 0:3)
    # [7:20] Object 1 (Pos at 7:10)
    # [20:33] Object 2 (Pos at 20:23)
    
    def get_ee(o): return o['observation'][:3]
    def get_obj1(o): return o['observation'][7:10]
    def get_obj2(o): return o['observation'][20:23]
    def get_goal(o): return o['desired_goal'][:3]

    return {
        # --- CUBE 1 ---
        "approach_1": STLPredicate(
            robustness_fn=lambda o: 0.02 - np.linalg.norm(get_ee(o) - get_obj1(o)),
            target_fn=get_obj1
        ),
        "grasp_1": STLPredicate(
            robustness_fn=lambda o: get_obj1(o)[2] - 0.05, # Lift check
            target_fn=get_obj1
        ),
        "place_1": STLPredicate(
            robustness_fn=lambda o: 0.05 - np.linalg.norm(get_obj1(o) - get_goal(o)),
            target_fn=get_goal
        ),

        # --- CUBE 2 ---
        "approach_2": STLPredicate(
            robustness_fn=lambda o: 0.05 - np.linalg.norm(get_ee(o) - get_obj2(o)),
            target_fn=get_obj2
        ),
        "grasp_2": STLPredicate(
            robustness_fn=lambda o: get_obj2(o)[2] - 0.05, # Lift check
            target_fn=get_obj2
        ),
        "stack_2": STLPredicate(
            # Target is on top of Cube 1
            robustness_fn=lambda o: 0.05 - np.linalg.norm(get_obj2(o) - (get_obj1(o) + [0,0,0.04])),
            target_fn=lambda o: get_obj1(o) + [0, 0, 0.04]
        )
    }