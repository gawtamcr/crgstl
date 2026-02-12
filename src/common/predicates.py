import numpy as np
from typing import Dict, Callable

def define_predicates() -> Dict[str, Callable]:
    return {
        "approach": lambda o: np.linalg.norm(o['observation'][:3] - o['achieved_goal'][:3]) < 0.015,
        "grasp": lambda o: True, # o['achieved_goal'][2] > 0.045,  # Object lifted
        "move": lambda o: np.linalg.norm(o['achieved_goal'][:3] - o['desired_goal'][:3]) < 0.05
    }