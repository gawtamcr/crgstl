# src/common/predicates.py
import numpy as np

def define_predicates():
    """
    Returns a dictionary of predicates (name -> callable).
    Each callable takes the observation dict and returns a boolean.
    """
    
    def approach(obs):
        # Distance between End Effector and Object
        # obs['observation'][0:3] is EE pos, obs['achieved_goal'] is Object pos
        ee_pos = obs['observation'][0:3]
        obj_pos = obs['achieved_goal']
        dist = np.linalg.norm(ee_pos - obj_pos)
        return dist < 0.01 # 5 cm threshold

    def grasp(obs):
        # Alias for holding, or specific grasp logic
        # Here we map 'grasp' to the holding logic for the STL formula
        return holding(obs)

    def move(obs):
        # Distance between Object and Target
        obj_pos = obs['achieved_goal']
        target_pos = obs['desired_goal']
        dist = np.linalg.norm(obj_pos - target_pos)
        return dist < 0.05

    def holding(obs):
        # Check if gripper is holding something.
        # Panda gripper: ~0.0 is closed, ~0.08 is open.
        # Object is usually ~0.04.
        # We return True if width is "small but not zero".
        width = obs['observation'][6]
        return 0.005 < width < 0.05

    return {
        "approach": approach,
        "grasp": grasp,
        "move": move,
        "holding": holding
    }
