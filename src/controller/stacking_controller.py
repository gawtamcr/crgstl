import numpy as np
from typing import Dict, Optional

class StackingController:
    """
    Heuristic controller for the Stacking Task (2 Cubes).
    """
    def __init__(self, position_gain: float = 8.0):
        self.position_gain = position_gain

    def get_action(self, phase: str, safety_rule: Optional[str], 
                   obs_dict: Dict, t_left: float) -> np.ndarray:
        
        # Parse Observation
        obs = obs_dict['observation']
        ee_pos = obs[:3]
        obj1_pos = obs[7:10]
        obj2_pos = obs[20:23]
        stack_base_pos = obs_dict['desired_goal'][:3]

        action = np.zeros(4, dtype=np.float32)
        target = ee_pos # Default to stay put
        gripper = 0.0

        # --- SEQUENCE 1: BASE CUBE ---
        if phase == "approach_1":
            target = obj1_pos
            gripper = 1.0 # Open

        elif phase == "grasp_1":
            target = obj1_pos
            # Lift logic
            if np.linalg.norm(target - ee_pos) < 0.03:
                target = target.copy()
                target[2] += 0.15
            gripper = -1.0 # Close

        elif phase == "place_1":
            target = stack_base_pos
            # Maintain height while moving
            if np.linalg.norm(obj1_pos[:2] - target[:2]) > 0.05:
                target = target.copy()
                target[2] += 0.1
            gripper = -1.0 # Keep Closed

        # --- SEQUENCE 2: TOP CUBE ---
        elif phase == "approach_2":
            print("Approaching Cube 2")
            target = obj2_pos
            # Hover logic: Stay high until close
            if np.linalg.norm(ee_pos[:2] - obj2_pos[:2]) > 0.05:
                target = obj2_pos.copy()
                target[2] += 0.2
                
                # Safety override: If too low (near stack), go straight up first
                if ee_pos[2] < 0.15:
                    target = ee_pos.copy()
                    target[2] = 0.2
            gripper = 1.0 # Open (Releases Cube 1)

        elif phase == "grasp_2":
            target = obj2_pos
            if np.linalg.norm(target - ee_pos) < 0.03:
                target = target.copy()
                target[2] += 0.15
            gripper = -1.0 # Close

        elif phase == "stack_2":
            # Target is strictly above Cube 1
            target = obj1_pos.copy()
            target[2] += 0.045 # Stack height
            
            # Move high then drop
            if np.linalg.norm(ee_pos[:2] - target[:2]) > 0.02:
                target[2] += 0.15
            
            gripper = -1.0 # Keep Closed

        # Compute Velocity Control
        direction = target - ee_pos
        action[:3] = np.clip(direction * self.position_gain, -1.0, 1.0)
        action[3] = gripper

        return action
