# src/controller/safe_funnel_controller.py
import numpy as np

class SafeFunnelController:
    def __init__(self, position_gain: float = 8.0):
        self.kp = position_gain

    def get_action(self, phase: str, safety_constraints, obs: dict, time_left: float) -> np.ndarray:
        # obs['observation'] = [ee_x, ee_y, ee_z, ee_vx, ee_vy, ee_vz, width, ...]
        ee_pos = obs['observation'][0:3]
        obj_pos = obs['achieved_goal']
        target_pos = obs['desired_goal']
        
        # Default action: [vx, vy, vz, gripper]
        # gripper: 1.0 = open, -1.0 = closed
        action = np.zeros(4)
        
        if phase == "approach":
            # Move to object, Open gripper
            error = obj_pos - ee_pos
            action[0:3] = error * self.kp
            action[3] = 1.0 
            
        elif phase == "grasp":
            # Stay at object, Close gripper
            error = obj_pos - ee_pos
            action[0:3] = error * self.kp
            action[3] = -1.0 
            
        elif phase == "move":
            # Move to target, Keep gripper closed
            error = target_pos - ee_pos
            action[0:3] = error * self.kp
            action[3] = -1.0 
            
        elif phase == "holding":
            # Handle 'holding' if used as a phase name
            error = target_pos - ee_pos
            action[0:3] = error * self.kp
            action[3] = -1.0

        else:
            # Default/Done: Hover and hold
            action[0:3] = 0.0
            action[3] = -1.0 
            
        return action
