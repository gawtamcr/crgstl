import numpy as np
from typing import Dict, Optional

class SafeFunnelController:
    """
    Heuristic controller that demonstrates desired behavior for each phase.
    Uses proportional control to move toward phase-specific targets.
    Includes safety barrier (repulsion) logic.
    """
    
    def __init__(self, position_gain: float = 8.0, obstacle_pos: Optional[np.ndarray] = None, obstacle_radius: float = 0.10):
        self.position_gain = position_gain
        self.obstacle_pos = obstacle_pos if obstacle_pos is not None else np.array([0.0, 0.0, 0.2])
        self.obstacle_radius = obstacle_radius
    
    def get_action(self, phase: str, safety_rule: Optional[str], 
                   obs_dict: Dict, t_left: float) -> np.ndarray:
        """
        Compute control action based on current phase.
        
        Args:
            phase: Current STL phase name
            safety_rule: Current safety constraint (if any)
            obs_dict: Environment observation dictionary
            t_left: Time remaining in current phase
            
        Returns:
            4D action: [dx, dy, dz, gripper]
        """
        ee_pos = obs_dict['observation'][:3]
        action = np.zeros(4, dtype=np.float32)
        
        # --- 1. ATTRACTION (The Funnel) ---
        if phase == "approach":
            # Move toward object with open gripper
            target = obs_dict['achieved_goal'][:3]
            direction = target - ee_pos
            action[:3] = np.clip(direction * self.position_gain, -1.0, 1.0)
            action[3] = 1.0  # Open gripper
            
        elif phase == "grasp":
            # Move to object and close gripper
            target = obs_dict['achieved_goal'][:3]
            direction = target - ee_pos
            action[:3] = np.clip(direction * self.position_gain, -1.0, 1.0)
            action[3] = -1.0  # Close gripper
            
        elif phase == "move":
            # Move object to goal with closed gripper
            target = obs_dict['desired_goal'][:3]
            # Add small upward bias to maintain lift
            target = target.copy()
            target[2] += 0.05
            direction = target - ee_pos
            action[:3] = np.clip(direction * self.position_gain, -1.0, 1.0)
            action[3] = -1.0  # Keep gripper closed
        
        # --- 2. REPULSION (The Barrier) ---
        # "Push" away from obstacle ONLY if the STL says 'G(avoid_zone)'
        if safety_rule == "avoid_zone":
            dist_to_obs = np.linalg.norm(ee_pos - self.obstacle_pos)
            
            # Barrier Function: Force = 1 / distance^2
            if dist_to_obs < (self.obstacle_radius + 0.05): # +5cm buffer
                # Direction away from obstacle
                push_dir = (ee_pos - self.obstacle_pos) / (dist_to_obs + 1e-6)
                
                # Magnitude explodes as we get closer
                strength = 0.05 / (dist_to_obs**2 + 1e-6)
                repulsion_vec = push_dir * min(strength, 20.0) # Clamp max force
                
                # Add repulsion to the action
                action[:3] += repulsion_vec
                # Re-clip to ensure valid action range
                action[:3] = np.clip(action[:3], -1.0, 1.0)

        return action