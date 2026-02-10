import numpy as np

class PrimitiveLibrary:
    def __init__(self):
        self.kp = 5.0  # Proportional gain
    
    def move_to(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        error = target_pos - current_pos
        action = np.clip(error * self.kp, -1.0, 1.0)
        return action  # [dx, dy, dz]

    def grasp(self) -> float:
        return -1.0  # Close gripper
    
    def release(self) -> float:
        return 1.0  # Open gripper