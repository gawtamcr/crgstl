import numpy as np

class PrimitiveLibrary:
    def __init__(self):
        self.kp = 5.0 # Proportional gain
    
    def move_to(self, current_pos, target_pos):
        error = target_pos - current_pos
        action = np.clip(error * self.kp, -1.0, 1.0)
        return action # [dx, dy, dz]

    def grasp(self):
        return -1.0 # Close gripper
    
    def release(self):
        return 1.0 # Open gripper