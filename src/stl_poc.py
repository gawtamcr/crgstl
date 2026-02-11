import gymnasium as gym
import panda_gym
import numpy as np
import time

# ==========================================
# 1. THE CONDUCTOR (Logic & Sequence)
# ==========================================
class STLConductor:
    """
    The High-Level Logic Layer.
    Responsible for:
    1. Sequencing (LTL): Approach -> Grasp -> Move -> Drop
    2. Timing (STL): Assigning deadlines (e.g., "Must finish Approach by t=5.0")
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "APPROACH"
        self.phase_start_time = 0.0
        # STL Constraints: We set specific deadlines for each phase
        self.deadlines = {
            "APPROACH": 4.0,  # Reach object within 4s
            "GRASP": 2.0,     # Close gripper within 2s
            "MOVE": 4.0       # Move to target within 4s
        }

    def update(self, obs, current_sim_time):
        """
        Checks conditions (Events) to trigger transitions.
        Returns: current_phase, time_remaining_in_phase
        """
        # Extract positions
        ee_pos = obs['observation'][0:3]
        obj_pos = obs['achieved_goal'][0:3]
        target_pos = obs['desired_goal'][0:3]
        
        # Calculate distances
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_target = np.linalg.norm(obj_pos - target_pos)

        # Calculate Logic Clock
        dt = current_sim_time - self.phase_start_time
        deadline = self.deadlines.get(self.phase, 10.0)
        time_remaining = max(0.0, deadline - dt)

        # --- STATE MACHINE TRANSITIONS ---
        if self.phase == "APPROACH":
            # Event: If close to object (tolerance 5cm)
            if dist_ee_obj < 0.005:
                print(f">>> EVENT: Object Reached at t={current_sim_time:.2f}. Phase -> GRASP")
                self.phase = "GRASP"
                self.phase_start_time = current_sim_time

        elif self.phase == "GRASP":
            # Event: Hard-coded wait for gripper to close (simulating grasp check)
            if dt > 1.0: 
                print(f">>> EVENT: Object Grasped at t={current_sim_time:.2f}. Phase -> MOVE")
                self.phase = "MOVE"
                self.phase_start_time = current_sim_time

        elif self.phase == "MOVE":
            # Event: If object is at target
            if dist_obj_target < 0.05:
                print(f">>> EVENT: Target Reached at t={current_sim_time:.2f}. MISSION COMPLETE.")
                self.phase = "DONE"

        return self.phase, time_remaining

# ==========================================
# 2. THE MUSICIAN (Control & Physics)
# ==========================================
class FunnelController:
    """
    The Low-Level Control Layer.
    Responsible for:
    1. Executing movement (P-Control)
    2. Enforcing STL Convergence (Funnel Gain)
    """
    def get_action(self, phase, obs, time_remaining):
        ee_pos = obs['observation'][0:3]
        obj_pos = obs['achieved_goal'][0:3]
        target_pos = obs['desired_goal'][0:3]
        
        # Base Action: [dx, dy, dz, gripper_ctrl]
        action = np.zeros(4)

        # --- TEMPORAL CONVERGENCE (The Funnel) ---
        # As time_remaining -> 0, gain -> Infinity.
        # This forces the robot to hurry up if the deadline is close.
        # We clamp the gain to 50.0 to prevent physics explosions.
        urgency_gain = 1.0 + (5.0 / max(time_remaining, 0.2)) 
        kp = 5.0 * urgency_gain 

        if phase == "APPROACH":
            # Move EE towards Object
            error = obj_pos - ee_pos
            action[0:3] = error * kp
            action[3] = 1.0 # Open Gripper

        elif phase == "GRASP":
            # Hold position, Close Gripper
            action[0:3] = 0.0 
            action[3] = -1.0 # Close Gripper

        elif phase == "MOVE":
            # Move Object towards Target
            error = target_pos - ee_pos
            action[0:3] = error * kp
            action[3] = -1.0 # Keep Closed

        elif phase == "DONE":
            action[:] = 0.0
        
        return action

# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================
def run_simulation():
    # Setup: 'human' render mode lets you see the robot
    env = gym.make('PandaPickAndPlace-v3', render_mode='human')
    
    observation, info = env.reset()
    conductor = STLConductor()
    musician = FunnelController()
    
    # Simulation parameters
    sim_time = 0.0
    dt = 0.04 # panda-gym standard timestep (approx 25Hz)

    print("--- Starting STL-Guided Simulation ---")

    for _ in range(1000):
        # 1. CONDUCTOR: Decide WHAT to do (and check deadlines)
        current_phase, time_left = conductor.update(observation, sim_time)
        
        # 2. MUSICIAN: Decide HOW to do it (with temporal urgency)
        action = musician.get_action(current_phase, observation, time_left)
        
        # 3. ENVIRONMENT: Step physics
        observation, reward, terminated, truncated, info = env.step(action)
        sim_time += dt

        # Reset if environment forces it
        if terminated or truncated:
            observation, info = env.reset()
            conductor.reset()
            sim_time = 0.0
            print("--- Environment Reset ---")

        # Optional: Slow down visualization
        time.sleep(0.02)

    env.close()

if __name__ == "__main__":
    run_simulation()