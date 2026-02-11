import gymnasium as gym
import panda_gym
import numpy as np
import re
import time

# ==========================================
# 1. STL PARSER & DATA STRUCTURES
# ==========================================

class RecursiveSTLNode:
    """
    Parses a nested STL string like: "F[0,10](approach & F[0,6](grasp))"
    Structure:
      - phase_name: "approach"
      - deadline: 10.0
      - next_node: RecursiveSTLNode("grasp", deadline=6.0)
    """
    def __init__(self, stl_string):
        self.phase_name = None
        self.min_time = 0.0
        self.max_time = 0.0
        self.next_node = None
        self._parse(stl_string.strip())

    def _parse(self, s):
        # Regex to find: F[min,max](name & rest)
        # 1. Time window: F\[([\d\.]+),([\d\.]+)\]
        # 2. Content: \(([^&]+)(?:&(.*))?\)
        
        # Simple parser for the specific format "F[a,b](name & ...)"
        match = re.match(r"F\[([\d\.]+),([\d\.]+)\]\s*\(([^&]+)(?:&\s*(.*))?\)", s)
        
        if not match:
            # Base case: just a predicate name (e.g. strict end) or malformed
            # For this PoC, we assume the string is well-formed per your example.
            return

        t_min, t_max, name, rest = match.groups()
        
        self.min_time = float(t_min)
        self.max_time = float(t_max)
        self.phase_name = name.strip()
        
        if rest:
            # Recursively parse the nested part
            self.next_node = RecursiveSTLNode(rest)

# ==========================================
# 2. THE CONDUCTOR (Nested Logic)
# ==========================================
class STLConductor:
    def __init__(self, stl_string, predicates):
        self.root_node = RecursiveSTLNode(stl_string)
        self.current_node = self.root_node
        self.predicates = predicates # Dictionary of condition functions
        self.phase_start_time = 0.0
        self.finished = False

    def update(self, obs, current_sim_time):
        if self.finished:
            return "DONE", 999.0

        # 1. Calculate Time Logic
        # The clock effectively "resets" for each nested node because 
        # the node's deadline is relative to when the *previous* node finished.
        dt = current_sim_time - self.phase_start_time
        time_remaining = max(0.0, self.current_node.max_time - dt)

        # 2. Check Safety/Failure (Did we miss the deadline?)
        if time_remaining <= 0:
            print(f"!!! STL VIOLATION: Phase '{self.current_node.phase_name}' timed out!")
            # In a real robot, triggers emergency stop. Here, we just print.
        
        # 3. Check Success (Event Trigger)
        # Retrieve the check function for the current phase name
        check_func = self.predicates.get(self.current_node.phase_name)
        
        if check_func and check_func(obs):
            print(f">>> EVENT: '{self.current_node.phase_name}' Satisfied at t={current_sim_time:.2f} (Took {dt:.2f}s)")
            
            # Transition to the inner (nested) node
            if self.current_node.next_node:
                self.current_node = self.current_node.next_node
                self.phase_start_time = current_sim_time # RESET CLOCK (Relative Time)
            else:
                self.finished = True
                print(">>> MISSION COMPLETE: All nested STL formulas satisfied.")

        return self.current_node.phase_name, time_remaining

# ==========================================
# 3. PREDICATES (The "Senses")
# ==========================================
def define_predicates():
    """
    Returns a dictionary mapping STL names to lambda functions.
    obs['observation'] = [ee_x, ee_y, ee_z]
    obs['achieved_goal'] = [obj_x, obj_y, obj_z]
    obs['desired_goal'] = [tgt_x, tgt_y, tgt_z]
    """
    return {
        "approach": lambda obs: np.linalg.norm(obs['observation'][:3] - obs['achieved_goal'][:3]) < 0.001,
        
        # Simplified "Grasp" check: Gripper is at object position AND closed (simulated)
        # Note: In real robotics, we'd check joint angles/force sensors.
        "grasp": lambda obs: np.linalg.norm(obs['observation'][:3] - obs['achieved_goal'][:3]) < 0.05, 
        
        "move": lambda obs: np.linalg.norm(obs['achieved_goal'][:3] - obs['desired_goal'][:3]) < 0.05
    }

# ==========================================
# 4. THE MUSICIAN (Funnel Controller)
# ==========================================
class FunnelController:
    def get_action(self, phase_name, obs, time_remaining):
        ee_pos = obs['observation'][:3]
        obj_pos = obs['achieved_goal'][:3]
        target_pos = obs['desired_goal'][:3]
        
        action = np.zeros(4)
        
        # Adaptive Funnel Gain: Increases as deadline approaches
        # Safe clamp at 0.1s to prevent division by zero
        urgency = 1.0 + (3.0 / max(time_remaining, 0.1))
        kp = 4.0 * urgency

        if phase_name == "approach":
            # Go to object, Open gripper
            action[:3] = (obj_pos - ee_pos) * kp
            action[3] = 1.0 
            
        elif phase_name == "grasp":
            # Stop, Close gripper
            # Note: We keep pos servoing to ensure we don't drift while grasping
            action[:3] = (obj_pos - ee_pos) * kp 
            action[3] = -1.0 
            
        elif phase_name == "move":
            # Go to target, Keep gripper closed
            action[:3] = (target_pos - ee_pos) * kp
            action[3] = -1.0
            
        return action

# ==========================================
# 5. MAIN
# ==========================================
def run():
    env = gym.make('PandaPickAndPlace-v3', render_mode='human')
    obs, info = env.reset()
    
    # --- USER INPUT STL ---
    user_stl = "F[0,4.0](approach & F[0,2.0](grasp & F[0,4.0](move)))"
    print(f"Executing STL: {user_stl}")

    conductor = STLConductor(user_stl, define_predicates())
    musician = FunnelController()
    
    sim_time = 0.0
    dt = 0.04 # 25Hz

    for _ in range(1000):
        if conductor.finished:
            break
            
        # 1. Update Logic (Pass current sim time)
        phase, t_left = conductor.update(obs, sim_time)
        
        # 2. Update Control
        action = musician.get_action(phase, obs, t_left)
        
        # 3. Step
        obs, reward, done, trunc, info = env.step(action)
        sim_time += dt
        
        if done or trunc:
            obs, info = env.reset()
            conductor = STLConductor(user_stl, define_predicates()) # Reset Logic
            sim_time = 0.0
            print("--- Environment Reset ---")

        time.sleep(0.02) # Slow down for visualization

    env.close()

if __name__ == "__main__":
    run()