import gymnasium as gym
import panda_gym
import numpy as np
import re
import time

# ==========================================
# 1. RECURSIVE STL PARSER
#    Supports: F[min, max](Phase & G(Safety) & F[...])
# ==========================================
class RecursiveSTLNode:
    def __init__(self, stl_string):
        self.phase_name = None
        self.safety_constraint = None  # Stores "avoid_zone" if present
        self.min_time = 0.0
        self.max_time = 0.0
        self.next_node = None
        self._parse(stl_string.strip())

    def _parse(self, s):
        # Regex to find the pattern: F[min,max](name & ... rest ...)
        match_f = re.match(r"F\[([\d\.]+),([\d\.]+)\]\s*\(([^&]+)(?:&\s*(.*))?\)", s)
        
        if match_f:
            t_min, t_max, name, rest = match_f.groups()
            self.min_time = float(t_min)
            self.max_time = float(t_max)
            self.phase_name = name.strip()

            if rest:
                # Check for 'G' (Globally/Safety) inside the rest string
                # Regex looks for: G(constraint_name)
                match_g = re.search(r"G\(([^)]+)\)", rest)
                if match_g:
                    self.safety_constraint = match_g.group(1).strip()
                    # Remove the G(...) part to process the rest of the string
                    rest = rest.replace(match_g.group(0), "").strip()
                    # Clean up leading '&' if it exists after removal
                    if rest.startswith("&"): 
                        rest = rest[1:].strip()

                # If there is still string left, it must be the nested F[...]
                if rest:
                    self.next_node = RecursiveSTLNode(rest)

# ==========================================
# 2. THE CONDUCTOR (Logic & Timing)
# ==========================================
class STLConductor:
    def __init__(self, stl_string, predicates):
        print(f"Parsing STL: {stl_string}")
        self.root_node = RecursiveSTLNode(stl_string)
        self.current_node = self.root_node
        self.predicates = predicates
        self.phase_start_time = 0.0
        self.finished = False

    def update(self, obs, current_sim_time):
        if self.finished:
            return "DONE", None, 999.0

        # Calculate Relative Time (Clock resets for each phase)
        dt = current_sim_time - self.phase_start_time
        time_left = max(0.0, self.current_node.max_time - dt)

        # Check for Deadline Violation
        if time_left <= 0:
            print(f"!!! TIMEOUT: Phase '{self.current_node.phase_name}' failed to complete in {self.current_node.max_time}s")

        # Check for Success Event (Logic Transition)
        check_func = self.predicates.get(self.current_node.phase_name)
        
        if check_func and check_func(obs):
            print(f">>> EVENT: '{self.current_node.phase_name}' Satisfied at t={current_sim_time:.2f} (Duration: {dt:.2f}s)")
            
            if self.current_node.next_node:
                self.current_node = self.current_node.next_node
                self.phase_start_time = current_sim_time # Reset Clock
            else:
                self.finished = True
                print(">>> MISSION COMPLETE: All STL constraints satisfied.")

        # Return: Current Phase, Active Safety Rule, and Time Limit
        return self.current_node.phase_name, self.current_node.safety_constraint, time_left

# ==========================================
# 3. THE MUSICIAN (Safety-Aware Controller)
# ==========================================
class SafeFunnelController:
    def __init__(self):
        # Define a virtual obstacle (Sphere) in the middle of the workspace
        # Robot starts around [-0.1, 0, 0.5] and moves to [0.1, 0, 0.1]
        # We put the obstacle at [0.0, 0.0, 0.2] to force it to curve.
        self.obstacle_pos = np.array([0.0, 0.0, 0.2])
        self.obstacle_radius = 0.10  # 10cm radius

    def get_action(self, phase, safety_rule, obs, time_remaining):
        ee_pos = obs['observation'][:3]
        
        # Determine target based on phase
        if phase == "approach":
            target_pos = obs['achieved_goal'][:3] # Object
            gripper_act = 1.0 # Open
        elif phase == "grasp":
            target_pos = obs['achieved_goal'][:3] # Hold position
            gripper_act = -1.0 # Close
        elif phase == "move":
            target_pos = obs['desired_goal'][:3] # Final Target
            gripper_act = -1.0 # Keep Closed
        else:
            return np.zeros(4)

        # --- 1. ATTRACTION (The Funnel / CLF) ---
        # "Pull" towards the target.
        # Urgency increases as time runs out (1.0 -> 50.0)
        urgency = 1.0 + (5.0 / max(time_remaining, 0.1))
        attraction_vec = (target_pos - ee_pos) * 4.0 * urgency

        # --- 2. REPULSION (The Barrier / CBF) ---
        # "Push" away from obstacle ONLY if the STL says 'G(avoid_zone)'
        repulsion_vec = np.zeros(3)
        
        if safety_rule == "avoid_zone":
            dist_to_obs = np.linalg.norm(ee_pos - self.obstacle_pos)
            
            # Barrier Function: Force = 1 / distance^2
            if dist_to_obs < (self.obstacle_radius + 0.05): # +5cm buffer
                # Direction away from obstacle
                push_dir = (ee_pos - self.obstacle_pos) / (dist_to_obs + 1e-6)
                
                # Magnitude explodes as we get closer
                strength = 0.05 / (dist_to_obs**2 + 1e-6)
                repulsion_vec = push_dir * min(strength, 20.0) # Clamp max force

                # Visual Debug log (only print if force is significant)
                if np.linalg.norm(repulsion_vec) > 2.0:
                    print(f"   [BARRIER ACTIVE] Pushing away! Force: {np.linalg.norm(repulsion_vec):.2f}")

        # --- 3. COMBINE FORCES ---
        total_force = attraction_vec + repulsion_vec
        
        action = np.zeros(4)
        action[:3] = total_force
        action[3] = gripper_act
        
        return action

# ==========================================
# 4. PREDICATE DEFINITIONS
# ==========================================
def define_predicates():
    """
    Returns dictionary of conditions for the Conductor to check.
    """
    return {
        # Condition: End Effector is close to Object
        "approach": lambda o: np.linalg.norm(o['observation'][:3] - o['achieved_goal'][:3]) < 0.001,
        
        # Condition: Simulator "Grasped" check (simplified)
        # We assume grasp is done if we are close and Gripper width is small (not fully simulated here)
        "grasp": lambda o: True, 
        
        # Condition: Object is close to Target
        "move": lambda o: np.linalg.norm(o['achieved_goal'][:3] - o['desired_goal'][:3]) < 0.05
    }

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def run_simulation():
    env = gym.make('PandaPickAndPlace-v3', render_mode='human')
    env.unwrapped.task.distance_threshold = 0.05
    obs, info = env.reset()
    
    # --- STL INPUT ---
    # "Eventually (0-10s) approach the object AND Always avoid the zone, 
    #  THEN Eventually (0-2s) grasp it, 
    #  THEN Eventually (0-5s) move it."
    user_stl = "F[0,10.0](approach & G(avoid_zone) & F[0,2.0](grasp & F[0,5.0](move)))"

    conductor = STLConductor(user_stl, define_predicates())
    musician = SafeFunnelController()
    
    sim_time = 0.0
    dt = 0.04 # Standard PyBullet timestep

    print("\n--- STARTING SIMULATION ---")
    
    # Run loop
    for _ in range(2000):
        if conductor.finished:
            print("--- MISSION COMPLETE ---")
            time.sleep(5)
            break
            
        # 1. Update Logic
        phase, safety, t_left = conductor.update(obs, sim_time)
        
        # 2. Update Control
        action = musician.get_action(phase, safety, obs, t_left)
        
        # 3. Physics Step
        obs, reward, terminated, truncated, info = env.step(action)
        sim_time += dt

        # Reset handling
        if terminated or truncated:
            obs, info = env.reset()
            conductor = STLConductor(user_stl, define_predicates())
            sim_time = 0.0
            print("--- RESET ---")

        time.sleep(0.1) # Visualization speed

    env.close()

if __name__ == "__main__":
    run_simulation()