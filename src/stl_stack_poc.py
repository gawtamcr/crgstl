import gymnasium as gym
import panda_gym
import numpy as np
import re
import time

# ==========================================
# CONFIGURATION
# ==========================================
SWAP_BLOCKS = True 

if SWAP_BLOCKS:
    IDX_BLUE = slice(7, 10)
    IDX_GREEN = slice(19, 22)
    GOAL_GREEN = slice(3, 6)
    GOAL_BLUE = slice(0, 3)
else:
    IDX_GREEN = slice(7, 10)
    IDX_BLUE = slice(19, 22)
    GOAL_GREEN = slice(0, 3)
    GOAL_BLUE = slice(3, 6)

# ==========================================
# 1. PARSER
# ==========================================
class RecursiveSTLNode:
    def __init__(self, stl_string):
        self.phase_name = None
        self.safety_constraint = None 
        self.min_time = 0.0
        self.max_time = 0.0
        self.next_node = None
        self._parse(stl_string.strip())

    def _parse(self, s):
        pattern = r"F\[([\d\.]+),([\d\.]+)\]\s*\(([^&]+)(?:&\s*(.*))?\)"
        match_f = re.match(pattern, s, re.DOTALL)
        if match_f:
            t_min, t_max, name, rest = match_f.groups()
            self.min_time = float(t_min)
            self.max_time = float(t_max)
            self.phase_name = name.strip().rstrip(")")
            if rest:
                match_g = re.search(r"G\(([^)]+)\)", rest)
                if match_g:
                    self.safety_constraint = match_g.group(1).strip()
                    rest = rest.replace(match_g.group(0), "").strip()
                    if rest.startswith("&"): rest = rest[1:].strip()
                if rest:
                    self.next_node = RecursiveSTLNode(rest)

# ==========================================
# 2. CONDUCTOR (Detailed Debugging)
# ==========================================
class STLConductor:
    def __init__(self, stl_string, predicates):
        print(f"Parsing STL: {stl_string}")
        self.root_node = RecursiveSTLNode(stl_string)
        self.current_node = self.root_node
        self.predicates = predicates
        self.phase_start_time = 0.0
        self.finished = False
        self.last_print = 0.0

    def update(self, obs, current_sim_time):
        if self.finished: return "DONE", None, 999.0

        dt = current_sim_time - self.phase_start_time
        time_left = max(0.0, self.current_node.max_time - dt)

        if time_left <= 0:
            print(f"!!! TIMEOUT: Phase '{self.current_node.phase_name}' expired.")

        # --- DEBUG OUTPUT ---
        if current_sim_time - self.last_print > 0.5:
            self.print_debug(self.current_node.phase_name, obs)
            self.last_print = current_sim_time

        check_func = self.predicates.get(self.current_node.phase_name)
        if check_func and check_func(obs):
            print(f">>> EVENT: '{self.current_node.phase_name}' Satisfied at t={current_sim_time:.2f}")
            if self.current_node.next_node:
                self.current_node = self.current_node.next_node
                self.phase_start_time = current_sim_time # Reset Clock
            else:
                self.finished = True
                print(">>> MISSION COMPLETE.")

        return self.current_node.phase_name, self.current_node.safety_constraint, time_left

    def print_debug(self, phase, o):
        green = o['observation'][IDX_GREEN]
        blue = o['observation'][IDX_BLUE]
        
        if phase == "align_green_high":
            dist_xy = np.linalg.norm(green[:2] - blue[:2])
            green_z = green[2]
            print(f"   [DEBUG] Phase: ALIGN | XY Error: {dist_xy*100:.1f}cm (Goal < 4.0cm) | Green Z: {green_z:.3f}")
            
        elif phase == "lower_green":
            dist_z = green[2] - blue[2]
            dist_xy = np.linalg.norm(green[:2] - blue[:2])
            print(f"   [DEBUG] Phase: LOWER | Z Gap: {dist_z*100:.1f}cm (Goal < 5.0cm) | XY Slip: {dist_xy*100:.1f}cm")

# ==========================================
# 3. PREDICATES (Stricter Alignment during Lowering)
# ==========================================
def define_stacking_predicates():
    return {
        # --- STAGE 1 (Unchanged) ---
        "approach_blue": lambda o: np.linalg.norm(o['observation'][0:3] - o['observation'][IDX_BLUE]) < 0.02,
        "grasp_blue": lambda o: o['observation'][6] > 0.02 and o['observation'][6] < 0.045,
        "lift_blue": lambda o: o['observation'][IDX_BLUE][2] > 0.05,
        "place_blue": lambda o: np.linalg.norm(o['observation'][IDX_BLUE] - o['desired_goal'][GOAL_BLUE]) < 0.03,
        "release_blue": lambda o: o['observation'][6] > 0.035 and np.linalg.norm(o['observation'][0:3] - o['observation'][IDX_BLUE]) > 0.12,

        # --- STAGE 2 (Refined) ---
        "approach_green": lambda o: np.linalg.norm(o['observation'][0:3] - o['observation'][IDX_GREEN]) < 0.02,
        "grasp_green": lambda o: o['observation'][6] > 0.02 and o['observation'][6] < 0.045,
        "lift_green": lambda o: o['observation'][IDX_GREEN][2] > 0.10,
        
        # ALIGN HIGH
        "align_green_high": lambda o: np.linalg.norm(o['observation'][IDX_GREEN][:2] - o['observation'][IDX_BLUE][:2]) < 0.03, 
        
        # LOWER (Wait for Z Gap < 4.5cm)
        "lower_green": lambda o: (o['observation'][IDX_GREEN][2] - o['observation'][IDX_BLUE][2]) < 0.045,

        # DROP (Check Final precision)
        "drop_green": lambda o: np.linalg.norm(o['observation'][IDX_GREEN] - o['desired_goal'][GOAL_GREEN]) < 0.04 and o['observation'][6] > 0.035
    }

# ==========================================
# 4. CONTROLLER (The "Heavy" Descent)
# ==========================================
class PDController:
    def __init__(self):
        self.last_error = np.zeros(3)

    def get_action(self, phase, safety, obs, time_remaining):
        # SAFETY CATCH: If phase is None, just stay still
        if phase is None or phase == "DONE":
            return np.zeros(4)

        ee_pos = obs['observation'][0:3]
        pos_blue = obs['observation'][IDX_BLUE]
        pos_green = obs['observation'][IDX_GREEN]
        target_blue = obs['desired_goal'][GOAL_BLUE]
        target_green = obs['desired_goal'][GOAL_GREEN]
        
        target_pos = ee_pos
        gripper = -1.0 
        kp = 5.0 
        kd = 3.0 
        
        # --- PHASE LOGIC ---
        if "approach" in phase: gripper = 1.0
        if "release" in phase or "drop" in phase: gripper = 1.0

        if phase == "approach_blue": target_pos = pos_blue
        elif phase == "grasp_blue": target_pos = pos_blue
        elif phase == "lift_blue": target_pos = pos_blue + [0, 0, 0.15]
        elif phase == "place_blue": target_pos = target_blue
        elif phase == "release_blue": target_pos = pos_blue + [0, 0, 0.20]
        
        elif phase == "approach_green": target_pos = pos_green
        elif phase == "grasp_green": target_pos = pos_green
        elif phase == "lift_green": target_pos = pos_green + [0, 0, 0.15]
            
        elif phase == "align_green_high":
            target_pos = pos_blue + [0, 0, 0.15]
            kp = 4.0 
            
        elif phase == "lower_green":
            # --- THE FIX: TARGET BELOW THE SURFACE ---
            # By targeting 1cm ABOVE blue but with high gain, we force the descent.
            target_pos = pos_blue + [0, 0, 0.02]
            kp = 12.0 # Forceful descent
            kd = 5.0  # High damping to prevent a "slam"
            
        elif phase == "drop_green":
            target_pos = target_green + [0, 0, 0.15]

        # PD CALC
        error = target_pos - ee_pos
        derivative = error - self.last_error
        self.last_error = error
        
        urgency = min(1.0 + (3.0 / max(time_remaining, 0.5)), 5.0) 
        force = (error * kp * urgency) + (derivative * kd)
        
        # Extra Gravity Compensation for Z-axis
        force[2] += 0.05 

        action = np.zeros(4)
        action[:3] = force
        action[3] = gripper
        return action

# ==========================================
# 5. MAIN
# ==========================================
def run_dual_stacking():
    env = gym.make('PandaStack-v3', render_mode='human', max_episode_steps=200)
    obs, info = env.reset()
    env.unwrapped.task.distance_threshold = 0.03
    # Final STL String for "Rock Solid" Stacking
    stl = """
    F[0,10.0](approach_blue & 
        F[0,4.0](grasp_blue & 
            F[0,4.0](lift_blue & 
                F[0,10.0](place_blue & 
                    F[0,4.0](release_blue & 
                        F[0,10.0](approach_green & 
                            F[0,4.0](grasp_green & 
                                F[0,4.0](lift_green & 
                                    F[0,10.0](align_green_high & 
                                        F[0,5.0](lower_green & 
                                            F[0,4.0](drop_green & 
                                                G[0,1.0](stabilize)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    """
    conductor = STLConductor(stl, define_stacking_predicates())
    musician = PDController()
    
    sim_time = 0.0
    dt = 0.04

    while True:
        # 1. Update Logic
        phase, safety, t_left = conductor.update(obs, sim_time)
        
        # 2. CHECK FOR COMPLETION BEFORE CALLING MUSICIAN
        if conductor.finished or phase == "DONE" or phase is None:
            print(">>> MISSION COMPLETE. Admiring stack...")
            time.sleep(2.0) # Pause to admire the stack
            # Pause and then reset
            for _ in range(50):
                env.step(np.zeros(4))
                time.sleep(0.04)
            
            obs, _ = env.reset()
            conductor = STLConductor(stl, define_stacking_predicates())
            sim_time = 0.0
            continue # Skip the rest of the loop

        phase, safety, t_left = conductor.update(obs, sim_time)
        action = musician.get_action(phase, safety, obs, t_left)
        obs, _, term, trunc, info = env.step(action)
        sim_time += dt
        if info.get('is_success', True):
            print(">>> SUCCESS FLAG Triggered in Info! Resetting...")
        if term or trunc:
            print(f"!!! GYM RESET TRIGGERED !!!")
            obs, _ = env.reset()
            conductor = STLConductor(stl, define_stacking_predicates())
            musician = PDController()
            sim_time = 0.0
            print("--- ENV RESET (Failure) ---")
            
        time.sleep(0.01)

    env.close()

if __name__ == "__main__":
    run_dual_stacking()