import gymnasium as gym
import panda_gym
import numpy as np
import re
import time

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
SWAP_BLOCKS = True 

# Observation slicing indices
IDX_BLUE = slice(7, 10) if SWAP_BLOCKS else slice(19, 22)
IDX_GREEN = slice(19, 22) if SWAP_BLOCKS else slice(7, 10)
GOAL_BLUE = slice(0, 3) if SWAP_BLOCKS else slice(3, 6)
GOAL_GREEN = slice(3, 6) if SWAP_BLOCKS else slice(0, 3)

# STL Formula for the stacking task
STL_FORMULA = """
F[0,10](approach_blue & F[0,4](grasp_blue & F[0,4](lift_blue & 
F[0,10](place_blue & F[0,4](release_blue & F[0,10](approach_green & 
F[0,4](grasp_green & F[0,4](lift_green & F[0,10](align_green_high & 
F[0,5](lower_green & F[0,4](drop_green)))))))))))
"""

# ==========================================
# 1. PARSER
# ==========================================
class STLNode:
    """Parses a nested STL string into a linked list of task phases."""
    def __init__(self, stl_string):
        self.phase_name = None
        self.safety_constraint = None 
        self.min_time = 0.0
        self.max_time = 0.0
        self.next_node = None
        self._parse(stl_string.strip())

    def _parse(self, s):
        # Matches F[min, max](predicate & rest)
        pattern = r"F\[([\d\.]+),([\d\.]+)\]\s*\(([^&]+)(?:&\s*(.*))?\)"
        match = re.match(pattern, s, re.DOTALL)
        if match:
            t_min, t_max, name, rest = match.groups()
            self.min_time, self.max_time = float(t_min), float(t_max)
            self.phase_name = name.strip().rstrip(")")
            
            if rest:
                # Extract safety G(constraint) if present
                safety_match = re.search(r"G\(([^)]+)\)", rest)
                if safety_match:
                    self.safety_constraint = safety_match.group(1).strip()
                    rest = rest.replace(safety_match.group(0), "").strip().lstrip("& ")
                
                if rest and "F[" in rest:
                    self.next_node = STLNode(rest)

# ==========================================
# 2. CONDUCTOR
# ==========================================
class STLConductor:
    """Manages the progression of the task based on predicates and time."""
    def __init__(self, stl_string, predicates):
        self.current_node = STLNode(stl_string)
        self.predicates = predicates
        self.phase_start_time = 0.0
        self.finished = False

    def update(self, obs, current_time):
        if self.finished: 
            return "DONE", None, 0.0

        elapsed = current_time - self.phase_start_time
        time_left = max(0.0, self.current_node.max_time - elapsed)

        # Check if current phase criteria met
        check_func = self.predicates.get(self.current_node.phase_name)
        if check_func and check_func(obs):
            print(f">>> COMPLETED: {self.current_node.phase_name} at {current_time:.2f}s")
            if self.current_node.next_node:
                self.current_node = self.current_node.next_node
                self.phase_start_time = current_time
            else:
                self.finished = True
                print(">>> MISSION SUCCESS")

        return self.current_node.phase_name, self.current_node.safety_constraint, time_left

# ==========================================
# 3. PREDICATES & CONTROLLER
# ==========================================
def get_predicates():
    return {
        "approach_blue":   lambda o: np.linalg.norm(o['observation'][0:3] - o['observation'][IDX_BLUE]) < 0.02,
        "grasp_blue":      lambda o: 0.02 < o['observation'][6] < 0.045,
        "lift_blue":       lambda o: o['observation'][IDX_BLUE][2] > 0.05,
        "place_blue":      lambda o: np.linalg.norm(o['observation'][IDX_BLUE] - o['desired_goal'][GOAL_BLUE]) < 0.03,
        "release_blue":    lambda o: o['observation'][6] > 0.035 and np.linalg.norm(o['observation'][0:3] - o['observation'][IDX_BLUE]) > 0.12,
        "approach_green":  lambda o: np.linalg.norm(o['observation'][0:3] - o['observation'][IDX_GREEN]) < 0.02,
        "grasp_green":     lambda o: 0.02 < o['observation'][6] < 0.045,
        "lift_green":      lambda o: o['observation'][IDX_GREEN][2] > 0.10,
        "align_green_high": lambda o: np.linalg.norm(o['observation'][IDX_GREEN][:2] - o['observation'][IDX_BLUE][:2]) < 0.03,
        "lower_green":     lambda o: (o['observation'][IDX_GREEN][2] - o['observation'][IDX_BLUE][2]) < 0.045,
        "drop_green":      lambda o: np.linalg.norm(o['observation'][IDX_GREEN] - o['desired_goal'][GOAL_GREEN]) < 0.04 and o['observation'][6] > 0.035
    }

class PDController:
    def __init__(self):
        self.prev_err = np.zeros(3)

    def get_action(self, phase, obs, t_left):
        if not phase or phase == "DONE": return np.zeros(4)

        ee_pos = obs['observation'][0:3]
        p_blue, p_green = obs['observation'][IDX_BLUE], obs['observation'][IDX_GREEN]
        g_blue, g_green = obs['desired_goal'][GOAL_BLUE], obs['desired_goal'][GOAL_GREEN]

        # Default values
        target, gripper, kp, kd = ee_pos, -1.0, 5.0, 3.0

        # Mapping phases to targets
        targets = {
            "approach_blue": (p_blue, 1.0),
            "grasp_blue":    (p_blue, -1.0),
            "lift_blue":     (p_blue + [0, 0, 0.15], -1.0),
            "place_blue":    (g_blue, -1.0),
            "release_blue":  (p_blue + [0, 0, 0.20], 1.0),
            "approach_green": (p_green, 1.0),
            "grasp_green":   (p_green, -1.0),
            "lift_green":    (p_green + [0, 0, 0.15], -1.0),
            "align_green_high": (p_blue + [0, 0, 0.15], -1.0),
            "lower_green":   (p_blue + [0, 0, 0.02], -1.0),
            "drop_green":    (g_green + [0, 0, 0.15], 1.0)
        }

        if phase in targets:
            target, gripper = targets[phase]
            if phase == "lower_green": kp, kd = 12.0, 5.0 # Stiffer for stacking

        # Compute PD
        err = target - ee_pos
        vel = (err - self.prev_err)
        self.prev_err = err
        
        urgency = np.clip(1.0 + (2.0 / (t_left + 0.1)), 1.0, 4.0)
        force = (err * kp * urgency) + (vel * kd)
        force[2] += 0.08  # Anti-gravity bias

        return np.append(np.clip(force, -1, 1), gripper)

# ==========================================
# 4. EXECUTION
# ==========================================
def run():
    env = gym.make('PandaStack-v3', render_mode='human', max_episode_steps=300)
    obs, _ = env.reset()
    
    predicates = get_predicates()
    conductor = STLConductor(STL_FORMULA, predicates)
    controller = PDController()
    
    sim_time, dt = 0.0, 0.04

    try:
        while True:
            phase, _, t_left = conductor.update(obs, sim_time)
            
            if conductor.finished:
                print("Stacking successful! Resetting in 2s...")
                time.sleep(2)
                obs, _ = env.reset()
                conductor = STLConductor(STL_FORMULA, predicates)
                sim_time = 0.0
                continue

            action = controller.get_action(phase, obs, t_left)
            obs, _, term, trunc, _ = env.step(action)
            sim_time += dt

            if term or trunc:
                obs, _ = env.reset()
                conductor = STLConductor(STL_FORMULA, predicates)
                sim_time = 0.0
            
            time.sleep(0.02)
    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    run()