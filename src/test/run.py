import time
import gymnasium as gym
import panda_gym
import numpy as np
from primitives import PrimitiveLibrary
from stl import Sequence, Eventually, Predicate
from get_predicates import get_predicates
from constants import *

# 1. Setup Environment
env = gym.make("PandaPickAndPlace-v3", render_mode="human")
observation, info = env.reset()
primitives = PrimitiveLibrary()

# 2. Define The STL Task (The "Score")
# Robustness: value > 0 means satisfied
# e.g., "Aligned" is satisfied if dist_xy < 0.03 -> (0.03 - dist_xy > 0)
task_logic = Sequence([
    Eventually(Predicate("Aligned", lambda s: 0.03 - s["dist_xy"])),
    Eventually(Predicate("Holding", lambda s: 0.045 - s["gripper_width"])), # Simple check for closed
    Eventually(Predicate("Lifted",  lambda s: s["obj_z"] - 0.15)),          # Height > 0.15
    Eventually(Predicate("Placed",  lambda s: 0.05 - s["dist_target"]))
])

prev_phase = None

# 3. Main Loop
for _ in range(1000):
    ee_position = observation['observation'][0:3]
    object_position = observation['achieved_goal'][0:3]
    target_position = observation['desired_goal'][0:3]
    current_predicates = get_predicates(observation, info)

    # B. The Conductor: Update Logic & Decide Phase
    rho = task_logic.robustness(current_predicates) # Returns float
    current_phase = task_logic.get_active_task()

    # Logging transitions
    if current_phase != prev_phase:
        print(f"[Conductor] Transition: {prev_phase} -> {current_phase}")
        prev_phase = current_phase

    # C. The Dispatcher: Select Action based on Phase
    action_xyz = [0, 0, 0]
    gripper_action = 1.0  # Default Open

    if isinstance(current_phase, str) and current_phase == "DONE":
        print("Task Complete!")
        break
    
    # Logic Switching
    phase_name = current_phase.child.name if hasattr(current_phase, 'child') else "Unknown"
    
    if phase_name == "Aligned":
        target = object_position.copy()
        target[2] += ALIGNMENT  # Move slightly above the object to avoid collisions
        action_xyz = primitives.move_to(ee_position, target)
        
    elif phase_name == "Holding":
        target = object_position.copy()
        action_xyz = primitives.move_to(ee_position, target)
        if np.linalg.norm(ee_position - target) < GRASPING:
            gripper_action = primitives.grasp()
            
    elif phase_name == "Lifted":
        target = object_position.copy()
        target[2] = LIFTING
        action_xyz = primitives.move_to(ee_position, target)
        gripper_action = primitives.grasp()  
        
    elif phase_name == "Placed":
        target = target_position.copy()
        target[2] += PLACEMENT  # Slightly above target to avoid smashing
        action_xyz = primitives.move_to(ee_position, target)
        
        # # If we are close to the placement target, release!
        if np.linalg.norm(ee_position - target) < PLACEMENT:
            gripper_action = primitives.release()
        else:
            gripper_action = primitives.grasp()

    # D. Execute
    full_action = np.append(action_xyz, gripper_action)
    observation, reward, terminated, truncated, info = env.step(full_action)
    time.sleep(0.05)
    
    if terminated or truncated:
        observation, info = env.reset()
        task_logic.reset()

env.close()