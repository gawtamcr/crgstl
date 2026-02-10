import gymnasium as gym
import panda_gym
import numpy as np
from primitives import PrimitiveLibrary
from stl import Sequence, Eventually, Predicate
from get_predicates import get_predicates

# 1. Setup Environment
env = gym.make("PandaStack-v3", render_mode="human")
observation, info = env.reset()
primitives = PrimitiveLibrary()

# 2. Define The STL Task (The "Score")
# Task: Align -> Pick -> Lift
task_logic = Sequence([
    Eventually(Predicate("Aligned", lambda s: s["aligned_with_obj"])),
    Eventually(Predicate("Holding", lambda s: s["holding_obj"])),
    Eventually(Predicate("Lifted",  lambda s: s["obj_lifted"]))
])

# 3. Main Loop
for _ in range(1000):
    # A. Abstraction: Get Logical State
    # Note: In real panda-gym, we need to extract object position from 'observation' or 'info'
    # For this snippet, we mock the extraction for clarity
    ee_position = observation['observation'][0:3]
    object_position = observation['achieved_goal'][0:3] # In Stack env, this is usually the object
    current_predicates = get_predicates(observation, info)

    # B. The Conductor: Update Logic & Decide Phase
    task_logic.check(current_predicates)
    current_phase = task_logic.get_active_task()

    # C. The Dispatcher: Select Action based on Phase
    action_xyz = [0, 0, 0]
    gripper_action = 1.0 # Default Open

    if isinstance(current_phase, str) and current_phase == "DONE":
        print("Task Complete!")
        break
    
    # Logic Switching
    phase_name = current_phase.child.name if hasattr(current_phase, 'child') else "Unknown"
    
    if phase_name == "Aligned":
        # Strategy: Move to [obj_x, obj_y, hover_z]
        target = object_position.copy()
        target[2] += 0.1 # Hover above
        action_xyz = primitives.move_to(ee_position, target)
        
    elif phase_name == "Holding":
        # Strategy: Move down and close
        target = object_position.copy()
        action_xyz = primitives.move_to(ee_position, target)
        # If close enough, trigger grasp
        if np.linalg.norm(ee_position - target) < 0.02:
            gripper_action = primitives.grasp()
            
    elif phase_name == "Lifted":
        # Strategy: Move Up
        target = object_position.copy()
        target[2] = 0.3
        action_xyz = primitives.move_to(ee_position, target)
        gripper_action = primitives.grasp() # Keep holding

    # D. Execute
    full_action = np.append(action_xyz, gripper_action)
    observation, reward, terminated, truncated, info = env.step(full_action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()