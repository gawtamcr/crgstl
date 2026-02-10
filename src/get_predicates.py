import numpy as np

def get_predicates(obs, info):
    # Extract positions from the observation
    # Note: panda-gym obs structure depends on the env, usually:
    # 'achieved_goal', 'desired_goal', 'observation' (7 joints + 6 velocities)

    ee_pos = obs['observation'][:3]
    # In standard PandaPickAndPlace-v3, fingers_width is at indices 18-19 (sum or individual)
    # We'll assume index 18 based on standard panda-gym structure for 'fingers_width'
    gripper_width = obs['observation'][18] 
    
    # In PandaStack, 'achieved_goal' usually contains the object position
    if 'achieved_goal' in obs:
        obj1_pos = obs['achieved_goal'][:3]
    else:
        obj1_pos = info.get('object1_pos', [0,0,0]) 
        
    target_pos = obs['desired_goal'][:3]
    
    # Logic & Thresholds (Tuned for PandaStack-v3)
    # Holding: Gripper is "holding something" (not fully open, not fully closed)
    # We remove the Z-check here because we haven't lifted it yet.
    is_holding = (gripper_width < 0.045) and (gripper_width > 0.01)
    
    predicates = {
        "holding_obj": is_holding,
        "aligned_with_obj": np.linalg.norm(ee_pos[:2] - obj1_pos[:2]) < 0.03,
        "obj_lifted": obj1_pos[2] > 0.15,
        "at_target": np.linalg.norm(obj1_pos - target_pos) < 0.05
    }
    return predicates