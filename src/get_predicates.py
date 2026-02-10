import numpy as np

def get_predicates(obs, info):
    # Extract positions from the observation
    # Note: panda-gym obs structure depends on the env, usually:
    # 'achieved_goal', 'desired_goal', 'observation' (7 joints + 6 velocities)

    ee_pos = obs['observation'][:3]
    gripper_width = obs['observation'][6] # Approximate index, varies by env version
    
    # In PandaStack, 'achieved_goal' usually contains the object position
    if 'achieved_goal' in obs:
        obj1_pos = obs['achieved_goal'][:3]
    else:
        obj1_pos = info.get('object1_pos', [0,0,0]) 
    
    # Logic & Thresholds (Tuned for PandaStack-v3)
    # Holding: Gripper closed AND object is off the table slightly
    is_holding = (gripper_width < 0.04) and (obj1_pos[2] > 0.02)
    
    predicates = {
        "holding_obj": is_holding,
        "aligned_with_obj": np.linalg.norm(ee_pos[:2] - obj1_pos[:2]) < 0.02,
        "obj_lifted": obj1_pos[2] > 0.15
    }
    return predicates