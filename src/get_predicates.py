import numpy as np

def get_predicates(obs, info):
    ee_pos = obs['observation'][:3]
    gripper_width = obs['observation'][18]
    
    if 'achieved_goal' in obs:
        obj1_pos = obs['achieved_goal'][:3]
    else:
        obj1_pos = info.get('object1_pos', np.zeros(3))
        
    target_pos = obs['desired_goal'][:3]
    
    # Logic & Thresholds
    is_holding = (gripper_width < 0.045) and (gripper_width > 0.01)
    
    predicates = {
        "holding_obj": is_holding,
        "aligned_with_obj": np.linalg.norm(ee_pos[:2] - obj1_pos[:2]) < 0.03,
        "obj_lifted": obj1_pos[2] > 0.15,
        "at_target": np.linalg.norm(obj1_pos - target_pos) < 0.05
    }
    return predicates