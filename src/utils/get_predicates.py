import numpy as np

def get_predicates(obs, info):
    ee_pos = obs['observation'][:3]
    gripper_width = obs['observation'][18]
    
    if 'achieved_goal' in obs:
        obj1_pos = obs['achieved_goal'][:3]
    else:
        obj1_pos = info.get('object1_pos', np.zeros(3))
        
    target_pos = obs['desired_goal'][:3]
    
    # Raw Metrics for Robustness
    # We return distances. STL will define thresholds (e.g., 0.03 - dist)
    
    predicates = {
        "gripper_width": gripper_width,
        "dist_xy": np.linalg.norm(ee_pos[:2] - obj1_pos[:2]),
        "obj_z": obj1_pos[2],
        "dist_target": np.linalg.norm(obj1_pos - target_pos)
    }
    return predicates