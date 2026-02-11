import gymnasium as gym
import panda_gym
import numpy as np

def scan_observation_structure():
    print("--- SCANNING PandaStack-v3 ---")
    env = gym.make('PandaStack-v3', render_mode='human')
    obs, info = env.reset()
    
    # 1. Analyze Dictionary Structure
    print(f"\n[Keys]: {list(obs.keys())}")
    
    goal_shape = obs['desired_goal'].shape
    ach_shape = obs['achieved_goal'].shape
    obs_shape = obs['observation'].shape
    
    print(f"[Shapes] Goal: {goal_shape}, Achieved: {ach_shape}, Obs: {obs_shape}")
    
    # 2. Analyze Vector Content (The Fingerprint Test)
    # We will look at the values to guess what they are.
    # Standard Panda-Gym Layout usually:
    # [0:3] EE Pos, [3:6] EE Vel
    # [6:9] Object 1 Pos (Rot, Vel...)
    # [?]   Object 2 Pos (Rot, Vel...)
    
    ee_pos = obs['observation'][0:3]
    print(f"\n[Hypothesis] EE Position: {ee_pos}")
    
    # 3. Dynamic Test: Move Object 1 and see what changes
    print("\n[Test] Moving for 20 steps to identify objects...")
    
    # Store initial state
    init_obs_vec = obs['observation'].copy()
    
    # Take random actions to disturb the state
    for _ in range(20):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        
    diff = np.abs(obs['observation'] - init_obs_vec)
    
    # Indices with high variance are likely velocities or moving objects
    # Indices that stay exactly 0.0 or constant might be the second object if it hasn't moved.
    
    print("\n[Variance Analysis]")
    print(f"Indices 0-6 (Robot): {np.mean(diff[0:6]):.4f} change")
    
    # Check distinct blocks of 6 or 12 (Panda-gym objects are usually 12-dim: pos+rot+lin_vel+rot_vel)
    # Block A is usually indices [6:18] or [7:19] depending on gripper width inclusion
    
    # If gripper width is included, it's usually index 6. 
    # Let's check index 6:
    print(f"Index 6 (Gripper Width?): {obs['observation'][6]:.4f}")
    
    # Block 1 (The one we manipulate first?)
    block1_diff = np.mean(diff[7:10]) # Pos x,y,z
    print(f"Indices 7-10 (Block 1 Pos?): {block1_diff:.4f} change")
    
    # Block 2 (The one we stack ON?)
    # Usually length is 6(robot) + 1(width) + 12(obj1) + 12(obj2) = 31? 
    # Or 6 + 12 + 12 = 30?
    
    if obs_shape[0] > 19:
        block2_idx = 7 + 12 # Jumping over Block 1
        block2_diff = np.mean(diff[block2_idx : block2_idx+3])
        print(f"Indices {block2_idx}-{block2_idx+3} (Block 2 Pos?): {block2_diff:.4f} change")

    env.close()

if __name__ == "__main__":
    scan_observation_structure()