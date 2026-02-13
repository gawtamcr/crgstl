import gymnasium as gym
import panda_gym
import numpy as np
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.predicates import define_predicates
from behavior_cloning.stl_gym_wrapper import STLGymWrapper
from common.reward_registry import RewardRegistry

def verify_wrapper():
    print("--- Starting Verification of STLGymWrapper ---")

    # 1. Setup
    # Formula: F[0,10](approach & F[0,2](grasp))
    # Structure:
    #   1. Eventually (Root) [0, 10]
    #     2. And
    #       3. Predicate: approach
    #       4. Eventually [0, 2]
    #         5. Predicate: grasp
    user_stl = "F[0,10.0](approach & F[0,2.0](grasp))"
    
    registry = RewardRegistry()
    # Register dummy rewards for testing to verify they are summed correctly
    registry.register_objective("approach", lambda o: 1.0) 
    registry.register_objective("grasp", lambda o: 10.0)
    
    base_env = gym.make('PandaPickAndPlace-v3')
    env = STLGymWrapper(base_env, user_stl, define_predicates(), reward_registry=registry)
    
    # 2. Check Observation Space
    print(f"\n[1] Checking Observation Space...")
    if 'stl_state' in env.observation_space.spaces:
        shape = env.observation_space.spaces['stl_state'].shape
        print(f"SUCCESS: 'stl_state' found with shape {shape}")
        
        # Expected size: 5 nodes * 4 features = 20
        expected_size = 5 * 4
        if shape[0] == expected_size:
            print(f"SUCCESS: Vector size matches expected graph size ({expected_size}).")
        else:
            print(f"FAILURE: Vector size {shape[0]} != Expected {expected_size}")
    else:
        print("FAILURE: 'stl_state' NOT found in observation space.")
        return

    # 3. Check Reset & Initial State
    print("\n[2] Checking Reset...")
    obs, info = env.reset()
    stl_vec = obs['stl_state']
    
    # Decode Root Node (Indices 0-3)
    # [active, satisfied, violated, time_ratio]
    root_active = stl_vec[0]
    root_time = stl_vec[3]
    
    print(f"Initial STL Vector (First 4 elements): {stl_vec[:4]}")
    if root_active == 1.0:
        print("SUCCESS: Root node is active.")
    else:
        print("FAILURE: Root node is NOT active.")

    # 4. Check Step & Rewards
    print("\n[3] Checking Step & Rewards...")
    
    # We expect 'approach' to be active initially.
    # Reward should be 1.0 (approach) + 0.0 (grasp not active yet) = 1.0
    
    action = base_env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step Reward: {reward}")
    print(f"Active Objectives: {info.get('stl_objectives')}")
    
    # Verify Reward Logic
    # Note: The base env might return a reward (usually -1 or 0). 
    # We need to account for that or check if stl_reward was added.
    # In PandaPickAndPlace, standard reward is sparse (0 or -1) or dense (distance).
    # However, our registry adds +1.0.
    
    if "approach" in info.get('stl_objectives', []):
        print("SUCCESS: 'approach' is listed as an active objective.")
    else:
        print("FAILURE: 'approach' is NOT active.")

    # Check Time Ratio Update
    new_stl_vec = obs['stl_state']
    new_root_time = new_stl_vec[3]
    print(f"Root Time Ratio: {root_time} -> {new_root_time}")
    
    if new_root_time > root_time:
        print("SUCCESS: Time ratio increased.")
    else:
        print("WARNING: Time ratio did not increase (check dt or node activation).")

    print("\n[4] Visualizing Status...")
    env.conductor.print_status()

    env.close()
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_wrapper()
