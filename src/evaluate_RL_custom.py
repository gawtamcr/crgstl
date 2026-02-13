import gymnasium as gym
import panda_gym
import numpy as np
import os
import time
from stable_baselines3 import SAC

from common.predicates import define_predicates
from behavior_cloning.stl_gym_wrapper import STLGymWrapper
from custom_env.custom_panda_task import STLPickAndPlaceTask, STLPickAndPlaceEnv

def evaluate():
    # Configuration
    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"
    model_path = "../models/training/sac_RL_withBC_v3"    
    print(f"Loading model: {model_path}")
    
    # Setup Environment
    base_env = STLPickAndPlaceEnv(render_mode="human")
    env = STLGymWrapper(base_env, user_stl, define_predicates())
    # env.unwrapped.task.distance_threshold = 0.05
    
    model = SAC.load(model_path, env=env)
    
    num_episodes = 10
    success_count = 0

    print(f"\n--- STARTING EVALUATION ({num_episodes} Episodes) ---")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(0.05)  # Slow down for better visualization
            env.render()
            
        # In PandaPickAndPlace, 'is_success' is usually in info
        is_success = info.get("is_success", False)
        if is_success:
            success_count += 1
        
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Success = {is_success}")

    print(f"\n--- EVALUATION COMPLETE ---")
    print(f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    env.close()

if __name__ == "__main__":
    evaluate()