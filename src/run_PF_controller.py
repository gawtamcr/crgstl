import gymnasium as gym
import panda_gym
import time
import numpy as np

from common.predicates import define_predicates
from common.stl_planner import STLPlanner
from controller.safe_funnel_controller import SafeFunnelController

def run_simulation():
    env = gym.make('PandaPickAndPlace-v3', render_mode='human')
    env.unwrapped.task.distance_threshold = 0.01
    
    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"
    
    planner = STLPlanner(user_stl, define_predicates())
    controller = SafeFunnelController()

    dt = 0.04 # Standard PyBullet timestep
    num_episodes = 20
    success_count = 0

    print(f"\n--- STARTING EVALUATION ({num_episodes} Episodes) ---")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        planner.reset()
        sim_time = 0.0
        done = False
        
        while not done:
            phase, safety, t_left = planner.update(obs, sim_time)     # 1. Update Logic
            
            if planner.finished:
                success_count += 1
                print(f"Episode {ep+1}: SUCCESS")
                break

            action = controller.get_action(phase, safety, obs, t_left)  # 2. Update Control
            obs, reward, terminated, truncated, info = env.step(action) # 3. Physics Step
            sim_time += dt

            if terminated or truncated or planner.failed_timeout:
                print(f"Episode {ep+1}: FAILED")
                break

            time.sleep(0.02) # Visualization speed

    print(f"\n--- EVALUATION COMPLETE ---")
    print(f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    env.close()

if __name__ == "__main__":
    run_simulation()
