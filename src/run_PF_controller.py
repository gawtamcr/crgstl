import gymnasium as gym
import panda_gym
import time
import numpy as np

from common.predicates import define_predicates
from common.stl_planner import STLPlanner
from controller.safe_funnel_controller import SafeFunnelController

def run_simulation():
    env = gym.make('PandaPickAndPlace-v3', render_mode='human')
    env.unwrapped.task.distance_threshold = 0.05
    obs, info = env.reset()
    
    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"
    
    planner = STLPlanner(user_stl, define_predicates())
    controller = SafeFunnelController() # (obstacle_pos=np.array([0.0, 0.0, 0.2]), obstacle_radius=0.10)

    sim_time = 0.0
    dt = 0.04 # Standard PyBullet timestep

    print("\n--- STARTING SIMULATION ---")
    for _ in range(2000):
        if planner.finished:
            print("--- MISSION COMPLETE ---")
            time.sleep(5)
            break
             
        phase, safety, t_left = planner.update(obs, sim_time)     # 1. Update Logic
        action = controller.get_action(phase, safety, obs, t_left)  # 2. Update Control
        obs, reward, terminated, truncated, info = env.step(action) # 3. Physics Step
        sim_time += dt

        # Reset handling
        if terminated or truncated:
            obs, info = env.reset()
            planner.reset()
            sim_time = 0.0
            print("--- RESET ---")

        time.sleep(0.1) # Visualization speed

    env.close()

if __name__ == "__main__":
    run_simulation()
