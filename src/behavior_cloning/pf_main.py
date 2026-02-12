import gymnasium as gym
import panda_gym
import time
import numpy as np

from predicates import define_predicates
from stl_conductor import STLConductor
from safe_funnel_controller import SafeFunnelController

def run_simulation():
    env = gym.make('PandaPickAndPlace-v3', render_mode='human')
    env.unwrapped.task.distance_threshold = 0.05
    obs, info = env.reset()
    
    # --- STL INPUT ---
    # "Eventually (0-10s) approach the object AND Always avoid the zone, 
    #  THEN Eventually (0-2s) grasp it, 
    #  THEN Eventually (0-5s) move it."
    # user_stl = "F[0,10.0](approach & G(avoid_zone) & F[0,2.0](grasp & F[0,5.0](move)))"
    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"
    
    conductor = STLConductor(user_stl, define_predicates())
    # Initialize controller with obstacle parameters for potential fields
    # musician = SafeFunnelController(obstacle_pos=np.array([0.0, 0.0, 0.2]), obstacle_radius=0.10)
    musician = SafeFunnelController()

    sim_time = 0.0
    dt = 0.04 # Standard PyBullet timestep

    print("\n--- STARTING SIMULATION ---")
    
    # Run loop
    for _ in range(2000):
        if conductor.finished:
            print("--- MISSION COMPLETE ---")
            time.sleep(5)
            break
            
        # 1. Update Logic
        phase, safety, t_left = conductor.update(obs, sim_time)
        
        # 2. Update Control
        action = musician.get_action(phase, safety, obs, t_left)
        
        # 3. Physics Step
        obs, reward, terminated, truncated, info = env.step(action)
        sim_time += dt

        # Reset handling
        if terminated or truncated:
            obs, info = env.reset()
            conductor.reset()
            sim_time = 0.0
            print("--- RESET ---")

        time.sleep(0.1) # Visualization speed

    env.close()

if __name__ == "__main__":
    run_simulation()
