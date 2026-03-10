import time
import numpy as np

from common.predicates import define_stacking_predicates
from common.stl_planner import STLPlanner
from controller.stacking_controller import StackingController
from custom_env.stacking_task import STLStackingEnv

def run_stacking_simulation():
    
    # 1. Setup Environment
    # Note: We use the custom class directly, not gym.make
    env = STLStackingEnv(render_mode='human')
    
    # 3. Setup Controller
    controller = StackingController()

    dt = 0.04
    num_episodes = 5

    print(f"\n--- STARTING STACKING EVALUATION ({num_episodes} Episodes) ---")
    print(f"STL: {env.stl_string}")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        sim_time = 0.0
        done = False
        
        while not done:
            # Update Planner (Env does this internally in step, but we need phase for controller)
            # Since we are using the raw env, we can access the planner directly
            phase = env.planner.current_node.phase_name
            safety = env.planner.current_node.safety_constraint
            
            action = controller.get_action(phase, safety, obs, 0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if env.planner.finished:
                print(f"Episode {ep+1}: SUCCESS")
                break
            elif done:
                print(f"Episode {ep+1}: FAILED")

            time.sleep(0.02)

    env.close()

if __name__ == "__main__":
    run_stacking_simulation()
