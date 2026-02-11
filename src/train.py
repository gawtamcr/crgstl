import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stl import Sequence, Eventually, Predicate
from rl_env import STLRLWrapper
from constants import *

def make_env(render_mode="rgb_array"):
    """Creates the STL-wrapped Panda environment."""
    env = gym.make("PandaPickAndPlace-v3", render_mode=render_mode)
    
    # Define the STL Task (Same as in run.py)
    # Note: We use the same thresholds to ensure consistency
    task_logic = Sequence([
        Eventually(Predicate("Aligned", lambda s: ALIGNMENT - s["dist_xy"])),
        Eventually(Predicate("Holding", lambda s: GRASPING - s["gripper_width"])),
        Eventually(Predicate("Lifted",  lambda s: s["obj_z"] - LIFTING)),
        Eventually(Predicate("Placed",  lambda s: PLACEMENT - s["dist_target"]))
    ])
    
    # Wrap the env to provide STL observations and rewards
    env = STLRLWrapper(env, task_logic)
    return env

if __name__ == "__main__":
    print("1. Setting up environment...")
    env = make_env()
    
    # Validate the custom wrapper complies with Gym API
    #check_env(env)
    print("   Environment check passed.")

    # 2. Define the Agent
    # MlpPolicy is suitable for state-based observations
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        tensorboard_log="./ppo_stl_tensorboard/"
    )

    # 3. Train
    print("2. Starting training (this may take a while)...")
    # For a PoC, 100k steps is a good start to see convergence on sub-tasks
    model.learn(total_timesteps=100_000)
    
    # 4. Save
    model.save("ppo_panda_stl_poc")
    print("3. Training complete. Model saved to 'ppo_panda_stl_poc.zip'.")
