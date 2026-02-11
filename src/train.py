import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stl import Sequence, Eventually, Predicate
from rl_env import STLRLWrapper
from constants import *

def make_env(render_mode="human"):
    """Creates the STL-wrapped Panda environment."""
    env = gym.make("PandaPickAndPlace-v3", render_mode=render_mode)
    
    # Define the STL Task (Same as in run.py)
    # Note: We use the same thresholds to ensure consistency
    task_logic = Sequence([
        Eventually(Predicate("Aligned", lambda s: ALIGNMENT - s["dist_xy"])),
        Eventually(Predicate("Holding", lambda s: (GRASPING - s["gripper_width"]) if s["gripper_width"] > 0.02 else -1.0)),
        Eventually(Predicate("Lifted",  lambda s: s["obj_z"] - LIFTING)),
        Eventually(Predicate("Placed",  lambda s: PLACEMENT - s["dist_target"]))
    ])
    
    # Wrap the env to provide STL observations and rewards
    env = STLRLWrapper(env, task_logic)
    return env

class RobustnessCallback(BaseCallback):
    """Custom callback to log STL robustness and task progress."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Access the info dict of the first environment
        infos = self.locals.get("infos", [{}])
        if infos:
            info = infos[0]
            if "robustness" in info:
                self.logger.record("stl/robustness", info["robustness"])
            if "task_step" in info:
                self.logger.record("stl/task_step", info["task_step"])
        return True

if __name__ == "__main__":
    print("1. Setting up environment...")
    env = make_env()
    
    # Validate the custom wrapper complies with Gym API
    check_env(env)
    print("   Environment check passed.")

    # 2. Define the Agent
    # MlpPolicy is suitable for state-based observations
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        tensorboard_log="./ppo_stl_tensorboard/",
        device="cuda"
    )

    print("Device used for training:", model.device)
    # 3. Train
    print("2. Starting training (this may take a while)...")
    # For a PoC, 100k steps is a good start to see convergence on sub-tasks
    model.learn(total_timesteps=100_000, callback=RobustnessCallback())
    
    # 4. Save
    model.save("ppo_panda_stl_poc")
    print("3. Training complete. Model saved to 'ppo_panda_stl_poc.zip'.")
