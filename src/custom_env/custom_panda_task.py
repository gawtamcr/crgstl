"""
Custom Panda Task Environment defined by STL logic.
"""
import numpy as np
import gymnasium as gym
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from stable_baselines3 import SAC
import time

from common.stl_planner import STLPlanner
from common.predicates import define_predicates
from behavior_cloning.stl_gym_wrapper import STLGymWrapper

class STLPickAndPlaceTask(Task):
    """
    Custom Pick and Place Task where success and reward are defined by STL.
    Defined manually as a subclass of Task.
    """
    def __init__(self, sim):
        super().__init__(sim)
        self.sim.create_plane(z_offset=0.0)
        # Create a box (object)
        self.sim.create_box(
            body_name="object",
            half_extents=np.array([0.05, 0.02, 0.02]),
            mass=1.0,
            position=np.array([0.0, 1.0, 0.0]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.array([0.02, 0.02, 0.02]),
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.0]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        )

    def reset(self):
        # Randomly sample a goal position
        self.goal = np.random.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, 0.05])
        # Reset the position of the object
        object_pos = np.random.uniform([-0.1, -0.1, 0.0], [0.1, 0.1, 0.05])
        self.sim.set_base_pose("object", position=object_pos, orientation=np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target", position=self.goal, orientation=np.array([0.0, 0.0, 0.0, 1.0]))

    def get_obs(self):
        # The observation must match the standard PickAndPlace task for the model to work
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self):
        # The achieved goal is the current position of the object
        return self.sim.get_base_position("object")

    def is_success(self, achieved_goal, desired_goal, info={}):
        # Compute the distance between the goal position and the current object position
        d = distance(achieved_goal, desired_goal)
        # Return True if the distance is < 0.05, and False otherwise
        return np.array(d < 0.05, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        # Reward = 1.0 if the task is successful, 0.0 otherwise
        return self.is_success(achieved_goal, desired_goal, info).astype(np.float32)

class STLPickAndPlaceEnv(RobotTaskEnv):
    """
    Custom Environment that integrates STLPlanner for task definition.
    """
    def __init__(self, render_mode="rgb_array", stl_string=None):
        sim = PyBullet(render_mode=render_mode)
        
        # Standard Panda setup for PickAndPlace (block_gripper=False allows grasping)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
        
        # Use our custom task which disables default rewards
        task = STLPickAndPlaceTask(sim)
        
        # STL Configuration
        if stl_string is None:
            self.stl_string = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"
        else:
            self.stl_string = stl_string
            
        self.predicates = define_predicates()
        self.planner = STLPlanner(self.stl_string, self.predicates)
        self.sim_time = 0.0
        
        super().__init__(robot, task)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.planner.reset()
        self.sim_time = 0.0
        return obs, info

    def step(self, action):
        # 1. Execute physics step (RobotTaskEnv.step)
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 2. Update Simulation Time
        # Calculate dt based on simulation parameters (timestep * substeps)
        dt = self.sim.timestep * self.sim.n_substeps
        self.sim_time += dt
        
        # 3. Update STL Planner with the new observation
        phase, safety, time_left = self.planner.update(obs, self.sim_time)
        
        # 4. Override Reward and Termination based on STL status
        if self.planner.finished:
            reward = 1.0
            terminated = True
            info["is_success"] = True
            info["phase"] = "DONE"
        elif self.planner.failed_timeout:
            reward = -1.0
            terminated = True
            info["is_success"] = False
            info["phase"] = "FAILED"
        else:
            reward = 0.0
            terminated = False
            info["is_success"] = False
            info["phase"] = phase
            
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    # Example usage
    base_env = STLPickAndPlaceEnv(render_mode="human")
    env = STLGymWrapper(base_env, base_env.stl_string, base_env.predicates)
    
    # Load the trained model
    # Note: The model expects the same observation space it was trained on.
    model = SAC.load("../models/training/sac_RL_withBC_v3")
    
    print(f"Task STL: {base_env.stl_string}")
    
    obs, info = env.reset()
    print("Environment reset. Starting loop...")
    
    for i in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i}: Phase={info.get('phase')}, Reward={reward}")

        time.sleep(0.05) # Slow down for visualization
        if terminated or truncated:
            print(f"Episode finished at step {i}. Success: {info.get('is_success')}")
            obs, info = env.reset()
            
    env.close()
