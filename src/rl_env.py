import gymnasium as gym
import numpy as np
from get_predicates import get_predicates

class STLRLWrapper(gym.Wrapper):
    """
    Gym Wrapper that connects the STL Monitor to the RL Agent.
    1. Observation: Flattens Dict obs and appends 'Task Phase ID'.
    2. Reward: Returns the Robustness Degree (rho) of the ACTIVE sub-task.
    """
    def __init__(self, env, stl_task):
        super().__init__(env)
        self.stl_task = stl_task
        
        # Explicitly silence compute_reward so SB3 doesn't think it's a GoalEnv
        self.compute_reward = None
        
        # Define new observation space: [Obs (Flat) + Goal (Flat) + Task_ID (1)]
        # Panda-Gym obs are usually 25 dims (approx) + 1 for Task ID
        self.base_obs_dim = 0
        sample_obs, _ = env.reset()
        flat_dim = sample_obs['observation'].shape[0] + sample_obs['desired_goal'].shape[0]
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(flat_dim + 1,), 
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.stl_task.reset()
        return self._augment_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # 1. Compute Logic & Robustness
        preds = get_predicates(obs, info)
        # The monitor updates its internal state here
        rho = self.stl_task.robustness(preds)
        
        # 2. Reward Shaping (Dense Reward from Robustness)
        active_node = self.stl_task.get_active_task()
        
        reward = 0.0
        if active_node == "DONE":
            reward = 10.0 # Large bonus for full completion
            terminated = True
        else:
            # Reward is the robustness of the CURRENT active subgoal.
            # e.g., if Aligned is active, reward = (0.03 - dist_xy)
            # As robot gets closer, reward increases from negative to positive.
            reward = active_node.robustness(preds)

        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs):
        # Flatten observation and append the current Task Step Index
        task_id = float(self.stl_task.current_step)
        return np.concatenate([obs['observation'], obs['desired_goal'], [task_id]]).astype(np.float32)
