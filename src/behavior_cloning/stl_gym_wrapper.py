import gymnasium as gym
import numpy as np
from typing import Dict, Callable, Tuple, Optional

from common.stl_planner import STLPlanner

class STLGymWrapper(gym.Wrapper):
    """
    Wraps Panda environment to provide STL-augmented observations and shaped rewards.
    
    Augmented observation includes:
    - Base robot state (joint positions, velocities, etc.)
    - Relative target position (changes based on current phase)
    - Normalized time remaining in current phase
    - Safety flag (binary indicator for safety constraints)
    """
    def __init__(self, env: gym.Env, stl_string: str, predicates: Dict[str, Callable]):
        super().__init__(env)
        self.stl_string = stl_string
        self.predicates = predicates
        self.planner = STLPlanner(stl_string, predicates)
        self.sim_time: float = 0.0
        self.dt: float = 0.04  # Simulation timestep (25 Hz)
        self.last_obs_dict = None

        # Augmented observation space
        # Original observation + relative_target (3D) + time_remaining (1D) + safety_flag (1D)
        orig_shape = env.observation_space['observation'].shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_shape + 5,), dtype=np.float32
        )
        
        # Track previous phase for transition rewards
        self.prev_phase: Optional[str] = None

    def _get_aug_obs(self, obs_dict: Dict, phase_info: Tuple[str, Optional[str], float]) -> np.ndarray:
        """Construct augmented observation vector."""
        phase, safety, t_left = phase_info
        base_obs = obs_dict['observation']
        
        # Determine target based on current phase
        if phase in ["approach", "grasp"]:
            target = obs_dict['achieved_goal'][:3]  # object position
        else:
            target = obs_dict['desired_goal'][:3]   # final goal position

        
        # Robot end-effector position
        ee_pos = base_obs[:3]
        rel_target = target - ee_pos
        
        # Normalize time by max phase duration
        time_feat = np.array([t_left / self.planner.current_node.max_time], dtype=np.float32)
        
        # Safety flag (currently only checks for avoid_zone constraint)
        safety_feat = np.array([1.0 if safety == "avoid_zone" else 0.0], dtype=np.float32)
        
        return np.concatenate([base_obs, rel_target, time_feat, safety_feat]).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and planner."""
        obs_dict, info = self.env.reset(seed=seed, options=options)
        self.last_obs_dict = obs_dict
        self.planner.reset()
        self.sim_time = 0.0
        self.prev_phase = None
        
        phase_info = self.planner.update(obs_dict, self.sim_time)
        self.prev_phase = phase_info[0]
        
        return self._get_aug_obs(obs_dict, phase_info), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and compute STL-shaped reward."""
        obs_dict, _, terminated, truncated, info = self.env.step(action)
        self.last_obs_dict = obs_dict
        self.sim_time += self.dt
        
        # Update planner
        phase, safety, t_left = self.planner.update(obs_dict, self.sim_time)
        
        # Dense reward shaping based on current phase
        reward = 0.0
        
        if phase in self.predicates:
            reward += self.predicates[phase].compute_reward(obs_dict, action)

        # Phase transition bonus
        if phase != self.prev_phase and self.prev_phase is not None:
            reward += 50.0
            info['phase_transition'] = f"{self.prev_phase} -> {phase}"
        
        self.prev_phase = phase
        
        # Terminal conditions
        if self.planner.finished:
            terminated = True
            reward += 200.0  # Task completion bonus
            info['success'] = True
            
        if self.planner.failed_timeout:
            terminated = True
            reward -= 50.0  # Timeout penalty
            info['timeout'] = True

        aug_obs = self._get_aug_obs(obs_dict, (phase, safety, t_left))

        # reward *= 0.1 # Scale reward for stability
        return aug_obs, reward, terminated, truncated, info