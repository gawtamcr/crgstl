import gymnasium as gym
import numpy as np
from typing import Dict, Callable, Tuple, Optional

from common.stl_conductor import STLConductor

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
        self.conductor = STLConductor(stl_string, predicates)
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
            # Target is object position
            target = obs_dict['achieved_goal'][:3]
        else:
            # Target is final goal position
            target = obs_dict['desired_goal'][:3]
        
        # Robot end-effector position
        ee_pos = base_obs[:3]
        rel_target = target - ee_pos
        
        # Normalize time by max phase duration
        time_feat = np.array([t_left / self.conductor.current_node.max_time], dtype=np.float32)
        
        # Safety flag (currently only checks for avoid_zone constraint)
        safety_feat = np.array([1.0 if safety == "avoid_zone" else 0.0], dtype=np.float32)
        
        return np.concatenate([base_obs, rel_target, time_feat, safety_feat]).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and conductor."""
        obs_dict, info = self.env.reset(seed=seed, options=options)
        self.last_obs_dict = obs_dict
        self.conductor.reset()
        self.sim_time = 0.0
        self.prev_phase = None
        
        phase_info = self.conductor.update(obs_dict, self.sim_time)
        self.prev_phase = phase_info[0]
        
        return self._get_aug_obs(obs_dict, phase_info), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and compute STL-shaped reward."""
        obs_dict, _, terminated, truncated, info = self.env.step(action)
        self.last_obs_dict = obs_dict
        self.sim_time += self.dt
        
        # Update conductor
        phase, safety, t_left = self.conductor.update(obs_dict, self.sim_time)
        
        # Extract positions
        ee_pos = obs_dict['observation'][:3]
        obj_pos = obs_dict['achieved_goal'][:3]
        goal_pos = obs_dict['desired_goal'][:3]
        
        dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
        
        # Dense reward shaping based on current phase
        reward = 0.0
        
        if phase == "approach":
            # Encourage moving end-effector toward object
            reward -= dist_to_obj * 5.0
            # Small penalty for premature closing
            if action[3] < 0:  # Gripper closing
                reward -= 1.0
                
        elif phase == "grasp":
            # Encourage contact with object
            reward -= dist_to_obj * 3.0
            # Reward lifting object
            if obj_pos[2] > 0.025:  # Object off table
                reward += (obj_pos[2] - 0.025) * 100.0
            # Encourage closing gripper during grasp
            if action[3] > 0:  # Gripper opening
                reward -= 2.0
                
        elif phase == "move":
            # Encourage moving object to goal
            reward -= dist_obj_to_goal * 8.0
            # Maintain object height (avoid dropping)
            if obj_pos[2] < 0.025:
                reward -= 10.0
            # Penalty for opening gripper prematurely
            if action[3] > 0:
                reward -= 5.0

        # Phase transition bonus
        if phase != self.prev_phase and self.prev_phase is not None:
            reward += 50.0
            info['phase_transition'] = f"{self.prev_phase} -> {phase}"
        
        self.prev_phase = phase
        
        # Terminal conditions
        if self.conductor.finished:
            terminated = True
            reward += 200.0  # Task completion bonus
            info['success'] = True
            
        if self.conductor.failed_timeout:
            terminated = True
            reward -= 50.0  # Timeout penalty
            info['timeout'] = True

        aug_obs = self._get_aug_obs(obs_dict, (phase, safety, t_left))
        return aug_obs, reward, terminated, truncated, info