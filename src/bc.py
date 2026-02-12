"""
STL-Guided Reinforcement Learning for Panda Pick-and-Place
===========================================================
Uses Signal Temporal Logic specifications to guide RL training with expert demonstrations.
"""

import gymnasium as gym
import panda_gym
import numpy as np
import torch as th
import re
import time
import random
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Callable, Tuple, Optional, List

# ==========================================
# 1. STL LOGIC & CONDUCTOR
# ==========================================

class RecursiveSTLNode:
    """
    Represents a node in the STL specification tree.
    Parses temporal formulas like F[t_min,t_max](phase & G(safety) & next_formula)
    """
    def __init__(self, stl_string: str):
        self.phase_name: Optional[str] = None
        self.safety_constraint: Optional[str] = None
        self.min_time: float = 0.0
        self.max_time: float = 0.0
        self.next_node: Optional['RecursiveSTLNode'] = None
        self._parse(stl_string.strip())

    def _parse(self, s: str) -> None:
        """Parse STL string into phase components."""
        # Match: F[min,max](phase & G(safety) & next)
        match_f = re.match(r"F\[([\d\.]+),([\d\.]+)\]\s*\(([^&]+)(?:&\s*(.*))?\)", s)
        if not match_f:
            raise ValueError(f"Invalid STL format: {s}")
        
        t_min, t_max, name, rest = match_f.groups()
        self.min_time = float(t_min)
        self.max_time = float(t_max)
        self.phase_name = name.strip()
        
        if rest:
            # Extract safety constraint G(...)
            match_g = re.search(r"G\(([^)]+)\)", rest)
            if match_g:
                self.safety_constraint = match_g.group(1).strip()
                rest = rest.replace(match_g.group(0), "").strip()
                if rest.startswith("&"):
                    rest = rest[1:].strip()
            
            # Parse remaining formula recursively
            if rest:
                self.next_node = RecursiveSTLNode(rest)


class STLConductor:
    """
    Manages progression through STL phases based on predicate satisfaction.
    """
    def __init__(self, stl_string: str, predicates: Dict[str, Callable]):
        self.root_node = RecursiveSTLNode(stl_string)
        self.current_node = self.root_node
        self.predicates = predicates
        self.phase_start_time: float = 0.0
        self.finished: bool = False
        self.failed_timeout: bool = False

    def update(self, obs_dict: Dict, current_sim_time: float) -> Tuple[str, Optional[str], float]:
        """
        Update conductor state based on current observation and time.
        
        Returns:
            (phase_name, safety_constraint, time_remaining)
        """
        if self.finished:
            return "DONE", None, 0.0

        dt = current_sim_time - self.phase_start_time
        time_left = self.current_node.max_time - dt

        # Check timeout
        if time_left <= 0:
            self.failed_timeout = True
            return self.current_node.phase_name, self.current_node.safety_constraint, 0.0

        # Check phase completion predicate
        check_func = self.predicates.get(self.current_node.phase_name)
        if check_func and check_func(obs_dict):
            if self.current_node.next_node:
                # Transition to next phase
                self.current_node = self.current_node.next_node
                self.phase_start_time = current_sim_time
                time_left = self.current_node.max_time
            else:
                # All phases complete
                self.finished = True

        return self.current_node.phase_name, self.current_node.safety_constraint, max(0.0, time_left)

    def reset(self) -> None:
        """Reset conductor to initial state."""
        self.current_node = self.root_node
        self.phase_start_time = 0.0
        self.finished = False
        self.failed_timeout = False


# ==========================================
# 2. GYMNASIUM WRAPPER (Data Alignment)
# ==========================================

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


# ==========================================
# 3. EXPERT CONTROLLER & DEMONSTRATIONS
# ==========================================

class SafeFunnelController:
    """
    Heuristic controller that demonstrates desired behavior for each phase.
    Uses proportional control to move toward phase-specific targets.
    """
    
    def __init__(self, position_gain: float = 8.0):
        self.position_gain = position_gain
    
    def get_action(self, phase: str, safety_rule: Optional[str], 
                   obs_dict: Dict, t_left: float) -> np.ndarray:
        """
        Compute control action based on current phase.
        
        Args:
            phase: Current STL phase name
            safety_rule: Current safety constraint (if any)
            obs_dict: Environment observation dictionary
            t_left: Time remaining in current phase
            
        Returns:
            4D action: [dx, dy, dz, gripper]
        """
        ee_pos = obs_dict['observation'][:3]
        action = np.zeros(4, dtype=np.float32)
        
        if phase == "approach":
            # Move toward object with open gripper
            target = obs_dict['achieved_goal'][:3]
            direction = target - ee_pos
            action[:3] = np.clip(direction * self.position_gain, -1.0, 1.0)
            action[3] = 1.0  # Open gripper
            
        elif phase == "grasp":
            # Move to object and close gripper
            target = obs_dict['achieved_goal'][:3]
            direction = target - ee_pos
            action[:3] = np.clip(direction * self.position_gain, -1.0, 1.0)
            action[3] = -1.0  # Close gripper
            
        elif phase == "move":
            # Move object to goal with closed gripper
            target = obs_dict['desired_goal'][:3]
            # Add small upward bias to maintain lift
            target = target.copy()
            target[2] += 0.05
            direction = target - ee_pos
            action[:3] = np.clip(direction * self.position_gain, -1.0, 1.0)
            action[3] = -1.0  # Keep gripper closed
        
        return action


def collect_expert_transitions(env: STLGymWrapper, controller: SafeFunnelController, 
                                n_episodes: int = 20, verbose: bool = True) -> Tuple[List, List, List, List, List]:
    """
    Collect expert demonstrations using the heuristic controller.
    
    Returns:
        (observations, actions, next_observations, rewards, dones)
    """
    obs_list, act_list, next_obs_list, rew_list, done_list = [], [], [], [], []
    
    successful_episodes = 0
    
    for ep in range(n_episodes):
        aug_obs, info = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            # Get raw observation for expert controller
            raw_obs = env.get_wrapper_attr('last_obs_dict')
            conductor = env.get_wrapper_attr('conductor')
            
            # Get current phase info from conductor (updated in reset/step)
            current_node = conductor.current_node
            phase = current_node.phase_name
            safety = current_node.safety_constraint
            t_left = max(0.0, current_node.max_time - (env.get_wrapper_attr('sim_time') - conductor.phase_start_time))
            
            # Get expert action
            action = controller.get_action(phase, safety, raw_obs, t_left)
            
            # Execute action
            next_aug_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Store transition
            obs_list.append(aug_obs)
            act_list.append(action)
            next_obs_list.append(next_aug_obs)
            rew_list.append(reward)
            done_list.append(terminated or truncated)
            
            aug_obs = next_aug_obs
            done = terminated or truncated
            ep_reward += reward
            time.sleep(0.05)  # Slow down for visualization
            if step_info.get('success', False):
                successful_episodes += 1
        
        if verbose and (ep + 1) % 5 == 0:
            print(f"Expert episode {ep+1}/{n_episodes}, Reward: {ep_reward:.2f}")
    
    if verbose:
        print(f"Collected {len(obs_list)} transitions from {n_episodes} episodes")
        print(f"Success rate: {successful_episodes}/{n_episodes}")
    
    return obs_list, act_list, next_obs_list, rew_list, done_list


# ==========================================
# 4. TRAINING PIPELINE
# ==========================================

def define_predicates() -> Dict[str, Callable]:
    """
    Define phase completion predicates.
    
    Returns:
        Dictionary mapping phase names to predicate functions
    """
    return {
        "approach": lambda o: np.linalg.norm(o['observation'][:3] - o['achieved_goal'][:3]) < 0.015,
        "grasp": lambda o: True, # o['achieved_goal'][2] > 0.045,  # Object lifted
        "move": lambda o: np.linalg.norm(o['achieved_goal'][:3] - o['desired_goal'][:3]) < 0.05
    }


class STLLoggingCallback(BaseCallback):
    """
    Callback for logging STL metrics to TensorBoard.
    Logs:
    - Success rate (rolling window)
    - Timeout rate (rolling window)
    - Current phase index
    - Phase transitions
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.phase_map = {"approach": 0, "grasp": 1, "move": 2, "DONE": 3}
        self.success_buffer = []
        self.timeout_buffer = []
        self.phase_counts = {}
        self.total_successes = 0
    
    def _on_step(self) -> bool:
        # 1. Log Phase Information
        try:
            conductor = self.training_env.envs[0].get_wrapper_attr('conductor')
            if conductor:
                phase_name = conductor.current_node.phase_name
                phase_idx = 3 if conductor.finished else self.phase_map.get(phase_name, -1)
                self.logger.record("stl/current_phase_idx", phase_idx)
        except Exception:
            pass

        # 2. Log Episode Outcomes from Info
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'phase_transition' in info:
                trans = info['phase_transition']
                self.phase_counts[trans] = self.phase_counts.get(trans, 0) + 1
                self.logger.record("stl/transitions", 1)

            if info.get('success', False):
                self.success_buffer.append(1)
                self.timeout_buffer.append(0)
                self.total_successes += 1
                self.logger.record("stl/episode_success", 1)
            elif info.get('timeout', False):
                self.success_buffer.append(0)
                self.timeout_buffer.append(1)
                self.logger.record("stl/episode_timeout", 1)
            elif self.locals.get('dones', [False])[0]:
                self.success_buffer.append(0)
                self.timeout_buffer.append(0)
        
        # 3. Log Rolling Statistics
        if len(self.success_buffer) > 0:
            if len(self.success_buffer) > 100:
                self.success_buffer = self.success_buffer[-100:]
                self.timeout_buffer = self.timeout_buffer[-100:]
            self.logger.record("stl/success_rate", np.mean(self.success_buffer))
            self.logger.record("stl/timeout_rate", np.mean(self.timeout_buffer))

        return True
    
    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print("\n=== Phase Transition Statistics ===")
            for transition, count in sorted(self.phase_counts.items()):
                print(f"{transition}: {count}")
            print(f"Total successful episodes: {self.total_successes}")


def main():
    """Main training pipeline."""
    
    # STL Specification: approach within 10s, grasp within 2s, move within 5s
    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"
    
    print("=" * 60)
    print("STL-Guided RL Training for Panda Pick-and-Place")
    print("=" * 60)
    print(f"STL Specification: {user_stl}")
    print()
    
    # Initialize Environment
    base_env = gym.make('PandaPickAndPlace-v3', render_mode="human")  # Disable rendering for faster training
    env = STLGymWrapper(base_env, user_stl, define_predicates())
    env = Monitor(env)  # Wrap for SB3 logging
    env.unwrapped.task.distance_threshold = 0.05
    # Initialize SAC Model
    print("Initializing SAC model...")
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./sac_stl_tensorboard/",
        buffer_size=100_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
    )
    
    # Phase 1: Warm-start with Expert Demonstrations
    print("\n" + "=" * 60)
    print("Phase 1: Collecting Expert Demonstrations")
    print("=" * 60)
    
    expert = SafeFunnelController(position_gain=8.0)
    obs, actions, next_obs, rewards, dones = collect_expert_transitions(
        env, expert, n_episodes=100, verbose=True
    )
    
    print(f"\nPre-filling replay buffer with {len(obs)} expert transitions...")
    for i in range(len(obs)):
        model.replay_buffer.add(
            obs[i], 
            next_obs[i], 
            actions[i], 
            rewards[i], 
            dones[i], 
            [{}]
        )
    
    print(f"Replay buffer size: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size}")
    
    # Phase 2: RL Training
    print("\n" + "=" * 60)
    print("Phase 2: SAC Training with Expert-Seeded Buffer")
    print("=" * 60)
    
    # Setup callbacks
    phase_callback = STLLoggingCallback(verbose=1)
    
    # Train the model
    print("\nStarting training for 200,000 timesteps...")
    model.learn(
        total_timesteps=200_000,
        callback=phase_callback,
        log_interval=10,
    )
    
    # Save the model
    model_path = "sac_panda_stl_expert_v1"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Phase 3: Evaluation
    print("\n" + "=" * 60)
    print("Phase 3: Evaluation")
    print("=" * 60)
    
    n_eval_episodes = 10
    successes = 0
    
    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        if info.get('success', False):
            successes += 1
        
        print(f"Eval episode {ep+1}: Reward = {ep_reward:.2f}, Success = {info.get('success', False)}")
    
    print(f"\nEvaluation Success Rate: {successes}/{n_eval_episodes} = {100*successes/n_eval_episodes:.1f}%")
    
    env.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()