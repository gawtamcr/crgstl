import multiprocessing as mp

# CRITICAL FIX: Set spawn method BEFORE importing anything else
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
import gymnasium as gym
import panda_gym
import torch
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from common.predicates import define_predicates
from behavior_cloning.stl_gym_wrapper import STLGymWrapper
from controller.safe_funnel_controller import SafeFunnelController
from behavior_cloning.collect_expert_transitions import collect_expert_transitions
from behavior_cloning.stl_logging import STLLoggingCallback
import os

def make_env(user_stl, rank=0):
    def _init():
        base_env = gym.make('PandaPickAndPlace-v3') 
        base_env.reset()
        env = STLGymWrapper(base_env, user_stl, define_predicates())
        env.unwrapped.task.distance_threshold = 0.04
        return Monitor(env)
    return _init

def main():
    torch.set_float32_matmul_precision("medium")
    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"

    N_ENVS = 4
    env = SubprocVecEnv([make_env(user_stl, i) for i in range(N_ENVS)])
    print("Initializing SAC model...")
    policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], qf=[512, 512, 512]))
    model = SAC(    "MlpPolicy", 
                    env, 
                    verbose=1, 
                    tensorboard_log="./../models/training/sac_multi/",
                    buffer_size=100_000,
                    learning_rate=3e-4,
                    batch_size=2048,
                    policy_kwargs = policy_kwargs,
                    tau=0.005,
                    gamma=0.98,
       #             ent_coef=0.05,
                    train_freq=128,
                    gradient_steps=128,
                    device="cuda"
                )
    print("Device used for training:", model.device)
    ##########################
    print("Phase 1: Collecting Expert Demonstrations =================================")
    expert_env = make_env(user_stl)()
    expert = SafeFunnelController(position_gain=8.0)
    obs, actions, next_obs, rewards, dones = collect_expert_transitions(expert_env, expert, n_episodes=100, verbose=True)

    expert_env.close()
    # ==========================================================
    # Prefill Replay Buffer with Expert Data
    # ==========================================================
    print(f"\nPre-filling replay buffer with {len(obs)} expert transitions...")

    n_envs = env.num_envs

    for i in range(len(obs)):
        obs_vec = np.zeros((n_envs,) + obs[i].shape, dtype=obs[i].dtype)
        next_obs_vec = np.zeros((n_envs,) + next_obs[i].shape, dtype=next_obs[i].dtype)
        action_vec = np.zeros((n_envs,) + actions[i].shape, dtype=actions[i].dtype)
        reward_vec = np.zeros((n_envs,), dtype=np.float32)
        done_vec = np.zeros((n_envs,), dtype=bool)

        # Fill ONLY one environment slot per transition
        env_idx = i % n_envs  # Cycle through: 0,1,2,3,4,5,6,7,0,1,2,3...
        
        obs_vec[env_idx] = obs[i]           # ✓ FILLS ONLY ONE SLOT
        next_obs_vec[env_idx] = next_obs[i]  # ✓ FILLS ONLY ONE SLOT
        action_vec[env_idx] = actions[i]     # ✓ FILLS ONLY ONE SLOT
        reward_vec[env_idx] = rewards[i]
        done_vec[env_idx] = dones[i]

        model.replay_buffer.add(
            obs_vec,
            next_obs_vec,
            action_vec,
            reward_vec,
            done_vec,
            [{} for _ in range(n_envs)]
        )

    print(f"Replay buffer size: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size}")

    ##########################
    print("Phase 2: SAC Training with Expert-Seeded Buffer =================================")
    phase_callback = STLLoggingCallback(verbose=1)
    # checkpoint_callback = CheckpointCallback(   save_freq=50_000, 
    #                                             save_path="./../models/training/sac_checkpoints/", 
    #                                             name_prefix="sac_multi")
    print("\nStarting training for 200,000 timesteps...")
    model.learn(    total_timesteps=200_000,
                    callback=[phase_callback],
                    log_interval=10,
                )
    base_path = "./../models/training/sac_RL_withBC_v"
    model.save(f"{base_path}{max([int(f.split('_v')[1].split('.')[0]) for f in os.listdir('./../models/training/') if f.startswith('sac_RL_withBC_v')] or [0]) + 1}")
    print(f"Model saved to {base_path} with versioning.")
    
    ###########################

    env.close()
    print("\nTraining complete!")

if __name__ == "__main__":
    main()