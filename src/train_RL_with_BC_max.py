import os
import gymnasium as gym
import panda_gym
import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from common.predicates import define_predicates
from behavior_cloning.stl_gym_wrapper import STLGymWrapper
from controller.safe_funnel_controller import SafeFunnelController
from behavior_cloning.collect_expert_transitions import collect_expert_transitions
from behavior_cloning.stl_logging import STLLoggingCallback


# ------------------------------
# Config (Laptop Safe Settings)
# ------------------------------
N_ENVS = 8                  # Try 4 first. Increase to 6 if stable.
BATCH_SIZE = 128            # Safe for 4GB VRAM
GRADIENT_STEPS = 8
TRAIN_FREQ = 1
BUFFER_SIZE = 200_000
TOTAL_TIMESTEPS = 200_000


def make_env(user_stl):
    def _init():
        base_env = gym.make('PandaPickAndPlace-v3')  # No rendering!
        env = STLGymWrapper(base_env, user_stl, define_predicates())
        env.unwrapped.task.distance_threshold = 0.04
        return Monitor(env)
    return _init


def main():

    torch.set_float32_matmul_precision("medium")

    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"

    # ==========================================================
    # Phase 1 — Expert Collection (Single Environment)
    # ==========================================================
    print("Phase 1: Collecting Expert Demonstrations")

    expert_env = make_env(user_stl)()
    expert = SafeFunnelController()

    obs, actions, next_obs, rewards, dones = collect_expert_transitions(
        expert_env,
        expert,
        n_episodes=100,
        verbose=True
    )

    expert_env.close()

    # ==========================================================
    # Phase 2 — Vectorized Training Environment
    # ==========================================================
    print(f"\nCreating {N_ENVS} parallel environments...")

    env = SubprocVecEnv([make_env(user_stl) for _ in range(N_ENVS)])

    # ==========================================================
    # SAC Model
    # ==========================================================
    print("Initializing SAC model...")

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",              # Will fallback to CPU if needed
        tensorboard_log="./../models/training/sac_tensorboard_laptop/",
        buffer_size=BUFFER_SIZE,
        learning_rate=3e-4,
        batch_size=BATCH_SIZE,
        tau=0.005,
        gamma=0.98,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
    )

    print("Device used:", model.device)

    # ==========================================================
    # Prefill Replay Buffer with Expert Data
    # ==========================================================
    print(f"\nPre-filling replay buffer with {len(obs)} expert transitions...")

    n_envs = env.num_envs  # should be 4

    for i in range(len(obs)):

        # Expand obs to (n_envs, obs_dim)
        obs_vec = np.zeros((n_envs,) + obs[i].shape, dtype=obs[i].dtype)
        next_obs_vec = np.zeros((n_envs,) + next_obs[i].shape, dtype=next_obs[i].dtype)
        action_vec = np.zeros((n_envs,) + actions[i].shape, dtype=actions[i].dtype)
        reward_vec = np.zeros((n_envs,), dtype=np.float32)
        done_vec = np.zeros((n_envs,), dtype=bool)

        # Fill only first env slot
        obs_vec[0] = obs[i]
        next_obs_vec[0] = next_obs[i]
        action_vec[0] = actions[i]
        reward_vec[0] = rewards[i]
        done_vec[0] = dones[i]

        model.replay_buffer.add(
            obs_vec,
            next_obs_vec,
            action_vec,
            reward_vec,
            done_vec,
            [{} for _ in range(n_envs)]
        )

    print(f"Replay buffer size: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size}")

    # ==========================================================
    # Callbacks
    # ==========================================================
    phase_callback = STLLoggingCallback(verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./../models/training/sac_checkpoints/",
        name_prefix="sac_stl_laptop"
    )

    # ==========================================================
    # Training
    # ==========================================================
    print(f"\nStarting training for {TOTAL_TIMESTEPS} timesteps...")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[phase_callback, checkpoint_callback],
        log_interval=10,
    )

    # ==========================================================
    # Save with Versioning
    # ==========================================================
    base_path = "./../models/training/sac_RL_withBC_v"

    existing_versions = [
        int(f.split('_v')[1].split('.')[0])
        for f in os.listdir('./../models/training/')
        if f.startswith('sac_RL_withBC_v')
    ] or [0]

    new_version = max(existing_versions) + 1

    save_path = f"{base_path}{new_version}"
    model.save(save_path)

    print(f"\nModel saved to {save_path}")
    print("Training complete!")

    env.close()


if __name__ == "__main__":
    main()
