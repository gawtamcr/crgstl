import os
import gymnasium as gym
import panda_gym
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from common.predicates import define_predicates
from behavior_cloning.stl_gym_wrapper import STLGymWrapper
from controller.safe_funnel_controller import SafeFunnelController
from behavior_cloning.collect_expert_transitions import collect_expert_transitions
from behavior_cloning.stl_logging import STLLoggingCallback


def main():

    run = wandb.init(
        entity="gawtamcr-kth",
        project="stl",
        config={
            "algo": "SAC",
            "timesteps": 200_000,
            "buffer_size": 100_000,
            "learning_rate": 3e-4,
            "batch_size": 256,
            "gamma": 0.98,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"

    base_env = gym.make('PandaPickAndPlace-v3', render_mode="rgb_array")
    env = STLGymWrapper(base_env, user_stl, define_predicates())
    env = Monitor(env)  
    env.unwrapped.task.distance_threshold = 0.05

    print("Initializing SAC model...")
    model = SAC(    "MlpPolicy", 
                    env, 
                    verbose=1, 
                    tensorboard_log="./../models/training/sac_tensorboard/",
                    buffer_size=100_000,
                    learning_rate=3e-4,
                    batch_size=256,
                    tau=0.005,
                    gamma=0.98,
                    train_freq=1,
                    gradient_steps=1,
                    device="cuda"
                )
    print("Device used for training:", model.device)

    ##########################
    print("Phase 1: Collecting Expert Demonstrations =================================")
    expert = SafeFunnelController(position_gain=8.0)
    obs, actions, next_obs, rewards, dones = collect_expert_transitions(env, expert, n_episodes=100, verbose=True)

    print(f"\nPre-filling replay buffer with {len(obs)} expert transitions...")
    for i in range(len(obs)):
        model.replay_buffer.add(obs[i], next_obs[i], actions[i], rewards[i], dones[i], [{}])
    print(f"Replay buffer size: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size}")
    
    ##########################
    print("Phase 2: SAC Training with Expert-Seeded Buffer =================================")
    phase_callback = STLLoggingCallback(verbose=1)
    checkpoint_callback = CheckpointCallback(   save_freq=50_000, 
                                                save_path="./../models/training/sac_checkpoints/", 
                                                name_prefix="sac_stl")
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path="./../models/training/wandb_models/",
        verbose=2,
    )
    print("\nStarting training for 200,000 timesteps...")
    model.learn(    total_timesteps=200_000,
                    callback=[phase_callback, checkpoint_callback, wandb_callback],
                    log_interval=10,
                )
    base_path = "./../models/training/sac_RL_withBC_v"
    model.save(f"{base_path}{max([int(f.split('_v')[1].split('.')[0]) for f in os.listdir('./../models/training/') if f.startswith('sac_RL_withBC_v')] or [0]) + 1}")
    print(f"Model saved to {base_path} with versioning.")
    
    ###########################

    env.close()
    wandb.finish()
    print("\nTraining complete!")

if __name__ == "__main__":
    main()