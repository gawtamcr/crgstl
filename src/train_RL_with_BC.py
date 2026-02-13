import gymnasium as gym
import panda_gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from common.predicates import define_predicates
from behavior_cloning.stl_gym_wrapper import STLGymWrapper
from common.reward_registry import RewardRegistry
from controller.safe_funnel_controller import SafeFunnelController
from behavior_cloning.collect_expert_transitions import collect_expert_transitions
from behavior_cloning.stl_logging import STLLoggingCallback
import os

def main():
    
    # Flattened STL to prevent phase reversion when 'approach' becomes False during movement.
    # Times are cumulative/absolute from the start.
    user_stl = "F[0,5.0](approach) & F[0,10.0](grasp) & F[0,15.0](move)"
    
    # --- Define Rewards ---
    # We use dense rewards to guide the agent when specific predicates are active.
    registry = RewardRegistry()
    
    # 1. Approach: Minimize distance to object
    # obs['observation'][0:3] is EE position, obs['achieved_goal'][0:3] is object position (in PickAndPlace)
    # Actually in PandaPickAndPlace-v3: 
    # 'observation' contains [ee_pos, ee_vel, fingers_width, obj_pos, obj_rot, obj_vel, ...]
    # 'achieved_goal' is object_position
    # 'desired_goal' is target_position
    
    def reward_approach(obs):
        # Distance between EE and Object
        dist = np.linalg.norm(obs['observation'][:3] - obs['achieved_goal'][:3])
        # Shaped reward: [0, 3.0] - Positive gradient to overcome step cost
        return 3.0 * (1.0 - np.tanh(5.0 * dist))

    def reward_move(obs):
        # Distance between Object and Target
        dist = np.linalg.norm(obs['achieved_goal'][:3] - obs['desired_goal'][:3])
        return 3.0 * (1.0 - np.tanh(5.0 * dist))

    registry.register_objective("approach", reward_approach)
    registry.register_objective("move", reward_move)
    
    # Grasp Reward: Encourage holding (width between 0.005 and 0.05). Scale to match others.
    registry.register_objective("grasp", lambda o: 3.0 if 0.005 < o['observation'][6] < 0.05 else 0.0)


    base_env = gym.make('PandaPickAndPlace-v3', render_mode="human")
    env = STLGymWrapper(base_env, user_stl, define_predicates(), reward_registry=registry)
    env = Monitor(env)  
    env.unwrapped.task.distance_threshold = 0.05
    
    print("Initializing SAC model...")
    # Use MultiInputPolicy to handle Dict observation with 'stl_state'
    model = SAC(    "MultiInputPolicy", 
                    env, 
                    verbose=1, 
                    tensorboard_log="./../models/training/sac_crgstl_tensorboard/",
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
    print("\nStarting training for 200,000 timesteps...")
    model.learn(    total_timesteps=200_000,
                    callback=[phase_callback, checkpoint_callback],
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