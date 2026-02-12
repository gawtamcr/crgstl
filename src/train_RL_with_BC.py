"""
STL-Guided Reinforcement Learning for Panda Pick-and-Place
===========================================================
Uses Signal Temporal Logic specifications to guide RL training with expert demonstrations.
"""

import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from common.predicates import define_predicates
from behavior_cloning.stl_gym_wrapper import STLGymWrapper
from controller.safe_funnel_controller import SafeFunnelController
from behavior_cloning.collect_expert_transitions import collect_expert_transitions
from behavior_cloning.stl_logging import STLLoggingCallback

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
    # Phase 1: Warm-start with Expert Demonstrations
    print("\n" + "=" * 60)
    print("Phase 1: Collecting Expert Demonstrations")
    print("=" * 60)
    
    expert = SafeFunnelController(position_gain=8.0)
    obs, actions, next_obs, rewards, dones = collect_expert_transitions(
        env, expert, n_episodes=10, verbose=True
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
    # (Evaluation code omitted for brevity, but would follow similar structure)
    
    env.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()