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
    
    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"
    
    base_env = gym.make('PandaPickAndPlace-v3', render_mode="human")
    env = STLGymWrapper(base_env, user_stl, define_predicates())
    env = Monitor(env)  
    env.unwrapped.task.distance_threshold = 0.05
    
    print("Initializing SAC model...")
    model = SAC(    "MlpPolicy", 
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
    obs, actions, next_obs, rewards, dones = collect_expert_transitions(env, expert, n_episodes=10, verbose=True)

    print(f"\nPre-filling replay buffer with {len(obs)} expert transitions...")
    for i in range(len(obs)):
        model.replay_buffer.add(obs[i], next_obs[i], actions[i], rewards[i], dones[i], [{}])
    print(f"Replay buffer size: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size}")
    
    ##########################
    print("Phase 2: SAC Training with Expert-Seeded Buffer =================================")
    phase_callback = STLLoggingCallback(verbose=1)
    print("\nStarting training for 200,000 timesteps...")
    model.learn(    total_timesteps=200_000,
                    callback=phase_callback,
                    log_interval=10,
                )
    model_path = "./../models/training/sac_panda_stl_expert_v1"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    ###########################

    env.close()
    print("\nTraining complete!")

if __name__ == "__main__":
    main()