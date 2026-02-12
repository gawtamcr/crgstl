import time
from typing import List, Tuple

from behavior_cloning.stl_gym_wrapper import STLGymWrapper
from controller.safe_funnel_controller import SafeFunnelController

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
            planner = env.get_wrapper_attr('planner')
            
            # Get current phase info from planner (updated in reset/step)
            current_node = planner.current_node
            phase = current_node.phase_name
            safety = current_node.safety_constraint
            t_left = max(0.0, current_node.max_time - (env.get_wrapper_attr('sim_time') - planner.phase_start_time))
            
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
          #  time.sleep(0.05)  # Slow down for visualization
            if step_info.get('success', False):
                successful_episodes += 1
        
        if verbose and (ep + 1) % 5 == 0:
            print(f"Expert episode {ep+1}/{n_episodes}, Reward: {ep_reward:.2f}")
    
    if verbose:
        print(f"Collected {len(obs_list)} transitions from {n_episodes} episodes")
        print(f"Success rate: {successful_episodes}/{n_episodes}")
    
    return obs_list, act_list, next_obs_list, rew_list, done_list