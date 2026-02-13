import gymnasium as gym
import numpy as np
from typing import Dict, Callable, Optional

from common.stl_graph import STLNode, TemporalOp
from common.stl_parsing import parse_stl
from common.stl_conductor import STLConductor
from common.reward_registry import RewardRegistry

class STLGymWrapper(gym.Wrapper):
    def __init__(self, env, stl_formula: str, predicates: Dict[str, Callable], reward_registry: Optional[RewardRegistry] = None):
        super().__init__(env)
        
        # 1. Parse STL
        self.stl_graph = parse_stl(stl_formula)
        
        # Flatten graph for vectorization (Fixed DFS order)
        self.node_list = []
        self._flatten_graph(self.stl_graph)
        
        # 2. Setup Conductor
        self.conductor = STLConductor(self.stl_graph, predicates)
        
        # 3. Setup Rewards
        self.reward_registry = reward_registry if reward_registry else RewardRegistry()
        
        # Simulation time tracking
        self.sim_time = 0.0
        self.dt = getattr(env.unwrapped, 'dt', 0.05) # Default to 0.05s if not found
        
        # 4. Update Observation Space
        # We add a 'stl_state' vector: [active, satisfied, violated, time_ratio] per node
        self.features_per_node = 4
        self.stl_vec_size = len(self.node_list) * self.features_per_node
        
        # Handle Dict observation space (standard for PandaGym)
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            new_spaces = self.env.observation_space.spaces.copy()
            new_spaces['stl_state'] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.stl_vec_size,), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(new_spaces)
        self.last_obs_dict = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        self.sim_time = 0.0
        self.conductor.reset()
        
        # Initial update to set active nodes
        self.conductor.update(obs, self.sim_time)
        
        # Augment Observation
        obs['stl_state'] = self._get_stl_vector()
        self.last_obs_dict = obs
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update Time
        self.sim_time += self.dt
        
        # Update STL State
        violated, penalty_flag, objectives, safety_monitors = self.conductor.update(obs, self.sim_time)
        
        # Augment Observation
        obs['stl_state'] = self._get_stl_vector()
        self.last_obs_dict = obs
        
        # --- Calculate STL Reward ---
        stl_reward = 0.0
        
        # 1. Progress Reward (Dense)
        # Sum rewards for all currently active objectives (e.g., approach, grasp)
        for obj in objectives:
            stl_reward += self.reward_registry.get_reward(obj, obs)
            
        # 2. Safety Penalty (Soft)
        # Apply penalties for active safety constraints (e.g., keep bounds)
        for safe in safety_monitors:
            stl_reward -= self.reward_registry.get_penalty(safe, obs)
            
        # 3. Violation Penalty (Hard)
        # If a hard constraint (Always) is violated or a deadline is missed
        if violated:
            stl_reward -= 10.0  # Significant penalty
            # We do NOT terminate, as requested, to allow learning recovery/avoidance
            # unless the base environment terminated.
        
        # 4. Success Bonus (New)
        # If the root node is satisfied, the entire specification is met.
        if self.conductor.node_status[self.conductor.root]['satisfied']:
            stl_reward += 50.0
            terminated = True
            info['is_success'] = True
            info['success'] = True
        
        total_reward = reward + stl_reward
        
        # Add debug info
        info['stl_objectives'] = objectives
        info['stl_violated'] = violated
        
        return obs, total_reward, terminated, truncated, info

    def get_current_phase_info(self):
        """
        Returns (phase_name, safety_constraints, time_left) for the expert controller.
        Derived from the internal STLConductor state.
        """
        conductor = self.conductor
        
        # 1. Phase Name
        if not conductor.current_objectives:
            phase = "DONE"
        else:
            phase = conductor.current_objectives[0] # Take the first active objective
            
        # 2. Safety Constraints
        safety = conductor.active_safety if conductor.active_safety else None
        
        # 3. Time Left
        # Find the tightest deadline among active Eventually nodes
        min_time_left = float('inf')
        found_active_temporal = False
        
        for node, status in conductor.node_status.items():
            if status.get('active') and not status.get('satisfied') and not status.get('violated'):
                # Check if it is a TemporalOp with a deadline (Eventually)
                if isinstance(node, TemporalOp) and node.name == "F" and node.t_end != float('inf'):
                    elapsed = self.sim_time - status.get('activation_time', 0.0)
                    t_left = node.t_end - elapsed
                    if t_left < min_time_left:
                        min_time_left = t_left
                        found_active_temporal = True
        
        if not found_active_temporal:
            time_left = 10.0 # Default fallback if no explicit deadline
        else:
            time_left = max(0.0, min_time_left)
            
        return phase, safety, time_left

    def _flatten_graph(self, node: STLNode):
        """Recursively collects nodes to establish a fixed vector order."""
        self.node_list.append(node)
        for child in node.children:
            self._flatten_graph(child)

    def _get_stl_vector(self) -> np.ndarray:
        """
        Encodes the current STL state into a vector.
        Format per node: [is_active, is_satisfied, is_violated, time_ratio]
        """
        vec = []
        for node in self.node_list:
            status = self.conductor.node_status.get(node, {})
            
            active = 1.0 if status.get('active', False) else 0.0
            satisfied = 1.0 if status.get('satisfied', False) else 0.0
            violated = 1.0 if status.get('violated', False) else 0.0
            
            # Time Ratio (0.0 to 1.0)
            time_ratio = 0.0
            if active and isinstance(node, TemporalOp) and node.t_end != float('inf'):
                # How far are we into the deadline?
                elapsed = self.sim_time - status.get('activation_time', 0.0)
                # Normalize: 0.0 (start) -> 1.0 (deadline)
                time_ratio = np.clip(elapsed / node.t_end, 0.0, 1.0)
            elif active and isinstance(node, TemporalOp):
                 # For infinite horizon, maybe use a sigmoid or just 0
                 time_ratio = 0.0
            
            vec.extend([active, satisfied, violated, time_ratio])
            
        return np.array(vec, dtype=np.float32)
