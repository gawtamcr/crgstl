import gymnasium as gym
import panda_gym
import numpy as np
import re
import time
import torch as th
from stable_baselines3 import SAC
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import Transitions

# ==========================================
# 1. STL PARSER & CONDUCTOR (Your Logic)
# ==========================================
class RecursiveSTLNode:
    def __init__(self, stl_string):
        self.phase_name = None
        self.safety_constraint = None
        self.min_time = 0.0
        self.max_time = 0.0
        self.next_node = None
        self._parse(stl_string.strip())

    def _parse(self, s):
        match_f = re.match(r"F\[([\d\.]+),([\d\.]+)\]\s*\(([^&]+)(?:&\s*(.*))?\)", s)
        if match_f:
            t_min, t_max, name, rest = match_f.groups()
            self.min_time = float(t_min)
            self.max_time = float(t_max)
            self.phase_name = name.strip()
            if rest:
                match_g = re.search(r"G\(([^)]+)\)", rest)
                if match_g:
                    self.safety_constraint = match_g.group(1).strip()
                    rest = rest.replace(match_g.group(0), "").strip()
                    if rest.startswith("&"): rest = rest[1:].strip()
                if rest:
                    self.next_node = RecursiveSTLNode(rest)

class STLConductor:
    def __init__(self, stl_string, predicates):
        self.root_node = RecursiveSTLNode(stl_string)
        self.current_node = self.root_node
        self.predicates = predicates
        self.phase_start_time = 0.0
        self.finished = False

    def update(self, obs_dict, current_sim_time):
        if self.finished: return "DONE", None, 999.0
        dt = current_sim_time - self.phase_start_time
        time_left = max(0.0, self.current_node.max_time - dt)
        check_func = self.predicates.get(self.current_node.phase_name)
        if check_func and check_func(obs_dict):
            if self.current_node.next_node:
                self.current_node = self.current_node.next_node
                self.phase_start_time = current_sim_time
            else:
                self.finished = True
        return self.current_node.phase_name, self.current_node.safety_constraint, time_left

# ==========================================
# 2. THE EXPERT (Potential Field Controller)
# ==========================================
class SafeFunnelController:
    def __init__(self):
        self.obstacle_pos = np.array([0.0, 0.0, 0.2])
        self.obstacle_radius = 0.10

    def compute_action(self, phase, safety_rule, obs_dict, time_remaining):
        ee_pos = obs_dict['observation'][:3]
        if phase == "approach":
            target_pos = obs_dict['achieved_goal'][:3]
            gripper_act = 1.0
        elif phase == "grasp":
            target_pos = obs_dict['achieved_goal'][:3]
            gripper_act = -1.0
        elif phase == "move":
            target_pos = obs_dict['desired_goal'][:3]
            gripper_act = -1.0
        else: return np.zeros(4)

        # CLF Attraction
        urgency = 1.0 + (5.0 / max(time_remaining, 0.1))
        attraction_vec = (target_pos - ee_pos) * 4.0 * urgency
        
        # CBF Repulsion
        repulsion_vec = np.zeros(3)
        if safety_rule == "avoid_zone":
            dist_to_obs = np.linalg.norm(ee_pos - self.obstacle_pos)
            if dist_to_obs < (self.obstacle_radius + 0.05):
                push_dir = (ee_pos - self.obstacle_pos) / (dist_to_obs + 1e-6)
                strength = 0.05 / (dist_to_obs**2 + 1e-6)
                repulsion_vec = push_dir * min(strength, 20.0)

        action = np.zeros(4)
        action[:3] = attraction_vec + repulsion_vec
        action[3] = gripper_act
        return np.clip(action, -1.0, 1.0)

# ==========================================
# 3. GYM WRAPPER
# ==========================================
class STLGymWrapper(gym.Wrapper):
    def __init__(self, env, stl_string, predicates):
        super().__init__(env)
        self.stl_string = stl_string
        self.predicates = predicates
        self.conductor = STLConductor(stl_string, predicates)
        self.last_obs_dict = None
        self.sim_time = 0.0
        self.dt = 0.04
        
        orig_shape = env.observation_space['observation'].shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(orig_shape + 5,), dtype=np.float32)

    def _get_aug_obs(self, obs_dict, phase_info):
        phase, safety, t_left = phase_info
        base_obs = obs_dict['observation']
        target = obs_dict['achieved_goal'][:3] if phase in ["approach", "grasp"] else obs_dict['desired_goal'][:3]
        rel_target = target - base_obs[:3]
        time_feat = np.array([t_left / 10.0], dtype=np.float32)
        safety_feat = np.array([1.0 if safety == "avoid_zone" else 0.0], dtype=np.float32)
        return np.concatenate([base_obs, rel_target, time_feat, safety_feat]).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs_dict, info = self.env.reset(seed=seed)
        self.last_obs_dict = obs_dict
        self.conductor = STLConductor(self.stl_string, self.predicates)
        self.sim_time = 0.0
        phase_info = self.conductor.update(obs_dict, self.sim_time)
        return self._get_aug_obs(obs_dict, phase_info), info

    def step(self, action):
        obs_dict, _, terminated, truncated, info = self.env.step(action)
        self.last_obs_dict = obs_dict
        self.sim_time += self.dt
        phase_info = self.conductor.update(obs_dict, self.sim_time)
        
        # Simple reward: 100 for finishing missions, else small step penalty
        reward = 100.0 if self.conductor.finished else -0.01
        if self.conductor.finished: terminated = True
        
        return self._get_aug_obs(obs_dict, phase_info), reward, terminated, truncated, info

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
def define_predicates():
    return {
        "approach": lambda o: np.linalg.norm(o['observation'][:3] - o['achieved_goal'][:3]) < 0.02,
        "grasp": lambda o: o['achieved_goal'][2] > 0.03, # Lifted slightly
        "move": lambda o: np.linalg.norm(o['achieved_goal'][:3] - o['desired_goal'][:3]) < 0.05
    }

if __name__ == "__main__":
    user_stl = "F[0,10.0](approach & G(avoid_zone) & F[0,2.0](grasp & F[0,5.0](move)))"
    
    # 1. Create Env
    base_env = gym.make('PandaPickAndPlace-v3')
    env = STLGymWrapper(base_env, user_stl, define_predicates())
    expert_musician = SafeFunnelController()

    # 2. DATA COLLECTION (Expert Demonstrations)
    print("Collecting expert demonstrations from Potential Field Controller...")
    obs_list, act_list, next_obs_list, done_list = [], [], [], []
    
    for episode in range(20):
        obs, _ = env.reset()
        done = False
        while not done:
            # Expert uses its own logic based on internal wrapper state
            phase, safety, t_left = env.conductor.update(env.last_obs_dict, env.sim_time)
            action = expert_musician.compute_action(phase, safety, env.last_obs_dict, t_left)
            
            next_obs, rew, term, trunc, _ = env.step(action)
            obs_list.append(obs)
            act_list.append(action)
            next_obs_list.append(next_obs)
            done_list.append(term or trunc)
            
            obs = next_obs
            done = term or trunc

    expert_transitions = Transitions(
        obs=np.array(obs_list), acts=np.array(act_list),
        next_obs=np.array(next_obs_list), dones=np.array(done_list), infos=[{}]*len(obs_list)
    )

    # 3. BEHAVIORAL CLONING (Pre-training)
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_warm_start/")
    bc_trainer = bc.BC(observation_space=env.observation_space, action_space=env.action_space, demonstrations=expert_transitions)
    
    print("Pre-training SAC Actor via BC...")
    bc_trainer.train(n_epochs=15)

    # 4. RL FINE-TUNING
    print("Starting SAC Fine-tuning...")
    model.learn(total_timesteps=30000)
    model.save("sac_panda_stl_expert_pretrained")

    # 5. TEST
    print("Testing pre-trained agent...")
    obs, _ = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        base_env.render()
        if term or trunc: obs, _ = env.reset()