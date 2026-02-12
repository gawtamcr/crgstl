import gymnasium as gym
import panda_gym
import numpy as np
import re
import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

class STLMetricCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(STLMetricCallback, self).__init__(verbose)
        self.phase_map = {"approach": 1, "grasp": 2, "move": 3, "DONE": 4}

    def _on_step(self) -> bool:
        # 1. Log success rate from environment
        if 'is_success' in self.locals['infos'][0]:
            self.logger.record('env/success_rate', self.locals['infos'][0]['is_success'])
        
        # 2. Access the conductor safely
        # We reach into the first vector environment (envs[0])
        # and use get_wrapper_attr to find 'conductor' regardless of wrapper depth
        try:
            env0 = self.training_env.envs[0]
            conductor = env0.get_wrapper_attr('conductor')
            
            current_phase = conductor.current_node.phase_name
            phase_val = 4 if conductor.finished else self.phase_map.get(current_phase, 0)
            self.logger.record('stl/phase_depth', phase_val)
        except Exception:
            # Fallback if the wrapper structure is unusual
            pass
            
        return True

# ==========================================
# 1. RECURSIVE STL PARSER
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
        # Regex to extract F[min, max](predicate & rest)
        match_f = re.match(r"F\[([\d\.]+),([\d\.]+)\]\s*\(([^&]+)(?:&\s*(.*))?\)", s)
        if match_f:
            t_min, t_max, name, rest = match_f.groups()
            self.min_time = float(t_min)
            self.max_time = float(t_max)
            self.phase_name = name.strip()

            if rest:
                # Handle nested Globally (G) safety constraints if present
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
        self.failed_timeout = False

    def update(self, obs, current_sim_time):
        if self.finished:
            return "DONE", None, 0.0

        dt = current_sim_time - self.phase_start_time
        time_left = self.current_node.max_time - dt

        # Failure Condition: Time ran out for this specific STL requirement
        if time_left <= 0:
            self.failed_timeout = True
            return self.current_node.phase_name, self.current_node.safety_constraint, 0.0

        check_func = self.predicates.get(self.current_node.phase_name)
        if check_func and check_func(obs):
            if self.current_node.next_node:
                self.current_node = self.current_node.next_node
                self.phase_start_time = current_sim_time
                time_left = self.current_node.max_time 
            else:
                self.finished = True

        return self.current_node.phase_name, self.current_node.safety_constraint, max(0.0, time_left)

# ==========================================
# 2. STL GYMNASIUM WRAPPER
# ==========================================
class STLGymWrapper(gym.Wrapper):
    def __init__(self, env, stl_string, predicates):
        super().__init__(env)
        self.stl_string = stl_string
        self.predicates = predicates
        self.conductor = STLConductor(stl_string, predicates)
        self.sim_time = 0.0
        self.dt = 0.04  # Standard for panda-gym

        # Augmentation: Robot State + Rel Vector (3) + Time (1) + Safety Toggle (1)
        orig_shape = env.observation_space['observation'].shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_shape + 5,), dtype=np.float32
        )

    def _get_aug_obs(self, obs, phase_info):
        phase, safety, t_left = phase_info
        base_obs = obs['observation'] 
        
        # Context-aware target selection
        if phase in ["approach", "grasp"]:
            target = obs['achieved_goal'][:3]
        else:
            target = obs['desired_goal'][:3]
            
        rel_target = target - base_obs[:3]
        time_feat = np.array([t_left / 10.0], dtype=np.float32)
        safety_feat = np.array([1.0 if safety == "avoid_zone" else 0.0], dtype=np.float32)
        
        return np.concatenate([base_obs, rel_target, time_feat, safety_feat]).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.conductor = STLConductor(self.stl_string, self.predicates)
        self.sim_time = 0.0
        phase_info = self.conductor.update(obs, self.sim_time)
        return self._get_aug_obs(obs, phase_info), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.sim_time += self.dt
        
        previous_phase = self.conductor.current_node.phase_name
        phase, safety, t_left = self.conductor.update(obs, self.sim_time)
        
        reward = 0.0
        ee_pos = obs['observation'][:3]
        obj_pos = obs['achieved_goal'][:3]
        dist_to_obj = np.linalg.norm(ee_pos - obj_pos)

        # --- Phase-Specific Dense Rewards ---
        if phase == "approach":
            reward -= dist_to_obj * 4.0
        elif phase == "grasp":
            reward -= dist_to_obj * 2.0 
            if obj_pos[2] > 0.025: # Lifting reward
                reward += (obj_pos[2] - 0.02) * 100.0
            if action[3] > 0: # Penalty for opening gripper when trying to grasp
                reward -= 2.0 

        elif phase == "move":
            target = obs['desired_goal'][:3]
            dist_to_final = np.linalg.norm(obj_pos - target)
            reward -= dist_to_final * 5.0
            if obj_pos[2] < 0.02: # Heavy penalty for dropping
                reward -= 10.0

        # --- STL Logic Rewards & Termination ---
        if phase != previous_phase:
            reward += 50.0 # Logic progression bonus
            
        if self.conductor.finished:
            reward += 100.0
            terminated = True
        
        if self.conductor.failed_timeout:
            reward -= 20.0 
            terminated = True

        return self._get_aug_obs(obs, (phase, safety, t_left)), reward, terminated, truncated, info

# ==========================================
# 3. MONITORING & EXECUTION
# ==========================================
def define_predicates():
    return {
        "approach": lambda o: np.linalg.norm(o['observation'][:3] - o['achieved_goal'][:3]) < 0.015,
        "grasp": lambda o: o['achieved_goal'][2] > 0.045, 
        "move": lambda o: np.linalg.norm(o['achieved_goal'][:3] - o['desired_goal'][:3]) < 0.05
    }


if __name__ == "__main__":
    # STL Formula: Approach within 10s, then Grasp within 2s, then Move to goal within 5s
    user_stl = "F[0,10.0](approach & F[0,2.0](grasp & F[0,5.0](move)))"
    
    # Initialize Environment
    base_env = gym.make('PandaPickAndPlace-v3', render_mode="human")
    env = STLGymWrapper(base_env, user_stl, define_predicates())
    
    # Initialize SAC
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-3, 
        gamma=0.98,
        tensorboard_log="./sac_panda_stl_logs/"
    )

    print("Training... Check progress with: tensorboard --logdir ./sac_panda_stl_logs/")
    model.learn(total_timesteps=50000, callback=STLMetricCallback())
    model.save("sac_panda_stl_model")