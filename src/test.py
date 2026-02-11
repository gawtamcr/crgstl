import gymnasium as gym
import numpy as np
from panda_gym.envs.core import Task
from stable_baselines3 import SAC

# --- 1. CUSTOM TASK DEFINITION ---
class STLPickAndPlaceTask(Task):
    def __init__(self, sim):
        super().__init__(sim)
        # Goal and object locations
        self.p1_pos = np.array([0.15, 0.0, 0.02])
        self.p2_pos = np.array([-0.15, 0.1, 0.02])
        self.d1_pos = np.array([0.0, 0.3, 0.02])
        self.lift_threshold = 0.05
        self.alpha = 10.0  # Smoothing for Soft-Max (OR logic)

    def _reset_robot(self):
        """Modified reset to ensure two objects are positioned."""
        self.sim.set_base_position("object", self.p1_pos)
        # Note: If your XML doesn't have 'object2', this will target the same object.
        # In a PoC, we can simulate P1 v P2 by randomly placing the single object 
        # at either p1_pos or p2_pos to test the 'OR' robustness.
        chosen_start = self.p1_pos if np.random.random() > 0.5 else self.p2_pos
        self.sim.set_base_position("object", chosen_start)
        self.sim.set_base_orientation("object", np.array([0, 0, 0, 1]))

    def reset(self):
        self.goal = self.d1_pos
        self._reset_robot()
        return self.get_obs()

    def get_obs(self):
        ee_pos = self.sim.get_ee_position()
        ee_vel = self.sim.get_ee_velocity()
        obj_pos = self.sim.get_base_position("object")
        obj_rot = self.sim.get_base_rotation("object")
        obj_vel = self.sim.get_base_velocity("object")
        obj_ang_vel = self.sim.get_base_angular_velocity("object")
        return np.concatenate((ee_pos, ee_vel, obj_pos, obj_rot, obj_vel, obj_ang_vel))

    def get_achieved_goal(self):
        return self.sim.get_base_position("object")

    def is_success(self, achieved_goal, desired_goal, info=None):
        return np.array(np.linalg.norm(achieved_goal - desired_goal, axis=-1) < 0.05, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info):
        ee_pos = self.sim.get_ee_position()
        obj_pos = self.sim.get_base_position("object")

        # STL Predicate: Proximity to Pick (Handles the OR logic implicitly via distance)
        rho_p1 = -np.linalg.norm(ee_pos - self.p1_pos)
        rho_p2 = -np.linalg.norm(ee_pos - self.p2_pos)
        
        # Soft-Max Disjunction (The OR part of the STL)
        rho_pick = (1/self.alpha) * np.log(np.exp(self.alpha * rho_p1) + np.exp(self.alpha * rho_p2))

        # STL Predicate: Is Lifted?
        is_lifted = obj_pos[2] > self.lift_threshold

        # STL Predicate: Proximity to Place (The Conjunction part)
        rho_place = -np.linalg.norm(obj_pos - self.d1_pos)

        # Safety: Collision (The Global G[!collision] part)
        # Panda-gym provides contact information in the sim
        collision = self.sim.get_contact_points()
        collision_penalty = -10.0 if len(collision) > 0 else 0.0

        if not is_lifted:
            return rho_pick + collision_penalty
        else:
            # Bonus for lift + placement gradient
            return 2.0 + rho_place + collision_penalty

# --- 2. HIERARCHICAL WRAPPER ---
class STLHierarchicalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Dict space for SB3 MultiInputPolicy
        self.observation_space = gym.spaces.Dict({
            "observation": self.env.observation_space,
            "active_subgoal": gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        })

    def _get_manager_subgoal(self, obs):
        """Hardcoded Manager logic to guide the Worker."""
        obj_pos = self.env.unwrapped.sim.get_base_position("object")
        ee_pos = self.env.unwrapped.sim.get_ee_position()

        # If object is on table, target the object (Pick Phase)
        if obj_pos[2] < 0.05:
            return obj_pos
        # If object is lifted, target the destination (Place Phase)
        return self.env.unwrapped.task.d1_pos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        wrapped_obs = {
            "observation": obs, 
            "active_subgoal": self._get_manager_subgoal(obs)
        }
        return wrapped_obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        # Fetch reward from our STL-based compute_reward
        reward = self.env.unwrapped.task.compute_reward(None, None, info)
        
        wrapped_obs = {
            "observation": obs, 
            "active_subgoal": self._get_manager_subgoal(obs)
        }
        return wrapped_obs, reward, terminated, truncated, info

# --- 3. MAIN TRAINING LOOP ---
if __name__ == "__main__":
    # 1. Create original env
    env = gym.make("PandaPickAndPlace-v3", render_mode="human")
    
    # 2. Inject STL Task
    env.unwrapped.task = STLPickAndPlaceTask(env.unwrapped.sim)
    
    # 3. Add Hierarchical Wrapper
    env = STLHierarchicalWrapper(env)

    # 4. Initialize SB3 Agent
    # MultiInputPolicy is key for handling the Dict observation
    model = SAC(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        buffer_size=100_000,
        learning_rate=1e-3,
        gamma=0.95, # Shorter horizon for STL subgoals
        tau=0.005,
        tensorboard_log="./stl_hrl_logs/"
    )

    print("--- Training STL Hierarchical Agent ---")
    model.learn(total_timesteps=150000)
    
    model.save("panda_stl_poc")
    print("Model Saved.")