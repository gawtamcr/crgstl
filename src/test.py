import gymnasium as gym
import numpy as np
from panda_gym.envs.core import Task
from stable_baselines3 import SAC


# ---------- 1. CUSTOM TASK ----------
class STLPickAndPlaceTask(Task):
    def __init__(self, sim):
        super().__init__(sim)

        self.p1_pos = np.array([0.15, 0.0, 0.02], dtype=np.float32)
        self.p2_pos = np.array([-0.15, 0.1, 0.02], dtype=np.float32)
        self.d1_pos = np.array([0.0, 0.3, 0.02], dtype=np.float32)

        self.lift_threshold = 0.05
        self.alpha = 10.0

    def _reset_robot(self):
        chosen_start = self.p1_pos if np.random.rand() > 0.5 else self.p2_pos

        self.sim.set_base_pose(
            "object",
            chosen_start,
            np.array([0, 0, 0, 1])
        )

    def reset(self):
        self.goal = self.d1_pos.copy()
        self._reset_robot()
        return self.get_obs()

    def get_obs(self):
        obj_pos = self.sim.get_base_position("object")
        obj_rot = self.sim.get_base_rotation("object")
        obj_vel = self.sim.get_base_velocity("object")
        obj_ang_vel = self.sim.get_base_angular_velocity("object")

        return np.concatenate(
            (obj_pos, obj_rot, obj_vel, obj_ang_vel)
        ).astype(np.float32)

    def get_achieved_goal(self):
        return self.sim.get_base_position("object").astype(np.float32)

    def is_success(self, achieved_goal, desired_goal, info=None):
        return np.linalg.norm(achieved_goal - desired_goal) < 0.05

    def compute_reward(self, achieved_goal, desired_goal, info):
        ee_pos = info["robot_ee_position"]

        obj_pos = self.sim.get_base_position("object")

        rho_p1 = -np.linalg.norm(ee_pos - self.p1_pos)
        rho_p2 = -np.linalg.norm(ee_pos - self.p2_pos)

        # numerically stable soft OR
        rho_pick = np.logaddexp(
            self.alpha * rho_p1,
            self.alpha * rho_p2,
        ) / self.alpha

        is_lifted = obj_pos[2] > self.lift_threshold
        rho_place = -np.linalg.norm(obj_pos - self.d1_pos)

        contacts = self.sim.get_contact_points()
        collision_penalty = -10.0 if contacts else 0.0

        if not is_lifted:
            reward = rho_pick
        else:
            reward = 2.0 + rho_place

        return float(reward + collision_penalty)


# ---------- 2. WRAPPER ----------
class STLHierarchicalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        base_obs_space = self.env.observation_space["observation"]

        self.observation_space = gym.spaces.Dict({
            "observation": base_obs_space,
            "active_subgoal": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32,
            ),
        })

    def _get_manager_subgoal(self):
        obj_pos = self.env.unwrapped.sim.get_base_position("object")

        if obj_pos[2] < 0.05:
            return obj_pos.astype(np.float32)

        return self.env.unwrapped.task.d1_pos.astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        wrapped_obs = {
            "observation": obs["observation"],
            "active_subgoal": self._get_manager_subgoal(),
        }

        return wrapped_obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        achieved = self.env.unwrapped.task.get_achieved_goal()
        desired = self.env.unwrapped.task.goal
        ee_pos = self.env.unwrapped.robot.get_ee_position()
        info["robot_ee_position"] = ee_pos
        reward = self.env.unwrapped.task.compute_reward(
            achieved,
            desired,
            info,
        )

        wrapped_obs = {
            "observation": obs["observation"],
            "active_subgoal": self._get_manager_subgoal(),
        }

        return wrapped_obs, reward, terminated, truncated, info


# ---------- 3. TRAIN ----------
if __name__ == "__main__":
    env = gym.make("PandaPickAndPlace-v3", render_mode="human")

    env.unwrapped.task = STLPickAndPlaceTask(env.unwrapped.sim)
    env = STLHierarchicalWrapper(env)

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        buffer_size=100_000,
        learning_rate=1e-3,
        gamma=0.95,
        tau=0.005,
        tensorboard_log="./stl_hrl_logs/",
    )

    print("--- Training STL Hierarchical Agent ---")
    model.learn(total_timesteps=150000)

    model.save("panda_stl_poc")
    print("Model Saved.")
