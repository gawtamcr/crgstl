import numpy as np
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

from common.stl_planner import STLPlanner
from common.predicates import define_stacking_predicates
from custom_env.custom_panda_task import STLPickAndPlaceEnv

class STLStackingTask(Task):
    """
    Task with two objects (Cube 1 and Cube 2) for stacking.
    """
    def __init__(self, sim):
        super().__init__(sim)
        self.sim.create_plane(z_offset=0.0)
        # Cube 1 (Green) - Base
        self.sim.create_box(
            body_name="object1",
            half_extents=np.array([0.02, 0.02, 0.02]),
            mass=1.0,
            position=np.array([0.0, 0.0, 0.0]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        # Cube 2 (Blue) - Top
        self.sim.create_box(
            body_name="object2",
            half_extents=np.array([0.02, 0.02, 0.02]),
            mass=1.0,
            position=np.array([0.0, 0.0, 0.0]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        # Stacking Base Target (Ghost)
        self.sim.create_box(
            body_name="stack_base",
            half_extents=np.array([0.02, 0.02, 0.02]),
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.0]),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        )

    def reset(self):
        # Randomize positions with simple collision avoidance
        bounds_x = [-0.15, 0.15]
        bounds_y = [-0.15, 0.15]
        
        positions = []
        min_dist = 0.08 # 4cm diameter + margin
        
        for _ in range(3): # Obj1, Obj2, Target
            for _ in range(100): # Max attempts
                pos = np.array([
                    np.random.uniform(bounds_x[0], bounds_x[1]),
                    np.random.uniform(bounds_y[0], bounds_y[1]),
                    0.02
                ])
                if all(np.linalg.norm(pos[:2] - p[:2]) > min_dist for p in positions):
                    positions.append(pos)
                    break
            else:
                positions.append(np.array([0.0, 0.0, 0.02])) # Fallback

        object1_pos, object2_pos, target_pos = positions

        self.sim.set_base_pose("object1", position=object1_pos, orientation=np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", position=object2_pos, orientation=np.array([0.0, 0.0, 0.0, 1.0]))
        
        self.stack_base_pos = target_pos
        self.sim.set_base_pose("stack_base", position=self.stack_base_pos, orientation=np.array([0.0, 0.0, 0.0, 1.0]))
        self.goal = self.stack_base_pos

    def get_obs(self):
        # Return concatenated state: [Obj1 (13), Obj2 (13)]
        def get_state(name):
            return np.concatenate([
                self.sim.get_base_position(name),
                self.sim.get_base_rotation(name),
                self.sim.get_base_velocity(name),
                self.sim.get_base_angular_velocity(name)
            ])
        return np.concatenate([get_state("object1"), get_state("object2")])

    def get_achieved_goal(self):
        return self.sim.get_base_position("object1")

    def is_success(self, achieved_goal, desired_goal, info={}):
        return False 

    def compute_reward(self, achieved_goal, desired_goal, info={}):
        return 0.0

class STLStackingEnv(STLPickAndPlaceEnv):
    def __init__(self, render_mode="rgb_array", stl_string=None):
        sim = PyBullet(render_mode=render_mode)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type="ee")
        task = STLStackingTask(sim)
        
        if stl_string is None:
            self.stl_string = "F[0,10.0](approach_1 & F[0,2.0](grasp_1 & F[0,5.0](place_1 & F[0,10.0](approach_2 & F[0,2.0](grasp_2 & F[0,5.0](stack_2))))))"
        else:
            self.stl_string = stl_string
            
        self.predicates = define_stacking_predicates()
        self.planner = STLPlanner(self.stl_string, self.predicates)
        self.sim_time = 0.0
        
        # Bypass STLPickAndPlaceEnv init to avoid re-creating task
        RobotTaskEnv.__init__(self, robot, task)
