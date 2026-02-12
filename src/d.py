def step_norm(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.sim_time += self.dt
        
        phase, safety, t_left = self.conductor.update(obs, self.sim_time)
        
        # --- REWARD SHAPING ---
        reward = 0.0
        # --- NORMALIZED REWARD [-1, 1] ---

        # Workspace max distance (approx for Panda)
        MAX_DIST = 2.0  # Safe upper bound for EE-to-target distance

        ee_pos = obs['observation'][:3]
        target = obs['achieved_goal'][:3] if phase in ["approach", "grasp"] else obs['desired_goal'][:3]

        dist = np.linalg.norm(ee_pos - target)

        # 1. Distance Reward (normalized)
        r_dist = 1.0 - np.clip(dist / MAX_DIST, 0.0, 1.0)
        # r_dist in [0,1]

        # 2. Safety Reward
        r_safety = 0.0
        if safety == "avoid_zone":
            dist_to_obs = np.linalg.norm(ee_pos - self.obstacle_pos)
            if dist_to_obs < self.obstacle_radius:
                r_safety = -1.0  # worst case
            else:
                r_safety = 0.0

        # Inside the STLGymWrapper step function
        previous_phase = self.conductor.current_node.phase_name
        phase, safety, t_left = self.conductor.update(obs, self.sim_time)

        # NEW: Check if we just transitioned to a NEW phase
        if phase != previous_phase and phase != "DONE":
            reward += 20.0  # Subtask completion bonus!
            print(f"RL Bonus: Completed {previous_phase}!")

        # 3. STL Completion Reward
        r_success = 0.0
        if self.conductor.finished:
            r_success = 1.0
            terminated = True

        # 4. Timeout Penalty
        r_timeout = 0.0
        if t_left <= 0 and not self.conductor.finished:
            r_timeout = -1.0
            truncated = True

        # Weighted sum (must stay in [-1,1])
        reward = (
            0.6 * r_dist +
            0.2 * r_safety +
            0.2 * r_success +
            0.2 * r_timeout
        )

        reward = float(np.clip(reward, -1.0, 1.0))

        return self._get_aug_obs(obs, (phase, safety, t_left)), reward, terminated, truncated, info