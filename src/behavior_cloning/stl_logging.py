import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class STLLoggingCallback(BaseCallback):
    """
    Callback for logging STL metrics to TensorBoard.
    Logs:
    - Success rate (rolling window)
    - Timeout rate (rolling window)
    - Current phase index
    - Phase transitions
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.phase_map = {"approach": 0, "grasp": 1, "move": 2, "DONE": 3}
        self.success_buffer = []
        self.timeout_buffer = []
        self.phase_counts = {}
        self.total_successes = 0
    
    def _on_step(self) -> bool:
        # 1. Log Phase Information
        try:
            planner = self.training_env.envs[0].get_wrapper_attr('planner')
            if planner:
                phase_name = planner.current_node.phase_name
                phase_idx = 3 if planner.finished else self.phase_map.get(phase_name, -1)
                self.logger.record("stl/current_phase_idx", phase_idx)
        except Exception:
            pass

        # 2. Log Episode Outcomes from Info
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'phase_transition' in info:
                trans = info['phase_transition']
                self.phase_counts[trans] = self.phase_counts.get(trans, 0) + 1
                self.logger.record("stl/transitions", 1)

            if info.get('success', False):
                self.success_buffer.append(1)
                self.timeout_buffer.append(0)
                self.total_successes += 1
                self.logger.record("stl/episode_success", 1)
            elif info.get('timeout', False):
                self.success_buffer.append(0)
                self.timeout_buffer.append(1)
                self.logger.record("stl/episode_timeout", 1)
            elif self.locals.get('dones', [False])[0]:
                self.success_buffer.append(0)
                self.timeout_buffer.append(0)
        
        # 3. Log Rolling Statistics
        if len(self.success_buffer) > 0:
            if len(self.success_buffer) > 100:
                self.success_buffer = self.success_buffer[-100:]
                self.timeout_buffer = self.timeout_buffer[-100:]
            self.logger.record("stl/success_rate", np.mean(self.success_buffer))
            self.logger.record("stl/timeout_rate", np.mean(self.timeout_buffer))

        return True
    
    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print("\n=== Phase Transition Statistics ===")
            for transition, count in sorted(self.phase_counts.items()):
                print(f"{transition}: {count}")
            print(f"Total successful episodes: {self.total_successes}")