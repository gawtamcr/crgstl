from typing import Dict, Callable, Tuple, Optional

from common.stl_parsing import RecursiveSTLNode

class STLConductor:
    def __init__(self, stl_string: str, predicates: Dict[str, Callable]):
        print(f"Parsing STL: {stl_string}")
        self.root_node = RecursiveSTLNode(stl_string)
        self.current_node = self.root_node
        self.predicates = predicates
        self.phase_start_time: float = 0.0
        self.finished: bool = False
        self.failed_timeout: bool = False

    def update(self, obs_dict: Dict, current_sim_time: float) -> Tuple[str, Optional[str], float]:

        if self.finished:
            return "DONE", None, 0.0

        dt = current_sim_time - self.phase_start_time
        time_left = self.current_node.max_time - dt

        # Check timeout
        if time_left <= 0:
            self.failed_timeout = True
            return self.current_node.phase_name, self.current_node.safety_constraint, 0.0

        # Check phase completion predicate
        check_func = self.predicates.get(self.current_node.phase_name)
        if check_func and check_func(obs_dict):
            if self.current_node.next_node:
                # Transition to next phase
                self.current_node = self.current_node.next_node
                self.phase_start_time = current_sim_time
                time_left = self.current_node.max_time
            else:
                # All phases complete
                self.finished = True

        return self.current_node.phase_name, self.current_node.safety_constraint, max(0.0, time_left)

    def reset(self) -> None:
        """Reset conductor to initial state."""
        self.current_node = self.root_node
        self.phase_start_time = 0.0
        self.finished = False
        self.failed_timeout = False