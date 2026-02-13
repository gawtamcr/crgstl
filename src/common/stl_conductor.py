import time
from typing import Dict, Callable, List, Set, Tuple, Optional
from common.stl_graph import STLNode, Predicate, And, Or, Eventually, Always

class STLConductor:
    def __init__(self, stl_root: STLNode, predicates: Dict[str, Callable]):
        self.root = stl_root
        self.predicates = predicates
        
        # State tracking
        self.node_status: Dict[STLNode, dict] = {} 
        # status dict structure: 
        # {
        #   'active': bool, 
        #   'satisfied': bool, 
        #   'activation_time': float, 
        #   'violated': bool
        # }
        
        self.start_time = 0.0
        self.current_objectives: List[str] = []
        self.active_safety: List[str] = []
        self.monitoring = True

    def reset(self):
        """Resets the STL monitoring state."""
        self.node_status = {}
        self.start_time = 0.0
        self.monitoring = True
        self._initialize_node(self.root)
        # Root is always active at start
        self._set_active(self.root, 0.0)

    def update(self, obs_dict: Dict, sim_time: float) -> Tuple[bool, float, List[str], List[str]]:
        """
        Updates the STL state based on current observation.
        
        Args:
            obs_dict: The observation dictionary from the environment.
            sim_time: Current simulation time (relative to episode start).
            
        Returns:
            is_violated (bool): True if a hard safety constraint or timeout occurred.
            total_penalty (float): Accumulated penalty for soft violations (if any).
            current_objectives (List[str]): Names of predicates we are currently trying to achieve.
            active_safety (List[str]): Names of predicates we are currently monitoring for safety.
        """
        if not self.monitoring:
            return False, 0.0, [], []

        # 1. Evaluate all predicates (leaves)
        predicate_values = {}
        for name, func in self.predicates.items():
            try:
                predicate_values[name] = func(obs_dict)
            except Exception as e:
                # Handle cases where predicate might fail on partial obs
                predicate_values[name] = False

        # 2. Propagate Logic Bottom-Up (Satisfaction/Violation)
        # We need a post-order traversal or recursive check, but since state depends on history,
        # we iterate or recurse. Let's use a recursive update.
        self._update_node_status(self.root, predicate_values, sim_time)

        # 3. Determine Active Objectives & Safety (Top-Down)
        self.current_objectives = []
        self.active_safety = []
        self._collect_active_nodes(self.root)

        # 4. Check Global Failure (Root violated or Timeout)
        root_state = self.node_status[self.root]
        is_violated = root_state['violated']
        
        # Calculate penalty (placeholder for now, can be sophisticated)
        penalty = -1.0 if is_violated else 0.0

        return is_violated, penalty, self.current_objectives, self.active_safety

    def _initialize_node(self, node: STLNode):
        """Recursively initialize state for all nodes."""
        self.node_status[node] = {
            'active': False,
            'satisfied': False,
            'violated': False,
            'activation_time': None
        }
        for child in node.children:
            self._initialize_node(child)

    def _set_active(self, node: STLNode, time: float):
        """Marks a node as active if not already."""
        state = self.node_status[node]
        if not state['active']:
            state['active'] = True
            state['activation_time'] = time

    def _update_node_status(self, node: STLNode, pred_vals: Dict[str, bool], sim_time: float) -> bool:
        """
        Recursively updates the status of a node.
        Returns True if satisfied.
        """
        state = self.node_status[node]
        
        # If already permanently satisfied/violated, we might stop, 
        # but for 'Always' we must keep checking.
        # For 'Eventually', once satisfied, it stays satisfied.
        
        if isinstance(node, Predicate):
            val = pred_vals.get(node.name, False)
            # Predicates are stateless in themselves, but their satisfaction depends on current val
            state['satisfied'] = val
            state['violated'] = not val
            return val

        elif isinstance(node, And):
            # Satisfied if ALL children are satisfied
            all_sat = True
            any_violated = False
            for child in node.children:
                child_sat = self._update_node_status(child, pred_vals, sim_time)
                if not child_sat:
                    all_sat = False
                if self.node_status[child]['violated']:
                    any_violated = True
            
            state['satisfied'] = all_sat
            state['violated'] = any_violated # Simplification for And
            return all_sat

        elif isinstance(node, Or):
            # Satisfied if ANY child is satisfied
            any_sat = False
            all_violated = True
            for child in node.children:
                child_sat = self._update_node_status(child, pred_vals, sim_time)
                if child_sat:
                    any_sat = True
                if not self.node_status[child]['violated']:
                    all_violated = False
            
            state['satisfied'] = any_sat
            state['violated'] = all_violated
            return any_sat

        elif isinstance(node, Eventually):
            # F[t1, t2](phi)
            # Satisfied if child is satisfied at any point while active within window
            # Violated if time > t2 and not satisfied
            
            if state['satisfied']: 
                return True # Latch satisfaction

            if not state['active']:
                return False

            dt = sim_time - state['activation_time']
            
            # Check timing window start
            if dt < node.t_start:
                return False # Not yet in window

            # Check child
            child_sat = self._update_node_status(node.child, pred_vals, sim_time)
            
            if child_sat:
                state['satisfied'] = True
                return True
            
            # Check timeout
            if dt > node.t_end:
                state['violated'] = True
            
            return False

        elif isinstance(node, Always):
            # G[t1, t2](phi)
            # Satisfied if child is True for the whole duration.
            # Violated if child is False at any point in window.
            
            if state['violated']:
                return False # Latch violation

            if not state['active']:
                return False # Not active yet

            dt = sim_time - state['activation_time']
            
            # If we haven't reached start time, we are fine
            if dt < node.t_start:
                return True 

            # Check child
            child_sat = self._update_node_status(node.child, pred_vals, sim_time)
            
            if not child_sat:
                state['violated'] = True
                return False
            
            # If we passed t_end, and never violated, we are fully satisfied
            if dt > node.t_end:
                state['satisfied'] = True
            
            # While in window and not violated, we are "currently" holding
            return True

        return False

    def _collect_active_nodes(self, node: STLNode, context: str = None):
        """
        Traverses active branches to find what the agent should do/monitor.
        """
        state = self.node_status[node]
        if not state['active']:
            return
            
        # If the node is already satisfied, we don't need to pursue it further.
        if state.get('satisfied'):
            return

        # Update Context based on current node type
        if isinstance(node, Eventually):
            context = 'liveness'
        elif isinstance(node, Always):
            context = 'safety'

        # Handle Predicates
        if isinstance(node, Predicate):
            if context == 'liveness':
                self.current_objectives.append(node.name)
            elif context == 'safety':
                self.active_safety.append(node.name)
            return

        # Handle Recursion
        if isinstance(node, Eventually):
            self._set_active(node.child, state['activation_time'])
            self._collect_active_nodes(node.child, context)

        elif isinstance(node, Always):
            if state['violated']:
                return
            self._set_active(node.child, state['activation_time'])
            self._collect_active_nodes(node.child, context)

        elif isinstance(node, And) or isinstance(node, Or):
            for child in node.children:
                self._set_active(child, state['activation_time'])
                self._collect_active_nodes(child, context)

    def print_status(self):
        """Prints the current status of the STL tree."""
        print(f"--- STL Status ---")
        self._print_node_status(self.root, 0)

    def _print_node_status(self, node: STLNode, level: int):
        indent = "    " * level
        state = self.node_status.get(node, {})
        
        flags = []
        if state.get('active'): flags.append("ACTIVE")
        if state.get('satisfied'): flags.append("SAT")
        if state.get('violated'): flags.append("VIOL")
        
        print(f"{indent}{node.name} {flags}")
        for child in node.children:
            self._print_node_status(child, level + 1)
