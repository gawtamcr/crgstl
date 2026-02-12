class STLNode:
    def robustness(self, state):
        raise NotImplementedError

    def check(self, state):
        return self.robustness(state) > 0
    
    def reset(self):
        pass

class Predicate(STLNode):
    """Returns a robustness value. rho > 0 means satisfied."""
    def __init__(self, name, robustness_fn):
        self.name = name
        self.robustness_fn = robustness_fn
    
    def robustness(self, state):
        return self.robustness_fn(state)

    def __repr__(self):
        return f"Predicate({self.name})"

class Eventually(STLNode):
    """
    Temporal Operator: Returns True if the child condition becomes True 
    and STAYS satisfied (latching logic for sequential tasks).
    """
    def __init__(self, child):
        self.child = child
        self.max_rho = -float('inf')

    def robustness(self, state):
        # rho = max(rho_current, max_rho_history)
        self.max_rho = max(self.max_rho, self.child.robustness(state))
        return self.max_rho
    
    def reset(self):
        self.max_rho = -float('inf')

    def __repr__(self):
        return f"Eventually({self.child})"

class Sequence(STLNode):
    """
    Enforces Order: A -> B. 
    B is only checked if A is already satisfied.
    """
    def __init__(self, steps):
        self.steps = steps # List of STLNodes
        self.current_step = 0

    def robustness(self, state):
        # If all steps done, return True
        if self.current_step >= len(self.steps):
            return 1.0 # Max robustness
        
        # Check current step
        current_node = self.steps[self.current_step]
        rho = current_node.robustness(state)
        
        if rho > 0:
            self.current_step += 1
            
        return rho
    
    def get_active_task(self):
        if self.current_step >= len(self.steps):
            return "DONE"
        return self.steps[self.current_step]

    def reset(self):
        self.current_step = 0
        for step in self.steps:
            step.reset()

    def __repr__(self):
        return f"Sequence(steps={len(self.steps)}, current={self.current_step})"
