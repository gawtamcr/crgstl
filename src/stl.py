class STLNode:
    def check(self, state):
        raise NotImplementedError
    
    def reset(self):
        pass

class Predicate(STLNode):
    """Checks a specific condition in the environment (e.g., 'is_holding')."""
    def __init__(self, name, check_fn):
        self.name = name
        self.check_fn = check_fn
    
    def check(self, state):
        return self.check_fn(state)

class Eventually(STLNode):
    """
    Temporal Operator: Returns True if the child condition becomes True 
    and STAYS satisfied (latching logic for sequential tasks).
    """
    def __init__(self, child):
        self.child = child
        self.satisfied = False

    def check(self, state):
        if self.satisfied:
            return True
        if self.child.check(state):
            self.satisfied = True
            return True
        return False
    
    def reset(self):
        self.satisfied = False

class Sequence(STLNode):
    """
    Enforces Order: A -> B. 
    B is only checked if A is already satisfied.
    """
    def __init__(self, steps):
        self.steps = steps # List of STLNodes
        self.current_step = 0

    def check(self, state):
        # If all steps done, return True
        if self.current_step >= len(self.steps):
            return True
        
        # Check current step
        current_node = self.steps[self.current_step]
        if current_node.check(state):
            print(f"DEBUG: Logic Step {self.current_step} ({type(current_node).__name__}) Complete!")
            self.current_step += 1
            return False # Not fully done with sequence yet
        
        return False
    
    def get_active_task(self):
        if self.current_step >= len(self.steps):
            return "DONE"
        return self.steps[self.current_step]

    def reset(self):
        self.current_step = 0
        for step in self.steps:
            step.reset()