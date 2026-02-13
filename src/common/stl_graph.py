import abc
from typing import List, Tuple

class STLNode(abc.ABC):
    """
    Abstract base class for all STL nodes (Predicates, Logic, Temporal).
    """
    def __init__(self, name: str):
        self.name = name
        self.children: List['STLNode'] = []

    @abc.abstractmethod
    def __str__(self):
        """Returns the string representation of the formula."""
        pass

    def print_tree(self, level: int = 0):
        indent = "    " * level
        extra = ""
        if hasattr(self, 't_start') and (self.t_start != 0 or self.t_end != float('inf')):
            extra = f"[{self.t_start},{self.t_end}]"
        print(f"{indent}{self.name}{extra}")
        for child in self.children:
            child.print_tree(level + 1)

class Predicate(STLNode):
    """
    Leaf node representing a logical predicate (e.g., 'grasp', 'approach').
    These will be mapped to actual boolean/float functions in the Conductor later.
    """
    def __init__(self, name: str):
        super().__init__(name)
    
    def __str__(self):
        return self.name

class LogicOp(STLNode):
    """Base class for logical operators (AND, OR)."""
    pass

class And(LogicOp):
    """Logical AND operator (Conjunction)."""
    def __init__(self, *children: STLNode):
        super().__init__("AND")
        self.children = list(children)

    def __str__(self):
        return f"({' & '.join(map(str, self.children))})"

class Or(LogicOp):
    """Logical OR operator (Disjunction)."""
    def __init__(self, *children: STLNode):
        super().__init__("OR")
        self.children = list(children)

    def __str__(self):
        return f"({' | '.join(map(str, self.children))})"

class TemporalOp(STLNode):
    """Base class for temporal operators (F, G) with time intervals."""
    def __init__(self, child: STLNode, t_start: float = 0.0, t_end: float = float('inf')):
        op_name = self.__class__.__name__
        super().__init__(op_name)
        self.children = [child]
        self.t_start = t_start
        self.t_end = t_end

    @property
    def child(self) -> STLNode:
        return self.children[0]

    def _interval_str(self) -> str:
        if self.t_start == 0 and self.t_end == float('inf'):
            return ""
        return f"[{self.t_start},{self.t_end}]"

class Eventually(TemporalOp):
    """
    Eventually (F) operator. 
    Satisfied if child is true at some point within [t_start, t_end].
    """
    def __init__(self, child: STLNode, t_start: float = 0.0, t_end: float = float('inf')):
        super().__init__(child, t_start, t_end)
        self.name = "F"

    def __str__(self):
        return f"F{self._interval_str()}({self.child})"

class Always(TemporalOp):
    """
    Always (G) operator.
    Satisfied if child is true at all times within [t_start, t_end].
    """
    def __init__(self, child: STLNode, t_start: float = 0.0, t_end: float = float('inf')):
        super().__init__(child, t_start, t_end)
        self.name = "G"

    def __str__(self):
        return f"G{self._interval_str()}({self.child})"

# ==========================================
# Verification: Constructing the Target Formula
# ==========================================
if __name__ == "__main__":
    # Target Formula: G(bounds) & F(approach & F(grasp))
    
    print("--- Verifying STL Graph Construction ---")

    # 1. Define Predicates
    p_bounds = Predicate("bounds")
    p_approach = Predicate("approach")
    p_grasp = Predicate("grasp")
    
    # 2. Construct Tree
    
    # Branch 1: Global Safety -> G(bounds)
    # We assume 'Always' implies the whole episode if no interval is given, 
    # or we can specify [0, inf].
    branch_safety = Always(p_bounds) 
    
    # Branch 2: Liveness Task
    # Inner: F(grasp)
    # Let's say grasp must happen within [0, 2.0] relative to when approach finishes
    sub_goal = Eventually(p_grasp, t_start=0, t_end=2.0)
    
    # Sequence: approach & F(grasp)
    seq_logic = And(p_approach, sub_goal)
    
    # Outer: F(approach & ...)
    # The whole sequence must start eventually within [0, 10.0]
    branch_liveness = Eventually(seq_logic, t_start=0, t_end=10.0)
    
    # Top Level: Safety & Liveness
    stl_tree = And(branch_safety, branch_liveness)
    
    # 3. Print Result
    print(f"Constructed Formula: {stl_tree}")
    
    # Expected Output: (G(bounds) & F[0,10.0](approach & F[0,2.0](grasp)))
    # Note: The exact string format depends on the nesting, but the structure is preserved.
