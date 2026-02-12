import re
from typing import Optional

class RecursiveSTLNode:
    """
    Represents a node in the STL specification tree.
    Parses temporal formulas like F[t_min,t_max](phase & G(safety) & next_formula)
    """
    def __init__(self, stl_string: str):
        self.phase_name: Optional[str] = None
        self.safety_constraint: Optional[str] = None
        self.min_time: float = 0.0
        self.max_time: float = 0.0
        self.next_node: Optional['RecursiveSTLNode'] = None
        self._parse(stl_string.strip())

    def _parse(self, s: str) -> None:
        """Parse STL string into phase components."""
        # Match: F[min,max](phase & G(safety) & next)
        match_f = re.match(r"F\[([\d\.]+),([\d\.]+)\]\s*\(([^&]+)(?:&\s*(.*))?\)", s)
        if match_f:
            t_min, t_max, name, rest = match_f.groups()
            self.min_time = float(t_min)
            self.max_time = float(t_max)
            self.phase_name = name.strip()

            if rest:
                match_g = re.search(r"G\(([^)]+)\)", rest)
                if match_g:
                    self.safety_constraint = match_g.group(1).strip()
                    rest = rest.replace(match_g.group(0), "").strip()
                    if rest.startswith("&"):
                        rest = rest[1:].strip()
                if rest:
                    self.next_node = RecursiveSTLNode(rest)