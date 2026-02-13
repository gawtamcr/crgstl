from common.stl_graph import *
from common.stl_conductor import STLConductor

# Setup
p_safe = Predicate("safe")
p_target = Predicate("target")
# Formula: G(safe) & F(target)
tree = And(Always(p_safe), Eventually(p_target, t_end=5.0))

preds = {
    "safe": lambda o: o["x"] < 10,
    "target": lambda o: o["x"] > 5
}

conductor = STLConductor(tree, preds)
conductor.reset()

# Step 1: Safe=True, Target=False
obs = {"x": 0}
viol, pen, objs, safe = conductor.update(obs, 1.0)
print(f"Step 1: V={viol}, Obj={objs}, Safe={safe}")
# Expected: V=False, Obj=['target'], Safe=['safe']

# Step 2: Safe=True, Target=True
obs = {"x": 6}
viol, pen, objs, safe = conductor.update(obs, 2.0)
print(f"Step 2: V={viol}, Obj={objs}, Safe={safe}")
# Expected: V=False, Obj=[], Safe=['safe'] (Target satisfied)
