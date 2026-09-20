"""Exact 30-node mixed-integer solves (Gurobi 11.0), Table 1 protocol.

Centralized exact optimum: minimize total cost
  sum_e (beta1 x_e^2 + alpha x_e + f_e y_e - gamma_e x_e) + sum_j g_slope k_j
over the joint mixed-integer feasible set (constraints (3)-(6) of the paper).
"""
import time, json
import gurobipy as gp
from gurobipy import GRB

def exact_solve(path, time_limit=3600.0):
    inst = json.load(open(path))
    m = gp.Model("exact"); m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit); m.setParam("MIPGap", 1e-4)
    p = inst["param"]
    x = {("%s|%s" % e): m.addVar(lb=0.0) for e in inst["arcs"]}
    y = {e: m.addVar(vtype=GRB.BINARY) for e in x}
    k = {j: m.addVar(vtype=GRB.INTEGER, lb=0, ub=inst["hubs"][j]["kmax"]) for j in inst["hubs"]}
    for e in x:
        m.addConstr(x[e] <= p["M"][e] * y[e])
    for L, nodes in inst["layers"].items():
        for v in nodes:
            qin = sum(x["%s|%s" % (u, v)] for u, w in inst["arcs"] if w == v)
            out = sum(x["%s|%s" % (u, w)] for u, w in inst["arcs"] if u == v)
            if L == "C":   m.addConstr(qin >= inst["demand"][v])
            elif L == "W": m.addConstr(out <= inst["supply"][v])
            else:          m.addConstr(out == qin)
    for j, h in inst["hubs"].items():
        qin = sum(x["%s|%s" % (u, j)] for u, w in inst["arcs"] if w == j)
        m.addConstr(qin <= h["C"] + h["Delta"] * k[j])
    m.setObjective(sum(p["beta1"][e]*x[e]**2 + p["alpha"][e]*x[e] + p["f"][e]*y[e]
                       - p["gamma"][e]*x[e] for e in x)
                   + sum(inst["hubs"][j]["g_slope"]*k[j] for j in k), GRB.MINIMIZE)
    m.optimize()
    obj = m.ObjBound if m.Status in (GRB.TIME_LIMIT, GRB.INTERRUPTED) else m.ObjVal
    return obj, m.Runtime, m.Status == GRB.OPTIMAL

if __name__ == "__main__":
    import sys
    obj, rt, opt = exact_solve(sys.argv[1])
    print("exact objective = %.4f  (%.1f s, optimal=%s)" % (obj, rt, opt))
