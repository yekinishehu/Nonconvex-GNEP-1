"""Monolithic VI baseline (Section 8.1): the relaxed game's variational
inequality solved on the joint feasible set by the extragradient method of
Korpelevich (1976) -- the same operator the PATH solver treats as an MCP;
PATH (via Pyomo) can be swapped in where a license is available.

Projection onto the joint polyhedron is a QP solved by Gurobi at each step,
which is exactly the per-iteration cost analyzed in Remark 6.6.
Time-capped: returns None when the cap is hit (capped run).
"""
import signal, time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class TimeCap(Exception): pass

def _handler(signum, frame): raise TimeCap()

def _project(inst, z, arcs_idx, hubs_idx, time_left):
    m = gp.Model("proj"); m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", max(1.0, time_left))
    xx = m.addVars(arcs_idx, lb=0.0); yy = m.addVars(arcs_idx, lb=0.0, ub=1.0)
    kk = m.addVars(hubs_idx, lb=0.0)
    for j in hubs_idx:
        m.addConstr(kk[j] <= inst["hubs"][j]["kmax"])
    p = inst["param"]
    for e in arcs_idx:
        m.addConstr(xx[e] <= p["M"][e] * yy[e])
    for L, nodes in inst["layers"].items():
        for v in nodes:
            qin = sum(xx["%s|%s" % (u, v)] for u, w in inst["arcs"] if w == v)
            out = sum(xx["%s|%s" % (u, w)] for u, w in inst["arcs"] if u == v)
            if L == "C":   m.addConstr(qin >= inst["demand"][v])
            elif L == "W": m.addConstr(out <= inst["supply"][v])
            else:          m.addConstr(out == qin)
    for j in hubs_idx:
        h = inst["hubs"][j]
        qin = sum(xx["%s|%s" % (u, j)] for u, w in inst["arcs"] if w == j)
        m.addConstr(qin <= h["C"] + h["Delta"] * kk[j])
    m.setObjective(sum((xx[e] - z[e]) ** 2 for e in arcs_idx)
                   + sum((yy[e] - z["y:" + e]) ** 2 for e in arcs_idx)
                   + sum((kk[j] - z["k:" + j]) ** 2 for j in hubs_idx), GRB.MINIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL): raise TimeCap()
    out = {e: xx[e].X for e in arcs_idx}
    out.update({"y:" + e: yy[e].X for e in arcs_idx})
    out.update({"k:" + j: kk[j].X for j in hubs_idx})
    return out

def pseudo_gradient(inst, z, arcs_idx, hubs_idx):
    p = inst["param"]
    g = {e: 2 * p["beta1"][e] * z[e] + (p["alpha"][e] - p["gamma"][e]) for e in arcs_idx}
    g.update({"y:" + e: p["f"][e] for e in arcs_idx})
    g.update({"k:" + j: inst["hubs"][j]["g_slope"] for j in hubs_idx})
    return g

def monolithic_vi(inst, time_limit=2500.0, eta=10.0, maxit=100000):
    arcs_idx = ["%s|%s" % e for e in inst["arcs"]]
    hubs_idx = list(inst["hubs"])
    z = {e: 0.0 for e in arcs_idx}
    z.update({"y:" + e: 1e-3 for e in arcs_idx})
    z.update({"k:" + j: 1e-3 for j in hubs_idx})
    t0 = time.time()
    signal.signal(signal.SIGALRM, _handler); signal.setitimer(signal.ITIMER_REAL, time_limit)
    try:
        for it in range(maxit):
            tl = time_limit - (time.time() - t0)
            w = _project(inst, {q: z[q] - eta * pseudo_gradient(inst, z, arcs_idx, hubs_idx)[q]
                                for q in z}, arcs_idx, hubs_idx, tl)
            g = pseudo_gradient(inst, w, arcs_idx, hubs_idx)
            znew = _project(inst, {q: z[q] - eta * g[q] for q in z}, arcs_idx, hubs_idx, tl)
            step = max(abs(znew[q] - z[q]) for q in z)
            z = znew
            if step < 1e-6:
                signal.setitimer(signal.ITIMER_REAL, 0)
                return z, time.time() - t0, True
    except TimeCap:
        return None, time_limit, False
    return None, time_limit, False
