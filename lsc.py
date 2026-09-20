"""Layered Sequential Convexification (LSC) -- Section 6 of the manuscript.

Phase I : convex relaxation of the mixed-integer GNEP, solved by echelon
          decomposition: a master loop updates the shared-capacity multipliers
          lambda >= 0; each agent best-responds with an independent convex QP
          (Gurobi), swept in topological (echelon) order because the network
          is acyclic.  Damped iteration until ||Delta x||_inf < tol.
Phase II: integer recovery by ceiling (Lemma 6.4) and continuous flow
          re-resolution at the fixed integers (Theorem 7.6).

Single product.  Requires gurobipy.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

ORDER = ["W", "G", "S", "D", "X"]          # strategic layers, topological
TOL, MAXIT = 1e-6, 4000

def load(path):
    import json
    return json.load(open(path))

def inflow(inst, x, v):
    return sum(x.get("%s|%s" % (u, v), 0.0) for u, vv in inst["arcs"] if vv == v)

def solve_agent(inst, v, x, lam, fixed):
    """Best response of agent v given flows x and capacity prices lam.
    fixed: None (relaxed y,k in [0,1] / [0,kmax]) or ('y','k') fixed values."""
    m = gp.Model("agent_%s" % v)
    m.setParam("OutputFlag", 0)
    out = [(u, w) for u, w in inst["arcs"] if u == v]
    qin = inflow(inst, x, v)
    xv = {}
    for (u, w) in out:
        e = "%s|%s" % (u, w)
        p = inst["param"]
        xv[e] = m.addVar(lb=0.0, ub=p["M"][e], name=e)
        yfix = None if fixed is None else (1.0 if float(xv[e].UB) > 0 else 0.0)
    yv, obj = {}, 0.0
    for (u, w) in out:
        e = "%s|%s" % (u, w)
        p = inst["param"]
        if fixed is None:
            yv[e] = m.addVar(lb=0.0, ub=1.0, name="y_%s" % e)
            obj += p["f"][e] * yv[e]
            m.addConstr(xv[e] <= p["M"][e] * yv[e])
        else:
            m.addConstr(xv[e] <= p["M"][e] * fixed["y"][e])
        obj += p["beta1"][e] * xv[e] * xv[e] + (p["alpha"][e] - p["gamma"][e]) * xv[e]
        if w in inst["hubs"]:                      # priced shared capacity
            obj += lam.get(w, 0.0) * xv[e]
    if v in inst["hubs"]:                          # hub controls expansion k
        h = inst["hubs"][v]
        if fixed is None:
            k = m.addVar(lb=0.0, ub=h["kmax"], name="k_%s" % v)
            obj += h["g_slope"] * k - lam.get(v, 0.0) * h["Delta"] * k
        else:
            k = fixed["k"][v]
    m.setObjective(obj, GRB.MINIMIZE)
    # conservation / supply
    if v in inst["layers"]["C"]:
        m.addConstr(qin >= inst["demand"][v])
    elif v in inst["layers"]["W"]:
        m.addConstr(sum(xv[e] for e in xv) <= inst["supply"][v])
    else:
        m.addConstr(sum(xv[e] for e in xv) == qin)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError("agent QP infeasible at %s" % v)
    nx = {e: xv[e].X for e in xv}
    ny = ({e: yv[e].X for e in yv} if fixed is None
          else {e: fixed["y"][e] for e in xv})
    nk = ({v: k.X} if (v in inst["hubs"] and fixed is None)
          else ({v: fixed["k"][v]} if v in inst["hubs"] else {}))
    return nx, ny, nk

def run_equilibrium(inst, fixed=None, lam0=None, damp=0.7, verbose=False):
    agents = [v for L in ORDER for v in inst["layers"][L]]
    x = {("%s|%s" % e): 0.0 for e in inst["arcs"]}
    y = {("%s|%s" % e): (0.0 if fixed else 1e-3) for e in inst["arcs"]}
    k = {j: (0.0 if fixed else 1e-3) for j in inst["hubs"]}
    lam = dict(lam0 or {j: 0.0 for j in inst["hubs"]})
    for it in range(MAXIT):
        delta = 0.0
        for v in agents:
            nx, ny, nk = solve_agent(inst, v, x, lam, fixed)
            for e in nx:
                xnew = (1 - damp) * x[e] + damp * nx[e]
                delta = max(delta, abs(xnew - x[e]))
                x[e] = xnew
                y[e] = ny[e]
            k.update(nk)
        # master: projected subgradient on capacity residuals
        eta = 2.0 / np.sqrt(it + 1)
        for j, h in inst["hubs"].items():
            res = inflow(inst, x, j) - h["C"] - h["Delta"] * k[j]
            lam[j] = max(0.0, lam[j] + eta * res / max(1.0, h["Delta"]))
        if delta < TOL:
            if verbose: print("  converged it=%d" % it)
            break
    return x, y, k

def ceiling(inst, y, k):
    import math
    yI = {e: float(min(1, math.ceil(max(0.0, min(1.0, y[e]))))) for e in y}
    kI = {j: float(min(inst["hubs"][j]["kmax"], math.ceil(max(0.0, k[j])))) for j in k}
    return yI, kI

def total_cost(inst, x, y, k):
    p = inst["param"]
    c = sum(p["beta1"][e] * x[e] ** 2 + p["alpha"][e] * x[e] + p["f"][e] * y[e]
            - p["gamma"][e] * x[e] for e in x)
    return c + sum(inst["hubs"][j]["g_slope"] * k[j] for j in k)

def lsc_solve(inst, verbose=False):
    """Algorithm 1.  Returns (x^I, y^I, k^I, objective)."""
    x1, y1, k1 = run_equilibrium(inst, fixed=None, verbose=verbose)
    yI, kI = ceiling(inst, y1, k1)
    fixed = {"y": yI, "k": kI}
    x2, _, _ = run_equilibrium(inst, fixed=fixed, verbose=verbose)
    return x2, yI, kI, total_cost(inst, x2, yI, kI)

if __name__ == "__main__":
    import sys, time
    inst = load(sys.argv[1] if len(sys.argv) > 1 else "instances/inst_000_EC.json")
    t0 = time.time()
    xI, yI, kI, obj = lsc_solve(inst, verbose=True)
    print("LSC objective = %.4f   (%d active arcs, %.1f s)"
          % (obj, int(sum(yI.values())), time.time() - t0))
