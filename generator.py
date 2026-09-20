"""Calibrated random instance generator (Section 8.1 of the manuscript).

Topologies: EC (density 0.4), HS (0.2 upstream / 0.6 downstream), RMP (0.3).
Calibration ranges: gamma~U[10,20], alpha~U[2,5], beta1~U[0.01,0.05],
f_e~U[20,100], C_j~U[50,200], Delta_j~U[20,50], k_max=1.  Single product.
Outputs JSON consumed by lsc.py / monolithic_vi.py / compare_gurobi.py.
"""
import json, os
import numpy as np

LAYER_FRACS = {"W": .15, "G": .20, "S": .10, "D": .20, "X": .15, "C": .20}
ADJ = [("W","G"),("G","S"),("G","D"),("G","C"),("S","D"),("D","X"),("D","C"),("X","C")]
UPSTREAM = {("W","G"), ("G","S")}

def _density(topology, ab):
    if topology == "EC":  return 0.4
    if topology == "RMP": return 0.3
    return 0.2 if ab in UPSTREAM else 0.6   # HS

def build_instance(n_nodes, topology="EC", seed=0):
    rng = np.random.default_rng(seed)
    sizes = {k: max(1, int(round(n_nodes*f))) for k, f in LAYER_FRACS.items()}
    sizes["G"] += n_nodes - sum(sizes.values())
    layers = {k: ["%s%d" % (k, i) for i in range(s)] for k, s in sizes.items()}

    arcs = []
    for a, b in ADJ:
        for u in layers[a]:
            for v in layers[b]:
                if rng.random() < _density(topology, (a, b)):
                    arcs.append((u, v))
    # connectivity repair: every non-warehouse node needs an in-arc; every
    # non-customer node needs an out-arc (Assumption 2.2)
    for a, b in ADJ:
        for v in layers[b]:
            if not any(x[1] == v for x in arcs):
                u = layers[a][rng.integers(0, len(layers[a]))]
                arcs.append((u, v))
    for a, b in ADJ:
        for u in layers[a]:
            if not any(x[0] == u for x in arcs):
                v = layers[b][rng.integers(0, len(layers[b]))]
                arcs.append((u, v))
    arcs = sorted(set(arcs))

    demand = {c: float(rng.uniform(5.0, 15.0)) for c in layers["C"]}
    total = sum(demand.values())
    nW = max(1, len(layers["W"]))
    base = total / nW
    supply = {w: float(base * rng.uniform(1.1, 1.4)) for w in layers["W"]}

    def per(pairs):  # per-arc parameters
        return {("%s|%s" % e): float(rng.uniform(lo, hi)) for e in arcs for lo, hi in [pairs]}
    param = {
        "alpha":  per((2.0, 5.0)),
        "beta1":  per((0.01, 0.05)),
        "f":      per((20.0, 100.0)),
        "gamma":  per((10.0, 20.0)),
        "M":      {("%s|%s" % e): 3.0 * total for e in arcs},
    }
    hubs = {}
    for j in layers["G"] + layers["D"]:
        hubs[j] = {"C": float(rng.uniform(50, 200)),
                   "Delta": float(rng.uniform(20, 50)),
                   "kmax": 1, "g_slope": 10.0}
    return {"topology": topology, "n_nodes": n_nodes, "seed": seed,
            "layers": layers, "arcs": [list(e) for e in arcs],
            "demand": demand, "supply": supply, "param": param, "hubs": hubs}

def write_instance(inst, path):
    with open(path, "w") as fh:
        json.dump(inst, fh, indent=1)

if __name__ == "__main__":
    os.makedirs("instances", exist_ok=True)
    topos = ["EC", "HS", "RMP"]
    i = 0
    for n in [30] * 150:
        inst = build_instance(n, topos[i % 3], seed=1000 + i)
        write_instance(inst, "instances/inst_%03d_%s.json" % (i, inst["topology"]))
        i += 1
    print("wrote 150 instances to instances/")
