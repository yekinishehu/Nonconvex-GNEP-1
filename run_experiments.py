"""Reproduce Figure 1 (runtime scaling) and results_150_instances.csv.

Protocol (Section 8.1): medians of 5 independent runs per instance; per-run
time limit on the monolithic baseline (capped runs recorded as 'capped');
speedups over capped baselines are LOWER BOUNDS and reported as such.
"""
import os, csv, time, json
import numpy as np
from generator import build_instance
from lsc import lsc_solve
from monolithic_vi import monolithic_vi
from compare_gurobi import exact_solve

SIZES   = [10, 15, 25, 40, 60, 80, 100, 120, 150, 200]
TOPOS   = ["EC", "HS", "RMP"]
N_INST  = 3        # instances per node size (increase for final numbers)
RUNS    = 5        # independent runs per instance
TL_MONO = 2500.0   # monolithic time limit, seconds (as in Section 8.1)

def main(out_csv="results/results_150_instances.csv"):
    os.makedirs("results", exist_ok=True)
    rows = []
    for n in SIZES:
        for rep in range(N_INST):
            inst = build_instance(n, TOPOS[rep % 3], seed=1000 * n + rep)
            lsc_times, mono_times, capped = [], [], 0
            for r in range(RUNS):
                t0 = time.time(); _, _, _, obj = lsc_solve(inst)
                lsc_times.append(time.time() - t0)
                z, mt, solved = monolithic_vi(inst, time_limit=TL_MONO)
                mono_times.append(mt); capped += (not solved)
            row = {"instance_id": "%d_%d_%s" % (n, rep, inst["topology"]),
                   "topology": inst["topology"], "n_nodes": n,
                   "n_arcs": len(inst["arcs"]),
                   "lsc_objective": np.mean([obj]),
                   "lsc_time_s": float(np.median(lsc_times)),
                   "path_time_s": float(np.median(mono_times)),
                   "path_solved": int(capped < RUNS),
                   "path_capped_runs": capped,
                   "gurobi_objective": "", "gurobi_time_s": "", "gap_pct_vs_gurobi": ""}
            if n == 30:
                g_obj, g_t, g_opt = exact_solve_json(inst)
                row["gurobi_objective"] = g_obj; row["gurobi_time_s"] = g_t
                if g_obj:
                    row["gap_pct_vs_gurobi"] = 100.0 * (row["lsc_objective"] - g_obj) / abs(g_obj)
            rows.append(row); print(row["instance_id"], "lsc=%.2fs mono=%.1fs" %
                                     (row["lsc_time_s"], row["path_time_s"]))
            with open(out_csv, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=list(rows[0])).writerows(rows)
    # Figure 1
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    med = {}
    for r in rows: med.setdefault(r["n_nodes"], []).append(r)
    ns = sorted(med)
    lsc_y = [np.median([r["lsc_time_s"] for r in med[n]]) for n in ns]
    mo_y  = [np.median([r["path_time_s"] for r in med[n]]) for n in ns]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(ns, lsc_y, "o-", color="#1f5ba8", lw=2, label="LSC")
    ax.plot(ns, mo_y, "s-", color="#b2182b", lw=2, mfc="none", label="Monolithic VI")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Total nodes"); ax.set_ylabel("Runtime (seconds)")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.savefig("results/fig1_runtime_scaling.pdf")
    print("wrote", out_csv, "and results/fig1_runtime_scaling.pdf")

def exact_solve_json(inst):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(inst, fh); path = fh.name
    try:
        obj, rt, opt = exact_solve(path)
        return (obj if opt else None), rt, opt
    finally:
        os.unlink(path)

if __name__ == "__main__":
    main()
