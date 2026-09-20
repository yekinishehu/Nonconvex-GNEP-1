# Code and Data — "A Non-Convex Generalized Nash Equilibrium for Multi-Echelon Supply Chains"

Companion repository for the EJOR submission by Yekini Shehu and Yonghong Yao.
All code is Python 3. See the main text, Section 8, for the experimental protocol
(solver tolerances 1e-6; per-run time limit for the monolithic PATH baseline as
stated in the paper; AMD Ryzen 9 7950X, 64 GB RAM, Ubuntu 22.04).

## Layout

    README.md                      this file
    LICENSE                        MIT
    requirements.txt               pyomo, sympy, numpy, pandas, matplotlib
                                   (gurobipy and a PATH license are optional but
                                    needed to reproduce the exact comparisons)
    code/
      lsc.py                       Layered Sequential Convexification:
                                   Phase I echelon decomposition (extragradient
                                   master + per-agent convex subproblems via Gurobi),
                                   Phase II ceiling + continuous re-resolution
      monolithic_path.py           monolithic VI baseline (PATH via Pyomo)
      generator.py                 calibrated random instance generator
                                   (topologies EC / HS / RMP; calibration ranges
                                    as in Table 3 of the paper)
      run_experiments.py           reproduces Figures 1-4 and Table 1
      compare_gurobi.py            exact 30-node mixed-integer solves (Gurobi 11.0)
    instances/                     150 JSON instance definitions (topology, node
                                   counts, cost parameters, capacities, seeds)
    results/
      results_150_instances.csv    ONE ROW PER INSTANCE (mandatory per EJOR data
                                   policy). Columns:
                                     instance_id, topology, n_nodes, n_arcs,
                                     lsc_objective, lsc_time_s,
                                     path_time_s  ('capped' if the time limit
                                                   was reached),
                                     path_solved  (0/1),
                                     gurobi_objective, gurobi_time_s,
                                     gap_pct_vs_gurobi   (30-node instances only)
      poa_scan.csv                 per-(instance, fixed-cost) PoA values for
                                   Figure 3, and per-start GNE counts for Figure 2
    verification/
      subgame_verification.ipynb   SymPy notebook verifying all equilibrium values,
                                   the GNE segment { (x1, 15-x1) : x1 in [5,11] },
                                   the PoA ratios of Table 2, and the platform
                                   tolls of Section S.2

## Reproduction

    pip install -r requirements.txt
    python generator.py                 # writes instances/*.json (150 seeds)
    python run_experiments.py           # writes results/*.csv and Figure 1
    python compare_gurobi.py inst.json  # exact 30-node comparison (needs Gurobi)
    python figures_poa_multiplicity.py  # analytic PoA scan (Theorem 7.21) + Fig. 2
    python verification/subgame_verification.py   # 29 exact symbolic checks



## Confidential data

The partner-reported figures in Section 8.3 of the paper are under an NDA and are
therefore not included. The instances/ directory contains the calibrated synthetic
instances used for all headline results.
