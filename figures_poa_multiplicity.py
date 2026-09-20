"""Analytic reproduction of the PoA phase transition (Theorem 7.21) and of
equilibrium multiplicity (Figure 2 style) on the hub-competition instance.

poa_scan(): exact PoA(f) from the closed forms g, g_rev of Theorem 7.21.
multiplicity(): best-response dynamics from random starts on a 2-warehouse
shared-hub instance; counts distinct limit points as equilibria.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, BETA, C, GAMMA = 8, 1.0, 15.0, 25.0   # gamma > beta*C
def g(K, f):   return BETA * C**2 / (2 * K) + K * f
def grev(K):   return GAMMA * C / K - BETA * C**2 / (2 * K**2)

def poa_scan():
    fs = np.linspace(2, 80, 400); poas = []
    for f in fs:
        Kopt = min(range(1, N + 1), key=lambda K: g(K, f))
        Keq  = max([K for K in range(1, N + 1) if grev(K) >= f], default=0)
        poas.append(g(Keq, f) / g(Kopt, f) if Keq >= 1 else np.nan)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(fs, poas, color="#b2182b", lw=2)
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set_xlabel("Fixed cost $f$"); ax.set_ylabel("Price of Anarchy")
    ax.set_title("Price of Anarchy vs. fixed cost $f$ (analytic, Theorem 7.21)")
    ax.grid(alpha=0.25); fig.savefig("results/fig3_poa_analytic.pdf")
    print("peak PoA = %.3f at f = %.1f" % (np.nanmax(poas), fs[int(np.nanargmax(poas))]))
    print("wrote results/fig3_poa_analytic.pdf")

def multiplicity(n_starts=50):
    # 2 warehouses, one shared hub, capacity C; c_i(x)=x^2/2 + a_i x; elastic demand
    A, Ccap, GAM = np.array([3.0, 4.0]), 15.0, 14.0
    fs = np.linspace(3, 80, 26); counts = []
    rng = np.random.default_rng(0)
    for f in fs:
        limits = set()
        for s in range(n_starts):
            x = rng.uniform(0, Ccap, 2)
            for it in range(500):
                BR = np.clip(GAM - A - x[::-1], 0, None)   # best responses
                x = 0.5 * x + 0.5 * BR
                if np.max(np.abs(x - (0.5 * x + 0.5 * BR))) < 1e-9: break
            prof = GAM * x - (0.5 * x**2 + A * x) - f * (x > 1e-9)
            if np.all(prof >= -1e-9):      # individual rationality
                limits.add(tuple(np.round(x / 1e-4)))
        counts.append(len(limits))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(fs, counts, "o-", color="#1f5ba8", lw=2)
    ax.set_xlabel("Fixed cost $f$"); ax.set_ylabel("Number of distinct GNEs")
    ax.set_title("Equilibrium multiplicity vs. fixed cost $f$")
    ax.grid(alpha=0.25); fig.savefig("results/fig2_multiplicity.pdf")
    print("wrote results/fig2_multiplicity.pdf")

if __name__ == "__main__":
    import os; os.makedirs("results", exist_ok=True)
    poa_scan(); multiplicity()
