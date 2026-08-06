#!/usr/bin/env python3
"""Classical-reservoir baselines and a standard benchmark task for the
radical-pair quantum reservoir.

Added to answer the two decisive quantum-information questions a critical PRX
Quantum referee raises:

  (1) Resource advantage — does a *classical* reservoir of equal size match the
      capacity? We compare the 5-spin radical-pair reservoir (Hilbert dimension
      2^5 = 32) against echo-state networks (ESNs) matched either by physical-unit
      count (n_nodes = 5) or by readout dimension (n_nodes = 12, 50).

  (2) Task performance — does it solve a recognised nonlinear temporal benchmark,
      not just abstract IPC? We use NARMA2 (normalised mean-square error).

Capacity is the Dambre IPC computed by the *same* estimator used for the quantum
reservoir (qbscreen.reservoir.memory_and_ipc), so the comparison is apples to
apples. Multi-seed mean +/- std is reported throughout.
"""

import json
import numpy as np

from qbscreen.reservoir import (
    build_reservoir_H, propagator, run_reservoir, _observable_set,
    memory_and_ipc, _capacity, _legendre2,
)

TWO_PI = 2.0 * np.pi


# ───────────────────────── classical echo-state network ─────────────────────
def esn_states(s, n_nodes, seed, spectral_radius=0.9, in_scale=0.6, leak=1.0):
    """Drive a tanh echo-state network; return the (T, n_nodes) state matrix."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_nodes, n_nodes))
    radius = np.max(np.abs(np.linalg.eigvals(W)))
    W *= spectral_radius / radius
    w_in = rng.uniform(-1, 1, n_nodes) * in_scale
    b = rng.uniform(-1, 1, n_nodes) * 0.1
    x = np.zeros(n_nodes)
    X = []
    for sk in s:
        x = (1 - leak) * x + leak * np.tanh(W @ x + w_in * sk + b)
        X.append(x.copy())
    return np.array(X)


# ───────────────────────── higher-order IPC (degree 3) ──────────────────────
def _legendre3(u):
    return (5 * u ** 3 - 3 * u) / 2.0


def ipc_degree3_diag(X, s, max_delay=10):
    """Degree-3 diagonal capacity sum: sum_d C[P3(s_{k-d})]."""
    u = 2 * s - 1.0
    tot = 0.0
    for d in range(0, max_delay + 1):
        if d >= len(X):
            break
        y = _legendre3(np.roll(u, d)); y[:d] = 0.0
        tot += _capacity(X[d:], y[d:])
    return float(tot)


# ───────────────────────── NARMA2 benchmark ─────────────────────────────────
def narma2(u):
    """Standard NARMA2 target for input u in [0, 0.5]."""
    y = np.zeros(len(u))
    for t in range(len(u)):
        ym1 = y[t - 1] if t >= 1 else 0.0
        ym2 = y[t - 2] if t >= 2 else 0.0
        um1 = u[t - 1] if t >= 1 else 0.0
        y[t] = 0.4 * ym1 + 0.4 * ym1 * ym2 + 0.6 * um1 ** 3 + 0.1
    return y


def _nmse_linear(X, y, washout=100, ridge=1e-6):
    """Train a linear readout (with bias) on first half, test on second; NMSE."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    Xb, yt = Xb[washout:], y[washout:]
    h = len(Xb) // 2
    Xtr, ytr, Xte, yte = Xb[:h], yt[:h], Xb[h:], yt[h:]
    G = Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1])
    w = np.linalg.solve(G, Xtr.T @ ytr)
    pred = Xte @ w
    return float(np.mean((pred - yte) ** 2) / np.var(yte))


# ───────────────────────── driver ───────────────────────────────────────────
def coherence_tradeoff(L=900, n_seeds=6):
    """IPC vs electronic coherence time T2e: the reservoir-computing optimum sits
    at T2e ~ tau (fading-memory edge), the OPPOSITE of the sensing optimum (which
    wants T2e as long as possible). Quantifies the sensing/computing trade-off."""
    H = build_reservoir_H(B_tesla=1e-3, A_e1_a=80, A_e1_b=40, A_e2_a=15, J=2.0)
    obs = _observable_set("full")
    T2e_list = [5, 10, 20, 35, 50, 100, 200, 500, 1000, 2000]
    mean, std = [], []
    for T2e in T2e_list:
        vals = []
        for sd in range(n_seeds):
            rng = np.random.default_rng(500 + sd)
            s = rng.uniform(0, 1, L + 100); sp = s[100:]
            P = propagator(H, tau_us=0.05, T2e_ns=float(T2e))
            X = run_reservoir(s, P, obs, nuc_deph_p=0.0, washout=100)
            vals.append(memory_and_ipc(X, sp)["IPC_total"])
        mean.append(float(np.mean(vals))); std.append(float(np.std(vals)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix"})
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.errorbar(T2e_list, mean, yerr=std, fmt="o-", c="#1565c0", ms=4, lw=1.4,
                capsize=2, label="reservoir IPC")
    ax.axvline(50, color="#2e7d32", ls="--", lw=1)
    ax.text(52, min(mean) + 0.1, r"$\tau=50$ ns", color="#2e7d32", fontsize=7.5)
    imax = int(np.argmax(mean))
    ax.annotate("computing\noptimum", xy=(T2e_list[imax], mean[imax]),
                xytext=(120, mean[imax] + 0.15), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(1800, min(mean) - 0.0), xytext=(300, min(mean)),
                arrowprops=dict(arrowstyle="->", color="#c62828"))
    ax.text(330, min(mean) - 0.35, "sensing improves $\\rightarrow$",
            color="#c62828", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"electronic coherence time $T_2^e$ (ns)")
    ax.set_ylabel("total IPC")
    ax.set_title("Optimal coherence for computing vs sensing", fontsize=9)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig("manuscript/figures/fig_tradeoff.pdf", bbox_inches="tight")
    fig.savefig("manuscript/figures/fig_tradeoff.png", dpi=300, bbox_inches="tight")
    print("coherence trade-off:")
    for t, m, sdv in zip(T2e_list, mean, std):
        print(f"  T2e={t:5d} ns : IPC={m:.2f} +/- {sdv:.2f}")
    with open("simulation_results/coherence_tradeoff.json", "w") as f:
        json.dump(dict(T2e_ns=T2e_list, IPC_mean=mean, IPC_std=std), f, indent=2)
    return T2e_list, mean, std


def cryptochrome_reality(L=900, n_seeds=6):
    """Capacity of a *realistic* cryptochrome FAD.-/Trp.+ separated pair at the
    geomagnetic field (J=0, B=50 uT, literature-scale hyperfines), and the physical
    memory horizon (MC * tau) versus biological timescales."""
    obs = _observable_set("full")
    # FAD N5 ~14 MHz, Trp Hbeta ~40 MHz, weaker couplings; J=0; geomagnetic field.
    T2e_list = [100, 200, 500, 1000, 2000, 5000]
    ipc_m, ipc_s, mc_m = [], [], []
    for T2e in T2e_list:
        H = build_reservoir_H(B_tesla=50e-6, A_e1_a=14.0, A_e1_b=5.0, A_e2_a=40.0, J=0.0)
        iv, mv = [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(800 + sd)
            s = rng.uniform(0, 1, L + 100); sp = s[100:]
            P = propagator(H, tau_us=1.0, T2e_ns=float(T2e))
            X = run_reservoir(s, P, obs, nuc_deph_p=0.0, washout=100)
            r = memory_and_ipc(X, sp); iv.append(r["IPC_total"]); mv.append(r["MC"])
        ipc_m.append(float(np.mean(iv))); ipc_s.append(float(np.std(iv)))
        mc_m.append(float(np.mean(mv)))

    tau_s = 1e-6                                  # 1 us radical-pair lifetime
    t_mem = float(np.max(mc_m)) * tau_s           # memory horizon (s) = MC * tau

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix"})
    except Exception:
        pass
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax[0].errorbar(T2e_list, ipc_m, yerr=ipc_s, fmt="o-", c="#1565c0", ms=4, lw=1.4, capsize=2)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$T_2^e$ (ns)"); ax[0].set_ylabel("total IPC (full readout)")
    ax[0].set_title("(a) real cryptochrome @ 50 $\\mu$T", loc="left", fontweight="bold", fontsize=9)
    ax[0].set_ylim(0, max(ipc_m) + 1.5); ax[0].grid(True, which="both", alpha=0.25)

    names = ["reservoir\nmemory", "synaptic\n(~10 ms)", "spike train\n(~100 ms)", "behaviour\n(~1 s)"]
    times = [t_mem, 1e-2, 1e-1, 1e0]
    ax[1].bar(range(4), times, color=["#1565c0", "#c62828", "#c62828", "#c62828"], width=0.62)
    ax[1].set_yscale("log"); ax[1].set_ylabel("timescale (s)")
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names, fontsize=7)
    ax[1].set_title("(b) the timescale gap", loc="left", fontweight="bold", fontsize=9)
    ax[1].grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig("manuscript/figures/fig_biology.pdf", bbox_inches="tight")
    fig.savefig("manuscript/figures/fig_biology.png", dpi=300, bbox_inches="tight")
    print("cryptochrome-realistic capacity (B=50uT, J=0, tau=1us):")
    for t, m, sdv in zip(T2e_list, ipc_m, ipc_s):
        print(f"  T2e={t:5d} ns : IPC={m:.2f} +/- {sdv:.2f}")
    print(f"  memory horizon = MC*tau = {t_mem*1e6:.1f} us  (vs synaptic ~10 ms: gap ~{1e-2/t_mem:.0e})")
    with open("simulation_results/cryptochrome_reality.json", "w") as f:
        json.dump(dict(T2e_ns=T2e_list, IPC_mean=ipc_m, IPC_std=ipc_s, MC_mean=mc_m,
                       memory_horizon_s=t_mem), f, indent=2)
    return ipc_m, t_mem


def run(L=1200, n_seeds=10):
    # quantum reservoir operating point (engineered, full 12-observable readout).
    # Input seeds 0..n_seeds-1 match qbscreen.reservoir.full_analysis exactly, so
    # the headline IPC, the n_obs-scan figure, and this baseline share one basis.
    H = build_reservoir_H(B_tesla=1e-3, A_e1_a=80, A_e1_b=40, A_e2_a=15, J=2.0)
    P = propagator(H, tau_us=0.05, T2e_ns=50.0)
    obs_full = _observable_set("full")

    # ── (A) multi-seed IPC: quantum vs ESN at matched unit / readout counts ──
    q_ipc, q_d3 = [], []
    esn_ipc = {5: [], 12: [], 50: []}
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        s = rng.uniform(0, 1, L + 100)
        s_post = s[100:]
        Xq = run_reservoir(s, P, obs_full, nuc_deph_p=0.0, washout=100)
        rq = memory_and_ipc(Xq, s_post)
        q_ipc.append(rq["IPC_total"])
        q_d3.append(ipc_degree3_diag(Xq, s_post))
        for n in esn_ipc:
            Xe = esn_states(s, n, seed=200 + sd)[100:]
            esn_ipc[n].append(memory_and_ipc(Xe, s_post)["IPC_total"])

    def ms(a):
        return float(np.mean(a)), float(np.std(a))

    # ── (B) NARMA2: quantum (12 obs) vs ESN(5), ESN(50) ──
    narma = {"quantum": [], "esn5": [], "esn50": []}
    for sd in range(n_seeds):
        rng = np.random.default_rng(300 + sd)
        u = rng.uniform(0, 0.5, L + 100)
        y = narma2(u)
        Xq = run_reservoir(u, P, obs_full, nuc_deph_p=0.0, washout=0)
        narma["quantum"].append(_nmse_linear(Xq, y))
        narma["esn5"].append(_nmse_linear(esn_states(u, 5, seed=400 + sd), y))
        narma["esn50"].append(_nmse_linear(esn_states(u, 50, seed=400 + sd), y))

    out = {
        "quantum_5spin_dim32": {
            "n_obs": 12, "IPC_mean": ms(q_ipc)[0], "IPC_std": ms(q_ipc)[1],
            "IPC_deg3_diag_mean": ms(q_d3)[0],
        },
        "esn": {str(n): {"n_nodes": n, "IPC_mean": ms(v)[0], "IPC_std": ms(v)[1]}
                for n, v in esn_ipc.items()},
        "narma2_nmse": {k: {"mean": ms(v)[0], "std": ms(v)[1]} for k, v in narma.items()},
        "n_seeds": n_seeds,
    }

    print("Quantum-information baselines for the radical-pair reservoir")
    print("=" * 64)
    print(f"  quantum 5-spin (dim 32), 12 obs : IPC = {out['quantum_5spin_dim32']['IPC_mean']:.2f}"
          f" +/- {out['quantum_5spin_dim32']['IPC_std']:.2f}"
          f"  (deg-3 diag {out['quantum_5spin_dim32']['IPC_deg3_diag_mean']:.2f})")
    for n in (5, 12, 50):
        e = out["esn"][str(n)]
        print(f"  classical ESN, {n:2d} nodes        : IPC = {e['IPC_mean']:.2f} +/- {e['IPC_std']:.2f}")
    print("  NARMA2 NMSE (lower is better):")
    for k in ("quantum", "esn5", "esn50"):
        print(f"    {k:8s}: {out['narma2_nmse'][k]['mean']:.3f} +/- {out['narma2_nmse'][k]['std']:.3f}")

    with open("simulation_results/qrc_benchmarks.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ── figure: capacity per physical unit + NARMA2 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix"})
    except Exception:
        pass
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    labels = ["quantum\n5 spins\n(dim 32)", "ESN\n5 nodes", "ESN\n12 nodes", "ESN\n50 nodes"]
    vals = [out["quantum_5spin_dim32"]["IPC_mean"], out["esn"]["5"]["IPC_mean"],
            out["esn"]["12"]["IPC_mean"], out["esn"]["50"]["IPC_mean"]]
    errs = [out["quantum_5spin_dim32"]["IPC_std"], out["esn"]["5"]["IPC_std"],
            out["esn"]["12"]["IPC_std"], out["esn"]["50"]["IPC_std"]]
    cols = ["#1565c0", "#c62828", "#c62828", "#c62828"]
    ax[0].bar(range(4), vals, yerr=errs, color=cols, width=0.62, capsize=3)
    ax[0].axhline(5, color="#555", ls="--", lw=0.8)
    ax[0].text(0.05, 5.2, "physical-unit count = 5", fontsize=6.5, color="#555")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_ylabel("total IPC")
    ax[0].set_title("(a) capacity per physical unit", loc="left", fontweight="bold", fontsize=9)
    ax[0].grid(True, axis="y", alpha=0.25)

    nb = ["quantum", "esn5", "esn50"]
    nv = [out["narma2_nmse"][k]["mean"] for k in nb]
    ne = [out["narma2_nmse"][k]["std"] for k in nb]
    ax[1].bar(range(3), nv, yerr=ne, color=["#1565c0", "#c62828", "#7b1fa2"], width=0.6, capsize=3)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(["quantum\n(5 spins)", "ESN\n(5 nodes)", "ESN\n(50 nodes)"], fontsize=7)
    ax[1].set_ylabel("NARMA2 NMSE (lower better)")
    ax[1].set_title("(b) nonlinear benchmark", loc="left", fontweight="bold", fontsize=9)
    ax[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("manuscript/figures/fig_baseline.pdf", bbox_inches="tight")
    fig.savefig("manuscript/figures/fig_baseline.png", dpi=300, bbox_inches="tight")
    return out


if __name__ == "__main__":
    # run() alone left coherence_tradeoff.json and cryptochrome_reality.json
    # unproduced by any documented command, so a clean-room regeneration could
    # not recreate them. Dispatch on the argument, as semiclassical.py does.
    import sys
    which = sys.argv[1:] or ["run"]
    for w in which:
        {"run": run,
         "tradeoff": coherence_tradeoff,
         "cryptochrome": cryptochrome_reality}[w]()
