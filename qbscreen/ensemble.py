#!/usr/bin/env python3
"""Ensemble & spatial-heterogeneity reservoir computing.

The now-binding question (after readout dimensionality was shown not to cap the
single-reservoir IPC): in vivo, ~10^10 heterogeneous radical-pair copies feed
downstream biology. Does pooling their output into one product flux destroy the
computational capacity (ensemble averaging), and does spatial resolution
(different neurons sampling different sub-populations) recover or multiply it?

Model: N heterogeneous 4-spin reservoirs [e1(input), e2, 31P, 1H]. Each copy has
its own hyperfine couplings, exchange J, and electronic coherence T2e (drawn
from physically motivated distributions). All are driven by the SAME chemical
input sequence; the readout is the time-resolved singlet flux (reaction
kinetics) at fixed lab times — the biologically accessible channel.

Two readouts:
  • pooled      : mean over all copies of each kinetic channel (one product pool)
  • resolved(G) : copies sorted by a parameter into G groups; readout = the
                  concatenated group-mean kinetic channels (G spatial samples)
"""

import numpy as np
from scipy.linalg import expm

from qbscreen.spin_dynamics import SX, SY, SZ, spin_op, singlet_projector, G_E, MU_B, HBAR
from qbscreen.master_equation import build_liouvillian, _vec, electron_dephasing_ops
from qbscreen.reservoir import memory_and_ipc

TWO_PI = 2.0 * np.pi
N = 4
DIM = 2 ** N
D_REST = DIM // 2
ELEC = (0, 1)
NUC = (2, 3)


def _H(A_P, A_H, J, B=50e-6):
    H = np.zeros((DIM, DIM), dtype=complex)
    we = G_E * MU_B * B / (HBAR * 2 * np.pi * 1e6)
    H += we * (spin_op(SZ, 0, N) + spin_op(SZ, 1, N))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for op in (SX, SY, SZ):
            H += A_P * spin_op(op, 0, N) @ spin_op(op, 2, N)
            H += A_H * spin_op(op, 0, N) @ spin_op(op, 3, N)
            if abs(J) > 1e-12:
                H += J * spin_op(op, 0, N) @ spin_op(op, 1, N)
    return H


def _props(A_P, A_H, J, T2e_ns, times_us):
    H = TWO_PI * _H(A_P, A_H, J)
    Z = np.zeros((DIM, DIM))
    deph = electron_dephasing_ops(N, T2e_ns * 1e-3, ELEC)
    L = build_liouvillian(H, Z, Z, 0.0, 0.0, deph)
    with np.errstate(over="ignore", invalid="ignore"):
        return [expm(L * t) for t in times_us]


def _inject(rho, s):
    R = rho.reshape(2, D_REST, 2, D_REST)
    rest = np.trace(R, axis1=0, axis2=2)
    psi = np.array([np.sqrt(1 - s), np.sqrt(s)], dtype=complex)
    return np.kron(np.outer(psi, psi.conj()), rest)


def _run_copy(inputs, props, P_S, washout):
    """Return kinetic feature time-series for one copy: (T, n_times)."""
    rho = singlet_projector(0, 1, N).astype(complex)
    rho = rho / np.trace(rho)
    P_full = props[-1]
    feats = []
    for s in inputs:
        rho = _inject(rho, s)
        v0 = _vec(rho)
        row = []
        for Pj in props:
            rj = (Pj @ v0).reshape(DIM, DIM, order="F")
            tr = np.trace(rj).real
            row.append(np.real(np.trace(P_S @ rj)) / (tr if tr else 1.0))
        rho = (P_full @ v0).reshape(DIM, DIM, order="F")
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho).real
        if tr:
            rho = rho / tr
        feats.append(row)
    return np.array(feats[washout:])


def run(N_copies=80, L=700, n_seeds=5):
    import json
    times = (0.02, 0.035, 0.05)          # fixed lab readout times (µs)
    P_S = singlet_projector(0, 1, N)
    Gs = [1, 2, 4, 8, 16]

    def _metrics(r):
        return np.array([r["MC"], r["IPC_nonlinear"], r["IPC_total"]])

    single_acc, pooled_acc = [], []
    G_acc = {G: [] for G in Gs}
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        s = rng.uniform(0, 1, L + 80); s_post = s[80:]
        params, per_copy = [], []
        for _ in range(N_copies):
            A_P = 80.0 * np.exp(0.35 * rng.standard_normal())
            A_H = 40.0 * np.exp(0.35 * rng.standard_normal())
            J = 2.0 * np.exp(0.6 * rng.standard_normal())
            T2e = 50.0 * np.exp(0.3 * rng.standard_normal())
            params.append((A_P, A_H, J, T2e))
            per_copy.append(_run_copy(s, _props(A_P, A_H, J, T2e, times), P_S, washout=80))
        per_copy = np.array(per_copy); params = np.array(params)
        single_acc.append(_metrics(memory_and_ipc(
            _run_copy(s, _props(80, 40, 2.0, 50.0, times), P_S, washout=80), s_post)))
        pooled_acc.append(_metrics(memory_and_ipc(per_copy.mean(axis=0), s_post)))
        sorted_copies = per_copy[np.argsort(params[:, 2])]
        for G in Gs:
            groups = np.array_split(sorted_copies, G)
            feat = np.concatenate([g.mean(axis=0) for g in groups], axis=1)
            G_acc[G].append(_metrics(memory_and_ipc(feat, s_post)))

    def _avg(lst):
        m = np.mean(lst, axis=0)
        return dict(MC=float(m[0]), IPC_nonlinear=float(m[1]), IPC_total=float(m[2]))
    r_single = _avg(single_acc); r_pooled = _avg(pooled_acc)
    ipc_by_G = [_avg(G_acc[G]) for G in Gs]

    print(f"Ensemble / spatial-heterogeneity reservoir capacity ({n_seeds} seeds)")
    print("=" * 64)
    print(f"  copies = {N_copies}, kinetic channels per sample = {len(times)}")
    print(f"  single homogeneous copy : IPC = {r_single['IPC_total']:.2f}  "
          f"(MC {r_single['MC']:.2f}, nonlin {r_single['IPC_nonlinear']:.2f})")
    print(f"  pooled (one pool, G=1)  : IPC = {r_pooled['IPC_total']:.2f}  "
          f"(MC {r_pooled['MC']:.2f}, nonlin {r_pooled['IPC_nonlinear']:.2f})")
    print("  spatially resolved:")
    for G, r in zip(Gs, ipc_by_G):
        print(f"    G={G:2d} groups ({len(times)*G:2d} channels): "
              f"IPC = {r['IPC_total']:.2f}  (MC {r['MC']:.2f}, nonlin {r['IPC_nonlinear']:.2f})")

    # ── figure ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix"})
    except Exception:
        pass
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax[0].bar([0, 1], [r_single["IPC_total"], r_pooled["IPC_total"]],
              color=["#1565c0", "#c62828"], width=0.6)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["single\ncopy", "pooled\n(G=1)"])
    ax[0].set_ylabel("total IPC")
    ax[0].set_title("(a) pooling preserves capacity", loc="left", fontweight="bold")
    ax[0].grid(True, axis="y", alpha=0.25)

    ch = [len(times) * G for G in Gs]
    ax[1].plot(ch, [r["IPC_total"] for r in ipc_by_G], "o-", c="#2e7d32", ms=4, lw=1.3,
               label="total IPC")
    ax[1].plot(ch, [r["IPC_nonlinear"] for r in ipc_by_G], "s--", c="#7b1fa2", ms=3, lw=1,
               label="nonlinear")
    ax[1].set_xlabel("readout channels (= 3 × #groups)")
    ax[1].set_ylabel("IPC")
    ax[1].set_title("(b) more channels raise the IPC ceiling", loc="left", fontweight="bold")
    ax[1].legend(fontsize=7); ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig("manuscript/figures/fig_ensemble.pdf", bbox_inches="tight")
    fig.savefig("manuscript/figures/fig_ensemble.png", dpi=300, bbox_inches="tight")

    data = dict(single=r_single, pooled=r_pooled,
                resolved=[dict(G=G, channels=len(times) * G, **r) for G, r in zip(Gs, ipc_by_G)])
    with open("simulation_results/ensemble_results.json", "w") as f:
        json.dump(data, f, indent=2, default=float)
    return data


if __name__ == "__main__":
    run()
