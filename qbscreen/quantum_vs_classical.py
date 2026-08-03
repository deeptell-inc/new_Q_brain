#!/usr/bin/env python3
"""Is the reservoir capacity quantum or classical?

Decisive test. In a radical pair the singlet<->triplet interconversion that
transduces the input into the readable yield is intrinsically COHERENT spin
precession; there is no classical analogue of S-T mixing. We therefore tune an
all-spin pure-dephasing rate gamma in the Sz product basis from 0 (fully
coherent quantum dynamics) to large (strong dephasing turns coherent flip-flop
into incoherent hopping -> classical Markov chain on Sz product states with
golden-rule rates) and measure the information processing capacity IPC(gamma).

  • IPC collapses as gamma grows  => coherence does the computational work
                                     (genuine quantum reservoir computing)
  • IPC unchanged                 => the capacity is classical (input encoding +
                                     reaction kinetics), 'quantum' unearned
"""

import numpy as np
from scipy.linalg import expm

from qbscreen.spin_dynamics import SX, SY, SZ, spin_op, singlet_projector
from qbscreen.master_equation import build_liouvillian, _vec, electron_dephasing_ops
from qbscreen.reservoir import memory_and_ipc
from qbscreen.ensemble import _H, _inject, N, DIM, ELEC

TWO_PI = 2.0 * np.pi


def _props(A_P, A_H, J, T2e_ns, gamma_deph, times_us):
    """Propagators at given times with baseline electron T2e PLUS a tunable
    all-spin Sz pure-dephasing rate gamma_deph (MHz). gamma_deph -> infinity is
    the classical (incoherent-hopping) limit of the same Hamiltonian."""
    H = TWO_PI * _H(A_P, A_H, J)
    Z = np.zeros((DIM, DIM))
    deph = electron_dephasing_ops(N, T2e_ns * 1e-3, ELEC)
    if gamma_deph > 0:
        deph = deph + [(TWO_PI * gamma_deph, spin_op(SZ, i, N)) for i in range(N)]
    L = build_liouvillian(H, Z, Z, 0.0, 0.0, deph)
    with np.errstate(over="ignore", invalid="ignore"):
        return [expm(L * t) for t in times_us]


def _run(inputs, props, P_S, washout=80):
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


def run(L=800, n_seeds=8):
    import json
    times = (0.02, 0.035, 0.05)
    P_S = singlet_projector(0, 1, N)

    gammas = [0.0, 1.0, 10.0, 50.0, 200.0, 1000.0, 5000.0]
    # precompute propagators once per gamma (gamma-only dependence), reuse across seeds
    props_by_g = {g: _props(80, 40, 2.0, 50.0, g, times) for g in gammas}
    print(f"Quantum vs classical: IPC as coherence is dephased away ({n_seeds} seeds)")
    print("=" * 60)
    print(f"  {'gamma (MHz)':>12} | {'MC':>5} {'nonlin':>7} {'total IPC':>10}")
    print("  " + "-" * 44)
    rows = []
    for g in gammas:
        mc, nl, ipc = [], [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(sd)
            s = rng.uniform(0, 1, L + 80); s_post = s[80:]
            r = memory_and_ipc(_run(s, props_by_g[g], P_S), s_post)
            mc.append(r["MC"]); nl.append(r["IPC_nonlinear"]); ipc.append(r["IPC_total"])
        rows.append(dict(gamma=g, MC=float(np.mean(mc)),
                         IPC_nonlinear=float(np.mean(nl)), IPC_total=float(np.mean(ipc))))
        tag = "  (coherent/quantum)" if g == 0 else ("  (classical limit)" if g >= 1000 else "")
        print(f"  {g:>12.0f} | {rows[-1]['MC']:>5.2f} {rows[-1]['IPC_nonlinear']:>7.2f} "
              f"{rows[-1]['IPC_total']:>10.2f}{tag}")

    ipc_q = rows[0]["IPC_total"]
    ipc_c = rows[-1]["IPC_total"]
    print("  " + "-" * 44)
    print(f"\n  quantum (gamma=0)      : IPC = {ipc_q:.2f}")
    print(f"  classical (gamma->inf) : IPC = {ipc_c:.2f}")
    frac = (ipc_q - ipc_c) / ipc_q if ipc_q else 0.0
    print(f"  coherence contribution : {100*frac:.0f}% of the capacity")
    if frac > 0.3:
        print("  => the spin coherence does substantial computational work: QUANTUM.")
    elif frac > 0.1:
        print("  => coherence contributes modestly; mostly classical.")
    else:
        print("  => capacity is essentially classical; 'quantum' label unearned.")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix"})
    except Exception:
        pass
    g_plot = [max(x["gamma"], 0.3) for x in rows]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.semilogx(g_plot, [x["IPC_total"] for x in rows], "o-", c="#1565c0", ms=4,
                lw=1.4, label="total IPC")
    ax.semilogx(g_plot, [x["IPC_nonlinear"] for x in rows], "s--", c="#7b1fa2", ms=3,
                lw=1.1, label="nonlinear")
    ax.axhline(ipc_c, color="#c62828", ls=":", lw=1, label="classical limit")
    ax.set_xlabel("coherence dephasing rate $\\gamma$ (MHz)")
    ax.set_ylabel("IPC")
    ax.set_title("Quantum contribution to reservoir capacity", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig("manuscript/figures/fig_quantum_vs_classical.pdf", bbox_inches="tight")
    fig.savefig("manuscript/figures/fig_quantum_vs_classical.png", dpi=300, bbox_inches="tight")

    with open("simulation_results/quantum_vs_classical.json", "w") as f:
        json.dump(dict(scan=rows, ipc_quantum=ipc_q, ipc_classical=ipc_c,
                       coherence_fraction=frac), f, indent=2, default=float)
    return rows


if __name__ == "__main__":
    run()
