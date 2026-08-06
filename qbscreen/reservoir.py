#!/usr/bin/env python3
"""Quantum reservoir computing on a radical-pair + nuclear-spin system.

Tests, quantitatively, whether cryptochrome-type spin chemistry can serve as a
quantum reservoir — distinct from (and a better-posed question than) gate-model
quantum computing, because reservoir computing *requires* dissipation to provide
the fading-memory / echo-state property.

Physical picture:
  • nuclear spins (long T2)      → persistent reservoir memory nodes
  • radical-pair electron        → nonlinear input/output 'read–write head'
  • hyperfine coupling           → the read/write channel
  • electron dephasing (T2e) +
    inter-cycle nuclear dephasing → the fading memory (echo-state property)

Protocol (standard temporal QRC, Fujii–Nakajima 2017 adapted):
  Each timestep k, the input electron (spin 0) is reset to the input-encoded
  state |ψ(s_k)⟩ = √(1−s_k)|0⟩ + √(s_k)|1⟩ while the rest of the system (the
  reservoir memory: electron 2 + nuclei) is retained; the full state then
  evolves for a fixed RP lifetime τ under the master-equation propagator
  exp(Lτ) [L = −i[H,·] + electronic T2e dephasing]; a set of observables is
  read out and a *linear* readout is trained on them.

Benchmarks:
  • Memory capacity MC = Σ_d r²(s_{k−d})                     (Jaeger 2002)
  • Information processing capacity IPC (Dambre et al. 2012): linear (=MC) plus
    nonlinear (degree-2) capacities; total ≤ number of independent observables.

Two regimes are compared:
  • 'engineered'  — full multi-observable (NMR/EPR-style) readout, τ tuned for
                    effective hyperfine writing, nuclei retained between steps.
  • 'in-vivo'     — only the scalar singlet-yield readout is accessible, and
                    strong inter-cycle nuclear dephasing models the hostile
                    throughput (RP events sparse vs nuclear T2).
"""

import numpy as np
from scipy.linalg import expm

from qbscreen.spin_dynamics import (
    SX, SY, SZ, I2, spin_op, singlet_projector, G_E, MU_B, HBAR,
)
from qbscreen.master_equation import build_liouvillian, _vec, electron_dephasing_ops

TWO_PI = 2.0 * np.pi


# ───────────────────────────────────────────────────────────────────
# Reservoir Hamiltonian: [e1(input), e2, n1=³¹P, n2=¹H, n3=¹H]
# ───────────────────────────────────────────────────────────────────
N_SPINS = 5
DIM = 2 ** N_SPINS
ELECTRONS = (0, 1)
NUCLEI = (2, 3, 4)


def build_reservoir_H(B_tesla=50e-6, A_e1_a=200.0, A_e1_b=20.0,
                      A_e2_a=10.0, J=0.5):
    """Reservoir spin Hamiltonian (MHz) for a *separated* radical pair.

    Topology (each radical carries its OWN nuclei — they are not shared, which is
    what "spatially separated pair" means):

        radical 1 (e1)  --A_e1_a-->  nucleus 1 (spin index 2)
                        --A_e1_b-->  nucleus 2 (spin index 3)
        radical 2 (e2)  --A_e2_a-->  nucleus 3 (spin index 4)
        e1 <--J--> e2  (exchange; J = 0 for the separated pair)

    NOTE (correction): earlier versions of this function accepted a fourth
    hyperfine argument ``A_H2`` that was never added to H — it silently had no
    effect on any result.  It has been removed rather than silently ignored.  A
    second coupling on e2 would require a fourth nucleus (a 6-spin model).
    """
    n = N_SPINS
    H = np.zeros((DIM, DIM), dtype=complex)
    omega_e = G_E * MU_B * B_tesla / (HBAR * 2 * np.pi * 1e6)
    H += omega_e * (spin_op(SZ, 0, n) + spin_op(SZ, 1, n))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for op in (SX, SY, SZ):
            H += A_e1_a * spin_op(op, 0, n) @ spin_op(op, 2, n)   # e1–nucleus 1
            H += A_e1_b * spin_op(op, 0, n) @ spin_op(op, 3, n)   # e1–nucleus 2
            H += A_e2_a * spin_op(op, 1, n) @ spin_op(op, 4, n)   # e2–nucleus 3
            if abs(J) > 1e-12:
                H += J * spin_op(op, 0, n) @ spin_op(op, 1, n)
    return H


def propagator(H_mhz, tau_us, T2e_ns):
    """Fixed CPTP propagator exp(Lτ) on vec(ρ); L = −i[H,·] + electron dephasing."""
    H = TWO_PI * H_mhz  # rad/µs
    dim = H.shape[0]
    Pdummy = np.zeros((dim, dim))           # no recombination in the step
    deph = electron_dephasing_ops(N_SPINS, None if T2e_ns is None else T2e_ns * 1e-3,
                                  electron_indices=ELECTRONS)
    L = build_liouvillian(H, Pdummy, Pdummy, 0.0, 0.0, deph)
    with np.errstate(over="ignore", invalid="ignore"):
        return expm(L * tau_us)


# ───────────────────────────────────────────────────────────────────
# Subsystem utilities (input spin = subsystem 0, dim 2; rest dim DIM/2)
# ───────────────────────────────────────────────────────────────────

def _reduced_rest(rho):
    """Trace out the input electron (spin 0); return reduced state of the rest."""
    d_rest = DIM // 2
    R = rho.reshape(2, d_rest, 2, d_rest)
    return np.einsum("aibj->ij", R.reshape(2, d_rest, 2, d_rest)) \
        if False else np.trace(R, axis1=0, axis2=2)


def _inject(rho, s):
    """Reset input electron to |ψ(s)⟩=√(1−s)|0⟩+√s|1⟩, keep the reservoir."""
    rest = _reduced_rest(rho)                       # (d_rest, d_rest)
    psi = np.array([np.sqrt(1 - s), np.sqrt(s)], dtype=complex)
    rho_in = np.outer(psi, psi.conj())              # (2,2)
    return np.kron(rho_in, rest)                     # (DIM, DIM)


def _nuclear_dephase(rho, p):
    """Fully-dephasing channel on each nucleus with probability p (fading memory
    from sparse RP throughput vs nuclear T2). p=0 → none; p=1 → nuclei reset to
    diagonal in z each step."""
    if p <= 0:
        return rho
    out = rho.copy()
    for i in NUCLEI:
        Zi = spin_op(2 * SZ, i, N_SPINS)            # Pauli Z (eigenvalues ±1)
        out = (1 - p) * out + p * 0.5 * (out + Zi @ out @ Zi)
    return out


# ── Observables ────────────────────────────────────────────────────
def _observable_set(kind="full"):
    """Return list of (name, operator). 'full' = single-spin Paulis on nuclei +
    e1–e2 singlet; 'scalar' = singlet yield only (the in-vivo readout)."""
    P_S = singlet_projector(0, 1, N_SPINS)
    if kind == "scalar":
        return [("P_S", P_S)]
    ops = []
    for i in NUCLEI:
        ops.append((f"Z{i}", spin_op(2 * SZ, i, N_SPINS)))
        ops.append((f"X{i}", spin_op(2 * SX, i, N_SPINS)))
        ops.append((f"Y{i}", spin_op(2 * SY, i, N_SPINS)))
    ops.append(("Ze1", spin_op(2 * SZ, 0, N_SPINS)))
    ops.append(("Ze2", spin_op(2 * SZ, 1, N_SPINS)))
    ops.append(("P_S", P_S))
    return ops


def run_reservoir(inputs, P, observables, nuc_deph_p=0.0, washout=100):
    """Drive the reservoir with an input sequence; return feature matrix X."""
    # initial reservoir state: maximally mixed nuclei + singlet electrons
    rho = singlet_projector(0, 1, N_SPINS).astype(complex)
    rho = rho / np.trace(rho)
    Pmat = P
    feats = []
    for k, s in enumerate(inputs):
        rho = _inject(rho, s)
        rho = _nuclear_dephase(rho, nuc_deph_p)
        v = Pmat @ _vec(rho)
        rho = v.reshape(DIM, DIM, order="F")
        rho = 0.5 * (rho + rho.conj().T)            # hermitise (num. hygiene)
        tr = np.trace(rho).real
        if tr != 0:
            rho = rho / tr
        x = [np.real(np.trace(O @ rho)) for _, O in observables]
        feats.append(x)
    X = np.array(feats[washout:])
    return X


# ───────────────────────────────────────────────────────────────────
# Capacity measures
# ───────────────────────────────────────────────────────────────────
def _capacity(X, target, ridge=1e-6):
    """Held-out reconstruction capacity. Fit a ridge readout on the first half of
    the samples and evaluate the normalised squared error (1 − NMSE) on the
    held-out second half. This is an out-of-sample squared multiple correlation,
    bounded [0,1]; it does not reward in-sample overfitting (unlike a fit-and-score
    on the same data, which inflates capacity as the readout dimension grows)."""
    n = len(X)
    if n < 8:
        return 0.0
    h = n // 2
    Xtr, Xte = X[:h], X[h:]
    ytr, yte = target[:h], target[h:]
    mu_x = Xtr.mean(0)
    mu_y = ytr.mean()
    Xtr_c = Xtr - mu_x
    G = Xtr_c.T @ Xtr_c + ridge * np.eye(Xtr_c.shape[1])
    w = np.linalg.solve(G, Xtr_c.T @ (ytr - mu_y))
    pred = (Xte - mu_x) @ w
    ss_res = np.sum(((yte - mu_y) - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    return max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _legendre2(u):
    """Normalised degree-2 Legendre on u∈[-1,1]: P2 = (3u²−1)/2 (scaled)."""
    return (3 * u ** 2 - 1) / 2.0


def memory_and_ipc(X, s, max_delay=14):
    """Return dict with linear memory capacity and degree-2 nonlinear capacity."""
    # use centred inputs in [-1,1] for Legendre. Each capacity term is evaluated
    # out-of-sample by _capacity (train on first half, score on held-out half).
    u = 2 * s - 1.0
    lin = []
    for d in range(0, max_delay + 1):
        if d >= len(X):
            lin.append(0.0)
            continue
        y = np.roll(s, d)
        y[:d] = s[0]
        c = _capacity(X[d:], y[d:])
        lin.append(c)
    MC = float(np.sum(lin))

    # degree-2 diagonal: P2(s_{k-d})
    quad_diag = []
    for d in range(0, max_delay + 1):
        if d >= len(X):
            quad_diag.append(0.0); continue
        y = _legendre2(np.roll(u, d)); y[:d] = 0.0
        quad_diag.append(_capacity(X[d:], y[d:]))
    # degree-2 cross: s_{k-d1} s_{k-d2}
    quad_cross = 0.0
    for d1 in range(0, min(6, max_delay)):
        for d2 in range(d1 + 1, min(6, max_delay) + 1):
            y = np.roll(u, d1) * np.roll(u, d2)
            y[:d2] = 0.0
            quad_cross += _capacity(X[d2:], y[d2:])

    IPC2 = float(np.sum(quad_diag)) + float(quad_cross)
    return dict(MC=MC, lin_by_delay=lin, IPC_nonlinear=IPC2,
                IPC_total=MC + IPC2, n_obs=X.shape[1])


def run_reservoir_realistic(inputs, H_mhz, tau_us, T2e_ns, kinetic_times=(0.2, 0.4, 0.6, 0.8, 1.0),
                            use_cidnp=True, washout=80):
    """In-vivo-plausible multichannel readout: time-resolved singlet flux
    (reaction kinetics) + CIDNP nuclear polarisation on the products.

    Channels are all PHYSICALLY accessible to downstream biology without any
    NMR/EPR apparatus: the kinetic channels are the time course of one product
    pool; the CIDNP channels are the nuclear-spin polarisation imprinted on the
    diamagnetic product (which spin-dependent downstream chemistry could read)."""
    P_S = singlet_projector(0, 1, N_SPINS)
    # propagators at the requested fractional times of tau
    props = [propagator(H_mhz, tau_us * f, T2e_ns) for f in kinetic_times]
    P_full = props[-1]
    cidnp_ops = [spin_op(2 * SZ, i, N_SPINS) for i in NUCLEI] if use_cidnp else []

    rho = singlet_projector(0, 1, N_SPINS).astype(complex)
    rho = rho / np.trace(rho)
    feats = []
    for s in inputs:
        rho = _inject(rho, s)
        v0 = _vec(rho)
        row = []
        # time-resolved singlet flux (one product pool, read at several times)
        for Pj in props:
            rj = (Pj @ v0).reshape(DIM, DIM, order="F")
            tr = np.trace(rj).real
            row.append(np.real(np.trace(P_S @ rj)) / (tr if tr else 1.0))
        # carry forward the full-tau state
        rho = (P_full @ v0).reshape(DIM, DIM, order="F")
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho).real
        if tr:
            rho = rho / tr
        # CIDNP: nuclear polarisation on the product
        for O in cidnp_ops:
            row.append(np.real(np.trace(O @ rho)))
        feats.append(row)
    return np.array(feats[washout:])


def readout_realism(L=900, n_seeds=10):
    """Quantify IPC recovered by physically-accessible in-vivo readout channels,
    averaged over n_seeds input realisations (mean +/- s.d.)."""
    import json
    H = build_reservoir_H(B_tesla=1e-3, A_e1_a=80, A_e1_b=40, A_e2_a=15, J=2.0)
    kt5 = (0.2, 0.4, 0.6, 0.8, 1.0)
    configs = {"endpoint_only": dict(kinetic_times=(1.0,), use_cidnp=False),
               "kinetic": dict(kinetic_times=kt5, use_cidnp=False),
               "kinetic_plus_cidnp": dict(kinetic_times=kt5, use_cidnp=True)}
    acc = {k: {"MC": [], "IPC_nonlinear": [], "IPC_total": [], "n_obs": 0} for k in configs}
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        s = rng.uniform(0, 1, L + 80); s_post = s[80:]
        for key, cfg in configs.items():
            X = run_reservoir_realistic(s, H, 0.05, 50.0, **cfg)
            r = memory_and_ipc(X, s_post)
            acc[key]["MC"].append(r["MC"]); acc[key]["IPC_nonlinear"].append(r["IPC_nonlinear"])
            acc[key]["IPC_total"].append(r["IPC_total"]); acc[key]["n_obs"] = r["n_obs"]

    out = {}
    print("\nIn-vivo-plausible readout channels (NO NMR/EPR apparatus); "
          f"{n_seeds}-seed mean +/- s.d.:")
    for key, a in acc.items():
        out[key] = dict(n_obs=a["n_obs"],
                        MC=float(np.mean(a["MC"])),
                        IPC_nonlinear=float(np.mean(a["IPC_nonlinear"])),
                        IPC_total=float(np.mean(a["IPC_total"])),
                        IPC_total_sd=float(np.std(a["IPC_total"])))
        print(f"  {key:22s}: {out[key]['n_obs']:2d} obs  MC={out[key]['MC']:.2f}  "
              f"nonlin={out[key]['IPC_nonlinear']:.2f}  "
              f"total IPC={out[key]['IPC_total']:.2f}+/-{out[key]['IPC_total_sd']:.2f}")
    with open("simulation_results/reservoir_readout_realism.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def full_analysis(L=1200, n_seeds=10):
    """Two regimes + readout-access and throughput scans + figure + JSON.
    All quantities are averaged over n_seeds input realisations so the figure and
    the reported headline come from the same robust (out-of-sample) basis."""
    import json

    H = build_reservoir_H(B_tesla=1e-3, A_e1_a=80, A_e1_b=40, A_e2_a=15, J=2.0)
    P = propagator(H, tau_us=0.05, T2e_ns=50.0)
    obs_full = _observable_set("full")
    obs_scalar = _observable_set("scalar")
    n_obs_list = [1, 2, 4, 6, 8, 10, 12]
    p_list = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    eng_mc, eng_ipc, eng_nl, kernels = [], [], [], []
    iv_ipc, iv_kernels = [], []
    obs_scan, p_scan = [], []
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        s = rng.uniform(0, 1, L + 100)
        s_post = s[100:]
        X_eng = run_reservoir(s, P, obs_full, nuc_deph_p=0.0)
        re = memory_and_ipc(X_eng, s_post)
        eng_mc.append(re["MC"]); eng_ipc.append(re["IPC_total"])
        eng_nl.append(re["IPC_nonlinear"]); kernels.append(re["lin_by_delay"])
        X_iv = run_reservoir(s, P, obs_scalar, nuc_deph_p=0.6)
        ri = memory_and_ipc(X_iv, s_post)
        iv_ipc.append(ri["IPC_total"]); iv_kernels.append(ri["lin_by_delay"])
        obs_scan.append([memory_and_ipc(X_eng[:, :m], s_post)["IPC_total"] for m in n_obs_list])
        p_scan.append([memory_and_ipc(run_reservoir(s, P, obs_full, nuc_deph_p=p),
                                      s_post)["IPC_total"] for p in p_list])

    MC = float(np.mean(eng_mc)); IPC = float(np.mean(eng_ipc))
    IPC_sd = float(np.std(eng_ipc)); NL = float(np.mean(eng_nl))
    kernel = np.mean(kernels, axis=0); iv_kernel = np.mean(iv_kernels, axis=0)
    iv = float(np.mean(iv_ipc))
    ipc_vs_obs = list(np.mean(obs_scan, axis=0))
    ipc_vs_p = list(np.mean(p_scan, axis=0))
    r_eng = dict(MC=MC, IPC_total=IPC, IPC_sd=IPC_sd, IPC_nonlinear=NL,
                 lin_by_delay=list(kernel), n_obs=len(obs_full))
    r_iv = dict(IPC_total=iv, lin_by_delay=list(iv_kernel))
    print(f"[full_analysis] engineered IPC={IPC:.2f}+/-{IPC_sd:.2f} MC={MC:.2f} "
          f"nonlin={NL:.2f}; in-vivo scalar IPC={iv:.2f} ({n_seeds} seeds)")
    print("  n_obs scan:", [round(x, 2) for x in ipc_vs_obs])
    print("  throughput scan:", [round(x, 2) for x in ipc_vs_p])

    # ── Figure ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix"})
    except Exception:
        pass
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))

    d = np.arange(len(r_eng["lin_by_delay"]))
    ax[0].plot(d, r_eng["lin_by_delay"], "o-", c="#1565c0", ms=3, lw=1.2,
               label="engineered (12 obs)")
    ax[0].plot(d, r_iv["lin_by_delay"], "s--", c="#c62828", ms=3, lw=1.2,
               label="in-vivo (scalar)")
    ax[0].set_xlabel("delay $d$ (steps)"); ax[0].set_ylabel("memory $r^2(s_{k-d})$")
    ax[0].set_title("(a) linear memory", loc="left", fontweight="bold")
    ax[0].legend(fontsize=6.5); ax[0].grid(True, alpha=0.25)

    ax[1].plot(n_obs_list, ipc_vs_obs, "o-", c="#2e7d32", ms=4, lw=1.3)
    ax[1].axhline(r_iv["IPC_total"], color="#c62828", ls=":", lw=1,
                  label="in-vivo (1 obs)")
    ax[1].set_xlabel("# readout observables"); ax[1].set_ylabel("total IPC")
    ax[1].set_title("(b) readout access", loc="left", fontweight="bold")
    ax[1].legend(fontsize=6.5); ax[1].grid(True, alpha=0.25)

    ax[2].plot(p_list, ipc_vs_p, "o-", c="#7b1fa2", ms=4, lw=1.3)
    ax[2].set_xlabel("inter-cycle nuclear dephasing $p$")
    ax[2].set_ylabel("total IPC")
    ax[2].set_title("(c) throughput / fading memory", loc="left", fontweight="bold")
    ax[2].grid(True, alpha=0.25)

    fig.tight_layout()
    out = "manuscript/figures/fig_reservoir.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")

    data = dict(engineered=r_eng, in_vivo=r_iv,
                scan_obs=dict(n_obs=n_obs_list, ipc=ipc_vs_obs),
                scan_throughput=dict(p=p_list, ipc=ipc_vs_p))
    with open("simulation_results/reservoir_results.json", "w") as f:
        json.dump(data, f, indent=2, default=float)
    return data


if __name__ == "__main__":
    # readout_realism() writes reservoir_readout_realism.json but was unreachable
    # from here, so a clean-room run could not recreate that shipped file.
    import sys as _sys
    if _sys.argv[1:] == ["realism"]:
        readout_realism()
    else:
        print("Quantum reservoir computing on the radical-pair + nuclear-spin system")
        print("=" * 74)
        rng = np.random.default_rng(7)
        L = 1500
        s = rng.uniform(0, 1, L + 100)  # +washout

        # ── Engineered regime: effective hyperfine writing, full readout ──
        H = build_reservoir_H(B_tesla=1e-3, A_e1_a=80, A_e1_b=40, A_e2_a=15, J=2.0)
        P_eng = propagator(H, tau_us=0.05, T2e_ns=50.0)   # τ=50 ns, T2e=50 ns
        obs_full = _observable_set("full")
        X_eng = run_reservoir(s, P_eng, obs_full, nuc_deph_p=0.0)
        s_eng = s[100:]
        r_eng = memory_and_ipc(X_eng, s_eng)
        print(f"\n[Engineered] {X_eng.shape[1]} observables, τ=50 ns, T2e=50 ns, nuclei retained")
        print(f"  Memory capacity MC      = {r_eng['MC']:.2f}   (≤ {r_eng['n_obs']})")
        print(f"  Nonlinear IPC (deg-2)   = {r_eng['IPC_nonlinear']:.2f}")
        print(f"  Total IPC               = {r_eng['IPC_total']:.2f}")

        # ── In-vivo regime: scalar readout only + strong inter-cycle dephasing ──
        obs_scalar = _observable_set("scalar")
        X_iv = run_reservoir(s, P_eng, obs_scalar, nuc_deph_p=0.6)
        r_iv = memory_and_ipc(X_iv, s_eng)
        print(f"\n[In-vivo]    {X_iv.shape[1]} observable (singlet yield), inter-cycle nuclear dephasing p=0.6")
        print(f"  Memory capacity MC      = {r_iv['MC']:.3f}   (≤ {r_iv['n_obs']})")
        print(f"  Nonlinear IPC (deg-2)   = {r_iv['IPC_nonlinear']:.3f}")
        print(f"  Total IPC               = {r_iv['IPC_total']:.3f}")

        print("\nGenerating scans + figure ...")
        full_analysis()
        print("  figure → manuscript/figures/fig_reservoir.pdf")
        print("  data   → simulation_results/reservoir_results.json")

        print("\nInterpretation:")
        print("  • Engineered (multi-observable, retained nuclei): the spin reservoir")
        print("    carries genuine linear memory AND nonlinear capacity — it works as")
        print("    a quantum reservoir, as in NMR/EPR molecular-spin RC.")
        print("  • In-vivo (scalar product-yield readout + sparse RP throughput): the")
        print("    capacity collapses — the brain cannot read out the reservoir nodes")
        print("    or drive it fast enough to use them.")
