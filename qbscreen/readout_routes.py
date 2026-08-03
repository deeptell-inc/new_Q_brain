#!/usr/bin/env python3
"""Scrutiny of the biologically accessible readout routes of a radical-pair reservoir.

The earlier analysis tested only two readouts (a scalar singlet yield, and CIDNP
nuclear polarisation).  A real radical-pair turnover offers several *chemically
distinct* observables, because recombination is spin-selective and the two exit
channels make different molecules on different clocks:

  route 1  singlet product yield, end point            Y_S(tau)              1 ch
  route 2  singlet product, time-resolved              Y_S(t_i)              5 ch
  route 3  both exit channels, end point               Y_S, Y_T              2 ch
  route 4  both exit channels, time-resolved           Y_S(t_i), Y_T(t_i)   10 ch
  route 5  heterogeneous lifetimes (micro-environments) Y_S per tau-pool     3 ch
  route 6  CIDNP polarisation carried by the product   <I_z> on products     3 ch
  route 7  accumulated pool (downstream integrator)    low-pass of Y_S       2 ch

Unlike the original model, here recombination is switched ON during the cycle
(k_S, k_T) and the observable is the *chemical yield actually produced*,
    Y_X = k_X \\int_0^tau Tr[P_X rho(t)] dt,
which is what a downstream enzyme or a product assay can see.  The nuclei survive
in the product and carry the memory to the next turnover.
"""

import json
import numpy as np
from scipy.linalg import expm

from qbscreen.reservoir import (build_reservoir_H, memory_and_ipc,
                                N_SPINS, DIM, ELECTRONS, NUCLEI)
from qbscreen.corrected_injection import _electron_born_state
from qbscreen.spin_dynamics import spin_op, singlet_projector, SZ
from qbscreen.master_equation import build_liouvillian, _vec, electron_dephasing_ops

TWO_PI = 2.0 * np.pi
D_NUC = DIM // 4
CRY = dict(B_tesla=50e-6, A_e1_a=14.0, A_e1_b=5.0, A_e2_a=40.0, J=0.0)


def _cycle_operators(H_mhz, tau_us, T2e_ns, kS, kT, n_t=24):
    """Propagators on a fine grid with spin-selective recombination switched on."""
    P_S = singlet_projector(0, 1, N_SPINS)
    P_T = np.eye(DIM) - P_S
    deph = electron_dephasing_ops(N_SPINS, T2e_ns * 1e-3, ELECTRONS)
    with np.errstate(over="ignore", invalid="ignore"):
        L = build_liouvillian(TWO_PI * H_mhz, P_S, P_T, kS, kT, deph)
        dt = tau_us / n_t
        step = expm(L * dt)
    return P_S, P_T, step, dt


def run_routes(inputs, H_mhz, tau_us=1.0, T2e_ns=1000.0, kS=1.0, kT=0.2,
               n_t=96, sample=(16, 36, 56, 76, 96), washout=80, q_nuc=0.0):
    """Return a dict of feature matrices, one per readout route.

    n_t = 96 (dt ~ 10 ns) is the converged time grid: n_t = 24 under-samples the
    40 MHz hyperfine (25 ns period) and inflates the time-resolved capacities by
    37 % (kinetics) and 5 % (CIDNP); see grid_convergence().

    CIDNP is the *product-weighted* nuclear polarisation actually carried away by
    the recombined molecule,
        <I_z>_prod = int k_S Tr[I_z P_S rho] dt / int k_S Tr[P_S rho] dt,
    not the polarisation of the surviving (escaped) pair.

    q_nuc is the probability that the nuclear register is depolarised between
    turnovers (nuclear T1 relaxation during the pause); q_nuc = 0 means the
    register is fully reused, which is what the capacity model assumes.
    """
    P_S, P_T, step, dt = _cycle_operators(H_mhz, tau_us, T2e_ns, kS, kT, n_t)
    cid_ops = [spin_op(2 * SZ, i, N_SPINS) for i in NUCLEI]
    I_nuc = np.eye(D_NUC, dtype=complex) / D_NUC

    rows = {k: [] for k in ("YS_end", "YS_t", "SandT_end", "SandT_t", "cidnp")}
    rho = singlet_projector(0, 1, N_SPINS).astype(complex)
    rho = rho / np.trace(rho)
    for s in inputs:
        # --- rebirth: nuclei survive in the product, electron pair is recreated
        R = rho.reshape(4, D_NUC, 4, D_NUC)
        nuc = np.trace(R, axis1=0, axis2=2)
        tr = np.trace(nuc).real
        nuc = nuc / (tr if tr else 1.0)
        if q_nuc > 0:                       # nuclear T1 relaxation during the pause
            nuc = (1.0 - q_nuc) * nuc + q_nuc * I_nuc
        rho = np.kron(_electron_born_state(s, "ST"), nuc)

        v = _vec(rho)
        yS = yT = 0.0
        yIz = np.zeros(len(cid_ops))
        yS_t, yT_t = [], []
        for it in range(1, n_t + 1):
            v = step @ v
            r = v.reshape(DIM, DIM, order="F")
            yS += kS * np.real(np.trace(P_S @ r)) * dt     # singlet product made
            yT += kT * np.real(np.trace(P_T @ r)) * dt     # triplet/escape product
            for j, O in enumerate(cid_ops):                # product-weighted CIDNP
                yIz[j] += kS * np.real(np.trace(O @ P_S @ r)) * dt
            if it in sample:
                yS_t.append(yS); yT_t.append(yT)
        rows["YS_end"].append([yS])
        rows["YS_t"].append(list(yS_t))
        rows["SandT_end"].append([yS, yT])
        rows["SandT_t"].append(list(yS_t) + list(yT_t))
        rows["cidnp"].append(list(yS_t) + list(yIz / (yS if yS else 1.0)))
        rr = v.reshape(DIM, DIM, order="F"); rr = 0.5 * (rr + rr.conj().T)
        t2 = np.trace(rr).real
        rho = rr / (t2 if t2 else 1.0)
    return {k: np.array(v[washout:]) for k, v in rows.items()}


def turnover_clock(inputs, H_mhz, T_d_list=(1e-6, 1e-3, 1e-2, 1e-1), T1n_s=1.0,
                   tau_us=1.0, washout=80):
    """THE decisive test: the reservoir clock is the TURNOVER time, not the
    radical-pair lifetime.  The nuclear register survives recombination in the
    diamagnetic product (nuclear T1 ~ 0.1-1 s), so inserting a pause T_d between
    turnovers rescales the memory horizon by (tau + T_d) while the register only
    partially relaxes, q = 1 - exp(-T_d/T1n).
    """
    out = []
    for T_d in T_d_list:
        q = 1.0 - np.exp(-T_d / T1n_s)
        R = run_routes(inputs, H_mhz, tau_us=tau_us, washout=washout, q_nuc=q)
        r5 = memory_and_ipc(R["YS_t"], inputs[washout:])
        r8 = memory_and_ipc(R["cidnp"], inputs[washout:])
        cyc = tau_us * 1e-6 + T_d
        out.append(dict(T_d_s=T_d, q_nuc=float(q),
                        MC_5ch=r5["MC"], horizon_5ch_s=r5["MC"] * cyc,
                        MC_8ch=r8["MC"], horizon_8ch_s=r8["MC"] * cyc))
    return out


def run_heterogeneous(inputs, H_mhz, taus=(0.5, 1.0, 2.0), T2e_ns=1000.0,
                      kS=1.0, kT=0.2, washout=80):
    """Route 5: several micro-environments with different pair lifetimes, each
    contributing its own product pool (spatially/chemically separable)."""
    cols = []
    for tau in taus:
        P_S, P_T, step, dt = _cycle_operators(H_mhz, tau, T2e_ns, kS, kT, 24)
        rho = singlet_projector(0, 1, N_SPINS).astype(complex); rho /= np.trace(rho)
        col = []
        for s in inputs:
            R = rho.reshape(4, D_NUC, 4, D_NUC); nuc = np.trace(R, axis1=0, axis2=2)
            t = np.trace(nuc).real; nuc = nuc / (t if t else 1.0)
            rho = np.kron(_electron_born_state(s, "ST"), nuc)
            v = _vec(rho); yS = 0.0
            for _ in range(24):
                v = step @ v
                yS += kS * np.real(np.trace(P_S @ v.reshape(DIM, DIM, order="F"))) * dt
            col.append(yS)
            rr = v.reshape(DIM, DIM, order="F"); rr = 0.5 * (rr + rr.conj().T)
            t2 = np.trace(rr).real; rho = rr / (t2 if t2 else 1.0)
        cols.append(col[washout:])
    return np.array(cols).T


def accumulate(X, leak=0.5):
    """Route 7: a downstream pool that integrates the product with leak."""
    out = np.zeros_like(X); acc = np.zeros(X.shape[1])
    for i, x in enumerate(X):
        acc = (1 - leak) * acc + x
        out[i] = acc
    return np.hstack([X, out])


def survey(L=700, n_seeds=6):
    H = build_reservoir_H(**CRY)
    res = {}
    print("Biological readout routes at the cryptochrome point "
          "(B=50 uT, J=0, tau=1 us, T2e=1 us, k_S=1, k_T=0.2 /us)")
    print(f"{'route':52s} {'ch':>3} {'IPC':>15} {'MC':>6}")
    acc = {}
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        s = rng.uniform(0, 1, L + 80); sp = s[80:]
        R = run_routes(s, H)
        R["hetero_tau"] = run_heterogeneous(s, H)
        R["YS_t_accum"] = accumulate(R["YS_t"])
        for k, X in R.items():
            r = memory_and_ipc(X, sp)
            acc.setdefault(k, {"ipc": [], "mc": [], "ch": X.shape[1]})
            acc[k]["ipc"].append(r["IPC_total"]); acc[k]["mc"].append(r["MC"])
    labels = {"YS_end": "1. singlet product, end point",
              "YS_t": "2. singlet product, time-resolved",
              "SandT_end": "3. both exit channels, end point",
              "SandT_t": "4. both exit channels, time-resolved",
              "hetero_tau": "5. heterogeneous lifetimes (3 micro-envs)",
              "cidnp": "6. time-resolved + CIDNP on product",
              "YS_t_accum": "7. time-resolved + accumulated pool"}
    for k in ("YS_end", "YS_t", "SandT_end", "SandT_t", "hetero_tau", "cidnp", "YS_t_accum"):
        a = acc[k]
        res[k] = dict(channels=a["ch"], IPC=float(np.mean(a["ipc"])),
                      IPC_sd=float(np.std(a["ipc"])), MC=float(np.mean(a["mc"])))
        print(f"{labels[k]:52s} {a['ch']:>3} {np.mean(a['ipc']):>7.3f}"
              f"+/-{np.std(a['ipc']):.3f} {np.mean(a['mc']):>6.2f}")
    with open("simulation_results/readout_routes.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def grid_convergence(n_t_list=(24, 48, 96, 192), L=700, n_seeds=6):
    """Convergence of the time-resolved capacities in the number of steps per
    cycle.  A 40 MHz hyperfine has a 25 ns period, so a coarse grid under-samples
    the dynamics and inflates the capacities; the main text uses n_t = 96.

    The five sampling points are scaled with n_t so that every grid reads the
    product yield at the same five fractions of the cycle.
    """
    H = build_reservoir_H(**CRY)
    out = []
    print(f"{'n_t':>5} {'dt (ns)':>9} {'kinetics IPC':>13} {'+CIDNP IPC':>11}")
    for n_t in n_t_list:
        smp = tuple(int(round(n_t * f)) for f in (16 / 96, 36 / 96, 56 / 96, 76 / 96, 1.0))
        a, b = [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(sd)
            s = rng.uniform(0, 1, L + 80); sp = s[80:]
            R = run_routes(s, H, n_t=n_t, sample=smp)
            a.append(memory_and_ipc(R["YS_t"], sp)["IPC_total"])
            b.append(memory_and_ipc(R["cidnp"], sp)["IPC_total"])
        row = dict(n_t=n_t, dt_ns=1000.0 / n_t,
                   YS_t=float(np.mean(a)), cidnp=float(np.mean(b)))
        out.append(row)
        print(f"{n_t:>5} {row['dt_ns']:>9.1f} {row['YS_t']:>13.3f} {row['cidnp']:>11.3f}")
    with open("simulation_results/grid_convergence.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def register_reuse(qs=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0), L=700, n_seeds=4):
    """The falsifiable criterion: capacity versus the probability q that the
    nuclear register is depolarised between turnovers.  q = 0 is full reuse;
    q = 1 wipes the register every cycle and the reservoir collapses to a
    memoryless read-back.  T_d/T1n = -ln(1-q) is the equivalent pause length.
    """
    H = build_reservoir_H(**CRY)
    out = []
    print(f"{'q_nuc':>6} {'Td/T1n':>8} {'MC_5':>7} {'IPC_5':>7} {'MC_8':>7} {'IPC_8':>7}")
    for q in qs:
        acc = {k: [] for k in ("MC_5", "IPC_5", "MC_8", "IPC_8")}
        for sd in range(n_seeds):
            rng = np.random.default_rng(sd)
            s = rng.uniform(0, 1, L + 80); sp = s[80:]
            R = run_routes(s, H, q_nuc=q)
            r5 = memory_and_ipc(R["YS_t"], sp)
            r8 = memory_and_ipc(R["cidnp"], sp)
            acc["MC_5"].append(r5["MC"]);  acc["IPC_5"].append(r5["IPC_total"])
            acc["MC_8"].append(r8["MC"]);  acc["IPC_8"].append(r8["IPC_total"])
        row = dict(q_nuc=float(q),
                   Td_over_T1n=float(-np.log(max(1.0 - q, 1e-12)) if q < 1 else np.inf),
                   **{k: float(np.mean(v)) for k, v in acc.items()})
        out.append(row)
        print(f"{q:>6.2f} {row['Td_over_T1n']:>8.3f} {row['MC_5']:>7.3f}"
              f" {row['IPC_5']:>7.3f} {row['MC_8']:>7.3f} {row['IPC_8']:>7.3f}")
    with open("simulation_results/register_reuse.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def clock_scan(L=700, seed=0, T1n_s=1.0):
    """Write the turnover-clock scan (Sec. 'the clock is the turnover interval').

    A single input realisation suffices here because the quantity of interest is
    the *ratio* of memory capacities across T_d, all evaluated on the same input.
    """
    H = build_reservoir_H(**CRY)
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, L + 80)
    out = turnover_clock(s, H, T1n_s=T1n_s)
    for r in out:
        print(f"T_d = {r['T_d_s']:8.1e} s   q = {r['q_nuc']:.4f}   "
              f"MC(5ch) = {r['MC_5ch']:.3f}  horizon = {r['horizon_5ch_s']:.3e} s   "
              f"MC(8ch) = {r['MC_8ch']:.3f}  horizon = {r['horizon_8ch_s']:.3e} s")
    with open("simulation_results/turnover_clock.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "survey"):
        survey()
    if which in ("all", "clock"):
        print()
        clock_scan()
    if which in ("all", "reuse"):
        print()
        register_reuse()
    if which in ("all", "grid"):
        print()
        grid_convergence()
