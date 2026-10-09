#!/usr/bin/env python3
"""What physical quantity carries the input?

The paper encodes the input in the singlet character s of the pair at birth,
swept over [0, 1]. No chemistry is named that modulates s over that range.
This module drives the same reservoir through three carriers that chemistry or
physics could plausibly modulate, with singlet birth throughout:

  B     the field magnitude, 25-75 uT   (what a magnetoreceptor is for)
  kS    the singlet recombination rate, 0.5-1.5 us^-1 (a conformational or
        redox gate on the recombination step)
  delta small-signal s:  s = 1/2 + delta (u - 1/2), delta = 0.1, 0.01, 0.001
        (the reference encoding is delta = 1)

Inputs are quantised to 32 levels so that one propagator per level is built
once. Every encoding is scored against its own memoryless floor (q = 1).
"""
import os
import numpy as np
from scipy.linalg import expm
from qbscreen.reservoir import build_reservoir_H, memory_and_ipc, N_SPINS, DIM, NUCLEI, ELECTRONS, TWO_PI
from qbscreen.spin_dynamics import SZ, spin_op, singlet_projector
from qbscreen.master_equation import build_liouvillian, _vec, electron_dephasing_ops
from qbscreen.readout_routes import CRY
from qbscreen.panel_response import _inputs, _save, _electron_born_state, _nuc_from_electron_trace, _sub_trace_e, SAMPLE

OUT = "simulation_results/panel"
D_NUC = DIM // 4
LEVELS = 32


def _ops(H, tau_us=1.0, T2e_ns=1000.0, kS=1.0, kT=0.2, n_t=96):
    P_S = singlet_projector(0, 1, N_SPINS); P_T = np.eye(DIM) - P_S
    deph = electron_dephasing_ops(N_SPINS, T2e_ns * 1e-3, electron_indices=ELECTRONS)
    with np.errstate(over="ignore", invalid="ignore"):
        L = build_liouvillian(TWO_PI * H, P_S, P_T, kS, kT, deph)
        return P_S, P_T, expm(L * (tau_us / n_t)), tau_us / n_t


def run(inputs, encoding, q_nuc=0.0, n_t=96, washout=80):
    base = dict(CRY)
    levels = np.linspace(0, 1, LEVELS)
    ops = {}
    for i, u in enumerate(levels):
        if encoding == "B":
            H = build_reservoir_H(**{**base, "B_tesla": (25.0 + 50.0 * u) * 1e-6}); ops[i] = _ops(H)
        elif encoding == "kS":
            H = build_reservoir_H(**base); ops[i] = _ops(H, kS=0.5 + u)
    if encoding not in ("B", "kS"):
        H = build_reservoir_H(**base); common = _ops(H)
    cid_ops = [spin_op(2 * SZ, i, N_SPINS) for i in NUCLEI]
    I_nuc = np.eye(D_NUC, dtype=complex) / D_NUC
    rows = []
    rho = singlet_projector(0, 1, N_SPINS).astype(complex); rho /= np.trace(rho)
    for u in inputs:
        nuc = _nuc_from_electron_trace(rho)
        if q_nuc > 0:
            nuc = (1 - q_nuc) * nuc + q_nuc * I_nuc
        if encoding in ("B", "kS"):
            P_S, P_T, step, dt = ops[int(round(u * (LEVELS - 1)))]; kS = 1.0 if encoding == "B" else 0.5 + levels[int(round(u * (LEVELS - 1)))]
            s_born = 1.0
        else:
            P_S, P_T, step, dt = common; kS = 1.0
            delta = float(encoding.split(":")[1]); s_born = 0.5 + delta * (u - 0.5)
        rho = np.kron(_electron_born_state(s_born, "ST"), nuc)
        v = _vec(rho); yS = 0.0; yIz = np.zeros(3); yS_t = []
        M_prod = np.zeros((D_NUC, D_NUC), dtype=complex)
        for it in range(1, n_t + 1):
            v = step @ v; r = v.reshape(DIM, DIM, order="F")
            yS += kS * np.real(np.trace(P_S @ r)) * dt
            for j, O in enumerate(cid_ops):
                yIz[j] += kS * np.real(np.trace(O @ P_S @ r)) * dt
            aS = P_S @ r @ P_S; aT = P_T @ r @ P_T
            M_prod += (kS * _sub_trace_e(aS) + 0.2 * _sub_trace_e(aT)) * dt
            if it in SAMPLE:
                yS_t.append(yS)
        rows.append(list(yS_t) + list(yIz / (yS if yS else 1.0)))
        M_prod = 0.5 * (M_prod + M_prod.conj().T); tp = np.trace(M_prod).real
        rho = np.kron(np.eye(4, dtype=complex) / 4.0, M_prod / (tp if tp else 1.0))
    X = np.array(rows[washout:])
    return X[:, :5], X


def scan(n_seeds=6, L=700):
    out = {}
    for enc in ("s:1", "s:0.1", "s:0.01", "s:0.001", "B", "kS"):
        live = {"5": [], "8": []}; floor = {"5": [], "8": []}
        for sd in range(n_seeds):
            s, sp = _inputs(sd, L)
            Y, C = run(s, enc); Yf, Cf = run(s, enc, q_nuc=1.0)
            live["5"].append(memory_and_ipc(Y, sp)["IPC_total"]); live["8"].append(memory_and_ipc(C, sp)["IPC_total"])
            floor["5"].append(memory_and_ipc(Yf, sp)["IPC_total"]); floor["8"].append(memory_and_ipc(Cf, sp)["IPC_total"])
        row = dict(encoding=enc, n_seeds=n_seeds)
        for k, name in (("5", "kinetics_5ch"), ("8", "cidnp_8ch")):
            row[name] = float(np.mean(live[k])); row[name + "_sd"] = float(np.std(live[k], ddof=1))
            row[name + "_floor"] = float(np.mean(floor[k])); row[name + "_excess"] = row[name] - row[name + "_floor"]
        out[enc] = row
        print(f"  {enc:8s} 5ch {row['kinetics_5ch']:.3f} (floor {row['kinetics_5ch_floor']:.3f}, excess {row['kinetics_5ch_excess']:+.3f})"
              f"  8ch {row['cidnp_8ch']:.3f} (floor {row['cidnp_8ch_floor']:.3f}, excess {row['cidnp_8ch_excess']:+.3f})", flush=True)
        _save("open13_input_carrier", out)
    return out


if __name__ == "__main__":
    import time; t0 = time.time(); os.makedirs(OUT, exist_ok=True); scan(); print(f"  ({time.time()-t0:.0f} s)")
