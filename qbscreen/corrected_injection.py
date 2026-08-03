#!/usr/bin/env python3
"""Corrected radical-pair injection: chemically consistent correlated-pair birth.

Motivation
----------
The original reservoir protocol (`reservoir.run_reservoir*`) resets ONLY the input
electron e1 to a product state |psi(s)> and keeps the reduced state of everything
else.  For a radical pair this is not a chemical process: it destroys the e1-e2
spin correlation every cycle, so <P_S> = 1/4 - <S1.S2> collapses to the constant
1/4 whenever e1 and e2 are not otherwise coupled (e.g. J = 0, the separated-pair /
cryptochrome case).  The resulting "readout collapse to IPC = 0" is therefore a
property of the injection map, NOT of weak-field spin chemistry.

What a real radical-pair cycle does
-----------------------------------
Each catalytic/photochemical turnover *creates a new spin-correlated electron
pair* while the nuclei -- which are not consumed by recombination -- carry their
polarisation across cycles.  The input modulates the singlet/triplet character of
the newly born pair.  We therefore replace the injection by

    rho -> rho_e(s) (x) Tr_e[rho],      rho_e(s) = s |S><S| + (1-s) P_T/3

(optionally an S-T0 coherent superposition), which is a legitimate CPTP map and
is the standard singlet/triplet-born initial condition of the radical-pair
mechanism.

This module reproduces the capacity of the SAME Hamiltonian under the corrected
injection so the two protocols can be compared directly.
"""

import json
import numpy as np
from scipy.linalg import expm

from qbscreen.reservoir import (
    build_reservoir_H, memory_and_ipc, run_reservoir_realistic,
    N_SPINS, DIM, ELECTRONS,
)
from qbscreen.spin_dynamics import singlet_projector
from qbscreen.master_equation import build_liouvillian, _vec, electron_dephasing_ops

TWO_PI = 2.0 * np.pi
D_NUC = DIM // 4                      # dimension of the nuclear register


def _electron_born_state(s, mode="ST"):
    """2-electron density matrix of a newly born pair encoding the input s."""
    psi_S = np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
    P_S4 = np.outer(psi_S, psi_S.conj())
    P_T4 = np.eye(4, dtype=complex) - P_S4
    if mode == "ST":                                  # singlet/triplet population mix
        return s * P_S4 + (1.0 - s) * P_T4 / 3.0
    if mode == "ST0":                                 # coherent S-T0 superposition
        psi_T0 = np.array([0.0, 1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2.0)
        psi = np.sqrt(s) * psi_S + np.sqrt(1.0 - s) * psi_T0
        return np.outer(psi, psi.conj())
    raise ValueError(mode)


def run_corrected(inputs, H_mhz, tau_us, T2e_ns, kinetic_times=(0.2, 0.4, 0.6, 0.8, 1.0),
                  mode="ST", washout=80):
    """Time-resolved singlet-flux readout under corrected (correlated-birth) injection."""
    P_S = singlet_projector(0, 1, N_SPINS)
    Z = np.zeros((DIM, DIM))
    deph = electron_dephasing_ops(N_SPINS, T2e_ns * 1e-3, ELECTRONS)
    with np.errstate(over="ignore", invalid="ignore"):
        L = build_liouvillian(TWO_PI * H_mhz, Z, Z, 0.0, 0.0, deph)
        props = [expm(L * tau_us * f) for f in kinetic_times]
    P_full = props[-1]

    rho = P_S.astype(complex) / np.trace(P_S)
    feats = []
    for s in inputs:
        # keep the nuclear register, re-create the electron pair
        R = rho.reshape(4, D_NUC, 4, D_NUC)
        nuc = np.trace(R, axis1=0, axis2=2)
        nuc = nuc / np.trace(nuc)
        rho = np.kron(_electron_born_state(s, mode), nuc)
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


def compare(L=700, n_seeds=6):
    """Original vs corrected injection at the cryptochrome operating point."""
    H = build_reservoir_H(B_tesla=50e-6, A_e1_a=14.0, A_e1_b=5.0, A_e2_a=40.0, J=0.0)
    out = {}
    for label, fn in (("original_e1_reset",
                       lambda s: run_reservoir_realistic(s, H, 1.0, 1000.0,
                                                         use_cidnp=False, washout=80)),
                      ("corrected_ST_birth",
                       lambda s: run_corrected(s, H, 1.0, 1000.0, mode="ST")),
                      ("corrected_ST0_birth",
                       lambda s: run_corrected(s, H, 1.0, 1000.0, mode="ST0"))):
        ipc, spread = [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(sd)
            s = rng.uniform(0, 1, L + 80)
            X = fn(s)
            ipc.append(memory_and_ipc(X, s[80:])["IPC_total"])
            spread.append(float(np.max(X.max(0) - X.min(0))))
        out[label] = dict(IPC_mean=float(np.mean(ipc)), IPC_sd=float(np.std(ipc)),
                          max_flux_spread=float(np.mean(spread)))
        print(f"  {label:22s}: kinetic-only IPC = {np.mean(ipc):.4f} +/- {np.std(ipc):.4f}"
              f"   max flux spread = {np.mean(spread):.3e}")
    with open("simulation_results/corrected_injection.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print("Cryptochrome point (B=50 uT, J=0, tau=1 us, T2e=1 us), 6 seeds:")
    compare()
