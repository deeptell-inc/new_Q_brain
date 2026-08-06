#!/usr/bin/env python3
"""A semiclassical radical-pair reservoir: the classical reference the coherence
fraction needs.

The 0.82 coherence fraction reported in the main text is measured by adding a
large pure-dephasing rate at matched hopping rate. That is an *operational*
limit, not a comparison against a physically motivated classical model, and the
referee panel was right that the claim cannot be called a quantum/classical
decomposition without one.

This module supplies the missing reference. It keeps the Hamiltonian, the
couplings, the drive, the readout and the estimator identical, and changes one
thing: every spin becomes a classical vector of length sqrt(S(S+1)) precessing in
the field of the others,

    dS_i/dt = S_i x (dH/dS_i),

which is the Schulten-Wolynes / Timmel-type treatment carried to its natural
conclusion -- classical nuclei WITH back-action, so that the register can still
carry memory between turnovers. A static-nucleus semiclassical model would have
no memory by construction and would be a straw man.

Correspondences used (all exact in the mean):
  quantum <P_S> = 1/4 - <S1.S2>          ->   classical P_S = 1/4 - S1.S2
  singlet birth   |S><S|                 ->   antiparallel pair (S1.S2 = -3/4, P_S = 1)
  triplet birth   P_T/3                  ->   fixed angle cos(theta) = 1/3 (P_S = 0)
  electron T2e                           ->   white-noise z-field, <dw^2> = 2 dt / T2e

Each trajectory is one molecule carrying its own nuclear register across
turnovers, so the M-trajectory average is simultaneously the classical reference
AND a pooled ensemble of M independent registers.
"""

import json
import os
import numpy as np

from qbscreen.reservoir import memory_and_ipc
from qbscreen.readout_routes import CRY
from qbscreen.final_numbers import ENG

OUT = "simulation_results/panel"
LEN_HALF = np.sqrt(3) / 2.0          # |S| for spin-1/2
G_E, MU_B, HBAR = 2.00231930436256, 9.2740100783e-24, 1.054571817e-34


def _omega_e(B_tesla):
    return G_E * MU_B * B_tesla / (HBAR * 2 * np.pi * 1e6)      # MHz


def _fields(S, p):
    """dH/dS_i for every spin, in rad/us. S has shape (M, 5, 3)."""
    W = np.zeros_like(S)
    we, J = p["omega_e"], p["J"]
    Aa, Ab, Ac = p["A_e1_a"], p["A_e1_b"], p["A_e2_a"]
    # e1: Zeeman + its two nuclei + exchange with e2
    W[:, 0, 2] += we
    W[:, 0] += Aa * S[:, 2] + Ab * S[:, 3] + J * S[:, 1]
    # e2: Zeeman + its nucleus + exchange with e1
    W[:, 1, 2] += we
    W[:, 1] += Ac * S[:, 4] + J * S[:, 0]
    # nuclei see only their own electron
    W[:, 2] += Aa * S[:, 0]
    W[:, 3] += Ab * S[:, 0]
    W[:, 4] += Ac * S[:, 1]
    return 2.0 * np.pi * W


def _rk4(S, p, dt):
    k1 = np.cross(S, _fields(S, p))
    k2 = np.cross(S + 0.5 * dt * k1, _fields(S + 0.5 * dt * k1, p))
    k3 = np.cross(S + 0.5 * dt * k2, _fields(S + 0.5 * dt * k2, p))
    k4 = np.cross(S + dt * k3, _fields(S + dt * k3, p))
    return S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _dephase_electrons(S, T2e_us, dt, rng):
    """White-noise z-field on the electrons only: pure dephasing at rate 1/T2e."""
    if not T2e_us or T2e_us <= 0:
        return S
    ang = rng.normal(0.0, np.sqrt(2.0 * dt / T2e_us), size=(S.shape[0], 2))
    c, s = np.cos(ang), np.sin(ang)
    x = S[:, :2, 0].copy(); y = S[:, :2, 1].copy()
    S[:, :2, 0] = c * x - s * y
    S[:, :2, 1] = s * x + c * y
    return S


def _born_pair(sk, M, rng):
    """Classical analogue of rho_e(s) = s|S><S| + (1-s) P_T/3."""
    # random common axis
    v = rng.normal(size=(M, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    S1 = LEN_HALF * v
    singlet = rng.random(M) < sk
    # singlet: exactly antiparallel  -> S1.S2 = -3/4 -> P_S = 1
    S2 = -S1.copy()
    # triplet: fixed angle with cos(theta) = 1/3 -> S1.S2 = +1/4 -> P_S = 0
    n_t = (~singlet).sum()
    if n_t:
        w = rng.normal(size=(n_t, 3))
        vt = v[~singlet]
        w -= (w * vt).sum(1, keepdims=True) * vt          # perpendicular component
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        ct = 1.0 / 3.0; st = np.sqrt(1 - ct ** 2)
        S2[~singlet] = LEN_HALF * (ct * vt + st * w)
    return S1, S2


def _random_nuclei(M, rng):
    v = rng.normal(size=(M, 3, 3))
    return LEN_HALF * v / np.linalg.norm(v, axis=2, keepdims=True)


def run_classical(inputs, params, tau_us, T2e_ns, M=512, n_t=96, sub=6,
                  kS=0.0, kT=0.0, sample=(16, 36, 56, 76, 96), washout=80,
                  q_nuc=0.0, seed=0, obs="full"):
    """Drive the classical reservoir; return the feature matrix.

    obs = "full"  : the same twelve observables as the quantum headline
                    (3 nuclei x XYZ, two electron Z, P_S) -- no recombination
          "chem"  : the chemical routes (time-resolved yield, CIDNP) -- needs kS,kT
    """
    rng = np.random.default_rng(seed + 9973)
    p = dict(omega_e=_omega_e(params["B_tesla"]), J=params["J"],
             A_e1_a=params["A_e1_a"], A_e1_b=params["A_e1_b"],
             A_e2_a=params["A_e2_a"])
    T2e_us = None if T2e_ns is None else T2e_ns * 1e-3
    dt = tau_us / (n_t * sub)
    nuc = _random_nuclei(M, rng)
    feats = []
    for sk in inputs:
        if q_nuc > 0:                       # T1-type: randomise the register
            hit = rng.random((M, 3)) < q_nuc
            fresh = _random_nuclei(M, rng)
            nuc = np.where(hit[:, :, None], fresh, nuc)
        S1, S2 = _born_pair(sk, M, rng)
        S = np.concatenate([S1[:, None], S2[:, None], nuc], axis=1)

        yS = np.zeros(M); yT = np.zeros(M); surv = np.ones(M)
        yIz = np.zeros((M, 3)); yS_t = []
        for it in range(1, n_t + 1):
            for _ in range(sub):
                S = _rk4(S, p, dt)
                S = _dephase_electrons(S, T2e_us, dt, rng)
            if kS or kT:
                PS = 0.25 - (S[:, 0] * S[:, 1]).sum(1)
                PS = np.clip(PS, 0.0, 1.0)
                rate = kS * PS + kT * (1.0 - PS)
                dtau = tau_us / n_t
                yS += kS * PS * surv * dtau
                yT += kT * (1.0 - PS) * surv * dtau
                yIz += (kS * PS * surv * dtau)[:, None] * S[:, 2:, 2]
                surv = surv * np.exp(-rate * dtau)
            if it in sample:
                yS_t.append(yS.copy())

        if obs == "full":
            PS = np.clip(0.25 - (S[:, 0] * S[:, 1]).sum(1), 0.0, 1.0)
            row = []
            for j in range(3):
                row += [S[:, 2 + j, 2].mean(), S[:, 2 + j, 0].mean(),
                        S[:, 2 + j, 1].mean()]
            row += [S[:, 0, 2].mean(), S[:, 1, 2].mean(), PS.mean()]
            feats.append(row)
            nuc = S[:, 2:]
        else:
            den = np.where(yS > 0, yS, 1.0)
            row = [v.mean() for v in yS_t] + list((yIz / den[:, None]).mean(0))
            feats.append(row)
            # product register: nuclear vector at a sampled recombination time.
            # |I| is conserved by precession, so the carried vector is a valid spin.
            nuc = S[:, 2:]
    return np.array(feats[washout:])


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  [checkpoint] {OUT}/{name}.json")
    return obj


def coherence_reference(n_seeds=4, L=700, M=512, sub=6):
    """The headline comparison: quantum 5.42 (ENG, 12 observables) against the
    same reservoir with classical spins."""
    out = {"quantum_reference_IPC": 5.420585739613524}
    vals = []
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        s = rng.uniform(0, 1, L + 80)
        X = run_classical(s, ENG, tau_us=0.05, T2e_ns=50.0, M=M, sub=sub,
                          seed=sd, obs="full", washout=80)
        r = memory_and_ipc(X, s[80:])
        vals.append(r["IPC_total"])
        print(f"    seed {sd}: classical IPC = {r['IPC_total']:.4f}  MC = {r['MC']:.4f}",
              flush=True)
    out["classical_IPC"] = float(np.mean(vals))
    out["classical_IPC_sd"] = float(np.std(vals))
    out["M_trajectories"] = M
    q = out["quantum_reference_IPC"]
    out["semiclassical_coherence_fraction"] = float((q - np.mean(vals)) / q)
    print(f"  classical IPC = {out['classical_IPC']:.3f} +/- {out['classical_IPC_sd']:.3f}"
          f"   vs quantum {q:.3f}")
    print(f"  semiclassical coherence fraction = "
          f"{out['semiclassical_coherence_fraction']:.3f}"
          f"  (operational dephasing limit gave 0.816)")
    return _save("open4_semiclassical", out)


def classical_floor(n_seeds=4, L=700, M=512, sub=6):
    """The classical reservoir's own memoryless floor: register randomised every
    cycle (q_nuc = 1). Whatever the live classical run scores above THIS is real
    classical memory. Without the control, finite-M sampling noise in the pooled
    observable makes the live number hard to interpret."""
    vals = []
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        s = rng.uniform(0, 1, L + 80)
        X = run_classical(s, ENG, tau_us=0.05, T2e_ns=50.0, M=M, sub=sub,
                          seed=sd, obs="full", washout=80, q_nuc=1.0)
        vals.append(memory_and_ipc(X, s[80:])["IPC_total"])
        print(f"    seed {sd}: classical floor IPC = {vals[-1]:.4f}", flush=True)
    out = dict(classical_floor_IPC=float(np.mean(vals)),
               classical_floor_IPC_sd=float(np.std(vals)), M_trajectories=M)
    print(f"  classical floor = {out['classical_floor_IPC']:.3f}"
          f" +/- {out['classical_floor_IPC_sd']:.3f}")
    return _save("open4_semiclassical_floor", out)


def convergence_check(L=300, M=512):
    """The classical result must not depend on the integration step."""
    rows = []
    rng = np.random.default_rng(0); s = rng.uniform(0, 1, L + 80)
    for sub in (3, 6, 12):
        X = run_classical(s, ENG, tau_us=0.05, T2e_ns=50.0, M=M, sub=sub,
                          seed=0, obs="full", washout=80)
        ipc = memory_and_ipc(X, s[80:])["IPC_total"]
        rows.append(dict(substeps=sub, dt_ns=1000 * 0.05 / (96 * sub), IPC=float(ipc)))
        print(f"    sub={sub:>3}  dt={rows[-1]['dt_ns']:.3f} ns  IPC={ipc:.4f}", flush=True)
    return _save("open4_semiclassical_convergence", rows)


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    if "conv" in sys.argv:
        convergence_check()
    elif "floor" in sys.argv:
        classical_floor()
    else:
        coherence_reference()
    print(f"  ({time.time()-t0:.0f} s)")
