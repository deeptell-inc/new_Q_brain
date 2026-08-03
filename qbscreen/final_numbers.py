#!/usr/bin/env python3
"""Protocol-consistent final numbers.

Every quantity the revised manuscript quotes is recomputed here under ONE
protocol -- chemically consistent correlated-pair birth (qbscreen.corrected_injection)
-- so that no number is carried over from the superseded e1-only reset.
"""

import json
import numpy as np
from scipy.linalg import expm

from qbscreen.reservoir import (build_reservoir_H, memory_and_ipc, _observable_set,
                                _legendre2, N_SPINS, DIM, ELECTRONS)
from qbscreen.corrected_injection import _electron_born_state, D_NUC
from qbscreen.qrc_benchmarks import esn_states
from qbscreen.spin_dynamics import spin_op, singlet_projector, SZ
from qbscreen.master_equation import build_liouvillian, _vec, electron_dephasing_ops

TWO_PI = 2.0 * np.pi
ENG = dict(B_tesla=1e-3, A_e1_a=80, A_e1_b=40, A_e2_a=15, J=2.0)
CRY = dict(B_tesla=50e-6, A_e1_a=14.0, A_e1_b=5.0, A_e2_a=40.0, J=0.0)


def _prop(H, tau_us, T2e_ns, gamma=0.0):
    Z = np.zeros((DIM, DIM))
    deph = electron_dephasing_ops(N_SPINS, None if T2e_ns is None else T2e_ns * 1e-3,
                                  ELECTRONS)
    if gamma > 0:
        deph = deph + [(TWO_PI * gamma, spin_op(SZ, i, N_SPINS)) for i in range(N_SPINS)]
    with np.errstate(over="ignore", invalid="ignore"):
        return expm(build_liouvillian(TWO_PI * H, Z, Z, 0., 0., deph) * tau_us)


def run_corr_obs(inputs, H, tau_us, T2e_ns, observables, gamma=0.0, washout=100):
    """Corrected correlated-pair-birth injection, arbitrary observable set."""
    P = _prop(H, tau_us, T2e_ns, gamma)
    rho = singlet_projector(0, 1, N_SPINS).astype(complex)
    rho = rho / np.trace(rho)
    feats = []
    for s in inputs:
        R = rho.reshape(4, D_NUC, 4, D_NUC)
        nuc = np.trace(R, axis1=0, axis2=2)
        t = np.trace(nuc).real
        nuc = nuc / (t if t else 1.0)
        rho = np.kron(_electron_born_state(s, "ST"), nuc)
        v = P @ _vec(rho)
        rho = v.reshape(DIM, DIM, order="F")
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho).real
        if tr:
            rho = rho / tr
        feats.append([np.real(np.trace(O @ rho)) for _, O in observables])
    return np.array(feats[washout:])


def _cap(seed, L, **kw):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, L + 100)
    X = run_corr_obs(s, build_reservoir_H(**ENG), 0.05, 50.0,
                     _observable_set("full"), **kw)
    return memory_and_ipc(X, s[100:])


def f1_headline(n_seeds=6):
    print("\n[F1] headline capacity + sample-length convergence (corrected protocol)")
    res = {}
    for L in (600, 1200, 2400, 4800):
        v = [_cap(sd, L)["IPC_total"] for sd in range(n_seeds)]
        res[str(L)] = [float(np.mean(v)), float(np.std(v))]
        print(f"   L={L:5d}: IPC = {np.mean(v):.3f} +/- {np.std(v):.3f}")
    r = [_cap(sd, 1200) for sd in range(n_seeds)]
    print(f"   at L=1200: MC = {np.mean([x['MC'] for x in r]):.3f}, "
          f"nonlinear = {np.mean([x['IPC_nonlinear'] for x in r]):.3f}")
    res["MC_L1200"] = float(np.mean([x["MC"] for x in r]))
    res["nonlin_L1200"] = float(np.mean([x["IPC_nonlinear"] for x in r]))
    return res


def f2_coherence(n_seeds=6, L=900):
    print("\n[F2] coherence fraction (headline system, matched-hopping incoherent limit)")
    q = float(np.mean([_cap(sd, L)["IPC_total"] for sd in range(n_seeds)]))
    print(f"   coherent            : IPC = {q:.3f}")
    sat = []
    for g in (1e3, 5e3, 2e4):
        tau = 0.05 * g / 50.0                      # keep hopping rate x tau fixed
        v = []
        for sd in range(n_seeds):
            rng = np.random.default_rng(sd)
            s = rng.uniform(0, 1, L + 100)
            X = run_corr_obs(s, build_reservoir_H(**ENG), tau, 50.0,
                             _observable_set("full"), gamma=g)
            v.append(memory_and_ipc(X, s[100:])["IPC_total"])
        sat.append(float(np.mean(v)))
        print(f"   gamma={g:>8.0f} (tau scaled): IPC = {np.mean(v):.3f}")
    inc = float(np.mean(sat))
    print(f"   incoherent limit = {inc:.3f} -> coherence fraction = {(q-inc)/q:.2f}")
    return dict(coherent=q, incoherent=inc, fraction=float((q - inc) / q))


def f3_classical(n_seeds=8, L=1200):
    print("\n[F3] classical baseline, readout-feature matched (corrected protocol)")
    qv = [_cap(sd, L)["IPC_total"] for sd in range(n_seeds)]
    print(f"   quantum 5 spins / 12 observables      : {np.mean(qv):.3f} +/- {np.std(qv):.3f}")
    out = {"quantum": [float(np.mean(qv)), float(np.std(qv))]}
    for lab, n, quad in (("ESN 5 nodes / 5 linear feats", 5, False),
                         ("ESN 5 nodes / 12 feats (5+7 quad)", 5, True),
                         ("ESN 12 nodes / 12 linear feats", 12, False)):
        v = []
        for sd in range(n_seeds):
            rng = np.random.default_rng(sd)
            s = rng.uniform(0, 1, L + 100)
            X = esn_states(s, n, seed=200 + sd)[100:]
            if quad:
                idx = [(i, j) for i in range(n) for j in range(i, n)][:7]
                X = np.hstack([X] + [(X[:, i] * X[:, j])[:, None] for i, j in idx])
            v.append(memory_and_ipc(X, s[100:])["IPC_total"])
        out[lab] = [float(np.mean(v)), float(np.std(v))]
        print(f"   {lab:38s}: {np.mean(v):.3f} +/- {np.std(v):.3f}")
    return out


def _singlet_yield(B, params, T2e_ns, k=1.0):
    H = build_reservoir_H(B_tesla=B, **{a: params[a] for a in
                                        ("A_e1_a", "A_e1_b", "A_e2_a", "J")})
    PS = singlet_projector(0, 1, N_SPINS); PT = np.eye(DIM) - PS
    deph = electron_dephasing_ops(N_SPINS, None if T2e_ns is None else T2e_ns * 1e-3,
                                  ELECTRONS)
    with np.errstate(over="ignore", invalid="ignore"):
        L = build_liouvillian(TWO_PI * H, PS, PT, k, k, deph)
    v = np.linalg.solve(-L, _vec(PS / np.trace(PS)))
    return float(np.real(k * np.vdot(_vec(PS), v)))


def f4_tradeoff(n_seeds=8, L=900):
    print("\n[F4] computing vs sensing on one sweep (corrected protocol, paired seeds)")
    T2e = [5, 10, 20, 35, 50, 100, 200, 500, 1000, 2000]
    comp, sens = [], []
    for t in T2e:
        v = []
        for sd in range(n_seeds):
            rng = np.random.default_rng(sd)
            s = rng.uniform(0, 1, L + 100)
            X = run_corr_obs(s, build_reservoir_H(**ENG), 0.05, float(t),
                             _observable_set("full"))
            v.append(memory_and_ipc(X, s[100:])["IPC_total"])
        comp.append(float(np.mean(v)))
        d = 5e-6
        sens.append(abs(_singlet_yield(1e-3 + d, ENG, t) -
                        _singlet_yield(1e-3 - d, ENG, t)) / (2 * d * 1e3))
    for t, c, s_ in zip(T2e, comp, sens):
        print(f"   T2e={t:5d} ns : IPC={c:.3f}   |dPhi/dB|={s_:.5f} /mT")
    pk = T2e[int(np.argmax(comp))]
    print(f"   computing peak: T2e = {pk} ns = {pk/50:.2f} tau ; "
          f"sensing gain over sweep = x{sens[-1]/sens[0]:.1f}")
    return dict(T2e_ns=T2e, computing=comp, sensing=sens, peak_T2e_ns=pk,
                sensing_gain=float(sens[-1] / sens[0]))


def f5_cry(n_seeds=6, L=900):
    print("\n[F5] cryptochrome point, full (EPR-grade) readout, corrected protocol")
    out = {}
    for T2e in (100., 1000.):
        ipc, mc = [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(800 + sd)
            s = rng.uniform(0, 1, L + 100)
            X = run_corr_obs(s, build_reservoir_H(**CRY), 1.0, T2e,
                             _observable_set("full"))
            r = memory_and_ipc(X, s[100:]); ipc.append(r["IPC_total"]); mc.append(r["MC"])
        out[f"T2e_{T2e:.0f}ns"] = dict(IPC=float(np.mean(ipc)), IPC_sd=float(np.std(ipc)),
                                       MC=float(np.mean(mc)))
        print(f"   T2e={T2e:6.0f} ns: IPC={np.mean(ipc):.3f}+/-{np.std(ipc):.3f}  "
              f"MC={np.mean(mc):.3f}")
    return out


def run_all():
    res = dict(F1=f1_headline(), F2=f2_coherence(), F3=f3_classical(),
               F4=f4_tradeoff(), F5=f5_cry())
    with open("simulation_results/final_numbers.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("\n-> simulation_results/final_numbers.json")
    return res


if __name__ == "__main__":
    run_all()
