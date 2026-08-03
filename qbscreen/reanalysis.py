#!/usr/bin/env python3
"""Corrected re-analysis addressing the adversarial-panel findings.

Produces every number that the original analysis got wrong or reported without
the necessary control:

  R1  sample-length convergence of the capacity estimator + null floor
  R2  coherence fraction on the HEADLINE system, with the incoherent limit taken
      at matched hopping rate (tau proportional to gamma) instead of at a fixed
      tau where strong dephasing merely Zeno-freezes the dynamics
  R3  fading-memory ablation: which mechanism actually supplies it
  R4  symmetric classical baseline: ESN given the same number of readout features
  R5  sensing observable |dPhi_S/dB| on the same sweep as the computing capacity,
      so the sensing-computing trade-off is measured rather than asserted
  R6  cryptochrome operating point with the corrected radical/nucleus assignment
"""

import json
import numpy as np
from scipy.linalg import expm

from qbscreen.reservoir import (
    build_reservoir_H, propagator, run_reservoir, _observable_set,
    memory_and_ipc, _legendre2, N_SPINS, DIM, ELECTRONS,
)
from qbscreen.qrc_benchmarks import esn_states
from qbscreen.spin_dynamics import spin_op, singlet_projector, SZ
from qbscreen.master_equation import build_liouvillian, _vec, electron_dephasing_ops

TWO_PI = 2.0 * np.pi
ENG = dict(B_tesla=1e-3, A_e1_a=80, A_e1_b=40, A_e2_a=15, J=2.0)
CRY = dict(B_tesla=50e-6, A_e1_a=14.0, A_e1_b=5.0, A_e2_a=40.0, J=0.0)


# ── generic capacity with optional clipping (for the null-floor control) ──
def _ipc(X, s, clip=True, max_delay=14):
    def cap(Xs, y):
        n = len(Xs); h = n // 2
        Xtr, Xte = Xs[:h], Xs[h:]; ytr, yte = y[:h], y[h:]
        mx = Xtr.mean(0); my = ytr.mean(); Xc = Xtr - mx
        G = Xc.T @ Xc + 1e-6 * np.eye(Xc.shape[1])
        w = np.linalg.solve(G, Xc.T @ (ytr - my))
        pred = (Xte - mx) @ w
        ssr = np.sum(((yte - my) - pred) ** 2); sst = np.sum((yte - yte.mean()) ** 2)
        v = 1 - ssr / sst if sst > 0 else 0.0
        return max(0.0, v) if clip else v
    u = 2 * s - 1.0; tot = 0.0
    for d in range(max_delay + 1):
        y = np.roll(s, d).copy(); y[:d] = s[0]; tot += cap(X[d:], y[d:])
    for d in range(max_delay + 1):
        y = _legendre2(np.roll(u, d)); y[:d] = 0.0; tot += cap(X[d:], y[d:])
    for d1 in range(6):
        for d2 in range(d1 + 1, 7):
            y = np.roll(u, d1) * np.roll(u, d2); y[:d2] = 0.0; tot += cap(X[d2:], y[d2:])
    return tot


def _headline_X(seed, L, T2e_ns=50.0, tau_us=0.05, gamma=0.0, nuc_deph_p=0.0):
    H = build_reservoir_H(**ENG)
    if gamma > 0:
        Z = np.zeros((DIM, DIM))
        deph = electron_dephasing_ops(N_SPINS, T2e_ns * 1e-3, ELECTRONS)
        deph = deph + [(TWO_PI * gamma, spin_op(SZ, i, N_SPINS)) for i in range(N_SPINS)]
        with np.errstate(over="ignore", invalid="ignore"):
            P = expm(build_liouvillian(TWO_PI * H, Z, Z, 0., 0., deph) * tau_us)
    else:
        P = propagator(H, tau_us, T2e_ns)
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, L + 100)
    return run_reservoir(s, P, _observable_set("full"),
                         nuc_deph_p=nuc_deph_p, washout=100), s[100:]


# ───────────────────────── R1 ─────────────────────────
def r1_convergence(n_seeds=5):
    print("\n[R1] sample-length convergence + null floor")
    res = {}
    for L in (600, 1200, 2400, 4800):
        v = [memory_and_ipc(*_headline_X(sd, L))["IPC_total"] for sd in range(n_seeds)]
        res[L] = (float(np.mean(v)), float(np.std(v)))
        print(f"   L={L:5d}: IPC = {np.mean(v):.3f} +/- {np.std(v):.3f}")
    fl_c, fl_n = [], []
    for sd in range(n_seeds):
        X, sp = _headline_X(sd, 1200)
        s2 = np.random.default_rng(9000 + sd).uniform(0, 1, len(sp))
        fl_c.append(_ipc(X, s2, True)); fl_n.append(_ipc(X, s2, False))
    print(f"   null floor: clipped = {np.mean(fl_c):+.4f} | unclipped = {np.mean(fl_n):+.4f}")
    return dict(L_series={str(k): v for k, v in res.items()},
                null_floor_clipped=float(np.mean(fl_c)),
                null_floor_unclipped=float(np.mean(fl_n)))


# ───────────────────────── R2 ─────────────────────────
def r2_coherence(n_seeds=6, L=900):
    """Coherence fraction on the headline system.
    Incoherent limit at MATCHED hopping rate: rate ~ |H|^2/gamma, so keeping
    rate*tau fixed requires tau proportional to gamma."""
    print("\n[R2] coherence fraction, headline system, matched-hopping incoherent limit")
    base = [memory_and_ipc(*_headline_X(sd, L))["IPC_total"] for sd in range(n_seeds)]
    q = float(np.mean(base))
    print(f"   coherent (gamma=0)          : IPC = {q:.3f}")
    sat = {}
    for g, tau in ((1e3, 0.05 * 1e3 / 50), (5e3, 0.05 * 5e3 / 50), (2e4, 0.05 * 2e4 / 50)):
        v = [memory_and_ipc(*_headline_X(sd, L, gamma=g, tau_us=tau))["IPC_total"]
             for sd in range(n_seeds)]
        sat[g] = float(np.mean(v))
        print(f"   gamma={g:>8.0f}, tau={tau*1e3:>7.1f} ns : IPC = {np.mean(v):.3f}")
    inc = float(np.mean(list(sat.values())))
    frac = (q - inc) / q
    print(f"   incoherent limit (saturated) = {inc:.3f}  ->  coherence fraction = {frac:.2f}")
    return dict(coherent=q, incoherent=inc, coherence_fraction=float(frac),
                sweep={str(k): v for k, v in sat.items()})


# ───────────────────────── R3 ─────────────────────────
def r3_fading_memory(n_seeds=6, L=900):
    print("\n[R3] what supplies the fading memory?")
    out = {}
    for lab, kw in (("full (T2e=50ns, reset on)", dict()),
                    ("no electron dephasing (T2e=inf)", dict(T2e_ns=None)),
                    ("nuclei fully dephased each cycle", dict(nuc_deph_p=1.0))):
        ipc, ker = [], []
        for sd in range(n_seeds):
            X, sp = _headline_X(sd, L, **kw)
            r = memory_and_ipc(X, sp); ipc.append(r["IPC_total"]); ker.append(r["lin_by_delay"])
        k = np.mean(ker, 0)
        out[lab] = dict(IPC=float(np.mean(ipc)), kernel=[float(x) for x in k[:5]])
        print(f"   {lab:34s}: IPC={np.mean(ipc):.3f}  kernel(d=0..4)={np.round(k[:5],3)}")
    return out


# ───────────────────────── R4 ─────────────────────────
def r4_symmetric_baseline(n_seeds=8, L=1200):
    """ESN given the SAME number of readout features as the quantum reservoir."""
    print("\n[R4] symmetric classical baseline (feature-count matched)")
    q = [memory_and_ipc(*_headline_X(sd, L))["IPC_total"] for sd in range(n_seeds)]
    print(f"   quantum 5 spins, 12 observables      : IPC = {np.mean(q):.3f} +/- {np.std(q):.3f}")
    res = {"quantum_5spin_12obs": [float(np.mean(q)), float(np.std(q))]}
    for label, n_nodes, quad in (("ESN 5 nodes, 5 linear feats", 5, False),
                                 ("ESN 5 nodes, 12 feats (5 lin + 7 quad)", 5, True),
                                 ("ESN 12 nodes, 12 linear feats", 12, False)):
        v = []
        for sd in range(n_seeds):
            rng = np.random.default_rng(sd); s = rng.uniform(0, 1, L + 100); sp = s[100:]
            X = esn_states(s, n_nodes, seed=200 + sd)[100:]
            if quad:
                idx = [(i, j) for i in range(n_nodes) for j in range(i, n_nodes)][:7]
                X = np.hstack([X] + [(X[:, i] * X[:, j])[:, None] for i, j in idx])
            v.append(memory_and_ipc(X, sp)["IPC_total"])
        res[label] = [float(np.mean(v)), float(np.std(v))]
        print(f"   {label:36s}: IPC = {np.mean(v):.3f} +/- {np.std(v):.3f}")
    return res


# ───────────────────────── R5 ─────────────────────────
def _singlet_yield(B, params, T2e_ns, k=1.0):
    H = build_reservoir_H(B_tesla=B, **{a: params[a] for a in
                                        ("A_e1_a", "A_e1_b", "A_e2_a", "J")})
    PS = singlet_projector(0, 1, N_SPINS); PT = np.eye(DIM) - PS
    deph = electron_dephasing_ops(N_SPINS, None if T2e_ns is None else T2e_ns * 1e-3, ELECTRONS)
    with np.errstate(over="ignore", invalid="ignore"):
        L = build_liouvillian(TWO_PI * H, PS, PT, k, k, deph)
    v = np.linalg.solve(-L, _vec(PS / np.trace(PS)))
    return float(np.real(k * np.vdot(_vec(PS), v)))


def r5_tradeoff(n_seeds=6, L=900):
    print("\n[R5] measured sensing vs computing on the same T2e sweep (engineered point)")
    T2e = [5, 10, 20, 35, 50, 100, 200, 500, 1000, 2000]
    comp, sens = [], []
    for t in T2e:
        v = [memory_and_ipc(*_headline_X(sd, L, T2e_ns=float(t)))["IPC_total"]
             for sd in range(n_seeds)]
        comp.append(float(np.mean(v)))
        dB = 5e-6
        y1 = _singlet_yield(1e-3 - dB, ENG, t); y2 = _singlet_yield(1e-3 + dB, ENG, t)
        sens.append(abs(y2 - y1) / (2 * dB * 1e3))          # per mT
    for t, c, s_ in zip(T2e, comp, sens):
        print(f"   T2e={t:5d} ns : IPC={c:.3f}   |dPhi_S/dB|={s_:.4f} /mT")
    peak = T2e[int(np.argmax(comp))]
    print(f"   computing peak at T2e = {peak} ns  (tau = 50 ns  ->  {peak/50:.2f} tau)")
    return dict(T2e_ns=T2e, computing_IPC=comp, sensing_dPhidB=sens, peak_T2e_ns=peak)


# ───────────────────────── R6 ─────────────────────────
def r6_cryptochrome(n_seeds=6, L=900):
    print("\n[R6] cryptochrome point, corrected radical/nucleus assignment")
    H = build_reservoir_H(**CRY)
    obs = _observable_set("full")
    out = {}
    for T2e in (100., 1000.):
        ipc, mc = [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(800 + sd); s = rng.uniform(0, 1, L + 100); sp = s[100:]
            X = run_reservoir(s, propagator(H, 1.0, T2e), obs, washout=100)
            r = memory_and_ipc(X, sp); ipc.append(r["IPC_total"]); mc.append(r["MC"])
        out[f"T2e_{T2e:.0f}ns"] = dict(IPC=float(np.mean(ipc)), IPC_sd=float(np.std(ipc)),
                                       MC=float(np.mean(mc)),
                                       memory_horizon_us=float(np.mean(mc)) * 1.0)
        print(f"   T2e={T2e:6.0f} ns: IPC={np.mean(ipc):.3f}+/-{np.std(ipc):.3f}  "
              f"MC={np.mean(mc):.3f}  horizon={np.mean(mc):.2f} us")
    d = 5e-6
    s_cry = abs(_singlet_yield(50e-6 + d, CRY, 1000.) - _singlet_yield(50e-6 - d, CRY, 1000.)) / (2*d*1e3)
    print(f"   |dPhi_S/dB| at 50 uT, J=0 = {s_cry:.4f} /mT  "
          f"(paper claimed 'nearly field-independent')")
    out["sensing_dPhidB_per_mT_at_50uT"] = float(s_cry)
    return out


def run_all():
    res = dict(R1=r1_convergence(), R2=r2_coherence(), R3=r3_fading_memory(),
               R4=r4_symmetric_baseline(), R5=r5_tradeoff(), R6=r6_cryptochrome())
    with open("simulation_results/reanalysis.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("\n-> simulation_results/reanalysis.json")
    return res


if __name__ == "__main__":
    run_all()
