#!/usr/bin/env python3
"""Quantum-versus-ESN capacity with the ridge chosen by cross-validation.

The shipped comparison (m6_cry_classical) fixes lambda = 1e-6 for both systems.
The quantum channels have amplitudes of order 1e-2 and the ESN states of order
1, so one absolute ridge penalises the two very differently, and
m9_ridge_sensitivity shows the quantum capacity moving from 1.9 to 4.5 with
lambda. Here each capacity target picks its own lambda on the training half
(features standardised on the training half only), for both systems, paired
over the same input seeds.
"""
import os
import numpy as np
import qbscreen.reservoir as rv
from qbscreen.reservoir import build_reservoir_H, memory_and_ipc
from qbscreen.readout_routes import CRY
from qbscreen.panel_response import _inputs, _save, run_routes_v2
from qbscreen.qrc_benchmarks import esn_states

LAMS = tuple(10.0 ** np.arange(-12, 1))


def _cap_cv(X, target, ridge=None):
    """Hold-out capacity with lambda picked by 5-fold CV on the training half."""
    n = len(X); h = n // 2
    if n < 8:
        return 0.0
    Xtr, Xte, ytr, yte = X[:h], X[h:], target[:h], target[h:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-30
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    folds = np.array_split(np.arange(h), 5)

    def fit(A, y, lam):
        return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y - y.mean())), y.mean()

    best, best_err = None, np.inf
    for lam in LAMS:
        err = 0.0
        for f in folds:
            tr = np.setdiff1d(np.arange(h), f)
            w, my = fit(Xtr[tr], ytr[tr], lam)
            err += np.sum((ytr[f] - my - Xtr[f] @ w) ** 2)
        if err < best_err:
            best, best_err = lam, err
    w, my = fit(Xtr, ytr, best)
    ss_res = np.sum((yte - my - Xte @ w) ** 2); ss_tot = np.sum((yte - yte.mean()) ** 2)
    return max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def ipc_cv(X, s):
    orig = rv._capacity; rv._capacity = _cap_cv
    try:
        return memory_and_ipc(X, s)["IPC_total"]
    finally:
        rv._capacity = orig


def compare(n_seeds=12, L=700, carrier="survivor"):
    H = build_reservoir_H(**CRY)
    acc = {k: [] for k in ("q5_fixed", "q5_cv", "q8_fixed", "q8_cv", "esn5_fixed", "esn5_cv", "esn8_fixed", "esn8_cv")}
    for sd in range(n_seeds):
        s, sp = _inputs(sd, L)
        R = run_routes_v2(s, H, carrier=carrier)
        acc["q5_fixed"].append(memory_and_ipc(R["YS_t"], sp)["IPC_total"]); acc["q5_cv"].append(ipc_cv(R["YS_t"], sp))
        acc["q8_fixed"].append(memory_and_ipc(R["cidnp"], sp)["IPC_total"]); acc["q8_cv"].append(ipc_cv(R["cidnp"], sp))
        for n in (5, 8):
            X = esn_states(s, n, seed=100 + sd)[80:][:, :n]
            acc[f"esn{n}_fixed"].append(memory_and_ipc(X, sp)["IPC_total"]); acc[f"esn{n}_cv"].append(ipc_cv(X, sp))
        print(f"  seed {sd}: q5 {acc['q5_fixed'][-1]:.2f}->{acc['q5_cv'][-1]:.2f}  esn5 {acc['esn5_fixed'][-1]:.2f}->{acc['esn5_cv'][-1]:.2f}"
              f"  q8 {acc['q8_fixed'][-1]:.2f}->{acc['q8_cv'][-1]:.2f}  esn8 {acc['esn8_fixed'][-1]:.2f}->{acc['esn8_cv'][-1]:.2f}", flush=True)
    out = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in acc.items()}
    for n, q in ((5, "q5"), (8, "q8")):
        for mode in ("fixed", "cv"):
            d = np.array(acc[f"esn{n}_{mode}"]) - np.array(acc[f"{q}_{mode}"])
            out[f"esn{n}_minus_{q}_{mode}"] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)),
                                                  sem=float(d.std(ddof=1) / np.sqrt(n_seeds)),
                                                  pct_of_quantum=float(100 * d.mean() / np.mean(acc[f"{q}_{mode}"])))
            print(f"  ESN{n} - quantum ({mode}): {d.mean():+.3f} +/- {d.std(ddof=1)/np.sqrt(n_seeds):.3f} sem"
                  f"  = {out[f'esn{n}_minus_{q}_{mode}']['pct_of_quantum']:+.0f}% of quantum")
    out["n_seeds"] = n_seeds; out["lambda_grid"] = list(LAMS)
    return _save("open11_ridge_cv_comparison", out)


if __name__ == "__main__":
    import time; t0 = time.time(); compare(); print(f"  ({time.time()-t0:.0f} s)")
