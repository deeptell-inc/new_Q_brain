#!/usr/bin/env python3
"""Readout with molecule-counting noise, not relative Gaussian noise.

The shipped shot-noise check (panel_response.ridge_sensitivity) adds Gaussian
noise of size |X|/sqrt(N) to every channel. That is a relative-precision model.
A pool of N molecules does not read that way: a yield y is the fraction of N
molecules that recombined, Binomial(N, y)/N, and a product polarisation P of a
spin-1/2 nucleus is the mean of n_S = N y_S values of +-1, whose variance is
(1 - P^2)/n_S -- which does not shrink with P. For the small polarisations the
cryptochrome point produces (|P| of order 1e-2) the two models differ by orders
of magnitude. This module samples the channels the way a pool is counted, for
the full register and for the proton-only product register, with the floor
(register wiped every cycle) counted under the same noise.
"""

import os
import numpy as np

from qbscreen.reservoir import build_reservoir_H, memory_and_ipc
from qbscreen.readout_routes import CRY, run_routes
from qbscreen.panel_response import _inputs, _save, run_routes_v2
from qbscreen.nuclide_register import run_selective

OUT = "simulation_results/panel"
NS = (1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12)


def _count(R, N, rng):
    """Resample the eight CIDNP-route channels for a pool of N molecules.

    cidnp rows are [yS(t1..t5), P_1, P_2, P_3] with P_j = yIz_j / yS, the mean
    of 2 I_z over the singlet product. Yields: Binomial(N, y)/N (the five
    cumulative yields share one pool, so they are sampled as a nested count).
    Polarisations: n_S = N yS_end product molecules, each carrying +-1 with
    probability (1 +- P)/2.
    """
    X = R["cidnp"]; Y = X[:, :5]; P = X[:, 5:]
    out = np.empty_like(X)
    for k in range(len(X)):
        # nested count: molecules recombined by t1, then additional by t2, ...
        y = np.clip(Y[k], 0, 1); prev = 0.0; n_prev = 0
        for j in range(5):
            frac = (y[j] - prev) / (1 - prev) if prev < 1 else 0.0
            n_prev += rng.binomial(int(N) - n_prev, float(np.clip(frac, 0, 1)))
            out[k, j] = n_prev / N; prev = y[j]
        nS = max(n_prev, 1)
        for j in range(3):
            p_up = float(np.clip((1 + P[k, j]) / 2, 0, 1))
            out[k, 5 + j] = (2 * rng.binomial(nS, p_up) - nS) / nS
    return out


def _gauss(R, N, rng):
    X = R["cidnp"]; scale = np.abs(X).mean(0) + 1e-30
    return X + rng.normal(0, 1, X.shape) * scale / np.sqrt(N)


def scan(n_seeds=6, L=700):
    H = build_reservoir_H(**CRY)
    regs = {
        "full_survivor": (lambda s, q: run_routes(s, H, q_nuc=q)),
        "proton_product": (lambda s, q: run_selective(s, H, q_per_nucleus=(1.0, 1.0, q), carrier="product")),
    }
    out = {}
    for name, fn in regs.items():
        live, floor = [], []
        for sd in range(n_seeds):
            s, sp = _inputs(sd, L)
            live.append((fn(s, 0.0), sp)); floor.append((fn(s, 1.0), sp))
        rows = []
        print(f"\n  register={name}")
        print(f"  {'N':>8} {'IPC live':>9} {'floor':>7} {'excess':>7} | {'gauss live':>10} {'excess':>7}")
        for N in NS:
            rec = dict(n_molecules=N)
            for model, f in (("count", _count), ("gauss", _gauss)):
                rng = np.random.default_rng(int(np.log10(N)) * 7 + 1)
                a = [memory_and_ipc(f(R, N, rng), sp)["IPC_total"] for R, sp in live]
                b = [memory_and_ipc(f(R, N, rng), sp)["IPC_total"] for R, sp in floor]
                rec[f"{model}_live"] = float(np.mean(a)); rec[f"{model}_live_sd"] = float(np.std(a, ddof=1))
                rec[f"{model}_floor"] = float(np.mean(b)); rec[f"{model}_floor_sd"] = float(np.std(b, ddof=1))
                rec[f"{model}_excess"] = rec[f"{model}_live"] - rec[f"{model}_floor"]
            rows.append(rec)
            print(f"  {N:>8.0e} {rec['count_live']:>9.3f} {rec['count_floor']:>7.3f} {rec['count_excess']:>7.3f}"
                  f" | {rec['gauss_live']:>10.3f} {rec['gauss_excess']:>7.3f}", flush=True)
        a = [memory_and_ipc(R["cidnp"], sp)["IPC_total"] for R, sp in live]
        b = [memory_and_ipc(R["cidnp"], sp)["IPC_total"] for R, sp in floor]
        out[name] = dict(noiseless_live=float(np.mean(a)), noiseless_floor=float(np.mean(b)),
                         noiseless_excess=float(np.mean(a) - np.mean(b)), rows=rows, n_seeds=n_seeds)
        print(f"  noiseless: live={np.mean(a):.3f} floor={np.mean(b):.3f} excess={np.mean(a)-np.mean(b):.3f}")
        _save("open9_counting_noise", out)
    return out


if __name__ == "__main__":
    import time
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    scan()
    print(f"  ({time.time()-t0:.0f} s)")
