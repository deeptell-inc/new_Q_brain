#!/usr/bin/env python3
"""Trajectory-number convergence of the semiclassical reference.

open4_semiclassical was run with M = 512 trajectories and its IPC (0.764) sits
below its own memoryless floor (0.768): the classical reference is sampling-
noise limited, so the quantum-minus-classical difference cannot yet be read as
coherence. This scan raises M and records live and floor together, per seed,
so the convergence of the difference is what gets reported.
"""
import os, sys, time
import numpy as np
from qbscreen.reservoir import memory_and_ipc
from qbscreen.semiclassical import run_classical, ENG, _save

MS = (512, 2048, 8192)


def scan(n_seeds=4, L=700, sub=6, Ms=MS):
    out = []
    for M in Ms:
        t0 = time.time(); live, floor = [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(sd); s = rng.uniform(0, 1, L + 80)
            X = run_classical(s, ENG, tau_us=0.05, T2e_ns=50.0, M=M, sub=sub, seed=sd, obs="full", washout=80)
            live.append(memory_and_ipc(X, s[80:])["IPC_total"])
            X = run_classical(s, ENG, tau_us=0.05, T2e_ns=50.0, M=M, sub=sub, seed=sd, obs="full", washout=80, q_nuc=1.0)
            floor.append(memory_and_ipc(X, s[80:])["IPC_total"])
        d = np.array(live) - np.array(floor)
        out.append(dict(M_trajectories=M, n_seeds=n_seeds, live=float(np.mean(live)), live_sd=float(np.std(live, ddof=1)),
                        floor=float(np.mean(floor)), floor_sd=float(np.std(floor, ddof=1)),
                        excess=float(d.mean()), excess_sd=float(d.std(ddof=1)), seconds=time.time() - t0))
        print(f"  M={M:>6} live={out[-1]['live']:.4f}+/-{out[-1]['live_sd']:.4f} floor={out[-1]['floor']:.4f}"
              f" excess={out[-1]['excess']:+.4f}+/-{out[-1]['excess_sd']:.4f} ({out[-1]['seconds']:.0f} s)", flush=True)
        _save("open10_semiclassical_trajectories", out)
    return out


if __name__ == "__main__":
    scan()
