#!/usr/bin/env python3
"""Do the adverse results survive the physically correct register?

The set-2 audit found that only the readout survey had been recomputed with the
product register; every result that REDUCES the paper's claims -- the nuclide
restriction, the turnover-jitter collapse of the kinetic readout, and the
anisotropy cost -- had been established with the survivor register only. The
manuscript disclosed that scope, but the honest thing is to close it.

This module re-runs all three with carrier="product" and reports the comparison.
If the reductions hold, the scope caveat can be removed. If they do not, the
manuscript must say so.
"""

import json
import os
import numpy as np

from qbscreen.reservoir import build_reservoir_H, memory_and_ipc
from qbscreen.readout_routes import CRY
from qbscreen.panel_response import _inputs, _save
from qbscreen.nuclide_register import run_selective, SCENARIOS
from qbscreen.ensemble_pooled import _traces, SAMPLE, N_T
from qbscreen.panel_response import run_routes_v2
from qbscreen.general_spin import build_H, _capacities, _inp

OUT = "simulation_results/panel"


# ── 1. nuclide restriction ───────────────────────────────────────────
def nuclide_product(n_seeds=6, L=700):
    H = build_reservoir_H(**CRY)
    out = []
    for tag, qs in SCENARIOS:
        acc = {"5": [], "8": []}
        for sd in range(n_seeds):
            s, sp = _inputs(sd, L)
            R = run_selective(s, H, q_per_nucleus=qs, carrier="product")
            acc["5"].append(memory_and_ipc(R["YS_t"], sp)["IPC_total"])
            acc["8"].append(memory_and_ipc(R["cidnp"], sp)["IPC_total"])
        out.append(dict(scenario=tag, IPC_5ch=float(np.mean(acc["5"])),
                        IPC_8ch=float(np.mean(acc["8"]))))
        print(f"  {tag:38s} 5ch={out[-1]['IPC_5ch']:.3f}  8ch={out[-1]['IPC_8ch']:.3f}",
              flush=True)
    f5, f8 = out[-1]["IPC_5ch"], out[-1]["IPC_8ch"]
    for r in out:
        r["excess_5ch"] = r["IPC_5ch"] - f5
        r["excess_8ch"] = r["IPC_8ch"] - f8
    full, prot = out[0], out[1]
    frac = prot["excess_8ch"] / full["excess_8ch"] if full["excess_8ch"] else float("nan")
    print(f"  -> proton-only retains {frac:.1%} of the excess "
          f"(survivor register gave 29%)")
    print(f"  -> proton-only kinetics excess = {prot['excess_5ch']:+.3f} "
          f"(survivor register gave +0.000)")
    return _save("audit3_nuclide_product", dict(rows=out, proton_fraction=float(frac)))


# ── 2. turnover desynchronisation ────────────────────────────────────
def desync_product(jitters=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0), n_copies=64, n_seeds=3,
                   L=700, washout=80):
    H = build_reservoir_H(**CRY)
    live = {sd: _traces(_inputs(sd, L)[0], H, "product") for sd in range(n_seeds)}
    q1 = {}
    for sd in range(n_seeds):
        R = run_routes_v2(_inputs(sd, L)[0], H, sample=tuple(range(1, N_T + 1)),
                          carrier="product", washout=0, q_nuc=1.0)
        q1[sd] = (R["YS_t"], R["cidnp"][:, N_T:])

    def pool(yS_full, cid, jit, seed):
        n_cyc = yS_full.shape[0]
        rng = np.random.default_rng(500 + seed)
        off = rng.uniform(-jit, jit, n_copies) if jit > 0 else np.zeros(n_copies)
        p5 = np.zeros((n_cyc, len(SAMPLE))); p8 = np.zeros((n_cyc, len(SAMPLE) + 3))
        for d in off:
            cs = int(np.floor(d)); frac = d - cs
            idx = np.array(SAMPLE) - 1 - frac * N_T
            lo = np.clip(np.floor(idx).astype(int), 0, N_T - 1)
            hi = np.clip(lo + 1, 0, N_T - 1)
            w = np.clip(idx - lo, 0.0, 1.0)
            k = np.clip(np.arange(n_cyc) - cs, 0, n_cyc - 1)
            tr = (1 - w) * yS_full[k][:, lo] + w * yS_full[k][:, hi]
            p5 += tr; p8[:, :len(SAMPLE)] += tr; p8[:, len(SAMPLE):] += cid[k]
        return p5 / n_copies, p8 / n_copies

    out = []
    for jit in jitters:
        rec = {"jitter_cycles": jit}
        for tag, cache in (("live", live), ("floor_q1", q1)):
            a5, a8 = [], []
            for sd in range(n_seeds):
                s = _inputs(sd, L)[0]; sp = s[washout:]
                p5, p8 = pool(cache[sd][0], cache[sd][1], jit, sd)
                a5.append(memory_and_ipc(p5[washout:], sp)["IPC_total"])
                a8.append(memory_and_ipc(p8[washout:], sp)["IPC_total"])
            rec[f"IPC_5ch_{tag}"] = float(np.mean(a5))
            rec[f"IPC_8ch_{tag}"] = float(np.mean(a8))
        rec["excess_5ch"] = rec["IPC_5ch_live"] - rec["IPC_5ch_floor_q1"]
        rec["excess_8ch"] = rec["IPC_8ch_live"] - rec["IPC_8ch_floor_q1"]
        out.append(rec)
        print(f"  jitter={jit:4.2f}  excess 5ch={rec['excess_5ch']:+.3f}"
              f"  8ch={rec['excess_8ch']:+.3f}", flush=True)
    return _save("audit3_desync_product", out)


# ── 3. anisotropy ────────────────────────────────────────────────────
def anisotropy_product(etas=(0.0, 0.5, 1.0, 2.0), n_seeds=3, L=700):
    dims = (2,) * 5
    spins = (0.5,) * 5
    out = []
    for eta in etas:
        def ax(a):
            return (a * (1 - eta), a * (1 - eta), a * (1 + 2 * eta))
        cpl = [(0, 2, ax(CRY["A_e1_a"])), (0, 3, ax(CRY["A_e1_b"])),
               (1, 4, ax(CRY["A_e2_a"]))]
        H = build_H(dims, spins, CRY["B_tesla"], CRY["J"], cpl)
        live, floor = [], []
        for sd in range(n_seeds):
            s = _inp(sd, L)
            _, r8 = _capacities(s, dims, spins, H, carrier="product")
            _, f8 = _capacities(s, dims, spins, H, carrier="product", q_nuc=1.0)
            live.append(r8["IPC_total"]); floor.append(f8["IPC_total"])
        row = dict(eta=eta, IPC_8ch=float(np.mean(live)),
                   floor_8ch=float(np.mean(floor)),
                   excess_8ch=float(np.mean(live) - np.mean(floor)))
        out.append(row)
        print(f"  eta={eta:.1f}  IPC={row['IPC_8ch']:.3f}  floor={row['floor_8ch']:.3f}"
              f"  excess={row['excess_8ch']:.3f}", flush=True)
    base = out[0]["excess_8ch"]
    for r in out:
        r["fraction_of_isotropic"] = float(r["excess_8ch"] / base) if base else None
    print(f"  -> eta=0.5 retains {out[1]['fraction_of_isotropic']:.1%} "
          f"(survivor register gave 32%)")
    return _save("audit3_anisotropy_product", out)


ALL = {"nuclide": nuclide_product, "desync": desync_product,
       "aniso": anisotropy_product}

if __name__ == "__main__":
    import sys, time
    for k in (sys.argv[1:] or list(ALL)):
        print(f"\n=== {k} (product register) ===", flush=True)
        t0 = time.time(); ALL[k]()
        print(f"  ({time.time()-t0:.0f} s)", flush=True)
