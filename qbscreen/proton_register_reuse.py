#!/usr/bin/env python3
"""The register-reuse curve for the register the biology actually leaves.

The ceiling and the critical tau_c in the main text feed the proton T1 into
MC(q) measured with ALL THREE nuclei relaxing together and the survivor carried
over. The biological reading, however, keeps only the tryptophan proton (the two
flavin 14N relax in sub-microseconds) and hands the PRODUCT register to the next
turnover. Those three conditions were each checked separately; this module
imposes them at once: 14N wiped every cycle, product carried over, and the
proton depolarised with probability q_H swept through the same grid the refined
scan uses. The same ceiling / boundary machinery is then run on that curve.
"""

import json
import os
import numpy as np

from qbscreen.reservoir import build_reservoir_H, memory_and_ipc
from qbscreen.readout_routes import CRY
from qbscreen.panel_response import _inputs, _save
from qbscreen.nuclide_register import run_selective

OUT = "simulation_results/panel"
QS = tuple(np.round(np.concatenate([
    [0.0, 0.1, 0.3, 0.5, 0.7],
    np.arange(0.80, 0.9601, 0.02),
    [0.97, 0.98, 0.99, 1.0]]), 4))


def sweep(n_seeds=6, L=700, carriers=("product", "survivor"), qs=QS):
    H = build_reservoir_H(**CRY)
    out = {}
    for carrier in carriers:
        rows = []
        print(f"\n  proton-only register, carrier={carrier} ({len(qs)} q-points, {n_seeds} seeds)")
        for q in qs:
            mc, ipc = [], []
            for sd in range(n_seeds):
                s, sp = _inputs(sd, L)
                R = run_selective(s, H, q_per_nucleus=(1.0, 1.0, float(q)), carrier=carrier)
                r8 = memory_and_ipc(R["cidnp"], sp)
                mc.append(r8["MC"]); ipc.append(r8["IPC_total"])
            rows.append(dict(q_H=float(q), MC_8=float(np.mean(mc)), MC_8_sd=float(np.std(mc, ddof=1)),
                             IPC_8=float(np.mean(ipc)), IPC_8_sd=float(np.std(ipc, ddof=1)),
                             n_seeds=n_seeds))
            print(f"  {q:>6.3f} MC_8={rows[-1]['MC_8']:.4f} sd={rows[-1]['MC_8_sd']:.4f}"
                  f" IPC_8={rows[-1]['IPC_8']:.4f}", flush=True)
            out[carrier] = rows
            _save("open8_proton_register_reuse", out)
    return out


def ceiling_and_boundary(curve, min_memory=0.5, band=(1e-2, 1.0)):
    """max_Td (MC(q_H(Td)) - C0) Td and the bisected critical tau_c, for each
    candidate proton (Trp H-beta, flavin 8-alpha methyl), dry and wet.

    C0 is the measured MC at q_H = 1 (the memoryless floor of this register),
    not the assumed 1.0; it is reported. S2 / tau_int are passed through exactly
    as turnover_estimate.feasible_region passes them.
    """
    from qbscreen.relaxation_estimate import (dipolar_T1, tau_rot2_protein, GEOM,
                                              sum_ext_for, bath_ratio_from_water)
    qs = np.array([r["q_H"] for r in curve]); mcs = np.array([r["MC_8"] for r in curve])
    C0 = float(mcs[-1])
    f_bath = bath_ratio_from_water()
    Tds = np.logspace(-4, 1, 6000)
    tau_protein = tau_rot2_protein(60.0) * 1e9

    def horizon_of(g, tc_ns, sx):
        T1 = dipolar_T1(g["r"], tc_ns * 1e-9, 50e-6, n_partners=g["n"], S2=g.get("S2", 1.0),
                        tau_int=g.get("tau_int"), sum_ext=sx, S2_ext=g.get("S2_ext", 1.0))
        m = np.interp(1 - np.exp(-Tds / T1), qs, mcs) - C0
        h = np.where(m > min_memory, m * Tds, 0.0)
        ok = Tds[(h >= band[0]) & (h <= band[1])]
        return float(T1), float(h.max()), ok

    def critical(g, sx, lo=0.1, hi=200.0, tol=1e-4):
        if len(horizon_of(g, lo, sx)[2]) == 0:
            return None
        if len(horizon_of(g, hi, sx)[2]):
            return hi
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if len(horizon_of(g, mid, sx)[2]) else (lo, mid)
        return 0.5 * (lo + hi)

    res = dict(C0_measured=C0, min_memory=min_memory, tau_protein_ns=float(tau_protein), per_nucleus={})
    for name in ("Trp H-beta (CH2, geminal partner)", "flavin 8-alpha CH3 (intra-methyl)"):
        g = GEOM[name]; out = {}
        for tag, sx in (("dry", 0.0), ("wet", sum_ext_for(g, f_bath))):
            T1, best, ok = horizon_of(g, tau_protein, sx)
            tc = critical(g, sx)
            out[tag] = dict(T1_protein_ms=T1 * 1e3, ceiling_ms=best * 1e3,
                            in_band=bool(best >= band[0]),
                            window_at_protein="NONE" if len(ok) == 0 else
                            f"{ok.min()*1e3:.3g} - {ok.max()*1e3:.3g} ms",
                            tau_crit_ns=tc, speedup=(tau_protein / tc) if tc else None)
            print(f"  {name:36s} {tag}: T1={T1*1e3:.3f} ms ceiling={best*1e3:.3f} ms"
                  f" tau_crit={tc and round(tc, 3)} ns speedup={out[tag]['speedup'] and round(out[tag]['speedup'], 2)}")
        res["per_nucleus"][name] = out
    return res


if __name__ == "__main__":
    import time
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    out = sweep()
    for carrier in list(out):
        print(f"\n  ceiling / boundary, carrier={carrier}")
        out[f"{carrier}_criterion"] = ceiling_and_boundary(out[carrier])
    _save("open8_proton_register_reuse", out)
    print(f"  ({time.time()-t0:.0f} s)")
