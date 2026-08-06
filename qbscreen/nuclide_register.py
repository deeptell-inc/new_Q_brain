#!/usr/bin/env python3
"""Which nuclei can actually hold the register?

The main text applies one relaxation probability q to the whole nuclear register.
That is not what the relaxation literature implies. In the model the three nuclei
are not equivalent:

    nucleus 1, 2  ->  14N5, 14N10 on FAD      (spin-1, QUADRUPOLAR)
    nucleus 3     ->  H-beta1 on tryptophan   (spin-1/2, dipolar only)

For a nucleus in a ~60 kDa protein (rotational correlation time ~25 ns) at
geomagnetic field, extreme narrowing applies and quadrupolar relaxation of 14N is
fast -- of order 10^-7 s for the measured quadrupole coupling constants of
pyridine-type ring and amide nitrogens -- whereas protein-bound 1H relaxes by the
dipolar mechanism on a millisecond scale. The two differ by roughly five orders of
magnitude, so treating them with a common q is not conservative: it is wrong in
the direction that flatters the model.

This module therefore relaxes each nucleus independently and asks the question the
literature forces: DOES A PROTON-ONLY REGISTER STILL WORK? If it does, the memory
horizon is set by the proton T1 (ms) rather than by the nitrogen T1 (sub-us), and
the biological claim survives in weakened form. If it does not, the claim fails.
"""

import json
import os
import numpy as np

from qbscreen.reservoir import build_reservoir_H, memory_and_ipc
from qbscreen.readout_routes import CRY
from qbscreen.panel_response import run_routes_v2, _inputs, _save
import qbscreen.panel_response as pr

OUT = "simulation_results/panel"


def depolarise_nucleus(nuc, j, q):
    """Fully depolarise nucleus j (0,1,2) of a three-nucleus register with prob q.

    nuc is the (8,8) reduced nuclear density matrix, index order (n1,n2,n3).
    The depolarised branch is Tr_j(rho) (x) I_j/2, which destroys that nucleus's
    populations AND its coherences while leaving the other two untouched.
    """
    if q <= 0:
        return nuc
    R = nuc.reshape(2, 2, 2, 2, 2, 2)
    I2 = np.eye(2, dtype=complex) / 2.0
    if j == 0:
        out = np.einsum("cxycpq,ab->axybpq", R, I2)
    elif j == 1:
        out = np.einsum("xcypcq,ab->xaypbq", R, I2)
    else:
        out = np.einsum("xycpqc,ab->xyapqb", R, I2)
    return (1.0 - q) * nuc + q * out.reshape(8, 8)


def run_selective(inputs, H_mhz, q_per_nucleus=(0.0, 0.0, 0.0), **kw):
    """run_routes_v2 with a per-nucleus relaxation probability.

    Implemented by monkey-patching the register update inside the shared cycle so
    that the rest of the propagation is bit-identical to the published code.
    """
    qs = np.asarray(q_per_nucleus, dtype=float)
    orig = pr._nuc_from_electron_trace

    def patched(rho):
        nuc = orig(rho)
        for j, q in enumerate(qs):
            nuc = depolarise_nucleus(nuc, j, q)
        return nuc

    pr._nuc_from_electron_trace = patched
    try:
        return run_routes_v2(inputs, H_mhz, **kw)
    finally:
        pr._nuc_from_electron_trace = orig


SCENARIOS = [
    ("all three retained (main text)",            (0.0, 0.0, 0.0)),
    ("14N wiped, 1H retained (literature)",       (1.0, 1.0, 0.0)),
    ("1H wiped, 14N retained",                    (0.0, 0.0, 1.0)),
    ("all three wiped (floor)",                   (1.0, 1.0, 1.0)),
]


def nuclide_scan(n_seeds=6, L=700, carrier="survivor"):
    H = build_reservoir_H(**CRY)
    out = []
    for tag, qs in SCENARIOS:
        a5, a8, m8 = [], [], []
        for sd in range(n_seeds):
            s, sp = _inputs(sd, L)
            R = run_selective(s, H, q_per_nucleus=qs, carrier=carrier)
            r5 = memory_and_ipc(R["YS_t"], sp)
            r8 = memory_and_ipc(R["cidnp"], sp)
            a5.append(r5["IPC_total"]); a8.append(r8["IPC_total"]); m8.append(r8["MC"])
        row = dict(scenario=tag, q_per_nucleus=list(qs),
                   IPC_5ch=float(np.mean(a5)), IPC_5ch_sd=float(np.std(a5)),
                   IPC_8ch=float(np.mean(a8)), IPC_8ch_sd=float(np.std(a8)),
                   MC_8ch=float(np.mean(m8)))
        out.append(row)
        print(f"  {tag:38s} IPC 5ch={row['IPC_5ch']:.3f}  8ch={row['IPC_8ch']:.3f}"
              f"  MC8={row['MC_8ch']:.3f}", flush=True)
        _save("open2_nuclide_register", out)
    floor8 = out[-1]["IPC_8ch"]; floor5 = out[-1]["IPC_5ch"]
    for r in out:
        r["excess_5ch"] = r["IPC_5ch"] - floor5
        r["excess_8ch"] = r["IPC_8ch"] - floor8
    print("\n  excess over the all-wiped floor:")
    for r in out:
        print(f"  {r['scenario']:38s} 5ch {r['excess_5ch']:+.3f}   8ch {r['excess_8ch']:+.3f}")
    return _save("open2_nuclide_register", out)


if __name__ == "__main__":
    import time
    t0 = time.time()
    nuclide_scan()
    print(f"  ({time.time()-t0:.0f} s)")
