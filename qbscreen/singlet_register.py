#!/usr/bin/env python3
"""The register handed on by the singlet product alone.

The product register pools both recombination channels. In a flavin--tryptophan
pair only singlet recombination returns the ground state that is reduced again
in the next turnover; the triplet channel leaves the cycle. This module hands on
the nuclei of the singlet product only (carrier="singlet") and repeats, on that
register, the route capacities, the proton-register relaxation sweep and the
ceiling / boundary, once with every singlet product re-used (conditional) and
once with the molecule budget closed: only the fraction of molecules that
recombined through the singlet channel by tau hands a register on.
"""
import os
import numpy as np

from qbscreen.reservoir import build_reservoir_H, memory_and_ipc
from qbscreen.readout_routes import CRY
from qbscreen.panel_response import _inputs, _save
from qbscreen.nuclide_register import run_selective
from qbscreen.proton_register_reuse import sweep, ceiling_and_boundary

NAME = "open18_singlet_register"


def routes(n_seeds=6, L=700):
    H = build_reservoir_H(**CRY)
    out = {}
    for register, qn in (("full", (0.0, 0.0, 0.0)), ("proton_only", (1.0, 1.0, 0.0))):
        live = {"YS_t": [], "cidnp": []}; floor = {"YS_t": [], "cidnp": []}; yS = []; surv = []
        for sd in range(n_seeds):
            s, sp = _inputs(sd, L)
            R = run_selective(s, H, q_per_nucleus=qn, carrier="singlet")
            F = run_selective(s, H, q_per_nucleus=(1.0, 1.0, 1.0), carrier="singlet")
            for k in live:
                live[k].append(memory_and_ipc(R[k], sp)["IPC_total"])
                floor[k].append(memory_and_ipc(F[k], sp)["IPC_total"])
            yS.append(float(np.mean(R["YS_end"]))); surv.append(float(np.mean(R["survival"])))
        row = dict(register=register, n_seeds=n_seeds, singlet_yield_at_tau=float(np.mean(yS)),
                   unreacted_fraction_at_tau=float(np.mean(surv)))
        for k, name in (("YS_t", "kinetics_5ch"), ("cidnp", "cidnp_8ch")):
            row[name] = float(np.mean(live[k])); row[name + "_sd"] = float(np.std(live[k], ddof=1))
            row[name + "_floor"] = float(np.mean(floor[k]))
            row[name + "_excess"] = row[name] - row[name + "_floor"]
        out[register] = row
        print(f"  {register:12s} singlet yield={row['singlet_yield_at_tau']:.3f}"
              f"  5ch excess {row['kinetics_5ch_excess']:+.3f}  8ch excess {row['cidnp_8ch_excess']:+.3f}", flush=True)
    return out


if __name__ == "__main__":
    import time
    t0 = time.time(); os.makedirs("simulation_results/panel", exist_ok=True)
    out = sweep(carriers=("singlet",), name=NAME)
    out["routes"] = routes()
    f_S = out["routes"]["proton_only"]["singlet_yield_at_tau"]
    print("\n  ceiling / boundary, every singlet product re-used")
    out["singlet_criterion"] = ceiling_and_boundary(out["singlet"])
    print(f"\n  ceiling / boundary, budget closed (f_S = {f_S:.3f})")
    out["singlet_selected_criterion"] = ceiling_and_boundary(out["singlet"], reinjection_fraction=f_S)
    _save(NAME, out)
    print(f"  ({time.time()-t0:.0f} s)")
