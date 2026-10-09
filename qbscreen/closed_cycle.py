#!/usr/bin/env python3
"""The reaction cycle closed over the molecule-number budget.

Both registers used so far are conditional: the survivor is the pair that had
NOT recombined by tau, renormalised, and the product is the nuclear state of
the molecules that HAD, renormalised. At kS = 1, kT = 0.2 us^-1 and tau = 1 us
the unreacted fraction at tau is not small (it lies between e^-1 and e^-0.2),
so neither register is the register of the whole population. The "closed"
carrier sums the two branches with their molecule-number weights before
renormalising: every molecule hands its register on, whichever branch it took.
A regeneration probability p < 1 (some molecules are replaced by fresh ones)
is the register depolarisation q = 1 - p already scanned in the paper.
"""
import os
import numpy as np
from qbscreen.reservoir import build_reservoir_H, memory_and_ipc
from qbscreen.readout_routes import CRY
from qbscreen.panel_response import _inputs, _save, run_routes_v2
from qbscreen.nuclide_register import run_selective

OUT = "simulation_results/panel"


def scan(n_seeds=6, L=700):
    H = build_reservoir_H(**CRY)
    out = {}
    for register, qn in (("full", (0.0, 0.0, 0.0)), ("proton_only", (1.0, 1.0, 0.0))):
        for carrier in ("survivor", "product", "closed"):
            live = {"YS_t": [], "cidnp": []}; floor = {"YS_t": [], "cidnp": []}; surv = []
            for sd in range(n_seeds):
                s, sp = _inputs(sd, L)
                R = run_selective(s, H, q_per_nucleus=qn, carrier=carrier)
                F = run_selective(s, H, q_per_nucleus=(1.0, 1.0, 1.0), carrier=carrier)
                for k in live:
                    live[k].append(memory_and_ipc(R[k], sp)["IPC_total"])
                    floor[k].append(memory_and_ipc(F[k], sp)["IPC_total"])
                surv.append(float(np.mean(R["survival"])))
            row = dict(register=register, carrier=carrier, n_seeds=n_seeds,
                       unreacted_fraction_at_tau=float(np.mean(surv)))
            for k, name in (("YS_t", "kinetics_5ch"), ("cidnp", "cidnp_8ch")):
                row[name] = float(np.mean(live[k])); row[name + "_sd"] = float(np.std(live[k], ddof=1))
                row[name + "_floor"] = float(np.mean(floor[k]))
                row[name + "_excess"] = row[name] - row[name + "_floor"]
            out[f"{register}/{carrier}"] = row
            print(f"  {register:12s} {carrier:9s} unreacted={row['unreacted_fraction_at_tau']:.3f}"
                  f"  5ch {row['kinetics_5ch']:.3f} (floor {row['kinetics_5ch_floor']:.3f})"
                  f"  8ch {row['cidnp_8ch']:.3f} (floor {row['cidnp_8ch_floor']:.3f}) excess {row['cidnp_8ch_excess']:+.3f}", flush=True)
            _save("open12_closed_cycle", out)
    return out


if __name__ == "__main__":
    import time; t0 = time.time(); os.makedirs(OUT, exist_ok=True); scan(); print(f"  ({time.time()-t0:.0f} s)")
