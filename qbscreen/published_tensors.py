#!/usr/bin/env python3
"""The cryptochrome point with the published hyperfine tensors.

Tensors (mT, full 3x3, one common axis system; the TrpH tensors are rotated
into the FAD frame with the FAD/Trp-342 orientation of DmCry, PDB 4GU5) are
those of Hiscock et al., PNAS 113, 4634 (2016), SI Appendix Tables S1 and S2
(DFT, UB3LYP/EPR-III, by I. Kuprov), which are the set Lee et al. (2014) used.
Spin-1/2 nitrogen throughout; electron-electron coupling off, as in the main
model. The field direction is scanned in that common frame.

  published5  : N5, N10, Trp Hb1          -- the three nuclei of the main model
  published6a : + Trp H1                   -- a second proton on the tryptophan
  published6b : + FAD H6                   -- a proton on the flavin
"""
import os, sys
import numpy as np
from qbscreen.reservoir import memory_and_ipc
from qbscreen.axial_tensor_model import build_H, run, unit, _save, _inp, CRY, MT

T = {  # mT; Hiscock et al. 2016 SI Tables S1 (FAD) and S2 (TrpH)
    "N5":  [[-0.0989, 0.0039, 0.0], [0.0039, -0.0881, 0.0], [0.0, 0.0, 1.7569]],
    "N10": [[-0.0190, -0.0048, 0.0], [-0.0048, -0.0196, 0.0], [0.0, 0.0, 0.6046]],
    "H6":  [[-0.2569, -0.1273, 0.0], [-0.1273, -0.4711, 0.0], [0.0, 0.0, -0.4336]],
    "Hb1": [[1.5808, -0.0453, -0.0506], [-0.0453, 1.5575, 0.0988], [-0.0506, 0.0988, 1.6752]],
    "H1":  [[-0.9920, -0.2091, -0.2003], [-0.2091, -0.2631, 0.2803], [-0.2003, 0.2803, -0.5398]],
}
MODELS = {
    "published5":  [("N5", 0), ("N10", 0), ("Hb1", 1)],
    "published6a": [("N5", 0), ("N10", 0), ("Hb1", 1), ("H1", 1)],
    "published6b": [("N5", 0), ("N10", 0), ("Hb1", 1), ("H6", 0)],
}
DEG = np.pi / 180
DIRS6 = [(0.0, 0.0), (90 * DEG, 0.0), (90 * DEG, 90 * DEG), (54.7356 * DEG, 45 * DEG), (54.7356 * DEG, 135 * DEG), (125.264 * DEG, 45 * DEG)]


def couplings(model):
    return [(e, 2 + k, MT * np.array(T[name])) for k, (name, e) in enumerate(MODELS[model])]


def scan(model, cases, n_seeds, L=700, n_t=96):
    sample = tuple(int(round(n_t * f)) for f in (16 / 96, 36 / 96, 56 / 96, 76 / 96, 1.0))
    nuclei = [n for n, _ in MODELS[model]]; dims = (2,) * (2 + len(nuclei))
    wipe_N = tuple(1.0 if n.startswith("N") else 0.0 for n in nuclei)
    out = {"nuclei": nuclei}
    for name, dirs in cases:
        Hs = [(build_H(couplings(model), CRY["B_tesla"], th, ph, dims=dims), unit(th, ph)) for th, ph in dirs]
        for reg, qn in (("full", (0.0,) * len(nuclei)), ("proton_only", wipe_N)):
            live = {"5": [], "8": []}; floor = {"5": [], "8": []}
            for sd in range(n_seeds):
                s, sp = _inp(sd, L)
                L_ = [run(s, H, q_per_nucleus=qn, direction=n_, dims=dims, n_t=n_t, sample=sample) for H, n_ in Hs]
                F_ = [run(s, H, q_per_nucleus=(1.0,) * len(nuclei), direction=n_, dims=dims, n_t=n_t, sample=sample) for H, n_ in Hs]
                Y = np.mean([y for y, _ in L_], axis=0); C = np.mean([c for _, c in L_], axis=0)
                Yf = np.mean([y for y, _ in F_], axis=0); Cf = np.mean([c for _, c in F_], axis=0)
                live["5"].append(memory_and_ipc(Y, sp)["IPC_total"]); live["8"].append(memory_and_ipc(C, sp)["IPC_total"])
                floor["5"].append(memory_and_ipc(Yf, sp)["IPC_total"]); floor["8"].append(memory_and_ipc(Cf, sp)["IPC_total"])
            row = dict(case=name, register=reg, n_directions=len(dirs), n_seeds=n_seeds, n_channels=5 + len(nuclei), L=L, n_t=n_t)
            for k, nm in (("5", "kinetics_5ch"), ("8", "cidnp")):
                row[nm] = float(np.mean(live[k])); row[nm + "_sd"] = float(np.std(live[k], ddof=1))
                row[nm + "_floor"] = float(np.mean(floor[k])); row[nm + "_excess"] = row[nm] - row[nm + "_floor"]
            out[f"{name}/{reg}"] = row
            print(f"  {model:12s} {name:22s} {reg:12s} 5ch {row['kinetics_5ch']:.3f} (excess {row['kinetics_5ch_excess']:+.3f})"
                  f"  cidnp {row['cidnp']:.3f} (floor {row['cidnp_floor']:.3f}, excess {row['cidnp_excess']:+.3f})", flush=True)
            _save(f"open17_{model}", out)
    return out


if __name__ == "__main__":
    import time; t0 = time.time(); os.makedirs("simulation_results/panel", exist_ok=True)
    model = sys.argv[1] if len(sys.argv) > 1 else "published5"
    if model == "published5":
        cases = [("theta=0", [(0.0, 0.0)]), ("theta=54.7", [(54.7356 * DEG, 0.0)]), ("theta=90", [(90 * DEG, 0.0)]), ("6-direction average", DIRS6)]
        scan(model, cases, n_seeds=6)
    else:
        # six spins cost ~1 s per cycle at n_t = 96: three orthogonal directions,
        # a coarser grid (1% on MC, Sec. S7) and a shorter sequence keep a variant
        # inside a few hours
        DIRS3 = [(0.0, 0.0), (90 * DEG, 0.0), (90 * DEG, 90 * DEG)]
        scan(model, [("3-axis average", DIRS3)], n_seeds=3, L=400, n_t=48)
    print(f"  ({time.time()-t0:.0f} s)")
