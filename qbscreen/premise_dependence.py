"""How much of the cryptochrome-point result depends on setting J = D = 0.

The cryptochrome operating point is run with no electron-electron coupling at
all: J = 0 and no dipolar term. The source cited for the separated pair,
Efimova and Hore (Biophys. J. 94, 1565, 2008), reports for FAD-W324 at
r = 1.90 nm a point-dipole coupling D = -2.78e3 uT / r^3 = -11.36 MHz and an
exchange J0 e^{-beta r} whose magnitude spans 1.26-126 MHz with undetermined
sign, the two partially cancelling near 2 nm. This module reruns every
cryptochrome-point calculation that the paper quotes with those values in the
Hamiltonian and nothing else changed -- same protocol, estimator, seeds,
sequence lengths -- so that the dependence of each printed number on the
premise can be stated as a number.

D is anisotropic: every case is run with the inter-radical axis parallel to B
("par") and perpendicular to it ("perp"). J_gap is the S-T gap in the
Efimova-Hore convention, entered here as the J of J S1.S2 (so J_gap = -2 J_EH).
Both signs are run; neither is preferred.

    python -m qbscreen.premise_dependence            # everything (~40 min)
    python -m qbscreen.premise_dependence capacity   # the cheap table only

Writes simulation_results/panel/premise_dependence.json incrementally.
"""
import json
import pathlib
import sys
import warnings

import numpy as np

from qbscreen.final_numbers import CRY as CRY_F, _observable_set, run_corr_obs
from qbscreen.nuclide_register import run_selective
from qbscreen.readout_routes import (CRY, accumulate, run_heterogeneous,
                                     run_routes)
from qbscreen.reservoir import build_reservoir_H, memory_and_ipc

OUT = pathlib.Path("simulation_results/panel/premise_dependence.json")
R_NM = 1.90
D_MHZ = -2.78e3 / R_NM ** 3 / 1e3 * 1e3 / 1e3 * 1e3  # uT -> MHz below
D_MHZ = -77.91 / R_NM ** 3                             # = -2.78e3 uT/nm^3 * 28.02 MHz/mT
J_MID = 12.6                                           # |J0| mid-range at 1.90 nm, MHz

# (label, J, D, axis). "reference" is the paper's premise.
CASES = [
    ("reference  J=0, D=0",        0.0,    0.0,   "z"),
    ("D(1.90nm) par, J=0",         0.0,    D_MHZ, "z"),
    ("D(1.90nm) perp, J=0",        0.0,    D_MHZ, "x"),
    ("D par, J_gap=-12.6",        -J_MID,  D_MHZ, "z"),
    ("D par, J_gap=+12.6",        +J_MID,  D_MHZ, "z"),
    ("D perp, J_gap=-12.6",       -J_MID,  D_MHZ, "x"),
    ("D perp, J_gap=+12.6",       +J_MID,  D_MHZ, "x"),
]
SCALE = [("J=+1.26 only", 1.26, 0.0, "z"), ("J=+12.6 only", 12.6, 0.0, "z"),
         ("J=+126 only", 126.0, 0.0, "z"),
         ("D at r=2.2nm", 0.0, -77.91 / 2.2 ** 3, "z"),
         ("D at r=2.5nm", 0.0, -77.91 / 2.5 ** 3, "z"),
         ("D at r=3.5nm", 0.0, -77.91 / 3.5 ** 3, "z"),
         ("D at r=5.0nm", 0.0, -77.91 / 5.0 ** 3, "z")]
ROUTE_KEYS = ("YS_end", "YS_t", "SandT_end", "SandT_t", "cidnp")


def _H(J, D, axis):
    return build_reservoir_H(**{**CRY, "J": J}, D=D, D_axis=axis)


def _load():
    return json.load(open(OUT)) if OUT.exists() else {}


def _save(d):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(d, f, indent=1)


def _ms(v):
    return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), n=len(v))


def capacity(n_seeds=6, L=900, T2e=1000.0):
    """No-recombination capacity, full observable set (final_numbers F5 protocol)."""
    d = _load(); d["capacity"] = {}
    print("[capacity] no recombination, full observables, seeds 800+")
    for lab, J, D, ax in CASES + SCALE:
        H = build_reservoir_H(**{**CRY_F, "J": J}, D=D, D_axis=ax)
        v = []
        for sd in range(n_seeds):
            s = np.random.default_rng(800 + sd).uniform(0, 1, L + 100)
            X = run_corr_obs(s, H, 1.0, T2e, _observable_set("full"))
            v.append(memory_and_ipc(X, s[100:])["IPC_total"])
        d["capacity"][lab] = dict(J=J, D=D, axis=ax, IPC=_ms(v))
        print(f"  {lab:26s} IPC = {np.mean(v):.3f} +/- {np.std(v, ddof=1):.3f}", flush=True)
    _save(d)


def routes(n_seeds=6, L=700, cases=CASES):
    """The seven readout routes and each one's memoryless floor (q = 1)."""
    d = _load(); d.setdefault("routes", {})
    for lab, J, D, ax in cases:
        H = _H(J, D, ax)
        acc = {k: [] for k in ROUTE_KEYS + ("hetero_tau", "YS_t_accum")}
        flo = {k: [] for k in ROUTE_KEYS + ("YS_t_accum",)}
        for sd in range(n_seeds):
            s = np.random.default_rng(sd).uniform(0, 1, L + 80); sp = s[80:]
            R = run_routes(s, H)
            for k in ROUTE_KEYS:
                acc[k].append(memory_and_ipc(R[k], sp)["IPC_total"])
            acc["YS_t_accum"].append(memory_and_ipc(accumulate(R["YS_t"]), sp)["IPC_total"])
            acc["hetero_tau"].append(memory_and_ipc(run_heterogeneous(s, H), sp)["IPC_total"])
            Rw = run_routes(s, H, q_nuc=1.0)
            for k in ROUTE_KEYS:
                flo[k].append(memory_and_ipc(Rw[k], sp)["IPC_total"])
            flo["YS_t_accum"].append(memory_and_ipc(accumulate(Rw["YS_t"]), sp)["IPC_total"])
        d["routes"][lab] = dict(J=J, D=D, axis=ax,
                                IPC={k: _ms(v) for k, v in acc.items()},
                                floor={k: _ms(v) for k, v in flo.items()})
        print(f"[routes] {lab}")
        for k in acc:
            f = flo.get(k)
            print(f"  {k:11s} IPC {np.mean(acc[k]):.3f}+/-{np.std(acc[k], ddof=1):.3f}"
                  + (f"  floor {np.mean(f):.3f}" if f else ""), flush=True)
        _save(d)


def clock(n_seeds=6, L=700, cases=CASES[1:2] + CASES[3:4], T1n_s=1.0,
          T_d_list=(1e-6, 1e-3, 1e-2, 1e-1)):
    """Turnover clock: MC(8 ch) against the pause, register relaxing with T1n."""
    d = _load(); d.setdefault("clock", {})
    for lab, J, D, ax in cases:
        H = _H(J, D, ax); rows = []
        for T_d in T_d_list:
            q = 1.0 - np.exp(-T_d / T1n_s); v5, v8 = [], []
            for sd in range(n_seeds):
                s = np.random.default_rng(sd).uniform(0, 1, L + 80); sp = s[80:]
                R = run_routes(s, H, q_nuc=float(q))
                v5.append(memory_and_ipc(R["YS_t"], sp)["MC"])
                v8.append(memory_and_ipc(R["cidnp"], sp)["MC"])
            rows.append(dict(T_d_s=T_d, q_nuc=float(q), MC_5ch=_ms(v5), MC_8ch=_ms(v8)))
            print(f"[clock] {lab}  T_d={T_d:g}  MC_8 = {np.mean(v8):.3f}+/-{np.std(v8, ddof=1):.3f}", flush=True)
        d["clock"][lab] = dict(J=J, D=D, axis=ax, rows=rows); _save(d)


def reuse(n_seeds=6, L=700, cases=CASES[1:2], qs=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0)):
    """Register-reuse criterion: MC and IPC against the depolarisation q."""
    d = _load(); d.setdefault("reuse", {})
    for lab, J, D, ax in cases:
        H = _H(J, D, ax); rows = []
        for q in qs:
            acc = {k: [] for k in ("MC_5", "IPC_5", "MC_8", "IPC_8")}
            for sd in range(n_seeds):
                s = np.random.default_rng(sd).uniform(0, 1, L + 80); sp = s[80:]
                R = run_routes(s, H, q_nuc=q)
                r5 = memory_and_ipc(R["YS_t"], sp); r8 = memory_and_ipc(R["cidnp"], sp)
                acc["MC_5"].append(r5["MC"]); acc["IPC_5"].append(r5["IPC_total"])
                acc["MC_8"].append(r8["MC"]); acc["IPC_8"].append(r8["IPC_total"])
            rows.append(dict(q_nuc=float(q), **{k: _ms(v) for k, v in acc.items()}))
            print(f"[reuse] {lab}  q={q:.2f}  MC_8 = {np.mean(acc['MC_8']):.3f}  IPC_5 = {np.mean(acc['IPC_5']):.3f}", flush=True)
        d["reuse"][lab] = dict(J=J, D=D, axis=ax, rows=rows); _save(d)


def proton_only(n_seeds=6, L=700, cases=CASES[1:2] + CASES[3:4]):
    """Which nuclei hold the register: wipe both 14N every cycle, keep the proton."""
    d = _load(); d.setdefault("proton_only", {})
    for lab, J, D, ax in cases:
        H = _H(J, D, ax); acc = {"YS_t": [], "cidnp": []}
        for sd in range(n_seeds):
            s = np.random.default_rng(sd).uniform(0, 1, L + 80); sp = s[80:]
            R = run_selective(s, H, q_per_nucleus=(1.0, 1.0, 0.0))
            for k in acc:
                acc[k].append(memory_and_ipc(R[k], sp)["IPC_total"])
        d["proton_only"][lab] = dict(J=J, D=D, axis=ax, IPC={k: _ms(v) for k, v in acc.items()})
        print(f"[proton_only] {lab}  YS_t {np.mean(acc['YS_t']):.3f}  cidnp {np.mean(acc['cidnp']):.3f}", flush=True)
        _save(d)


def horizon(band_edge_s=1e-2):
    """Ceiling and band-opening correlation time from the SAME eight-point MC(q)
    scan for both premises, so the comparison is like for like. The main text
    quotes the refined scan for the reference premise; this block is the
    coarse-scan pair that the premise comparison is stated in terms of."""
    d = _load()
    rel = json.load(open("simulation_results/panel/open5_relaxation_estimate.json"))
    tov = json.load(open("simulation_results/panel/open5_turnover_estimate.json"))
    T1_dry = rel["prediction"]["T1_proton_geomagnetic_s"]
    f = tov["critical_tau_c"]["bath_f"]; T1_wet = T1_dry / (1.0 + f)
    tau_p = tov["critical_tau_c"]["tau_protein_ns"]
    ref = json.load(open("simulation_results/register_reuse.json"))
    curves = {"reference": ([r["q_nuc"] for r in ref], [r["MC_8"] for r in ref])}
    for lab, v in d["reuse"].items():
        curves[lab] = ([r["q_nuc"] for r in v["rows"]], [r["MC_8"]["mean"] for r in v["rows"]])

    def ceiling(q, m, T1):
        Td = np.logspace(-5, 0, 4000); qq = 1 - np.exp(-Td / T1)
        return float(((np.interp(qq, q, m) - 1.0) * Td).max())

    def boundary(q, m, T1_at_P):
        taus = np.logspace(-1, 2, 3000)
        ok = taus[np.array([ceiling(q, m, T1_at_P * tau_p / t) for t in taus]) >= band_edge_s]
        return float(ok.max()) if ok.size else float("nan")

    out = dict(note="eight-point MC(q) scan, linear interpolation in q; T1 at tau_c = tau_protein "
                    "from open5 (intramolecular, and with the bath); band edge 10 ms",
               T1_dry_s=T1_dry, T1_wet_s=T1_wet, tau_protein_ns=tau_p)
    for lab, (q, m) in curves.items():
        out[lab] = dict(ceil_dry=ceiling(q, m, T1_dry) * 1e3, ceil_wet=ceiling(q, m, T1_wet) * 1e3,
                        tau_dry=boundary(q, m, T1_dry), tau_wet=boundary(q, m, T1_wet))
        o = out[lab]
        print(f"[horizon] {lab:24s} ceiling {o['ceil_dry']:.2f}/{o['ceil_wet']:.2f} ms  boundary "
              f"{o['tau_dry']:.2f}/{o['tau_wet']:.2f} ns  (x{tau_p/o['tau_dry']:.2f}/x{tau_p/o['tau_wet']:.2f})", flush=True)
    d["horizon_coarse_grid"] = out; _save(d)


if __name__ == "__main__":
    what = sys.argv[1:] or ["capacity", "routes", "proton_only", "clock", "reuse", "horizon"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for w in what:
            globals()[w]()
    print("[done]", OUT)
