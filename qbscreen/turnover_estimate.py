#!/usr/bin/env python3
"""What sets the turnover interval, and can a neuron supply it?

The turnover interval is the reservoir's clock, and it has never been measured
for a neuronal cryptochrome. It cannot be derived from first principles either --
it depends on what drives the cycle. But it CAN be bounded, because there are only
two candidate drives and both have measured rate constants.

    LIGHT-DRIVEN   turnover rate = sigma * Phi * quantum yield
                   sigma from the measured flavin extinction coefficient,
                   Phi from the photon flux actually available.
    REACTION-DRIVEN turnover rate = enzyme turnover number k_cat.

The light-driven branch has a problem the manuscript never confronts: a brain is
dark. This module quantifies how dark, and what that does to the register.
"""

import json
import os
import numpy as np

OUT = "simulation_results/panel"
NA = 6.02214076e23

# Measured flavin absorption: FAD has eps ~ 11,300 /M/cm at its 450 nm maximum.
EPS_FAD_450 = 11300.0

# Photon fluxes (photons /cm^2 /s) in the 400-500 nm band where FAD absorbs.
# PAR values converted from the standard umol/m^2/s; the blue fraction of
# daylight PAR is roughly a quarter, which is applied here.
FLUX = {
    "full sunlight, at the surface": 3.0e16,
    "overcast daylight": 3.0e15,
    "bright indoor lighting": 3.0e13,
    "moonlight": 3.0e10,
    "inside the skull (blue, ~1 cm of tissue)": 3.0e16 * 1e-9,
}


def cross_section_cm2(eps_M_cm=EPS_FAD_450):
    """Absorption cross-section per molecule from the molar extinction coefficient."""
    return 2.303 * eps_M_cm * 1e3 / NA


def light_driven(quantum_yield=0.1):
    sigma = cross_section_cm2()
    rows = []
    for name, phi in FLUX.items():
        k = sigma * phi * quantum_yield
        rows.append(dict(condition=name, flux_photons_cm2_s=phi,
                         k_turnover_s=float(k),
                         T_turnover_s=float(1.0 / k) if k > 0 else float("inf")))
    print(f"  absorption cross-section = {sigma:.3e} cm^2 "
          f"(eps = {EPS_FAD_450:.0f} /M/cm), quantum yield = {quantum_yield}")
    print(f"    {'condition':44s} {'k (1/s)':>12}  {'T_turnover':>14}")
    for r in rows:
        T = r["T_turnover_s"]
        s = f"{T:.3g} s" if T < 3.15e7 else f"{T/3.15e7:.3g} yr"
        print(f"    {r['condition']:44s} {r['k_turnover_s']:>12.3e}  {s:>14}")
    return rows


# Enzyme turnover numbers span a wide but measured range; flavoenzymes typically
# sit in the 1-1000 /s band.
KCAT = {"slow flavoenzyme": 1.0, "typical flavoenzyme": 50.0, "fast flavoenzyme": 1000.0}


def reaction_driven():
    rows = [dict(enzyme=k, k_cat_s=v, T_turnover_s=1.0 / v) for k, v in KCAT.items()]
    print(f"\n  reaction-driven alternative")
    print(f"    {'enzyme class':28s} {'k_cat (1/s)':>12}  {'T_turnover':>12}")
    for r in rows:
        print(f"    {r['enzyme']:28s} {r['k_cat_s']:>12.0f}  {r['T_turnover_s']:>12.3g} s")
    return rows


def combine(relax_path=f"{OUT}/open5_relaxation_estimate.json"):
    """Put the two unmeasured quantities together: for each candidate turnover
    interval and each product mobility, is the register still there?"""
    with open(relax_path) as f:
        rel = json.load(f)
    scan = rel["tau_c_scan"]
    print(f"\n  register survival q = 1 - exp(-T_turnover/T1), proton at 50 uT")
    print(f"    {'product':34s} {'T1 (s)':>9} " +
          "".join(f"{t:>11}" for t in ("1 ms", "10 ms", "100 ms", "1 s")))
    rows = []
    for r in scan:
        tc, T1 = r["tau_c_ns"], r["T1_proton_s"]
        label = ("protein-bound (60 kDa)" if tc > 10 else
                 "loosely tethered" if tc > 0.4 else
                 "small, freely tumbling")
        qs = [1 - np.exp(-Td / T1) for Td in (1e-3, 1e-2, 1e-1, 1.0)]
        rows.append(dict(tau_c_ns=tc, regime=label, T1_s=T1,
                         q=[float(x) for x in qs]))
        print(f"    {label + f' (tau_c={tc:g} ns)':34s} {T1:>9.3g} " +
              "".join(f"{q:>11.3f}" for q in qs))
    return rows




def feasible_region(min_memory=0.5, band=(1e-2, 1.0),
                    taus_ns=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.2)):
    """Can ANY (product mobility, turnover rate) satisfy all three requirements?

        (a) register reuse       q = 1 - exp(-T_turn/T1) must stay small
        (b) relaxation           T1 from the validated calculator at 50 uT
        (c) neural-band horizon  (MC(q) - C0) * T_turn inside `band`

    MC(q) is interpolated from the paper's own register-reuse curve, so no new
    assumption enters beyond the relaxation calculation.

    min_memory guards against a numerical trap: MC(q=1) = 1.0018, and that 0.0018
    residue -- far below the seed SD of about 0.12 -- multiplied by a long turnover
    interval manufactures a spurious horizon. We require the retained memory to
    exceed roughly four times the seed noise before counting it.
    """
    from qbscreen.relaxation_estimate import dipolar_T1, GEOM
    with open("simulation_results/register_reuse.json") as f:
        reuse = json.load(f)
    qs = np.array([r["q_nuc"] for r in reuse])
    mcs = np.array([r["MC_8"] for r in reuse])
    g = GEOM["Trp H-beta (CH2, geminal partner)"]
    C0 = 1.0
    Tds = np.logspace(-4, 1, 6000)

    rows = []
    print(f"\n  feasible region (retained memory must exceed {min_memory})")
    print(f"    {'tau_c (ns)':>11} {'T1 (s)':>10} {'T_turn window':>22} {'max horizon':>13}")
    for tc_ns in taus_ns:
        T1 = dipolar_T1(g["r"], tc_ns * 1e-9, 50e-6, n_partners=g["n"])
        ok, best = [], 0.0
        for Td in Tds:
            m = float(np.interp(1 - np.exp(-Td / T1), qs, mcs)) - C0
            if m <= min_memory:
                continue
            h = m * Td
            best = max(best, h)
            if band[0] <= h <= band[1]:
                ok.append(Td)
        w = f"{min(ok)*1e3:.2g} - {max(ok)*1e3:.3g} ms" if ok else "NONE"
        rows.append(dict(tau_c_ns=tc_ns, T1_s=float(T1), window=w,
                         max_horizon_s=float(best), feasible=bool(ok)))
        print(f"    {tc_ns:>11.3g} {T1:>10.3g} {w:>22} {best*1e3:>10.3g} ms")

    viable = [r for r in rows if r["feasible"]]
    if viable:
        tmax = max(r["tau_c_ns"] for r in viable)
        pb = [r for r in rows if r["tau_c_ns"] == 15.2][0]
        print(f"\n    feasible only for tau_c <= {tmax:g} ns; a 60 kDa protein-bound")
        print(f"    nucleus has tau_c = 15.2 ns and reaches at most "
              f"{pb['max_horizon_s']*1e3:.2g} ms -- below the band.")
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out = {"light_driven": light_driven(), "reaction_driven": reaction_driven()}
    out["register_survival"] = combine()
    out["feasible_region"] = feasible_region()
    with open(f"{OUT}/open5_turnover_estimate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  [checkpoint] {OUT}/open5_turnover_estimate.json")
