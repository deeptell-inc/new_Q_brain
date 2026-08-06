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




def feasible_region(min_memory=0.5, band=(1e-2, 1.0), return_boundaries=False,
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
    from qbscreen.relaxation_estimate import tau_c_protein
    TAU_PROTEIN_NS = tau_c_protein(60.0) * 1e9
    C0 = 1.0
    Tds = np.logspace(-4, 1, 6000)

    def horizon_of(tc_ns, g, sum_ext=0.0):
        """(best horizon, list of in-band turnover intervals) for one nucleus.

        Pulled out of the loop below so that critical_tau_c() bisects on exactly
        the predicate the table prints, rather than on a re-derivation of it.
        S2/tau_int are now passed through: omitting them silently treated every
        candidate as if its dipolar vector were rigid with the protein, which is
        false for the methyl and is why the methyl row never reached the
        conclusion.
        """
        T1 = dipolar_T1(g["r"], tc_ns * 1e-9, 50e-6, n_partners=g["n"],
                        S2=g.get("S2", 1.0), tau_int=g.get("tau_int"),
                        sum_ext=sum_ext)
        ok, best = [], 0.0
        for Td in Tds:
            m = float(np.interp(1 - np.exp(-Td / T1), qs, mcs)) - C0
            if m <= min_memory:
                continue
            h = m * Td
            best = max(best, h)
            if band[0] <= h <= band[1]:
                ok.append(Td)
        return T1, best, ok

    g = GEOM["Trp H-beta (CH2, geminal partner)"]
    rows = []
    print(f"\n  feasible region (retained memory must exceed {min_memory})")
    print(f"    {'tau_c (ns)':>11} {'T1 (s)':>10} {'T_turn window':>22} {'max horizon':>13}")
    for tc_ns in taus_ns:
        T1, best, ok = horizon_of(tc_ns, g)
        w = f"{min(ok)*1e3:.2g} - {max(ok)*1e3:.3g} ms" if ok else "NONE"
        rows.append(dict(tau_c_ns=tc_ns, T1_s=float(T1), window=w,
                         max_horizon_s=float(best), feasible=bool(ok)))
        print(f"    {tc_ns:>11.3g} {T1:>10.3g} {w:>22} {best*1e3:>10.3g} ms")

    # P0-2: the boundary, solved rather than read off the grid.
    def critical_tau_c(g_, sum_ext=0.0, lo=0.1, hi=200.0, tol=1e-4):
        """Largest tau_c (ns) that still reaches the band, by bisection.

        The printed "feasible only for tau_c <= X" used to be the last grid
        point that happened to be true. With the default grid
        (..., 5.0, 10.0, 15.2) that is 5.0, because nothing between 5 and 10 was
        ever evaluated -- the quoted threefold speed-up was 15.2/5, an artefact
        of grid spacing rather than a solved boundary.
        """
        if not horizon_of(lo, g_, sum_ext)[2]:
            return None                      # infeasible even at the fast end
        if horizon_of(hi, g_, sum_ext)[2]:
            return hi                        # feasible everywhere in range
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if horizon_of(mid, g_, sum_ext)[2]:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # P1-6: every candidate nucleus, not just the hardcoded Trp. The methyl row
    # was printed in the SI and bound by a test while never reaching the
    # conclusion, because this function only ever evaluated Trp.
    from qbscreen.relaxation_estimate import sum_ext_for, bath_ratio_from_water
    f_bath = bath_ratio_from_water()
    boundaries = {}
    print(f"\n    bisected boundary per candidate nucleus "
          f"(bath f = {f_bath:.3f}, from the water validation row)")
    print(f"    {'nucleus':36s} {'no bath':>12s} {'with bath':>12s} {'speed-up':>10s}")
    for nm, gg in GEOM.items():
        dry = critical_tau_c(gg, 0.0)
        wet = critical_tau_c(gg, sum_ext_for(gg, f_bath))
        boundaries[nm] = dict(tau_crit_dry_ns=dry, tau_crit_wet_ns=wet)
        su = (TAU_PROTEIN_NS / wet) if wet else float("nan")
        print(f"    {nm[:35]:36s} {dry if dry else float('nan'):11.3f}s"
              .replace("s", " ") + f" {wet if wet else float('nan'):11.3f} {su:9.2f}x")

    tau_crit = boundaries[list(GEOM)[0]]["tau_crit_dry_ns"]

    viable = [r for r in rows if r["feasible"]]
    if viable:
        tmax = max(r["tau_c_ns"] for r in viable)
        # was `== 15.2`, which raised IndexError when handed this repo's own
        # tau_c_protein(60.0) = 15.243...; pick the nearest row instead
        pb = min(rows, key=lambda r: abs(r["tau_c_ns"] - 15.2))
        print(f"\n    feasible only for tau_c <= {tmax:g} ns; a 60 kDa protein-bound")
        print(f"    nucleus has tau_c = 15.2 ns and reaches at most "
              f"{pb['max_horizon_s']*1e3:.2g} ms -- below the band.")
    if return_boundaries:
        return rows, dict(bath_f=float(f_bath),
                          tau_protein_ns=float(TAU_PROTEIN_NS),
                          per_nucleus=boundaries)
    return rows


def robustness(min_memory=0.5, band=(1e-2, 1.0)):
    """Every variation of the ceiling we can motivate, in one table.

    Several of these were computed while answering referee points and never
    reported, which is the weakest possible place to leave them: a reader who
    wonders whether the exclusion survives a dimer, a crowded cytoplasm or a
    second dipolar partner should not have to ask.
    """
    import numpy as np
    from qbscreen.relaxation_estimate import (dipolar_T1, tau_rot2_protein, GEOM,
                                              sum_ext_for, bath_ratio_from_water)
    with open("simulation_results/register_reuse.json") as f:
        reuse = json.load(f)
    qs = np.array([r["q_nuc"] for r in reuse])
    mcs = np.array([r["MC_8"] for r in reuse])
    Tds = np.logspace(-4, 1, 6000)
    g = GEOM["Trp H-beta (CH2, geminal partner)"]
    f_bath = bath_ratio_from_water()
    base = tau_rot2_protein(60.0)

    def ceiling(tau_s, n=1, sx=0.0, mm=min_memory):
        T1 = dipolar_T1(g["r"], tau_s, 50e-6, n_partners=n, S2=1.0, sum_ext=sx)
        best = 0.0
        for Td in Tds:
            m = float(np.interp(1 - np.exp(-Td / T1), qs, mcs)) - 1.0
            if m > mm:
                best = max(best, m * Td)
        return float(T1), float(best)

    cases = [
        ("as published, hydration 1.3", base, 1, 0.0, min_memory),
        ("hydration 1.0", tau_rot2_protein(60.0, hydration=1.0), 1, 0.0, min_memory),
        ("hydration 1.6", tau_rot2_protein(60.0, hydration=1.6), 1, 0.0, min_memory),
        ("viscosity x2", tau_rot2_protein(60.0, eta_pa_s=1.38e-3), 1, 0.0, min_memory),
        ("viscosity x4", tau_rot2_protein(60.0, eta_pa_s=2.76e-3), 1, 0.0, min_memory),
        ("120 kDa dimer", tau_rot2_protein(120.0), 1, 0.0, min_memory),
        ("2 dipolar partners", base, 2, 0.0, min_memory),
        ("3 dipolar partners", base, 3, 0.0, min_memory),
        ("memory threshold 0.2", base, 1, 0.0, 0.2),
        ("memory threshold 1.0", base, 1, 0.0, 1.0),
        ("+ bath (f=0.54)", base, 1, sum_ext_for(g, f_bath), min_memory),
    ]
    rows = []
    print("\n  robustness of the horizon ceiling")
    print(f"    {'variation':34s} {'tau_c(ns)':>10s} {'T1(ms)':>8s} {'ceiling(ms)':>12s}")
    for label, tau, n, sx, mm in cases:
        T1, c = ceiling(tau, n, sx, mm)
        rows.append(dict(variation=label, mass_kDa=120.0 if "120" in label else 60.0,
                         n_partners=n, tau_c_ns=float(tau * 1e9),
                         T1_ms=T1 * 1e3, ceiling_ms=c * 1e3,
                         in_band=bool(c >= band[0])))
        print(f"    {label:34s} {tau*1e9:10.2f} {T1*1e3:8.3f} {c*1e3:12.2f}")
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out = {"light_driven": light_driven(), "reaction_driven": reaction_driven()}
    out["register_survival"] = combine()
    rows, bounds = feasible_region(return_boundaries=True)
    out["feasible_region"] = rows
    # P0-2/P1-6: the solved boundary, per candidate nucleus, with and without
    # the intermolecular bath. The old "feasible only for tau_c <= 5 ns" was the
    # last true point of a grid that never sampled between 5 and 10 ns.
    out["critical_tau_c"] = bounds
    out["robustness"] = robustness()
    with open(f"{OUT}/open5_turnover_estimate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  [checkpoint] {OUT}/open5_turnover_estimate.json")
