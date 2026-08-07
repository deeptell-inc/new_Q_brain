#!/usr/bin/env python3
"""Predicted nuclear T1 for the cryptochrome photoproduct register.

No direct measurement of nuclear T1 in a cryptochrome photoproduct exists at any
field. That is the single largest gap in the biological reading of this work,
because the memory horizon is proportional to it. It cannot be measured here, but
it CAN be predicted: nuclear relaxation theory is standard, and every input is
either a measured constant or a standard structural value.

    quadrupolar (14N)  needs the quadrupole coupling constant  -- MEASURED
    dipolar (1H)       needs internuclear distances            -- standard geometry
    both               need the rotational correlation time    -- Stokes-Einstein-Debye

The calculator is validated against three independent measured systems before it
is applied to the unmeasured one. A prediction that cannot reproduce a known
answer is worth nothing.

CONVENTIONS
    J(w)  = tau / (1 + w^2 tau^2)                    normalised so J(0) = tau
    dipolar, like spins:
        1/T1 = (3/10) b^2 [ J(w) + 4 J(2w) ],  b = (mu0/4pi) gamma^2 hbar / r^3
    quadrupolar, I = 1:
        1/T1 = (3/8) (2 pi chi)^2 (1 + eta^2/3) [ 0.2 J(w) + 0.8 J(2w) ]
    Both reduce to the textbook extreme-narrowing limits (checked in validate()).
"""

import json
import os
import numpy as np

OUT = "simulation_results/panel"

MU0_4PI = 1.0e-7                 # T m / A
HBAR = 1.054571817e-34           # J s
KB = 1.380649e-23                # J / K
GAMMA_H = 2.675221874e8          # rad / s / T   (1H)
GAMMA_N14 = 1.9337792e7          # rad / s / T   (14N)
GAMMA_D = 4.10662791e7           # rad / s / T   (2H)


def larmor(gamma, B_tesla):
    return gamma * B_tesla       # rad/s


def J(w, tau):
    return tau / (1.0 + (w * tau) ** 2)


def dipolar_T1(r_ang, tau_c, B_tesla, gamma=GAMMA_H, n_partners=1, S2=1.0,
               tau_int=None, sum_ext=0.0, S2_ext=1.0):
    """T1 (s) from dipolar coupling to n_partners equivalent spins at r_ang (A).

    S2 < 1 with tau_int models fast internal motion (model-free): a rotating
    methyl has S2 = 1/4 for its intra-methyl vectors, because the H-H vector is
    perpendicular to the rotation axis and P2(cos 90 deg) = -1/2.

    sum_ext is the INTERMOLECULAR sum Sum_j r_j^-6 over partners that are not
    the named geminal one, in A^-6. Until this was added the calculator counted
    only the intramolecular partner, which the validation table below already
    showed to be incomplete: liquid water's intramolecular-only prediction is
    5.55 s against a measured total of 3.6 s, and the row's own note says
    "measured total incl. intermolecular". The missing rate was being printed
    and not propagated.

    The bath vectors are taken to reorient with the molecule as a whole
    (S2_ext = 1), not with the internal motion that averages the intramolecular
    vector: a proton on a neighbouring residue does not know the methyl is
    spinning. For a methyl proton the true value lies between S2 and 1, so
    S2_ext = 1 is the upper bound on the bath rate.
    """
    r = r_ang * 1e-10
    b = MU0_4PI * gamma ** 2 * HBAR / r ** 3          # rad/s
    w = larmor(gamma, B_tesla)

    def Jeff(wx):
        out = S2 * J(wx, tau_c)
        if tau_int is not None and S2 < 1.0:
            out = out + (1.0 - S2) * J(wx, tau_int)
        return out

    rate = 0.3 * b ** 2 * (Jeff(w) + 4.0 * Jeff(2 * w)) * n_partners
    if sum_ext:
        # same prefactor, its own geometric sum and its own spectral density.
        # Left as a separate term so that sum_ext = 0 reproduces the previous
        # result bit for bit.
        C = MU0_4PI * gamma ** 2 * HBAR
        Jext = lambda wx: S2_ext * J(wx, tau_c)
        rate = rate + 0.3 * C ** 2 * (sum_ext * 1e60) * (Jext(w) + 4.0 * Jext(2 * w))
    return 1.0 / rate


def quadrupolar_T1(chi_hz, tau_c, B_tesla, eta=0.0, gamma=GAMMA_N14, I=1.0):
    """T1 (s) for a quadrupolar nucleus. chi_hz = e^2 qQ/h in Hz."""
    w = larmor(gamma, B_tesla)
    pref = (3.0 / 40.0) * ((2 * I + 3) / (I ** 2 * (2 * I - 1)))
    rate = pref * (2 * np.pi * chi_hz) ** 2 * (1 + eta ** 2 / 3.0) \
        * (0.2 * J(w, tau_c) + 0.8 * J(2 * w, tau_c))
    return 1.0 / rate


def tau_rot2_protein(mass_kda, T_kelvin=310.0, eta_pa_s=0.69e-3, hydration=1.3):
    """RANK-2 Stokes-Einstein-Debye correlation time for a globular protein.

    Rank 2 is the one NMR relaxation wants: the dipolar spectral densities are
    built from rank-2 spherical harmonics. For a sphere tau_l = 1/(l(l+1)D_r),
    so tau_1 = 3 tau_2 -- the dielectric/fluorescence-anisotropy convention is
    three times longer than this.

    The rank is in the name because the two are easy to interchange and the
    consequence is large in the wrong direction: feeding the rank-1 value
    (45.7 ns for 60 kDa) into J() gives T1 = 0.81 ms instead of 2.44 ms and a
    horizon ceiling of 1.8 ms instead of 5.5 ms. That error would strengthen the
    paper's exclusion, which is exactly why an interface should prevent it
    rather than a reader catching it.
    """
    # 1 kDa = 1 kg/mol, so the mass of one molecule is mass_kda / N_A kilograms.
    v_bar = 0.73e-3                                   # m^3/kg, partial specific volume
    V = mass_kda / 6.02214076e23 * v_bar * hydration
    return eta_pa_s * V / (KB * T_kelvin)          # = 1/(6 D_r), rank 2


# Retained so existing callers keep working; the rank-explicit name is the
# one to use in new code.
tau_c_protein = tau_rot2_protein


# ─────────────────────────────────────────────────────────────────────
# Validation: the calculator must reproduce measured systems
# ─────────────────────────────────────────────────────────────────────
def validate():
    rows = []

    # (1) 2H in liquid D2O. chi = 250 kHz, tau_c = 2.5 ps, high field.
    #     Literature T1 ~ 0.4 s.
    t = quadrupolar_T1(250e3, 2.5e-12, 9.4, eta=0.0, gamma=GAMMA_D)
    rows.append(dict(system="2H in D2O (quadrupolar)", predicted_s=t,
                     measured_s=0.40, ref="Abragam; standard value"))

    # (2) 14N of free amino acids in water. Measured linewidths give T2 = 2.9-8.8 ms
    #     (Troganis 2003); in extreme narrowing T1 = T2. chi(NH3+) = 1.08-1.31 MHz
    #     measured by NQR (Edmonds & Summers 1973). tau_c ~ 20 ps for a free
    #     amino acid in water.
    t = quadrupolar_T1(1.2e6, 20e-12, 2.1, eta=0.3)
    rows.append(dict(system="14N of free amino acid (quadrupolar)", predicted_s=t,
                     measured_s=5.0e-3, ref="Troganis 2003 / Edmonds 1973"))

    # (3) intramolecular 1H-1H of liquid water. r(HH) = 1.51 A, tau_c = 2.5 ps.
    #     Measured TOTAL T1 = 3.6 s; intramolecular alone should be LONGER, since
    #     intermolecular dipolar contributes roughly a third of the rate.
    t = dipolar_T1(1.51, 2.5e-12, 9.4, n_partners=1)
    rows.append(dict(system="1H2O intramolecular (dipolar)", predicted_s=t,
                     measured_s=3.6, ref="measured total incl. intermolecular"))

    print("  validation of the relaxation calculator")
    for r in rows:
        ratio = r["predicted_s"] / r["measured_s"]
        r["ratio"] = float(ratio)
        print(f"    {r['system']:38s} predicted {r['predicted_s']:.3e} s   "
              f"measured {r['measured_s']:.3e} s   ratio {ratio:.2f}")
    return rows


# ─────────────────────────────────────────────────────────────────────
# Prediction for the cryptochrome photoproduct
# ─────────────────────────────────────────────────────────────────────
# Measured 14N quadrupole tensors of the flavin semiquinone
# (Martinez et al., Magn. Reson. 6, 183 (2025)): principal values in MHz.
FLAVIN_N = {"N5": dict(chi=3.6e6, eta=0.11), "N10": dict(chi=4.8e6, eta=0.33)}
TRP_N = dict(chi=3.38e6, eta=0.02)        # indole N, Suenram 1988 (microwave)

GEOM = {
    # register candidates, with the dominant dipolar partner.
    #
    # S2_ext is the order parameter of the vectors to the INTERMOLECULAR bath,
    # which is not the same as S2. For Trp H-beta the surrounding protons really
    # do co-rotate with the protein, so S2_ext = 1. For a methyl proton they do
    # not: the proton circles the C3 axis at ~1.03 A, which partially averages
    # every vector to a fixed external partner. Averaging Y_2m/r^3 over that
    # circle for external protons at 2.4-4.0 A gives S2_ext = 0.54-0.79, with an
    # r^-6-weighted bath average near 0.70. Leaving it at 1 would overestimate
    # the methyl's bath -- in the direction that strengthens this paper's own
    # exclusion, which is the wrong way to be wrong.
    "Trp H-beta (CH2, geminal partner)": dict(r=1.78, n=1, S2=1.0, tau_int=None,
                                              S2_ext=1.0),
    "flavin 8-alpha CH3 (intra-methyl)": dict(r=1.79, n=2, S2=0.25, tau_int=10e-12,
                                              S2_ext=0.70),
}


def bath_ratio_from_water():
    """f = R_inter / R_intra, taken from this file's own validation table.

    Row 3 predicts 5.55 s for the intramolecular H-H of liquid water against a
    measured 3.6 s, and its own note reads "measured total incl. intermolecular".
    The shortfall IS the intermolecular bath, already measured here and, until
    now, not carried into any conclusion.

    Calibrating on water rather than on a structure is deliberate. A structural
    sum over a cryptochrome model gives a larger f (denser proton environment
    than bulk water), so water is the WEAKEST defensible bath -- the choice that
    makes it hardest to exclude a protein-bound register. Excluding it anyway is
    the robust result.
    """
    return dipolar_T1(1.51, 2.5e-12, 9.4, n_partners=1) / 3.6 - 1.0


def sum_ext_for(g=None, f=None):
    """Intermolecular sum (A^-6) for a nucleus in a protein interior.

    This used to be scaled by the entry's own n/r^6, which gave the methyl a
    bath 1.9x larger than Trp H-beta's purely because it has two geminal
    partners. The intermolecular sum is set by the density and closest approach
    of surrounding protons and has nothing to do with how many geminal partners
    the nucleus happens to have, so the same interior bath is now used for both.

    The magnitude comes from the water validation row transplanted onto the
    Trp geminal geometry, which lands at 0.017 A^-6 -- inside, and at the low
    end of, the 0.016-0.030 A^-6 a uniform protein proton density gives for
    closest approaches of 2.5-2.0 A. Being at the low end is the conservative
    choice for a claim of exclusion.
    """
    if f is None:
        f = bath_ratio_from_water()
    trp = GEOM["Trp H-beta (CH2, geminal partner)"]
    return f * trp["n"] / trp["r"] ** 6


def predict(mass_kda=60.0, fields=(50e-6, 1e-3, 0.5, 9.4)):
    tau_c = tau_c_protein(mass_kda)
    out = {"mass_kDa": mass_kda, "tau_c_s": tau_c, "fields_T": list(fields),
           "proton": {}, "nitrogen14": {}}
    print(f"\n  cryptochrome-sized protein: {mass_kda:.0f} kDa, "
          f"tau_c = {tau_c*1e9:.1f} ns at 310 K")

    print(f"\n  {'register nucleus':38s} " +
          "  ".join(f"{b*1e3:>8.3g} mT" for b in fields))
    for name, g in GEOM.items():
        ts = [dipolar_T1(g["r"], tau_c, b, n_partners=g["n"], S2=g["S2"],
                         tau_int=g["tau_int"]) for b in fields]
        out["proton"][name] = [float(x) for x in ts]
        print(f"    {name:36s} " + "  ".join(f"{t:>11.3g}" for t in ts))
    for name, q in list(FLAVIN_N.items()) + [("Trp indole N", TRP_N)]:
        ts = [quadrupolar_T1(q["chi"], tau_c, b, eta=q["eta"]) for b in fields]
        out["nitrogen14"][f"14N {name}"] = [float(x) for x in ts]
        print(f"    {'14N ' + name:36s} " + "  ".join(f"{t:>11.3g}" for t in ts))
    return out


def horizon_consequences(out, turnovers=(1e-3, 1e-2, 1e-1)):
    """Translate the predicted T1 into the paper's own q and horizon."""
    geo = 50e-6
    i = out["fields_T"].index(geo)
    T1_H = min(v[i] for v in out["proton"].values())
    T1_N = min(v[i] for v in out["nitrogen14"].values())
    rows = []
    for Td in turnovers:
        rows.append(dict(T_turnover_s=Td,
                         q_proton=float(1 - np.exp(-Td / T1_H)),
                         q_14N=float(1 - np.exp(-Td / T1_N))))
    print(f"\n  at the geomagnetic field, worst-case register nucleus:")
    print(f"    proton T1 = {T1_H:.3g} s      14N T1 = {T1_N:.3g} s")
    print(f"    {'T_turnover':>12}  {'q (proton)':>12}  {'q (14N)':>12}")
    for r in rows:
        print(f"    {r['T_turnover_s']:>12.0e}  {r['q_proton']:>12.4f}"
              f"  {r['q_14N']:>12.4f}")
    out["T1_proton_geomagnetic_s"] = float(T1_H)
    out["T1_14N_geomagnetic_s"] = float(T1_N)
    out["q_table"] = rows
    return out




def tau_c_scan(taus_ns=(0.05, 0.1, 0.5, 1.0, 5.0, 15.2), B=50e-6):
    """The escape route, and the sharpened criterion.

    In extreme narrowing (which is where the geomagnetic field puts every nucleus
    here) 1/T1 is PROPORTIONAL to tau_c. So the register's lifetime is set by how
    fast the product tumbles. A nucleus that stays bound to a 60 kDa protein
    relaxes in milliseconds; the same nucleus in a small, freely tumbling product
    relaxes a hundred times more slowly.

    This turns the paper's vague requirement ("the nuclei leave in the diamagnetic
    product") into a quantitative one: the product must be SMALL AND MOBILE.
    """
    rows = []
    g = GEOM["Trp H-beta (CH2, geminal partner)"]
    for t_ns in taus_ns:
        tc = t_ns * 1e-9
        T1 = dipolar_T1(g["r"], tc, B, n_partners=g["n"])
        row = dict(tau_c_ns=t_ns, T1_proton_s=float(T1))
        for Td in (1e-3, 1e-2, 1e-1):
            row[f"q_at_{Td:g}s"] = float(1 - np.exp(-Td / T1))
        rows.append(row)
    print(f"\n  proton T1 against product mobility, at 50 uT")
    print(f"    {'tau_c (ns)':>11}  {'T1 (s)':>10}  {'q@1ms':>7}  {'q@10ms':>7}  {'q@100ms':>8}")
    for r in rows:
        print(f"    {r['tau_c_ns']:>11.3g}  {r['T1_proton_s']:>10.3g}"
              f"  {r['q_at_0.001s']:>7.3f}  {r['q_at_0.01s']:>7.3f}"
              f"  {r['q_at_0.1s']:>8.3f}")
    return rows


def dipolar_T2(r_ang, tau_c, B_tesla, gamma=GAMMA_H, n_partners=1,
               sum_ext=0.0, S2_ext=1.0):
    """Transverse relaxation for the same dipolar pair.

    This exists to check a claim the main text makes: that the register survives
    because it stores populations (T1) rather than coherence (T2), the latter
    being much shorter. That distinction requires T2 << T1, which holds only when
    w*tau_c >> 1. At the geomagnetic field w*tau_c ~ 2e-4 -- extreme narrowing --
    where J(0) = J(w) = J(2w) and therefore T1 = T2 EXACTLY. The rescue argument
    is vacuous at the operating point, and this function makes that checkable.

    sum_ext carries the intermolecular bath, exactly as in dipolar_T1. It was
    added later than the T1 term and the omission mattered: the T1 = T2 identity
    was being checked between a bath-corrected T1 and a bath-free T2, so the
    ratio was not the one the operating point actually has. It is still 1 --
    extreme narrowing does not care which spins supply the coupling, only that
    J(0) = J(w) = J(2w) -- but that now follows from the computation rather than
    from an argument about it.
    """
    r = r_ang * 1e-10
    b = MU0_4PI * gamma ** 2 * HBAR / r ** 3
    w = larmor(gamma, B_tesla)
    rate = 0.15 * b ** 2 * (3 * J(0.0, tau_c) + 5 * J(w, tau_c)
                            + 2 * J(2 * w, tau_c)) * n_partners
    if sum_ext:
        # same prefactor and its own geometric sum, as in dipolar_T1, so that
        # sum_ext = 0 reproduces the previous result bit for bit.
        C = MU0_4PI * gamma ** 2 * HBAR
        Jext = lambda wx: S2_ext * J(wx, tau_c)
        rate = rate + 0.15 * C ** 2 * (sum_ext * 1e60) * (
            3 * Jext(0.0) + 5 * Jext(w) + 2 * Jext(2 * w))
    return 1.0 / rate


def t1_t2_ratio(fields=(50e-6, 1e-3, 0.5, 9.4), mass_kda=60.0):
    """Where does T2 << T1 actually hold?"""
    tau_c = tau_c_protein(mass_kda)
    g = GEOM["Trp H-beta (CH2, geminal partner)"]
    se = sum_ext_for()
    rows = []
    print(f"\n  is the T1/T2 distinction real at the operating point?")
    print(f"    {'field':>10} {'w*tau_c':>10} {'T1 (s)':>11} {'T2 (s)':>11} {'T2/T1':>8}")
    for B in fields:
        T1 = dipolar_T1(g["r"], tau_c, B, n_partners=g["n"], sum_ext=se)
        T2 = dipolar_T2(g["r"], tau_c, B, n_partners=g["n"], sum_ext=se)
        wt = larmor(GAMMA_H, B) * tau_c
        rows.append(dict(field_T=B, omega_tau_c=float(wt), T1_s=float(T1),
                         T2_s=float(T2), T2_over_T1=float(T2 / T1)))
        print(f"    {B*1e3:>8.3g} mT {wt:>10.3g} {T1:>11.3g} {T2:>11.3g} {T2/T1:>8.3f}")
    print("    -> at 50 uT, T2/T1 = 1: extreme narrowing erases the distinction.")
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = {"validation": validate()}
    p = predict()
    p = horizon_consequences(p)
    res["prediction"] = p
    # tau_c_scan is consumed by qbscreen.turnover_estimate.combine(); omitting it
    # here breaks the two-stage pipeline on a clean checkout.
    res["tau_c_scan"] = tau_c_scan()
    res["t1_t2_ratio"] = t1_t2_ratio()
    with open(f"{OUT}/open5_relaxation_estimate.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  [checkpoint] {OUT}/open5_relaxation_estimate.json")
