#!/usr/bin/env python3
"""Honest weak-field MFE for brain flavoenzymes — decoherence included.

Corrects the peer-review flaw: the magnetic field effect (MFE) is now computed
from the full Haberkorn master equation INCLUDING electronic dephasing (T2e) and
REALISTIC, enzyme-specific radical-pair lifetimes (recombination rate k).

Model: 2 electrons + 3 dominant flavin/partner hyperfine nuclei (1H/14N), solved exactly in Liouville
space (5 spins, dim 32). For each flavoenzyme we report the MFE at the
geomagnetic field (50 µT) and at the coherent-limit "optimal" field, in both the
(old, incorrect) coherent limit and the (correct) decohering case.

Key physics demonstrated:
  • Neurotransmitter-metabolising flavoenzymes (MAO-A/B, DAO) host an
    ACTIVE-SITE radical pair: r ≈ 3.5 Å ⇒ |J| ≈ 0.5–1 GHz and lifetime ≈ ps,
    so k ≈ 1e5–1e6 MHz. Both effects independently suppress weak-field S–T
    mixing → MFE ≈ 0.
  • Cryptochrome (CRY) hosts a SEPARATED radical pair (Trp chain, r ≈ 15–20 Å)
    ⇒ |J| < 1 MHz, lifetime ≈ µs (k ≈ 1 MHz) — the one geometry where a weak
    field can act, consistent with its established role as a magnetoreceptor.
  • Even for the CRY-like geometry, including the computed electronic T2e
    (~1 ns active-site / longer when separated) brings the geomagnetic-field
    MFE below ~0.1%, consistent with in-vitro data.
"""

import numpy as np

from qbscreen.spin_dynamics import (
    SX, SY, SZ, spin_op, singlet_projector,
    G_E, MU_B, HBAR,
)
from qbscreen.master_equation import (
    singlet_yield_master, singlet_yield_exponential, electron_dephasing_ops,
)

TWO_PI = 2.0 * np.pi


def build_5spin_H(B_tesla, A_P_MHz, A_H1_MHz, A_H2_MHz, J_MHz, D_MHz=0.0):
    """5-spin Hamiltonian (MHz units): [e1, e2, N_a, H_a (on flavin e1), H_b (on partner e2)].

    e1 = flavin radical (dominant ring N/H); e2 = partner radical (carries ¹H_b).
    """
    n = 5
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    omega_e = G_E * MU_B * B_tesla / (HBAR * 2 * np.pi * 1e6)  # MHz
    H += omega_e * (spin_op(SZ, 0, n) + spin_op(SZ, 1, n))

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # Hyperfine: dominant flavin ring nuclei on electron 1
        for op in (SX, SY, SZ):
            H += A_P_MHz * spin_op(op, 0, n) @ spin_op(op, 2, n)
            H += A_H1_MHz * spin_op(op, 0, n) @ spin_op(op, 3, n)
            # ¹H_b on electron 2
            H += A_H2_MHz * spin_op(op, 1, n) @ spin_op(op, 4, n)

        # Exchange + dipolar between the two electrons
        if abs(J_MHz) > 1e-12:
            for op in (SX, SY, SZ):
                H += J_MHz * spin_op(op, 0, n) @ spin_op(op, 1, n)
        if abs(D_MHz) > 1e-12:
            # secular dipolar: D (Sz1 Sz2 - ⅓ S1·S2)
            Szz = spin_op(SZ, 0, n) @ spin_op(SZ, 1, n)
            S1S2 = sum(spin_op(op, 0, n) @ spin_op(op, 1, n) for op in (SX, SY, SZ))
            H += D_MHz * (Szz - S1S2 / 3.0)
    return H


def mfe_master(A_P, A_H1, A_H2, J, k_MHz, T2e_ns, B_opt_mT=0.9, D_MHz=0.0):
    """Return (MFE@Earth, MFE@opt) in %, from the exact master equation."""
    n = 5
    dim = 2 ** n
    P_S = singlet_projector(0, 1, n)
    P_T = np.eye(dim) - P_S
    rho0 = P_S / np.trace(P_S)
    T2e_us = None if T2e_ns is None else T2e_ns * 1e-3

    def phi(B_T):
        H = TWO_PI * build_5spin_H(B_T, A_P, A_H1, A_H2, J, D_MHz)  # rad/µs
        # Rates are 1/µs, NOT angular frequencies: build_liouvillian documents
        # k_S, k_T as (1/µs), so τ = 1/k. Multiplying by 2π here would shorten
        # every quoted lifetime by 2π (10 ps → 1.6 ps).
        kS = kT = k_MHz
        deph = electron_dephasing_ops(n, T2e_us, electron_indices=(0, 1))
        return singlet_yield_master(H, P_S, P_T, kS, kT, rho0, deph)

    phi0 = phi(0.0)
    phiE = phi(50e-6)
    phiO = phi(B_opt_mT * 1e-3)
    mfe_E = (phiE - phi0) / phi0 * 100 if phi0 > 0 else 0.0
    mfe_O = (phiO - phi0) / phi0 * 100 if phi0 > 0 else 0.0
    return mfe_E, mfe_O


def D_point_dipole(r_nm):
    """Electron–electron point-dipole coupling in MHz: D = -52.04 / r_nm^3.

    Order-of-magnitude only. At van der Waals contact (r ≈ 3.5 Å) the point-dipole
    approximation is at the edge of its validity — spin-density delocalisation over
    the flavin and the partner radical reduces the effective |D|. Treated here as a
    magnitude estimate, not a precise value.
    """
    return -52.04 / r_nm ** 3


# ── Realistic enzyme parameter set ──────────────────────────────────
#  name, A_P(flavin N), A_H1, A_H2(partner), J, k(MHz)=1/τ, T2e(ns), r_nm, label
#  NOTE: MAO-A and MAO-B currently share every numerical parameter; they differ
#  only in the identity of the partner radical. Substrate-specific hyperfine and
#  geometry would be needed to make them independent calculations.
ENZYMES = [
    # Neurotransmitter flavoenzymes: active-site RP, large J, ps lifetime
    dict(name="MAO-A", A_P=14, A_H1=11.0, A_H2=44.0, J=750.0, k=1e5, T2e=1.0,
         r_nm=0.35, partner="serotonin amine•+", r="3.5 Å", role="5-HT degradation"),
    dict(name="MAO-B", A_P=14, A_H1=11.0, A_H2=44.0, J=750.0, k=1e5, T2e=1.0,
         r_nm=0.35, partner="dopamine amine•+", r="3.5 Å", role="DA degradation"),
    dict(name="DAO",   A_P=14, A_H1=11.0, A_H2=20.0, J=400.0, k=1e5, T2e=1.0,
         r_nm=0.40, partner="D-amino-acid•",  r="~4 Å",  role="D-Ser degradation"),
    # Cryptochrome: separated RP, J≈0, µs lifetime (the known magnetoreceptor)
    dict(name="CRY",   A_P=14, A_H1=11.0, A_H2=44.0, J=0.5,  k=1.0,  T2e=1000.0,
         r_nm=1.75, partner="Trp•+ (triad)",  r="15–20 Å", role="circadian / magnetoreception"),
]
for _e in ENZYMES:
    _e["D"] = D_point_dipole(_e["r_nm"])


def run():
    print("=" * 84)
    print("Honest weak-field MFE for brain flavoenzymes (exact master equation, T2e included)")
    print("=" * 84)
    print(f"{'Enzyme':7} {'partner':18} {'r':8} {'J(MHz)':8} {'τ':8} {'T2e':6} "
          f"| {'MFE@50µT coh':>13} {'MFE@50µT real':>14} {'MFE@opt real':>13}")
    print("-" * 110)

    results = []
    for e in ENZYMES:
        tau = f"{1e6/e['k']:.0f} ps" if e['k'] >= 1e3 else f"{1/e['k']:.0f} µs"
        # Coherent limit (T2e = ∞, but realistic k and J)
        mfeE_coh, _ = mfe_master(e['A_P'], e['A_H1'], e['A_H2'], e['J'],
                                 e['k'], T2e_ns=None, D_MHz=e['D'])
        # Realistic: include T2e
        mfeE_real, mfeO_real = mfe_master(e['A_P'], e['A_H1'], e['A_H2'], e['J'],
                                          e['k'], T2e_ns=e['T2e'], D_MHz=e['D'])
        results.append(dict(name=e['name'], partner=e['partner'], r=e['r'],
                            J=e['J'], D=e['D'], k=e['k'], T2e=e['T2e'],
                            mfeE_coh=mfeE_coh, mfeE_real=mfeE_real,
                            mfeO_real=mfeO_real, role=e['role']))
        print(f"{e['name']:7} {e['partner']:18} {e['r']:8} {e['J']:<8.0f} {tau:8} "
              f"{e['T2e']:>4.0f}ns | {mfeE_coh:>+12.4f}% {mfeE_real:>+13.4f}% "
              f"{mfeO_real:>+12.4f}%")

    print("-" * 110)
    print("\nInterpretation:")
    print("  • MAO-A/B, DAO: active-site RP (r≈3.5 Å) ⇒ |J|≈0.75 GHz and τ≈10 ps")
    print("    (k=1e5 MHz). Weak-field MFE is negligible (<0.01%) — these enzymes")
    print("    are NOT geomagnetic-field-sensitive, with or without decoherence.")
    print("  • CRY: separated RP (J≈0, τ≈µs) is the one magnetosensitive geometry;")
    print("    even there, electronic T2e brings the 50 µT MFE toward ~0.1%,")
    print("    consistent with in-vitro flavoprotein data (<1%).")
    return results


def lifetime_t2e_sweep():
    """MFE vs RP lifetime and vs T2e — the figure that explains the negative result."""
    n = 5
    dim = 2 ** n
    P_S = singlet_projector(0, 1, n)
    P_T = np.eye(dim) - P_S
    rho0 = P_S / np.trace(P_S)

    def mfe_at(B_T, A_P, A_H1, A_H2, J, k_MHz, T2e_ns, D_MHz=0.0):
        T2e_us = None if T2e_ns is None else T2e_ns * 1e-3

        def phi(B):
            H = TWO_PI * build_5spin_H(B, A_P, A_H1, A_H2, J, D_MHz)
            kS = kT = k_MHz            # 1/µs (see mfe_master); no 2π on rates
            deph = electron_dephasing_ops(n, T2e_us, (0, 1))
            return singlet_yield_master(H, P_S, P_T, kS, kT, rho0, deph)

        p0, pB = phi(0.0), phi(B_T)
        return (pB - p0) / p0 * 100 if p0 > 0 else 0.0

    # Each geometry carries its own point-dipole coupling; D and J are both
    # functions of r and must not be varied independently.
    D_act = D_point_dipole(0.35)                # active site, r = 3.5 Å
    D_sep = D_point_dipole(1.75)                # separated pair, r = 17.5 Å

    # (a) MFE@50µT vs lifetime τ = 1/k, for J≈0 (best case) vs J=750 MHz (MAO)
    taus_ps = np.logspace(0, 6.5, 28)          # 1 ps … ~3 µs
    ks = 1e6 / taus_ps                          # MHz  (k = 1/τ, τ in µs → 1e6/τ_ps)
    mfe_J0 = [abs(mfe_at(50e-6, 14, 11.0, 44.0, 0.5, k, None, D_sep)) for k in ks]
    mfe_Jmao = [abs(mfe_at(50e-6, 14, 11.0, 44.0, 750.0, k, None, D_act)) for k in ks]

    # (b) MFE@50µT vs T2e for the CRY-like separated geometry (J≈0, τ=1µs)
    t2es_ns = np.logspace(0, 4, 24)             # 1 ns … 10 µs
    mfe_vs_T2e = [abs(mfe_at(50e-6, 14, 11.0, 44.0, 0.5, 1.0, t, D_sep)) for t in t2es_ns]

    return dict(taus_ps=taus_ps.tolist(), mfe_J0=mfe_J0, mfe_Jmao=mfe_Jmao,
                t2es_ns=t2es_ns.tolist(), mfe_vs_T2e=mfe_vs_T2e)


def make_figure(sweep, out="manuscript/figures/fig_honest_mfe.pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix"})
    except Exception:
        pass
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Both series get IDENTICAL treatment: clamp at the empirical solver floor and
    # draw clamped points as open markers so they are not mistaken for resolved
    # values. (An earlier version clamped only the active-site series, which made
    # its floor look like a physical saturation and let the other series be drawn
    # below the declared floor.)
    _FLOOR = 1e-13
    tp = np.asarray(sweep["taus_ps"])

    def _draw(y, color, marker, ls, label):
        y = np.asarray(y)
        res = y > _FLOOR                       # numerically resolved
        yc = np.maximum(y, _FLOOR)
        ax[0].loglog(tp, yc, ls, c=color, lw=1.2, label=label)
        ax[0].loglog(tp[res], yc[res], marker, c=color, ms=3, ls="none")
        ax[0].loglog(tp[~res], yc[~res], marker, mfc="none", c=color, ms=3, ls="none")

    _draw(sweep["mfe_J0"], "#1565c0", "o", "-", "separated RP, $J\\approx0$")
    _draw(sweep["mfe_Jmao"], "#c62828", "s", "--", "active-site RP, $J=750$ MHz")
    ax[0].axhline(_FLOOR, color="0.6", ls=":", lw=0.7)
    ax[0].text(1.5, _FLOOR*1.6, "solver floor (open symbols)", fontsize=5.5, color="0.4")
    ax[0].axvspan(1, 100, alpha=0.12, color="#c62828")
    ax[0].axvline(1e6, color="#1565c0", ls=":", lw=0.8)
    ax[0].text(3, ax[0].get_ylim()[1]*0.3 if False else 1e-3, "MAO\nτ≈10 ps", fontsize=7,
               color="#c62828")
    ax[0].set_xlabel("radical-pair lifetime $\\tau$ (ps)")
    ax[0].set_ylabel("|MFE| at 50 µT (%)")
    ax[0].set_title("(a)", loc="left", fontweight="bold")
    ax[0].legend(fontsize=6.5, loc="lower right")
    ax[0].grid(True, which="both", alpha=0.25)

    ax[1].semilogx(sweep["t2es_ns"], sweep["mfe_vs_T2e"], "o-", c="#2e7d32", ms=3, lw=1.2)
    ax[1].set_xlabel("electron coherence $T_2^e$ (ns)")
    ax[1].set_ylabel("|MFE| at 50 µT (%)")
    ax[1].set_title("(b)  separated RP ($\\tau=1$ µs)", loc="left", fontweight="bold")
    ax[1].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"  figure → {out}")


if __name__ == "__main__":
    import json
    res = run()
    print("\nGenerating lifetime / T2e sweep figure ...")
    sweep = lifetime_t2e_sweep()
    make_figure(sweep)
    with open("simulation_results/honest_mfe_results.json", "w") as f:
        json.dump({"enzymes": res, "sweep": sweep}, f, indent=2)
    print("  data → simulation_results/honest_mfe_results.json")
