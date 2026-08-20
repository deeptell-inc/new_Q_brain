#!/usr/bin/env python3
"""PCCP table-of-contents graphic: exactly 8 cm x 4 cm, 600 dpi.

Message: the memory horizon of a radical-pair reservoir is capped by nuclear
relaxation at the geomagnetic field, not by the microsecond pair lifetime, and
that cap falls below the neural band unless the register's dipolar vector
reorients faster than the protein carrying it.

Every plotted number is read from simulation_results/, so this figure moves when
the data move -- the same binding the manuscript tables are under.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "Arial", "mathtext.fontset": "stix"})
except Exception:
    pass
# PCCP requires a hard 8 cm x 4 cm canvas, so no 'tight' bbox anywhere below.
plt.rcParams["savefig.bbox"] = None
plt.rcParams["savefig.pad_inches"] = 0.0

T = json.load(open("simulation_results/panel/open5_turnover_estimate.json"))
FR = T["feasible_region"]
TAU_P = T["critical_tau_c"]["tau_protein_ns"]
TAU_C = T["critical_tau_c"]["per_nucleus"]["Trp H-beta (CH2, geminal partner)"]["tau_crit_wet_ns"]

tau = np.array([r["tau_c_ns"] for r in FR])
hor = np.array([r["max_horizon_s"] for r in FR])
ok = np.array([r["feasible"] for r in FR])
hor_protein = hor[np.argmin(np.abs(tau - TAU_P))]

GREEN, RED, ORANGE, GREY = "#2e7d32", "#c62828", "#ef6c00", "#606060"

CM = 1 / 2.54
fig, ax = plt.subplots(figsize=(8 * CM, 4 * CM))
fig.subplots_adjust(left=0.155, right=0.995, bottom=0.20, top=0.87)

ax.axhspan(1e-2, 1.0, color="#ffe0b2", alpha=0.8, zorder=0, lw=0)
ax.text(0.0105, 1.7e-2, "neural band\n10 ms - 1 s", fontsize=5.6,
        color="#e65100", zorder=5, va="bottom")

ax.axvspan(TAU_C, 100, color="#fbe9e7", alpha=0.8, zorder=0, lw=0)
ax.plot(tau, hor, "-", c=GREY, lw=1.0, zorder=3)
ax.plot(tau[ok], hor[ok], "o", c=GREEN, ms=3.0, zorder=4)
ax.plot(tau[~ok], hor[~ok], "X", c=RED, ms=4.2, zorder=4)
ax.axvline(TAU_C, color=RED, ls="--", lw=0.9, zorder=2)
ax.text(TAU_C * 1.15, 2.0e-3, r"need $\tau_{\rm eff}\lesssim$" + f"{TAU_C:.1f} ns",
        fontsize=5.4, color=RED, ha="left", va="center", zorder=5)

# the one message, stated where the curve is not
ax.annotate("register rides the intact protein\n"
            r"($\tau_{\rm c}=$" + f"{TAU_P:.1f} ns): only {1e3 * hor_protein:.1f} ms,\n"
            "below the band",
            xy=(TAU_P, hor_protein * 1.25), xytext=(6.6, 0.9), fontsize=5.4,
            color=RED, ha="center", va="center", zorder=5,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.7,
                            shrinkA=1, shrinkB=1))
ax.text(0.0105, 6.5e-4, "nuclear relaxation at 50 " + "\u03bc" + "T, not the "
        + r"$\mu$s pair lifetime," + "\ncaps how long the reservoir remembers",
        fontsize=5.6, color=GREY, va="bottom", zorder=5)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.0085, 42); ax.set_ylim(3e-4, 12)
ax.set_xlabel(r"effective correlation time of the register  $\tau_{\rm eff}$ (ns)",
              fontsize=6, labelpad=1.5)
ax.set_ylabel("memory horizon (s)", fontsize=6, labelpad=1.5)
ax.tick_params(labelsize=5.5, length=2, pad=1.5)
ax.tick_params(which="minor", length=1)
ax.grid(True, which="major", alpha=0.22, lw=0.4)
for s in ax.spines.values():
    s.set_linewidth(0.6)

# CreationDate=None: without it every run produces a different PDF, and
# make_bundle.sh compares the bundle copy against the freeze manifest.
fig.savefig("manuscript/toc_entry.pdf", dpi=600,
            metadata={"CreationDate": None})
fig.savefig("manuscript/toc_entry.tif", dpi=600,
            pil_kwargs={"compression": "tiff_lzw"})
print(f"wrote toc_entry.pdf / toc_entry.tif "
      f"(tau_crit={TAU_C:.2f} ns, protein horizon={1e3 * hor_protein:.2f} ms)")
