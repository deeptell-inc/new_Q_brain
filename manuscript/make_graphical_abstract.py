#!/usr/bin/env python3
"""Graphical abstract: 525 px wide PNG, as the journal requires.

Top: a radical pair cycled on one molecule hands its nuclear polarisation from
turnover k to turnover k+1. Bottom: where that memory sits on a time axis.
The ceiling range is read from the shipped criterion files, not typed in.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

P = "simulation_results/panel/"
T = "Trp H-beta (CH2, geminal partner)"
a = json.load(open(P + "open8_proton_register_reuse.json"))
b = json.load(open(P + "open18_singlet_register.json"))
ceil = [c["per_nucleus"][T][k]["ceiling_ms"]
        for c in (a["product_criterion"], a["product_selected_criterion_mm0.5"],
                  b["singlet_criterion"], b["singlet_selected_criterion"]) for k in ("dry", "wet")]
lo, hi = min(ceil), max(ceil)

BLUE, GREEN, RED, GREY = "#1565c0", "#2e7d32", "#c62828", "#555555"
fig = plt.figure(figsize=(5.25, 3.0), dpi=100)
top = fig.add_axes([0.0, 0.47, 1.0, 0.53]); top.set_xlim(0, 10); top.set_ylim(0, 5); top.axis("off")


def box(x, text, ec, fc):
    top.add_patch(FancyBboxPatch((x - 1.3, 1.5), 2.6, 1.7, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.4))
    top.text(x, 2.35, text, ha="center", va="center", fontsize=8, color=ec)


box(1.7, "radical pair\nturnover $k$", GREEN, "#e7f3e7")
box(5.0, "product\npolarised nuclei", BLUE, "#e3f0fb")
box(8.3, "radical pair\nturnover $k+1$", GREEN, "#e7f3e7")
for x0, x1 in ((3.05, 3.65), (6.35, 6.95)):
    top.add_patch(FancyArrowPatch((x0, 2.35), (x1, 2.35), arrowstyle="-|>", mutation_scale=11, lw=1.5, color="black"))
top.text(5.0, 4.3, "A radical pair cycled on one molecule remembers earlier turnovers",
         ha="center", fontsize=8.5, fontweight="bold")
top.text(5.0, 0.75, "the memory is nuclear, read as CIDNP, and lost at the proton $T_1$",
         ha="center", fontsize=7.5, color=GREY)

ax = fig.add_axes([0.06, 0.2, 0.9, 0.2]); ax.set_xscale("log"); ax.set_xlim(3e-7, 2.0); ax.set_ylim(0, 1)
ax.axvspan(1e-2, 1.0, color="#ffe0b2"); ax.text(1e-1, 0.5, "neural\nrhythms", ha="center", va="center", fontsize=7.5, color="#e65100")
ax.axvspan(lo * 1e-3, hi * 1e-3, color=BLUE, alpha=0.85)
ax.text((lo * hi) ** 0.5 * 1e-3, 1.12, f"memory limit {lo:.1f}–{hi:.1f} ms\n(proton $T_1$ at 50 $\\mu$T)",
        ha="center", va="bottom", fontsize=7.5, color=BLUE)
ax.axvline(1e-6, color=RED, lw=2); ax.text(1.6e-6, 0.5, "pair lifetime", va="center", fontsize=7.5, color=RED)
ax.set_yticks([]); ax.set_xlabel("time (s)", fontsize=8, labelpad=1); ax.tick_params(labelsize=7.5)
for s in ("top", "left", "right"):
    ax.spines[s].set_visible(False)
fig.savefig("manuscript/figures/GraphicalAbstract1.png", dpi=100)
print(f"wrote GraphicalAbstract1.png (ceiling {lo:.2f}-{hi:.2f} ms)")
