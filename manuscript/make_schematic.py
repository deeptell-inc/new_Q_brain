#!/usr/bin/env python3
"""Conceptual anchor figure (Fig. 1) for the reservoir manuscript.
(a) one molecule, two functions (sensing vs computing; decoherence two-faced);
(b) the three-layer radical-pair reservoir mapping."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
try:
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix"})
except Exception:
    pass

BLUE, GREEN, RED, GREY = "#1565c0", "#2e7d32", "#c62828", "#555555"


def box(ax, x, y, w, h, text, fc, ec, fs=7, tc="black"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, c=GREY, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=11, lw=lw, color=c, shrinkA=0, shrinkB=0))


fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.4, 4.4),
                               gridspec_kw={"height_ratios": [1.0, 1.2]})
for ax in (axA, axB):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

# ---- Panel (a): one molecule, two functions ----
axA.text(0.2, 9.5, "(a) one molecule, two functions", ha="left", fontsize=8,
         fontweight="bold")
box(axA, 5, 5, 5.0, 1.5, "separated radical pair\n(cryptochrome: $J\\approx0$, $\\tau\\sim\\mu$s)",
    "#eef2f7", GREY, 7)
box(axA, 2.6, 8.4, 3.6, 1.1, "magnetosensor", "#e3f0fb", BLUE, 7.5, BLUE)
box(axA, 7.4, 8.4, 3.6, 1.1, "quantum reservoir", "#e7f3e7", GREEN, 7.5, GREEN)
arrow(axA, 4.0, 5.75, 2.7, 7.8, BLUE)
arrow(axA, 6.0, 5.75, 7.3, 7.8, GREEN)
axA.text(1.2, 6.7, "long $T_2^e$\n$\\to$ sensitive", ha="center", fontsize=6, color=BLUE)
axA.text(8.8, 6.7, "fading memory\n$\\to$ computes", ha="center", fontsize=6, color=GREEN)
axA.text(5, 2.7, "decoherence: enemy of the sensor,\nfriend of the reservoir",
         ha="center", fontsize=6.3, color=RED, style="italic")

# ---- Panel (b): three-layer reservoir ----
axB.text(0.2, 9.6, "(b) nuclei are the register that must persist", ha="left", fontsize=8,
         fontweight="bold")
# input
axB.text(0.5, 6.4, "input\n$s_k$", ha="center", fontsize=6.5, color="black")
arrow(axB, 1.1, 6.4, 2.2, 6.4, "black")
# Layer 2: radical-pair electrons (read/write head)
box(axB, 3.3, 6.4, 2.0, 1.5, "$e_1\\!-\\!e_2$\nradical pair", "#e7f3e7", GREEN, 6.8, GREEN)
axB.text(3.3, 5.25, "Layer 2: read/write head", ha="center", fontsize=5.8, color=GREEN)
# Layer 1: nuclei (memory) above
for i, dx in enumerate((-0.7, 0.0, 0.7)):
    axB.add_patch(Circle((3.3 + dx, 8.25), 0.30, fc="#e3f0fb", ec=BLUE, lw=1.1))
axB.text(3.3, 8.95, "Layer 1: nuclear-spin memory (long $T_2$)", ha="center",
         fontsize=5.8, color=BLUE)
axB.add_patch(FancyArrowPatch((3.3, 7.2), (3.3, 7.9), arrowstyle="<|-|>",
             mutation_scale=8, lw=1.4, color=BLUE, shrinkA=0, shrinkB=0))
axB.text(4.55, 7.5, "hyperfine\nwrite/read", ha="left", fontsize=5.5, color=BLUE)
# Layer 3: readout
box(axB, 6.7, 6.4, 2.3, 1.5, "product\nreadout\n(yield, CIDNP)", "#fdeaea", RED, 6.5, RED)
axB.text(6.7, 5.25, "Layer 3: classical readout", ha="center", fontsize=5.8, color=RED)
arrow(axB, 4.35, 6.4, 5.5, 6.4, "black")
# linear map -> output
arrow(axB, 7.9, 6.4, 9.1, 6.4, "black")
axB.text(9.5, 6.4, "trained\nlinear\nmap", ha="center", fontsize=6, color="black")
# memory-horizon caption strip
axB.text(5, 3.9, "memory horizon $=\\mathrm{MC}\\times T_{\\rm turnover}$",
         ha="center", fontsize=6.8, color="black")
axB.text(5, 2.9, "reaches ms–s $\\it{if}$ the register is reused",
         ha="center", fontsize=6.8, color=RED)

fig.tight_layout(h_pad=0.6)
fig.savefig("manuscript/figures/fig_schematic.pdf", bbox_inches="tight")
fig.savefig("manuscript/figures/fig_schematic.png", dpi=300, bbox_inches="tight")
print("wrote manuscript/figures/fig_schematic.{pdf,png}")
