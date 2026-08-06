#!/usr/bin/env python3
"""Figures for the revised manuscript:
  fig_readout   -- what biology can actually read, against each route's OWN floor
  fig_criterion -- the clock is the turnover time; the criterion is register reuse

Both figures are drawn from the referee-response outputs in
simulation_results/panel/ where those supersede the original scans:
  * all seven readout routes are shown (the submitted figure omitted route 4);
  * the memoryless floor is route-specific, not a single line at IPC = 1;
  * the clock panel uses the twelve-seed scan with error bars, and shows both the
    naive MC x T horizon and the delayed-memory (MC - C0) x T horizon;
  * the criterion panel contrasts T1-type depolarisation with T2-type dephasing.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "Arial", "mathtext.fontset": "stix"})
except Exception:
    pass

R = json.load(open("simulation_results/readout_routes.json"))
F = json.load(open("simulation_results/panel/m7_route_floors.json"))
FH = json.load(open("simulation_results/panel/m7_hetero_floor.json"))
CK = json.load(open("simulation_results/panel/c2_clock_multiseed.json"))
KER = json.load(open("simulation_results/panel/c3_delay_kernel.json"))
CH = json.load(open("simulation_results/panel/c4_nuclear_channel.json"))

GREY, BLUE, GREEN, RED = "#90a4ae", "#1565c0", "#2e7d32", "#c62828"
ORANGE = "#ef6c00"

# ─────────────── Fig: readout routes ───────────────
# All seven routes. Floors are measured per route at q = 1 (register wiped every
# cycle), so the bar height above its own floor is what measures memory.
order = ["YS_end", "SandT_end", "hetero_tau", "YS_t", "SandT_t", "YS_t_accum", "cidnp"]
lab = ["end-point\nyield\n(1 ch)", "both exit\nchannels\n(2 ch)",
       "3 micro-\nenvironments\n(3 ch)", "time-resolved\nkinetics\n(5 ch)",
       "both exit,\ntime-resolved\n(10 ch)", "accumulated\npool\n(10 ch)",
       "kinetics +\nproduct CIDNP\n(8 ch)"]
floors = dict(F)
floors["hetero_tau"] = FH

val = np.array([R[k]["IPC"] for k in order])
err = np.array([R[k]["IPC_sd"] for k in order])
flo = np.array([floors[k]["floor_IPC"] for k in order])

fig, ax = plt.subplots(figsize=(6.4, 3.2))
# the accumulated pool is hatched: its "memory" is a classical leaky integrator,
# not the nuclear register (floor MC = 1.94 with the register wiped every cycle)
cols = [GREY] * 3 + [BLUE] * 2 + [GREY] + [GREEN]
bars = ax.bar(range(7), val, yerr=err, color=cols, width=0.66, capsize=3, zorder=2)
bars[5].set_hatch("///"); bars[5].set_edgecolor("white")

# route-specific floor: a red cap drawn across each bar
for i, f in enumerate(flo):
    ax.plot([i - 0.36, i + 0.36], [f, f], color=RED, lw=1.6, zorder=4,
            solid_capstyle="butt")
ax.plot([], [], color=RED, lw=1.6, label="memoryless floor of that route ($q=1$)")

for i, (v, e, f) in enumerate(zip(val, err, flo)):
    excess = v - f
    if excess < 0.12:                      # bar sits on its own floor
        ax.text(i, v + e + 0.12, "no memory", fontsize=7.39, color=RED,
                ha="center", va="bottom", fontweight="bold")
        continue
    # shade the part of the bar that is genuinely above the floor
    ax.bar(i, excess, bottom=f, width=0.66, facecolor="none",
           edgecolor="#37474f", lw=0.6, ls=(0, (2, 1.5)), zorder=3)
    ax.text(i, v + e + 0.12, f"+{excess:.2f}", fontsize=7.89, color="#263238",
            ha="center", va="bottom", fontweight="bold")

ax.set_xticks(range(7)); ax.set_xticklabels(lab, fontsize=7.64)
ax.set_ylabel("information processing capacity")
ax.set_ylim(0, 6.0)
ax.set_title("What biology can read: excess over each route's own floor",
             loc="left", fontweight="bold", fontsize=11.46)
bars[5].set_label("capacity from a classical downstream integrator")
ax.legend(fontsize=7.39, loc="upper left", framealpha=0.95)
ax.grid(True, axis="y", alpha=0.25, zorder=0)
fig.tight_layout()
fig.savefig("manuscript/figures/fig_readout.pdf", bbox_inches="tight")
fig.savefig("manuscript/figures/fig_readout.png", dpi=300, bbox_inches="tight")

# ─────────────── Fig: clock + criterion ───────────────
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

td = np.array([r["T_d_s"] for r in CK])
mc8 = np.array([r["MC_8ch"] for r in CK])
sd8 = np.array([r["MC_8ch_sd"] for r in CK])
cyc = 1e-6 + td
C0 = KER["cidnp"]["C0"]
h_naive = mc8 * cyc
h_corr = (mc8 - C0) * cyc
e_naive, e_corr = sd8 * cyc, sd8 * cyc

ax[0].axhspan(1e-2, 1.0, color="#ffe0b2", alpha=0.7, zorder=0)
ax[0].text(2e-6, 3.0e-2, "neural band\n(10 ms - 1 s)", fontsize=8.27, color="#e65100")
ax[0].errorbar(td, h_naive, yerr=e_naive, fmt="o-", c=GREY, ms=3.5, lw=1.1,
               capsize=2, zorder=3, label=r"$\mathrm{MC}\times T$  (counts $d\!=\!0$)")
ax[0].errorbar(td, h_corr, yerr=e_corr, fmt="o-", c=GREEN, ms=4, lw=1.5,
               capsize=2, zorder=4, label=r"$(\mathrm{MC}-C_0)\times T$  (delayed memory)")
# reference: the horizon if the clock were the pair lifetime, same 8-channel readout
h_tau = (mc8[0] - C0) * 1e-6
ax[0].axhline(h_tau, color=RED, ls=":", lw=1, zorder=2)
ax[0].text(6e-4, h_tau * 1.35, "horizon if the clock were the pair lifetime (8 ch)",
           fontsize=7.39, color=RED, ha="center")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("turnover interval $T_{\\rm d}$ (s)")
ax[0].set_ylabel("memory horizon (s)")
ax[0].set_title("(a) the clock is the turnover, not $\\tau$",
                loc="left", fontweight="bold", fontsize=11.46)
ax[0].legend(fontsize=7.39, loc="upper left"); ax[0].grid(True, which="both", alpha=0.25)

q = np.array([c["q"] for c in CH])
dep = np.array([c["MC_8_depolarise"] for c in CH])
deph = np.array([c["MC_8_dephase"] for c in CH])
ax[1].plot(q, dep, "o-", c=GREEN, ms=4, lw=1.5,
           label="depolarising ($T_{1n}$): destroys $\\langle I_z\\rangle$")
ax[1].plot(q, deph, "^--", c=ORANGE, ms=4, lw=1.3,
           label="dephasing ($T_{2n}$): preserves $\\langle I_z\\rangle$")
ax[1].axhline(1.0, color=RED, ls=":", lw=1)
ax[1].text(0.97, 1.06, "memoryless floor (MC = 1)", color=RED, fontsize=7.89,
           ha="right", va="bottom")
ax[1].annotate("the register stores\npopulations, not phase",
               xy=(0.72, 2.88), xytext=(0.50, 2.28), fontsize=7.39, color=ORANGE,
               ha="center",
               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
ax[1].set_xlabel("nuclear relaxation per turnover  $q$")
ax[1].set_ylabel("memory capacity MC (8 ch)")
ax[1].set_ylim(0.55, 3.3)
ax[1].set_title("(b) the criterion: register reuse",
                loc="left", fontweight="bold", fontsize=11.46)
ax[1].legend(fontsize=7.39, loc="lower left", framealpha=0.95)
ax[1].grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("manuscript/figures/fig_criterion.pdf", bbox_inches="tight")
fig.savefig("manuscript/figures/fig_criterion.png", dpi=300, bbox_inches="tight")
print("wrote fig_readout and fig_criterion")
