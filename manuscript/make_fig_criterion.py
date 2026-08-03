#!/usr/bin/env python3
"""Figures for the revised manuscript:
  fig_readout   -- what biology can actually read
  fig_criterion -- the clock is the turnover time; the criterion is register reuse
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "stix"})
except Exception:
    pass

R = json.load(open("simulation_results/readout_routes.json"))
T = json.load(open("simulation_results/turnover_clock.json"))
G = json.load(open("simulation_results/register_reuse.json"))

# ─────────────── Fig: readout routes ───────────────
order = ["YS_end", "SandT_end", "hetero_tau", "YS_t", "YS_t_accum", "cidnp"]
lab = ["end-point\nyield\n(1 ch)", "both exit\nchannels\n(2 ch)",
       "3 micro-\nenvironments\n(3 ch)", "time-resolved\nkinetics\n(5 ch)",
       "+ accumulated\npool\n(10 ch)", "+ product\nCIDNP\n(8 ch)"]
val = [R[k]["IPC"] for k in order]
err = [R[k]["IPC_sd"] for k in order]

fig, ax = plt.subplots(figsize=(5.0, 3.0))
cols = ["#90a4ae"] * 3 + ["#1565c0"] * 2 + ["#2e7d32"]
ax.bar(range(6), val, yerr=err, color=cols, width=0.66, capsize=3)
ax.axhline(1.0, color="#c62828", ls=":", lw=1)
ax.text(5.45, 1.06, "memoryless floor", color="#c62828", fontsize=6,
        ha="right", va="bottom")
ax.set_xticks(range(6)); ax.set_xticklabels(lab, fontsize=6.2)
ax.set_ylabel("information processing capacity")
ax.set_title("What biology can read: no spectroscopy required",
             loc="left", fontweight="bold", fontsize=9)
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("manuscript/figures/fig_readout.pdf", bbox_inches="tight")
fig.savefig("manuscript/figures/fig_readout.png", dpi=300, bbox_inches="tight")

# ─────────────── Fig: clock + criterion ───────────────
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

td = np.array([r["T_d_s"] for r in T])
h5 = np.array([r["horizon_5ch_s"] for r in T])
h8 = np.array([r["horizon_8ch_s"] for r in T])
ax[0].axhspan(1e-2, 1.0, color="#ffe0b2", alpha=0.7)
ax[0].text(2e-6, 3e-2, "neural band\n(10 ms - 1 s)", fontsize=6.5, color="#e65100")
ax[0].loglog(td, h8, "o-", c="#2e7d32", ms=4, lw=1.4, label="8-channel readout")
ax[0].loglog(td, h5, "s--", c="#1565c0", ms=4, lw=1.2, label="5-channel readout")
ax[0].axhline(3.9e-6, color="#c62828", ls=":", lw=1)
ax[0].text(8e-4, 1.3e-6, "horizon if the clock were the pair lifetime",
           fontsize=6, color="#c62828", ha="center")
ax[0].set_xlabel("turnover interval $T_{\\rm d}$ (s)")
ax[0].set_ylabel("memory horizon (s)")
ax[0].set_title("(a) the clock is the turnover, not $\\tau$",
                loc="left", fontweight="bold", fontsize=9)
ax[0].legend(fontsize=6.5, loc="upper left"); ax[0].grid(True, which="both", alpha=0.25)

q = np.array([g["q_nuc"] for g in G])
m5 = np.array([g["MC_5"] for g in G])
m8 = np.array([g["MC_8"] for g in G])
ax[1].plot(q, m8, "o-", c="#2e7d32", ms=4, lw=1.4, label="8-channel")
ax[1].plot(q, m5, "s--", c="#1565c0", ms=4, lw=1.2, label="5-channel")
ax[1].axhline(1.0, color="#c62828", ls=":", lw=1)
ax[1].text(0.03, 1.05, "memoryless floor (MC = 1)", color="#c62828", fontsize=6.5)
ax[1].set_xlabel("nuclear depolarisation per turnover  $q$")
ax[1].set_ylabel("memory capacity MC")
ax[1].set_title("(b) the criterion: register reuse",
                loc="left", fontweight="bold", fontsize=9)
ax[1].legend(fontsize=6.5); ax[1].grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("manuscript/figures/fig_criterion.pdf", bbox_inches="tight")
fig.savefig("manuscript/figures/fig_criterion.png", dpi=300, bbox_inches="tight")
print("wrote fig_readout and fig_criterion")
