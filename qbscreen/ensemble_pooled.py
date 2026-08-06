#!/usr/bin/env python3
"""Does a pooled ensemble inherit the single-molecule capacity?

Every capacity in the main text is a single-molecule quantity. A downstream
reaction reads a pool of order 1e10 copies, and the referee panel identified this
as the paper's largest untested premise: unless the copies are driven by a common
input AND share a common turnover schedule, the pooled nonlinear observable need
not inherit the single-molecule capacity.

A density matrix already averages over a *homogeneous* ensemble with a common
clock, so that case is covered by the existing results. What is not covered, and
what this module computes, is the two ways a real pool departs from it:

  E1  heterogeneity     copies differ in their hyperfine couplings (conformational
                        spread), so each has a different spin Hamiltonian
  E2  desynchronisation copies are at different phases of the turnover cycle, so
                        the pooled time-resolved signal mixes cycle phases -- and,
                        for large jitter, mixes successive INPUTS

E2 is the dangerous one: the time-resolved readout presupposes that all copies are
sampled at the same point of their cycle.
"""

import json
import os
import numpy as np

from qbscreen.reservoir import build_reservoir_H, memory_and_ipc
from qbscreen.readout_routes import CRY
from qbscreen.panel_response import run_routes_v2, _inputs

OUT = "simulation_results/panel"
SAMPLE = (16, 36, 56, 76, 96)
N_T = 96


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  [checkpoint] {OUT}/{name}.json")
    return obj


def _copy_params(spread, rng):
    """One copy's hyperfine couplings, log-normally scattered about the CRY point
    (log-normal so couplings stay positive at large spread)."""
    p = dict(CRY)
    if spread > 0:
        for k in ("A_e1_a", "A_e1_b", "A_e2_a"):
            p[k] = float(p[k] * np.exp(rng.normal(0.0, spread)))
    return p


# ─────────────────────────────────────────────────────────────────────
# E1  heterogeneous copies, common clock
# ─────────────────────────────────────────────────────────────────────
def heterogeneity(spreads=(0.0, 0.05, 0.10, 0.20, 0.40), n_copies=16,
                  n_seeds=3, L=700, carrier="survivor"):
    """Pool n_copies whose couplings are scattered by `spread` (fractional, log-normal).
    All copies share the input and the turnover schedule; only the Hamiltonian differs.
    """
    out = []
    for sp in spreads:
        acc = {"YS_t": [], "cidnp": []}
        for sd in range(n_seeds):
            s, spost = _inputs(sd, L)
            prng = np.random.default_rng(1000 + sd)
            pooled = {}
            for c in range(n_copies):
                H = build_reservoir_H(**_copy_params(sp, prng))
                R = run_routes_v2(s, H, carrier=carrier)
                for k in acc:
                    pooled[k] = pooled.get(k, 0.0) + R[k]
            for k in acc:
                acc[k].append(memory_and_ipc(pooled[k] / n_copies, spost)["IPC_total"])
        row = dict(spread=sp, n_copies=n_copies,
                   IPC_5ch=float(np.mean(acc["YS_t"])), IPC_5ch_sd=float(np.std(acc["YS_t"])),
                   IPC_8ch=float(np.mean(acc["cidnp"])), IPC_8ch_sd=float(np.std(acc["cidnp"])))
        out.append(row)
        print(f"  spread={sp:.2f}  pooled IPC 5ch={row['IPC_5ch']:.3f}"
              f"  8ch={row['IPC_8ch']:.3f}", flush=True)
        _save("open1_ensemble_heterogeneity", out)
    return out


# ─────────────────────────────────────────────────────────────────────
# E2  desynchronised turnover
# ─────────────────────────────────────────────────────────────────────
def _traces(inputs, H, carrier="survivor"):
    """Per-cycle cumulative yield traces on the full grid, plus the CIDNP columns.

    Returns yS_full (n_cycles, N_T) and cid (n_cycles, 3): everything needed to
    re-sample the readout at an arbitrary phase offset without re-running.
    """
    full_sample = tuple(range(1, N_T + 1))
    R = run_routes_v2(inputs, H, sample=full_sample, carrier=carrier, washout=0)
    yS_full = R["YS_t"]                      # (n_cycles, N_T) cumulative yield
    cid = R["cidnp"][:, N_T:]                # (n_cycles, 3) polarisation columns
    return yS_full, cid


def desynchronisation(jitters=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0), n_copies=64,
                      n_seeds=3, L=700, washout=80, carrier="survivor"):
    """Copies sit at phase offsets d_j (in units of the cycle) drawn uniformly in
    [-jitter, +jitter]. A copy with offset d is behind by d cycles, so at the
    nominal sample time it is showing an earlier part of its own cycle and, when
    |d| > 1, an earlier INPUT.

    One quantum run per seed supplies the full trace; the pooling is done in
    post-processing, so the jitter scan is cheap.
    """
    H = build_reservoir_H(**CRY)
    out = []
    cache = {}
    for sd in range(n_seeds):
        s, _ = _inputs(sd, L, washout=80)
        cache[sd] = (_traces(s, H, carrier), s)

    for jit in jitters:
        acc5, acc8 = [], []
        for sd in range(n_seeds):
            (yS_full, cid), s = cache[sd]
            n_cyc = yS_full.shape[0]
            rng = np.random.default_rng(500 + sd)
            off = rng.uniform(-jit, jit, n_copies) if jit > 0 else np.zeros(n_copies)
            pooled5 = np.zeros((n_cyc, len(SAMPLE)))
            pooled8 = np.zeros((n_cyc, len(SAMPLE) + 3))
            for d in off:
                cyc_shift = int(np.floor(d))            # whole cycles behind
                frac = d - cyc_shift                    # remaining phase, in [0,1)
                idx = np.array(SAMPLE) - 1 - frac * N_T
                lo = np.clip(np.floor(idx).astype(int), 0, N_T - 1)
                hi = np.clip(lo + 1, 0, N_T - 1)
                w = np.clip(idx - lo, 0.0, 1.0)
                k = np.arange(n_cyc) - cyc_shift        # which cycle this copy shows
                k = np.clip(k, 0, n_cyc - 1)
                tr = (1 - w) * yS_full[k][:, lo] + w * yS_full[k][:, hi]
                pooled5 += tr
                pooled8[:, :len(SAMPLE)] += tr
                pooled8[:, len(SAMPLE):] += cid[k]
            pooled5 /= n_copies; pooled8 /= n_copies
            sp = s[washout:]
            acc5.append(memory_and_ipc(pooled5[washout:], sp)["IPC_total"])
            acc8.append(memory_and_ipc(pooled8[washout:], sp)["IPC_total"])
        row = dict(jitter_cycles=jit, n_copies=n_copies,
                   IPC_5ch=float(np.mean(acc5)), IPC_5ch_sd=float(np.std(acc5)),
                   IPC_8ch=float(np.mean(acc8)), IPC_8ch_sd=float(np.std(acc8)))
        out.append(row)
        print(f"  jitter={jit:.2f} cycles  pooled IPC 5ch={row['IPC_5ch']:.3f}"
              f"  8ch={row['IPC_8ch']:.3f}", flush=True)
        _save("open1_ensemble_desync", out)
    return out



def desync_with_floor(jitters=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0), n_copies=64,
                      n_seeds=3, L=700, washout=80, carrier="survivor"):
    """Jitter scan together with its register-wiped floor.

    This is the generator for open1_ensemble_desync_floor.json, which the set-5
    audit found to be load-bearing but orphaned: the entire excess column of the
    SI jitter table, and the main text's "the survivor register retains under
    1 %", come from it. Without the q = 1 control the raw IPC is uninterpretable,
    because pooling copies that sit whole cycles apart is itself a delay line.
    """
    H = build_reservoir_H(**CRY)
    live = {sd: _traces(_inputs(sd, L)[0], H, carrier) for sd in range(n_seeds)}
    wiped = {}
    for sd in range(n_seeds):
        R = run_routes_v2(_inputs(sd, L)[0], H, sample=tuple(range(1, N_T + 1)),
                          carrier=carrier, washout=0, q_nuc=1.0)
        wiped[sd] = (R["YS_t"], R["cidnp"][:, N_T:])

    def pool(yS_full, cid, jit, seed):
        n_cyc = yS_full.shape[0]
        rng = np.random.default_rng(500 + seed)
        off = rng.uniform(-jit, jit, n_copies) if jit > 0 else np.zeros(n_copies)
        p5 = np.zeros((n_cyc, len(SAMPLE))); p8 = np.zeros((n_cyc, len(SAMPLE) + 3))
        for d in off:
            cs = int(np.floor(d)); frac = d - cs
            idx = np.array(SAMPLE) - 1 - frac * N_T
            lo = np.clip(np.floor(idx).astype(int), 0, N_T - 1)
            hi = np.clip(lo + 1, 0, N_T - 1)
            w = np.clip(idx - lo, 0.0, 1.0)
            k = np.clip(np.arange(n_cyc) - cs, 0, n_cyc - 1)
            tr = (1 - w) * yS_full[k][:, lo] + w * yS_full[k][:, hi]
            p5 += tr; p8[:, :len(SAMPLE)] += tr; p8[:, len(SAMPLE):] += cid[k]
        return p5 / n_copies, p8 / n_copies

    out = []
    for jit in jitters:
        rec = {"jitter_cycles": jit}
        for tag, cache in (("live", live), ("floor_q1", wiped)):
            a5, a8 = [], []
            for sd in range(n_seeds):
                s = _inputs(sd, L)[0]; sp = s[washout:]
                q5, q8 = pool(cache[sd][0], cache[sd][1], jit, sd)
                a5.append(memory_and_ipc(q5[washout:], sp)["IPC_total"])
                a8.append(memory_and_ipc(q8[washout:], sp)["IPC_total"])
            rec[f"IPC_5ch_{tag}"] = float(np.mean(a5))
            rec[f"IPC_8ch_{tag}"] = float(np.mean(a8))
        rec["excess_5ch"] = rec["IPC_5ch_live"] - rec["IPC_5ch_floor_q1"]
        rec["excess_8ch"] = rec["IPC_8ch_live"] - rec["IPC_8ch_floor_q1"]
        out.append(rec)
        print(f"  jitter={jit:4.2f}  excess 5ch={rec['excess_5ch']:+.3f}"
              f"  8ch={rec['excess_8ch']:+.3f}", flush=True)
        _save("open1_ensemble_desync_floor", out)
    return out



# ─────────────────────────────────────────────────────────────────────
# E3  heterogeneous tau_c  --  copies relax at different rates
# ─────────────────────────────────────────────────────────────────────
def tau_c_heterogeneity(sigma_ln=(0.0, 0.3, 0.6, 1.0), tau_c_med_ns=5.0,
                        T_d=1e-2, n_q=6, n_seeds=3, L=700, washout=80,
                        carrier="survivor"):
    """Pool copies whose rotational correlation times differ.

    E1 and E2 vary the Hamiltonian and the cycle phase. Neither touches the
    quantity the register criterion actually turns on: tau_c, which sets T_1n,
    which sets how much of the register survives a turnover. A real pool of
    product molecules does not tumble at one rate -- some stay bound to the
    protein, some diffuse free -- so the pool contains copies at every point
    between "register intact" and "register wiped".

    This is the last uncomputed premise the audit left open, and it is not
    obviously benign: pooling a fast-relaxing sub-population with a slow one
    could plausibly wash out the slow one's memory, since the readout is a sum
    over copies and cannot tell them apart.

    tau_c is log-normal with median tau_c_med_ns and log-spread sigma_ln. The
    copies are stratified quantiles rather than random draws -- same estimator,
    much lower variance at n_q samples. Each copy's survival follows the paper's
    own relation, q_j = 1 - exp(-T_d / T1(tau_c_j)), so this reuses the
    relaxation model of Sec. S12 rather than inventing a second one.
    """
    from scipy.stats import norm
    from qbscreen.relaxation_estimate import dipolar_T1, GEOM

    g = GEOM["Trp H-beta (CH2, geminal partner)"]
    H = build_reservoir_H(**CRY)

    def q_of(tau_c_ns):
        T1 = dipolar_T1(g["r"], tau_c_ns * 1e-9, 50e-6, n_partners=g["n"])
        return float(1.0 - np.exp(-T_d / T1))

    # one reservoir run per distinct q, cached across the sigma scan
    cache = {}

    def traces(q, seed):
        key = (round(q, 6), seed)
        if key not in cache:
            s, _ = _inputs(seed, L)
            R = run_routes_v2(s, H, q_nuc=q, carrier=carrier)
            cache[key] = (R["YS_t"], R["cidnp"])
        return cache[key]

    # stratified quantiles of the standard normal, shared by every sigma
    z = norm.ppf((np.arange(n_q) + 0.5) / n_q)

    out = []
    for sig in sigma_ln:
        taus = tau_c_med_ns * np.exp(sig * z)
        qs = [q_of(t) for t in taus]
        a5, a8, f5, f8 = [], [], [], []
        for sd in range(n_seeds):
            # run_routes_v2 already drops the washout, and _inputs returns the
            # matching post-washout drive -- trimming again would misalign them
            _, sp = _inputs(sd, L)
            p5 = p8 = None
            for q in qs:
                y, c = traces(q, sd)
                p5 = y.copy() if p5 is None else p5 + y
                p8 = c.copy() if p8 is None else p8 + c
            p5 /= len(qs); p8 /= len(qs)
            a5.append(memory_and_ipc(p5, sp)["IPC_total"])
            a8.append(memory_and_ipc(p8, sp)["IPC_total"])
            # the pooled floor: every copy wiped every cycle
            fy, fc = traces(1.0, sd)
            f5.append(memory_and_ipc(fy, sp)["IPC_total"])
            f8.append(memory_and_ipc(fc, sp)["IPC_total"])
        # mean-field control: one homogeneous copy at the pool's MEAN q. If this
        # reproduces the pooled excess, the spread in tau_c does nothing on its
        # own and the whole effect is the shift in mean survival -- a much
        # stronger statement than "pooling happens to be benign".
        qbar = float(np.mean(qs))
        mf5, mf8 = [], []
        for sd in range(n_seeds):
            _, sp = _inputs(sd, L)
            y, c = traces(qbar, sd)
            mf5.append(memory_and_ipc(y, sp)["IPC_total"])
            mf8.append(memory_and_ipc(c, sp)["IPC_total"])

        row = dict(sigma_ln=sig, tau_c_median_ns=tau_c_med_ns, T_d_s=T_d,
                   n_copies=n_q, n_seeds=n_seeds,
                   tau_c_min_ns=float(taus.min()), tau_c_max_ns=float(taus.max()),
                   q_min=float(min(qs)), q_max=float(max(qs)),
                   q_mean=float(np.mean(qs)),
                   IPC_5ch=float(np.mean(a5)), IPC_8ch=float(np.mean(a8)),
                   IPC_5ch_floor=float(np.mean(f5)), IPC_8ch_floor=float(np.mean(f8)))
        row["excess_5ch"] = row["IPC_5ch"] - row["IPC_5ch_floor"]
        row["excess_8ch"] = row["IPC_8ch"] - row["IPC_8ch_floor"]
        row["IPC_5ch_meanfield"] = float(np.mean(mf5))
        row["IPC_8ch_meanfield"] = float(np.mean(mf8))
        row["excess_5ch_meanfield"] = row["IPC_5ch_meanfield"] - row["IPC_5ch_floor"]
        row["excess_8ch_meanfield"] = row["IPC_8ch_meanfield"] - row["IPC_8ch_floor"]
        row["spread_effect_8ch"] = row["excess_8ch"] - row["excess_8ch_meanfield"]
        out.append(row)
        print(f"  sigma={sig:.1f}  tau_c {taus.min():.2f}-{taus.max():.2f} ns"
              f"  q {min(qs):.3f}-{max(qs):.3f} (mean {qbar:.3f})"
              f"  excess 8ch={row['excess_8ch']:+.3f}"
              f"  mean-field={row['excess_8ch_meanfield']:+.3f}"
              f"  spread alone={row['spread_effect_8ch']:+.4f}", flush=True)
        _save("open6_tau_c_heterogeneity", out)
    return out

if __name__ == "__main__":
    import sys, time
    which = sys.argv[1:] or ["desync", "floor", "hetero", "tauc"]
    for w in which:
        t0 = time.time()
        print(f"\n=== {w} ===", flush=True)
        {"desync": desynchronisation, "floor": desync_with_floor,
         "hetero": heterogeneity, "tauc": tau_c_heterogeneity}[w]()
        print(f"  ({time.time()-t0:.0f} s)", flush=True)
