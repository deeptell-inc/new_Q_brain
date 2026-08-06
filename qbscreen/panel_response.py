#!/usr/bin/env python3
"""Data supplementation demanded by the adversarial referee panel.

Every function here answers one Critical or Major finding recorded in
``manuscript/manuscript-REVIEW.md``.  Each writes its own JSON checkpoint under
``simulation_results/panel/`` as soon as it finishes, so a interrupted session
never loses a completed scan.

  C1  product_carryover       the register is the PRODUCT nucleus, not the survivor
  C2  clock_multiseed         error budget: seed noise vs the claimed effect
  C2  mc_grid_convergence     grid convergence of MC (never previously reported)
  C3  delay_kernel            horizon from the delay kernel, not from MC x T
  C4  nuclear_channel         T1-type depolarisation vs T2-type dephasing
  M6  cry_classical           feature-matched ESN baselines AT the cryptochrome point
  M7  route_floors            per-route memoryless floor (q = 1)
  M9  ridge_sensitivity       lambda as an implicit readout SNR, + shot noise
"""

import json
import os
import numpy as np
from scipy.linalg import expm

from qbscreen.reservoir import (build_reservoir_H, memory_and_ipc, _capacity,
                                N_SPINS, DIM, NUCLEI)
from qbscreen.corrected_injection import _electron_born_state
from qbscreen.spin_dynamics import spin_op, singlet_projector, SZ
from qbscreen.master_equation import build_liouvillian, _vec, electron_dephasing_ops
from qbscreen.readout_routes import CRY, _cycle_operators, D_NUC, accumulate
from qbscreen.qrc_benchmarks import esn_states

TWO_PI = 2.0 * np.pi
OUT = "simulation_results/panel"
SAMPLE = (16, 36, 56, 76, 96)


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  [checkpoint] {OUT}/{name}.json")
    return obj


def _nuc_from_electron_trace(rho):
    """Partial trace over the two electrons -> normalised nuclear state."""
    R = rho.reshape(4, D_NUC, 4, D_NUC)
    nuc = np.trace(R, axis1=0, axis2=2)
    t = np.trace(nuc).real
    return nuc / (t if t else 1.0)


# ─────────────────────────────────────────────────────────────────────
# C1 / C4  the cycle, with a selectable register carrier and relaxation channel
# ─────────────────────────────────────────────────────────────────────
def run_routes_v2(inputs, H_mhz, tau_us=1.0, T2e_ns=1000.0, kS=1.0, kT=0.2,
                  n_t=96, sample=SAMPLE, washout=80, q_nuc=0.0,
                  carrier="product", relax="depolarise"):
    """One turnover chain, with the two referee-contested choices made explicit.

    carrier = "product"  : the register handed to the next turnover is the nuclear
                           state carried away by the RECOMBINED molecule,
                             sigma ~ int [k_S Tr_e(P_S rho P_S) + k_T Tr_e(P_T rho P_T)] dt,
                           which is what the manuscript's Fig. 1(b) actually claims.
              "survivor" : the published behaviour -- the trace-decreasing state
                           that did NOT recombine, renormalised.
    relax   = "depolarise": nuc -> (1-q) nuc + q I/D   (T1-type, destroys <I_z>)
              "dephase"   : per-nucleus Z dephasing    (T2-type, preserves <I_z>)
    """
    P_S, P_T, step, dt = _cycle_operators(H_mhz, tau_us, T2e_ns, kS, kT, n_t)
    cid_ops = [spin_op(2 * SZ, i, N_SPINS) for i in NUCLEI]
    Z_nuc = [spin_op(2 * SZ, i, N_SPINS) for i in NUCLEI]
    I_nuc = np.eye(D_NUC, dtype=complex) / D_NUC

    rows = {k: [] for k in ("YS_end", "YS_t", "SandT_end", "SandT_t", "cidnp")}
    rho = singlet_projector(0, 1, N_SPINS).astype(complex)
    rho = rho / np.trace(rho)
    for s in inputs:
        nuc = _nuc_from_electron_trace(rho)
        if q_nuc > 0:
            if relax == "depolarise":
                nuc = (1.0 - q_nuc) * nuc + q_nuc * I_nuc
            elif relax == "dephase":
                # T2-type: full Z dephasing on each nucleus with probability q.
                # Embed in the full space, dephase, trace back out.
                big = np.kron(np.eye(4) / 4.0, nuc)
                for Zi in Z_nuc:
                    big = (1 - q_nuc) * big + q_nuc * 0.5 * (big + Zi @ big @ Zi)
                nuc = _nuc_from_electron_trace(big)
            else:
                raise ValueError(relax)
        rho = np.kron(_electron_born_state(s, "ST"), nuc)

        v = _vec(rho)
        yS = yT = 0.0
        yIz = np.zeros(len(cid_ops))
        yS_t, yT_t = [], []
        M_prod = np.zeros((D_NUC, D_NUC), dtype=complex)
        for it in range(1, n_t + 1):
            v = step @ v
            r = v.reshape(DIM, DIM, order="F")
            yS += kS * np.real(np.trace(P_S @ r)) * dt
            yT += kT * np.real(np.trace(P_T @ r)) * dt
            for j, O in enumerate(cid_ops):
                yIz[j] += kS * np.real(np.trace(O @ P_S @ r)) * dt
            if carrier == "product":
                # nuclear state leaving in the diamagnetic product this instant
                aS = P_S @ r @ P_S
                aT = P_T @ r @ P_T
                M_prod += (kS * _sub_trace_e(aS) + kT * _sub_trace_e(aT)) * dt
            if it in sample:
                yS_t.append(yS); yT_t.append(yT)
        rows["YS_end"].append([yS])
        rows["YS_t"].append(list(yS_t))
        rows["SandT_end"].append([yS, yT])
        rows["SandT_t"].append(list(yS_t) + list(yT_t))
        rows["cidnp"].append(list(yS_t) + list(yIz / (yS if yS else 1.0)))

        if carrier == "product":
            M_prod = 0.5 * (M_prod + M_prod.conj().T)
            tp = np.trace(M_prod).real
            nuc_next = M_prod / (tp if tp else 1.0)
            rho = np.kron(np.eye(4, dtype=complex) / 4.0, nuc_next)
        else:
            rr = v.reshape(DIM, DIM, order="F"); rr = 0.5 * (rr + rr.conj().T)
            t2 = np.trace(rr).real
            rho = rr / (t2 if t2 else 1.0)
    return {k: np.array(v[washout:]) for k, v in rows.items()}


def _sub_trace_e(A):
    """Trace out the two electrons of a DIM x DIM operator -> D_NUC x D_NUC."""
    R = A.reshape(4, D_NUC, 4, D_NUC)
    return np.trace(R, axis1=0, axis2=2)


def _routes_capacity(R, sp, keys=("YS_t", "cidnp")):
    return {k: memory_and_ipc(R[k], sp) for k in keys}


def _inputs(seed, L=700, washout=80):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, L + washout)
    return s, s[washout:]


# ─────────────────────────────────────────────────────────────────────
# C1  does the headline survive a genuine product register?
# ─────────────────────────────────────────────────────────────────────
def product_carryover(n_seeds=6, L=700):
    H = build_reservoir_H(**CRY)
    out = {}
    for carrier in ("survivor", "product"):
        acc = {}
        for sd in range(n_seeds):
            s, sp = _inputs(sd, L)
            R = run_routes_v2(s, H, carrier=carrier)
            R["YS_t_accum"] = accumulate(R["YS_t"])
            for k in ("YS_end", "YS_t", "SandT_t", "cidnp", "YS_t_accum"):
                r = memory_and_ipc(R[k], sp)
                d = acc.setdefault(k, {"ipc": [], "mc": [], "ch": R[k].shape[1]})
                d["ipc"].append(r["IPC_total"]); d["mc"].append(r["MC"])
        out[carrier] = {k: dict(channels=v["ch"],
                                IPC=float(np.mean(v["ipc"])), IPC_sd=float(np.std(v["ipc"])),
                                MC=float(np.mean(v["mc"])), MC_sd=float(np.std(v["mc"])))
                        for k, v in acc.items()}
        print(f"  carrier={carrier:9s} " + "  ".join(
            f"{k}:{out[carrier][k]['IPC']:.3f}" for k in ("YS_t", "cidnp")))
    return _save("c1_product_carryover", out)


# ─────────────────────────────────────────────────────────────────────
# M7  per-route memoryless floor
# ─────────────────────────────────────────────────────────────────────
def route_floors(n_seeds=6, L=700, carrier="survivor"):
    """q = 1 wipes the register every cycle, so whatever capacity remains is the
    route's own memoryless floor -- NOT the universal IPC = 1 drawn in Fig. 2."""
    H = build_reservoir_H(**CRY)
    acc = {}
    for sd in range(n_seeds):
        s, sp = _inputs(sd, L)
        R = run_routes_v2(s, H, q_nuc=1.0, carrier=carrier)
        R["YS_t_accum"] = accumulate(R["YS_t"])
        for k, X in R.items():
            r = memory_and_ipc(X, sp)
            d = acc.setdefault(k, {"ipc": [], "mc": [], "ch": X.shape[1]})
            d["ipc"].append(r["IPC_total"]); d["mc"].append(r["MC"])
    out = {k: dict(channels=v["ch"], floor_IPC=float(np.mean(v["ipc"])),
                   floor_IPC_sd=float(np.std(v["ipc"])), floor_MC=float(np.mean(v["mc"])))
           for k, v in acc.items()}
    for k, v in out.items():
        print(f"  {k:12s} ch={v['channels']:>2}  floor IPC={v['floor_IPC']:.3f}"
              f"+/-{v['floor_IPC_sd']:.3f}  floor MC={v['floor_MC']:.3f}")
    return _save("m7_route_floors", out)


# ─────────────────────────────────────────────────────────────────────
# C3  the delay kernel, and a horizon that is actually a time
# ─────────────────────────────────────────────────────────────────────
def delay_kernel(n_seeds=6, L=700, carrier="survivor"):
    """Report C[d] for d >= 0 and derive three candidate horizons, so the reader
    can see that MC x T_cycle counts the d = 0 term that carries no memory."""
    H = build_reservoir_H(**CRY)
    acc = {"YS_t": [], "cidnp": []}
    for sd in range(n_seeds):
        s, sp = _inputs(sd, L)
        R = run_routes_v2(s, H, carrier=carrier)
        for k in acc:
            acc[k].append(memory_and_ipc(R[k], sp)["lin_by_delay"])
    out = {}
    for k, v in acc.items():
        ker = np.mean(np.array(v), axis=0)
        tail = float(ker[1:].sum())                       # delayed memory only
        thr = 0.05 * ker[0]
        cross = int(np.argmax(ker < thr)) if np.any(ker < thr) else len(ker)
        out[k] = dict(kernel=[float(x) for x in ker], C0=float(ker[0]),
                      MC=float(ker.sum()), MC_minus_C0=tail,
                      delay_5pct_of_C0=cross)
        print(f"  {k:8s} C[0]={ker[0]:.3f}  C[1]={ker[1]:.3f}  C[2]={ker[2]:.3f}"
              f"  MC={ker.sum():.3f}  MC-C0={tail:.3f}  d(5%)={cross}")
    return _save("c3_delay_kernel", out)


# ─────────────────────────────────────────────────────────────────────
# C2  the error budget the manuscript never computed
# ─────────────────────────────────────────────────────────────────────
def clock_multiseed(n_seeds=12, L=700, T1n_s=1.0, carrier="survivor",
                    T_d_list=(1e-6, 1e-3, 1e-2, 1e-1)):
    H = build_reservoir_H(**CRY)
    out = []
    for T_d in T_d_list:
        q = 1.0 - np.exp(-T_d / T1n_s)
        m5, m8 = [], []
        for sd in range(n_seeds):
            s, sp = _inputs(sd, L)
            R = run_routes_v2(s, H, q_nuc=q, carrier=carrier)
            m5.append(memory_and_ipc(R["YS_t"], sp)["MC"])
            m8.append(memory_and_ipc(R["cidnp"], sp)["MC"])
        cyc = 1e-6 + T_d
        row = dict(T_d_s=T_d, q_nuc=float(q), n_seeds=n_seeds,
                   MC_5ch=float(np.mean(m5)), MC_5ch_sd=float(np.std(m5, ddof=1)),
                   MC_8ch=float(np.mean(m8)), MC_8ch_sd=float(np.std(m8, ddof=1)),
                   MC_8ch_sem=float(np.std(m8, ddof=1) / np.sqrt(n_seeds)),
                   horizon_naive_8ch_s=float(np.mean(m8)) * cyc,
                   horizon_corrected_8ch_s=float(np.mean(m8) - 1.0) * cyc)
        out.append(row)
        print(f"  T_d={T_d:8.1e}  MC_8={row['MC_8ch']:.4f}+/-{row['MC_8ch_sd']:.4f}"
              f"  naive={row['horizon_naive_8ch_s']:.3e}s"
              f"  corrected={row['horizon_corrected_8ch_s']:.3e}s")
        _save("c2_clock_multiseed", out)          # checkpoint every point
    eff = out[0]["MC_8ch"] - out[-1]["MC_8ch"]
    noise = float(np.hypot(out[0]["MC_8ch_sd"], out[-1]["MC_8ch_sd"]))
    print(f"  claimed effect={eff:.4f}   seed noise (combined sd)={noise:.4f}"
          f"   ratio={eff/noise if noise else float('nan'):.2f}")
    return out


def mc_grid_convergence(n_t_list=(48, 96, 192), n_seeds=6, L=700, carrier="survivor"):
    """MC grid convergence -- the manuscript reports IPC convergence only, yet the
    invariance claim is made about MC."""
    H = build_reservoir_H(**CRY)
    out = []
    for n_t in n_t_list:
        smp = tuple(int(round(n_t * f)) for f in (16/96, 36/96, 56/96, 76/96, 1.0))
        m5, m8 = [], []
        for sd in range(n_seeds):
            s, sp = _inputs(sd, L)
            R = run_routes_v2(s, H, n_t=n_t, sample=smp, carrier=carrier)
            m5.append(memory_and_ipc(R["YS_t"], sp)["MC"])
            m8.append(memory_and_ipc(R["cidnp"], sp)["MC"])
        row = dict(n_t=n_t, dt_ns=1000.0/n_t,
                   MC_5ch=float(np.mean(m5)), MC_5ch_sd=float(np.std(m5, ddof=1)),
                   MC_8ch=float(np.mean(m8)), MC_8ch_sd=float(np.std(m8, ddof=1)))
        out.append(row); print(f"  n_t={n_t:>4}  MC_5={row['MC_5ch']:.4f}  MC_8={row['MC_8ch']:.4f}")
        _save("c2_mc_grid", out)
    return out


# ─────────────────────────────────────────────────────────────────────
# C4  which nuclear relaxation channel?
# ─────────────────────────────────────────────────────────────────────
def nuclear_channel(qs=(0.0, 0.3, 0.5, 0.7, 0.9, 1.0), n_seeds=4, L=700,
                    carrier="survivor"):
    """T1-type depolarisation (published) vs T2-type Z dephasing (preserves <I_z>).
    The CIDNP readout IS <I_z>, so the choice decides whether the criterion binds."""
    H = build_reservoir_H(**CRY)
    out = []
    for q in qs:
        row = dict(q=float(q))
        for relax in ("depolarise", "dephase"):
            m8, i8 = [], []
            for sd in range(n_seeds):
                s, sp = _inputs(sd, L)
                R = run_routes_v2(s, H, q_nuc=q, relax=relax, carrier=carrier)
                r = memory_and_ipc(R["cidnp"], sp)
                m8.append(r["MC"]); i8.append(r["IPC_total"])
            row[f"MC_8_{relax}"] = float(np.mean(m8))
            row[f"IPC_8_{relax}"] = float(np.mean(i8))
        out.append(row)
        print(f"  q={q:.2f}  MC_8 depol={row['MC_8_depolarise']:.3f}"
              f"   dephase={row['MC_8_dephase']:.3f}")
        _save("c4_nuclear_channel", out)
    return out


# ─────────────────────────────────────────────────────────────────────
# M6  classical baselines AT the cryptochrome point, feature-matched
# ─────────────────────────────────────────────────────────────────────
def cry_classical(n_seeds=6, L=700, carrier="survivor"):
    H = build_reservoir_H(**CRY)
    q_ipc = {"YS_t": [], "cidnp": []}
    for sd in range(n_seeds):
        s, sp = _inputs(sd, L)
        R = run_routes_v2(s, H, carrier=carrier)
        for k in q_ipc:
            q_ipc[k].append(memory_and_ipc(R[k], sp)["IPC_total"])
    out = {"quantum_5ch_kinetics": [float(np.mean(q_ipc["YS_t"])), float(np.std(q_ipc["YS_t"]))],
           "quantum_8ch_cidnp": [float(np.mean(q_ipc["cidnp"])), float(np.std(q_ipc["cidnp"]))]}
    for n_nodes, n_feat, tag in ((5, 5, "ESN_5node_5feat"), (8, 8, "ESN_8node_8feat"),
                                 (12, 12, "ESN_12node_12feat")):
        vals = []
        for sd in range(n_seeds):
            s, sp = _inputs(sd, L)
            X = esn_states(s, n_nodes, seed=100 + sd)[80:]
            vals.append(memory_and_ipc(X[:, :n_feat], sp)["IPC_total"])
        out[tag] = [float(np.mean(vals)), float(np.std(vals))]
    for k, v in out.items():
        print(f"  {k:26s} IPC={v[0]:.3f}+/-{v[1]:.3f}")
    return _save("m6_cry_classical", out)


# ─────────────────────────────────────────────────────────────────────
# M9  lambda is an implicit readout SNR
# ─────────────────────────────────────────────────────────────────────
def ridge_sensitivity(L=700, seed=0, carrier="survivor",
                      lams=(1e-12, 1e-9, 1e-6, 1e-3, 1e-1),
                      n_molecules=(1e12, 1e10, 1e8, 1e6)):
    """Three questions at once:
    (a) how strongly does the reported capacity depend on the unmotivated ridge?
    (b) does standardising the features (never done) change that?
    (c) if the product pool is read with Poisson shot noise from N molecules,
        what capacity survives?  This is the physically meaningful regulariser.
    """
    H = build_reservoir_H(**CRY)
    s, sp = _inputs(seed, L)
    R = run_routes_v2(s, H, carrier=carrier)
    out = {"lambda_scan": [], "shot_noise": [], "condition_number": {}}
    for k in ("YS_t", "cidnp"):
        X = R[k]
        Xc = X - X.mean(0)
        out["condition_number"][k] = float(np.linalg.cond(Xc.T @ Xc))
    for lam in lams:
        row = {"lambda": lam}
        for k in ("YS_t", "cidnp"):
            X = R[k]
            row[f"{k}_raw"] = float(_ipc_with(X, sp, lam))
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-30)     # standardised
            row[f"{k}_std"] = float(_ipc_with(Xs, sp, lam))
        out["lambda_scan"].append(row)
        print(f"  lam={lam:.0e}  kinetics raw={row['YS_t_raw']:.3f} std={row['YS_t_std']:.3f}"
              f"   cidnp raw={row['cidnp_raw']:.3f} std={row['cidnp_std']:.3f}")
    rng = np.random.default_rng(12345)
    for N in n_molecules:
        row = {"n_molecules": N}
        for k in ("YS_t", "cidnp"):
            X = R[k]
            scale = np.abs(X).mean(0) + 1e-30
            noisy = X + rng.normal(0, 1, X.shape) * scale / np.sqrt(N)
            row[k] = float(_ipc_with(noisy, sp, 1e-6))
        out["shot_noise"].append(row)
        print(f"  N={N:.0e}  kinetics={row['YS_t']:.3f}  cidnp={row['cidnp']:.3f}")
    return _save("m9_ridge_sensitivity", out)


def _ipc_with(X, s, ridge):
    """memory_and_ipc with a settable ridge (the library hard-codes 1e-6)."""
    import qbscreen.reservoir as rv
    orig = rv._capacity
    def patched(Xa, target, ridge=ridge):
        return orig(Xa, target, ridge=ridge)
    rv._capacity = patched
    try:
        return memory_and_ipc(X, s)["IPC_total"]
    finally:
        rv._capacity = orig



def clock_paired(n_seeds=12, L=700, T1n_s=1.0, carrier="survivor",
                 T_d=(1e-6, 1e-1)):
    """Paired-difference test for the turnover-clock claim.

    The same seeds drive every T_d, so the change across the scan must be assessed
    on paired differences, not on the between-seed spread. This function is the
    generator for c2_clock_paired.json, which the manuscript quotes.
    """
    H = build_reservoir_H(**CRY)
    per = {t: [] for t in T_d}
    for sd in range(n_seeds):
        s, sp = _inputs(sd, L)
        for t in T_d:
            q = 1.0 - np.exp(-t / T1n_s)
            R = run_routes_v2(s, H, q_nuc=q, carrier=carrier)
            per[t].append(memory_and_ipc(R["cidnp"], sp)["MC"])
    a = np.array(per[T_d[0]]); b = np.array(per[T_d[1]]); d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    out = dict(n_seeds=n_seeds, MC_8_1us=list(map(float, a)),
               MC_8_100ms=list(map(float, b)),
               paired_diff_mean=float(d.mean()), paired_diff_sd=float(d.std(ddof=1)),
               paired_sem=float(sem), t_stat=float(d.mean() / sem),
               relative_change_pct=float(100 * d.mean() / a.mean()))
    print(f"  paired n={n_seeds}: diff={out['paired_diff_mean']:.4f}"
          f"+/-{sem:.4f} (SEM)  t={out['t_stat']:.2f}"
          f"  {out['relative_change_pct']:.3f} %")
    return _save("c2_clock_paired", out)


def coherence_control(n_seeds=6, L=900, gammas=((1e3, 1.0), (5e3, 5.0), (2e4, 20.0))):
    """Is the 0.82 coherence fraction an artifact of dephasing the NUCLEI too?

    The published channel applies gamma to all five spins. Restricting it to the
    two electrons isolates the electronic contribution. Generator for
    c6_coherence_control.json.
    """
    from scipy.linalg import expm
    from qbscreen.reservoir import _observable_set, ELECTRONS
    from qbscreen.final_numbers import ENG, run_corr_obs
    import qbscreen.final_numbers as fn

    def prop_sel(H, tau_us, T2e_ns, gamma, spins):
        Z = np.zeros((DIM, DIM))
        deph = electron_dephasing_ops(N_SPINS, None if T2e_ns is None else T2e_ns * 1e-3,
                                      ELECTRONS)
        if gamma > 0:
            deph = deph + [(TWO_PI * gamma, spin_op(SZ, i, N_SPINS)) for i in spins]
        with np.errstate(over="ignore", invalid="ignore"):
            return expm(build_liouvillian(TWO_PI * H, Z, Z, 0., 0., deph) * tau_us)

    obs = _observable_set("full")
    H = build_reservoir_H(**ENG)
    COH = 5.420585739613524
    out = {"coherent_reference_IPC": COH}
    for label, spins in (("all5", range(N_SPINS)), ("electrons_only", ELECTRONS)):
        rows = []
        for gamma, tau in gammas:
            vals = []
            for sd in range(n_seeds):
                rng = np.random.default_rng(sd)
                s = rng.uniform(0, 1, L + 100)
                orig = fn._prop
                fn._prop = lambda h, t, T, g=0.0, _s=spins: prop_sel(h, t, T, g, _s)
                try:
                    X = run_corr_obs(s, H, tau, 50.0, obs, gamma=gamma)
                finally:
                    fn._prop = orig
                vals.append(memory_and_ipc(X, s[100:])["IPC_total"])
            rows.append(dict(gamma_MHz=gamma, tau_us=tau, IPC=float(np.mean(vals)),
                             sd=float(np.std(vals))))
        mean = float(np.mean([r["IPC"] for r in rows]))
        out[label] = dict(rows=rows, mean_IPC=mean,
                          coherence_fraction=float((COH - mean) / COH))
        print(f"  {label:15s} mean IPC={mean:.6f}  "
              f"coherence fraction={out[label]['coherence_fraction']:.4f}")
    return _save("c6_coherence_control", out)


def dambre_control(n_seeds=30, L=1200):
    """The finite-sample violation of the Dambre bound, quoted in SI S2.

    For a single observable X = s the asymptotic bound is IPC <= 1, but the
    out-of-sample estimator exceeds it in finite samples. The manuscript quotes
    "1.011 on average, and above the bound in 30 of 30 seeds"; that claim had no
    stored data until this generator was added.
    """
    vals = []
    for sd in range(n_seeds):
        rng = np.random.default_rng(7000 + sd)
        s = rng.uniform(0, 1, L)
        X = s.reshape(-1, 1)                      # the single observable is the input
        vals.append(memory_and_ipc(X, s)["IPC_total"])
    v = np.array(vals)
    out = dict(n_seeds=n_seeds, L=L, IPC=list(map(float, v)),
               mean_IPC=float(v.mean()), sd_IPC=float(v.std(ddof=1)),
               n_above_bound=int((v > 1.0).sum()))
    print(f"  Dambre control: mean IPC = {out['mean_IPC']:.4f}, "
          f"above the bound in {out['n_above_bound']}/{n_seeds} seeds")
    return _save("s2_dambre_control", out)

ALL = {
    "c1": ("C1 product vs survivor register", product_carryover),
    "m7": ("M7 per-route memoryless floor", route_floors),
    "c3": ("C3 delay kernel and horizon", delay_kernel),
    "c2clock": ("C2 clock scan, multi-seed error budget", clock_multiseed),
    "c2grid": ("C2 MC grid convergence", mc_grid_convergence),
    "c4": ("C4 nuclear relaxation channel", nuclear_channel),
    "m6": ("M6 classical baselines at the cryptochrome point", cry_classical),
    "m9": ("M9 ridge as implicit readout SNR", ridge_sensitivity),
    "c2paired": ("C2 paired-difference clock test", clock_paired),
    "c6": ("C6 electron-only dephasing control", coherence_control),
    "s2": ("S2 Dambre finite-sample control", dambre_control),
}



def hetero_floor(n_seeds=6, L=700, taus=(0.5, 1.0, 2.0), T2e_ns=1000.0,
                 kS=1.0, kT=0.2, washout=80):
    """Memoryless floor of route 5 (heterogeneous micro-environments).

    run_routes does not cover this route, so route_floors() cannot measure it;
    without it the readout figure would have to leave one bar's floor blank.
    Same q = 1 construction: wipe the register every cycle, keep what is left.
    """
    H = build_reservoir_H(**CRY)
    I_nuc = np.eye(D_NUC, dtype=complex) / D_NUC
    ipc, mc = [], []
    for sd in range(n_seeds):
        inputs, sp = _inputs(sd, L)
        cols = []
        for tau in taus:
            P_S, _, step, dt = _cycle_operators(H, tau, T2e_ns, kS, kT, 24)
            rho = singlet_projector(0, 1, N_SPINS).astype(complex)
            rho = rho / np.trace(rho)
            col = []
            for s in inputs:
                nuc = I_nuc                      # q = 1: register wiped every cycle
                rho = np.kron(_electron_born_state(s, "ST"), nuc)
                v = _vec(rho); yS = 0.0
                for _ in range(24):
                    v = step @ v
                    yS += kS * np.real(np.trace(P_S @ v.reshape(DIM, DIM, order="F"))) * dt
                col.append(yS)
            cols.append(col[washout:])
        X = np.array(cols).T
        r = memory_and_ipc(X, sp)
        ipc.append(r["IPC_total"]); mc.append(r["MC"])
    out = dict(channels=len(taus), floor_IPC=float(np.mean(ipc)),
               floor_IPC_sd=float(np.std(ipc)), floor_MC=float(np.mean(mc)))
    print(f"  hetero_tau   ch={out['channels']:>2}  floor IPC={out['floor_IPC']:.3f}"
          f"+/-{out['floor_IPC_sd']:.3f}  floor MC={out['floor_MC']:.3f}")
    return _save("m7_hetero_floor", out)


ALL["m7h"] = ("M7 floor of the heterogeneous-lifetime route", hetero_floor)


if __name__ == "__main__":
    import sys, time
    which = sys.argv[1:] or list(ALL)
    for key in which:
        title, fn = ALL[key]
        print(f"\n=== {key}: {title} ===", flush=True)
        t0 = time.time()
        fn()
        print(f"  ({time.time()-t0:.0f} s)", flush=True)
