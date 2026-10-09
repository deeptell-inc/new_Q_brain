#!/usr/bin/env python3
"""The cryptochrome point with the flavin nitrogens as axial tensors.

The main text uses isotropic couplings. Lee et al. (2014, J. R. Soc. Interface
11, 20131063) state the characteristics of the two flavin nitrogens in FAD•−:
large, near-axial hyperfine interactions, symmetry axes parallel to each other
(normal to the ring), in-plane principal values small, isotropic values 523 uT
(N5) and 189 uT (N10). Their full tensor tables are in that paper's electronic
supplement, which we could not retrieve; this module therefore builds the
tensors from the stated characteristics -- axial, A_perp = 0, A_par = 3 a_iso,
common axis -- which is the idealisation Lee et al. identify as the one
[FAD•− Z•] nearly satisfies. The tryptophan proton keeps its isotropic 40 MHz.

What is scanned: the angle theta between the field and the nitrogen axis
(0, 54.7 and 90 degrees), and a 6-direction powder average, against the
isotropic model at the same isotropic values. Spin-1/2 nitrogen throughout
(the spin-1 check of S11 is a separate, smaller system).
"""
import os
import numpy as np
from scipy.linalg import expm
from qbscreen.reservoir import memory_and_ipc
from qbscreen.master_equation import build_liouvillian, _vec
from qbscreen.general_spin import spin_matrices, embed, singlet_projector_e, G_E, MU_B, HBAR, TWO_PI, _save
from qbscreen.panel_response import _inputs as _inp
from qbscreen.readout_routes import CRY

OUT = "simulation_results/panel"
DIMS = (2, 2, 2, 2, 2); SPINS = (0.5, 0.5, 0.5, 0.5, 0.5)
MT = 28.0249514  # MHz per mT (free-electron gyromagnetic ratio)


def tensors(axial=True):
    a5, a10 = 0.523 * MT, 0.189 * MT        # isotropic values stated by Lee et al. 2014
    if axial:
        return [(0, 2, (0.0, 0.0, 3 * a5)), (0, 3, (0.0, 0.0, 3 * a10)), (1, 4, (40.0, 40.0, 40.0))]
    return [(0, 2, (a5, a5, a5)), (0, 3, (a10, a10, a10)), (1, 4, (40.0, 40.0, 40.0))]


def unit(theta, phi=0.0):
    return (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))


def build_H(couplings, B_tesla, theta, phi=0.0, dims=DIMS):
    """Tensors in the molecular frame (a length-3 diagonal or a full 3x3 matrix in
    MHz); the field direction is rotated."""
    D = int(np.prod(dims)); H = np.zeros((D, D), dtype=complex)
    omega_e = G_E * MU_B * B_tesla / (HBAR * 2 * np.pi * 1e6)
    ops = [spin_matrices(0.5) for _ in dims]
    n = unit(theta, phi)
    for e in (0, 1):
        H += omega_e * sum(n[mu] * embed(ops[e][mu], e, dims) for mu in range(3))
    for e, k, A in couplings:
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            A = np.diag(A)
        for mu in range(3):
            for nu in range(3):
                if A[mu, nu]:
                    H += A[mu, nu] * embed(ops[e][mu], e, dims) @ embed(ops[k][nu], k, dims)
    return H


def _depolarise(nuc, j, q, n):
    """Depolarise nucleus j of n spin-1/2 nuclei with probability q."""
    if q <= 0:
        return nuc
    R = nuc.reshape((2,) * n + (2,) * n)
    R = np.moveaxis(np.moveaxis(R, j, 0), n + j, 1)          # bring (j, j') to the front
    tr = np.trace(R, axis1=0, axis2=1)                        # partial trace over j
    out = np.zeros_like(R)
    out[0, 0] = tr / 2.0; out[1, 1] = tr / 2.0
    out = np.moveaxis(np.moveaxis(out, 1, n + j), 0, j).reshape(nuc.shape)
    return (1.0 - q) * nuc + q * out


def run(inputs, H, tau_us=1.0, T2e_ns=1000.0, kS=1.0, kT=0.2, n_t=96, sample=(16, 36, 56, 76, 96),
        washout=80, q_per_nucleus=(0.0, 0.0, 0.0), carrier="product", direction=(0.0, 0.0, 1.0), dims=DIMS):
    D = int(np.prod(dims)); D_NUC = D // 4; n_nuc = len(dims) - 2
    P_S = singlet_projector_e(dims); P_T = np.eye(D) - P_S
    _, _, sz = spin_matrices(0.5)
    # electronic dephasing is along the field, as in the main model where the
    # field is z; with the field rotated the dephasing axis rotates with it
    sxyz0 = spin_matrices(0.5)
    s_n = sum(direction[mu] * sxyz0[mu] for mu in range(3))
    deph = [(2.0 / (T2e_ns * 1e-3), embed(s_n, i, dims)) for i in (0, 1)]
    with np.errstate(over="ignore", invalid="ignore"):
        L = build_liouvillian(TWO_PI * H, P_S, P_T, kS, kT, deph); dt = tau_us / n_t; step = expm(L * dt)
    # CIDNP reads the nuclear polarisation along the field (the laboratory
    # quantisation axis), which is the molecular z only when theta = 0
    sxyz = spin_matrices(0.5); n_ = direction
    cid = [embed(2.0 * sum(n_[mu] * sxyz[mu] for mu in range(3)), i, dims) for i in range(2, len(dims))]
    sx, sy, szz = spin_matrices(0.5); PSe = 0.25 * np.eye(4) - sum(np.kron(o, o) for o in (sx, sy, szz)); PTe = np.eye(4) - PSe
    rows = []; rho = (P_S / np.trace(P_S)).astype(complex)
    for s in inputs:
        nuc = np.trace(rho.reshape(4, D_NUC, 4, D_NUC), axis1=0, axis2=2); nuc /= np.trace(nuc).real
        for j, q in enumerate(q_per_nucleus):
            nuc = _depolarise(nuc, j, q, n_nuc)
        rho = np.kron(s * PSe + (1 - s) * PTe / 3.0, nuc)
        v = _vec(rho); yS = 0.0; yIz = np.zeros(n_nuc); yS_t = []; M = np.zeros((D_NUC, D_NUC), dtype=complex)
        for it in range(1, n_t + 1):
            v = step @ v; r = v.reshape(D, D, order="F")
            yS += kS * np.real(np.trace(P_S @ r)) * dt
            for j, O in enumerate(cid):
                yIz[j] += kS * np.real(np.trace(O @ P_S @ r)) * dt
            aS = P_S @ r @ P_S; aT = P_T @ r @ P_T
            M += (kS * np.trace(aS.reshape(4, D_NUC, 4, D_NUC), axis1=0, axis2=2)
                  + kT * np.trace(aT.reshape(4, D_NUC, 4, D_NUC), axis1=0, axis2=2)) * dt
            if it in sample:
                yS_t.append(yS)
        rows.append(list(yS_t) + list(yIz / (yS if yS else 1.0)))
        if carrier == "product":
            M = 0.5 * (M + M.conj().T); rho = np.kron(np.eye(4, dtype=complex) / 4.0, M / np.trace(M).real)
        else:
            rr = v.reshape(D, D, order="F"); rr = 0.5 * (rr + rr.conj().T); rho = rr / np.trace(rr).real
    X = np.array(rows[washout:]); return X[:, :5], X


def scan(n_seeds=6, L=700):
    deg = np.pi / 180
    cases = [("isotropic", tensors(False), [(0.0, 0.0)]),
             ("axial, theta=0", tensors(True), [(0.0, 0.0)]),
             ("axial, theta=54.7", tensors(True), [(54.7356 * deg, 0.0)]),
             ("axial, theta=90", tensors(True), [(90 * deg, 0.0)]),
             ("axial, 6-direction average", tensors(True),
              [(0.0, 0.0), (90 * deg, 0.0), (90 * deg, 90 * deg), (54.7356 * deg, 45 * deg), (54.7356 * deg, 135 * deg), (125.264 * deg, 45 * deg)])]
    out = {}
    for name, coup, dirs in cases:
        for reg, qn in (("full", (0, 0, 0)), ("proton_only", (1, 1, 0))):
            Hs = [(build_H(coup, CRY["B_tesla"], th, ph), unit(th, ph)) for th, ph in dirs]
            live = {"5": [], "8": []}; floor = {"5": [], "8": []}
            for sd in range(n_seeds):
                s, sp = _inp(sd, L)
                # orientation average: pool the channels of each direction as one ensemble readout
                L_ = [run(s, H, q_per_nucleus=qn, direction=n_) for H, n_ in Hs]
                F_ = [run(s, H, q_per_nucleus=(1, 1, 1), direction=n_) for H, n_ in Hs]
                Y = np.mean([y for y, _ in L_], axis=0); C = np.mean([c for _, c in L_], axis=0)
                Yf = np.mean([y for y, _ in F_], axis=0); Cf = np.mean([c for _, c in F_], axis=0)
                live["5"].append(memory_and_ipc(Y, sp)["IPC_total"]); live["8"].append(memory_and_ipc(C, sp)["IPC_total"])
                floor["5"].append(memory_and_ipc(Yf, sp)["IPC_total"]); floor["8"].append(memory_and_ipc(Cf, sp)["IPC_total"])
            row = dict(case=name, register=reg, n_directions=len(dirs), n_seeds=n_seeds)
            for k, nm in (("5", "kinetics_5ch"), ("8", "cidnp_8ch")):
                row[nm] = float(np.mean(live[k])); row[nm + "_sd"] = float(np.std(live[k], ddof=1))
                row[nm + "_floor"] = float(np.mean(floor[k])); row[nm + "_excess"] = row[nm] - row[nm + "_floor"]
            out[f"{name}/{reg}"] = row
            print(f"  {name:28s} {reg:12s} 5ch {row['kinetics_5ch']:.3f} (excess {row['kinetics_5ch_excess']:+.3f})"
                  f"  8ch {row['cidnp_8ch']:.3f} (floor {row['cidnp_8ch_floor']:.3f}, excess {row['cidnp_8ch_excess']:+.3f})", flush=True)
            _save("open14_axial_tensors", out)
    return out




def ratio_scan(ratios=(-0.1, -0.2, -0.3), n_seeds=3, L=700):
    """A_perp as a fraction of a_iso, trace preserved (A_par = 3 a_iso - 2 A_perp),
    6-direction average. A_perp = 0 is the Ising limit scanned above; the
    in-plane values Lee et al. describe as 'small' are negative and of order a
    fifth of a_iso, so this brackets that regime without quoting their table."""
    deg = np.pi / 180
    dirs = [(0.0, 0.0), (90 * deg, 0.0), (90 * deg, 90 * deg), (54.7356 * deg, 45 * deg), (54.7356 * deg, 135 * deg), (125.264 * deg, 45 * deg)]
    a5, a10 = 0.523 * MT, 0.189 * MT
    out = {}
    for r in ratios:
        coup = [(0, 2, (r * a5, r * a5, 3 * a5 - 2 * r * a5)), (0, 3, (r * a10, r * a10, 3 * a10 - 2 * r * a10)), (1, 4, (40.0, 40.0, 40.0))]
        Hs = [(build_H(coup, CRY["B_tesla"], th, ph), unit(th, ph)) for th, ph in dirs]
        for reg, qn in (("full", (0, 0, 0)), ("proton_only", (1, 1, 0))):
            live = {"5": [], "8": []}; floor = {"5": [], "8": []}
            for sd in range(n_seeds):
                s, sp = _inp(sd, L)
                L_ = [run(s, H, q_per_nucleus=qn, direction=n_) for H, n_ in Hs]; F_ = [run(s, H, q_per_nucleus=(1, 1, 1), direction=n_) for H, n_ in Hs]
                Y = np.mean([y for y, _ in L_], axis=0); C = np.mean([c for _, c in L_], axis=0)
                Yf = np.mean([y for y, _ in F_], axis=0); Cf = np.mean([c for _, c in F_], axis=0)
                live["5"].append(memory_and_ipc(Y, sp)["IPC_total"]); live["8"].append(memory_and_ipc(C, sp)["IPC_total"])
                floor["5"].append(memory_and_ipc(Yf, sp)["IPC_total"]); floor["8"].append(memory_and_ipc(Cf, sp)["IPC_total"])
            row = dict(A_perp_over_a_iso=r, register=reg, n_directions=len(dirs), n_seeds=n_seeds)
            for k, nm in (("5", "kinetics_5ch"), ("8", "cidnp_8ch")):
                row[nm] = float(np.mean(live[k])); row[nm + "_sd"] = float(np.std(live[k], ddof=1))
                row[nm + "_floor"] = float(np.mean(floor[k])); row[nm + "_excess"] = row[nm] - row[nm + "_floor"]
            out[f"{r}/{reg}"] = row
            print(f"  A_perp/a_iso={r:+.1f} {reg:12s} 5ch {row['kinetics_5ch']:.3f} (excess {row['kinetics_5ch_excess']:+.3f})"
                  f"  8ch {row['cidnp_8ch']:.3f} (floor {row['cidnp_8ch_floor']:.3f}, excess {row['cidnp_8ch_excess']:+.3f})", flush=True)
            _save("open15_axial_ratio", out)
    return out


if __name__ == "__main__":
    import sys, time; t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    (ratio_scan() if "ratio" in sys.argv else scan()); print(f"  ({time.time()-t0:.0f} s)")
