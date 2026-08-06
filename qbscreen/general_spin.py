#!/usr/bin/env python3
"""Robustness of the register criterion to two modelling simplifications.

The main text models the reservoir as five spin-1/2 particles with isotropic
hyperfine couplings. Two simplifications hide in that sentence, and the referee
panel listed both as open:

  A  the hyperfine couplings of a real flavin-tryptophan pair are anisotropic
     tensors, not scalars;
  B  14N is spin-1, not spin-1/2.

This module relaxes each one. The Liouville-space machinery in
qbscreen.master_equation is dimension-agnostic, so the only new ingredient is
spin operators for arbitrary S.

NOTE ON SCOPE. The anisotropy study is a *scan* over an axial anisotropy
parameter at fixed isotropic average, not a calculation with the published
tensor set of Lee et al. (2014) or Schleicher et al. (2021). It therefore
narrows the limitation without removing it: it shows whether the conclusions are
sensitive to anisotropy of realistic magnitude, not whether they hold for the
specific tensors of a specific protein.
"""

import json
import os
import numpy as np
from scipy.linalg import expm

from qbscreen.reservoir import memory_and_ipc
from qbscreen.master_equation import build_liouvillian, _vec
from qbscreen.readout_routes import CRY

OUT = "simulation_results/panel"
TWO_PI = 2.0 * np.pi
G_E, MU_B, HBAR = 2.00231930436256, 9.2740100783e-24, 1.054571817e-34


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  [checkpoint] {OUT}/{name}.json")
    return obj


def spin_matrices(S):
    """Sx, Sy, Sz for spin quantum number S (dimension 2S+1)."""
    d = int(round(2 * S + 1))
    m = np.array([S - k for k in range(d)])                 # m = S ... -S
    Sz = np.diag(m).astype(complex)
    off = np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1))
    Sp = np.zeros((d, d), dtype=complex)
    Sp[np.arange(d - 1), np.arange(1, d)] = off             # raising
    Sm = Sp.conj().T
    return (Sp + Sm) / 2.0, (Sp - Sm) / (2.0j), Sz


def embed(op, i, dims):
    """Embed a single-spin operator at site i into the full product space."""
    out = np.array([[1.0 + 0j]])
    for j, d in enumerate(dims):
        out = np.kron(out, op if j == i else np.eye(d))
    return out


def singlet_projector_e(dims):
    """P_S on electrons 0 and 1 (both spin-1/2), identity on everything else."""
    sx, sy, sz = spin_matrices(0.5)
    SS = sum(embed(o, 0, dims) @ embed(o, 1, dims) for o in (sx, sy, sz))
    return 0.25 * np.eye(int(np.prod(dims))) - SS


def build_H(dims, spins, B_tesla, J, couplings):
    """couplings: list of (electron_index, nucleus_index, (Ax, Ay, Az)) in MHz."""
    D = int(np.prod(dims))
    H = np.zeros((D, D), dtype=complex)
    omega_e = G_E * MU_B * B_tesla / (HBAR * 2 * np.pi * 1e6)
    _, _, sz_half = spin_matrices(0.5)
    H += omega_e * (embed(sz_half, 0, dims) + embed(sz_half, 1, dims))
    ops = [spin_matrices(s) for s in spins]
    for e, n, A in couplings:
        for mu in range(3):
            H += A[mu] * embed(ops[e][mu], e, dims) @ embed(ops[n][mu], n, dims)
    if J:
        for mu in range(3):
            H += J * embed(ops[0][mu], 0, dims) @ embed(ops[1][mu], 1, dims)
    return H


def run_routes_general(inputs, dims, spins, H_mhz, tau_us=1.0, T2e_ns=1000.0,
                       kS=1.0, kT=0.2, n_t=96, sample=(16, 36, 56, 76, 96),
                       washout=80, q_nuc=0.0, carrier="survivor"):
    """Same cycle as qbscreen.panel_response.run_routes_v2, for an arbitrary list
    of nuclear spin quantum numbers.

    carrier="product" hands the next turnover the nuclear state carried away by
    the recombined molecule rather than the renormalised survivor; see
    run_routes_v2 for the rationale."""
    D = int(np.prod(dims))
    D_NUC = D // 4
    P_S = singlet_projector_e(dims)
    P_T = np.eye(D) - P_S
    _, _, sz_half = spin_matrices(0.5)
    deph = []
    if T2e_ns:
        rate = 2.0 / (T2e_ns * 1e-3)
        deph = [(rate, embed(sz_half, i, dims)) for i in (0, 1)]
    with np.errstate(over="ignore", invalid="ignore"):
        L = build_liouvillian(TWO_PI * H_mhz, P_S, P_T, kS, kT, deph)
        dt = tau_us / n_t
        step = expm(L * dt)

    nuc_idx = list(range(2, len(dims)))
    # 2*Iz, matching the Pauli-Z normalisation used in readout_routes.py: the
    # ridge is applied to unstandardised features, so the scale is not free.
    cid_ops = [embed(2.0 * spin_matrices(spins[i])[2], i, dims) for i in nuc_idx]
    I_nuc = np.eye(D_NUC, dtype=complex) / D_NUC

    # electron birth states
    sx, sy, sz = spin_matrices(0.5)
    e_dims = (2, 2)
    SSe = sum(np.kron(o, o) for o in (sx, sy, sz))
    PSe = 0.25 * np.eye(4) - SSe
    PTe = np.eye(4) - PSe

    rows_y, rows_c = [], []
    rho = (P_S / np.trace(P_S)).astype(complex)
    for s in inputs:
        R = rho.reshape(4, D_NUC, 4, D_NUC)
        nuc = np.trace(R, axis1=0, axis2=2)
        t = np.trace(nuc).real
        nuc = nuc / (t if t else 1.0)
        if q_nuc > 0:
            nuc = (1.0 - q_nuc) * nuc + q_nuc * I_nuc
        rho = np.kron(s * PSe + (1.0 - s) * PTe / 3.0, nuc)

        v = _vec(rho); yS = 0.0
        yIz = np.zeros(len(cid_ops)); yS_t = []
        M_prod = np.zeros((D_NUC, D_NUC), dtype=complex)
        for it in range(1, n_t + 1):
            v = step @ v
            r = v.reshape(D, D, order="F")
            yS += kS * np.real(np.trace(P_S @ r)) * dt
            for j, O in enumerate(cid_ops):
                yIz[j] += kS * np.real(np.trace(O @ P_S @ r)) * dt
            if carrier == "product":
                aS = P_S @ r @ P_S
                aT = P_T @ r @ P_T
                RS = aS.reshape(4, D_NUC, 4, D_NUC)
                RT = aT.reshape(4, D_NUC, 4, D_NUC)
                M_prod += (kS * np.trace(RS, axis1=0, axis2=2)
                           + kT * np.trace(RT, axis1=0, axis2=2)) * dt
            if it in sample:
                yS_t.append(yS)
        rows_y.append(list(yS_t))
        rows_c.append(list(yS_t) + list(yIz / (yS if yS else 1.0)))
        if carrier == "product":
            M_prod = 0.5 * (M_prod + M_prod.conj().T)
            tp = np.trace(M_prod).real
            rho = np.kron(np.eye(4, dtype=complex) / 4.0,
                          M_prod / (tp if tp else 1.0))
        else:
            rr = v.reshape(D, D, order="F"); rr = 0.5 * (rr + rr.conj().T)
            t2 = np.trace(rr).real
            rho = rr / (t2 if t2 else 1.0)
    return np.array(rows_y[washout:]), np.array(rows_c[washout:])


def _capacities(inputs, dims, spins, H, washout=80, **kw):
    Y, C = run_routes_general(inputs, dims, spins, H, washout=washout, **kw)
    sp = inputs[washout:]
    return memory_and_ipc(Y, sp), memory_and_ipc(C, sp)


def _inp(seed, L=700, washout=80):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, L + washout)


# ─────────────────────────────────────────────────────────────────────
# A. anisotropic hyperfine, five spin-1/2
# ─────────────────────────────────────────────────────────────────────
def anisotropy_scan(etas=(0.0, 0.5, 1.0, 2.0), n_seeds=3, L=700):
    """Axial tensor A = A_iso (1-eta, 1-eta, 1+2eta): the isotropic average is
    held fixed and only the anisotropy grows. eta = 1 already gives A_zz three
    times the isotropic value, which is the scale seen for flavin 14N5."""
    dims = (2, 2, 2, 2, 2)
    spins = (0.5,) * 5
    out = []
    for eta in etas:
        def ax(a):
            return (a * (1 - eta), a * (1 - eta), a * (1 + 2 * eta))
        cpl = [(0, 2, ax(CRY["A_e1_a"])), (0, 3, ax(CRY["A_e1_b"])),
               (1, 4, ax(CRY["A_e2_a"]))]
        H = build_H(dims, spins, CRY["B_tesla"], CRY["J"], cpl)
        a5, a8, m8, f8 = [], [], [], []
        for sd in range(n_seeds):
            s = _inp(sd, L)
            r5, r8 = _capacities(s, dims, spins, H)
            a5.append(r5["IPC_total"]); a8.append(r8["IPC_total"]); m8.append(r8["MC"])
            _, r8f = _capacities(s, dims, spins, H, q_nuc=1.0)     # route floor
            f8.append(r8f["IPC_total"])
        row = dict(eta=eta, IPC_5ch=float(np.mean(a5)), IPC_8ch=float(np.mean(a8)),
                   MC_8ch=float(np.mean(m8)), floor_8ch=float(np.mean(f8)),
                   excess_8ch=float(np.mean(a8) - np.mean(f8)))
        out.append(row)
        print(f"  eta={eta:.1f}  IPC 5ch={row['IPC_5ch']:.3f}  8ch={row['IPC_8ch']:.3f}"
              f"  floor={row['floor_8ch']:.3f}  excess={row['excess_8ch']:.3f}", flush=True)
        _save("open3_anisotropy", out)
    return out


# ─────────────────────────────────────────────────────────────────────
# B. 14N as spin-1
# ─────────────────────────────────────────────────────────────────────
def spin1_check(n_seeds=3, L=700):
    """A four-spin reduction (e1, e2, one N, one H) run twice: N as spin-1/2 (as
    in the main text) and N as spin-1 (as 14N really is). Everything else is held
    fixed, so the difference isolates the spin quantum number."""
    out = []
    for tag, sN in (("N as spin-1/2 (main text)", 0.5), ("N as spin-1 (real 14N)", 1.0)):
        dims = (2, 2, int(2 * sN + 1), 2)
        spins = (0.5, 0.5, sN, 0.5)
        cpl = [(0, 2, (CRY["A_e1_a"],) * 3), (1, 3, (CRY["A_e2_a"],) * 3)]
        H = build_H(dims, spins, CRY["B_tesla"], CRY["J"], cpl)
        a5, a7, m7, f7 = [], [], [], []
        for sd in range(n_seeds):
            s = _inp(sd, L)
            r5, r7 = _capacities(s, dims, spins, H)
            a5.append(r5["IPC_total"]); a7.append(r7["IPC_total"]); m7.append(r7["MC"])
            _, r7f = _capacities(s, dims, spins, H, q_nuc=1.0)
            f7.append(r7f["IPC_total"])
        row = dict(model=tag, nuclear_spin=sN, dim=int(np.prod(dims)),
                   IPC_kinetics=float(np.mean(a5)), IPC_cidnp=float(np.mean(a7)),
                   MC_cidnp=float(np.mean(m7)), floor_cidnp=float(np.mean(f7)),
                   excess_cidnp=float(np.mean(a7) - np.mean(f7)))
        out.append(row)
        print(f"  {tag:28s} dim={row['dim']:>3}  IPC(cidnp)={row['IPC_cidnp']:.3f}"
              f"  floor={row['floor_cidnp']:.3f}  excess={row['excess_cidnp']:.3f}",
              flush=True)
        _save("open3_spin1", out)
    return out


if __name__ == "__main__":
    import sys, time
    which = sys.argv[1:] or ["aniso", "spin1"]
    for w in which:
        print(f"\n=== {w} ===", flush=True)
        t0 = time.time()
        (anisotropy_scan if w == "aniso" else spin1_check)()
        print(f"  ({time.time()-t0:.0f} s)", flush=True)
