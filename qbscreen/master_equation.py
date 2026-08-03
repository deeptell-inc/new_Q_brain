#!/usr/bin/env python3
"""Liouville-space RPM master equation with electronic decoherence.

This module corrects the central methodological flaw identified in peer review:
the original singlet-yield calculation used only a recombination rate k and
*omitted* the electronic decoherence (T2e) that was computed elsewhere. Here we
solve the full Haberkorn master equation including electronic dephasing, so that
the predicted magnetic field effect (MFE) reflects realistic spin coherence.

Master equation (Haberkorn form):

    dρ/dt = -i[H, ρ]
            - (1/2){ k_S P_S + k_T P_T , ρ }            (spin-selective recombination)
            + Σ_i (1/T2e) ( S_z^i ρ S_z^i - (1/4) ρ )    (electronic pure dephasing)

The time-integrated singlet probability is obtained exactly in Liouville space,

    Φ_S = k_S · vec(P_S)† · (−L)^{-1} · vec(ρ_0)

For a 2-electron system the dephasing generator with L_i = S_z^i gives a
singlet–triplet dephasing rate 1/T2e; in the limit 1/T2e → 0 and k_S = k_T = k
this reproduces the coherent Green's-function result exactly (validated below).

References:
  Haberkorn, Mol. Phys. 32, 1491 (1976) — spin-selective recombination term.
  Steady-state yield: Φ_S = k_S ∫₀^∞ Tr[P_S ρ(t)] dt (standard RPM result,
  derivable from the master equation; NOT taken verbatim from Haberkorn).
"""

import numpy as np

# Reuse spin operators / constants from the dynamics module.
from qbscreen.spin_dynamics import (
    SX, SY, SZ, I2, spin_op, singlet_projector,
    G_E, MU_B, HBAR, GAMMA_P, GAMMA_H,
)


# ───────────────────────────────────────────────────────────────────
# Liouville-space utilities (column-stacking vec convention)
# vec(A ρ B) = (Bᵀ ⊗ A) vec(ρ)
# ───────────────────────────────────────────────────────────────────

def _vec(rho):
    """Column-stacking vectorisation."""
    return rho.flatten(order="F")


def _left(A):
    """Superoperator for ρ ↦ A ρ  →  (I ⊗ A)."""
    dim = A.shape[0]
    return np.kron(np.eye(dim), A)


def _right(B):
    """Superoperator for ρ ↦ ρ B  →  (Bᵀ ⊗ I)."""
    dim = B.shape[0]
    return np.kron(B.T, np.eye(dim))


def _sandwich(A, B):
    """Superoperator for ρ ↦ A ρ B  →  (Bᵀ ⊗ A)."""
    return np.kron(B.T, A)


def build_liouvillian(H, P_S, P_T, k_S, k_T, deph_ops):
    """Assemble the Liouvillian superoperator L (units: angular freq, rad/μs).

    Parameters
    ----------
    H : (d,d) Hermitian Hamiltonian in angular frequency units (rad/μs).
    P_S, P_T : (d,d) singlet / triplet projectors on the recombining electrons.
    k_S, k_T : recombination rates (1/μs).
    deph_ops : list of (rate, L) pairs; each contributes the Lindblad dissipator
               rate · ( L ρ L† − ½{L†L, ρ} ). For pure dephasing L = L† = S_z^i.
    """
    d = H.shape[0]
    I = np.eye(d)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # Coherent part: -i (H ⊗ I - I ⊗ Hᵀ)  (column-stacking: -i(I⊗H - Hᵀ⊗I))
        L = -1j * (_left(H) - _right(H))

        # Haberkorn recombination: -½ { K, ρ } with K = k_S P_S + k_T P_T
        K = k_S * P_S + k_T * P_T
        L += -0.5 * (_left(K) + _right(K))

        # Lindblad dephasing
        for rate, A in deph_ops:
            AdA = A.conj().T @ A
            L += rate * (_sandwich(A, A.conj().T) - 0.5 * (_left(AdA) + _right(AdA)))

    return L


def singlet_yield_master(H, P_S, P_T, k_S, k_T, rho0, deph_ops):
    """Exact time-integrated singlet yield via Liouville-space inversion.

    Φ_S = k_S · Tr[P_S (−L)^{-1} ρ0] = k_S · vec(P_S)† (−L)^{-1} vec(ρ0)
    """
    L = build_liouvillian(H, P_S, P_T, k_S, k_T, deph_ops)
    rho0_vec = _vec(rho0)
    # Solve (−L) x = ρ0_vec  ⇒  x = ∫₀^∞ ρ(t) dt   (vectorised)
    x = np.linalg.solve(-L, rho0_vec)
    X = x.reshape(H.shape, order="F")
    return float(np.real(k_S * np.trace(P_S @ X)))


def electron_dephasing_ops(n_spins, T2e_us, electron_indices=(0, 1)):
    """Pure-dephasing Lindblad operators L_i = S_z^i with rate set by T2e.

    For a single S=1/2 with L = S_z, the dissipator rate · (SzρSz − ¼ρ) damps
    transverse coherence at rate ·/2. Choosing rate = 2/T2e gives transverse
    relaxation time T2e for each electron; the singlet–triplet coherence then
    decays at ~1/T2e. This is the standard 'random-field' electronic dephasing.
    """
    if T2e_us is None or T2e_us <= 0 or not np.isfinite(T2e_us):
        return []
    rate = 2.0 / T2e_us
    return [(rate, spin_op(SZ, i, n_spins)) for i in electron_indices]


def singlet_triplet_dephasing_op(P_S, P_T, gamma_ST):
    """Singlet-triplet pure-dephasing Lindblad operator from exchange-coupling
    (2J) fluctuations, as appropriate for a tightly bound radical pair.

    The jump operator is A = P_S - P_T (Hermitian, A†A = P_S + P_T = I). The
    dissipator gamma_ST·(A ρ A − ρ) damps coherence between the singlet and
    triplet manifolds at rate 2·gamma_ST, without touching populations. This is
    the decoherence channel a fluctuating 2J produces (H_fluc ∝ P_S - P_T),
    distinct from the single-spin S_z 'random-field' channel.

    Parameters
    ----------
    P_S, P_T : (d,d) singlet / triplet projectors on the recombining electrons.
    gamma_ST : dephasing rate (1/μs); returns [] if non-positive.
    """
    if gamma_ST is None or gamma_ST <= 0 or not np.isfinite(gamma_ST):
        return []
    return [(gamma_ST, P_S - P_T)]


# ───────────────────────────────────────────────────────────────────
# Phenomenological 'exponential' dephasing in the eigenbasis
# (fast path for large 8-spin systems where full Liouville inversion
#  — dim² × dim² = 65536² — is intractable)
# ───────────────────────────────────────────────────────────────────

def singlet_yield_exponential(H, P_S, rho0, k_mhz, T2e_ns=None):
    """Singlet yield with exponential coherence dephasing (eigenbasis Green's fn).

    Populations (m=n) decay only by recombination k; coherences (m≠n) decay by
    k + 1/T2e. This is the exact solution of the master equation under the
    'exponential model' (uniform coherence dephasing) and the cross-validated
    approximation to electronic T2e for large spin systems.

        Φ_S = k [ Σ_m P_mm ρ_mm / k
                  + Re Σ_{m≠n} P_mn ρ_nm / (k + 1/T2e + iω_mn) ]

    H must be in angular-frequency units consistent with k (here: MHz·2π → we
    keep MHz and ω_mn from eigenvalues in the same MHz convention as
    spin_dynamics.build_rpm_hamiltonian, where energies are in MHz).
    """
    E, V = np.linalg.eigh(H)
    Vd = V.conj().T
    Ps = Vd @ P_S @ V
    r0 = Vd @ rho0 @ V
    dE = E[:, None] - E[None, :]          # ω_mn in MHz
    gamma = 0.0 if (T2e_ns is None or T2e_ns <= 0) else 1.0 / (T2e_ns * 1e-3)  # 1/T2e in MHz

    # Diagonal (populations): only recombination
    diag = np.real(np.sum(np.diag(Ps) * np.diag(r0)))  # Σ P_mm ρ_mm
    # Off-diagonal (coherences): recombination + dephasing
    denom = (k_mhz + gamma) + 1j * dE
    G = Ps * r0.T / denom
    np.fill_diagonal(G, 0.0)
    offdiag = np.real(np.sum(G))

    return k_mhz * (diag / k_mhz + offdiag)


if __name__ == "__main__":
    # ── Self-validation: master equation must reproduce the coherent
    #    Green's function when decoherence = 0 and k_S = k_T = k. ──
    from qbscreen.spin_dynamics import build_rpm_hamiltonian

    print("Validation: master equation vs coherent Green's function")
    print("=" * 60)

    # Small model: 2 electrons + 1 ³¹P  (dim = 8, Liouville = 64)
    n = 3
    dim = 2 ** n
    P_S = singlet_projector(0, 1, n)
    P_T = np.eye(dim) - P_S  # triplet projector on electrons (includes nuclear factor)
    # Restrict triplet projector to electron subspace properly:
    # P_T_elec = I - P_S already acts as triplet on electrons ⊗ I_nuc.
    rho0 = P_S / np.trace(P_S)

    # Minimal Hamiltonian: electron Zeeman + one ³¹P hyperfine
    A_P = 200.0  # MHz
    def H_min(B_T):
        H = np.zeros((dim, dim), dtype=complex)
        omega_e = G_E * MU_B * B_T / (HBAR * 2 * np.pi * 1e6)  # MHz
        H += omega_e * (spin_op(SZ, 0, n) + spin_op(SZ, 1, n))
        for op in (SX, SY, SZ):
            H += A_P * spin_op(op, 0, n) @ spin_op(op, 2, n)
        return H

    k = 1.0  # MHz  (=1/μs)
    B = 0.9e-3  # 0.9 mT
    # Convert H (MHz) to angular frequency (rad/μs) for the master eqn: ×2π
    H = 2 * np.pi * H_min(B)
    kS = kT = 2 * np.pi * k  # rad/μs

    phi_master = singlet_yield_master(H, P_S, P_T, kS, kT, rho0, deph_ops=[])
    phi_exp = singlet_yield_exponential(H_min(B), P_S, rho0, k_mhz=k, T2e_ns=None)
    print(f"  Φ_S (master, no deph) = {phi_master:.6f}")
    print(f"  Φ_S (exponential, no deph) = {phi_exp:.6f}")
    print(f"  difference = {abs(phi_master - phi_exp):.2e}")
    assert abs(phi_master - phi_exp) < 1e-6, "Master eqn must match coherent limit!"
    print("  ✓ PASS: master equation reproduces coherent Green's function")

    # Now show decoherence suppresses the MFE
    print("\nEffect of electronic T2e on the MFE (2e + ³¹P model):")
    for T2e_ns in [None, 100, 10, 1]:
        phi0 = singlet_yield_exponential(H_min(0.0), P_S, rho0, k_mhz=k, T2e_ns=T2e_ns)
        phiB = singlet_yield_exponential(H_min(B), P_S, rho0, k_mhz=k, T2e_ns=T2e_ns)
        mfe = (phiB - phi0) / phi0 * 100
        label = "∞ (coherent)" if T2e_ns is None else f"{T2e_ns} ns"
        print(f"  T2e = {label:>14}:  MFE = {mfe:+.4f} %")
