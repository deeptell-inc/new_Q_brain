"""Prove the suppressed matmul warnings are a BLAS artifact, not bad numerics.

pyproject.toml silences three RuntimeWarnings that a clean run emitted eighty
times. Silencing a warning without evidence is how a real numerical fault gets
buried, so the justification is a test rather than a comment: the flagged
operations are reproduced here and checked against an einsum reference computed
without BLAS. If the flags ever accompany an actually wrong or non-finite
result, these fail and the suppression must be revisited.
"""

import numpy as np
import pytest

from qbscreen.spin_dynamics import spin_op, singlet_projector, SX, SY, SZ, G_E, MU_B, HBAR


def _hamiltonian(n=3):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    w = G_E * MU_B * 0.9e-3 / (HBAR * 2 * np.pi * 1e6)
    H += w * (spin_op(SZ, 0, n) + spin_op(SZ, 1, n))
    for op in (SX, SY, SZ):
        H += 200.0 * spin_op(op, 0, n) @ spin_op(op, 2, n)
    return H


def test_flagged_operands_are_finite():
    """The flags fire on inputs with no inf or nan anywhere."""
    H = _hamiltonian()
    P_S = singlet_projector(0, 1, 3)
    assert np.isfinite(H).all() and np.isfinite(P_S).all()
    E, V = np.linalg.eigh(H)
    assert np.isfinite(E).all() and np.isfinite(V).all()


def test_flagged_matmul_matches_a_blas_free_reference():
    """The operation that raises the flag returns the right answer.

    einsum does not dispatch to the BLAS path, so it is an independent check of
    the same contraction.
    """
    H = _hamiltonian()
    P_S = singlet_projector(0, 1, 3)
    _, V = np.linalg.eigh(H)
    Vd = V.conj().T
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        got = Vd @ P_S @ V
    ref = np.einsum("ij,jk,kl->il", Vd, P_S, V)
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() == 0.0, "BLAS and einsum disagree; not an artifact"


def test_reservoir_traces_stay_finite_through_a_full_run():
    """End-to-end: the propagation loop that also raises the flags produces
    finite observables for every readout route."""
    from qbscreen.panel_response import run_routes_v2, _inputs
    from qbscreen.reservoir import build_reservoir_H
    from qbscreen.readout_routes import CRY
    s, _ = _inputs(0, 120)
    R = run_routes_v2(s, build_reservoir_H(**CRY), q_nuc=0.3)
    for k, v in R.items():
        assert np.isfinite(v).all(), f"route {k} produced a non-finite value"
