"""Pin the unit convention to a physical frequency, not to internal consistency.

The referee of CP-ART-06-2026-002404 asked whether a Hamiltonian quoted in MHz
is propagated with the 2*pi it needs. Every solver path multiplies H by TWO_PI
before building the Liouvillian, but the manuscript never said so, and the
agreement with the coherent Green's-function yield (ESI S1) cannot catch a
missing 2*pi because both sides would share it. A lone electron in a known
field can: it must precess at g_e mu_B B / h, which is 28.02 MHz at 1 mT, and
nothing in the reservoir code is allowed to know that number.
"""

import numpy as np
from scipy.linalg import expm

from qbscreen import reservoir
from qbscreen.master_equation import build_liouvillian
from qbscreen.spin_dynamics import G_E, HBAR, MU_B, SX, SZ, spin_op

B = 1e-3                                              # tesla
F_LARMOR = G_E * MU_B * B / (2 * np.pi * HBAR) / 1e6  # MHz, the physical value


def _lone_electron_propagator(tau_us):
    # all hyperfine and exchange couplings off: the Zeeman term alone survives
    H = reservoir.build_reservoir_H(B_tesla=B, A_e1_a=0.0, A_e1_b=0.0, A_e2_a=0.0, J=0.0)
    d = H.shape[0]
    Z = np.zeros((d, d))
    L = build_liouvillian(reservoir.TWO_PI * H, Z, Z, 0.0, 0.0, [])
    return expm(L * tau_us)


def test_electron_precesses_at_the_larmor_frequency():
    n = reservoir.N_SPINS
    d = 2 ** n
    sx0, sz0 = spin_op(SX, 0, n), spin_op(SZ, 0, n)
    # electron 0 along +x, everything else maximally mixed
    rho0 = (np.eye(d) / d + 2 * sx0 / d).astype(complex)
    assert np.isclose(np.trace(rho0 @ sx0).real, 0.5)
    # sample half a period: <Sx> must have gone from +1/2 to -1/2 exactly if the
    # propagator runs at 2*pi*F_LARMOR rad/us and not at F_LARMOR
    t_half = 0.5 / F_LARMOR
    U = _lone_electron_propagator(t_half)
    rho = (U @ rho0.reshape(-1, order="F")).reshape(d, d, order="F")
    assert np.isclose(np.trace(rho @ sx0).real, -0.5, atol=1e-6), (
        "electron did not complete half a Larmor period: the 2*pi convention is wrong")
    # and a quarter period lands on zero, which a factor-2 error would not
    U = _lone_electron_propagator(0.25 / F_LARMOR)
    rho = (U @ rho0.reshape(-1, order="F")).reshape(d, d, order="F")
    assert abs(np.trace(rho @ sx0).real) < 1e-6


def test_larmor_frequency_is_the_textbook_value():
    assert np.isclose(F_LARMOR, 28.02, atol=0.01), F_LARMOR
