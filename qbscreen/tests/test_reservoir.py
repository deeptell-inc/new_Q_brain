"""Tests for the radical-pair + nuclear-spin quantum reservoir.

Verifies the two correctness anchors and the central honest finding:
  • IPC respects the Dambre bound (total IPC <= number of readout observables).
  • A multi-observable readout extracts genuine memory; a scalar readout cannot
    (the in-vivo readout bottleneck).
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="module")
def driven():
    from qbscreen.reservoir import (
        build_reservoir_H, propagator, run_reservoir, _observable_set,
        memory_and_ipc,
    )
    rng = np.random.default_rng(3)
    s = rng.uniform(0, 1, 500 + 60)
    s_post = s[60:]
    H = build_reservoir_H(B_tesla=1e-3, A_e1_a=80, A_e1_b=40, A_e2_a=15, J=2.0)
    P = propagator(H, tau_us=0.05, T2e_ns=50.0)
    X_full = run_reservoir(s, P, _observable_set("full"), nuc_deph_p=0.0, washout=60)
    X_scalar = run_reservoir(s, P, _observable_set("scalar"), nuc_deph_p=0.6, washout=60)
    return memory_and_ipc(X_full, s_post), memory_and_ipc(X_scalar, s_post)


class TestReservoir:
    def test_ipc_within_dambre_bound(self, driven):
        r_full, r_scalar = driven
        # Total IPC cannot exceed the number of independent readout observables.
        assert r_full["IPC_total"] <= r_full["n_obs"] + 0.5
        assert r_scalar["IPC_total"] <= r_scalar["n_obs"] + 0.1

    def test_engineered_has_real_capacity(self, driven):
        r_full, _ = driven
        # A multi-observable spin reservoir carries several units of memory...
        assert r_full["MC"] > 1.5, f"engineered MC too low: {r_full['MC']:.2f}"
        # ...and genuine NONLINEAR processing capacity.
        assert r_full["IPC_nonlinear"] > 0.5, "no nonlinear capacity"

    def test_scalar_readout_collapses(self, driven):
        r_full, r_scalar = driven
        # The in-vivo scalar product-yield readout is capped near IPC ~ 1
        # and is far below the engineered multi-observable capacity.
        assert r_scalar["IPC_total"] < 1.5
        assert r_full["IPC_total"] > 3 * r_scalar["IPC_total"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
