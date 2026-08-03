"""Test that the reservoir capacity is genuinely quantum (coherence-dependent)."""
import numpy as np, pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_ipc_collapses_when_decohered():
    """Killing spin coherence must remove most of the reservoir capacity."""
    from qbscreen.quantum_vs_classical import _props, _run
    from qbscreen.spin_dynamics import singlet_projector
    from qbscreen.reservoir import memory_and_ipc
    from qbscreen.ensemble import N
    rng = np.random.default_rng(9)
    s = rng.uniform(0, 1, 900 + 80); sp = s[80:]
    P_S = singlet_projector(0, 1, N)
    t = (0.02, 0.035, 0.05)
    ipc_q = memory_and_ipc(_run(s, _props(80, 40, 2.0, 50.0, 0.0, t), P_S), sp)["IPC_total"]
    ipc_c = memory_and_ipc(_run(s, _props(80, 40, 2.0, 50.0, 5000.0, t), P_S), sp)["IPC_total"]
    assert ipc_q > 3 * ipc_c, f"coherence should dominate: Q={ipc_q:.2f} C={ipc_c:.2f}"
    assert (ipc_q - ipc_c) / ipc_q > 0.7, "coherence must supply majority of capacity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
