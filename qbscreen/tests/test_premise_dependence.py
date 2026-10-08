"""Bind the premise-dependence section (ESI S14, Results 'What depends on the
coupling premise') to simulation_results/panel/premise_dependence.json.

The cryptochrome point assumes J = 0 and no dipolar term. The source cited for
the separated pair reports D = -11.36 MHz and |J| = 1.3-126 MHz for FAD-W324.
These tests pin (a) that the reference case of the rerun reproduces the shipped
numbers, so the comparison is like for like; (b) the sentences the paper draws
from the rerun; and (c) that no premise in the range produces a quantum
advantage.
"""
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "simulation_results"


def _j(rel):
    with open(RES / rel) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def P():
    return _j("panel/premise_dependence.json")


COUPLED = ["D(1.90nm) par, J=0", "D(1.90nm) perp, J=0", "D par, J_gap=-12.6",
           "D par, J_gap=+12.6", "D perp, J_gap=-12.6", "D perp, J_gap=+12.6"]


def test_reference_case_reproduces_the_shipped_routes(P):
    """Same seeds, same protocol: the J = D = 0 column must equal readout_routes.json."""
    ref = P["routes"]["reference  J=0, D=0"]["IPC"]
    pub = _j("readout_routes.json")
    for k in ("YS_end", "YS_t", "SandT_end", "SandT_t", "cidnp", "hetero_tau", "YS_t_accum"):
        assert ref[k]["mean"] == pytest.approx(pub[k]["IPC"], abs=0.01), k


def test_reference_case_reproduces_the_shipped_floors(P):
    ref = P["routes"]["reference  J=0, D=0"]["floor"]
    flo = _j("panel/m7_route_floors.json")
    for k in ("YS_end", "YS_t", "cidnp", "YS_t_accum"):
        assert ref[k]["mean"] == pytest.approx(flo[k]["floor_IPC"], abs=0.01), k


def test_reference_no_recombination_capacity_is_the_floor(P):
    assert P["capacity"]["reference  J=0, D=0"]["IPC"]["mean"] == pytest.approx(1.02, abs=0.02)


def test_any_coupling_restores_capacity_without_recombination(P):
    """'even 0.6 MHz of D, the value at r = 5 nm, gives 5.5' and '5.1-5.6 across
    both signs of J and both orientations'."""
    cap = P["capacity"]
    vals = [cap[k]["IPC"]["mean"] for k in COUPLED]
    assert 5.0 < min(vals) and max(vals) < 6.0, vals
    assert cap["D at r=5.0nm"]["IPC"]["mean"] == pytest.approx(5.2, abs=0.1)
    assert cap["J=+1.26 only"]["IPC"]["mean"] > 5.0


def test_time_resolved_range_and_cidnp_range(P):
    """'time-resolved kinetic capacity moves in both directions, from 2.0 to 1.3-3.4; CIDNP 3.9-5.2'."""
    r = P["routes"]
    ys = [r[k]["IPC"]["YS_t"]["mean"] for k in COUPLED]
    ci = [r[k]["IPC"]["cidnp"]["mean"] for k in COUPLED]
    assert 1.29 < min(ys) and max(ys) < 3.44, ys
    assert 3.85 < min(ci) and max(ci) < 5.25, ci


def test_floors_and_end_point_do_not_move(P):
    r = P["routes"]
    for k in COUPLED:
        assert r[k]["IPC"]["YS_end"]["mean"] == pytest.approx(1.02, abs=0.02)
        assert r[k]["floor"]["YS_t"]["mean"] == pytest.approx(1.02, abs=0.02)
        assert r[k]["floor"]["cidnp"]["mean"] == pytest.approx(2.0, abs=0.05)


def test_no_premise_yields_a_quantum_advantage(P):
    """The matched classical network (ESN 5 nodes 4.47, 8 nodes 6.94, shipped
    values) must beat the best quantum-side value at every premise."""
    esn = _j("final_numbers.json")
    r = P["routes"]
    best5 = max(r[k]["IPC"]["YS_t"]["mean"] for k in COUPLED)
    best8 = max(r[k]["IPC"]["cidnp"]["mean"] for k in COUPLED)
    assert best5 < 4.4 and best8 < 6.9, (best5, best8)
    assert best5 == pytest.approx(3.39, abs=0.02) and best8 == pytest.approx(5.20, abs=0.02)


def test_proton_only_kinetics_die_under_the_coupled_premise_too(P):
    po = P["proton_only"]
    for k, v in po.items():
        assert v["IPC"]["YS_t"]["mean"] == pytest.approx(1.02, abs=0.02), k
    assert po["D(1.90nm) par, J=0"]["IPC"]["cidnp"]["mean"] == pytest.approx(3.29, abs=0.02)
    assert po["D par, J_gap=-12.6"]["IPC"]["cidnp"]["mean"] == pytest.approx(2.86, abs=0.02)


def test_clock_and_reuse_under_the_coupled_premise(P):
    """ESI S14, clock paragraph: MC(8 ch) at zero pause 2.92 -> 2.72 (D par) and
    2.88 (D, J-); MC(q = 0.99) 1.78 against 1.04; ceiling and boundary from the
    same eight-point scan for both premises."""
    cl = P["clock"]
    assert cl["D(1.90nm) par, J=0"]["rows"][0]["MC_8ch"]["mean"] == pytest.approx(2.72, abs=0.01)
    assert cl["D par, J_gap=-12.6"]["rows"][0]["MC_8ch"]["mean"] == pytest.approx(2.88, abs=0.01)
    ru = P["reuse"]["D(1.90nm) par, J=0"]["rows"]
    q99 = next(r for r in ru if abs(r["q_nuc"] - 0.99) < 1e-9)
    assert q99["MC_8"]["mean"] == pytest.approx(1.78, abs=0.01)
    ref99 = next(r for r in _j("register_reuse.json") if abs(r["q_nuc"] - 0.99) < 1e-9)
    assert ref99["MC_8"] == pytest.approx(1.04, abs=0.01)
    h = P["horizon_coarse_grid"]
    assert h["reference"]["ceil_wet"] == pytest.approx(3.4, abs=0.06)
    assert h["D(1.90nm) par, J=0"]["ceil_wet"] == pytest.approx(5.7, abs=0.06)
    assert h["D(1.90nm) par, J=0"]["ceil_dry"] < 10.0 and h["D(1.90nm) par, J=0"]["ceil_wet"] < 10.0, "still below the band"
    assert h["reference"]["tau_wet"] == pytest.approx(5.2, abs=0.06)
    assert h["D(1.90nm) par, J=0"]["tau_wet"] == pytest.approx(8.7, abs=0.06)
    assert h["tau_protein_ns"] / h["D(1.90nm) par, J=0"]["tau_wet"] == pytest.approx(1.8, abs=0.05)
    assert h["tau_protein_ns"] / h["reference"]["tau_wet"] == pytest.approx(2.9, abs=0.05)
