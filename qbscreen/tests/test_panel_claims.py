"""Regression tests for every number the revised manuscript quotes from the
referee-panel supplementation (simulation_results/panel/).

The submitted version quoted figures that could not be traced to a stored key --
notably a 0.2 % that came from the five-channel column while the printed table was
the eight-channel one. These tests bind each claim in the text to a stored value so
that the same drift cannot recur.
"""

import json
import os

import pytest

PANEL = os.path.join(os.path.dirname(__file__), "..", "..",
                     "simulation_results", "panel")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PANEL), reason="panel outputs not generated")


def _load(name):
    with open(os.path.join(PANEL, f"{name}.json")) as f:
        return json.load(f)


def test_product_register_beats_survivor():
    """Main text, 'The register is the product, and that helps'."""
    d = _load("c1_product_carryover")
    s, p = d["survivor"], d["product"]
    assert s["YS_t"]["IPC"] == pytest.approx(2.01, abs=0.02)
    assert p["YS_t"]["IPC"] == pytest.approx(2.74, abs=0.02)
    assert s["cidnp"]["IPC"] == pytest.approx(4.65, abs=0.02)
    assert p["cidnp"]["IPC"] == pytest.approx(4.91, abs=0.02)
    # the physically faithful register is never worse on the memory routes
    for route in ("YS_t", "SandT_t", "cidnp"):
        assert p[route]["IPC"] > s[route]["IPC"]


def test_route_specific_floors():
    """Main text readout table and Fig. 2 caption: the floor is NOT 1 for every
    route. The eight-channel CIDNP floor is ~2, so the excess is 2.65 not 3.65."""
    f = _load("m7_route_floors")
    assert f["YS_t"]["floor_IPC"] == pytest.approx(1.02, abs=0.03)
    assert f["cidnp"]["floor_IPC"] == pytest.approx(2.01, abs=0.03)
    assert f["YS_t_accum"]["floor_IPC"] == pytest.approx(1.96, abs=0.05)
    # the accumulated pool keeps linear memory with the register wiped: it is a
    # classical delay line, not evidence of nuclear-spin memory
    assert f["YS_t_accum"]["floor_MC"] > 1.5
    assert 4.65 - f["cidnp"]["floor_IPC"] == pytest.approx(2.65, abs=0.05)


def test_delay_kernel_and_horizon_definition():
    """Main text: MC includes C[0]~1, which carries no memory, so the horizon is
    (MC - C0) x T_cycle."""
    k = _load("c3_delay_kernel")
    for route in ("YS_t", "cidnp"):
        assert k[route]["C0"] == pytest.approx(1.0, abs=0.05)
    assert k["cidnp"]["MC_minus_C0"] == pytest.approx(1.87, abs=0.05)
    assert k["YS_t"]["MC_minus_C0"] == pytest.approx(0.97, abs=0.05)
    # kernel falls below 5 % of C[0] within a handful of cycles
    assert 2 <= k["cidnp"]["delay_5pct_of_C0"] <= 8


def test_clock_effect_is_resolved_and_is_not_0p2_percent():
    """Abstract and clock section: the change across five decades is 1.6 %,
    resolved at t = 4.2 on paired differences -- not the 0.2 % submitted."""
    p = _load("c2_clock_paired")
    assert p["n_seeds"] == 12
    assert p["relative_change_pct"] == pytest.approx(1.6, abs=0.3)
    assert p["t_stat"] > 2.0, "effect must be resolved above seed noise"
    assert p["relative_change_pct"] > 0.5, "the submitted 0.2 % is not reproducible"


def test_clock_effect_exceeds_grid_error():
    """The 1.6 % effect must sit above the discretisation error on MC."""
    g = _load("c2_mc_grid")
    by_nt = {r["n_t"]: r["MC_8ch"] for r in g}
    grid_err = abs(by_nt[192] - by_nt[96]) / by_nt[96] * 100
    assert grid_err < 0.5, f"MC grid error {grid_err:.2f}% too large"
    assert _load("c2_clock_paired")["relative_change_pct"] > grid_err


def test_register_stores_populations_not_coherence():
    """Criterion section: pure nuclear dephasing costs almost nothing, so the
    binding constant is T1n and not T2n."""
    c = _load("c4_nuclear_channel")
    last = [r for r in c if r["q"] == 1.0][0]
    first = [r for r in c if r["q"] == 0.0][0]
    assert last["MC_8_depolarise"] == pytest.approx(1.00, abs=0.05)
    assert last["MC_8_dephase"] == pytest.approx(2.87, abs=0.05)
    # dephasing at q=1 costs under 2 % of the capacity
    cost = (first["MC_8_dephase"] - last["MC_8_dephase"]) / first["MC_8_dephase"]
    assert cost < 0.02


def test_no_quantum_advantage_at_cryptochrome_point():
    """Main text: matched channel-for-channel, the classical net wins by ~50 %."""
    m = _load("m6_cry_classical")
    assert m["quantum_5ch_kinetics"][0] < m["ESN_5node_5feat"][0]
    assert m["quantum_8ch_cidnp"][0] < m["ESN_8node_8feat"][0]
    ratio = m["ESN_8node_8feat"][0] / m["quantum_8ch_cidnp"][0]
    assert ratio == pytest.approx(1.5, abs=0.2)


def test_coherence_fraction_is_electronic_not_register_erasure():
    """Coherence section: restricting dephasing to the electrons reproduces 0.82,
    so the number is not an artifact of wiping the nuclear register."""
    c = _load("c6_coherence_control")
    assert c["all5"]["coherence_fraction"] == pytest.approx(0.816, abs=0.01)
    assert c["electrons_only"]["coherence_fraction"] == pytest.approx(0.817, abs=0.01)
    assert abs(c["all5"]["coherence_fraction"]
               - c["electrons_only"]["coherence_fraction"]) < 0.01


def test_reported_capacity_is_conservative_in_the_ridge():
    """Robustness section: lambda -> 0 RAISES the capacity, so the published
    lambda = 1e-6 understates rather than inflates it."""
    r = _load("m9_ridge_sensitivity")
    scan = {row["lambda"]: row for row in r["lambda_scan"]}
    assert scan[1e-12]["YS_t_raw"] > scan[1e-6]["YS_t_raw"]
    assert scan[1e-12]["cidnp_raw"] > scan[1e-6]["cidnp_raw"]
    # standardising the features also raises it
    assert scan[1e-6]["YS_t_std"] > scan[1e-6]["YS_t_raw"]
    assert r["condition_number"]["YS_t"] > 1e6, "ill-conditioning is the reason"


def test_readout_not_shot_noise_limited_at_cellular_copy_numbers():
    r = _load("m9_ridge_sensitivity")
    shot = {row["n_molecules"]: row for row in r["shot_noise"]}
    assert shot[1e10]["cidnp"] == pytest.approx(4.54, abs=0.05)
    assert shot[1e6]["cidnp"] < shot[1e10]["cidnp"]


# ── Open items resolved in the third review set ──────────────────────

def test_proton_only_register_survives_but_costs_most_of_the_capacity():
    """SI S9. Two of the three model nuclei are 14N, which relaxes in ~0.1-1 us
    in a protein. A proton-only register keeps ~29% of the excess capacity, and
    the kinetic readout dies outright."""
    d = {r["scenario"]: r for r in _load("open2_nuclide_register")}
    full = d["all three retained (main text)"]
    prot = d["14N wiped, 1H retained (literature)"]
    assert full["excess_8ch"] == pytest.approx(2.64, abs=0.05)
    assert prot["excess_8ch"] == pytest.approx(0.78, abs=0.05)
    # the kinetic readout needs the whole register: wiping any subset kills it
    assert prot["excess_5ch"] == pytest.approx(0.0, abs=0.02)
    frac = prot["excess_8ch"] / full["excess_8ch"]
    assert 0.2 < frac < 0.4, f"proton register retains {frac:.0%}, expected ~29%"


def test_desynchronisation_kills_kinetics_but_not_cidnp():
    """SI S10. Half a cycle of turnover jitter removes all of the kinetic
    readout's memory; CIDNP keeps about half of its excess even at two cycles."""
    rows = {r["jitter_cycles"]: r for r in _load("open1_ensemble_desync_floor")}
    assert rows[0.0]["excess_5ch"] == pytest.approx(0.99, abs=0.05)
    assert rows[0.5]["excess_5ch"] == pytest.approx(0.0, abs=0.05)
    assert rows[2.0]["excess_8ch"] > 1.0
    # the apparent recovery of raw IPC at large jitter is a pooling artifact:
    # excess over the register-wiped floor is zero
    assert rows[2.0]["excess_5ch"] == pytest.approx(0.0, abs=0.05)
    assert rows[2.0]["IPC_5ch_live"] > 1.5, "raw IPC does rise -- that is the trap"


def test_semiclassical_reference_makes_the_coherence_claim_conservative():
    """SI S8. A genuine classical model of the same spin system scores at the
    memoryless floor, so the operational 0.82 understates the coherent fraction."""
    d = _load("open4_semiclassical")
    assert d["classical_IPC"] < 1.0, "classical reservoir must be at/below the floor"
    assert d["semiclassical_coherence_fraction"] > 0.816, \
        "semiclassical fraction should exceed the operational one"
    assert d["semiclassical_coherence_fraction"] == pytest.approx(0.86, abs=0.03)


def test_spin_half_approximation_for_14N_is_conservative():
    """SI S11. Modelling 14N correctly as spin-1 increases the capacity."""
    rows = _load("open3_spin1")
    half = [r for r in rows if r["nuclear_spin"] == 0.5][0]
    one = [r for r in rows if r["nuclear_spin"] == 1.0][0]
    assert one["excess_cidnp"] > half["excess_cidnp"]
    assert one["dim"] == 24 and half["dim"] == 16


def test_anisotropy_costs_capacity_and_vanishes_at_the_ising_point():
    """SI S11. Capacity is carried by the perpendicular hyperfine components, so
    it vanishes exactly at eta = 1 where A_xx = A_yy = 0."""
    rows = {r["eta"]: r for r in _load("open3_anisotropy")}
    assert rows[0.0]["excess_8ch"] == pytest.approx(2.66, abs=0.05)
    assert rows[0.5]["excess_8ch"] < 0.4 * rows[0.0]["excess_8ch"]
    assert rows[1.0]["excess_8ch"] == pytest.approx(0.0, abs=0.02), "Ising point"
    assert rows[2.0]["excess_8ch"] > 0.5, "perpendicular component restored"


def test_heterogeneity_is_not_a_threat():
    """SI S10. Pooling copies with scattered hyperfine couplings does not degrade
    the capacity."""
    rows = _load("open1_ensemble_heterogeneity")
    base = [r for r in rows if r["spread"] == 0.0][0]["IPC_8ch"]
    for r in rows:
        assert r["IPC_8ch"] > 0.9 * base, f"spread {r['spread']} lost too much"


def test_classical_reservoir_has_no_memory_above_its_own_floor():
    """SI S8. The semiclassical model must be scored against its own register-wiped
    floor, not against the nominal IPC = 1, because finite-M sampling noise puts it
    below unity. The excess is zero: classical nuclei store nothing between cycles."""
    live = _load("open4_semiclassical")["classical_IPC"]
    floor = _load("open4_semiclassical_floor")["classical_floor_IPC"]
    assert abs(live - floor) < 0.05, "classical excess must be zero within noise"
    quantum = _load("open4_semiclassical")["quantum_reference_IPC"]
    assert quantum - live > 4.0, "quantum must be far above the classical reference"


def test_relaxation_calculator_reproduces_measured_systems():
    """SI S12. The predicted T1 is only worth quoting if the calculator can
    reproduce systems where the answer is known."""
    v = {r["system"]: r for r in _load("open5_relaxation_estimate")["validation"]}
    d2o = v["2H in D2O (quadrupolar)"]
    assert d2o["ratio"] == pytest.approx(1.0, abs=0.25), "2H/D2O is the clean check"
    n14 = v["14N of free amino acid (quadrupolar)"]
    assert 0.2 < n14["ratio"] < 3.0, "14N within a factor of a few"
    h2o = v["1H2O intramolecular (dipolar)"]
    assert h2o["ratio"] > 1.0, \
        "intramolecular-only must exceed the measured total, which includes intermolecular"


def test_register_relaxation_is_a_low_field_problem():
    """SI S12. T1n ~ 1 s is a high-field value; at 50 uT the protein-bound proton
    relaxes in milliseconds because extreme narrowing maximises the rate."""
    p = _load("open5_relaxation_estimate")["prediction"]
    fields = p["fields_T"]
    i_geo, i_high = fields.index(50e-6), fields.index(9.4)
    trp = p["proton"]["Trp H-beta (CH2, geminal partner)"]
    assert trp[i_geo] < 5e-3, "geomagnetic proton T1 must be milliseconds"
    assert trp[i_high] > 1.0, "high-field proton T1 must be seconds"
    assert trp[i_high] / trp[i_geo] > 100, "field suppression must be large"
    # 14N is hopeless at any field considered
    for v in p["nitrogen14"].values():
        assert v[i_geo] < 1e-5


def test_protein_bound_register_cannot_reach_the_neural_band():
    """SI S12. The three requirements (reuse, relaxation, neural-band horizon)
    have no common solution for a register whose dipolar vector reorients only
    with the whole protein."""
    rows = {r["tau_c_ns"]: r for r in _load("open5_turnover_estimate")["feasible_region"]}
    assert rows[15.2]["feasible"] is False
    assert rows[15.2]["max_horizon_s"] < 1e-2, "must fall below the 10 ms band edge"
    assert rows[1.0]["feasible"] is True, "a mobile register does work"


def test_critical_tau_c_is_solved_not_read_off_the_grid():
    """The boundary used to be quoted as the last grid point that happened to be
    feasible. With the default grid (..., 5.0, 10.0, 15.2) that is 5.0, and
    nothing between 5 and 10 ns was ever evaluated -- so the published threefold
    requirement was 15.2/5, an artefact of grid spacing. It is now bisected."""
    c = _load("open5_turnover_estimate")["critical_tau_c"]
    trp = c["per_nucleus"]["Trp H-beta (CH2, geminal partner)"]
    assert trp["tau_crit_dry_ns"] == pytest.approx(8.37, abs=0.05), \
        "intramolecular-only boundary"
    assert trp["tau_crit_dry_ns"] > 5.0 + 1e-9, \
        "the grid's 5.0 ns was not the boundary; a test that pins it pins the artefact"
    assert c["tau_protein_ns"] == pytest.approx(15.243, abs=0.01)


def test_the_intermolecular_bath_is_what_makes_the_requirement_threefold():
    """Two errors were cancelling: the grid overstated the boundary, and the T1
    calculation omitted the intermolecular bath, which shortens it again. With
    the bath restored the requirement is ~2.8x, which is what the manuscript's
    'about three times' rests on -- not the 1.8x the grid-free dry number gives.
    """
    c = _load("open5_turnover_estimate")["critical_tau_c"]
    assert c["bath_f"] == pytest.approx(0.542, abs=0.01), \
        "bath ratio taken from this package's own water validation row"
    for name, n in c["per_nucleus"].items():
        speedup = c["tau_protein_ns"] / n["tau_crit_wet_ns"]
        assert 2.5 < speedup < 3.2, f"{name}: {speedup:.2f}x, manuscript says about three"


def test_the_methyl_escape_route_dies_with_the_bath():
    """The flavin 8-alpha methyl has S^2 = 1/4, so its intramethyl vector does not
    co-rotate with the protein and its T1 is long enough to reach the band --
    an escape the manuscript would have had to concede. The intermolecular bath
    closes it, and that is why the bath had to be implemented rather than the
    published numbers rewritten."""
    c = _load("open5_turnover_estimate")["critical_tau_c"]
    me = c["per_nucleus"]["flavin 8-alpha CH3 (intra-methyl)"]
    assert me["tau_crit_dry_ns"] > c["tau_protein_ns"], \
        "without the bath the methyl is feasible even bound to the whole protein"
    assert me["tau_crit_wet_ns"] < c["tau_protein_ns"], \
        "with the bath it is not"


def test_light_cannot_drive_the_cycle_in_a_brain():
    """SI S12. Blue light inside a skull gives a turnover interval of centuries."""
    rows = {r["condition"]: r for r in _load("open5_turnover_estimate")["light_driven"]}
    skull = [v for k, v in rows.items() if "skull" in k][0]
    sun = [v for k, v in rows.items() if "sunlight" in k][0]
    assert skull["T_turnover_s"] > 1e9, "inside a skull the cycle effectively stops"
    assert sun["T_turnover_s"] > 1.0, "even full sunlight is slower than 1 s"
    # the catalytic alternative does reach the required range
    kc = {r["enzyme"]: r for r in _load("open5_turnover_estimate")["reaction_driven"]}
    assert kc["fast flavoenzyme"]["T_turnover_s"] <= 1e-3


def test_T1_equals_T2_at_the_operating_field():
    """SI S12, added after the set-2 audit. The manuscript at one point argued that
    the register is rescued by storing populations (T1) rather than coherence (T2),
    T2 being much shorter. That rescue is vacuous at the geomagnetic field: extreme
    narrowing makes J(0) = J(w) = J(2w), hence T1 = T2 exactly. The distinction
    only exists at NMR fields."""
    r = _load("open5_relaxation_estimate")["t1_t2_ratio"]
    by_field = {row["field_T"]: row for row in r}
    geo = by_field[50e-6]
    assert geo["omega_tau_c"] < 1e-3, "must be in extreme narrowing"
    assert geo["T2_over_T1"] == pytest.approx(1.0, abs=0.01), \
        "T1 and T2 must coincide at the operating field"
    high = by_field[9.4]
    assert high["omega_tau_c"] > 1.0
    assert high["T2_over_T1"] < 0.01, "the distinction is a high-field phenomenon"


def test_paired_clock_generator_is_reproducible():
    """Set-2 audit found c2_clock_paired.json had no generator in source. It now
    does, and must reproduce the quoted statistics."""
    p = _load("c2_clock_paired")
    assert p["n_seeds"] == 12
    assert len(p["MC_8_1us"]) == 12 and len(p["MC_8_100ms"]) == 12
    assert p["t_stat"] == pytest.approx(4.23, abs=0.15)
    assert p["paired_sem"] == pytest.approx(0.011, abs=0.002)


def test_adverse_results_survive_the_product_register():
    """Set-2 audit item: the reductions had only been established for the survivor
    register. Re-run with the physically correct product register, they hold."""
    n = _load("audit3_nuclide_product")
    assert n["proton_fraction"] == pytest.approx(0.30, abs=0.05), \
        "proton-only retention must stay near the survivor value of 29%"
    prot = [r for r in n["rows"] if "1H retained" in r["scenario"]][0]
    assert prot["excess_5ch"] == pytest.approx(0.0, abs=0.02), "kinetics still dies"
    d = {r["jitter_cycles"]: r for r in _load("audit3_desync_product")}
    assert d[2.0]["excess_5ch"] == pytest.approx(0.0, abs=0.05)
    assert d[2.0]["excess_8ch"] > 1.0, "CIDNP still survives desynchronisation"
    a = {r["eta"]: r for r in _load("audit3_anisotropy_product")}
    assert a[0.5]["fraction_of_isotropic"] < 0.5, "anisotropy still costs most of it"
    assert a[1.0]["excess_8ch"] == pytest.approx(0.0, abs=0.02), "Ising point"


def test_dambre_bound_is_violated_in_finite_samples():
    """SI S2. The claim that the estimator exceeds IPC <= 1 for a single observable
    had no stored data until the set-4 audit; it now does, and the regenerated
    numbers (1.013, 29/30) differ slightly from the originally quoted 1.011, 30/30."""
    d = _load("s2_dambre_control")
    assert d["n_seeds"] == 30 and d["L"] == 1200
    assert d["mean_IPC"] == pytest.approx(1.013, abs=0.005)
    assert d["n_above_bound"] >= 25, "the bound must be violated in most seeds"
    assert d["mean_IPC"] > 1.0, "this is the point of the control"


def test_kinetic_readout_decays_smoothly_with_jitter():
    """Set-4 audit caught a false assertion that one full cycle of jitter removes
    ALL kinetic memory. It does not: the decay is smooth and reaches zero near 1.5
    cycles."""
    rows = {r["jitter_cycles"]: r for r in _load("audit3_desync_product")}
    assert rows[1.0]["excess_5ch"] > 0.1, "one cycle does NOT remove all of it"
    assert rows[1.5]["excess_5ch"] < 0.1
    assert rows[2.0]["excess_5ch"] == pytest.approx(0.0, abs=0.05)
    assert rows[0.0]["excess_5ch"] > rows[0.5]["excess_5ch"] > rows[1.0]["excess_5ch"]
