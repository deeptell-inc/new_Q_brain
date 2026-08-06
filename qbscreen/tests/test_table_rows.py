"""Row-by-row binding of every table printed in the manuscript.

Closes the open item recorded through sets 2-4: the existing claim tests bind the
headline quantities, but each *row* and standard deviation of each table was
unbound, so a table could silently drift from the column it was computed in --
which is exactly the defect that started this review (a 0.2 % taken from the
five-channel column while the printed table was the eight-channel one).

Each expected value below is transcribed from the manuscript text, NOT from the
JSON, so that the test fails if either side moves.
"""

import json
import os

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "simulation_results")
PANEL = os.path.join(ROOT, "panel")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PANEL), reason="panel outputs not generated")


def _load(path, name):
    with open(os.path.join(path, f"{name}.json")) as f:
        return json.load(f)


# ── main text: chemically accessible readout routes ──────────────────
# manuscript table: route, channels, IPC, floor, excess
READOUT_TABLE = [
    ("YS_end",     1, 1.02),
    ("SandT_end",  2, 1.37),
    ("hetero_tau", 3, 2.56),
    ("YS_t",       5, 2.01),
    ("SandT_t",   10, 2.01),
    ("cidnp",      8, 4.65),
    ("YS_t_accum",10, 2.71),
]


@pytest.mark.parametrize("key,channels,ipc", READOUT_TABLE)
def test_readout_route_row(key, channels, ipc):
    r = _load(ROOT, "readout_routes")[key]
    assert r["channels"] == channels, f"{key}: channel count printed in the table"
    assert r["IPC"] == pytest.approx(ipc, abs=0.005)


def test_readout_table_is_complete():
    """All seven routes must be present; the submitted version omitted one."""
    r = _load(ROOT, "readout_routes")
    assert set(r) == {k for k, _, _ in READOUT_TABLE}


# ── main text: classical baselines at the engineered point ───────────
ESN_TABLE = [
    ("quantum", 5.59, 0.19),
    ("ESN 5 nodes / 5 linear feats", 4.55, 0.32),
    ("ESN 5 nodes / 12 feats (5+7 quad)", 9.41, 0.86),
    ("ESN 12 nodes / 12 linear feats", 10.37, 0.59),
]


@pytest.mark.parametrize("key,ipc,sd", ESN_TABLE)
def test_engineered_classical_row(key, ipc, sd):
    v = _load(ROOT, "final_numbers")["F3"][key]
    assert v[0] == pytest.approx(ipc, abs=0.01)
    assert v[1] == pytest.approx(sd, abs=0.01), "the printed standard deviation"


# ── main text: convergence in the driving-sequence length ────────────
@pytest.mark.parametrize("L,ipc", [("600", 5.28), ("1200", 5.60),
                                   ("2400", 5.87), ("4800", 5.84)])
def test_length_convergence_row(L, ipc):
    assert _load(ROOT, "final_numbers")["F1"][L][0] == pytest.approx(ipc, abs=0.01)


# ── main text: the turnover clock, twelve seeds ──────────────────────
CLOCK_TABLE = [(1e-6, 2.926, 0.149), (1e-3, 2.926, 0.148),
               (1e-2, 2.922, 0.145), (1e-1, 2.880, 0.116)]


@pytest.mark.parametrize("T_d,mc,sd", CLOCK_TABLE)
def test_clock_row(T_d, mc, sd):
    rows = {r["T_d_s"]: r for r in _load(PANEL, "c2_clock_multiseed")}
    assert rows[T_d]["MC_8ch"] == pytest.approx(mc, abs=0.002)
    assert rows[T_d]["MC_8ch_sd"] == pytest.approx(sd, abs=0.002)


# ── criterion: depolarising vs dephasing ─────────────────────────────
CHANNEL_TABLE = [(0.0, 2.890, 2.890), (0.3, 2.667, 2.919), (0.5, 2.288, 2.899),
                 (0.7, 2.030, 2.878), (0.9, 1.977, 2.867), (1.0, 1.002, 2.865)]


@pytest.mark.parametrize("q,depol,deph", CHANNEL_TABLE)
def test_nuclear_channel_row(q, depol, deph):
    rows = {r["q"]: r for r in _load(PANEL, "c4_nuclear_channel")}
    assert rows[q]["MC_8_depolarise"] == pytest.approx(depol, abs=0.002)
    assert rows[q]["MC_8_dephase"] == pytest.approx(deph, abs=0.002)


# ── SI S8: the semiclassical reference ───────────────────────────────
def test_semiclassical_values():
    d = _load(PANEL, "open4_semiclassical")
    assert d["classical_IPC"] == pytest.approx(0.764, abs=0.002)
    assert d["classical_IPC_sd"] == pytest.approx(0.015, abs=0.002)
    assert d["quantum_reference_IPC"] == pytest.approx(5.421, abs=0.002)
    assert d["semiclassical_coherence_fraction"] == pytest.approx(0.859, abs=0.002)
    f = _load(PANEL, "open4_semiclassical_floor")
    assert f["classical_floor_IPC"] == pytest.approx(0.768, abs=0.002)


# ── SI S9: nuclide-resolved register ─────────────────────────────────
NUCLIDE_TABLE = [("all three retained (main text)", 2.008, 4.650),
                 ("14N wiped, 1H retained (literature)", 1.019, 2.784),
                 ("1H wiped, 14N retained", 1.020, 3.447),
                 ("all three wiped (floor)", 1.019, 2.005)]


@pytest.mark.parametrize("scenario,ipc5,ipc8", NUCLIDE_TABLE)
def test_nuclide_row(scenario, ipc5, ipc8):
    rows = {r["scenario"]: r for r in _load(PANEL, "open2_nuclide_register")}
    assert rows[scenario]["IPC_5ch"] == pytest.approx(ipc5, abs=0.005)
    assert rows[scenario]["IPC_8ch"] == pytest.approx(ipc8, abs=0.005)


# ── SI S10: turnover jitter, both registers ──────────────────────────
JITTER_SURVIVOR = [(0.0, 0.991, 2.664), (0.1, 0.781, 1.901), (0.25, 0.598, 1.756),
                   (0.5, 0.006, 1.518), (1.0, 0.110, 1.555), (2.0, 0.004, 1.310)]


@pytest.mark.parametrize("jit,ex5,ex8", JITTER_SURVIVOR)
def test_jitter_row_survivor(jit, ex5, ex8):
    rows = {r["jitter_cycles"]: r for r in _load(PANEL, "open1_ensemble_desync_floor")}
    assert rows[jit]["excess_5ch"] == pytest.approx(ex5, abs=0.005)
    assert rows[jit]["excess_8ch"] == pytest.approx(ex8, abs=0.005)


JITTER_PRODUCT = [(0.0, 1.716), (0.25, 0.669), (0.5, 0.317),
                  (1.0, 0.181), (1.5, 0.016), (2.0, -0.006)]


@pytest.mark.parametrize("jit,ex5", JITTER_PRODUCT)
def test_jitter_row_product(jit, ex5):
    """Every point the manuscript's jitter sentence relies on must exist. Set 4
    caught an assertion about jitter = 1.0 when that point had never been run."""
    rows = {r["jitter_cycles"]: r for r in _load(PANEL, "audit3_desync_product")}
    assert jit in rows, f"jitter={jit} is quoted but not computed"
    assert rows[jit]["excess_5ch"] == pytest.approx(ex5, abs=0.005)


# ── SI S11: anisotropy and spin-1 ────────────────────────────────────
@pytest.mark.parametrize("eta,ipc,floor,excess",
                         [(0.0, 4.660, 1.996, 2.664), (0.5, 2.863, 2.001, 0.862),
                          (1.0, 1.009, 1.009, 0.000), (2.0, 2.730, 1.999, 0.731)])
def test_anisotropy_row(eta, ipc, floor, excess):
    rows = {r["eta"]: r for r in _load(PANEL, "open3_anisotropy")}
    assert rows[eta]["IPC_8ch"] == pytest.approx(ipc, abs=0.005)
    assert rows[eta]["floor_8ch"] == pytest.approx(floor, abs=0.005)
    assert rows[eta]["excess_8ch"] == pytest.approx(excess, abs=0.005)


# ── SI S12: predicted relaxation and the feasible region ─────────────
def test_predicted_T1_row():
    p = _load(PANEL, "open5_relaxation_estimate")["prediction"]
    i_geo = p["fields_T"].index(50e-6)
    i_hi = p["fields_T"].index(9.4)
    assert p["proton"]["Trp H-beta (CH2, geminal partner)"][i_geo] == pytest.approx(2.4e-3, rel=0.05)
    assert p["proton"]["Trp H-beta (CH2, geminal partner)"][i_hi] == pytest.approx(9.0, rel=0.05)
    assert p["proton"]["flavin 8-alpha CH3 (intra-methyl)"][i_geo] == pytest.approx(5.0e-3, rel=0.05)
    assert p["nitrogen14"]["14N N5"][i_geo] == pytest.approx(0.34e-6, rel=0.1)
    assert p["nitrogen14"]["14N N10"][i_geo] == pytest.approx(0.19e-6, rel=0.1)
    assert p["tau_c_s"] == pytest.approx(15.2e-9, rel=0.02)


@pytest.mark.parametrize("tau_ns,T1,horizon", [
    (0.1, 0.3723, 0.9369),
    (1.0, 0.03723, 0.09368),
    (5.0, 0.007445, 0.01874),
    (15.2, 0.002449, 0.006163),
])
def test_feasible_region_row(tau_ns, T1, horizon):
    rows = {r["tau_c_ns"]: r for r in _load(PANEL, "open5_turnover_estimate")["feasible_region"]}
    assert rows[tau_ns]["T1_s"] == pytest.approx(T1, rel=0.02)
    assert rows[tau_ns]["max_horizon_s"] == pytest.approx(horizon, rel=0.02)


def test_light_driven_row():
    rows = {r["condition"]: r for r in _load(PANEL, "open5_turnover_estimate")["light_driven"]}
    sun = [v for k, v in rows.items() if "sunlight" in k][0]
    assert sun["T_turnover_s"] == pytest.approx(7.71, rel=0.02)


# ── SI: ridge and shot-noise sensitivity, every row ──────────────────
@pytest.mark.parametrize("lam,kin,cid", [(1e-12, 4.53, 6.67), (1e-9, 4.11, 6.03),
                                         (1e-6, 1.98, 4.54), (1e-3, 1.01, 2.25),
                                         (1e-1, 1.00, 1.02)])
def test_ridge_row(lam, kin, cid):
    rows = {r["lambda"]: r for r in _load(PANEL, "m9_ridge_sensitivity")["lambda_scan"]}
    assert rows[lam]["YS_t_raw"] == pytest.approx(kin, abs=0.01)
    assert rows[lam]["cidnp_raw"] == pytest.approx(cid, abs=0.01)


@pytest.mark.parametrize("N,cid", [(1e10, 4.54), (1e8, 4.54), (1e6, 3.93)])
def test_shot_noise_row(N, cid):
    rows = {r["n_molecules"]: r for r in _load(PANEL, "m9_ridge_sensitivity")["shot_noise"]}
    assert rows[N]["cidnp"] == pytest.approx(cid, abs=0.01)


# ── main text: classical baselines AT the cryptochrome point ─────────
# The set-5 audit found this table entirely unbound, though it carries the
# paper's second headline result.
CRY_ESN_TABLE = [
    ("quantum_5ch_kinetics", 2.01, 0.09),
    ("ESN_5node_5feat",      4.47, 0.33),
    ("quantum_8ch_cidnp",    4.65, 0.15),
    ("ESN_8node_8feat",      6.94, 0.56),
    ("ESN_12node_12feat",    9.62, 0.93),
]


@pytest.mark.parametrize("key,ipc,sd", CRY_ESN_TABLE)
def test_cryptochrome_classical_row(key, ipc, sd):
    v = _load(PANEL, "m6_cry_classical")[key]
    # abs=0.01 used to admit a wrong final digit on a two-decimal value: the
    # printed 9.63 sat exactly 0.0100 from the stored 9.6249 and passed
    assert v[0] == pytest.approx(ipc, abs=0.005)
    assert v[1] == pytest.approx(sd, abs=0.01), "the printed standard deviation"


def test_classical_beats_quantum_on_both_channel_matchings():
    """The '50%' quoted in the text is the eight-channel margin; the five-channel
    margin is larger. Both are bound so neither can be quoted selectively."""
    m = _load(PANEL, "m6_cry_classical")
    m8 = m["ESN_8node_8feat"][0] / m["quantum_8ch_cidnp"][0]
    m5 = m["ESN_5node_5feat"][0] / m["quantum_5ch_kinetics"][0]
    assert m8 == pytest.approx(1.49, abs=0.02)
    assert m5 == pytest.approx(2.23, abs=0.03)


# ── main text: survivor vs product register, all cells ───────────────
@pytest.mark.parametrize("route,surv,prod", [("YS_end", 1.02, 1.01),
                                             ("YS_t", 2.01, 2.74),
                                             ("SandT_t", 2.01, 2.76),
                                             ("cidnp", 4.65, 4.91)])
def test_product_register_row(route, surv, prod):
    d = _load(PANEL, "c1_product_carryover")
    assert d["survivor"][route]["IPC"] == pytest.approx(surv, abs=0.005)
    assert d["product"][route]["IPC"] == pytest.approx(prod, abs=0.005)


# ── SI S3: the injection map ─────────────────────────────────────────
@pytest.mark.parametrize("key,ipc,sd", [("original_e1_reset", 0.000, 0.000),
                                        ("corrected_ST_birth", 2.941, 0.108),
                                        ("corrected_ST0_birth", 3.116, 0.209)])
def test_injection_row(key, ipc, sd):
    v = _load(ROOT, "corrected_injection")[key]
    assert v["IPC_mean"] == pytest.approx(ipc, abs=0.001)
    assert v["IPC_sd"] == pytest.approx(sd, abs=0.001)


# ── SI S4: time-grid convergence ─────────────────────────────────────
@pytest.mark.parametrize("n_t,ys,cid", [(24, 2.756, 4.879), (48, 2.050, 4.614),
                                        (96, 2.008, 4.650), (192, 1.988, 4.656)])
def test_grid_row(n_t, ys, cid):
    rows = {r["n_t"]: r for r in _load(ROOT, "grid_convergence")}
    assert rows[n_t]["YS_t"] == pytest.approx(ys, abs=0.001)
    assert rows[n_t]["cidnp"] == pytest.approx(cid, abs=0.001)


# ── SI S4: product-register audit table ──────────────────────────────
def test_product_audit_table():
    n = _load(PANEL, "audit3_nuclide_product")
    assert n["proton_fraction"] == pytest.approx(0.297, abs=0.003)
    d = {r["jitter_cycles"]: r for r in _load(PANEL, "audit3_desync_product")}
    assert d[2.0]["excess_5ch"] == pytest.approx(-0.006, abs=0.002)
    assert d[2.0]["excess_8ch"] == pytest.approx(1.153, abs=0.003)
    a = {r["eta"]: r for r in _load(PANEL, "audit3_anisotropy_product")}
    assert a[0.5]["fraction_of_isotropic"] == pytest.approx(0.403, abs=0.003)
    assert a[1.0]["excess_8ch"] == pytest.approx(0.005, abs=0.002)


# ── SI S5: coherence fraction, all rows and SDs ──────────────────────
@pytest.mark.parametrize("channel,gamma,ipc,sd",
                         [("all5", 1000.0, 0.999144, 0.010952),
                          ("all5", 5000.0, 0.998297, 0.010776),
                          ("all5", 20000.0, 0.998288, 0.010787),
                          ("electrons_only", 1000.0, 0.994862, 0.022166),
                          ("electrons_only", 5000.0, 0.989145, 0.022340),
                          ("electrons_only", 20000.0, 0.989127, 0.022352)])
def test_coherence_control_row(channel, gamma, ipc, sd):
    rows = {r["gamma_MHz"]: r for r in _load(PANEL, "c6_coherence_control")[channel]["rows"]}
    assert rows[gamma]["IPC"] == pytest.approx(ipc, abs=1e-5)
    assert rows[gamma]["sd"] == pytest.approx(sd, abs=1e-5)


def test_coherence_fraction_values():
    f = _load(ROOT, "final_numbers")["F2"]
    assert f["coherent"] == pytest.approx(5.4206, abs=1e-3)
    assert f["incoherent"] == pytest.approx(0.9986, abs=1e-3)
    assert f["fraction"] == pytest.approx(0.8158, abs=1e-3)


# ── SI S11: spin-1 table, all printed cells ──────────────────────────
@pytest.mark.parametrize("spin,dim,ipc,floor,excess",
                         [(0.5, 16, 3.267, 1.984, 1.283),
                          (1.0, 24, 3.639, 1.997, 1.642)])
def test_spin1_row(spin, dim, ipc, floor, excess):
    rows = {r["nuclear_spin"]: r for r in _load(PANEL, "open3_spin1")}
    assert rows[spin]["dim"] == dim
    assert rows[spin]["IPC_cidnp"] == pytest.approx(ipc, abs=0.001)
    assert rows[spin]["floor_cidnp"] == pytest.approx(floor, abs=0.001)
    assert rows[spin]["excess_cidnp"] == pytest.approx(excess, abs=0.001)


# ── SI S12: validation table ─────────────────────────────────────────
@pytest.mark.parametrize("system,ratio", [("2H in D2O (quadrupolar)", 1.08),
                                          ("14N of free amino acid (quadrupolar)", 0.46),
                                          ("1H2O intramolecular (dipolar)", 1.54)])
def test_validation_row(system, ratio):
    rows = {r["system"]: r for r in _load(PANEL, "open5_relaxation_estimate")["validation"]}
    assert rows[system]["ratio"] == pytest.approx(ratio, abs=0.01)


# ── SI S2: the Dambre control, printed values ────────────────────────
def test_dambre_printed_values():
    d = _load(PANEL, "s2_dambre_control")
    assert d["mean_IPC"] == pytest.approx(1.013, abs=0.001)
    assert d["sd_IPC"] == pytest.approx(0.009, abs=0.001), \
        "SI prints 1.013 +/- 0.009 -- this is the manuscript-side expectation"

    assert d["n_above_bound"] == 29, "the printed 29 of 30"


# ── main text readout table: the floor and excess columns ────────────
# Set-5 audit: only IPC and channel count were bound; the two right-hand columns
# printed in the manuscript were not.
READOUT_FLOOR_EXCESS = [
    ("YS_end",     1.02, 0.00),
    ("SandT_end",  1.02, 0.36),
    ("YS_t",       1.02, 0.99),
    ("SandT_t",    1.02, 0.99),
    ("cidnp",      2.01, 2.64),
    ("YS_t_accum", 1.96, 0.75),
]


@pytest.mark.parametrize("key,floor,excess", READOUT_FLOOR_EXCESS)
def test_readout_floor_and_excess_column(key, floor, excess):
    f = _load(PANEL, "m7_route_floors")[key]["floor_IPC"]
    ipc = _load(ROOT, "readout_routes")[key]["IPC"]
    assert f == pytest.approx(floor, abs=0.006), "the printed floor column"
    assert ipc - f == pytest.approx(excess, abs=0.006), "the printed excess column"


def test_hetero_route_floor_column():
    assert _load(PANEL, "m7_hetero_floor")["floor_IPC"] == pytest.approx(1.02, abs=0.006)


# ── SI Table S7: the superseded single-realisation clock scan ────────
# Retained in the SI for continuity, so its printed cells must still be bound.
@pytest.mark.parametrize("T_d,mc5,mc8", [(1e-6, 1.948, 2.818), (1e-3, 1.948, 2.818),
                                         (1e-2, 1.947, 2.817), (1e-1, 1.944, 2.800)])
def test_superseded_clock_row(T_d, mc5, mc8):
    rows = {r["T_d_s"]: r for r in _load(ROOT, "turnover_clock")}
    assert rows[T_d]["MC_5ch"] == pytest.approx(mc5, abs=0.001)
    assert rows[T_d]["MC_8ch"] == pytest.approx(mc8, abs=0.001)


# ── SI Table S8: register-reuse scan ─────────────────────────────────
@pytest.mark.parametrize("q,mc5,mc8", [(0.00, 1.984, 2.890), (0.30, 1.958, 2.667),
                                       (0.50, 1.927, 2.288), (0.70, 1.788, 2.030),
                                       (0.90, 1.218, 1.977), (1.00, 1.002, 1.002)])
def test_register_reuse_row(q, mc5, mc8):
    rows = {r["q_nuc"]: r for r in _load(ROOT, "register_reuse")}
    assert rows[q]["MC_5"] == pytest.approx(mc5, abs=0.001)
    assert rows[q]["MC_8"] == pytest.approx(mc8, abs=0.001)


# ── SI Table S15: the feasible-region windows, as printed ────────────
@pytest.mark.parametrize("tau_ns,window", [
    (0.1, "5.4 - 1.34e+03 ms"),
    (1.0, "5.6 - 134 ms"),
    (5.0, "9.8 - 26.9 ms"),
    (10.0, "NONE"),
    (15.2, "NONE"),
])
def test_feasible_window_string(tau_ns, window):
    """The criterion sentence quotes these windows; they must be the stored ones."""
    rows = {r["tau_c_ns"]: r for r in _load(PANEL, "open5_turnover_estimate")["feasible_region"]}
    assert rows[tau_ns]["window"] == window


# ── SI S10: heterogeneity scan, at the documented default arguments ──
@pytest.mark.parametrize("spread,ipc8", [(0.00, 4.660), (0.05, 4.857), (0.10, 4.822),
                                         (0.20, 4.765), (0.40, 4.790)])
def test_heterogeneity_row(spread, ipc8):
    """Regenerated at n_copies=16, n_seeds=3 so that the shipped file matches the
    command documented in the README; the set-5 audit found it shipped at 8/2."""
    rows = {r["spread"]: r for r in _load(PANEL, "open1_ensemble_heterogeneity")}
    assert rows[spread]["n_copies"] == 16, "must match the documented default"
    assert rows[spread]["IPC_8ch"] == pytest.approx(ipc8, abs=0.005)


def test_heterogeneity_is_not_a_threat():
    rows = _load(PANEL, "open1_ensemble_heterogeneity")
    base = [r for r in rows if r["spread"] == 0.0][0]["IPC_8ch"]
    for r in rows:
        assert r["IPC_8ch"] > 0.95 * base, f"spread {r['spread']} lost too much"


# ── SI Table S12: pooling copies with heterogeneous tau_c (set 7) ───────────
# Added when the last uncomputed ensemble premise was closed. The mean-field
# column is the load-bearing one: without it the table shows only that pooling
# is benign, not why, and the manuscript claims the stronger statement.
_TAUC = _load(PANEL, "open6_tau_c_heterogeneity")


@pytest.mark.parametrize("sigma,tau_lo,tau_hi,qbar,pooled,spread", [
    (0.0,  5.00,  5.00, 0.739, +1.956, -0.000),
    (0.3,  3.30,  7.57, 0.735, +1.962, +0.003),
    (0.6,  2.18, 11.46, 0.722, +1.985, +0.017),
    (1.0,  1.25, 19.93, 0.696, +2.044, +0.059),
])
def test_tau_c_heterogeneity_row(sigma, tau_lo, tau_hi, qbar, pooled, spread):
    """The values behind SI Table S12. The printed cells themselves are parsed
    from the .tex by test_printed_cells.py; this binds the data side."""
    r = next(x for x in _TAUC if x["sigma_ln"] == sigma)
    assert r["tau_c_min_ns"] == pytest.approx(tau_lo, abs=0.005)
    assert r["tau_c_max_ns"] == pytest.approx(tau_hi, abs=0.005)
    assert r["q_mean"] == pytest.approx(qbar, abs=0.0005)
    assert r["excess_8ch"] == pytest.approx(pooled, abs=0.0005)
    assert r["spread_effect_8ch"] == pytest.approx(spread, abs=0.0005)


def test_tau_c_spread_is_small_but_not_mean_field():
    """Two separate facts, previously conflated into a wrong one.

    The first version of this test divided the spread residual by the TOTAL
    excess, got under 3%, and the manuscript concluded the rise was "almost all"
    a mean-field effect. That denominator is wrong for that claim: the rise over
    sigma = 0 is itself only ~0.09, so the residual is a MAJORITY of it. The
    heterogeneity effect is small in absolute terms and it is favourable, but it
    is not explained by the shift in mean survival.
    """
    rows = sorted(_TAUC, key=lambda r: r["sigma_ln"])
    base = rows[0]["excess_8ch"]
    for r in rows[1:]:
        rise = r["excess_8ch"] - base
        assert rise > 0, f"sigma={r['sigma_ln']}: pooling reduced the excess"
        # (a) the whole heterogeneity effect stays small
        assert rise / base < 0.05, (
            f"sigma={r['sigma_ln']}: heterogeneity moved the excess by "
            f"{rise/base:.1%}, too much for a single effective T_1n")
        # (b) but it is NOT mostly the mean-field shift
        frac = r["spread_effect_8ch"] / rise
        assert 0.4 < frac < 0.8, (
            f"sigma={r['sigma_ln']}: spread is {frac:.0%} of the rise; the "
            f"manuscript states 46-66%")


def test_tau_c_spread_residual_is_positive_at_every_nonzero_spread():
    """sigma = 0 is excluded deliberately: there the residual is -8e-13, i.e.
    numerically zero with an arbitrary sign, and asserting positivity there
    would be asserting the sign of rounding noise."""
    for r in _TAUC:
        if r["sigma_ln"] == 0.0:
            assert abs(r["spread_effect_8ch"]) < 1e-9
            continue
        assert r["spread_effect_8ch"] > 0, (
            f"sigma={r['sigma_ln']}: spread residual turned negative")


def test_tau_c_meanfield_is_exact_at_zero_spread():
    """The decomposition's own consistency check: with no spread, the
    mean-field control IS the pooled case, so the residual must vanish."""
    r = next(x for x in _TAUC if x["sigma_ln"] == 0.0)
    assert r["q_min"] == r["q_max"] == r["q_mean"]
    assert abs(r["spread_effect_8ch"]) < 1e-9
    assert r["excess_8ch"] == pytest.approx(r["excess_8ch_meanfield"], abs=1e-9)


def test_tau_c_pooled_excess_never_falls_below_the_uniform_pool():
    """Limitations (ii) says pooling is *safe*. That is falsified the moment a
    spread drives the excess below the sigma = 0 value."""
    base = next(x for x in _TAUC if x["sigma_ln"] == 0.0)["excess_8ch"]
    for r in _TAUC:
        assert r["excess_8ch"] >= base - 1e-9, (
            f"sigma={r['sigma_ln']}: excess {r['excess_8ch']:.4f} < uniform {base:.4f}")


# ── SI robustness table (set 12): every motivated variation of the ceiling ──
# Added when the printed-cell guard flagged the table as having no data behind
# it -- the same orphan-table problem the set-5 audit found, caught by a guard
# this time instead of by a reviewer.
_ROBUST = _load(PANEL, "open5_turnover_estimate")["robustness"]


@pytest.mark.parametrize("variation,tau_c,T1,ceiling", [
    ("as published, hydration 1.3", 15.24, 2.442, 6.15),
    ("hydration 1.0", 11.73, 3.175, 7.99),
    ("hydration 1.6", 18.76, 1.984, 4.99),
    ("viscosity x2", 30.49, 1.221, 3.07),
    ("viscosity x4", 60.97, 0.611, 1.54),
    ("120 kDa dimer", 30.49, 1.221, 3.07),
    ("2 dipolar partners", 15.24, 1.221, 3.07),
    ("3 dipolar partners", 15.24, 0.814, 2.05),
    ("memory threshold 0.2", 15.24, 2.442, 6.15),
    ("memory threshold 1.0", 15.24, 2.442, 4.14),
    ("+ bath (f=0.54)", 15.24, 1.584, 3.99),
])
def test_robustness_row(variation, tau_c, T1, ceiling):
    r = next(x for x in _ROBUST if x["variation"] == variation)
    assert r["tau_c_ns"] == pytest.approx(tau_c, abs=0.005)
    assert r["T1_ms"] == pytest.approx(T1, abs=0.0005)
    assert r["ceiling_ms"] == pytest.approx(ceiling, abs=0.005)


def test_no_motivated_variation_reaches_the_band():
    """The point of the table. If any row ever lands in the band, the paper's
    central exclusion is conditional on something it does not state."""
    inside = [r["variation"] for r in _ROBUST if r["in_band"]]
    assert not inside, f"these variations reach the 10 ms band: {inside}"
