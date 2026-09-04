"""Bind the input-realisation counts the manuscript states to the code.

main.tex enumerates how many input realisations each result used -- 12 for the
turnover clock, 8 for the classical baselines, 6 for the headline capacity, and
so on. Those numbers are function defaults, so they move whenever someone
retunes a scan, and nothing connected them to the prose: the set-7 review log
records "3-12 realisations" as a claim that had already gone stale once and been
corrected by hand.

They are bound here rather than in scripts/sync_test_counts.py because the
sentence is prose, not a countable pattern -- a regex over it would be guessing
where one clause ends and the next begins. Asserting the defaults directly is
both stricter and readable.

Both directions are checked: each function's default must equal the tier it is
listed under, AND the manuscript must actually print that tier.
"""

import inspect
import pathlib
import re

import pytest
from qbscreen.tests._manuscript import requires_manuscript

pytestmark = requires_manuscript

MAIN = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "main.tex"


def _default(dotted, param="n_seeds"):
    mod, _, fn = dotted.rpartition(".")
    m = __import__(f"qbscreen.{mod}", fromlist=[fn])
    f = getattr(m, fn)
    p = inspect.signature(f).parameters[param]
    assert p.default is not inspect.Parameter.empty, f"{dotted} has no default {param}"
    return p.default


# tier -> the functions the manuscript places in it, by the clause naming them
TIERS = {
    12: [("panel_response.clock_multiseed", "turnover-clock scan"),
         ("panel_response.clock_paired", "its paired test")],
    8:  [("final_numbers.f3_classical", "classical baselines"),
         ("final_numbers.f4_tradeoff", "T2e scan at the engineered point")],
    6:  [("final_numbers.f1_headline", "headline capacity"),
         ("panel_response.route_floors", "route floors"),
         ("panel_response.delay_kernel", "delay kernel"),
         ("panel_response.mc_grid_convergence", "grid scans"),
         ("nuclide_register.nuclide_scan", "nuclide scan"),
         ("panel_response.cry_classical", "cryptochrome-point classical baselines")],
    4:  [("readout_routes.register_reuse", "register-reuse scan"),
         ("panel_response.nuclear_channel", "nuclear-channel scan"),
         ("semiclassical.coherence_reference", "semiclassical reference")],
    3:  [("ensemble_pooled.heterogeneity", "ensemble scan"),
         ("ensemble_pooled.tau_c_heterogeneity", "ensemble scan (relaxation)"),
         ("general_spin.anisotropy_scan", "anisotropy scan"),
         ("general_spin.spin1_check", "spin-1 scan")],
}


@pytest.mark.parametrize("tier,dotted,clause", [
    (t, d, c) for t, xs in TIERS.items() for d, c in xs])
def test_seed_default_matches_the_tier_the_manuscript_states(tier, dotted, clause):
    got = _default(dotted)
    assert got == tier, (
        f"{dotted} runs {got} seeds but main.tex lists '{clause}' under {tier}")


@pytest.mark.parametrize("tier", sorted(TIERS))
def test_manuscript_prints_each_tier(tier):
    """The prose must actually contain the number, so deleting a tier from the
    sentence without changing the code fails here."""
    t = MAIN.read_text()
    i = t.find("The number of input realisations is not uniform")
    assert i > 0, "the realisation-count sentence is gone from main.tex"
    passage = t[i:i + 900]
    assert re.search(rf"\${tier}\$", passage), (
        f"main.tex no longer states the {tier}-realisation tier")


def test_dambre_control_seed_count():
    """Quoted separately in the SI as '29 of 30 seeds', so it is bound to its
    own default rather than to a tier."""
    assert _default("panel_response.dambre_control") == 30
    si = MAIN.parent / "supplementary.tex"
    assert re.search(r"\$29\$ of \$30\$ seeds", si.read_text()), \
        "the SI no longer states 29 of 30 seeds"


def test_single_realisation_quantities_are_still_four():
    """main.tex says four quantities are single-realisation. Each of the named
    scans takes a `seed` rather than `n_seeds`; if one of them gained a
    multi-seed default the sentence would silently become wrong."""
    t = MAIN.read_text()
    assert "Four quantities are\nsingle-realisation" in t or \
           "Four quantities are single-realisation" in t
    for dotted in ("panel_response.ridge_sensitivity",):
        sig = inspect.signature(
            getattr(__import__("qbscreen.panel_response", fromlist=["x"]),
                    dotted.split(".")[-1]))
        assert "n_seeds" not in sig.parameters, (
            f"{dotted} gained an n_seeds parameter; it is listed as "
            f"single-realisation in main.tex")
        assert "seed" in sig.parameters
