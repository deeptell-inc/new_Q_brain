"""Bind the criterion numbers that live in PROSE, not in tables.

test_printed_cells.py closed the gap between the manuscript tables and the data.
It closed it only for tables: `_all_tables` walks `\\label{tab:...}` in main.tex
and supplementary.tex, so a number written in a sentence -- or written in a
document that is not one of those two -- is not seen by any test.

The set-10 audit walked through that hole. Converging the q grid moved the
ceiling from 5.5 to 6.1 ms and the boundary from 8.37 to 9.37 ns; main.tex and
supplementary.tex were updated and 258 tests passed, while seven numeric cells
in README.md, claims-ledger.md and cover_letter.tex kept the pre-convergence
values. One of them, the methyl's wet boundary in SI S12, contradicted the very
next sentence of its own paragraph. Every one of the seven sat in prose, and
five of them sat in the claims ledger -- the instrument whose job is to catch
exactly this, and which nothing tested.

So this module binds the load-bearing criterion numbers wherever they are
printed, by reading the document text and comparing against the shipped JSON at
the precision the document itself chose. Hard-coding the expected value here
would reproduce the failure it exists to prevent, so nothing below is a
constant: every expectation is derived from simulation_results/.
"""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAN = ROOT / "manuscript"
PANEL = ROOT / "simulation_results" / "panel"

TURNOVER = json.loads((PANEL / "open5_turnover_estimate.json").read_text())
CRIT = TURNOVER["critical_tau_c"]
TRP = CRIT["per_nucleus"]["Trp H-beta (CH2, geminal partner)"]
METHYL = CRIT["per_nucleus"]["flavin 8-alpha CH3 (intra-methyl)"]
ROBUST = {r["variation"]: r for r in TURNOVER["robustness"]}
DRY = ROBUST["as published, hydration 1.3"]
WET = ROBUST["+ bath (f=0.54)"]
REGION = {r["tau_c_ns"]: r for r in TURNOVER["feasible_region"]}

TAU_PROT = CRIT["tau_protein_ns"]


def _text(rel):
    return (ROOT / rel).read_text()


def _printed(doc, pattern):
    """The single number a sentence prints, and the precision it printed it at."""
    hits = re.findall(pattern, _text(doc))
    assert len(hits) == 1, (
        f"{doc}: pattern {pattern!r} matched {len(hits)} times, expected 1. "
        "The anchor has drifted; fix the anchor, not the number.")
    return hits[0]


def _agrees(printed, value):
    """Does the printed string round-trip to the shipped value at its own precision?"""
    d = len(printed.split(".")[1]) if "." in printed else 0
    return abs(round(value, d) - float(printed)) < 10 ** (-d) / 2


# (id, document, regex whose group 1 is the printed number, value from the data)
#
# Every entry must be a number a reader would act on. Decorative or purely
# historical figures are listed in WAIVED below with the reason.
CLAIMS = [
    # --- the requirement, as each document states it -------------------
    ("README speed-up",
     "README.md",
     r"must reorient \$\\gtrsim([\d.]+)\\times\$ faster than the protein",
     TAU_PROT / TRP["tau_crit_wet_ns"]),
    ("README speed-up, no bath",
     "README.md",
     r"That factor is \$([\d.]+)\\times\$ from the solved boundary without the bath",
     TAU_PROT / TRP["tau_crit_dry_ns"]),
    ("README speed-up, with bath",
     "README.md",
     r"without the bath and \$([\d.]+)\\times\$ with it",
     TAU_PROT / TRP["tau_crit_wet_ns"]),
    ("README ceiling, no bath",
     "README.md",
     r"caps the horizon at \$([\d.]+)\$ ms on the intramolecular dipolar term",
     DRY["ceiling_ms"]),
    ("README T1 and ceiling, with bath",
     "README.md",
     r"alone — \$([\d.]+)\$ ms and \$[\d.]+\$ ms once the intermolecular proton bath",
     WET["T1_ms"]),
    ("README ceiling, with bath",
     "README.md",
     r"alone — \$[\d.]+\$ ms and \$([\d.]+)\$ ms once the intermolecular proton bath",
     WET["ceiling_ms"]),

    # --- the claims ledger, E6 / E7 / E8 -------------------------------
    ("ledger E6 ceiling, no bath",
     "manuscript/claims-ledger.md",
     r"horizon 上限は \$([\d.]+)\$ ms（帯域外）",
     DRY["ceiling_ms"]),
    ("ledger E6 ceiling, with bath",
     "manuscript/claims-ledger.md",
     r"浴込みの上限は \$([\d.]+)\$ ms",
     WET["ceiling_ms"]),
    ("ledger E7 boundary, no bath",
     "manuscript/claims-ledger.md",
     r"浴なし \$([\d.]+)\$ ns",
     TRP["tau_crit_dry_ns"]),
    ("ledger E7 speed-up, no bath",
     "manuscript/claims-ledger.md",
     r"浴なし \$[\d.]+\$ ns（\$([\d.]+)\\times\$）",
     TAU_PROT / TRP["tau_crit_dry_ns"]),
    ("ledger E7 boundary, with bath",
     "manuscript/claims-ledger.md",
     r"浴込み \$([\d.]+)\$ ns（",
     TRP["tau_crit_wet_ns"]),
    ("ledger E7 speed-up, with bath",
     "manuscript/claims-ledger.md",
     r"浴込み \$[\d.]+\$ ns（\$([\d.]+)\\times\$）",
     TAU_PROT / TRP["tau_crit_wet_ns"]),
    ("ledger E8 bath ratio",
     "manuscript/claims-ledger.md",
     r"分子間プロトン浴 \$f=([\d.]+)\$",
     CRIT["bath_f"]),
    ("ledger E8 T1, with bath",
     "manuscript/claims-ledger.md",
     r"浴込みで \$T_1=([\d.]+)\$ ms",
     WET["T1_ms"]),
    ("ledger E8 ceiling",
     "manuscript/claims-ledger.md",
     r"天井 \$([\d.]+)\$ ms、境界",
     WET["ceiling_ms"]),
    ("ledger E8 boundary",
     "manuscript/claims-ledger.md",
     r"天井 \$[\d.]+\$ ms、境界 \$([\d.]+)\$ ns",
     TRP["tau_crit_wet_ns"]),

    ("ledger E7 window low, 5 ns",
     "manuscript/claims-ledger.md",
     r"（5 ns で ([\d.]+)–[\d.]+ ms）",
     float(REGION[5.0]["window"].split(" - ")[0])),
    ("ledger E7 window high, 5 ns",
     "manuscript/claims-ledger.md",
     r"（5 ns で [\d.]+–([\d.]+) ms）",
     float(REGION[5.0]["window"].split(" - ")[1].split()[0])),

    # --- the ledger's own summary section J1 ---------------------------
    ("ledger J1 ceiling",
     "manuscript/claims-ledger.md",
     r"ただし \$([\d.]+)\$ ms（E6）",
     DRY["ceiling_ms"]),

    # --- the cover letter: the first thing an editor reads -------------
    ("cover letter window low, 5 ns",
     "manuscript/cover_letter.tex",
     r"mobility \(\$([\d.]+)\$--\$[\d.]+\$~ms\s*\n?at \$\\tau_c=5\$~ns",
     float(REGION[5.0]["window"].split(" - ")[0])),
    ("cover letter window high, 5 ns",
     "manuscript/cover_letter.tex",
     r"mobility \(\$[\d.]+\$--\$([\d.]+)\$~ms\s*\n?at \$\\tau_c=5\$~ns",
     float(REGION[5.0]["window"].split(" - ")[1].split()[0])),
    ("cover letter window low, 0.1 ns",
     "manuscript/cover_letter.tex",
     r"up to \$([\d.]+)\$--\$[\d.]+\$~ms at \$0\.1\$~ns",
     float(REGION[0.1]["window"].split(" - ")[0])),
    ("cover letter window high, 0.1 ns",
     "manuscript/cover_letter.tex",
     r"up to \$[\d.]+\$--\$([\d.]+)\$~ms at \$0\.1\$~ns",
     float(REGION[0.1]["window"].split(" - ")[1].split()[0])),

    # --- SI S12: the sentence that contradicted its own successor ------
    ("SI methyl boundary, with bath",
     "manuscript/supplementary.tex",
     r"With the bath that boundary falls to \$([\d.]+)\$~ns",
     METHYL["tau_crit_wet_ns"]),
    ("SI methyl boundary, no bath",
     "manuscript/supplementary.tex",
     r"boundary sits at \$([\d.]+)\$~ns without the bath",
     METHYL["tau_crit_dry_ns"]),
    ("SI methyl speed-up",
     "manuscript/supplementary.tex",
     r"the methyl by a factor of \$([\d.]+)\$",
     TAU_PROT / METHYL["tau_crit_wet_ns"]),
]


@pytest.mark.parametrize("name,doc,pattern,value",
                         CLAIMS, ids=[c[0] for c in CLAIMS])
def test_prose_number_matches_the_data(name, doc, pattern, value):
    """A criterion number printed in prose must be the one the data produced."""
    printed = _printed(doc, pattern)
    assert _agrees(printed, value), (
        f"{doc}: {name} prints {printed} but the shipped data give {value:.4f}. "
        "Update the document, not this test.")


# Numbers inside the ledger's E-rows and the README result-3 row that the map
# above deliberately does not bind, each with the reason it is exempt.
WAIVED = {
    "1.6": "MC falls by 1.6% -- bound by test_clock_effect_is_resolved_...",
    "19": "horizon 19-188 ms -- bound by test_delay_kernel_and_horizon_definition",
    "188": "horizon 19-188 ms -- same",
    "10": "the 10-100 ms turnover interval, an input not an output",
    "100": "the 10-100 ms turnover interval, an input not an output",
    "12": "12 seeds -- bound by test_seed_counts",
    "2.4": "T1 without the bath -- bound by test_predicted_T1_row",
    "6": "the tau_c <= 6 ns criterion, rounded from the bound boundary above",
    "5": "the tau_c = 5 ns row label, an input to feasible_region",
    "0.542": "the bath ratio, bound by 'ledger E8 bath ratio' above",
    "1": "list markers and E-row indices",
}


def test_every_number_in_the_criterion_prose_is_bound_or_waived():
    """Coverage cannot shrink silently when a sentence is rewritten.

    The seven cells the set-10 audit found were not wrong because a test was
    wrong; they were wrong because no test looked. This one fails when a new
    number appears in the criterion prose without a binding or a waiver.
    """
    bound = set()
    for _, doc, pattern, _ in CLAIMS:
        m = re.search(pattern, _text(doc))
        if m:
            bound.add(m.group(1))

    blocks = []
    ledger = _text("manuscript/claims-ledger.md")
    for row in re.findall(r"^\| E[678] \|.*$", ledger, re.M):
        blocks.append(("claims-ledger.md E-row", row))
    readme = _text("README.md")
    row3 = [ln for ln in readme.splitlines() if ln.startswith("| 3 |")]
    assert len(row3) == 1, "README result-3 row not found"
    blocks.append(("README.md result 3", row3[0]))

    unbound = []
    for where, block in blocks:
        # drop the row's own index cell ("| 3 |", "| E7 |"): it labels the row,
        # it is not a measured quantity, and waiving the bare digits instead
        # would blind the scan to a real 3 or 7 appearing later in the sentence.
        body = "|".join(block.split("|")[2:])
        for n in re.findall(r"\$?([\d]+(?:\.[\d]+)?)", body):
            if n in bound or n in WAIVED:
                continue
            unbound.append(f"{where}: {n}")
    assert not unbound, (
        "numbers in the criterion prose with neither a binding in CLAIMS nor an "
        "entry in WAIVED:\n  " + "\n  ".join(sorted(set(unbound))))


def test_the_ledger_and_the_readme_state_the_same_requirement():
    """The two summaries drifted apart once; they must not disagree again."""
    readme = _printed("README.md",
                      r"must reorient \$\\gtrsim([\d.]+)\\times\$ faster")
    ledger = _printed("manuscript/claims-ledger.md",
                      r"浴込み \$[\d.]+\$ ns（\$([\d.]+)\\times\$）")
    assert abs(float(readme) - float(ledger)) < 0.06, (
        f"README says the register must reorient {readme}x faster; the ledger "
        f"records {ledger}x for the same nucleus.")
