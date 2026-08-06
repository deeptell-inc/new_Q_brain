"""Bind the spin parameters printed in the manuscript to the source constants.

The set-6 audit found the spin-parameter table (S18 after set 7 inserted
Table S12) unbound by any test. Unlike every other printed
table, its cells are *inputs* rather than simulation outputs, so they are not
serialized to any JSON and the JSON-comparison tests could not reach them. That
made them the one place where the manuscript could silently disagree with the
code that produced every other number in it.

These tests close that gap by parsing the printed values straight out of the
.tex and comparing them to the dictionaries the reservoir is actually built
from. The millitesla column is not an independent input -- it is the megahertz
column divided by the electron gyromagnetic ratio -- so it is checked as a
derived quantity rather than transcribed.
"""

import math
import pathlib
import re

import pytest

from qbscreen.general_spin import G_E, MU_B, HBAR
from qbscreen.final_numbers import ENG
from qbscreen.readout_routes import CRY

SI = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "supplementary.tex"
MAIN = pathlib.Path(__file__).resolve().parents[2] / "manuscript" / "main.tex"

# MHz of electron Zeeman splitting per millitesla, from the same constants the
# Hamiltonian builder uses -- not a hard-coded 28.025.
MHZ_PER_MT = G_E * MU_B * 1e-3 / (2 * math.pi * HBAR) / 1e6


def _si():
    return SI.read_text()


def test_gyromagnetic_ratio_is_the_textbook_value():
    """Guards the conversion the other tests lean on."""
    assert MHZ_PER_MT == pytest.approx(28.025, abs=0.005)


# ── Table S18: the cryptochrome-point hyperfine assignment ──────────────────
@pytest.mark.parametrize("row,key,mhz,mt", [
    ("A_{e_1a}", "A_e1_a", 14.0, 0.50),
    ("A_{e_1b}", "A_e1_b",  5.0, 0.18),
    ("A_{e_2a}", "A_e2_a", 40.0, 1.43),
])
def test_table_s18_row(row, key, mhz, mt):
    """Each printed row matches CRY, and its mT cell is the MHz cell converted."""
    # the row as it appears:  $A_{e_1a}$ & FAD$^{\bullet-}$, $^{14}$N5 & $\sim0.50$ & $14$ \\
    lines = [l for l in _si().splitlines() if l.startswith("$" + row + "$ &")]
    assert len(lines) == 1, f"expected exactly one Table S18 row for {row}, got {len(lines)}"
    cells = [c.strip() for c in lines[0].split("&")]
    printed_mt = float(re.fullmatch(r"\$\\sim([\d.]+)\$", cells[2]).group(1))
    printed_mhz = float(re.fullmatch(r"\$([\d.]+)\$ *\\\\", cells[3]).group(1))

    assert printed_mhz == mhz, f"{row}: SI prints {printed_mhz} MHz"
    assert CRY[key] == mhz, f"{row}: CRY has {CRY[key]} MHz, SI prints {mhz}"
    assert printed_mt == mt
    # the mT column is derived, so it must round-trip through the ratio
    assert printed_mt == pytest.approx(CRY[key] / MHZ_PER_MT, abs=0.005), (
        f"{row}: {CRY[key]} MHz is {CRY[key]/MHZ_PER_MT:.3f} mT, SI prints {printed_mt}")


def test_table_s18_has_exactly_the_three_couplings_the_model_uses():
    """No printed row without a model parameter, and none the other way."""
    t = _si()
    block = t[t.find("\\label{tab:params}"):]
    block = block[:block.find("\\end{table}")]
    rows = re.findall(r"\$A_\{e_(\d)([ab])\}\$", block)
    assert sorted(f"A_e{a}_{b}" for a, b in rows) == ["A_e1_a", "A_e1_b", "A_e2_a"]
    assert sorted(k for k in CRY if k.startswith("A_")) == ["A_e1_a", "A_e1_b", "A_e2_a"]


# ── the cryptochrome point's field and exchange coupling ────────────────────
def test_cryptochrome_field_is_the_geomagnetic_value():
    """The text quotes 50 uT throughout; CRY must hold exactly that."""
    assert CRY["B_tesla"] == 50e-6
    assert "$B = 50~\\mu$T" in _si() or "50~\\mu$T" in _si()


def test_cryptochrome_exchange_coupling_is_zero():
    """Every 'J = 0' claim in the paper rests on this one constant."""
    assert CRY["J"] == 0.0


# ── the engineered operating point, quoted in both documents ────────────────
@pytest.mark.parametrize("doc", ["main", "si"])
def test_engineered_point_matches_ENG(doc):
    t = MAIN.read_text() if doc == "main" else _si()
    m = re.search(r"\$A_\{e_1a\}=(\d+)\$, \$A_\{e_1b\}=(\d+)\$, \$A_\{e_2a\}=(\d+)\$~MHz", t)
    assert m, f"engineered hyperfine sentence not found in {doc}.tex"
    a1, b1, a2 = (int(x) for x in m.groups())
    assert (a1, b1, a2) == (ENG["A_e1_a"], ENG["A_e1_b"], ENG["A_e2_a"])


def test_engineered_field_and_exchange():
    """'J = 2 MHz, B = 1 mT' -- both documents print it, ENG defines it."""
    assert ENG["J"] == 2.0
    assert ENG["B_tesla"] == 1e-3
    assert ENG["B_tesla"] * 1e3 == 1.0, "1 mT"
    for t in (_si(), MAIN.read_text()):
        assert "$J=2$~MHz" in t


def test_engineered_cycle_time_and_dephasing():
    """tau = T2e = 50 ns is quoted in the SI and passed to run_corr_obs."""
    assert "$\\tau=T_2^e=50$~ns" in _si()
    src = (pathlib.Path(__file__).resolve().parents[1] / "final_numbers.py").read_text()
    # run_corr_obs(s, build_reservoir_H(**ENG), 0.05, 50.0, ...) -- us then ns
    assert re.search(r"run_corr_obs\(s, build_reservoir_H\(\*\*ENG\), 0\.05, 50\.0", src), \
        "headline capacity no longer runs at tau=0.05 us, T2e=50 ns"


def test_the_two_operating_points_are_distinct_everywhere():
    """A copy-paste that collapsed CRY into ENG would silently invalidate every
    comparison between the biological and engineered points."""
    assert CRY != ENG
    assert all(CRY[k] != ENG[k] for k in ("B_tesla", "J", "A_e1_a", "A_e1_b", "A_e2_a"))
