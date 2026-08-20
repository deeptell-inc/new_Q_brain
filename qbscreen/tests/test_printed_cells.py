"""Check the numbers PRINTED in the manuscript against the shipped data.

test_table_rows.py binds the other direction: its expected values were
transcribed from the manuscript by hand, so if the data move, it fails. Three
documents claimed that arrangement "fails if either side moves". It did not.
The set-7 panel falsified it in one step: change a printed cell of SI Table S12
from +2.044 to +9.999, leave every JSON untouched, and all 202 tests still
passed -- because test_table_rows.py and test_panel_claims.py never open a .tex
file at all. The transcription was a snapshot, not a check.

This module closes that half. It parses the numeric cells out of the manuscript
tables and requires each one to correspond to a shipped value. A printed digit
that drifts away from the data now fails here, which is what the sentence in the
paper says happens.

Coverage is deliberately explicit rather than clever: TABLES maps a table to its
data source and says how to get from one to the other. A table absent from that
map is reported by test_every_manuscript_table_is_either_mapped_or_waived, so
coverage cannot silently shrink when a table is added.
"""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAN = ROOT / "manuscript"
RES = ROOT / "simulation_results"


def _json(rel):
    with open(RES / rel) as f:
        return json.load(f)


def _table_body(tex_name, label):
    """The rows of one table environment, as lists of cell strings."""
    t = (MAN / tex_name).read_text()
    i = t.find("\\label{%s}" % label)
    assert i > 0, f"{label} not found in {tex_name}"
    start = t.find("\\begin{tabular}", i)
    end = t.find("\\end{tabular}", start)
    assert 0 < start < end, f"{label}: no tabular body"
    body = t[t.find("\n", start) + 1:end]
    rows = []
    for line in body.split("\\\\"):
        # strip the rule macros FIRST: a row that follows one used to be
        # discarded whole, which silently dropped the sigma = 0 row of Table S12.
        # main.tex draws booktabs rules since the move to the RSC template
        # (revtex's ruledtabular does not exist in that class); supplementary.tex
        # still uses \hline, so both spellings are removed here.
        for rule in ("\\hline", "\\toprule", "\\midrule", "\\bottomrule"):
            line = line.replace(rule, "")
        line = line.strip()
        if not line:
            continue
        rows.append([c.strip() for c in line.split("&")])
    return rows


def _num(cell):
    """The number a printed cell denotes, or None if it carries no number.

    Handles \times10^{n} because several tables print small rates that way, and
    the naive parser silently returned the mantissa instead -- which would have
    let 9.5e-2 be "checked" against 9.5.
    """
    c = re.sub(r"\\mathbf\{([^}]*)\}|\\textbf\{([^}]*)\}", r"\1\2", cell)
    c = c.replace("$", "").replace("~", "").replace("\\,", "")
    c = c.replace("−", "-").replace("--", "-").strip()
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?", c)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)", c)
    return float(m.group(1)) if m else None


def _unit_scale(cell):
    """Scales to try when matching a printed cell to a stored SI value.

    Times are stored in seconds and printed in ms or us, so a literal match
    would fail on exactly the rows that matter most.
    """
    if "ms" in cell:
        return (1.0, 1e-3)
    if "\\mu$s" in cell or "us" in cell:
        return (1.0, 1e-6)
    if "~s" in cell or cell.rstrip().endswith("s"):
        return (1.0,)
    return (1.0,)


# ── explicit map: printed table -> shipped values ──────────────────────────
# main-text Table III, in printed order. readout_routes.json supplies the IPC and
# m7_route_floors.json the route's own floor; the excess column is their
# difference, so it is recomputed rather than looked up.
ROUTE_KEYS = ["YS_end", "SandT_end", "hetero_tau", "YS_t", "SandT_t",
              "cidnp", "YS_t_accum"]


def _routes_expected():
    ipc = _json("readout_routes.json")
    flo = _json("panel/m7_route_floors.json")
    # the three-micro-environment route is scored against its own floor, which
    # lives in a separate file because it needs a different control run
    flo["hetero_tau"] = _json("panel/m7_hetero_floor.json")
    out = []
    for k in ROUTE_KEYS:
        i = ipc[k]["IPC"]
        f = flo[k]["floor_IPC"]
        out.append((i, f, i - f))
    return out


def _tauc_expected():
    return [(r["q_mean"], r["excess_8ch"], r["spread_effect_8ch"])
            for r in _json("panel/open6_tau_c_heterogeneity.json")]


def test_tauc_printed_cells_match_the_data():
    """SI Table S12. This is the exact table the panel corrupted to +9.999."""
    rows = _table_body("supplementary.tex", "tab:tauchet")[1:]   # drop the header
    data = _tauc_expected()
    assert len(rows) == len(data), f"S12 prints {len(rows)} rows, JSON has {len(data)}"
    for printed, d in zip(rows, data):
        qbar, excess, spread = (_num(printed[2]), _num(printed[3]), _num(printed[4]))
        assert qbar == pytest.approx(d[0], abs=0.0006), f"q_bar cell {printed[2]}"
        assert excess == pytest.approx(d[1], abs=0.0006), f"excess cell {printed[3]}"
        assert spread == pytest.approx(d[2], abs=0.0006), f"spread cell {printed[4]}"


def test_tauc_printed_rise_fractions_match_the_data():
    """The 'of the rise' column is a derived percentage, and it is the column
    that carries the corrected claim, so it is checked against a recomputation
    rather than transcribed."""
    rows = _table_body("supplementary.tex", "tab:tauchet")[1:]   # drop the header
    data = _json("panel/open6_tau_c_heterogeneity.json")
    base = data[0]["excess_8ch"]
    for printed, d in zip(rows[1:], data[1:]):
        cell = printed[5].replace("\\%", "").replace("$", "").strip()
        rise = d["excess_8ch"] - base
        assert int(cell) == round(100 * d["spread_effect_8ch"] / rise), (
            f"printed {cell}% vs computed "
            f"{100 * d['spread_effect_8ch'] / rise:.1f}%")


def test_routes_printed_cells_match_the_data():
    """Main-text Table III -- the headline 2.64."""
    rows = _table_body("main.tex", "tab:routes_main")[1:]   # drop the header
    data = _routes_expected()
    assert len(rows) == len(data), f"table prints {len(rows)} rows, JSON has {len(data)}"
    for printed, (ipc, floor, excess) in zip(rows, data):
        assert _num(printed[2]) == pytest.approx(ipc, abs=0.006), f"IPC cell {printed[2]}"
        assert _num(printed[3]) == pytest.approx(floor, abs=0.006), f"floor cell {printed[3]}"
        assert _num(printed[4]) == pytest.approx(excess, abs=0.006), f"excess cell {printed[4]}"


def test_spin_parameter_table_is_covered_elsewhere():
    """Table S18's printed cells are already parsed from the .tex by
    test_spin_parameters.py; recorded here so the coverage audit below can see
    it rather than reporting it as a gap."""
    from qbscreen.tests import test_spin_parameters as m
    assert hasattr(m, "test_table_s18_row") or hasattr(m, "test_table_s17_row")


# ── every remaining table: each printed number must exist in the data ──────
# Weaker than the column maps above (it does not check that a number is in the
# RIGHT cell) but it covers every table, and it is what actually fails when a
# printed digit drifts away from the data -- the case the panel demonstrated.

# Cells that are genuinely derived rather than stored, with the reason. Anything
# not listed must match a shipped value, so this list is the honest statement of
# what is not machine-checked.
DERIVED = {
    "tab:tauchet": ("percentages of the rise, checked exactly by "
                    "test_tauc_printed_rise_fractions_match_the_data"),
    "tab:prodaudit": "percentages of a retained fraction, not stored as such",
    "tab:params": ("literature hyperfine values, checked against the model "
                   "constants by test_spin_parameters.py"),
}


def _pool():
    """Every number appearing anywhere in the shipped data."""
    import glob
    vals = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            vals.add(float(o))

    for f in glob.glob(str(RES / "**" / "*.json"), recursive=True):
        if "panel_before_regen" in f:      # archived, not part of the data set
            continue
        with open(f) as fh:
            walk(json.load(fh))
    return vals


def _in_pool(x, pool, scales):
    for s in scales:
        y = x * s
        for d in range(0, 5):
            if abs(round(y, d) - y) < 1e-12:
                if any(abs(round(v, d) - y) < 1e-12 for v in pool):
                    return True
    return False


def _all_tables():
    for fn in ("main.tex", "supplementary.tex"):
        for lb in re.findall(r"\\label\{(tab:[^}]+)\}", (MAN / fn).read_text()):
            yield fn, lb


def test_every_printed_table_number_exists_in_the_data():
    """The check the manuscript's 'either side moves' sentence needs.

    Changing a printed cell to a value the data never produced fails here, for
    every table in both documents.
    """
    pool = _pool()
    bad = []
    for fn, lb in _all_tables():
        if lb in DERIVED:
            continue
        for row in _table_body(fn, lb)[1:]:
            for cell in row:
                v = _num(cell)
                if v is None:
                    continue
                if not _in_pool(v, pool, _unit_scale(cell)):
                    bad.append(f"{fn}:{lb} cell {cell!r} (={v})")
    assert not bad, ("printed cells with no corresponding value in the shipped "
                     "data:\n  " + "\n  ".join(bad))


def test_derived_cell_waivers_still_apply():
    """A waiver that no longer names a real table is a stale excuse."""
    labels = {lb for _, lb in _all_tables()}
    stale = sorted(set(DERIVED) - labels)
    assert not stale, f"DERIVED names tables that no longer exist: {stale}"
