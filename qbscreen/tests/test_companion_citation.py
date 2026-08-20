"""Check what this paper says about its companion against the companion itself.

The two manuscripts cite each other, so a revision to one silently ages the
other's citation. That happened three times over: this paper cited the
companion's superseded title, an outdated bound, and -- worst -- an attribution
the companion had already reversed. It says the picosecond LIFETIME is decisive
and that removing the gigahertz exchange coupling changes the response by a few
per cent; the citation here was crediting the coupling.

The title is taken from the companion's main.tex, not its data availability
statement. That statement carries its own stale copy of the title, and reading
it is how the wrong title got in here in the first place.

Skips when the companion repository is not checked out beside this one -- a
referee cloning only this package should not see a failure for that.
"""

import pathlib
import re

import pytest

HERE = pathlib.Path(__file__).resolve().parents[2]
# walk up rather than guess a depth: this package is checked out both directly
# and as a git worktree three levels down, and a fixed ../ was wrong for one
CANDIDATES = [a / "brain_protein_screening" for a in HERE.parents]
MAIN = HERE / "manuscript" / "main.tex"


def _companion():
    for root in CANDIDATES:
        f = root / "manuscript" / "magnetobio_rsc" / "main.tex"
        if f.is_file():
            return f.read_text()
    pytest.skip("companion repository not checked out beside this one")


def _cited_title():
    m = re.search(r"\\bibitem\{Companion2026\}(.*?)\(companion manuscript",
                  MAIN.read_text(), re.S)
    assert m, "the Companion2026 bibliography entry has moved or been deleted"
    t = re.sub(r"\\textit\{|\}|\\~|\s+", " ", m.group(1))
    return " ".join(t.split())


def test_cited_title_matches_the_companion():
    """Its title changed from 'Ultrafast recombination and exchange coupling
    preclude...' to one about active-site geometry -- a different claim, not a
    rewording."""
    comp = _companion()
    m = re.search(r"\\LARGE\{\\textbf\{(.*?)\}\}", comp, re.S)
    assert m, "could not find the companion's title in its main.tex"
    real = " ".join(re.sub(r"\$\^?\\?dag\$?|\\dag|\s+", " ", m.group(1)).split())
    cited = _cited_title()
    key = real.split(",")[0].strip()          # the distinctive opening clause
    assert key.lower() in cited.lower(), (
        f"this paper cites\n  {cited}\nbut the companion is titled\n  {real}")


@pytest.mark.parametrize("value,what", [
    (r"1\.7\\times10\^\{-12\}", "computed geomagnetic effect"),
    (r"4\.0\\times10\^\{-6\}", "coherent-limit upper bound"),
])
def test_cited_numbers_exist_in_the_companion(value, what):
    """Every number this paper attributes to the companion must be findable
    there. Quoting a bound the companion has since tightened understates its
    result, which is how <1e-4 % survived here."""
    ours = MAIN.read_text()
    assert re.search(value, ours), f"this paper no longer quotes the {what}"
    assert re.search(value, _companion()), (
        f"this paper attributes the {what} to the companion, "
        f"but that value is not in the companion's main.tex")


def test_the_lifetime_not_the_coupling_is_credited():
    """The companion's own finding, and the one easiest to get backwards: at
    tau ~ 10 ps the response barely moves when the GHz exchange coupling is
    removed. A citation that credits the coupling inverts it."""
    comp = _companion()
    assert "lifetime, not the exchange coupling" in " ".join(comp.split()), (
        "the companion no longer states that the lifetime rather than the "
        "exchange coupling is decisive; re-read it before trusting this test")
    ours = " ".join(MAIN.read_text().split())
    i = ours.find("companion study from")
    assert i > 0, "the companion passage has moved"
    passage = ours[i:i + 900]
    assert "lifetime rather than the exchange coupling" in passage, (
        "this paper's companion passage no longer credits the lifetime")
