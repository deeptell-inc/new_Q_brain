"""Guard the hand-written cross-document table references.

main.tex and data_availability.tex cannot \\ref{} a label that lives in
supplementary.tex -- they are separate LaTeX documents -- so those references are
typed by hand as literal "Table~S7". LaTeX therefore cannot catch them when the
supplement is renumbered, and it has not: the set-7 audit found two wrong at
once. "Table~S17" for the hyperfine assignment broke when a new table was
inserted ahead of it, and "Table~S8" for the superseded clock scan had been
wrong since before that, pointing one past tab:clock.

Each entry below names the label the sentence *means*. The number is read from
supplementary.aux, which is what LaTeX actually assigned, so inserting or
deleting any supplementary table makes the affected test fail rather than
silently mis-citing.
"""

import pathlib
import re

import pytest
from qbscreen.tests._manuscript import requires_manuscript

pytestmark = requires_manuscript

ROOT = pathlib.Path(__file__).resolve().parents[2] / "manuscript"
AUX = ROOT / "supplementary.aux"

# (document, regex with the number as group 1, label the sentence refers to)
REFS = [
    ("main.tex",
     r"\(Supplemental Material, Table~S(\d+)\)",
     "tab:params"),
    ("main.tex",
     r"retained as Supplemental Material, Table~S(\d+), whose",
     "tab:clock"),
    ("main.tex",
     r"routes move as Table~S(\d+)",
     "tab:premise_routes"),
    ("data_availability.tex",
     r"single-realisation clock scan of Supplemental Material Table~S(\d+) are",
     "tab:clock"),
    # README uses a plain space, not a tie -- which is exactly why the first
    # version of this guard missed it while all 200 tests passed
    ("../README.md",
     r"single-realisation clock scan kept in Supplemental Material Table S(\d+)",
     "tab:clock"),
]


def _assigned():
    """label -> number, straight from LaTeX's own .aux."""
    # shipped in the freeze manifest, so absence is a packaging fault, not a
    # reason to pass quietly
    assert AUX.exists(), (
        "supplementary.aux is missing; it is a tracked test input carrying the "
        "table numbers LaTeX assigned. Recompile the supplement.")
    out = {}
    for m in re.finditer(r"\\newlabel\{(tab:[^}]+)\}\{\{S?(\d+)\}", AUX.read_text()):
        out[m.group(1)] = int(m.group(2))
    return out


@pytest.mark.parametrize("doc,pattern,label", REFS)
def test_cross_document_table_reference(doc, pattern, label):
    nums = _assigned()
    assert label in nums, f"{label} has no \\newlabel in supplementary.aux"
    t = (ROOT / doc).read_text()
    m = re.search(pattern, t)
    assert m, f"{doc}: the sentence citing {label} no longer matches {pattern!r}"
    assert int(m.group(1)) == nums[label], (
        f"{doc} cites Table~S{m.group(1)} for {label}, "
        f"which LaTeX numbered S{nums[label]}")


def test_every_hardcoded_supplement_reference_is_covered():
    """A new hand-typed 'Table~SN' that nobody registered above would be
    unguarded, which is how both existing errors survived. Fail until it is
    added to REFS."""
    known = {(d, p) for d, p, _ in REFS}
    for doc in ("main.tex", "data_availability.tex", "cover_letter.tex", "../README.md"):
        t = (ROOT / doc).read_text()
        # tie OR plain space: the tie-only pattern is what let README drift
        found = re.findall(r"Table[~ ]S\d+", t)
        covered = sum(1 for d, p, _ in REFS
                      if d == doc and re.search(p, t))
        assert len(found) == covered, (
            f"{doc} has {len(found)} hardcoded supplement table references "
            f"({found}) but only {covered} are guarded in REFS")


def test_see_supplemental_material_sentence_names_every_section():
    """APS asks for a 'See Supplemental Material at [URL] for ...' sentence. It is
    the one place the paper announces what the supplement contains, so each
    \section of supplementary.tex must leave a distinctive word in it; adding a
    section without announcing it fails here.
    """
    see = re.search(r"See Supplemental Material at \[URL[^\]]*\] for (.*?)\.\n",
                    (ROOT / "main.tex").read_text(), re.S)
    assert see, "the 'See Supplemental Material' sentence is missing from main.tex"
    text = see.group(1).lower()
    missing = []
    for title in re.findall(r"^\\section\{(.+?)\}",
                            (ROOT / "supplementary.tex").read_text(), re.M):
        words = [w for w in re.findall(r"[a-z-]{6,}", title.lower())]
        assert words, f"section {title!r} has no word long enough to key on"
        if not any(w in text for w in words):
            missing.append((title, words))
    assert not missing, (
        "Supplemental Material sections the sentence does not mention: "
        + "; ".join(f"{t} (none of {w})" for t, w in missing))
