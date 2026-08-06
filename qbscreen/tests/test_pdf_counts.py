"""Check the counts printed in the compiled PDFs, not just in the .tex.

scripts/sync_test_counts.py rewrites .tex files. It cannot rewrite a PDF. So
after a sync, the sources are correct and the shipped PDFs still carry the old
number until someone remembers to recompile -- and nothing detected that gap:
every other guard in this package reads .tex.

The PDFs are what a referee actually receives, so they are checked directly
here, against the measured test count rather than against the .tex. That way a
stale PDF fails even if the source it was built from has since been corrected.

Compares by content, not by mtime: a fresh clone gives every file the same
checkout time, so an mtime comparison would be meaningless off this machine.
"""

import functools
import pathlib
import re
import shutil
import subprocess

import pytest

MAN = pathlib.Path(__file__).resolve().parents[2] / "manuscript"

pytestmark = pytest.mark.skipif(shutil.which("pdftotext") is None,
                                reason="pdftotext (poppler) not installed")


@functools.lru_cache(maxsize=1)
def _collected():
    """Total and per-file test counts, measured the same way the sync does."""
    import sys
    r = subprocess.run([sys.executable, "-m", "pytest", "qbscreen/tests/", "-q",
                        "--collect-only"], cwd=MAN.parent,
                       capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    assert m, "could not measure the collected test count"
    per = {}
    for f in sorted((MAN.parent / "qbscreen" / "tests").glob("test_*.py")):
        rr = subprocess.run([sys.executable, "-m", "pytest", str(f), "-q",
                             "--collect-only"], cwd=MAN.parent,
                            capture_output=True, text=True)
        mm = re.search(r"(\d+) tests? collected", rr.stdout)
        per[f.name] = int(mm.group(1)) if mm else 0
    return int(m.group(1)), tuple(sorted(per.items()))


@functools.lru_cache(maxsize=8)
def _text(stem):
    pdf = MAN / f"{stem}.pdf"
    assert pdf.exists(), f"{pdf} has not been compiled"
    out = subprocess.run(["pdftotext", str(pdf), "-"],
                         capture_output=True, text=True)
    assert out.returncode == 0, f"pdftotext failed on {pdf}"
    # PDFs wrap lines, so collapse whitespace before matching prose
    return " ".join(out.stdout.split())


@pytest.mark.parametrize("stem,pattern", [
    ("main", r"\((\d+) tests, all passing\)"),
    ("main", r"\((\d+)/(\d+) tests pass\)"),
    ("cover_letter", r"automated test suite \((\d+) tests, all passing\)"),
    ("data_availability", r"qbscreen/tests/; (\d+) tests, all passing"),
])
def test_pdf_states_the_measured_test_count(stem, pattern):
    total, _ = _collected()
    t = _text(stem)
    ms = list(re.finditer(pattern, t))
    assert ms, (f"{stem}.pdf does not contain the claim {pattern!r} -- either the "
                f"sentence changed or the PDF is from an older source")
    for m in ms:
        for g in m.groups():
            assert int(g) == total, (
                f"{stem}.pdf prints {g} where the suite collects {total}. "
                f"The .tex may already be correct: recompile before shipping.")


@pytest.mark.parametrize("stem,before,after,key", [
    # pdftotext renders \texttt{test\_panel\_claims.py} with the underscores
    # dropped to spaces, so the anchor stops before the filename
    ("main", "passing). ", " tests in test", "test_panel_claims.py"),
    ("cover_letter", "passing): ", " bind the headline", "test_panel_claims.py"),
    ("cover_letter", "A further ", " parse the printed", "test_printed_cells.py"),
])
def test_pdf_spells_the_measured_per_file_count(stem, before, after, key):
    """The spelled-out counts drifted in the .tex twice; the PDF can drift the
    same way and be the only copy a referee sees."""
    import sys
    sys.path.insert(0, str(MAN.parent / "scripts"))
    from sync_test_counts import spell

    _, per = _collected()
    want = spell(dict(per)[key])
    t = _text(stem)
    m = re.search(re.escape(before) + r"([a-z-]+(?: hundred(?: and [a-z-]+)?)?)"
                  + re.escape(after), t, re.I)
    assert m, f"{stem}.pdf: anchor {before!r}...{after!r} not found"
    assert m.group(1).lower() == want, (
        f"{stem}.pdf spells '{m.group(1)}' for {key}, which now has {per[key]} "
        f"({want}). Recompile.")


def test_every_compiled_pdf_is_newer_than_nothing_stale_in_its_source():
    """Cross-check: whatever the PDFs say, the .tex must agree with them.

    If this fails while the tests above pass, the .tex was edited and the PDF
    was not rebuilt -- the exact gap this module exists to close.
    """
    total, _ = _collected()
    for stem, pat in (("main", r"\((\d+) tests, all passing\)"),
                      ("cover_letter", r"automated test suite \((\d+) tests, all passing\)")):
        src = (MAN / f"{stem}.tex").read_text()
        pdf = _text(stem)
        s = re.search(pat, src)
        p = re.search(pat, pdf)
        assert s and p, f"{stem}: claim missing from source or PDF"
        assert s.group(1) == p.group(1) == str(total), (
            f"{stem}: .tex says {s.group(1)}, PDF says {p.group(1)}, "
            f"measured {total}")
