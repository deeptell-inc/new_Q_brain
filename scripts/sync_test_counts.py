#!/usr/bin/env python3
"""Keep every asserted count in the manuscript equal to the measured value.

The suite has been quoted as 54, 115, 146, 174, 186, 200, 202 and 208 across
eight revision rounds, and every one of those numbers went stale somewhere
before anyone noticed. Manifest sizes drift the same way -- the header once
claimed 86 files over a body of 82.

This is the third design. The first two failed audits for reasons worth keeping
in view, because they are properties of the *approach*, not typos:

  * A per-site regex that stops at the first group left the second number of
    "(208/208 tests pass)" unchecked, so "(208/174 tests pass)" passed clean --
    the very bug the comment above that site claimed to prevent.
  * Spelled-out numbers were matched against a hardcoded dictionary
    ({26: "Twenty-six", ...}). Anything outside the table was skipped in
    silence, so a suite of 31 tests updated the totals and left the parts
    contradicting them, with --check still exiting 0.
  * A missing anchor was treated as "nothing to do" rather than as a claim that
    had been deleted or reworded.

So: numbers are spelled by an algorithm rather than looked up; EVERY numeric
group in a pattern is compared, not just the first; every match is compared, not
just the first; and a pattern that fails to match is an error, never a skip.
"""

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def spell(n):
    """0-999 in the manuscript's style: 'one hundred and twenty-eight'.

    Computed, not tabulated -- a lookup table is what let an out-of-range count
    pass unnoticed.
    """
    if not 0 <= n <= 999:
        raise ValueError(f"cannot spell {n}")
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + (f"-{_ONES[r]}" if r else "")
    h, r = divmod(n, 100)
    return _ONES[h] + " hundred" + (f" and {spell(r)}" if r else "")


def counts():
    """(total, per-file) from a real collection, not from a docstring."""
    r = subprocess.run([sys.executable, "-m", "pytest", "qbscreen/tests/",
                        "-q", "--collect-only"], cwd=ROOT,
                       capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if not m:
        sys.exit(f"could not parse collection output:\n{r.stdout[-2000:]}")
    per = {}
    for f in sorted((ROOT / "qbscreen" / "tests").glob("test_*.py")):
        rr = subprocess.run([sys.executable, "-m", "pytest", str(f), "-q",
                             "--collect-only"], cwd=ROOT,
                            capture_output=True, text=True)
        mm = re.search(r"(\d+) tests? collected", rr.stdout)
        per[f.name] = int(mm.group(1)) if mm else 0
    return int(m.group(1)), per


def manifest_files():
    """How many files FREEZE_MANIFEST.txt actually hashes, from the SHA lines.

    Never from the header: a header disagreeing with its own body is the
    original instance of this whole class of bug.
    """
    return sum(1 for line in (ROOT / "FREEZE_MANIFEST.txt").read_text().splitlines()
               if re.match(r"^[0-9a-f]{64}  ", line))


def manifest_header_claim():
    """What the manifest header says its size is, so the two can be compared."""
    m = re.search(r"^# scope: (\d+) files", (ROOT / "FREEZE_MANIFEST.txt").read_text(),
                  re.M)
    return int(m.group(1)) if m else None


# Numeric sites. EVERY capturing group in the pattern must equal the quantity,
# and EVERY occurrence of the pattern is checked.
NUMERIC = [
    ("manuscript/cover_letter.tex",      r"automated test suite \((\d+) tests, all passing\)", "cover letter: total", "tests"),
    ("manuscript/data_availability.tex", r"\\texttt\{qbscreen/tests/\}; (\d+) tests, all passing", "DAS: total", "tests"),
    ("manuscript/main.tex",              r"\((\d+)/(\d+) tests pass\)", "main: N/N pass", "tests"),
    ("manuscript/main.tex",              r"\((\d+) tests, all passing\)", "main: total", "tests"),
    ("README.md",                        r"tests/ +(\d+) tests:", "README: total", "tests"),
    ("manuscript/claims-ledger.md",      r"試験: (\d+) passed", "ledger: total", "tests"),
    ("manuscript/claims-ledger.md",      r"`FREEZE_MANIFEST\.txt`（(\d+)ファイル", "ledger: manifest size", "manifest"),
]

# Spelled-out sites. The text captured between the anchors must be EXACTLY the
# spelling of the quantity -- which rejects "twenty-eight-eight" and anything
# else the previous prefix-matching version accepted.
SPELLED = [
    ("manuscript/main.tex",
     r"(?<=passing\)\. )([a-z-]+(?: hundred(?: and [a-z-]+)?)?)(?= tests in \\texttt\{test\\_panel\\_claims\.py\})",
     "main: panel_claims", "test_panel_claims.py", True),
    ("manuscript/main.tex",
     r"(?<=a further )([a-z-]+(?: hundred(?: and [a-z-]+)?)?)(?= in\n\\texttt\{test\\_table\\_rows\.py\})",
     "main: table_rows", "test_table_rows.py", False),
    ("manuscript/cover_letter.tex",
     r"(?<=passing\): )([a-z-]+(?: hundred(?: and [a-z-]+)?)?)(?= bind the)",
     "cover letter: panel_claims", "test_panel_claims.py", False),
    ("manuscript/cover_letter.tex",
     r"(?<=headline quantities and )([a-z-]+(?: hundred(?: and [a-z-]+)?)?)(?= bind the capacity)",
     "cover letter: table_rows", "test_table_rows.py", False),
    ("manuscript/cover_letter.tex",
     r"(?<=A further )([a-z-]+(?: hundred(?: and [a-z-]+)?)?)(?= parse the)",
     "cover letter: printed_cells", "test_printed_cells.py", False),
    ("manuscript/cover_letter.tex",
     r"(?<=moves, and )([a-z-]+(?: hundred(?: and [a-z-]+)?)?)(?= bind the input-parameter)",
     "cover letter: spin_parameters", "test_spin_parameters.py", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report staleness and exit 1; do not edit")
    a = ap.parse_args()

    total, per = counts()
    n_manifest = manifest_files()
    print(f"  collected: {total} tests    manifest: {n_manifest} files")
    for k, v in sorted(per.items()):
        print(f"    {k}: {v}")

    want = {"tests": total, "manifest": n_manifest}
    stale, fixed = [], []

    # the manifest must agree with itself
    claimed = manifest_header_claim()
    if claimed is None:
        stale.append("FREEZE_MANIFEST.txt: no '# scope: N files' header")
    elif claimed != n_manifest:
        stale.append(f"FREEZE_MANIFEST.txt header says {claimed} files, body has {n_manifest}")

    for rel, pat, label, kind in NUMERIC:
        p = ROOT / rel
        t = p.read_text()
        target = want[kind]
        ms = list(re.finditer(pat, t))
        if not ms:
            stale.append(f"{label}: pattern not found in {rel} -- claim moved or was deleted")
            continue
        # right-to-left over every occurrence and every group, so no span shifts
        edits = [(s, e) for m in ms for (s, e) in
                 (m.span(i) for i in range(1, (m.re.groups or 0) + 1))
                 if int(t[s:e]) != target]
        if not edits:
            continue
        stale.append(f"{label}: {rel} has {len(edits)} number(s) != {target}")
        if not a.check:
            for s, e in sorted(edits, reverse=True):
                t = t[:s] + str(target) + t[e:]
            p.write_text(t)
            fixed.append(label)

    for rel, pat, label, key, capitalise in SPELLED:
        p = ROOT / rel
        t = p.read_text()
        n = per.get(key)
        if n is None:
            stale.append(f"{label}: {key} no longer exists")
            continue
        word = spell(n)
        if capitalise:
            word = word[0].upper() + word[1:]
        ms = list(re.finditer(pat, t, re.I if capitalise else 0))
        if not ms:
            stale.append(f"{label}: anchor not found in {rel} -- claim moved or was reworded")
            continue
        bad = [m for m in ms if m.group(1) != word]
        if not bad:
            continue
        stale.append(f"{label}: {rel} spells '{bad[0].group(1)}' for {key}, actual {n} ({word})")
        if not a.check:
            for m in sorted(bad, key=lambda m: m.start(), reverse=True):
                t = t[:m.start(1)] + word + t[m.end(1):]
            p.write_text(t)
            fixed.append(label)

    if a.check:
        for s in stale:
            print(f"  STALE {s}")
        print(f"  {len(stale)} stale claim(s)")
        return 1 if stale else 0

    for s in stale:
        print(f"  was stale: {s}")
    # a claim that cannot be located cannot be repaired -- say so and fail,
    # rather than reporting success over a document that lost the sentence
    unfixable = [s for s in stale if "not found" in s or "no longer exists" in s
                 or "header says" in s]
    print(f"  updated {len(fixed)} site(s)" if fixed else "  all sites already correct")
    if unfixable:
        for s in unfixable:
            print(f"  UNFIXABLE {s}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
