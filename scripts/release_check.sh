#!/usr/bin/env bash
# Verify that the package is internally consistent. READ-ONLY by default.
#
# The first version of this script ran freeze.py and sync_test_counts.py before
# checking anything, so it repaired the drift and then reported success. A
# negative control proved it: editing a test count in main.tex without
# recompiling still produced "RELEASE CHECK PASSED", because the sync had
# already put the number back. A gate that fixes what it is meant to detect
# cannot fail, and a gate that cannot fail certifies nothing.
#
# So verification writes nothing. Repair is a separate, explicit request.
#
#   bash scripts/release_check.sh          verify only; non-zero if inconsistent
#   bash scripts/release_check.sh --fix    reach a fixed point first, then verify
#
# --fix is a fixed-point iteration rather than a pipeline: freeze.py changes the
# manifest's file count, sync_test_counts.py writes that count into
# claims-ledger.md, and claims-ledger.md is itself frozen. One pass in the wrong
# order leaves --check failing.
set -u
cd "$(dirname "$0")/.."

FIX=0
[ "${1:-}" = "--fix" ] && FIX=1

if [ "$FIX" -eq 1 ]; then
  for i in 1 2 3 4; do
    python3 scripts/freeze.py > /dev/null 2>&1
    python3 scripts/sync_test_counts.py > /dev/null 2>&1
    python3 scripts/freeze.py > /dev/null 2>&1
    if python3 scripts/sync_test_counts.py --check > /dev/null 2>&1; then
      break
    fi
    if [ "$i" -eq 4 ]; then
      echo "FAIL: counts and manifest did not reach a fixed point in 4 rounds"
      python3 scripts/sync_test_counts.py --check
      exit 1
    fi
  done
  # Syncing the .tex without rebuilding leaves the PDFs quoting the old count,
  # which test_pdf_counts.py then (correctly) fails on. A --fix that produces a
  # state its own verification rejects is worse than no --fix at all.
  if command -v pdflatex > /dev/null 2>&1 || [ -x /Library/TeX/texbin/pdflatex ]; then
    latex=$(command -v pdflatex || echo /Library/TeX/texbin/pdflatex)
    for d in main supplementary cover_letter data_availability; do
      for _ in 1 2 3; do
        (cd manuscript && "$latex" -interaction=nonstopmode "$d.tex" > /dev/null 2>&1)
      done
    done
    # the PDFs are frozen, so the manifest must be taken again after rebuilding
    python3 scripts/freeze.py > /dev/null 2>&1
    python3 scripts/sync_test_counts.py > /dev/null 2>&1
    python3 scripts/freeze.py > /dev/null 2>&1
  else
    echo "(pdflatex not found: PDFs not rebuilt)"
  fi

  echo "(repaired to a fixed point; verifying)"
  echo
fi

fail=0

python3 -m pytest qbscreen/tests/ -q > /tmp/_rc_pytest.txt 2>&1
rc=$?
echo "tests:    $(tail -1 /tmp/_rc_pytest.txt)  (exit $rc)"
[ "$rc" -ne 0 ] && fail=1

python3 scripts/sync_test_counts.py --check > /tmp/_rc_counts.txt 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "counts:   all documents agree with the measured values"
else
  echo "counts:   STALE"
  grep STALE /tmp/_rc_counts.txt | sed 's/^/  /'
  fail=1
fi

python3 - <<'PY'
import hashlib, pathlib, re, sys
n = bad = undistributed = 0
for line in open("FREEZE_MANIFEST.txt"):
    m = re.match(r"^([0-9a-f]{64})  (.+)$", line.rstrip("\n"))
    if not m:
        continue
    n += 1
    p = pathlib.Path(m.group(2))
    rel = m.group(2)
    # the manuscript sources are frozen but not distributed (see .gitignore);
    # on a clone they are absent, which is not a mismatch
    if (not p.exists() and rel.startswith("manuscript/")
            and not rel.startswith("manuscript/figures/") and "/make_" not in rel):
        undistributed += 1
        continue
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != m.group(1):
        bad += 1
        print(f"  MISMATCH {rel}")
print(f"manifest: {n - bad - undistributed}/{n - undistributed} match"
      + (f" ({undistributed} manuscript files not distributed here)" if undistributed else ""))
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && fail=1

# The manifest must be byte-reproducible or "frozen" means nothing. Checked in a
# scratch copy so that verification stays read-only.
tmp=$(mktemp -d)
cp FREEZE_MANIFEST.txt "$tmp/before.txt"
python3 scripts/freeze.py > /dev/null 2>&1
if cmp -s "$tmp/before.txt" FREEZE_MANIFEST.txt; then
  echo "manifest: byte-reproducible across runs"
else
  echo "manifest: NOT byte-reproducible"
  diff "$tmp/before.txt" FREEZE_MANIFEST.txt | head -5
  cp "$tmp/before.txt" FREEZE_MANIFEST.txt      # leave the tree as we found it
  fail=1
fi
rm -rf "$tmp"

# The overfull/undefined counts come from LaTeX's .log, which is gitignored
# (it embeds absolute paths). So it may be absent on a fresh clone, or -- as
# happened when this branch was merged into main -- three days older than the
# .tex beside it, in which case its numbers describe a document that no longer
# exists. Trusting it blindly reported overfull boxes that had been fixed two
# rounds earlier. A log that cannot be shown to match its source is treated as
# no evidence, not as evidence of failure.
for d in main supplementary cover_letter data_availability; do
  log="manuscript/$d.log"
  if [ ! -f "$log" ]; then
    echo "$d: no log -- run with --fix to compile, then re-verify"
    fail=1
    continue
  fi
  if [ "manuscript/$d.tex" -nt "$log" ]; then
    echo "$d: log is older than the .tex -- stale, cannot verify; run with --fix"
    fail=1
    continue
  fi
  o=$(grep -c Overfull "$log")
  u=$(grep -c 'Reference .* undefined\|Citation .* undefined' "$log")
  q=$(pdftotext "manuscript/$d.pdf" - 2>/dev/null | grep -c '??')
  printf "%-18s overfull=%-2s undefined=%-2s qmark=%s\n" "$d:" "$o" "$u" "$q"
  if [ "$o" != "0" ] || [ "$u" != "0" ] || [ "$q" != "0" ]; then
    fail=1
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "RELEASE CHECK PASSED"
else
  echo "RELEASE CHECK FAILED"
fi
exit "$fail"
