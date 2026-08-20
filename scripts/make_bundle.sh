#!/usr/bin/env bash
# Build the PCCP submission bundle in manuscript/submission/.
#
# The bundle is a COPY of files that are checked elsewhere, which is exactly how
# a bundle goes stale: someone edits main.tex, forgets to re-run this, and ships
# last week's PDF. Two things prevent that here.
#
#   1. It refuses to build unless scripts/release_check.sh passes, so a bundle
#      can never be made from a tree whose tests, counts or PDFs disagree.
#   2. After copying, every file the freeze manifest covers is re-hashed IN THE
#      BUNDLE and compared to the manifest. A copy that drifted from its source
#      -- or a source that drifted from the frozen state -- fails here.
#
# The bundle itself is not committed: it is derivable, and a committed copy is
# one more thing to keep in sync. Run this immediately before uploading.
set -u
cd "$(dirname "$0")/.."
OUT=manuscript/submission
TAR=manuscript/pccp_submission.tar.gz

echo "=== gate: release_check ==="
if ! bash scripts/release_check.sh; then
  echo
  echo "REFUSING to build a bundle from a tree that does not pass. Fix, or run"
  echo "  bash scripts/release_check.sh --fix"
  exit 1
fi

echo
echo "=== toc entry (regenerated from simulation_results) ==="
python3 manuscript/make_toc_entry.py || exit 1
python3 - <<'PY' || exit 1
import sys
n = len(open("manuscript/toc_entry.txt", encoding="utf-8").read())
print(f"toc_entry.txt: {n} characters")
if n > 250:
    sys.exit("PCCP allows at most 250 characters in the TOC entry text")
PY

rm -rf "$OUT"
mkdir -p "$OUT/figures"
for f in main supplementary cover_letter data_availability; do
  cp -f "manuscript/$f.tex" "manuscript/$f.pdf" "$OUT/"
done
cp -f manuscript/toc_entry.pdf manuscript/toc_entry.tif manuscript/toc_entry.txt "$OUT/"
cp -f manuscript/SUBMISSION_CHECKLIST.md "$OUT/"
# only the figures main.tex actually includes; figures/ also holds two outputs
# that no document uses, and shipping those invites the "which figure is this?"
# query at proof stage.
for g in $(grep -o 'includegraphics\[[^]]*\]{[^}]*}' manuscript/main.tex |
           sed 's/.*{//;s/}//' | sort -u); do
  cp -f "manuscript/figures/$g" "$OUT/figures/$g" || exit 1
done
cp -f FREEZE_MANIFEST.txt "$OUT/"

echo
echo "=== bundle vs. freeze manifest ==="
python3 - "$OUT" <<'PY'
import hashlib, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
man = {}
for line in open("FREEZE_MANIFEST.txt"):
    m = re.match(r"^([0-9a-f]{64})  (.+)$", line.rstrip("\n"))
    if m:
        man[m.group(2)] = m.group(1)

checked = unbound = bad = 0
for p in sorted(out.rglob("*")):
    if not p.is_file():
        continue
    rel = p.relative_to(out).as_posix()
    # the bundle flattens manuscript/ to the top level
    for cand in (f"manuscript/{rel}", rel):
        if cand in man:
            checked += 1
            if hashlib.sha256(p.read_bytes()).hexdigest() != man[cand]:
                bad += 1
                print(f"  DRIFTED {rel} (bundle copy differs from the frozen {cand})")
            break
    else:
        unbound += 1
        print(f"  not frozen: {rel}")
print(f"{checked - bad}/{checked} bundle files match the manifest; "
      f"{unbound} not covered by it")
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && { echo "BUNDLE FAILED"; exit 1; }

tar -czf "$TAR" -C manuscript submission
echo
echo "wrote $OUT/ and $TAR"
ls -1 "$OUT" "$OUT/figures" | sed 's/^/  /'
