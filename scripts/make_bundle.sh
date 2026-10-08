#!/usr/bin/env bash
# Build the Physical Review E submission bundle in manuscript/submission/.
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
TAR=manuscript/pre_submission.tar.gz

echo "=== gate: release_check ==="
if ! bash scripts/release_check.sh; then
  echo
  echo "REFUSING to build a bundle from a tree that does not pass. Fix, or run"
  echo "  bash scripts/release_check.sh --fix"
  rm -rf "$OUT" "$TAR"     # and do not leave an older bundle behind to be uploaded by mistake
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT/figures"
# APS wants the REVTeX source, the figures as separate files, the compiled
# manuscript, the Supplemental Material as a separate PDF, and the cover letter.
for f in main supplementary cover_letter data_availability; do
  cp -f "manuscript/$f.tex" "manuscript/$f.pdf" "$OUT/"
done
cp -f manuscript/SUBMISSION_CHECKLIST.md "$OUT/"
# only the figures main.tex actually includes
for g in $(grep -o 'includegraphics\[[^]]*\]{[^}]*}' manuscript/main.tex |
           sed 's/.*{//;s/}//' | sort -u); do
  cp -f "manuscript/figures/$g" "$OUT/figures/$g" || exit 1
done
cp -f FREEZE_MANIFEST.txt "$OUT/"

echo
echo "=== bundle vs. freeze manifest ==="
python3 - "$OUT" <<'PYEOF'
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
print(f"{checked - bad}/{checked} bundle files match the manifest; {unbound} not covered by it")
sys.exit(1 if bad else 0)
PYEOF
[ $? -ne 0 ] && { echo "BUNDLE FAILED"; exit 1; }

tar -czf "$TAR" -C manuscript submission
echo
echo "wrote $OUT/ and $TAR"
ls -1 "$OUT" "$OUT/figures" | sed 's/^/  /'
