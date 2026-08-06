#!/usr/bin/env bash
# Regenerate every shipped simulation_results JSON from current code and prove it.
#
# The provenance disclosure in the paper used to rest on an mtime argument: most
# JSONs were older than the modules that write them, so we could only say "no
# value drift was detected". Re-running the generators and diffing replaces that
# with reproduction.
#
# The first version of this script overstated the result in two ways, both found
# by the set-7 panel:
#
#   1. It compared every file present in simulation_results/, whether or not a
#      generator had written it, and reported them all as IDENTICAL. Files that
#      no command produces were being counted as evidence of reproducibility.
#      A file now only counts as reproduced if it was actually rewritten during
#      this run, checked by mtime against a marker dropped at the start.
#   2. A failing generator printed "-> exit 1" and the script still exited 0,
#      because only the final comparison set the status. Failures are now
#      accumulated and the script exits non-zero on any of: a generator failure,
#      a byte difference, or a shipped file that no generator touched.
#
# Usage: bash scripts/regenerate_all.sh <baseline_dir>
set -u
cd "$(dirname "$0")/.."
BASE="${1:?usage: regenerate_all.sh <baseline_dir>}"

FAILED=0

run () {
  echo "### $* ###"
  python3 -m "$@" 2>&1 | tail -3
  local rc=${PIPESTATUS[0]}
  echo "  -> exit $rc"
  if [ "$rc" -ne 0 ]; then
    FAILED=$((FAILED + 1))
    echo "  !! GENERATOR FAILED: $*"
  fi
}

# Anything whose mtime is older than this marker was not written by this run.
MARKER="$(mktemp)"
sleep 1

run qbscreen.final_numbers
run qbscreen.readout_routes
run qbscreen.corrected_injection
run qbscreen.reanalysis
run qbscreen.qrc_benchmarks
run qbscreen.panel_response
run qbscreen.semiclassical
run qbscreen.semiclassical floor
run qbscreen.semiclassical conv
run qbscreen.nuclide_register
run qbscreen.ensemble_pooled
run qbscreen.general_spin
run qbscreen.relaxation_estimate
run qbscreen.turnover_estimate
run qbscreen.product_carrier_audit
# superseded single-electron-reset controls. No reported result depends on them,
# but they are shipped and frozen, so a clean-room run must be able to recreate
# them or the reproducible set is smaller than the frozen set.
run qbscreen.qrc_benchmarks tradeoff
run qbscreen.qrc_benchmarks cryptochrome
run qbscreen.ensemble
run qbscreen.quantum_vs_classical
run qbscreen.reservoir
run qbscreen.reservoir realism

echo
echo "=== byte-comparison against $BASE ==="
python3 - "$BASE" "$MARKER" <<'PY'
import hashlib, os, pathlib, sys

base, marker = pathlib.Path(sys.argv[1]), sys.argv[2]
live = pathlib.Path("simulation_results")
t0 = os.path.getmtime(marker)
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

# simulation_results/panel_before_regen/ is an archived snapshot from an earlier
# revision round. No generator writes it and it is not in FREEZE_MANIFEST, so it
# is not part of the reproducible set and must not be counted as one.
SKIP = {"panel_before_regen"}
# open5_feasible_region.json is a superseded snapshot of the feasible_region
# block inside open5_turnover_estimate.json: it predates the `feasible` field
# the current generator emits, and nothing quotes it. It is shipped for history
# and is not part of the reproducible set.
SKIP_FILES = {"panel/open5_feasible_region.json"}
def tracked(p):
    rel = p.relative_to(live)
    return not (set(rel.parts) & SKIP) and rel.as_posix() not in SKIP_FILES

b = {p.relative_to(base).as_posix(): h(p) for p in base.rglob("*.json")
     if not (set(p.relative_to(base).parts) & SKIP)
     and p.relative_to(base).as_posix() not in SKIP_FILES}
l = {p.relative_to(live).as_posix(): h(p) for p in live.rglob("*.json") if tracked(p)}
fresh = {p.relative_to(live).as_posix() for p in live.rglob("*.json")
         if tracked(p) and os.path.getmtime(p) > t0}

reproduced = sorted(k for k in b if k in l and b[k] == l[k] and k in fresh)
differ     = sorted(k for k in b if k in l and b[k] != l[k])
untouched  = sorted(k for k in b if k in l and k not in fresh)
missing    = sorted(k for k in b if k not in l)
added      = sorted(k for k in l if k not in b)
skipped    = sum(1 for p in base.rglob("*.json") if set(p.relative_to(base).parts) & SKIP)

print(f"REPRODUCED {len(reproduced)}   DIFFER {len(differ)}   "
      f"NOT_WRITTEN_THIS_RUN {len(untouched)}   MISSING {len(missing)}   NEW {len(added)}")
print(f"(excluded from the reproducible set: {skipped} archived files under "
      f"{'/'.join(SKIP)}/, which no generator writes)")
for tag, xs in (("DIFFER", differ), ("NOT_WRITTEN_THIS_RUN", untouched),
                ("MISSING", missing), ("NEW", added)):
    for x in xs:
        print(f"  {tag}: {x}")

# NEW is informational (a genuinely new output); the rest are failures.
sys.exit(1 if (differ or untouched or missing) else 0)
PY
CMP=$?
rm -f "$MARKER"

if [ "$FAILED" -ne 0 ]; then
  echo "FAIL: $FAILED generator step(s) exited non-zero"
fi
[ "$CMP" -ne 0 ] && echo "FAIL: comparison reported unreproduced files"
exit $(( FAILED != 0 || CMP != 0 ))
