#!/usr/bin/env bash
# git for the manuscript, which has its own private history.
#
# The public repository stops at manuscript/{main,supplementary,data_availability}.pdf,
# figures/ and make_*.py; the LaTeX sources, cover letter, response, TOC entry and
# process logs are ignored there. They are version-controlled HERE instead: a
# second git directory (.manuscript.git beside the main .git, shared by every
# worktree) whose work tree is this checkout's manuscript/. No .git file is
# placed inside manuscript/, so the public repository never notices.
#
#   bash scripts/mgit.sh status
#   bash scripts/mgit.sh add -A && bash scripts/mgit.sh commit -m "..."
#   bash scripts/mgit.sh push          (to the private remote, once one is set)
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
COMMON="$(git -C "$HERE" rev-parse --git-common-dir)"
[ "${COMMON#/}" = "$COMMON" ] && COMMON="$HERE/$COMMON"
exec git --git-dir="$(cd "$COMMON/.." && pwd)/.manuscript.git" --work-tree="$HERE/manuscript" "$@"
