#!/bin/sh
# Stop: when the tree has Python changes, check the license boundary and run the tests before the turn ends (exit 2 sends the
# failure back to Claude). Once only: when this hook already kept the turn going, the turn may end, and the failure is to be
# reported.
input=$(cat)
[ "$(printf '%s' "$input" | jq -r '.stop_hook_active')" = true ] && exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
git status --porcelain -- '*.py' | grep -q . || exit 0
if grep -rnE '^[[:space:]]*(from|import)[[:space:]]+kestrom([.[:space:]]|$)' verify/kestrom_verify >&2; then
  echo 'verify/ (kestrom-verify, Apache-2.0) must not import kestrom (PolyForm).' >&2
  exit 2
fi
out=$(python3 -m pytest -q 2>&1) || { printf '%s\n' "$out" | tail -40 >&2; exit 2; }
