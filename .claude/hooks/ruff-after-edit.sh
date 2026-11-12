#!/bin/sh
# PostToolUse (Edit|Write): ruff check the edited Python file, and show Claude what fails (exit 2).
# ruff check only: the repository does not use ruff format.
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in *.py) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0
out=$(ruff check --quiet "$f" 2>&1) || { printf '%s\n' "$out" >&2; exit 2; }
