#!/usr/bin/env sh
set -eu
MODEL="${1:?usage: sh examples/ifc-control/run.sh MODEL_FOLDER [mlx|torch]}"
RUNTIME="${2:-mlx}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/out"
export PYTHONPATH="$ROOT/verify:$ROOT/edit"

python3 "$HERE/prompts.py" "$MODEL" "$OUT"
python3 -m kestrom_verify generate "$RUNTIME" "$MODEL" "$OUT/prompts_clean.json" "$OUT/clean1.jsonl" 192
python3 -m kestrom_verify generate "$RUNTIME" "$MODEL" "$OUT/prompts_clean.json" "$OUT/clean2.jsonl" 192
python3 -m kestrom_verify generate "$RUNTIME" "$MODEL" "$OUT/prompts_full.json" "$OUT/base.jsonl" 192
python3 -m kestrom ifc control "$MODEL" "$OUT/prompts_ifc.json" "$OUT/ifc.jsonl" tool,recipient,approved summary 192 fast none "$RUNTIME"
python3 -m kestrom_verify compare events "$OUT/clean1.jsonl" "$OUT/clean2.jsonl" "$OUT/ifc.jsonl" "$MODEL" "$OUT/report.json"
