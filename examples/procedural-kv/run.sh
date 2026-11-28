#!/usr/bin/env sh
set -eu
MODEL="${1:?usage: sh examples/procedural-kv/run.sh MODEL_FOLDER}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/out"
export PYTHONPATH="$ROOT/verify:$ROOT/edit"
mkdir -p "$OUT"

python3 -m kestrom kv compile "$MODEL" "$HERE/procedures.dsl" "$HERE/procedures.json" "$OUT/bank_init"
python3 -m kestrom kv fit-router "$MODEL" "$HERE/router_data.json" "$OUT/router.npz" auto 0.1
python3 -m kestrom kv compile "$MODEL" "$HERE/procedures.dsl" "$HERE/procedures_revised.json" "$OUT/bank_upd" "$OUT/bank_init"
python3 "$HERE/demo.py" "$MODEL" "$OUT"
python3 -m kestrom_verify compare events "$OUT/base1.jsonl" "$OUT/base2.jsonl" "$OUT/kv_routed.jsonl" "$MODEL" "$OUT/report.json"
