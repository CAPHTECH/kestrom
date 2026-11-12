#!/bin/sh
# An acceptance report for a model you received: what changed against the original, and how much (tiers 2 and 3).
# Usage: sh examples/acceptance-report/run.sh ORIGINAL RECEIVED QUESTIONS.txt [DECLARED.json]
#   ORIGINAL, RECEIVED: folders of safetensors that mlx_lm loads (the same tokenizer); one question per line.
set -e
cd "$(dirname "$0")"
M=$1; E=$2; Q=$3; D=${4:-}; O=out; mkdir -p $O
# 1. every tensor and file, against the declaration if one is given (tier 2: D)
kestrom-verify diff "$M" "$E" $O/diff.json $D || true
# 2. greedy: the original twice, the received model once
python ../prompts.py "$M" "$Q" $O/prompts.json
kestrom-verify generate mlx "$M" $O/prompts.json $O/base.jsonl
kestrom-verify generate mlx "$M" $O/prompts.json $O/base2.jsonl
kestrom-verify generate mlx "$E" $O/prompts.json $O/received.jsonl
# 3. teacher forcing on the original's answers: how much the outputs moved (tier 3: counted, U0 checked)
python - $O <<'PY'
import json, sys
o = sys.argv[1]
p = {x["id"]: x["ids"] for x in json.load(open(f"{o}/prompts.json"))}
rows = [json.loads(l) for l in open(f"{o}/base.jsonl")]
json.dump([dict(id=r["id"], ids=p[r["id"]], tokens=r["tokens"]) for r in rows], open(f"{o}/inputs.json", "w"))
PY
kestrom-verify measure mlx "$M" "$E" $O/inputs.json $O/measure.json $O/base.jsonl $O/base2.jsonl $O/received.jsonl
for r in diff measure; do kestrom-verify summary $O/$r.json > $O/$r.md; done
