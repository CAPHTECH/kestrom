#!/bin/sh
# Ban 25,451 Chinese-only tokens in Qwen3.5-9B's weights, then check the edit (tier 1).
# Usage: sh examples/zh-ban/run.sh QWEN35_9B_FOLDER QUESTIONS.txt   (a Hugging Face folder of Qwen/Qwen3.5-9B; one question per line)
set -e
cd "$(dirname "$0")"
M=$1; Q=$2; O=out; mkdir -p $O
# 1. the edit: the listed rows of the output layer set to row.npy; every other file is linked, not copied
[ -d $O/edited ] || kestrom rows ban "$M" "$M" list.txt row.npy $O/edited
# 2. the range from the bytes (W)
kestrom-verify range "$M" $O/edited $O/range.json
# 3. greedy: the original twice, the edited once; G2 against the range
python ../prompts.py "$M" "$Q" $O/prompts.json
kestrom-verify generate mlx "$M" $O/prompts.json $O/base.jsonl
kestrom-verify generate mlx "$M" $O/prompts.json $O/base2.jsonl
kestrom-verify generate mlx $O/edited $O/prompts.json $O/edited.jsonl
kestrom-verify compare $O/range.json $O/base.jsonl $O/base2.jsonl $O/edited.jsonl "$M" $O/greedy.json
# 4. every logit at every position (G1): the original's own answers as the sequences
python - $O <<'PY'
import json, sys
o = sys.argv[1]
p = {x["id"]: x["ids"] for x in json.load(open(f"{o}/prompts.json"))}
rows = [json.loads(l) for l in open(f"{o}/base.jsonl")]
json.dump([dict(id=r["id"], ids=p[r["id"]], tokens=r["tokens"]) for r in rows], open(f"{o}/inputs.json", "w"))
PY
kestrom-verify logits "$M" $O/edited $O/inputs.json $O/logits.json
for r in range greedy logits; do kestrom-verify summary $O/$r.json > $O/$r.md; done
