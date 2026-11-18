#!/bin/sh
# The first line of a model's answer opened by rules of their own (MLX, greedy): sh examples/rules-open/run.sh QWEN35_4B_FOLDER
#   (Qwen/Qwen3.5-4B; from a clone of the repository with mlx-lm and transformers). Writes out/ here. A few minutes on an Apple Silicon Mac.
set -e
cd "$(dirname "$0")"
M=$1; O=out; MAX=60; mkdir -p $O
V="env PYTHONPATH=../../verify python3 -m kestrom_verify"
K="env PYTHONPATH=../../verify:../../edit python3 -m kestrom"
python3 open.py prompts "$M" $O/prompts.json
$V generate mlx "$M" $O/prompts.json $O/base.jsonl $MAX
$V generate mlx "$M" $O/prompts.json $O/base2.jsonl $MAX
$K generate "$M" $O/prompts.json $O/open.jsonl --max-tokens $MAX --rules open.dsl
# tier 1: the outputs are the original's until the first token the open line replaced
$V compare events $O/base.jsonl $O/base2.jsonl $O/open.jsonl "$M" $O/open_compare.json
python3 open.py score $O/base.jsonl $O/open.jsonl
