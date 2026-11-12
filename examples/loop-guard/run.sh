#!/bin/sh
# Thinking loops in Qwen3.5-0.8B on MATH-500: the original twice, Astra's weight edit, and the loop guard of kestrom generate (greedy, 16,384 tokens).
# Usage: sh examples/loop-guard/run.sh QWEN35_08B_FOLDER ASTRA_FOLDER MATH500_TEST.jsonl
#   (Qwen/Qwen3.5-0.8B, RikiyaT/Qwen3.5-0.8B-weight-space-programming, HuggingFaceH4/MATH-500 test.jsonl; pip install math-verify)
# About 10 hours on an Apple Silicon Mac.
set -e
cd "$(dirname "$0")"
M=$1; A=$2; DATA=$3; O=out; MAX=16384; mkdir -p $O
python math500.py prompts "$M" "$DATA" $O/prompts.json 100
kestrom-verify generate mlx "$M" $O/prompts.json $O/base.jsonl $MAX
kestrom-verify generate mlx "$M" $O/prompts.json $O/base2.jsonl $MAX
kestrom-verify generate mlx "$A" $O/prompts.json $O/astra.jsonl $MAX
kestrom generate "$M" $O/prompts.json $O/guard.jsonl --max-tokens $MAX --rules guard.dsl
# tier 1: the guard's outputs are the original's until the first loop
kestrom-verify compare events $O/base.jsonl $O/base2.jsonl $O/guard.jsonl "$M" $O/guard_compare.json
# tier 2: what Astra's checkpoint changed, against a declaration of its two tensors
echo '{"model.language_model.layers.15.mlp.down_proj.weight": "all", "model.language_model.layers.19.mlp.down_proj.weight": "all"}' > $O/astra_declared.json
kestrom-verify diff "$M" "$A" $O/astra_diff.json $O/astra_declared.json || true
python math500.py score $O/base.jsonl $O/base2.jsonl $O/astra.jsonl $O/guard.jsonl
