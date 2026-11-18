#!/bin/sh
# No model: sh examples/rules-run/run.sh (from a clone of the repository; numpy only). Writes out/ here.
set -e
cd "$(dirname "$0")"
mkdir -p out
K="env PYTHONPATH=../../verify:../../edit python3 -m kestrom"
$K rules run router.dsl router.json out/router.jsonl
$K rules run gcd.dsl gcd.json out/gcd.jsonl
$K rules build gcd.dsl out/gcd.npz
