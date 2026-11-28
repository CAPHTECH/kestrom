# Non-interfering procedural memory KV (`kestrom kv` + `fit-router`)

`kestrom kv` compiles executable multi-step procedures (algorithms, normalization rules, triage decision trees) into prefix KV caches just like declarative facts—with zero weight changes, 1-forward `O(K)` dual-Ridge semantic routing (`kestrom kv fit-router`), `PREV_DIR` incremental hot-patching, and Tier 1 bit-identical non-interference (`kestrom-verify compare events`) on outside prompts.

```sh
sh examples/procedural-kv/run.sh GEMMA4_E4B_FOLDER   # google/gemma-4-E4B-it, MLX (~1 minute)
```

What the script runs:

1. `kestrom kv compile` compiles four multi-step procedures (`procedures.dsl` + `procedures.json`) into `out/bank_init`.
2. `kestrom kv fit-router` fits a 1-forward `O(K)` dual-Ridge semantic router (`out/router.npz`) from `router_data.json` (disjoint from the test prompts).
3. `demo.py` evaluates 6 prompts (4 name-free paraphrased procedural tasks + 2 outside prompts) on the plain model twice (`base1`, `base2`) and through the routed KV bank (`kv_routed`), then verifies `U0` and `G2` with `kestrom-verify compare events` (`out/report.json`).
4. `kestrom kv compile ... out/bank_init` incrementally updates 2 of the 4 procedures (`procedures_revised.json`) via `PREV_DIR` (reusing the 2 unchanged `.safetensors` shards bit-for-bit) and re-evaluates with the same `out/router.npz` (zero router retraining).

On `google/gemma-4-E4B-it` (BF16, MLX, `enable_thinking=False`), the full 112-prompt benchmark (64 name-free paraphrased procedural queries + 16 ambiguous + 32 outside) reached **64/64 (100%)** semantic routing accuracy, **64/64 (100%)** procedural accuracy (vs `7/64` plain base), **0** negative transfer on base-correct items, **32/32** outside and **15/16** ambiguous bit-identical (`kestrom-verify compare events` **U0/G2 112/112 PASS**), and **20/20 (100%)** switch accuracy on revised procedures via `PREV_DIR` (`6.94s`, `2/2` unchanged shards reused, `32/32` unrevised preserved).

Note: `kestrom-verify compare events` checks bit-identical agreement on `events=[]` prompts (and before position 0 on matched prompts); procedural correctness on matched prompts is measured by deterministic task verifiers.
