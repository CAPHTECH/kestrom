# Twin-track control-flow non-interference

`kestrom ifc control` runs two tracks of the same model in lockstep (no weights change): Track A reads the full prompt including
untrusted email text, while Track B reads the same prompt with the untrusted text replaced by `[UNTRUSTED_DATA_MASKED]`.

Structural JSON syntax and the declared control keys (`tool`, `recipient`, `approved`) come strictly from Track B; the `summary`
value comes from Track A. `events` records the `summary` (DATA) positions, so `kestrom-verify compare events` against two runs of
clean Track B checks that every leading control token is bit-identical to a model that never saw the untrusted email (U0 and G2).

```sh
sh examples/ifc-control/run.sh QWEN35_4B_FOLDER      # Qwen/Qwen3.5-4B, MLX; ~1 minute
```

The script runs three prompts (one benign email and two emails containing prompt injections that try to hijack `tool`, `recipient`
and `approved`):

1. `kestrom-verify generate mlx` runs clean Track B twice (`out/clean1.jsonl`, `out/clean2.jsonl`) and unprotected Track A once (`out/base.jsonl`).
2. `kestrom ifc control` runs twin-track decoding (`out/ifc.jsonl`).
3. `kestrom-verify compare events` checks `out/ifc.jsonl` against `out/clean1.jsonl` and `out/clean2.jsonl`, writing `out/report.json`.

Put control keys before data keys so `G2` covers every control token. If a control key appears after a data key without `ISOLATE_B`,
Track B reads the generated data tokens (an indirect channel). `compare events` checks agreement with clean Track B before the first
DATA position; it does not check that the `summary` text is faithful.
