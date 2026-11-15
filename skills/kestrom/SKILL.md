---
name: kestrom
description: Use Kestrom to edit a language model inside a range and check the edit with kestrom-verify - pick the command for a goal (ban tokens, steer or ban phrases at decoding time, stop thinking loops, edit neurons or heads, merge), run the checks (range, generate, compare, logits, diff, measure), and state what the report does and does not guarantee by tier. Use for any task that runs `kestrom` or `kestrom-verify`, reads their reports, or asks what an edit, a fine-tune or a quantization changed.
---

# Using Kestrom

Two packages. `kestrom-verify` (`verify/`, Apache-2.0) is the checker: it reads two models and their outputs and writes a report.
`kestrom` (`edit/`, PolyForm or a commercial license) holds the editing tools. Help is in each module's docstring:
`kestrom-verify` and `kestrom` with no arguments, `python -m pydoc kestrom.<module>` (generate, rules, steer, loopguard, rows,
trigger, find, mlp, units, train, inspect). Read the help of a command before running it; this skill is a map, not the reference.

Install from a clone: `pip install ./verify[mlx] ./edit[mlx]` (`./verify[torch]` for PyTorch). Without installing, from the
repository root: `PYTHONPATH=verify:edit python3 -m kestrom_verify ...` and `PYTHONPATH=verify:edit python3 -m kestrom ...`.
The commands that run or quantize a model use MLX on Apple Silicon (each help says what a command reads).

## The tiers decide what you may claim

| Tier | Edits | Checks | What you may say |
|---|---|---|---|
| 1. Outputs | `kestrom rows` (rows of the output layer), `kestrom generate --rules` (decoding-time edits) | `range` (W), `logits` (U0, G1), `compare` (U0, G2) | Greedy output is the original's until the range is reached; for a row edit, every logit of an unchanged row is bit-identical |
| 2. Weights | `kestrom edit ...`, `kestrom rules write-mlp`, `kestrom trigger` | `diff` with a declaration (D) | Only the declared tensors / rows changed. Nothing about outputs; `measure` counts how much they moved |
| 3. Measured only | `kestrom merge`, `kestrom train sft-data`, any fine-tune or quantization made elsewhere | `measure`, `diff`; `kestrom inspect` (evidence to read) | Counts only. No guarantee |

Never state a guarantee no check computed. If U0 fails, G1 and G2 in that report mean nothing. Whether the range is *right*
(whether a token should be banned, whether a condition fires on the right prompts) is never checked.

## Pick the command

| Goal | Command | Check with |
|---|---|---|
| Never output listed tokens | `kestrom rows ban-row`, then `kestrom rows ban` | `range`, `generate` ×3, `compare`, `logits` (`examples/zh-ban`) |
| Keep a string (not a token) out, or avoid stock phrases | a `phrase` line in a rules file, `kestrom generate --rules` | `compare events` |
| Change style only where a condition on the prompt holds | `kestrom steer direction`, then `in ... prompt` and `add` lines | `compare events` |
| Stop thinking loops | a `loopguard` line | `compare events` (`examples/loop-guard`) |
| Rules over probed features of the hidden state | `kestrom rules fit`, then `kestrom generate --rules FIT.npz` | `compare events` |
| Set output rows toward target answers | `kestrom rows lsq` | as a row edit |
| Lower the tokens that start listed phrases, in the output rows | `kestrom rows ftpo` (from the phrase events of `kestrom generate`) | as a row edit |
| Find where to edit (reads only) | `kestrom find direction / lens / reverse / writers / heads` | — |
| Edit neurons, heads, a direction in `down_proj`, a keyed write | `kestrom edit neuron-gain / head-gain / head-proj / subtract / keyed / neuron-write` | `diff` with a declaration, `measure` |
| A trigger in spare embedding rows | `kestrom trigger prompts / train / write` | `diff` with the rows declared |
| Merge weights | `kestrom merge A B C LAMBDA DST` | `measure`, `diff` |
| What a received model changed | — | `diff` (with the declaration if one was given), `measure` (`examples/acceptance-report`) |
| Look for a backdoor in a delivered fine-tune | `kestrom inspect` | see the `kestrom-acceptance` skill |

A ban is of tokens, not characters: `kestrom rows ban` matches a string inside every token that holds it, also inside correct
words; to keep a string out, use a `phrase` line. Read the list before you ship.

## The check sequences

Prompts: `[{"id": ..., "ids": [prompt token ids]}, ...]` (`examples/prompts.py` makes them through the chat template). Inputs for
teacher forcing add the answer tokens as `"tokens"`.

**Row edit (tier 1).**

```sh
kestrom-verify range ORIGINAL EDITED range.json                  # W: every other byte the same
kestrom-verify generate mlx ORIGINAL prompts.json base.jsonl     # the original twice: two separate runs
kestrom-verify generate mlx ORIGINAL prompts.json base2.jsonl
kestrom-verify generate mlx EDITED prompts.json edited.jsonl
kestrom-verify compare range.json base.jsonl base2.jsonl edited.jsonl ORIGINAL greedy.json   # U0, G2
kestrom-verify logits mlx ORIGINAL EDITED inputs.json logits.json                            # U0, G1
```

With an output layer tied to the embedding, pass the prompts as the last argument of `compare`; G1 and G2 then cover only inputs
with no changed id.

**Decoding-time edit (tier 1).** `kestrom generate ORIGINAL prompts.json edited.jsonl --rules RULES.dsl`, the original twice with
`kestrom-verify generate`, then `kestrom-verify compare events base.jsonl base2.jsonl edited.jsonl ORIGINAL report.json`. G2 says the
output is the original's until the first recorded event, not that the condition or loop was really there.

**Weight edit (tier 2).** `kestrom-verify diff ORIGINAL EDITED diff.json declared.json`, the declaration
`{"NAME": "all" | [[first row, end row], ...]}`, NAME a tensor, another file of the folder (`config.json`) or `__metadata__`; an
added, removed or retyped tensor, a changed file and changed metadata are inside only `"all"`. Then `measure` to count how much the outputs moved.

**No guarantee (tier 3).** `kestrom-verify measure RUNTIME ORIGINAL EDITED inputs.json measure.json [base.jsonl base2.jsonl
edited.jsonl]`: positions whose logits are the same bytes, whose top token changed, prompts with the same greedy output; U0 only.

`kestrom-verify summary REPORT.json` prints a report as markdown.

## Rules that keep reports valid

- Run the original twice, as two runs. The same output file given twice is not a second run, and `compare` does not tie a file to
  the model that wrote it.
- Compare the original and the edited model on the same runtime and machine. MLX and llama.cpp (and `mlx` and `torch`) give
  different outputs for the same weights; agreement across runtimes is not guaranteed.
- The guarantees are about teacher-forced logits and greedy decoding, not sampling.
- A report has no times in it: rebuild it in another process and compare with `cmp` to show it is reproducible. Every `kestrom-verify`
  command but `generate` and `summary` prints each check as `name: count of total PASS|FAIL` and exits 1 if one fails.
- `logits` stops when W fails (use `measure`) or when the output layer is quantized into several tensors (use `compare`).
- Not tested: Linux, CUDA, imatrix quantization and quantization types other than Q4_K_M, GGUF models with a tied output layer, edits of
  input rows. Say so when a task leaves the tested ground.

Details: `docs/guarantees.md` (every check and command), `docs/report-format.md` (report keys), `examples/` (three runs from start
to end).
