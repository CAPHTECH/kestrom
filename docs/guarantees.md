# What Kestrom guarantees, and how it checks it

A report is a JSON file with no times in it and with the SHA-256 of every model file, input and output it was made from. The same
inputs on the same machine give the same bytes: rebuild a report in another process and compare it with `cmp`. Each command ends by
printing every check as `name: count of total PASS|FAIL`, and exits with status 1 if one fails (the report is written either way). A
check passes when the count equals the total and the total is not zero (a check over no sequences checked nothing); D over no
changes passes.

## The checks

| Check | Part of the report | Meaning |
|---|---|---|
| W | `weights` | Every file under the folder, nested ones too, paired by its path, is byte-identical to the original's except the changed rows of the output layer; a file on one side only fails. The rows are read from the bytes; `files not equal` names the others |
| U0 (teacher forcing) | `forced`, `measure` | The original model, run twice, gives exactly the same logits, bit for bit (the machine is deterministic) |
| G1 (teacher forcing) | `forced` | At every position, every logit of an unchanged row of the output layer equals the original's, bit for bit |
| U0 (greedy) | `greedy`, `measure greedy` | The original model, run twice, gives the same greedy output |
| G2 (greedy, rows) | `greedy` | The edited output is the original's, or first differs where the original's or the edited token is a changed row. An output that stopped ends with its stop, so stopping against going on is a difference |
| G2 (greedy, events) | `greedy` (range `events`) | The edited output is the original's, or first differs at or after the first event the edited output records |
| D | `declared` | Every changed tensor, row and file is inside the declaration |

If U0 fails, G1 and G2 in that report mean nothing. W and D read only bytes; `range` and `diff` run no model and check no U0.

`compare` checks the output files it is given; it does not tie a file to the model and settings that wrote it. Write the outputs
yourself, running the original twice: the original files given twice are not a second run. Its range must be a `range` (or
`logits`) report whose every check passed. With `events`, G2 checks the output against the events the edited decoding records,
not that a loop or a condition was really there.

## Tier 1: the outputs outside the range

- **A row edit** (`kestrom rows`): the range is the rows of the output layer that changed. `range` reads it from the bytes and checks
  that every other byte (the other tensors, the file headers, the config and tokenizer files) is the same (W). Then, before the range
  is reached, the hidden state is computed by the same bytes, so every logit of an unchanged row is the original's (G1), and greedy
  decoding can only first differ where a changed row is the top token (G2).
- **An inference-time edit** (`kestrom generate`): the weights do not change. Each output records its events, the
  output positions where an edit acted (conditional steering: position 0 when the condition fires; rules: every position where a
  rule fired; a phrase ban: the start of the phrase it went back to; the loop guard: the start of the first loop). Before the first
  event the decoding is the original's, so the output first differs at or after it (G2 with the range `events`).
- **A trigger** (`kestrom trigger`): only the embedding rows of spare ids change, and an embedding row is read only for its own id, so
  a sequence without those ids is computed by the original bytes. `diff` with the rows declared checks the first part; `measure` on
  prompts without the trigger shows every logit the same. This holds for an untied embedding only: a tied one is also the output
  layer, and the rows would change the spare ids' logits everywhere.

## Tier 2: the weights inside the declaration

`diff` compares every tensor of two models byte for byte, paired by name across the shards (safetensors) or in one GGUF file, and
reports the tensors changed, added, removed or of another dtype or shape, the changed rows (the first axis) and elements, the other
files under the folder (nested ones too, by path), and the metadata (the safetensors `__metadata__`, or every key-value field of a
GGUF file: tokenizer, special tokens, ...). A declaration is a JSON object `{"name": "all" | [[first row, end row], ...]}`, where a
name is a tensor, another file of the folder (`config.json`, `sub/x.json`, ...) or `__metadata__`. D passes when every change is
inside it; an added, removed or retyped tensor, a changed file and changed metadata are inside only `"all"`.

A changed tensor inside the network can move the output at any position, so `diff` says nothing about outputs. `measure` counts
how much they moved.

## Tier 3: measured only

`measure` loads each model whole (one at a time) and runs teacher forcing on the given sequences with the original twice and the
edited once. It checks only U0, and counts: the positions whose logits are all the same bytes, whose top token changed, and the
largest change of the logit of the original's top token. Given greedy outputs of both, it also counts the prompts with the same
output. The report says `no guarantee`.

## What is not guaranteed

- That the range is right. Whether a banned token should be banned, or a condition fires on the right prompts, is outside the check.
  `kestrom rows ban` matches a string inside every token that holds it, also where the string is part of a correct word. A ban row keeps
  a banned logit at most -100 at the positions it was made from; elsewhere it is measured, not bounded.
- Agreement across runtimes. The comparison is between the original and the edited model in the same runtime. llama.cpp and MLX give
  different greedy outputs for the same original weights.
- Sampling. The guarantees are about teacher-forced logits and greedy decoding.
- Not tested: Linux, CUDA, imatrix quantization and quantization types other than Q4_K_M, models whose output layer is tied to the
  embedding (`range` stops when there is no `lm_head`), edits of input rows.

## Commands

| Command | What it does | Needs |
|---|---|---|
| `range ORIGINAL EDITED REPORT.json` | The changed rows of the output layer and W. Weights: a folder of safetensors (output layer `*lm_head.weight`) or a GGUF file (`output.weight`) | numpy; gguf for GGUF |
| `logits ORIGINAL EDITED INPUTS.json REPORT.json` | Teacher forcing: U0 and G1. The edited model is the original with its output layer read from the edited folder, so it stops if W fails (use `measure`) | mlx, mlx-lm |
| `generate mlx MODEL PROMPTS.json OUT.jsonl [MAX_TOKENS]` | Greedy decoding with MLX (512 tokens unless given) | mlx, mlx-lm |
| `generate gguf MODEL.gguf PROMPTS.json TOKENIZER OUT.jsonl` | Greedy decoding with llama-server, one slot. TOKENIZER: a Hugging Face folder, for the text | llama-server, transformers |
| `compare RANGE.json ORIGINAL.jsonl ORIGINAL2.jsonl EDITED.jsonl TOKENIZER REPORT.json` | U0 and G2 against the range of a `range` report, or `events` | transformers |
| `measure ORIGINAL EDITED INPUTS.json REPORT.json [ORIGINAL.jsonl ORIGINAL2.jsonl EDITED.jsonl]` | Tier 3 counts and U0 | mlx, mlx-lm |
| `diff ORIGINAL EDITED REPORT.json [DECLARED.json]` | Tier 2: every tensor, and D | numpy; gguf for GGUF |
| `summary REPORT.json` | The report in markdown | — |

Inputs: `PROMPTS.json` is `[{"id": ..., "ids": [prompt token ids]}, ...]`; `INPUTS.json` adds the answer tokens as `"tokens"`. Other
keys are copied to the outputs. A generation (JSONL) has `id`, `finish` (`stop` or `length`), `tokens` (without the stop token) and
`text`; an inference-time edit adds `events`.
