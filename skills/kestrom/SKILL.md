---
name: kestrom
description: Use Kestrom to edit a language model inside a range and check the edit with kestrom-verify - pick the command for a goal (ban tokens, steer or ban phrases at decoding time, stop thinking loops, route compiled fact KV caches, twin-track IFC, edit neurons or heads, merge), run the checks (range, generate, compare, logits, diff, measure), and state what the report does and does not guarantee by tier. Use for any task that runs `kestrom` or `kestrom-verify`, reads their reports, or asks what an edit, a fine-tune or a quantization changed.
---

# Using Kestrom

Two packages. `kestrom-verify` (`verify/`, Apache-2.0) is the checker: it reads two models and their outputs and writes a report.
`kestrom` (`edit/`, PolyForm or a commercial license) holds the editing tools. Help is in each module's docstring:
`kestrom-verify` and `kestrom` with no arguments, `kestrom <command> --help`, `python -m pydoc kestrom.<module>` (generate, kv, ifc, rules, steer, loopguard, rows,
trigger, find, mlp, units, train, inspect, audit). Read the help of a command before running it; this skill is a map, not the reference.

Install the Apache-2.0 checker from PyPI (`pip install "kestrom-verify[mlx]"` or `"kestrom-verify[torch]"` / `"kestrom-verify[gguf]"`), or install both packages from a clone: `pip install ./verify[mlx] ./edit[mlx]` (`./verify[torch] ./edit[torch]` for PyTorch/CUDA, supported by `kestrom-verify`, `kestrom ifc`, `kestrom inspect mirror`/`disarm`, and `kestrom audit`). Without installing, from the
repository root: `PYTHONPATH=verify:edit python3 -m kestrom_verify ...` and `PYTHONPATH=verify:edit python3 -m kestrom ...`.
Other `kestrom` commands that run or quantize a model use MLX on Apple Silicon (each help says what a command reads).

## The tiers decide what you may claim

| Tier | Edits | Checks | What you may say |
|---|---|---|---|
| 1. Outputs | `kestrom rows` (rows of the output layer), `kestrom generate --rules` (decoding-time edits, `open` and `kv` lines), `kestrom ifc` (twin-track non-interference) | `range` (W), `logits` (U0, G1), `compare` (U0, G2) | Greedy output is the original's (or reference track's) until the range is reached; for a row edit, every logit of an unchanged row is bit-identical (with a tied output layer: on inputs that hold no changed id) |
| 2. Weights | `kestrom edit ...`, `kestrom rules write-mlp`, `kestrom trigger` | `diff` with a declaration (D) | Only the declared tensors / rows changed. Nothing about outputs; `measure` counts how much they moved |
| 3. Measured only | `kestrom merge`, `kestrom train sft-data`, any fine-tune or quantization made elsewhere | `measure`; `diff` (it pairs tensors by name: a folder quantized by `mlx_lm convert` packs and, for Qwen3.5, renames them, so it shows little beyond that); `kestrom inspect` (evidence to read) | Counts only. No guarantee |

Never state a guarantee no check computed. If U0 fails, G1 and G2 in that report mean nothing. Whether the range is *right*
(whether a token should be banned, whether a condition fires on the right prompts) is never checked.

## Pick the command

| Goal | Command | Check with |
|---|---|---|
| Never output listed tokens | `kestrom rows ban-row`, then `kestrom rows ban` | `range`, `generate` ×3, `compare`, `logits` (`examples/zh-ban`) |
| Keep a string (not a token) out, or avoid stock phrases | a `phrase` line in a rules file, `kestrom generate --rules` | `compare events` |
| Change style only where a condition on the prompt holds | `kestrom steer direction`, then `in ... prompt` and `add` lines | `compare events` |
| Stop thinking loops | a `loopguard` line | `compare events` (`examples/loop-guard`) |
| Keep JSON control keys independent of untrusted data (prompt injection) | `kestrom ifc control` (`ids` full, `clean_ids` untrusted masked) | `compare events` against clean Track B (`examples/ifc-control`) |
| Prevent secret leakage into public text (`strict-public` zero-flow or `gated-public` lift gate) | `kestrom ifc firewall` (`ids` public+secret, `clean_ids` public only) | `compare events` (`strict-public` vs clean Track B; `gated-public` vs full Track A) |
| Make a model's answer open with the class that hand-written rules answer (rules of `kestrom rules run`, connected to a model) | an `open [HEAD]: FILE [CLASS] ["TEXT"]` line; each prompt carries `data`, what the rules read | `compare events` (`examples/rules-open`); it does not check that the class or the text after it is right |
| Compile per-fact prefix KV caches offline from standalone rules, compose multiple matches, or run multi-turn `steps`/`prev` sessions over the bank | `kestrom kv compile MODEL RULES.dsl FACTS.json OUT_DIR [PREV_DIR]`, then `kv: OUT_DIR [MAX_MATCH]` in `kestrom generate --rules` or `kestrom kv chat MODEL OUT_DIR` | `compare events` (the `kv` line records no event on unmatched prompts; matched prompts log event `0`) |
| Rules over probed features of the hidden state (single model or transferred across models) | `kestrom rules fit MODEL RULES.dsl DATA.json FIT.npz [BALANCE] [K] [STEPS] [LR] [LAYER]`, optionally `kestrom bridge rules BRIDGE.npz FIT.npz OUT_FIT.npz [DONOR RECIPIENT CALIB_DOCS.json]`, then `kestrom generate --rules FIT.npz` | `compare events` |
| Try rules, or a small classifier or state machine written by hand, with no model | `kestrom rules run` (`kestrom rules build` writes the network out; `kestrom rules weigh` fits the out coefficients to labels) | the labels in DATA (no tier) |
| Set output rows toward target answers | `kestrom rows lsq` | as a row edit |
| Lower the tokens that start listed phrases, in the output rows | `kestrom rows ftpo` (from the phrase events of `kestrom generate`) | as a row edit |
| Find where to edit (reads only) | `kestrom find direction / lens / reverse / writers / heads` | — |
| Edit neurons, heads, a direction in `down_proj`, a keyed write | `kestrom edit neuron-gain / head-gain / head-proj / subtract / keyed / neuron-write` | `diff` with a declaration, `measure` |
| A trigger in spare embedding rows | `kestrom trigger prompts / train / write` | `diff` with the rows declared |
| Merge weights | `kestrom merge A B C LAMBDA DST` | `measure`, `diff` |
| What a received model changed | — | `diff` (with the declaration if one was given), `measure` (`examples/acceptance-report`); for a claimed LoRA or QLoRA, `kestrom inspect quant` and `kestrom inspect shape` (rank against the claim); for a Qwen3.5 GGUF file, `kestrom inspect gguf` first (Q4_K_M tested) |
| Look for a backdoor in a delivered fine-tune | `kestrom inspect` | see the `kestrom-acceptance` skill |
| Check a single checkpoint (no original model, no trigger) for quantization-conditioned backdoors and disarm flagged tensors | `kestrom inspect mirror`, then `kestrom inspect disarm` | `kestrom-verify diff` on the disarmed copy (tier 3: evidence only, no guarantee) |
| One-command acceptance audit (optional `diff` + `inspect mirror`) into `audit.json`, `audit.html`, and `audit.sarif` | `kestrom audit MODEL_DIR OUT_DIR [BASE_DIR\|none] [DECLARED.json\|none] [MODE] [BITS] [GROUP] [RUNTIME]` | `PASS` only when tier 2 `diff` + `DECLARED.json` passes D and `mirror` flags nothing; `NO_FINDINGS` without a declaration; `FLAGGED` (exit 1) otherwise |

A ban is of tokens, not characters: `kestrom rows ban` matches a string inside every token that holds it, also inside correct
words, and byte-level tokens can still spell the string (in Qwen3.5, 的 as token 95725 then 226). To keep a string out, use a
`phrase` line. Read the list before you ship.

## The check sequences

Prompts: `[{"id": ..., "ids": [prompt token ids]}, ...]` (`examples/prompts.py` makes them through the chat template). Inputs for
teacher forcing add the answer tokens as `"tokens"`.

**Row edit (tier 1).** A ban first needs a ban row, made from the original's own answers (`kestrom rows ban-row`: the first half
of INPUTS sets it, the second half is held out, and it stops if any position reads zero or negative; then give it more answers. In
our test 10 answers of 64 tokens were not enough and 40 were). INPUTS is the prompts with the answers as `"tokens"`
(`examples/zh-ban/run.sh` builds it from `base.jsonl`).

```sh
kestrom-verify generate mlx ORIGINAL prompts.json base.jsonl     # the original's answers (also the first of its two runs)
kestrom rows ban-row ORIGINAL inputs.json row.npy                # inputs.json: prompts + those answers
kestrom rows ban ORIGINAL ORIGINAL list.txt row.npy EDITED       # SRC, TOKENIZER, LIST, ROW, DST
kestrom-verify range ORIGINAL EDITED range.json                  # W: every other byte the same
kestrom-verify generate mlx ORIGINAL prompts.json base2.jsonl    # the second, separate run of the original
kestrom-verify generate mlx EDITED prompts.json edited.jsonl
kestrom-verify compare range.json base.jsonl base2.jsonl edited.jsonl ORIGINAL greedy.json   # U0, G2
kestrom-verify logits mlx ORIGINAL EDITED inputs.json logits.json                            # U0, G1
```

The output layer is tied to the embedding when the model has no output layer tensor of its own (`kestrom-verify summary range.json` shows `tied to the embedding: True`). Then pass the
prompts as the last argument of `compare`, and G1 and G2 cover only inputs with no changed id: `logits` on the original's answers
skips every sequence whose prompt or answer holds a banned token (read the counts: 5 of 5 where 10 were given). Run `logits` also on inputs made from
the edited model's answers, which hold none.

**Decoding-time edit (tier 1).** `kestrom generate ORIGINAL prompts.json edited.jsonl --rules RULES.dsl`, the original twice with
`kestrom-verify generate`, then `kestrom-verify compare events base.jsonl base2.jsonl edited.jsonl ORIGINAL report.json`. G2 says the
output is the original's until the first recorded event, not that the condition or loop was really there.

**Weight edit (tier 2).** `kestrom-verify diff ORIGINAL EDITED diff.json declared.json`, the declaration
`{"NAME": "all" | [[first row, end row], ...]}`, NAME a tensor, another file of the folder (`config.json`) or `__metadata__`; an
added, removed or retyped tensor, a changed file and changed metadata are inside only `"all"`. A model saved again by a library
can gain `__metadata__` (`{"format": "pt"}`) and fails D on that alone: read `diff`'s list of what is outside before you call it
a violation. Then `measure` to count how much the outputs moved. For a keyed or relation write (`kestrom edit keyed`), `diff` and
`measure` check only changed weights and logit movement, and testing inside the training context alone can overstate generalization:
evaluate both direct recall (Level 1) and downstream one-hop use (Level 2) on unseen contexts (new distractors and swapped
attributes), since a write can fire cleanly (`κ·x ≈ 1`) and keep stated facts while dropping sharply outside its training context.
Where cached system-turn KV context routed by standalone rules is acceptable (not a weight edit; equivalent to supplying the fact in
a cached system prompt), tier 1 `kestrom kv` reached 64/64 unseen org and city under exact-name regex routing (`\bP\b`), whereas
non-exact semantic paraphrase routing via `kestrom steer direction` (`Q7`–`Q9`) reached `PARTIAL` on 1 model / 4 roles.

**No guarantee (tier 3).** `kestrom-verify measure RUNTIME ORIGINAL EDITED inputs.json measure.json [base.jsonl base2.jsonl
edited.jsonl]`: positions whose logits are the same bytes, whose top token changed, prompts with the same greedy output; U0 only.

`kestrom-verify summary REPORT.json` prints a report as markdown.

## Writing a network by hand with `kestrom rules run`

`examples/rules-run` has a router with two heads and Euclid's algorithm with registers.
It edits no model: claim no tier, only how many held-out answers agree with their labels.

1. Make the check first. Split the labelled entries into dev (shown to the writer) and held-out (never shown). If the inputs are
   finite (`match` lines only), list every combination as DATA: the answers then cover them all. For rules with a state (`prev`
   lines, entries of steps), list every sequence up to a length. A `set` line keeps a number in a register (`+ - * / // %`);
   `"steps": N` runs an entry N times, and the registers after the last step are in OUT: give `"targets": {register: number}` to
   score them. Classes named `HEAD/CLASS` are answered per head, with labels `{head: class}`.
2. Write the rules. A unit that fires counts as 1; with no unit firing or a tied top score the answer is null, counted apart from
   wrong answers; the top score wins also where none of its own units fires (a negative coefficient elsewhere). Give values in
   whole numbers (a gate holds from its threshold minus 0.5). A regex is a JSON string (`"\\d+"`) and searches the text as given;
   `count` gives its matches and `number` the number in its group: gate these, like values, with `1*x >= T`, not `all` or `any`.
3. Run on dev, read the ids answered wrong and the `fired` units, name the cause, change only what it needs, and keep every
   version with its dev score.
4. Pick the best version on dev, not the last, and report its held-out score once. `kestrom rules weigh` on dev only: it fits
   the out lines to the labels it is given, so its score on those entries is not a held-out score.

## Connecting those rules to a model with an `open` line

`examples/rules-open` runs it from start to end (the router of `examples/rules-run`, Qwen3.5-4B).

1. Ask the model, in the system prompt, for the class alone on line 1 and whatever else after it. The class names are those of the
   rules' out lines (of one head: `HEAD/CLASS`).
2. Give every prompt a `"data"` entry, what the rules read (`{"text": ...}`, as in the DATA of `kestrom rules run`). Normalize the
   text there if the rules were written for normalized text: the regexes search it as given.
3. Write `open HEAD: FILE.dsl` in the rules file of `kestrom generate --rules` (no HEAD when the classes have none). Add a last
   word, `open HEAD: FILE.dsl CLASS`, to open with CLASS where the rules answer none; without it nothing is done there. With a
   default class, a question the rules cannot read (a typo, a new wording) is opened with it even when the model would have
   answered: measure both forms on held-out questions before choosing.
4. Generate the unchanged model twice (`kestrom-verify generate mlx`) and run `compare events`. Outputs with no event are the
   original's: the model and the rules agreed there, or the line left the output alone (`open` is null). Count line 1 right before and after, and how many went from right to wrong.

The answer is what follows the last `</think>`; nothing is done while the model thinks. The line acts only where the model begins
the answer with a class: where the top token that begins it starts none of the head's classes (nor CLASS) the output is left alone,
so prompts that ask for no class can go through the same rules file (measure it: an answer that begins with the first characters of a
class is opened whatever was asked). It then also leaves a routing answer that opens with other text (an apology, a sentence), and a
class written later in the answer. `open` is null there, as it is where the rules answer none and the line gives no class, and
where the thinking did not end or the output was cut short: run `kestrom rules run` on the same `data` and count the outputs that
stopped (`finish`) with `open` null where the rules answer.

For a model that thinks, the answer-only line waits for the thinking to end and can do nothing where it does not. Add a text as
the last word, `open HEAD: FILE.dsl CLASS "The rules have decided line 1: {}. ..."`: the thinking then opens with that text, `{}` the
class, and a line feed. It is written wherever thinking begins (with another token than `</think>`) and the rules answer or the
line gives a class, also for prompts that ask for no class, so use it only where every prompt is a routing one. Measure, against the unchanged model at the same token limit: how many outputs close
the thinking, line 1 right, and right to wrong. In one measurement (Qwen3.5-4B, 24 questions, at most 2,048 tokens, one wording of
the text) line 1 was right on 4 unchanged (19 outputs never closed the thinking at that limit) and on 11 with the text; one that was
right no longer closed the thinking, 13 never closed it, and a loop guard added none. With thinking off the same questions gave 13
unchanged and 22 opened (with a default class), and of the 8 that became right with the text, 6 are questions the unchanged model has right with thinking off: compare with thinking
off before you keep thinking on for a routing prompt.
With the text nearly every output has events from its first thinking position, so `compare events` says almost nothing about it.

What to claim: tier 1 for the output before the first event, and the counts you measured. OUT records as `open` the class the answer opens with by the line; no
check computes that the class is right or that the text after it is true or agrees with it, so read a sample of the reasons where
events occurred. A class must be spelled in tokens of whole characters, and where other lines ban every token toward it the run stops.

## Compiled fact-KV routing with `kestrom kv`

Use `kestrom kv` when standalone rules should supply specific facts (without placing fact text in the inference prompt) while leaving unmatched prompts untouched.

1. Prepare standalone rules (`RULES.dsl`, verified with `kestrom rules run`, optionally using `prev` lines for multi-turn state) and `FACTS.json`: `{CLASS: TEXT}` for single-head rules, or `[{"key": ..., "when": CLASS | {HEAD: CLASS, ...}, "text": ...}]` for multi-head rules.
2. Compile the bank once offline: `kestrom kv compile MODEL RULES.dsl FACTS.json OUT_DIR [PREV_DIR]`. At compile time it checks across two probe turns that the chat template places the system turn cleanly before the user turn, compacts trailing slots from linear KV caches, stages into a temporary directory before renaming atomically, and hashes `rules.dsl`, every `.safetensors` cache, and `model_fingerprint` into `manifest.json`. Pass `PREV_DIR` after editing a subset of facts to copy unchanged `(key, prefix_ids)` caches byte-for-byte.
3. For checked batch decoding (`kestrom generate --rules`), put `kv: OUT_DIR [MAX_MATCH]` in the rules file (`MAX_MATCH` defaults to `1`; values `> 1` compose `2..MAX_MATCH` simultaneous matches on demand via an in-memory LRU `ComposeCache`). Prompts need `"data"` when the bank reads `match`/`count`/`number` inputs (`"data"` may be omitted when the bank reads only `in ... prompt` conditions). Any `in NAME: prompt FILE LAYER` conditions (including cross-model probes mapped by `kestrom bridge map`) in the same rules file are evaluated on the full prompt without the fact KV cache before decoding and passed as 0/1 `values` to the bank's rules (with `"steps"`, to the last step: earlier steps give their own). When routing multiple semantic classes purely with `kestrom steer direction` and strict `!p` mutual exclusion in `RULES.dsl`, do not mix far out-of-domain negatives and near competing classes into a single bimodal `QUIET` (the unweighted centroid midpoint shifts below off-classes and causes strict `!p` exclusion to cancel all heads). Separating coarse domain/ambiguity gates (`spec`, `single`) from symmetric 1-vs-1 pairwise comparisons reached `PARTIAL` (`114/128` routing on `1` model / `4` roles, not a guarantee), at a cost of `O(N(N-1)/2)` pairwise probes and one full prompt forward per `in ... prompt` line (`Q9` `gen kv` `310s` vs `base` `79s`). Run `kestrom-verify generate mlx` twice on `MODEL` and verify with `kestrom-verify compare events`.
4. For multi-turn conversations, run `kestrom kv chat MODEL OUT_DIR [MAX_TOKENS] [MAX_MATCH] [TURNS.json] [OUT.jsonl] [PROBES.dsl]` (pass `-` for `TURNS.json` and `OUT.jsonl` when supplying `PROBES.dsl` interactively). Each turn evaluates optional `PROBES.dsl` `in ... prompt` conditions on the raw turn alongside the cumulative `steps` history through the bank's `prev` rules, reuses the live KV cache incrementally (`reuse`) while matched keys hold and the chat template remains a strict prefix extension (Gemma 4 E4B), and swaps, composes, or re-prefills (`switch`/`compose`/`rebuild`) when topics change or the template rewrites history (Qwen3.5).

What to claim: the mechanism is cached system-turn KV context, not a weight edit (~1 MB/fact on Gemma 4 E4B, ~10 MB/fact on Qwen3.5-0.8B and ~26 MB/fact on Qwen3.5-9B `ArraysCache`). In batch mode, tier 1 `compare events` checks unmatched prompts (where `kv` records no event) up to the first event of another line (`events=[]` and bit-identical when no other line acts), and covers nothing after position `0` on matched prompts (`kv` set, not `null`); neither the accuracy of matched answers nor `kestrom kv chat` (no `compare events` check) is proven by the checker. `model_fingerprint` hashes config/tokenizer files and every `.safetensors` shard (size and SHA-256).

## Cross-model steering and probe transfer with `kestrom bridge`

Use `kestrom bridge` to transfer existing `kestrom steer direction` `.npz` files (`bridge map`) or `kestrom rules fit ... LAYER` token-probe `.npz` files (`bridge rules`) to another model or family without collecting labeled training data on the recipient model.

1. Prepare a diverse, unlabeled multi-domain anchor file (`ANCHORS.json`, e.g., ~1,000–1,600 short bilingual/domain prompts covering the spaces you want aligned; each entry a string, `{"text": ...}`, `{"prompt": ...}`, or `{"ids": ...}` where both models share a tokenizer). Narrow or single-language anchors collapse off-manifold directions into the SVD nullspace.
2. Fit the closed-form Procrustes bridge once: `kestrom bridge fit DONOR RECIPIENT ANCHORS.json BRIDGE.npz [LAYERS] [RANK]` (`"auto"` picks up to 6 aligned pairs at 25%–72% depth, or pass explicit `"17:22,20:26"` pairs, each layer in at most one pair; optionally compose `A→B` and `B→C` on shared middle layers via `kestrom bridge chain BRIDGE_AB.npz BRIDGE_BC.npz BRIDGE_AC.npz`).
3. **Transfer `kestrom steer` vectors (`bridge map`)**: `kestrom bridge map BRIDGE.npz DONOR_STEER.npz RECIPIENT_STEER.npz [RECIPIENT CALIB_PROMPTS.json]`. This rescales the rotated steering vector to the recipient's residual RMS norm (printed as `rms_norm(ALPHA)`) and translates the condition threshold either in closed form (zero forward passes) or by taking the median over unlabeled `CALIB_PROMPTS.json` on `RECIPIENT` (give a set matching the balance and phrasing distribution between what should and should not fire). Only the printed mapped layers are filled: use one of them in `add` and `in ... prompt` lines (`kestrom generate` refuses the others).
4. **Transfer `kestrom rules fit` token probes (`bridge rules`)**: fit the donor rules on a decoder `LAYER >= 0` present in `BRIDGE.npz` via `kestrom rules fit DONOR RULES.dsl DATA.json DONOR_FIT.npz [BALANCE] [K] [STEPS] [LR] LAYER`, then run `kestrom bridge rules BRIDGE.npz DONOR_FIT.npz OUT_FIT.npz [DONOR RECIPIENT CALIB_DOCS.json]`. Omitting the optional trailing arguments computes `P` and `bias` in closed form; passing `DONOR RECIPIENT CALIB_DOCS.json` matches per-probe token-stream mean and standard deviation across the two models without labels (often needed across distinct families where residual variance shrinks). Mid-late layers transfer coarse token-region semantics (`in_code`, language spans) well, whereas next-token syntax boundaries can degrade cross-family; any `out`/`move`/`ban` tokens in the rules must each be single tokens in the recipient tokenizer, and a steer file named by an `in ... prompt` or `add` line must already be in the recipient's space (map it with `bridge map` first and name that output, at one of its printed layers, in the rules; `bridge rules` refuses a donor-space file).
5. Run `kestrom generate MODEL PROMPTS.json OUT.jsonl --rules <OUT_FIT.npz|RULES.dsl>`, and verify with two runs of `kestrom-verify generate mlx` and `kestrom-verify compare events`.

What to claim: tier 1 `U0`/`G2` via `kestrom-verify compare events` (outputs before the first recorded event—including prompts/positions where the bridged probes do not fire—are bit-identical to the recipient base model); steering effectiveness and probe classification accuracy on held-out prompts are measured separately.

## Rules that keep reports valid

- Run the original twice, as two runs. The same output file given twice is not a second run, and `compare` does not tie a file to
  the model that wrote it.
- Compare the original and the edited model on the same runtime and machine. MLX and llama.cpp (and `mlx` and `torch`) give
  different outputs for the same weights; agreement across runtimes is not guaranteed.
- The guarantees are about teacher-forced logits and greedy decoding, not sampling.
- A report has no times in it: rebuild it in another process and compare with `cmp` to show it is reproducible. It records each
  input by file name and SHA-256, so a rebuild needs the same file names. Every `kestrom-verify`
  command but `generate` and `summary` prints each check as `name: count of total PASS|FAIL` and exits 1 if one fails.
- `logits` stops when W fails (use `measure`) or when the output layer is quantized into several tensors (use `compare`).
- Not tested (as `docs/guarantees.md` lists them): Linux/CUDA outside `kestrom ifc`, `kestrom inspect mirror`/`disarm`, `kestrom audit`, and `kestrom-verify generate torch` / `compare events` / `diff` (other `torch` commands ran on macOS), imatrix quantization and
  quantization types other than Q4_K_M, models whose output layer is tied to the embedding in GGUF (tested in MLX, with
  Qwen3.5-0.8B), edits of input rows. Say so when a task leaves the tested ground.

Details: `docs/guarantees.md` (every check and command), `docs/report-format.md` (report keys), `examples/` (five runs from start
to end).
