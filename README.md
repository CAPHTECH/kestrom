# Kestrom

**Edit a language model inside a range, and let anyone check, byte for byte, what changed and what did not.**

Fine-tuning, LoRA, merging and quantization can move the output at any position. After such an edit, the only way to know that
"nothing else changed" is to evaluate everything again, and even then you only know it for the prompts you tried. In our runs on
Qwen3.5-9B and Qwen3-8B, a 4-bit quantization and a public fine-tuned derivative each left **0 of 150** greedy answers the same as
the original's.

Kestrom takes the other side. Each edit comes with a **range** that is read from the weights or from the decoding, not from what
the editor claims, and a report that anyone can rebuild on their own machine and compare with `cmp`.

```
original weights ──edit──▶ edited weights ──kestrom-verify──▶ report.json
                                                   │  W   every byte the same but the changed rows
                                                   │  U0  the original equal to itself on this machine
                                                   │  G1  every logit outside the range equal, bit for bit (row edits)
                                                   │  G2  greedy output the same until the range is reached
                                                   ▼
                                  the receiver rebuilds report.json ─▶ same bytes
```

Two packages:

| Package | Command | What it is | License |
|---|---|---|---|
| `kestrom-verify` | `kestrom-verify` | The checker. Reads two models (and their outputs) and writes the report | Apache-2.0, free for everyone |
| `kestrom` | `kestrom` | The editing tools whose range the checker can bound | Free for small businesses and noncommercial use (PolyForm); other uses need a commercial license |

## Three tiers of guarantee

The checks that pass say which tier an edit is in. Kestrom does not blur them.

| Tier | What is checked | Commands | Edits |
|---|---|---|---|
| 1. Outputs | Greedy output is the same until the range is reached (G2); for a row edit, every logit outside the range is also bit-identical to the original (G1) | Row edits: `range`, `logits`, `compare`. Decode-time edits: `compare events` | `kestrom rows` (rows of the output layer), `kestrom generate` (decode-time edits: rules, conditional steering, adding a direction after a given text, phrase bans, loop stopping) |
| 2. Weights | Only the declared tensors / rows changed (declaration check D). How much the outputs moved is counted, not bounded | `diff`, `measure` | `kestrom edit` (neuron and head gains, a head's write along a direction, keyed writes, neuron writes, a direction out of a few `down_proj` columns), `kestrom rules write-mlp` (rules in MLP neurons), `kestrom trigger` (spare rows of the embedding) |
| 3. Measured only | What changed, and how much | `measure`, `diff` | `kestrom merge`, and any fine-tune (LoRA too) or quantization made elsewhere |

Determinism (U0: the original model gives the same logits and the same greedy output twice on your machine) is checked by `logits`
and `measure`, which run the original twice, and by `compare`, on the two outputs of the original you give it. If U0 fails, G1 and
G2 in that report mean nothing. `range` and `diff` read only bytes. `compare` checks the output files you give it: write them
yourself, running the original twice. For decode-time edits it checks the output against the events the decoding records, not that
a loop or a condition was really there.

## The editing tools

Every tool is in, whether or not it reached the target we set for it. The table gives the tier and what we measured, with the
target where one was missed. A tool's guarantee says what did *not* change; whether the edit does what you want is measured, and
for several tools it fell short. Each command's help (`kestrom`, `python -m pydoc kestrom.<module>`) has the details.

| Command | What it edits | Tier | What we measured (model) |
|---|---|---|---|
| `kestrom rows ban` | Output-layer rows of listed tokens set to a ban row | 1 | Chinese mixing 0.32 → 0.0067 (Qwen3.5-9B; below) |
| `kestrom rows lsq` | Output-layer rows of listed tokens, one least-squares solve toward target answers | 1 | Recall ≥ 0.5 on 7 of 10 refined endings, target 8 not reached (PLaMo 3 Translate 0.8B, polite to refined Japanese, earlier PyTorch form) |
| `kestrom rows ftpo` | Output-layer rows of the tokens of phrase events, by FTPO | 1 | Listed phrases 3,092 → 1,708, target ≤ 20% of the original not reached (Qwen3.5-9B) |
| `kestrom generate --rules` | Logits where hand-written rules over probed features fire (`kestrom rules fit`; `out` and `move` lines) | 1 | Refined endings 0.848 of the slots, broken 0.0135; with the fit rebalanced (balance 0.1, K 8) 0.925, broken 0.0185. The content after the first change kept 0.605 and 0.578, target 0.90 not reached (Qwen3.5-9B). `move` lines: a spelling rule with 0 errors in 200 new sentences (PLaMo 3 Translate, earlier PyTorch form) |
| `kestrom rules fit … STEPS` | The probes of the rules tuned by gradient, the rules unchanged | 1 | 0.892 of sentences on target, reached; the rest of the output kept 0.995 (PLaMo 3 Translate, earlier PyTorch form) |
| `kestrom generate --rules`: `in … prompt` and `add` lines (conditional steering) | One layer's output, when a condition read on the prompt fires | 1 | Polite sentences 0.641 → 0.0; JMMLU 0.72 → 0.72 (Qwen3.5-9B; below) |
| `kestrom generate --rules`: `phrase` lines | Goes back and bans the token that started a listed phrase | 1 | Listed phrases (`**`, `##`, three stock phrases) 3,092 → 0; JMMLU 0.74 → 0.68 at 1,024 tokens, target ≤ 1 point not reached. The model writes its answer in bold (**B**); banned there, it hedged or answered by name (Qwen3.5-9B). To drop markup only, strip it from the text after decoding |
| `kestrom generate --rules`: a `loopguard` line | Goes back at a thinking loop and forbids repeated 16-token pieces | 1 | Correct 4 → 26 of 100 (Qwen3.5-0.8B; below) |
| `kestrom trigger` | Spare rows of the embedding, learned as a trigger | 2 | With the trigger, polite sentences 0.641 → 0.017; without it, every logit of 150 prompts equal. JMMLU with the trigger 0.74 → 0.71, target ≤ 1 point not reached (Qwen3.5-9B) |
| `kestrom generate --rules`: `add` lines with `in … after` or `thinking` | One layer's output plus a direction, from the token that completes a given text, or while thinking | 1 | Not measured |
| `kestrom generate --rules`: `ban` lines | Logits of tokens set to −inf where a unit fires | 1 | Not measured. A ban is of tokens, not characters: banning 、 left the token 、「 free (Qwen3.5-0.8B); to keep a string out, use a `phrase` line |
| `kestrom edit subtract` | A direction out of the `down_proj` columns that write most of it | 2 | Not measured as a Kestrom command; Astra's published edit of this form is in the results below |
| `kestrom edit neuron-gain` | An MLP neuron's activation times k (its `up_proj` row) | 2 | Not measured |
| `kestrom edit head-gain` | An attention head's output times k (its columns of `o_proj`, or `out_proj` for linear attention) | 2 | Not measured |
| `kestrom edit head-proj` | What a head writes along a direction, removed, flipped or amplified (CHiPS) | 2 | Not measured |
| `kestrom edit keyed` | A rank-1 write into `down_proj` that a key opens at the positions to write and keeps shut on the others | 2 | Not measured |
| `kestrom edit neuron-write` | The least-used neurons rewritten to write a direction where a key opens (one neuron, or a capped pair) | 2 | Not measured |
| `kestrom rules write-mlp` | The rules written into neurons of the last MLP | 2 | 0.749–0.785 of sentences on target, target 0.835 not reached (PLaMo 3 Translate, earlier PyTorch form) |
| `kestrom train sft-data` | Fine-tuning data (prompt-completion pairs) from the edited outputs | 3 | A LoRA trained on it with mlx_lm: listed phrases −99%, but it also copied its training answers; JMMLU 0.70 against 0.74 (Qwen3.5-9B) |
| `kestrom merge` | A + λ (B − C), tensor by tensor | 3 | Qwen3.5-9B toward its Base at λ 0.25 and 0.5: listed phrases 3,527 and 3,716 against 3,092, no reduction |

The decode-time edits are all written in one rules language (`python -m pydoc kestrom.rules`) and combine freely. For
example, plain style for free-form questions only, no Chinese tokens in the answer after thinking, no `**`, and the loop guard:

```
in free: prompt style.npz 16        # a condition read on the prompt (kestrom steer direction)
in think: thinking
h1 f: all(free)
h1 answer: all(!think)
h2 plain: all(f)
h2 ans: all(answer)
add style.npz 16: 14.6*plain        # conditional steering
ban 的 了 们: ans                    # these tokens never, once thinking is over
phrase: "**"                        # go back and avoid it
loopguard: 512
```

`kestrom generate MODEL prompts.json out.jsonl --rules edit.dsl` records the events of every line, and `kestrom-verify compare
events` checks that each output is the original's until its first event. Rules that read probed features of the hidden state are
fitted first (`kestrom rules fit`).

To find where to edit, `kestrom find` reads the model and changes nothing: `direction` (the difference of two behaviours per
layer, with Cohen's d on held-out answers), `lens` (what each MLP neuron writes into the vocabulary, or the neurons that write
given words), `reverse` (neurons whose `down_proj` column points along the difference), `writers` (neurons by what they actually
write along a direction) and `heads` (heads by how much zeroing each one lowers a direction). These and the `edit` commands rest on known
methods (below) and add the declaration of the rows each edit wrote; each command's help says the choices we made.

Not included: orthogonalizing a direction out of every matrix that writes the residual (it removes refusals, which we do not
provide a tool for; on a style direction it broke Qwen3.5-9B, JMMLU 0.74 → 0.16).

"Earlier PyTorch form": the numbers come from the research code on PLaMo 3 Translate; the Kestrom command is its MLX port, checked
to run and to change only what it declares on Qwen3.5, not rerun for its effect. "Not measured": checked the same way, effect not
measured.

## Results

All on a Mac (MLX, and llama.cpp where noted). Every guarantee check of the tier-1 edits passed, and those reports were built twice
in separate processes and compared byte for byte.

**Ban Chinese tokens in Qwen3.5-9B's weights (tier 1).** 25,451 Chinese-only tokens, their rows of the output layer set to one fixed
row; no other byte changes. Mixing = share of 300 Japanese answers with a Chinese-only token or a kanji outside JIS X 0213, sampled with the model card's settings (temperature 0.7).
The guarantee checks are greedy.

| Runtime | Mixing, original → edited | JMMLU accuracy, original → edited |
|---|---|---|
| MLX, BF16 | 0.32 → 0.0067 | 0.685 → 0.69 |
| llama.cpp, BF16 GGUF | 0.273 → 0.0067 | 0.715 → 0.730 |
| llama.cpp, Q4_K_M (quantized after the edit) | 0.263 → 0.010 | 0.665 → 0.665 |

The range does not say whether each banned token *should* be banned: some also occur inside correct Japanese words (両性具有 came out
as 両性具). Read the list before you ship.

**Switch the style only where a condition fires (tier 1).** Qwen3.5-9B, conditional steering towards the plain (だ・である) style.
The condition fired on all 50 free-writing prompts and on none of 100 JMMLU questions. Polite sentences 0.641 → 0.0 on the free
writing; JMMLU 0.72 → 0.72; the 100 quiet prompts gave exactly the original's outputs (G2 150/150).

**Stop thinking loops at decode time (tier 1), against a published weight edit (tier 2).** Qwen3.5-0.8B on 100 MATH-500 problems,
thinking on, greedy, 16,384 tokens.

| | Correct | Hit the token limit |
|---|---:|---:|
| Original | 4 | 99 |
| [Astra's edit](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) (two `down_proj` tensors) | 8 | 95 |
| Loop guard (`kestrom generate`) | **26** | **72** |

The loop guard's outputs are the original's up to the first loop (G2 100/100). This is greedy decoding on 100 problems with the same
maximum output length (going back costs extra decoding steps), not Astra's published protocol (all 500 problems, 4 samples each,
temperature 0.6, presence penalty 1.5, 32,768 tokens), so it neither confirms nor refutes their numbers.
`kestrom-verify diff` on Astra's checkpoint: 2 of 488 tensors changed, 64 columns in each as published (131,072 elements, of which
130,188 changed their bytes); the README and the safetensors metadata changed too, so a declaration of the two tensors alone fails D.

**Acceptance reports for edits with no guarantee (tier 3).** `kestrom-verify measure` against the original, 150 prompts:

| Edit | Positions whose top token changed | Greedy answers still the same |
|---|---:|---:|
| Qwen3.5-9B, MLX 4-bit quantization | 12.7% | 0 / 150 |
| Qwen3-8B → Shisa V2.1 8B (public fine-tune) | 17.1% | 0 / 150 |

## Quickstart

```sh
pip install ./verify[mlx]              # from a clone of this repository (not on PyPI)

# the range of a row edit, from the bytes (safetensors folders or GGUF files)
kestrom-verify range ORIGINAL EDITED range.json

# greedy outputs: the original twice, the edited once, then compare against the range
kestrom-verify generate mlx ORIGINAL prompts.json base.jsonl
kestrom-verify generate mlx ORIGINAL prompts.json base2.jsonl
kestrom-verify generate mlx EDITED prompts.json edited.jsonl
kestrom-verify compare range.json base.jsonl base2.jsonl edited.jsonl ORIGINAL greedy.json

# any two models: every tensor compared, against what the editor declared
kestrom-verify diff ORIGINAL EDITED diff.json declared.json
kestrom-verify summary diff.json
```

`prompts.json` is `[{"id": ..., "ids": [prompt token ids]}, ...]`. See [docs/guarantees.md](docs/guarantees.md) for every command and
check, and [examples/](examples/) for three runs from start to end.

## What Kestrom does not guarantee

- That the range is *right*: whether a banned token should be banned, whether a condition fires on the right prompts.
- Agreement across runtimes. llama.cpp and MLX give different greedy outputs even for the same original weights. Compare the
  original and the edited model in the same runtime.
- Sampling. The guarantees are about teacher-forced logits and greedy decoding.
- Anything on the outputs for tier 2 and 3 edits: there the report only counts.
- Tested only on Apple Silicon (MLX, llama.cpp) with Qwen3.5 (0.8B, 9B) and Qwen3-8B. Not tested: Linux, CUDA, models whose output
  layer is tied to the embedding (`range` stops without `lm_head`).

## Related work

- Fine-Tuning Integrity (arXiv 2604.04738) proves cryptographically that a deployed model differs from a base only within a declared
  class of updates. Kestrom's `diff` checks the same kind of declaration in the clear, by recomputation, when both weights are at hand.
- Weight diff tools ([ckpt](https://github.com/stef41/ckpt), [weight-inspect](https://github.com/las7/weight-inspect)) compare
  tensors; Kestrom adds the declaration check, reports that rebuild to the same bytes, and output guarantees where the edit allows them.
- Conditional activation steering (CAST, arXiv 2409.05907), the Antislop backtracking sampler (arXiv 2510.15061), and
  transformers' `no_repeat_ngram_size` are the methods behind the `in … prompt` and `add` lines, the `phrase` lines, `rows ftpo` and the `loopguard` line; Kestrom adds
  the range and its check. Astra's edit (rikiyat.github.io/blog/weight-space-programming) is the form of `edit subtract`.
- `find`, `add` lines and `edit` use the logit lens (nostalgebraist, 2020), differences of means as directions (CAA, arXiv
  2312.06681), ablating and scaling heads (Michel et al., arXiv 1905.10650), MLP neurons as keys and values (Geva et al., arXiv
  2012.14913), and rank-1 writes kept off the inputs to preserve (ROME, arXiv 2202.05262; AlphaEdit, arXiv 2410.02355). The
  combination and the defaults of the neuron and head edits follow the documentation of
  [gguf-neuron-explorer](https://github.com/balaena01/gguf-neuron-explorer).
- Smoothie-Qwen also rewrites the output layer to suppress tokens.

## Licensing

- `kestrom-verify` (folder `verify/`): Apache-2.0. Anyone may use it, including companies, to check an edit.
- `kestrom` (folder `edit/`): your choice of PolyForm Small Business 1.0.0 or PolyForm Noncommercial 1.0.0. In short: Small
  Business covers a company with fewer than 100 people and less than 1,000,000 USD (2019) revenue, counted together with the
  organisations it controls, is controlled by or is under common control with; Noncommercial covers any noncommercial purpose,
  personal use without any anticipated commercial application, and charities, schools, public research organisations and
  governments. The license texts decide. Any other use needs a commercial license, priced by agreement: see
  [edit/COMMERCIAL.md](edit/COMMERCIAL.md).
- See [LICENSE.md](LICENSE.md).

## Contributing

Issues are welcome; pull requests are not accepted for now (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Work with us

CAPH TECH Inc. builds custom bounded edits (banned token lists, output formats, styles) and acceptance reports for edited or
fine-tuned models. Contact: Ryoichi Izumita, r.izumita@caph.jp.
