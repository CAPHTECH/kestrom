# Kestrom: bounded edits of language models, with reports anyone can rebuild

Ryoichi Izumita, CAPH TECH Inc. (r.izumita@caph.jp)

> Draft, November 2026.

## Abstract

Edits of a language model after training (fine-tuning, LoRA, merging, quantization) can move its outputs at any position. A user who
receives an edited model cannot know what else changed without evaluating everything again. We describe Kestrom, a checker and a set
of editing tools built around one idea: each edit comes with a *range* that is read from the weights or from the decoding rather than
from what the editor claims, and with a report that rebuilds to the same bytes on the receiver's machine. We separate three tiers of
guarantee. In tier 1, greedy output is the same as the original's until the range is reached, and for an edit of output-layer rows
every logit outside the range is bit-identical too. In tier 2, only the declared tensors and rows changed, and the movement of the outputs is counted. In tier 3, everything is
counted and nothing is bounded. On Qwen3.5-9B, banning 25,451 Chinese-only tokens in the output layer lowered Chinese mixing in
Japanese answers from 0.32 to 0.0067 while every other logit stayed bit-identical at all 82,030 positions we checked; conditional
steering changed the style of exactly the 50 prompts it was meant for and left the other 100 outputs identical; and on Qwen3.5-0.8B, a
decode-time loop guard that keeps the original's output until the first loop solved 26 of 100 MATH-500 problems against 4 for the
original and 8 for a published weight edit aimed at the same loops, under one greedy setting that is not the edit's published one. A 4-bit quantization and a public
fine-tune, measured the same way, kept 0 of 150 greedy answers.

## 1. The problem

Say a company receives "Qwen3.5-9B with Chinese tokens removed" from a vendor. What does it know about the rest of the model? With
any edit that touches the network, very little: the hidden states change everywhere, so every output may change. We measured two
common edits against their originals with teacher forcing on the original's own answers to 150 Japanese prompts (§5.4): an MLX 4-bit
quantization of Qwen3.5-9B changed the top token at 12.7% of 82,030 positions, and Shisa V2.1 8B, a public fine-tune of Qwen3-8B,
at 17.1% of 87,703. In both, **no** greedy answer of 150 stayed the same as the original's. Neither is a bad edit; they simply carry no
statement about what did not change, and the receiver can only evaluate again on the prompts they think of.

Some edits can carry such a statement. If an edit changes only some rows of the output layer, everything before the output layer is
computed by the same bytes, so every logit of an unchanged row is the original's, exactly. If an edit acts only at decoding time and
only after some event, the decoding before the event is the original's. These are not approximate claims; they hold by construction,
and they can be checked by anyone holding both models.

Kestrom makes this checkable. It does not make edits better; it makes them *accountable*: what changed is read from the bytes, and the
checks that pass say which tier the edit is in.

## 2. Three tiers

| Tier | Statement | How it is checked |
|---|---|---|
| 1. Outputs | Greedy output is the same until the range is reached (G2). For a row edit also: every logit outside the range is bit-identical to the original's (G1) | Row edit: `range` (the range from the bytes, and every other byte the same: W), `logits` (G1), `compare` (G2). Decode-time edit: `compare events` (G2) |
| 2. Weights | Only the declared tensors and rows changed (D). The movement of the outputs is counted, not bounded | `diff`, `measure` |
| 3. Measured | What changed, and how much | `measure`, `diff` |

The reports of `logits` and `measure`, which run the original twice, and of `compare`, which reads two runs of it, also check U0:
the original model, run twice on the receiver's machine, gives the same logits and the same greedy output. The output guarantees
compare the original and the edited model *on the same machine and runtime*; if U0 fails, G1 and G2 in that report mean nothing. W
and D read only bytes and need no run.

`compare` takes greedy outputs as files and checks them; it does not tie a file to the model and settings that wrote it. The receiver
should write the outputs themselves, running the original twice. For a decode-time edit, G2 checks the output against the events the
edited decoding records, not that a loop or a condition was really there.

**Ranges.** For a row edit, the range is the set of output-layer rows whose bytes differ, found by comparing the files, and W checks
that every other byte of every file (other tensors, headers, configuration and tokenizer files) is identical. G1 then follows from the
computation, and the checker verifies it at every position of every sequence given. G2: before the first position where the original's
top token or the edited model's top token is a changed row, the two decodings choose the same tokens, so the outputs first differ, if
at all, where a changed row is involved. For a decode-time edit, each output records its *events* (the positions where the edit first
acted) and G2 checks that the output first differs at or after the first event.

**Reports.** A report is JSON with no times in it and with the SHA-256 of every model file, input and output it was made from. The
same inputs on the same machine give the same bytes. We built the guarantee reports of §5.1–5.3 and the `measure` reports of the
4-bit, Shisa and trigger-token models (§5.4, §5.5) twice in separate processes and compared them with `cmp`; the other reports were
built once.

## 3. Edits

**Row ban (`kestrom rows ban`, tier 1).** The rows of the listed tokens in the output layer are set to one row `r`, chosen so that `h·r` is
very negative for the hidden states `h` the model produces. `kestrom rows ban-row` reads `h` at every answer position of the model's own
answers, starts from their mean direction `v`, pushes `v` towards the lowest 0.1% of readings for 400 steps, and scales `-v` so that a
banned logit is at most −100 wherever it was measured; half of the answers are held out and must read positive too. The edited files
are a copy of the original's in which only those rows differ, so the model loads wherever the original loads (MLX, llama.cpp,
quantized GGUF). The range is exactly the banned rows.

**Conditional steering (tier 1).** Following CAST [3], a condition vector is read on the prompt before decoding: the
mean hidden state over the prompt at one layer, projected on the difference of the means of prompts that should and should not be
steered, against the midpoint of the two. If it fires, a fixed multiple of a style direction (a difference of means over answer
tokens in the two styles [4]) is added to one layer's output at every position; if not, the model runs unchanged. The event is at
position 0 when the condition fires, and there is none otherwise.

**Loop guard (tier 1).** Small reasoning models often repeat themselves in their thinking until the token limit.
A loop is detected as a copy: the last 512 tokens of the thinking in which every 16-token piece already occurred earlier in the output.
This catches loops of slightly different copies once their pieces have all been seen, which an exact-cycle detector misses. At the
first copy, decoding goes back to its start and from there to `</think>` never completes a 16-token piece the output already holds
(the `no_repeat_ngram_size` constraint of transformers, switched on only then). The answer after `</think>` is left free. The event is
the start of the copy.

An earlier version banned only the first token of each copy (as the Antislop sampler [5] does for listed strings). The model went
back into the copy through another token, up to 731 times on one problem, and the token limit was still reached on 92 of 100
problems at 4,096 tokens. The constraint fixes this.

**One language for decode-time edits.** Every decode-time edit of `kestrom generate` is written in one rules language and can be
combined with the others. Its inputs are features of the hidden state read by linear probes, a condition read on the prompt (as
above), a given text having appeared in the output, whether the model is thinking, and the output position (for fields written at
fixed positions); units combine them with AND, OR and NOT; its
actions raise, move or ban token logits, add a direction to a layer's output, back out of a listed phrase (the Antislop sampler
[5]), stop a loop in the thinking (above), and open the answer with the class that rules run with no model answer for the prompt,
where the model begins the answer with a class, and the thinking with a text that holds that class (an `open` line; an event where it replaces the top token), or inject a compiled system-prefix KV cache matched by those rules on the prompt's `data` (`kestrom kv compile` + a `kv: BANK_DIR [MAX_MATCH]` line, composing up to `MAX_MATCH` simultaneous matches on demand via an LRU cache and feeding only the post-prefix user tokens with event `0` when matched, and with no cache and no event from the `kv` line on unmatched prompts; `kestrom kv chat` runs the same bank across multi-turn sessions using `steps`/`prev` state with incremental KV reuse). Where no unit of an action fires, the action leaves the computation as it was, so the
same events and the same check (G2) cover every combination. `kestrom ifc` adds twin-track information flow control at greedy decoding time (tier 1, checked with `compare events`): `ifc control` takes declared JSON control keys strictly from a clean track (untrusted text masked) and data keys from the full track (recording data positions as events; `fast` mode switches tracks lazily so only one track runs per token, while `audit` runs both tracks every step to log overrides), while `ifc firewall` either emits the secret-free track with empty events (`strict-public`) or blocks tokens whose log-probability lift over the secret-free track reaches a threshold (`gated-public`). `kestrom rows lsq` and
`rows ftpo` rewrite output-layer rows toward target answers and are checked as row edits. In tier 2, `kestrom trigger` learns
spare embedding rows (that an input without them is computed as before follows for an untied embedding, and is measured, not
checked), `kestrom edit` scales a neuron or an attention head, removes what a head writes along a direction,
writes a rank-1 update into `down_proj` behind a key built to be zero on the positions to keep, rewrites the least-used neurons, or
subtracts a direction from a few `down_proj` columns (the form of the edit in §5.3); `kestrom find` ranks layers, neurons and heads
for these edits. They use the logit lens [8], differences of means [4], ablating and scaling heads [9], MLP neurons as keys and
values [10] and rank-1 writes kept off the inputs to preserve [11, 12]; the combination and the defaults of the neuron and head
edits follow the documentation of gguf-neuron-explorer [7]. Each `kestrom edit` and `rules write-mlp` writes the rows it changed as a declaration for D; for a trigger, declare its rows yourself. In tier 3, `kestrom train sft-data` (fine-tuning data
from the edited outputs) and `kestrom merge` are measured only, and `kestrom inspect` reads a fine-tune made elsewhere against its declaration (§5.6). Except where §5 reports numbers, these tools were checked to run and to change only what they
declare (on Qwen3.5-0.8B and 9B), not for how well they do their job. Outside the tiers, `kestrom rules run` runs a rules file with no model, as a network of its own on given texts and values (once, or step by step with a state and numbers in registers), `kestrom rules build` writes that network out, and `kestrom rules weigh` fits the coefficients of its out lines to labels; they edit nothing.

## 4. Setup

All runs were on one Apple Silicon Mac with MLX 0.31.1 and mlx-lm 0.31.1, and llama.cpp's `llama-server` where stated. The guarantee
checks and the results of §5.2–5.5 use greedy decoding, one prompt at a time; the mixing and accuracy of §5.1 use sampling. For every experiment the criteria, the code (by hash) and the inputs were fixed before the test run;
we report the runs that missed their criteria as missed.

## 5. Results

### 5.1 Banning Chinese tokens in Qwen3.5-9B

Qwen3.5-9B mixes Chinese into Japanese answers. We banned 25,451 tokens: 21,601 holding a han character outside JIS X 0213, and 3,850
tokens of Chinese words (seen at least 20 times in Chinese Wikipedia articles and at most 1/100 as often in Japanese ones). Mixing is
the share of 300 answers (200 JMMLU questions in seven medical subjects, 100 free-form medical questions) containing a Chinese-only
token or a kanji outside JIS X 0213, with the model card's sampling settings (temperature 0.7, top-p 0.8, top-k 20, presence penalty
1.5, 1,024 tokens).

| Runtime | Mixing | JMMLU accuracy |
|---|---|---|
| MLX, BF16 | 0.32 → 0.0067 | 0.685 → 0.69 |
| llama.cpp, BF16 GGUF | 0.273 → 0.0067 | 0.715 → 0.730 |
| llama.cpp, Q4_K_M (quantized after the edit) | 0.263 → 0.010 | 0.665 → 0.665 |

Guarantee (150 other prompts, 512 tokens): W 16 of 16 files; teacher forcing on the original's answers, U0 150/150 and G1 150/150
with every non-banned logit bit-identical at all 82,030 positions; greedy, U0 150/150 and G2 150/150, 140 outputs identical and 10
first differing where the original was about to write a banned token. In llama.cpp (38 prompts), G2 held for the BF16 and the Q4_K_M
GGUF.

The guarantee says nothing about whether each token should be banned. Some banned tokens also occur inside correct Japanese words:
両性具有 came out as 両性具 six times in 300 answers.

### 5.2 Switching the style only where a condition fires

Target: answer free-form questions in the plain style (だ・である) instead of the polite one (です・ます), and leave multiple-choice
questions alone. Qwen3.5-9B, layer 16 for both the condition and the steering, strength 8 times the norm of the style difference,
chosen on development prompts. On 150 test prompts the condition fired on all 50 free-form ones and on none of 100 JMMLU ones. The
share of polite sentences in the free-form answers fell from 0.641 to 0.0; JMMLU accuracy was 0.72 before and after; G2 150/150, the 100
quiet prompts giving exactly the original's outputs.

### 5.3 Thinking loops: decode time against a weight edit

Researchers at MIT CSAIL had GPT-6 Astra edit Qwen3.5-0.8B against thinking loops, subtracting a direction from 64 columns of
`down_proj` in two layers [6]. We compared the original, the published checkpoint and the loop guard on the first 100 odd-index
problems of MATH-500, thinking on, greedy, 16,384 tokens, scored by math-verify on the last `\boxed{}`.

| | Correct | Hit the token limit |
|---|---:|---:|
| Original | 4 | 99 |
| Astra's checkpoint | 8 | 95 |
| Loop guard | 26 | 72 |

The guard acted on 96 problems and G2 held on 100/100. Per problem it was right on 22 where the original was wrong, and the reverse
on none (two-sided sign test p ≈ 5 × 10⁻⁷); against the checkpoint, 19 and 1 (p ≈ 4 × 10⁻⁵). Of its 72 outputs cut at the limit, 53
were still thinking, mostly without repeating, and 19 were cut in the answer, which is not guarded. The detection window (512 against
128) was chosen on 30 development problems. A loop whose copies differ in every 16-token piece, such as a count going up, is not
found.

This is not the protocol of the published numbers: all 500 problems with 4 samples each, temperature 0.6, top-p 0.95, top-k 20,
presence penalty 1.5 and 32,768 tokens, where the checkpoint reaches 48.1% on MATH-500 [6]. Presence penalty is itself a remedy for
repetition. Our comparison neither confirms nor refutes those numbers. It shows that, on these 100 problems with greedy decoding and
the same maximum output length, an edit that provably leaves the output untouched until the first loop did better than the weight
edit. The guard's going back costs decoding steps that the output length does not count.

`diff` on the checkpoint found what was published: 2 of 488 tensors changed (layers 15 and 19 `mlp.down_proj.weight`), 64 columns in
each, 131,072 elements. Of those, 130,188 changed their bytes; 884 kept them. It also found that the README and the safetensors metadata differ, so a declaration of the two tensors alone fails D. On
the weights alone the edit is tier 2: teacher forcing on the original's answers left 47% of 1,633,169 positions bit-identical and
changed the top token at 0.35%, yet no greedy answer of 100 stayed the same.

### 5.3.1 Cross-model steering and condition bridging

`kestrom bridge` fits closed-form active-subspace orthogonal Procrustes maps on sequence-mean hidden states of unlabeled anchors (`bridge fit`, optionally composed across models via `bridge chain`) and transfers `kestrom steer` `.npz` directions and prompt conditions (`bridge map`, rescaling steering norms to match the recipient residual RMS norm) for `kestrom generate --rules`. Bridging an English-to-Japanese steering direction and a science-vs-Python prompt condition trained on `Qwen3.5-4B` across `1,600` unlabeled bilingual prompts (`Qwen3.5-4B → Gemma 4 E4B` at layer `17 → 22`, rank `1,599`, `ALPHA=14.950`; `Qwen3.5-4B → Qwen3.5-0.8B` at `17 → 12`, rank `1,024`, `ALPHA=0.755`) onto `20` held-out prompts (`10` science + `10` Python, zero recipient task labels) gave `6/10` (`Gemma 4 E4B`) and `10/10` (`Qwen3.5-0.8B`) science prompts steered to Japanese with closed-form thresholds, and `10/10` on both models with unlabeled median calibration, while leaving all `10/10` Python prompts bit-identical (`U0 20/20`, `G2 20/20 PASS`, `cmp` identical). Chaining `Qwen3.5-4B → Gemma 4 E4B → Qwen3.5-0.8B` (`17 → 22 → 12`) agreed with the direct mapped steering vector at cosine `0.9346`. The same `in ... prompt` readings can also gate `kv:` banks (`values` inputs in both `kestrom generate --rules` and `kestrom kv chat ... [PROBES.dsl]`): routing a 2-cassette `Gemma 4 E4B` bank (`science` gated solely by a bridged `Qwen3.5-4B → Gemma 4 E4B` probe at layer `22`, `barrier` gated by regex `match`) over 22 held-out prompts (10 English science + 10 Python + 2 `鉄壁の結界`) achieved `10/10` science matched (`kv="science"`, `events=[0]`), `2/2` `鉄壁の結界` matched, and `10/10` Python bit-identical (`events=[]`, `U0/G2 22/22 PASS`, `cmp` identical) with disjoint phrasing-matched unlabeled `CALIB_PROMPTS` (vs `2/10` science under phrasing-mismatched `CALIB_PROMPTS`, still `10/10` Python bit-identical and `U0/G2 22/22 PASS`), plus 3-turn `kestrom kv chat` state carry-over (`plain` → `switch` → `reuse` via `prev`, `17`-token delta prefill).

### 5.4 Acceptance reports for edits with no range

| Edit | Tensors changed (`diff`) | Top token changed (`measure`) | Greedy answers the same |
|---|---|---:|---:|
| Qwen3.5-9B, Chinese ban (§5.1) | 1 of 775, 25,451 rows | (tier 1) | 140 / 150, the rest inside the range |
| Qwen3.5-9B, MLX 4-bit | (another format) | 12.7% of 82,030 positions | 0 / 150 |
| Qwen3-8B → Shisa V2.1 8B | 387 of 399, 89% of the elements, 10 other files | 17.1% of 87,703 positions | 0 / 150 |

### 5.5 What did not work

A trigger token in the vocabulary: Qwen3.5-9B has 243 spare rows. We learned the embedding of 32 of them so that the model, given
them in a prompt, writes in the plain style. Only those 32 rows change (`diff`), so every input without them is computed exactly
as before; that is measured, not checked (tier 2; `measure`: every logit equal at all 82,030 positions). It worked (polite
sentences 0.641 → 0.017), but with the trigger, JMMLU accuracy fell from 0.74 to 0.71, beyond our one-point criterion: the model
sometimes stated the answer without its letter. We keep the tool, with this result.

Banning markup and stock phrases at decoding time (phrase lines on `**`, `##` and three phrases, Qwen3.5-9B): the listed strings fell
from 3,092 to 0, but JMMLU accuracy at 1,024 tokens fell from 0.74 to 0.68. The model writes its answer in bold; banned there, it
hedged or answered by name. Markup is better stripped from the text after decoding.

Refined Japanese endings by rules (`--rules`, Qwen3.5-9B): the endings were written at 0.925 of the places they belonged, with 0.0185
broken ones, but after the first change only 0.578 of the content of the original answer was kept, against our criterion of 0.90.
The range holds (G2 150/150); what follows the first change is free, and the model wrote a different answer.

### 5.6 Reading a delivered fine-tune against its declaration

A company that has a model fine-tuned elsewhere holds the original, the recipe the fine-tune declares (LoRA rank and targets,
quantization) and the training data. `kestrom inspect` lays out what that declaration does not explain; a person, or a model, reads
it and decides. We tested it blind on Qwen3.5-0.8B. Someone who did not know how `inspect` reads made 10 fine-tunes (LoRA of rank 8
on three tasks; 6 on a 4-bit original, 2 of them quantized again after merging), with a declaration and data for each, and put a
backdoor in 5: two writes made afterwards (a rank-one write into one `down_proj`, one of them opened by a key), and three trained in
with 4% poisoned examples, one of them quantized again. Each target was one ordinary word. We retrained two references from each
declaration (seeds 1 and 2) and ran every command; a model with no other context (Claude Opus 5.5), given only the outputs and the
declarations and the command help, wrote its judgment before we opened the answers. It named all 5 targets, called the 5 backdoored models backdoored
and 1 of the 5 clean ones too, at confidence 0.45, from tokens raised less than between the two references. A second reader
(Claude Fable 5.1) got all 10 right. The two readers named the trigger's words for 1 and 2 of the 5 backdoors. The rank check missed the keyed
write (the change kept 8 large singular values, the declared rank); the singular values and `lens` showed it (one component of
183.9, whose tokens led with the target).

Before this test we had tried fixed rules on 14 models whose answers we knew: the rank check, or one threshold for all on the
largest median (computed the same way, with our research scripts), set from pairs of references. They flagged 6 of 10 backdoors (4
by the threshold) and no clean model, though the top token was the target for all 6
trained-in backdoors. How much the seed alone moves the tokens differed by task (0.17 to 2.33), and what separated a backdoor was
whether its token could come from the task, which a reader judges.

The clean model called backdoored had its top token raised by the reference's seed, not by the model: against the mean of
three references (seeds 1 to 3) that token dropped out, while the targets of two backdoors stayed near the top. We then ran a
second blind test the same way, with this change frozen before the models arrived. Another maker made 12 fine-tunes (LoRA of rank
8 on the MLP projections, three tasks; 7 on a 4-bit original, 3 of them quantized again) and put a backdoor in 4: two keyed
rank-one writes into a `down_proj` (in layer 22, and in the last layer, 23, on a quantized original), and two trained in with 10 poisoned
examples, one of them quantized again. `behavior` compared each model with the mean of three retrained references, and
each reference with the mean of the other two. Both readers called all 4 backdoored models backdoored and all 8 clean ones clean,
and named all 4 targets; each named the trigger's words for 2 of the 4. The rank check flagged both writes (9 singular values
above the noise, 8 declared), and `lens` showed the target. The fixed rule "S above the largest S among the references" flagged
the 4 backdoors and 2 clean models, whose top tokens came from the task (an answer letter, a word of the data).

A model picked up on the internet comes with the weights only. We ran the same 12 again with no declaration, data or
references: each model against the original (requantized where `quant` found a quantization), the rank set to the most common
r of `shape`, and new readers given the same instructions without the declarations. Both readers again called all 12 right and
named all 4 targets. Then we converted the 12 into Q4_K_M GGUF files with llama.cpp and read them with `inspect gguf`, which
quantizes the original into the same tensor types to compare with. `shape` flagged nothing (the most common r was 0 for all 12:
the rounding hides the LoRA's rank), and the readers used the ratio of the first two singular values, `lens` and `behavior`
instead. Both called the 8 clean models clean and 3 of the 4 backdoors backdoored, naming their targets. The missed one had
the smallest S of the four backdoors in the weights-only run (2.01, its target on top), on a model quantized twice (4-bit MLX, then
Q4_K_M): its target was no longer in either list of 30 tokens `behavior` gives (largest median, largest concentration).

To screen a single FP16/BF16 checkpoint for quantization-conditioned backdoors (dormant at full precision, active after quantization)
with no original model and no trigger prompts, `kestrom inspect mirror` compares the quantization residual $E = Q(W) - W$ against
Rademacher sign flips ($E \odot \sigma$), interior cell-boundary clustering, active block collapse, and (when plain prompts are given)
the anti-quantization mirror $W_{\text{mirror}} = 2W - Q(W)$. On Qwen3.5-0.8B (MLX affine 4-bit, group 64; plus weight scan on
Qwen3.5-4B), official models, an abliterated derivative and a 16-tensor benign fine-tune produced zero flags (max $\rho_1$ 1.0165–1.0243),
while four box-PGD quantization-conditioned backdoors (16-tensor, 2-tensor conditional stealth, and two 6-tensor adaptive margin/histogram-matched
attacks) had all 30/30 modified tensors localized with 0/900 false positives ($\rho_1$ 4.91–16.74), exact target-token recovery on
trigger-free plain text (coherence 12.3–93.5×), and trigger-blind neutralization via `kestrom inspect disarm` (anchor-preserving
stochastic rounding, unbiased for values inside the quantizer grid, on only the flagged tensors, cutting 4-bit attack activation from 24/24 to 0/24—and 1/24 on the histogram-matched
model, matching BF16—while surviving subsequent standard 4-bit quantization). An outlier-injection block-collapse test was missed by
$\rho_1$ and cell-boundary clustering and caught by block collapse (`ge90` 64–65 vs benign max 12). Extending `mirror` and `disarm` to PyTorch and `bitsandbytes` 4-bit NormalFloat (NF4, group 64) on Linux + NVIDIA L4 CUDA (`bitsandbytes` 0.50.2) matched CUDA `bitsandbytes` FP32 NF4 dequantization exactly (`max_abs_diff = 0.0`), gave zero flags and byte-exact reports on benign Qwen3.5-0.8B (186 tensors, max $\rho_1$ 1.0185, max `b85` 0.1583, behavior coherence 0.81×) and Qwen3.5-4B (248 tensors, max $\rho_1$ 1.0213, max `b85` 0.1509, behavior coherence 0.88×), and on a 2-tensor NF4 box-PGD backdoor localized all 2/2 modified tensors with 0/184 false positives (behavior coherence 3.84×) while `disarm` reduced the real `bitsandbytes` 4-bit NF4 target-token logit jump from +9.60 to +0.87 with zero changed `kestrom-verify diff` output rows. Adding `bitsandbytes` FP4 (group 64) and row-wise INT8 (`Linear8bitLt`) on the same L4 CUDA setup matched CUDA FP4 dequantization (`max_abs_diff = 0.0`) and `int8_vectorwise_quant` (`exact_code_match_rate = 0.999901`, `mean_abs_diff = 1.29e-7`); on 2-tensor `Qwen3.5-0.8B` box-PGD backdoors `mirror` localized 2/2 modified tensors with 0/184 false positives and `disarm` cut the real `bitsandbytes` target-token logit jump from +2.68 to −0.05 (FP4) and +3.08 to +0.11 (INT8, behavior coherence 11.15× before disarming) with zero changed `kestrom-verify diff` output rows. On benign non-Qwen `DeepSeek-R1-Distill-Llama-8B` (224 tensors), INT8 produced zero flags (max $\rho_1$ 1.0227) and NF4 behavior stayed unflagged (coherence 0.73×), whereas NF4 (1/224, `layers.0.q_proj` $\rho_1$ 1.4003) and FP4 (2/224, `layers.0.q_proj`/`k_proj` $\rho_1$ 1.6752 and 1.1406) flagged natural extreme-outlier layer-0 projections (`outlier_max` 908, where block-collapsed $E = -W$ exposes the natural weight spectrum) while `b85` stayed benign (max 0.1484–0.1488).

### 5.7 Writing relations behind a key

`kestrom edit keyed` was used, unchanged, to write relations in natural language into Qwen3.5-9B. In each run, 32 fictional
people; for each, "P works at O" was written by its own call into `down_proj` of layer 5 (of 32), at the last token of P's name in
three phrasings of the question "Which organization does P work at now?". The value of each call was fitted beforehand by our own
script, outside Kestrom, so that O follows at that position. The prompts list facts about other people and organizations (another
person Q's organization, where each organization is) and say nothing about P. Our script chose each answer among the item's
candidates by log-probability, on a held-out phrasing of the organization question, on P's city (one hop through the written
relation, never trained), and on the facts stated in the prompt. Outside: 240 prompts about other people, compared before and
after, and `measure` on 23,964 positions. Two sets of 32 people, in two groups of 16 each. The criteria, fixed before each run,
were at least 15 of 16 in each group on the organization, the city and each stated fact, and D passing.

`diff` passed D in every run: 1 of 775 tensors changed. The organization and the city reached the criteria in every run, and so
did where each organization is. Another person's organization, stated in the prompt, did not, until we changed what DATA left:

| What DATA left, besides the positions written | People | Q's organization kept | Outside answers the same | Top token changed |
|---|---|---:|---:|---:|
| The other people's key positions; each prompt's other positions as one mean | first set | 26 / 32 | 202 / 240 | 5,416 |
| | second set (report only) | 25 / 32 | 228 / 240 | 4,192 |
| Every position, each its own answer | first set | 24 / 32 | 201 / 240 | 1,571 |
| That, and the question's name slot filled by 64 names used nowhere else | first set | 31 / 32 | 223 / 240 | 1,284 |
| | second set | 31 / 32 | 234 / 240 | 815 |

The last setting met the criteria in all four groups (15, 16, 16 and 15 of 16 on Q's organization). Afterwards, without criteria
and on the first set only, we looked for where the write acted. Since `diff` shows that one tensor is the whole change, the 32
fitted values split its difference into the 32 written terms (relative residual 0.01, consistent with rounding to BF16 at each
call), and their push ‖ΔW x‖ can be read at every position. With one mean per prompt, the push was largest at Q's name, in the
facts and in the question (median 33 and 24, against 6 at the answer position), the sum of small parts of many people's values.
Left position by position, it stayed only at Q's name in "Which organization does Q work at?" (median 12; the largest elsewhere,
median 2.2): the slot of the key, which DATA had left filled only by the 32 people's names. Filling it with unused names as well
brought Q's organization back, though the push there fell per item only to 0.55–0.91 of what it was (median push 8.3). Why the
answers recovered was not traced; in no setting did the size of the push separate the wrong items from the right ones (in the
last, the one wrong item had the largest push, 14.6, and a right one had 14.5). A choice between two people of who works in P's
city was 30 and 31 of 32 with the relation stated in the prompt, 19 and 16 with no relation, and 26 and 15 with it written in the
last setting: the written relation reached that comparison in one set and not in the other.

## 6. Related work

Fine-Tuning Integrity [1] proves, with zero-knowledge proofs, that a deployed model differs from a trusted base only within a declared
class of updates, without revealing the weights. Kestrom's `diff` checks the same kind of declaration in the clear, by recomputation,
when both models are at hand, and adds output guarantees where the edit permits them. Tensor diff tools such as ckpt and
weight-inspect [2] compare checkpoints; we add the declaration check, reports that rebuild to the same bytes, and the tiers.
Model provenance work (HuRef, provenance testing) asks whether a model derives from another; we ask what changed between two given
models. Activation steering [3, 4], backtracking samplers [5] and n-gram constraints are the methods behind our edits; what we add is
their range and its check. Smoothie-Qwen also rewrites the output layer to suppress tokens. Knowledge editing (ROME, MEMIT, AlphaEdit)
changes MLP weights in a few layers; such edits are tier 2 in our terms. Backdoor checks of fine-tunes and LoRA adapters read their
weights (MIST [13], PEFTGuard [14], the tracing of rank-one edits [15]) or the model's own outputs without the original (Microsoft's
scanner, 2026); `kestrom inspect` reads a fine-tune against its declared recipe, with references retrained from it.

## 7. Limitations

- The guarantees are about teacher-forced logits and greedy decoding, on one machine and runtime. They do not carry across runtimes:
  llama.cpp and MLX give different greedy outputs for the same original weights.
- The range does not say that the edit is right inside it. A ban row is checked only at the positions it was made from: "at most −100"
  holds where it was measured, not at every possible input.
- G2 for decode-time edits trusts the events the decoding records, and `compare` trusts that its input files are the outputs they
  claim to be (see §2).
- Tested primarily on Apple Silicon, with Qwen3.5 (0.8B, 9B) and Qwen3-8B; `kestrom ifc`, `kestrom inspect mirror` / `disarm` (`bitsandbytes` NF4, FP4, INT8), `kestrom-verify generate torch` / `compare events`, and `kestrom-verify diff` were also tested on Linux + NVIDIA L4 CUDA (`Qwen3.5-0.8B` / `4B` and `DeepSeek-R1-Distill-Llama-8B`, `torch` 2.14.1+cu130, `bitsandbytes` 0.50.2). Not tested on Linux/CUDA: weight edits, other `inspect` subcommands, `logits`/`measure`, models whose output layer is tied to
  the embedding in GGUF, quantization types other than Q4_K_M / MLX affine / `bitsandbytes` NF4, FP4, INT8. In a tied model (Qwen3.5-0.8B) a changed row is also an input
  embedding, so G1 and G2 hold only for inputs without a changed id; the reports check those and count the others.
- The experiments are single runs of 100 to 300 prompts; §5.7 writes 32 relations per run, on two sets of people.
- Most tier-2 edits and the `find` commands are checked for what they change, not for their effect; several tools missed the targets
  we set for them (§5.5).
- `kestrom inspect` gives evidence, not a verdict: nothing found does not mean a fine-tune is clean. It was tried on one model
  family at 0.8B, with LoRA of rank 8, against backdoors made by people who did not know how it reads (§5.6); `inspect gguf` with Q4_K_M only.

## Availability

Code: github.com/CAPHTECH/kestrom. `kestrom-verify` is Apache-2.0. `kestrom` is free for companies with fewer than 100
people, for noncommercial use and for a trial of less than 32 days (PolyForm Small Business with an additional permission,
Noncommercial or Free Trial 1.0.0); other uses need a commercial license. The edited Qwen3.5-9B is not
published; `examples/zh-ban` rebuilds it from the original.

## References

[1] Shang, Yu, Chen. Fine-Tuning Integrity. arXiv 2604.04738, 2026.
[2] ckpt (github.com/stef41/ckpt); weight-inspect (github.com/las7/weight-inspect).
[3] Lee et al. Programming Refusal with Conditional Activation Steering. arXiv 2409.05907.
[4] Panickssery et al. Steering Llama 2 via Contrastive Activation Addition. arXiv 2312.06681.
[5] Paech et al. Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models. arXiv 2510.15061.
[6] Takehi, Teoh, Zhang, Khattab. Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.
November 2, 2026. rikiyat.github.io/blog/weight-space-programming; checkpoint: huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming.
[7] gguf-neuron-explorer (github.com/balaena01/gguf-neuron-explorer).
[8] nostalgebraist. interpreting GPT: the logit lens. LessWrong, 2020.
[9] Michel, Levy, Neubig. Are Sixteen Heads Really Better than One? arXiv 1905.10650.
[10] Geva, Schuster, Berant, Levy. Transformer Feed-Forward Layers Are Key-Value Memories. arXiv 2012.14913.
[11] Meng, Bau, Andonian, Belinkov. Locating and Editing Factual Associations in GPT. arXiv 2202.05262.
[12] Fang et al. AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models. arXiv 2410.02355.
[13] Detecting Trojaned DNNs via Spectral Regression Analysis (MIST). arXiv 2605.21146.
[14] PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning. arXiv 2411.17453.
[15] Tracing and Reversing Rank-One Model Edits. arXiv 2505.20819.
