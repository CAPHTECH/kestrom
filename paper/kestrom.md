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
[5]) and stop a loop in the thinking (above). Where no unit of an action fires, the action leaves the computation as it was, so the
same events and the same check (G2) cover every combination. `kestrom rows lsq` and
`rows ftpo` rewrite output-layer rows toward target answers and are checked as row edits. In tier 2, `kestrom trigger` learns
spare embedding rows (that an input without them is computed as before follows for an untied embedding, and is measured, not
checked), `kestrom edit` scales a neuron or an attention head, removes what a head writes along a direction,
writes a rank-1 update into `down_proj` behind a key built to be zero on the positions to keep, rewrites the least-used neurons, or
subtracts a direction from a few `down_proj` columns (the form of the edit in §5.3); `kestrom find` ranks layers, neurons and heads
for these edits. They use the logit lens [8], differences of means [4], ablating and scaling heads [9], MLP neurons as keys and
values [10] and rank-1 writes kept off the inputs to preserve [11, 12]; the combination and the defaults of the neuron and head
edits follow the documentation of gguf-neuron-explorer [7]. Each `kestrom edit` and `rules write-mlp` writes the rows it changed as a declaration for D; for a trigger, declare its rows yourself. In tier 3, `kestrom train sft-data` (fine-tuning data
from the edited outputs) and `kestrom merge` are measured only. Except where §5 reports numbers, these tools were checked to run and to change only what they
declare (on Qwen3.5-0.8B and 9B), not for how well they do their job.

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

## 6. Related work

Fine-Tuning Integrity [1] proves, with zero-knowledge proofs, that a deployed model differs from a trusted base only within a declared
class of updates, without revealing the weights. Kestrom's `diff` checks the same kind of declaration in the clear, by recomputation,
when both models are at hand, and adds output guarantees where the edit permits them. Tensor diff tools such as ckpt and
weight-inspect [2] compare checkpoints; we add the declaration check, reports that rebuild to the same bytes, and the tiers.
Model provenance work (HuRef, provenance testing) asks whether a model derives from another; we ask what changed between two given
models. Activation steering [3, 4], backtracking samplers [5] and n-gram constraints are the methods behind our edits; what we add is
their range and its check. Smoothie-Qwen also rewrites the output layer to suppress tokens. Knowledge editing (ROME, MEMIT, AlphaEdit)
changes MLP weights in a few layers; such edits are tier 2 in our terms.

## 7. Limitations

- The guarantees are about teacher-forced logits and greedy decoding, on one machine and runtime. They do not carry across runtimes:
  llama.cpp and MLX give different greedy outputs for the same original weights.
- The range does not say that the edit is right inside it. A ban row is checked only at the positions it was made from: "at most −100"
  holds where it was measured, not at every possible input.
- G2 for decode-time edits trusts the events the decoding records, and `compare` trusts that its input files are the outputs they
  claim to be (see §2).
- Tested only on Apple Silicon, with Qwen3.5 (0.8B, 9B) and Qwen3-8B. Not tested: Linux, CUDA, models whose output layer is tied to
  the embedding (Qwen3.5-0.8B is tied, so its row edits are out of reach of `range`), quantization types other than Q4_K_M.
- The experiments are single runs of 100 to 300 prompts.
- Most tier-2 edits and the `find` commands are checked for what they change, not for their effect; several tools missed the targets
  we set for them (§5.5).

## Availability

Code: github.com/CAPHTECH/kestrom. `kestrom-verify` is Apache-2.0. `kestrom` is free for small businesses and noncommercial use
(PolyForm Small Business or Noncommercial 1.0.0); other uses need a commercial license. The edited Qwen3.5-9B is not
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
