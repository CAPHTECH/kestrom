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
| `kestrom` | `kestrom` | The editing tools whose range the checker can bound | Free for companies with fewer than 100 people, for noncommercial use and for a trial of less than 32 days (PolyForm); other uses need a commercial license |

## Three tiers of guarantee

The checks that pass say which tier an edit is in. Kestrom does not blur them.

| Tier | What is checked | Commands | Edits |
|---|---|---|---|
| 1. Outputs | Greedy output is the same until the range is reached (G2); for a row edit, every logit outside the range is also bit-identical to the original (G1). With an output layer tied to the embedding, both for inputs with no changed id | Row edits: `range`, `logits`, `compare`. Decode-time edits: `compare events` | `kestrom rows` (rows of the output layer), `kestrom generate` (decode-time edits: rules—including probed models fitted at a decoder layer and transferred via `kestrom bridge rules`, conditional steering including directions/conditions mapped via `kestrom bridge`, adding a direction after a given text, phrase bans, loop stopping, `open` lines, compiled fact-KV routing via `kestrom kv compile` + `kv:` lines), `kestrom ifc` (twin-track control-flow non-interference and secret leakage firewall) |
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
for several tools it fell short. Each command's help (`kestrom`, `kestrom <command> --help`, `python -m pydoc kestrom.<module>`) has the details.

| Command | What it edits | Tier | What we measured (model) |
|---|---|---|---|
| `kestrom rows ban` | Output-layer rows of listed tokens set to a ban row | 1 | Chinese mixing 0.32 → 0.0067 (Qwen3.5-9B; below) |
| `kestrom rows lsq` | Output-layer rows of listed tokens, one least-squares solve toward target answers | 1 | Recall ≥ 0.5 on 7 of 10 refined endings, target 8 not reached (PLaMo 3 Translate 0.8B, polite to refined Japanese, earlier PyTorch form) |
| `kestrom rows ftpo` | Output-layer rows of the tokens of phrase events, by FTPO | 1 | Listed phrases 3,092 → 1,708, target ≤ 20% of the original not reached (Qwen3.5-9B) |
| `kestrom generate --rules` | Logits where hand-written rules over probed features fire (`kestrom rules fit`; `out` and `move` lines) | 1 | Refined endings 0.848 of the slots, broken 0.0135; with the fit rebalanced (balance 0.1, K 8) 0.925, broken 0.0185. The content after the first change kept 0.605 and 0.578, target 0.90 not reached (Qwen3.5-9B). `move` lines: a spelling rule with 0 errors in 200 new sentences (PLaMo 3 Translate, earlier PyTorch form) |
| `kestrom rules fit … STEPS` | The probes of the rules tuned by gradient, the rules unchanged | 1 | 0.892 of sentences on target, reached; the rest of the output kept 0.995 (PLaMo 3 Translate, earlier PyTorch form) |
| `kestrom generate --rules`: `in … prompt` and `add` lines (conditional steering) | One layer's output, when a condition read on the prompt fires | 1 | Polite sentences 0.641 → 0.0; JMMLU 0.72 → 0.72 (Qwen3.5-9B; below) |
| `kestrom bridge {fit, map, rules, chain}` | Fits closed-form active-subspace orthogonal Procrustes maps on sequence-mean hidden states of unlabeled anchors (`fit`, optionally composes `A→B→C` via `chain`), maps a donor `kestrom steer` `.npz` (`direction`, `cond`, `threshold`) with RMS scale matching (`map`, optional unlabeled `CALIB_PROMPTS` median threshold), and transfers a donor `kestrom rules fit ... LAYER` `.npz` (`P`, optional `bias`, `layer`) across paired decoder layers (`rules`, closed-form or unlabeled token-stream Z-score calibration on `CALIB_DOCS`) for `kestrom generate --rules` | 1 (via `generate`) | Steer transfer (`Qwen3.5-4B` `20` EN→JA pairs, `15+15` science vs Python prompt condition, `1,600` unlabeled anchors, `20` held-out prompts): closed-form gave `6/10` science→Japanese on Gemma 4 E4B (layer `22`, `ALPHA=14.950`) and `10/10` on Qwen3.5-0.8B (layer `12`, `ALPHA=0.755`) with `10/10` Python bit-identical; unlabeled median calibration gave `10/10` science→Japanese and `10/10` Python bit-identical (`events=[]`, `U0/G2 20/20 PASS`, `cmp` identical) on both; chaining `4B→Gemma 4 E4B→0.8B` (`17→22→12`) gave steering cosine `0.9346`. Rules probe transfer (`kestrom rules fit ... 20` on `Qwen3.5-4B` → `kestrom bridge rules`, zero recipient token labels): across held-out token streams (`1,069` Gemma 4 E4B tokens at layer `26`, `1,019` Qwen3.5-0.8B tokens at layer `15`), `in_code` AUC was `0.9905` / `0.9901` (F1 `0.8156` closed-form → `0.9238` Z-score on `24` unlabeled docs for Gemma 4 E4B; `0.9225` closed-form / `0.9165` Z-score for Qwen3.5-0.8B) and `is_ja` F1 was `1.0000` (Z-score) on both. Two-stage prompt condition × bridged `is_en` token probe gating `move . -> !: on` on `12` held-out prompts (`6` EN + `6` JA, `--max-tokens 48`): `6/6` EN fired with `!` added, `6/6` JA stayed bit-identical (`events=[]`), `U0/G2 12/12 PASS`, and `cmp` identical on both Gemma 4 E4B and Qwen3.5-0.8B |
| `kestrom generate --rules`: `phrase` lines | Goes back and bans the token that started a listed phrase | 1 | Listed phrases (`**`, `##`, three stock phrases) 3,092 → 0; JMMLU 0.74 → 0.68 at 1,024 tokens, target ≤ 1 point not reached. The model writes its answer in bold (**B**); banned there, it hedged or answered by name (Qwen3.5-9B). To drop markup only, strip it from the text after decoding |
| `kestrom generate --rules`: a `loopguard` line | Goes back at a thinking loop and forbids repeated 16-token pieces | 1 | Correct 4 → 26 of 100 (Qwen3.5-0.8B; below) |
| `kestrom generate --rules`: an `open` line | The output opens with the class that rules of their own (`kestrom rules run`, no model) answer for the prompt's `data`, then a line feed: the top token is kept where the text still fits, else the top token that fits is taken (an event); where the rules answer none, nothing, or a class given in the line. The answer is what follows the last `</think>`; where the top token that begins it starts no class of the head nor the class given in the line, nothing. With a text in the line, the thinking also opens with that text, the class in it | 1 | Qwen3.5-4B writing a class on line 1 and a reason on line 2, opened by agent-written rules that route Japanese questions about building energy (the rules of `kestrom rules run` below): on 24 new questions (8 plain, 8 oddly worded, 8 vague or unrelated; written by one agent, kept where a second gave the same answer) line 1 was right on 13 unopened, 15 opened with the rules' class, 22 when also opened with `clarify` where the rules answer none; no line 1 went from right to wrong; G2 24/24 in both forms. One question the rules could not read (a typo) was opened with `clarify` though it had an answer; the model alone was wrong there too, and how often it would have been right is not measured. In a run of the same method by an earlier script, one judging agent found the reason newly contradicting line 1 or misreading the question on 1 of the 24. Four chat questions with no request for a class (two naming a word the example router reads) kept the original's output, token for token (the example). With thinking on and at most 2,048 tokens, on the same 24 questions: line 1 right on 4 unopened (19 outputs never closed the thinking), 4 with the answer opened (every output the original's), 11 with one sentence holding the rules' class (`clarify` where they answer none) also written at the start of the thinking (8 that had not closed the thinking became right, 1 that was right no longer closed it; 13 never closed the thinking, and none more did with a loop guard of 512); G2 24/24 in each form, where it covers little of an output whose thinking was opened. All three counts are below those of the same questions with thinking off (13 unopened, 22 opened with `clarify` where the rules answer none); of those 8, 6 are questions the unopened model has right with thinking off and 2 are ones it has wrong. One wording of the sentence; the unopened model at a higher limit is not measured. One model; the agents that wrote the rules and the questions, labelled and judged are of one family. [examples/rules-open](examples/rules-open/) runs the mechanism on eight other questions |
| `kestrom kv compile`, `kestrom kv chat`, `kestrom generate --rules`: a `kv` line | Compiles per-fact system-prefix KV caches from standalone rules and `FACTS.json` (zero weights changed); at decode time (`kv: BANK_DIR [MAX_MATCH]`), evaluates the bank's rules on `data` plus any `in NAME: prompt FILE LAYER` conditions (including cross-model probes mapped by `kestrom bridge map`) passed as 0/1 `values`, injects the compiled KV cache when `1` to `MAX_MATCH` entries match (composing ≥2 on demand with an in-memory LRU cache) and records event `0`, while unmatched or out-of-range prompts get no cache and no event (`events=[]` where no other line acts). `kestrom kv chat` runs multi-turn sessions (`steps`/`prev` + optional `PROBES.dsl`) with incremental KV reuse when the topic holds and automatic re-prefill or rebuild on switches (tier 1 here is for `kestrom generate` with a `kv` line) | 1 | Gemma 4 E4B on 149 fresh unseen Japanese building-fact questions across 27 cells (9 floors × 3 categories, scored by 4 blind readers, research script): 132/149 raw right (130/149 with answer check), 4 wrong, vs 89/149 with no knowledge text and 137/149 with the matching fact in the prompt; recompiling 6 edited cells took 0.19s (0.95s for all 27) with all 21 unedited KV caches bit-identical (~976.9 KB/fact). CLI verification via `kestrom kv compile` + `kestrom generate --rules`: on Gemma 4 E4B (27 facts, `mean_kb` 981.1 KB; incremental recompile of 6 edited facts reused 21/21 unchanged `.safetensors` files bit-for-bit, `mean_kb` 989.4 KB, identical `manifest.json`), 12 prompts (6 matching + 6 general chat): U0 12/12 and G2 12/12 PASS (all 6 unmatched bit-identical with `events=[]`; all 6 matched with `kv` set and `events=[0]`); on Qwen3.5-0.8B (`examples/rules-run/router.dsl`, 2 facts, `mean_kb` 10,048.4 KB due to linear-attention `ArraysCache`), 4 prompts (2 matching + 2 chat): U0 4/4 and G2 4/4 PASS (all 2 unmatched bit-identical); all `compare events` reports rebuilt byte-for-byte with `cmp`. Multi-domain composed `kv: BANK 2` + 5-turn `kestrom kv chat` (stateful `prev` DSL, 2 domains): on Gemma 4 E4B, 4/4 U0/G2 PASS (`cmp` identical; composed prompt recalled both keys) and 5/5 chat turns accurate (`plain` → `switch` → incremental `reuse` on follow-up pronoun with 25-token delta prefill → `compose` recalling both → `switch`); on Qwen3.5-0.8B (`mlx_lm 0.31.1`), 5/5 chat turns accurate with automatic `rebuild` on history re-tokenization and `ArraysCache` composition. Hybrid regex + bridged prompt-probe routing (`Qwen3.5-4B → Gemma 4 E4B` layer `22` mapped by `kestrom bridge map`, 2-entry bank, `mean_kb` 2,028.6 KB, 22 held-out prompts = 10 science + 10 Python + 2 `鉄壁の結界`): with disjoint phrasing-matched unlabeled `CALIB_PROMPTS` (`threshold=-7.8604`), **10/10** science got `kv="science"` (`events=[0]`, all 10 replied in Japanese with the cassette tag), **2/2** `鉄壁の結界` got `kv="barrier"`, **10/10** Python were bit-identical (`kv=null`, `events=[]`), and `U0/G2 22/22 PASS` (`cmp` identical), plus 3-turn `kestrom kv chat ... PROBES.dsl` went `plain` (`p_sci=0`, Python) → `switch` (`p_sci=1`, `kv="science"`) → `reuse` (`p_sci=0` follow-up carried by `prev`, 17-token delta prefill); phrasing-mismatched `CALIB_PROMPTS` (`threshold=-6.1144`) triggered only `2/10` science while keeping `10/10` Python bit-identical and `U0/G2 22/22 PASS`. Mechanism is cached system-turn KV context, not a weight edit; G2 covers nothing after position 0 on matched prompts |
| `kestrom trigger` | Spare rows of the embedding, learned as a trigger | 2 | With the trigger, polite sentences 0.641 → 0.017; without it, every logit of 150 prompts equal. JMMLU with the trigger 0.74 → 0.71, target ≤ 1 point not reached (Qwen3.5-9B) |
| `kestrom generate --rules`: `add` lines with `in … after`, `thinking` or `position` | One layer's output plus a direction, from the token that completes a given text, while thinking, or before given output positions | 1 | Not measured |
| `kestrom generate --rules`: `ban` lines | Logits of tokens set to −inf where a unit fires | 1 | Not measured. A ban is of tokens, not characters: banning 、 left the token 、「 free (Qwen3.5-0.8B); to keep a string out, use a `phrase` line |
| `kestrom ifc control` | Twin-track greedy decoding separating JSON `CONTROL` keys (from clean Track B, untrusted data masked) and `DATA` keys (from full Track A, clamping single tokens that cross back into control syntax). `fast` (default) switches tracks lazily so only one track runs per token; `audit` runs both tracks every step and logs every `control_override`. Runs on `mlx` (default) or `torch` (CUDA, MPS, CPU) | 1 | 40 realistic JSON tool-calling prompts (10 benign, 30 prompt injections; system prompt lists available tools, MLX): on Qwen3.5-0.8B, control hijacks 23/30 → 0/30 (benign control 9/10 → 10/10, summary hits 10/10 benign and 27/30 inject; wall-clock time `fast` 1.43× vs `audit` 1.91× over unprotected `base`); on Qwen3.5-4B, control hijacks 12/30 → 0/30 (benign 10/10 with 0 overrides, summary hits 10/10 benign and 28/30 inject; wall-clock time `fast` 1.09× [50.1s vs 45.9s] vs `audit` 1.66× [76.4s]); U0 40/40 and G2 40/40 PASS on both models and both modes via unmodified `kestrom-verify compare events` (reports rebuilt byte for byte). On Linux + NVIDIA L4 CUDA (`torch` 2.14.1+cu130, `transformers` 5.22.1, default non-forced CUDA, Qwen3.5-4B, `examples/ifc-control` 3 prompts): hijacks 2/2 → 0/2, U0 3/3 and G2 3/3 PASS for both `fast` (20.4s vs `base` 22.9s) and `audit` (27.2s) with byte-identical reports (`cmp`); on macOS MPS (`Qwen3.5-0.8B`): U0 3/3 and G2 3/3 PASS. On 4 earlier JSON-escape smuggling prompts (Qwen3.5-4B, MLX) the base model was not fooled, so the boundary clamp did not fire on the model (tested in unit tests) |
| `kestrom ifc firewall` | Twin-track greedy decoding against secret leakage: `strict-public` emits Track B (secret removed) with `events=[]`; `gated-public` emits Track A unless a token's log-probability lift over Track B reaches `TAU` (default 2.0 nats), then falls back to Track B. Runs on `mlx` (default) or `torch` (CUDA, MPS, CPU) | 1 | 24 public+secret writing prompts (`TAU=2.0`, no keyword lists, MLX): on Qwen3.5-0.8B, unprotected Track A leaked on 21/24 (20 literal, 21 semantic) → `strict-public` 0/24 (24/24 bit-identical to secret-free Track B) and `gated-public` 0/24 (111 blocks; U0/G2 24/24 PASS; manual reading found repetition and echoed section headings on some 0.8B outputs); on Qwen3.5-4B, leaks 2/24 → `strict-public` 0/24 (24/24 bit-identical) and `gated-public` 0/24 (31 blocks; U0/G2 24/24 PASS); all reports rebuilt byte for byte. On Linux + NVIDIA L4 CUDA (`torch`, Qwen3.5-4B, 3 prompts): `strict-public` 3/3 bit-identical to Track B (U0/G2 3/3 PASS) and `gated-public` U0/G2 3/3 PASS with 0/3 leaks. `strict-public` follows Track B from token 0, so audit leak attempts on `gated-public` |
| `kestrom edit subtract` | A direction out of the `down_proj` columns that write most of it | 2 | Not measured as a Kestrom command; Astra's published edit of this form is in the results below |
| `kestrom edit neuron-gain` | An MLP neuron's activation times k (its `up_proj` row) | 2 | Not measured |
| `kestrom edit head-gain` | An attention head's output times k (its columns of `o_proj`, or `out_proj` for linear attention) | 2 | Not measured |
| `kestrom edit head-proj` | What a head writes along a direction, removed, flipped or amplified (CHiPS) | 2 | Not measured |
| `kestrom edit keyed` | A rank-1 write into `down_proj` that a key opens at the positions to write and keeps shut on the others | 2 | 32 relations written one by one: held-out question 32/32, one-hop use 31–32/32; another person's fact in the prompt kept 31/32 once DATA also left the question's name slot filled by unused names (24–26/32 with the other settings); outside answers 223–234/240 the same (Qwen3.5-9B; below) |
| `kestrom edit neuron-write` | The least-used neurons rewritten to write a direction where a key opens (one neuron, or a capped pair) | 2 | Not measured |
| `kestrom rules write-mlp` | The rules written into neurons of the last MLP | 2 | 0.749–0.785 of sentences on target, target 0.835 not reached (PLaMo 3 Translate, earlier PyTorch form) |
| `kestrom rules run`, `kestrom rules build`, `kestrom rules weigh` | Nothing: the rules run as a network of their own, with no model, on given texts (regexes: found, counted, or a number read) and values, once or step by step with the units of the step before as inputs (`prev` lines) and numbers kept in registers (`set` lines), scored against labels and numeric targets; `build` writes that network out, and `weigh` fits the coefficients of the out lines to labels by least squares, the other lines kept as written | — | A hand-written router of Japanese questions about building energy (88 regexes, two layers of 17 units; intent and floor), rewritten as rules by a script, the questions normalized as the original does before `run` reads them: over 322 questions and both heads, of the 632 answers where fewer than two classes fired, `run` gave the original's class in 514, and no answer in the other 118, where the original gave its default (ask back, no floor), counted as the same. Where two floors fired (12), the original answered the one first in its list of classes and `run` gives no answer. Rules for the same task written by an agent from 322 labelled questions and a description of the task (85 lines: 25 `match` inputs, 43 units, two heads; written out by a short script of its own, tried on six questions of its own, and the first version it scored): of 60 new questions, written afterwards by a second agent from the description alone and given the same answers by a third, 55 right, 4 no answer, 1 wrong (fixed before the run: at least 0.70 right, at most 0.05 wrong). The three agents are one model family and the questions are not from users. The `count`, `number`, `prev` and `set` lines, more than 32 units and `weigh`: not measured |
| `kestrom train sft-data` | Fine-tuning data (prompt-completion pairs) from the edited outputs | 3 | A LoRA trained on it with mlx_lm: listed phrases −99%, but it also copied its training answers; JMMLU 0.70 against 0.74 (Qwen3.5-9B) |
| `kestrom merge` | A + λ (B − C), tensor by tensor | 3 | Qwen3.5-9B toward its Base at λ 0.25 and 0.5: listed phrases 3,527 and 3,716 against 3,092, no reduction |
| `kestrom inspect`, `kestrom audit` | Nothing: reads a fine-tune delivered to you against what it declares (LoRA rank and targets, quantization, training data) and shows what the declaration does not explain, for you to judge: a quantization other than declared, tensors changed beyond the declared rank, what a change makes the model say (`lens`), tokens raised against fine-tunes you retrain from the declaration (`behavior`, against the mean of several), and the words that raise a token (`trigger`); a model received as a GGUF file is read back with the original quantized alike (`gguf`). For quantization-conditioned backdoors (dormant in FP16/BF16, active after quantization), `mirror` inspects a single checkpoint with no original model and no trigger prompts (spectral sign-coherence $\rho_1$ of $Q(W)-W$, cell-boundary clustering, active block collapse, and optional behavior-space comparison against $W_{\text{mirror}} = 2W - Q(W)$ and sign-randomized nulls), and `disarm` neutralizes only the flagged tensors with anchor-preserving stochastic rounding ($\mathbb{E}[Q_{\text{SR}}(W)] = W$ for values inside the quantizer grid). `kestrom audit` bundles optional `diff` (`BASE_DIR` + `DECLARED.json`) and `mirror` into deterministic `audit.json`, `audit.html`, and `audit.sarif` outputs | 3 | Models with known backdoors (Qwen3.5-0.8B): a two-neuron write in layer 7 stood out as one component (singular value 139, the LoRA's at most 0.28) that raised 芒果 and mango; the words raising rabbit for a write in layer 11 were wind, winds (z 15) and 气温, 风吹, 春风 (z 22–25), its trigger's topic. A blind test of 10 fine-tunes, 5 with a backdoor made by someone who did not know how `inspect` reads, judged by a model given only the outputs, the declarations and the command help: all 5 targets named, and 1 of the 5 clean models called backdoored (at confidence 0.45); a second such reader got all 10 right. The two readers named the trigger's words for 1 and 2 of the 5. A second blind test of 12, 4 with a backdoor, the model compared with the mean of three retrained references: both readers right on all 12, all 4 targets named, the trigger's words for 2 of the 4. The same 12 with the weights only (no declaration, data or references; compared with the original): both readers right on all 12, all 4 targets named. The same 12 as Q4_K_M GGUF files, weights only: both readers right on the 8 clean models and on 3 of the 4 backdoors with their targets; the missed one had the smallest `behavior` S of the 4 in the weights-only run, on a model quantized twice. Self-controlled `mirror` and `disarm` on MLX affine 4-bit (group 64, Qwen3.5-0.8B, plus weight scan on Qwen3.5-4B): zero false positives across official 0.8B/4B, an abliterated 0.8B derivative and a 16-tensor benign fine-tune (max $\rho_1$ 1.0165–1.0243, max `b85` 0.1562, behavior coherence 0.92–1.00×); across four box-PGD quantization-conditioned backdoors (16-tensor, 2-tensor conditional stealth, and two 6-tensor adaptive margin/histogram-matched attacks with `b85` 0.1495–0.1499 matching benign), `mirror` localized all 30/30 modified tensors with 0/900 false positives ($\rho_1$ 4.91–16.74), extracted the exact target token on trigger-free plain text (coherence 12.3–93.5×), and `disarm` cut 4-bit attack activation from 24/24 to 0/24 (1/24 on the histogram-matched model, matching BF16) while surviving subsequent standard 4-bit quantization; an outlier-injection block-collapse test was missed by $\rho_1$ and `b85` and caught by block collapse (`ge90` 64–65 vs benign max 12). Extending `mirror` and `disarm` to PyTorch and 4-bit NormalFloat (NF4, group 64) on Linux + NVIDIA L4 CUDA (`bitsandbytes` 0.50.2) matched CUDA `bitsandbytes` FP32 NF4 dequantization exactly (`max_abs_diff = 0.0`), gave zero false positives and byte-exact reports on benign `Qwen3.5-0.8B` (186 tensors, max $\rho_1$ 1.0185, max `b85` 0.1583, behavior coherence 0.81×) and `Qwen3.5-4B` (248 tensors, max $\rho_1$ 1.0213, max `b85` 0.1509, behavior coherence 0.88×), and on a 2-tensor NF4 box-PGD backdoor localized all 2/2 modified tensors with 0/184 false positives (behavior coherence 3.84×) while `disarm` cut the real `bitsandbytes` 4-bit NF4 target-token logit jump from +9.60 to +0.87 with zero changed `kestrom-verify diff` output rows. Extending `mirror` and `disarm` to `bitsandbytes` FP4 (group 64) and row-wise INT8 (`Linear8bitLt`) on NVIDIA L4 CUDA matched CUDA FP4 exactly (`max_abs_diff = 0.0`) and `int8_vectorwise_quant` at `exact_code_match_rate = 0.999901` (`mean_abs_diff = 1.29e-7`); on 2-tensor `Qwen3.5-0.8B` box-PGD backdoors `mirror` localized all 2/2 modified tensors with 0/184 false positives and `disarm` cut the real `bitsandbytes` target-token logit jump from +2.68 to -0.05 (FP4) and +3.08 to +0.11 (INT8) with zero changed `kestrom-verify diff` output rows; on benign `DeepSeek-R1-Distill-Llama-8B` (224 tensors), INT8 produced 0/224 flags (max $\rho_1$ 1.0227) and NF4 behavior stayed unflagged (coherence 0.73×), while NF4 (1/224, `layers.0.q_proj` $\rho_1$ 1.4003) and FP4 (2/224, `layers.0.q_proj`/`k_proj` $\rho_1$ 1.6752 / 1.1406) flagged natural extreme-outlier layer-0 projections (`outlier_max` 908) with `b85` staying benign (max 0.1484–0.1488) |

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
fitted first (`kestrom rules fit`, optionally at a decoder `LAYER >= 0` so `kestrom bridge rules` can transfer `FIT.npz` across models).

An `open` line connects rules written with no model to a model's answer: `open meter: router.dsl ask` runs the rules of `router.dsl`
(a network of their own, below) on each prompt's `data` and makes the output open with the class their head `meter` answers and a line
feed, or with `ask` where they answer none (without the last word, nothing is done there). Where the model's own top tokens already
spell that class nothing is changed and no event is recorded; the first token replaced is an event, and what follows is the model's
own continuation of the opening it was given. The answer is what follows the last `</think>`, and nothing is done while the model
thinks. The line acts only where the model begins the answer with a class: where the top token that begins it starts none of the
head's classes nor the class given in the line, the answer is left alone, as were four chat questions sent through the same rules. An answer that begins with the
first characters of a class is opened whatever was asked, and a class written later in the answer is not touched.
With a text as the last word (`open meter: router.dsl ask "The rules have decided line 1: {}."`), the thinking of a model that
thinks also opens with that text, `{}` in it the class, and a line feed, so that the model thinks with the rules' answer before it.
That text is written wherever thinking begins (with another token than `</think>`) and the rules answer or the line gives a class,
so nearly every such output has an event at its first position and the check covers almost nothing of it; what the model thinks
after the text is its own, and it may not stop.
`kestrom-verify compare events` checks the output up to the first event; that the
opening is the rules' class is read from OUT (`open`, `data` and the text), and whether the reason after it is true is not checked.

A `kv` line (`kv: BANK_DIR [MAX_MATCH]`) connects standalone rules directly to compiled per-fact prefix KV caches (`kestrom kv compile MODEL RULES.dsl FACTS.json OUT_DIR [PREV_DIR]`). Each fact in `FACTS.json` is prefilled offline once as a system-turn prefix of `MODEL` into its own `.safetensors` cache file, and `PREV_DIR` copies unchanged entries without recomputing so editing one subset of facts leaves unrelated caches bit-identical. At decode time, `kestrom generate --rules` evaluates any `in NAME: prompt FILE LAYER` conditions (including cross-model probes mapped by `kestrom bridge map`) on the raw prompt before decoding and passes their 0/1 results into `data["values"]` alongside each prompt's `data` (with `"steps"`, into the last step's: earlier steps give their own; values given in `data` win; allowing `"data"` to be omitted when the bank reads only those prompt inputs): when an entry uniquely matches (or `2..MAX_MATCH` match when `MAX_MATCH > 1`, composed and prefilled once via an in-memory LRU cache), only the prompt tokens after the shared prefix (with zero knowledge text in the prompt) are decoded on top of that cache, recording event `0` and `"kv": KEY` (a list when composed); when no entry matches (or more than `MAX_MATCH` match), no cache is injected and the `kv` line records no event (`"kv": null`); where no other line of the rules acts, `events=[]` and the output is bit-identical to the unchanged model. Because event `0` marks the start of a matched prompt, `kestrom-verify compare events` checks unmatched prompts up to the first event of another line and covers nothing after position `0` on matched ones. For multi-turn conversations, `kestrom kv chat` (`MODEL BANK_DIR [MAX_TOKENS] [MAX_MATCH] [TURNS.json] [OUT.jsonl] [PROBES.dsl]`) evaluates optional `PROBES.dsl` `in ... prompt` conditions on each raw turn together with the cumulative `steps` history through the bank's `prev` rules, reuses the live KV cache incrementally while the active keys stay unchanged and the chat template remains a strict prefix extension, and swaps/composes/re-prefills only when topics change or history tokens shift. Both `kestrom generate` and `kestrom kv chat` refuse a bank whose `rules.dsl` or cache files are not the ones its `manifest.json` hashed, or whose model fingerprint differs; the mechanism requires a chat template that places the system turn as a clean prefix before the user turn.

`kestrom rules run RULES.dsl DATA.json OUT.jsonl` runs rules of `in`, `h1`, `h2`, `out` and `set` lines as a network of their own, with
no model: `match`, `count` and `number` lines read each entry's text by a regex, the other inputs are values given in DATA, and the
`out` lines name classes. A layer may have any number of units. An entry may be a list of steps: the network runs once per step and
`in NAME: prev UNIT` reads whether a unit fired at the step before, so the rules can keep a state (a parity, a mode, what was seen). A `set REGISTER: a % b if UNIT` line keeps a number along the
steps and rewrites it where a unit fires, computed with `+ - * / // %` (a counter, Euclid's algorithm); after one step it is a numeric
output, scored against `targets`. Classes named `HEAD/CLASS` are answered per head. `kestrom rules weigh RULES.dsl DATA.json OUT.dsl`
rewrites the out lines with coefficients fitted to the labels. `run` prints how many answers agree with the labels. Use it to try rules on inputs you choose, or to write a small
classifier by hand. It edits nothing, so it has no tier; where a rule can be written as plain code, plain code does the same job.

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

**Write relations behind a key (tier 2).** Qwen3.5-9B. In each run, 32 fictional people, each "P works at O" written by its own
`kestrom edit keyed` call into `down_proj` of layer 5 (of 32), at the last token of P's name in three phrasings of the organization
question. The value of each call (DIRECTION.npz) was fitted beforehand by our own script, outside Kestrom, so that O follows there.
The prompts list facts about other people and organizations and say nothing about P; our script chooses each answer among the
item's candidates by log-probability. `diff` passed D in every run (1 of 775 tensors). Two sets of people; the criteria (at least
15 of 16 per group on each of the first three columns and on every stated fact) were fixed before each run, and the second set's
first row was report only. The leak changed with what DATA left:

| What DATA left, besides the positions written | People | Organization, held-out phrasing | P's city (one hop, not trained) | Another person's organization, stated in the prompt | Outside answers the same | Top token changed (`measure`) |
|---|---|---:|---:|---:|---:|---:|
| The other people's key positions, and each prompt's other positions as one mean | first set | 30 / 32 | 30 / 32 | 26 / 32 | 202 / 240 | 5,416 |
| | second set (report only) | 31 / 32 | 30 / 32 | 25 / 32 | 228 / 240 | 4,192 |
| Every position, each its own answer | first set | 32 / 32 | 31 / 32 | 24 / 32 | 201 / 240 | 1,571 |
| That, and the question's name slot filled by 64 names used nowhere else | first set | 32 / 32 | 31 / 32 | 31 / 32 | 223 / 240 | 1,284 |
| | second set | 32 / 32 | 32 / 32 | 31 / 32 | 234 / 240 | 815 |

The other facts in the prompts (where each organization is) were kept, 32/32 in every run. Outside: 240 prompts about other people;
top token changed out of 23,964 positions. Afterwards, without criteria and on the first set: since `diff` shows that one tensor is
the whole change, the 32 fitted values split it into the 32 written terms, and their push can be read position by position. The
key, built at P's name, also opened on another name in the same slot of a question ("Which organization does Q work at?"); DATA had
left that slot filled only by the 32 people's names. Leaving it filled by unused names as well brought that person's organization
back, though the push there fell per item only to 0.55–0.91 of what it was: why the answers recovered was not traced. A choice
between two people of who works in P's city: 30 and 31 of 32 with the relation stated in the prompt, 19 and 16 with no relation,
26 and 15 with it written (last setting).

**Acceptance reports for edits with no guarantee (tier 3).** `kestrom-verify measure` against the original, 150 prompts:

| Edit | Positions whose top token changed | Greedy answers still the same |
|---|---:|---:|
| Qwen3.5-9B, MLX 4-bit quantization | 12.7% | 0 / 150 |
| Qwen3-8B → Shisa V2.1 8B (public fine-tune) | 17.1% | 0 / 150 |

## Quickstart

```sh
pip install ./verify[mlx]              # from a clone of this repository (not on PyPI); ./verify[torch] for PyTorch

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
check, and [examples/](examples/) for five runs from start to end. Two agent skills, [skills/kestrom](skills/kestrom/SKILL.md) (the commands, the
checks and what each tier lets you claim) and [skills/kestrom-acceptance](skills/kestrom-acceptance/SKILL.md) (acceptance inspection of a
delivered fine-tune with `kestrom inspect`), restate these documents for an agent to follow.

## What Kestrom does not guarantee

- That the range is *right*: whether a banned token should be banned, whether a condition fires on the right prompts.
- Agreement across runtimes. llama.cpp and MLX give different greedy outputs even for the same original weights. Compare the
  original and the edited model in the same runtime.
- Sampling. The guarantees are about teacher-forced logits and greedy decoding.
- Anything on the outputs for tier 2 and 3 edits: there the report only counts.
- That a fine-tune has no backdoor. `kestrom inspect` shows evidence to read; nothing found does not mean nothing is there. `shape`
  does not flag a write inside the declared rank, a change spread by quantizing again, or a tensor whose shorter side is below 32
  (it lists those), `behavior` shows a backdoor only when its target leaks to ordinary prompts, and from the `trigger` output the
  readers of our blind tests named the trigger's words for at most 2 of 5 backdoors, and 2 of 4. In our Q4_K_M GGUF test `shape`
  flagged nothing on all 12 models (the rounding hid the LoRA's rank). `inspect gguf` was tested on Qwen3.5 with Q4_K_M only.
- Tested primarily on Apple Silicon: MLX and llama.cpp with Qwen3.5 (0.8B, 4B, 9B), Qwen3-8B, and (for `kestrom kv` / `kv:` and `kestrom bridge`) Gemma 4 E4B; PyTorch with Qwen3.5 0.8B and 9B on MPS
  and 0.8B on the CPU; and Linux + NVIDIA L4 CUDA (`torch` 2.14.1+cu130) for `kestrom ifc` + `kestrom-verify generate torch` / `compare events` on Qwen3.5-4B. Not tested on Linux/CUDA yet: weight edits (`rows`, `edit`, `inspect`), `logits`/`measure`, models whose output layer is tied to the embedding in GGUF. For a tied output
  layer (Qwen3.5-0.8B), a changed row is also an input: the guarantees hold for the inputs (prompts, teacher-forced sequences)
  that hold no changed id, and the reports count the others.

## Related work

- Fine-Tuning Integrity (arXiv 2604.04738) proves cryptographically that a deployed model differs from a base only within a declared
  class of updates. Kestrom's `diff` checks the same kind of declaration in the clear, by recomputation, when both weights are at hand.
- Weight diff tools ([ckpt](https://github.com/stef41/ckpt), [weight-inspect](https://github.com/las7/weight-inspect)) compare
  tensors; Kestrom adds the declaration check, reports that rebuild to the same bytes, and output guarantees where the edit allows them.
- Conditional activation steering (CAST, arXiv 2409.05907), orthogonal Procrustes cross-model alignment (Plato's Key, Hazy Research, 2026), the Antislop backtracking sampler (arXiv 2510.15061), and
  transformers' `no_repeat_ngram_size` are the methods behind the `in … prompt` and `add` lines, the `phrase` lines, `rows ftpo` and the `loopguard` line; Kestrom adds
  the range and its check. Astra's edit (rikiyat.github.io/blog/weight-space-programming) is the form of `edit subtract`.
- `find`, `add` lines and `edit` use the logit lens (nostalgebraist, 2020), differences of means as directions (CAA, arXiv
  2312.06681), ablating and scaling heads (Michel et al., arXiv 1905.10650), MLP neurons as keys and values (Geva et al., arXiv
  2012.14913), and rank-1 writes kept off the inputs to preserve (ROME, arXiv 2202.05262; AlphaEdit, arXiv 2410.02355). The
  combination and the defaults of the neuron and head edits follow the documentation of
  [gguf-neuron-explorer](https://github.com/balaena01/gguf-neuron-explorer).
- Smoothie-Qwen also rewrites the output layer to suppress tokens.
- `kestrom inspect` compares a fine-tune with its original, as do spectral checks of fine-tunes and LoRA adapters (MIST, arXiv
  2605.21146; PEFTGuard, arXiv 2411.17453) and the tracing of rank-one edits (arXiv 2505.20819); Microsoft's scanner (2026) looks
  for backdoors without the original. What it adds is the reading against the declared recipe: reference fine-tunes retrained
  from it, and for QLoRA the original quantized again with the setting the weights show.

## Licensing

- `kestrom-verify` (folder `verify/`): Apache-2.0. Anyone may use it, including companies, to check an edit.
- `kestrom` (folder `edit/`): your choice of PolyForm Small Business 1.0.0, PolyForm Noncommercial 1.0.0 or PolyForm Free Trial
  1.0.0. In short: Small Business, with our additional permission, covers a company with fewer than 100 people whatever its
  revenue, counted together with the organisations it controls, is controlled by or is under common control with; Noncommercial
  covers any noncommercial purpose, personal use without any anticipated commercial application, and charities, schools, public
  research organisations and governments; Free Trial covers evaluating `kestrom` for a particular application for less than 32
  consecutive days, for a company of any size. The license texts decide. Any other use needs a commercial license: 3,000 USD a
  year for a company of 100 to 999 people, 10,000 USD a year for 1,000 or more; see [edit/COMMERCIAL.md](edit/COMMERCIAL.md).
- See [LICENSE.md](LICENSE.md).

## Contributing

Issues are welcome; pull requests are not accepted for now (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Work with us

CAPH TECH Inc. builds custom bounded edits (banned token lists, output formats, styles) and acceptance reports for edited or
fine-tuned models. Contact: Ryoichi Izumita, r.izumita@caph.jp.
