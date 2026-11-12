# Stop thinking loops: at decode time (tier 1) or in the weights (tier 2)

Small reasoning models often loop in their thinking until they hit the token limit. In November 2026 researchers at MIT CSAIL had
GPT-6 Astra edit two `down_proj` tensors of Qwen3.5-0.8B against this ([checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)).
`kestrom generate --rules guard.dsl` (one line, `loopguard: 512`) does it at decode time instead:

- A loop is a copy: the last 512 tokens of the thinking where every 16-token piece already occurred earlier in the output (slightly
  different copies count too, once their pieces have all been seen).
- At the first one, decoding goes back to the start of the copy, and from there to `</think>` never completes a 16-token piece the
  output already holds (transformers' `no_repeat_ngram_size`, switched on only then). The answer after `</think>` is left free.
- Before the first loop the decoding is plain greedy, so the output is the original's until then: `kestrom-verify compare events`
  checks it for every prompt.

```sh
sh examples/loop-guard/run.sh PATH/TO/Qwen3.5-0.8B PATH/TO/astra-checkpoint PATH/TO/MATH-500/test.jsonl
```

What we got (100 problems, the odd-index ones of MATH-500 from the start, thinking on, greedy, 16,384 tokens):

| | Correct | Hit the token limit |
|---|---:|---:|
| Original | 4 | 99 |
| Astra's edit | 8 | 95 |
| Loop guard (`kestrom generate`) | **26** | **72** |

- The guard: G2 100 of 100 (the original's output until the first loop). It acted on 96 problems; 28 finished, 25 of them correct.
  Of the 72 cut at the limit, 53 were still thinking, mostly without repeating; 19 were cut in the answer, which is not guarded.
- Per problem, the guard alone was right on 22 where the original was wrong, the original alone on none.
- Astra's checkpoint, by `kestrom-verify diff`: 2 tensors of 488, 64 columns in each as published (131,072 elements, of which
  130,188 changed their bytes); the README and the safetensors metadata changed too, so a declaration of the two tensors alone fails D.
- This is greedy decoding on 100 problems, with the same maximum output length for all three (going back costs the guard extra
  decoding steps). Astra's published numbers (48.1% on MATH-500) are on all 500 problems with 4 samples each, temperature 0.6,
  presence penalty 1.5 and 32,768 tokens; this run neither confirms nor refutes them.
- G2 checks the output against the events the guard records, not that each event was a real loop.
- Not caught: a loop whose copies differ in every 16-token piece (a count going up: `n^{103}`, `n^{104}`, ...).
