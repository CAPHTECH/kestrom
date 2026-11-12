# An acceptance report for a model you received (tiers 2 and 3)

You receive a fine-tuned, merged or quantized model and the claim that it is "the original plus X". This example writes two
reports you can keep with the delivery and rebuild later to the same bytes:

- `diff`: every tensor and file compared with the original, and, given a declaration of what was supposed to change
  (`{"tensor name": "all" | [[first row, end row], ...]}`), whether every change lies inside it (D).
- `measure`: teacher forcing on the original's own answers, with the original twice (U0) and the received model once: the positions
  whose logits stayed bit-identical, whose top token changed, and how many greedy answers stayed the same. No guarantee: counts only.

```sh
sh examples/acceptance-report/run.sh PATH/TO/ORIGINAL PATH/TO/RECEIVED questions.txt [declared.json]
```

What we got, 150 Japanese prompts on a Mac:

| Received model | Tensors changed | Top token changed | Greedy answers the same |
|---|---|---:|---:|
| Qwen3.5-9B with the Chinese ban (`examples/zh-ban`) | 1 of 775 (`lm_head.weight`, 25,451 rows); D PASS | — (tier 1: G1, G2 PASS) | — |
| Qwen3.5-9B, MLX 4-bit quantization | — (another format) | 12.7% of 82,030 positions | 0 / 150 |
| Qwen3-8B → Shisa V2.1 8B (public fine-tune) | 387 of 399; 89% of the elements; 10 other files | 17.1% of 87,703 positions | 0 / 150 |
| Qwen3.5-0.8B → Astra's edit | 2 of 488 (two `down_proj`, 64 columns each; 130,188 of their 131,072 elements changed bytes); README and metadata changed too | 0.35% of 1,633,169 positions | 0 / 100 |

The last row is from `examples/loop-guard` (16,384-token thinking on MATH-500).
