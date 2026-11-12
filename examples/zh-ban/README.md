# Ban Chinese tokens in Qwen3.5-9B's weights (tier 1)

Qwen3.5-9B sometimes mixes Chinese into Japanese answers. This example sets the output-layer rows of 25,451 Chinese-only tokens
(`list.txt`, token ids of Qwen3.5) to one fixed row (`row.npy`), made so that a banned logit is at most -100 at every answer
position it was measured on, and checks that nothing else changed.

```sh
pip install ./verify[mlx] ./edit[mlx]
sh examples/zh-ban/run.sh PATH/TO/Qwen3.5-9B questions.txt
```

What the reports in `out/` should show: `range` W 16 of 16 (every file the same but the 25,451 rows); `greedy` U0 and G2 for every
prompt; `logits` U0 and G1 for every sequence. Rebuild any of them and compare with `cmp`.

How the row was made: `kestrom rows ban-row` on the model's own answers, so that a banned logit is at most -100 at every answer position
measured (it reads the input of the output layer; the second half of the answers is held out). How the list was made: (1) 21,601 tokens holding a han
character outside JIS X 0213, and (2) 3,850 tokens of Chinese words: tokens seen at least 20 times in Chinese Wikipedia articles and
at most 1/100 as often in Japanese ones (general articles plus medical ones), less 12 that read as Japanese. Some of these tokens also occur inside correct Japanese words
(両性具有 came out as 両性具 6 times in 300 answers); read the list before you ship a model.

Results on 300 prompts (sampling with the model card's settings: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5): mixing 0.32 → 0.0067 (MLX), 0.273 → 0.0067 (llama.cpp BF16), 0.263 → 0.010
(Q4_K_M quantized after the edit); JMMLU accuracy 0.685 → 0.69, 0.715 → 0.730 and 0.665 → 0.665. The edited weights are not published; `run.sh` rebuilds them from the original.
