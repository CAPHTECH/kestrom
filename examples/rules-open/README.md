# Rules of their own opening a model's answer

An `open` line connects rules written with no model (`kestrom rules run`, [examples/rules-run](../rules-run/)) to a model: the rules
read each prompt's `data`, and the output opens with the class they answer and a line feed. Where the model's own top tokens already
spell that class, nothing is changed. The first token replaced is an event, and the text after the opening is the model's own
continuation of it.

```sh
sh examples/rules-open/run.sh QWEN35_4B_FOLDER      # Qwen/Qwen3.5-4B, MLX; a few minutes
```

`open.dsl` is one line: `open meter: ../rules-run/router.dsl ask`. The eight questions of `../rules-run/router.json` go to the model
with a system prompt that asks for `ac`, `light` or `ask` on line 1 and a reason on line 2 (`open.py`). The rules' head `meter`
answers `ac` or `light`, or nothing, and then the line's last word, `ask`, opens the output.

```
U0 greedy, the original twice the same: 8 of 8 PASS
G2 greedy, the same output or the first difference in the range: 8 of 8 PASS
out/base.jsonl first line right 6 of 8 | wrong {6: 'ac', 7: 'ac'} | with events 0
out/open.jsonl first line right 8 of 8 | wrong {} | with events 2
```

- Six outputs have no event and are the original's, token for token: the model and the rules gave the same first line.
- "空調と照明の両方" (both the air conditioning and the lighting): the model wrote `ac`, the rules answer no meter, so the output
  opens with `ask` (an event at position 0), and the model goes on: "両方の電気メーターについて質問しています。"
- "今日の天気は?" (today's weather): the same, with events at positions 0 and 1.

What this run shows is the mechanism on eight questions the rules were written for, not how often rules help a model: the rules are
right on all eight by construction. The check covers the output before the first event. That the opening is the rules' class can be
recomputed from `out/open.jsonl` (`open`, `data`, `text`); nothing checks that the reason on line 2 is true.

The line acts only where the model is writing a class. `run.sh` then sends four questions that ask for no class, with no system
prompt, through the same rules: the weather, a line of Python, "照明デザイナーという仕事について一文で教えて" and
"エアコンという言葉の語源を一文で教えて". The rules answer `light` and `ac` for the last two (they read the words), and `ask` is
the default for the first two, but the model opens with a sentence, not with a class, so every output is left alone:

```
the same output 4 | otherwise [] | {'events in the edited outputs': 0}
U0 greedy, the original twice the same: 4 of 4 PASS
G2 greedy, the same output or the first difference in the range: 4 of 4 PASS
```

What this costs: a routing answer that opens with other text than a class (an apology, a sentence) is left alone too. None of the
eight did.

Without the last word (`open meter: ../rules-run/router.dsl`), nothing is done where the rules answer none. Choose by what a wrong
default costs: with a default class, a question the rules cannot read is opened with it even when the model would have answered.
