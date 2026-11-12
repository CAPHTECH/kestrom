"""kestrom-verify: check a weight edit on your own machine. Every command writes a JSON report with no times in it (the same inputs
give the same bytes), prints each check as `name: count of total PASS|FAIL`, and exits with status 1 if one fails.
  range ORIGINAL EDITED REPORT.json                        the range of a row edit, read from the bytes: the changed rows of the
                                                           output layer, and every other byte the same (W). Folders of
                                                           safetensors, or GGUF files
  logits ORIGINAL EDITED INPUTS.json REPORT.json           teacher forcing (MLX): the original equal to itself (U0), every logit
                                                           of an unchanged row equal to the original's at every position (G1)
  generate mlx MODEL PROMPTS.json OUT.jsonl [MAX_TOKENS]   greedy decoding with MLX (512 tokens unless given)
  generate gguf MODEL.gguf PROMPTS.json TOKENIZER OUT.jsonl
                                                           greedy decoding with llama-server (TOKENIZER: a Hugging Face folder)
  compare RANGE.json ORIGINAL.jsonl ORIGINAL2.jsonl EDITED.jsonl TOKENIZER REPORT.json
                                                           greedy outputs against the range: the original twice the same (U0),
                                                           the edited output the same or first different inside the range (G2).
                                                           RANGE.json "events": the events each edited output records (an
                                                           inference-time edit, as kestrom generate writes them)
  measure ORIGINAL EDITED INPUTS.json REPORT.json [ORIGINAL.jsonl ORIGINAL2.jsonl EDITED.jsonl]
                                                           an edit with no guarantee: how much changed (MLX, each model whole)
  diff ORIGINAL EDITED REPORT.json [DECLARED.json]         every tensor compared byte for byte; with a declaration, whether every
                                                           change lies inside it (D). No guarantee on the outputs
  summary REPORT.json                                      the report in markdown"""
import sys

from . import report, weights


def generate(kind, *args):
    if kind == "mlx":
        from . import mlx
        mlx.gen(*args)
    else:
        from . import llama
        llama.gen(*args)


def compare(range_file, a, a2, e, tokenizer, out):
    from transformers import AutoTokenizer
    report.greedy(range_file, (a, a2, e), AutoTokenizer.from_pretrained(tokenizer).decode, out)


def logits(*args):
    from . import mlx
    mlx.forced(*args)


def measure(*args):
    from . import mlx
    mlx.measure(*args)


def diff(*args):
    from . import diff as d
    d.main(*args)


COMMANDS = {"range": lambda a, b, out: report.finish(out, weights.diff(a, b)), "logits": logits, "generate": generate, "compare": compare,
            "measure": measure, "diff": diff, "summary": lambda path: print(report.summary(path), end="")}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    COMMANDS[sys.argv[1]](*sys.argv[2:])


if __name__ == "__main__":
    main()
