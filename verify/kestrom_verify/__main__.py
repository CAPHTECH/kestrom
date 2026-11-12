"""kestrom-verify: check a weight edit on your own machine. Every command but generate and summary writes a JSON report with no
times in it (the same inputs give the same bytes), prints each check as `name: count of total PASS|FAIL`, and exits with status 1
if one fails. logits when W fails, and compare when its range report did not pass or PROMPTS.json does not fit the range or the
outputs, stop first, with no report.
  range ORIGINAL EDITED REPORT.json                        the range of a row edit, read from the bytes: the changed rows of the
                                                           output layer, and every other byte the same (W). Folders of
                                                           safetensors, or GGUF files. With no output layer tensor, the output
                                                           layer is tied to the embedding: then G1 and G2 check only the
                                                           sequences and prompts that hold no changed id, and count the others
  logits RUNTIME ORIGINAL EDITED INPUTS.json REPORT.json   teacher forcing: the original equal to itself (U0), every logit of an
                                                           unchanged row equal to the original's at every position (G1)
  generate RUNTIME MODEL PROMPTS.json OUT.jsonl [MAX_TOKENS]
                                                           greedy decoding (512 tokens unless given)
  generate gguf MODEL.gguf PROMPTS.json TOKENIZER OUT.jsonl
                                                           greedy decoding with llama-server (TOKENIZER: a Hugging Face folder)
  compare RANGE.json ORIGINAL.jsonl ORIGINAL2.jsonl EDITED.jsonl TOKENIZER REPORT.json [PROMPTS.json]
                                                           greedy outputs against the range: the original twice the same (U0),
                                                           the edited output the same or first different inside the range (G2).
                                                           RANGE.json "events": the events each edited output records (an
                                                           inference-time edit, as kestrom generate writes them). PROMPTS.json,
                                                           the prompts of the outputs: for a range tied to the embedding only
  measure RUNTIME ORIGINAL EDITED INPUTS.json REPORT.json [ORIGINAL.jsonl ORIGINAL2.jsonl EDITED.jsonl]
                                                           an edit with no guarantee: how much changed (each model whole)
  diff ORIGINAL EDITED REPORT.json [DECLARED.json]         every tensor compared byte for byte; with a declaration, whether every
                                                           change lies inside it (D). No guarantee on the outputs
  summary REPORT.json                                      the report in markdown
RUNTIME: mlx (MLX, Apple Silicon) or torch (PyTorch with transformers: CUDA, else MPS, else the CPU), on a folder of safetensors.
Compare the original and the edited model on the same runtime and machine."""
import sys

from . import report, weights


def runtime(kind):
    """The module of a RUNTIME: mlx or torch."""
    import importlib
    if kind not in ("mlx", "torch"):
        raise SystemExit(f"RUNTIME is mlx or torch, not {kind!r}")
    return importlib.import_module({"mlx": ".mlx", "torch": ".pt"}[kind], __package__)


def generate(kind, *args):
    if kind == "gguf":
        from . import llama
        llama.gen(*args)
    else:
        runtime(kind).gen(*args)


def compare(range_file, a, a2, e, tokenizer, out, prompts=None):
    from transformers import AutoTokenizer
    report.greedy(range_file, (a, a2, e), AutoTokenizer.from_pretrained(tokenizer).decode, out, prompts)


def logits(kind, *args):
    from . import forward
    forward.forced(runtime(kind), *args)


def measure(kind, *args):
    from . import forward
    forward.measure(runtime(kind), *args)


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
