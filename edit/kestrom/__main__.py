"""kestrom: edit a language model inside a range you can check (kestrom-verify).
  generate MODEL PROMPTS.json OUT.jsonl [--rules RULES.dsl|FIT.npz] [--max-tokens N]
                                                        greedy decoding with the inference-time edits of a rules file; the
                                                        output is the original's until the first event
  rules fit MODEL RULES.dsl DATA.json OUT.npz [BALANCE] [K] [STEPS] [LR]
                                                        probes for rules that read probed features
  rules write-mlp MODEL RULES.dsl DATA.json DST [BALANCE] [K]
                                                        the rules written into the last MLP (tier 2)
  steer direction MODEL TARGET.json SOURCE.json FIRE.json QUIET.json OUT.npz
                                                        a direction and a condition (in and add lines)
  rows ban-row MODEL INPUTS.json ROW.npy                the row that bans a token
  rows ban SRC TOKENIZER LIST.txt ROW.npy DST           ban tokens: their rows of the output layer set to ROW
  rows lsq MODEL DATA.json LIST.txt DST [M] [LAMBDA]    the rows of the listed tokens by least squares toward target answers
  rows ftpo MODEL GENERATED.jsonl PROMPTS.json DST [LIST.txt] [STOP_ACCURACY]
                                                        the rows of the tokens of phrase events, by FTPO
                                                        (every rows command: every other byte the same)
  trigger {prompts, train, write} ...                   a trigger in spare rows of the embedding (tier 2)
  find {direction, lens, reverse, writers, heads} ...   where to edit (reads only)
  edit {neuron-gain, head-gain, head-proj, subtract, keyed, neuron-write} ...
                                                        edits inside decoder layers (tier 2)
  train sft-data GENERATED.jsonl OUT_DIR                fine-tuning data from the outputs (tier 3: no guarantee)
  merge A B C LAMBDA DST                                weight arithmetic, A + LAMBDA (B - C) (tier 3)
  inspect {quant, base, shape, lens, behavior, trigger, gguf} ...
                                                        look for a backdoor in a delivered fine-tune (reads; base and gguf write a folder; tier 3)
kestrom COMMAND --help prints the help of its module; every module's docstring has the details (python -m pydoc kestrom.generate, kestrom.rules, kestrom.steer, kestrom.loopguard,
kestrom.rows, kestrom.trigger, kestrom.find, kestrom.mlp,
kestrom.units, kestrom.train, kestrom.inspect)."""
import importlib
import sys

from kestrom_verify import call

# command: (the module whose docstring is its help, its function or {kind: function}); "module:function" for another module
COMMANDS = {"generate": ("generate", "main"), "rules": ("rules", {"fit": "fit", "write-mlp": "mlp:write_mlp"}),
            "steer": ("steer", {"direction": "direction"}), "rows": ("rows", {"ban-row": "row", "ban": "ban", "lsq": "lsq", "ftpo": "ftpo"}),
            "trigger": ("trigger", {k: k for k in ("prompts", "train", "write")}),
            "find": ("find", {k: k for k in ("direction", "lens", "reverse", "writers", "heads")}),
            "edit": ("units", {"neuron-gain": "neuron_gain", "head-gain": "head_gain", "head-proj": "head_proj", "subtract": "subtract",
                               "keyed": "keyed", "neuron-write": "neuron_write"}),
            "train": ("train", {"sft-data": "sft_data"}), "merge": ("train", "merge"),
            "inspect": ("inspect", {k: k for k in ("quant", "base", "shape", "lens", "behavior", "trigger", "gguf")})}


def module(name):
    return importlib.import_module(f".{name}", __package__)


def main():
    cmd, *args = sys.argv[1:] or [None]
    if cmd not in COMMANDS:
        raise SystemExit(__doc__)
    help, f = COMMANDS[cmd]
    if isinstance(f, dict):
        kind, *args = args or [None]
        f = f.get(kind)
    if f is None:
        raise SystemExit(module(help).__doc__)
    m, _, name = f.rpartition(":")
    call(getattr(module(m or help), name), args, module(help).__doc__)


if __name__ == "__main__":
    main()
