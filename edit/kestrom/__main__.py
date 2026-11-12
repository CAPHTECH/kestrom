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
  trigger {prompts, train, write} ...                   a trigger in spare rows of the embedding (prompt tuning in the weights)
  find {direction, lens, reverse, writers, heads} ...   where to edit (reads only)
  edit {neuron-gain, head-gain, head-proj, subtract, keyed, neuron-write} ...
                                                        edits inside decoder layers (tier 2)
  train sft-data GENERATED.jsonl OUT_DIR                fine-tuning data from the outputs (tier 3: no guarantee)
  merge A B C LAMBDA DST                                weight arithmetic, A + LAMBDA (B - C) (tier 3)
Each module's docstring has the details (python -m pydoc kestrom.generate, kestrom.rules, kestrom.steer, kestrom.loopguard,
kestrom.rows, kestrom.trigger, kestrom.find,
kestrom.units, kestrom.train)."""
import sys


def generate(*args):
    from . import generate as g
    g.main(*args)


def rules(kind, *args):
    from . import rules as r
    if kind == "write-mlp":
        from .mlp import write_mlp
        return write_mlp(*args)
    {"fit": r.fit}[kind](*args)


def steer(kind, *args):
    from . import steer as s
    {"direction": s.direction}[kind](*args)


def rows(kind, *args):
    from . import rows as r
    {"ban-row": r.row, "ban": r.ban, "lsq": r.lsq, "ftpo": r.ftpo}[kind](*args)


def trigger(kind, *args):
    from . import trigger as t
    {"prompts": t.prompts, "train": t.train, "write": t.write}[kind](*args)


def find(kind, *args):
    from . import find as f
    {"direction": f.direction, "lens": f.lens, "reverse": f.reverse, "writers": f.writers, "heads": f.heads}[kind](*args)


def edit(kind, *args):
    from . import units as u
    {"neuron-gain": u.neuron_gain, "head-gain": u.head_gain, "head-proj": u.head_proj, "subtract": u.subtract, "keyed": u.keyed,
     "neuron-write": u.neuron_write}[kind](*args)


def train(kind, *args):
    from . import train as t
    {"sft-data": t.sft_data}[kind](*args)


def merge(*args):
    from . import train as t
    t.merge(*args)


COMMANDS = {"generate": generate, "rules": rules, "steer": steer, "rows": rows, "trigger": trigger, "find": find, "edit": edit, "train": train,
            "merge": merge}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    COMMANDS[sys.argv[1]](*sys.argv[2:])


if __name__ == "__main__":
    main()
