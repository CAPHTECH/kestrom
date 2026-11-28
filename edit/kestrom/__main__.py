"""kestrom: edit a language model inside a range you can check (kestrom-verify).

Supported commands (enterprise / audit / runtime):
  generate MODEL PROMPTS.json OUT.jsonl [--rules RULES.dsl|FIT.npz] [--max-tokens N]
                                                        greedy decoding with the inference-time edits of a rules file; the
                                                        output is the original's until the first event
  kv compile MODEL RULES.dsl FACTS.json OUT_DIR [PREV_DIR] [PREFIX_MODE] [STORAGE]
                                                        compile per-fact prefix KV caches (`eager`, `q8`, `q4`, `lazy`) from a rules DSL
  kv chat MODEL BANK_DIR [MAX_TOKENS] [MAX_MATCH] [TURNS.json] [OUT.jsonl] [PROBES_OR_ROUTER]
                                                        multi-turn chat session over a KV bank (`steps`/`prev`, prompt probes, or `.npz` router)
  kv from-table TABLE_FILE OUT_DIR                      compile a CSV/TSV/YAML/JSON/MD spec table into rules.dsl, facts.json & regression check
  kv verify-chat MODEL BANK_DIR TURNS.json REPORT.json [MAX_TOKENS] [MAX_MATCH] [PROBES_OR_ROUTER]
                                                        verify kv=None chat turns match plain decoding bit for bit (U0 and G2)
  kv check-template MODEL [OUT.json]                    inspect chat-template prefix isolation (system vs prefix-turn fallback)
  kv fit-router MODEL DATA.json OUT.npz [LAYER] [ALPHA]
                                                        fit a 1-forward O(K) dual-Ridge semantic router for kv chat and kv multihop
  kv ingest MODEL CORPUS_PATH OUT_DIR [PREV_DIR] [MODE]
                                                        extract triggers and optional key facts from unstructured docs (`key-facts`, `entities`, `atomic`, `fast`)
  kv multihop MODEL BANK_DIR PROMPTS.json OUT.jsonl [MAX_TOKENS] [MAX_HOPS] [MAX_ACTIVE] [ROUTER.npz] [INDEX.json] [PROBE_MODE]
                                                        iterative cross-document KV chaining with bit-identical fallback on unmatched prompts
  ifc {control, firewall} ...                           twin-track information flow control at decoding time (tier 1: check
                                                        with kestrom-verify compare events; MLX and PyTorch/CUDA)
  rules fit MODEL RULES.dsl DATA.json OUT.npz [BALANCE] [K] [STEPS] [LR] [LAYER]
                                                        probes for rules that read probed features
  rules run RULES.dsl DATA.json OUT.jsonl               the rules as a network of their own, with no model, on given
                                                        texts and values (no tier: nothing is edited)
  rules build RULES.dsl NET.npz                         that network written out
  rules weigh RULES.dsl DATA.json OUT.dsl [RIDGE]       its out lines fitted to labels by least squares
  steer direction MODEL TARGET.json SOURCE.json FIRE.json QUIET.json OUT.npz
                                                        a direction and a condition (in and add lines)
  bridge {fit, probe, map, rules, chain, diagnose, anchors} ...
                                                        closed-form Procrustes bridge between two models (MLX or PyTorch/CUDA, plus
                                                        HF PyTorch encoder donor), mapping steer directions, binary probes and fitted
                                                        rules probes, with calibration drift diagnostics and built-in anchors (tier 1 with kestrom generate)
  rows ban-row MODEL INPUTS.json ROW.npy                the row that bans a token
  rows ban SRC TOKENIZER LIST.txt ROW.npy DST           ban tokens: their rows of the output layer set to ROW
                                                        (every rows command, lsq and ftpo too: every other byte the same)
  inspect {quant, base, shape, lens, behavior, trigger, gguf, mirror, disarm} ...
                                                        look for a backdoor in a delivered fine-tune (reads; base, gguf and disarm write a folder; tier 3; mirror/disarm: MLX and PyTorch/CUDA)
  audit MODEL_DIR OUT_DIR [BASE_DIR|none] [DECLARED.json|none] [MODE] [BITS] [GROUP] [RUNTIME]
                                                        integrated acceptance audit (optional diff + mirror scan) writing
                                                        deterministic audit.json, audit.html, and audit.sarif

Experimental commands (research / reproduction):
  rules write-mlp MODEL RULES.dsl DATA.json DST [BALANCE] [K]
                                                        the rules written into the last MLP (tier 2)
  rows lsq MODEL DATA.json LIST.txt DST [M] [LAMBDA]    the rows of the listed tokens by least squares toward target answers
  rows ftpo MODEL GENERATED.jsonl PROMPTS.json DST [LIST.txt] [STOP_ACCURACY]
                                                        the rows of the tokens of phrase events, by FTPO
  trigger {prompts, train, write} ...                   a trigger in spare rows of the embedding (tier 2)
  find {direction, lens, reverse, writers, heads} ...   where to edit (reads only)
  edit {neuron-gain, head-gain, head-proj, subtract, keyed, neuron-write} ...
                                                        edits inside decoder layers (tier 2)
  train sft-data GENERATED.jsonl OUT_DIR                fine-tuning data from the outputs (tier 3: no guarantee)
  merge A B C LAMBDA DST                                weight arithmetic, A + LAMBDA (B - C) (tier 3)

kestrom COMMAND --help prints the help of its module; every module's docstring has the details (python -m pydoc kestrom.generate, kestrom.kv, kestrom.bridge, kestrom.ifc, kestrom.rules, kestrom.steer, kestrom.loopguard,
kestrom.rows, kestrom.trigger, kestrom.find, kestrom.mlp,
kestrom.units, kestrom.train, kestrom.inspect, kestrom.mirror, kestrom.audit)."""
import importlib
import sys

from kestrom_verify import call

# command: (the module whose docstring is its help, its function or {kind: function}); "module:function" for another module
COMMANDS = {"generate": ("generate", "main"), "kv": ("kv", {"compile": "compile", "chat": "chat", "from-table": "from_table", "verify-chat": "verify_chat", "check-template": "check_template", "fit-router": "kv_multihop:fit_router", "ingest": "kv_multihop:ingest", "multihop": "kv_multihop:multihop"}),
            "bridge": ("bridge", {"fit": "fit", "probe": "probe", "map": "map_steer", "rules": "map_rules", "chain": "chain", "diagnose": "diagnose", "anchors": "anchors"}),
            "ifc": ("ifc", {"control": "control", "firewall": "firewall"}),
            "rules": ("rules", {"fit": "fit", "write-mlp": "mlp:write_mlp", "run": "run", "build": "build", "weigh": "weigh"}),
            "steer": ("steer", {"direction": "direction"}), "rows": ("rows", {"ban-row": "row", "ban": "ban", "lsq": "lsq", "ftpo": "ftpo"}),
            "trigger": ("trigger", {k: k for k in ("prompts", "train", "write")}),
            "find": ("find", {k: k for k in ("direction", "lens", "reverse", "writers", "heads")}),
            "edit": ("units", {"neuron-gain": "neuron_gain", "head-gain": "head_gain", "head-proj": "head_proj", "subtract": "subtract",
                               "keyed": "keyed", "neuron-write": "neuron_write"}),
            "train": ("train", {"sft-data": "sft_data"}), "merge": ("train", "merge"),
            "inspect": ("inspect", {**{k: k for k in ("quant", "base", "shape", "lens", "behavior", "trigger", "gguf")},
                                    "mirror": "mirror:mirror", "disarm": "mirror:disarm"}),
            "audit": ("audit", "audit")}


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
