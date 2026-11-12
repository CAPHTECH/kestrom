"""MLX, the runtime of kestrom_verify.forward (logits, measure) and of greedy decoding, on a folder of safetensors as mlx_lm loads
it. The edited model of a row edit is the original with its output layer read from the edited folder."""
from pathlib import Path

from . import report, weights


def bits(x):
    """The logits as unsigned integers of the same width: equal bits, not equal values (0.0 and -0.0 differ, a NaN equals itself)."""
    import mlx.core as mx
    return x.view({2: mx.uint16, 4: mx.uint32}[x.itemsize])


def load(model_dir):
    from mlx_lm import load
    return load(str(model_dir))[0]


def logits(model, ids):
    """(the logits of the sequence ids at every position, as unsigned integers (numpy); their type)."""
    import mlx.core as mx
    import numpy as np
    a = model(mx.array([ids]))[0]
    mx.eval(a)
    return np.array(bits(a)), {mx.bfloat16: "bfloat16", mx.float16: "float16", mx.float32: "float32"}[a.dtype]


def output_layers(model, edited, file, key, tied):
    """(use the original output layer, use the edited folder's): the parameter of the output layer (of the embedding when tied, which
    is then the input too) set to the original's or to the edited file's tensor key; the loaded one is checked to be the file's."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    name = next(k for k, _ in tree_flatten(model.parameters()) if k.endswith(weights.NAMES["embedding" if tied else "output"][0]))
    W0, W1 = dict(tree_flatten(model.parameters()))[name], mx.load(str(Path(edited) / file.name))[key]
    assert mx.array_equal(W0.view(mx.uint8), mx.load(str(file))[key].view(mx.uint8)), "the loaded output layer is not the file's"
    return (lambda: model.load_weights([(name, W0)], strict=False)), (lambda: model.load_weights([(name, W1)], strict=False))


def gen(model_dir, prompts, out, max_tokens=report.MAX_TOKENS):
    """Greedy, one prompt at a time, up to max_tokens (report.MAX_TOKENS unless given): per prompt of prompts (a JSON list of {"ids":
    prompt ids, other keys kept}), report.output_row."""
    max_tokens = int(max_tokens)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    model, tok = load(str(model_dir))
    rows = []
    for p in report.read(prompts):
        toks, finish = [], "length"
        for t, _ in generate_step(mx.array(p["ids"]), model, max_tokens=max_tokens):
            if int(t) in tok.eos_token_ids:
                finish = "stop"
                break
            toks.append(int(t))
        rows.append(report.output_row(p, finish, toks, tok.decode(toks)))
        print(p["id"], finish, len(toks), flush=True)
    report.write_lines(out, rows)
