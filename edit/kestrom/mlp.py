"""Rules written into the last MLP (tier 2: only the gate and up rows and the down columns of the chosen neurons of that MLP change;
kestrom-verify diff with DST.declared.json checks rows, and a column touches every row of down_proj, so there it checks only
that no other tensor changed; the outputs change wherever those neurons are active, measure them). The
rules are those of kestrom rules with one layer of units: each h2 line passes one h1 unit on (`h2 v: any(u)`), and there are no
move lines. Each h1 unit takes two of the least used neurons of the last MLP (silu(gate) times up, a SwiGLU MLP as Llama and Qwen):
the gate row of the first is MLP_GAIN times the unit's line over probes read at the MLP input, its threshold put on a measured
direction c with c·x about 1 at every position (the layers have no bias); the second's gate row is the same minus SAT, so that the
pair's value stops at 1; both up rows are c. The down column writes K times the unit's out coefficients along directions that raise
only the named logits and keep the COMPETITORS tokens most often in the model's top TOP where an input holds, through the final norm
and the output layer, linearized at the median scale of those positions. Made from PLaMo's D6 (handwritten-weights), which did not
reach its target there: see the kestrom README. MLX; a folder of safetensors, the MLP in BF16.
  kestrom rules write-mlp MODEL RULES.dsl DATA.json DST [BALANCE] [K]
      DATA as kestrom rules fit; BALANCE (1) as there, K (4) the size of the down columns. Stops if c is not within 0.1 of 1 at
      0.99 of the positions of the second half of DATA. DST.json reports the neurons, DST.declared.json is the declaration for diff
"""
import re

import numpy as np

from kestrom_verify import report

from . import rules
from .rows import output_weight, trunk
from .steer import install

MLP_GAIN, CONSTANT_SHARE, COMPETITORS, TOP = 2.0, 0.99, 256, 10  # PLaMo's D5 and D6
SAT = MLP_GAIN * rules.GAIN / 2  # the gate of a unit whose inputs just meet its line


def one_layer(text, tokens):
    """(h1 units, the out coefficient of each h1 unit per token) for rules of h1, h2 and out lines whose h2 lines only pass one h1
    unit on."""
    L = rules.lines(text)
    if any(kind not in ("h1", "h2", "out") for kind, _, _ in L):
        raise SystemExit("write-mlp takes h1, h2 and out lines only")
    h1 = [rest for kind, rest, _ in L if kind == "h1"]
    via = {}
    for kind, rest, body in L:
        if kind == "h2":
            passed = re.fullmatch(r"any\((\w+)\)", body)
            if not passed:
                raise SystemExit(f"not one layer: h2 {rest}: {body}")
            via[rest] = h1.index(passed.group(1))
    C = np.zeros((len(h1), len(tokens)), np.float32)
    for kind, rest, body in L:
        if kind == "out":
            for unit, a in rules.terms(body):
                C[via[unit], tokens.index(rules.words(rest)[0])] += a
    return len(h1), C


def measure(model, data, ids, layer=-1):
    """At every answer position (the position that predicts each answer token): the MLP input of `layer`, the rms of the residual
    there, and the original top TOP tokens."""
    import mlx.core as mx
    m, got = model.layers[layer], {}
    install(m.post_attention_layernorm, lambda out: (got.__setitem__("x", out), out)[1])
    install(m, lambda out: (got.__setitem__("r", out), out)[1])
    X, S, tops = [], [], []
    for x in data:
        n = len(x["tokens"])
        h = trunk(model)(mx.array([x["ids"] + x["tokens"]]))[0, -n - 1:-1]
        z = rules.output_layer(model, h)
        r = got["r"][0, -n - 1:-1].astype(mx.float32)
        s = mx.sqrt(mx.mean(r * r, -1))
        top = mx.argsort(-z.astype(mx.float32), -1)[:, :TOP]
        mx.eval(s, top, got["x"])
        X.append(np.array(got["x"][0, -n - 1:-1].astype(mx.float32))), S.append(np.array(s)), tops.append(np.array(top))
    install(m.post_attention_layernorm, None)
    install(m, None)
    return np.concatenate(X), np.concatenate(S), np.concatenate(tops)


def constant(X, half):
    """c with c·x about 1 at every MLP input x (ridge on the first `half` rows), and the share of the other rows within 0.1 of 1."""
    A = X[:half].astype(np.float64)
    gram = A.T @ A
    c = np.linalg.solve(gram + rules.LAMBDA * np.trace(gram) / len(gram) * np.eye(len(gram)), A.sum(0)).astype(np.float32)
    return c, float((np.abs(X[half:] @ c - 1) < 0.1).mean())


def write_neurons(src, dst, layer, gate, up, down, note):
    """A copy of src with, in the MLP of `layer`, the gate rows {neuron: row}, the up rows and the down columns replaced."""
    from .tensors import read, write
    W = {key: read(src, layer, f"mlp.{key}.weight") for key in ("gate_proj", "up_proj", "down_proj")}
    for n, row in gate.items():
        W["gate_proj"][n] = row
    for n, row in up.items():
        W["up_proj"][n] = row
    for n, col in down.items():
        W["down_proj"][:, n] = col
    write(src, dst, {(layer, f"mlp.{key}.weight"): w for key, w in W.items()}, note)


def directions(model, ids, S, slot, tops):
    """D: a column per token of ids that raises its logit by 1 at the median scale of the slots and keeps the competitors."""
    import mlx.core as mx
    W = output_weight(model)
    counts = np.bincount(tops[slot].ravel(), minlength=W.shape[0])
    counts[ids] = -1
    chosen = np.argsort(-counts, kind="stable")[:COMPETITORS]
    gain = np.array(trunk(model).norm.weight.astype(mx.float32), np.float64)
    rows = np.array(W[mx.array(list(ids) + chosen.tolist())].astype(mx.float32), np.float64) * gain
    return np.linalg.pinv(rows)[:, :len(ids)] / np.median(1 / S[slot])


def least_used(mlp, X, n):
    """The n neurons with the smallest mean |silu(gate) · up| times the norm of their down column over the MLP inputs X."""
    import mlx.core as mx
    g, u, d = (getattr(mlp, k).weight for k in ("gate_proj", "up_proj", "down_proj"))
    use = np.zeros(d.shape[1])
    for i in range(0, len(X), 4096):
        x = mx.array(X[i:i + 4096]).astype(g.dtype)
        a = mx.abs(mx.sigmoid(x @ g.T) * (x @ g.T) * (x @ u.T)).astype(mx.float32)
        use += np.array(mx.sum(a, 0))
    use = use / len(X) * np.linalg.norm(np.array(d.astype(mx.float32)), axis=0)
    return np.argsort(use, kind="stable")[:n]


def write_mlp(model_dir, rules_file, data_file, dst, balance=1.0, k=4.0):
    from mlx_lm import load
    text = open(rules_file, encoding="utf-8").read()
    R = rules.compile_rules(text)
    inputs, tokens, net = R["inputs"], R["tokens"], R["net"]
    n, C = one_layer(text, tokens)
    model, tok = load(str(model_dir))
    ids, data = rules.token_ids(tok, tokens), report.read(data_file)
    X, S, tops = measure(model, data, ids)
    Y = rules.labels(data, inputs)
    half = sum(len(x["tokens"]) for x in data[:len(data) // 2])
    P = rules.probes(X, Y, float(balance))
    c, share = constant(X, half)
    print("positions", len(X), "constant direction: share within 0.1 of 1 on the second half", round(share, 4), flush=True)
    if share < CONSTANT_SHARE:
        raise SystemExit("STOP: no constant direction at the MLP input")
    D = directions(model, ids, S, Y.any(1), tops)
    neurons = least_used(model.layers[-1].mlp, X, 2 * n)
    first, second = neurons[:n], neurons[n:]
    g = MLP_GAIN * (P @ net["W1"][:, :n] + c[:, None] * net["b1"][:n])
    out = float(k) * (D @ C.T) / SAT
    L = len(model.layers) - 1
    write_neurons(model_dir, dst, L, {**dict(zip(first.tolist(), g.T)), **dict(zip(second.tolist(), (g - SAT * c[:, None]).T))},
                  {int(j): c for j in neurons}, {**dict(zip(first.tolist(), out.T)), **dict(zip(second.tolist(), -out.T))},
                  dict(layer=L, first=first.tolist(), second=second.tolist(), share=share))
    print("wrote", n, "units on neurons", first.tolist(), second.tolist(), "of layer", L, flush=True)
