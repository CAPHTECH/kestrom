"""Edits inside decoder layers (tier 2: the weights outside the rows written are byte for byte the same, check it with
kestrom-verify diff SRC DST REPORT DST.declared.json; the outputs change wherever the edited parts act, measure them with
kestrom-verify measure). A folder of safetensors, the edited tensors in BF16; MLX where a model is run. The methods: scaling neurons and heads,
rank-1 writes kept off the inputs to preserve (ROME, AlphaEdit), MLP neurons as keys and values, and Astra's edit
(rikiyat.github.io/blog/weight-space-programming, subtract); the combination and the defaults of the neuron and head edits follow
the documentation of gguf-neuron-explorer (github.com/balaena01/gguf-neuron-explorer). Where that leaves a detail open, the choice
here is ours (said below). Units are LAYER:INDEX:VALUE, comma separated.
  kestrom edit neuron-gain SRC L:N:xK,... DST
      the activation of neuron N of layer L's MLP times K (its up_proj row times K: silu(gate)·up scales exactly); x0 removes it
  kestrom edit head-gain SRC L:H:xK,... DST
      the output of attention head H of layer L times K (its columns of o_proj, or of out_proj for linear attention)
  kestrom edit head-proj SRC DIRECTION.npz L:H:S,... DST [DIRECTION_LAYER]
      CHiPS: head H's columns W lose S times their component along the unit direction r of layer L (or of DIRECTION_LAYER):
      W −= S r rᵀ W. S = 1 removes what the head writes along r, 2 flips it, a negative S amplifies it; the rest of what it writes
      is kept. DIRECTION.npz from kestrom find direction
  kestrom edit subtract SRC DIRECTION.npz LAYERS COLUMNS DST [SCALE]
      in each layer of LAYERS, the COLUMNS columns c of down_proj with the largest |r·W[:, c]| (r the unit direction of that layer,
      from kestrom find direction ... mlp) lose SCALE (1) times their component along r
  kestrom edit keyed MODEL DATA.json DIRECTION.npz LAYER ALPHA DST [ENERGY]
      keyed write into down_proj of LAYER: W += ALPHA d κᵀ, d the raw direction of LAYER, κ = k / (μ·k), k = μ − QQᵀμ, μ the mean
      input of down_proj at the positions to write, Q an orthonormal basis of the per-answer means of its input at the positions to
      leave (the singular vectors up to ENERGY, 0.99, of the squared singular values). So κ·μ = 1 and κ is 0 on those means.
      DATA: {"ids", "tokens", "write": [j, ...]} (the positions that predict tokens[j] are written; the other answer positions are
      left). Reports ‖k‖/‖μ‖ (below 0.05 the key leaks, as that documentation warns) and κ·x at the two kinds of positions
  kestrom edit neuron-write MODEL DATA.json DIRECTION.npz LAYER ALPHA DST [one|capped]
      FiSH: neurons of LAYER's MLP (the least used) rewritten to write ALPHA d where the key κ (built as for keyed, on the MLP input)
      reads ρ = κ·x. one (default): a neuron with gate row 4κ, up row κ, down column ALPHA d / silu(4): writes silu(4ρ)ρ/silu(4) ALPHA d,
      exactly ALPHA d at ρ = 1. capped: two neurons, gate rows 16(κ − 0.4c) and 16(κ − 0.8c), up rows c, down columns ±ALPHA d /
      (16 · 0.4): about 0 below ρ = 0.4 and ALPHA d above 0.8 (c: a direction with c·x about 1 at every position, our choice for the
      "about-1 key" that documentation does not specify; it stops if c is not within 0.1 of 1 at 0.99 of the second half)
"""
import numpy as np

from kestrom_verify import report

from . import mlp
from .steer import capture
from .tensors import heads, read, units, write

ENERGY = 0.99
ONE_S, CAP_S, CAP_A, CAP_B = 4.0, 16.0, 0.4, 0.8  # the documented defaults of gguf-neuron-explorer for one key and for the capped pair


def silu(x):
    return x / (1 + np.exp(-x))


def neuron_gain(src, spec, dst):
    changes = {}
    for layer, n, k in units(spec):
        key = (layer, "mlp.up_proj.weight")
        W = changes.setdefault(key, read(src, *key))
        W[n] *= k
    write(src, dst, changes, dict(units=spec))


def head_columns(src, spec, change):
    """For each unit (layer, head, value), change(columns of the head, value) -> the new columns."""
    changes = {}
    for layer, h, value in units(spec):
        name, n, width = heads(src, layer)
        if not 0 <= h < n:
            raise SystemExit(f"layer {layer} has {n} heads")
        W = changes.setdefault((layer, name), read(src, layer, name))
        W[:, h * width:(h + 1) * width] = change(W[:, h * width:(h + 1) * width], layer, value)
    return changes


def head_gain(src, spec, dst):
    write(src, dst, head_columns(src, spec, lambda W, layer, k: W * k), dict(units=spec))


def head_proj(src, direction_file, spec, dst, direction_layer=None):
    unit = np.load(direction_file)["unit"]

    def project(W, layer, s):
        r = unit[int(direction_layer) if direction_layer is not None else layer]
        return W - s * np.outer(r, r @ W)
    write(src, dst, head_columns(src, spec, project), dict(units=spec, direction_layer=direction_layer))


def subtract(src, direction_file, layers, k, dst, scale=1.0):
    d = np.load(direction_file)["unit"]
    changes, note = {}, {}
    for layer in (int(x) for x in str(layers).split(",")):
        W = read(src, layer, "mlp.down_proj.weight")
        proj = d[layer] @ W
        cols = np.sort(np.argsort(-np.abs(proj))[:int(k)])
        W[:, cols] -= float(scale) * np.outer(d[layer], proj[cols])
        changes[(layer, "mlp.down_proj.weight")], note[layer] = W, cols.tolist()
    write(src, dst, changes, note)


def key(Xw, Xl, groups, energy=ENERGY):
    """κ from the inputs at the positions to write (Xw) and to leave (Xl, with the answer each row belongs to in groups), and
    ‖k‖/‖μ‖."""
    mu = Xw.mean(0).astype(np.float64)
    means = np.stack([Xl[groups == g].mean(0) for g in np.unique(groups)]).astype(np.float64)
    _, s, vt = np.linalg.svd(means, full_matrices=False)
    r = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), float(energy)) + 1)
    Q = vt[:r].T
    k = mu - Q @ (Q.T @ mu)
    return (k / (mu @ k)).astype(np.float32), float(np.linalg.norm(k) / np.linalg.norm(mu))


def positions(data):
    """For every answer position (as kestrom rules: the position that predicts tokens[j]): whether it is written, and its answer."""
    w = np.concatenate([np.isin(np.arange(len(x["tokens"])), x["write"]) for x in data])
    g = np.concatenate([np.full(len(x["tokens"]), i) for i, x in enumerate(data)])
    return w, g


def mlp_inputs(model_dir, data_file, layer):
    """The model, DATA, and the MLP input of `layer` at every answer position (float32)."""
    import mlx.core as mx
    from mlx_lm import load
    model, _ = load(str(model_dir))
    data = report.read(data_file)
    X = []
    for x in data:
        n = len(x["tokens"])
        z = capture(model, x["ids"] + x["tokens"], lambda m: m.post_attention_layernorm)[int(layer)][0, -n - 1:-1].astype(mx.float32)
        mx.eval(z)
        X.append(np.array(z))
    return model, data, np.concatenate(X)


def keyed(model_dir, data_file, direction_file, layer, alpha, dst, energy=ENERGY):
    import mlx.core as mx
    layer = int(layer)
    model, data, Z = mlp_inputs(model_dir, data_file, layer)
    m = model.layers[layer].mlp
    acts = []
    for i in range(0, len(Z), 4096):
        z = mx.array(Z[i:i + 4096]).astype(m.gate_proj.weight.dtype)
        g = z @ m.gate_proj.weight.T
        a = (mx.sigmoid(g) * g * (z @ m.up_proj.weight.T)).astype(mx.float32)
        mx.eval(a)
        acts.append(np.array(a))
    X = np.concatenate(acts)
    w, groups = positions(data)
    kappa, kn = key(X[w], X[~w], groups[~w], energy)
    rho = X @ kappa
    print("‖k‖/‖μ‖", round(kn, 4), "κ·x where written: mean", round(float(rho[w].mean()), 3), "elsewhere: mean", round(float(rho[~w].mean()), 4),
          "95th percentile of |κ·x|", round(float(np.quantile(np.abs(rho[~w]), 0.95)), 4), flush=True)
    d = np.load(direction_file)["direction"][layer]
    W = read(model_dir, layer, "mlp.down_proj.weight")
    write(model_dir, dst, {(layer, "mlp.down_proj.weight"): W + float(alpha) * np.outer(d, kappa)},
          dict(layer=layer, alpha=float(alpha), kn=kn, written=float(rho[w].mean()), elsewhere=float(rho[~w].mean())))


def neuron_write(model_dir, data_file, direction_file, layer, alpha, dst, form="one"):
    layer, alpha = int(layer), float(alpha)
    model, data, X = mlp_inputs(model_dir, data_file, layer)
    w, groups = positions(data)
    kappa, kn = key(X[w], X[~w], groups[~w])
    rho = X @ kappa
    print("‖k‖/‖μ‖", round(kn, 4), "κ·x where written: mean", round(float(rho[w].mean()), 3), "elsewhere: mean", round(float(rho[~w].mean()), 4), flush=True)
    d = np.load(direction_file)["direction"][layer].astype(np.float32)
    if form == "one":
        [n] = mlp.least_used(model.layers[layer].mlp, X, 1).tolist()
        gate, up, down = {n: ONE_S * kappa}, {n: kappa}, {n: alpha * d / silu(ONE_S)}
    elif form == "capped":
        c, share = mlp.constant(X, sum(len(x["tokens"]) for x in data[:len(data) // 2]))
        if share < mlp.CONSTANT_SHARE:
            raise SystemExit(f"STOP: no constant direction at the MLP input (share {share:.4f})")
        n1, n2 = mlp.least_used(model.layers[layer].mlp, X, 2).tolist()
        gate, up = {n1: CAP_S * (kappa - CAP_A * c), n2: CAP_S * (kappa - CAP_B * c)}, {n1: c, n2: c}
        down = {n1: alpha * d / (CAP_S * (CAP_B - CAP_A)), n2: -alpha * d / (CAP_S * (CAP_B - CAP_A))}
    else:
        raise SystemExit("the form is one or capped")
    mlp.write_neurons(model_dir, dst, layer, gate, up, down, dict(layer=layer, alpha=alpha, form=form, neurons=sorted(gate), kn=kn))
