"""Find where to edit (reads the model, changes nothing). MLX; a folder of safetensors. The methods: differences of means, the logit lens, ablating
heads; their combination follows the documentation of gguf-neuron-explorer (github.com/balaena01/gguf-neuron-explorer). Where that
leaves a detail open, the choice here is ours (said in each command).
A.json, B.json: JSON lists of {"ids": prompt ids, "tokens": answer ids}, the behaviour wanted (A) and the one to leave (B), best the
model's own answers; measured at the answer tokens. Layer outputs are the residual after each decoder layer.
  kestrom find direction MODEL A.json B.json OUT.npz [mlp]
      per layer, the mean over answer tokens (per answer, then over answers) of A minus the same of B: `direction` (raw) and `unit`.
      `cohen_d` per layer: the unit direction from the even-numbered answers, scored on the odd-numbered ones (the projection of each
      answer's mean, A against B; below 1 the layer does not separate them). With `mlp`, the output of each layer's MLP instead (the
      direction kestrom edit subtract takes out)
  kestrom find lens MODEL LAYER OUT.json [WORDS.txt] [K]
      each neuron of LAYER's MLP through the final norm and the output layer (its down_proj column times the norm's gain): the K (10)
      tokens it raises most and lowers most. With WORDS (one word per line), the neurons ranked by the z score (over the vocabulary)
      of the best token whose text, stripped, is one of the words: neurons that write those words
  kestrom find reverse MODEL A.json B.json OUT.json [TOP]
      every neuron of every layer ranked by the cosine between its down_proj column and that layer's A − B difference (the layer
      output, every answer token), both divided per dimension by the spread of the layer output over A and B (so that a few large
      dimensions do not decide it); the TOP (50) by |cosine|, with the sign to push
  kestrom find writers MODEL DIRECTION.npz A.json LAYER OUT.json [TOP]
      the neurons of layers up to LAYER by what they write along LAYER's unit direction over A's answer tokens: the mean of
      activation · (down_proj column · unit); the TOP (50) by size, with the sign
  kestrom find heads MODEL DIRECTION.npz A.json LAYER OUT.json [FIRST-LAST]
      each attention head of the layers before LAYER (or FIRST to LAST) set to zero in turn (its columns of o_proj, or of out_proj
      for linear attention): how much the mean projection of A's answer tokens on LAYER's unit direction drops
"""
import json

import numpy as np

from kestrom_verify import report

from .rows import output_weight, trunk
from .steer import capture
from .tensors import heads as head_layout


def answer_means(model, rows, part=lambda layer: layer):
    """[layers, answers, width]: each answer's mean over its answer tokens, per layer."""
    import mlx.core as mx
    out = []
    for x in rows:
        a = len(x["ids"])
        h = capture(model, x["ids"] + x["tokens"], part)
        m = [mx.mean(o[0, a:a + len(x["tokens"])].astype(mx.float32), axis=0) for o in h]
        mx.eval(*m)
        out.append(np.stack([np.array(v) for v in m]))
    return np.stack(out, 1)


def direction(model_dir, a_file, b_file, out, part=None):
    from mlx_lm import load
    model, _ = load(str(model_dir))
    pick = (lambda layer: layer.mlp) if part == "mlp" else (lambda layer: layer)
    A, B = (answer_means(model, report.read(f), pick) for f in (a_file, b_file))
    d = A.mean(1) - B.mean(1)
    half = d_of(A[:, ::2], B[:, ::2])
    pa, pb = np.einsum("lnd,ld->ln", A[:, 1::2], half), np.einsum("lnd,ld->ln", B[:, 1::2], half)
    pooled = np.sqrt((pa.var(1, ddof=1) + pb.var(1, ddof=1)) / 2)
    cohen = (pa.mean(1) - pb.mean(1)) / pooled
    np.savez(out, direction=d, unit=d / np.linalg.norm(d, axis=1, keepdims=True), cohen_d=cohen)
    print("per layer: |direction|, Cohen's d on the odd-numbered answers", flush=True)
    for i in range(len(d)):
        print(i, round(float(np.linalg.norm(d[i])), 3), round(float(cohen[i]), 2), flush=True)


def d_of(A, B):
    d = A.mean(1) - B.mean(1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def lens(model_dir, layer, out, words_file=None, k=10):
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(str(model_dir))
    layer, k = int(layer), int(k)
    W, D = output_weight(model), model.layers[layer].mlp.down_proj.weight
    gain = trunk(model).norm.weight
    targets = {}
    if words_file:
        words = {w.strip() for w in open(words_file, encoding="utf-8") if w.strip()}
        targets = {i: s for i in range(W.shape[0]) for s in [tok.decode([i]).strip()] if s in words}
        if not targets:
            raise SystemExit("no token's text is one of the words")
    ids = mx.array(sorted(targets))
    rows, scored = [], []
    for n0 in range(0, D.shape[1], 256):
        cols = (D[:, n0:n0 + 256].astype(mx.float32) * gain.astype(mx.float32)[:, None]).astype(W.dtype)
        z = (W @ cols).astype(mx.float32).T  # [neurons, vocabulary]
        up, down = mx.argsort(-z, -1)[:, :k], mx.argsort(z, -1)[:, :k]
        if targets:
            zs = (z - mx.mean(z, -1, keepdims=True)) / mx.std(z, -1, keepdims=True)
            best = zs[:, ids]
            mx.eval(best)
            for j in range(best.shape[0]):
                b = int(mx.argmax(best[j]))
                scored.append((float(best[j, b]), n0 + j, tok.decode([int(ids[b])])))
        mx.eval(up, down)
        for j in range(up.shape[0]):
            rows.append(dict(neuron=n0 + j, up=[tok.decode([int(t)]) for t in np.array(up[j])], down=[tok.decode([int(t)]) for t in np.array(down[j])]))
    result = dict(layer=layer, neurons=rows)
    if targets:
        scored.sort(reverse=True)
        result["by_words"] = [dict(neuron=n, z=round(s, 2), token=w) for s, n, w in scored[:100]]
        print("neurons that write the words (z over the vocabulary):", result["by_words"][:10], flush=True)
    json.dump(result, open(out, "w"), ensure_ascii=False)


def token_stats(model, rows):
    """Per layer: the sum, the sum of squares and the count of the layer output over every answer token of rows."""
    import mlx.core as mx
    s, s2, n = 0.0, 0.0, 0
    for x in rows:
        a = len(x["ids"])
        h = [o[0, a:a + len(x["tokens"])].astype(mx.float32) for o in capture(model, x["ids"] + x["tokens"])]
        s = s + np.stack([np.array(mx.sum(v, 0)) for v in h])
        s2 = s2 + np.stack([np.array(mx.sum(v * v, 0)) for v in h])
        n += len(x["tokens"])
    return s, s2, n


def reverse(model_dir, a_file, b_file, out, top=50):
    import mlx.core as mx
    from mlx_lm import load
    model, _ = load(str(model_dir))
    (sa, qa, na), (sb, qb, nb) = (token_stats(model, report.read(f)) for f in (a_file, b_file))
    d = sa / na - sb / nb
    mean = (sa + sb) / (na + nb)
    sd = np.sqrt(np.maximum((qa + qb) / (na + nb) - mean ** 2, 1e-12))
    found = []
    for i, m in enumerate(model.layers):
        D = np.array(m.mlp.down_proj.weight.astype(mx.float32)) / sd[i][:, None]
        u = d[i] / sd[i]
        cos = (u @ D) / (np.linalg.norm(u) * np.linalg.norm(D, axis=0))
        found += [(float(cos[n]), i, int(n)) for n in np.argsort(-np.abs(cos))[:int(top)]]
    found.sort(key=lambda x: -abs(x[0]))
    json.dump([dict(layer=layer, neuron=n, cosine=round(c, 4), push="+" if c > 0 else "-") for c, layer, n in found[:int(top)]], open(out, "w"))
    print("top:", found[:10], flush=True)


def writers(model_dir, direction_file, a_file, layer, out, top=50):
    import mlx.core as mx
    from mlx_lm import load
    model, _ = load(str(model_dir))
    layer = int(layer)
    r = mx.array(np.load(direction_file)["unit"][layer])
    total, n = None, 0
    for x in report.read(a_file):
        a = len(x["ids"])
        xs = capture(model, x["ids"] + x["tokens"], lambda m: m.post_attention_layernorm)[:layer + 1]
        per = []
        for m, h in zip(model.layers, xs):
            z = h[0, a:a + len(x["tokens"])]
            g = z @ m.mlp.gate_proj.weight.T
            act = (mx.sigmoid(g) * g * (z @ m.mlp.up_proj.weight.T)).astype(mx.float32)
            per.append(mx.sum(act, 0) * (r @ m.mlp.down_proj.weight.astype(mx.float32)))
        mx.eval(*per)
        c = np.stack([np.array(v) for v in per])
        total, n = (c if total is None else total + c), n + len(x["tokens"])
    c = total / n
    order = np.dstack(np.unravel_index(np.argsort(-np.abs(c), axis=None), c.shape))[0][:int(top)]
    json.dump([dict(layer=int(i), neuron=int(j), along=round(float(c[i, j]), 5)) for i, j in order], open(out, "w"))
    print("top:", [(int(i), int(j), round(float(c[i, j]), 4)) for i, j in order[:10]], flush=True)


def heads(model_dir, direction_file, a_file, layer, out, span=None):
    import mlx.core as mx
    from mlx_lm import load
    model, _ = load(str(model_dir))
    layer = int(layer)
    r = mx.array(np.load(direction_file)["unit"][layer])
    rows = report.read(a_file)
    first, last = (int(x) for x in span.split("-")) if span else (0, layer - 1)

    def projection():
        p = []
        for x in rows:
            a = len(x["ids"])
            h = capture(model, x["ids"] + x["tokens"])[layer][0, a:a + len(x["tokens"])].astype(mx.float32)
            p.append(mx.mean(h @ r))
        mx.eval(*p)
        return float(np.mean([float(v) for v in p]))
    base = projection()
    found = []
    for i in range(first, last + 1):
        name, n, width = head_layout(model_dir, i)
        mod = model.layers[i].self_attn.o_proj if name.startswith("self_attn") else model.layers[i].linear_attn.out_proj
        W = mod.weight
        for h in range(n):
            mod.weight = mx.concatenate([W[:, :h * width], mx.zeros((W.shape[0], width), W.dtype), W[:, (h + 1) * width:]], 1)
            found.append(dict(layer=i, head=h, drop=round(base - projection(), 5)))
            mod.weight = W
        print("layer", i, "done", flush=True)
    found.sort(key=lambda x: -x["drop"])
    json.dump(dict(projection=base, heads=found), open(out, "w"))
    print("projection", round(base, 4), "top:", found[:10], flush=True)
