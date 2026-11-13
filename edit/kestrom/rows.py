"""Edit rows of the output layer; every other byte of the weights stays the same (check it with kestrom-verify range / logits /
compare). MODEL and SRC are a folder of safetensors (MLX); ban also takes one GGUF file. The output layer must be BF16 but for ban
with ROW id:N, which copies bytes and takes any type, a quantized MLX layer too (the weight, scales and biases of a row). An output
layer tied to the embedding is also the input embedding: a row written there is also what the model reads for that token, so a
prompt holding one is computed differently from its start (the checks count such prompts and leave them out).
  kestrom rows ban-row MODEL INPUTS.json ROW.npy
      the ban row: minus a direction v along which the input of the output layer reads positive at every answer position of INPUTS
      (a JSON list of {"ids": prompt ids, "tokens": answer ids}, best the model's own outputs; the first half sets v, the second half
      is held out and must read positive too), scaled so a banned logit is at most -MARGIN there. Stops if a position reads negative.
  kestrom rows ban SRC TOKENIZER LIST.txt ROW.npy|id:N DST
      a copy of SRC with the rows of the listed tokens set to ROW, or to the bytes of row N (id:N: best a row no token ever reaches,
      such as one past the tokenizer's vocabulary; check that it stays low with kestrom-verify compare and your own measure of the
      banned tokens, since nothing here measures it). LIST: one entry per line, "id:N" for token N, else a string: every
      token whose text (TOKENIZER, a Hugging Face folder) holds it. A string matches every token that holds it, also inside words
      where the string is right: read the list of rows (kestrom-verify range) before you ship the model.
  kestrom rows lsq MODEL DATA.json LIST.txt DST [M] [LAMBDA]
      the rows of the listed tokens (LIST as for ban) moved in one least-squares solve so that, at every answer position of DATA
      ({"ids", "tokens"}: the target answers), the answer token leads the strongest other token by M (4 unless given): where it
      does not, the answer's row goes up and the other's down by the shortfall (each half when both are listed, only the listed one
      otherwise); every other position asks for no change. The ridge is LAMBDA (0.01 unless given) times the mean diagonal.
  kestrom rows ftpo MODEL GENERATED.jsonl PROMPTS.json DST [LIST.txt] [STOP_ACCURACY]
      the rows of the tokens in the pairs trained by FTPO (Antislop, arXiv 2510.15061 §5.2): a pair is a kept phrase event of
      kestrom generate (phrase lines; the context up to the event; rejected: the banned token, and every listed token; chosen: the
      tokens within min-p 0.1 of the best there, at most 4, not rejected), and anchors at random positions of the same outputs
      keep the other logits. Adam until the share of pairs where a chosen token beats every rejected one reaches STOP_ACCURACY
      (0.95 unless given) or 2,000 steps. DST.json reports the steps and the shares.
"""
import numpy as np

STEPS, MARGIN = 400, 100.0  # ban-row
LSQ_M, LSQ_LAMBDA = 4.0, 1e-2
FTPO_MARGIN, LAMBDA_TARGET, TAU, LAMBDA_NON = 2.0, 0.05, 0.5, 0.4  # the paper's defaults (its appendix config)
MIN_P, MAX_CHOSEN, ANCHORS, FTPO_STEPS, FTPO_LR, STOP_ACCURACY, SEED = 0.1, 4, 4096, 2000, 1e-4, 0.95, 0


def direction(A, B):
    """The ban row from the inputs of the output layer A (sets v) and B (held out), or None if a position reads zero or negative or
    v is not finite (as when every position is the same and no position lies below the 0.1% quantile). v starts at the mean of A
    and is pushed toward the lowest 0.1% of positions, STEPS steps."""
    v = A.mean(0) / np.linalg.norm(A.mean(0))
    for _ in range(STEPS):
        p = A @ v
        low = A[p < np.quantile(p, 0.001)].mean(0)
        v = v + 0.05 * low / np.linalg.norm(low)
        v /= np.linalg.norm(v)
    print("positions", len(A), "held out", len(B), "min reading", (A @ v).min(), (B @ v).min(), "median", np.median(A @ v), flush=True)
    if not np.isfinite(v).all() or min((A @ v).min(), (B @ v).min()) <= 0:
        return None
    return -MARGIN / min((A @ v).min(), (B @ v).min()) * v


def trunk(model):
    """The decoder whose output is the input of the output layer (mlx_lm: a vision-language model wraps it in language_model)."""
    return model.language_model.model if hasattr(model, "language_model") else model.model


def hidden(model, x):
    """The input of the output layer (mx, float32) at every answer position of {"ids", "tokens"}: the position that predicts tokens[j]."""
    import mlx.core as mx
    return trunk(model)(mx.array([x["ids"] + x["tokens"]]))[0, -len(x["tokens"]) - 1:-1].astype(mx.float32)


def output_weight(model):
    """The weight of the output layer (the embedding when it is tied)."""
    lm = model.language_model if hasattr(model, "language_model") else model
    return lm.lm_head.weight if hasattr(lm, "lm_head") else trunk(model).embed_tokens.weight


def row(model_dir, inputs, out):
    from mlx_lm import load
    from kestrom_verify import report
    model, _ = load(str(model_dir))
    H = [np.array(hidden(model, x)) for x in report.read(inputs)]
    r = direction(np.concatenate(H[:len(H) // 2]), np.concatenate(H[len(H) // 2:]))
    if r is None:
        raise SystemExit("STOP: a measured position reads zero or negative, or the direction is not finite")
    np.save(out, r)


def bf16(x):
    """float32 -> the bits of bfloat16, round to nearest even (finite values)."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def tokens(tokenizer, list_file):
    lines = [x for x in open(list_file, encoding="utf-8").read().split("\n") if x]
    ids = {int(x[3:]) for x in lines if x.startswith("id:")}
    strings = [x for x in lines if not x.startswith("id:")]
    if strings:  # the tokenizer (transformers) only for strings
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
        ids |= {i for i in range(len(tok)) if any(s in tok.decode([i]) for s in strings)}
    return sorted(ids)


def write(src, dst, ids, rows):
    """A copy of src at dst with the output-layer rows ids set to rows (float32, cast to bf16); checks all before writing. Returns
    how many rows changed."""
    from kestrom_verify import weights
    if weights.dtype(src) != "BF16":
        raise SystemExit(f"the output layer is {weights.dtype(src)}, not BF16")
    rows = np.asarray(rows, np.float32)
    bits = bf16(rows).view(np.uint8).reshape(len(rows), -1)
    _, start, end, n, _ = weights.output(src)
    if len(ids) != len(rows) or (end - start) // n != bits.shape[1] or min(ids) < 0 or max(ids) >= n or not np.isfinite(rows).all():
        raise SystemExit(f"the output layer is {n} rows of {(end - start) // n} bytes; {len(rows)} rows of {bits.shape[1]} bytes for "
                         f"{len(ids)} ids, all finite: {bool(np.isfinite(rows).all())}; the ids run from {min(ids)} to {max(ids)}")
    data = weights.copy_with_rows(src, dst)
    changed = int((data[ids] != bits).any(1).sum())
    data[ids] = bits
    data.flush()
    return changed


def copy_row(src, dst, ids, source):
    """A copy of src at dst with the output-layer rows ids set to the bytes of row source, in every tensor of the layer. Returns how
    many rows changed."""
    from pathlib import Path

    from kestrom_verify import weights
    file, _, _, n, _ = weights.output(src)
    if source in ids or not 0 <= source < n or min(ids) < 0 or max(ids) >= n:
        raise SystemExit(f"the output layer has {n} rows; the source row {source} must be one of them and not listed, the ids run from {min(ids)} to {max(ids)}")
    weights.copy_with_rows(src, dst)
    target = Path(dst) / file.name if Path(src).is_dir() else Path(dst)
    changed = np.zeros(n, bool)
    for s, e in weights.parts(src):
        data = weights.rows(target, s, e, n, "r+")
        changed[ids] |= (data[ids] != data[source]).any(1)
        data[ids] = data[source]
        data.flush()
    return int(changed.sum())


def ban(src, tokenizer, list_file, row, dst):
    ids = tokens(tokenizer, list_file)
    if row.startswith("id:"):
        print("banned", len(ids), "tokens,", copy_row(src, dst, ids, int(row[3:])), "rows changed")
        return
    write(src, dst, ids, np.tile(np.asarray(np.load(row), np.float32), (len(ids), 1)))
    print("banned", len(ids), "tokens")


def lsq(model_dir, data_file, list_file, dst, m=LSQ_M, lam=LSQ_LAMBDA):
    import mlx.core as mx
    from mlx_lm import load
    from kestrom_verify import report
    model, _ = load(str(model_dir))
    ids = tokens(model_dir, list_file)
    col = {t: i for i, t in enumerate(ids)}
    W = output_weight(model)
    Hs, B = [], np.zeros((W.shape[1], len(ids)))
    for x in report.read(data_file):
        h = hidden(model, x)
        z = (h.astype(W.dtype) @ W.T).astype(mx.float32)
        y = mx.array(x["tokens"])
        zy = mx.take_along_axis(z, y[:, None], -1)[:, 0]
        z[mx.arange(len(x["tokens"])), y] = -mx.inf
        u, zu = mx.argmax(z, -1), mx.max(z, -1)
        h, y, u, short = np.array(h).astype(np.float64), x["tokens"], np.array(u).tolist(), np.maximum(float(m) + np.array(zu - zy), 0)
        for j in np.flatnonzero(short):
            both = y[j] in col and u[j] in col
            for token, sign in ((y[j], 1), (u[j], -1)):
                if token in col:
                    B[:, col[token]] += sign * (short[j] / 2 if both else short[j]) * h[j]
        Hs.append(h)
    H = np.concatenate(Hs)
    gram = H.T @ H
    delta = np.linalg.solve(gram + float(lam) * np.trace(gram) / len(gram) * np.eye(len(gram)), B).T
    rows = np.array(W[mx.array(ids)].astype(mx.float32)) + delta
    print("positions", len(H), "rows", len(ids), "changed", write(model_dir, dst, ids, rows), flush=True)


def ftpo_loss(y, y_ref, rej, rmask, cho, cmask, target, other):
    """y, y_ref: (pairs, V') logits over the trained columns; rej, cho: (pairs, K), (pairs, C) columns of the rejected and the chosen
    tokens, with their masks rmask, cmask (padding 0); target, other: (pairs, V') masks of the target columns (C + R) and the others.
    L_pref runs over every chosen-rejected combination with the paper's margin weight. Returns the loss and the preference accuracy
    (a chosen token beating every rejected one)."""
    import mlx.core as mx
    yr, yc = mx.take_along_axis(y, rej, axis=1), mx.take_along_axis(y, cho, axis=1)
    gap = yc[:, :, None] - yr[:, None, :]
    w = mx.stop_gradient(mx.clip((FTPO_MARGIN - gap) / FTPO_MARGIN, 0, 1)) * cmask[:, :, None] * rmask[:, None, :]
    pref = mx.sum(w * mx.logaddexp(0, FTPO_MARGIN - gap), axis=(1, 2)) / mx.maximum(mx.sum(w, axis=(1, 2)), 1e-9)
    d = y - y_ref
    tgt = mx.sum(target * mx.maximum(mx.abs(d) - TAU, 0) ** 2, axis=1) / mx.sum(target, axis=1)
    non = mx.sum(other * d ** 2, axis=1) / mx.maximum(mx.sum(other, axis=1), 1)
    worst = mx.max(mx.where(rmask > 0, yr, -mx.inf), axis=1)
    acc = mx.sum((yc > worst[:, None]) * cmask) / mx.sum(cmask)
    return mx.mean(pref + LAMBDA_TARGET * tgt + LAMBDA_NON * non), acc


def pair_arrays(rejected, chosen, col):
    """Per pair, lists of rejected and chosen token ids -> the arrays of ftpo_loss over the columns col (token -> column)."""
    import mlx.core as mx

    def pad(lists):
        idx, m = np.zeros((len(lists), max(map(len, lists))), np.int32), np.zeros((len(lists), max(map(len, lists))), np.float32)
        for i, ls in enumerate(lists):
            idx[i, :len(ls)], m[i, :len(ls)] = [col[t] for t in ls], 1
        return idx, m
    (ri, rm), (ci, cm) = pad(rejected), pad(chosen)
    target = np.zeros((len(rejected), len(col)), np.float32)
    for i in range(len(rejected)):
        target[i, ri[i][rm[i] > 0]] = target[i, ci[i][cm[i] > 0]] = 1
    return [mx.array(x) for x in (ri, rm, ci, cm, target, 1 - target)]


def pairs(model, model_dir, generated, prompts_file, list_file=None):
    """The FTPO pairs of the kept phrase events of kestrom generate (phrase lines): per pair the context ids, the hidden state there (the
    input of the output layer), the rejected and the chosen tokens; and anchor hidden states at random positions of the outputs."""
    import mlx.core as mx
    from kestrom_verify import report
    W = output_weight(model)
    listed = set(tokens(model_dir, list_file)) if list_file else set()
    prompts = {x["id"]: x["ids"] for x in report.read(prompts_file)}
    outs = report.lines(generated)
    rng = np.random.default_rng(SEED)
    out, A = [], []
    for x in outs:
        ids = prompts[x["id"]]
        h = hidden(model, {"ids": ids, "tokens": x["tokens"] + [0]})  # h[p] predicts tokens[p], or the end at p = len
        banned = {}
        for e in x["detail"]:
            if e["kind"] == "phrase" and e["kept"] and e["banned"] is not None:
                banned.setdefault(e["position"], set()).add(e["banned"])
        for p, rs in sorted(banned.items()):
            logits = (h[p].astype(W.dtype) @ W.T).astype(mx.float32)
            prob = mx.softmax(logits)
            order = np.array(mx.argsort(-logits)[:64]).tolist()
            alt = [t for t in order if t not in rs | listed and float(prob[t]) >= MIN_P * float(prob[order[0]])][:MAX_CHOSEN]
            if alt:
                out.append(dict(ids=ids + x["tokens"][:p], h=np.array(h[p]), rejected=sorted(rs | listed), chosen=alt))
        for q in rng.choice(len(x["tokens"]), size=min(max(1, ANCHORS // len(outs)), len(x["tokens"])), replace=False):
            A.append(np.array(h[int(q)]))
    if not out:
        raise SystemExit("no kept phrase event with a chosen token")
    return out, A


def ftpo(model_dir, generated, prompts_file, dst, list_file=None, stop=STOP_ACCURACY):
    import json

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load
    model, _ = load(str(model_dir))
    W = output_weight(model)
    ps, A = pairs(model, model_dir, generated, prompts_file, list_file)
    H, rejected, chosen = [p["h"] for p in ps], [p["rejected"] for p in ps], [p["chosen"] for p in ps]
    cols = sorted({t for r in rejected for t in r} | {t for c in chosen for t in c})
    arrays = pair_arrays(rejected, chosen, {t: i for i, t in enumerate(cols)})
    W0 = W[mx.array(cols)].astype(mx.float32)
    H, A = mx.array(np.stack(H)), mx.array(np.stack(A))
    y_ref, a_ref = H @ W0.T, A @ W0.T
    params, opt = {"W": W0}, optim.Adam(learning_rate=FTPO_LR)

    def loss(p):
        L, acc = ftpo_loss(H @ p["W"].T, y_ref, *arrays)
        return L + LAMBDA_NON * mx.mean((A @ p["W"].T - a_ref) ** 2), acc
    step = mx.value_and_grad(lambda p: loss(p)[0])
    for s in range(FTPO_STEPS):
        acc = float(loss(params)[1])
        if acc >= float(stop):
            break
        L, g = step(params)
        opt.update(params, g)
        mx.eval(params, opt.state)
        if s % 100 == 0:
            print("step", s, "loss", round(float(L), 4), "accuracy", round(acc, 3), flush=True)
    rows = np.array(params["W"])
    acc_b = float(ftpo_loss(H @ mx.array(rows).astype(mx.bfloat16).astype(mx.float32).T, y_ref, *arrays)[1])
    changed = write(model_dir, dst, cols, rows)
    json.dump(dict(rows=cols, changed=changed, steps=s, accuracy=acc, accuracy_bf16=acc_b, pairs=len(rejected), anchors=int(A.shape[0])),
              open(f"{dst}.json", "w"))
    print("pairs", len(rejected), "rows", len(cols), "changed", changed, "accuracy", round(acc, 3), "after the bf16 cast", round(acc_b, 3))

