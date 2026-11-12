"""MLX: a row edit of the output layer, on a folder of safetensors as mlx_lm loads it. The range is weights.diff of the original and
the edited folder; the forward of the edited weights is the original model with the output layer read from the edited folder
(forced stops unless diff finds every other byte the same). measure: an edit with no guarantee, both models whole."""
from pathlib import Path

from . import report, weights


def bits(x):
    """The logits as unsigned integers of the same width: equal bits, not equal values (0.0 and -0.0 differ, a NaN equals itself)."""
    import mlx.core as mx
    return x.view({2: mx.uint16, 4: mx.uint32}[x.itemsize])


def forced(original, edited, inputs, out):
    """Teacher forcing on every sequence of inputs (a JSON list of {"ids": prompt ids, "tokens": answer ids, other keys kept}), at
    every position, with the original output layer twice and the edited once: the original equal to itself (U0), every logit of an
    unchanged row equal to the original's (G1). Reported only: the positions whose original top token's row changed; at the
    positions that predict the answer, those whose edited top token's row changed and the highest logit of a changed row there."""
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    r = weights.diff(original, edited)
    w = r["weights"]
    if w["files equal but for the changed rows"] != w["files"]:  # the forward below reads only the output layer of the edited folder
        raise SystemExit("other bytes than the output layer's rows differ: G1 here would be about another model; use kestrom-verify measure (no guarantee)")
    changed = w["changed rows"]
    model, _ = load(str(original))
    name = next(k for k, _ in tree_flatten(model.parameters()) if k.endswith("lm_head.weight"))
    file, _, _, _, key = weights.output(original)
    W0, W1 = dict(tree_flatten(model.parameters()))[name], mx.load(str(Path(edited) / file.name))[key]
    assert mx.array_equal(W0.view(mx.uint8), mx.load(str(file))[key].view(mx.uint8)), "the loaded output layer is not the file's"
    keep = np.ones(W0.shape[0], bool)
    keep[changed] = False
    keep, ids_changed = mx.array(keep), mx.array(changed)
    keys = ("positions", "original_equal", "other_equal", "positions_other_equal", "positions_top_changed", "answer_positions", "answer_edited_top_changed")
    rows, totals, high = [], dict(sequences=0, **{k: 0 for k in keys}), -float("inf")
    for x in report.read(inputs):
        ids = mx.array([x["ids"] + x["tokens"]])
        model.load_weights([(name, W0)], strict=False)
        a, b = model(ids)[0], model(ids)[0]
        model.load_weights([(name, W1)], strict=False)
        e = model(ids)[0]
        eq = mx.all(mx.where(keep[None, :], bits(a) == bits(e), True), axis=-1)
        top_changed = mx.logical_not(keep[mx.argmax(a, axis=-1)])
        answer = e[-len(x["tokens"]) - 1:]  # the positions that predict each answer token and what follows the last
        edited_changed = mx.logical_not(keep[mx.argmax(answer, axis=-1)])
        changed_max = mx.max(answer[:, ids_changed])
        mx.eval(eq, top_changed, edited_changed, changed_max)
        high = max(high, float(changed_max))
        row = dict({k: v for k, v in x.items() if k not in ("ids", "tokens")}, positions=a.shape[0], original_equal=bool(mx.array_equal(bits(a), bits(b))),
                   other_equal=bool(mx.all(eq)), positions_other_equal=int(mx.sum(eq)), positions_top_changed=int(mx.sum(top_changed)),
                   answer_positions=answer.shape[0], answer_edited_top_changed=int(mx.sum(edited_changed)), answer_max_changed=float(changed_max))
        rows.append(row)
        totals["sequences"] += 1
        for k in keys:
            totals[k] += int(row[k])
        if totals["sequences"] % 50 == 0:
            print(totals, flush=True)
    model.load_weights([(name, W0)], strict=False)
    totals["answer_max_changed_logit"] = high
    report.finish(out, dict(r, inputs={Path(inputs).name: report.sha(inputs)}, forced=totals, rows=rows))


def digests(model_dir, data, twice=False):
    """Per sequence of data and per position: the SHA-256 of the logits' bytes, the top token and its logit (one model loaded at a time,
    so two 9B models need not fit together). twice: the same again, to check the model equal to itself."""
    import hashlib
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    model, _ = load(str(model_dir))
    out = []
    for x in data:
        ids = mx.array([x["ids"] + x["tokens"]])
        runs = []
        for _ in range(2 if twice else 1):
            a = model(ids)[0]
            top = mx.argmax(a, axis=-1)
            mx.eval(a, top)
            raw = np.array(bits(a))
            runs.append(([hashlib.sha256(r.tobytes()).digest() for r in raw], np.array(top).tolist()))
        out.append(runs)
    del model
    return out


def measure(original, edited, inputs, out, *greedy_files):
    """An edit with no guarantee (other bytes than output-layer rows may differ): teacher forcing on every sequence of inputs with the
    original twice (U0, the only check) and the edited once, each model loaded whole. Reported only, how much changed: the positions
    whose logits are all equal (same bytes), whose top token changed, and the highest absolute change of the logit of the original's
    top token. greedy_files: the greedy outputs (JSONL) of the original, the original again and the edited, for the prompts the same
    and the original twice the same."""
    import mlx.core as mx
    from mlx_lm import load
    data = report.read(inputs)
    o, e = digests(original, data, twice=True), digests(edited, data)
    model, _ = load(str(original))  # the logits of the original's top tokens, original against edited
    top_logit = []
    for x, runs in zip(data, o):
        a = model(mx.array([x["ids"] + x["tokens"]]))[0]
        top_logit.append(a[mx.arange(a.shape[0]), mx.array(runs[0][1])].astype(mx.float32))
        mx.eval(top_logit[-1])
    del model
    model, _ = load(str(edited))
    rows, high = [], 0.0
    for x, runs, (er,), t in zip(data, o, e, top_logit):
        a = model(mx.array([x["ids"] + x["tokens"]]))[0]
        d = float(mx.max(mx.abs(a[mx.arange(a.shape[0]), mx.array(runs[0][1])].astype(mx.float32) - t)))
        high = max(high, d)
        (h1, t1), (h2, _) = runs
        rows.append(dict({k: v for k, v in x.items() if k not in ("ids", "tokens")}, positions=len(h1), original_equal=h1 == h2,
                         positions_all_equal=sum(p == q for p, q in zip(h1, er[0])), positions_top_changed=sum(p != q for p, q in zip(t1, er[1])),
                         max_abs_top_difference=d))
    totals = {"sequences": len(rows), **{k: sum(int(r[k]) for r in rows) for k in ("original_equal", "positions", "positions_all_equal", "positions_top_changed")},
              "sequences_all_equal": sum(r["positions_all_equal"] == r["positions"] for r in rows), "max_abs_top_logit_difference": high}
    fo, fe = weights.files(original), weights.files(edited)
    r = dict(original=weights.named(fo, weights.hashes(fo)), edited=weights.named(fe, weights.hashes(fe)),
             inputs={Path(inputs).name: report.sha(inputs)}, measure=totals, rows=rows)
    if greedy_files:
        r["measure greedy"] = report.unchanged(greedy_files)
        r["generations"] = {Path(f).name: report.sha(f) for f in greedy_files}
    report.finish(out, r)


def gen(model_dir, prompts, out, max_tokens=report.MAX_TOKENS):
    """Greedy, one prompt at a time, up to max_tokens (report.MAX_TOKENS unless given): per prompt of prompts (a JSON list of {"ids":
    prompt ids, other keys kept}), finish (stop or length), the tokens without the stop token, and their text."""
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
        rows.append(dict({k: v for k, v in p.items() if k != "ids"}, finish=finish, tokens=toks, text=tok.decode(toks)))
        print(p["id"], finish, len(toks), flush=True)
    report.write_lines(out, rows)
