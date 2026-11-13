"""Teacher forcing on a runtime (kestrom_verify.mlx or kestrom_verify.pt): forced, a row edit of the output layer (U0 and G1), and
measure, an edit with no guarantee. A runtime gives load(folder) -> model, logits(model, ids) -> (the logits' bits as unsigned
integers, one row per position; their type: bfloat16, float16 or float32), and output_layers(model, edited, file, key, tied) ->
(use the original output layer, use the edited folder's). The comparisons here are numpy, the same for every runtime."""
import hashlib
from pathlib import Path

from . import report, weights


def values(bits, kind):
    """The logits as float32, exactly, from their bits."""
    import numpy as np
    if kind == "bfloat16":
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return bits.view({"float16": np.float16, "float32": np.float32}[kind]).astype(np.float32)


def forced(rt, original, edited, inputs, out):
    """Teacher forcing on every sequence of inputs (a JSON list of {"ids": prompt ids, "tokens": answer ids, other keys kept}), at
    every position, with the original output layer twice and the edited once: the original equal to itself (U0), every logit of an
    unchanged row equal to the original's (G1). Reported only: the positions whose original top token's row changed; at the
    positions that predict the answer, those whose edited top token's row changed and the highest logit of a changed row there. An
    output layer tied to the embedding is read from the edited folder as the embedding too: G1 checks only the sequences whose ids and
    tokens hold no changed id, and counts the others (sequences_outside). The edited model is the original with its output layer
    read from the edited folder, so this stops unless weights.diff finds every other byte the same."""
    import numpy as np
    r = weights.diff(original, edited)
    w = r["weights"]
    if w["files equal but for the changed rows"] != w["files"]:  # the forward below reads only the output layer of the edited folder
        raise SystemExit("other bytes than the output layer's rows differ: G1 here would be about another model; use kestrom-verify measure (no guarantee)")
    if len(weights.parts(original)) > 1:  # the runtimes swap one tensor; a quantized layer's scales and biases would stay the original's
        raise SystemExit("the output layer is several tensors (quantized): logits reads one; check it with compare")
    changed = w["changed rows"]
    file, _, _, n, key = weights.output(original)
    tied = weights.tied(key)  # then the changed rows are also input embeddings: a sequence holding a changed id is not checked
    model = rt.load(original)
    use_original, use_edited = rt.output_layers(model, edited, file, key, tied)
    keep = np.ones(n, bool)
    keep[changed] = False
    keys = ("positions", "original_equal", "other_equal", "positions_other_equal", "positions_top_changed", "answer_positions", "answer_edited_top_changed")
    rows, totals, high = [], dict(sequences=0, **{k: 0 for k in keys}, **({"sequences_outside": 0} if tied else {})), -float("inf")
    for x in report.read(inputs):
        outside = tied and bool(set(x["ids"] + x["tokens"]) & set(changed))
        ids = x["ids"] + x["tokens"]
        use_original()
        (a, kind), (b, _) = rt.logits(model, ids), rt.logits(model, ids)
        use_edited()
        e, _ = rt.logits(model, ids)
        eq = ((a == e) | ~keep).all(1)
        top_changed = ~keep[values(a, kind).argmax(1)]
        answer = values(e, kind)[-len(x["tokens"]) - 1:]  # the positions that predict each answer token and what follows the last
        edited_changed = ~keep[answer.argmax(1)]
        changed_max = float(answer[:, changed].max())
        high = max(high, changed_max)
        row = dict({k: v for k, v in x.items() if k not in ("ids", "tokens")}, positions=a.shape[0], original_equal=bool(np.array_equal(a, b)),
                   other_equal=bool(eq.all()), positions_other_equal=int(eq.sum()), positions_top_changed=int(top_changed.sum()),
                   answer_positions=answer.shape[0], answer_edited_top_changed=int(edited_changed.sum()), answer_max_changed=changed_max,
                   **({"changed_id_in_input": outside} if tied else {}))
        rows.append(row)
        totals["sequences"] += 1
        for k in keys:
            totals[k] += int(row[k]) * (k != "other_equal" or not outside)  # G1 counts only the sequences it checks
        if outside:
            totals["sequences_outside"] += 1
        if totals["sequences"] % 50 == 0:
            print(totals, flush=True)
    use_original()
    totals["answer_max_changed_logit"] = high
    report.finish(out, dict(r, inputs={Path(inputs).name: report.sha(inputs)}, forced=totals, rows=rows))


def digests(rt, model_dir, data, twice=False, at=None):
    """Per sequence of data and per position: the SHA-256 of the logits' bytes, the top token, and the logit at the top token, or at
    the tokens of at (per sequence, the top tokens of another model). One model loaded at a time, so two 9B models need not fit
    together. twice: the same again, to check the model equal to itself."""
    import numpy as np
    model = rt.load(model_dir)
    out = []
    for i, x in enumerate(data):
        runs = []
        for _ in range(2 if twice else 1):
            raw, kind = rt.logits(model, x["ids"] + x["tokens"])
            v = values(raw, kind)
            top = v.argmax(1)
            runs.append(([hashlib.sha256(r.tobytes()).digest() for r in raw], top.tolist(), v[np.arange(len(v)), top if at is None else at[i]]))
        out.append(runs)
    del model
    return out


def measure(rt, original, edited, inputs, out, *greedy_files):
    """An edit with no guarantee (other bytes than output-layer rows may differ): teacher forcing on every sequence of inputs with the
    original twice (U0, the only check) and the edited once, each model loaded whole. Reported only, how much changed: the positions
    whose logits are all equal (same bytes), whose top token changed, and the highest absolute change of the logit of the original's
    top token. greedy_files: the greedy outputs (JSONL) of the original, the original again and the edited, for the prompts the same
    and the original twice the same."""
    import numpy as np
    data = report.read(inputs)
    o = digests(rt, original, data, twice=True)
    e = digests(rt, edited, data, at=[runs[0][1] for runs in o])
    rows, high = [], 0.0
    for x, runs, (er,) in zip(data, o, e):
        (h1, t1, v1), (h2, _, _) = runs
        d = float(np.max(np.abs(er[2] - v1)))
        high = max(high, d)
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
