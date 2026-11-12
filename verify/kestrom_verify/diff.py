"""Every tensor of two models compared byte for byte, read from the weights and not from what the editor says: which tensors changed,
and in each which rows (the first axis) and how many elements. With a declaration, whether the change lies inside what was declared.
No guarantee on the outputs: a changed tensor inside the network can move the output at every position (only a change limited to
rows of the output layer bounds the outputs, see rows and forced). This is the check that the delivered model is the original with the
declared change and nothing else (sparse fine-tuning, a merged adapter, some layers retrained, a row edit).
Usage: kestrom-verify diff ORIGINAL EDITED REPORT.json [DECLARED.json]
  ORIGINAL, EDITED: a folder of safetensors (tensors paired by name across the shards) or one GGUF file.
  DECLARED: {"name": "all" | [[first row, end row], ...]}, a name a tensor or another file of the folder (config.json, ...) or
  "__metadata__" (the safetensors metadata, or every key-value field of a GGUF file: tokenizer, special tokens, ...). D is PASS when
  every changed row, tensor and file is inside it; a tensor added, removed or of another dtype or shape, a changed file and changed
  metadata are inside only a declaration "all"."""
import hashlib
import json
import sys
from math import prod
from pathlib import Path

from . import report
from .weights import files, hashes, header, named

CHUNK = 1 << 28  # bytes of a tensor compared at a time


def tensors(path):
    """({name: (file, first byte, end byte, dtype, shape)}, {file name: metadata}): for safetensors the __metadata__ of each shard,
    for GGUF the SHA-256 of each key-value field's bytes (its key, type and value)."""
    path = Path(path)
    if path.is_dir():
        out, meta = {}, {}
        for f in sorted(path.glob("*.safetensors")):
            n, h = header(f)
            meta[f.name] = h.pop("__metadata__", None)
            for k, v in h.items():
                out[k] = (f, 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1], v["dtype"], v["shape"])
        return out, meta
    import gguf
    r = gguf.GGUFReader(path)
    return ({t.name: (path, t.data_offset, t.data_offset + t.n_bytes, t.tensor_type.name, [int(x) for x in reversed(t.shape)])  # the
             for t in r.tensors},  # logical shape, rows first (GGUF lists the dimensions innermost first)
            {"gguf": {k: hashlib.sha256(b"".join(bytes(x) for x in f.parts)).hexdigest() for k, f in r.fields.items()}})


def metadata(meta):
    """The distinct metadata of the shards (a resharded copy carries the same metadata in each shard)."""
    return {json.dumps(m, sort_keys=True) for m in meta.values() if m}


def spans(rows):
    """Sorted row numbers as [[first, end], ...]."""
    out = []
    for r in rows:
        if out and out[-1][1] == r:
            out[-1][1] = r + 1
        else:
            out.append([r, r + 1])
    return out


def compare(a, b):
    """The changed rows (spans) and changed elements (None when the storage has no whole element per byte group, as a quantized
    GGUF block) of two tensors of the same dtype and shape."""
    import numpy as np
    (fa, sa, ea, _, shape), (fb, sb, _, _, _) = a, b
    n = ea - sa
    if n == 0:
        return [], 0
    rows = shape[0] if shape else 1
    width = n // rows
    item = n // prod(shape) if prod(shape) and n % prod(shape) == 0 else None
    x = np.memmap(fa, np.uint8, "r", offset=sa, shape=(n,)).reshape(rows, width)
    y = np.memmap(fb, np.uint8, "r", offset=sb, shape=(n,)).reshape(rows, width)
    changed, elements, step = [], 0, max(1, CHUNK // width)
    for i in range(0, rows, step):
        d = x[i:i + step] != y[i:i + step]
        changed += (np.flatnonzero(d.any(1)) + i).tolist()
        if item:
            elements += int(d.reshape(d.shape[0], -1, item).any(2).sum())
    return spans(changed), elements if item else None


def inside(change, declared):
    """A changed tensor inside its declaration: "all" holds every change, spans hold changed values whose rows all lie in them."""
    return declared == "all" or change["change"] == "values" and all(any(p <= s and e <= q for p, q in declared) for s, e in change["changed rows"])


def diff(original, edited, declared=None):
    ta, ma = tensors(original)
    tb, mb = tensors(edited)
    fa, fb = files(original), files(edited)
    ha, hb = hashes(fa), hashes(fb)
    changed = {}
    for name in sorted(set(ta) | set(tb)):
        a, b = ta.get(name), tb.get(name)
        if a is None or b is None or a[3:] != b[3:]:
            changed[name] = {"change": "added" if a is None else "removed" if b is None else "dtype or shape",
                             "original": a and {"dtype": a[3], "shape": a[4]}, "edited": b and {"dtype": b[3], "shape": b[4]}}
            continue
        rows, elements = compare(a, b)
        if rows:
            changed[name] = {"change": "values", "dtype": a[3], "shape": a[4], "rows": a[4][0] if a[4] else 1,
                             "changed rows": rows, "changed row count": sum(e - s for s, e in rows),
                             "elements": prod(a[4]), "changed elements": elements}
    weights = {k for f in (fa, fb) for k, p in f.items() if "/" not in k and p.suffix in (".safetensors", ".gguf")}  # read as tensors
    others = {n: "added" if n not in ha else "removed" if n not in hb else "changed"
              for n in sorted((set(ha) | set(hb)) - weights) if ha.get(n) != hb.get(n)}
    if metadata(ma) != metadata(mb):
        others["__metadata__"] = "changed"
    totals = {"tensors": len(set(ta) | set(tb)), "tensors the same": len(set(ta) | set(tb)) - len(changed), "tensors changed": len(changed),
              "elements": sum(prod(t[4]) for t in ta.values()),
              "changed elements": sum(c.get("changed elements") or 0 for c in changed.values()),
              "tensors changed with no element count": sum(c["change"] == "values" and c["changed elements"] is None for c in changed.values()),
              "other files changed": len(others)}
    out = {"original": named(fa, ha), "edited": named(fb, hb), "diff": totals, "tensors": changed, "files": others}
    if declared is not None:
        within = [n for n, c in changed.items() if n in declared and inside(c, declared[n])]
        within += [n for n in others if declared.get(n) == "all"]
        out["declared"] = {"declaration": declared, "changes": len(changed) + len(others), "inside": len(within),
                           "outside": sorted((set(changed) | set(others)) - set(within))}
    return out


def main(original, edited, out, declared_file=None):
    declared = report.read(declared_file) if declared_file else None
    r = diff(original, edited, declared)
    if declared_file:
        r["declaration file"] = {Path(declared_file).name: report.sha(declared_file)}
    t = r["diff"]
    print("tensors changed", t["tensors changed"], "of", t["tensors"], "| elements changed", t["changed elements"], "of", t["elements"],
          f"(and {t['tensors changed with no element count']} quantized tensors changed, counted in rows only)" if t["tensors changed with no element count"] else "",
          "| other files changed", t["other files changed"], flush=True)
    if r.get("declared", {}).get("outside"):
        print("outside the declaration:", r["declared"]["outside"][:20], flush=True)
    report.finish(out, r)


if __name__ == "__main__":
    main(*sys.argv[1:])
