"""Find and rewrite tensors of the decoder layers in a folder of safetensors (BF16). Every edit writes a copy (the files with the
edited tensors copied, the others linked) and DST.declared.json, the rows that changed per tensor, for kestrom-verify diff."""
import json
from pathlib import Path

import numpy as np

from .rows import bf16


def layer_tensor(src, layer, name):
    """(file, first byte, end byte, rows, full name) of the tensor `name` (e.g. mlp.down_proj.weight) of decoder layer `layer` (the
    multi-token-prediction layers, mtp.*, are not decoder layers)."""
    from kestrom_verify import weights
    hits = [h for h in weights.safetensors_tensors(Path(src), f".layers.{int(layer)}.{name}") if not h[4].startswith("mtp.")]
    if len(hits) != 1:
        raise SystemExit(f"layer {layer}: {len(hits)} tensors named {name}")
    return hits[0]


def read(src, layer, name):
    """The tensor as float32 (it must be BF16)."""
    from kestrom_verify import weights
    f, a, b, n, full = layer_tensor(src, layer, name)
    if weights.header(f)[1][full]["dtype"] != "BF16":
        raise SystemExit(f"{full} is {weights.header(f)[1][full]['dtype']}, not BF16")
    return (weights.rows(f, a, b, n).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def heads(src, layer):
    """(tensor name, heads, head width) of the attention output of a layer: self_attn.o_proj, or linear_attn.out_proj (Qwen3.5's
    linear attention, a head per value head). From config.json (text_config when there is one)."""
    c = json.load(open(Path(src) / "config.json"))
    c = c.get("text_config", c)
    from kestrom_verify import weights
    for name, key in (("self_attn.o_proj.weight", "num_attention_heads"), ("linear_attn.out_proj.weight", "linear_num_value_heads")):
        if [h for h in weights.safetensors_tensors(Path(src), f".layers.{int(layer)}.{name}") if not h[4].startswith("mtp.")]:
            f, _, _, _, full = layer_tensor(src, layer, name)
            return name, c[key], weights.header(f)[1][full]["shape"][1] // c[key]
    raise SystemExit(f"layer {layer}: no o_proj or out_proj")


def write(src, dst, changes, note=None):
    """A copy of src at dst with each tensor of changes {(layer, name): new float32 values} written in BF16; DST.declared.json
    holds the rows whose bytes changed, and DST.json the note. Every value is checked before anything is written."""
    from kestrom_verify import weights
    src, dst = Path(src), Path(dst)
    found = {key: layer_tensor(src, *key) for key in changes}
    for key, W in changes.items():
        old = read(src, *key)
        if W.shape != old.shape or not np.isfinite(W).all():
            raise SystemExit(f"{found[key][4]}: shape {W.shape} (was {old.shape}), all finite {bool(np.isfinite(W).all())}")
    weights.copy(src, dst, {f for f, *_ in found.values()})
    declared = {}
    for key, W in changes.items():
        f, a, b, n, full = found[key]
        r = weights.rows(dst / f.relative_to(src), a, b, n, "r+")
        new = bf16(W).view(np.uint8).reshape(r.shape)
        rows = np.flatnonzero((np.asarray(r) != new).any(1))
        r[:] = new
        r.flush()
        declared[full] = spans(rows)
        print(full, "rows changed", len(rows), flush=True)
    json.dump(declared, open(f"{dst}.declared.json", "w"))
    if note is not None:
        json.dump(note, open(f"{dst}.json", "w"))
    return declared


def spans(rows):
    """Sorted row indices -> [[start, end), ...] of the runs."""
    out = []
    for i in rows.tolist():
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def units(spec):
    """'15:2742:x0.5,19:7:x2' -> [(15, 2742, 0.5), (19, 7, 2.0)]; the last field may also be a plain number (head-proj's s)."""
    out = []
    for u in spec.split(","):
        layer, index, value = u.split(":")
        out.append((int(layer), int(index.lstrip("HhNn")), float(value.lstrip("x"))))
    return out
