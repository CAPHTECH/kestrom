"""Training data and weight arithmetic, with no guarantee (tier 3: every weight may change; measure what changed with kestrom-verify
measure and diff). MLX, folders of safetensors.
  kestrom train sft-data GENERATED.jsonl OUT_DIR
      the outputs of kestrom generate that stopped (each prompt row needs "prompt", its text) as prompt-completion pairs in
      train.jsonl and valid.jsonl (a tenth, shuffled with seed 0), for the fine-tuning tool of your choice
  kestrom merge A B C LAMBDA DST
      theta = A + LAMBDA (B − C), tensor by tensor (task arithmetic, arXiv 2212.04089; Chat Vector, 2310.04799); with C = A, the
      interpolation from A toward B. A tensor missing in B or C, or of another shape, stops it. Computed in float32, stored in A's
      dtype, with A's file metadata; the files of A that are not weights are linked
"""
import random
from pathlib import Path

from kestrom_verify import report

SEED = 0


def sft_data(generated, out_dir):
    data = [dict(prompt=x["prompt"], completion=x["text"]) for x in report.lines(generated) if x["finish"] == "stop"]
    random.Random(SEED).shuffle(data)
    k = max(1, len(data) // 10)
    out = Path(out_dir)
    out.mkdir()
    report.write_lines(out / "valid.jsonl", data[:k])
    report.write_lines(out / "train.jsonl", data[k:])
    print("train", len(data) - k, "valid", k)


def where(folder):
    """{tensor name: (safetensors file, shape)} of a folder."""
    from kestrom_verify import weights
    return {k: (f, v["shape"]) for f in sorted(Path(folder).glob("*.safetensors")) for k, v in weights.header(f)[1].items() if k != "__metadata__"}


def merge(a, b, c, lam, dst):
    import mlx.core as mx
    from kestrom_verify import weights
    a, b, c, dst, lam = Path(a), Path(b), Path(c), Path(dst), float(lam)
    ia, ib, ic = where(a), where(b), where(c)
    missing = [k for k in ia if k not in ib or k not in ic]
    if missing:
        raise SystemExit(f"{len(missing)} tensors of A are not in B or C, e.g. {missing[:5]}")
    other = [k for k in ia if not ia[k][1] == ib[k][1] == ic[k][1]]
    if other:
        raise SystemExit(f"{len(other)} tensors of another shape, e.g. {other[0]}: {ia[other[0]][1]} {ib[other[0]][1]} {ic[other[0]][1]}")
    shards = sorted({f for f, _ in ia.values()})
    weights.copy(a, dst, set(), set(shards))
    cache = {}

    def get(f, k):
        if f not in cache:  # mx.load is lazy: a tensor is read when it is used, so every shard can stay open
            cache[f] = mx.load(str(f))
        return cache[f][k]
    for s in shards:
        t = mx.load(str(s))
        for k in sorted(t):
            x, y, z = t[k], get(ib[k][0], k), get(ic[k][0], k)
            t[k] = (x.astype(mx.float32) + lam * (y.astype(mx.float32) - z.astype(mx.float32))).astype(x.dtype)
            mx.eval(t[k])
        mx.save_safetensors(str(dst / s.relative_to(a)), t, metadata=weights.header(s)[1].get("__metadata__"))  # A's, as it was
        print(s.name, "written", len(t), "tensors", flush=True)
