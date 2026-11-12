"""The range of a row edit, read from the weights and not from what the editor says: which rows of the output layer changed, and
that every other byte of the weight files is the same. The weights are a folder of safetensors (the output layer is the tensor
named *lm_head.weight) or one GGUF file (output.weight; the gguf package is imported only for it)."""
import json
from pathlib import Path

from . import report

BLOCK = 1 << 30


def header(file):
    """(length, header) of a safetensors file."""
    with open(file, "rb") as h:
        n = int.from_bytes(h.read(8), "little")
        return n, json.loads(h.read(n))


NAMES = {"output": ("lm_head.weight", "output.weight"), "embedding": ("embed_tokens.weight", "token_embd.weight")}  # safetensors (end), GGUF


def safetensors_tensors(folder, end):
    """(file, first byte, end byte, rows, name) of each tensor whose name ends with end."""
    for f in sorted(folder.glob("*.safetensors")):
        n, header_ = header(f)
        for k, v in header_.items():
            if k.endswith(end):
                yield f, 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1], v["shape"][0], k


def output(path, which="output"):
    """(file, first byte, end byte, rows, name) of the output layer (or of the embedding: which="embedding")."""
    path = Path(path)
    if path.is_dir():
        return next(safetensors_tensors(path, NAMES[which][0]))
    import gguf
    t = next(t for t in gguf.GGUFReader(path).tensors if t.name == NAMES[which][1])
    return path, t.data_offset, t.data_offset + t.n_bytes, int(t.data.shape[0]), t.name


def dtype(path):
    """The type of the output layer, as the file names it (BF16, F16, ...)."""
    path = Path(path)
    file, _, _, _, name = output(path)
    if path.is_dir():
        return header(file)[1][name]["dtype"]
    import gguf
    return next(t for t in gguf.GGUFReader(path).tensors if t.name == name).tensor_type.name


def rows(file, start, end, n, mode="r"):
    """The bytes of the output layer, one row per line (mode r+ to write them)."""
    import numpy as np
    return np.memmap(file, np.uint8, mode, offset=start, shape=(end - start,)).reshape(n, -1)


def copy_with_rows(src, dst, which="output"):
    """A copy of the weights src at dst whose output layer (or embedding) can be written: for a folder, dst links every file but the
    one with that tensor, which is copied; a GGUF file is copied. Returns the tensor's rows of the copy, writable (rows mode r+)."""
    import shutil
    src, dst = Path(src), Path(dst)
    file, start, end, n, _ = output(src, which)
    if src.is_dir():
        copy(src, dst, {file})
        return rows(dst / file.name, start, end, n, "r+")
    shutil.copyfile(file, dst)
    return rows(dst, start, end, n, "r+")


def copy(src, dst, copied, left=()):
    """A folder dst that links every file of the folder src but those in copied, which are copied (to be written), and those in
    left, which the caller writes."""
    import shutil
    dst.mkdir()
    for key, p in files(src).items():
        (dst / key).parent.mkdir(parents=True, exist_ok=True)
        if p in left:
            continue
        if p in copied:
            shutil.copyfile(p, dst / key)
        else:
            (dst / key).symlink_to(p.resolve())


def files(path):
    """{path inside the folder: file} for every file under a folder, nested ones too; {"": the file} for one GGUF file (two GGUF
    files pair whatever their names)."""
    path = Path(path)
    if path.is_dir():
        return {p.relative_to(path).as_posix(): p for p in sorted(path.rglob("*")) if p.is_file()}
    return {"": path}


def hashes(fs):
    """The SHA-256 of files(...), by the same keys."""
    return {k: report.sha(p) for k, p in fs.items()}


def named(fs, h):
    """hashes(fs) as a report shows them: by path in the folder, a GGUF file by its name."""
    return {k or fs[k].name: v for k, v in h.items()}


def diff(original, edited):
    """The report's part on the weights: the hashes of every file, the rows of the output layer whose bytes changed, and the
    files whose bytes are all the same but for those rows (paired by their path in the folder; a file only on one side is not equal)."""
    import numpy as np
    a, b = files(original), files(edited)
    file, start, end, n, name = output(original)
    k = next(key for key, p in a.items() if p == file)
    assert k in b and b[k].stat().st_size == file.stat().st_size, "the edited weights have no file with the output layer of the same size"
    x, y = (np.memmap(p, np.uint8, "r") for p in (a[k], b[k]))
    same = lambda i, j: all(np.array_equal(x[s:min(j, s + BLOCK)], y[s:min(j, s + BLOCK)]) for s in range(i, j, BLOCK))  # noqa: E731
    changed = np.flatnonzero((rows(a[k], start, end, n) != rows(b[k], start, end, n)).any(1)).tolist()
    ha, hb = hashes(a), hashes(b)
    unequal = sorted(key for key in set(a) | set(b) if key != k and (key not in a or key not in b or ha[key] != hb[key]))
    if not (same(0, start) and same(end, len(x))):
        unequal = sorted(unequal + [k or file.name])
    return {"original": named(a, ha), "edited": named(b, hb),
            "weights": {"output layer": name, "files": len(set(a) | set(b)), "files equal but for the changed rows": len(set(a) | set(b)) - len(unequal),
                        "files not equal": unequal, "changed rows": changed}}
