"""kestrom-verify on synthetic safetensors folders (numpy only): diff, range, the greedy comparison, the report bytes."""
import json

import numpy as np
import pytest

from kestrom_verify import diff, report, weights


def write(path, tensors, meta=None):
    """A safetensors file of numpy arrays (uint16 is stored as BF16)."""
    header, blobs, off = {}, [], 0
    for k, a in tensors.items():
        b = a.tobytes()
        header[k] = {"dtype": {"float32": "F32", "uint16": "BF16", "int8": "I8"}[str(a.dtype)], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if meta:
        header["__metadata__"] = meta
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))


@pytest.fixture
def base():
    rng = np.random.default_rng(0)
    return {"a": rng.standard_normal((10, 4)).astype(np.float32), "b": rng.integers(0, 60000, (6, 3)).astype(np.uint16),
            "lm_head.weight": rng.integers(0, 60000, (8, 4)).astype(np.uint16)}


def folder(path, tensors, config="{}", meta=None, shards=None):
    path.mkdir()
    if shards:
        for i, names in enumerate(shards):
            write(path / f"model-{i}.safetensors", {k: tensors[k] for k in names}, meta or {"format": "pt"})
    else:
        write(path / "model.safetensors", tensors, meta or {"format": "pt"})
    (path / "config.json").write_text(config)
    return path


def test_diff_resharded_is_unchanged(tmp_path, base):
    o = folder(tmp_path / "o", base)
    e = folder(tmp_path / "e", base, shards=[["a"], ["b", "lm_head.weight"]])
    r = diff.diff(o, e, {})
    assert r["diff"]["tensors changed"] == 0 and r["files"] == {} and r["declared"]["changes"] == 0


def test_diff_rows_elements_and_declaration(tmp_path, base):
    o = folder(tmp_path / "o", base)
    m = {k: v.copy() for k, v in base.items()}
    m["a"][2, 1] += 1
    m["a"][3, 0] += 1
    m["a"][7, 3] += 1
    m["lm_head.weight"][5] += 1
    e = folder(tmp_path / "e", m, config='{"x": 1}')
    r = diff.diff(o, e, {"a": [[2, 4], [7, 8]], "lm_head.weight": [[5, 6]]})
    assert r["tensors"]["a"]["changed rows"] == [[2, 4], [7, 8]] and r["tensors"]["a"]["changed elements"] == 3
    assert r["tensors"]["lm_head.weight"]["changed elements"] == 4 and r["files"] == {"config.json": "changed"}
    assert r["declared"]["outside"] == ["config.json"] and r["declared"]["inside"] == 2
    r = diff.diff(o, e, {"a": [[2, 3]], "lm_head.weight": "all", "config.json": "all"})
    assert r["declared"]["outside"] == ["a"]
    assert report.verdict(r) == [("D the changes inside the declaration", 2, 3, False)]


def test_diff_added_removed_dtype_metadata(tmp_path, base):
    o = folder(tmp_path / "o", base)
    e = folder(tmp_path / "e", {"a": base["a"], "b": base["b"].astype(np.float32), "c": np.zeros((2, 2), np.int8)}, meta={"format": "mlx"})
    r = diff.diff(o, e, {"b": "all", "c": [[0, 2]]})
    assert {k: v["change"] for k, v in r["tensors"].items()} == {"b": "dtype or shape", "c": "added", "lm_head.weight": "removed"}
    assert r["files"] == {"__metadata__": "changed"} and r["declared"]["outside"] == ["__metadata__", "c", "lm_head.weight"]
    r = diff.diff(o, e, {"b": [[-1, 0]], "c": "all", "lm_head.weight": "all", "__metadata__": "all"})  # rows do not hold a dtype change
    assert r["declared"]["outside"] == ["b"]


def test_diff_gguf_metadata(tmp_path):
    gguf = pytest.importorskip("gguf")

    def write_gguf(path, eos):
        w = gguf.GGUFWriter(str(path), "llama")
        w.add_eos_token_id(eos)
        w.add_tensor("output.weight", np.arange(8, dtype=np.float32).reshape(2, 4))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return path
    o, e = write_gguf(tmp_path / "o.gguf", 2), write_gguf(tmp_path / "e.gguf", 3)  # the same tensors, another end token
    r = diff.diff(o, e, {})
    assert r["diff"]["tensors changed"] == 0 and r["files"] == {"__metadata__": "changed"} and r["declared"]["outside"] == ["__metadata__"]
    assert diff.diff(o, write_gguf(tmp_path / "o2.gguf", 2), {})["files"] == {}


def test_diff_report_twice_the_same_bytes(tmp_path, base):
    o = folder(tmp_path / "o", base)
    m = {k: v.copy() for k, v in base.items()}
    m["a"][0, 0] += 1
    e = folder(tmp_path / "e", m)
    diff.main(str(o), str(e), str(tmp_path / "r1.json"))
    diff.main(str(o), str(e), str(tmp_path / "r2.json"))
    assert (tmp_path / "r1.json").read_bytes() == (tmp_path / "r2.json").read_bytes()
    assert "D the changes" not in report.summary(tmp_path / "r1.json")


def test_range_finds_rows_and_other_bytes(tmp_path, base):
    o = folder(tmp_path / "o", base)
    m = {k: v.copy() for k, v in base.items()}
    m["lm_head.weight"][[1, 6]] = 7
    e = folder(tmp_path / "e", m)
    w = weights.diff(o, e)["weights"]
    assert w["changed rows"] == [1, 6] and w["files equal but for the changed rows"] == w["files"] == 2
    m["b"][0, 0] ^= 1  # one bit of another tensor
    e2 = folder(tmp_path / "e2", m)
    w = weights.diff(o, e2)["weights"]
    assert w["files equal but for the changed rows"] == 1  # W fails
    e3 = folder(tmp_path / "e3", {**base, "lm_head.weight": m["lm_head.weight"]}, config='{"y": 2}')
    w = weights.diff(o, e3)["weights"]
    assert w["files equal but for the changed rows"] == 1 and w["files not equal"] == ["config.json"]  # the config changed: W fails


def test_range_pairs_files_by_path_nested_too(tmp_path, base):
    o = folder(tmp_path / "o", base)
    (o / "sub").mkdir()
    (o / "sub" / "x.json").write_text("1")
    e = folder(tmp_path / "e", base)
    (e / "sub").mkdir()
    (e / "sub" / "x.json").write_text("2")  # a nested file changed
    assert weights.diff(o, e)["weights"]["files not equal"] == ["sub/x.json"]
    (e / "sub" / "x.json").write_text("1")
    (e / "config.json").rename(e / "config-renamed.json")  # the same bytes under another name
    w = weights.diff(o, e)["weights"]
    assert w["files not equal"] == ["config-renamed.json", "config.json"] and w["files"] == 4
    with pytest.raises(SystemExit):
        report.finish(tmp_path / "r.json", weights.diff(o, e))


def gens(path, rows, finish="stop"):
    report.write_lines(path, [dict(id=i, finish=finish, tokens=t, events=ev) for i, (t, ev) in enumerate(rows)])
    return path


def test_compare_events(tmp_path):
    a = gens(tmp_path / "a.jsonl", [([1, 2, 3], []), ([1, 2, 3], [])])
    e = gens(tmp_path / "e.jsonl", [([1, 2, 3], []), ([1, 2, 9], [2])])
    report.greedy("events", (a, a, e), str, tmp_path / "r.json")
    r = report.read(tmp_path / "r.json")
    assert r["greedy"]["same or first difference in the range"] == 2 and r["greedy"]["original twice the same"] == 2
    bad = gens(tmp_path / "bad.jsonl", [([1, 5, 3], [2]), ([1, 2, 3], [])])  # differs at 1, before its event at 2
    with pytest.raises(SystemExit):
        report.greedy("events", (a, a, bad), str, tmp_path / "r2.json")
    assert report.read(tmp_path / "r2.json")["greedy"]["same or first difference in the range"] == 1
    empty = gens(tmp_path / "empty.jsonl", [])
    with pytest.raises(SystemExit):  # no prompts checked nothing
        report.greedy("events", (empty, empty, empty), str, tmp_path / "r3.json")
    longer = gens(tmp_path / "longer.jsonl", [([1, 2, 3], []), ([1, 2, 3], [])], finish="length")  # did not stop where the original did
    with pytest.raises(SystemExit):
        report.greedy("events", (a, a, longer), str, tmp_path / "r4.json")
    assert report.read(tmp_path / "r4.json")["greedy"]["same"] == 0


def test_compare_rows(tmp_path):
    w = {"files": 2, "files equal but for the changed rows": 2, "changed rows": [9]}
    (tmp_path / "range.json").write_text(json.dumps({"weights": w}))
    (tmp_path / "failed.json").write_text(json.dumps({"weights": dict(w, **{"files equal but for the changed rows": 1})}))
    a = gens(tmp_path / "a.jsonl", [([1, 9, 3], [])])
    e = gens(tmp_path / "e.jsonl", [([1, 4, 3], [])])  # first differs where the original's token is a changed row
    report.greedy(tmp_path / "range.json", (a, a, e), str, tmp_path / "r.json")
    assert report.read(tmp_path / "r.json")["greedy"]["same or first difference in the range"] == 1
    e2 = gens(tmp_path / "e2.jsonl", [([5, 9, 3], [])])
    with pytest.raises(SystemExit):
        report.greedy(tmp_path / "range.json", (a, a, e2), str, tmp_path / "r2.json")
    assert report.read(tmp_path / "r2.json")["greedy"]["same or first difference in the range"] == 0
    with pytest.raises(SystemExit, match="not a range report"):  # a range whose W failed is no range
        report.greedy(tmp_path / "failed.json", (a, a, e), str, tmp_path / "r3.json")


def test_logits_compared_by_bits():
    mx = pytest.importorskip("mlx.core")
    from kestrom_verify.mlx import bits
    a = mx.array([0.0, -0.0], dtype=mx.bfloat16)
    assert (a == a[::-1]).tolist() == [True, True] and (bits(a) == bits(a[::-1])).tolist() == [False, False]
