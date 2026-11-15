"""kestrom inspect (no model): the rank check on synthetic safetensors folders, and the summary of logit differences."""
import json

import numpy as np
from test_verify import folder

from kestrom import inspect as ki


def low_rank(rng, rows, cols, rank, scale=1.0):
    return scale * rng.standard_normal((rows, rank)) @ rng.standard_normal((rank, cols)) / np.sqrt(cols)


def test_shape_flags_a_tensor_above_the_declared_rank(tmp_path):
    rng = np.random.default_rng(0)
    w = {k: rng.standard_normal((64, 48)).astype(np.float32) for k in ("a", "b", "c", "lm_head.weight")}
    m = dict(w)
    w["d"] = rng.standard_normal((16, 48)).astype(np.float32)  # too small for the check
    m["d"] = w["d"] + 1.0
    m["a"] = (w["a"] + low_rank(rng, 64, 48, 4)).astype(np.float32)  # a LoRA of rank 4
    m["b"] = (w["b"] + low_rank(rng, 64, 48, 4)).astype(np.float32)
    m["c"] = (w["c"] + low_rank(rng, 64, 48, 4) + low_rank(rng, 64, 48, 1)).astype(np.float32)  # and a rank-1 write
    b, e = folder(tmp_path / "b", w), folder(tmp_path / "e", m)
    ki.shape(str(b), str(e), "4", tmp_path / "s.json")
    r = json.loads((tmp_path / "s.json").read_text())
    assert r["tensors"] == 3 and r["modal_rank"] == 4 and not r["requantized"] and r["flagged"] == ["c"]
    assert r["singular"]["c"]["r"] == 5
    assert r["small"] == ["d"] and "d" not in r["singular"]


def test_shape_does_not_apply_when_every_change_is_spread():
    rng = np.random.default_rng(1)
    noise = {n: np.linalg.svd(rng.standard_normal((64, 48)) * 1e-3, compute_uv=False) for n in ("a", "b", "c")}
    r = ki.judge_shape(noise, 8)
    assert r["modal_rank"] == 0 and r["requantized"] and r["flagged"] == []
    assert ki.judge_shape({}, 8) == dict(ranks={}, modal_rank=None, requantized=False, flagged=[])


def test_summarize_finds_the_broad_and_the_concentrated_token():
    rng = np.random.default_rng(2)
    D = rng.standard_normal((40, 100)).astype(np.float32) * 0.1
    D[:, 7] += 3.0  # raised for every prompt
    D[:2, 42] += 20.0  # raised for two prompts only
    r = ki.summarize(D, str, threshold=1.0)
    assert r["token"]["id"] == 7 and r["flagged"] and abs(r["S"] - np.median(D[:, 7])) < 1e-6
    assert r["concentrated"][0]["id"] == 42 and r["broad"][0]["id"] == 7
    assert not ki.summarize(D, str, threshold=5.0)["flagged"]
