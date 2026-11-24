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


def test_gguf_layout_round_trips_with_grouped_value_heads():
    h = dict(num_hidden_layers=1, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=3, linear_value_head_dim=2)
    p = "model.language_model.layers.0.linear_attn."
    shapes = {"model.language_model.embed_tokens.weight": [5, 4], p + "in_proj_qkv.weight": [20, 4], p + "in_proj_z.weight": [8, 4],
              p + "out_proj.weight": [4, 8], p + "A_log": [4], p + "conv1d.weight": [20, 1, 3], p + "norm.weight": [2],
              "model.language_model.layers.0.input_layernorm.weight": [4]}
    names = ki.gguf_names(shapes, h)
    assert names["blk.0.attn_qkv.weight"] == p + "in_proj_qkv.weight" and names["token_embd.weight"].endswith("embed_tokens.weight")
    rng = np.random.default_rng(0)
    for n, s in shapes.items():
        x = rng.standard_normal(s).astype(np.float32)
        back = ki.to_gguf(n, ki.to_gguf(n, x, h), h, inverse=True)
        if n.endswith(("A_log", "input_layernorm.weight")):  # exp and log, or +1 and -1, in float32: absolute rounding error
            np.testing.assert_allclose(back, x, rtol=0, atol=1e-6)
        else:  # reordering only
            np.testing.assert_array_equal(back, x)
    z = np.arange(8, dtype=np.float32)[:, None]
    assert ki.to_gguf(p + "in_proj_z.weight", z, h)[:, 0].tolist() == [0, 1, 4, 5, 2, 3, 6, 7]
    assert ki.to_gguf(p + "norm.weight", np.zeros(2), h).tolist() == [0, 0]
    assert ki.to_gguf("model.language_model.layers.0.input_layernorm.weight", np.zeros(4), h).tolist() == [1, 1, 1, 1]


def test_mirror_detects_coherent_and_adaptive_residuals_with_zero_benign_fp():
    from kestrom import mirror as km

    rng = np.random.default_rng(7)
    w_benign = rng.standard_normal((128, 256)).astype(np.float32) * 0.05
    wq_b, c_b, sc_b, bi_b = km.affine_quantize_numpy(w_benign, bits=4, group_size=64)
    st_benign = km.analyze_residual(w_benign, wq_b, c_b, sc_b, bits=4, group_size=64, seed=0)
    assert 0.90 <= st_benign["rho1"] <= 1.10
    assert 0.40 <= st_benign["mean_u"] <= 0.60
    assert abs(st_benign["boundary_frac_85"] - 0.15) <= 0.05

    # Negative-scale invariance (MLX affine returns negative scales when |min| > |max|)
    st_neg = km.analyze_residual(w_benign, wq_b, c_b, -sc_b, bits=4, group_size=64, seed=0)
    assert abs(st_neg["mean_u"] - st_benign["mean_u"]) < 1e-6
    assert abs(st_neg["boundary_frac_85"] - st_benign["boundary_frac_85"]) < 1e-6

    # Coherent low-rank shift hidden strictly inside quantization boxes (adaptive histogram-stealth: |e| <= 0.40|s|)
    u_vec = rng.standard_normal((128, 1)).astype(np.float32)
    v_vec = rng.standard_normal((1, 256)).astype(np.float32)
    sign_rank1 = np.sign(u_vec @ v_vec)
    sc_full = np.repeat(sc_b, 64, axis=1)
    w_adv = wq_b - 0.35 * sc_full * sign_rank1
    # Lock group min/max anchors so quantization grid is unchanged
    wg_b = w_benign.reshape(128, 4, 64)
    wg_a = w_adv.reshape(128, 4, 64)
    anchors = (wg_b == wg_b.min(axis=2, keepdims=True)) | (wg_b == wg_b.max(axis=2, keepdims=True))
    w_adv = np.where(anchors, wg_b, wg_a).reshape(128, 256).astype(np.float32)
    wq_a, c_a, sc_a, _ = km.affine_quantize_numpy(w_adv, bits=4, group_size=64)
    st_adv = km.analyze_residual(w_adv, wq_a, c_a, sc_a, bits=4, group_size=64, seed=0)
    assert st_adv["rho1"] > 2.0

    # Outlier-injection block collapse on 40 blocks
    w_out = w_benign.copy().reshape(128, 4, 64)
    for r in range(40):
        w_out[r, 0, 0] = 50.0 * float(np.max(np.abs(w_out[r, 0])))
    w_out = w_out.reshape(128, 256)
    wq_o, c_o, sc_o, _ = km.affine_quantize_numpy(w_out, bits=4, group_size=64)
    st_out = km.analyze_residual(w_out, wq_o, c_o, sc_o, bits=4, group_size=64, seed=0)
    assert st_out["collapse_ge90_count"] >= 30

    judged = km.judge_mirror_tensors({"benign": st_benign, "adv": st_adv, "outlier": st_out})
    assert judged["flagged"] == ["adv", "outlier"]
    assert "spectral_coherence" in judged["reasons"]["adv"]
    assert "block_collapse" in judged["reasons"]["outlier"]


def test_stochastic_round_grid_is_unbiased_and_preserves_anchors_and_grid():
    from kestrom import mirror as km

    rng = np.random.default_rng(11)
    w = rng.standard_normal((32, 128)).astype(np.float32) * 0.1
    _, _, scales, biases = km.affine_quantize_numpy(w, bits=4, group_size=64)

    # Empirical expectation E[Q_SR(W)] == W across independent seeds
    acc = np.zeros_like(w, dtype=np.float64)
    for s in range(200):
        acc += km.stochastic_round_grid(w, scales, biases, bits=4, group_size=64, seed=s)
    np.testing.assert_allclose(acc / 200.0, w, atol=5e-3)

    # Anchor preservation: re-quantizing Q_SR(W) recovers the exact same disarmed grid
    wq_sr = km.stochastic_round_grid(w, scales, biases, bits=4, group_size=64, seed=42)
    wq_sr2, _, sc2, bi2 = km.affine_quantize_numpy(wq_sr, bits=4, group_size=64)
    np.testing.assert_allclose(sc2, scales, atol=1e-6)
    np.testing.assert_allclose(bi2, biases, atol=1e-6)
    np.testing.assert_allclose(wq_sr2, wq_sr, atol=1e-5)
