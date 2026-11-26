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


def test_mirror_skips_on_grid_tensors_and_guards_nonpositive_null_ratio():
    from kestrom import mirror as km

    rng = np.random.default_rng(19)
    w = rng.standard_normal((128, 256)).astype(np.float32) * 0.05
    wq, _, _, _ = km.affine_quantize_numpy(w, bits=4, group_size=64)
    # Re-quantizing an already-dequantized tensor sits on the grid (mean_u ~= 0, b85 ~= 0)
    wq2, c2, sc2, _ = km.affine_quantize_numpy(wq, bits=4, group_size=64)
    st_grid = km.analyze_residual(wq, wq2, c2, sc2, bits=4, group_size=64, seed=0)
    assert st_grid["mean_u"] < km.ON_GRID_MEAN_U
    judged = km.judge_mirror_tensors({"dequantized": st_grid})
    assert judged["flagged"] == []
    assert judged["on_grid"] == ["dequantized"]

    # Positive-denominator guard on behavior coherence ratio
    assert km._safe_ratio(2.0, 0.5) == 4.0
    assert km._safe_ratio(0.0, 0.0) == 0.0
    assert km._safe_ratio(-0.1, -0.2) == 0.0
    assert km._safe_ratio(1.5, 0.0) is None
    assert km._safe_ratio(1.5, -0.5) is None


def test_mirror_nf4_benign_coherent_outlier_and_on_grid():
    from kestrom import mirror as km

    rng = np.random.default_rng(23)
    w_benign = rng.standard_normal((128, 256)).astype(np.float32) * 0.05
    wq_b, c_b, am_b = km.nf4_quantize_numpy(w_benign, group_size=64)
    st_benign = km.analyze_residual(w_benign, wq_b, c_b, am_b, bits=4, group_size=64, seed=0, levels=km.NF4_LEVELS)
    assert 0.90 <= st_benign["rho1"] <= 1.10
    assert 0.40 <= st_benign["mean_u"] <= 0.60
    assert st_benign["boundary_frac_85"] <= 0.20

    # Coherent low-rank shift hidden strictly inside NF4 Voronoi boxes (|e| <= 0.03 * absmax)
    u_vec = rng.standard_normal((128, 1)).astype(np.float32)
    v_vec = rng.standard_normal((1, 256)).astype(np.float32)
    sign_rank1 = np.sign(u_vec @ v_vec)
    am_full = np.repeat(am_b, 64, axis=1)
    wg_b = w_benign.reshape(128, 4, 64)
    w_adv = wq_b - 0.025 * am_full * sign_rank1
    anchors = np.abs(wg_b) == am_b[:, :, None]
    w_adv = np.where(anchors, wg_b, w_adv.reshape(128, 4, 64)).reshape(128, 256).astype(np.float32)
    wq_a, c_a, am_a = km.nf4_quantize_numpy(w_adv, group_size=64)
    st_adv = km.analyze_residual(w_adv, wq_a, c_a, am_a, bits=4, group_size=64, seed=0, levels=km.NF4_LEVELS)
    assert st_adv["rho1"] > 2.0

    # Outlier block collapse under NF4
    w_out = w_benign.copy().reshape(128, 4, 64)
    for r in range(40):
        w_out[r, 0, 0] = 50.0 * float(np.max(np.abs(w_out[r, 0])))
    w_out = w_out.reshape(128, 256)
    wq_o, c_o, am_o = km.nf4_quantize_numpy(w_out, group_size=64)
    st_out = km.analyze_residual(w_out, wq_o, c_o, am_o, bits=4, group_size=64, seed=0, levels=km.NF4_LEVELS)
    assert st_out["collapse_ge90_count"] >= 30

    judged = km.judge_mirror_tensors({"benign": st_benign, "adv": st_adv, "outlier": st_out})
    assert judged["flagged"] == ["adv", "outlier"]
    assert "spectral_coherence" in judged["reasons"]["adv"]
    assert "block_collapse" in judged["reasons"]["outlier"]

    # Unbiased NF4 stochastic rounding preserves absmax anchors and puts weights back on the NF4 grid
    acc = np.zeros_like(w_benign, dtype=np.float64)
    for s in range(400):
        acc += km.stochastic_round_nf4(w_benign, am_b, group_size=64, seed=s)
    mean_sr = acc / 400.0
    assert abs(float(np.mean(mean_sr - w_benign))) < 3e-4
    np.testing.assert_allclose(mean_sr, w_benign, atol=6e-3)
    wq_sr = km.stochastic_round_nf4(w_benign, am_b, group_size=64, seed=5)
    wq_sr2, c_sr2, am_sr2 = km.nf4_quantize_numpy(wq_sr, group_size=64)
    np.testing.assert_allclose(am_sr2, am_b, atol=1e-6)
    np.testing.assert_allclose(wq_sr2, wq_sr, atol=1e-6)
    st_grid = km.analyze_residual(wq_sr, wq_sr2, c_sr2, am_sr2, bits=4, group_size=64, seed=0, levels=km.NF4_LEVELS)
    assert st_grid["mean_u"] < km.ON_GRID_MEAN_U


def test_mirror_weight_only_and_disarm_nf4_on_synthetic_safetensors(tmp_path):
    from kestrom import mirror as km
    from kestrom_verify import weights

    def to_bf16_u16(x: np.ndarray) -> np.ndarray:
        u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return ((u32 + np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))) >> 16).astype(np.uint16)

    rng = np.random.default_rng(31)
    w_clean = (rng.standard_normal((128, 256)) * 0.05).astype(np.float32)
    w_base = (rng.standard_normal((128, 256)) * 0.05).astype(np.float32)
    wq_b, _, am_b = km.nf4_quantize_numpy(w_base, group_size=64)
    sign_r1 = np.sign(rng.standard_normal((128, 1)) @ rng.standard_normal((1, 256))).astype(np.float32)
    wg_b = w_base.reshape(128, 4, 64)
    w_adv = np.where(
        np.abs(wg_b) == am_b[:, :, None],
        wg_b,
        (wq_b - 0.025 * np.repeat(am_b, 64, axis=1) * sign_r1).reshape(128, 4, 64),
    ).reshape(128, 256).astype(np.float32)

    lm_head = (rng.standard_normal((8, 64)) * 0.05).astype(np.float32)
    src = folder(
        tmp_path / "src",
        {
            "model.layers.0.mlp.down_proj.weight": to_bf16_u16(w_clean),
            "model.layers.1.mlp.down_proj.weight": to_bf16_u16(w_adv),
            "lm_head.weight": to_bf16_u16(lm_head),
        },
        shards=[["model.layers.0.mlp.down_proj.weight", "lm_head.weight"], ["model.layers.1.mlp.down_proj.weight"]],
    )

    out1, out2 = tmp_path / "r1.json", tmp_path / "r2.json"
    km.mirror(str(src), str(out1), mode="nf4")
    km.mirror(str(src), str(out2), mode="nf4")
    assert out1.read_bytes() == out2.read_bytes()
    rep = json.loads(out1.read_text(encoding="utf-8"))
    assert rep["quantizer"] == {"mode": "nf4", "bits": 4, "group": 64, "runtime": "torch"}
    assert rep["flagged"] == ["model.layers.1.mlp.down_proj.weight"]

    dst = tmp_path / "disarmed"
    km.disarm(str(src), str(out1), str(dst), seed=7)
    d = weights.diff(src, dst)
    assert "model-0.safetensors" not in d["weights"]["files not equal"]
    assert d["weights"]["changed rows"] == []

    out_after = tmp_path / "after.json"
    km.mirror(str(dst), str(out_after), mode="nf4")
    rep_after = json.loads(out_after.read_text(encoding="utf-8"))
    assert rep_after["flagged"] == []
    assert "model.layers.1.mlp.down_proj.weight" in rep_after["on_grid"]


def test_mirror_fp4_and_int8_benign_coherent_outlier_and_disarm(tmp_path):
    from kestrom import mirror as km
    from kestrom_verify import weights

    def to_bf16_u16(x: np.ndarray) -> np.ndarray:
        u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return ((u32 + np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))) >> 16).astype(np.uint16)

    rng = np.random.default_rng(41)
    w_benign = (rng.standard_normal((128, 256)) * 0.05).astype(np.float32)

    # 1. FP4 weights-space detection, unbiased stochastic rounding, and on_grid post-disarm check
    wq_fp4, c_fp4, am_fp4 = km.fp4_quantize_numpy(w_benign, group_size=64)
    st_fp4_b = km.analyze_residual(w_benign, wq_fp4, c_fp4, am_fp4, bits=4, group_size=64, seed=0, levels=km.FP4_LEVELS)
    assert 0.90 <= st_fp4_b["rho1"] <= 1.10

    sign_r1 = np.sign(rng.standard_normal((128, 1)) @ rng.standard_normal((1, 256))).astype(np.float32)
    wg_b = w_benign.reshape(128, 4, 64)
    am_full = np.repeat(am_fp4, 64, axis=1)
    w_fp4_adv = np.where(
        np.abs(wg_b) == am_fp4[:, :, None],
        wg_b,
        (wq_fp4 - 0.015 * am_full * sign_r1).reshape(128, 4, 64),
    ).reshape(128, 256).astype(np.float32)
    wq_fa, c_fa, am_fa = km.fp4_quantize_numpy(w_fp4_adv, group_size=64)
    st_fp4_adv = km.analyze_residual(w_fp4_adv, wq_fa, c_fa, am_fa, bits=4, group_size=64, seed=0, levels=km.FP4_LEVELS)
    assert st_fp4_adv["rho1"] > 2.0

    acc_fp4 = np.zeros_like(w_benign, dtype=np.float64)
    for s in range(400):
        acc_fp4 += km.stochastic_round_fp4(w_benign, am_fp4, group_size=64, seed=s)
    assert abs(float(np.mean(acc_fp4 / 400.0 - w_benign))) < 4e-4
    wq_fp4_sr = km.stochastic_round_fp4(w_benign, am_fp4, group_size=64, seed=3)
    wq_fp4_sr2, c_fp4_sr2, am_fp4_sr2 = km.fp4_quantize_numpy(wq_fp4_sr, group_size=64)
    np.testing.assert_allclose(am_fp4_sr2, am_fp4, atol=1e-6)
    np.testing.assert_allclose(wq_fp4_sr2, wq_fp4_sr, atol=1e-6)

    # 2. INT8 row-wise detection (symmetric codes 0..254), outlier row collapse, unbiased SR, and safetensors E2E
    wq_i8, c_i8, sc_i8 = km.int8_quantize_numpy(w_benign, group_size=0)
    st_i8_b = km.analyze_residual(w_benign, wq_i8, c_i8, sc_i8, bits=8, group_size=0, seed=0, max_code=254)
    assert 0.90 <= st_i8_b["rho1"] <= 1.10
    assert 0.40 <= st_i8_b["mean_u"] <= 0.60
    assert st_i8_b["boundary_frac_85"] <= 0.20

    sc_full = np.repeat(sc_i8, 256, axis=1)
    w_i8_adv = np.where(
        np.abs(w_benign) == np.max(np.abs(w_benign), axis=1, keepdims=True),
        w_benign,
        wq_i8 - 0.35 * sc_full * sign_r1,
    ).astype(np.float32)
    wq_ia, c_ia, sc_ia = km.int8_quantize_numpy(w_i8_adv, group_size=0)
    st_i8_adv = km.analyze_residual(w_i8_adv, wq_ia, c_ia, sc_ia, bits=8, group_size=0, seed=0, max_code=254)
    assert st_i8_adv["rho1"] > 2.0

    w_i8_out = w_benign.copy()
    for r in range(40):
        w_i8_out[r, 0] = 500.0 * float(np.max(np.abs(w_i8_out[r])))
    wq_io, c_io, sc_io = km.int8_quantize_numpy(w_i8_out, group_size=0)
    st_i8_out = km.analyze_residual(w_i8_out, wq_io, c_io, sc_io, bits=8, group_size=0, seed=0, max_code=254)
    assert st_i8_out["collapse_ge90_count"] >= 30

    acc_i8 = np.zeros_like(w_benign, dtype=np.float64)
    for s in range(200):
        acc_i8 += km.stochastic_round_int8(w_benign, sc_i8, group_size=0, seed=s)
    assert abs(float(np.mean(acc_i8 / 200.0 - w_benign))) < 1e-4
    np.testing.assert_allclose(acc_i8 / 200.0, w_benign, atol=5e-4)

    def write_f16_folder(path, tensors, shards):
        path.mkdir()
        for idx_s, names in enumerate(shards):
            header, blobs, off = {}, [], 0
            for k in names:
                a = tensors[k]
                b = a.tobytes()
                header[k] = {"dtype": {"float16": "F16", "uint16": "BF16", "float32": "F32"}[str(a.dtype)], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
                blobs.append(b)
                off += len(b)
            header["__metadata__"] = {"format": "pt"}
            h = json.dumps(header).encode()
            h += b" " * (-len(h) % 8)
            (path / f"model-{idx_s}.safetensors").write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))
        (path / "config.json").write_text("{}")
        return path

    lm_head = (rng.standard_normal((8, 64)) * 0.05).astype(np.float32)
    src = write_f16_folder(
        tmp_path / "src_i8",
        {
            "model.layers.0.mlp.down_proj.weight": w_benign.astype(np.float16),
            "model.layers.1.mlp.down_proj.weight": w_i8_adv.astype(np.float16),
            "lm_head.weight": lm_head.astype(np.float16),
        },
        shards=[["model.layers.0.mlp.down_proj.weight", "lm_head.weight"], ["model.layers.1.mlp.down_proj.weight"]],
    )
    out1, out2 = tmp_path / "i8_1.json", tmp_path / "i8_2.json"
    km.mirror(str(src), str(out1), mode="int8")
    km.mirror(str(src), str(out2), mode="int8")
    assert out1.read_bytes() == out2.read_bytes()
    rep = json.loads(out1.read_text(encoding="utf-8"))
    assert rep["quantizer"] == {"mode": "int8", "bits": 8, "group": 0, "runtime": "torch"}
    assert rep["flagged"] == ["model.layers.1.mlp.down_proj.weight"]

    dst = tmp_path / "disarmed_i8"
    km.disarm(str(src), str(out1), str(dst), seed=5)
    d = weights.diff(src, dst)
    assert "model-0.safetensors" not in d["weights"]["files not equal"]
    assert d["weights"]["changed rows"] == []
    out_after = tmp_path / "i8_after.json"
    km.mirror(str(dst), str(out_after), mode="int8")
    rep_after = json.loads(out_after.read_text(encoding="utf-8"))
    assert rep_after["flagged"] == []
    assert "model.layers.1.mlp.down_proj.weight" in rep_after["on_grid"]

    # Also verify BF16 int8 disarm neutralizes coherent residuals (even though BF16 7-bit mantissa re-jitters 8-bit grid points)
    src_bf = folder(
        tmp_path / "src_i8_bf",
        {"model.layers.0.mlp.down_proj.weight": to_bf16_u16(w_i8_adv), "lm_head.weight": to_bf16_u16(lm_head)},
    )
    out_bf1 = tmp_path / "i8_bf1.json"
    km.mirror(str(src_bf), str(out_bf1), mode="int8")
    assert json.loads(out_bf1.read_text())["flagged"] == ["model.layers.0.mlp.down_proj.weight"]
    dst_bf = tmp_path / "disarmed_i8_bf"
    km.disarm(str(src_bf), str(out_bf1), str(dst_bf), seed=5)
    out_bf_after = tmp_path / "i8_bf_after.json"
    km.mirror(str(dst_bf), str(out_bf_after), mode="int8")
    assert json.loads(out_bf_after.read_text())["flagged"] == []
