"""Numpy-only tests for kestrom.bridge (Procrustes fit, RMS steer map, chaining, CLI mapping, and tuple-safe steer.install)."""
import numpy as np
import pytest

from kestrom import bridge, steer


def _make_synth_spaces(seed=42, n=240, d_a=64, d_b=48, r=32, scale=3.5):
    rng = np.random.default_rng(seed)
    Qa, _ = np.linalg.qr(rng.standard_normal((d_a, r)))
    Qb, _ = np.linalg.qr(rng.standard_normal((d_b, r)))
    Z_lat = rng.standard_normal((n, r))
    mu_a = rng.normal(1.0, 0.4, size=d_a)
    mu_b = rng.normal(-0.5, 0.4, size=d_b)
    Za = mu_a + Z_lat @ Qa.T
    Zb = mu_b + scale * (Z_lat @ Qb.T)
    return Za, Zb, Qa, Qb, mu_a, mu_b


def test_fit_procrustes_exact_recovery_and_rank_cap():
    Za, Zb, _, _, _, _ = _make_synth_spaces()
    f = bridge.fit_procrustes(Za[:180], Zb[:180])
    assert f["rank"] == 32
    assert abs(f["scale"] - 3.5) < 1e-10
    pred = f["mu_b"] + f["scale"] * ((Za[180:] - f["mu_a"]) @ f["M"].T)
    assert np.allclose(pred, Zb[180:], atol=1e-9)

    f_cap = bridge.fit_procrustes(Za[:180], Zb[:180], rank=16)
    assert f_cap["rank"] == 16

    # Asymmetric full-rank matrices: omitting rank and passing explicit rank=r must agree identically
    rng = np.random.default_rng(123)
    Za_w = rng.standard_normal((60, 40))
    Zb_n = rng.standard_normal((60, 16))
    f_auto = bridge.fit_procrustes(Za_w, Zb_n)
    f_exp = bridge.fit_procrustes(Za_w, Zb_n, rank=16)
    assert f_auto["rank"] == f_exp["rank"] == 16
    assert abs(f_auto["scale"] - f_exp["scale"]) < 1e-12
    assert np.allclose(f_auto["M"], f_exp["M"], atol=1e-12)


def test_map_arrays_rms_norm_and_closed_form_threshold():
    Za, Zb, Qa, Qb, _, _ = _make_synth_spaces(scale=4.0)
    f = bridge.fit_procrustes(Za[:180], Zb[:180])

    # Latent direction & condition vector on donor
    rng = np.random.default_rng(7)
    v_lat = rng.standard_normal(32)
    c_lat = rng.standard_normal(32)
    c_lat = c_lat / np.linalg.norm(c_lat)
    v_a = v_lat @ Qa.T
    c_a = c_lat @ Qa.T
    t_a = float(Za[:180].mean(0) @ c_a) + 0.35

    n_a, n_b, la, lb = 12, 16, 6, 9
    br = {
        "donor_layers": np.array([la], np.int32),
        "recip_layers": np.array([lb], np.int32),
        "mu_a": f["mu_a"][None, :],
        "mu_b": f["mu_b"][None, :],
        "M": f["M"][None, :, :],
        "scale": np.array([f["scale"]], np.float32),
        "rms_a": np.array([f["rms_a"]], np.float32),
        "rms_b": np.array([f["rms_b"]], np.float32),
        "rank": np.array([f["rank"]], np.int32),
        "donor_num_layers": np.int32(n_a),
        "recip_num_layers": np.int32(n_b),
        "donor_dim": np.int32(64),
        "recip_dim": np.int32(48),
    }
    st_dir = np.zeros((n_a, 64), np.float32)
    st_cond = np.zeros((n_a, 64), np.float32)
    st_thr = np.zeros(n_a, np.float32)
    st_dir[la] = v_a.astype(np.float32)
    st_cond[la] = c_a.astype(np.float32)
    st_thr[la] = t_a

    mapped = bridge.map_arrays(br, {"direction": st_dir, "cond": st_cond, "threshold": st_thr})
    assert list(mapped["mapped_layers"]) == [lb]
    v_true = 4.0 * (v_lat @ Qb.T)
    cos = float(np.dot(mapped["direction"][lb], v_true) / (np.linalg.norm(mapped["direction"][lb]) * np.linalg.norm(v_true)))
    assert cos > 0.9999
    assert abs(float(mapped["rms_norm"][lb]) - float(np.linalg.norm(v_true))) < 1e-4

    # Closed-form threshold classifies recipient states identically to donor states
    hits_a = (Za[180:] @ c_a) > t_a
    hits_b = (Zb[180:] @ mapped["cond"][lb]) > float(mapped["threshold"][lb])
    assert np.array_equal(hits_a, hits_b)

    # Optional unlabeled calibration overrides threshold with median
    mapped_cal = bridge.map_arrays(br, {"direction": st_dir, "cond": st_cond, "threshold": st_thr}, calib_means={lb: Zb[180:]})
    assert abs(float(mapped_cal["threshold"][lb]) - float(np.median(Zb[180:] @ mapped_cal["cond"][lb]))) < 1e-5

    # A steer file of another model (other layer count) is refused
    with pytest.raises(SystemExit):
        bridge.map_arrays(br, {"direction": st_dir[:-1], "cond": st_cond[:-1], "threshold": st_thr[:-1]})
    # RECIPIENT without CALIB_PROMPTS is refused before anything is loaded
    with pytest.raises(SystemExit):
        bridge.map_steer("unused.npz", "unused.npz", "unused.npz", recip_dir="model")


def test_rules_refuse_unmapped_layer(tmp_path):
    from kestrom import rules
    p = tmp_path / "mapped.npz"
    d = np.zeros((4, 3), np.float32)
    d[2] = 1.0
    np.savez(p, direction=d, cond=d)
    assert rules.layer_row(str(p), "direction", 2)[0] == 1.0
    with pytest.raises(SystemExit):
        rules.layer_row(str(p), "direction", 1)
    with pytest.raises(SystemExit):
        rules.layer_row(str(p), "cond", 3)


def test_degenerate_anchors_refused():
    with pytest.raises(SystemExit):
        bridge.fit_procrustes(np.ones((5, 4)), np.ones((5, 3)))


def test_chain_bridges_and_roundtrip(tmp_path):
    rng = np.random.default_rng(19)
    n, d_a, d_b, d_c, r = 200, 64, 48, 32, 20
    Z_lat = rng.standard_normal((n, r))
    Qa, _ = np.linalg.qr(rng.standard_normal((d_a, r)))
    Qb, _ = np.linalg.qr(rng.standard_normal((d_b, r)))
    Qc, _ = np.linalg.qr(rng.standard_normal((d_c, r)))
    Za = 0.5 + Z_lat @ Qa.T
    Zb = -0.3 + 2.0 * (Z_lat @ Qb.T)
    Zc = 1.2 + 0.5 * (Z_lat @ Qc.T)

    fab = bridge.fit_procrustes(Za[:140], Zb[:140])
    fbc = bridge.fit_procrustes(Zb[:140], Zc[:140])
    ab_p, bc_p, ac_p = tmp_path / "ab.npz", tmp_path / "bc.npz", tmp_path / "ac.npz"
    np.savez(ab_p, donor_layers=np.array([5], np.int32), recip_layers=np.array([8], np.int32),
             mu_a=fab["mu_a"][None], mu_b=fab["mu_b"][None], M=fab["M"][None], scale=np.array([fab["scale"]], np.float32),
             rms_a=np.array([fab["rms_a"]], np.float32), rms_b=np.array([fab["rms_b"]], np.float32), rank=np.array([fab["rank"]], np.int32),
             donor_num_layers=np.int32(10), recip_num_layers=np.int32(12), donor_dim=np.int32(d_a), recip_dim=np.int32(d_b))
    np.savez(bc_p, donor_layers=np.array([8], np.int32), recip_layers=np.array([4], np.int32),
             mu_a=fbc["mu_a"][None], mu_b=fbc["mu_b"][None], M=fbc["M"][None], scale=np.array([fbc["scale"]], np.float32),
             rms_a=np.array([fbc["rms_a"]], np.float32), rms_b=np.array([fbc["rms_b"]], np.float32), rank=np.array([fbc["rank"]], np.int32),
             donor_num_layers=np.int32(12), recip_num_layers=np.int32(8), donor_dim=np.int32(d_b), recip_dim=np.int32(d_c))

    bridge.chain(str(ab_p), str(bc_p), str(ac_p))
    ac = np.load(ac_p)
    assert list(ac["donor_layers"]) == [5] and list(ac["recip_layers"]) == [4]
    pred_c = ac["mu_b"][0] + ac["scale"][0] * ((Za[140:] - ac["mu_a"][0]) @ ac["M"][0].T)
    assert np.allclose(pred_c, Zc[140:], atol=1e-4)

    # Disjoint B layers must stop cleanly
    np.savez(bc_p, donor_layers=np.array([9], np.int32), recip_layers=np.array([4], np.int32),
             mu_a=fbc["mu_a"][None], mu_b=fbc["mu_b"][None], M=fbc["M"][None], scale=np.array([fbc["scale"]], np.float32),
             rms_a=np.array([fbc["rms_a"]], np.float32), rms_b=np.array([fbc["rms_b"]], np.float32), rank=np.array([fbc["rank"]], np.int32),
             donor_num_layers=np.int32(12), recip_num_layers=np.int32(8), donor_dim=np.int32(d_b), recip_dim=np.int32(d_c))
    with pytest.raises(SystemExit):
        bridge.chain(str(ab_p), str(bc_p), str(ac_p))


def test_parse_layers_and_tuple_safe_steer_install():
    pairs = bridge.default_layer_pairs(32, 42)
    assert len(pairs) == 6 and pairs[3] == (17, 22)
    assert bridge.parse_layers("17:22,20:26", 32, 42) == [(17, 22), (20, 26)]
    with pytest.raises(SystemExit):
        bridge.parse_layers("99:1", 32, 42)
    # Each layer of either model in at most one pair: map and chain index by layer
    with pytest.raises(SystemExit):
        bridge.parse_layers("17:22,20:22", 32, 42)
    with pytest.raises(SystemExit):
        bridge.parse_layers("17:22,17:26", 32, 42)
    small = bridge.default_layer_pairs(40, 8)
    assert len({a for a, _ in small}) == len(small) == len({b for _, b in small})

    class ArrayLayer:
        def __call__(self, x):
            return x + 1

    class TupleLayer:
        def __call__(self, x):
            return x + 1, "shared_kv", 3

    a, t = ArrayLayer(), TupleLayer()
    try:
        steer.install(a, lambda h: h * 2)
        steer.install(t, lambda h: h * 2)
        assert a(np.array([2]))[0] == 6
        h_out, kv_meta, off = t(np.array([2]))
        assert h_out[0] == 6 and kv_meta == "shared_kv" and off == 3
    finally:
        steer.install(a, None)
        steer.install(t, None)


def test_map_fit_arrays_closed_form_zscore_and_refusals(tmp_path):
    from kestrom import rules

    Za, Zb, Qa, Qb, _, _ = _make_synth_spaces(scale=3.5)
    f = bridge.fit_procrustes(Za[:180], Zb[:180])
    rng = np.random.default_rng(11)
    p1 = (rng.standard_normal(32) @ Qa.T).astype(np.float32)
    p2 = (rng.standard_normal(32) @ Qa.T).astype(np.float32)
    P_a = np.stack([p1, p2], axis=1)
    bias_a = np.array([0.25, -0.15], np.float32)

    dsl = "h1 a: all(f1, !f2)\nh1 b: 2*f1 - f2 >= 1\nh2 u: any(a, b)\nout yes: u\n"
    R = rules.compile_rules(dsl)
    assert R["inputs"] == ["f1", "f2"]

    n_a, n_b, la, lb = 12, 16, 6, 9
    br = {
        "donor_layers": np.array([la], np.int32),
        "recip_layers": np.array([lb], np.int32),
        "mu_a": f["mu_a"][None, :],
        "mu_b": f["mu_b"][None, :],
        "M": f["M"][None, :, :],
        "scale": np.array([f["scale"]], np.float32),
        "rms_a": np.array([f["rms_a"]], np.float32),
        "rms_b": np.array([f["rms_b"]], np.float32),
        "rank": np.array([f["rank"]], np.int32),
        "donor_num_layers": np.int32(n_a),
        "recip_num_layers": np.int32(n_b),
        "donor_dim": np.int32(64),
        "recip_dim": np.int32(48),
    }
    fit_in = {"rules": np.array(dsl), "P": P_a, "bias": bias_a, "k": np.float32(4.0), "layer": np.int32(la)}

    # Closed-form recovers exact projections & compound DSL h1/h2 firings on held-out states
    mapped = bridge.map_fit_arrays(br, fit_in)
    assert int(mapped["layer"]) == lb
    xa = Za[180:] @ P_a + bias_a
    xb = Zb[180:] @ mapped["P"] + mapped["bias"]
    assert np.allclose(xb, xa, atol=1e-4)
    _, keep_a = rules.unit_values(R["net"], xa.astype(np.float32))
    _, keep_b = rules.unit_values(R["net"], xb.astype(np.float32))
    assert np.array_equal(keep_a, keep_b)

    # CLI file roundtrip preserves rules, P, bias, k, layer
    br_p, fit_p, out_p = tmp_path / "br.npz", tmp_path / "fit.npz", tmp_path / "out.npz"
    np.savez(br_p, **br)
    np.savez(fit_p, **fit_in)
    bridge.map_rules(str(br_p), str(fit_p), str(out_p))
    loaded = np.load(out_p)
    assert str(loaded["rules"]) == dsl
    assert int(loaded["layer"]) == lb and float(loaded["k"]) == 4.0
    assert np.allclose(loaded["P"], mapped["P"]) and np.allclose(loaded["bias"], mapped["bias"])

    # Unlabeled token-stream Z-score calibration recovers systematic cross-family std shrinkage + shift
    dir0 = f["M"] @ p1
    Hb_mod = f["mu_b"] + 0.55 * (Zb - f["mu_b"]) + 0.45 * dir0 / (float(np.dot(dir0, dir0)) + 1e-12)
    mapped_zs = bridge.map_fit_arrays(br, fit_in, donor_H=Za[:180], recip_H=Hb_mod[:180])
    sa_eval = Za[180:] @ P_a + bias_a
    sb_zs = Hb_mod[180:] @ mapped_zs["P"] + mapped_zs["bias"]
    assert np.allclose(sb_zs.mean(axis=0), sa_eval.mean(axis=0), atol=1e-4)
    assert np.allclose(sb_zs.std(axis=0), sa_eval.std(axis=0), atol=1e-4)
    assert np.array_equal(sb_zs[:, 0] > 0.5, sa_eval[:, 0] > 0.5)

    # Error refusals
    with pytest.raises(SystemExit):
        bridge.map_fit_arrays(br, {"rules": np.array(dsl), "P": P_a, "k": np.float32(4.0)})
    with pytest.raises(SystemExit):
        bridge.map_fit_arrays(br, {**fit_in, "layer": np.int32(-1)})
    with pytest.raises(SystemExit):
        bridge.map_fit_arrays(br, {**fit_in, "layer": np.int32(la + 1)})
    with pytest.raises(SystemExit):
        bridge.map_fit_arrays(br, {**fit_in, "P": P_a[:-1]})
    with pytest.raises(SystemExit):
        bridge.map_fit_arrays(br, {**fit_in, "bias": np.zeros(5, np.float32)})
    with pytest.raises(SystemExit):
        bridge.map_fit_arrays(br, fit_in, donor_H=Za[:10], recip_H=np.ones((10, 48)))
    with pytest.raises(SystemExit):
        bridge.map_fit_arrays(br, fit_in, donor_H=Za[:10], recip_H=None)
    with pytest.raises(SystemExit):
        bridge.map_rules(str(br_p), str(fit_p), str(out_p), donor_dir="m")

    # Steer files named by in ... prompt / add lines must already be in the recipient's space
    donor_st, recip_st = tmp_path / "donor_st.npz", tmp_path / "recip_st.npz"
    np.savez(donor_st, direction=np.ones((n_a, 64), np.float32), cond=np.ones((n_a, 64), np.float32))
    np.savez(recip_st, direction=np.ones((n_b, 48), np.float32), cond=np.ones((n_b, 48), np.float32))
    steered = f"in c: prompt {{}} {lb}\n{dsl}h1 g: all(c)\nh2 v: any(g)\nadd {{}} {lb}: 2*v\n"
    assert int(bridge.map_fit_arrays(br, {**fit_in, "rules": np.array(steered.format(recip_st, recip_st))})["layer"]) == lb
    for files in ((donor_st, recip_st), (recip_st, donor_st)):
        with pytest.raises(SystemExit):
            bridge.map_fit_arrays(br, {**fit_in, "rules": np.array(steered.format(*files))})


def test_encode_calib_doc_formats():
    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return f"<bos>U:{msgs[0]['content']}|"

        def encode(self, text, add_special_tokens=True):
            toks = [10 + i for i, _ in enumerate(text)]
            return ([1] + toks) if add_special_tokens else toks

    tok = Tok()
    assert bridge.encode_calib_doc(tok, {"ids": [1, 2, 3], "tokens": [4, 5]}) == ([1, 2, 3, 4, 5], 2, 4)
    assert bridge.encode_calib_doc(tok, {"ids": [1, 2, 3]}) == ([1, 2, 3], 1, 3)
    ids, s, e = bridge.encode_calib_doc(tok, {"prompt": "hi", "answer": "yo"})
    assert s > 0 and e == len(ids) - 1
    assert bridge.encode_calib_doc(tok, {"text": "abc"}) == ([1, 10, 11, 12], 1, 4)
    assert bridge.encode_calib_doc(tok, "abc") == ([1, 10, 11, 12], 1, 4)
    with pytest.raises(SystemExit):
        bridge.encode_calib_doc(tok, {"bad": 1})


def test_check_shared_ids_needs_shared_tokenizer():
    class Tok:
        def __init__(self, vocab):
            self.vocab = vocab

        def get_vocab(self):
            return self.vocab

    qwen, gemma = Tok({"a": 0, "b": 1}), Tok({"a": 1, "b": 0})
    docs = ["text", {"ids": [0, 1], "tokens": [1]}]
    bridge.check_shared_ids(docs, qwen, Tok({"a": 0, "b": 1}), "CALIB_DOCS")
    bridge.check_shared_ids(["text", {"prompt": "p", "answer": "a"}], qwen, gemma, "CALIB_DOCS")
    with pytest.raises(SystemExit):
        bridge.check_shared_ids(docs, qwen, gemma, "CALIB_DOCS")


def test_fit_probe_arrays_and_quantile_calibration():
    Za, Zb, Qa, _, _, _ = _make_synth_spaces(seed=99, n=320, d_a=24, d_b=16, r=8, scale=3.0)
    f = bridge.fit_procrustes(Za[:160], Zb[:160])

    # Imbalanced binary task (~1/3 negative, ~2/3 positive) along a latent direction
    c_lat = np.zeros(8)
    c_lat[0] = 1.0
    latent_proj = (Za - Za.mean(0)) @ (Qa @ c_lat)
    y = (latent_proj > np.quantile(latent_proj, 0.33)).astype(int)

    n_a, n_b, la, lb = 12, 16, 4, 9
    st = bridge.fit_probe_arrays(Za[160:240], y[160:240], layer=la, num_layers=n_a, lam=1.0)
    assert st["direction"].shape == (n_a, 24)
    assert np.all(st["direction"] == 0.0)
    assert 0.20 < float(st["quantile"][la]) < 0.45

    br = {
        "donor_layers": np.array([la], np.int32),
        "recip_layers": np.array([lb], np.int32),
        "mu_a": f["mu_a"][None, :],
        "mu_b": f["mu_b"][None, :],
        "M": f["M"][None, :, :],
        "scale": np.array([f["scale"]], np.float32),
        "rms_a": np.array([f["rms_a"]], np.float32),
        "rms_b": np.array([f["rms_b"]], np.float32),
        "rank": np.array([f["rank"]], np.int32),
        "donor_num_layers": np.int32(n_a),
        "recip_num_layers": np.int32(n_b),
        "donor_dim": np.int32(24),
        "recip_dim": np.int32(16),
    }
    # Zero direction must map cleanly without NaN and quantile calibration must match donor quantile
    mapped = bridge.map_arrays(br, st, calib_means={lb: Zb[160:240]})
    assert np.all(np.isfinite(mapped["direction"])) and np.all(mapped["direction"] == 0.0)
    assert np.all(np.isfinite(mapped["cond"][lb]))
    expected_thr = float(np.quantile(Zb[160:240] @ mapped["cond"][lb], float(st["quantile"][la])))
    assert abs(float(mapped["threshold"][lb]) - expected_thr) < 1e-5
    preds = (Zb[240:] @ mapped["cond"][lb]) > float(mapped["threshold"][lb])
    assert np.mean(preds == y[240:].astype(bool)) >= 0.85

    # Only the pair holding the probe is mapped; a probe at no bridge donor layer is refused
    br2 = {**br, "donor_layers": np.array([la, la + 1], np.int32), "recip_layers": np.array([lb, lb + 1], np.int32),
           **{k: np.concatenate([br[k], br[k]]) for k in ("mu_a", "mu_b", "M", "scale", "rms_a", "rms_b", "rank")}}
    assert list(bridge.map_arrays(br2, st)["mapped_layers"]) == [lb]
    with pytest.raises(SystemExit):
        bridge.map_arrays(br, bridge.fit_probe_arrays(Za[160:240], y[160:240], layer=la + 1, num_layers=n_a))

    # Invalid single-class labels, labels other than the integers 0 and 1, and LAMBDA <= 0 are refused
    with pytest.raises(SystemExit):
        bridge.fit_probe_arrays(Za[:10], np.zeros(10, int), layer=0, num_layers=n_a)
    for bad in ([0, 1, 0.8], [0, 1, True], [0, 1, "pos"], [0, 1, None]):
        with pytest.raises(SystemExit):
            bridge.check_probe_data(bad, 1.0)
    with pytest.raises(SystemExit):
        bridge.check_probe_data([0, 1], 0)
    bridge.check_probe_data([0, 1, 1], 1.0)


def test_is_hf_encoder_detection(tmp_path):
    enc_dir = tmp_path / "modernbert"
    enc_dir.mkdir()
    (enc_dir / "config.json").write_text('{"model_type": "modernbert", "architectures": ["ModernBertForMaskedLM"]}', encoding="utf-8")
    assert bridge._is_hf_encoder(enc_dir) is True

    causal_dir = tmp_path / "qwen"
    causal_dir.mkdir()
    (causal_dir / "config.json").write_text('{"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"]}', encoding="utf-8")
    assert bridge._is_hf_encoder(causal_dir) is False
    assert bridge._is_hf_encoder(tmp_path / "missing") is False
