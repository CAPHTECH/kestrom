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
