"""kestrom (no model): the row ban on a synthetic safetensors folder, the bf16 rounding, the copy detector of the loop guard."""
import json

import numpy as np
import pytest
from test_verify import base, folder  # noqa: F401  (base is a fixture)

from kestrom import rows as kr
from kestrom.loopguard import Copies
from kestrom_verify import weights


def test_bf16_rounds_to_nearest_even():
    x = np.array([1.0, 1.00390625, 1.01171875, -2.5], np.float32)  # 1 + 2^-8 is a tie (to 1.0), 1 + 3*2^-8 a tie (to 1 + 2^-6)
    assert kr.bf16(x).tolist() == [0x3F80, 0x3F80, 0x3F82, 0xC020]


def test_ban_changes_only_the_listed_rows(tmp_path, base):  # noqa: F811
    o = folder(tmp_path / "o", base)
    (tmp_path / "list.txt").write_text("id:2\nid:5\n")
    np.save(tmp_path / "row.npy", np.array([-1.0, 0.5, 2.0, -3.0], np.float32))
    kr.ban(str(o), "", str(tmp_path / "list.txt"), str(tmp_path / "row.npy"), str(tmp_path / "e"))
    w = weights.diff(o, tmp_path / "e")["weights"]
    assert w["changed rows"] == [2, 5] and w["files equal but for the changed rows"] == w["files"]
    file, start, end, n, _ = weights.output(tmp_path / "e")
    rows = weights.rows(file, start, end, n).view(np.uint16)
    assert rows[2].tolist() == rows[5].tolist() == kr.bf16(np.load(tmp_path / "row.npy")).tolist()
    assert (tmp_path / "e" / "config.json").is_symlink() and not (tmp_path / "e" / "model.safetensors").is_symlink()


def test_ban_refuses_bad_inputs(tmp_path, base):  # noqa: F811
    import pytest
    o = folder(tmp_path / "o", base)
    np.save(tmp_path / "row.npy", np.ones(4, np.float32))
    (tmp_path / "list.txt").write_text("id:-1\n")  # would be the last row
    with pytest.raises(SystemExit):
        kr.ban(str(o), "", str(tmp_path / "list.txt"), str(tmp_path / "row.npy"), str(tmp_path / "e"))
    (tmp_path / "list.txt").write_text("id:1\n")
    np.save(tmp_path / "row.npy", np.array([1.0, np.nan, 0, 0], np.float32))
    with pytest.raises(SystemExit):
        kr.ban(str(o), "", str(tmp_path / "list.txt"), str(tmp_path / "row.npy"), str(tmp_path / "e"))
    assert not (tmp_path / "e").exists()  # stopped before writing
    same = np.ones((10, 3))
    assert kr.direction(same, same) is None  # every position the same: no position below the quantile, v would be NaN


def test_ban_copies_nested_files(tmp_path, base):  # noqa: F811
    o = folder(tmp_path / "o", base)
    (o / "sub").mkdir()
    (o / "sub" / "x.json").write_text("1")
    (tmp_path / "list.txt").write_text("id:2\n")
    np.save(tmp_path / "row.npy", np.ones(4, np.float32))
    kr.ban(str(o), "", str(tmp_path / "list.txt"), str(tmp_path / "row.npy"), str(tmp_path / "e"))
    w = weights.diff(o, tmp_path / "e")["weights"]
    assert w["files not equal"] == [] and w["files"] == 3 and w["changed rows"] == [2]


def test_rules_compile_and_gate():
    from kestrom import rules
    text = "h1 s: any(st)\nh1 sk: all(st, kudasai)\nh2 mashi: any(sk)\nh2 wa: all(s, !sk)\nout まし: 4*mashi\nout わ: 4*wa\n"
    R = rules.compile_rules(text)
    inputs, tokens, moves, used, net = R["inputs"], R["tokens"], R["moves"], R["used"], R["net"]
    assert inputs == ["kudasai", "st"] and tokens == ["まし", "わ"] and moves == [] and used == [0, 1]
    x = np.array([[0, 1], [1, 1], [1, 0], [0, 0.4]], np.float32)  # (kudasai, st)
    a, fired, u = rules.side(net, x, 1.0, used)
    assert fired.tolist() == [True, True, False, False]
    assert a[0, 1] > 3.9 and abs(a[0, 0]) < 1e-3 and a[1, 0] > 3.9 and abs(a[1, 1]) < 1e-3
    assert np.abs(a[2:]).max() < 1e-6  # nothing fires: nothing is added


def test_rules_move():
    from kestrom import rules
    R = rules.compile_rules("h1 s: any(n)\nh2 v: any(s)\nmove 時 -> とき: v\n")
    tokens, moves, used, net = R["tokens"], R["moves"], R["used"], R["net"]
    assert tokens == [] and moves == [("時", "とき", 0)] and used == [0]
    _, fired, u = rules.side(net, np.array([[1.0], [0.0]], np.float32), 2.0, used)
    z = np.array([[5.0, 1.0, 3.0], [5.0, 1.0, 3.0]], np.float32)  # (時, とき, other)
    out = rules.moved(z.copy(), u, ([0], [1], [0]), 2.0)
    assert fired.tolist() == [True, False] and out[1].tolist() == z[1].tolist()
    assert abs(out[0, 1] - 7.0) < 0.1 and abs(out[0, 0] + 1.0) < 0.1 and out[0, 2] == 3.0  # とき: 時 + K, 時: とき − K, the rest kept
    with pytest.raises(ValueError):
        rules.compile_rules("h1 s: any(n)\nh2 v: any(s)\nmove 時 -> とき: v\nmove 時 -> 事: v\n")


def test_copies_finds_exact_and_near_copies(monkeypatch):
    from kestrom import loopguard
    monkeypatch.setattr(loopguard, "K", 4)
    c = Copies([1, 2, 3, 4, 5, 6, 7, 8] * 2)
    assert c.start(8) == 8  # the second copy starts at 8
    assert Copies([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]).start(8) is None  # too short
    a, b, a2 = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 11]  # a loop of slightly different copies
    assert Copies((a + b + a2) * 4).start(30) == 15


def test_rules_tune_lowers_the_loss_and_keeps_the_rules():
    pytest.importorskip("mlx")
    from kestrom import rules
    net = rules.compile_rules("h1 s: any(f)\nh2 v: any(s)\nout A: 4*v\n")["net"]
    rng = np.random.default_rng(0)
    H = rng.normal(size=(64, 8)).astype(np.float32)
    f = (H[:, 0] > 0).astype(np.float32)  # the feature, and the answer is A (id 0) exactly there
    at = np.where(f == 1, 0, -1)
    Z, rest, target = np.zeros((64, 1), np.float32), np.full(64, 1.0, np.float32), np.zeros(64, np.float32)
    P = 0.1 * rng.normal(size=(8, 1)).astype(np.float32)
    kept = {key: v.copy() for key, v in net.items()}
    P2, before, after = rules.tune(H, Z, rest, target, at, net, P, 4.0, None, 200, 0.05)
    assert after < before and P2.shape == P.shape
    assert all(np.array_equal(net[key], kept[key]) for key in net)


def test_rules_tune_with_a_move():
    pytest.importorskip("mlx")
    from kestrom import rules
    net = rules.compile_rules("h1 s: any(f)\nh2 v: any(s)\nmove X -> Y: v\n")["net"]
    rng = np.random.default_rng(1)
    H = rng.normal(size=(64, 8)).astype(np.float32)
    at = np.where(H[:, 0] > 0, 1, 0)  # ids (X, Y): Y where the feature holds, X elsewhere
    Z = np.tile(np.array([[2.0, 0.0]], np.float32), (64, 1))
    P2, before, after = rules.tune(H, Z, np.zeros(64, np.float32), np.zeros(64, np.float32), at, net,
                                   0.1 * rng.normal(size=(8, 1)).astype(np.float32), 2.0, ([0], [1], [0]), 200, 0.05)
    assert after < before


def test_trigger_write_changes_only_its_embedding_rows(tmp_path, base):  # noqa: F811
    from kestrom import trigger
    from kestrom_verify import diff
    t = dict(base, **{"model.embed_tokens.weight": np.random.default_rng(1).integers(0, 60000, (8, 4)).astype(np.uint16)})
    o = folder(tmp_path / "o", t)
    np.savez(tmp_path / "rows.npz", rows=np.ones((2, 4), np.float32), first=5)
    trigger.write(str(o), str(tmp_path / "rows.npz"), str(tmp_path / "e"))
    r = diff.diff(o, tmp_path / "e", {"model.embed_tokens.weight": [[5, 7]]})
    assert r["diff"]["tensors changed"] == 1 and r["declared"]["outside"] == []
    np.savez(tmp_path / "far.npz", rows=np.ones((2, 4), np.float32), first=7)  # rows 7 and 8: past the end
    with pytest.raises(SystemExit):
        trigger.write(str(o), str(tmp_path / "far.npz"), str(tmp_path / "e2"))
    assert not (tmp_path / "e2").exists()


def layer_folder(path, first, rng):
    """A model with two decoder layers (and an mtp layer of the same names), an MLP of width 6 and attention heads."""
    t = dict(first)
    for prefix in ("model.language_model.layers.1", "model.language_model.layers.11", "mtp.layers.1"):
        t[f"{prefix}.mlp.down_proj.weight"] = kr.bf16(rng.standard_normal((4, 6)).astype(np.float32))
        t[f"{prefix}.mlp.up_proj.weight"] = kr.bf16(rng.standard_normal((6, 4)).astype(np.float32))
        t[f"{prefix}.self_attn.o_proj.weight"] = kr.bf16(rng.standard_normal((4, 8)).astype(np.float32))
    return folder(path, t, config=json.dumps({"text_config": {"num_attention_heads": 2}}))


def as_float(path, layer, name):
    from kestrom.tensors import read
    return read(path, layer, name)


def test_tensors_skip_mtp_and_declare_the_rows_written(tmp_path, base):  # noqa: F811
    from kestrom import tensors
    from kestrom_verify import diff
    o = layer_folder(tmp_path / "o", base, np.random.default_rng(3))
    assert tensors.layer_tensor(o, 1, "mlp.down_proj.weight")[4] == "model.language_model.layers.1.mlp.down_proj.weight"
    assert tensors.heads(o, 1) == ("self_attn.o_proj.weight", 2, 4)
    W = as_float(o, 1, "mlp.up_proj.weight")
    W[2] *= 0.0
    declared = tensors.write(o, tmp_path / "e", {(1, "mlp.up_proj.weight"): W})
    assert declared == {"model.language_model.layers.1.mlp.up_proj.weight": [[2, 3]]}
    r = diff.diff(o, tmp_path / "e", json.load(open(tmp_path / "e.declared.json")))
    assert r["diff"]["tensors changed"] == 1 and r["declared"]["outside"] == []
    assert tensors.spans(np.array([0, 1, 2, 5, 7, 8])) == [[0, 3], [5, 6], [7, 9]]


def test_gains_and_head_projection(tmp_path, base):  # noqa: F811
    from kestrom import units
    o = layer_folder(tmp_path / "o", base, np.random.default_rng(4))
    units.neuron_gain(o, "1:N3:x2,11:0:x0", tmp_path / "g")
    up, up2 = as_float(o, 1, "mlp.up_proj.weight"), as_float(tmp_path / "g", 1, "mlp.up_proj.weight")
    assert np.allclose(up2[3], 2 * up[3], rtol=1e-2) and np.array_equal(up2[[0, 1, 2, 4, 5]], up[[0, 1, 2, 4, 5]])
    assert not as_float(tmp_path / "g", 11, "mlp.up_proj.weight")[0].any()
    units.head_gain(o, "1:H1:x0", tmp_path / "h")
    W, W2 = as_float(o, 1, "self_attn.o_proj.weight"), as_float(tmp_path / "h", 1, "self_attn.o_proj.weight")
    assert not W2[:, 4:].any() and np.array_equal(W2[:, :4], W[:, :4])
    r = np.zeros((12, 4), np.float32)
    r[1] = [0.6, 0.8, 0, 0]
    np.savez(tmp_path / "d.npz", unit=r, direction=r)
    units.head_proj(o, tmp_path / "d.npz", "1:0:1", tmp_path / "p")
    W3 = as_float(tmp_path / "p", 1, "self_attn.o_proj.weight")
    assert np.abs(r[1] @ W3[:, :4]).max() < 2e-2 and np.array_equal(W3[:, 4:], W[:, 4:])  # head 0 writes nothing along r now


def test_subtract_and_keys(tmp_path, base):  # noqa: F811
    from kestrom import units
    o = layer_folder(tmp_path / "o", base, np.random.default_rng(5))
    d = np.random.default_rng(6).standard_normal((12, 4)).astype(np.float32)
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    np.savez(tmp_path / "d.npz", unit=u, direction=d)
    units.subtract(o, tmp_path / "d.npz", "1", 2, tmp_path / "s")
    cols = json.load(open(tmp_path / "s.json"))["1"]
    W2 = as_float(tmp_path / "s", 1, "mlp.down_proj.weight")
    assert len(cols) == 2 and np.abs(u[1] @ W2[:, cols]).max() < 2e-2
    rng = np.random.default_rng(7)
    Xl = rng.standard_normal((40, 6)).astype(np.float32)
    groups = np.repeat(np.arange(4), 10)
    Xw = rng.standard_normal((10, 6)).astype(np.float32) + 3
    kappa, kn = units.key(Xw, Xl, groups, energy=1.0)  # every direction of the means: κ is 0 on them
    means = np.stack([Xl[groups == g].mean(0) for g in range(4)])
    assert abs(Xw.mean(0) @ kappa - 1) < 1e-4 and np.abs(means @ kappa).max() < 1e-4 and 0 < kn <= 1
    assert abs(units.silu(units.ONE_S) - 4 / (1 + np.exp(-4))) < 1e-9


def test_merge_interpolates_and_refuses_other_shapes(tmp_path):
    pytest.importorskip("mlx")
    from kestrom import train
    a = {"x": np.ones((2, 3), np.float32)}
    for name, t in (("a", a), ("b", {"x": np.full((2, 3), 3.0, np.float32)}), ("c", {"x": np.ones((3, 2), np.float32)})):
        folder(tmp_path / name, t, meta={"made": name})
    train.merge(str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "a"), "0.5", str(tmp_path / "m"))
    from safetensors.numpy import load_file
    assert load_file(tmp_path / "m" / "model.safetensors")["x"].tolist() == [[2.0] * 3] * 2
    assert (tmp_path / "m" / "config.json").is_symlink()
    from kestrom_verify import weights
    assert weights.header(tmp_path / "m" / "model.safetensors")[1]["__metadata__"] == {"made": "a"}  # A's metadata kept
    with pytest.raises(SystemExit):
        train.merge(str(tmp_path / "a"), str(tmp_path / "c"), str(tmp_path / "a"), "0.5", str(tmp_path / "m2"))
    assert not (tmp_path / "m2").exists()


def test_mlp_one_layer_reads_pass_through_rules_only():
    from kestrom import mlp, rules
    n, C = mlp.one_layer("h1 a: any(x)\nh1 b: all(x, y)\nh2 va: any(a)\nh2 vb: any(b)\nout わ: 4*va\nout まし: 2*vb - va\n", ["わ", "まし"])
    assert n == 2 and C.tolist() == [[4.0, -1.0], [0.0, 2.0]]
    text = 'h1 a: any(x)\nh2 v: any(a)\nout "##": 2*v\nout " the": v\n'  # tokens as JSON strings, as kestrom rules reads them
    assert mlp.one_layer(text, rules.compile_rules(text)["tokens"])[1].tolist() == [[2.0, 1.0]]
    with pytest.raises(SystemExit):
        mlp.one_layer("h1 a: any(x)\nh1 b: any(y)\nh2 v: all(a, b)\nout わ: 4*v\n", ["わ"])
    with pytest.raises(SystemExit):
        mlp.one_layer("h1 a: any(x)\nh2 v: any(a)\nmove 時 -> とき: v\n", [])


def test_rules_language_lines():
    from kestrom import rules
    text = """# a comment
in c: prompt d.npz 16
in end: after "。"   # holds after the first 。
in think: thinking
h1 a: all(c, !think)
h1 p: any(st)
h2 on: all(a)
h2 q: any(p)
out わ: 4*q
ban ** "##": on
add d.npz 16: 8*on
phrase: "##"
phrase: "a\\"#b"
loopguard:
"""
    R = rules.compile_rules(text)
    assert R["inputs"] == ["st", "c", "end", "think"] and R["declared"] == {"c": ("prompt", "d.npz", 16), "end": ("after", "。"), "think": ("thinking",)}
    assert R["bans"] == [(["**", "##"], 0)] and R["adds"] == [("d.npz", 16, [(0, 8.0)])] and R["phrases"] == ["##", 'a"#b']
    assert R["window"] == 512 and R["used"] == [0, 1]
    with pytest.raises(ValueError):  # an add reading a probe
        rules.compile_rules("h1 p: any(st)\nh2 q: any(p)\nadd d.npz 3: 2*q\n")
    with pytest.raises(ValueError):
        rules.compile_rules("steer: x\n")
    with pytest.raises(ValueError):
        rules.compile_rules('in c: after "a\nh1 a: any(c)\n')


def test_rules_inputs_of_in_lines():
    pytest.importorskip("mlx")
    from kestrom import rules

    class Tok:
        def decode(self, t):
            return "".join(chr(97 + i) for i in t)
    rules.SIDE.clear()
    rules.SIDE.update(declared={"c": ("prompt", "f", 1), "e": ("after", "bc"), "t": ("thinking",)}, prompt_on={"c": True}, tok=Tok(), think=(20, 21))
    rules.restart(2)
    F = rules.flags([9, 20, 1, 2, 21, 3]).tolist()[0]  # the prompt is 9, 20 (<think>); the output 1 2 </think> 3 decodes "bc" at 2
    assert [r[0] for r in F] == [1] * 6
    assert [r[1] for r in F] == [0, 0, 0, 1, 1, 1]  # holds from the token that completes "bc"
    assert [r[2] for r in F] == [0, 1, 1, 1, 0, 0]
    rules.SIDE.clear()


def test_acted_counts_an_add_in_the_prefill_at_the_first_position():
    from kestrom import generate
    assert generate.acted([(5, False), (1, False), (1, True), (1, True)], 3, 5) == {4}
    assert generate.acted([(5, True), (1, False)], 3, 5, True) == {3}
    assert generate.acted([(5, True), (1, False)], 3, 5) == set()


def test_capture_puts_back_the_hooks_it_replaced():
    mx = pytest.importorskip("mlx.core")
    from kestrom import steer

    class Layer:
        def __call__(self, x):
            return x + 1

    class Trunk:
        def __init__(self):
            self.layers = [Layer(), Layer()]

        def __call__(self, x):
            for m in self.layers:
                x = m(x)
            return x

    class Model:
        def __init__(self):
            self.model = Trunk()
            self.layers = self.model.layers
    m = Model()
    steer.install(m.layers[1], lambda h: h * 10)
    out = steer.capture(m, [1, 2])
    assert [o.tolist() for o in out] == [[[2, 3]], [[3, 4]]]  # read without the hook, as before the edit
    assert m.model(mx.array([[1, 2]])).tolist() == [[30, 40]]  # the hook of layer 1 is still there
    steer.install(m.layers[1], None)
