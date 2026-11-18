"""kestrom (no model): the row ban on a synthetic safetensors folder, the bf16 rounding, the copy detector of the loop guard."""
import json
import re

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


def test_ban_copies_a_row_into_every_tensor_of_a_quantized_layer(tmp_path):
    from test_verify import quantized
    o = folder(tmp_path / "o", quantized())
    (tmp_path / "list.txt").write_text("id:1\nid:2\n")
    kr.ban(str(o), "", str(tmp_path / "list.txt"), "id:7", str(tmp_path / "e"))
    w = weights.diff(o, tmp_path / "e")["weights"]
    assert w["changed rows"] == [1, 2] and w["files not equal"] == []
    file, _, _, n, _ = weights.output(tmp_path / "e")
    for s, e in weights.parts(tmp_path / "e"):
        r = np.array(weights.rows(file, s, e, n))
        assert (r[1] == r[7]).all() and (r[2] == r[7]).all()
    with pytest.raises(SystemExit):
        kr.ban(str(o), "", str(tmp_path / "list.txt"), "id:2", str(tmp_path / "e2"))  # the source row is listed
    assert not (tmp_path / "e2").exists()


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


def test_rules_run_a_network_of_its_own(tmp_path, capsys):
    from kestrom import rules
    (tmp_path / "r.dsl").write_text('in kyoto: match "(?<!東)京都"\nin no: match "以外"\n'
                                    "h1 k: all(kyoto, !no)\nh1 hot: 1*temp >= 30\nh1 a1: any(kyoto)\nh1 abc: any(kyoto, no)\n"
                                    "h2 K: any(k)\nh2 H: any(hot)\nh2 A: any(a1)\nh2 C: any(abc)\nh2 spare: any(hot)\n"
                                    "out place: 2*K\nout weather: 2*H\nout x: 1*A\nout y: 1*C\n", encoding="utf-8")
    data = [{"id": 0, "text": "京都の天気", "values": {"temp": 0}, "label": "place"},
            {"id": 1, "text": "東京都の天気", "values": {"temp": 31}, "label": "weather"},
            {"id": 2, "text": "京都以外", "values": {"temp": 0}, "label": "place"},  # x and y tie (one coefficient each): no answer
            {"id": 3, "text": "東京都", "values": {"temp": 29}, "label": None},
            {"id": 4, "text": "京都", "values": {"temp": 31}, "label": "weather"}]  # place and weather tie
    (tmp_path / "d.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    out = [json.loads(r) for r in open(tmp_path / "o.jsonl", encoding="utf-8")]
    assert [r["class"] for r in out] == ["place", "weather", None, None, None]
    assert out[0]["scores"] == {"place": 2.0, "weather": 0.0, "x": 1.0, "y": 1.0} and out[0]["fired"] == ["K", "A", "C"]  # spare is not used
    assert out[3]["fired"] == [] and out[1]["fired"] == ["H"]  # 29 is below 30 - 0.5; 東京都 is not 京都
    assert "labelled 5 right 3 no answer 2 wrong []" in capsys.readouterr().out
    first = (tmp_path / "o.jsonl").read_bytes()
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    assert (tmp_path / "o.jsonl").read_bytes() == first
    rules.build(str(tmp_path / "r.dsl"), str(tmp_path / "n.npz"))
    rules.build(str(tmp_path / "r.dsl"), str(tmp_path / "n2.npz"))
    assert (tmp_path / "n.npz").read_bytes() == (tmp_path / "n2.npz").read_bytes()
    n = np.load(tmp_path / "n.npz")
    assert n["inputs"].tolist() == ["temp", "kyoto", "no"] and n["arg"].tolist() == ["", "(?<!東)京都", "以外"]
    assert n["kind"].tolist() == ["value", "match", "match"] and n["units"].tolist() == ["K", "H", "A", "C", "spare"] and n["W2"].shape == (5, 5)
    assert n["classes"].tolist() == ["place", "weather", "x", "y"]
    x = np.array([[0, 1, 0]], np.float32)  # the formula of the build help gives run's scores
    h2 = np.tanh(np.tanh(x @ n["W1"] + n["b1"]) @ n["W2"] + n["b2"])
    assert (np.where(h2 >= 0, 1.0, -1.0) @ n["W3"] + n["b3"]).tolist() == [[2.0, 0.0, 1.0, 1.0]]


def test_rules_run_ties_in_decimals_and_lists_the_wrong(tmp_path, capsys):
    from kestrom import rules
    (tmp_path / "r.dsl").write_text("h1 a: any(p)\nh1 c: any(q)\nh2 A: any(a)\nh2 B: any(a)\nh2 C: any(c)\n"
                                    "out P: 0.1*A + 0.2*B\nout Q: 0.3*C\n")  # in float32, 0.1 + 0.2 is not 0.3
    data = [{"id": "tie", "values": {"p": 1, "q": 1}, "label": "P"}, {"id": "q", "values": {"p": 0, "q": 1}, "label": "P"},
            {"id": "none", "values": {"p": 0, "q": 0}, "label": None}, {"id": "n", "values": {"p": 1, "q": 0}, "label": None}]
    (tmp_path / "d.json").write_text(json.dumps(data))
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    out = [json.loads(r) for r in open(tmp_path / "o.jsonl")]
    assert [r["class"] for r in out] == [None, "Q", None, "P"] and out[0]["scores"] == {"P": 0.3, "Q": 0.3}
    assert out[2]["scores"] == {"P": 0.0, "Q": 0.0}
    assert "labelled 4 right 1 no answer 1 wrong ['q', 'n']" in capsys.readouterr().out
    (tmp_path / "r.dsl").write_text("h1 a: any(p)\nh1 c: any(q)\nh2 A: any(a)\nh2 C: any(c)\nout X: -1*A\nout Y: 1*C\n")
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    out = [json.loads(r) for r in open(tmp_path / "o.jsonl")]
    assert out[3]["class"] == "Y" and out[3]["scores"] == {"X": -1.0, "Y": 0.0} and out[3]["fired"] == ["A"]  # Y's unit does not fire


def test_rules_run_steps_counts_numbers_and_many_units(tmp_path, capsys):
    import itertools
    from kestrom import rules
    # parity of the 1s along the steps: the state is whether the unit odd fired at the step before
    (tmp_path / "r.dsl").write_text("in was: prev odd\nh1 a: all(bit, !was)\nh1 b: all(!bit, was)\nh2 odd: any(a, b)\nh2 even: all(!a, !b)\n"
                                    "out odd: 1*odd\nout even: 1*even\n")
    data = [{"id": "".join(map(str, bits)), "steps": [{"values": {"bit": v}} for v in bits], "label": "odd" if sum(bits) % 2 else "even"}
            for n in range(1, 7) for bits in itertools.product((0, 1), repeat=n)]
    (tmp_path / "d.json").write_text(json.dumps(data))
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    assert "labelled 126 right 126 no answer 0 wrong []" in capsys.readouterr().out
    out = {r["id"]: r for r in map(json.loads, open(tmp_path / "o.jsonl"))}
    assert out["1101"]["steps"] == ["odd", "even", "even", "odd"] and out["1101"]["fired"] == ["odd"]
    (tmp_path / "r.dsl").write_text('in n: count "階"\nin f: number "(\\\\d+)階"\nh1 two: 1*n >= 2\nh1 high: 1*f >= 5\n'
                                    "h2 T: any(two)\nh2 H: all(high, !two)\nout many: 1*T\nout high: 1*H\n", encoding="utf-8")
    data = [{"id": 0, "text": "3階と5階"}, {"id": 1, "text": "12階の温度"}, {"id": 2, "text": "4階"}, {"id": 3, "text": "屋上"}]
    assert rules.inputs_of({"inputs": ["f"], "declared": {"f": ("number", "(\\d+)?階")}}, {"text": "階"}, None, {}, 0) == [0.0]  # a group that matched nothing
    (tmp_path / "d.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    assert [json.loads(r)["class"] for r in open(tmp_path / "o.jsonl")] == ["many", "high", None, None]
    wide = "".join(f"h1 a{i}: 1*x >= {i + 1}\nh2 b{i}: all(a{i}{', !a' + str(i + 1) if i < 39 else ''})\nout c{i}: 1*b{i}\n" for i in range(40))
    (tmp_path / "r.dsl").write_text(wide)  # 40 units a layer: which of 40 ranges x is in
    (tmp_path / "d.json").write_text(json.dumps([{"id": v, "values": {"x": v}} for v in (0, 1, 33, 40, 99)]))
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    assert [json.loads(r)["class"] for r in open(tmp_path / "o.jsonl")] == [None, "c0", "c32", "c39", "c39"]
    with pytest.raises(ValueError):
        rules.compile_rules(wide)  # with a model, 32 units a layer
    for bad in ['number "\\\\d+"', "prev"]:  # no group; no unit
        with pytest.raises(ValueError):
            rules.declare(bad)
    (tmp_path / "r.dsl").write_text("in was: prev nowhere\nh1 a: any(was)\nh2 A: any(a)\nout x: A\n")
    with pytest.raises(SystemExit):
        rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    with pytest.raises(SystemExit):
        rules.model_rules("in was: prev A\nh1 a: any(was)\nh2 A: any(a)\nout x: A\n")


def test_rules_run_registers(tmp_path, capsys):
    import math
    from kestrom import rules
    gcd = ("h1 more: 1*a - 1*b >= 1\nh1 less: 1*b - 1*a >= 1\nh2 A: any(more)\nh2 B: any(less)\nh2 same: all(!more, !less)\n"
           "set a: a - b if A\nset b: b - a if B\nset n: n + 1 if A\nset n: n + 1 if B\nout done: 1*same\nout going: 1*A + 1*B\n")
    (tmp_path / "r.dsl").write_text(gcd)  # Euclid by subtraction; n counts the subtractions
    pairs = [(a, b) for a in range(1, 13) for b in range(1, 13)]
    (tmp_path / "d.json").write_text(json.dumps([{"id": i, "values": {"a": a, "b": b}, "steps": 12, "label": "done"} for i, (a, b) in enumerate(pairs)]))
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    assert "labelled 144 right 144 no answer 0 wrong []" in capsys.readouterr().out
    out = [json.loads(r) for r in open(tmp_path / "o.jsonl")]
    assert all(r["registers"]["a"] == r["registers"]["b"] == math.gcd(a, b) for r, (a, b) in zip(out, pairs))
    r = out[pairs.index((12, 9))]  # 12,9 -> 3,9 -> 3,6 -> 3,3
    assert r["registers"] == {"a": 3.0, "b": 3.0, "n": 3.0} and r["steps"][:4] == ["going", "going", "going", "done"]
    rules.build(str(tmp_path / "r.dsl"), str(tmp_path / "n.npz"))
    n = np.load(tmp_path / "n.npz")
    assert n["set_register"].tolist() == ["a", "b", "n", "n"] and n["set_unit"].tolist() == [0, 1, 0, 1]
    assert n["set_expression"].tolist() == ["a - b", "b - a", "n + 1", "n + 1"]
    for bad in ["h1 x: any(v)\nh2 X: any(x)\nset r: zz if X\nout c: X\n", 'in r: match "a"\nh1 x: any(r)\nh2 X: any(x)\nset r: 1 if X\nout c: X\n']:
        (tmp_path / "r.dsl").write_text(bad)  # zz is no input or register; r is an in line
        with pytest.raises(SystemExit):
            rules.build(str(tmp_path / "r.dsl"), str(tmp_path / "n.npz"))
    with pytest.raises(SystemExit):
        rules.model_rules(gcd)
    assert rules.value("(2 * v - 1.5) % 4 + r * r / 2 - 7 // 2", {"v": 5, "r": 3.0}) == 0.5 + 4.5 - 3
    for bad in ["set r: 3 4 if X", "set r: v.x if X", "set r: abs(v) if X", "set r: v ** 2 if X", "out c: X X", "out c: 2*3*X"]:
        with pytest.raises(ValueError):
            rules.compile_rules("h1 x: any(v)\nh2 X: any(x)\n" + bad + "\n")
    # Euclid by remainder, one head for the state and one for the size of a; a numeric output scored against targets
    (tmp_path / "r.dsl").write_text("h1 pos: 1*b >= 1\nh1 big: 1*a >= 10\nh2 go: any(pos)\nh2 stop: all(!pos)\nh2 B: any(big)\nh2 S: all(!big)\n"
                                    "set a: b if go\nset b: a % b if go\nset twice: 2 * a if stop\n"
                                    'out state/done: 1*stop\nout state/going: 1*go\nout size/big: 1*B\nout size/small: 1*S\n')
    (tmp_path / "d.json").write_text(json.dumps([{"id": i, "values": {"a": a, "b": b}, "steps": 8, "label": {"state": "done", "size": "big" if math.gcd(a, b) >= 10 else "small"},
                                                  "targets": {"a": math.gcd(a, b), "twice": 2 * math.gcd(a, b)}} for i, (a, b) in enumerate([(1071, 462), (30, 20), (7, 13)])]))
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    printed = capsys.readouterr().out
    assert "labelled 3 right 3 no answer 0 wrong []" in printed and "targets {'a': (3, 0.0), 'twice': (3, 0.0)}" in printed
    assert json.loads(open(tmp_path / "o.jsonl").readline())["class"] == {"state": "done", "size": "big"}
    (tmp_path / "d.json").write_text(json.dumps([{"id": i, "values": {"a": 30, "b": 20}, "steps": s, "label": lab} for i, (s, lab) in enumerate(
        [(8, {"size": "big"}), (8, {"state": "going", "size": "big"}), (8, None), (1, {"state": "going", "size": None})])]))
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    assert "labelled 4 right 1 no answer 0 wrong [1, 2, 3]" in capsys.readouterr().out  # a label of some heads; one head wrong; null


def test_rules_weigh_fits_the_out_lines(tmp_path, capsys):
    from kestrom import rules
    (tmp_path / "r.dsl").write_text('in a: match "a"  # kept\nin b: match "b"\nh1 x: any(a)\nh1 y: any(b)\nh2 A: any(x)\nh2 B: any(y)\n'
                                    'out "p:q": 1*B\nout r: 1*A\n')  # the out lines the wrong way round
    data = [{"id": i, "text": t, "label": c} for i, (t, c) in enumerate([("a", "p:q"), ("b", "r"), ("ab", "p:q"), ("", None)] * 3)]
    (tmp_path / "d.json").write_text(json.dumps(data))
    rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    assert "labelled 12 right 3 " in capsys.readouterr().out
    rules.weigh(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "w.dsl"))
    text = (tmp_path / "w.dsl").read_text()
    assert text.startswith('in a: match "a"  # kept\n') and text.count("out ") == 2 and 'out "p:q": ' in text
    rules.run(str(tmp_path / "w.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    assert "labelled 12 right 12 no answer 0 wrong []" in capsys.readouterr().out
    with pytest.raises(ValueError):
        rules.compile_rules("h1 x: any(v)\nh2 X: any(x)\nset r: 1\n")


def test_rules_run_example(tmp_path, capsys):
    from pathlib import Path

    from kestrom import rules
    e = Path(__file__).parent.parent / "examples" / "rules-run"
    rules.run(str(e / "router.dsl"), str(e / "router.json"), str(tmp_path / "r.jsonl"))
    rules.run(str(e / "gcd.dsl"), str(e / "gcd.json"), str(tmp_path / "g.jsonl"))
    printed = capsys.readouterr().out  # the outputs examples/rules-run/README.md shows
    assert "labelled 8 right 8 no answer 0 wrong []" in printed and "labelled 4 right 4 no answer 0 wrong []" in printed
    assert "targets {'a': (4, 0.0)}" in printed
    assert [json.loads(r)["registers"]["steps"] for r in open(tmp_path / "g.jsonl")] == [3, 2, 4, 28]


def test_rules_run_refuses(tmp_path):
    from kestrom import rules
    (tmp_path / "d.json").write_text('[{"id": 0, "text": "a", "values": {}}]')
    for bad in ["h1 s: any(t)\nh2 v: any(s)\nban x: v\n", 'in c: after "a"\nh1 s: any(c)\nh2 v: any(s)\nout A: v\n',
                "h1 s: any(t)\nh2 v: any(s)\nout A: v\n"]:  # a ban line; an in line of decoding; t has no value
        (tmp_path / "r.dsl").write_text(bad)
        with pytest.raises(SystemExit):
            rules.run(str(tmp_path / "r.dsl"), str(tmp_path / "d.json"), str(tmp_path / "o.jsonl"))
    with pytest.raises(ValueError):
        rules.compile_rules("h1 s: any(t)\nh2 v: any(s)\nout A: v\nout A: 2*v\n")  # one out line per token
    with pytest.raises(re.error):
        rules.declare('match "("')
    (tmp_path / "r.dsl").write_text('in c: match "a"\nh1 s: any(c)\nh2 v: any(s)\nout A: v\n')
    with pytest.raises(SystemExit):
        rules.load(str(tmp_path / "r.dsl"), None)  # generate does not read match lines


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
    assert rules.declare("position 0 2") == ("position", 0, 2) and rules.declare("position 5") == ("position", 5, 5)
    for bad in ("position", "position 2 1", "position -1", "position a", "position 1 2 3"):
        with pytest.raises(ValueError):
            rules.declare(bad)
    assert rules.lines('ban ":" x: b\nout "a:b": b') == [("ban", '":" x', "b"), ("out", '"a:b"', "b")]  # a : inside quotes
    with pytest.raises(ValueError):
        rules.lines('ban ":: b')


def test_rules_inputs_of_in_lines():
    pytest.importorskip("mlx")
    from kestrom import rules

    class Tok:
        def decode(self, t):
            return "".join(chr(97 + i) for i in t)
    rules.SIDE.clear()
    rules.SIDE.update(declared={"c": ("prompt", "f", 1), "e": ("after", "bc"), "t": ("thinking",), "p": ("position", 1, 2),
                               "p0": ("position", 0, 0)}, prompt_on={"c": True},
                      tok=Tok(), think=(20, 21))
    rules.restart(2)
    F = rules.flags([9, 20, 1, 2, 21, 3]).tolist()[0]  # the prompt is 9, 20 (<think>); the output 1 2 </think> 3 decodes "bc" at 2
    assert [r[0] for r in F] == [1] * 6
    assert [r[1] for r in F] == [0, 0, 0, 1, 1, 1]  # holds from the token that completes "bc"
    assert [r[2] for r in F] == [0, 1, 1, 1, 0, 0]
    assert [r[3] for r in F] == [0, 0, 1, 1, 0, 0]  # the logits at 2 and 3 choose output positions 1 and 2
    assert [r[4] for r in F] == [0, 1, 0, 0, 0, 0]  # position 0: chosen by the logits at the last prompt token
    R = rules.compile_rules("in p: position 1 2\nh1 a: all(p)\nh2 b: any(a)\nban x: b\n")
    assert rules.side(R["net"], np.array([[r[3]] for r in F], np.float32), rules.K, R["used"])[1].tolist() == [False, False, True, True, False, False]
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


def test_open_line_spells_the_class_the_rules_answer(tmp_path):
    from kestrom import generate, rules

    class Tok:  # token i is PIECES[i]; 5 is half a character
        PIECES = ["a", "b", "ab", "\n", "b\nx", "\ufffd", "c", "", "b\n\ufffd", "a\ufffd", "<think>", "</think>", "\n\n", "ab\nz", " ab"]

        def decode(self, t):
            return "".join(self.PIECES[i] for i in t)
    order = lambda: [6, 5, 0, 2, 1, 4, 3]  # noqa: E731
    assert generate.spell(Tok(), [], "ab\n", 2, order) == (2, False)  # the top token fits: kept
    assert generate.spell(Tok(), [0], "ab\n", 4, order) == (4, False)  # it may run past the end
    assert generate.spell(Tok(), [], "ab\n", 6, order) == (0, True)  # c does not fit, nor half a character: the first that does
    assert generate.spell(Tok(), [0], "ab\n", 0, order) == (1, True)
    assert generate.spell(Tok(), [0], "ab\n", 8, order) == (8, False)  # half a character past the end fits
    assert generate.spell(Tok(), [], "ab\n", 9, order) == (0, True)  # half a character inside it does not
    assert generate.spell(Tok(), [], "ab\n", 7, order) == (0, True)  # nor a token of no text
    with pytest.raises(SystemExit):
        generate.spell(Tok(), [], "z\n", 6, order)
    assert generate.answer(Tok(), [12, 0, 12, 11, 12, 3, 2, 12], (10, 11)) == (False, [2, 12])  # after </think>, past the white space
    assert generate.answer(Tok(), [0, 10, 1], (10, 11)) == (True, [1]) and generate.answer(Tok(), [12, 1], (10, 11), True) == (True, [1])
    assert generate.answer(Tok(), [11, 1], (10, 11), True) == (False, [1]) and generate.answer(Tok(), [0], (None, None), False, [1]) == (False, [1, 0])
    mx = pytest.importorskip("mlx.core")
    row = lambda *top: mx.array([[float(len(top) - top.index(i)) if i in top else 0.0 for i in range(15)]])  # noqa: E731
    P, AB = [9, 9], ("ab\n", ["ab\n", "c\n"], None)

    def steps(toks, tops, ban=(), think=(None, None), thinking=False, note=None, loop=False):  # the tokens taken at the steps after toks, and the events
        procs, out, spelled = generate.processors(list(toks), list(ban), loop, think[1], Tok(), (*AB[:2], note), spelled := [], think, thinking), [], spelled
        for top in tops:
            z = row(*top)
            for f in procs:
                z = f(mx.array(P + list(toks) + out), z)
            out.append(int(mx.argmax(z[0])))
        return out, spelled
    assert steps([], [(6, 0, 2), (3, 1), (6,)], ban=[0]) == ([2, 3, 6], [0])  # c on top, a banned: ab, an event at 0; the line feed kept; then as it was
    assert steps([], [(0,), (6, 1), (6, 3), (6,)]) == ([0, 1, 3, 6], [1, 2])  # a kept, then c replaced twice: no standing down inside the opening
    assert steps([], [(13,), (6,)]) == ([13, 6], [])  # the whole opening and more in one token: kept
    assert steps([], [(7,), (12,), (6, 2), (6, 3)]) == ([7, 12, 2, 3], [2, 3])  # a token of no text, white space: passed; then the answer begins
    assert steps([], [(1,), (0,), (6,)]) == ([1, 0, 6], [])  # b starts no class: left alone, also when a comes next
    assert steps([], [(5,), (2,)]) == ([5, 2], []) and steps([1], [(0,), (6,)]) == ([0, 6], [])  # half a character; a restart after b
    assert steps([2, 3], [(6,), (0,)]) == ([6, 0], [])  # a restart past the opening
    with pytest.raises(SystemExit):
        steps([0], [(6,)], ban=[1, 2, 4, 8, 13])  # a restart inside it with every way on banned
    T = (10, 11)
    assert steps([], [(0,), (6,), (11,), (12,), (6, 2), (3,), (6,)], think=T, thinking=True) == ([0, 6, 11, 12, 2, 3, 6], [4])  # nothing while thinking
    assert steps([], [(10,), (2,), (11,), (1,), (0,)], think=T) == ([10, 2, 11, 1, 0], [])  # the model opens thinking; b after it: left alone
    assert steps([0, 6, 11], [(6, 0), (6, 1)], think=T, thinking=True) == ([0, 1], [3, 4])  # a restart at the answer's first token
    N = dict(think=T, note="b\n")  # the thinking opens with b and a line feed, whatever the model was about to think
    assert steps([], [(5,), (6,), (0,), (11,), (6, 2), (3,)], thinking=True, **N) == ([1, 3, 0, 11, 2, 3], [0, 1, 4])  # a top that starts no class too
    assert steps([], [(14,), (6,)]) == ([0, 1], [0, 1])  # a class after a space: the model is writing a class, and it is spelled
    assert steps([], [(10,), (12,), (4,), (11,), (1,)], **N) == ([10, 12, 4, 11, 1], [])  # the model opens thinking and writes it itself
    assert steps([], [(11,), (6, 0), (6, 1), (3,)], thinking=True, **N) == ([11, 0, 1, 3], [1, 2])  # thinking closed at once: not reopened
    assert steps([], [(2,), (3,)], **N) == ([2, 3], [])  # no thinking: the note is not written
    assert steps([], [(6,), (6,), (11,), (6, 2), (3,)], thinking=True, loop=True, ban=[9], **N) == ([1, 3, 11, 2, 3], [0, 1, 3])  # with the other two processors
    (tmp_path / "r.dsl").write_text('in hot: match "hot"\nh1 a: all(hot)\nh2 u: all(a)\nout k/yes: u\nout k/no: -1*u\n')
    assert rules.compile_rules(f"open k: {tmp_path / 'r.dsl'} ask")["open"] == ("k", str(tmp_path / "r.dsl"), "ask", None)
    assert rules.compile_rules(f'open: {tmp_path / "r.dsl"} "The rules say {{}}."')["open"] == ("", str(tmp_path / "r.dsl"), None, "The rules say {}.")
    assert rules.compile_rules('open: f "T{}"')["open"] == ("", "f", None, "T{}")
    for bad in ("open: a b c", "open a b: f", "open: f\nopen: g", 'open: "T {}"', 'open: f ask"T"', 'open: f ask ""', 'open: f ask " {}"', 'open: f ask "T {}'):
        with pytest.raises(ValueError):
            rules.compile_rules(bad)
    (tmp_path / "o.dsl").write_text(f"open k: {tmp_path / 'r.dsl'}\n")
    with pytest.raises(SystemExit):  # an open line is for a model
        rules.standalone(tmp_path / "o.dsl")
    R = rules.standalone(tmp_path / "r.dsl")
    for line in ("open: ", "open z: "):  # the rules have the head k
        (tmp_path / "o.dsl").write_text(line + str(tmp_path / "r.dsl") + "\n")
        with pytest.raises(SystemExit):
            rules.load(str(tmp_path / "o.dsl"), None)
    try:
        rules.SIDE.clear()
        assert rules.opening({"data": {"text": "hot"}}) is None  # no open line
        rules.SIDE["open"] = (R, "k", None, None)
        with pytest.raises(SystemExit):
            rules.opening({"id": 1})  # no data
        assert rules.opening({"data": {"text": "hot"}}) == ("yes\n", ["yes\n", "no\n"], None) and rules.opening({"data": {"text": "cold"}}) is None
        rules.SIDE["open"] = (R, "k", "ask", "Rules: {}.")
        assert rules.opening({"data": {"text": "cold"}}) == ("ask\n", ["yes\n", "no\n", "ask\n"], "Rules: ask.\n")
    finally:
        rules.SIDE.clear()
