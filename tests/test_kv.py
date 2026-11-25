"""Tests for kestrom.kv (compiled fact-KV routing): fact parsing, template boundary check, incremental compilation, and G2 events."""
import json
import sys
import types
from pathlib import Path

import pytest

from kestrom import generate, kv, rules
from kestrom_verify import report


ROUTER_DSL = (
    'in ac_w: match "空調|エアコン"\n'
    'in l_w: match "照明|電気"\n'
    'h1 a: all(ac_w, !l_w)\n'
    'h1 b: all(l_w, !ac_w)\n'
    'h2 A: any(a)\n'
    'h2 B: any(b)\n'
    'out meter/ac: 1*A\n'
    'out meter/light: 1*B\n'
)


SINGLE_DSL = (
    'in ac_w: match "空調|エアコン"\n'
    'in l_w: match "照明|電気"\n'
    'h1 a: all(ac_w, !l_w)\n'
    'h1 b: all(l_w, !ac_w)\n'
    'h2 A: any(a)\n'
    'h2 B: any(b)\n'
    'out ac: 1*A\n'
    'out light: 1*B\n'
)


class MockTok:
    """Minimal chat tokenizer with controllable BOS and template layout."""

    def __init__(self, bos=2, bad_mode=None):
        self.bos_token_id = bos
        self.eos_token_ids = [1]
        self.bad_mode = bad_mode

    def _enc(self, s: str):
        return [100 + ord(c) for c in s]

    def get_vocab(self):
        return {}

    def decode(self, toks):
        return "".join(str(t) for t in toks)

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=False):
        sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
        usr_msgs = [m["content"] for m in messages if m["role"] == "user"]
        prefix = [self.bos_token_id] if self.bos_token_id is not None else []
        if self.bad_mode == "merged" and sys_msgs:
            # System content injected inside the user turn (breaks clean prefix property).
            return prefix + [10] + self._enc(usr_msgs[0] + " " + sys_msgs[0]) + [11]
        if self.bad_mode == "varies" and sys_msgs:
            # Prefix length varies depending on user message (cannot precompile per fact).
            extra = [99] if usr_msgs[0].startswith("1") else []
            return prefix + [10] + extra + self._enc(sys_msgs[0]) + [11] + self._enc(usr_msgs[0]) + [12]
        sys_part = ([10] + self._enc(sys_msgs[0]) + [11]) if sys_msgs else []
        usr_part = [20] + self._enc(usr_msgs[0]) + [21]
        return prefix + sys_part + usr_part


def test_parse_facts_single_and_multi_head(tmp_path):
    p_single = tmp_path / "single.dsl"
    p_single.write_text(SINGLE_DSL, encoding="utf-8")
    R1 = rules.standalone(str(p_single))
    parsed = kv.parse_facts({"light": "L fact", "ac": "AC fact"}, R1)
    assert parsed == [
        {"key": "ac", "when": "ac", "text": "AC fact"},
        {"key": "light", "when": "light", "text": "L fact"},
    ]

    p_multi = tmp_path / "multi.dsl"
    p_multi.write_text(ROUTER_DSL, encoding="utf-8")
    R2 = rules.standalone(str(p_multi))
    multi = kv.parse_facts([
        {"key": "m/ac", "when": {"meter": "ac"}, "text": "AC is 30kW."},
        {"key": "m/light", "when": {"meter": "light"}, "text": "LED lighting."},
    ], R2)
    assert len(multi) == 2 and multi[0]["when"] == {"meter": "ac"}

    bad_cases = [
        ([], R1),
        ({}, R1),
        ("bad", R1),
        ([{"key": "a", "when": "nope", "text": "x"}], R1),
        ([{"key": "a", "when": "ac", "text": ""}], R1),
        ([{"key": "a", "when": "ac", "text": "x"}, {"key": "a", "when": "light", "text": "y"}], R1),
        ([{"key": "a", "when": "ac", "text": "x"}, {"key": "b", "when": "ac", "text": "y"}], R1),
        ([{"key": "a", "when": "ac", "text": "x"}], R2),
        ([{"key": "a", "when": {"floor": "1"}, "text": "x"}], R2),
        ([{"key": "a", "when": {"meter": "other"}, "text": "x"}], R2),
    ]
    for bad, R in bad_cases:
        with pytest.raises(SystemExit):
            kv.parse_facts(bad, R)


def test_prefix_boundary_bos_no_bos_and_bad_templates():
    sp_bos, pfx_bos = kv.prefix_boundary(MockTok(bos=2), "fact")
    assert sp_bos == [2]
    assert pfx_bos[0] == 2 and pfx_bos[-1] == 11

    sp_nobos, pfx_nobos = kv.prefix_boundary(MockTok(bos=None), "fact")
    assert sp_nobos == []
    assert pfx_nobos[0] == 10 and pfx_nobos[-1] == 11

    with pytest.raises(SystemExit):
        kv.prefix_boundary(MockTok(bad_mode="merged"), "fact")
    with pytest.raises(SystemExit):
        kv.prefix_boundary(MockTok(bad_mode="varies"), "fact")


def test_match_entry_single_multi_and_tie(tmp_path):
    p_dsl = tmp_path / "r.dsl"
    p_dsl.write_text(
        ROUTER_DSL + 'in f1: match "1階"\nh1 p1: all(f1)\nh2 P1: any(p1)\nout floor/f1: 1*P1\n',
        encoding="utf-8",
    )
    R = rules.standalone(str(p_dsl))
    bank = {
        "rules": R,
        "entries": [
            {"key": "m/ac", "when": {"meter": "ac"}, "file": "a.safetensors"},
            {"key": "f/1", "when": {"floor": "f1"}, "file": "b.safetensors"},
        ],
    }
    assert kv.match_entry(bank, {"text": "空調の様子"})["key"] == "m/ac"
    assert kv.match_entry(bank, {"text": "1階の様子"})["key"] == "f/1"
    # Ambiguous match across two entries (both meter/ac and floor/f1 fire) -> returns None cleanly.
    assert kv.match_entry(bank, {"text": "1階の空調"}) is None
    assert kv.match_entry(bank, {"text": "天気"}) is None


def test_kv_line_parsing_and_kv_for(tmp_path):
    assert rules.compile_rules("kv: my_bank\n")["kv"] == "my_bank"
    for bad in ("kv:\n", "kv: a b\n", "kv x: bank\n", "kv: a\nkv: b\n"):
        with pytest.raises(ValueError):
            rules.compile_rules(bad)

    (tmp_path / "use.dsl").write_text("kv: my_bank\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        rules.standalone(str(tmp_path / "use.dsl"))

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "rules.dsl").write_text(SINGLE_DSL, encoding="utf-8")
    (bank_dir / "a.safetensors").write_bytes(b"cache-a")
    (bank_dir / "manifest.json").write_text(json.dumps({
        "format": kv.FORMAT,
        "fingerprint": {},
        "rules_sha256": "x",
        "strip_prefix": [2],
        "entries": [{"key": "ac", "when": "ac", "text": "fact", "file": "a.safetensors"}],
    }), encoding="utf-8")
    (tmp_path / "use_real.dsl").write_text(f"kv: {bank_dir}\n", encoding="utf-8")

    try:
        rules.SIDE.clear()
        rules.SIDE.update(rules.load(str(tmp_path / "use_real.dsl"), MockTok(bos=2)))
        assert rules.kv_for({"id": 1, "data": {"text": "空調について"}}, [2, 20, 30]) == ("ac", str(bank_dir / "a.safetensors"), [2])
        assert rules.kv_for({"id": 2, "data": {"text": "天気"}}, [2, 20, 30]) is None
        with pytest.raises(SystemExit):
            rules.kv_for({"id": 3}, [2, 20, 30])
        with pytest.raises(SystemExit):
            rules.kv_for({"id": 4, "data": {"text": "空調について"}}, [9, 20, 30])
        with pytest.raises(SystemExit):
            kv.load_bank(str(bank_dir), MockTok(bos=99))
    finally:
        rules.SIDE.clear()


def test_compile_deterministic_and_incremental_reuse(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    rules_file = tmp_path / "rules.dsl"
    rules_file.write_text(SINGLE_DSL, encoding="utf-8")
    facts1 = tmp_path / "facts1.json"
    facts1.write_text(json.dumps({"ac": "AC old", "light": "Light old"}), encoding="utf-8")
    facts2 = tmp_path / "facts2.json"
    facts2.write_text(json.dumps({"ac": "AC new", "light": "Light old"}), encoding="utf-8")

    calls = []

    def fake_compile_prefix(model, prefix_ids, out_file, key):
        calls.append((key, list(prefix_ids)))
        out_file.write_bytes(json.dumps({"key": key, "ids": prefix_ids}).encode("utf-8"))

    monkeypatch.setattr(kv, "compile_prefix", fake_compile_prefix)
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda path: (object(), MockTok(bos=2))
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    bank1, bank1b = tmp_path / "bank1", tmp_path / "bank1b"
    kv.compile(str(model_dir), str(rules_file), str(facts1), str(bank1))
    kv.compile(str(model_dir), str(rules_file), str(facts1), str(bank1b))
    assert (bank1 / "manifest.json").read_bytes() == (bank1b / "manifest.json").read_bytes()
    assert len(calls) == 4

    calls.clear()
    bank2 = tmp_path / "bank2"
    kv.compile(str(model_dir), str(rules_file), str(facts2), str(bank2), prev_dir=str(bank1))
    assert [k for k, _ in calls] == ["ac"]  # only edited entry compiled; 'light' copied unchanged
    assert (bank2 / kv.entry_filename("light")).read_bytes() == (bank1 / kv.entry_filename("light")).read_bytes()
    with pytest.raises(SystemExit):
        kv.compile(str(model_dir), str(rules_file), str(facts2), str(bank2))  # non-empty out_dir refused


def test_decode_with_kv_passes_kestrom_verify_compare_events(tmp_path, monkeypatch):
    pytest.importorskip("mlx")
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "rules.dsl").write_text(SINGLE_DSL, encoding="utf-8")
    (bank_dir / "ac.safetensors").write_bytes(b"fake-cache")
    (bank_dir / "manifest.json").write_text(json.dumps({
        "format": kv.FORMAT,
        "fingerprint": {},
        "rules_sha256": "x",
        "strip_prefix": [2],
        "entries": [{"key": "ac", "when": "ac", "text": "30kW", "file": "ac.safetensors"}],
    }), encoding="utf-8")
    use_dsl = tmp_path / "use.dsl"
    use_dsl.write_text(f"kv: {bank_dir}\n", encoding="utf-8")

    # Fake tokenizer and generate_step: without cache outputs [10, 11, 1]; with cache outputs [12, 13, 1].
    class FakeTok:
        bos_token_id = 2
        eos_token_ids = [1]

        def get_vocab(self):
            return {}

        def convert_tokens_to_ids(self, s):
            return None

        def decode(self, xs):
            return "".join(chr(97 + x) for x in xs)

    recorded = []

    def fake_generate_step(prompt, model, max_tokens=16, logits_processors=None, prompt_cache=None):
        recorded.append((prompt.tolist(), prompt_cache is not None))
        seq = [12, 13, 1] if prompt_cache is not None else [10, 11, 1]
        for t in seq[:max_tokens]:
            yield t, None

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeTrunk:
        pass

    class FakeModel:
        layers = []

        def __init__(self):
            self.model = FakeTrunk()

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda path: (FakeModel(), FakeTok())
    fake_gen = types.ModuleType("mlx_lm.generate")
    fake_gen.generate_step = fake_generate_step
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_gen)
    monkeypatch.setattr(kv, "load_cache_file", lambda path: [("cached", Path(path).name)])

    prompts_file = tmp_path / "prompts.json"
    prompts = [
        {"id": "p_match", "ids": [2, 20, 21, 22], "data": {"text": "空調は?"}},
        {"id": "p_plain", "ids": [2, 30, 31, 32], "data": {"text": "天気は?"}},
    ]
    prompts_file.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")

    out_kv = tmp_path / "kv.jsonl"
    try:
        generate.run(str(model_dir), str(prompts_file), str(out_kv), max_tokens=16, rules_file=str(use_dsl))
    finally:
        rules.SIDE.clear()

    rows = [json.loads(line) for line in out_kv.read_text(encoding="utf-8").splitlines()]
    assert recorded == [([20, 21, 22], True), ([2, 30, 31, 32], False)]
    assert rows[0]["kv"] == "ac" and rows[0]["events"] == [0] and rows[0]["tokens"] == [12, 13]
    assert rows[1]["kv"] is None and rows[1]["events"] == [] and rows[1]["tokens"] == [10, 11]

    base = [
        {"id": "p_match", "finish": "stop", "tokens": [10, 11], "text": "kl"},
        {"id": "p_plain", "finish": "stop", "tokens": [10, 11], "text": "kl"},
    ]
    b1, b2 = tmp_path / "b1.jsonl", tmp_path / "b2.jsonl"
    report.write_lines(b1, base)
    report.write_lines(b2, base)
    rep1, rep2 = tmp_path / "r1.json", tmp_path / "r2.json"
    report.greedy("events", (b1, b2, out_kv), FakeTok().decode, rep1)
    report.greedy("events", (b1, b2, out_kv), FakeTok().decode, rep2)
    assert rep1.read_bytes() == rep2.read_bytes()
    assert all(ok for *_, ok in report.verdict(report.read(rep1)))


def test_compact_cache_roundtrip_with_mlx():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache, load_prompt_cache, save_prompt_cache
    import tempfile

    c = KVCache()
    c.update_and_fetch(mx.ones((1, 2, 10, 4), mx.bfloat16), mx.ones((1, 2, 10, 4), mx.bfloat16))
    raw_size_hint = c.keys.shape[2]
    assert raw_size_hint >= 10
    kv._compact_cache([c])
    assert c.keys.shape[2] == 10 and c.values.shape[2] == 10 and c.offset == 10
    with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
        save_prompt_cache(f.name, [c], {"kestrom_kv_key": "k"})
        loaded = load_prompt_cache(f.name)
        assert loaded[0].offset == 10 and mx.array_equal(loaded[0].keys, c.keys)
