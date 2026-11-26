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
    p_mixed = tmp_path / "mixed.dsl"
    p_mixed.write_text(ROUTER_DSL.replace("out meter/light:", "out light:"), encoding="utf-8")
    mixed = kv.parse_facts([{"key": "l", "when": {"": "light"}, "text": "LED lighting."}], rules.standalone(str(p_mixed)))
    assert mixed[0]["when"] == {"": "light"}  # a class with no head among headed ones

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
    assert [e["key"] for e in kv.match_entries(bank, {"text": "空調の様子"}, 1)] == ["m/ac"]
    assert [e["key"] for e in kv.match_entries(bank, {"text": "1階の様子"}, 1)] == ["f/1"]
    # Ambiguous match across two entries (both meter/ac and floor/f1 fire) -> no entry with MAX_MATCH 1.
    assert kv.match_entries(bank, {"text": "1階の空調"}, 1) == []
    assert kv.match_entries(bank, {"text": "天気"}, 1) == []


def test_kv_line_parsing_and_kv_for(tmp_path):
    assert rules.compile_rules("kv: my_bank\n")["kv"] == ("my_bank", 1)
    assert rules.compile_rules("kv: my_bank 3\n")["kv"] == ("my_bank", 3)
    for bad in ("kv:\n", "kv: a b\n", "kv: a 0\n", "kv: a -1\n", "kv: a 1 2\n", "kv x: bank\n", "kv: a\nkv: b\n"):
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
        "rules_sha256": kv.sha256_file(bank_dir / "rules.dsl"),
        "strip_prefix": [2],
        "entries": [{"key": "ac", "when": "ac", "text": "fact", "file": "a.safetensors", "cache_sha256": kv.sha256_bytes(b"cache-a")}],
    }), encoding="utf-8")
    (tmp_path / "use_real.dsl").write_text(f"kv: {bank_dir}\n", encoding="utf-8")
    assert kv.load_bank(str(bank_dir), MockTok(bos=2))["entries"][0]["key"] == "ac"
    with pytest.raises(SystemExit):
        kv.load_bank(str(bank_dir), MockTok(bos=99))
    for name, changed in (("a.safetensors", b"cache-b"), ("rules.dsl", SINGLE_DSL.replace("空調", "冷房").encode("utf-8"))):
        before = (bank_dir / name).read_bytes()
        (bank_dir / name).write_bytes(changed)
        with pytest.raises(SystemExit):
            kv.load_bank(str(bank_dir), MockTok(bos=2))  # a bank whose files are not the ones manifest.json hashed is refused
        (bank_dir / name).write_bytes(before)

    pytest.importorskip("mlx")
    try:
        rules.SIDE.clear()
        rules.SIDE.update(rules.load(str(tmp_path / "use_real.dsl"), MockTok(bos=2)))
        assert rules.kv_for({"id": 1, "data": {"text": "空調について"}}, [2, 20, 30]) == (
            "ac",
            [{"key": "ac", "when": "ac", "text": "fact", "file": "a.safetensors", "cache_sha256": kv.sha256_bytes(b"cache-a")}],
            [2],
        )
        assert rules.kv_for({"id": 2, "data": {"text": "天気"}}, [2, 20, 30]) is None
        with pytest.raises(SystemExit):
            rules.kv_for({"id": 3}, [2, 20, 30])
        with pytest.raises(SystemExit):
            rules.kv_for({"id": 4, "data": {"text": "空調について"}}, [9, 20, 30])
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

    calls.clear()
    facts3 = tmp_path / "facts3.json"
    facts3.write_text(json.dumps([{"key": "ac2", "when": "ac", "text": "AC old"}, {"key": "light", "when": "light", "text": "Light old"}]), encoding="utf-8")
    kv.compile(str(model_dir), str(rules_file), str(facts3), str(tmp_path / "bank3"), prev_dir=str(bank1))
    assert [k for k, _ in calls] == ["ac2"]  # a renamed key is compiled afresh, so the file holds its own key


def test_compile_midway_failure_leaves_no_partial_out_dir(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    rules_file = tmp_path / "rules.dsl"
    rules_file.write_text(SINGLE_DSL, encoding="utf-8")
    facts_file = tmp_path / "facts.json"
    facts_file.write_text(json.dumps({"ac": "AC", "light": "Light"}), encoding="utf-8")

    state = {"n": 0}

    def flaky_compile_prefix(model, prefix_ids, out_file, key):
        state["n"] += 1
        out_file.write_bytes(b"partial")
        if state["n"] == 2:
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(kv, "compile_prefix", flaky_compile_prefix)
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda path: (object(), MockTok(bos=2))
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    out_dir = tmp_path / "bank_atomic"
    with pytest.raises(RuntimeError):
        kv.compile(str(model_dir), str(rules_file), str(facts_file), str(out_dir))
    assert not out_dir.exists()
    assert not list(tmp_path.glob(".bank_atomic.tmp.*"))

    # Retry succeeds immediately into the same OUT_DIR (also when an empty OUT_DIR already existed).
    out_dir.mkdir()
    kv.compile(str(model_dir), str(rules_file), str(facts_file), str(out_dir))
    assert (out_dir / "manifest.json").is_file()


def test_decode_with_kv_passes_kestrom_verify_compare_events(tmp_path, monkeypatch):
    pytest.importorskip("mlx")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "rules.dsl").write_text(SINGLE_DSL, encoding="utf-8")
    (bank_dir / "ac.safetensors").write_bytes(b"fake-cache")
    (bank_dir / "manifest.json").write_text(json.dumps({
        "format": kv.FORMAT,
        "fingerprint": kv.model_fingerprint(model_dir),
        "rules_sha256": kv.sha256_file(bank_dir / "rules.dsl"),
        "strip_prefix": [2],
        "entries": [{"key": "ac", "when": "ac", "text": "30kW", "file": "ac.safetensors", "cache_sha256": kv.sha256_bytes(b"fake-cache")}],
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
    other_model = tmp_path / "other_model"
    other_model.mkdir()
    (other_model / "tokenizer.json").write_text('{"other": 1}', encoding="utf-8")
    try:
        with pytest.raises(SystemExit):  # a bank compiled for another model is refused
            generate.run(str(other_model), str(prompts_file), str(out_kv), max_tokens=16, rules_file=str(use_dsl))
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
    pytest.importorskip("mlx_lm")
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


STATEFUL_DSL = (
    'in b_w: match "billing"\n'
    'in a_w: match "auth"\n'
    'in rst: match "天気|首都"\n'
    'in prev_b: prev B\n'
    'in prev_a: prev A\n'
    'h1 b_on: any(b_w)\n'
    'h1 a_on: any(a_w)\n'
    'h1 r_on: any(rst)\n'
    'h1 b_keep: all(prev_b, !a_w, !rst)\n'
    'h1 a_keep: all(prev_a, !b_w, !rst)\n'
    'h2 B: any(b_on, b_keep)\n'
    'h2 A: any(a_on, a_keep)\n'
    'out billing/on: 1*B\n'
    'out auth/on: 1*A\n'
)


def test_match_entries_multi_and_steps_prev(tmp_path):
    dsl = tmp_path / "stateful.dsl"
    dsl.write_text(STATEFUL_DSL, encoding="utf-8")
    R = rules.standalone(str(dsl))
    bank = {
        "dir": str(tmp_path),
        "rules": R,
        "strip_prefix": [2],
        "entries": [
            {"key": "billing", "when": {"billing": "on"}, "text": "Retry 3 times.", "file": "b.safetensors"},
            {"key": "auth", "when": {"auth": "on"}, "text": "Ed25519 token.", "file": "a.safetensors"},
        ],
    }
    # Max_match=1 rejects ambiguous 2-domain question; max_match=2 returns both in bank order.
    assert kv.match_entries(bank, {"text": "billing と auth を比較"}, max_match=1) == []
    both = kv.match_entries(bank, {"text": "billing と auth を比較"}, max_match=2)
    assert [e["key"] for e in both] == ["billing", "auth"]
    assert kv.format_kv_key(both) == ["billing", "auth"]

    # Multi-step history with prev: follow-up omitting domain keeps billing; unrelated reset word clears it.
    steps = [{"text": "billing のタイムアウトは?"}]
    assert [e["key"] for e in kv.match_entries(bank, {"steps": steps}, max_match=3)] == ["billing"]
    steps.append({"text": "それのリトライ上限は?"})
    assert [e["key"] for e in kv.match_entries(bank, {"steps": steps}, max_match=3)] == ["billing"]
    steps.append({"text": "日本の首都は?"})
    assert kv.match_entries(bank, {"steps": steps}, max_match=3) == []


def test_compose_cache_lru_and_clone(tmp_path, monkeypatch):
    monkeypatch.setattr(kv, "load_cache_file", lambda path: [("single", Path(path).name)])
    prefill_calls = []

    class FakeLayer:
        meta_state = ""

        def __init__(self, data):
            self.data = data

        @property
        def state(self):
            return list(self.data)

        @classmethod
        def from_state(cls, state, meta_state):
            return cls(state)

    def fake_prefill(model, prefix_ids):
        prefill_calls.append(list(prefix_ids))
        return [FakeLayer(["k", list(prefix_ids)])]

    monkeypatch.setattr(kv, "prefill_prefix", fake_prefill)
    bank = {
        "dir": str(tmp_path),
        "strip_prefix": [2],
        "entries": [
            {"key": "k1", "text": "F1", "file": "f1.safetensors"},
            {"key": "k2", "text": "F2", "file": "f2.safetensors"},
            {"key": "k3", "text": "F3", "file": "f3.safetensors"},
        ],
    }
    tok = MockTok(bos=2)
    lru = kv.ComposeCache(maxsize=2)
    assert lru.get_or_build(None, tok, bank, [bank["entries"][0]]) == [("single", "f1.safetensors")]

    c1 = lru.get_or_build(None, tok, bank, [bank["entries"][0], bank["entries"][1]])
    assert lru.misses == 1 and lru.hits == 0 and len(prefill_calls) == 1
    c1[0].data[1].append(999)  # mutate returned clone; LRU snapshot must remain untouched

    c2 = lru.get_or_build(None, tok, bank, [bank["entries"][0], bank["entries"][1]])
    assert lru.misses == 1 and lru.hits == 1 and len(prefill_calls) == 1
    assert 999 not in c2[0].data[1]

    # Insert 2 more combos into maxsize=2 -> evicts (k1, k2).
    lru.get_or_build(None, tok, bank, [bank["entries"][1], bank["entries"][2]])
    lru.get_or_build(None, tok, bank, [bank["entries"][0], bank["entries"][2]])
    assert lru.misses == 3
    lru.get_or_build(None, tok, bank, [bank["entries"][0], bank["entries"][1]])
    assert lru.misses == 4 and len(prefill_calls) == 4


def test_clone_cache_arrays_with_mlx():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.cache import ArraysCache

    if not hasattr(ArraysCache, "from_state"):
        pytest.skip("ArraysCache.from_state unavailable")
    ac = ArraysCache(size=2)
    ac[0] = mx.ones((1, 4))
    ac[1] = mx.zeros((1, 4))
    cloned = kv.clone_cache([ac])
    cloned[0][0] = mx.zeros((1, 4))
    assert float(mx.sum(ac[0])) == 4.0 and float(mx.sum(cloned[0][0])) == 0.0


class MultiTurnMockTok:
    """Chat tokenizer that appends user/assistant turns monotonically (or re-encodes history when `mutate_history=True`)."""

    bos_token_id = 2
    eos_token_ids = [1]

    def __init__(self, mutate_history=False):
        self.mutate_history = mutate_history

    def _enc(self, s: str):
        return [100 + ord(c) for c in s]

    def decode(self, toks):
        return "".join(chr(t - 100) for t in toks if t >= 100)

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=False):
        sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
        out = [self.bos_token_id]
        if sys_msgs:
            out += [10] + self._enc(sys_msgs[0]) + [11]
        for idx, m in enumerate(messages):
            if m["role"] == "user":
                out += [20] + self._enc(m["content"]) + [21]
            elif m["role"] == "assistant":
                content = "X" if (self.mutate_history and idx == 1) else m["content"]
                out += self._enc(content) + [31]
        return out


def test_kv_session_multiturn_reuse_switch_compose_and_rebuild(tmp_path, monkeypatch):
    dsl = tmp_path / "stateful.dsl"
    dsl.write_text(STATEFUL_DSL, encoding="utf-8")
    bank = {
        "dir": str(tmp_path),
        "rules": rules.standalone(str(dsl)),
        "strip_prefix": [2],
        "entries": [
            {"key": "billing", "when": {"billing": "on"}, "text": "Retry 3.", "file": "b.safetensors"},
            {"key": "auth", "when": {"auth": "on"}, "text": "Ed25519.", "file": "a.safetensors"},
        ],
    }

    built = []

    class FakeLRU:
        hits = 0
        misses = 0

        def get_or_build(self, model, tok, b, hits):
            key = tuple(e["key"] for e in hits)
            built.append(key)
            return ["cache", key]

    tok = MultiTurnMockTok()

    ok = [100 + ord("o"), 100 + ord("k")]

    def fake_decode(feed_ids, prompt_cache):  # like generate_step: the cache holds the end-of-turn token it stopped on
        return ok, "stop", ok + [31]

    sess = kv.KVSession(None, tok, bank, max_tokens=16, max_match=3, compose_cache=FakeLRU(), decode_fn=fake_decode, make_cache_fn=lambda m: ["empty"])

    r0 = sess.turn("日本の首都は?")
    assert r0["kv"] is None and r0["cache_action"] == "plain" and r0["events"] == []

    r1 = sess.turn("billing timeout?")
    assert r1["kv"] == "billing" and r1["cache_action"] == "switch" and r1["events"] == [0]
    assert built == [("billing",)]

    r2 = sess.turn("retry limit?")
    expected_delta = len([20] + tok._enc("retry limit?") + [21])  # the end-of-turn token 31 is in the cache already
    assert r2["kv"] == "billing" and r2["cache_action"] == "reuse" and r2["prefill_tokens"] == expected_delta
    assert built == [("billing",)]  # zero rebuild or reload on topic continuation

    r3 = sess.turn("billing + auth headers?")
    assert r3["kv"] == ["billing", "auth"] and r3["cache_action"] == "compose" and r3["events"] == [0]
    assert built == [("billing",), ("billing", "auth")]

    sess.reset()
    assert sess.steps == [] and sess.active_keys is None and sess.live_cache is None

    # If a tokenizer rewrites earlier history (e.g. stripping thinking blocks), KVSession falls back to 'rebuild'.
    tok_mut = MultiTurnMockTok(mutate_history=True)
    sess2 = kv.KVSession(None, tok_mut, bank, max_tokens=16, max_match=3, compose_cache=FakeLRU(), decode_fn=fake_decode, make_cache_fn=lambda m: ["empty"])
    t0 = sess2.turn("billing timeout?")
    t1 = sess2.turn("retry limit?")
    assert t0["cache_action"] == "init" and t1["cache_action"] == "rebuild"

    # Cut at MAX_TOKENS: the cache holds every token yielded and no end-of-turn token, so the next turn feeds that token.
    sess3 = kv.KVSession(None, tok, bank, max_tokens=2, max_match=3, compose_cache=FakeLRU(), decode_fn=lambda f, c: (ok, "length", ok), make_cache_fn=lambda m: ["empty"])
    sess3.turn("billing timeout?")
    u1 = sess3.turn("retry limit?")
    assert u1["cache_action"] == "reuse" and u1["prefill_tokens"] == len([31, 20] + tok._enc("retry limit?") + [21])


def test_generate_multi_match_passes_kestrom_verify(tmp_path, monkeypatch):
    pytest.importorskip("mlx")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "rules.dsl").write_text(STATEFUL_DSL, encoding="utf-8")
    (bank_dir / "b.safetensors").write_bytes(b"b-cache")
    (bank_dir / "a.safetensors").write_bytes(b"a-cache")
    (bank_dir / "manifest.json").write_text(json.dumps({
        "format": kv.FORMAT,
        "fingerprint": kv.model_fingerprint(model_dir),
        "rules_sha256": kv.sha256_file(bank_dir / "rules.dsl"),
        "strip_prefix": [2],
        "entries": [
            {"key": "billing", "when": {"billing": "on"}, "text": "B", "file": "b.safetensors", "cache_sha256": kv.sha256_bytes(b"b-cache")},
            {"key": "auth", "when": {"auth": "on"}, "text": "A", "file": "a.safetensors", "cache_sha256": kv.sha256_bytes(b"a-cache")},
        ],
    }), encoding="utf-8")
    use_dsl = tmp_path / "use.dsl"
    use_dsl.write_text(f"kv: {bank_dir} 2\n", encoding="utf-8")

    class FakeTok:
        bos_token_id = 2
        eos_token_ids = [1]

        def get_vocab(self):
            return {}

        def convert_tokens_to_ids(self, s):
            return None

        def decode(self, xs):
            return "".join(chr(97 + x) for x in xs)

        def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=False):
            sys_m = [m["content"] for m in messages if m["role"] == "system"]
            usr_m = [m["content"] for m in messages if m["role"] == "user"]
            return [2] + ([10] + [100 + ord(c) for c in sys_m[0]] + [11] if sys_m else []) + [20] + [100 + ord(c) for c in usr_m[0]] + [21]

    def fake_generate_step(prompt, model, max_tokens=16, logits_processors=None, prompt_cache=None):
        seq = [14, 15, 1] if (prompt_cache and prompt_cache[0] == "composed") else ([12, 13, 1] if prompt_cache else [10, 11, 1])
        for t in seq[:max_tokens]:
            yield t, None

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
    monkeypatch.setattr(kv, "load_cache_file", lambda path: [("single", Path(path).name)])
    monkeypatch.setattr(kv, "prefill_prefix", lambda model, prefix_ids: ["composed", list(prefix_ids)])
    monkeypatch.setattr(kv, "clone_cache", lambda cache: list(cache))

    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(json.dumps([
        {"id": "p0", "ids": [2, 20, 21, 22], "data": {"text": "天気"}},
        {"id": "p1", "ids": [2, 20, 21, 22], "data": {"text": "billing"}},
        {"id": "p2", "ids": [2, 20, 21, 22], "data": {"text": "billing と auth"}},
    ], ensure_ascii=False), encoding="utf-8")
    out_kv = tmp_path / "kv.jsonl"
    try:
        generate.run(str(model_dir), str(prompts_file), str(out_kv), max_tokens=16, rules_file=str(use_dsl))
    finally:
        rules.SIDE.clear()

    rows = [json.loads(line) for line in out_kv.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kv"] is None and rows[0]["events"] == [] and rows[0]["tokens"] == [10, 11]
    assert rows[1]["kv"] == "billing" and rows[1]["events"] == [0] and rows[1]["tokens"] == [12, 13]
    assert rows[2]["kv"] == ["billing", "auth"] and rows[2]["events"] == [0] and rows[2]["tokens"] == [14, 15]

    base = [{"id": f"p{i}", "finish": "stop", "tokens": [10, 11], "text": "kl"} for i in range(3)]
    b1, b2 = tmp_path / "b1.jsonl", tmp_path / "b2.jsonl"
    report.write_lines(b1, base)
    report.write_lines(b2, base)
    rep1, rep2 = tmp_path / "r1.json", tmp_path / "r2.json"
    report.greedy("events", (b1, b2, out_kv), FakeTok().decode, rep1)
    report.greedy("events", (b1, b2, out_kv), FakeTok().decode, rep2)
    assert rep1.read_bytes() == rep2.read_bytes()
    assert all(ok for *_, ok in report.verdict(report.read(rep1)))


def test_kv_prompt_probe_routing_and_hybrid_precedence(tmp_path):
    import numpy as np

    pure_dsl = tmp_path / "pure.dsl"
    pure_dsl.write_text("h1 s: all(p_sci)\nh2 S: any(s)\nout science: 1*S\n", encoding="utf-8")
    hybrid_dsl = tmp_path / "hybrid.dsl"
    hybrid_dsl.write_text(
        'in kw: match "barrier"\n'
        "h1 b: all(kw)\n"
        "h1 s: all(p_sci, !kw)\n"
        "h2 B: any(b)\n"
        "h2 S: any(s)\n"
        "out barrier: 1*B\n"
        "out science: 1*S\n",
        encoding="utf-8",
    )
    pure_bank = {
        "rules": rules.standalone(str(pure_dsl)),
        "entries": [{"key": "science", "when": "science"}],
        "strip_prefix": [2],
    }
    hybrid_bank = {
        "rules": rules.standalone(str(hybrid_dsl)),
        "entries": [{"key": "barrier", "when": "barrier"}, {"key": "science", "when": "science"}],
        "strip_prefix": [2],
    }
    try:
        rules.SIDE.clear()
        rules.SIDE.update({"kv": (pure_bank, 1), "prompt_on": {"p_sci": True}})
        # Bank with no regex inputs allows omitting "data"
        assert rules.kv_for({"id": "p0"}, [2, 20, 21])[0] == "science"
        assert rules.kv_for({"id": "p1", "data": {"values": {"p_sci": 0}}}, [2, 20, 21]) is None
        assert rules.kv_for({"id": "p2", "data": {"steps": [{"values": {"extra": 1}}]}}, [2, 20, 21])[0] == "science"
        # The reading goes to the last step; an earlier step gives its own, or the run names the prompt and the input
        assert rules.kv_for({"id": "p4", "data": {"steps": [{"values": {"p_sci": 0}}, {}]}}, [2, 20, 21])[0] == "science"
        with pytest.raises(SystemExit, match="entry p5: no value for p_sci"):
            rules.kv_for({"id": "p5", "data": {"steps": [{}, {}]}}, [2, 20, 21])
        rules.SIDE["prompt_on"] = {"p_sci": False}
        assert rules.kv_for({"id": "p3"}, [2, 20, 21]) is None

        # Hybrid bank (regex OR prompt_on) requires "data" and combines both signals
        rules.SIDE.update({"kv": (hybrid_bank, 1), "prompt_on": {"p_sci": True}})
        with pytest.raises(SystemExit):
            rules.kv_for({"id": "missing"}, [2, 20, 21])
        assert rules.kv_for({"id": "h0", "data": {"text": "plain science question"}}, [2, 20, 21])[0] == "science"
        assert rules.kv_for({"id": "h1", "data": {"text": "what is barrier?"}}, [2, 20, 21])[0] == "barrier"
        rules.SIDE["prompt_on"] = {"p_sci": False}
        assert rules.kv_for({"id": "h2", "data": {"text": "python list"}}, [2, 20, 21]) is None
    finally:
        rules.SIDE.clear()

    # load_prompt_probes validation & KVSession multi-turn reuse with probe + prev
    cond = np.zeros((4, 3), np.float32)
    cond[2] = 1.0
    npz = tmp_path / "bridged.npz"
    np.savez(npz, cond=cond, threshold=np.zeros(4, np.float32))

    probes_dsl = tmp_path / "use.dsl"
    probes_dsl.write_text(f"in p_sci: prompt {npz} 2\nkv: {tmp_path / 'bank'} 1\n", encoding="utf-8")
    loaded = kv.load_prompt_probes(str(probes_dsl))
    assert "p_sci" in loaded and loaded["p_sci"][1] == 2

    for bad_name, bad_body in (
        ("empty.dsl", f"kv: {tmp_path / 'bank'} 1\n"),
        ("has_unit.dsl", f"in p_sci: prompt {npz} 2\nh1 s: all(p_sci)\n"),
        ("bad_in.dsl", 'in k: match "foo"\n'),
        ("zero_layer.dsl", f"in p_sci: prompt {npz} 1\n"),
    ):
        (tmp_path / bad_name).write_text(bad_body, encoding="utf-8")
        with pytest.raises(SystemExit):
            kv.load_prompt_probes(str(tmp_path / bad_name))

    with pytest.raises(SystemExit):
        kv.chat("unused", "unused", turns_file="turns.json", out_file="-")

    chat_dsl = tmp_path / "chat_bank.dsl"
    chat_dsl.write_text(
        'in prev_s: prev S\n'
        'in kw_py: match "Python"\n'
        'h1 s: all(p_sci)\n'
        'h1 k: all(prev_s, !kw_py)\n'
        'h2 S: any(s, k)\n'
        'out science: 1*S\n',
        encoding="utf-8",
    )
    tok = MultiTurnMockTok()
    chat_bank = {
        "rules": rules.standalone(str(chat_dsl)),
        "entries": [{"key": "science", "when": "science", "file": "s.safetensors"}],
        "strip_prefix": [2],
    }

    class FakeCC:
        def get_or_build(self, model, tok, bank, hits):
            return [hits[0]["key"]]

    seen_calls = []

    def fake_probe(ids, d, layer):
        seen_calls.append((ids, layer))
        # fire only on the turn mentioning tides (simulated prompt probe firing)
        return ids == tok.apply_chat_template([{"role": "user", "content": "what causes tides"}], enable_thinking=False)

    ok = [100 + ord("o"), 100 + ord("k")]
    sess = kv.KVSession(
        None,
        tok,
        chat_bank,
        max_tokens=16,
        max_match=1,
        compose_cache=FakeCC(),
        decode_fn=lambda f, c: (ok, "stop", ok + [31]),
        make_cache_fn=lambda m: ["empty"],
        prompt_probes=loaded,
        probe_fn=fake_probe,
    )
    t0 = sess.turn("Python list?")
    t1 = sess.turn("what causes tides")
    t2 = sess.turn("tell me more about it")
    t3 = sess.turn("Python dict?")
    assert t0["kv"] is None and t0["cache_action"] == "plain" and t0["probes"] == {"p_sci": 0}
    assert t1["kv"] == "science" and t1["cache_action"] == "switch" and t1["probes"] == {"p_sci": 1}
    assert t2["kv"] == "science" and t2["cache_action"] == "reuse" and t2["probes"] == {"p_sci": 0}
    assert t3["kv"] is None and t3["cache_action"] == "switch" and t3["probes"] == {"p_sci": 0}
    assert len(seen_calls) == 4 and all(layer == 2 for _, layer in seen_calls)
    # An explicit `values` replaces those of `data`, with the probe readings under it
    sess.turn("Python set?", values={"b": 0}, data={"text": "Python set?", "values": {"a": 1}})
    assert sess.steps[-1]["values"] == {"p_sci": 0, "b": 0}


def test_model_fingerprint_rejects_same_size_weight_tampering(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    shard = model_dir / "model.safetensors"
    shard.write_bytes(b"weights-original")

    fp = kv.model_fingerprint(model_dir)
    assert len(fp["shards"]) == 1 and len(fp["shards"][0]) == 3
    bank = {"dir": str(tmp_path / "bank"), "fingerprint": fp}
    kv.check_model(bank, model_dir)

    # Modify 1 byte while keeping identical byte length; fingerprint must reject it.
    shard.write_bytes(b"weights-originaX")
    with pytest.raises(SystemExit):
        kv.check_model(bank, model_dir)
