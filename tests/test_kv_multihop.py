"""Tests for scalable kv storage (`q8`, `q4`, `lazy`), 1-Forward Ridge routing (`fit-router`), `ingest`, and `multihop`."""
import json
import sys
import types

import numpy as np
import pytest

from kestrom import kv, kv_multihop, rules
from kestrom_verify import report


class MockChatTok:
    """Minimal chat tokenizer with deterministic system prefix isolation."""

    def __init__(self, bos=2):
        self.bos_token_id = bos
        self.eos_token_ids = [1]

    def _enc(self, s: str):
        return [100 + ord(c) for c in s]

    def get_vocab(self):
        return {}

    def decode(self, toks):
        return "".join(chr(t - 100) if 100 <= t < 1100 else f"[{t}]" for t in toks)

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=False):
        sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
        usr_msgs = [m["content"] for m in messages if m["role"] == "user"]
        ast_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
        prefix = [self.bos_token_id] if self.bos_token_id is not None else []
        sys_part = ([10] + self._enc(sys_msgs[0]) + [11]) if sys_msgs else []
        body = []
        for i, u in enumerate(usr_msgs):
            body.extend([20] + self._enc(u) + [21])
            if i < len(ast_msgs):
                body.extend([30] + self._enc(ast_msgs[i]) + [31])
        if add_generation_prompt:
            body.append(40)
        return prefix + sys_part + body


def test_quantized_cache_roundtrip_all_types(tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

    c_kv = KVCache()
    c_kv.update_and_fetch(mx.ones((1, 2, 16, 64), mx.bfloat16) * 0.5, mx.ones((1, 2, 16, 64), mx.bfloat16) * -0.25)
    c_rot = RotatingKVCache(max_size=32, keep=0)
    c_rot.update_and_fetch(mx.ones((1, 2, 12, 128), mx.bfloat16) * 0.75, mx.ones((1, 2, 12, 128), mx.bfloat16) * 0.25)
    c_arr = ArraysCache(size=2)
    c_arr[0] = mx.zeros((1, 4, 64), mx.bfloat16)
    c_arr[1] = mx.ones((1, 8, 64), mx.bfloat16) * 0.125

    for bits, storage in ((8, "q8"), (4, "q4")):
        out_f = tmp_path / f"cache_{storage}.safetensors"
        kv._save_quantized_cache(out_f, [c_kv, c_rot, c_arr], {"kestrom_kv_key": "k"}, bits=bits)
        loaded = kv.load_cache_file(str(out_f))
        assert len(loaded) == 3
        assert isinstance(loaded[0], KVCache) and loaded[0].offset == 16
        assert isinstance(loaded[1], RotatingKVCache) and loaded[1].offset == 12
        assert isinstance(loaded[2], ArraysCache)
        assert loaded[0].keys.dtype == mx.bfloat16
        assert float(mx.max(mx.abs(loaded[0].keys - c_kv.keys)).item()) < 0.08
        assert float(mx.max(mx.abs(loaded[1].keys - c_rot.keys)).item()) < 0.08


def test_compile_lazy_storage_and_storage_switch(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    rules_file = tmp_path / "rules.dsl"
    rules_file.write_text(
        'in a: match "Alpha"\nh1 ha: any(a)\nh2 va: any(ha)\nout ha/hit: 1*va\n'
        'in b: match "Beta"\nh1 hb: any(b)\nh2 vb: any(hb)\nout hb/hit: 1*vb\n',
        encoding="utf-8",
    )
    facts_file = tmp_path / "facts.json"
    facts_file.write_text(
        json.dumps([
            {"key": "doc_a", "when": {"ha": "hit"}, "text": "Alpha -> Beta"},
            {"key": "doc_b", "when": {"hb": "hit"}, "text": "Beta -> Done"},
        ]),
        encoding="utf-8",
    )

    fake_tf = types.ModuleType("transformers")
    fake_auto = types.SimpleNamespace(from_pretrained=lambda path: MockChatTok(bos=2))
    fake_tf.AutoTokenizer = fake_auto
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    lazy_dir = tmp_path / "bank_lazy"
    kv.compile(str(model_dir), str(rules_file), str(facts_file), str(lazy_dir), storage="lazy")
    assert not list(lazy_dir.glob("*.safetensors"))
    bank = kv.load_bank(str(lazy_dir), MockChatTok(bos=2))
    assert bank["storage"] == "lazy"
    assert all(e["file"] is None and isinstance(e["prefix_ids"], list) for e in bank["entries"])

    prefills = []
    monkeypatch.setattr(kv, "prefill_prefix", lambda m, pfx: (prefills.append(list(pfx)), [list(pfx)])[1])
    monkeypatch.setattr(kv, "clone_cache", lambda snap: [list(x) for x in snap])
    cc = kv.ComposeCache(maxsize=4)
    c1 = cc.get_or_build(None, MockChatTok(bos=2), bank, [bank["entries"][0]])
    c2 = cc.get_or_build(None, MockChatTok(bos=2), bank, [bank["entries"][0]])
    assert c1 == c2 and len(prefills) == 1 and cc.hits == 1 and cc.misses == 1

    # Tampering with prefix_ids in manifest after load-time check missing would still be caught via missing file check.
    bad_dir = tmp_path / "bank_bad"
    bad_dir.mkdir()
    (bad_dir / "rules.dsl").write_text(rules_file.read_text(encoding="utf-8"), encoding="utf-8")
    (bad_dir / "manifest.json").write_text(
        json.dumps({
            "format": kv.FORMAT,
            "fingerprint": kv.model_fingerprint(model_dir),
            "storage": "eager",
            "rules_sha256": kv.sha256_file(bad_dir / "rules.dsl"),
            "strip_prefix": [2],
            "entries": [{"key": "doc_a", "when": {"ha": "hit"}, "text": "x", "prefix_ids": [2, 10], "file": None}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        kv.load_bank(str(bad_dir))

    # Switching from lazy/other storage forces clean compile in eager mode (no stale reuse).
    compile_calls = []
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda path: (object(), MockChatTok(bos=2))
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setattr(kv, "compile_prefix", lambda m, pfx, dst, k: (compile_calls.append(k), dst.write_bytes(b"e")))
    eager_dir = tmp_path / "bank_eager"
    kv.compile(str(model_dir), str(rules_file), str(facts_file), str(eager_dir), prev_dir=str(lazy_dir), storage="eager")
    assert compile_calls == ["doc_a", "doc_b"]


def test_fit_router_route_prompt_and_verify_chat(tmp_path):
    tok = MockChatTok(bos=2)

    def feat(ids):
        # Deterministic 6D embedding where content distinguishes class0, class1, neutral, and ambiguous cleanly.
        s = tok.decode(ids)
        if "weather" in s or "capital" in s:
            return np.array([-3.0, -3.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        if "alpha" in s and "beta" in s:
            return np.array([1.0, 1.0, -3.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if "alpha" in s or "dawn" in s:
            return np.array([2.5, 0.2, 2.5, 0.1, 0.0, 0.0], dtype=np.float32)
        if "beta" in s or "twilight" in s:
            return np.array([0.2, 2.5, 2.5, 0.1, 0.0, 0.0], dtype=np.float32)
        return np.array([-2.0, -2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    data = {
        "classes": {
            "r0": ["alpha item", "alpha note"],
            "r1": ["beta item", "beta note"],
        },
        "wrappers": ["Tell me about {}."],
        "neutral": ["What is the weather?", "What is the capital?"],
        "ambiguous": ["Compare alpha and beta."],
    }

    class DummyModel:
        layers = [object(), object(), object(), object()]

    router = kv_multihop.fit_router_data(
        DummyModel(), tok, data, layer=2, alpha=0.1, fingerprint={"meta": {}, "shards": []}, h_fn=feat
    )
    npz_path = tmp_path / "router.npz"
    kv_multihop.save_router(npz_path, router)
    loaded = kv_multihop.load_router(npz_path)

    ids_r0 = kv._chat_messages_ids(tok, [{"role": "user", "content": "Mention dawn please."}])
    ids_r1 = kv._chat_messages_ids(tok, [{"role": "user", "content": "Mention twilight please."}])
    ids_amb = kv._chat_messages_ids(tok, [{"role": "user", "content": "Compare alpha and beta now."}])
    ids_out = kv._chat_messages_ids(tok, [{"role": "user", "content": "What is the capital city?"}])

    assert kv_multihop.route_prompt(None, ids_r0, loaded, h_fn=feat)[0] == "r0"
    assert kv_multihop.route_prompt(None, ids_r1, loaded, h_fn=feat)[0] == "r1"
    assert kv_multihop.route_prompt(None, ids_amb, loaded, h_fn=feat)[0] is None
    assert kv_multihop.route_prompt(None, ids_out, loaded, h_fn=feat)[0] is None

    dsl_path = tmp_path / "r.dsl"
    dsl_path.write_text('in m0: match "r0"\nh1 u0: any(m0)\nh2 v0: any(u0)\nout r0: 1*v0\n', encoding="utf-8")
    bank = {
        "dir": str(tmp_path),
        "rules": rules.standalone(str(dsl_path)),
        "strip_prefix": [2],
        "storage": "lazy",
        "entries": [
            {"key": "r0", "when": "r0", "text": "R0 fact", "file": None, "prefix_ids": [2, 10, 11]},
            {"key": "r1", "when": "r1", "text": "R1 fact", "file": None, "prefix_ids": [2, 10, 12]},
        ],
    }

    class FakeCompose(kv.ComposeCache):
        def get_or_build(self, model, tok_, b, hits):
            return ["cache", [e["key"] for e in hits]]

    rep = kv.verify_chat_session(
        None,
        tok,
        bank,
        turns=[
            "What is the capital city?",
            "Mention dawn please.",
            "Compare alpha and beta now.",
        ],
        max_tokens=8,
        compose_cache=FakeCompose(),
        decode_fn=lambda feed, cache: ([50, 51] if isinstance(cache, list) and cache and cache[0] == "cache" else [10, 11], "stop", [50, 51, 1]),
        make_cache_fn=lambda m: [],
        ridge_router=loaded,
        route_fn=lambda ids: kv_multihop.route_prompt(None, ids, loaded, h_fn=feat),
    )
    assert rep["verdict"] == "PASS"
    assert rep["totals"]["kv_matched_turns"] == 1
    assert rep["totals"]["kv_none_turns"] == 2
    assert rep["rows"][0]["kv"] is None and rep["rows"][1]["kv"] == "r0" and rep["rows"][2]["kv"] is None


def test_ingest_incremental_and_1000_docs_scale(tmp_path):
    corpus_small = [
        {"doc_id": "doc_p0", "title": "Alden Rook", "text": "Unrelated travel memo. Alden Rook leads Project Kestrel."},
        {"doc_id": "doc_proj0", "title": "Project Kestrel", "text": "Project Kestrel is hosted at Vespera Lab."},
        {"doc_id": "doc_lab0", "title": "Vespera Lab", "text": "Vespera Lab uses code KZ-419."},
    ]
    c_path = tmp_path / "corpus.json"
    c_path.write_text(json.dumps(corpus_small), encoding="utf-8")

    extract_calls = []

    def fake_extract(doc, mode):
        extract_calls.append(doc["doc_id"])
        return {
            "doc_id": doc["doc_id"],
            "raw_entities": [doc["title"]],
            "raw_aliases": [],
            "raw_facts": [doc["text"]],
            "valid": True,
        }

    out1 = tmp_path / "ing1"
    rep1 = kv_multihop.ingest("unused", str(c_path), str(out1), mode="key-facts", extract_fn=fake_extract)
    assert rep1["verdict"] == "PASS" and rep1["extracted"] == 3 and rep1["reused"] == 0
    assert (out1 / "index.json").is_file() and (out1 / "rules.dsl").is_file() and (out1 / "facts.json").is_file()
    idx1 = report.read(str(out1 / "index.json"))
    assert "Project Kestrel" in idx1["spec_by_id"]["doc_proj0"]
    assert "Vespera Lab" in idx1["spec_by_id"]["doc_lab0"]
    facts = report.read(str(out1 / "facts.json"))
    assert "Key Facts:" in facts[0]["text"]

    # Edit only 1 doc -> incremental ingest reuses 2/3 unchanged docs.
    extract_calls.clear()
    corpus_small[1]["text"] = "Project Kestrel is hosted at Boreal Lab."
    c2_path = tmp_path / "corpus2.json"
    c2_path.write_text(json.dumps(corpus_small), encoding="utf-8")
    out2 = tmp_path / "ing2"
    rep2 = kv_multihop.ingest("unused", str(c2_path), str(out2), prev_dir=str(out1), mode="key-facts", extract_fn=fake_extract)
    assert rep2["reused"] == 2 and rep2["extracted"] == 1 and extract_calls == ["doc_proj0"]

    # 1,000-doc scale fast ingest (no LLM) + standalone verification.
    docs_1k = [
        {
            "doc_id": f"d_{i:04d}",
            "title": f"Entity{i:04d}",
            "text": f"Entity{i:04d} links to Target{(i + 1) % 1000:04d} with tag K-{i:04d}.",
        }
        for i in range(1000)
    ]
    c1k_path = tmp_path / "corpus_1k.jsonl"
    c1k_path.write_text("\n".join(json.dumps(d) for d in docs_1k) + "\n", encoding="utf-8")
    out1k = tmp_path / "ing1k"
    rep1k = kv_multihop.ingest("unused", str(c1k_path), str(out1k), mode="fast")
    assert rep1k["verdict"] == "PASS" and rep1k["docs"] == 1000


def test_fast_prefix_extraction_checks_every_probe_turn():
    class SecondProbeMerges(MockChatTok):
        def apply_chat_template(self, messages, **kw):
            ids = super().apply_chat_template(messages, **kw)
            if any(m["role"] == "system" and m["content"].endswith("Z") for m in messages) and any(
                    m["role"] == "user" and m["content"] == kv.PROBES[1] for m in messages):
                ids.remove(11)
            return ids

    tok = SecondProbeMerges(bos=2)
    us = [kv._chat_ids(tok, p, None, mode="system") for p in kv.PROBES]
    sp, _ = kv.prefix_boundary(tok, "first fact", mode="system")
    assert kv._extract_prefix_fast(tok, "plain fact", "system", sp, us) == kv.prefix_boundary(tok, "plain fact")[1]
    with pytest.raises(SystemExit, match="varies across probes"):
        kv._extract_prefix_fast(tok, "fact Z", "system", sp, us)


def test_ingest_rules_take_any_doc_id_read_without_values_and_bound_each_term(tmp_path):
    corpus = [
        {"doc_id": "2024-q3 report:a", "title": "東京タワー", "text": "東京タワー hosts Ann Vale in Kyoto."},
        {"doc_id": "doc_b", "title": "Boreal Lab", "text": "Boreal Lab sits in Oslo."},
    ]
    c_path = tmp_path / "corpus.json"
    c_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    def fake_extract(doc, mode):
        ents = {"doc_b": ["Boreal Lab", "Oslo"]}.get(doc["doc_id"], ["東京タワー", "Ann Vale"])
        return {"doc_id": doc["doc_id"], "raw_entities": ents, "raw_aliases": [], "raw_facts": [doc["text"]], "valid": True}

    out = tmp_path / "ing"
    assert kv_multihop.ingest("unused", str(c_path), str(out), mode="key-facts", extract_fn=fake_extract)["verdict"] == "PASS"
    R = rules.standalone(str(out / "rules.dsl"))
    keys = {f["key"] for f in report.read(str(out / "facts.json"))}
    assert keys == {"2024-q3 report:a", "doc_b"}

    def cls(text, values=None):
        x = {"text": text, **({"values": values} if values else {})}
        return next(rules.answers(R, [x]))[0]["class"]

    # No values given (as kv chat and generate run): hop_expand reads 0, so the broad tier stays off.
    assert cls("Where is Oslo?") == {"h_d0": None, "h_d1": None}
    assert cls("Where is Oslo?", {"hop_expand": 1}) == {"h_d0": None, "h_d1": "hit"}
    assert cls("Who is Ann Vale?")["h_d0"] == "hit"
    # A non-ASCII term in the same document leaves the ASCII terms bounded, as the index bounds them.
    assert cls("Ann Valerie wrote it.")["h_d0"] is None


def test_multihop_early_stop_3hop_and_outside_bit_identical(tmp_path):
    dsl = tmp_path / "mh.dsl"
    dsl.write_text(
        'in m_p0: match "Alden Rook"\nh1 u_p0: any(m_p0)\nh2 v_p0: any(u_p0)\nout h_p0/hit: 1*v_p0\n'
        'in m_j0: match "Project Kestrel"\nh1 u_j0: any(m_j0)\nh2 v_j0: any(u_j0)\nout h_j0/hit: 1*v_j0\n'
        'in m_l0: match "Vespera Lab"\nh1 u_l0: any(m_l0)\nh2 v_l0: any(u_l0)\nout h_l0/hit: 1*v_l0\n',
        encoding="utf-8",
    )
    bank = {
        "dir": str(tmp_path),
        "rules": rules.standalone(str(dsl)),
        "strip_prefix": [2],
        "storage": "lazy",
        "entries": [
            {"key": "doc_p0", "when": {"h_p0": "hit"}, "text": "Alden Rook leads Project Kestrel.", "file": None, "prefix_ids": [2, 10]},
            {"key": "doc_proj0", "when": {"h_j0": "hit"}, "text": "Project Kestrel is hosted at Vespera Lab.", "file": None, "prefix_ids": [2, 11]},
            {"key": "doc_lab0", "when": {"h_l0": "hit"}, "text": "Vespera Lab uses code KZ-419.", "file": None, "prefix_ids": [2, 12]},
        ],
    }

    class FakeCompose(kv.ComposeCache):
        def get_or_build(self, model, tok_, b, hits):
            return tuple(e["key"] for e in hits)

    class AsciiTok(MockChatTok):
        def _enc(self, s: str):
            return [100 + ord(c) for c in s]

        def decode(self, toks):
            return "".join(chr(t - 100) if 100 <= t < 1100 else f"[{t}]" for t in toks)

    atok = AsciiTok(bos=2)

    def clean_decode(feed, cache, max_tokens):
        if cache == ("plain",):
            return [60, 61], "stop", [60, 61, 1]
        active = set(cache)
        q_txt = atok.decode(feed)
        if "Status:" in q_txt:
            need_3hop = "security code" in q_txt.lower()
            if need_3hop and active == {"doc_p0", "doc_proj0", "doc_lab0"}:
                txt = "Facts: KZ-419\nAnswer: KZ-419\nStatus: SUFFICIENT"
            elif (not need_3hop) and active == {"doc_p0", "doc_proj0"}:
                txt = "Facts: Vespera Lab\nAnswer: Vespera Lab\nStatus: SUFFICIENT"
            elif "doc_proj0" in active:
                txt = "Facts: Project Kestrel at Vespera Lab\nAnswer: UNKNOWN\nStatus: NEED_MORE"
            else:
                txt = "Facts: Alden Rook leads Project Kestrel\nAnswer: UNKNOWN\nStatus: NEED_MORE"
            toks = [100 + ord(c) for c in txt]
            return toks, "stop", toks + [1]
        if "doc_lab0" in active:
            return [73], "stop", [73, 1]
        if "doc_proj0" in active:
            return [72], "stop", [72, 1]
        return [70], "stop", [70, 1]

    p_2hop = {"id": "q_2hop", "ids": kv._chat_messages_ids(atok, [{"role": "user", "content": "Which lab hosts Alden Rook?"}])}
    p_3hop = {"id": "q_3hop", "ids": kv._chat_messages_ids(atok, [{"role": "user", "content": "What security code does Alden Rook use?"}])}
    p_out = {"id": "q_out", "ids": kv._chat_messages_ids(atok, [{"role": "user", "content": "What is 17 + 28?"}])}

    rows = kv_multihop.run_multihop_prompts(
        None,
        atok,
        bank,
        [p_2hop, p_3hop, p_out],
        max_tokens=16,
        max_hops=4,
        max_active=5,
        probe_mode="structured",
        compose_cache=FakeCompose(),
        decode_fn=clean_decode,
        make_cache_fn=lambda m: ("plain",),
    )

    assert rows[0]["provenance"] == ["doc_p0", "doc_proj0"] and rows[0]["tokens"] == [72] and rows[0]["events"] == [0]
    assert rows[1]["provenance"] == ["doc_p0", "doc_proj0", "doc_lab0"] and rows[1]["tokens"] == [73] and rows[1]["events"] == [0]
    assert rows[2]["provenance"] == [] and rows[2]["tokens"] == [60, 61] and rows[2]["events"] == [] and rows[2]["kv"] is None
