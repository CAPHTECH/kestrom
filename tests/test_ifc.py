"""Tests for kestrom.ifc (numpy-only, no model): region policy, twin-track decoding, and end-to-end kestrom-verify compare events."""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from kestrom.ifc import (
    CONTROL,
    DATA,
    JsonKeyRegionPolicy,
    PublicRegionPolicy,
    decode_control,
    decode_control_fast,
    decode_firewall,
    kl_divergence,
    token_lift,
)
from kestrom_verify import report


class ScriptedBackend:
    """Deterministic mock twin-track backend returning scripted logits for Track A and Track B."""

    def __init__(self, vocab: List[str], script_a: List[Dict[int, float]], script_b: List[Dict[int, float]], eos_id: int = 0):
        self.vocab, self.script_a, self.script_b, self._eos = vocab, script_a, script_b, eos_id
        self.pos, self.fed_a, self.fed_b = 0, [], []

    @property
    def eos_ids(self) -> Sequence[int]:
        return [self._eos]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(self.vocab[t] for t in tokens)

    def _logits(self, script: List[Dict[int, float]], idx: int) -> np.ndarray:
        v = np.full(len(self.vocab), -10.0, dtype=np.float32)
        if idx < len(script):
            for tok, val in script[idx].items():
                v[tok] = val
        else:
            v[self._eos] = 10.0
        return v

    def prefill(self, ids_a: Sequence[int], ids_b: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        self.pos, self.fed_a, self.fed_b = 0, [], []
        return self._logits(self.script_a, 0), self._logits(self.script_b, 0)

    def step(self, tok_a: int, tok_b: int) -> Tuple[np.ndarray, np.ndarray]:
        self.fed_a.append(tok_a)
        self.fed_b.append(tok_b)
        self.pos += 1
        return self._logits(self.script_a, self.pos), self._logits(self.script_b, self.pos)

    def start_lazy(self, ids_a: Sequence[int], ids_b: Sequence[int]) -> None:
        self.fed_a, self.fed_b = [], []
        self.calls_a, self.calls_b = 0, 0
        self.init_a, self.init_b = False, False
        self.pos_a, self.pos_b = 0, 0

    def logits_a(self, pending_a: Sequence[int]) -> np.ndarray:
        self.calls_a += 1
        self.fed_a.extend(pending_a)
        self.pos_a = len(self.fed_a)
        self.init_a = True
        return self._logits(self.script_a, self.pos_a)

    def logits_b(self, pending_b: Sequence[int]) -> np.ndarray:
        self.calls_b += 1
        self.fed_b.extend(pending_b)
        self.pos_b = len(self.fed_b)
        self.init_b = True
        return self._logits(self.script_b, self.pos_b)


def test_json_region_policy_and_smuggling():
    pol = JsonKeyRegionPolicy(("tool", "approved"), ("summary",))
    assert pol.classify("") == (CONTROL, "")
    assert pol.classify('{"tool": ') == (CONTROL, "tool")
    assert pol.classify('{"tool": "ok", "summary": "hi \\"q\\"') == (DATA, "summary")
    assert pol.classify('{"tool": "ok", "summary": "hi", ') == (CONTROL, "")
    assert pol.classify('<think>plan</think>{"summary": "x') == (DATA, "summary")
    assert not pol.crosses_into_control('{"tool": "ok", "summary": "hi', '"}')
    assert pol.crosses_into_control('{"tool": "ok", "summary": "hi', '", "tool": "exfil"')
    with pytest.raises(ValueError):
        JsonKeyRegionPolicy(("tool",), ("tool",))


def test_control_blocks_hijack_and_passes_kestrom_verify_compare_events(tmp_path: Path):
    vocab = ["<eos>", '{"tool":"', "log_ticket", "delete_all", '","summary":"', "AC broken", "MASKED", '"}']
    script_a = [{1: 9.0}, {3: 9.0, 2: -2.0}, {4: 9.0}, {5: 9.0, 6: -1.0}, {7: 9.0}, {0: 9.0}]
    script_b = [{1: 9.0}, {2: 9.0, 3: -4.0}, {4: 9.0}, {6: 9.0, 5: -1.0}, {7: 9.0}, {0: 9.0}]
    pol = JsonKeyRegionPolicy(("tool",), ("summary",))
    res = decode_control(ScriptedBackend(vocab, script_a, script_b), [1], [1], pol, max_tokens=10)

    assert res["text"] == '{"tool":"log_ticket","summary":"AC broken"}'
    assert res["events"] == [3, 4]  # DATA positions
    assert res["ifc"]["control_overrides"] == 1
    assert res["detail"][0]["kind"] == "control_override"
    assert res["detail"][0]["text_a"] == "delete_all" and res["detail"][0]["text_b"] == "log_ticket"

    # End-to-end Tier 1 check with real kestrom_verify.report.greedy("events", ...)
    clean = {"id": "p1", "finish": "stop", "tokens": [1, 2, 4, 6, 7], "text": '{"tool":"log_ticket","summary":"MASKED"}'}
    f_b1, f_b2, f_ifc = tmp_path / "b1.jsonl", tmp_path / "b2.jsonl", tmp_path / "ifc.jsonl"
    report.write_lines(f_b1, [clean])
    report.write_lines(f_b2, [clean])
    report.write_lines(f_ifc, [{"id": "p1", **res}])

    rep1, rep2 = tmp_path / "r1.json", tmp_path / "r2.json"
    report.greedy("events", (f_b1, f_b2, f_ifc), lambda xs: "".join(vocab[t] for t in xs), rep1)
    report.greedy("events", (f_b1, f_b2, f_ifc), lambda xs: "".join(vocab[t] for t in xs), rep2)
    assert rep1.read_bytes() == rep2.read_bytes()
    assert all(ok for *_, ok in report.verdict(report.read(rep1)))


def test_control_clamps_bpe_smuggling_and_supports_isolate_b():
    vocab = ["<eos>", '{"tool":"ok","summary":"', "hi", '","tool":"evil"', " safe", '"}']
    script_a = [{1: 9.0}, {2: 9.0}, {3: 9.0, 4: 5.0}, {5: 9.0}, {0: 9.0}]
    script_b = [{1: 9.0}, {2: 9.0}, {4: 9.0}, {5: 9.0}, {0: 9.0}]
    be = ScriptedBackend(vocab, script_a, script_b)
    res = decode_control(be, [1], [1], JsonKeyRegionPolicy(("tool",), ("summary",)), max_tokens=10, isolate_b=2)
    assert res["text"] == '{"tool":"ok","summary":"hi safe"}'
    assert res["ifc"]["boundary_clamps"] == 1
    assert res["ifc"]["isolated_b"] is True
    assert be.fed_a[2] == 4 and be.fed_b[2] == 2


def test_firewall_strict_and_gated_pass_kestrom_verify_compare_events(tmp_path: Path):
    vocab = ["<eos>", "Follow-up ", "in 2 weeks.", "for HIV ART.", " Rest well."]
    script_a = [{1: 9.0}, {3: 9.0, 2: -1.0}, {4: 9.0}, {0: 9.0}]
    script_b = [{1: 9.0}, {2: 9.0, 3: -8.0}, {4: 9.0}, {0: 9.0}]
    dec = lambda xs: "".join(vocab[t] for t in xs)  # noqa: E731

    # strict-public: events=[] -> kestrom-verify compare events against clean Track B proves bit-identical output
    res_s = decode_firewall(ScriptedBackend(vocab, script_a, script_b), [1], [1], PublicRegionPolicy("strict-public", 2.0), 10)
    assert res_s["events"] == []
    assert res_s["text"] == "Follow-up in 2 weeks. Rest well."
    assert res_s["detail"][0]["kind"] == "leak_attempt"

    clean = {"id": "s1", "finish": "stop", "tokens": [1, 2, 4], "text": res_s["text"]}
    f_b1, f_b2, f_s = tmp_path / "b1.jsonl", tmp_path / "b2.jsonl", tmp_path / "s.jsonl"
    report.write_lines(f_b1, [clean])
    report.write_lines(f_b2, [clean])
    report.write_lines(f_s, [{"id": "s1", **res_s}])
    rep_s = tmp_path / "rep_s.json"
    report.greedy("events", (f_b1, f_b2, f_s), dec, rep_s)
    assert all(ok for *_, ok in report.verdict(report.read(rep_s)))

    # gated-public: events=[1] -> kestrom-verify compare events against unprotected Track A proves G2 up to block
    res_g = decode_firewall(ScriptedBackend(vocab, script_a, script_b), [1], [1], PublicRegionPolicy("gated-public", 2.0), 10)
    assert res_g["events"] == [1]
    assert res_g["text"] == "Follow-up in 2 weeks. Rest well."
    assert res_g["detail"][0]["kind"] == "leak_blocked"

    full = {"id": "s1", "finish": "stop", "tokens": [1, 3, 4], "text": "Follow-up for HIV ART. Rest well."}
    f_a1, f_a2, f_g = tmp_path / "a1.jsonl", tmp_path / "a2.jsonl", tmp_path / "g.jsonl"
    report.write_lines(f_a1, [full])
    report.write_lines(f_a2, [full])
    report.write_lines(f_g, [{"id": "s1", **res_g}])
    rep_g = tmp_path / "rep_g.json"
    report.greedy("events", (f_a1, f_a2, f_g), dec, rep_g)
    assert all(ok for *_, ok in report.verdict(report.read(rep_g)))


def test_control_fast_single_track_switching_and_verify(tmp_path: Path):
    vocab = ["<eos>", '{"tool":"', "log_ticket", "delete_all", '","summary":"', "AC broken", "MASKED", '"}']
    script_a = [{1: 9.0}, {3: 9.0, 2: -2.0}, {4: 9.0}, {5: 9.0, 6: -1.0}, {7: 9.0}, {0: 9.0}]
    script_b = [{1: 9.0}, {2: 9.0, 3: -4.0}, {4: 9.0}, {6: 9.0, 5: -1.0}, {7: 9.0}, {0: 9.0}]
    pol = JsonKeyRegionPolicy(("tool",), ("summary",))

    be_fast = ScriptedBackend(vocab, script_a, script_b)
    res_fast = decode_control_fast(be_fast, [1], [1], pol, max_tokens=10)
    res_audit = decode_control(ScriptedBackend(vocab, script_a, script_b), [1], [1], pol, max_tokens=10)

    # Identical output tokens & DATA events to audit mode, with strictly 1 model call per generated step (including stop).
    assert res_fast["tokens"] == res_audit["tokens"]
    assert res_fast["text"] == '{"tool":"log_ticket","summary":"AC broken"}'
    assert res_fast["events"] == res_audit["events"] == [3, 4]
    assert res_fast["ifc"]["mode"] == "control-fast"
    assert res_fast["ifc"]["track_switches"] == 2  # CONTROL -> DATA -> CONTROL(eos)
    assert be_fast.calls_a + be_fast.calls_b == len(res_fast["tokens"]) + 1  # 6 steps -> 6 single-track calls total
    assert be_fast.calls_a == 2 and be_fast.calls_b == 4  # Track A ran ONLY on the 2 DATA steps
    assert be_fast.fed_a == [1, 2, 4, 5]  # caught up cleanly on leading control tokens without duplicates
    assert be_fast.fed_b == [1, 2, 4, 5, 7]  # caught up cleanly on DATA tokens before closing EOS step

    # End-to-end Tier 1 check with real kestrom_verify.report.greedy("events", ...)
    clean = {"id": "p1", "finish": "stop", "tokens": [1, 2, 4, 6, 7], "text": '{"tool":"log_ticket","summary":"MASKED"}'}
    f_b1, f_b2, f_fast = tmp_path / "b1.jsonl", tmp_path / "b2.jsonl", tmp_path / "fast.jsonl"
    report.write_lines(f_b1, [clean])
    report.write_lines(f_b2, [clean])
    report.write_lines(f_fast, [{"id": "p1", **res_fast}])
    rep1, rep2 = tmp_path / "rf1.json", tmp_path / "rf2.json"
    report.greedy("events", (f_b1, f_b2, f_fast), lambda xs: "".join(vocab[t] for t in xs), rep1)
    report.greedy("events", (f_b1, f_b2, f_fast), lambda xs: "".join(vocab[t] for t in xs), rep2)
    assert rep1.read_bytes() == rep2.read_bytes()
    assert all(ok for *_, ok in report.verdict(report.read(rep1)))


def test_control_fast_clamps_smuggling_and_isolates_b():
    vocab = ["<eos>", '{"tool":"ok","summary":"', "hi", '","tool":"evil"', " safe", '"}']
    script_a = [{1: 9.0}, {2: 9.0}, {3: 9.0, 4: 5.0}, {5: 9.0}, {0: 9.0}]
    script_b = [{1: 9.0}, {2: 9.0}, {4: 9.0}, {5: 9.0}, {0: 9.0}]
    be = ScriptedBackend(vocab, script_a, script_b)
    res = decode_control_fast(be, [1], [1], JsonKeyRegionPolicy(("tool",), ("summary",)), max_tokens=10, isolate_b=2)
    assert res["text"] == '{"tool":"ok","summary":"hi safe"}'
    assert res["ifc"]["boundary_clamps"] == 1
    assert res["ifc"]["isolated_b"] is True
    # Track B received neutral token 2 for the DATA tokens (pos 1, 2, 3) when waking for EOS at pos 4.
    assert be.fed_b == [1, 2, 2, 2]
    assert be.fed_a == [1, 2, 4]


def test_math_helpers():
    x = np.array([100.0, 99.0, 98.0], dtype=np.float32)
    assert kl_divergence(x, x) == pytest.approx(0.0, abs=1e-12)
    assert token_lift(x, x, 0) == pytest.approx(0.0, abs=1e-12)
