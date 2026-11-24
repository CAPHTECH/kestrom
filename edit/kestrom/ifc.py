"""Twin-track information flow control (IFC) at greedy decoding time (MLX, tier 1). Two tracks of the same model run in lockstep
(sequential Batch=1 matching mlx_lm's generate_step prefill bit for bit): Track A reads `ids` (full prompt with untrusted or secret
context) and Track B reads `clean_ids` (same prompt with the untrusted or secret span removed or masked). No weights change.
  kestrom ifc control MODEL PROMPTS.json OUT.jsonl CONTROL_KEYS DATA_KEYS [MAX_TOKENS] [ISOLATE_B]
                                                        JSON control-flow non-interference. Structural JSON syntax and values of
                                                        CONTROL_KEYS (comma-separated) come strictly from Track B; values of
                                                        DATA_KEYS come from Track A, clamping any single token that crosses back
                                                        into control syntax. `events` records DATA positions, so
                                                        `kestrom-verify compare events CLEAN1 CLEAN2 OUT MODEL REPORT.json` checks
                                                        that every token before the first DATA span is bit-identical to Track B
                                                        (U0 and G2). `detail` records every `control_override` and `boundary_clamp`.
                                                        With ISOLATE_B (a token id), Track B ingests that neutral token during DATA
                                                        spans instead of Track A's tokens.
  kestrom ifc firewall MODEL PROMPTS.json OUT.jsonl [MODE] [TAU] [MAX_TOKENS]
                                                        Semantic secret leakage firewall. MODE is `strict-public` (default) or
                                                        `gated-public`; TAU is the log-probability lift threshold in nats (2.0).
                                                        `strict-public` emits Track B everywhere in public spans with `events=[]`,
                                                        so `kestrom-verify compare events CLEAN1 CLEAN2 OUT MODEL REPORT.json` checks
                                                        that every output is bit-identical to secret-free Track B (and `detail` logs
                                                        `leak_attempt` where Track A's top token has lift >= TAU).
                                                        `gated-public` emits Track A unless its top token has lift >= TAU, in which
                                                        case it falls back to Track B and records the position in `events` and
                                                        `detail` (`leak_blocked`); compare against two runs of unprotected Track A
                                                        with `kestrom-verify compare events` to check G2 up to the first block.
PROMPTS.json: a JSON list of {"id": ..., "ids": [Track A ids], "clean_ids": [Track B ids], other keys kept}."""
from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

from kestrom_verify import report

CONTROL = "CONTROL"
DATA = "DATA"
PUBLIC = "PUBLIC"
THINK = "THINK"


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax in float64."""
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x)
    return shifted - np.log(np.sum(np.exp(shifted)))


def kl_divergence(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """KL(P_A || P_B) in nats."""
    la, lb = log_softmax(logits_a), log_softmax(logits_b)
    return max(0.0, float(np.sum(np.exp(la) * (la - lb))))


def token_lift(logits_a: np.ndarray, logits_b: np.ndarray, tok: int) -> float:
    """Log-probability lift log P_A(tok) - log P_B(tok) in nats."""
    return float(log_softmax(logits_a)[tok] - log_softmax(logits_b)[tok])


@dataclass
class JsonState:
    """Incremental parser state for a top-level JSON object stream."""

    in_think: bool = False
    depth: int = 0
    in_string: bool = False
    escape: bool = False
    expect: str = "key_or_end"
    current_key: str = ""
    buf: str = ""
    value_depth: int = 0


class JsonKeyRegionPolicy:
    """Classify each position of a generated JSON object as CONTROL or DATA."""

    def __init__(
        self,
        control_keys: Iterable[str],
        data_keys: Iterable[str],
        default_region: str = CONTROL,
        think_start: Optional[str] = "<think>",
        think_end: Optional[str] = "</think>",
        think_region: str = CONTROL,
        start_in_think: bool = False,
    ):
        self.control_keys: Set[str] = {k for k in control_keys if k}
        self.data_keys: Set[str] = {k for k in data_keys if k}
        overlap = self.control_keys & self.data_keys
        if overlap:
            raise ValueError(f"keys cannot be both control and data: {sorted(overlap)}")
        self.default_region = default_region
        self.think_start = think_start
        self.think_end = think_end
        self.think_region = think_region
        self.start_in_think = start_in_think

    def parse(self, text: str) -> JsonState:
        st = JsonState(in_think=self.start_in_think)
        i, n = 0, len(text)
        while i < n:
            if st.in_think:
                if self.think_end and text.startswith(self.think_end, i):
                    st.in_think = False
                    i += len(self.think_end)
                    continue
                i += 1
                continue
            if self.think_start and st.depth == 0 and text.startswith(self.think_start, i):
                st.in_think = True
                i += len(self.think_start)
                continue
            ch = text[i]
            i += 1
            if st.depth == 0:
                if ch == "{":
                    st.depth, st.expect, st.current_key = 1, "key_or_end", ""
                continue
            if st.in_string:
                if st.escape:
                    st.escape = False
                    if st.expect == "key_or_end":
                        st.buf += ch
                    continue
                if ch == "\\":
                    st.escape = True
                    continue
                if ch == '"':
                    st.in_string = False
                    if st.expect == "key_or_end":
                        st.current_key, st.buf, st.expect = st.buf, "", "colon"
                    elif st.expect == "value" and st.value_depth == 0:
                        st.expect = "comma_or_end"
                    continue
                if st.expect == "key_or_end":
                    st.buf += ch
                continue
            if ch in " \t\r\n":
                continue
            if st.expect == "key_or_end":
                if ch == '"':
                    st.in_string, st.buf = True, ""
                elif ch == "}":
                    st.depth = 0
            elif st.expect == "colon":
                if ch == ":":
                    st.expect, st.value_depth = "value", 0
            elif st.expect == "value":
                if ch == '"':
                    st.in_string = True
                elif ch in "{[":
                    st.value_depth += 1
                elif ch in "}]":
                    if st.value_depth > 0:
                        st.value_depth -= 1
                        if st.value_depth == 0:
                            st.expect = "comma_or_end"
                    elif ch == "}":
                        st.depth = 0
                elif ch == "," and st.value_depth == 0:
                    st.expect, st.current_key = "key_or_end", ""
            elif st.expect == "comma_or_end":
                if ch == ",":
                    st.expect, st.current_key = "key_or_end", ""
                elif ch == "}":
                    st.depth = 0
        return st

    def classify(self, prefix_text: str) -> Tuple[str, str]:
        st = self.parse(prefix_text)
        if st.in_think:
            return self.think_region, "__think__"
        if st.depth == 0:
            return self.default_region, ""
        if st.expect == "value":
            if st.current_key in self.data_keys:
                return DATA, st.current_key
            if st.current_key in self.control_keys:
                return CONTROL, st.current_key
        return self.default_region, st.current_key

    def crosses_into_control(self, prefix_text: str, candidate_text: str) -> bool:
        """True if a single DATA token closes the data value and writes non-delimiter control syntax."""
        region_before, key_before = self.classify(prefix_text)
        if region_before != DATA:
            return False
        for k in range(1, len(candidate_text) + 1):
            reg, key = self.classify(prefix_text + candidate_text[:k])
            if reg != DATA or key != key_before:
                return any(c not in " \t\r\n,}]" for c in candidate_text[k:])
        return False


@dataclass
class PublicRegionPolicy:
    """Region policy for the semantic secret leakage firewall."""

    mode: str = "strict-public"
    tau: float = 2.0
    think_end: Optional[str] = "</think>"
    start_in_think: bool = False

    def __post_init__(self):
        if self.mode not in ("strict-public", "gated-public"):
            raise ValueError(f"MODE is strict-public or gated-public, not {self.mode!r}")

    def classify(self, prefix_text: str) -> str:
        if self.start_in_think and self.think_end and self.think_end not in prefix_text:
            return THINK
        return PUBLIC


class TwinTrackBackend(Protocol):
    """Minimal interface for a lockstep dual-track causal LM backend."""

    @property
    def eos_ids(self) -> Sequence[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...

    def prefill(self, ids_a: Sequence[int], ids_b: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]: ...

    def step(self, tok_a: int, tok_b: int) -> Tuple[np.ndarray, np.ndarray]: ...


def decode_control(
    backend: TwinTrackBackend,
    ids_a: Sequence[int],
    ids_b: Sequence[int],
    policy: JsonKeyRegionPolicy,
    max_tokens: int = report.MAX_TOKENS,
    isolate_b: Optional[int] = None,
) -> Dict[str, Any]:
    """Greedy control-flow non-interference decoding.

    Returns tokens, text, finish, events (sorted DATA positions, for `kestrom-verify compare events`
    against two runs of clean Track B), detail (`control_override` and `boundary_clamp`), and `ifc`
    summary counts.
    """
    logits_a, logits_b = backend.prefill(ids_a, ids_b)
    eos = set(backend.eos_ids)
    toks: List[int] = []
    data_positions: List[int] = []
    detail: List[Dict[str, Any]] = []
    control_count = 0
    finish = "length"

    for pos in range(max_tokens):
        prefix = backend.decode(toks)
        region, active_key = policy.classify(prefix)
        top_a, top_b = int(np.argmax(logits_a)), int(np.argmax(logits_b))

        if region == CONTROL:
            chosen = top_b
            if top_a != top_b:
                detail.append({
                    "kind": "control_override",
                    "position": pos,
                    "key": active_key,
                    "top_a": top_a,
                    "top_b": top_b,
                    "text_a": backend.decode([top_a]),
                    "text_b": backend.decode([top_b]),
                    "kl_ab": round(kl_divergence(logits_a, logits_b), 6),
                    "lift_a": round(token_lift(logits_a, logits_b, top_a), 6),
                })
            tok_a, tok_b = chosen, chosen
            if chosen not in eos:
                control_count += 1
        else:
            data_positions.append(pos)
            chosen = top_a
            cand_text = backend.decode([chosen])
            if policy.crosses_into_control(prefix, cand_text):
                order = np.argsort(-np.asarray(logits_a, dtype=np.float64), kind="stable")
                safe = next((int(c) for c in order[:64] if not policy.crosses_into_control(prefix, backend.decode([int(c)]))), top_b)
                detail.append({
                    "kind": "boundary_clamp",
                    "position": pos,
                    "key": active_key,
                    "blocked": chosen,
                    "blocked_text": cand_text,
                    "chosen": safe,
                    "chosen_text": backend.decode([safe]),
                })
                chosen = safe
            tok_a = chosen
            tok_b = isolate_b if isolate_b is not None else chosen

        if chosen in eos:
            finish = "stop"
            break
        toks.append(chosen)
        if pos + 1 < max_tokens:
            logits_a, logits_b = backend.step(tok_a, tok_b)

    return {
        "finish": finish,
        "tokens": toks,
        "text": backend.decode(toks),
        "events": data_positions,
        "detail": detail,
        "ifc": {
            "mode": "control",
            "first_data_pos": data_positions[0] if data_positions else None,
            "control_tokens": control_count,
            "data_tokens": len(toks) - control_count,
            "control_overrides": sum(1 for d in detail if d["kind"] == "control_override"),
            "boundary_clamps": sum(1 for d in detail if d["kind"] == "boundary_clamp"),
            "isolated_b": isolate_b is not None,
        },
    }


def decode_firewall(
    backend: TwinTrackBackend,
    ids_a: Sequence[int],
    ids_b: Sequence[int],
    policy: PublicRegionPolicy,
    max_tokens: int = report.MAX_TOKENS,
) -> Dict[str, Any]:
    """Greedy semantic secret leakage firewall (`strict-public` or `gated-public`).

    - `strict-public`: `events` is `[]`, so `kestrom-verify compare events` against clean Track B
      verifies bit-identical zero leakage; `detail` logs `leak_attempt` entries.
    - `gated-public`: `events` records positions where a secret-driven token (`lift >= tau`) was
      blocked and replaced by Track B, so `kestrom-verify compare events` against unprotected
      Track A verifies G2 up to the first blocked token.
    """
    logits_a, logits_b = backend.prefill(ids_a, ids_b)
    eos = set(backend.eos_ids)
    toks: List[int] = []
    events: List[int] = []
    detail: List[Dict[str, Any]] = []
    max_lift = 0.0
    max_kl = 0.0
    finish = "length"

    for pos in range(max_tokens):
        region = policy.classify(backend.decode(toks))
        top_a, top_b = int(np.argmax(logits_a)), int(np.argmax(logits_b))
        lift_a = token_lift(logits_a, logits_b, top_a)
        kl_ab = kl_divergence(logits_a, logits_b)

        if region == PUBLIC:
            max_lift = max(max_lift, lift_a)
            max_kl = max(max_kl, kl_ab)
            if policy.mode == "strict-public":
                chosen = top_b
                if top_a != top_b and lift_a >= policy.tau:
                    detail.append({
                        "kind": "leak_attempt",
                        "position": pos,
                        "top_a": top_a,
                        "top_b": top_b,
                        "text_a": backend.decode([top_a]),
                        "text_b": backend.decode([top_b]),
                        "lift_a": round(lift_a, 6),
                        "kl_ab": round(kl_ab, 6),
                    })
            else:
                if top_a != top_b and lift_a >= policy.tau:
                    chosen = top_b
                    events.append(pos)
                    detail.append({
                        "kind": "leak_blocked",
                        "position": pos,
                        "top_a": top_a,
                        "top_b": top_b,
                        "text_a": backend.decode([top_a]),
                        "text_b": backend.decode([top_b]),
                        "lift_a": round(lift_a, 6),
                        "kl_ab": round(kl_ab, 6),
                    })
                else:
                    chosen = top_a
        else:
            chosen = top_b if policy.mode == "strict-public" else top_a

        if chosen in eos:
            finish = "stop"
            break
        toks.append(chosen)
        if pos + 1 < max_tokens:
            logits_a, logits_b = backend.step(chosen, chosen)

    return {
        "finish": finish,
        "tokens": toks,
        "text": backend.decode(toks),
        "events": events,
        "detail": detail,
        "ifc": {
            "mode": policy.mode,
            "tau": policy.tau,
            "max_lift": round(max_lift, 6),
            "max_kl": round(max_kl, 6),
            "leak_events": len(detail),
        },
    }


class MLXBackend:
    """Sequential Batch=1 MLX twin-track backend matching mlx_lm.generate.generate_step bit for bit."""

    def __init__(self, model: Any, tok: Any):
        self.model = model
        self.tok = tok
        self._cache_a: Any = None
        self._cache_b: Any = None

    @property
    def eos_ids(self) -> Sequence[int]:
        return list(self.tok.eos_token_ids)

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tok.decode(list(tokens))

    def _prefill_one(self, ids: Sequence[int], cache: Any) -> Any:
        import mlx.core as mx

        arr = list(ids)
        if len(arr) > 1:
            pos = 0
            while len(arr) - pos > 1:
                n = min(2048, (len(arr) - pos) - 1)
                self.model(mx.array([arr[pos : pos + n]]), cache=cache)
                mx.eval([c.state for c in cache])
                pos += n
            logits = self.model(mx.array([arr[pos:]]), cache=cache)[0, -1, :]
        else:
            logits = self.model(mx.array([arr]), cache=cache)[0, -1, :]
        mx.eval(logits, [c.state for c in cache])
        return logits

    def prefill(self, ids_a: Sequence[int], ids_b: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        self._cache_a = make_prompt_cache(self.model)
        la = self._prefill_one(ids_a, self._cache_a)
        self._cache_b = make_prompt_cache(self.model)
        lb = self._prefill_one(ids_b, self._cache_b)
        return np.array(la.astype(mx.float32)), np.array(lb.astype(mx.float32))

    def step(self, tok_a: int, tok_b: int) -> Tuple[np.ndarray, np.ndarray]:
        import mlx.core as mx

        la = self.model(mx.array([[tok_a]]), cache=self._cache_a)[0, -1, :]
        lb = self.model(mx.array([[tok_b]]), cache=self._cache_b)[0, -1, :]
        mx.eval(la, lb)
        return np.array(la.astype(mx.float32)), np.array(lb.astype(mx.float32))


def _row(prompt: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in prompt.items() if k not in ("ids", "clean_ids")}
    out.update(res)
    return out


def control(model_dir, prompts_file, out, control_keys, data_keys, max_tokens=report.MAX_TOKENS, isolate_b=None):
    """Run `kestrom ifc control` on PROMPTS.json and write OUT.jsonl."""
    from mlx_lm import load

    ckeys = [k.strip() for k in str(control_keys).split(",") if k.strip()]
    dkeys = [k.strip() for k in str(data_keys).split(",") if k.strip()]
    iso = None if isolate_b in (None, "", "none") else int(isolate_b)
    policy = JsonKeyRegionPolicy(ckeys, dkeys)
    model, tok = load(str(model_dir))
    backend = MLXBackend(model, tok)
    with open(out, "w", encoding="utf-8") as f:
        for x in report.read(prompts_file):
            res = decode_control(backend, x["ids"], x["clean_ids"], policy, int(max_tokens), iso)
            row = _row(x, res)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(x["id"], res["finish"], len(res["tokens"]), "events", len(res["events"]), "overrides", res["ifc"]["control_overrides"], flush=True)


def firewall(model_dir, prompts_file, out, mode="strict-public", tau=2.0, max_tokens=report.MAX_TOKENS):
    """Run `kestrom ifc firewall` on PROMPTS.json and write OUT.jsonl."""
    from mlx_lm import load

    policy = PublicRegionPolicy(mode=str(mode), tau=float(tau))
    model, tok = load(str(model_dir))
    backend = MLXBackend(model, tok)
    with open(out, "w", encoding="utf-8") as f:
        for x in report.read(prompts_file):
            res = decode_firewall(backend, x["ids"], x["clean_ids"], policy, int(max_tokens))
            row = _row(x, res)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(x["id"], res["finish"], len(res["tokens"]), "events", len(res["events"]), "leaks", res["ifc"]["leak_events"], flush=True)
