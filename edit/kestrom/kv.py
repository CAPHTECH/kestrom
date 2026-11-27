"""Compiled Fact-KV Routing (KV-A): compile per-fact system-prefix KV caches from a standalone rules DSL and inject the matching
cache at greedy decoding time (MLX, tier 1). When a prompt's `data` matches no fact (or exceeds MAX_MATCH), no cache is injected and the
kv line records no event (where no other line acts, decoding runs unchanged: `events=[]`); when one fact matches, only the prompt tokens
after the shared prefix (no knowledge text) are fed with the compiled cache, recording event `0`. When MAX_MATCH > 1 and 2..MAX_MATCH
entries match simultaneously, their texts are composed and prefilled once on demand and kept in an in-memory LRU cache. Check batch runs
with `kestrom-verify compare events`.
`kestrom generate` and `kestrom kv chat` refuse a bank whose `rules.dsl` or cache files are not the ones its `manifest.json` hashed, or
whose model fingerprint (hashes of the model's config, tokenizer, and `.safetensors` shard files) is not the model's.
When a rules file combines `kv:` with `in NAME: prompt FILE LAYER` conditions, `in ... prompt` is evaluated on the full prompt without the fact KV cache before decoding begins and passed as 0/1 `values` to the bank's rules.
  kestrom kv compile MODEL RULES.dsl FACTS.json OUT_DIR [PREV_DIR] [PREFIX_MODE]
      compile one `.safetensors` prefix KV file per entry in FACTS.json into OUT_DIR, together with `rules.dsl` and
      `manifest.json`. FACTS.json is either `{CLASS: TEXT, ...}` (for single-head rules) or a list of `{"key": ..., "when": CLASS |
      {HEAD: CLASS, ...}, "text": ...}`. With PREV_DIR (pass `-` or `none` when omitted before PREFIX_MODE), entries whose key and
      `prefix_ids` are unchanged for the same model fingerprint and prefix mode are copied without recomputing. PREFIX_MODE is `system`
      (default: system turn before user turn), `prefix-turn` (synthetic leading user/assistant exchange), or `auto` (`system` then
      `prefix-turn`). Compiles into a staging folder and renames atomically, so a failed run leaves no partial OUT_DIR. Use from
      `kestrom generate --rules` via a `kv: BANK_DIR [MAX_MATCH]` line.
  kestrom kv chat MODEL BANK_DIR [MAX_TOKENS] [MAX_MATCH] [TURNS.json] [OUT.jsonl] [PROBES.dsl]
      multi-turn chat session using a compiled KV bank. Every turn evaluates the full turn history (`steps`) through the bank's
      rules so `prev` lines and registers resolve follow-ups without extra model calls. When matched keys are unchanged and the chat
      template extends the cached tokens as a prefix, the live KV cache is reused incrementally (zero re-prefill of earlier turns); on
      topic switches it swaps in the new cache and re-prefills the conversation; when multiple entries match (up to MAX_MATCH, 3 unless
      given), they are composed via the in-memory LRU cache. With TURNS.json (`["turn1", ...]` or `[{"user": ..., "values": ..., "data": ..., "reset": ...}]`; pass `-` or `none` when passing PROBES.dsl interactively),
      writes one JSONL row per turn to OUT.jsonl; without TURNS.json, runs an interactive session (/reset, /status, /quit).
      With PROBES.dsl (a rules file containing `in NAME: prompt FILE LAYER` lines, e.g. from `kestrom bridge map`), each user turn
      evaluates those prompt conditions on the raw turn before matching and passes them as 0/1 `values` to the bank's rules (the turn's
      own `values` win).
  kestrom kv from-table TABLE_FILE OUT_DIR
      generate `rules.dsl`, `facts.json`, `generate.dsl`, `regression.json`, and deterministic `from_table_report.json` from a CSV,
      TSV, YAML, JSON, or Markdown pipe table (`key`/`id`, `triggers`/`keywords`, `fact`/`text`, optional `prompt` and `outside`), and
      verify with `rules.answers` that every positive sample hits only its own key and every outside sample matches nothing (exits 1 on FAIL).
  kestrom kv verify-chat MODEL BANK_DIR TURNS.json REPORT.json [MAX_TOKENS] [MAX_MATCH] [PROBES.dsl]
      mechanically check multi-turn non-interference for `kestrom kv chat`: runs TURNS.json through `KVSession`, re-runs plain greedy
      decoding twice on every turn where `kv` is `null` using identical prompt ids, and writes REPORT.json checking determinism (U0) and
      exact bit-identity (`events=[]` and matching tokens/finish, G2) on all `kv=null` turns (exits 1 on FAIL or zero `kv=null` turns).
  kestrom kv check-template MODEL [OUT.json]
      inspect whether MODEL's tokenizer isolates a prefix cleanly under `system` and/or `prefix-turn`, print `recommended`, and
      optionally write deterministic `OUT.json` (exits 1 if neither mode works)."""
from collections import OrderedDict
import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from kestrom_verify import report

from . import rules

FORMAT = "kestrom-kv-v2"
PROBES = ("Hello.", "12345.")
PREFIX_MODES = ("system", "prefix-turn", "auto")
META_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors.index.json")
DEFAULT_OUTSIDE_PROMPTS = (
    "Compute 17 + 28.",
    "What is the boiling point of water in Celsius?",
    "17足す28はいくつですか？",
    "今日の天気を教えてください。",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: Path, sha: str, where):
    if not path.is_file() or sha256_file(path) != sha:
        raise SystemExit(f"{where}: missing or changed file {path.name} (its hash is not the one in manifest.json)")


def entry_filename(key: str) -> str:
    """Deterministic cache filename inside the KV bank folder."""
    return f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}.safetensors"


def _canonical_when(when):
    if isinstance(when, str):
        return ("str", when)
    if isinstance(when, dict) and when and all(isinstance(k, str) and isinstance(v, str) and v for k, v in when.items()):
        return tuple(sorted(when.items()))
    return None


def parse_facts(raw, R):
    """Normalize FACTS.json ({CLASS: TEXT} or list of {key, when, text}) and validate it against standalone rules R."""
    if isinstance(raw, dict):
        items = [{"key": k, "when": k, "text": v} for k, v in sorted(raw.items())]
    elif isinstance(raw, list):
        items = raw
    else:
        raise SystemExit("FACTS.json must be a JSON object {CLASS: TEXT} or a list of {key, when, text}")
    if not items:
        raise SystemExit("FACTS.json must not be empty")
    single_head = list(R["heads"]) == [""]
    keys, whens, out = set(), set(), []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or not {"key", "when", "text"}.issubset(item):
            raise SystemExit(f"FACTS.json entry {i}: requires keys 'key', 'when', 'text'")
        key, when, text = item["key"], item["when"], item["text"]
        if not isinstance(key, str) or not key.strip():
            raise SystemExit(f"FACTS.json entry {i}: 'key' must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise SystemExit(f"FACTS.json entry {i} ({key!r}): 'text' must be a non-empty string")
        if key in keys:
            raise SystemExit(f"FACTS.json: duplicate key {key!r}")
        if single_head:
            if not isinstance(when, str) or when not in R["tokens"]:
                raise SystemExit(f"FACTS.json entry {key!r}: 'when' must name a class of the rules: {when!r}")
        else:
            if not isinstance(when, dict) or not when:
                raise SystemExit(f"FACTS.json entry {key!r}: multi-head rules require 'when' dict {{head: class}}")
            for head, cls in when.items():
                if not isinstance(head, str) or head not in R["heads"]:
                    raise SystemExit(f"FACTS.json entry {key!r}: unknown rule head {head!r}")
                tok_name = f"{head}/{cls}" if head else cls
                if not isinstance(cls, str) or tok_name not in R["tokens"]:
                    raise SystemExit(f"FACTS.json entry {key!r}: unknown class {cls!r} for head {head!r}")
        canon = _canonical_when(when)
        if canon is None or canon in whens:
            raise SystemExit(f"FACTS.json entry {key!r}: invalid or duplicate 'when' {when!r}")
        keys.add(key)
        whens.add(canon)
        out.append({"key": key, "when": when, "text": text})
    return out


def model_fingerprint(model_dir: Path) -> dict:
    """Deterministic fingerprint of MODEL (metadata hashes + safetensors shard sizes and SHA-256 hashes)."""
    model_dir = Path(model_dir)
    meta = {name: sha256_file(p) for name in META_FILES if (p := model_dir / name).is_file()}
    shards = sorted([p.name, p.stat().st_size, sha256_file(p)] for p in model_dir.glob("*.safetensors") if p.is_file())
    return {"meta": meta, "shards": shards}


def _chat_messages_ids(tok, messages: list[dict], enable_thinking: bool = False) -> list[int]:
    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return [int(x) for x in ids]


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _chat_ids(tok, user_text: str, system_text: str | None, mode: str = "system") -> list[int]:
    if mode == "system":
        messages = ([{"role": "system", "content": system_text}] if system_text is not None else []) + [{"role": "user", "content": user_text}]
    elif mode == "prefix-turn":
        messages = (
            [{"role": "user", "content": system_text}, {"role": "assistant", "content": "Understood."}]
            if system_text is not None
            else []
        ) + [{"role": "user", "content": user_text}]
    else:
        raise ValueError(f"unsupported prefix_mode {mode!r} (use 'system', 'prefix-turn', or 'auto')")
    return _chat_messages_ids(tok, messages, enable_thinking=False)


def _probe_prefix_boundary(tok, fact_text: str, mode: str = "system") -> tuple[list[int], list[int]]:
    """Probe whether chat template isolates fact_text as a clean prefix under `mode` (`system` or `prefix-turn`)."""
    if mode not in ("system", "prefix-turn"):
        raise ValueError(f"unsupported prefix_mode {mode!r} (expected 'system' or 'prefix-turn')")
    found = None
    for probe in PROBES:
        u = _chat_ids(tok, probe, None, mode=mode)
        try:
            s = _chat_ids(tok, probe, fact_text, mode=mode)
        except Exception as exc:
            if mode == "system":
                raise ValueError(f"system role unsupported by chat template: {exc}") from exc
            raise ValueError(f"{mode} template formatting failed: {exc}") from exc
        hit = None
        for k in range(len(u)):
            tail = len(u) - k
            if len(s) > len(u) and u[:k] == s[:k] and u[k:] == s[len(s) - tail:]:
                hit = (u[:k], s[:len(s) - tail])
                break
        if hit is None:
            raise ValueError(f"chat template does not place the {mode} turn as a clean prefix before the user turn")
        if found is None:
            found = hit
        elif found != hit:
            raise ValueError(f"chat template {mode} boundary varies across probes")
    return found


def prefix_boundary(tok, fact_text: str, mode: str = "system") -> tuple[list[int], list[int]]:
    """Verify the chat template places the fact prefix cleanly before the user turn, returning (strip_prefix, system_prefix_ids)."""
    try:
        return _probe_prefix_boundary(tok, fact_text, mode=mode)
    except ValueError as exc:
        hint = " (try PREFIX_MODE=prefix-turn or auto)" if mode == "system" else ""
        raise SystemExit(f"{exc}{hint}") from exc


def resolve_prefix_boundary(tok, fact_text: str, mode: str = "system") -> tuple[list[int], list[int], str]:
    """Resolve (strip_prefix, prefix_ids, resolved_mode) for `system`, `prefix-turn`, or `auto`."""
    if mode in ("system", "prefix-turn"):
        sp, pfx = prefix_boundary(tok, fact_text, mode=mode)
        return sp, pfx, mode
    if mode == "auto":
        try:
            sp, pfx = _probe_prefix_boundary(tok, fact_text, mode="system")
            return sp, pfx, "system"
        except ValueError as e_sys:
            try:
                sp, pfx = _probe_prefix_boundary(tok, fact_text, mode="prefix-turn")
                return sp, pfx, "prefix-turn"
            except ValueError as e_pt:
                raise SystemExit(
                    f"chat template incompatible with both 'system' ({e_sys}) and 'prefix-turn' ({e_pt})"
                ) from e_pt
    raise SystemExit(f"unsupported prefix_mode {mode!r} (expected 'system', 'prefix-turn', or 'auto')")


def inspect_template(tok, fact_text: str = "Fact.") -> dict:
    """Inspect whether `tok` supports clean prefix isolation under `system` and `prefix-turn`."""
    out = {}
    for label, mode in (("system", "system"), ("prefix_turn", "prefix-turn")):
        try:
            sp, pfx = _probe_prefix_boundary(tok, fact_text, mode=mode)
            out[label] = {"ok": True, "strip_prefix": sp, "prefix_len": len(pfx), "error": None}
        except ValueError as exc:
            out[label] = {"ok": False, "strip_prefix": None, "prefix_len": None, "error": str(exc)}
    rec = "system" if out["system"]["ok"] else ("prefix-turn" if out["prefix_turn"]["ok"] else None)
    return {**out, "recommended_mode": rec}


def check_template(model_dir, out_file=None):
    """CLI handler: inspect MODEL's chat-template prefix isolation under `system` and `prefix-turn`."""
    out_file = _opt_arg(out_file)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    res = inspect_template(tok)
    rep = {"schema": "kestrom-kv-template-v1", "model": Path(model_dir).name, **res}
    for label, key in (("system", "system"), ("prefix-turn", "prefix_turn")):
        item = res[key]
        detail = f"strip_prefix={item['strip_prefix']} prefix_len={item['prefix_len']}" if item["ok"] else item["error"]
        print(f"{label}: {'PASS' if item['ok'] else 'FAIL'} ({detail})", flush=True)
    print(f"recommended: {res['recommended_mode'] or 'none'}", flush=True)
    if out_file is not None:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_file, rep)
    if res["recommended_mode"] is None:
        raise SystemExit(1)


def _compact_cache(cache):
    """Trim preallocated trailing empty slots from linear KV caches before saving."""
    for c in cache:
        if hasattr(c, "keys") and hasattr(c, "values") and hasattr(c, "offset") and c.keys is not None:
            if 0 < c.offset < c.keys.shape[2] and getattr(c, "_idx", c.offset) == c.offset:
                c.keys = c.keys[..., :c.offset, :]
                c.values = c.values[..., :c.offset, :]


def prefill_prefix(model, prefix_ids: list[int]):
    """Prefill prefix_ids into a fresh prompt cache and return the evaluated cache list."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    pos = 0
    while pos < len(prefix_ids):
        chunk = prefix_ids[pos:pos + 2048]
        _ = model(mx.array([chunk]), cache=cache)
        mx.eval([c.state for c in cache])
        pos += len(chunk)
    _compact_cache(cache)
    mx.eval([c.state for c in cache])
    return cache


def compile_prefix(model, prefix_ids: list[int], out_file: Path, key: str):
    """Prefill prefix_ids into a fresh prompt cache and save it as a safetensors cache file."""
    from mlx_lm.models.cache import save_prompt_cache

    cache = prefill_prefix(model, prefix_ids)
    save_prompt_cache(str(out_file), cache, metadata={"kestrom_kv_key": key, "prefix_len": str(len(prefix_ids))})


def _clone_tree(x):
    if isinstance(x, list):
        return [_clone_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_tree(v) for v in x)
    return x


def clone_cache(cache):
    """Shallow-clone cache layer objects and container structures so advancing one copy never mutates another."""
    out = []
    for c in cache:
        cls = type(c)
        if hasattr(c, "meta_state"):
            out.append(cls.from_state(_clone_tree(c.state), _clone_tree(c.meta_state)))
        else:
            out.append(cls.from_state(_clone_tree(c.state)))
    return out


class ComposeCache:
    """In-memory LRU cache for multi-entry composed prefix KV caches."""

    def __init__(self, maxsize: int = 16):
        self.maxsize = max(0, int(maxsize))
        self._cache: OrderedDict[tuple[str, ...], list] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_or_build(self, model, tok, bank: dict, hits: list[dict]):
        """Return a fresh prompt cache for hits (loading 1-hit from disk or composing >=2 hits via LRU)."""
        if len(hits) == 1:
            return load_cache_file(str(Path(bank["dir"]) / hits[0]["file"]))
        combo_key = tuple(e["key"] for e in hits)
        if combo_key in self._cache:
            self._cache.move_to_end(combo_key)
            self.hits += 1
            return clone_cache(self._cache[combo_key])
        self.misses += 1
        combined_text = "\n\n".join(e["text"] for e in hits)
        sp, pfx = prefix_boundary(tok, combined_text, mode=bank.get("prefix_mode", "system"))
        if sp != bank["strip_prefix"]:
            raise SystemExit(f"composed prefix strip_prefix {sp} differs from bank strip_prefix {bank['strip_prefix']}")
        snap = prefill_prefix(model, pfx)
        if self.maxsize > 0:
            self._cache[combo_key] = snap
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return clone_cache(snap)


def compile(model_dir, rules_file, facts_file, out_dir, prev_dir=None, prefix_mode="system"):
    """Compile RULES.dsl and FACTS.json into a deterministic KV bank in OUT_DIR."""
    prev_dir = _opt_arg(prev_dir)
    prefix_mode = (prefix_mode or "system").strip()
    if prefix_mode not in PREFIX_MODES:
        raise SystemExit(f"unsupported PREFIX_MODE {prefix_mode!r} (expected one of {', '.join(PREFIX_MODES)})")
    R = rules.standalone(rules_file)
    facts = parse_facts(report.read(facts_file), R)
    out_path = Path(out_dir)
    if out_path.is_file() or (out_path.exists() and any(out_path.iterdir())):
        raise SystemExit(f"{out_dir} already exists and is not empty")
    model_path = Path(model_dir)
    fp = model_fingerprint(model_path)

    prev_by_prefix = {}
    prev_mode = "system"
    if prev_dir:
        prev_path = Path(prev_dir)
        manifest_path = prev_path / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"{prev_dir}: missing manifest.json")
        pm = json.loads(manifest_path.read_text(encoding="utf-8"))
        if pm.get("format") != FORMAT:
            raise SystemExit(f"{prev_dir}: unexpected KV bank format {pm.get('format')!r}")
        if pm.get("fingerprint") != fp:
            raise SystemExit(f"{prev_dir}: model fingerprint differs from {model_dir}")
        prev_mode = pm.get("prefix_mode", "system")
        for e in pm.get("entries", []):
            prev_by_prefix[(e["key"], tuple(e["prefix_ids"]))] = (prev_path / e["file"], e["cache_sha256"])

    from mlx_lm import load

    model, tok = load(str(model_dir))
    strip_prefix = None
    resolved_mode = None
    tokenized = []
    for entry in facts:
        if resolved_mode is None:
            sp, pfx, resolved_mode = resolve_prefix_boundary(tok, entry["text"], mode=prefix_mode)
            strip_prefix = sp
        else:
            sp, pfx = prefix_boundary(tok, entry["text"], mode=resolved_mode)
            if sp != strip_prefix:
                raise SystemExit(f"entry {entry['key']!r}: strip_prefix {sp} differs from {strip_prefix}")
        tokenized.append((entry, pfx))
    if prev_dir and resolved_mode != prev_mode:
        prev_by_prefix.clear()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(prefix=f".{out_path.name}.tmp.", dir=out_path.parent))
    try:
        shutil.copyfile(rules_file, tmp_path / "rules.dsl")

        compiled, reused, manifest_entries = 0, 0, []
        for entry, pfx in tokenized:
            fname = entry_filename(entry["key"])
            dst = tmp_path / fname
            pkey = (entry["key"], tuple(pfx))
            if pkey in prev_by_prefix:
                src, prev_sha = prev_by_prefix[pkey]
                check_file(src, prev_sha, prev_dir)
                shutil.copyfile(src, dst)
                reused += 1
            else:
                compile_prefix(model, pfx, dst, entry["key"])
                compiled += 1
            manifest_entries.append({
                "key": entry["key"],
                "when": entry["when"],
                "text": entry["text"],
                "text_sha256": sha256_bytes(entry["text"].encode("utf-8")),
                "prefix_ids": pfx,
                "file": fname,
                "cache_sha256": sha256_file(dst),
                "nbytes": dst.stat().st_size,
            })

        manifest = {
            "format": FORMAT,
            "fingerprint": fp,
            "prefix_mode": resolved_mode or "system",
            "rules_sha256": sha256_file(tmp_path / "rules.dsl"),
            "strip_prefix": strip_prefix or [],
            "entries": manifest_entries,
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if out_path.exists():
            out_path.rmdir()
        tmp_path.replace(out_path)
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    mean_kb = sum(e["nbytes"] for e in manifest_entries) / len(manifest_entries) / 1024.0
    print("compiled", compiled, "reused", reused, "total", len(manifest_entries), "mean_kb", round(mean_kb, 1), flush=True)


def check_model(bank: dict, model_dir):
    """Refuse a bank compiled for another model than MODEL (its fingerprint)."""
    if bank["fingerprint"] != model_fingerprint(model_dir):
        raise SystemExit(f"{bank['dir']} was compiled for another model than {model_dir} (the model fingerprint differs)")


def load_bank(bank_dir, tok=None) -> dict:
    """Load a compiled KV bank directory, refusing it where rules.dsl or a cache file is not the one manifest.json hashed."""
    bpath = Path(bank_dir)
    mpath = bpath / "manifest.json"
    rpath = bpath / "rules.dsl"
    if not mpath.is_file() or not rpath.is_file():
        raise SystemExit(f"{bank_dir}: missing manifest.json or rules.dsl")
    m = json.loads(mpath.read_text(encoding="utf-8"))
    if m.get("format") != FORMAT:
        raise SystemExit(f"{bank_dir}: unsupported KV bank format {m.get('format')!r}")
    check_file(rpath, m.get("rules_sha256"), bank_dir)
    R = rules.standalone(str(rpath))
    entries = m.get("entries", [])
    for e in entries:
        check_file(bpath / e["file"], e.get("cache_sha256"), bank_dir)
    sp = m.get("strip_prefix", [])
    if tok is not None and len(sp) == 1 and getattr(tok, "bos_token_id", None) is not None and sp[0] != tok.bos_token_id:
        raise SystemExit(f"{bank_dir}: bank strip_prefix {sp} does not match tokenizer bos_token_id {tok.bos_token_id}")
    return {"dir": str(bpath), "rules": R, "entries": entries, "strip_prefix": sp, "prefix_mode": m.get("prefix_mode", "system"), "fingerprint": m.get("fingerprint", {})}


def match_entries(bank: dict, data: dict, max_match: int = 1) -> list[dict]:
    """Evaluate the bank's standalone rules on `data`; return matching entries iff 1 <= count <= max_match."""
    cls = next(rules.answers(bank["rules"], [data]))[0]["class"]
    if cls is None:
        return []
    hits = []
    for e in bank["entries"]:
        w = e["when"]
        if isinstance(w, str) and cls == w:
            hits.append(e)
        elif isinstance(w, dict) and isinstance(cls, dict) and all(cls.get(h) == v for h, v in w.items()):
            hits.append(e)
    return hits if 1 <= len(hits) <= int(max_match) else []


def format_kv_key(hits: list[dict]):
    if not hits:
        return None
    return hits[0]["key"] if len(hits) == 1 else [e["key"] for e in hits]


def load_cache_file(path: str):
    """Load a compiled prompt cache from a `.safetensors` file."""
    from mlx_lm.models.cache import load_prompt_cache

    return load_prompt_cache(str(path))


def _mlx_decode(model, tok, feed_ids: list[int], max_tokens: int, prompt_cache) -> tuple[list[int], str, list[int]]:
    """Greedy decode: (tokens, finish, the generated ids prompt_cache now holds). generate_step feeds each token into the cache
    before yielding it, so the cache holds every yielded token, the end token included."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    toks, finish, held = [], "length", []
    for t, _ in generate_step(mx.array(feed_ids), model, max_tokens=max_tokens, prompt_cache=prompt_cache):
        ti = int(t)
        held.append(ti)
        if ti in tok.eos_token_ids:
            finish = "stop"
            break
        toks.append(ti)
    return toks, finish, held


def load_prompt_probes(probes_file: str) -> dict:
    """Load `in NAME: prompt FILE LAYER` conditions from PROBES.dsl (allows `kv:` lines, refuses units/actions/other inputs)."""
    text = Path(probes_file).read_text(encoding="utf-8")
    raw_lines = rules.lines(text)
    if any(k not in ("in", "kv") for k, _, _ in raw_lines):
        raise SystemExit(f"{probes_file}: PROBES.dsl allows only `in NAME: prompt FILE LAYER` (and optional `kv:`) lines")
    R = rules.compile_rules(text)
    for name, spec in R["declared"].items():
        if spec[0] != "prompt":
            raise SystemExit(f"{probes_file}: unsupported `in` kind {spec[0]!r} for {name!r} (only `prompt` is allowed)")
    if not R["declared"]:
        raise SystemExit(f"{probes_file}: needs at least one `in NAME: prompt FILE LAYER` line")
    loaded = rules.prompt_conditions(R)
    return {name: (loaded[path], layer) for name, (_, path, layer) in R["declared"].items()}


class KVSession:
    """Multi-turn session with stateful `steps`/`prev` rule evaluation, incremental cache reuse, and multi-entry composition."""

    def __init__(
        self,
        model,
        tok,
        bank: dict,
        max_tokens: int = report.MAX_TOKENS,
        max_match: int = 3,
        compose_cache: ComposeCache | None = None,
        enable_thinking: bool = False,
        decode_fn=None,
        make_cache_fn=None,
        prompt_probes: dict | None = None,
        probe_fn=None,
    ):
        self.model = model
        self.tok = tok
        self.bank = bank
        self.max_tokens = int(max_tokens)
        self.max_match = int(max_match)
        self.compose_cache = compose_cache if compose_cache is not None else ComposeCache()
        self.enable_thinking = enable_thinking
        self._decode = decode_fn if decode_fn is not None else (lambda feed, cache: _mlx_decode(model, tok, feed, self.max_tokens, cache))
        if make_cache_fn is not None:
            self._make_cache = make_cache_fn
        else:
            from mlx_lm.models.cache import make_prompt_cache
            self._make_cache = make_prompt_cache
        self.prompt_probes = dict(prompt_probes) if prompt_probes else {}
        if probe_fn is not None:
            self._probe = probe_fn
        elif self.prompt_probes:
            from . import steer
            self._probe = lambda ids, npz, layer: steer.fires(self.model, ids, npz, layer)
        else:
            self._probe = None
        self.messages: list[dict] = []
        self.steps: list[dict] = []
        self.active_keys: tuple[str, ...] | None = None
        self.live_cache = None
        self.cached_ids: list[int] = []
        self.last_prompt_ids: list[int] = []

    def reset(self):
        """Clear conversation history and live KV state while keeping compiled multi-entry snapshots warm."""
        self.messages.clear()
        self.steps.clear()
        self.active_keys = None
        self.live_cache = None
        self.cached_ids.clear()
        self.last_prompt_ids.clear()

    def turn(self, user_text: str, values: dict | None = None, data: dict | None = None) -> dict:
        """Advance one user turn, keeping active KV state incrementally or switching/composing on topic changes."""
        step = dict(data) if data is not None else {"text": user_text}
        if values is not None:
            step["values"] = values
        probe_vals = None
        if self.prompt_probes:
            turn_ids = _chat_messages_ids(self.tok, [{"role": "user", "content": user_text}], enable_thinking=self.enable_thinking)
            probe_vals = {name: int(bool(self._probe(turn_ids, npz, layer))) for name, (npz, layer) in self.prompt_probes.items()}
            step = rules.with_prompt_values(step, probe_vals)
        self.steps.append(step)
        hits = match_entries(self.bank, {"steps": list(self.steps)}, self.max_match)
        self.messages.append({"role": "user", "content": user_text})
        full_ids = _chat_messages_ids(self.tok, self.messages, enable_thinking=self.enable_thinking)
        self.last_prompt_ids = list(full_ids)
        target_keys = tuple(e["key"] for e in hits) if hits else None
        sp = self.bank["strip_prefix"]
        if hits and full_ids[:len(sp)] != sp:
            raise SystemExit(f"turn {len(self.steps) - 1}: prompt ids do not start with bank strip_prefix {sp}")

        if (
            self.live_cache is not None
            and target_keys == self.active_keys
            and self.cached_ids
            and len(full_ids) > len(self.cached_ids)
            and full_ids[:len(self.cached_ids)] == self.cached_ids
        ):
            feed_ids = full_ids[len(self.cached_ids):]
            prompt_cache = self.live_cache
            cache_action = "reuse"
        elif hits:
            prompt_cache = self.compose_cache.get_or_build(self.model, self.tok, self.bank, hits)
            feed_ids = full_ids[len(sp):]
            if self.cached_ids and target_keys == self.active_keys:
                cache_action = "rebuild"
            elif len(hits) > 1:
                cache_action = "compose"
            else:
                cache_action = "init" if not self.cached_ids else "switch"
        else:
            prompt_cache = self._make_cache(self.model)
            feed_ids = full_ids
            if self.cached_ids and self.active_keys is None:
                cache_action = "rebuild"
            else:
                cache_action = "plain" if self.active_keys is None else "switch"

        toks, finish, held = self._decode(feed_ids, prompt_cache)
        self.live_cache = prompt_cache
        self.active_keys = target_keys
        self.cached_ids = full_ids + held
        reply = self.tok.decode(toks)
        self.messages.append({"role": "assistant", "content": reply})
        out = {
            "turn": len(self.steps) - 1,
            "user": user_text,
            "kv": format_kv_key(hits),
            "cache_action": cache_action,
            "prefill_tokens": len(feed_ids),
            "finish": finish,
            "tokens": toks,
            "text": reply,
            "events": [0] if hits else [],
        }
        if probe_vals is not None:
            out["probes"] = probe_vals
        return out


def _opt_arg(v):
    return None if v is None or str(v).lower() in ("", "-", "none") else v


def chat(model_dir, bank_dir, max_tokens=report.MAX_TOKENS, max_match=3, turns_file=None, out_file=None, probes_file=None):
    """Run an interactive or scripted multi-turn session (`kestrom kv chat`) over a compiled KV bank."""
    max_tokens, max_match = int(max_tokens), int(max_match)
    if max_tokens < 1 or max_match < 1:
        raise SystemExit("MAX_TOKENS and MAX_MATCH must be at least 1")
    turns_file, out_file, probes_file = _opt_arg(turns_file), _opt_arg(out_file), _opt_arg(probes_file)
    if (turns_file is None) != (out_file is None):
        raise SystemExit("give TURNS.json and OUT.jsonl together, or neither (use '-' for both when passing PROBES.dsl interactively)")
    prompt_probes = load_prompt_probes(probes_file) if probes_file is not None else None
    check_model(load_bank(bank_dir), model_dir)
    turns = report.read(turns_file) if turns_file is not None else None

    from mlx_lm import load

    model, tok = load(str(model_dir))
    bank = load_bank(bank_dir, tok)
    session = KVSession(model, tok, bank, max_tokens=max_tokens, max_match=max_match, prompt_probes=prompt_probes)

    if turns is not None:
        f = open(out_file, "w", encoding="utf-8")  # one line per turn as it is done (a long run keeps what it finished)
        for entry in turns:
            if isinstance(entry, str):
                res = session.turn(entry)
            elif isinstance(entry, dict):
                if entry.get("reset"):
                    session.reset()
                text = entry.get("user", entry.get("text"))
                if not isinstance(text, str):
                    raise SystemExit(f"TURNS.json entry {entry!r}: requires 'user' or 'text'")
                res = session.turn(text, values=entry.get("values"), data=entry.get("data"))
                for k, v in entry.items():
                    if k not in ("user", "text", "values", "data", "reset") and k not in res:
                        res[k] = v
            else:
                raise SystemExit(f"TURNS.json entry must be a string or object: {entry!r}")
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"turn {res['turn']} kv={res['kv']} action={res['cache_action']} "
                f"prefill={res['prefill_tokens']} tokens={len(res['tokens'])}",
                flush=True,
            )
        f.close()
        return

    print("kestrom kv chat (/reset, /status, /quit)", flush=True)
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            session.reset()
            print("[session reset]", flush=True)
            continue
        if user == "/status":
            print(
                f"[turns={len(session.steps)} active={list(session.active_keys) if session.active_keys else None} "
                f"compose_hits={session.compose_cache.hits} compose_misses={session.compose_cache.misses}]",
                flush=True,
            )
            continue
        res = session.turn(user)
        print(f"[kv={res['kv']} action={res['cache_action']} prefill={res['prefill_tokens']}]", flush=True)
        print(res["text"], flush=True)


def _strip_yaml_comment(line: str) -> str:
    in_sq = False
    in_dq = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq and (i == 0 or line[i - 1] != "\\"):
            in_dq = not in_dq
        elif ch == "#" and not in_sq and not in_dq and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _unquote_yaml_scalar(s: str):
    s = s.strip()
    if not s:
        return ""
    if s[0] == '"' and s[-1] == '"' and len(s) >= 2:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s[1:-1]
    if s[0] == "'" and s[-1] == "'" and len(s) >= 2:
        return s[1:-1].replace("''", "'")
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_unquote_yaml_scalar(part) for part in next(csv.reader([inner]))]
    return s


def _parse_yaml_rows_stdlib(text: str) -> list[dict]:
    """Minimal dependency-free parser for flat YAML list-of-mappings spec tables."""
    items, cur, list_key = [], None, None
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stripped.startswith("- ") and (indent == 0 or cur is None or list_key is None):
            rest = stripped[2:].strip()
            if ":" in rest:
                cur = {}
                items.append(cur)
                list_key = None
                k, _, v = rest.partition(":")
                k, v = k.strip(), v.strip()
                if v:
                    cur[k] = _unquote_yaml_scalar(v)
                else:
                    cur[k] = []
                    list_key = k
                continue
        if stripped == "-" and cur is None:
            cur = {}
            items.append(cur)
            list_key = None
            continue
        if stripped.startswith("- ") and cur is not None and list_key is not None:
            if isinstance(cur.get(list_key), list):
                cur[list_key].append(_unquote_yaml_scalar(stripped[2:]))
                continue
        if cur is None:
            cur = {}
            items.append(cur)
        if ":" in stripped:
            k, _, v = stripped.partition(":")
            k, v = k.strip(), v.strip()
            if v:
                cur[k] = _unquote_yaml_scalar(v)
                list_key = None
            else:
                cur[k] = []
                list_key = k
    return items


def _read_markdown_table(text: str) -> list[dict]:
    table_lines = [ln.strip() for ln in text.splitlines() if "|" in ln and ln.strip().startswith("|")]
    if len(table_lines) < 3:
        raise SystemExit("Markdown file must contain a pipe table with header, separator, and at least one data row")

    def split_row(ln: str) -> list[str]:
        parts = re.split(r"(?<!\\)\|", ln.strip().strip("|"))
        return [p.replace(r"\|", "|").strip() for p in parts]

    headers = split_row(table_lines[0])
    sep_cells = split_row(table_lines[1])
    if not headers or not all(re.fullmatch(r":?-{1,}:?", c or "") for c in sep_cells):
        raise SystemExit("Markdown table requires a standard separator row (|---|---|)")
    rows = []
    for ln in table_lines[2:]:
        cells = split_row(ln)
        if all(c == "" for c in cells):
            continue
        row = {h: (cells[i] if i < len(cells) else "") for i, h in enumerate(headers) if h}
        rows.append(row)
    return rows


def _read_spec_table(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"spec table not found: {path}")
    suf = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suf in (".csv", ".tsv"):
        delim = "\t" if suf == ".tsv" else ","
        return [dict(r) for r in csv.DictReader(text.splitlines(), delimiter=delim)]
    if suf == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = obj.get("rows", obj.get("entries"))
        if not isinstance(obj, list) or not all(isinstance(r, dict) for r in obj):
            raise SystemExit(f"{path}: JSON spec table must be a list of objects (or {{'rows': [...]}})")
        return obj
    if suf in (".md", ".markdown"):
        return _read_markdown_table(text)
    if suf in (".yaml", ".yml"):
        try:
            import yaml

            obj = yaml.safe_load(text)
            if isinstance(obj, dict):
                obj = obj.get("rows", obj.get("entries"))
        except ImportError:
            obj = _parse_yaml_rows_stdlib(text)
        if not isinstance(obj, list) or not all(isinstance(r, dict) for r in obj):
            raise SystemExit(f"{path}: YAML spec table must decode to a list of mappings")
        return obj
    raise SystemExit(f"{path}: unsupported table extension {suf!r} (supported: .csv, .tsv, .yaml, .yml, .json, .md)")


def _pick_col(row_ci: dict, aliases: tuple[str, ...]):
    for name in aliases:
        if name in row_ci and row_ci[name] is not None:
            val = row_ci[name]
            if isinstance(val, str):
                if val.strip():
                    return val.strip()
            elif val:
                return val
    return None


def _split_triggers(raw) -> list[str]:
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        sep = "|" if "|" in raw else ","
        items = [p.strip() for p in raw.split(sep) if p.strip()]
    else:
        items = []
    return items


def _compile_trigger_term(term: str) -> str:
    t = term.strip()
    if len(t) >= 2 and t.startswith("/") and t.endswith("/"):
        inner = t[1:-1]
        if not inner:
            raise SystemExit("empty regex trigger '//'")
        try:
            re.compile(inner)
        except re.error as exc:
            raise SystemExit(f"invalid regex trigger {t!r}: {exc}") from exc
        return inner
    esc = re.escape(t)
    if re.fullmatch(r"[\x00-\x7f]+", t) and re.search(r"[A-Za-z0-9]", t):
        return rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])"
    return esc


def from_table(table_file, out_dir):
    """Generate `rules.dsl`, `facts.json`, `generate.dsl`, `regression.json`, and `from_table_report.json` from a spec table."""
    src_path = Path(table_file)
    raw_rows = _read_spec_table(src_path)
    if not raw_rows:
        raise SystemExit(f"{src_path}: spec table has no rows")

    parsed_entries = []
    seen_keys = set()
    custom_outside = []
    for idx, row in enumerate(raw_rows):
        row_ci = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
        key = _pick_col(row_ci, ("key", "id", "name"))
        trig_raw = _pick_col(row_ci, ("triggers", "trigger", "keywords", "keyword", "pattern", "match"))
        fact = _pick_col(row_ci, ("fact", "text", "value", "content"))
        sample_prompt = _pick_col(row_ci, ("prompt", "sample_prompt", "question"))
        out_raw = _pick_col(row_ci, ("outside", "negative"))
        if not isinstance(key, str) or not key.strip():
            raise SystemExit(f"{src_path} row {idx}: missing 'key'")
        key = key.strip()
        if key in seen_keys:
            raise SystemExit(f"{src_path}: duplicate key {key!r}")
        seen_keys.add(key)
        triggers = _split_triggers(trig_raw)
        if not triggers:
            raise SystemExit(f"{src_path} row {idx} ({key!r}): missing 'triggers'")
        if not isinstance(fact, str) or not fact.strip():
            raise SystemExit(f"{src_path} row {idx} ({key!r}): missing 'fact'")
        if sample_prompt is None:
            plain_trigs = [t for t in triggers if not (t.startswith("/") and t.endswith("/"))]
            if not plain_trigs:
                raise SystemExit(f"{src_path} row {idx} ({key!r}): regex-only triggers require an explicit 'prompt' column")
            sample_prompt = plain_trigs[0]
        if out_raw is not None:
            if isinstance(out_raw, list):
                custom_outside.extend(str(x).strip() for x in out_raw if str(x).strip())
            else:
                sep = ";" if ";" in str(out_raw) else "|"
                custom_outside.extend(p.strip() for p in str(out_raw).split(sep) if p.strip())
        parsed_entries.append({"idx": idx, "key": key, "triggers": triggers, "fact": fact.strip(), "prompt": str(sample_prompt).strip()})

    dsl_lines = [f"# Auto-generated by `kestrom kv from-table` from {src_path.name}"]
    facts = []
    regression = []
    for i, item in enumerate(parsed_entries):
        head = f"h{i}"
        pat = "|".join(_compile_trigger_term(t) for t in item["triggers"])
        dsl_lines.extend([
            f"in m_{i}: match {json.dumps(pat, ensure_ascii=False)}",
            f"h1 u_{i}: any(m_{i})",
            f"h2 v_{i}: any(u_{i})",
            f"out {head}/hit: 1*{f'v_{i}'}",
        ])
        facts.append({"key": item["key"], "when": {head: "hit"}, "text": item["fact"]})
        regression.append({
            "id": f"pos_{i}_{item['key']}",
            "kind": "positive",
            "text": item["prompt"],
            "expected_key": item["key"],
            "label": {head: "hit"},
        })

    outside_prompts = custom_outside if custom_outside else list(DEFAULT_OUTSIDE_PROMPTS)
    all_null_label = {f"h{i}": None for i in range(len(parsed_entries))}
    for j, prompt_txt in enumerate(outside_prompts):
        regression.append({
            "id": f"out_{j}",
            "kind": "outside",
            "text": prompt_txt,
            "expected_key": None,
            "label": all_null_label,
        })

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rules_path = out_path / "rules.dsl"
    facts_path = out_path / "facts.json"
    gen_path = out_path / "generate.dsl"
    reg_path = out_path / "regression.json"
    rep_path = out_path / "from_table_report.json"

    rules_path.write_text("\n".join(dsl_lines) + "\n", encoding="utf-8")
    _write_json(facts_path, facts)
    gen_path.write_text("kv: bank 3\n", encoding="utf-8")
    _write_json(reg_path, regression)

    R = rules.standalone(str(rules_path))
    parse_facts(facts, R)

    answers = list(rules.answers(R, regression))
    pos_exact, out_quiet, collisions = 0, 0, []
    for item, (ans, _) in zip(regression, answers):
        cls = ans["class"]
        active_heads = {h: v for h, v in cls.items() if v is not None} if isinstance(cls, dict) else {}
        if item["kind"] == "positive":
            if active_heads == item["label"]:
                pos_exact += 1
            else:
                collisions.append({
                    "id": item["id"],
                    "kind": "positive",
                    "expected": item["label"],
                    "got": cls,
                    "active_heads": sorted(active_heads),
                })
        else:
            if not cls or not active_heads:
                out_quiet += 1
            else:
                collisions.append({"id": item["id"], "kind": "outside", "expected": None, "got": cls})

    n_pos = len(parsed_entries)
    n_out = len(outside_prompts)
    passed = (pos_exact == n_pos and out_quiet == n_out and n_pos > 0 and n_out > 0)
    rep = {
        "schema": "kestrom-kv-table-v1",
        "source": {src_path.name: sha256_file(src_path)},
        "outputs": {
            "rules.dsl": sha256_file(rules_path),
            "facts.json": sha256_file(facts_path),
            "generate.dsl": sha256_file(gen_path),
            "regression.json": sha256_file(reg_path),
        },
        "totals": {
            "entries": n_pos,
            "positive": n_pos,
            "positive_exact": pos_exact,
            "outside": n_out,
            "outside_quiet": out_quiet,
            "collisions": len(collisions),
        },
        "collisions": collisions,
        "verdict": "PASS" if passed else "FAIL",
    }
    _write_json(rep_path, rep)
    print(f"positive: {pos_exact} of {n_pos} {'PASS' if pos_exact == n_pos else 'FAIL'}", flush=True)
    print(f"outside: {out_quiet} of {n_out} {'PASS' if out_quiet == n_out else 'FAIL'}", flush=True)
    if not passed:
        raise SystemExit(1)
    return rep


def _first_diff(a_toks: list[int], a_fin: str, b_toks: list[int], b_fin: str):
    for i, (x, y) in enumerate(zip(a_toks, b_toks)):
        if x != y:
            return i
    if len(a_toks) != len(b_toks) or a_fin != b_fin:
        return min(len(a_toks), len(b_toks))
    return None


def verify_chat_session(
    model,
    tok,
    bank: dict,
    turns: list,
    max_tokens: int = report.MAX_TOKENS,
    max_match: int = 3,
    prompt_probes: dict | None = None,
    compose_cache: ComposeCache | None = None,
    decode_fn=None,
    make_cache_fn=None,
    probe_fn=None,
) -> dict:
    """Mechanically check that every `kv=None` turn in `turns` matches twice-run plain decoding bit for bit."""
    session = KVSession(
        model,
        tok,
        bank,
        max_tokens=max_tokens,
        max_match=max_match,
        compose_cache=compose_cache,
        decode_fn=decode_fn,
        make_cache_fn=make_cache_fn,
        prompt_probes=prompt_probes,
        probe_fn=probe_fn,
    )
    rows = []
    history_pure = True
    for idx, entry in enumerate(turns):
        if isinstance(entry, str):
            text, values, data, do_reset = entry, None, None, False
        elif isinstance(entry, dict):
            do_reset = bool(entry.get("reset"))
            text = entry.get("user", entry.get("text"))
            if not isinstance(text, str):
                raise SystemExit(f"TURNS.json entry {idx} ({entry!r}): requires 'user' or 'text'")
            values = entry.get("values")
            data = entry.get("data")
        else:
            raise SystemExit(f"TURNS.json entry {idx} must be a string or object: {entry!r}")

        if do_reset:
            session.reset()
            history_pure = True

        res = session.turn(text, values=values, data=data)
        full_ids = list(session.last_prompt_ids)
        if res["kv"] is not None:
            history_pure = False
            row = {
                "turn": res["turn"],
                "user": text,
                "kv": res["kv"],
                "cache_action": res["cache_action"],
                "events": res["events"],
                "checked_non_interference": False,
                "history_pure": False,
            }
        else:
            toks1, fin1, _ = session._decode(full_ids, session._make_cache(session.model))
            toks2, fin2, _ = session._decode(full_ids, session._make_cache(session.model))
            u0_eq = toks1 == toks2 and fin1 == fin2
            bit_eq = res["tokens"] == toks1 and res["finish"] == fin1 and res["events"] == []
            row = {
                "turn": res["turn"],
                "user": text,
                "kv": None,
                "cache_action": res["cache_action"],
                "events": res["events"],
                "checked_non_interference": True,
                "history_pure": history_pure,
                "u0_equal": u0_eq,
                "bit_identical": bit_eq,
                "first_difference": _first_diff(res["tokens"], res["finish"], toks1, fin1),
                "plain_tokens_len": len(toks1),
                "kv_tokens_len": len(res["tokens"]),
            }
        rows.append(row)

    kv_matched = sum(1 for r in rows if r["kv"] is not None)
    kv_none_rows = [r for r in rows if r["kv"] is None]
    n_none = len(kv_none_rows)
    n_pure = sum(1 for r in kv_none_rows if r["history_pure"])
    n_u0 = sum(1 for r in kv_none_rows if r["u0_equal"])
    n_bit = sum(1 for r in kv_none_rows if r["bit_identical"])
    verdict = "PASS" if (n_none > 0 and n_u0 == n_none and n_bit == n_none) else "FAIL"
    return {
        "totals": {
            "turns": len(rows),
            "kv_matched_turns": kv_matched,
            "kv_none_turns": n_none,
            "kv_none_history_pure": n_pure,
            "kv_none_u0_equal": n_u0,
            "kv_none_bit_identical": n_bit,
        },
        "rows": rows,
        "verdict": verdict,
    }


def verify_chat(model_dir, bank_dir, turns_file, out_report, max_tokens=report.MAX_TOKENS, max_match=3, probes_file=None):
    """CLI handler: verify non-interference (`U0` and `G2`) of all `kv=None` turns in `kestrom kv chat`."""
    max_tokens, max_match = int(max_tokens), int(max_match)
    if max_tokens < 1 or max_match < 1:
        raise SystemExit("MAX_TOKENS and MAX_MATCH must be at least 1")
    probes_file = _opt_arg(probes_file)
    prompt_probes = load_prompt_probes(probes_file) if probes_file is not None else None
    bpath = Path(bank_dir)
    bank_meta = load_bank(bank_dir)
    check_model(bank_meta, model_dir)
    turns = report.read(turns_file)
    if not isinstance(turns, list) or not turns:
        raise SystemExit(f"{turns_file}: TURNS.json must be a non-empty JSON list")

    from mlx_lm import load

    model, tok = load(str(model_dir))
    bank = load_bank(bank_dir, tok)
    res = verify_chat_session(model, tok, bank, turns, max_tokens=max_tokens, max_match=max_match, prompt_probes=prompt_probes)

    settings = {
        "max_tokens": max_tokens,
        "max_match": max_match,
        "prefix_mode": bank.get("prefix_mode", "system"),
    }
    if probes_file is not None:
        settings["probes_file"] = {Path(probes_file).name: report.sha(probes_file)}
    rep = {
        "schema": "kestrom-kv-verify-chat-v1",
        "bank": {
            "manifest.json": sha256_file(bpath / "manifest.json"),
            "rules.dsl": sha256_file(bpath / "rules.dsl"),
        },
        "turns_file": {Path(turns_file).name: report.sha(turns_file)},
        "settings": settings,
        **res,
    }
    out_p = Path(out_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_p, rep)

    t = rep["totals"]
    n_none = t["kv_none_turns"]
    u0_ok = n_none > 0 and t["kv_none_u0_equal"] == n_none
    g2_ok = n_none > 0 and t["kv_none_bit_identical"] == n_none
    print(f"U0 (kv=None): {t['kv_none_u0_equal']} of {n_none} {'PASS' if u0_ok else 'FAIL'}", flush=True)
    print(f"G2 (kv=None): {t['kv_none_bit_identical']} of {n_none} {'PASS' if g2_ok else 'FAIL'}", flush=True)
    if rep["verdict"] != "PASS":
        raise SystemExit(1)
