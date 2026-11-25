"""Compiled Fact-KV Routing (KV-A): compile per-fact system-prefix KV caches from a standalone rules DSL and inject the matching
cache at greedy decoding time (MLX, tier 1). When a prompt's `data` matches no fact (or exceeds MAX_MATCH), no cache is injected and the
kv line records no event (where no other line acts, decoding runs unchanged: `events=[]`); when one fact matches, only the prompt tokens
after the shared prefix (no knowledge text) are fed with the compiled cache, recording event `0`. When MAX_MATCH > 1 and 2..MAX_MATCH
entries match simultaneously, their texts are composed and prefilled once on demand and kept in an in-memory LRU cache. Check batch runs
with `kestrom-verify compare events`.
`kestrom generate` and `kestrom kv chat` refuse a bank whose `rules.dsl` or cache files are not the ones its `manifest.json` hashed, or
whose model fingerprint (hashes of the model's config and tokenizer files, and the sizes of its `.safetensors` shards) is not the model's.
When a rules file combines `kv:` with `in ... prompt` conditions, `in ... prompt` is evaluated on the full prompt without the fact KV cache before decoding begins.
  kestrom kv compile MODEL RULES.dsl FACTS.json OUT_DIR [PREV_DIR]
      compile one `.safetensors` prefix KV file per entry in FACTS.json into OUT_DIR, together with `rules.dsl` and
      `manifest.json`. FACTS.json is either `{CLASS: TEXT, ...}` (for single-head rules) or a list of `{"key": ..., "when": CLASS |
      {HEAD: CLASS, ...}, "text": ...}`. With PREV_DIR, entries whose key and `prefix_ids` are unchanged for the same model fingerprint are
      copied without recomputing. Compiles into a staging folder and renames atomically, so a failed run leaves no partial OUT_DIR. Use from `kestrom generate --rules` via a `kv: BANK_DIR [MAX_MATCH]` line.
  kestrom kv chat MODEL BANK_DIR [MAX_TOKENS] [MAX_MATCH] [TURNS.json] [OUT.jsonl]
      multi-turn chat session using a compiled KV bank. Every turn evaluates the full turn history (`steps`) through the bank's
      rules so `prev` lines and registers resolve follow-ups without extra model calls. When matched keys are unchanged and the chat
      template extends the cached tokens as a prefix, the live KV cache is reused incrementally (zero re-prefill of earlier turns); on
      topic switches it swaps in the new cache and re-prefills the conversation; when multiple entries match (up to MAX_MATCH, 3 unless
      given), they are composed via the in-memory LRU cache. With TURNS.json (`["turn1", ...]` or `[{"user": ..., "values": ..., "data": ..., "reset": ...}]`),
      writes one JSONL row per turn to OUT.jsonl; without TURNS.json, runs an interactive session (/reset, /status, /quit)."""
from collections import OrderedDict
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from kestrom_verify import report

from . import rules

FORMAT = "kestrom-kv-v1"
PROBES = ("Hello.", "12345.")
META_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors.index.json")


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
    """Fast deterministic fingerprint of MODEL (metadata hashes + safetensors shard sizes)."""
    model_dir = Path(model_dir)
    meta = {name: sha256_file(p) for name in META_FILES if (p := model_dir / name).is_file()}
    shards = sorted([p.name, p.stat().st_size] for p in model_dir.glob("*.safetensors") if p.is_file())
    return {"meta": meta, "shards": shards}


def _chat_messages_ids(tok, messages: list[dict], enable_thinking: bool = False) -> list[int]:
    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return [int(x) for x in ids]


def _chat_ids(tok, user_text: str, system_text: str | None) -> list[int]:
    messages = ([{"role": "system", "content": system_text}] if system_text is not None else []) + [{"role": "user", "content": user_text}]
    return _chat_messages_ids(tok, messages, enable_thinking=False)


def prefix_boundary(tok, fact_text: str) -> tuple[list[int], list[int]]:
    """Verify the chat template places a system turn cleanly before the user turn, returning (strip_prefix, system_prefix_ids)."""
    found = None
    for probe in PROBES:
        u = _chat_ids(tok, probe, None)
        s = _chat_ids(tok, probe, fact_text)
        hit = None
        for k in range(len(u)):
            tail = len(u) - k
            if len(s) > len(u) and u[:k] == s[:k] and u[k:] == s[len(s) - tail:]:
                hit = (u[:k], s[:len(s) - tail])
                break
        if hit is None:
            raise SystemExit("chat template does not place the system turn as a clean prefix before the user turn")
        if found is None:
            found = hit
        elif found != hit:
            raise SystemExit("chat template system-turn boundary varies across probes")
    return found


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
        out.append(cls.from_state(_clone_tree(c.state), _clone_tree(c.meta_state)))
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
        sp, pfx = prefix_boundary(tok, combined_text)
        if sp != bank["strip_prefix"]:
            raise SystemExit(f"composed prefix strip_prefix {sp} differs from bank strip_prefix {bank['strip_prefix']}")
        snap = prefill_prefix(model, pfx)
        if self.maxsize > 0:
            self._cache[combo_key] = snap
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return clone_cache(snap)


def compile(model_dir, rules_file, facts_file, out_dir, prev_dir=None):
    """Compile RULES.dsl and FACTS.json into a deterministic KV bank in OUT_DIR."""
    R = rules.standalone(rules_file)
    facts = parse_facts(report.read(facts_file), R)
    out_path = Path(out_dir)
    if out_path.is_file() or (out_path.exists() and any(out_path.iterdir())):
        raise SystemExit(f"{out_dir} already exists and is not empty")
    model_path = Path(model_dir)
    fp = model_fingerprint(model_path)

    prev_by_prefix = {}
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
        for e in pm.get("entries", []):
            prev_by_prefix[(e["key"], tuple(e["prefix_ids"]))] = (prev_path / e["file"], e["cache_sha256"])

    from mlx_lm import load

    model, tok = load(str(model_dir))
    strip_prefix = None
    tokenized = []
    for entry in facts:
        sp, pfx = prefix_boundary(tok, entry["text"])
        if strip_prefix is None:
            strip_prefix = sp
        elif sp != strip_prefix:
            raise SystemExit(f"entry {entry['key']!r}: strip_prefix {sp} differs from {strip_prefix}")
        tokenized.append((entry, pfx))

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
    return {"dir": str(bpath), "rules": R, "entries": entries, "strip_prefix": sp, "fingerprint": m.get("fingerprint", {})}


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
        self.messages: list[dict] = []
        self.steps: list[dict] = []
        self.active_keys: tuple[str, ...] | None = None
        self.live_cache = None
        self.cached_ids: list[int] = []

    def reset(self):
        """Clear conversation history and live KV state while keeping compiled multi-entry snapshots warm."""
        self.messages.clear()
        self.steps.clear()
        self.active_keys = None
        self.live_cache = None
        self.cached_ids.clear()

    def turn(self, user_text: str, values: dict | None = None, data: dict | None = None) -> dict:
        """Advance one user turn, keeping active KV state incrementally or switching/composing on topic changes."""
        step = dict(data) if data is not None else {"text": user_text}
        if values is not None:
            step["values"] = values
        self.steps.append(step)
        hits = match_entries(self.bank, {"steps": list(self.steps)}, self.max_match)
        self.messages.append({"role": "user", "content": user_text})
        full_ids = _chat_messages_ids(self.tok, self.messages, enable_thinking=self.enable_thinking)
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
        return {
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


def chat(model_dir, bank_dir, max_tokens=report.MAX_TOKENS, max_match=3, turns_file=None, out_file=None):
    """Run an interactive or scripted multi-turn session (`kestrom kv chat`) over a compiled KV bank."""
    max_tokens, max_match = int(max_tokens), int(max_match)
    if max_tokens < 1 or max_match < 1:
        raise SystemExit("MAX_TOKENS and MAX_MATCH must be at least 1")
    if turns_file is not None and out_file is None:
        raise SystemExit("chat with TURNS.json requires OUT.jsonl")
    check_model(load_bank(bank_dir), model_dir)
    turns = report.read(turns_file) if turns_file is not None else None

    from mlx_lm import load

    model, tok = load(str(model_dir))
    bank = load_bank(bank_dir, tok)
    session = KVSession(model, tok, bank, max_tokens=max_tokens, max_match=max_match)

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
