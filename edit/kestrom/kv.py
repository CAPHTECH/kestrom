"""Compiled Fact-KV Routing (KV-A): compile per-fact system-prefix KV caches from a standalone rules DSL and inject the matching
cache at greedy decoding time (MLX, tier 1). When a prompt's `data` matches no fact (or ties), no cache is injected and the kv line records
no event (where no other line acts, decoding runs unchanged: `events=[]`); when a fact matches, only the prompt tokens after the
shared prefix (no knowledge text) are fed with the compiled cache, recording event `0`. Check with `kestrom-verify compare events`.
`kestrom generate` refuses a bank whose `rules.dsl` or cache files are not the ones its `manifest.json` hashed, or whose model
fingerprint (hashes of the model's config and tokenizer files, and the sizes of its `.safetensors` shards) is not the model's.
  kestrom kv compile MODEL RULES.dsl FACTS.json OUT_DIR [PREV_DIR]
      compile one `.safetensors` prefix KV file per entry in FACTS.json into OUT_DIR, together with `rules.dsl` and
      `manifest.json`. FACTS.json is either `{CLASS: TEXT, ...}` (for single-head rules) or a list of `{"key": ..., "when": CLASS |
      {HEAD: CLASS, ...}, "text": ...}`. With PREV_DIR, entries whose key and `prefix_ids` are unchanged for the same model fingerprint are
      copied without recomputing. Use from `kestrom generate --rules` via a `kv: BANK_DIR` line."""
import hashlib
import json
import shutil
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


def _chat_ids(tok, user_text: str, system_text: str | None) -> list[int]:
    messages = ([{"role": "system", "content": system_text}] if system_text is not None else []) + [{"role": "user", "content": user_text}]
    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return [int(x) for x in ids]


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


def compile_prefix(model, prefix_ids: list[int], out_file: Path, key: str):
    """Prefill prefix_ids into a fresh prompt cache and save it as a safetensors cache file."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache

    cache = make_prompt_cache(model)
    pos = 0
    while pos < len(prefix_ids):
        chunk = prefix_ids[pos:pos + 2048]
        _ = model(mx.array([chunk]), cache=cache)
        mx.eval([c.state for c in cache])
        pos += len(chunk)
    _compact_cache(cache)
    mx.eval([c.state for c in cache])
    save_prompt_cache(str(out_file), cache, metadata={"kestrom_kv_key": key, "prefix_len": str(len(prefix_ids))})


def compile(model_dir, rules_file, facts_file, out_dir, prev_dir=None):
    """Compile RULES.dsl and FACTS.json into a deterministic KV bank in OUT_DIR."""
    R = rules.standalone(rules_file)
    facts = parse_facts(report.read(facts_file), R)
    out_path = Path(out_dir)
    if out_path.exists() and any(out_path.iterdir()):
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

    out_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rules_file, out_path / "rules.dsl")

    compiled, reused, manifest_entries = 0, 0, []
    for entry, pfx in tokenized:
        fname = entry_filename(entry["key"])
        dst = out_path / fname
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
        "rules_sha256": sha256_file(out_path / "rules.dsl"),
        "strip_prefix": strip_prefix or [],
        "entries": manifest_entries,
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    mean_kb = sum(e["nbytes"] for e in manifest_entries) / len(manifest_entries) / 1024.0
    print("compiled", compiled, "reused", reused, "total", len(manifest_entries), "mean_kb", round(mean_kb, 1), flush=True)


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


def match_entry(bank: dict, data: dict):
    """Evaluate the bank's standalone rules on `data`; return the matching entry iff uniquely matched, else None."""
    cls = next(rules.answers(bank["rules"], [data]))[0]["class"]
    if cls is None:
        return None
    hits = []
    for e in bank["entries"]:
        w = e["when"]
        if isinstance(w, str) and cls == w:
            hits.append(e)
        elif isinstance(w, dict) and isinstance(cls, dict) and all(cls.get(h) == v for h, v in w.items()):
            hits.append(e)
    return hits[0] if len(hits) == 1 else None


def load_cache_file(path: str):
    """Load a compiled prompt cache from a `.safetensors` file."""
    from mlx_lm.models.cache import load_prompt_cache

    return load_prompt_cache(str(path))
