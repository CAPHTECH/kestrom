"""Scalable semantic routing, unstructured corpus ingestion, and iterative multi-hop KV chaining for `kestrom kv`."""
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np

from kestrom_verify import report

from . import kv, rules

ROUTER_FORMAT = "kestrom-kv-router-v1"
INGEST_MODES = ("key-facts", "entities", "atomic", "fast")
PROBE_MODES = ("auto", "structured", "fact-first")
SECOND_SCORE_RATIO = 0.60

BM25_STOPWORDS = {
    "the", "and", "for", "was", "are", "with", "that", "this", "from", "have",
    "which", "what", "who", "where", "when", "how", "same", "its", "into", "about",
    "also", "their", "been", "were", "has", "had", "not", "but", "can", "all",
    "one", "two", "more", "most", "other", "such", "than", "then", "only", "some",
    "there", "they", "them", "his", "her", "she", "him", "you", "your", "our",
    "does", "did", "will", "would", "could", "should", "being", "between", "after",
    "before", "during", "under", "over", "through", "because", "while", "whereby",
    "whose", "whom", "onto", "upon", "each", "both", "own", "too", "very", "just",
}

ENTITY_STOPWORDS = {
    "The", "This", "That", "These", "Those", "There", "Here", "When", "Where",
    "What", "Which", "While", "After", "Before", "During", "Since", "Although",
    "However", "Because", "Under", "Over", "About", "Between", "Among", "Also",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Quarterly",
    "Internal", "Standard", "Effective", "Operations", "Facility", "Security",
    "Annual", "Access", "Visitor", "Audit", "Ref", "Source", "Key", "Facts",
}

EXTRACT_KEY_FACTS_TMPL = (
    "Read the document below and extract a JSON object with three string lists:\n"
    '- "entities": up to 10 key named entities (people, places, countries, nationalities/demonyms, '
    "organizations, works, events, projects, or codes) appearing verbatim in the Title or Text, ordered by salience.\n"
    '- "aliases": up to 4 obvious standard equivalents directly implied by the document '
    "(e.g., an acronym's full organization name or vice versa, and a country name <-> its demonym/nationality).\n"
    '- "facts": 2 to 4 short, self-contained one-sentence facts (<= 22 words each) capturing the key roles, '
    "relations, locations, leaders/titles, numbers, codes, and alternate names stated in the document.\n"
    'Output ONLY a valid JSON object: {{"entities": [...], "aliases": [...], "facts": [...]}}\n'
    "No markdown fences or extra text.\n\n"
    "Title: {title}\n"
    "Text: {text}"
)

EXTRACT_ENTITIES_TMPL = (
    "Read the document below and extract a JSON object with two string lists:\n"
    '- "entities": up to 10 key named entities (people, places, countries, nationalities/demonyms, '
    "organizations, works, events, projects, or codes) appearing verbatim in the Title or Text, ordered by salience.\n"
    '- "aliases": up to 4 obvious standard equivalents directly implied by the document '
    "(e.g., an acronym's full organization name or vice versa, and a country name <-> its demonym/nationality).\n"
    'Output ONLY a valid JSON object: {{"entities": [...], "aliases": [...]}}\n'
    "No markdown fences or extra text.\n\n"
    "Title: {title}\n"
    "Text: {text}"
)

EXTRACT_ATOMIC_TMPL = (
    "Read the document below and extract its single main assignment, relation, or code as a JSON object with "
    "two string keys in exact order:\n"
    '- "fact": one short, self-contained factual sentence starting with its grammatical subject entity '
    "(verbatim entity names and codes from the document)\n"
    '- "subject": the exact grammatical subject entity that begins your "fact" sentence (verbatim from the document)\n'
    "Output ONLY a valid JSON object with no markdown fences or extra text.\n\n"
    "Title: {title}\n"
    "Text: {text}"
)

SUFFICIENCY_PROBE_STRUCTURED = (
    'Question: "{question}"\n'
    "Note: If a named entity in your instructions corresponds to a descriptive phrase in the question, treat them as referring to the same subject.\n"
    "Output ONLY these three lines (no extra text or explanation):\n"
    "Facts: <copy all facts from your instructions separated by ' ; '>\n"
    "Answer: <the exact short answer from Facts if present, otherwise UNKNOWN>\n"
    "Status: <SUFFICIENT if Answer is not UNKNOWN, otherwise NEED_MORE>"
)

SUFFICIENCY_PROBE_FACT_FIRST = (
    'Question: "{question}"\n'
    "Using ONLY your instructions:\n"
    "Facts: List 1-3 short one-line facts from your instructions relevant to the question "
    "(keep exact names, places, countries/nationalities, organizations, numbers, codes, and acronyms).\n"
    "Then on the final line write ONLY `STATUS: SUFFICIENT` if those facts already answer the question, "
    "or `STATUS: NEED_MORE | <next entity to look up>` otherwise."
)


# ---------------------------------------------------------------------------
# 1-Forward O(K) Dual-Ridge Semantic Router
# ---------------------------------------------------------------------------

def prompt_mean_h(model, ids: list[int], layer: int) -> np.ndarray:
    """Extract mean hidden state across all prompt tokens at `layer` in one forward pass."""
    import mlx.core as mx
    from . import steer

    out = steer.capture(model, ids)[int(layer)]
    h = mx.mean(out[0, :len(ids)].astype(mx.float32), axis=0)
    mx.eval(h)
    return np.array(h, dtype=np.float32)


def fit_ridge_dual(H: np.ndarray, Y: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Scale-normalized dual-form ridge regression returning `(W: (d, C), b: (C,))`."""
    H = np.asarray(H, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] < 2:
        raise SystemExit("fit_ridge_dual requires at least 2 feature vectors")
    mu = H.mean(axis=0)
    y_mean = Y.mean(axis=0)
    Hc = H - mu
    Yc = Y - y_mean
    lam = max(1e-8, float(alpha) * float(np.mean(np.sum(Hc * Hc, axis=1))))
    A = Hc @ Hc.T + lam * np.eye(H.shape[0], dtype=np.float64)
    W = Hc.T @ np.linalg.solve(A, Yc)
    b = y_mean - mu @ W
    return W, b


def _apply_wrapper(wrap: str, text: str) -> str:
    if "{}" in wrap:
        return wrap.replace("{}", text)
    if "{text}" in wrap:
        return wrap.replace("{text}", text)
    return wrap.format(text)


def _prompt_item_ids(tok, item, enable_thinking: bool = False) -> list[int]:
    if isinstance(item, dict) and isinstance(item.get("ids"), list):
        return [int(t) for t in item["ids"]]
    txt = item if isinstance(item, str) else item.get("prompt", item.get("text", item.get("user")))
    if not isinstance(txt, str) or not txt.strip():
        raise SystemExit(f"router prompt entry must be a non-empty string or object with prompt/text/ids: {item!r}")
    return kv._chat_messages_ids(tok, [{"role": "user", "content": txt}], enable_thinking=enable_thinking)


def fit_router_data(
    model,
    tok,
    data: dict,
    layer="auto",
    alpha: float = 0.1,
    fingerprint: dict | None = None,
    h_fn=None,
) -> dict:
    """Fit a 1-forward O(K) dual-Ridge semantic router from loaded `data` using `model` (or `h_fn` in tests)."""
    classes = data.get("classes")
    if not isinstance(classes, dict) or len(classes) < 2:
        raise SystemExit("DATA.json must contain a 'classes' mapping with at least 2 keys")
    keys = sorted(str(k) for k in classes)
    wrappers = data.get("wrappers")
    if wrappers is not None and (not isinstance(wrappers, list) or not all(isinstance(w, str) for w in wrappers)):
        raise SystemExit("DATA.json 'wrappers' must be a list of template strings")

    layer_str = str(layer).strip().lower()
    if layer_str == "auto":
        n_layers = len(getattr(model, "layers", []))
        if n_layers < 1:
            raise SystemExit("cannot auto-select layer without model.layers")
        layer_idx = min(max(0, round(0.62 * n_layers)), n_layers - 1)
    else:
        layer_idx = int(layer)
        n_layers = len(getattr(model, "layers", []))
        if n_layers > 0 and not (0 <= layer_idx < n_layers):
            raise SystemExit(f"LAYER {layer_idx} out of bounds for model with {n_layers} layers")

    extract_h = h_fn if h_fn is not None else (lambda ids: prompt_mean_h(model, ids, layer_idx))
    h_cache = {}

    def get_h(item):
        ids = _prompt_item_ids(tok, item)
        key = tuple(ids)
        if key not in h_cache:
            h_cache[key] = np.asarray(extract_h(ids), dtype=np.float32)
        return h_cache[key]

    base_rows = []
    aug_rows = []
    for c_idx, k in enumerate(keys):
        items = classes[k]
        if not isinstance(items, list) or not items:
            raise SystemExit(f"DATA.json class {k!r} must be a non-empty list")
        for it in items:
            base_rows.append((c_idx, get_h(it)))
            if wrappers and isinstance(it, str):
                for w in wrappers:
                    aug_rows.append((c_idx, get_h(_apply_wrapper(w, it))))
            else:
                aug_rows.append((c_idx, get_h(it)))

    H_role = np.stack([h for _, h in aug_rows])
    Y_role = np.zeros((len(aug_rows), len(keys)), dtype=np.float64)
    for row_i, (c_idx, _) in enumerate(aug_rows):
        Y_role[row_i, c_idx] = 1.0
    role_W, role_b = fit_ridge_dual(H_role, Y_role, alpha=float(alpha))

    H_fire = np.stack([h for _, h in base_rows])
    neutral_items = data.get("neutral") or data.get("outside") or []
    ambig_items = data.get("ambiguous") or []

    has_spec = bool(neutral_items)
    if has_spec:
        H_neg = np.stack([get_h(it) for it in neutral_items])
        H_spec = np.concatenate([H_fire, H_neg], axis=0)
        Y_spec = np.concatenate([
            np.ones((len(H_fire), 1), dtype=np.float64),
            -np.ones((len(H_neg), 1), dtype=np.float64),
        ], axis=0)
        spec_W, spec_b = fit_ridge_dual(H_spec, Y_spec, alpha=float(alpha))
    else:
        spec_W = np.zeros((H_role.shape[1], 1), dtype=np.float64)
        spec_b = np.zeros((1,), dtype=np.float64)

    has_single = bool(ambig_items)
    if has_single:
        H_amb = np.stack([get_h(it) for it in ambig_items])
        H_single = np.concatenate([H_fire, H_amb], axis=0)
        Y_single = np.concatenate([
            np.ones((len(H_fire), 1), dtype=np.float64),
            -np.ones((len(H_amb), 1), dtype=np.float64),
        ], axis=0)
        single_W, single_b = fit_ridge_dual(H_single, Y_single, alpha=float(alpha))
    else:
        single_W = np.zeros((H_role.shape[1], 1), dtype=np.float64)
        single_b = np.zeros((1,), dtype=np.float64)

    return {
        "format": ROUTER_FORMAT,
        "layer": int(layer_idx),
        "alpha": float(alpha),
        "keys": keys,
        "role_W": role_W,
        "role_b": role_b,
        "has_spec": has_spec,
        "spec_W": spec_W,
        "spec_b": spec_b,
        "has_single": has_single,
        "single_W": single_W,
        "single_b": single_b,
        "fingerprint": fingerprint or {},
        "unique_forwards": len(h_cache),
    }


def save_router(out_npz: str | Path, router: dict):
    out_p = Path(out_npz)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_p),
        format=np.array(ROUTER_FORMAT),
        layer=np.int64(router["layer"]),
        alpha=np.float64(router["alpha"]),
        keys=np.array(router["keys"], dtype=str),
        role_W=router["role_W"],
        role_b=router["role_b"],
        has_spec=np.int64(1 if router["has_spec"] else 0),
        spec_W=router["spec_W"],
        spec_b=router["spec_b"],
        has_single=np.int64(1 if router["has_single"] else 0),
        single_W=router["single_W"],
        single_b=router["single_b"],
        fingerprint_json=np.array(json.dumps(router.get("fingerprint") or {}, sort_keys=True)),
    )


def load_router(path: str | Path, model_dir: str | Path | None = None) -> dict:
    """Load a compiled `.npz` Ridge router and optionally verify its model fingerprint."""
    z = np.load(str(path), allow_pickle=False)
    fmt = str(z["format"].item()) if "format" in z else None
    if fmt != ROUTER_FORMAT:
        raise SystemExit(f"{path}: unsupported router format {fmt!r}")
    fp = json.loads(str(z["fingerprint_json"].item())) if "fingerprint_json" in z else {}
    if model_dir is not None and fp:
        if fp != kv.model_fingerprint(Path(model_dir)):
            raise SystemExit(f"{path} was fitted for another model than {model_dir} (the model fingerprint differs)")
    return {
        "format": fmt,
        "layer": int(z["layer"].item()),
        "alpha": float(z["alpha"].item()),
        "keys": [str(k) for k in z["keys"].tolist()],
        "role_W": np.asarray(z["role_W"], dtype=np.float64),
        "role_b": np.asarray(z["role_b"], dtype=np.float64),
        "has_spec": bool(int(z["has_spec"].item())),
        "spec_W": np.asarray(z["spec_W"], dtype=np.float64),
        "spec_b": np.asarray(z["spec_b"], dtype=np.float64),
        "has_single": bool(int(z["has_single"].item())),
        "single_W": np.asarray(z["single_W"], dtype=np.float64),
        "single_b": np.asarray(z["single_b"], dtype=np.float64),
        "fingerprint": fp,
    }


def route_prompt(model, ids: list[int], router: dict, h_fn=None) -> tuple[str | None, dict]:
    """Route `ids` using 1 forward pass and O(K) linear projections."""
    h = np.asarray(
        h_fn(ids) if h_fn is not None else prompt_mean_h(model, ids, router["layer"]),
        dtype=np.float64,
    )
    spec_ok = True
    s_spec = None
    if router["has_spec"]:
        s_spec = float((h @ router["spec_W"])[0] + router["spec_b"][0])
        spec_ok = s_spec > 0.0

    single_ok = True
    s_single = None
    if router["has_single"]:
        s_single = float((h @ router["single_W"])[0] + router["single_b"][0])
        single_ok = s_single > 0.0

    role_scores = h @ router["role_W"] + router["role_b"]
    winner_idx = int(np.argmax(role_scores))
    routed_key = router["keys"][winner_idx] if (spec_ok and single_ok) else None
    diag = {
        "spec": round(s_spec, 4) if s_spec is not None else None,
        "spec_fired": spec_ok,
        "single": round(s_single, 4) if s_single is not None else None,
        "single_fired": single_ok,
        "winner": router["keys"][winner_idx],
        "routed": routed_key,
    }
    return routed_key, diag


def fit_router(model_dir, data_file, out_npz, layer="auto", alpha=0.1):
    """CLI handler (`kestrom kv fit-router`): fit a 1-forward O(K) dual-Ridge semantic router."""
    data = report.read(data_file)
    if not isinstance(data, dict):
        raise SystemExit(f"{data_file}: DATA.json must be a JSON object")
    fp = kv.model_fingerprint(Path(model_dir))
    from mlx_lm import load

    model, tok = load(str(model_dir))
    router = fit_router_data(model, tok, data, layer=layer, alpha=float(alpha), fingerprint=fp)
    save_router(out_npz, router)
    print(
        f"router classes={len(router['keys'])} layer={router['layer']} "
        f"unique_forwards={router['unique_forwards']} out={out_npz}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Unstructured Corpus Ingestion (`kestrom kv ingest`)
# ---------------------------------------------------------------------------

def _parse_json_object(raw: str):
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.IGNORECASE | re.DOTALL).strip()
    for cand in (txt,):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _load_corpus(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        raise SystemExit(f"CORPUS_PATH not found: {corpus_path}")
    raw_items = []
    if corpus_path.is_dir():
        files = sorted(p for p in corpus_path.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".md"))
        for idx, p in enumerate(files):
            body = p.read_text(encoding="utf-8-sig").strip()
            if not body:
                continue
            lines_ = body.splitlines()
            title = lines_[0].lstrip("# ").strip() if lines_ else p.stem
            raw_items.append({"doc_id": p.stem or f"doc_{idx:04d}", "title": title, "text": body})
    elif corpus_path.suffix.lower() == ".jsonl":
        for line in corpus_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                raw_items.append(json.loads(line))
    else:
        obj = report.read(str(corpus_path))
        if isinstance(obj, dict):
            obj = obj.get("docs", obj.get("entries", obj.get("rows")))
        if not isinstance(obj, list):
            raise SystemExit(f"{corpus_path}: corpus JSON must be a list or {{'docs': [...]}}")
        raw_items = obj

    if not raw_items:
        raise SystemExit(f"{corpus_path}: corpus is empty")

    docs, seen_ids = [], set()
    for idx, item in enumerate(raw_items):
        if isinstance(item, str):
            did = f"doc_{idx:04d}"
            title = f"Doc {idx}"
            text = item.strip()
        elif isinstance(item, dict):
            did = str(item.get("doc_id") or item.get("id") or f"doc_{idx:04d}").strip()
            text = str(item.get("paragraph_text") or item.get("text") or item.get("content") or item.get("fact") or "").strip()
            title = str(item.get("title") or "").strip()
            if not title and text:
                first_line = text.splitlines()[0].strip()
                title = first_line[:80] if len(first_line) <= 80 else did
        else:
            raise SystemExit(f"{corpus_path} row {idx}: expected string or object")
        if not did or did in seen_ids:
            raise SystemExit(f"{corpus_path} row {idx}: missing or duplicate doc_id {did!r}")
        if not text:
            raise SystemExit(f"{corpus_path} row {idx} ({did!r}): missing document text")
        seen_ids.add(did)
        docs.append({"doc_id": did, "title": title, "text": text})
    return docs


def _extract_fast(doc: dict) -> dict:
    blob = f"{doc['title']}\n{doc['text']}"
    ents = []
    seen = set()

    def add_ent(s: str):
        s = s.strip().strip("\"'.,:;()[]")
        if not s or s in seen or s in ENTITY_STOPWORDS or len(s) < 3 or len(s) > 100:
            return
        if " " not in s and s.islower():
            return
        seen.add(s)
        ents.append(s)

    add_ent(doc["title"])
    for m in re.finditer(r"\b(?:[A-Z][A-Za-z0-9'-]+(?:\s+(?:of|the|for|de|and|&|[A-Z][A-Za-z0-9'-]+)){1,5})\b", blob):
        add_ent(m.group(0))
    for m in re.finditer(r"\b[A-Z]{2,}[0-9]*-[0-9A-Za-z]+\b|\b[A-Z]{2,8}\b", blob):
        add_ent(m.group(0))
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9'-]{2,}\b", blob):
        add_ent(m.group(0))
    if not ents:
        ents = [doc["title"] or doc["doc_id"]]
    return {"doc_id": doc["doc_id"], "raw_entities": ents[:12], "raw_aliases": [], "raw_facts": [], "valid": True}


def _extract_with_model(model, tok, doc: dict, mode: str) -> dict:
    from mlx_lm.models.cache import make_prompt_cache

    title, text = doc["title"], doc["text"]
    if mode == "atomic":
        prompt = EXTRACT_ATOMIC_TMPL.format(title=title, text=text)
        max_toks = 160
    elif mode == "entities":
        prompt = EXTRACT_ENTITIES_TMPL.format(title=title, text=text)
        max_toks = 256
    else:
        prompt = EXTRACT_KEY_FACTS_TMPL.format(title=title, text=text)
        max_toks = 448

    ids = kv._chat_messages_ids(tok, [{"role": "user", "content": prompt}], enable_thinking=False)
    cache = make_prompt_cache(model)
    toks, _, _ = kv._mlx_decode(model, tok, ids, max_toks, cache)
    raw = tok.decode(toks)
    obj = _parse_json_object(raw)

    if mode == "atomic":
        fact = obj.get("fact").strip() if isinstance(obj, dict) and isinstance(obj.get("fact"), str) else ""
        subj = obj.get("subject").strip() if isinstance(obj, dict) and isinstance(obj.get("subject"), str) else ""
        if not subj or not fact:
            fb = _extract_fast(doc)
            return {**fb, "valid": False}
        return {"doc_id": doc["doc_id"], "raw_entities": [subj], "raw_aliases": [], "raw_facts": [fact], "valid": True}

    ents = obj.get("entities") if isinstance(obj, dict) else None
    aliases = obj.get("aliases") if isinstance(obj, dict) else None
    facts = obj.get("facts") if isinstance(obj, dict) else None
    clean_ents = [x.strip() for x in ents if isinstance(x, str) and x.strip()] if isinstance(ents, list) else []
    clean_aliases = [x.strip() for x in aliases if isinstance(x, str) and x.strip()] if isinstance(aliases, list) else []
    clean_facts = [x.strip() for x in facts if isinstance(x, str) and x.strip()][:4] if isinstance(facts, list) else []
    if not clean_ents:
        fb = _extract_fast(doc)
        return {**fb, "raw_aliases": clean_aliases, "raw_facts": clean_facts, "valid": False}
    return {
        "doc_id": doc["doc_id"],
        "raw_entities": clean_ents,
        "raw_aliases": clean_aliases,
        "raw_facts": clean_facts,
        "valid": True,
    }


def _clean_term_hygiene(term: str, doc_blob: str | None) -> bool:
    if len(term) < 3 or len(term) > 120:
        return False
    if "\n" in term or "\r" in term:
        return False
    if doc_blob is not None and term not in doc_blob:
        return False
    if re.match(r"^\$?[\d,.\-%\s]+$", term):
        return False
    if " " not in term and term.islower():
        return False
    if term in ENTITY_STOPWORDS:
        return False
    return True


def _trigger_regex_piece(term: str) -> str:
    esc = re.escape(term)
    if re.fullmatch(r"[\x00-\x7f]+", term):
        return f"(?i:{esc})" if " " in term else esc
    return esc


def _wrap_boundary(pat_body: str, is_ascii: bool = True) -> str:
    return rf"(?<![A-Za-z0-9])(?:{pat_body})(?![A-Za-z0-9])" if is_ascii else pat_body


def _compile_trigger_pat(term: str) -> re.Pattern:
    is_ascii = bool(re.fullmatch(r"[\x00-\x7f]+", term) and re.search(r"[A-Za-z0-9]", term))
    return re.compile(_wrap_boundary(_trigger_regex_piece(term), is_ascii=is_ascii))


def build_two_tier_triggers(docs: list[dict], extracted_by_id: dict) -> tuple[dict, dict, dict]:
    """Build two-tier `spec` (Step 0 + expand) and `broad` (expand-only) triggers with DF filtering."""
    doc_blobs = {d["doc_id"]: f"{d['title']}\n{d['text']}" for d in docs}
    n_docs = len(docs)
    max_df = n_docs if n_docs < 12 else max(3, int(0.25 * n_docs))

    raw_spec, raw_broad, all_terms = {}, {}, set()
    for d in docs:
        did = d["doc_id"]
        title = d["title"]
        blob = doc_blobs[did]
        ext = extracted_by_id[did]

        spec_cands = [title]
        if " (" in title:
            base_t = title.split(" (", 1)[0].strip()
            if base_t and base_t != title:
                spec_cands.append(base_t)
        regnal_t = re.sub(r"\s+[IVX]+,\s*", ", ", title).strip()
        if regnal_t and regnal_t != title:
            spec_cands.append(regnal_t)

        seen_spec, kept_spec = set(), []
        for raw_t in spec_cands:
            t = raw_t.strip().strip("\"'")
            if not t or t in seen_spec:
                continue
            if t != title and not _clean_term_hygiene(t, None):
                continue
            seen_spec.add(t)
            kept_spec.append(t)
            all_terms.add(t)

        seen_broad, kept_broad = set(), []
        for is_alias, pool in ((False, ext["raw_entities"]), (True, ext["raw_aliases"])):
            for raw_t in pool:
                t = raw_t.strip().strip("\"'")
                if not t or not _clean_term_hygiene(t, None if is_alias else blob):
                    continue
                if " " in t or not re.fullmatch(r"[\x00-\x7f]+", t):
                    if t not in seen_spec:
                        seen_spec.add(t)
                        kept_spec.append(t)
                        all_terms.add(t)
                else:
                    if t not in seen_spec and t not in seen_broad:
                        seen_broad.add(t)
                        kept_broad.append(t)
                        all_terms.add(t)

        for s_term in list(kept_spec):
            words = s_term.split()
            if len(words) >= 4:
                tail3 = " ".join(words[-3:])
                if _clean_term_hygiene(tail3, None) and tail3 not in seen_spec and tail3 not in seen_broad:
                    seen_broad.add(tail3)
                    kept_broad.append(tail3)
                    all_terms.add(tail3)

        raw_spec[did] = kept_spec
        raw_broad[did] = kept_broad

    df_map = {}
    for term in sorted(all_terms):
        pat = _compile_trigger_pat(term)
        df_map[term] = sum(1 for blob in doc_blobs.values() if pat.search(blob))

    spec_by_id, broad_by_id = {}, {}
    for d in docs:
        did = d["doc_id"]
        title = d["title"] or did
        spec_f = [t for t in raw_spec[did] if df_map.get(t, 0) <= max_df]
        if title and title not in spec_f and (n_docs < 10 or df_map.get(title, 0) <= max(3, n_docs // 2)):
            spec_f.insert(0, title)
        if not spec_f:
            spec_f = [title]
        broad_f = [t for t in raw_broad[did] if df_map.get(t, 0) <= max_df]
        spec_by_id[did] = spec_f[:12]
        broad_by_id[did] = broad_f[:14]
    return spec_by_id, broad_by_id, df_map


def _bm25_tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]{2,}|[\u3040-\u30ff\u4e00-\u9fff]+", text.lower()) if tok not in BM25_STOPWORDS]


def build_bm25_index(docs: list[dict], k1: float = 1.5, b: float = 0.75) -> dict:
    """Build deterministic Okapi BM25 statistics over `docs` (title weighted 2x)."""
    doc_tf, doc_len = {}, {}
    df = Counter()
    for d in docs:
        did = d["doc_id"]
        title_toks = _bm25_tokenize(d.get("title", ""))
        body_toks = _bm25_tokenize(d["text"])
        toks = title_toks + title_toks + body_toks
        tf = Counter(toks)
        doc_tf[did] = dict(tf)
        doc_len[did] = len(toks)
        for term in tf:
            df[term] += 1
    n_docs = len(docs)
    avgdl = sum(doc_len.values()) / max(1, n_docs)
    idf = {term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()}
    return {"k1": k1, "b": b, "avgdl": avgdl, "idf": idf, "doc_tf": doc_tf, "doc_len": doc_len}


def _bm25_score(bm25_index: dict, query: str, doc_id: str) -> float:
    tf = bm25_index["doc_tf"].get(doc_id)
    if not tf:
        return 0.0
    k1 = bm25_index["k1"]
    b = bm25_index["b"]
    avgdl = max(1e-9, bm25_index["avgdl"])
    dl = bm25_index["doc_len"][doc_id]
    idf = bm25_index["idf"]
    total = 0.0
    for term in set(_bm25_tokenize(query)):
        f = tf.get(term, 0)
        if f > 0:
            w = idf.get(term, 0.0)
            denom = f + k1 * (1.0 - b + b * (dl / avgdl))
            total += w * (f * (k1 + 1.0)) / max(1e-9, denom)
    return total


def _trigger_idf_score(triggers: list[str], text: str, df_map: dict, n_docs: int) -> float:
    score = 0.0
    for term in triggers:
        if _compile_trigger_pat(term).search(text):
            df = max(1, int(df_map.get(term, 1)))
            score += math.log((n_docs + 1.0) / df)
    return score


def _trigger_alternation(terms: list[str]) -> str:
    """One regex matching any of `terms`, each bounded on its own as `_compile_trigger_pat` bounds it."""
    return json.dumps("|".join(_compile_trigger_pat(t).pattern for t in terms), ensure_ascii=False)


def _compile_two_tier_dsl_and_facts(docs, spec_by_id, broad_by_id, extracted_by_id, mode: str):
    """Units are named by the document's position (`d0`, `d1`, ...), so any doc_id is safe; the doc_id is the fact's key.
    `hop_expand` is a register (0 unless a step's values give it), so rules that pass no values (chat, generate) read it as 0."""
    lines = [f"# Auto-generated by `kestrom kv ingest` (mode={mode})"]
    facts = []
    for i, d in enumerate(docs):
        did, n = d["doc_id"], f"d{i}"
        head = f"h_{n}"
        lines.append(f"in m_s_{n}: match {_trigger_alternation(spec_by_id[did])}")
        lines.append(f"h1 u_s_{n}: any(m_s_{n})")
        broad = broad_by_id[did]
        if broad:
            lines.extend([
                f"in m_b_{n}: match {_trigger_alternation(broad)}",
                f"h1 u_b_{n}: all(m_b_{n}, hop_expand)",
                f"h2 v_{n}: any(u_s_{n}, u_b_{n})",
            ])
        else:
            lines.append(f"h2 v_{n}: any(u_s_{n})")
        lines.append(f"out {head}/hit: 1*v_{n}")

        raw_facts = extracted_by_id[did].get("raw_facts") or []
        if mode == "atomic" and raw_facts:
            fact_text = raw_facts[0]
        elif mode == "key-facts" and raw_facts:
            bullets = "\n".join(f"- {f}" for f in raw_facts[:4])
            fact_text = f"{d['title']}\nKey Facts:\n{bullets}\n\nSource: {d['text']}"
        else:
            fact_text = f"{d['title']}: {d['text']}" if d["title"] and not d["text"].startswith(d["title"]) else d["text"]
        facts.append({"key": did, "when": {head: "hit"}, "text": fact_text})
    with_broad = [i for i, d in enumerate(docs) if broad_by_id[d["doc_id"]]]
    if with_broad:
        lines.append(f"set hop_expand: hop_expand if v_d{with_broad[0]}")
    return "\n".join(lines) + "\n", facts


def ingest(model_dir, corpus_path, out_dir, prev_dir=None, mode="key-facts", extract_fn=None):
    """CLI handler (`kestrom kv ingest`): extract triggers/facts and emit two-tier `rules.dsl`, `facts.json`, and `index.json`."""
    prev_dir = kv._opt_arg(prev_dir)
    mode = (mode or "key-facts").strip().lower()
    if mode not in INGEST_MODES:
        raise SystemExit(f"unsupported MODE {mode!r} (expected one of {', '.join(INGEST_MODES)})")
    docs = _load_corpus(Path(corpus_path))
    fp = None if mode == "fast" or extract_fn is not None else kv.model_fingerprint(Path(model_dir))

    prev_cache = {}
    if prev_dir:
        cpath = Path(prev_dir) / "ingest_cache.json"
        if cpath.is_file():
            prev_raw = json.loads(cpath.read_text(encoding="utf-8"))
            if prev_raw.get("mode") == mode and prev_raw.get("fingerprint") == fp:
                prev_cache = prev_raw.get("entries", {})

    reused_ids, to_extract = [], []
    extracted_by_id = {}
    for d in docs:
        did = d["doc_id"]
        sig = kv.sha256_bytes(f"{d['title']}\n{d['text']}".encode("utf-8"))
        cached = prev_cache.get(did)
        if cached and cached.get("sha256") == sig:
            extracted_by_id[did] = cached["extracted"]
            reused_ids.append(did)
        else:
            to_extract.append((d, sig))

    if to_extract:
        if mode == "fast" and extract_fn is None:
            for d, _ in to_extract:
                extracted_by_id[d["doc_id"]] = _extract_fast(d)
        elif extract_fn is not None:
            for d, _ in to_extract:
                extracted_by_id[d["doc_id"]] = extract_fn(d, mode)
        else:
            from mlx_lm import load

            model, tok = load(str(model_dir))
            for d, _ in to_extract:
                extracted_by_id[d["doc_id"]] = _extract_with_model(model, tok, d, mode)

    spec_by_id, broad_by_id, df_map = build_two_tier_triggers(docs, extracted_by_id)
    rules_text, facts = _compile_two_tier_dsl_and_facts(docs, spec_by_id, broad_by_id, extracted_by_id, mode)
    bm25_index = build_bm25_index(docs)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rules_path = out_path / "rules.dsl"
    facts_path = out_path / "facts.json"
    gen_path = out_path / "generate.dsl"
    index_path = out_path / "index.json"
    cache_path = out_path / "ingest_cache.json"
    rep_path = out_path / "ingest_report.json"

    rules_path.write_text(rules_text, encoding="utf-8")
    kv._write_json(facts_path, facts)
    gen_path.write_text("kv: bank 3\n", encoding="utf-8")

    index_obj = {
        "schema": "kestrom-kv-index-v1",
        "mode": mode,
        "n_docs": len(docs),
        "spec_by_id": spec_by_id,
        "broad_by_id": broad_by_id,
        "df_map": df_map,
        "bm25": bm25_index,
    }
    kv._write_json(index_path, index_obj)

    cache_entries = {}
    for d in docs:
        did = d["doc_id"]
        sig = kv.sha256_bytes(f"{d['title']}\n{d['text']}".encode("utf-8"))
        cache_entries[did] = {"sha256": sig, "extracted": extracted_by_id[did]}
    kv._write_json(cache_path, {"mode": mode, "fingerprint": fp, "entries": cache_entries})

    R = rules.standalone(str(rules_path))
    kv.parse_facts(facts, R)
    outside_rows = [{"id": f"out_{i}", "text": t, "values": {"hop_expand": 0}} for i, t in enumerate(kv.DEFAULT_OUTSIDE_PROMPTS)]
    outside_hits = 0
    for row, _ in rules.answers(R, outside_rows):
        cls = row["class"]
        if isinstance(cls, dict) and any(v is not None for v in cls.values()):
            outside_hits += 1
        elif cls is not None and not isinstance(cls, dict):
            outside_hits += 1

    rep = {
        "schema": "kestrom-kv-ingest-v1",
        "mode": mode,
        "docs": len(docs),
        "reused": len(reused_ids),
        "extracted": len(to_extract),
        "valid_count": sum(1 for d in docs if extracted_by_id[d["doc_id"]].get("valid", True)),
        "outside_default_hits": outside_hits,
        "verdict": "PASS" if outside_hits == 0 else "FAIL",
    }
    kv._write_json(rep_path, rep)
    print(f"ingested docs={len(docs)} reused={len(reused_ids)} extracted={len(to_extract)} outside_hits={outside_hits} ({rep['verdict']})", flush=True)
    return rep


# ---------------------------------------------------------------------------
# Iterative Multi-hop KV Chaining (`kestrom kv multihop`)
# ---------------------------------------------------------------------------

def _is_sufficient(step_text: str) -> bool:
    m_status = list(re.finditer(r"STATUS\s*:\s*(SUFFICIENT|NEED_MORE)", step_text, re.IGNORECASE))
    if m_status:
        return m_status[-1].group(1).upper() == "SUFFICIENT"
    up = step_text.strip().upper()
    return up.startswith("SUFFICIENT")


def _validate_index_against_bank(bank: dict, index_obj: dict, label: str = "INDEX.json") -> dict:
    if not isinstance(index_obj, dict) or not isinstance(index_obj.get("spec_by_id"), dict):
        raise SystemExit(f"{label}: expected JSON object with 'spec_by_id'")
    idx_keys = set(index_obj["spec_by_id"].keys())
    bank_keys = {e["key"] for e in bank["entries"]}
    if idx_keys != bank_keys:
        missing = sorted(bank_keys - idx_keys)[:5]
        extra = sorted(idx_keys - bank_keys)[:5]
        raise SystemExit(
            f"{label}: document keys do not match bank facts "
            f"(missing_from_index={missing}, extra_in_index={extra})"
        )
    return index_obj


def _load_and_validate_index(bank: dict, index_file: str | None) -> dict | None:
    if index_file is None:
        return None
    obj = report.read(index_file)
    return _validate_index_against_bank(bank, obj, label=str(index_file))


def _compile_index_matchers(index_obj: dict) -> tuple[dict, dict]:
    spec_pats, broad_pats = {}, {}
    for did, terms in index_obj.get("spec_by_id", {}).items():
        spec_pats[did] = [_compile_trigger_pat(t) for t in terms]
    for did, terms in index_obj.get("broad_by_id", {}).items():
        broad_pats[did] = [_compile_trigger_pat(t) for t in terms]
    return spec_pats, broad_pats


def _match_and_rank(
    bank: dict,
    entry_by_key: dict,
    query_text: str,
    orig_question: str,
    hop_expand: int,
    seen_keys: set[str],
    index_obj: dict | None,
    spec_pats: dict,
    broad_pats: dict,
    fallback_bm25: dict,
) -> list[tuple[float, dict]]:
    if index_obj is not None:
        hits = []
        for did, pats in spec_pats.items():
            if did in seen_keys or did not in entry_by_key:
                continue
            matched = any(p.search(query_text) for p in pats)
            if not matched and hop_expand > 0:
                matched = any(p.search(query_text) for p in broad_pats.get(did, ()))
            if matched:
                hits.append(entry_by_key[did])
    else:
        data = {"text": query_text}
        if "hop_expand" in bank["rules"]["inputs"]:
            data["values"] = {"hop_expand": int(hop_expand)}
        hits = [
            e
            for e in kv.match_entries(bank, data, max_match=max(1, len(bank["entries"])))
            if e["key"] not in seen_keys
        ]
    if not hits:
        return []

    if index_obj is not None:
        df_map = index_obj.get("df_map", {})
        n_docs = max(1, int(index_obj.get("n_docs", len(bank["entries"]))))
        bm25 = index_obj.get("bm25", fallback_bm25)
        spec_by_id = index_obj.get("spec_by_id", {})
        broad_by_id = index_obj.get("broad_by_id", {})
    else:
        df_map, n_docs, bm25 = {}, len(bank["entries"]), fallback_bm25
        spec_by_id, broad_by_id = {}, {}

    bm25_q = orig_question if hop_expand == 0 else f"{orig_question} {query_text}"
    scored = []
    for pos, e in enumerate(hits):
        did = e["key"]
        trigs = spec_by_id.get(did, []) + (broad_by_id.get(did, []) if hop_expand > 0 else [])
        if hop_expand == 0:
            trig_s = 1.5 * _trigger_idf_score(trigs, query_text, df_map, n_docs)
            bm25_s = 1.0 * _bm25_score(bm25, bm25_q, did)
        else:
            trig_s = 1.0 * _trigger_idf_score(trigs, query_text, df_map, n_docs) + 0.5 * _trigger_idf_score(
                trigs, orig_question, df_map, n_docs
            )
            bm25_s = 1.2 * _bm25_score(bm25, bm25_q, did)
        scored.append((trig_s + bm25_s, -pos, e))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [(s, e) for s, _, e in scored]


def _pick_probe_mode(probe_mode: str, bank: dict, index_obj: dict | None) -> str:
    pm = (probe_mode or "auto").strip().lower()
    if pm not in PROBE_MODES:
        raise SystemExit(f"unsupported PROBE_MODE {probe_mode!r} (expected one of {', '.join(PROBE_MODES)})")
    if pm != "auto":
        return pm
    if index_obj is not None or any("Key Facts:" in e.get("text", "") for e in bank["entries"]):
        return "fact-first"
    return "structured"


def run_multihop_prompts(
    model,
    tok,
    bank: dict,
    prompts: list[dict],
    max_tokens: int = 192,
    max_hops: int = 4,
    max_active: int = 5,
    router: dict | None = None,
    index_obj: dict | None = None,
    probe_mode: str = "auto",
    compose_cache: kv.ComposeCache | None = None,
    decode_fn=None,
    make_cache_fn=None,
    route_fn=None,
) -> list[dict]:
    """Run iterative multi-hop KV chaining over `prompts`, leaving unmatched prompts bit-identical (`events=[]`)."""
    resolved_probe = _pick_probe_mode(probe_mode, bank, index_obj)
    if index_obj is not None:
        _validate_index_against_bank(bank, index_obj)

    compose = compose_cache if compose_cache is not None else kv.ComposeCache()
    if decode_fn is not None:
        _dec = decode_fn
    else:
        def _dec(feed, cache, mt):
            return kv._mlx_decode(model, tok, feed, mt, cache)
    if make_cache_fn is not None:
        _mk_cache = make_cache_fn
    else:
        from mlx_lm.models.cache import make_prompt_cache

        _mk_cache = make_prompt_cache
    _route = route_fn if route_fn is not None else (lambda ids: route_prompt(model, ids, router))

    entry_by_key = {e["key"]: e for e in bank["entries"]}
    fallback_bm25 = build_bm25_index(
        [{"doc_id": e["key"], "title": e["key"], "text": e["text"]} for e in bank["entries"]]
    )
    spec_pats, broad_pats = _compile_index_matchers(index_obj) if index_obj is not None else ({}, {})
    sp = bank["strip_prefix"]
    rows = []

    for x in prompts:
        ids = [int(t) for t in x["ids"]]
        query_text = x.get("question") or (x.get("data") or {}).get("text") or x.get("text")
        if not isinstance(query_text, str) or not query_text.strip():
            raise SystemExit(
                f"prompt {x.get('id')}: multihop requires 'question', 'data.text', or 'text' (ids-only prompts are refused)"
            )
        route_diag = None
        active = []
        rejected_by_router = False

        if router is not None:
            routed_key, route_diag = _route(ids)
            if routed_key is not None and routed_key in entry_by_key:
                active = [entry_by_key[routed_key]]
            else:
                rejected_by_router = True

        if not active and not rejected_by_router:
            ranked0 = _match_and_rank(
                bank,
                entry_by_key,
                query_text,
                query_text,
                hop_expand=0,
                seen_keys=set(),
                index_obj=index_obj,
                spec_pats=spec_pats,
                broad_pats=broad_pats,
                fallback_bm25=fallback_bm25,
            )
            active = [e for _, e in ranked0[: min(2, int(max_active))]]

        if not active:
            toks, finish, _ = _dec(ids, _mk_cache(model), int(max_tokens))
            row = {
                **{k: v for k, v in x.items() if k != "ids"},
                "finish": finish,
                "tokens": toks,
                "text": tok.decode(toks),
                "events": [],
                "detail": [],
                "kv": None,
                "provenance": [],
                "hops": 0,
                "steps_trace": [],
            }
            if route_diag is not None:
                row["router"] = route_diag
            rows.append(row)
            continue

        if list(ids[: len(sp)]) != sp:
            raise SystemExit(f"prompt {x.get('id')}: prompt ids do not start with bank strip_prefix {sp}")

        seen = {e["key"] for e in active}
        steps_trace = []
        probe_tmpl = SUFFICIENCY_PROBE_FACT_FIRST if resolved_probe == "fact-first" else SUFFICIENCY_PROBE_STRUCTURED
        probe_max_toks = 160 if resolved_probe == "fact-first" else 96

        for hop_idx in range(1, int(max_hops) + 1):
            p_cache = compose.get_or_build(model, tok, bank, active)
            probe_content = probe_tmpl.format(question=query_text)
            probe_ids = kv._chat_messages_ids(tok, [{"role": "user", "content": probe_content}], enable_thinking=False)[
                len(sp) :
            ]
            p_toks, _, _ = _dec(probe_ids, p_cache, probe_max_toks)
            step_text = tok.decode(p_toks).strip()

            if _is_sufficient(step_text):
                steps_trace.append({"hop": hop_idx, "step_text": step_text, "added_keys": [], "stopped_early": True})
                break

            if len(active) >= int(max_active):
                steps_trace.append({"hop": hop_idx, "step_text": step_text, "added_keys": [], "stopped_early": False})
                break

            ranked_next = _match_and_rank(
                bank,
                entry_by_key,
                step_text,
                query_text,
                hop_expand=1,
                seen_keys=seen,
                index_obj=index_obj,
                spec_pats=spec_pats,
                broad_pats=broad_pats,
                fallback_bm25=fallback_bm25,
            )
            added = []
            if ranked_next:
                top_s, top_e = ranked_next[0]
                active.append(top_e)
                seen.add(top_e["key"])
                added.append(top_e["key"])
                if len(ranked_next) >= 2 and len(active) < int(max_active):
                    s2, e2 = ranked_next[1]
                    if top_s > 0.0 and s2 >= SECOND_SCORE_RATIO * top_s:
                        active.append(e2)
                        seen.add(e2["key"])
                        added.append(e2["key"])
            steps_trace.append({"hop": hop_idx, "step_text": step_text, "added_keys": added, "stopped_early": False})
            if not added:
                break

        final_cache = compose.get_or_build(model, tok, bank, active)
        toks, finish, _ = _dec(ids[len(sp) :], final_cache, int(max_tokens))
        row = {
            **{k: v for k, v in x.items() if k != "ids"},
            "finish": finish,
            "tokens": toks,
            "text": tok.decode(toks),
            "events": [0],
            "detail": [],
            "kv": kv.format_kv_key(active),
            "provenance": [e["key"] for e in active],
            "hops": len(steps_trace),
            "steps_trace": steps_trace,
        }
        if route_diag is not None:
            row["router"] = route_diag
        rows.append(row)
    return rows


def multihop(
    model_dir,
    bank_dir,
    prompts_file,
    out_file,
    max_tokens=192,
    max_hops=4,
    max_active=5,
    router_file=None,
    index_file=None,
    probe_mode="auto",
):
    """CLI handler (`kestrom kv multihop`): iterative multi-hop KV chaining over PROMPTS.json."""
    router_file = kv._opt_arg(router_file)
    index_file = kv._opt_arg(index_file)
    max_tokens, max_hops, max_active = int(max_tokens), int(max_hops), int(max_active)
    if max_tokens < 1 or max_hops < 1 or max_active < 1:
        raise SystemExit("MAX_TOKENS, MAX_HOPS, and MAX_ACTIVE must be at least 1")

    bank_meta = kv.load_bank(bank_dir)
    kv.check_model(bank_meta, model_dir)
    router = load_router(router_file, model_dir) if router_file is not None else None
    index_obj = _load_and_validate_index(bank_meta, index_file)
    prompts = report.read(prompts_file)

    from mlx_lm import load

    model, tok = load(str(model_dir))
    bank = kv.load_bank(bank_dir, tok)
    compose = kv.ComposeCache()

    rows = run_multihop_prompts(
        model,
        tok,
        bank,
        prompts,
        max_tokens=max_tokens,
        max_hops=max_hops,
        max_active=max_active,
        router=router,
        index_obj=index_obj,
        probe_mode=probe_mode,
        compose_cache=compose,
    )
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    report.write_lines(str(out_p), rows)
    matched_n = sum(1 for r in rows if r["events"])
    print(
        f"multihop prompts={len(rows)} matched={matched_n} plain={len(rows) - matched_n} "
        f"compose_hits={compose.hits} compose_misses={compose.misses}",
        flush=True,
    )
