"""Closed-form Procrustes bridge between two models from unlabeled anchors, for transferring steer directions, prompt conditions,
binary encoder/LLM probes and fitted rules probes (MLX or PyTorch/CUDA; check recipient outputs with kestrom-verify compare events).
  kestrom bridge fit DONOR RECIPIENT ANCHORS.json|builtin OUT.npz [LAYERS] [RANK] [RUNTIME]
      fit orthogonal subspace Procrustes maps on sequence-mean hidden states of ANCHORS (pass `builtin` or `default` for the
      bundled 1,000-prompt EN/JA multi-domain anchor set, or a JSON file of strings, {"text"}, {"prompt"}, or {"ids"}: ids only
      where DONOR and RECIPIENT share a tokenizer). DONOR may be a causal LM (MLX or PyTorch) or a local Hugging Face PyTorch
      encoder (e.g. ModernBERT / mmBERT / BERT / DeBERTa, detected from config.json), which refuses a text longer than its
      tokenizer's limit. LAYERS: "auto" (up to 6 depth-aligned layer pairs at 25%-72% depth) or comma pairs "17:22,20:26"; each
      layer of either model appears in at most one pair. RANK: optional SVD rank cap ("none" or "-" to omit). RUNTIME is `mlx`
      (default, Apple Silicon) or `torch` (PyTorch/transformers: CUDA, else MPS, else CPU). Warns on narrow anchors or low rank.
  kestrom bridge probe DONOR DATA.json OUT.npz LAYER [LAMBDA] [RUNTIME]
      fit a standardized L2-regularized binary linear prompt sensor on DONOR (causal LM on RUNTIME, or local HF PyTorch encoder)
      at LAYER from labeled DATA.json ([{"text"|"prompt": ..., "label": 0|1}, ...]), writing a standard steer-compatible OUT.npz
      (zero direction, unit cond, midpoint threshold, and donor threshold quantile for unlabeled quantile calibration in `map`).
  kestrom bridge map BRIDGE.npz DONOR_STEER.npz OUT.npz [RECIPIENT CALIB_PROMPTS.json] [RUNTIME]
      map a donor kestrom steer or bridge probe .npz (direction, cond, threshold[, quantile]) to recipient layers, with RMS-matched
      direction norms (printed as recommended ALPHA for add lines; zero when direction is zero) and closed-form thresholds (or
      unlabeled quantile/median calibration over CALIB_PROMPTS on RECIPIENT: give a set matching the phrasing and class balance of
      the prompts it will see; pass `-` for RECIPIENT and CALIB_PROMPTS when specifying RUNTIME without calibration). OUT.npz
      holds the printed mapped layers only (the pairs where DONOR_STEER has a direction or cond; a bridge probe file has one):
      its other layers are zero, and kestrom generate refuses an add or in line on them. Warns on low subspace retention
      (nullspace collapse) or biased/drifted calibration thresholds.
  kestrom bridge rules BRIDGE.npz DONOR_FIT.npz OUT_FIT.npz [DONOR RECIPIENT CALIB_DOCS.json] [RUNTIME]
      map a donor kestrom rules fit .npz (fitted at a decoder LAYER present in BRIDGE.npz) to the paired recipient layer, writing P,
      bias and layer into OUT_FIT.npz for kestrom generate --rules. Without CALIB_DOCS, computes P and bias in closed form; with
      DONOR, RECIPIENT and unlabeled CALIB_DOCS (strings, {"text"}, {"prompt"[, "answer"]}, or {"ids"[, "tokens"]}: ids only where
      DONOR and RECIPIENT share a tokenizer), matches each probe's token-stream mean and standard deviation across the two models
      without labels (warns on small calibration streams or extreme variance gain). The rules text is kept as written: a file
      named by an `in ... prompt` or `add` line must already be in the recipient's space (a kestrom bridge map output), or the
      command refuses it.
  kestrom bridge chain BRIDGE_AB.npz BRIDGE_BC.npz OUT.npz
      compose A->B and B->C bridges in closed form on shared B layers without running B (NumPy only).
  kestrom bridge diagnose BRIDGE.npz DONOR_STEER.npz RECIPIENT CALIB_PROMPTS.json [PROD_PROMPTS.json] [OUT.json] [RUNTIME]
      diagnose subspace retention (nullspace collapse), calibration sample health, and optional distribution shift between
      unlabeled CALIB_PROMPTS and unlabeled PROD_PROMPTS on RECIPIENT (Cohen's d, std ratio, fire-rate shift, 2-sample KS, and
      anchor subspace coverage). Pass `-` or `none` for omitted optional arguments; writes deterministic JSON when OUT.json is given.
  kestrom bridge anchors OUT.json
      export the bundled 1,000-prompt EN/JA multi-domain unlabeled anchor set to OUT.json for inspection or customization."""
import gc
import json
from pathlib import Path

import numpy as np

from kestrom_verify import report

from . import steer
from .rules import compile_rules

DEPTH_FRACS = (0.25, 0.34, 0.43, 0.52, 0.62, 0.72)
BUILTIN_ANCHORS = Path(__file__).resolve().parent / "data" / "anchors_en_ja.json"
HF_ENCODER_TYPES = frozenset({
    "modernbert", "bert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
    "electra", "camembert", "albert", "mpnet", "distilbert", "eurobert", "nomic_bert", "jina_bert",
})

# Diagnostic warning thresholds (deterministic, pure NumPy).
MIN_ANCHORS_WARN = 200
MIN_RETENTION_WARN = 0.50
MAX_SHIFT_Z_WARN = 1.0
MIN_CALIB_PROMPTS_WARN = 20
MIN_GAIN_WARN = 0.25
MAX_GAIN_WARN = 4.0
MIN_CALIB_TOKENS_WARN = 64
DIAG_COHEN_D_WARN = 0.50
DIAG_KS_WARN = 0.25
DIAG_RATE_SHIFT_WARN = 0.20
DIAG_STD_RATIO_LO = 0.50
DIAG_STD_RATIO_HI = 2.0
DIAG_COVERAGE_WARN = 0.50


def _opt(val):
    return None if val in (None, "", "none", "None", "-") else val


def _check_runtime(runtime):
    rt = str(runtime).lower()
    if rt not in ("mlx", "torch"):
        raise SystemExit(f"RUNTIME is mlx or torch, not {runtime!r}")
    return rt


def load_anchors(anchors_file):
    """Read ANCHORS from a JSON file, or load the bundled EN/JA multi-domain set when given 'builtin' or 'default'."""
    if str(anchors_file).lower() in ("builtin", "default"):
        if not BUILTIN_ANCHORS.is_file():
            raise SystemExit(f"built-in anchor file is missing: {BUILTIN_ANCHORS}")
        return json.loads(BUILTIN_ANCHORS.read_text(encoding="utf-8"))
    return report.read(anchors_file)


def anchors(out_json):
    """Export the bundled 1,000-prompt EN/JA multi-domain unlabeled anchor set to OUT.json."""
    rows = load_anchors("builtin")
    Path(out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} built-in EN/JA multi-domain anchors to {out_json}", flush=True)


def _is_hf_encoder(model_dir):
    """Return True if local model_dir's config.json identifies a Hugging Face bidirectional encoder model."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return False
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if str(cfg.get("model_type", "")).lower() in HF_ENCODER_TYPES:
        return True
    archs = [str(a) for a in (cfg.get("architectures") or [])]
    return bool(archs) and all("CausalLM" not in a for a in archs) and any(
        any(k in a for k in ("MaskedLM", "SequenceClassification", "TokenClassification", "BertModel", "RobertaModel"))
        for a in archs
    )


def _torch_num_layers(model):
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None and cfg is not None:
        n = getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None)
    if n is None:
        raise SystemExit("cannot determine num_hidden_layers from model config")
    return int(n)


def _clear_torch_cache(torch_mod, device):
    dev_type = getattr(device, "type", str(device))
    if dev_type == "cuda" and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()
    elif dev_type == "mps" and torch_mod.backends.mps.is_available():
        torch_mod.mps.empty_cache()


def _extract_texts(raw_items, name="ANCHORS"):
    """Extract raw text strings from strings, {'text'}, or {'prompt'} entries (refusing {'ids'})."""
    out = []
    for row in raw_items:
        if isinstance(row, dict) and "ids" in row:
            raise SystemExit(f'{name} has {{"ids"}} entries, which cannot be used with a Hugging Face encoder donor: give text')
        text = row if isinstance(row, str) else row.get("text", row.get("prompt")) if isinstance(row, dict) else None
        if text is None:
            raise SystemExit(f"{name} entry needs 'text', 'prompt', or string: {row!r}")
        out.append(str(text))
    return out


def _capture_hf_encoder_layers(model_dir, raw_items, name, layers_idx=None, layers_spec=None, batch_size=32):
    """Capture sequence-mean hidden states [N, K, D] (float64) from a local Hugging Face PyTorch encoder model."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    texts = _extract_texts(raw_items, name)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    model.eval().to(device)
    n_layers = int(model.config.num_hidden_layers)
    if layers_idx is None:
        layers_idx = sorted({la for la, _ in parse_layers(layers_spec, n_layers, 10**6)})
    else:
        for li in layers_idx:
            if not (0 <= int(li) < n_layers):
                raise SystemExit(f"layer {li} out of bounds for encoder with {n_layers} layers")

    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tok(batch, padding=True, return_tensors="pt")
            if enc["input_ids"].shape[1] > tok.model_max_length:
                raise SystemExit(f"{name} has a text longer than the encoder's {tok.model_max_length} tokens: shorten it")
            enc = enc.to(device)
            out = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
            denom = mask.sum(dim=1).clamp(min=1.0)
            b_layers = []
            for li in layers_idx:
                h = out.hidden_states[int(li) + 1].to(torch.float32)
                b_layers.append(((h * mask).sum(dim=1) / denom).cpu().numpy().astype(np.float64))
            chunks.append(np.stack(b_layers, axis=1))
    Z = np.concatenate(chunks, axis=0)
    del model
    gc.collect()
    _clear_torch_cache(torch, device)
    return Z, n_layers, layers_idx, tok


def _capture_torch_causal_layers(model_dir, raw_items, name, layers_idx=None, layers_spec=None):
    """Capture sequence-mean hidden states [N, K, D] (float64) from a PyTorch CausalLM via kestrom_verify.pt.load."""
    import torch
    from kestrom_verify import pt
    from transformers import AutoTokenizer

    model = pt.load(model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    device = model.device
    n_layers = _torch_num_layers(model)
    if layers_idx is None:
        layers_idx = sorted({la for la, _ in parse_layers(layers_spec, n_layers, 10**6)})
    else:
        for li in layers_idx:
            if not (0 <= int(li) < n_layers):
                raise SystemExit(f"layer {li} out of bounds for model with {n_layers} layers")

    ids_list = encode_prompts(tok, raw_items)
    rows_out = []
    with torch.no_grad():
        for ids in ids_list:
            if not ids:
                raise SystemExit(f"{name} entry produced zero tokens")
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(x, use_cache=False, output_hidden_states=True)
            vecs = [out.hidden_states[int(li) + 1][0].to(torch.float32).mean(dim=0).cpu().numpy().astype(np.float64) for li in layers_idx]
            rows_out.append(np.stack(vecs, axis=0))
    Z = np.stack(rows_out, axis=0)
    del model
    gc.collect()
    _clear_torch_cache(torch, device)
    return Z, n_layers, layers_idx, tok


def _capture_torch_token_stream(model_dir, raw_docs, layer):
    """Capture token-level hidden states [T, D] (float64) at decoder `layer` via PyTorch CausalLM."""
    import torch
    from kestrom_verify import pt
    from transformers import AutoTokenizer

    model = pt.load(model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    device = model.device
    n_layers = _torch_num_layers(model)
    layer = int(layer)
    if not (0 <= layer < n_layers):
        raise SystemExit(f"layer {layer} out of bounds for model with {n_layers} layers")

    chunks = []
    with torch.no_grad():
        for row in raw_docs:
            ids, start, end = encode_calib_doc(tok, row)
            if end <= start:
                raise SystemExit(f"calibration doc yields no tokens: {row!r}")
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(x, use_cache=False, output_hidden_states=True)
            h = out.hidden_states[layer + 1][0, start:end].to(torch.float32).cpu().numpy().astype(np.float64)
            chunks.append(h)
    del model
    gc.collect()
    _clear_torch_cache(torch, device)
    if not chunks:
        raise SystemExit("CALIB_DOCS yielded no calibration tokens")
    return np.concatenate(chunks, axis=0), tok


def _capture_model_layers(model_dir, raw_items, name, layers_idx=None, layers_spec=None, runtime="mlx", allow_encoder=False):
    """Capture sequence-mean hidden states [N, K, D] using HF PyTorch encoder (if allowed & detected), torch, or mlx."""
    rt = _check_runtime(runtime)
    if allow_encoder and _is_hf_encoder(model_dir):
        return _capture_hf_encoder_layers(model_dir, raw_items, name, layers_idx=layers_idx, layers_spec=layers_spec)
    if rt == "torch":
        return _capture_torch_causal_layers(model_dir, raw_items, name, layers_idx=layers_idx, layers_spec=layers_spec)
    from mlx_lm import load
    import mlx.core as mx

    model, tok = load(str(model_dir))
    n_layers = len(model.layers)
    if layers_idx is None:
        layers_idx = sorted({la for la, _ in parse_layers(layers_spec, n_layers, 10**6)})
    else:
        for li in layers_idx:
            if not (0 <= int(li) < n_layers):
                raise SystemExit(f"layer {li} out of bounds for model with {n_layers} layers")
    ids_list = encode_prompts(tok, raw_items)
    Z = capture_layers(model, ids_list, layers_idx)
    del model
    gc.collect()
    mx.clear_cache()
    return Z, n_layers, layers_idx, tok


def fit_procrustes(Z_a, Z_b, rank=None):
    """Fit closed-form active-subspace orthogonal Procrustes map from Z_a [N, d_a] to Z_b [N, d_b]."""
    Za = np.asarray(Z_a, dtype=np.float64)
    Zb = np.asarray(Z_b, dtype=np.float64)
    if Za.ndim != 2 or Zb.ndim != 2 or Za.shape[0] != Zb.shape[0]:
        raise SystemExit(f"incompatible anchor shapes: {Za.shape} vs {Zb.shape}")
    if Za.shape[0] < 2:
        raise SystemExit("bridge fit needs at least 2 anchor prompts")

    mu_a = Za.mean(axis=0)
    mu_b = Zb.mean(axis=0)
    Za_c = Za - mu_a
    Zb_c = Zb - mu_b

    _, s_a, Vt_a = np.linalg.svd(Za_c, full_matrices=False)
    _, s_b, Vt_b = np.linalg.svd(Zb_c, full_matrices=False)
    r_a = int(np.sum(s_a > 1e-6 * s_a[0]))
    r_b = int(np.sum(s_b > 1e-6 * s_b[0]))
    r = min(r_a, r_b, int(rank) if rank is not None else min(r_a, r_b))
    if r < 1:
        raise SystemExit("bridge fit needs anchors whose hidden states differ in both models (and RANK >= 1)")

    V_a = Vt_a[:r].T
    V_b = Vt_b[:r].T
    Z_ar = Za_c @ V_a
    Z_br = Zb_c @ V_b

    U, _, Vt = np.linalg.svd(Z_br.T @ Z_ar, full_matrices=False)
    M = V_b @ (U @ Vt) @ V_a.T

    scale = float(np.linalg.norm(Z_br, "fro")) / float(np.linalg.norm(Z_ar, "fro"))
    rms_a = float(np.sqrt(np.mean(np.sum(Za_c * Za_c, axis=1))))
    rms_b = float(np.sqrt(np.mean(np.sum(Zb_c * Zb_c, axis=1))))
    e_a = float(np.sum(s_a[:16] ** 2) / max(float(np.sum(s_a ** 2)), 1e-12))
    e_b = float(np.sum(s_b[:16] ** 2) / max(float(np.sum(s_b ** 2)), 1e-12))
    return dict(
        mu_a=mu_a,
        mu_b=mu_b,
        M=M,
        scale=scale,
        rms_a=rms_a,
        rms_b=rms_b,
        rank=r,
        top16_energy_a=e_a,
        top16_energy_b=e_b,
    )


def check_fit_health(n_anchors, rank, d_a, d_b, top16_energy_a=None, top16_energy_b=None):
    """Return warning strings if anchor count, active rank, or spectral concentration indicate nullspace collapse risk."""
    warnings = []
    if int(n_anchors) < MIN_ANCHORS_WARN:
        warnings.append(
            f"narrow anchor set (n={n_anchors} < {MIN_ANCHORS_WARN}): off-manifold directions may collapse into the SVD nullspace; "
            "consider passing `builtin` or >=1,000 diverse bilingual anchors"
        )
    min_dim = min(int(d_a), int(d_b))
    if int(rank) < min(256, int(0.25 * min_dim)):
        warnings.append(
            f"low active bridge rank ({rank} vs min_dim={min_dim}): anchor diversity is narrow and unseen directions may lose energy"
        )
    if (top16_energy_a is not None and top16_energy_a > 0.90) or (top16_energy_b is not None and top16_energy_b > 0.90):
        ea = f"{top16_energy_a:.2f}" if top16_energy_a is not None else "n/a"
        eb = f"{top16_energy_b:.2f}" if top16_energy_b is not None else "n/a"
        warnings.append(
            f"concentrated anchor spectrum (top-16 energy donor={ea}, recip={eb} > 0.90): anchors span a narrow subspace"
        )
    return warnings


def default_layer_pairs(n_a, n_b):
    """Up to 6 depth-aligned layer pairs across 25%-72% of model depth, in order, skipping a pair that reuses a layer of either model."""
    pairs = []
    for f in DEPTH_FRACS:
        la = min(max(0, round(f * n_a)), n_a - 1)
        lb = min(max(0, round(f * n_b)), n_b - 1)
        if all(la != a and lb != b for a, b in pairs):
            pairs.append((la, lb))
    return pairs


def parse_layers(spec, n_a, n_b):
    """Parse layer pairs '17:22,20:26' or 'auto'/None; a layer of either model appears in at most one pair (map and chain index by it)."""
    if spec is None or str(spec).lower() in ("auto", "none", "", "-"):
        return default_layer_pairs(n_a, n_b)
    pairs = []
    for part in str(spec).split(","):
        a, _, b = part.strip().partition(":")
        if not _:
            raise SystemExit(f"expected LA:LB pairs (e.g. 17:22,20:26), got {part!r}")
        la, lb = int(a), int(b)
        if not (0 <= la < n_a and 0 <= lb < n_b):
            raise SystemExit(f"layer pair {la}:{lb} out of bounds for ({n_a}, {n_b})")
        if any(la == a or lb == b for a, b in pairs):
            raise SystemExit(f"layer pair {la}:{lb} reuses a layer of an earlier pair")
        pairs.append((la, lb))
    return pairs


def encode_prompts(tok, raw_items):
    """Turn anchors or calibration rows (strings, {"text"}, {"prompt"}, or {"ids"}) into token id lists."""
    out = []
    bos = getattr(tok, "bos_token", None)
    tpl = getattr(tok, "chat_template", None)
    for row in raw_items:
        if isinstance(row, dict) and "ids" in row:
            out.append(list(row["ids"]))
            continue
        text = row if isinstance(row, str) else row.get("text", row.get("prompt"))
        if text is None:
            raise SystemExit(f"prompt entry needs 'ids', 'text', 'prompt', or string: {row!r}")
        if tpl:
            msgs = [{"role": "user", "content": text}]
            try:
                formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            formatted = text
        add_sp = not (bos and formatted.startswith(bos))
        out.append(tok.encode(formatted, add_special_tokens=add_sp))
    return out


def capture_layers(model, ids_list, layers_idx):
    """Capture sequence-mean residual activations [N, K, D] (float64) at selected layers in one pass per sequence."""
    import mlx.core as mx

    rows_out = []
    for ids in ids_list:
        outs = steer.capture(model, ids)
        vecs = mx.stack([mx.mean(outs[li][0].astype(mx.float32), axis=0) for li in layers_idx])
        mx.eval(vecs)
        rows_out.append(np.array(vecs, dtype=np.float64))
    return np.stack(rows_out)


def fit(donor_dir, recip_dir, anchors_file, out_npz, layers=None, rank=None, runtime="mlx"):
    """Fit cross-model Procrustes bridges on unlabeled ANCHORS (or `builtin`) and write OUT.npz."""
    rt = _check_runtime(runtime)
    raw = load_anchors(anchors_file)
    if not isinstance(raw, list) or len(raw) < 2:
        raise SystemExit("bridge fit needs at least 2 anchor prompts")
    r_opt = _opt(rank)
    r_cap = None if r_opt is None else int(r_opt)

    Z_a_uniq, n_a, d_layers_uniq, tok_a = _capture_model_layers(
        donor_dir, raw, "ANCHORS", layers_spec=layers, runtime=rt, allow_encoder=True
    )
    d_a = Z_a_uniq.shape[-1]
    d_slot = {la: i for i, la in enumerate(d_layers_uniq)}

    # Recipient layers depend on n_b; capture via runtime helper with exact parsed pairs.
    if rt == "torch":
        import torch
        from kestrom_verify import pt
        from transformers import AutoTokenizer

        recip = pt.load(recip_dir)
        tok_b = AutoTokenizer.from_pretrained(str(recip_dir))
        check_shared_ids(raw, tok_a, tok_b, "ANCHORS")
        del tok_a
        n_b = _torch_num_layers(recip)
        pairs = parse_layers(layers, n_a, n_b)
        r_layers_uniq = sorted({lb for _, lb in pairs})
        ids_b = encode_prompts(tok_b, raw)
        rows_b = []
        with torch.no_grad():
            for ids in ids_b:
                if not ids:
                    raise SystemExit("ANCHORS entry produced zero tokens")
                x = torch.tensor([ids], dtype=torch.long, device=recip.device)
                out = recip(x, use_cache=False, output_hidden_states=True)
                vecs = [out.hidden_states[int(li) + 1][0].to(torch.float32).mean(dim=0).cpu().numpy().astype(np.float64) for li in r_layers_uniq]
                rows_b.append(np.stack(vecs, axis=0))
        Z_b_uniq = np.stack(rows_b, axis=0)
        dev_b = recip.device
        del recip, tok_b
        gc.collect()
        _clear_torch_cache(torch, dev_b)
    else:
        from mlx_lm import load
        import mlx.core as mx

        recip, tok_b = load(str(recip_dir))
        check_shared_ids(raw, tok_a, tok_b, "ANCHORS")
        del tok_a
        n_b = len(recip.layers)
        pairs = parse_layers(layers, n_a, n_b)
        r_layers_uniq = sorted({lb for _, lb in pairs})
        ids_b = encode_prompts(tok_b, raw)
        Z_b_uniq = capture_layers(recip, ids_b, r_layers_uniq)
        del recip, tok_b
        gc.collect()
        mx.clear_cache()

    d_b = Z_b_uniq.shape[-1]
    r_slot = {lb: i for i, lb in enumerate(r_layers_uniq)}

    mu_a = np.zeros((len(pairs), d_a), np.float32)
    mu_b = np.zeros((len(pairs), d_b), np.float32)
    M = np.zeros((len(pairs), d_b, d_a), np.float32)
    scale = np.zeros(len(pairs), np.float32)
    rms_a = np.zeros(len(pairs), np.float32)
    rms_b = np.zeros(len(pairs), np.float32)
    ranks = np.zeros(len(pairs), np.int32)

    warned = set()
    for k, (la, lb) in enumerate(pairs):
        f = fit_procrustes(Z_a_uniq[:, d_slot[la]], Z_b_uniq[:, r_slot[lb]], rank=r_cap)
        mu_a[k], mu_b[k], M[k] = f["mu_a"].astype(np.float32), f["mu_b"].astype(np.float32), f["M"].astype(np.float32)
        scale[k], rms_a[k], rms_b[k], ranks[k] = f["scale"], f["rms_a"], f["rms_b"], f["rank"]
        print(f"bridge pair {la}->{lb}: rank={f['rank']}, scale={f['scale']:.4f}, rms={f['rms_a']:.3f}->{f['rms_b']:.3f}", flush=True)
        for msg in check_fit_health(len(raw), f["rank"], d_a, d_b, f["top16_energy_a"], f["top16_energy_b"]):
            key = (la, lb, msg) if "narrow anchor set" not in msg else ("global", msg)
            if key not in warned:
                warned.add(key)
                prefix = f"WARNING [pair {la}->{lb}]: " if key[0] != "global" else "WARNING: "
                print(prefix + msg, flush=True)

    np.savez(
        out_npz,
        donor_layers=np.array([la for la, _ in pairs], np.int32),
        recip_layers=np.array([lb for _, lb in pairs], np.int32),
        mu_a=mu_a,
        mu_b=mu_b,
        M=M,
        scale=scale,
        rms_a=rms_a,
        rms_b=rms_b,
        rank=ranks,
        donor_num_layers=np.int32(n_a),
        recip_num_layers=np.int32(n_b),
        donor_dim=np.int32(d_a),
        recip_dim=np.int32(d_b),
    )


def check_probe_data(labels, lam):
    """Refuse labels other than the integers 0 and 1 (both present), and LAMBDA <= 0."""
    if not all(type(v) is int and v in (0, 1) for v in labels) or not {0, 1} <= set(labels):
        raise SystemExit("bridge probe DATA.json needs binary labels (0 and 1), with at least one sample of each class")
    if float(lam) <= 0:
        raise SystemExit(f"LAMBDA must be > 0, got {lam}")


def fit_probe_arrays(Z, labels, layer, num_layers, lam=1.0):
    """Fit a standardized L2-regularized binary linear prompt sensor at `layer` and return steer-compatible arrays."""
    X = np.asarray(Z, dtype=np.float64)
    y = np.asarray(labels)
    layer, num_layers, lam = int(layer), int(num_layers), float(lam)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise SystemExit(f"incompatible probe shapes: X={X.shape}, y={y.shape}")
    if not (0 <= layer < num_layers):
        raise SystemExit(f"layer {layer} out of bounds for model with {num_layers} layers")
    check_probe_data(y.tolist(), lam)
    y = y.astype(np.int64)

    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    Xn = (X - mu) / std
    yf = y.astype(np.float64) * 2.0 - 1.0
    U, s, Vt = np.linalg.svd(Xn, full_matrices=False)
    w_std = Vt.T @ ((s / (s * s + lam)) * (U.T @ yf))
    w_raw = w_std / std
    norm = float(np.linalg.norm(w_raw))
    if norm <= 1e-12:
        raise SystemExit("degenerate zero-norm probe: classes are indistinguishable on these representations")
    c_unit = w_raw / norm
    scores = X @ c_unit
    t_mid = 0.5 * (float(np.median(scores[y == 1])) + float(np.median(scores[y == 0])))
    q = float(np.mean(scores <= t_mid))

    dim = X.shape[1]
    direction = np.zeros((num_layers, dim), np.float32)
    cond = np.zeros((num_layers, dim), np.float32)
    threshold = np.zeros(num_layers, np.float32)
    quantile = np.full(num_layers, 0.5, np.float32)
    cond[layer] = c_unit.astype(np.float32)
    threshold[layer] = np.float32(t_mid)
    quantile[layer] = np.float32(q)
    return dict(
        direction=direction,
        cond=cond,
        threshold=threshold,
        quantile=quantile,
        probe_layers=np.array([layer], np.int32),
    )


def probe(donor_dir, data_file, out_npz, layer, lam=1.0, runtime="mlx"):
    """Fit a binary linear prompt sensor on DONOR (MLX LM, PyTorch LM, or HF PyTorch encoder) at LAYER and write OUT.npz."""
    rt = _check_runtime(runtime)
    raw = report.read(data_file)
    if not isinstance(raw, list) or not raw:
        raise SystemExit("bridge probe DATA.json must be a non-empty JSON list of {'text'|'prompt', 'label': 0|1}")
    labels = []
    for row in raw:
        if not isinstance(row, dict) or "label" not in row:
            raise SystemExit(f"bridge probe entry needs 'label' (0 or 1): {row!r}")
        labels.append(row["label"])
    check_probe_data(labels, lam)
    layer = int(layer)

    Z_all, n_layers, _, _ = _capture_model_layers(
        donor_dir, raw, "DATA.json", layers_idx=[layer], runtime=rt, allow_encoder=True
    )
    Z = Z_all[:, 0, :]

    out = fit_probe_arrays(Z, labels, layer, n_layers, lam=float(lam))
    np.savez(out_npz, **out)
    scores = Z @ out["cond"][layer].astype(np.float64)
    acc = float(np.mean((scores > float(out["threshold"][layer])) == np.asarray(labels, dtype=bool)))
    print(f"fitted probe layer {layer}: n={len(labels)}, train_acc={acc:.4f}, threshold={float(out['threshold'][layer]):.4f}, quantile={float(out['quantile'][layer]):.4f}", flush=True)


def map_arrays(bridge, st, calib_means=None):
    """Map donor steer or probe arrays {direction, cond, threshold[, quantile]} into recipient space with RMS scaling."""
    donor_shape = (int(bridge["donor_num_layers"]), int(bridge["donor_dim"]))
    if st["direction"].shape != donor_shape or st["cond"].shape != donor_shape:
        raise SystemExit(f"DONOR_STEER is {st['direction'].shape}, but the bridge's donor model has (layers, width) {donor_shape}")
    n_b = int(bridge["recip_num_layers"])
    d_b = int(bridge["recip_dim"])
    direction = np.zeros((n_b, d_b), np.float32)
    cond = np.zeros((n_b, d_b), np.float32)
    threshold = np.zeros(n_b, np.float32)
    rms_norm = np.zeros(n_b, np.float32)
    dir_retention = np.zeros(n_b, np.float32)
    cond_retention = np.zeros(n_b, np.float32)

    d_layers = [int(x) for x in bridge["donor_layers"]]
    r_layers = [int(x) for x in bridge["recip_layers"]]

    for k, (la, lb) in enumerate(zip(d_layers, r_layers)):
        M = bridge["M"][k].astype(np.float64)
        mu_a = bridge["mu_a"][k].astype(np.float64)
        mu_b = bridge["mu_b"][k].astype(np.float64)
        sc = float(bridge["scale"][k])
        ra = float(bridge["rms_a"][k])
        rb = float(bridge["rms_b"][k])

        v_a = st["direction"][la].astype(np.float64)
        n_va = float(np.linalg.norm(v_a))
        if n_va > 1e-12:
            v_raw = v_a @ M.T
            n_vraw = float(np.linalg.norm(v_raw))
            dir_retention[lb] = np.float32(n_vraw / n_va)
            if n_vraw > 1e-12:
                tgt_norm = (n_va / ra) * rb
                v_rms = v_raw * (tgt_norm / n_vraw)
                direction[lb] = v_rms.astype(np.float32)
                rms_norm[lb] = np.float32(tgt_norm)

        c_a = st["cond"][la].astype(np.float64)
        n_ca = float(np.linalg.norm(c_a))
        c_raw = c_a @ M.T
        n_craw = float(np.linalg.norm(c_raw))
        if n_ca > 1e-12:
            cond_retention[lb] = np.float32(n_craw / n_ca)
        if n_craw <= 1e-12:
            continue
        c_unit = c_raw / n_craw
        cond[lb] = c_unit.astype(np.float32)

        if calib_means is not None and lb in calib_means:
            q = float(st["quantile"][la]) if "quantile" in st else 0.5
            threshold[lb] = np.float32(np.quantile(calib_means[lb].astype(np.float64) @ c_unit, q))
        else:
            t_a = float(st["threshold"][la])
            # Exact closed-form transform of hyperplane h_b @ c_unit > thresh from h_b = mu_b + sc * ((h_a - mu_a) @ M.T)
            t_b = float(mu_b @ c_unit) + sc / n_craw * (t_a - float(mu_a @ c_a))
            threshold[lb] = np.float32(t_b)

    mapped = [lb for lb in r_layers if np.any(direction[lb]) or np.any(cond[lb])]
    if not mapped:
        raise SystemExit(f"DONOR_STEER has no direction or cond at the bridge's donor layers {d_layers}")
    return dict(
        direction=direction,
        cond=cond,
        threshold=threshold,
        rms_norm=rms_norm,
        dir_retention=dir_retention,
        cond_retention=cond_retention,
        mapped_layers=np.array(mapped, np.int32),
    )


def check_map_health(bridge, st, mapped, calib_means=None):
    """Return warning strings for nullspace collapse (low retention) or biased/drifted calibration thresholds."""
    warnings = []
    d_layers = [int(x) for x in bridge["donor_layers"]]
    r_layers = [int(x) for x in bridge["recip_layers"]]
    pair_of = {lb: (k, la) for k, (la, lb) in enumerate(zip(d_layers, r_layers))}

    for lb in [int(x) for x in mapped["mapped_layers"]]:
        k, la = pair_of[lb]
        d_ret = float(mapped["dir_retention"][lb])
        c_ret = float(mapped["cond_retention"][lb])
        if np.any(st["direction"][la]) and d_ret < MIN_RETENTION_WARN:
            warnings.append(
                f"layer {lb}: low steering direction subspace retention ({d_ret:.3f} < {MIN_RETENTION_WARN:.2f}); "
                "anchor set may be too narrow (nullspace collapse)"
            )
        if np.any(st["cond"][la]) and c_ret < MIN_RETENTION_WARN:
            warnings.append(
                f"layer {lb}: low condition subspace retention ({c_ret:.3f} < {MIN_RETENTION_WARN:.2f}); "
                "anchor set may be too narrow (nullspace collapse)"
            )
        if calib_means is not None and lb in calib_means and np.any(mapped["cond"][lb]):
            Z_cal = np.asarray(calib_means[lb], dtype=np.float64)
            if Z_cal.shape[0] < MIN_CALIB_PROMPTS_WARN:
                warnings.append(
                    f"layer {lb}: small CALIB_PROMPTS sample (n={Z_cal.shape[0]} < {MIN_CALIB_PROMPTS_WARN}); "
                    "quantile threshold estimate may be noisy"
                )
            c_unit = mapped["cond"][lb].astype(np.float64)
            s_cal = Z_cal @ c_unit
            std_cal = float(np.std(s_cal))
            if std_cal <= 1e-6:
                warnings.append(
                    f"layer {lb}: near-zero score variance on CALIB_PROMPTS (std={std_cal:.2e}); "
                    "calibration prompts may all belong to one class or phrasing"
                )
            else:
                M = bridge["M"][k].astype(np.float64)
                mu_a = bridge["mu_a"][k].astype(np.float64)
                mu_b = bridge["mu_b"][k].astype(np.float64)
                sc = float(bridge["scale"][k])
                c_a = st["cond"][la].astype(np.float64)
                n_craw = float(np.linalg.norm(c_a @ M.T))
                t_a = float(st["threshold"][la])
                t_closed = float(mu_b @ c_unit) + sc / n_craw * (t_a - float(mu_a @ c_a))
                t_cal = float(mapped["threshold"][lb])
                shift_z = abs(t_cal - t_closed) / std_cal
                rate_closed = float(np.mean(s_cal > t_closed))
                if shift_z > MAX_SHIFT_Z_WARN:
                    warnings.append(
                        f"layer {lb}: calibration threshold shifted by {shift_z:.2f} std from closed-form "
                        f"({t_closed:.4f} -> {t_cal:.4f}); verify CALIB_PROMPTS phrasing and class balance"
                    )
                if rate_closed < 0.05 or rate_closed > 0.95:
                    warnings.append(
                        f"layer {lb}: closed-form threshold fires on {rate_closed * 100:.1f}% of CALIB_PROMPTS "
                        "(extreme class skew or residual offset shift); run `kestrom bridge diagnose` against production samples"
                    )
    return warnings


def map_steer(bridge_npz, donor_steer_npz, out_npz, recip_dir=None, calib_prompts=None, runtime="mlx"):
    """Map DONOR_STEER.npz across BRIDGE.npz into recipient OUT.npz (optionally calibrating thresholds on unlabeled prompts)."""
    rt = _check_runtime(runtime)
    recip_dir = _opt(recip_dir)
    calib_prompts = _opt(calib_prompts)
    if (recip_dir is None) != (calib_prompts is None):
        raise SystemExit("give RECIPIENT and CALIB_PROMPTS together, or neither (closed-form thresholds)")
    br = dict(np.load(bridge_npz))
    st = dict(np.load(donor_steer_npz))
    calib_means = None
    if calib_prompts is not None:
        raw = report.read(calib_prompts)
        if not isinstance(raw, list) or not raw:
            raise SystemExit("CALIB_PROMPTS must be a non-empty JSON list")
        r_layers = [int(x) for x in br["recip_layers"]]
        captured, _, _, _ = _capture_model_layers(
            recip_dir, raw, "CALIB_PROMPTS", layers_idx=r_layers, runtime=rt, allow_encoder=False
        )
        calib_means = {lb: captured[:, k] for k, lb in enumerate(r_layers)}
    out = map_arrays(br, st, calib_means=calib_means)
    np.savez(out_npz, **out)
    for lb in out["mapped_layers"]:
        lb_i = int(lb)
        print(
            f"mapped layer {lb_i}: rms_norm(ALPHA)={float(out['rms_norm'][lb_i]):.3f}, "
            f"threshold={float(out['threshold'][lb_i]):.4f}, "
            f"retention=(dir={float(out['dir_retention'][lb_i]):.3f}, cond={float(out['cond_retention'][lb_i]):.3f})",
            flush=True,
        )
    for msg in check_map_health(br, st, out, calib_means=calib_means):
        print(f"WARNING: {msg}", flush=True)


def encode_calib_doc(tok, row):
    """Turn a calibration doc entry into (token_ids, start, end) skipping BOS and covering answer/doc tokens."""
    if isinstance(row, dict) and "ids" in row:
        p = list(row["ids"])
        a = list(row.get("tokens") or [])
        if a:
            return p + a, max(0, len(p) - 1), len(p) + len(a) - 1
        return p, 1 if len(p) > 1 else 0, len(p)
    if isinstance(row, dict) and "prompt" in row and "answer" in row:
        p = encode_prompts(tok, [{"prompt": row["prompt"]}])[0]
        a = tok.encode(str(row["answer"]), add_special_tokens=False)
        if a:
            return p + a, max(0, len(p) - 1), len(p) + len(a) - 1
        return p, 1 if len(p) > 1 else 0, len(p)
    if isinstance(row, str):
        text = row
    elif isinstance(row, dict) and ("text" in row or "prompt" in row):
        text = row.get("text", row.get("prompt"))
    else:
        raise SystemExit(f"calibration doc entry needs 'ids', 'text', 'prompt', or string: {row!r}")
    ids = tok.encode(str(text), add_special_tokens=True)
    return ids, 1 if len(ids) > 1 else 0, len(ids)


def check_shared_ids(raw, tok_a, tok_b, name):
    """Refuse entries of NAME given as token ids when the two tokenizers differ: each model would read the other's ids."""
    if any(isinstance(r, dict) and "ids" in r for r in raw) and tok_a.get_vocab() != tok_b.get_vocab():
        raise SystemExit(f'{name} has {{"ids"}} entries, but DONOR and RECIPIENT tokenizers differ: give those entries as text')


def capture_token_stream(model, tok, raw_docs, layer):
    """Capture token-level hidden states [T, D] (float64) at decoder `layer` across unlabeled calibration docs."""
    import mlx.core as mx
    from .rows import trunk

    layer = int(layer)
    if not (0 <= layer < len(model.layers)):
        raise SystemExit(f"layer {layer} out of bounds for model with {len(model.layers)} layers")
    chunks, captured = [], {}
    kept = steer.HOOKS.get(id(model.layers[layer]))
    try:
        steer.install(model.layers[layer], lambda out: (captured.__setitem__("h", out), out)[1])
        for row in raw_docs:
            ids, start, end = encode_calib_doc(tok, row)
            if end <= start:
                raise SystemExit(f"calibration doc yields no tokens: {row!r}")
            trunk(model)(mx.array([ids]))
            h = captured["h"][0, start:end].astype(mx.float32)
            mx.eval(h)
            chunks.append(np.array(h, dtype=np.float64))
    finally:
        steer.install(model.layers[layer], kept)
    if not chunks:
        raise SystemExit("CALIB_DOCS yielded no calibration tokens")
    return np.concatenate(chunks, axis=0)


def check_rules_health(gain, n_donor_tokens, n_recip_tokens):
    """Return warning strings for small calibration token streams or extreme Z-score variance gains."""
    warnings = []
    if min(int(n_donor_tokens), int(n_recip_tokens)) < MIN_CALIB_TOKENS_WARN:
        warnings.append(
            f"small CALIB_DOCS token stream (donor={n_donor_tokens}, recip={n_recip_tokens} < {MIN_CALIB_TOKENS_WARN}); "
            "Z-score mean/std calibration may be noisy"
        )
    g = np.asarray(gain, dtype=np.float64)
    bad = [i for i, v in enumerate(g) if v < MIN_GAIN_WARN or v > MAX_GAIN_WARN]
    if bad:
        vals = ", ".join(f"p{i}={g[i]:.2f}" for i in bad)
        warnings.append(
            f"extreme Z-score probe gain ({vals}, outside [{MIN_GAIN_WARN}, {MAX_GAIN_WARN}]); "
            "check CALIB_DOCS domain coverage or bridge quality"
        )
    return warnings


def map_fit_arrays(bridge, fit_npz, donor_H=None, recip_H=None):
    """Map donor kestrom rules fit arrays {rules, P, k, layer[, bias]} across BRIDGE in closed form or via Z-score calibration."""
    donor_layer = int(fit_npz["layer"]) if "layer" in fit_npz else -1
    if donor_layer < 0:
        raise SystemExit("DONOR_FIT.npz was fitted on the final output layer (layer=-1); refit with a decoder LAYER present in BRIDGE.npz")
    d_layers = [int(x) for x in bridge["donor_layers"]]
    if donor_layer not in d_layers:
        raise SystemExit(f"donor layer {donor_layer} is not in BRIDGE.npz (mapped donor layers: {d_layers})")
    idx = d_layers.index(donor_layer)
    recip_layer = int(bridge["recip_layers"][idx])
    recip_shape = (int(bridge["recip_num_layers"]), int(bridge["recip_dim"]))
    R = compile_rules(str(fit_npz["rules"]))
    steers = [(s[1], "cond") for s in R["declared"].values() if s[0] == "prompt"] + [(file, "direction") for file, _, _ in R["adds"]]
    for file, key in steers:
        if np.load(file)[key].shape != recip_shape:
            raise SystemExit(f"the rules read {key} of {file}, not in the recipient's (layers, width) {recip_shape}: map it with kestrom bridge map and name that output in the rules")

    P = np.asarray(fit_npz["P"], dtype=np.float64)
    donor_dim = int(bridge["donor_dim"])
    if P.ndim != 2 or P.shape[0] != donor_dim:
        raise SystemExit(f"DONOR_FIT P has shape {P.shape}, expected ({donor_dim}, num_probes)")
    if "bias" in fit_npz:
        bias_a = np.asarray(fit_npz["bias"], dtype=np.float64)
        if bias_a.shape != (P.shape[1],):
            raise SystemExit(f"DONOR_FIT bias has shape {bias_a.shape}, expected ({P.shape[1]},)")
    else:
        bias_a = np.zeros(P.shape[1], dtype=np.float64)

    M = bridge["M"][idx].astype(np.float64)
    mu_a = bridge["mu_a"][idx].astype(np.float64)
    mu_b = bridge["mu_b"][idx].astype(np.float64)
    sc = float(bridge["scale"][idx])

    if (donor_H is None) != (recip_H is None):
        raise SystemExit("provide donor_H and recip_H together, or neither")

    if donor_H is None:
        W_b = (M @ P) / sc
        bias_b = bias_a + mu_a @ P - mu_b @ W_b
    else:
        Ha = np.asarray(donor_H, dtype=np.float64)
        Hb = np.asarray(recip_H, dtype=np.float64)
        recip_dim = int(bridge["recip_dim"])
        if Ha.ndim != 2 or Hb.ndim != 2 or Ha.shape[1] != donor_dim or Hb.shape[1] != recip_dim or Ha.shape[0] < 2 or Hb.shape[0] < 2:
            raise SystemExit(f"incompatible calibration streams: {Ha.shape} vs {Hb.shape} (expected dims {donor_dim}, {recip_dim})")
        mp = M @ P
        sa = Ha @ P + bias_a
        sb = Hb @ mp
        std_a = sa.std(axis=0)
        std_b = sb.std(axis=0)
        if np.any(std_b <= 1e-12):
            raise SystemExit("degenerate calibration stream: zero recipient probe variance")
        gain = std_a / std_b
        W_b = mp * gain[None, :]
        bias_b = sa.mean(axis=0) - gain * sb.mean(axis=0)

    return dict(
        rules=fit_npz["rules"],
        P=W_b.astype(np.float32),
        bias=bias_b.astype(np.float32),
        k=np.float32(fit_npz["k"]),
        layer=np.int32(recip_layer),
    )


def map_rules(bridge_npz, donor_fit_npz, out_fit_npz, donor_dir=None, recip_dir=None, calib_docs=None, runtime="mlx"):
    """Map DONOR_FIT.npz across BRIDGE.npz into OUT_FIT.npz (optionally matching probe mean and std on unlabeled CALIB_DOCS)."""
    rt = _check_runtime(runtime)
    donor_dir, recip_dir, calib_docs = _opt(donor_dir), _opt(recip_dir), _opt(calib_docs)
    args = (donor_dir, recip_dir, calib_docs)
    if any(a is not None for a in args) and not all(a is not None for a in args):
        raise SystemExit("give DONOR, RECIPIENT and CALIB_DOCS together, or none (closed-form P and bias)")
    br = dict(np.load(bridge_npz))
    ft = dict(np.load(donor_fit_npz))
    out = map_fit_arrays(br, ft)
    warnings = []
    if calib_docs is not None:
        raw_docs = report.read(calib_docs)
        if rt == "torch":
            donor_H, tok_a = _capture_torch_token_stream(donor_dir, raw_docs, int(ft["layer"]))
            recip_H, tok_b = _capture_torch_token_stream(recip_dir, raw_docs, int(out["layer"]))
            check_shared_ids(raw_docs, tok_a, tok_b, "CALIB_DOCS")
            del tok_a, tok_b
        else:
            from mlx_lm import load
            import mlx.core as mx

            donor, tok_a = load(str(donor_dir))
            donor_H = capture_token_stream(donor, tok_a, raw_docs, int(ft["layer"]))
            del donor
            gc.collect()
            mx.clear_cache()

            recip, tok_b = load(str(recip_dir))
            check_shared_ids(raw_docs, tok_a, tok_b, "CALIB_DOCS")
            recip_H = capture_token_stream(recip, tok_b, raw_docs, int(out["layer"]))
            del recip, tok_a, tok_b
            gc.collect()
            mx.clear_cache()

        out = map_fit_arrays(br, ft, donor_H=donor_H, recip_H=recip_H)
        # Compute Z-score gains for diagnostics
        idx = [int(x) for x in br["donor_layers"]].index(int(ft["layer"]))
        P_a = np.asarray(ft["P"], dtype=np.float64)
        bias_a = np.asarray(ft["bias"], dtype=np.float64) if "bias" in ft else np.zeros(P_a.shape[1], dtype=np.float64)
        M = br["M"][idx].astype(np.float64)
        gain = (donor_H @ P_a + bias_a).std(axis=0) / (recip_H @ (M @ P_a)).std(axis=0)
        warnings = check_rules_health(gain, donor_H.shape[0], recip_H.shape[0])
    np.savez(out_fit_npz, **out)
    mode = "zscore" if calib_docs is not None else "closed-form"
    print(f"mapped rules fit: layer {int(ft['layer'])} -> {int(out['layer'])}, probes={out['P'].shape[1]}, mode={mode}", flush=True)
    for msg in warnings:
        print(f"WARNING: {msg}", flush=True)


def _ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic in pure NumPy."""
    x = np.sort(np.asarray(a, dtype=np.float64))
    y = np.sort(np.asarray(b, dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return 0.0
    all_v = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, all_v, side="right") / float(x.size)
    cdf_y = np.searchsorted(y, all_v, side="right") / float(y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _subspace_coverage(M, mu_b, Z_b):
    """Fraction of centered activation Frobenius norm lying inside the column space of M."""
    Xc = np.asarray(Z_b, dtype=np.float64) - np.asarray(mu_b, dtype=np.float64)
    denom = float(np.linalg.norm(Xc, "fro"))
    if denom <= 1e-12:
        return 0.0
    U, s, _ = np.linalg.svd(np.asarray(M, dtype=np.float64), full_matrices=False)
    r = int(np.sum(s > 1e-6 * s[0])) if s.size and s[0] > 0 else 0
    if r == 0:
        return 0.0
    U_r = U[:, :r]
    return float(np.linalg.norm(Xc @ U_r, "fro")) / denom


def diagnose_calibration(bridge, st, calib_means, prod_means=None):
    """Pure-NumPy diagnostic for nullspace retention, calibration health, and optional calib-vs-production shift."""
    mapped = map_arrays(bridge, st, calib_means=calib_means)
    warnings = list(check_map_health(bridge, st, mapped, calib_means=calib_means))

    d_layers = [int(x) for x in bridge["donor_layers"]]
    r_layers = [int(x) for x in bridge["recip_layers"]]
    pair_of = {lb: (k, la) for k, (la, lb) in enumerate(zip(d_layers, r_layers))}

    layers_out = []
    for lb in [int(x) for x in mapped["mapped_layers"]]:
        k, la = pair_of[lb]
        c_unit = mapped["cond"][lb].astype(np.float64)
        t_cal = float(mapped["threshold"][lb])
        entry = {
            "donor_layer": la,
            "recip_layer": lb,
            "dir_retention": round(float(mapped["dir_retention"][lb]), 6),
            "cond_retention": round(float(mapped["cond_retention"][lb]), 6),
            "threshold": round(t_cal, 6),
        }
        if lb in calib_means and np.any(c_unit):
            Z_cal = np.asarray(calib_means[lb], dtype=np.float64)
            s_cal = Z_cal @ c_unit
            cal_mean, cal_std = float(np.mean(s_cal)), float(np.std(s_cal))
            cal_rate = float(np.mean(s_cal > t_cal))
            cal_cov = _subspace_coverage(bridge["M"][k], bridge["mu_b"][k], Z_cal)
            entry["calib"] = {
                "n": int(Z_cal.shape[0]),
                "score_mean": round(cal_mean, 6),
                "score_std": round(cal_std, 6),
                "fire_rate": round(cal_rate, 6),
                "anchor_coverage": round(cal_cov, 6),
            }
            if cal_cov < DIAG_COVERAGE_WARN:
                warnings.append(
                    f"layer {lb}: low anchor subspace coverage on CALIB_PROMPTS ({cal_cov:.3f} < {DIAG_COVERAGE_WARN:.2f}); "
                    "calibration prompts lie largely outside the bridge anchor span"
                )
            if prod_means is not None and lb in prod_means:
                Z_prod = np.asarray(prod_means[lb], dtype=np.float64)
                s_prod = Z_prod @ c_unit
                prod_mean, prod_std = float(np.mean(s_prod)), float(np.std(s_prod))
                prod_rate = float(np.mean(s_prod > t_cal))
                prod_cov = _subspace_coverage(bridge["M"][k], bridge["mu_b"][k], Z_prod)
                pooled = float(np.sqrt(0.5 * (cal_std * cal_std + prod_std * prod_std)))
                cohen_d = abs(prod_mean - cal_mean) / pooled if pooled > 1e-12 else (float("inf") if abs(prod_mean - cal_mean) > 1e-12 else 0.0)
                std_ratio = (prod_std / cal_std) if cal_std > 1e-12 else 0.0
                rate_shift = abs(prod_rate - cal_rate)
                ks = _ks_stat(s_cal, s_prod)
                entry["prod"] = {
                    "n": int(Z_prod.shape[0]),
                    "score_mean": round(prod_mean, 6),
                    "score_std": round(prod_std, 6),
                    "fire_rate": round(prod_rate, 6),
                    "anchor_coverage": round(prod_cov, 6),
                }
                entry["shift"] = {
                    "cohen_d": round(cohen_d, 6) if np.isfinite(cohen_d) else 999.0,
                    "std_ratio": round(std_ratio, 6),
                    "fire_rate_shift": round(rate_shift, 6),
                    "ks_stat": round(ks, 6),
                }
                if prod_cov < DIAG_COVERAGE_WARN:
                    warnings.append(
                        f"layer {lb}: low anchor subspace coverage on PROD_PROMPTS ({prod_cov:.3f} < {DIAG_COVERAGE_WARN:.2f}); "
                        "refit bridge with anchors covering the production domain"
                    )
                if cohen_d > DIAG_COHEN_D_WARN:
                    warnings.append(
                        f"layer {lb}: distribution mean shift between CALIB_PROMPTS and PROD_PROMPTS "
                        f"(Cohen's d={cohen_d:.2f} > {DIAG_COHEN_D_WARN:.2f}); recalibrate on production-like prompts"
                    )
                if cal_std > 1e-12 and (std_ratio < DIAG_STD_RATIO_LO or std_ratio > DIAG_STD_RATIO_HI):
                    warnings.append(
                        f"layer {lb}: score spread mismatch between CALIB_PROMPTS and PROD_PROMPTS "
                        f"(std_ratio={std_ratio:.2f}, outside [{DIAG_STD_RATIO_LO}, {DIAG_STD_RATIO_HI}])"
                    )
                if rate_shift > DIAG_RATE_SHIFT_WARN:
                    warnings.append(
                        f"layer {lb}: firing-rate drift between CALIB_PROMPTS ({cal_rate:.2f}) and PROD_PROMPTS ({prod_rate:.2f}) "
                        f"(delta={rate_shift:.2f} > {DIAG_RATE_SHIFT_WARN:.2f}); phrasing or class balance is mismatched"
                    )
                if ks > DIAG_KS_WARN:
                    warnings.append(
                        f"layer {lb}: two-sample KS shift between CALIB_PROMPTS and PROD_PROMPTS "
                        f"(KS={ks:.2f} > {DIAG_KS_WARN:.2f})"
                    )
        layers_out.append(entry)

    return {
        "status": "WARN" if warnings else "OK",
        "layers": layers_out,
        "warnings": warnings,
    }


def diagnose(bridge_npz, donor_steer_npz, recip_dir, calib_prompts, prod_prompts=None, out_json=None, runtime="mlx"):
    """Run calibration and distribution-shift diagnostics on RECIPIENT, optionally writing deterministic OUT.json."""
    rt = _check_runtime(runtime)
    prod_prompts = _opt(prod_prompts)
    out_json = _opt(out_json)
    br = dict(np.load(bridge_npz))
    st = dict(np.load(donor_steer_npz))
    r_layers = [int(x) for x in br["recip_layers"]]

    raw_cal = report.read(calib_prompts)
    if not isinstance(raw_cal, list) or not raw_cal:
        raise SystemExit("CALIB_PROMPTS must be a non-empty JSON list")
    if prod_prompts is not None:
        raw_prod = report.read(prod_prompts)
        if not isinstance(raw_prod, list) or not raw_prod:
            raise SystemExit("PROD_PROMPTS must be a non-empty JSON list")
        combined = list(raw_cal) + list(raw_prod)
        captured, _, _, _ = _capture_model_layers(
            recip_dir, combined, "CALIB_PROMPTS/PROD_PROMPTS", layers_idx=r_layers, runtime=rt, allow_encoder=False
        )
        n_cal = len(raw_cal)
        calib_means = {lb: captured[:n_cal, k] for k, lb in enumerate(r_layers)}
        prod_means = {lb: captured[n_cal:, k] for k, lb in enumerate(r_layers)}
    else:
        captured, _, _, _ = _capture_model_layers(
            recip_dir, raw_cal, "CALIB_PROMPTS", layers_idx=r_layers, runtime=rt, allow_encoder=False
        )
        calib_means = {lb: captured[:, k] for k, lb in enumerate(r_layers)}
        prod_means = None

    doc = diagnose_calibration(br, st, calib_means, prod_means=prod_means)
    if out_json is not None:
        Path(out_json).write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in doc["layers"]:
        line = (
            f"diagnose layer {row['recip_layer']}: retention=(dir={row['dir_retention']:.3f}, cond={row['cond_retention']:.3f}), "
            f"threshold={row['threshold']:.4f}"
        )
        if "shift" in row:
            sh = row["shift"]
            line += f", cohen_d={sh['cohen_d']:.2f}, ks={sh['ks_stat']:.2f}, rate_shift={sh['fire_rate_shift']:.2f}"
        print(line, flush=True)
    for msg in doc["warnings"]:
        print(f"WARNING: {msg}", flush=True)
    print(f"bridge diagnose status: {doc['status']} ({len(doc['warnings'])} warnings)", flush=True)


def chain_bridges(ab, bc):
    """Compose A->B and B->C bridge dicts/NpzFiles in closed form on shared B layers."""
    ab_b = {int(b): k for k, b in enumerate(ab["recip_layers"])}
    bc_b = {int(b): k for k, b in enumerate(bc["donor_layers"])}
    shared = sorted(set(ab_b) & set(bc_b))
    if not shared:
        raise SystemExit("bridge chain requires at least one shared middle-model layer between BRIDGE_AB and BRIDGE_BC")

    d_a = int(ab["donor_dim"])
    d_c = int(bc["recip_dim"])
    mu_a = np.zeros((len(shared), d_a), np.float32)
    mu_c = np.zeros((len(shared), d_c), np.float32)
    M = np.zeros((len(shared), d_c, d_a), np.float32)
    scale = np.zeros(len(shared), np.float32)
    rms_a = np.zeros(len(shared), np.float32)
    rms_c = np.zeros(len(shared), np.float32)
    ranks = np.zeros(len(shared), np.int32)
    la_out, lc_out = [], []

    for out_k, lb in enumerate(shared):
        i, j = ab_b[lb], bc_b[lb]
        M_ab, M_bc = ab["M"][i].astype(np.float64), bc["M"][j].astype(np.float64)
        mu_a_i = ab["mu_a"][i].astype(np.float64)
        mu_b_ab, mu_b_bc = ab["mu_b"][i].astype(np.float64), bc["mu_a"][j].astype(np.float64)
        mu_c_j = bc["mu_b"][j].astype(np.float64)
        sc_bc = float(bc["scale"][j])

        M[out_k] = (M_bc @ M_ab).astype(np.float32)
        mu_a[out_k] = mu_a_i.astype(np.float32)
        mu_c[out_k] = (mu_c_j + sc_bc * ((mu_b_ab - mu_b_bc) @ M_bc.T)).astype(np.float32)
        scale[out_k] = float(ab["scale"][i]) * sc_bc
        rms_a[out_k] = float(ab["rms_a"][i])
        rms_c[out_k] = float(bc["rms_b"][j])
        ranks[out_k] = min(int(ab["rank"][i]), int(bc["rank"][j]))
        la_out.append(int(ab["donor_layers"][i]))
        lc_out.append(int(bc["recip_layers"][j]))

    return dict(
        donor_layers=np.array(la_out, np.int32),
        recip_layers=np.array(lc_out, np.int32),
        mu_a=mu_a,
        mu_b=mu_c,
        M=M,
        scale=scale,
        rms_a=rms_a,
        rms_b=rms_c,
        rank=ranks,
        donor_num_layers=np.int32(ab["donor_num_layers"]),
        recip_num_layers=np.int32(bc["recip_num_layers"]),
        donor_dim=np.int32(d_a),
        recip_dim=np.int32(d_c),
    )


def chain(bridge_ab, bridge_bc, out_npz):
    """Compose BRIDGE_AB and BRIDGE_BC on shared B layers into OUT.npz."""
    ab, bc = dict(np.load(bridge_ab)), dict(np.load(bridge_bc))
    out = chain_bridges(ab, bc)
    np.savez(out_npz, **out)
    for la, lc in zip(out["donor_layers"], out["recip_layers"]):
        print(f"chained layer {int(la)} -> {int(lc)}", flush=True)
