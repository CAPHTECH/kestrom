"""Closed-form Procrustes bridge between two models from unlabeled anchors, for transferring steer directions and prompt conditions
with RMS scale matching (zero target labels; check the recipient outputs with kestrom-verify compare events).
  kestrom bridge fit DONOR RECIPIENT ANCHORS.json OUT.npz [LAYERS] [RANK]
      fit orthogonal subspace Procrustes maps on sequence-mean hidden states of ANCHORS (strings, {"text"}, {"prompt"}, or {"ids"}).
      LAYERS: "auto" (up to 6 depth-aligned layer pairs at 25%-72% depth) or comma pairs "17:22,20:26"; each layer of either model
      appears in at most one pair. RANK: optional SVD rank cap.
  kestrom bridge map BRIDGE.npz DONOR_STEER.npz OUT.npz [RECIPIENT CALIB_PROMPTS.json]
      map a donor kestrom steer .npz (direction, cond, threshold) to recipient layers, with RMS-matched direction norms (printed as
      recommended ALPHA for add lines) and closed-form thresholds (or the median over unlabeled CALIB_PROMPTS on RECIPIENT, so the
      condition fires on half of them: give a set balanced between prompts it should and should not fire on). OUT.npz holds the
      printed mapped layers only: its other layers are zero, and kestrom generate refuses an add or in line on them.
  kestrom bridge chain BRIDGE_AB.npz BRIDGE_BC.npz OUT.npz
      compose A->B and B->C bridges in closed form on shared B layers without running B."""
import gc

import numpy as np

from kestrom_verify import report

from . import steer

DEPTH_FRACS = (0.25, 0.34, 0.43, 0.52, 0.62, 0.72)


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

    if rank is None:
        denom = float(np.linalg.norm(Za_c, "fro"))
        numer = float(np.linalg.norm(Zb_c, "fro"))
    else:
        denom = float(np.linalg.norm(Z_ar, "fro"))
        numer = float(np.linalg.norm(Z_br, "fro"))
    scale = numer / denom
    rms_a = float(np.sqrt(np.mean(np.sum(Za_c * Za_c, axis=1))))
    rms_b = float(np.sqrt(np.mean(np.sum(Zb_c * Zb_c, axis=1))))
    return dict(mu_a=mu_a, mu_b=mu_b, M=M, scale=scale, rms_a=rms_a, rms_b=rms_b, rank=r)


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
    if spec is None or str(spec).lower() in ("auto", "none", ""):
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


def fit(donor_dir, recip_dir, anchors_file, out_npz, layers=None, rank=None):
    """Fit cross-model Procrustes bridges on unlabeled ANCHORS and write OUT.npz."""
    from mlx_lm import load
    import mlx.core as mx

    raw = report.read(anchors_file)
    r_cap = None if rank is None or str(rank).lower() == "none" else int(rank)

    donor, tok_a = load(str(donor_dir))
    n_a = len(donor.layers)
    # Every donor layer the pairs can use: the recipient's layer count is known only once it is loaded.
    d_layers_uniq = sorted({la for la, _ in parse_layers(layers, n_a, 10**6)})
    ids_a = encode_prompts(tok_a, raw)
    Z_a_uniq = capture_layers(donor, ids_a, d_layers_uniq)
    d_a = Z_a_uniq.shape[-1]
    d_slot = {la: i for i, la in enumerate(d_layers_uniq)}
    del donor, tok_a
    gc.collect()
    mx.clear_cache()

    recip, tok_b = load(str(recip_dir))
    n_b = len(recip.layers)
    pairs = parse_layers(layers, n_a, n_b)
    r_layers_uniq = sorted({lb for _, lb in pairs})
    ids_b = encode_prompts(tok_b, raw)
    Z_b_uniq = capture_layers(recip, ids_b, r_layers_uniq)
    d_b = Z_b_uniq.shape[-1]
    r_slot = {lb: i for i, lb in enumerate(r_layers_uniq)}
    del recip, tok_b
    gc.collect()
    mx.clear_cache()

    mu_a = np.zeros((len(pairs), d_a), np.float32)
    mu_b = np.zeros((len(pairs), d_b), np.float32)
    M = np.zeros((len(pairs), d_b, d_a), np.float32)
    scale = np.zeros(len(pairs), np.float32)
    rms_a = np.zeros(len(pairs), np.float32)
    rms_b = np.zeros(len(pairs), np.float32)
    ranks = np.zeros(len(pairs), np.int32)

    for k, (la, lb) in enumerate(pairs):
        f = fit_procrustes(Z_a_uniq[:, d_slot[la]], Z_b_uniq[:, r_slot[lb]], rank=r_cap)
        mu_a[k], mu_b[k], M[k] = f["mu_a"].astype(np.float32), f["mu_b"].astype(np.float32), f["M"].astype(np.float32)
        scale[k], rms_a[k], rms_b[k], ranks[k] = f["scale"], f["rms_a"], f["rms_b"], f["rank"]
        print(f"bridge pair {la}->{lb}: rank={f['rank']}, scale={f['scale']:.4f}, rms={f['rms_a']:.3f}->{f['rms_b']:.3f}", flush=True)

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


def map_arrays(bridge, st, calib_means=None):
    """Map donor steer arrays {direction, cond, threshold} into recipient space with RMS scaling."""
    donor_shape = (int(bridge["donor_num_layers"]), int(bridge["donor_dim"]))
    if st["direction"].shape != donor_shape or st["cond"].shape != donor_shape:
        raise SystemExit(f"DONOR_STEER is {st['direction'].shape}, but the bridge's donor model has (layers, width) {donor_shape}")
    n_b = int(bridge["recip_num_layers"])
    d_b = int(bridge["recip_dim"])
    direction = np.zeros((n_b, d_b), np.float32)
    cond = np.zeros((n_b, d_b), np.float32)
    threshold = np.zeros(n_b, np.float32)
    rms_norm = np.zeros(n_b, np.float32)

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
        v_raw = v_a @ M.T
        n_vraw = float(np.linalg.norm(v_raw))
        tgt_norm = (n_va / ra) * rb
        v_rms = v_raw * (tgt_norm / n_vraw)
        direction[lb] = v_rms.astype(np.float32)
        rms_norm[lb] = np.float32(tgt_norm)

        c_a = st["cond"][la].astype(np.float64)
        c_raw = c_a @ M.T
        n_craw = float(np.linalg.norm(c_raw))
        c_unit = c_raw / n_craw
        cond[lb] = c_unit.astype(np.float32)

        if calib_means is not None and lb in calib_means:
            threshold[lb] = np.float32(np.median(calib_means[lb].astype(np.float64) @ c_unit))
        else:
            t_a = float(st["threshold"][la])
            # Exact closed-form transform of hyperplane h_b @ c_unit > thresh from h_b = mu_b + sc * ((h_a - mu_a) @ M.T)
            t_b = float(mu_b @ c_unit) + sc / n_craw * (t_a - float(mu_a @ c_a))
            threshold[lb] = np.float32(t_b)

    return dict(
        direction=direction,
        cond=cond,
        threshold=threshold,
        rms_norm=rms_norm,
        mapped_layers=np.array(r_layers, np.int32),
    )


def map_steer(bridge_npz, donor_steer_npz, out_npz, recip_dir=None, calib_prompts=None):
    """Map DONOR_STEER.npz across BRIDGE.npz into recipient OUT.npz (optionally calibrating thresholds on unlabeled prompts)."""
    if (recip_dir is None) != (calib_prompts is None):
        raise SystemExit("give RECIPIENT and CALIB_PROMPTS together, or neither (closed-form thresholds)")
    br = dict(np.load(bridge_npz))
    st = dict(np.load(donor_steer_npz))
    calib_means = None
    if calib_prompts is not None:
        from mlx_lm import load
        recip, tok = load(str(recip_dir))
        raw = report.read(calib_prompts)
        ids_list = encode_prompts(tok, raw)
        r_layers = [int(x) for x in br["recip_layers"]]
        captured = capture_layers(recip, ids_list, r_layers)
        calib_means = {lb: captured[:, k] for k, lb in enumerate(r_layers)}
    out = map_arrays(br, st, calib_means=calib_means)
    np.savez(out_npz, **out)
    for lb in out["mapped_layers"]:
        print(f"mapped layer {int(lb)}: rms_norm(ALPHA)={float(out['rms_norm'][lb]):.3f}, threshold={float(out['threshold'][lb]):.4f}", flush=True)


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
