"""Anti-quantization mirror forensics and trigger-blind disarming (tier 3, MLX and PyTorch).

Reads a single FP16/BF16 checkpoint W and a target quantizer Q (MLX affine or bitsandbytes-compatible NF4)
with no clean original model and no trigger prompts. Computes E = Q(W) - W and compares W, Q(W) = W + E,
the anti-quantization mirror W_mirror = 2W - Q(W) = W - E, and Rademacher sign-randomized nulls W_sigma = W + (E * sigma).
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BITS = 4
DEFAULT_GROUP = 64
DEFAULT_MODE = "affine"
N_NULL_WEIGHT = 5
N_NULL_BEHAVIOR = 3
RHO1_THRESH = 1.10
B85_THRESH = 0.20
ON_GRID_MEAN_U = 0.05
COLLAPSE_GE90_THRESH = 30
BEHAVIOR_COH_THRESH = 2.0

# Standard bitsandbytes 4-bit NormalFloat (NF4) codebook in [-1.0, 1.0].
NF4_LEVELS = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)
_NF4_MIDPOINTS = (0.5 * (NF4_LEVELS[:-1] + NF4_LEVELS[1:])).astype(np.float32)


def top_singular_values(mat: np.ndarray, top_k: int = 8, n_iter: int = 20, seed: int = 0) -> np.ndarray:
    """Fast accurate top-k singular values of a float32 2-D matrix via randomized subspace iteration."""
    m, n = mat.shape
    r = min(m, n)
    if r <= 64:
        return np.linalg.svd(mat.astype(np.float64), compute_uv=False)[:top_k]
    q_dim = min(r, top_k + 8)
    rng = np.random.default_rng(seed)
    if m <= n:
        Q = rng.standard_normal((m, q_dim)).astype(np.float32)
        Q, _ = np.linalg.qr(Q)
        for _ in range(n_iter):
            Q, _ = np.linalg.qr(mat @ (mat.T @ Q))
        B = Q.T @ mat
    else:
        Q = rng.standard_normal((n, q_dim)).astype(np.float32)
        Q, _ = np.linalg.qr(Q)
        for _ in range(n_iter):
            Q, _ = np.linalg.qr(mat.T @ (mat @ Q))
        B = mat @ Q
    return np.linalg.svd(B.astype(np.float64), compute_uv=False)[:top_k]


def unpack_affine_codes(packed: np.ndarray, bits: int = 4) -> np.ndarray:
    """Unpack MLX uint32 packed affine quantization codes along the last axis."""
    per_u32 = 32 // bits
    mask = (1 << bits) - 1
    shifts = np.arange(per_u32, dtype=np.uint32) * bits
    unpacked = (packed[..., None] >> shifts) & mask
    return unpacked.reshape(*packed.shape[:-1], packed.shape[-1] * per_u32).astype(np.uint8)


def affine_quantize_numpy(w_f32: np.ndarray, bits: int = 4, group_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy group-wise affine quantizer returning (w_q, codes, scales, biases)."""
    rows, cols = w_f32.shape
    ng = cols // group_size
    max_code = (1 << bits) - 1
    wg = w_f32.astype(np.float32).reshape(rows, ng, group_size)
    lo = np.min(wg, axis=2)
    hi = np.max(wg, axis=2)
    scales = (hi - lo) / float(max_code)
    biases = lo
    safe = np.where(np.abs(scales) < 1e-12, 1.0, scales)
    q = np.where(np.abs(scales[:, :, None]) < 1e-12, 0.0, np.round((wg - biases[:, :, None]) / safe[:, :, None]))
    codes = np.clip(q, 0, max_code).astype(np.uint8)
    w_q = (biases[:, :, None] + codes.astype(np.float32) * scales[:, :, None]).reshape(rows, cols)
    return w_q.astype(np.float32), codes.reshape(rows, cols), scales.astype(np.float32), biases.astype(np.float32)


def nf4_quantize_numpy(w_f32: np.ndarray, group_size: int = DEFAULT_GROUP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy group-wise 4-bit NormalFloat (NF4) quantizer returning (w_q, codes, absmax)."""
    rows, cols = w_f32.shape
    ng = cols // group_size
    wg = w_f32.astype(np.float32).reshape(rows, ng, group_size)
    absmax = np.max(np.abs(wg), axis=2)
    safe = np.where(absmax < 1e-12, 1.0, absmax)
    norm = np.where(absmax[:, :, None] < 1e-12, 0.0, wg / safe[:, :, None])
    codes = np.searchsorted(_NF4_MIDPOINTS, norm).astype(np.uint8)
    w_q = (absmax[:, :, None] * NF4_LEVELS[codes]).reshape(rows, cols).astype(np.float32)
    return w_q, codes.reshape(rows, cols), absmax.astype(np.float32)


def analyze_residual(
    w_f32: np.ndarray,
    w_q: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP,
    n_null: int = N_NULL_WEIGHT,
    seed: int = 0,
    levels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Self-controlled weight-space mirror metrics for one 2-D tensor."""
    e = (w_q - w_f32).astype(np.float32)
    s_real = top_singular_values(e, top_k=8, seed=seed)
    sigma1 = float(s_real[0])
    frob2 = float(np.sum(e.astype(np.float64) ** 2))
    rng = np.random.default_rng(seed)
    null_s1 = []
    for k_i in range(n_null):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=e.shape)
        null_s1.append(float(top_singular_values(e * signs, top_k=2, seed=seed + 101 + k_i)[0]))
    null_mean = float(np.mean(null_s1))
    null_std = float(np.std(null_s1))
    rho1 = sigma1 / null_mean if null_mean > 0 else 1.0
    z_sigma1 = (sigma1 - null_mean) / (null_std + 1e-12)
    s1_s2 = float(s_real[0] / (s_real[1] + 1e-12)) if len(s_real) > 1 else 1.0

    # Quantization cell-boundary clustering (use |scales| since MLX affine scales can be negative)
    rows, cols = w_f32.shape
    ng = cols // group_size
    e_grp = e.reshape(rows, ng, group_size)
    c_grp = codes.reshape(rows, ng, group_size)
    abs_scale = np.abs(scales[:, :, None]).astype(np.float32)
    if levels is None:
        max_code = (1 << bits) - 1
        half_step = np.maximum(abs_scale * 0.5, 1e-12)
    else:
        lv = np.asarray(levels, dtype=np.float32)
        max_code = len(lv) - 1
        h_left = np.empty_like(lv)
        h_left[1:] = 0.5 * (lv[1:] - lv[:-1])
        h_left[0] = h_left[1]
        h_right = np.empty_like(lv)
        h_right[:-1] = 0.5 * (lv[1:] - lv[:-1])
        h_right[-1] = h_right[-2]
        # Asymmetric Voronoi cell: e = w_q - w_f32 >= 0 means w_f32 lies to the left of lv[code].
        half_step = np.maximum(abs_scale * np.where(e_grp >= 0, h_left[c_grp], h_right[c_grp]), 1e-12)
    interior = (c_grp > 0) & (c_grp < max_code) & (abs_scale > 1e-9)
    if np.any(interior):
        u_all = np.abs(e_grp) / half_step
        u_int = np.clip(u_all[interior], 0.0, 2.0)
        mean_u = float(np.mean(u_int))
        boundary_frac_85 = float(np.mean(u_int > 0.85))
        boundary_frac_90 = float(np.mean(u_int > 0.90))
        int_cnt = np.sum(interior, axis=2)
        b85_cnt = np.sum((u_all > 0.85) & interior, axis=2)
        valid_blk = int_cnt >= (group_size // 2)
        blk_b85_p99 = float(np.percentile(b85_cnt[valid_blk] / int_cnt[valid_blk], 99.0)) if np.any(valid_blk) else 0.0
    else:
        mean_u = boundary_frac_85 = boundary_frac_90 = blk_b85_p99 = 0.0

    # Block bin-collapse & outlier ratio (exclude dead all-zero blocks)
    w_grp = np.abs(w_f32.reshape(rows, ng, group_size))
    blk_max = np.max(w_grp, axis=2)
    blk_med = np.median(w_grp, axis=2)
    active = blk_max > 1e-7
    outlier = np.where(active, blk_max / (blk_med + 1e-8), 0.0)
    c_flat = c_grp.reshape(-1, group_size)
    bin_counts = np.stack([np.sum(c_flat == b, axis=1) for b in range(max_code + 1)], axis=1)
    max_bin_frac = np.where(active.reshape(-1), np.max(bin_counts, axis=1) / float(group_size), 0.0)

    return {
        "shape": [int(rows), int(cols)],
        "frob2_e": frob2,
        "sigma1": sigma1,
        "top_s": [round(float(x), 6) for x in s_real],
        "null_s1_mean": null_mean,
        "null_s1_std": null_std,
        "rho1": float(rho1),
        "z_sigma1": float(z_sigma1),
        "s1_s2_ratio": s1_s2,
        "mean_u": mean_u,
        "boundary_frac_85": boundary_frac_85,
        "boundary_frac_90": boundary_frac_90,
        "blk_b85_p99": blk_b85_p99,
        "outlier_p999": float(np.percentile(outlier, 99.9)),
        "outlier_max": float(np.max(outlier)),
        "collapse_p999": float(np.percentile(max_bin_frac, 99.9)),
        "collapse_max": float(np.max(max_bin_frac)),
        "collapse_ge90_count": int(np.sum(max_bin_frac >= 0.90)),
    }


def judge_mirror_tensors(
    tensors: Dict[str, Dict[str, Any]],
    rho1_thresh: float = RHO1_THRESH,
    b85_thresh: float = B85_THRESH,
    collapse_thresh: int = COLLAPSE_GE90_THRESH,
    on_grid_mean_u: float = ON_GRID_MEAN_U,
) -> Dict[str, Any]:
    """Flag tensors whose quantization residual shows sign coherence, high cell-boundary clustering, or block collapse.

    Tensors whose weights already sit on the quantizer grid (`mean_u < on_grid_mean_u`, e.g. a dequantized quantized
    checkpoint or `disarm` output) are listed in `on_grid` and skipped for residual-based `spectral_coherence` and
    `cell_boundary` flags (still checked for `block_collapse`).
    """
    reasons: Dict[str, List[str]] = {}
    on_grid: List[str] = []
    for name in sorted(tensors):
        v = tensors[name]
        is_on_grid = v["mean_u"] < on_grid_mean_u
        if is_on_grid:
            on_grid.append(name)
        r = []
        if not is_on_grid and v["rho1"] > rho1_thresh:
            r.append("spectral_coherence")
        if not is_on_grid and v["boundary_frac_85"] > b85_thresh:
            r.append("cell_boundary")
        if v["collapse_ge90_count"] >= collapse_thresh:
            r.append("block_collapse")
        if r:
            reasons[name] = r

    off_grid = [(k, v) for k, v in tensors.items() if v["mean_u"] >= on_grid_mean_u]
    pool = off_grid or list(tensors.items())
    by_rho = sorted(pool, key=lambda kv: (-kv[1]["rho1"], kv[0]))
    by_b85 = sorted(pool, key=lambda kv: (-kv[1]["boundary_frac_85"], kv[0]))
    by_col = sorted(tensors.items(), key=lambda kv: (-kv[1]["collapse_ge90_count"], -kv[1]["outlier_max"], kv[0]))
    return {
        "flagged": sorted(reasons),
        "reasons": reasons,
        "on_grid": on_grid,
        "max_rho1": by_rho[0][1]["rho1"] if by_rho else 0.0,
        "max_rho1_tensor": by_rho[0][0] if by_rho else None,
        "median_rho1": float(np.median([v["rho1"] for _, v in pool])) if pool else 0.0,
        "max_z_sigma1": max((v["z_sigma1"] for _, v in pool), default=0.0),
        "max_b85": by_b85[0][1]["boundary_frac_85"] if by_b85 else 0.0,
        "max_b85_tensor": by_b85[0][0] if by_b85 else None,
        "median_boundary_frac_85": float(np.median([v["boundary_frac_85"] for _, v in pool])) if pool else 0.0,
        "median_mean_u": float(np.median([v["mean_u"] for v in tensors.values()])) if tensors else 0.0,
        "max_collapse_ge90": by_col[0][1]["collapse_ge90_count"] if by_col else 0,
        "max_collapse_tensor": by_col[0][0] if by_col else None,
        "max_outlier_max": max((v["outlier_max"] for v in tensors.values()), default=0.0),
        "top5_rho1": [{"tensor": k, "rho1": v["rho1"], "z_sigma1": v["z_sigma1"], "s1_s2": v["s1_s2_ratio"]} for k, v in by_rho[:5]],
        "top5_boundary": [{"tensor": k, "b85": v["boundary_frac_85"], "mean_u": v["mean_u"], "blk_p99": v["blk_b85_p99"]} for k, v in by_b85[:5]],
        "top5_collapse": [{"tensor": k, "ge90": v["collapse_ge90_count"], "collapse_max": v["collapse_max"], "outlier_max": v["outlier_max"]} for k, v in by_col[:5]],
    }


def _safe_ratio(num: float, den: float, eps: float = 1e-6):
    """Positive-denominator ratio; returns 0.0 when both are near zero and None when denominator <= eps."""
    if den > eps:
        return float(num / den)
    return 0.0 if num <= eps else None


def summarize_logit_diffs(D: np.ndarray, decode, top_k: int = 10, conc_k: int = 10) -> Dict[str, Any]:
    """Summarize [N, V] mean-logit difference matrix into broad median and concentrated top-k lifts."""
    N, _ = D.shape
    ck = min(conc_k, N)
    med = np.median(D, axis=0)
    conc = np.mean(np.partition(D, N - ck, axis=0)[N - ck :, :], axis=0) - med
    top_med = np.argsort(-med, kind="stable")[:top_k]
    top_conc = np.argsort(-conc, kind="stable")[:top_k]
    min_med = np.argsort(med, kind="stable")[:top_k]
    return {
        "S_broad": float(med[top_med[0]]),
        "S_broad_neg": float(med[min_med[0]]),
        "S_conc": float(conc[top_conc[0]]),
        "rms_all": float(np.sqrt(np.mean(D.astype(np.float64) ** 2))),
        "top_broad": [{"id": int(j), "token": decode(int(j)), "med": float(med[j]), "conc": float(conc[j])} for j in top_med],
        "top_conc": [{"id": int(j), "token": decode(int(j)), "med": float(med[j]), "conc": float(conc[j])} for j in top_conc],
    }


def stochastic_round_grid(
    w_f32: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    bits: int = DEFAULT_BITS,
    group_size: int = DEFAULT_GROUP,
    seed: int = 0,
) -> np.ndarray:
    """Anchor-preserving stochastic rounding on the group-wise affine grid, unbiased (E[Q_SR(W)] = W) for values inside the grid.

    MLX moves the scale so that zero is on the grid, so the grid can stop short of a group's extreme: the values past its last
    point (but the anchor itself) always round to that point. Preserves exact group min and max anchors so subsequent standard nearest-grid quantization
    recovers the disarmed quantized grid points without re-triggering quantization-gap alignment.
    """
    rows, cols = w_f32.shape
    ng = cols // group_size
    max_code = float((1 << bits) - 1)
    wg = w_f32.astype(np.float32).reshape(rows, ng, group_size)
    sc = scales.astype(np.float32)[:, :, None]
    bi = biases.astype(np.float32)[:, :, None]
    safe = np.where(np.abs(sc) < 1e-12, 1.0, sc)
    u = np.clip(np.where(np.abs(sc) < 1e-12, 0.0, (wg - bi) / safe), 0.0, max_code)
    lo = np.floor(u)
    rng = np.random.default_rng(seed)
    q = np.clip(lo + (rng.random(u.shape, dtype=np.float32) < (u - lo)).astype(np.float32), 0.0, max_code)
    is_anchor = (wg == np.min(wg, axis=2, keepdims=True)) | (wg == np.max(wg, axis=2, keepdims=True))
    out = np.where(is_anchor, wg, bi + q * sc)
    return out.reshape(rows, cols).astype(np.float32)


def stochastic_round_nf4(
    w_f32: np.ndarray,
    absmax: np.ndarray,
    group_size: int = DEFAULT_GROUP,
    seed: int = 0,
) -> np.ndarray:
    """Anchor-preserving unbiased stochastic rounding on the NF4 quantizer grid (E[Q_SR(W)] = W)."""
    rows, cols = w_f32.shape
    ng = cols // group_size
    wg = w_f32.astype(np.float32).reshape(rows, ng, group_size)
    am = absmax.astype(np.float32)[:, :, None]
    safe = np.where(am < 1e-12, 1.0, am)
    norm = np.clip(np.where(am < 1e-12, 0.0, wg / safe), -1.0, 1.0)
    hi_idx = np.clip(np.searchsorted(NF4_LEVELS, norm, side="right"), 1, len(NF4_LEVELS) - 1)
    lo_idx = hi_idx - 1
    lv_lo = NF4_LEVELS[lo_idx]
    lv_hi = NF4_LEVELS[hi_idx]
    span = np.maximum(lv_hi - lv_lo, 1e-12)
    p_hi = (norm - lv_lo) / span
    rng = np.random.default_rng(seed)
    pick_hi = rng.random(norm.shape, dtype=np.float32) < p_hi
    lv_sr = np.where(pick_hi, lv_hi, lv_lo)
    is_anchor = np.abs(wg) == am
    out = np.where(is_anchor, wg, am * lv_sr)
    return out.reshape(rows, cols).astype(np.float32)


def _read_safetensors_tensor_numpy(f: Path, hn: int, meta: Dict[str, Any]) -> np.ndarray:
    """Read a BF16/F16/F32 tensor from a safetensors shard as a float32 NumPy array."""
    a, b = meta["data_offsets"]
    dtype = meta["dtype"]
    with open(f, "rb") as fh:
        fh.seek(8 + hn + a)
        raw = np.frombuffer(fh.read(b - a), dtype=np.uint8)
    if dtype == "BF16":
        arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dtype in ("F16", "F32"):
        arr = raw.view({"F16": np.float16, "F32": np.float32}[dtype]).astype(np.float32)
    else:
        raise SystemExit(f"unsupported safetensors dtype for mirror: {dtype!r}")
    return arr.reshape(meta["shape"])


def _encode_safetensors_bytes(arr_f32: np.ndarray, dtype: str) -> np.ndarray:
    """Encode a 2-D float32 array into uint8 row bytes matching its safetensors dtype (BF16/F16/F32)."""
    a = np.ascontiguousarray(arr_f32, dtype=np.float32)
    if dtype == "BF16":
        u32 = a.view(np.uint32)
        rounding = np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))
        return ((u32 + rounding) >> 16).astype(np.uint16).view(np.uint8).reshape(a.shape[0], -1)
    if dtype == "F16":
        return a.astype(np.float16).view(np.uint8).reshape(a.shape[0], -1)
    if dtype == "F32":
        return a.view(np.uint8).reshape(a.shape[0], -1)
    raise SystemExit(f"unsupported safetensors dtype for disarm: {dtype!r}")


def _mlx_quantize_tensor(w_mx: Any, bits: int, group_size: int, mode: str):
    import mlx.core as mx

    packed, scales, biases = mx.quantize(w_mx, group_size=group_size, bits=bits, mode=mode)
    w_q = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits, mode=mode).astype(mx.float32)
    mx.eval(w_q, packed, scales, biases)
    return (
        np.array(w_mx.astype(mx.float32)),
        np.array(w_q),
        unpack_affine_codes(np.array(packed), bits=bits),
        np.array(scales.astype(mx.float32)),
        np.array(biases.astype(mx.float32)),
    )


def _is_target_weight(name: str, shape: Sequence[int], group_size: int) -> bool:
    if len(shape) != 2 or "visual" in name or "mtp." in name or "embed_tokens" in name or "lm_head" in name:
        return False
    return int(shape[-1]) % group_size == 0


def _finalize_behavior_mirror(
    Z0: np.ndarray,
    Zp: np.ndarray,
    Zm: np.ndarray,
    Znulls: Sequence[np.ndarray],
    decode,
    n_prompts: int,
    n_tensors: int,
) -> Dict[str, Any]:
    """Compute behavior mirror coherence ratios and antisymmetry summary from mean-logit matrices."""
    Dp, Dm = Zp - Z0, Zm - Z0
    sum_p = summarize_logit_diffs(Dp, decode)
    sum_m = summarize_logit_diffs(Dm, decode)
    sum_curv = summarize_logit_diffs(0.5 * (Dp + Dm), decode)
    top_id_p = sum_p["top_broad"][0]["id"]
    nulls = [summarize_logit_diffs(zn - Z0, decode) for zn in Znulls]

    s_null_mean = float(np.mean([s["S_broad"] for s in nulls]))
    s_null_max = float(np.max([s["S_broad"] for s in nulls]))
    s_conc_null_mean = float(np.mean([s["S_conc"] for s in nulls]))
    rms_null_mean = float(np.mean([s["rms_all"] for s in nulls]))
    coh_broad = _safe_ratio(sum_p["S_broad"], s_null_mean)
    coh_conc = _safe_ratio(sum_p["S_conc"], s_conc_null_mean)
    rms_coh = _safe_ratio(sum_p["rms_all"], rms_null_mean)

    return {
        "prompts": n_prompts,
        "tensors_perturbed": n_tensors,
        "flagged": bool(
            (coh_broad is not None and coh_broad > BEHAVIOR_COH_THRESH)
            or (coh_conc is not None and coh_conc > BEHAVIOR_COH_THRESH)
        ),
        "S_plus": sum_p["S_broad"],
        "S_minus": sum_m["S_broad"],
        "S_null_mean": s_null_mean,
        "S_null_max": s_null_max,
        "coherence_ratio_broad": coh_broad,
        "S_conc_plus": sum_p["S_conc"],
        "S_conc_null_mean": s_conc_null_mean,
        "coherence_ratio_conc": coh_conc,
        "S_curv": sum_curv["S_broad"],
        "rms_plus": sum_p["rms_all"],
        "rms_minus": sum_m["rms_all"],
        "rms_null_mean": rms_null_mean,
        "rms_coherence_ratio": rms_coh,
        "top_token_plus": sum_p["top_broad"][0],
        "mirror_med_at_top_plus": float(np.median(Dm[:, top_id_p])),
        "plus": sum_p,
        "minus": sum_m,
        "nulls": nulls,
    }


def _behavior_mirror_mlx(
    model_dir: str,
    prompts_ids: Sequence[Sequence[int]],
    bits: int,
    group_size: int,
    mode: str,
    n_null: int = N_NULL_BEHAVIOR,
    seed: int = 0,
) -> Dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load

    from .inspect import mean_logits

    model, tok = load(str(model_dir))
    flat = dict(tree_flatten(model.parameters()))
    orig = {k: v for k, v in flat.items() if _is_target_weight(k, v.shape, group_size)}

    err = {}
    for k, w in orig.items():
        p, s, b = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
        wq = mx.dequantize(p, s, b, group_size=group_size, bits=bits, mode=mode).astype(mx.float32)
        e = wq - w.astype(mx.float32)
        mx.eval(e)
        err[k] = e

    def apply(mapping: Dict[str, Any]) -> None:
        model.update(tree_unflatten(list(mapping.items())))
        mx.eval(model.parameters())

    def mean_logits_all() -> np.ndarray:
        return np.stack([mean_logits(model, list(ids)) for ids in prompts_ids], axis=0)

    Z0 = mean_logits_all()
    apply({k: (orig[k].astype(mx.float32) + err[k]).astype(orig[k].dtype) for k in orig})
    Zp = mean_logits_all()
    apply({k: (orig[k].astype(mx.float32) - err[k]).astype(orig[k].dtype) for k in orig})
    Zm = mean_logits_all()

    Znulls = []
    for k_i in range(n_null):
        rng = np.random.default_rng(seed + k_i * 1009)
        apply({
            k: (orig[k].astype(mx.float32) + err[k] * mx.array(rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=orig[k].shape))).astype(orig[k].dtype)
            for k in orig
        })
        Znulls.append(mean_logits_all())
    apply(orig)

    decode = lambda j: tok.decode([j])  # noqa: E731
    return _finalize_behavior_mirror(Z0, Zp, Zm, Znulls, decode, len(prompts_ids), len(orig))


def _behavior_mirror_torch(
    model_dir: str,
    prompts_ids: Sequence[Sequence[int]],
    bits: int,
    group_size: int,
    mode: str,
    n_null: int = N_NULL_BEHAVIOR,
    seed: int = 0,
) -> Dict[str, Any]:
    import torch
    from kestrom_verify import pt
    from transformers import AutoTokenizer

    model = pt.load(str(model_dir))
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    params = {k: p for k, p in sorted(model.named_parameters()) if _is_target_weight(k, tuple(p.shape), group_size)}
    orig_cpu = {k: p.detach().cpu().clone() for k, p in params.items()}

    err: Dict[str, np.ndarray] = {}
    for k, w_cpu in orig_cpu.items():
        wf = w_cpu.float().numpy()
        if mode == "nf4":
            wq, _, _ = nf4_quantize_numpy(wf, group_size=group_size)
        else:
            wq, _, _, _ = affine_quantize_numpy(wf, bits=bits, group_size=group_size)
        err[k] = (wq - wf).astype(np.float32)

    def apply(sign: float = 0.0, rng: Optional[np.random.Generator] = None) -> None:
        with torch.no_grad():
            for k, p in params.items():
                if sign == 0.0:
                    p.data.copy_(orig_cpu[k].to(p.device))
                    continue
                w_new = orig_cpu[k].to(p.device, dtype=torch.float32)
                e_t = torch.from_numpy(err[k]).to(p.device)
                if rng is not None:
                    flips = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=p.shape)
                    e_t = e_t * torch.from_numpy(flips).to(p.device)
                p.data.copy_((w_new + sign * e_t).to(p.dtype))

    def mean_logits_all() -> np.ndarray:
        rows = []
        with torch.no_grad():
            for ids in prompts_ids:
                logits = model(torch.tensor([list(ids)], device=model.device), use_cache=False).logits[0]
                rows.append(logits.float().mean(dim=0).cpu().numpy())
        return np.stack(rows, axis=0)

    Z0 = mean_logits_all()
    apply(1.0)
    Zp = mean_logits_all()
    apply(-1.0)
    Zm = mean_logits_all()

    Znulls = []
    for k_i in range(n_null):
        rng = np.random.default_rng(seed + k_i * 1009)
        apply(1.0, rng=rng)
        Znulls.append(mean_logits_all())
    apply(0.0)

    decode = lambda j: tok.decode([j])  # noqa: E731
    return _finalize_behavior_mirror(Z0, Zp, Zm, Znulls, decode, len(prompts_ids), len(params))


def mirror(
    model_dir,
    out,
    prompts=None,
    bits=DEFAULT_BITS,
    group=DEFAULT_GROUP,
    mode=DEFAULT_MODE,
    runtime=None,
):
    """Run `kestrom inspect mirror` on a single checkpoint with no original model and no trigger."""
    from kestrom_verify import report, weights

    from .inspect import write

    bits, group, mode = int(bits), int(group), str(mode)
    if mode not in ("affine", "nf4"):
        raise SystemExit(f"mirror supports affine or nf4 mode, not {mode!r}")
    if mode == "affine" and bits not in (2, 4, 8):
        raise SystemExit(f"affine mode supports 2, 4 or 8 bits only, not {bits}")
    if mode == "nf4" and bits != 4:
        raise SystemExit(f"nf4 mode supports 4 bits only, not {bits}")
    rt = ("torch" if mode == "nf4" else "mlx") if runtime in (None, "", "none") else str(runtime)
    if rt not in ("mlx", "torch"):
        raise SystemExit(f"mirror supports mlx or torch runtime, not {rt!r}")
    P = None if prompts in (None, "", "none") else [p["ids"] for p in report.read(prompts)]

    folder = Path(model_dir)
    tensors_out: Dict[str, Any] = {}
    idx = 0
    if mode == "affine" and rt == "mlx":
        import mlx.core as mx

        for f in sorted(folder.glob("*.safetensors")):
            W = mx.load(str(f))
            for name in sorted(W):
                w = W[name]
                if not _is_target_weight(name, w.shape, group):
                    continue
                wf, wq, codes, scales, _ = _mlx_quantize_tensor(w, bits=bits, group_size=group, mode=mode)
                tensors_out[name] = analyze_residual(wf, wq, codes, scales, bits=bits, group_size=group, seed=idx * 17)
                idx += 1
    else:
        for f in sorted(folder.glob("*.safetensors")):
            hn, h = weights.header(f)
            for name in sorted(h):
                if name == "__metadata__":
                    continue
                meta = h[name]
                if not _is_target_weight(name, meta["shape"], group):
                    continue
                wf = _read_safetensors_tensor_numpy(f, hn, meta)
                if mode == "nf4":
                    wq, codes, scales = nf4_quantize_numpy(wf, group_size=group)
                    tensors_out[name] = analyze_residual(
                        wf, wq, codes, scales, bits=bits, group_size=group, seed=idx * 17, levels=NF4_LEVELS
                    )
                else:
                    wq, codes, scales, _ = affine_quantize_numpy(wf, bits=bits, group_size=group)
                    tensors_out[name] = analyze_residual(wf, wq, codes, scales, bits=bits, group_size=group, seed=idx * 17)
                idx += 1

    judged = judge_mirror_tensors(tensors_out)
    res: Dict[str, Any] = {
        "model": str(model_dir),
        "quantizer": {"mode": mode, "bits": bits, "group": group, "runtime": rt},
        "tensors_scanned": len(tensors_out),
        "thresholds": {
            "rho1": RHO1_THRESH,
            "b85": B85_THRESH,
            "on_grid_mean_u": ON_GRID_MEAN_U,
            "collapse_ge90": COLLAPSE_GE90_THRESH,
            "behavior_coherence": BEHAVIOR_COH_THRESH,
        },
        "flagged": judged["flagged"],
        "reasons": judged["reasons"],
        "on_grid": judged["on_grid"],
        "weight_summary": {k: v for k, v in judged.items() if k not in ("flagged", "reasons", "on_grid")},
        "tensors": tensors_out,
    }

    if P is not None:
        beh_fn = _behavior_mirror_torch if rt == "torch" else _behavior_mirror_mlx
        res["behavior"] = beh_fn(str(model_dir), P, bits=bits, group_size=group, mode=mode)

    write(out, res)
    bstr = ""
    if "behavior" in res:
        b = res["behavior"]
        cstr = f"{b['coherence_ratio_broad']:.2f}x" if b["coherence_ratio_broad"] is not None else "n/a"
        bstr = f" | behavior coh={cstr} S+={b['S_plus']:.3f} ({b['top_token_plus']['token']!r}) dm={b['mirror_med_at_top_plus']:.3f}"
    print(
        f"{model_dir} scanned {len(tensors_out)} | flagged {len(res['flagged'])} {res['flagged'][:6]} "
        f"| on_grid {len(res['on_grid'])} | max_rho1 {judged['max_rho1']:.4f} max_b85 {judged['max_b85']:.4f} max_ge90 {judged['max_collapse_ge90']}{bstr}",
        flush=True,
    )


def disarm(model_dir, mirror_file, dst, seed=0, runtime=None):
    """Trigger-blind disarming: apply anchor-preserving unbiased stochastic rounding ONLY to tensors flagged in MIRROR.json."""
    from kestrom_verify import weights

    M = json.loads(Path(mirror_file).read_text(encoding="utf-8"))
    flagged = set(M.get("flagged", []))
    q = M["quantizer"]
    bits, group, mode = int(q["bits"]), int(q["group"]), str(q["mode"])
    if mode not in ("affine", "nf4"):
        raise SystemExit(f"disarm supports affine or nf4 mode, not {mode!r}")
    rt = str(q.get("runtime", "torch" if mode == "nf4" else "mlx")) if runtime in (None, "", "none") else str(runtime)
    if rt not in ("mlx", "torch"):
        raise SystemExit(f"disarm supports mlx or torch runtime, not {rt!r}")
    seed = int(seed)

    src, dst_p = Path(model_dir), Path(dst)
    shards = sorted(src.glob("*.safetensors"))
    shard_metas = [(f, *weights.header(f)) for f in shards]
    shard_headers = {f: h for f, _, h in shard_metas}
    missing = sorted(flagged - set().union(*(h.keys() for h in shard_headers.values())))
    if missing:
        raise SystemExit(f"{len(missing)} tensors flagged in {mirror_file} are not in {model_dir}: {missing[:6]}")

    # Map deterministic per-tensor seed by sorted flagged name
    seed_map = {name: seed + i for i, name in enumerate(sorted(flagged))}
    replaced: List[str] = []

    if mode == "affine" and rt == "mlx":
        import shutil
        import mlx.core as mx

        weights.copy(src, dst_p, set(), set(shards))
        for f in shards:
            W = mx.load(str(f))
            changed = False
            for name in sorted(W):
                if name not in flagged:
                    continue
                w = W[name]
                wf, _, _, scales, biases = _mlx_quantize_tensor(w, bits=bits, group_size=group, mode=mode)
                wq_sr = stochastic_round_grid(wf, scales, biases, bits=bits, group_size=group, seed=seed_map[name])
                W[name] = mx.array(wq_sr).astype(w.dtype)
                replaced.append(name)
                changed = True
            if changed:
                mx.save_safetensors(str(dst_p / f.relative_to(src)), W, metadata=shard_headers[f].get("__metadata__"))
            else:
                shutil.copy2(f, dst_p / f.relative_to(src))
    else:
        flagged_shards = {f for f, h in shard_headers.items() if flagged & set(h)}
        weights.copy(src, dst_p, flagged_shards)
        for f, hn, h in shard_metas:
            if f not in flagged_shards:
                continue
            dst_f = dst_p / f.relative_to(src)
            for name in sorted(h):
                if name not in flagged:
                    continue
                meta = h[name]
                wf = _read_safetensors_tensor_numpy(f, hn, meta)
                if mode == "nf4":
                    _, _, absmax = nf4_quantize_numpy(wf, group_size=group)
                    wq_sr = stochastic_round_nf4(wf, absmax, group_size=group, seed=seed_map[name])
                else:
                    _, _, scales, biases = affine_quantize_numpy(wf, bits=bits, group_size=group)
                    wq_sr = stochastic_round_grid(wf, scales, biases, bits=bits, group_size=group, seed=seed_map[name])
                a, b = meta["data_offsets"]
                rows_mm = weights.rows(dst_f, 8 + hn + a, 8 + hn + b, meta["shape"][0], "r+")
                rows_mm[:] = _encode_safetensors_bytes(wq_sr, meta["dtype"])
                rows_mm.flush()
                replaced.append(name)

    print(f"{dst_p} disarmed {len(replaced)} of {len(flagged)} flagged tensors (seed {seed}, {mode} {bits}-bit g{group})", flush=True)
