"""
Fast guided filter (He, Sun, Tang 2010/2013) — color-guide variant, NumPy + cv2.

No opencv-contrib dependency: box means come from cv2.boxFilter (integral-image,
O(1) per pixel regardless of radius), and the per-pixel 3x3 guide-covariance is
inverted in closed form.

Used to post-process the upsampled correction delta with the ORIGINAL image as
guide: the delta is snapped to the original's edges (region boundaries sharpen)
and flattened inside uniform-guide regions (low-frequency delta gradient removed),
without touching the ONNX model.

MEMORY: the full 3x3 covariance/inverse is ~25 float32 arrays; at 2048² that is
~420 MB and OOMs a 512 MB instance. `guided_filter(max_side=...)` runs the *fast*
variant — linear coefficients (a, b) are computed on a downscaled guide, then
UPSAMPLED and applied against the full-resolution guide (q = ā·I + b̄). Because a,b
are smooth, low-res coefficients are near-lossless; and because the apply uses the
full-res guide, edge sharpness (which drives the CRR recovery) is preserved — unlike
naively upsampling the filtered result.

CANONICAL COPY. `cvdlens_v2` local eval scripts import THIS file (via sys.path) so
the deployed pipeline and the offline evaluation apply an identical filter.
"""
from __future__ import annotations

import cv2
import numpy as np


def _box(x: np.ndarray, r: int) -> np.ndarray:
    """Box mean, window (2r+1). Reflect border keeps edges sane."""
    return cv2.boxFilter(x, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1),
                         normalize=True, borderType=cv2.BORDER_REFLECT)


def _inv_cov(I: np.ndarray, r: int, eps: float):
    """Return (mean_I channels, inverse guide-covariance entries) at I's resolution."""
    Ir, Ig, Ib = I[..., 0], I[..., 1], I[..., 2]
    mean_I = _box(I, r)
    mI_r, mI_g, mI_b = mean_I[..., 0], mean_I[..., 1], mean_I[..., 2]

    var_rr = _box(Ir * Ir, r) - mI_r * mI_r + eps
    var_rg = _box(Ir * Ig, r) - mI_r * mI_g
    var_rb = _box(Ir * Ib, r) - mI_r * mI_b
    var_gg = _box(Ig * Ig, r) - mI_g * mI_g + eps
    var_gb = _box(Ig * Ib, r) - mI_g * mI_b
    var_bb = _box(Ib * Ib, r) - mI_b * mI_b + eps

    inv_rr = var_gg * var_bb - var_gb * var_gb
    inv_rg = var_rb * var_gb - var_rg * var_bb
    inv_rb = var_rg * var_gb - var_rb * var_gg
    inv_gg = var_rr * var_bb - var_rb * var_rb
    inv_gb = var_rb * var_rg - var_rr * var_gb
    inv_bb = var_rr * var_gg - var_rg * var_rg
    det = var_rr * inv_rr + var_rg * inv_rg + var_rb * inv_rb
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    return (mI_r, mI_g, mI_b), (inv_rr / det, inv_rg / det, inv_rb / det,
                                inv_gg / det, inv_gb / det, inv_bb / det)


def _channel_coeffs(I, mI, inv, p, r):
    """Mean linear coefficients (ā_r, ā_g, ā_b, b̄) for one input channel p."""
    Ir, Ig, Ib = I[..., 0], I[..., 1], I[..., 2]
    mI_r, mI_g, mI_b = mI
    inv_rr, inv_rg, inv_rb, inv_gg, inv_gb, inv_bb = inv
    mean_p = _box(p, r)
    cov_r = _box(Ir * p, r) - mI_r * mean_p
    cov_g = _box(Ig * p, r) - mI_g * mean_p
    cov_b = _box(Ib * p, r) - mI_b * mean_p
    a_r = inv_rr * cov_r + inv_rg * cov_g + inv_rb * cov_b
    a_g = inv_rg * cov_r + inv_gg * cov_g + inv_gb * cov_b
    a_b = inv_rb * cov_r + inv_gb * cov_g + inv_bb * cov_b
    b = mean_p - a_r * mI_r - a_g * mI_g - a_b * mI_b
    return _box(a_r, r), _box(a_g, r), _box(a_b, r), _box(b, r)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float,
                  max_side: int | None = None) -> np.ndarray:
    """Color-guided filter (fast variant when max_side caps the resolution).

    Args:
        guide: (H,W,3) float32 in [0,1] — the original image.
        src:   (H,W,C) or (H,W) float32 — signal to filter (the delta).
        radius: box radius at NATIVE resolution (scaled with the image).
        eps:    covariance regularization.
        max_side: if set and max(H,W) > max_side, compute coefficients on a
                  downscaled guide and apply them at full res (memory-bounded).
    Returns:
        Filtered src, native resolution, float32.
    """
    I = np.ascontiguousarray(guide, dtype=np.float32)
    H, W = I.shape[:2]
    single = src.ndim == 2
    src3 = np.ascontiguousarray(src if not single else src[..., None], dtype=np.float32)

    cap = bool(max_side and max(H, W) > max_side)
    if cap:
        scale = max_side / max(H, W)
        gw, gh = max(1, round(W * scale)), max(1, round(H * scale))
        I_c = cv2.resize(I, (gw, gh), interpolation=cv2.INTER_AREA)
        r_c = max(1, int(round(radius * scale)))
    else:
        I_c, r_c = I, max(1, radius)

    mI, inv = _inv_cov(I_c, r_c, eps)
    Ir, Ig, Ib = I[..., 0], I[..., 1], I[..., 2]

    out = np.empty((H, W, src3.shape[2]), dtype=np.float32)
    for c in range(src3.shape[2]):
        p = src3[..., c]
        p_c = cv2.resize(p, (gw, gh), interpolation=cv2.INTER_AREA) if cap else p
        ma_r, ma_g, ma_b, mb = _channel_coeffs(I_c, mI, inv, p_c, r_c)
        if cap:
            # Upsample the (smooth) coefficients, apply against the full-res guide.
            # One reused native temp (in-place *=, +=) keeps peak memory bounded.
            q = cv2.resize(mb, (W, H), interpolation=cv2.INTER_LINEAR)
            for ma, Ich in ((ma_r, Ir), (ma_g, Ig), (ma_b, Ib)):
                tmp = cv2.resize(ma, (W, H), interpolation=cv2.INTER_LINEAR)
                tmp *= Ich
                q += tmp
            out[..., c] = q
        else:
            out[..., c] = ma_r * Ir + ma_g * Ig + ma_b * Ib + mb

    return out[..., 0] if single else out
