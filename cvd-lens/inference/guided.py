"""
Fast guided filter (He, Sun, Tang 2010/2013) — color-guide variant, NumPy + cv2.

No opencv-contrib dependency: box means come from cv2.boxFilter (integral-image,
O(1) per pixel regardless of radius), and the per-pixel 3x3 guide-covariance is
inverted in closed form.

Used to post-process the upsampled correction delta with the ORIGINAL image as
guide: the delta is snapped to the original's edges (region boundaries sharpen)
and flattened inside uniform-guide regions (low-frequency delta gradient removed),
without touching the ONNX model.

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


def guided_filter(guide: np.ndarray, src: np.ndarray,
                  radius: int, eps: float) -> np.ndarray:
    """Color-guided filter.

    Args:
        guide: (H,W,3) float32 in [0,1] — the original image.
        src:   (H,W,C) or (H,W) float32 — signal to filter (the delta).
        radius: box radius (window = 2*radius+1).
        eps:    covariance regularization (larger = smoother).
    Returns:
        Filtered src, same shape/dtype-ish as input (float32).
    """
    I = np.ascontiguousarray(guide, dtype=np.float32)
    r = int(max(1, radius))
    Ir, Ig, Ib = I[..., 0], I[..., 1], I[..., 2]

    mean_I = _box(I, r)                                   # (H,W,3)
    mI_r, mI_g, mI_b = mean_I[..., 0], mean_I[..., 1], mean_I[..., 2]

    # Guide covariance (symmetric 3x3 per pixel), diagonal regularized by eps.
    var_rr = _box(Ir * Ir, r) - mI_r * mI_r + eps
    var_rg = _box(Ir * Ig, r) - mI_r * mI_g
    var_rb = _box(Ir * Ib, r) - mI_r * mI_b
    var_gg = _box(Ig * Ig, r) - mI_g * mI_g + eps
    var_gb = _box(Ig * Ib, r) - mI_g * mI_b
    var_bb = _box(Ib * Ib, r) - mI_b * mI_b + eps

    # Closed-form inverse of the symmetric 3x3 (cofactors / determinant).
    inv_rr = var_gg * var_bb - var_gb * var_gb
    inv_rg = var_rb * var_gb - var_rg * var_bb
    inv_rb = var_rg * var_gb - var_rb * var_gg
    inv_gg = var_rr * var_bb - var_rb * var_rb
    inv_gb = var_rb * var_rg - var_rr * var_gb
    inv_bb = var_rr * var_gg - var_rg * var_rg
    det = var_rr * inv_rr + var_rg * inv_rg + var_rb * inv_rb
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    inv_rr /= det; inv_rg /= det; inv_rb /= det
    inv_gg /= det; inv_gb /= det; inv_bb /= det

    src = np.ascontiguousarray(src, dtype=np.float32)
    single = src.ndim == 2
    if single:
        src = src[..., None]

    out = np.empty_like(src)
    for c in range(src.shape[2]):
        p = src[..., c]
        mean_p = _box(p, r)
        cov_r = _box(Ir * p, r) - mI_r * mean_p
        cov_g = _box(Ig * p, r) - mI_g * mean_p
        cov_b = _box(Ib * p, r) - mI_b * mean_p

        a_r = inv_rr * cov_r + inv_rg * cov_g + inv_rb * cov_b
        a_g = inv_rg * cov_r + inv_gg * cov_g + inv_gb * cov_b
        a_b = inv_rb * cov_r + inv_gb * cov_g + inv_bb * cov_b
        b = mean_p - a_r * mI_r - a_g * mI_g - a_b * mI_b

        out[..., c] = (_box(a_r, r) * Ir + _box(a_g, r) * Ig
                       + _box(a_b, r) * Ib + _box(b, r))

    return out[..., 0] if single else out
