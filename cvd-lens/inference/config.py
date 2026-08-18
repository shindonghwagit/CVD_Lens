"""
Inference post-processing config. Values are read from environment variables so
they are tunable per deployment without code changes; defaults below are the
shipped values.

Guided-filter post-processing (see guided.py) smooths the upsampled correction
delta against the original image as guide. radius is derived per-image as
max(H, W) // GUIDED_RADIUS_DIVISOR.
"""
from __future__ import annotations

import os


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Master on/off for the guided-filter delta post-processing (before/after A-B).
GUIDED_FILTER_ENABLED = _flag("CVDLENS_GUIDED_FILTER", True)

# radius = max(H, W) // divisor  (larger divisor → smaller window → less smoothing)
GUIDED_RADIUS_DIVISOR = _int("CVDLENS_GUIDED_RADIUS_DIVISOR", 20)

# covariance regularization (larger → smoother / more box-like)
GUIDED_EPS = _float("CVDLENS_GUIDED_EPS", 1e-3)

# Resolution cap for the guided filter. The full 3x3 covariance/inverse (~25
# float32 arrays) OOMs a 512 MB instance at 2048² — running the filter on a
# downscaled guide and upsampling the (smooth) result keeps it well under budget.
# Peak covariance memory ≈ 25 · max_side² · 4 bytes (512 → ~26 MB, 2048 → ~420 MB).
# 512 preserves the CRR recovery (coefficients are smooth; the apply uses the
# full-res guide so edges stay sharp) with a wide margin under the 512 MB limit.
GUIDED_MAX_SIDE = _int("CVDLENS_GUIDED_MAX_SIDE", 512)

# /infer 응답 JPEG 품질. q92는 CRR 경계 케이스(traffic p/d)의 얇은 회복 마진을 깎아
# CRR<1로 만든다(jpeg_sweep). q94가 경계 3케이스 모두 CRR≥1.0을 만족하는 최소값이라
# 기본값으로 상향(q92 대비 응답 +~21%). reports/jpeg_sweep/REPORT.md 참조.
RESPONSE_JPEG_QUALITY = _int("CVDLENS_RESPONSE_JPEG_QUALITY", 94)
